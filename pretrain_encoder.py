import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import torchmetrics
import torch.optim.lr_scheduler as lr_scheduler 
import matplotlib
# matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
import numpy as np
from torchvision.transforms import Resize
import tifffile as tiff
import os
import sys
from torch.utils.data import Dataset
import albumentations as A 
from torch.utils.data import random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from transformers import ConvNextV2Config, ConvNextV2Model
from timm.models.layers import trunc_normal_, DropPath
from transformers import ConvNextConfig, ConvNextModel
from transformers import Swinv2Config, Swinv2Model
from convnextv2 import convnextv2_atto
from transformers import Swinv2Config, Swinv2Model, UperNetConfig, UperNetForSemanticSegmentation, Swinv2ForMaskedImageModeling
from transformers import AutoConfig




def collate_fn(inputs):    
    # hyperspectral
    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i for i in inputs], dim=0)
    batch["mask"] = torch.stack([mask_generator() for i in inputs], dim=0)
    return batch   

    # hsi_pixel_values = torch.stack([item for item in inputs], dim=0)
    # return {"hsi_pixel_values": hsi_pixel_values}

def spectral_angle_mapper_loss(predicted, target):
    # Flatten the tensors to (batch_size, num_channels, -1)
    predicted_flat = predicted.view(predicted.size(0), predicted.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # Compute the dot product between predicted and target
    dot_product = torch.sum(predicted_flat * target_flat, dim=1)
    
    # Compute the norms of predicted and target
    norm_predicted = torch.norm(predicted_flat, dim=1)
    norm_target = torch.norm(target_flat, dim=1)
    
    # Compute the spectral angle
    cos_theta = dot_product / (norm_predicted * norm_target + 1e-8)  # Add a small value to avoid division by zero
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    
    # Compute the mean spectral angle over all pixels
    sam_loss = torch.mean(theta)
    
    return sam_loss

class BaseSegmentationModel(L.LightningModule):
        def __init__(self,  learning_rate = 1e-3, num_channels=12, num_workers=4, train_dataset=None, val_dataset=None,  batch_size=2, training_epochs = 100 ):
            super().__init__()
            
            self.learning_rate = learning_rate
            # self.batch_size = batch_size override in dataloaders
            self.num_workers = num_workers
            self.num_channels = num_channels
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.training_epochs = training_epochs

            
            self.save_hyperparameters()
            
            # self.loss_fn = torch.nn.MSELoss()
            # self.loss_fn = spectral_angle_mapper_loss


        def forward(self, hsi_img):
            raise NotImplementedError("Subclasses should implement this method")
        
        def log_data(self, step_type, loss):       

            if step_type == "train":
                pass
                optimizer = self.optimizers()
                lr = optimizer.param_groups[0]['lr']
                self.log(f"{step_type}_learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            elif step_type == "val":
                pass

            else:
                raise ValueError("step_type must be one of 'train', 'val'")
            
            self.log(f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        def hsi_step(self, batch):
            
            hsi_pixel_values = batch["hsi_pixel_values"]
            mask = batch["mask"]
            # print( mask.shape)
            # logits, spectral_adapted_pixel_values = self.forward(hsi_pixel_values, mask)
            # return logits, spectral_adapted_pixel_values 
            loss = self.forward(hsi_pixel_values, mask)
            return loss 
            
        def training_step(self, batch):
            
            step_type = "train"      
            # logits, hsi_img = self.hsi_step(batch)
            # loss = self.loss_fn(logits, hsi_img) 
            loss = self.hsi_step(batch)
            self.log_data(step_type, loss)
            return loss
                
        def validation_step(self, batch):
                
            step_type = "val"
            # logits, hsi_img = self.hsi_step(batch)
            # loss = self.loss_fn(logits, hsi_img) 
            loss = self.hsi_step(batch)
            self.log_data(step_type,  loss)
            return loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.05, betas=(0.9, 0.95))
            
            # Warmup scheduler
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=40)
            
            # Cosine annealing warm restarts scheduler
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.training_epochs - 40, eta_min=1e-6)

            # Combine the warmup and cosine annealing schedulers
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[40])

            return {
            'optimizer': optimizer,
            # "lr_scheduler": scheduler
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor for learning rate adjustment
                'interval': 'epoch',    # How often to apply the scheduler
                'frequency': 1          # Frequency of the scheduler
            }
             }     
              
        def train_dataloader(self):
            
            return  DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=self.num_workers, drop_last=True)
        
        def val_dataloader(self):
            
            return  DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=self.num_workers, drop_last=True)
        
class fractal_dataset(Dataset):
    def __init__(self,  root_dir,  transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform


        self.img_dir =  root_dir
       
        
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.' + 'tiff')]
        self.num_images = len( self.img_names  ) 

        # print("Number of images in the dataset: ", self.num_images)
        
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        
        file_name = str(idx) + '.tiff'
        hsi_img = tiff.imread(os.path.join(self.img_dir, file_name))
        
        # print(hsi_img.shape,file_name, idx, hsi_img.dtype)
        
        hsi_img = np.transpose(hsi_img, (1, 2, 0)) # transpose to x,y,channels for albumnetations
        # print(hsi_img.shape,file_name, idx)
        
        # apply transformations  # must be in x,y,channels format        
        if self.transform:            
            transformed = self.transform(image = hsi_img)
        
            hsi_img = torch.tensor(transformed['image'])
        else:
            hsi_img = torch.tensor(hsi_img)
              
        #convert from x,y,channels to channels, x, y
        hsi_img = hsi_img.permute(2,0,1)
        
        #convert from uint8 to float32
        hsi_img = hsi_img.float()
            
        return hsi_img      


class convnext2_model(BaseSegmentationModel):
    def __init__(self,  learning_rate=1.5e-4, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, batch_size=2, training_epochs=100, image_size=256, patch_size=4):
        super().__init__( learning_rate, num_channels, num_workers, train_dataset, val_dataset, batch_size, training_epochs)
        
        
        self.patch_size = patch_size
        # large config should be around 198m parameters
        embed_dim = 192 # 128 is base , 352 is huge 
        decoder_embed_dim = embed_dim*8 #512
        
        
        # # isntantiate the model the same way as used for training 
        # seg_head = UperNetConfig(
            
        #     # backbone="convnextv2_config/convnextv2_backbone", 
        #     backbone="facebook/convnextv2-large-22k-384", 
        #     use_pretrained_backbone=False,
            
        #     # backbone_config=backbone_configuration, 
            
        #     num_labels = 2,    
        #     out_features=["stage1", "stage2", "stage3", "stage4"],
        #     use_auxiliary_head=False,
        #     num_channels= num_channels,   
        #     image_size=image_size,   
        #     patch_size=patch_size,       
        # )                   
        # self.backbone_upernet = UperNetForSemanticSegmentation(seg_head)
        
        
        backbone_config = {
            "backbone": "facebook/convnextv2-large-22k-384",
            "use_pretrained_backbone": True,
            "num_channels": num_channels,
            "image_size": image_size,
            "patch_size": patch_size,
        }
        
        
        # Instantiate the backbone model
        self.backbone = ConvNextV2Model.from_pretrained(
            backbone_config["backbone"],
            num_channels=backbone_config["num_channels"],
            image_size=backbone_config["image_size"],
            patch_size=backbone_config["patch_size"],
            ignore_mismatched_sizes=True
        )


        # Modify the patch embeddings if necessary
        self.backbone.embeddings.patch_embeddings = nn.Conv2d(num_channels, 192, kernel_size=(4, 4), stride=(4, 4))
        self.backbone.embeddings.num_channels = num_channels
        self.backbone.embeddings.patch_embeddings.num_channels = num_channels

        # Save the backbone configuration if needed
        self.backbone.config.save_pretrained("convnextv2_config")
        self.backbone.train()
        # self.backbone_upernet.backbone.embeddings.patch_embeddings = nn.Conv2d(num_channels, 192, kernel_size=(4, 4), stride=(4, 4))
        # self.backbone_upernet.backbone.embeddings.num_channels = num_channels
        # self.backbone_upernet.backbone.embeddings.patch_embeddings.num_channels = num_channels
        
        
        # self.backbone = self.backbone_upernet.backbone
        # print(self.backbone.config)
        # sys.exit()
        
       
        
        # self.backbone.config.save_pretrained("convnextv2_config")
        
        
        # print(self.backbone.config.num_channels, self.backbone.embeddings.num_channels )
        
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim, embed_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim, num_channels, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Assuming the original image has pixel values in the range [0, 1]
        )
        
        # print(self.backbone)
        # print(self.backbone.config)
        
        # self.backbone = convnextv2_atto(in_chans= num_channels)
        
        # add decode head 
        
        # # decoder
        # self.proj = nn.Conv2d(
        #     in_channels=embed_dim*8, 
        #     out_channels=decoder_embed_dim, 
        #     kernel_size=1)
        # # mask tokens
        # # self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        # decoder = [Block(
        #     dim=decoder_embed_dim, 
        #     drop_path=0.) for i in range(1)]
        # self.decoder = nn.Sequential(*decoder)
        # # pred
        # self.pred = nn.Conv2d(
        #     in_channels=decoder_embed_dim,
        #     out_channels=patch_size ** 2 * num_channels,
        #     kernel_size=1)

        # # # print(self.decoder)
    
    def forward(self, hsi_img):
        
        # print(hsi_img.shape)
        x = self.backbone(hsi_img)
        # x = x.hidden_states
        # print(x.last_hidden_state.shape, x.pooler_output.shape)
        x = x.last_hidden_state
        x = self.decoder(x)
        # print(x.shape)
        
        return x

class SpectralAdapter(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAdapter, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=7, stride=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x

class SpectralAdapter_new(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAdapter_new, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 200, kernel_size=1 )
        self.bn1 = nn.BatchNorm2d(200)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(200, 150, kernel_size=1 )
        self.bn2 = nn.BatchNorm2d(150)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(150, 128, kernel_size=1 )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # x = self.global_pool(x)
        # x = x.view(x.size(0), -1)  # Flatten the output
        return x

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mim.py    
class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=256, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())   
    
class swin2_model(BaseSegmentationModel):
    def __init__(self,  learning_rate=1.5e-4, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, batch_size=2, training_epochs=100, image_size=256, patch_size=4, mask_ratio = 0.60):
        super().__init__( learning_rate, num_channels, num_workers, train_dataset, val_dataset, batch_size, training_epochs)
        
        
       
        
        num_patches = (image_size // patch_size) ** 2
        
        
        #spectral adapter 
        
        #need to verify the spectral adapter for 1d convolutions to match paper, this is just a temp one right now 
        self.spectral_adapter = SpectralAdapter_new(num_channels)
        
        # swin2 model 

        # num_channels is 128 as output from spectral adapter 
        
        # configuration = Swinv2Config(
        #     num_channels=128,
        #     patch_size=patch_size,
        #     image_size=image_size,
        #     embed_dim=192,
        #     depths=[2, 2, 18, 2],
        #     num_heads=[6, 12, 24, 48],
        #     window_size=12,
        #     pretrained_window_sizes=[0,0,0,0],
        #     mlp_ratio=4.0,
        #     attention_probs_dropout_prob=0.0,
        #     drop_path_rate=0.1, 
        #     encoder_stride=32,
        #     hidden_act="gelu",
        #     hidden_dropout_prob=0.0,
        #     hidden_size=1536,
        #     initializer_range=0.02,
        #     layer_norm_eps=1e-05,
        #     num_layers =4,
        #     out_features=["stage1", "stage2", "stage3", "stage4"],
        #     out_indices=[1, 2, 3, 4],
        #     path_norm = True,
        #     qkv_bias = True,
        #     stage_names=["stem","stage1", "stage2", "stage3", "stage4"],
        #     torch_dtype="float32",
        #     use_absolute_embeddings=False
                        
        # )
        
        
        # configuration = {
        #     "backbone": "microsoft/swinv2-large-patch4-window12-192-22k",
        #     "use_pretrained_backbone": True,
        #     "num_channels": 128,
        #     "image_size": image_size,
        #     "patch_size": patch_size,
        # }
        
        # configuration = AutoConfig.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k", id2label={0: "background", 1: "object"}, label2id={"background": 0, "object": 1}, num_channels = 128, image_size = image_size)
        
        # # self.backbone = Swinv2Model(configuration)
        # # print(self.backbone)
        # self.backbone = Swinv2ForMaskedImageModeling(configuration)
        # # self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k", id2label={0: "background", 1: "object"}, label2id={"background": 0, "object": 1})
        
        # # print(self.backbone.config)
        # # print(self.backbone.swinv2.embeddings)
        # # add decode head 
        
        # # already in the model
        
        # # decoder used for SimMIM
        
        
        # use pretrained model instead of just the config for it 
        
        # Load pre-trained SwinV2 model
        self.backbone = Swinv2ForMaskedImageModeling.from_pretrained(
            "microsoft/swinv2-large-patch4-window12-192-22k",
            ignore_mismatched_sizes=True, id2label={0: "background", 1: "object"}, label2id={"background": 0, "object": 1}, image_size=image_size, num_channels=128
        )
        
        
        # print(self.backbone.swinv2.embeddings)
        # sys.exit()
        # Adjust the input channels if necessary
        if self.backbone.config.num_channels != 128:
            self.backbone.swinv2.embeddings.patch_embeddings.projection = nn.Conv2d(
                128,
                self.backbone.config.embed_dim,
                kernel_size=self.backbone.config.patch_size,
                stride=self.backbone.config.patch_size
            )
            self.backbone.config.num_channels = 128
        
        # print(self.backbone.swinv2.embeddings.mask_token)
        # print(self.backbone.config)
        # sys.exit()
        
    def forward(self, hsi_img, mask):
        
        # print(hsi_img.shape)
        
        # Reshape the input tensor to (batch_size, channels, height * width)
        # batch_size, channels, height, width = hsi_img.shape
        # hsi_img = hsi_img.view(batch_size , height * width, channels)
    
        # print(hsi_img.shape)
        feature_img = self.spectral_adapter(hsi_img)
        # print(feature_img.shape)
        # Reshape the output back to (batch_size, new_channels, height, width)
        # x = x.view(batch_size, x.size(1), height, width)
        # print(x.shape)

        x = self.backbone(feature_img, bool_masked_pos = mask)
        # print(x.reconstruction.shape)
        # print(x.loss)
       
        
        # print(x.last_hidden_state.shape, hsi_img.shape)
        # print(x.reconstruction.shape, feature_img.shape)
        
        # return x.reconstruction, feature_img
        return x.loss

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/README.md 
# use simmim
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mim_no_trainer.py


# test out loading hsi image 
datset_dir = "output"
batch_size = 8
accumulate_grad_batches = 32 # want batch size to be 256
num_workers = 4 
initial_lr =  1e-3 #swin2 paper used 1e-3, convnextv2 1.5e-4 # 8e-4 simmim paper
grad_clip_val = 5 
# Define the split ratio
train_ratio = 0.85 # % used for training
# val_ratio = 0.3 % not used 
mask_ratio = 0.60 # recommended .90 used for mae, 0.6 simmim (standard for rgb)

max_epochs = 200
patch_size = 4

full_dataset = fractal_dataset(root_dir=datset_dir)
test_img = full_dataset[0]
# print(test_img.shape, test_img.dtype)
# # plot a layer of the image
# plt.imshow(test_img[50])
# plt.show()
num_channels = test_img.shape[0]
img_height = 256
img_width = 256

mask_generator = MaskGenerator(input_size=img_height, mask_patch_size=32, model_patch_size=patch_size, mask_ratio=mask_ratio)

torch.cuda.empty_cache()

test_transform = A.Compose([
    A.Resize(width=img_width, height=img_height), 
    A.Normalize(normalization="image", max_pixel_value=255.0)
])

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),  
    A.Resize(width=img_width, height=img_height), 
    A.Normalize(normalization="image", max_pixel_value=255.0),
    A.ChannelDropout(channel_drop_range=(1, 10), fill_value=0, p=0.5)
])

# Calculate the lengths for each split
train_len = int(len(full_dataset) * train_ratio)
val_len = len(full_dataset) - train_len

# Perform the split
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

# Apply the transformations to the respective datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform


# test = val_dataset.__getitem__(0)
# print(test.shape, test.dtype, test.min(), test.max())

# verify dataloader 
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers, drop_last=True)

# for batch in val_loader:
#     print(batch["hsi_pixel_values"].shape)


# # Print the sizes of the datasets
# print(f'Train dataset size: {len(train_dataset)}')
# print(f'Validation dataset size: {len(val_dataset)}')

# model = convnext2_model(learning_rate=initial_lr, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, batch_size=batch_size, image_size=img_height, training_epochs=max_epochs, patch_size=patch_size)

model = swin2_model(learning_rate=initial_lr, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, batch_size=batch_size, image_size=img_height, training_epochs=max_epochs, patch_size=patch_size)
# model = swin2_model.load_from_checkpoint('lightning_logs/version_18/checkpoints/lowest_val_loss_hsi.ckpt')

checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")
# Set the float32 matmul precision to 'medium' or 'high'
torch.set_float32_matmul_precision('medium')

trainer = L.Trainer(
    max_epochs=max_epochs,
    accumulate_grad_batches=accumulate_grad_batches, 
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=20), 
        checkpoint_callback_val_loss ], 
    accelerator="gpu", 
    devices="auto", 
    gradient_clip_val=grad_clip_val, 
    precision="16-mixed" ) # 


model.hparams.learning_rate = initial_lr  # learning_rate
model.hparams.batch_size = batch_size


# sample_hsi_img = torch.rand(batch_size, num_channels, img_height, img_width)#.to("cuda")
# sample_mask = torch.stack([mask_generator() for i in range(batch_size)], dim=0)#.to("cuda")
# # print( sample_mask.shape)
# output = model.forward(sample_hsi_img, sample_mask)


# sys.exit()

trainer.fit(model)

# # # load the model and save as huggingface model
model = swin2_model.load_from_checkpoint('lightning_logs/version_19/checkpoints/lowest_val_loss_hsi.ckpt')

# # only save backbone encoder
# print(model.spectral_adapter, model.backbone)
# backbone = model.backbone
torch.save(model.state_dict(), "swinv2_config/full_model.pth")
model.backbone.save_pretrained("swinv2_config")
print("model saved")