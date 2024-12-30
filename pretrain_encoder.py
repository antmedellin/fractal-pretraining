import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import torchmetrics
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler 
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
from transformers import SwinConfig, SwinModel
from transformers import  UperNetConfig, UperNetForSemanticSegmentation,  SwinForMaskedImageModeling
from transformers import AutoConfig
from os.path import join


def collate_fn(inputs):    
    # hyperspectral
    batch = dict()
    batch["hsi_pixel_values"] = torch.stack([i for i in inputs], dim=0)
    batch["mask"] = torch.stack([mask_generator() for i in inputs], dim=0)
    return batch   

    # hsi_pixel_values = torch.stack([item for item in inputs], dim=0)
    # return {"hsi_pixel_values": hsi_pixel_values}

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
    
    
    
    
class enmap_dataset(Dataset):
    def __init__(self,  root_dir,    transform=None):
        
        # image_set # train ,test, validation
        self.transform = transform
      

        self.img_dir =  root_dir
        
        
        
        self.img_names = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.tif'):
                    # self.img_labels.append(os.path.join(root, file))
                    # self.img_names.append(os.path.join(self.img_dir, root.split("/")[-1], file))
                    img_path = os.path.join(root, file)
                    if os.path.exists(img_path):
                        self.img_names.append(img_path)
        
        
        # print(len(self.img_names))
        # sys.exit()
        self.num_images = len( self.img_names  ) 

        # sort img_names and img_labels
        self.img_names.sort()
        

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        
        hsi_name, ext_hsi = os.path.splitext(self.img_names[idx])
        
        
        hsi_path = join(self.img_dir, self.img_names[idx])
        hsi_img = tiff.imread(hsi_path)  #  x,y,channels for albumnetations
        
        
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
    
class swin_model(BaseSegmentationModel):
    def __init__(self,  learning_rate=1.5e-4, num_channels=204, num_workers=4, train_dataset=None, val_dataset=None, batch_size=2, training_epochs=100, image_size=256, patch_size=4, mask_ratio = 0.60, backbone_config ="microsoft_swin_model_not_pretrained" ):
        super().__init__( learning_rate, num_channels, num_workers, train_dataset, val_dataset, batch_size, training_epochs)
        
 

        self.backbone_config = backbone_config
        self.backbone = SwinForMaskedImageModeling.from_pretrained(
            # "microsoft_swin_model",
            # "microsoft_swin_fractal_base",
            # "microsoft/swin-large-patch4-window7-224",
            
            self.backbone_config,
            ignore_mismatched_sizes=True, image_size=image_size, num_channels=num_channels, label2id={}, id2label={}
        )
        
        
                      
        # print(self.backbone.config)
        
        # Adjust the input channels if necessary

        #     self.backbone.swinv2.embeddings.patch_embeddings.projection = nn.Conv2d(
        #         num_channels,
        #         self.backbone.config.embed_dim,
        #         kernel_size=self.backbone.config.patch_size,
        #         stride=self.backbone.config.patch_size
        #     )
        #     self.backbone.config.num_channels = num_channels
        
        # self.backbone.save_pretrained("microsoft_swin_fractal_base")

      
        self.backbone.train()
        
    def forward(self, hsi_img, mask):
        
        # print(hsi_img.shape)
        
        # Reshape the input tensor to (batch_size, channels, height * width)
        # batch_size, channels, height, width = hsi_img.shape
        # hsi_img = hsi_img.view(batch_size , height * width, channels)
    
        # print(hsi_img.shape)
        # feature_img = self.spectral_adapter(hsi_img)
        # print(feature_img.shape)
        # Reshape the output back to (batch_size, new_channels, height, width)
        # x = x.view(batch_size, x.size(1), height, width)
        # print(x.shape)

        x = self.backbone(hsi_img, bool_masked_pos = mask)
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
datset_dir = "output_test"
full_dataset = fractal_dataset(root_dir=datset_dir)

# datset_dir = '/workspaces/enmap/enmap_mini'
# full_dataset = enmap_dataset(root_dir=datset_dir)



batch_size = 16
accumulate_grad_batches = int(256/batch_size) # want batch size to be 256 # need to also factor in number of gpus
num_workers =  os.cpu_count() or 1 
initial_lr =  1e-3 #swin2 paper used 1e-3,  # 8e-4 simmim paper
grad_clip_val = 5 
# Define the split ratio
train_ratio = 0.85 # % used for training
mask_ratio = 0.60 # 0.6 simmim (standard for rgb)

max_epochs =  200
patch_size = 4


test_img = full_dataset[0]
# print(test_img.shape, test_img.dtype)
# # plot a layer of the image
# plt.imshow(test_img[50])
# plt.show()
num_channels = test_img.shape[0]
img_height =224# 256 
img_width = 224#256 # 128

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
print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')


model = swin_model(learning_rate=initial_lr, num_channels= num_channels, num_workers=num_workers,  train_dataset=train_dataset,val_dataset=val_dataset, batch_size=batch_size, image_size=img_height, training_epochs=max_epochs, patch_size=patch_size)
# model = swin_model.load_from_checkpoint('lightning_logs/version_16/checkpoints/lowest_val_loss_hsi.ckpt').to("cuda")

checkpoint_callback_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="lowest_val_loss_hsi")

# Callback to save the model every 10 epochs
checkpoint_callback_every_10_epochs = ModelCheckpoint(
    every_n_epochs=10,
    save_top_k=-1,  # Save all checkpoints
    filename="epoch_{epoch:02d}"
)


# Set the float32 matmul precision to 'medium' or 'high'
torch.set_float32_matmul_precision('medium')

trainer = L.Trainer(
    max_epochs=max_epochs,
    accumulate_grad_batches=accumulate_grad_batches, 
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=20), 
        checkpoint_callback_val_loss, checkpoint_callback_every_10_epochs ], 
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

# # load the model and save as huggingface model
# model = swin_model.load_from_checkpoint('lightning_logs/version_68/checkpoints/lowest_val_loss_hsi.ckpt')

# only save backbone encoder
backbone = model.backbone
model.backbone.save_pretrained("microsoft_swin_fractal_pretrained_224")
print("model saved")
