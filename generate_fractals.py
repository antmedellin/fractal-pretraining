
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from fractal_learning.fractals import ifs, diamondsquare
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tifffile as tiff

# Add Gaussian noise
def add_gaussian_noise(image, mean=0.0, std=0.03):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create a 1D Gaussian kernel
        kernel = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
        kernel = kernel / kernel.sum()

        # Reshape to depthwise convolutional weight
        self.register_buffer('weight', kernel.view(1, 1, -1).repeat(channels, 1, 1))

    def forward(self, x):
        if x.size(2) > self.kernel_size:
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
            x = F.conv1d(x, self.weight, groups=self.channels)
        return x

# Define the decoder model
class Decoder(nn.Module):
    def __init__(self, num_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(3, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, num_channels)
        self.smoothing = GaussianSmoothing(num_channels, kernel_size=20, sigma=10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.unsqueeze(1)  # Add channel dimension for convolution
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_channels, length)
        x = self.smoothing(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, length, num_channels)
        x = x.squeeze(1)  # Remove channel dimension after smoothing
        return x
    
def normalize_to_latent_space(image, latent_min, latent_max):
    # Normalize the image to the range [0, 1]
    image_min = image.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
    image_max = image.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
    normalized_image = (image - image_min) / (image_max - image_min + 1e-5)  # Adding a small value to avoid division by zero
    
    # Scale and shift the normalized image to the range defined by latent_min and latent_max
    latent_range = latent_max - latent_min
    scaled_image = normalized_image * latent_range.unsqueeze(1).unsqueeze(2) + latent_min.unsqueeze(1).unsqueeze(2)
    
    return scaled_image

   
# Generate 10,000 random spectral curves
num_curves = 10000
num_points = 204  # Number of points in each spectral curve
sigma = 6  # Standard deviation for Gaussian kernel
num_curves_to_plot = 5
num_epochs = 1000


system = ifs.sample_system()
points = ifs.iterate(system, 100000)

# render images in binary, grayscale, and color
binary_image = ifs.render(points, binary=True)
gray_image = ifs.render(points, binary=False)
color_image = ifs.colorize(gray_image)

# create a random colored background
background = diamondsquare.colorized_ds()

# create a composite image
composite = background.copy()
composite[gray_image.nonzero()] = color_image[gray_image.nonzero()]


# Assuming composite is a NumPy array of shape (H, W, C)
composite_tensor = torch.tensor(composite.transpose(2, 0, 1), dtype=torch.float32) / 255.0



# Add Gaussian noise
noisy_image = add_gaussian_noise(composite_tensor)

# # Convert back to NumPy array for visualization
# noisy_image = noisy_image.permute(1, 2, 0).numpy()

noisy_image_np = noisy_image.permute(1, 2, 0).numpy()

# plt.figure()
# plt.imshow(composite)
# plt.title('Composite Image')

# plt.figure()
# plt.imshow(noisy_image_np)
# plt.title('Noisy Image')

# plt.show()
# sys.exit()





# Generate random spectral curves
spectral_curves = np.random.rand(num_curves, num_points)

# convert to dtype float16
spectral_curves = spectral_curves.astype(np.float32)

# Apply Gaussian filter to smooth the curves

smoothed_curves = gaussian_filter1d(spectral_curves, sigma=sigma, axis=1)



# Assuming smoothed_curves is a NumPy array of shape (num_curves, num_points)

# Select 5 random indices
indices = np.random.choice(smoothed_curves.shape[0], num_curves_to_plot, replace=False)

# # Plot the selected curves
# plt.figure(figsize=(10, 6))
# for idx in indices:
#     plt.plot(smoothed_curves[idx], label=f'Curve {idx}')

# plt.xlabel('Point Index')
# plt.ylabel('Value')
# plt.title('Smoothed Spectral Curves')
# plt.legend()
# plt.show()


# sys.exit()





# Use PCA to reduce the dimensionality to 2D (similar to a latent space)
pca = PCA(n_components=3)
latent_space = pca.fit_transform(smoothed_curves)



decoder = Decoder(num_points)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(decoder.parameters())

# Convert data to PyTorch tensors
latent_space_tensor = torch.tensor(latent_space, dtype=torch.float32)
smoothed_curves_tensor = torch.tensor(smoothed_curves, dtype=torch.float32)

# Train the decoder

batch_size = num_curves
num_batches = len(latent_space_tensor) // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_latent_space = latent_space_tensor[start_idx:end_idx]
        batch_smoothed_curves = smoothed_curves_tensor[start_idx:end_idx]

        # Forward pass
        # print(batch_latent_space.shape)
        outputs = decoder(batch_latent_space)
        loss = criterion(outputs, batch_smoothed_curves)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate new points in the latent space
new_points = np.random.uniform(low=latent_space.min(axis=0), high=latent_space.max(axis=0), size=(num_curves_to_plot, 3))
new_points_tensor = torch.tensor(new_points, dtype=torch.float32)

# Decode the new points to generate spectral curves
generated_curves_tensor = decoder(new_points_tensor)
generated_curves = generated_curves_tensor.detach().numpy()

# sigma_new =2.5
# generated_curves = gaussian_filter1d(generated_curves, sigma=sigma_new, axis=1)


# # Plot the generated spectral curves
# plt.figure(figsize=(10, 8))
# for curve in generated_curves:
#     plt.plot(curve)
# plt.xlabel('Wavelength')
# plt.ylabel('Intensity')
# plt.title('Generated Spectral Curves')
# plt.show()

# print the max and min values of the latent space in each dimension
latent_min = latent_space.min(axis=0)
latent_max = latent_space.max(axis=0)
latent_min = torch.tensor(latent_min)
latent_max = torch.tensor(latent_max)

# print('latent space max and min values')
# print(latent_min)
# print(latent_max)

#  normalize the noisy image to latent space for each dimension


# print max and min of noisy image before normalization
# print('Noisy image shape:', noisy_image.shape)

# Compute max and min values for each channel
channel_max = noisy_image.view(noisy_image.size(0), -1).max(dim=1).values
channel_min = noisy_image.view(noisy_image.size(0), -1).min(dim=1).values

# Print max and min values for each channel
# for i in range(noisy_image.size(0)):  # Iterate over channels
#     print(f'Channel {i}: max = {channel_max[i].item()}, min = {channel_min[i].item()}')
    
# Normalize the noisy image to the latent space
normalized_image = normalize_to_latent_space(noisy_image, latent_min, latent_max)

# print('Normalized image shape:', normalized_image.shape)

# Compute max and min values for each channel
channel_max = normalized_image.view(normalized_image.size(0), -1).max(dim=1).values
channel_min = normalized_image.view(normalized_image.size(0), -1).min(dim=1).values

# Print max and min values for each channel
# for i in range(normalized_image.size(0)):  # Iterate over channels
#     print(f'Channel {i}: max = {channel_max[i].item()}, min = {channel_min[i].item()}')

# pass each value into the decoder to generate the spectral curve
# Decode the normalized image to generate spectral curves
# Flatten the normalized image tensor
flattened_image = normalized_image.view(normalized_image.size(0), -1)
flattened_image = flattened_image.permute(1, 0)

# print('flattened_image shape:', flattened_image.shape)


# Pass the flattened image into the decoder to generate the spectral curves
generated_curves_tensor = decoder(flattened_image)

# print('Generated curves tensor shape:', generated_curves_tensor.shape)

# convert back to the right shape 
generated_curves = generated_curves_tensor.permute(1, 0).detach().numpy()
generated_curves = generated_curves.reshape(num_points, noisy_image.shape[1], noisy_image.shape[2])

# print('generated curves shape:', generated_curves.shape)

# plot the image  take 3 channels to plot
# plt.figure(figsize=(10, 6))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(generated_curves[i*20], cmap='gray')
#     plt.axis('off')
#     plt.title(f'Channel {i*20}')
# plt.show()

# save the hyperspectral image to a file

# normalize the generated curves to the range [0, 1] for the image as a whole 
generated_curves = (generated_curves - generated_curves.min()) / (generated_curves.max() - generated_curves.min())

# print(generated_curves.shape[0])

# Print max and min values for each channel
# for i in range(generated_curves.shape[0]):  # Iterate over channels
#     print(f'Channel {i}: max = {generated_curves[i].max()}, min = {generated_curves[i].min()}')



# save image 
# Save the hyperspectral image to a TIFF file
# Convert the normalized image to 16-bit
generated_curves_8bit = (generated_curves * 256).astype(np.uint8)

# Save the hyperspectral image to a TIFF file
tiff.imwrite('generated_curves_8bit.tiff', generated_curves_8bit)

print('Multi-channel TIFF image saved as generated_curves.tiff')
