import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


class ImageDataset(Dataset):
    def __init__(self, coordinates, pixel_values):
        self.coordinates = coordinates.astype(np.float32)
        self.pixel_values = pixel_values.astype(np.float32)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        pixel = self.pixel_values[idx]
        return coord, pixel
    
    
class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class BabySINE(nn.Module):
    def __init__(self, 
                 input_dim=2, 
                 output_dim=3,
                 hidden_dim=256,
                 num_layers=4,
                 omega_0=30,
                 sigma=10.0,
                 use_dropout=False,
                 dropout_rate=0.1):
        super().__init__()
        self.sigma = sigma
        self.omega_0 = omega_0
        
        layers = []
        # Input layer with sine activation
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(Sine(omega_0))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Keep output in [0,1] for RGB
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """SIREN-style weight initialization"""
        with torch.no_grad():
            for idx, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    # Calculate fan_in
                    fan_in = layer.weight.size(1)
                    
                    # First layer initialization
                    if idx == 0:  # First linear layer
                        bound = 1 / fan_in
                    else:
                        # bound = torch.sqrt(torch.tensor(6.0 / fan_in)) / self.omega_0
                        bound = torch.sqrt(torch.tensor(6.0 / fan_in, device=layer.weight.device)) / self.omega_0

                    # Uniform initialization
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, sigma=None):
        """Forward pass with optional sigma scaling"""
        if sigma is None:
            sigma = self.sigma
        # Scale input coordinates by sigma
        x = x * sigma
        return self.network(x)

    def train_model(self, 
                    dataloader, 
                    num_epochs=100, 
                    lr=1e-4, 
                    device=None,
                    criterion=nn.MSELoss(),
                    sigma=10.0):
        # Automatically detect the device (MPS for macOS, CUDA for Windows/Linux, or CPU)
        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        print(f"Using device: {device}")
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        self.to(device)
        
        losses = []
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            
            for batch_coords, batch_pixels in dataloader:
                batch_coords = batch_coords.to(device)
                batch_pixels = batch_pixels.to(device)
                
                # Forward pass with sigma scaling
                pred_pixels = self(batch_coords, sigma)
                
                # Compute loss
                loss = criterion(pred_pixels, batch_pixels)
                total_loss += loss.item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    
def list_png_files(data_folder):
    """List all PNG files in the specified folder."""
    return [f for f in os.listdir(data_folder) if f.endswith(".png")]
    
def load_image(data_folder, img_file_path):
    """Load and preprocess an image."""
    img_path = os.path.join(data_folder, img_file_path)
    img_original = Image.open(img_path)
    img_np_original = np.array(img_original)

    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np_original)
    plt.axis("off")
    plt.title(f"Original Image: {img_file_path}")
    plt.show()

    # Get image dimensions
    height_target, width_target, channels = img_np_original.shape
    print(f"Image dimensions: {height_target}x{width_target}, {channels} channels")

    return img_np_original, height_target, width_target, channels
    
# def pixel_coordinates_normalized(image, downsize_factor): 
#     """Generate normalized coordinates and pixel values for a downsampled image."""
#     print(f"The original image has shape: {image.shape}")
#     x, y = image.shape[:2]
#     resized_image = cv.resize(image, (y // downsize_factor, x // downsize_factor))
#     resized_x, resized_y = resized_image.shape[:2]
#     xs = np.linspace(0, 1, resized_x)  # x coordinates (0 to 1)
#     ys = np.linspace(0, 1, resized_y)  # y coordinates (0 to 1)

#     xx, yy = np.meshgrid(xs, ys, indexing="ij")
#     coordinates = np.stack((xx, yy), axis=-1)
#     coordinates = coordinates.reshape(-1, 2) 
#     resized_image = resized_image / 255.0
#     norm_resized_image = (resized_image - np.mean(resized_image)) / np.std(resized_image)
#     pixel_values = norm_resized_image.reshape(-1, 3)
    
#     return coordinates, pixel_values, norm_resized_image, resized_x, resized_y

def pixel_coordinates_normalized(image, downsize_factor): 
    """Generate normalized coordinates and pixel values for a downsampled image."""
    print(f"The original image has shape: {image.shape}")
    x, y = image.shape[:2]

    # Generate high-resolution coordinates for inference
    xs_hr = np.linspace(-1, 1, x)  # x coordinates (-1 to 1)
    ys_hr = np.linspace(-1, 1, y)  # y coordinates (-1 to 1)
    xx_hr, yy_hr = np.meshgrid(xs_hr, ys_hr, indexing="ij")
    high_res_coordinates = np.stack((xx_hr, yy_hr), axis=-1).reshape(-1, 2)

    # Normalize pixel values for the high-resolution image
    high_res_image = image / 255.0
    high_res_pixel_values = high_res_image.reshape(-1, 3)

    # Downsample the image for training
    resized_image = cv.resize(image, (y // downsize_factor, x // downsize_factor))
    resized_x, resized_y = resized_image.shape[:2]
    print(f"The downsampled image has shape: {resized_image.shape}")

    # Generate low-resolution coordinates for training
    xs_lr = np.linspace(-1, 1, resized_x)  # x coordinates (-1 to 1)
    ys_lr = np.linspace(-1, 1, resized_y)  # y coordinates (-1 to 1)
    xx_lr, yy_lr = np.meshgrid(xs_lr, ys_lr, indexing="ij")
    low_res_coordinates = np.stack((xx_lr, yy_lr), axis=-1).reshape(-1, 2)

    # Normalize pixel values for the low-resolution image
    low_res_image = resized_image / 255.0
    low_res_pixel_values = low_res_image.reshape(-1, 3)

    return (
        low_res_coordinates, 
        low_res_pixel_values, 
        high_res_coordinates, 
        high_res_pixel_values, 
        (resized_x, resized_y), 
        (x, y)
    )
def plot_image(image, title=None):
    """Plot an image with optional title."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{title} Image: ")
    plt.show()

    # Get image dimensions
    height_target, width_target, channels = image.shape
    print(f"Image dimensions: {height_target}x{width_target}, {channels} channels")