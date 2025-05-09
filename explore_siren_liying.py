# %% [markdown]
# # SIREN (Implicit Neural Representations with Periodic Activation Functions)
# 
# This notebook explores the SIREN architecture for learning implicit neural representations of images.
# SIREN uses periodic activation functions (sine) to represent signals with fine details and their derivatives.
# 
# Original paper: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

# %% [markdown]
# ## Imports and Setup
# 
# Import necessary libraries and set up device (CPU/GPU) for computation.

# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from collections import OrderedDict

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2
import time

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## Utility Functions
# 
# Helper functions for coordinate grid generation and image processing.

# %%
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    Args:
        sidelen: int - side length of the grid
        dim: int - dimensionality of the grid
    Returns:
        torch.Tensor: Flattened grid coordinates
    '''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_mgrid_2d(height, width, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    Args:
        width: int - number of points along x-axis
        height: int - number of points along y-axis
        dim: int - dimensionality of the grid
    Returns:
        torch.Tensor: Flattened grid coordinates
    '''
    if dim != 2:
        raise ValueError("This function currently supports only 2D grids.")

    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)

    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

# %% [markdown]
# ## SIREN Architecture
# 
# Implementation of the SIREN network with periodic activation functions.

# %%
class SineLayer(nn.Module):
    """Sine activation layer with frequency scaling.
    
    If is_first=True, omega_0 is a frequency factor which multiplies the activations before the
    nonlinearity. Different signals may require different omega_0 in the first layer.
    
    If is_first=False, the weights are divided by omega_0 to keep the magnitude of
    activations constant but boost gradients to the weight matrix.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    """SIREN network architecture.
    
    Args:
        in_features: int - input dimension (e.g., 2 for 2D coordinates)
        hidden_features: int - number of hidden units
        hidden_layers: int - number of hidden layers
        out_features: int - output dimension (e.g., 3 for RGB)
        outermost_linear: bool - whether to use linear activation in the last layer
        first_omega_0: float - frequency factor for first layer
        hidden_omega_0: float - frequency factor for hidden layers
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                    is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns model output and intermediate activations for visualization."""
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

# %% [markdown]
# ## Differential Operators
# 
# Implementation of differential operators using autograd for computing gradients and laplacian.

# %%
def laplace(y, x):
    """Compute the laplacian of y with respect to x."""
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    """Compute the divergence of y with respect to x."""
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    """Compute the gradient of y with respect to x."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

# %% [markdown]
# ## Image Processing Utilities
# 
# Functions for loading and processing images.

# %%
def get_cameraman_tensor(sidelength):
    """Load and process the cameraman image."""
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def get_image_tensor(image_path, H, W):
    """Load and process an image from file."""
    img = Image.open(image_path)
    transform = Compose([
        Resize((H, W)),
        ToTensor(),
    ])
    img = transform(img)
    img = img * 2. - 1.
    return img

# %% [markdown]
# ## Image Fitting Dataset
# 
# Dataset class for fitting images with SIREN.

# %%
class ImageFitting(Dataset):
    """Dataset for fitting images with SIREN.
    
    Args:
        sidelength: int - size of the image
    """
    def __init__(self, sidelength):
        super().__init__()
        image_path = 'data/DIV2K_train_HR/0001.png'
        
        # Load image to get dimensions
        img = Image.open(image_path)
        # Resize to target size
        img = img.resize((sidelength, sidelength), Image.LANCZOS)
        self.H, self.W = img.size
        
        # Process image
        img = get_image_tensor(image_path, self.H, self.W)
        self.pixels = img.permute(1, 2, 0).contiguous().view(-1, 3)
        self.coords = get_mgrid_2d(self.H, self.W, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels

# %% [markdown]
# ## Training Setup
# 
# Initialize the model and training parameters.

# %%
img_size = 512
cameraman = ImageFitting(img_size)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)
img_siren.to(device)

total_steps = 100
steps_til_summary = 25
optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

# %% [markdown]
# ## Training Loop
# 
# Train the SIREN model to fit the image.

# %%
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.to(device), ground_truth.to(device)

for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    loss = ((model_output - ground_truth)**2).mean()

    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        with torch.no_grad():
            output_view = model_output.view(cameraman.H, cameraman.W, 3)
            output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
            output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
            plt.imshow(output_view)
            plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

# %% [markdown]
# ## Testing and Visualization
# 
# Test the trained model with different resolutions and save results.

# %%
save_folder = 'siren_fitting_results'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Test with original resolution
with torch.no_grad():
    coords = get_mgrid_2d(cameraman.H, cameraman.W, 2).unsqueeze(0).to(device)
    model_output, _ = img_siren(coords)
    output_view = model_output.view(cameraman.H, cameraman.W, 3)
    output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
    output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
    plt.imshow(output_view)
    plt.show()
    cv2.imwrite(os.path.join(save_folder, 'org_size.png'), output_view[:, :, ::-1])

# Test with double resolution
with torch.no_grad():
    target_H = cameraman.H * 2
    target_W = cameraman.W * 2
    coords = get_mgrid_2d(target_H, target_W, 2).unsqueeze(0).to(device)
    model_output, _ = img_siren(coords)
    output_view = model_output.view(target_H, target_W, 3)
    output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
    output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
    plt.imshow(output_view)
    plt.show()
    cv2.imwrite(os.path.join(save_folder, 'double_size.png'), output_view[:, :, ::-1])

# Clean up memory
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Test with 2.5x resolution
with torch.no_grad():
    target_H = int(cameraman.H * 2.5)
    target_W = int(cameraman.W * 2.5)
    coords = get_mgrid_2d(target_H, target_W, 2).unsqueeze(0).to(device)
    model_output, _ = img_siren(coords)
    output_view = model_output.view(target_H, target_W, 3)
    output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
    output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
    plt.imshow(output_view)
    plt.show()
    cv2.imwrite(os.path.join(save_folder, 'HW2.5_size.png'), output_view[:, :, ::-1])
    del output_view

# %% [markdown]
# ## Patch-based Image Fitting
# 
# Implementation of patch-based image fitting using SIREN with neighborhood conditioning.

# %%
class SingleImagePatchDataset(Dataset):
    """Dataset for patch-based image fitting.
    
    Args:
        image_path: str - path to the image
        img_size: tuple - target image size
        patch_size: int - size of the patch
    """
    def __init__(self, image_path, img_size=(256, 256), patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

        # Load and process image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        self.img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        self.img_tensor = self.img_tensor.permute(2, 0, 1)

        # Create coordinates grid
        self.coords = get_mgrid_2d(img_size[1], img_size[0])

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        c, h, w = self.img_tensor.shape
        y = idx // w
        x = idx % w

        padded_img = F.pad(self.img_tensor.unsqueeze(0),
                          (self.pad_size,)*4, mode='reflect')[0]
        patch = padded_img[:, y:y+self.patch_size, x:x+self.patch_size]
        patch = patch.reshape(3, -1)
        patch = torch.cat([patch[:, :4], patch[:, 5:]], dim=1)
        patch = patch.flatten()

        return {
            'coord': self.coords[idx],
            'patch': patch,
            'target': self.img_tensor[:, y, x]
        }

class ConditionedSiren(nn.Module):
    """SIREN network with patch conditioning.
    
    Args:
        patch_size: int - size of the patch
        hidden_dim: int - number of hidden units
    """
    def __init__(self, patch_size=3, hidden_dim=256):
        super().__init__()
        # Calculate input dimension: 2 for coordinates + 3*(patch_size^2 - 1) for patch (excluding center)
        input_dim = 2 + 3 * (patch_size * patch_size - 1)  # 2 for coord + RGB patch (excluding center)

        self.net = nn.Sequential(
            SineLayer(input_dim, hidden_dim, is_first=True),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, coord, patch):
        x = torch.cat([coord, patch], dim=-1)
        return self.net(x)

# %% [markdown]
# ## Patch-based Training
# 
# Training loop for patch-based image fitting.

# %%
# Training setup
dataset = SingleImagePatchDataset("data/DIV2K_train_HR/0001.png")
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = ConditionedSiren().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

visualize_every = 5
pad = dataset.pad_size

# Get original image for comparison
original_img = (dataset.img_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2

# Create figure
plt.figure(figsize=(12, 6))

# Training loop
for epoch in range(5):
    for batch in dataloader:
        coords = batch['coord'].to(device)
        patches = batch['patch'].to(device)
        targets = batch['target'].to(device)

        preds = model(coords, patches)
        loss = F.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % visualize_every == 0:
        model.eval()
        with torch.no_grad():
            all_coords = dataset.coords.to(device)
            
            # Process patches to match dataset format
            padded_img = F.pad(dataset.img_tensor.unsqueeze(0), (pad,)*4, mode='reflect')[0]
            patches = []
            for y in range(dataset.img_tensor.shape[1]):
                for x in range(dataset.img_tensor.shape[2]):
                    patch = padded_img[:, y:y+dataset.patch_size, x:x+dataset.patch_size]
                    patch = patch.reshape(3, -1)
                    patch = torch.cat([patch[:, :4], patch[:, 5:]], dim=1)  # Remove center pixel
                    patch = patch.flatten()
                    patches.append(patch)
            patches = torch.stack(patches).to(device)
            
            preds = model(all_coords, patches)
            reconstructed = preds.cpu().numpy().reshape(dataset.img_tensor.shape[1],
                                                      dataset.img_tensor.shape[2], 3)
            reconstructed = (reconstructed + 1) / 2

            plt.subplot(1, 2, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed.clip(0, 1))
            plt.title(f"Epoch {epoch}")
            plt.axis('off')

            plt.tight_layout()
            plt.pause(0.01)
            plt.draw()

        model.train()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# %% [markdown]
# ## Multi-Image Dataset
# 
# Dataset class for handling multiple images with patch-based fitting.

# %%
class MultiImagePatchDataset(Dataset):
    """Dataset for patch-based fitting of multiple images.
    
    Args:
        image_paths: list - paths to the images
        img_size: tuple - target image size
        patch_size: int - size of the patch
        is_train: bool - whether this is training data
    """
    def __init__(self, image_paths, img_size=(256, 256), patch_size=3, is_train=True):
        super().__init__()
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.img_size = img_size
        self.is_train = is_train
        
        # Load and process all images
        self.images = []
        self.coords_list = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize(img_size)
            img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1)
            self.images.append(img_tensor)
            
            # Create coordinates grid for this image
            coords = get_mgrid_2d(img_size[1], img_size[0])
            self.coords_list.append(coords)
        
        # Calculate total number of patches
        self.total_patches = len(image_paths) * img_size[0] * img_size[1]
        
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        # Determine which image and which patch within that image
        img_idx = idx // (self.img_size[0] * self.img_size[1])
        patch_idx = idx % (self.img_size[0] * self.img_size[1])
        
        img_tensor = self.images[img_idx]
        coords = self.coords_list[img_idx]
        
        # Get patch coordinates
        y = patch_idx // self.img_size[1]
        x = patch_idx % self.img_size[1]
        
        # Extract patch with reflection padding
        padded_img = F.pad(img_tensor.unsqueeze(0),
                          (self.pad_size,)*4, mode='reflect')[0]
        patch = padded_img[:, y:y+self.patch_size, x:x+self.patch_size]
        patch = patch.reshape(3, -1)
        patch = torch.cat([patch[:, :4], patch[:, 5:]], dim=1)  # Remove center pixel
        patch = patch.flatten()
        
        return {
            'coord': coords[patch_idx],
            'patch': patch,
            'target': img_tensor[:, y, x],
            'img_idx': img_idx
        }

# %% [markdown]
# ## Multi-Image Training
# 
# Training loop for fitting multiple images with a single network.

# %%
# Configuration parameters
config = {
    'image_size': (64, 64),    # Even smaller image size
    'patch_size': 3,
    'batch_size': 64,          # Much smaller batch size
    'num_workers': 0,          # No workers
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'visualize_every': 1,
    'dataset_percentage': 0.05,  # Use even less data
    'hidden_dim': 64,          # Smaller network
    'num_hidden_layers': 2,
    'save_dir': 'siren_multi_fitting_results'
}

# Get all image paths
train_path = 'data/DIV2K_train_HR'
valid_path = 'data/DIV2K_valid_HR'

train_images = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.png')]
valid_images = [os.path.join(valid_path, f) for f in os.listdir(valid_path) if f.endswith('.png')]

# Use only a percentage of the dataset
train_images = train_images[:int(len(train_images) * config['dataset_percentage'])]
valid_images = valid_images[:int(len(valid_images) * config['dataset_percentage'])]

print(f"Using {len(train_images)} training images and {len(valid_images)} validation images")

# Create datasets with smaller image size
train_dataset = MultiImagePatchDataset(
    train_images, 
    img_size=config['image_size'], 
    patch_size=config['patch_size'], 
    is_train=True
)
valid_dataset = MultiImagePatchDataset(
    valid_images, 
    img_size=config['image_size'], 
    patch_size=config['patch_size'], 
    is_train=False
)

# Create dataloaders with smaller batch size
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=config['num_workers'],
    pin_memory=True
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'],
    pin_memory=True
)

# Initialize model with smaller architecture
model = ConditionedSiren(
    patch_size=config['patch_size'],
    hidden_dim=config['hidden_dim']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Create save directory
os.makedirs(config['save_dir'], exist_ok=True)

# Save configuration
with open(os.path.join(config['save_dir'], 'config.txt'), 'w') as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

# Training loop
best_val_loss = float('inf')
for epoch in range(config['num_epochs']):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        coords = batch['coord'].to(device)
        patches = batch['patch'].to(device)
        targets = batch['target'].to(device)
        
        preds = model(coords, patches)
        loss = F.mse_loss(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            coords = batch['coord'].to(device)
            patches = batch['patch'].to(device)
            targets = batch['target'].to(device)
            
            preds = model(coords, patches)
            loss = F.mse_loss(preds, targets)
            val_loss += loss.item()
    
    val_loss /= len(valid_dataloader)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, os.path.join(config['save_dir'], 'best_model.pth'))
    
    # Save visualization results
    if epoch % config['visualize_every'] == 0:
        model.eval()
        with torch.no_grad():
            # Process one validation image at a time
            for img_idx in range(min(1, len(valid_images))):  # Only process one image
                # Get all patches for this image
                img_patches = []
                img_coords = []
                for batch in valid_dataloader:
                    mask = batch['img_idx'] == img_idx
                    if mask.any():
                        img_patches.append(batch['patch'][mask])
                        img_coords.append(batch['coord'][mask])
                
                if img_patches:
                    img_patches = torch.cat(img_patches).to(device)
                    img_coords = torch.cat(img_coords).to(device)
                    
                    # Reconstruct image
                    preds = model(img_coords, img_patches)
                    reconstructed = preds.cpu().numpy().reshape(*config['image_size'], 3)
                    reconstructed = (reconstructed + 1) / 2
                    
                    # Save original and reconstructed images separately
                    original_img = (valid_dataset.images[img_idx].permute(1, 2, 0).cpu().numpy() + 1) / 2
                    plt.imsave(os.path.join(config['save_dir'], f'original_epoch_{epoch+1}.png'), original_img)
                    plt.imsave(os.path.join(config['save_dir'], f'reconstructed_epoch_{epoch+1}.png'), reconstructed.clip(0, 1))
                    
                    # Clear memory
                    del img_patches, img_coords, preds, reconstructed, original_img
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()

# %%
