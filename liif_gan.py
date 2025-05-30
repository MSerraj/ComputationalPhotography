import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Device setup - works for CUDA, MPS, and CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
        
device = get_device()
print(f"Using device: {device}")

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#------------------------------------------------------------------------------
# Coordinate-based Utilities
#------------------------------------------------------------------------------

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_mgrid_2d(height, width, dim=2):
    '''Generates a flattened grid of (x,y) coordinates in a range of -1 to 1.'''
    if dim != 2:
        raise ValueError("This function currently supports only 2D grids.")
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    mgrid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

#------------------------------------------------------------------------------
# SIREN Architecture with Patch Conditioning
#------------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Sine activation layer with frequency scaling."""
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

class LocalFeatureExtractor(nn.Module):
    """Local feature extractor for patch-based conditioning"""
    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # Further reduce spatial dimensions
        return x

class PatchConditionedSiren(nn.Module):
    """SIREN network with patch conditioning for super-resolution"""
    def __init__(self, 
                coord_dim=2, 
                patch_feature_dim=64, 
                hidden_features=256, 
                hidden_layers=3, 
                out_features=3,
                first_omega_0=30, 
                hidden_omega_0=30.):
        super().__init__()
        
        # Feature extractor for patch conditioning
        self.feature_extractor = LocalFeatureExtractor(in_channels=3, feature_dim=patch_feature_dim)
        
        # SIREN network with dynamic sizing of first layer
        self.coord_dim = coord_dim
        self.patch_feature_dim = patch_feature_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        # Create the network immediately with known input size
        # Input is coordinates (2) + patch features (64)
        self.net = self._create_network(coord_dim + patch_feature_dim)
    
    def _create_network(self, input_size):
        """Create the network with correct input size determined at runtime"""
        layers = []
        
        # First layer
        layers.append(SineLayer(input_size, self.hidden_features,
                              is_first=True, omega_0=self.first_omega_0))
        
        # Hidden layers
        for i in range(self.hidden_layers):
            layers.append(SineLayer(self.hidden_features, self.hidden_features,
                                  is_first=False, omega_0=self.hidden_omega_0))
        
        # Output layer
        final_linear = nn.Linear(self.hidden_features, self.out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                                        np.sqrt(6 / self.hidden_features) / self.hidden_omega_0)
        layers.append(final_linear)
        
        network = nn.Sequential(*layers)
        # Ensure network is on the same device as the model
        network = network.to(self.feature_extractor.conv1.weight.device)
        return network
        
    def forward(self, coords, patch):
        """
        Forward pass through the network.
        
        Args:
            coords: Coordinates of shape [batch_size, num_points, 2]
            patch: Input image patch of shape [batch_size, 3, h, w]
            
        Returns:
            RGB values for each coordinate
        """
        # Extract features from the patch
        patch_features = self.feature_extractor(patch)  # [batch_size, feature_dim, h', w']
        batch_size = coords.shape[0]
        
        # Global average pool to get a single feature vector per channel
        patch_features = F.adaptive_avg_pool2d(patch_features, 1)  # [batch_size, feature_dim, 1, 1]
        patch_features = patch_features.squeeze(-1).squeeze(-1)  # [batch_size, feature_dim]
        
        # Expand patch features to match number of coordinates
        num_points = coords.shape[1]
        expanded_features = patch_features.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, num_points, features]
        
        # Ensure expanded_features has the same batch size as coords
        if expanded_features.shape[0] != coords.shape[0]:
            expanded_features = expanded_features.expand(coords.shape[0], -1, -1)
        
        # Combine coordinates and features
        combined = torch.cat([coords, expanded_features], dim=2)  # [batch_size, num_points, coord_dim + features]
        
        # Process through network
        output = self.net(combined.reshape(-1, combined.shape[-1]))  # [batch_size * num_points, out_features]
        output = output.reshape(batch_size, num_points, -1)  # [batch_size, num_points, out_features]
        
        # Apply sigmoid to get values in [0, 1] range
        return torch.sigmoid(output)

#------------------------------------------------------------------------------
# Discriminator Architecture
#------------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Patch-based discriminator for GAN training"""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        # Input layer
        layers.extend(discriminator_block(in_channels, 64, 1, normalize=False))
        # Downsampling
        layers.extend(discriminator_block(64, 64, 2))
        layers.extend(discriminator_block(64, 128, 1))
        layers.extend(discriminator_block(128, 128, 2))
        layers.extend(discriminator_block(128, 256, 1))
        layers.extend(discriminator_block(256, 256, 2))
        layers.extend(discriminator_block(256, 512, 1))
        
        # Output layer
        layers.append(nn.Conv2d(512, 1, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, img):
        return self.model(img)

#------------------------------------------------------------------------------
# Dataset for Patch-Based Training
#------------------------------------------------------------------------------

class PatchBasedDataset(Dataset):
    """Dataset for patch-based training with arbitrary resolution"""
    def __init__(self, image_paths, patch_size=64, scale_factor=4):
        super().__init__()
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.lr_patch_size = patch_size // scale_factor
        
        # Transformations
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Skip images that are too small
        if w < self.patch_size or h < self.patch_size:
            # Get next image
            return self.__getitem__((idx + 1) % len(self.image_paths))
        
        # Random crop for HR patch
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        hr_patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
        
        # Create LR patch by downsampling
        lr_patch = hr_patch.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        
        # Convert to tensors
        hr_tensor = self.to_tensor(hr_patch)
        lr_tensor = self.to_tensor(lr_patch)
        
        # Generate coordinate grid for HR patch
        coords = get_mgrid_2d(self.patch_size, self.patch_size)
        
        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'coords': coords
        }

#------------------------------------------------------------------------------
# Training Functions
#------------------------------------------------------------------------------

def train_liif_gan(generator, discriminator, dataloader, num_epochs=100, 
                   lr=0.0002, betas=(0.5, 0.999), save_interval=10, output_dir='liif_gan_results'):
    """Train the LIIF-GAN model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    # Prepare a fixed batch for visualization
    fixed_batch = next(iter(dataloader))
    fixed_lr = fixed_batch['lr'][:4].to(device)
    fixed_hr = fixed_batch['hr'][:4].to(device)
    fixed_coords = fixed_batch['coords'].to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch in dataloader:
                # Move data to device
                lr_imgs = batch['lr'].to(device)
                hr_imgs = batch['hr'].to(device)
                coords = batch['coords'].to(device)
                batch_size = lr_imgs.size(0)
                
                # Create ground truth labels
                fake_sample = discriminator(hr_imgs)
                valid = torch.ones_like(fake_sample).to(device)
                fake = torch.zeros_like(fake_sample).to(device)
                
                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate SR images
                sr_imgs = generator(coords, lr_imgs)
                # Reshape the output to match the HR image dimensions
                if sr_imgs.dim() == 3:  # If output is [batch_size, num_points, channels]
                    batch_size, num_points, channels = sr_imgs.shape
                    height = width = int(num_points ** 0.5)  # Assuming square images
                    sr_imgs = sr_imgs.reshape(batch_size, height, width, channels)
                    sr_imgs = sr_imgs.permute(0, 3, 1, 2)  # [batch, channels, height, width]
                elif sr_imgs.dim() == 2:  # If output is [batch_size, num_points*channels]
                    # Calculate the expected dimensions
                    batch_size = sr_imgs.shape[0]
                    sr_h, sr_w = hr_imgs.shape[2], hr_imgs.shape[3]
                    sr_imgs = sr_imgs.reshape(batch_size, 3, sr_h, sr_w)
                
                # Adversarial loss
                pred_fake = discriminator(sr_imgs)
                loss_GAN = criterion_GAN(pred_fake, valid)
                
                # Pixel-wise loss
                loss_pixel = criterion_pixel(sr_imgs, hr_imgs)
                
                # Total generator loss
                lambda_pixel = 100  # Weight for pixel loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel
                
                loss_G.backward()
                optimizer_G.step()
                
                # -----------------
                # Train Discriminator
                # -----------------
                optimizer_D.zero_grad()
                
                # Real loss
                pred_real = discriminator(hr_imgs)
                loss_real = criterion_GAN(pred_real, valid)
                
                # Fake loss
                pred_fake = discriminator(sr_imgs.detach())
                loss_fake = criterion_GAN(pred_fake, fake)
                
                # Total discriminator loss
                loss_D = (loss_real + loss_fake) / 2
                
                loss_D.backward()
                optimizer_D.step()
                
                # Update progress bar
                epoch_g_loss += loss_G.item()
                epoch_d_loss += loss_D.item()
                pbar.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
                pbar.update()
            
            # Save model checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_epoch_{epoch+1}.pth")
            
            # Generate and save sample SR images
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    # Process fixed batch
                    fixed_sr = generator(fixed_coords, fixed_lr)
                    # Reshape if needed
                    if fixed_sr.dim() == 3:  # [batch_size, num_points, channels]
                        batch_size, num_points, channels = fixed_sr.shape
                        height = width = int(num_points ** 0.5)  # Assuming square images
                        fixed_sr = fixed_sr.reshape(batch_size, height, width, channels)
                        fixed_sr = fixed_sr.permute(0, 3, 1, 2)  # [batch, channels, height, width]
                    elif fixed_sr.dim() == 2:  # [batch_size, pixels*channels]
                        # Calculate the expected dimensions
                        batch_size = fixed_sr.shape[0]
                        sr_h, sr_w = fixed_hr.shape[2], fixed_hr.shape[3]
                        fixed_sr = fixed_sr.reshape(batch_size, 3, sr_h, sr_w)
                    
                    # Create image grid
                    img_grid = make_grid(
                        torch.cat([
                            F.interpolate(fixed_lr, scale_factor=4, mode='bicubic'),
                            fixed_sr,
                            fixed_hr
                        ], dim=3),
                        nrow=4, normalize=True
                    )
                    
                    # Save grid
                    save_image(img_grid, f"{output_dir}/epoch_{epoch+1}.png")
    
    return generator, discriminator

#------------------------------------------------------------------------------
# Inference Function
#------------------------------------------------------------------------------

def upscale_image(generator, img_path, output_path=None, scale_factor=None):
    """Upscale an image using the trained generator"""
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # Get dimensions
    _, _, h, w = img_tensor.shape
    
    # Set scale factor if not provided
    if scale_factor is None:
        scale_factor = 4
    
    # Target dimensions
    target_h, target_w = h * scale_factor, w * scale_factor
    
    # Generate coordinate grid for target resolution
    coords = get_mgrid_2d(target_h, target_w).unsqueeze(0).to(device)
    
    # Process image patch by patch for memory efficiency
    patch_size = 128
    stride = 96  # Overlap to avoid boundary artifacts
    
    # Initialize output tensor
    output = torch.zeros(1, 3, target_h, target_w).to(device)
    
    # Initialize weight tensor for blending overlapping patches
    weights = torch.zeros(1, 1, target_h, target_w).to(device)
    
    # Process patches
    with torch.no_grad():
        for y in range(0, target_h, stride):
            for x in range(0, target_w, stride):
                # Get patch coordinates
                h_end = min(y + patch_size, target_h)
                w_end = min(x + patch_size, target_w)
                h_start = max(0, h_end - patch_size)
                w_start = max(0, w_end - patch_size)
                
                # Extract patch coordinates
                patch_coords = coords[:, (h_start * target_w + w_start):(h_end * target_w + w_end)]
                
                # Create weight mask (simple linear falloff)
                weight = torch.ones(1, 1, h_end - h_start, w_end - w_start).to(device)
                
                # Generate patch
                patch_result = generator(patch_coords, img_tensor)
                # Reshape output based on dimensions
                if patch_result.dim() == 3:  # [batch, points, 3]
                    patch_result = patch_result.reshape(1, h_end - h_start, w_end - w_start, 3)
                    patch_result = patch_result.permute(0, 3, 1, 2)  # [batch, 3, h, w]
                else:  # patch_result is [batch, points*3]
                    patch_result = patch_result.reshape(1, 3, h_end - h_start, w_end - w_start)
                
                # Add to output with weight
                output[:, :, h_start:h_end, w_start:w_end] += patch_result * weight
                weights[:, :, h_start:h_end, w_start:w_end] += weight
    
    # Normalize by weights
    output = output / (weights + 1e-8)
    
    # Convert to image
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    
    # Save if output path is provided
    if output_path:
        output_img.save(output_path)
    
    return output_img

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def main():
    # Parameters
    data_dir = 'data/DIV2K_train_HR'
    output_dir = 'liif_gan_results'
    patch_size = 128
    batch_size = 16
    num_epochs = 100
    save_interval = 10
    
    # Create image paths list
    image_paths = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
                  if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    
    # Create dataset and dataloader
    dataset = PatchBasedDataset(image_paths, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize generator and discriminator
    generator = PatchConditionedSiren().to(device)
    discriminator = Discriminator().to(device)
    
    # Train model
    generator, discriminator = train_liif_gan(
        generator, 
        discriminator, 
        dataloader, 
        num_epochs=num_epochs, 
        save_interval=save_interval,
        output_dir=output_dir
    )
    
    # Save final models
    torch.save(generator.state_dict(), f"{output_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_final.pth")
    
    # Test upscaling
    test_img_path = 'data/DIV2K_valid_HR/0801.png'
    upscale_image(generator, test_img_path, f"{output_dir}/test_upscaled.png")

if __name__ == "__main__":
    main() 