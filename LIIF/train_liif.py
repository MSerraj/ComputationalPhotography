import os
import argparse
import random
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# Utility function to generate a 2D grid of coordinates
def get_mgrid_2d(height, width, dim=2):
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, dim)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                         np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ImprovedConditionedSiren(nn.Module):
    def __init__(self, patch_size=3, hidden_dim=256):
        super().__init__()
        input_dim = 2 + 3*(patch_size**2)  # 2 coords + patch values

        # Initial layer
        self.input_layer = SineLayer(input_dim, hidden_dim, is_first=True)

        # Multiple SIREN layers with skip connections
        self.layers = nn.ModuleList([
            SineLayer(hidden_dim, hidden_dim) for _ in range(4)
        ])

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Skip connection mixing parameters
        self.skip_weights = nn.Parameter(torch.ones(8))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, coord, patch, patch_size=3):
        x = torch.cat([coord, patch], dim=-1)
        x = self.input_layer(x)

        # Store intermediate outputs for skip connections
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        # Weighted sum of all intermediate outputs
        weights = self.softmax(self.skip_weights)
        x = sum(w * output for w, output in zip(weights, intermediates))
        out = self.output_layer(x)
        
        index_rgb = math.floor(patch_size/2)
        offset = patch_size ** 2
        patch_rgb = torch.cat((patch[:, index_rgb:index_rgb+1], 
                             patch[:, index_rgb+offset:index_rgb+offset+1], 
                             patch[:, index_rgb+2*offset:index_rgb+2*offset+1]), dim=1)

        return out + patch_rgb

class SuperResolutionDatasetCropping(Dataset):
    def __init__(self, hr_image_path, crop_size=(256, 256), scale_factor=8, patch_size=3, upscale_method=Image.BILINEAR, random_crop=True, random_scale_factor=False, min_scale_factor=1.5, max_scale_factor=8.0):
        super().__init__()
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.random_scale_factor = random_scale_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.random_crop = random_crop
        self.upscale_method = upscale_method
        self.hr_image_path = hr_image_path
        
        # Generate random scale factor if enabled
        if random_scale_factor:
            self.scale_factor = random.uniform(min_scale_factor, max_scale_factor)
            print(f"Using random scale factor: {self.scale_factor:.2f}")
        else:
            self.scale_factor = scale_factor
            
        self.crop_size = crop_size
        
        # Store crop parameters for lazy loading
        self.crop_width, self.crop_height = crop_size
        
        # Get the total number of pixels (samples) for this image
        self.total_pixels = self.crop_width * self.crop_height
        
        # We'll load and process the image on first access
        self._hr_tensor = None
        self._lr_tensor = None
        self._coords = None
        self._padded_lr = None
        self._hr_img = None
        self._lr_img = None

    def _load_and_process_image(self):
        """Load and process the image on first access"""
        if self._hr_tensor is not None:
            return  # Already loaded
            
        # Load HR image at full resolution
        hr_img = Image.open(self.hr_image_path).convert('RGB')

        # Get original image dimensions
        orig_width, orig_height = hr_img.size

        # Make sure the image is large enough for cropping
        if orig_width < self.crop_width or orig_height < self.crop_height:
            # If image is too small, resize it to at least the crop size
            scale = max(self.crop_width / orig_width, self.crop_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            hr_img = hr_img.resize((new_width, new_height), Image.BICUBIC)
            orig_width, orig_height = new_width, new_height

        # Crop the HR image
        if self.random_crop:
            # Random crop for training variability
            left = random.randint(0, orig_width - self.crop_width)
            top = random.randint(0, orig_height - self.crop_height)
        else:
            # Center crop for consistent validation
            left = (orig_width - self.crop_width) // 2
            top = (orig_height - self.crop_height) // 2

        hr_img = hr_img.crop((left, top, left + self.crop_width, top + self.crop_height))

        # Calculate LR size (handle fractional scale factors)
        lr_width = int(self.crop_width / self.scale_factor)
        lr_height = int(self.crop_height / self.scale_factor)
        
        # Ensure minimum size of 1 pixel
        lr_width = max(1, lr_width)
        lr_height = max(1, lr_height)

        # Create LR image: downscale first
        small_lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)

        # Then upscale back to original size with specified interpolation to create blurry image
        lr_img = small_lr_img.resize(self.crop_size, self.upscale_method)

        # Convert to tensors and normalize to [-1, 1]
        self._hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        self._lr_tensor = torch.from_numpy(np.array(lr_img)).float() / 127.5 - 1.0

        # Permute to CxHxW
        self._hr_tensor = self._hr_tensor.permute(2, 0, 1)
        self._lr_tensor = self._lr_tensor.permute(2, 0, 1)

        # Create coordinate grid for HR image
        h_hr, w_hr = self._hr_tensor.shape[1], self._hr_tensor.shape[2]
        self._coords = get_mgrid_2d(h_hr, w_hr)

        # Precompute padded LR for patch extraction
        self._padded_lr = F.pad(self._lr_tensor.unsqueeze(0), (self.pad_size,) * 4, mode='reflect')[0]

        # Store the original HR and LR images for visualization
        self._hr_img = hr_img
        self._lr_img = lr_img

    def __len__(self):
        return self.total_pixels

    def __getitem__(self, idx):
        # Load image on first access
        self._load_and_process_image()
        
        # Calculate pixel coordinates from flat index
        h, w = self._hr_tensor.shape[1], self._hr_tensor.shape[2]
        y = idx // w
        x = idx % w

        # Extract LR patch directly from the same position in the blurry image
        patch = self._padded_lr[:, y:y + self.patch_size, x:x + self.patch_size]
        patch = patch.reshape(3, -1)  # [3, patch_size²]

        # Use complete patch including center pixel
        patch = patch.flatten()  # [3*patch_size²]

        return {
            'coord': self._coords[idx],
            'patch': patch,
            'target': self._hr_tensor[:, y, x]
        }

    def get_images(self):
        """Return the HR and LR images as PIL Images for inspection"""
        self._load_and_process_image()
        return self._hr_img, self._lr_img

def train_super_resolution(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare datasets
    data_folder = args.data_dir
    image_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder)
                  if fname.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.sort()

    # Split train/val
    train_paths = image_paths[:args.num_train_images]
    val_paths = image_paths[args.num_train_images:args.num_train_images + args.num_val_images]

    print(f"Creating datasets for {len(train_paths)} training and {len(val_paths)} validation images...")
    print(f"Training images: {[os.path.basename(p) for p in train_paths]}")
    print(f"Validation images: {[os.path.basename(p) for p in val_paths]}")
    
    if args.use_random_scale:
        print(f"Using random scale factors between {args.min_scale_factor} and {args.max_scale_factor} for training")
    else:
        print(f"Using fixed scale factor: {args.scale_factor}")

    # Create datasets with cropping approach
    train_datasets = [SuperResolutionDatasetCropping(path, crop_size=args.crop_size, 
                                                   scale_factor=args.scale_factor,
                                                   patch_size=args.patch_size,
                                                   random_crop=True,
                                                   random_scale_factor=args.use_random_scale,
                                                   min_scale_factor=args.min_scale_factor,
                                                   max_scale_factor=args.max_scale_factor) 
                     for path in train_paths]
    val_datasets = [SuperResolutionDatasetCropping(path, crop_size=args.crop_size,
                                                 scale_factor=args.scale_factor,
                                                 patch_size=args.patch_size,
                                                 random_crop=False,
                                                 random_scale_factor=False)
                   for path in val_paths]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model = ImprovedConditionedSiren(
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        first_batch = True
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            coords = batch['coord'].to(device)
            patches = batch['patch'].to(device)
            targets = batch['target'].to(device)

            preds = model(coords, patches)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                coords = batch['coord'].to(device)
                patches = batch['patch'].to(device)
                targets = batch['target'].to(device)

                preds = model(coords, patches)
                val_loss += F.mse_loss(preds, targets).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print("-" * 50)

        # Plot and save loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
        plt.close()

    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LIIF Super Resolution model")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save-interval", type=int, default=5, help="Interval between model saving")
    parser.add_argument("--scale-factor", type=int, default=2, help="Upscaling factor")
    
    # Custom type for tuple argument
    def parse_tuple(s):
        try:
            # Remove parentheses and split by comma
            s = s.strip('()')
            return tuple(map(int, s.split(',')))
        except:
            raise argparse.ArgumentTypeError("Tuple must be in format '(x,y)'")
    
    parser.add_argument("--crop-size", type=parse_tuple, default=(256, 256), 
                       help="Crop size as tuple (width,height)")
    
    parser.add_argument("--patch-size", type=int, default=3, help="Patch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-train-images", type=int, default=6, help="Number of training images to use")
    parser.add_argument("--num-val-images", type=int, default=2, help="Number of validation images to use")
    
    # Random scale factor parameters
    parser.add_argument("--use-random-scale", action="store_true", help="Enable random scale factors for training")
    parser.add_argument("--min-scale-factor", type=float, default=1.5, help="Minimum scale factor for random scaling")
    parser.add_argument("--max-scale-factor", type=float, default=8.0, help="Maximum scale factor for random scaling")
    
    # Paths
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/liif_train", help="Path to save models and results")
    
    args = parser.parse_args()
    train_super_resolution(args) 