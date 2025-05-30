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

        # Initial layer with higher capacity
        self.input_layer = SineLayer(input_dim, hidden_dim, is_first=True)

        # More SIREN layers for better representation learning
        self.layers = nn.ModuleList([
            SineLayer(hidden_dim, hidden_dim) for _ in range(6)  # Increased from 4 to 6
        ])

        # Additional intermediate layers for better feature processing
        self.intermediate_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Skip connection mixing parameters (increased for more layers)
        self.skip_weights = nn.Parameter(torch.ones(6))  # Changed from 8 to 6
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
        
        # Additional processing layers
        for layer in self.intermediate_layers:
            x = layer(x)
        
        out = self.output_layer(x)
        
        # Residual connection with center pixel
        index_rgb = math.floor(patch_size/2)
        offset = patch_size ** 2
        patch_rgb = torch.cat((patch[:, index_rgb:index_rgb+1], 
                             patch[:, index_rgb+offset:index_rgb+offset+1], 
                             patch[:, index_rgb+2*offset:index_rgb+2*offset+1]), dim=1)

        return out + patch_rgb

class SuperResolutionDatasetCropping(Dataset):
    def __init__(self, hr_image_path, crop_size=(256, 256), scale_factor=8, patch_size=3, upscale_method=Image.BILINEAR, random_crop=True):
        super().__init__()
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.scale_factor = scale_factor
        self.crop_size = crop_size

        # Load HR image at full resolution
        hr_img = Image.open(hr_image_path).convert('RGB')

        # Get original image dimensions
        orig_width, orig_height = hr_img.size
        crop_width, crop_height = crop_size

        # Make sure the image is large enough for cropping
        if orig_width < crop_width or orig_height < crop_height:
            # If image is too small, resize it to at least the crop size
            scale = max(crop_width / orig_width, crop_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            hr_img = hr_img.resize((new_width, new_height), Image.BICUBIC)
            orig_width, orig_height = new_width, new_height

        # Crop the HR image
        if random_crop:
            # Random crop for training variability
            left = random.randint(0, orig_width - crop_width)
            top = random.randint(0, orig_height - crop_height)
        else:
            # Center crop for consistent validation
            left = (orig_width - crop_width) // 2
            top = (orig_height - crop_height) // 2

        hr_img = hr_img.crop((left, top, left + crop_width, top + crop_height))

        # Calculate LR size
        lr_width = crop_width // scale_factor
        lr_height = crop_height // scale_factor

        # Create LR image: downscale first
        small_lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)

        # Then upscale back to original size with specified interpolation to create blurry image
        lr_img = small_lr_img.resize(crop_size, upscale_method)

        # Convert to tensors and normalize to [-1, 1]
        self.hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 127.5 - 1.0
        self.lr_tensor = torch.from_numpy(np.array(lr_img)).float() / 127.5 - 1.0

        # Permute to CxHxW
        self.hr_tensor = self.hr_tensor.permute(2, 0, 1)
        self.lr_tensor = self.lr_tensor.permute(2, 0, 1)

        # Create coordinate grid for HR image
        h_hr, w_hr = self.hr_tensor.shape[1], self.hr_tensor.shape[2]
        self.coords = get_mgrid_2d(h_hr, w_hr)

        # Precompute all data
        self.data = []
        padded_lr = F.pad(self.lr_tensor.unsqueeze(0), (self.pad_size,) * 4, mode='reflect')[0]
        c, h, w = self.hr_tensor.shape

        for idx in range(self.coords.shape[0]):
            y = idx // w
            x = idx % w

            # Extract LR patch directly from the same position in the blurry image
            patch = padded_lr[:, y:y + self.patch_size, x:x + self.patch_size]
            patch = patch.reshape(3, -1)  # [3, patch_size²]

            # Use complete patch including center pixel
            patch = patch.flatten()  # [3*patch_size²]

            self.data.append({
                'coord': self.coords[idx],
                'patch': patch,
                'target': self.hr_tensor[:, y, x]
            })

        # Store the original HR and LR images for visualization
        self.hr_img = hr_img
        self.lr_img = lr_img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_images(self):
        """Return the HR and LR images as PIL Images for inspection"""
        return self.hr_img, self.lr_img

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

    # Create datasets with cropping approach
    train_datasets = [SuperResolutionDatasetCropping(path, crop_size=args.crop_size, 
                                                   scale_factor=args.scale_factor,
                                                   patch_size=args.patch_size,
                                                   random_crop=True) 
                     for path in train_paths]
    val_datasets = [SuperResolutionDatasetCropping(path, crop_size=args.crop_size,
                                                 scale_factor=args.scale_factor,
                                                 patch_size=args.patch_size,
                                                 random_crop=False)
                   for path in val_paths]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Initialize model
    model = ImprovedConditionedSiren(
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer with improved settings - less aggressive weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # More patient learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=8, verbose=True, min_lr=1e-7
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    early_stop_patience = 20  # Increased patience

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            coords = batch['coord'].to(device, non_blocking=True)
            patches = batch['patch'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            preds = model(coords, patches)
            
            # Use both MSE and L1 loss for better training
            mse_loss = F.mse_loss(preds, targets)
            l1_loss = F.l1_loss(preds, targets)
            loss = mse_loss + 0.1 * l1_loss  # Combined loss

            optimizer.zero_grad()
            loss.backward()
            
            # More conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                coords = batch['coord'].to(device, non_blocking=True)
                patches = batch['patch'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                preds = model(coords, patches)
                # Use same loss as training for consistency
                mse_loss = F.mse_loss(preds, targets)
                l1_loss = F.l1_loss(preds, targets)
                loss = mse_loss + 0.1 * l1_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            patience_counter = 0
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.2e}")
        print(f"Patience: {patience_counter}/{early_stop_patience}")
        print(f"Train/Val Ratio: {train_loss/val_loss:.3f}")  # Monitor over/underfitting

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Plot and save loss curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(train_losses) > 1:
            plt.plot(range(1, len(train_losses)), [train_losses[i] - train_losses[i-1] 
                                                  for i in range(1, len(train_losses))], 
                    label='Train Loss Change', color='blue')
            plt.plot(range(1, len(val_losses)), [val_losses[i] - val_losses[i-1] 
                                                for i in range(1, len(val_losses))], 
                    label='Val Loss Change', color='red')
            plt.xlabel('Epochs')
            plt.ylabel('Loss Change')
            plt.legend()
            plt.title('Loss Change Per Epoch')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")

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
    
    # Paths
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/liif_train", help="Path to save models and results")
    
    args = parser.parse_args()
    train_super_resolution(args) 