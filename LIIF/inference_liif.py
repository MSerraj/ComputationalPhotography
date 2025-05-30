import os
import argparse
import torch
import math
from torch import nn
import torch.nn.functional as F
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

class SuperResolutionInference:
    def __init__(self, model_path, patch_size=3, hidden_dim=256, device=None):
        """
        Initialize the super-resolution inference class.
        
        Args:
            model_path: Path to the trained model (.pth file)
            patch_size: Patch size used during training
            hidden_dim: Hidden dimension used during training
            device: Device to run inference on (cuda/cpu/mps)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ImprovedConditionedSiren(
            patch_size=patch_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        
        print(f"Model loaded successfully from {model_path}")
    
    def preprocess_image(self, image_path, target_size=None, scale_factor=2):
        """
        Preprocess the input image for super-resolution.
        
        Args:
            image_path: Path to input image
            target_size: Target size for the output (if None, uses scale_factor)
            scale_factor: Factor to upscale the image
            
        Returns:
            lr_tensor: Low resolution tensor
            target_height: Target height for output
            target_width: Target width for output
        """
        # Load image
        hr_img = Image.open(image_path).convert('RGB')
        original_width, original_height = hr_img.size
        
        if target_size is not None:
            target_width, target_height = target_size
        else:
            target_width = original_width * scale_factor
            target_height = original_height * scale_factor
        
        # Create LR image by downscaling and then upscaling
        lr_img = hr_img.resize((target_width, target_height), Image.BILINEAR)
        
        # Convert to tensor and normalize to [-1, 1]
        lr_tensor = torch.from_numpy(np.array(lr_img)).float() / 127.5 - 1.0
        lr_tensor = lr_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return lr_tensor, target_height, target_width, hr_img
    
    def extract_patches_and_coords(self, lr_tensor, target_height, target_width):
        """
        Extract patches and coordinates for the LIIF model.
        
        Args:
            lr_tensor: Low resolution tensor [C, H, W]
            target_height: Target output height
            target_width: Target output width
            
        Returns:
            coords: Coordinate grid
            patches: Extracted patches
        """
        # Create coordinate grid for target resolution
        coords = get_mgrid_2d(target_height, target_width)
        
        # Pad the LR tensor for patch extraction
        padded_lr = F.pad(lr_tensor.unsqueeze(0), (self.pad_size,) * 4, mode='reflect')[0]
        
        # Extract patches for each coordinate
        patches = []
        c, h, w = lr_tensor.shape
        
        for idx in range(coords.shape[0]):
            y = idx // target_width
            x = idx % target_width
            
            # Ensure we don't go out of bounds
            y = min(y, h - 1)
            x = min(x, w - 1)
            
            # Extract patch
            patch = padded_lr[:, y:y + self.patch_size, x:x + self.patch_size]
            patch = patch.reshape(3, -1).flatten()  # [3*patch_size²]
            patches.append(patch)
        
        patches = torch.stack(patches)
        
        return coords, patches
    
    def super_resolve(self, image_path, output_path=None, target_size=None, scale_factor=2, batch_size=1024):
        """
        Perform super-resolution on an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (if None, will be auto-generated)
            target_size: Target size (width, height) for output
            scale_factor: Upscaling factor if target_size is None
            batch_size: Batch size for inference
            
        Returns:
            sr_image: Super-resolved PIL Image
            lr_image: Low-resolution PIL Image
        """
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        lr_tensor, target_height, target_width, original_img = self.preprocess_image(
            image_path, target_size, scale_factor
        )
        
        print(f"Target resolution: {target_width}x{target_height}")
        
        # Extract patches and coordinates
        coords, patches = self.extract_patches_and_coords(lr_tensor, target_height, target_width)
        
        # Initialize output array
        sr_array = np.zeros((target_height, target_width, 3))
        
        # Run inference in batches
        print("Running super-resolution inference...")
        with torch.no_grad():
            for i in tqdm(range(0, len(coords), batch_size)):
                # Get batch
                batch_coords = coords[i:i+batch_size].to(self.device)
                batch_patches = patches[i:i+batch_size].to(self.device)
                
                # Run inference
                batch_preds = self.model(batch_coords, batch_patches)
                batch_preds = batch_preds.cpu().numpy()
                
                # Place predictions in output array
                for j in range(len(batch_preds)):
                    idx = i + j
                    if idx < len(coords):
                        y = idx // target_width
                        x = idx % target_width
                        if y < target_height and x < target_width:
                            sr_array[y, x] = batch_preds[j]
        
        # Convert to [0, 1] range and create PIL image
        sr_array = (sr_array + 1) / 2
        sr_array = np.clip(sr_array, 0, 1)
        sr_image = Image.fromarray((sr_array * 255).astype(np.uint8))
        
        # Create LR image for comparison
        lr_array = (lr_tensor.permute(1, 2, 0).numpy() + 1) / 2
        lr_array = np.clip(lr_array, 0, 1)
        lr_image = Image.fromarray((lr_array * 255).astype(np.uint8))
        
        # Save output if path provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_super_resolved.png"
        
        sr_image.save(output_path)
        print(f"Super-resolved image saved to: {output_path}")
        
        return sr_image, lr_image, original_img
    
    def visualize_results(self, sr_image, lr_image, original_img=None):
        """
        Visualize the super-resolution results.
        
        Args:
            sr_image: Super-resolved PIL Image
            lr_image: Low-resolution PIL Image
            original_img: Original PIL Image (optional)
        """
        if original_img is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(lr_image)
            axes[1].set_title('Low Resolution')
            axes[1].axis('off')
            
            axes[2].imshow(sr_image)
            axes[2].set_title('Super Resolution')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(lr_image)
            axes[0].set_title('Low Resolution Input')
            axes[0].axis('off')
            
            axes[1].imshow(sr_image)
            axes[1].set_title('Super Resolution Output')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="LIIF Super Resolution Inference")
    
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to the trained model file (.pth)")
    parser.add_argument("--input-image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save output image (optional)")
    parser.add_argument("--scale-factor", type=int, default=2,
                       help="Upscaling factor")
    parser.add_argument("--target-width", type=int, default=None,
                       help="Target width (overrides scale-factor)")
    parser.add_argument("--target-height", type=int, default=None,
                       help="Target height (overrides scale-factor)")
    parser.add_argument("--patch-size", type=int, default=3,
                       help="Patch size used in training")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension used in training")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for inference")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization of results")
    
    args = parser.parse_args()
    
    # Determine target size
    target_size = None
    if args.target_width is not None and args.target_height is not None:
        target_size = (args.target_width, args.target_height)
    
    # Initialize inference class
    sr_inference = SuperResolutionInference(
        model_path=args.model_path,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim
    )
    
    # Perform super-resolution
    sr_image, lr_image, original_img = sr_inference.super_resolve(
        image_path=args.input_image,
        output_path=args.output_path,
        target_size=target_size,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size
    )
    
    # Visualize if requested
    if args.visualize:
        sr_inference.visualize_results(sr_image, lr_image, original_img)

if __name__ == "__main__":
    main() 