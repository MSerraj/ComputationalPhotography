import torch
import torch.nn as nn
import torch.nn.functional as F
from liif_gan import PatchConditionedSiren, get_mgrid_2d, get_device

device = get_device()
print(f"Using device: {device}")

def test_generator_shapes():
    """Test the PatchConditionedSiren shapes with different input dimensions"""
    
    # Initialize generator
    generator = PatchConditionedSiren(
        coord_dim=2,
        patch_feature_dim=64,
        hidden_features=256,
        hidden_layers=3,
        out_features=3
    ).to(device)
    
    # Test case 1: Single coordinate point per batch
    print("\nTest case 1: Single coordinate point per batch")
    batch_size = 2
    lr_size = 8
    coords_2d = torch.randn(batch_size, 2).to(device)  # [batch_size, coord_dim]
    lr_patch = torch.randn(batch_size, 3, lr_size, lr_size).to(device)  # [batch_size, 3, h, w]
    
    print(f"Input coords shape: {coords_2d.shape}")
    print(f"Input patch shape: {lr_patch.shape}")
    
    output_2d = generator(coords_2d, lr_patch)
    print(f"Output shape: {output_2d.shape}")
    
    # Test case 2: Batch of multiple coordinate points
    print("\nTest case 2: Batch of multiple coordinate points")
    patch_size = 32
    num_points = 10  # Just use a few points for testing
    coords_3d = torch.randn(batch_size, num_points, 2).to(device)  # [batch_size, num_points, coord_dim]
    
    print(f"Input coords shape: {coords_3d.shape}")
    print(f"Input patch shape: {lr_patch.shape}")
    
    output_3d = generator(coords_3d, lr_patch)
    print(f"Output shape: {output_3d.shape}")
    
    # Test case 3: Real-world scenario with coordinate grid
    print("\nTest case 3: Real-world scenario with coordinate grid")
    hr_size = lr_size * 4  # 4x upscaling
    # Get coordinates on the correct device
    coords_grid = get_mgrid_2d(hr_size, hr_size).to(device)  # [hr_size*hr_size, 2]
    coords_grid = coords_grid.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, hr_size*hr_size, 2]
    
    print(f"Input coords shape: {coords_grid.shape}")
    print(f"Input patch shape: {lr_patch.shape}")
    print(f"Coord device: {coords_grid.device}, Patch device: {lr_patch.device}")
    
    output_grid = generator(coords_grid, lr_patch)
    print(f"Output shape: {output_grid.shape}")
    
    # Reshape output to image
    if output_grid.dim() == 3:  # [batch, points, 3]
        output_img = output_grid.reshape(batch_size, hr_size, hr_size, 3)
        output_img = output_img.permute(0, 3, 1, 2)  # [batch, 3, h, w]
        print(f"Reshaped output image shape: {output_img.shape}")

    # Test case 4: Inference with a single image but many coordinates
    print("\nTest case 4: Single image, many coordinates (inference)")
    batch_size = 1
    lr_patch = torch.randn(batch_size, 3, lr_size, lr_size).to(device)
    
    # This simulates upscaling to 4x the input resolution
    upscale_factor = 4
    target_h, target_w = lr_size * upscale_factor, lr_size * upscale_factor
    coords = get_mgrid_2d(target_h, target_w).to(device)
    coords = coords.unsqueeze(0).to(device)  # [1, target_h*target_w, 2]
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Input patch shape: {lr_patch.shape}")
    print(f"Coord device: {coords.device}, Patch device: {lr_patch.device}")
    
    with torch.no_grad():
        output = generator(coords, lr_patch)
    
    print(f"Output shape: {output.shape}")
    
    # Reshape for visualization
    if output.dim() == 3:  # [batch, points, 3]
        output_img = output.reshape(batch_size, target_h, target_w, 3)
        output_img = output_img.permute(0, 3, 1, 2)  # [batch, 3, h, w]
        print(f"Reshaped output image shape: {output_img.shape}")

if __name__ == "__main__":
    test_generator_shapes() 