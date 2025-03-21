import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pickle
import argparse

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding to capture higher-frequency details.
    This helps the network better represent high-frequency details.
    """
    def __init__(self, input_dim=2, num_frequencies=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * 2 * num_frequencies
        
    def forward(self, x):
        # x has shape [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # Create frequencies for sinusoidal encoding
        # Using powers of 2 for frequencies: 1, 2, 4, 8, ...
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=x.device).float()
        
        # Reshape for broadcasting
        # [batch_size, input_dim] -> [batch_size, input_dim, 1]
        x_reshaped = x.view(batch_size, self.input_dim, 1)
        
        # [num_frequencies] -> [1, 1, num_frequencies]
        freqs_reshaped = freqs.view(1, 1, self.num_frequencies)
        
        # Computing sin and cos embeddings
        # [batch_size, input_dim, num_frequencies]
        args = x_reshaped * freqs_reshaped
        
        # Compute sin and cos
        embedding_sin = torch.sin(args)
        embedding_cos = torch.cos(args)
        
        # Flatten the last two dimensions
        # [batch_size, input_dim, num_frequencies] -> [batch_size, input_dim * num_frequencies]
        embedding_sin = embedding_sin.reshape(batch_size, -1)
        embedding_cos = embedding_cos.reshape(batch_size, -1)
        
        # Concatenate sin and cos embeddings
        # [batch_size, input_dim * num_frequencies * 2]
        embedding = torch.cat([embedding_sin, embedding_cos], dim=-1)
        
        return embedding

class ImprovedINRModel(nn.Module):
    """
    Improved Implicit Neural Representation model with positional encoding.
    """
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=256, num_layers=6, 
                 dropout_rate=0.0, use_positional_encoding=True, num_frequencies=10):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPositionalEncoding(
                input_dim=input_dim, 
                num_frequencies=num_frequencies
            )
            encoding_dim = self.positional_encoding.output_dim
        else:
            encoding_dim = input_dim
        
        # Network architecture
        layers = []
        
        # First layer
        layers.append(nn.Linear(encoding_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1] range
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Apply positional encoding if used
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
            
        return self.network(x)
    
    def train_model(self, dataloader, num_epochs=100, lr=1e-3, weight_decay=1e-5,
                    device='cpu', criterion=nn.MSELoss(), scheduler_gamma=0.95):
        """
        Train the model with learning rate scheduling and weight decay.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        
        self.train()
        self.to(device)
        
        losses = []
        
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            
            for batch_coords, batch_pixels in dataloader:
                batch_coords = batch_coords.to(device)
                batch_pixels = batch_pixels.to(device)
                
                # Forward pass
                pred_pixels = self(batch_coords)
                
                # Compute loss
                loss = criterion(pred_pixels, batch_pixels)
                total_loss += loss.item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Apply learning rate scheduling
            scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        return losses

def get_normalized_coordinates(h, w):
    """Generate normalized coordinate grid (0 to 1) for an image"""
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X, Y], axis=-1)  # [h, w, 2]
    return coords.reshape(-1, 2)  # Flatten to [h*w, 2]

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, coordinates, pixel_values):
        self.coordinates = torch.from_numpy(coordinates.astype(np.float32))
        self.pixel_values = torch.from_numpy(pixel_values.astype(np.float32))
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        pixel = self.pixel_values[idx]
        return coord, pixel

def downsample_image(img, factor):
    """Downsample an image by a given factor"""
    w, h = img.size
    return img.resize((w // factor, h // factor), Image.LANCZOS)

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM metrics between original and reconstructed images"""
    # Convert to numpy arrays if they're not already
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    # Make sure images are in the right format for the metrics
    original = original.astype(np.float32) / 255.0 if original.max() > 1.0 else original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32) / 255.0 if reconstructed.max() > 1.0 else reconstructed.astype(np.float32)
    
    # Calculate metrics
    psnr_value = psnr(original, reconstructed, data_range=1.0)
    ssim_value = ssim(original, reconstructed, channel_axis=2, data_range=1.0)
    
    return psnr_value, ssim_value

def train_inr_for_image(img_path, downscale_factor=4, target_scale=None, 
                        hidden_dim=256, num_layers=6, num_epochs=200, 
                        batch_size=4096, learning_rate=1e-3, weight_decay=1e-5,
                        use_positional_encoding=True, num_frequencies=10,
                        scheduler_gamma=0.95, device=None, output_dir='results'):
    """
    Train an INR model for a single image.
    
    Args:
        img_path: Path to the image
        downscale_factor: Factor to downsample the image for training
        target_scale: Optional different scale for upsampling (default: same as downscale)
        hidden_dim: Hidden dimension size of the model
        num_layers: Number of layers in the model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        use_positional_encoding: Whether to use positional encoding
        num_frequencies: Number of frequencies for positional encoding
        scheduler_gamma: Gamma value for learning rate scheduler
        device: Device to train on (default: auto-detect)
        output_dir: Directory to save results
        
    Returns:
        model: Trained INR model
        metrics: Dictionary of metrics including PSNR and SSIM
        images: Dictionary containing original, downsampled, and reconstructed images
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set target scale if not provided
    if target_scale is None:
        target_scale = downscale_factor
    
    # Load the image
    img_original = Image.open(img_path)
    img_np_original = np.array(img_original)
    
    # Get original dimensions
    height_original, width_original, channels = img_np_original.shape
    print(f"Original image dimensions: {height_original}x{width_original}, {channels} channels")
    
    # Save the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np_original)
    plt.axis("off")
    plt.title(f"Original Image")
    plt.savefig(os.path.join(output_dir, 'original.png'), bbox_inches='tight')
    plt.close()
    
    # Downsample image
    img_downsampled = downsample_image(img_original, downscale_factor)
    img_np_downsampled = np.array(img_downsampled)
    height_downsampled, width_downsampled, _ = img_np_downsampled.shape
    print(f"Downsampled image dimensions: {height_downsampled}x{width_downsampled}")
    
    # Save downsampled image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np_downsampled)
    plt.axis("off")
    plt.title(f"Downsampled Image (/{downscale_factor})")
    plt.savefig(os.path.join(output_dir, 'downsampled.png'), bbox_inches='tight')
    plt.close()
    
    # Normalize pixel values to [0, 1]
    img_np_downsampled_normalized = img_np_downsampled.astype(np.float32) / 255.0
    
    # Generate coordinates
    normalized_coordinates = get_normalized_coordinates(height_downsampled, width_downsampled)
    pixel_values = img_np_downsampled_normalized.reshape(-1, channels)
    
    # Create dataset and dataloader
    dataset = ImageDataset(normalized_coordinates, pixel_values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = ImprovedINRModel(
        input_dim=2, 
        output_dim=channels, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers,
        dropout_rate=0.0,  # We'll use weight decay instead of dropout
        use_positional_encoding=use_positional_encoding,
        num_frequencies=num_frequencies
    )
    
    # Train model
    criterion = nn.MSELoss()
    losses = model.train_model(
        dataloader, 
        num_epochs=num_epochs, 
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device, 
        criterion=criterion,
        scheduler_gamma=scheduler_gamma
    )
    
    # Plot and save training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.yscale('log')  # Log scale makes it easier to see progress
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    # Calculate upsampled dimensions based on target scale
    height_target = height_downsampled * target_scale
    width_target = width_downsampled * target_scale
    print(f"Target upsampled dimensions: {height_target}x{width_target}")
    
    # Generate coordinates for the target upsampled image
    target_coords_normalized = get_normalized_coordinates(height_target, width_target)
    target_coords_tensor = torch.from_numpy(target_coords_normalized).float().to(device)
    
    # Generate the upsampled image
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid CUDA out of memory
        batch_size = 10000
        num_batches = (len(target_coords_tensor) + batch_size - 1) // batch_size
        pred_pixels_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(target_coords_tensor))
            batch = target_coords_tensor[start_idx:end_idx]
            pred_pixels = model(batch).cpu().numpy()
            pred_pixels_list.append(pred_pixels)
        
        pred_pixels = np.concatenate(pred_pixels_list, axis=0)
    
    # Reshape and convert to image
    reconstructed_img = pred_pixels.reshape(height_target, width_target, channels)
    reconstructed_img_uint8 = (reconstructed_img * 255).clip(0, 255).astype(np.uint8)
    
    # Save reconstructed image
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_img_uint8)
    plt.axis("off")
    plt.title(f"INR Reconstructed Image (x{target_scale})")
    plt.savefig(os.path.join(output_dir, 'reconstructed.png'), bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    # We need to resize the original image to match the target dimensions for fair comparison
    img_original_resized = img_original.resize((width_target, height_target), Image.LANCZOS)
    img_np_original_resized = np.array(img_original_resized)
    
    # Save the resized original for comparison
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np_original_resized)
    plt.axis("off")
    plt.title(f"Original Image (resized)")
    plt.savefig(os.path.join(output_dir, 'original_resized.png'), bbox_inches='tight')
    plt.close()
    
    psnr_value, ssim_value = calculate_metrics(img_np_original_resized, reconstructed_img_uint8)
    
    # Compile results
    metrics = {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'final_loss': losses[-1]
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"PSNR: {psnr_value:.2f} dB\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")
        f.write(f"Final training loss: {losses[-1]:.6f}\n")
    
    # Generate images at different scales (to demonstrate continuous scaling)
    scales = [1.5, 2.0, 3.0, 4.0]
    upscaled_images = {}
    
    # Create directory for different scales
    scales_dir = os.path.join(output_dir, 'scales')
    os.makedirs(scales_dir, exist_ok=True)
    
    for scale in scales:
        # Calculate dimensions at this scale
        height = int(height_downsampled * scale)
        width = int(width_downsampled * scale)
        
        # Generate coordinates
        coords = get_normalized_coordinates(height, width)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        
        # Generate the image in batches
        model.eval()
        with torch.no_grad():
            batch_size = 10000
            num_batches = (len(coords_tensor) + batch_size - 1) // batch_size
            pred_pixels_list = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(coords_tensor))
                batch = coords_tensor[start_idx:end_idx]
                pred_pixels = model(batch).cpu().numpy()
                pred_pixels_list.append(pred_pixels)
            
            pred_pixels = np.concatenate(pred_pixels_list, axis=0)
        
        # Reshape and convert to image
        upscaled_img = pred_pixels.reshape(height, width, channels)
        upscaled_img_uint8 = (upscaled_img * 255).clip(0, 255).astype(np.uint8)
        upscaled_images[scale] = upscaled_img_uint8
        
        # Save image at this scale
        plt.figure(figsize=(10, 10))
        plt.imshow(upscaled_img_uint8)
        plt.axis("off")
        plt.title(f"Scale {scale}x: {width}x{height}")
        plt.savefig(os.path.join(scales_dir, f'scale_{scale:.1f}x.png'), bbox_inches='tight')
        plt.close()
    
    # Compare with traditional methods
    # Target scale
    scale = target_scale
    target_width = int(width_downsampled * scale)
    target_height = int(height_downsampled * scale)
    
    # Create directory for comparison
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Upscale using traditional methods
    img_downsampled_pil = Image.fromarray(img_np_downsampled)
    
    upscaled_nearest = np.array(img_downsampled_pil.resize((target_width, target_height), Image.NEAREST))
    upscaled_bilinear = np.array(img_downsampled_pil.resize((target_width, target_height), Image.BILINEAR))
    upscaled_bicubic = np.array(img_downsampled_pil.resize((target_width, target_height), Image.BICUBIC))
    upscaled_lanczos = np.array(img_downsampled_pil.resize((target_width, target_height), Image.LANCZOS))
    
    # Calculate metrics
    psnr_nearest, ssim_nearest = calculate_metrics(img_np_original_resized, upscaled_nearest)
    psnr_bilinear, ssim_bilinear = calculate_metrics(img_np_original_resized, upscaled_bilinear)
    psnr_bicubic, ssim_bicubic = calculate_metrics(img_np_original_resized, upscaled_bicubic)
    psnr_lanczos, ssim_lanczos = calculate_metrics(img_np_original_resized, upscaled_lanczos)
    
    # Save comparison images
    plt.figure(figsize=(10, 10))
    plt.imshow(upscaled_nearest)
    plt.axis("off")
    plt.title(f"Nearest: PSNR={psnr_nearest:.2f}, SSIM={ssim_nearest:.4f}")
    plt.savefig(os.path.join(comparison_dir, 'nearest.png'), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(upscaled_bilinear)
    plt.axis("off")
    plt.title(f"Bilinear: PSNR={psnr_bilinear:.2f}, SSIM={ssim_bilinear:.4f}")
    plt.savefig(os.path.join(comparison_dir, 'bilinear.png'), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(upscaled_bicubic)
    plt.axis("off")
    plt.title(f"Bicubic: PSNR={psnr_bicubic:.2f}, SSIM={ssim_bicubic:.4f}")
    plt.savefig(os.path.join(comparison_dir, 'bicubic.png'), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(upscaled_lanczos)
    plt.axis("off")
    plt.title(f"Lanczos: PSNR={psnr_lanczos:.2f}, SSIM={ssim_lanczos:.4f}")
    plt.savefig(os.path.join(comparison_dir, 'lanczos.png'), bbox_inches='tight')
    plt.close()
    
    # Save INR result in comparison folder too
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_img_uint8)
    plt.axis("off")
    plt.title(f"INR: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}")
    plt.savefig(os.path.join(comparison_dir, 'inr.png'), bbox_inches='tight')
    plt.close()
    
    # Save comparison metrics to file
    with open(os.path.join(comparison_dir, 'comparison_metrics.txt'), 'w') as f:
        f.write("Method\t\tPSNR (dB)\tSSIM\n")
        f.write("-" * 40 + "\n")
        f.write(f"Nearest\t\t{psnr_nearest:.2f}\t\t{ssim_nearest:.4f}\n")
        f.write(f"Bilinear\t{psnr_bilinear:.2f}\t\t{ssim_bilinear:.4f}\n")
        f.write(f"Bicubic\t\t{psnr_bicubic:.2f}\t\t{ssim_bicubic:.4f}\n")
        f.write(f"Lanczos\t\t{psnr_lanczos:.2f}\t\t{ssim_lanczos:.4f}\n")
        f.write(f"INR (ours)\t{psnr_value:.2f}\t\t{ssim_value:.4f}\n")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, 'inr_model.pth'))
    
    # Save results dictionary
    results = {
        'metrics': metrics,
        'losses': losses,
        'parameters': {
            'downscale_factor': downscale_factor,
            'target_scale': target_scale,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_epochs': num_epochs,
            'image_name': os.path.basename(img_path),
        }
    }
    
    with open(os.path.join(output_dir, 'inr_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"All results saved to {output_dir}/")
    
    return model, metrics, losses

def main():
    parser = argparse.ArgumentParser(description='INR-based Super Resolution')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor')
    parser.add_argument('--target_scale', type=float, default=None, help='Target scale for super-resolution (default: same as downscale)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for regularization')
    parser.add_argument('--no_positional_encoding', action='store_false', dest='use_positional_encoding', help='Disable positional encoding')
    parser.add_argument('--num_frequencies', type=int, default=10, help='Number of frequencies for positional encoding')
    parser.add_argument('--scheduler_gamma', type=float, default=0.98, help='Gamma for learning rate scheduler')
    
    args = parser.parse_args()
    
    # Train the model
    train_inr_for_image(
        img_path=args.image,
        downscale_factor=args.downscale_factor,
        target_scale=args.target_scale,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_positional_encoding=args.use_positional_encoding,
        num_frequencies=args.num_frequencies,
        scheduler_gamma=args.scheduler_gamma,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 