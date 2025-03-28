import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Define ResidualBlock for Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features)
        )
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        return self.prelu(x + self.block(x))

# Define Generator
class Generator(nn.Module):
    def __init__(self, scale_factor, num_res_blocks=16):
        super(Generator, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling layers
        upsample_layers = []
        for _ in range(int(np.log2(scale_factor))):
            upsample_layers.extend([
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])
        self.upsampling = nn.Sequential(*upsample_layers)
        
        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2)
        out4 = out1 + out3
        out5 = self.upsampling(out4)
        out = self.conv3(out5)
        return torch.tanh(out)

def upscale_image(image_path, weights_path, output_path, scale_factor, num_res_blocks=16):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize generator and load weights
    generator = Generator(scale_factor=scale_factor, num_res_blocks=num_res_blocks).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()
    print(f"Loaded generator weights from {weights_path}")
    
    # Load the image
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size
    print(f"Original image size: {orig_width}x{orig_height}")
    
    # Check if the image is too large
    max_size = 128  # Maximum size for processing at once
    if orig_width > max_size or orig_height > max_size:
        print(f"Image is large, resizing to maximum dimension of {max_size} for processing")
        # Calculate new dimensions maintaining aspect ratio
        if orig_width > orig_height:
            new_width = max_size
            new_height = int(orig_height * (max_size / orig_width))
        else:
            new_height = max_size
            new_width = int(orig_width * (max_size / orig_height))
        
        img = img.resize((new_width, new_height), Image.BICUBIC)
        print(f"Resized to: {new_width}x{new_height}")
    
    # Image transformation for network input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Convert image to tensor
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate upscaled image
    with torch.no_grad():
        print("Upscaling image...")
        upscaled_tensor = generator(input_tensor)
        # Denormalize
        upscaled_tensor = (upscaled_tensor + 1) / 2.0
        upscaled_tensor = upscaled_tensor.clamp(0, 1)
    
    # Convert to PIL image and save
    upscaled_img = transforms.ToPILImage()(upscaled_tensor.squeeze(0).cpu())
    upscaled_width, upscaled_height = upscaled_img.size
    print(f"Upscaled image size: {upscaled_width}x{upscaled_height}")
    upscaled_img.save(output_path)
    print(f"Saved upscaled image to {output_path}")
    
    # For comparison, also save a bicubic upscaled version
    bicubic_path = os.path.splitext(output_path)[0] + "_bicubic" + os.path.splitext(output_path)[1]
    bicubic_img = img.resize((upscaled_width, upscaled_height), Image.BICUBIC)
    bicubic_img.save(bicubic_path)
    print(f"Saved bicubic upscaled image to {bicubic_path}")
    
    # Create comparison image
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Bicubic Upscaling')
    plt.imshow(bicubic_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('GAN Upscaling')
    plt.imshow(upscaled_img)
    plt.axis('off')
    
    comparison_path = os.path.splitext(output_path)[0] + "_comparison.png"
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    print(f"Saved comparison image to {comparison_path}")

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Upscale the image
    upscale_image(
        image_path=args.image_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        scale_factor=args.scale_factor,
        num_res_blocks=args.num_res_blocks
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained GAN for image upscaling")
    
    # Required arguments
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to the generator weights")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the upscaled image")
    
    # Optional arguments
    parser.add_argument("--scale-factor", type=int, default=4, help="Upscaling factor (must match the trained model)")
    parser.add_argument("--num-res-blocks", type=int, default=16, help="Number of residual blocks in generator (must match the trained model)")
    
    args = parser.parse_args()
    main(args)