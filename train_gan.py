import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
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

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(Discriminator, self).__init__()
        
        self.input_shape = input_shape
        in_channels, in_height, in_width = input_shape
        
        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        # First layer without normalization
        layers.extend(discriminator_block(in_channels, 64, 1, normalize=False))
        # Add downsampling blocks
        layers.extend(discriminator_block(64, 64, 2))
        layers.extend(discriminator_block(64, 128, 1))
        layers.extend(discriminator_block(128, 128, 2))
        layers.extend(discriminator_block(128, 256, 1))
        layers.extend(discriminator_block(256, 256, 2))
        layers.extend(discriminator_block(256, 512, 1))
        layers.extend(discriminator_block(512, 512, 2))
        
        # Output layer
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(512, 1024, 1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(1024, 1, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, img):
        return self.model(img)

# Custom Dataset
class UpscaleDataset(Dataset):
    def __init__(self, data_dir, scale_factor=4, hr_size=128):
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                           if f.endswith(('jpg', 'jpeg', 'png'))]
        
        # Use Resize to ensure all images have the same dimensions
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((self.lr_size, self.lr_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        
        # Create HR and LR versions
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)
        
        return {'lr': lr_img, 'hr': hr_img}

# Training function
def train(dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion_GAN, 
          criterion_content, epochs, device, output_dir, save_interval):
    # Sample images for visualization
    sample_batch = next(iter(dataloader))
    fixed_lr = sample_batch['lr'][:5].to(device)
    fixed_hr = sample_batch['hr'][:5].to(device)
    
    # Lists to store loss values
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Configure input
                imgs_lr = batch['lr'].to(device)
                imgs_hr = batch['hr'].to(device)
                batch_size = imgs_lr.size(0)
                
                # Adversarial ground truths
                valid = torch.ones((batch_size, 1, 1, 1), requires_grad=False).to(device)
                fake = torch.zeros((batch_size, 1, 1, 1), requires_grad=False).to(device)
                
                # ------------------
                #  Train Generator
                # ------------------
                optimizer_G.zero_grad()
                
                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)
                
                # Adversarial loss (make discriminator think generated images are real)
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
                
                # Content loss (pixel-wise difference between generated and real HR images)
                loss_content = criterion_content(gen_hr, imgs_hr)
                
                # Total generator loss
                loss_G = loss_content + 1e-3 * loss_GAN
                
                loss_G.backward()
                optimizer_G.step()
                
                # -----------------------
                #  Train Discriminator
                # -----------------------
                optimizer_D.zero_grad()
                
                # Loss of real and fake images
                loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
                
                # Total discriminator loss
                loss_D = (loss_real + loss_fake) / 2
                
                loss_D.backward()
                optimizer_D.step()
                
                # Update progress bar
                epoch_g_loss += loss_G.item()
                epoch_d_loss += loss_D.item()
                pbar.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
                pbar.update()
            
            # Record average epoch losses
            g_losses.append(epoch_g_loss / len(dataloader))
            d_losses.append(epoch_d_loss / len(dataloader))
            
            # Save model checkpoints
            if (epoch+1) % save_interval == 0:
                torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch+1}.pth")
                torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_epoch_{epoch+1}.pth")
            
            # Generate and save sample images
            with torch.no_grad():
                gen_imgs = generator(fixed_lr)
                # Denormalize
                gen_imgs = (gen_imgs + 1) / 2.0
                fixed_hr_norm = (fixed_hr + 1) / 2.0
                fixed_lr_upscaled = torch.nn.functional.interpolate(fixed_lr, scale_factor=args.scale_factor, mode='nearest')
                fixed_lr_norm = (fixed_lr_upscaled + 1) / 2.0
                
                # Create image grid
                img_grid = make_grid(torch.cat([fixed_lr_norm, gen_imgs, fixed_hr_norm], -1), nrow=5)
                save_image(img_grid, f"{output_dir}/epoch_{epoch+1}.png", normalize=False)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curves.png")
    plt.close()
    
    return generator, discriminator

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = UpscaleDataset(args.data_dir, scale_factor=args.scale_factor, hr_size=args.hr_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Dataset size: {len(dataset)} images")
    
    # Initialize generator and discriminator
    generator = Generator(scale_factor=args.scale_factor, num_res_blocks=args.num_res_blocks).to(device)
    discriminator = Discriminator(input_shape=(3, args.hr_size, args.hr_size)).to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_content = nn.L1Loss().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=args.betas)
    
    # Train the models
    generator, discriminator = train(
        dataloader=dataloader,
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        criterion_GAN=criterion_GAN,
        criterion_content=criterion_content,
        epochs=args.epochs,
        device=device,
        output_dir=args.output_dir,
        save_interval=args.save_interval
    )
    
    # Save final models
    torch.save(generator.state_dict(), f"{args.output_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{args.output_dir}/discriminator_final.pth")
    print(f"Training completed. Models saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for image upscaling")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--betas", type=tuple, default=(0.5, 0.999), help="Adam betas")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--save-interval", type=int, default=10, help="Interval between model saving")
    parser.add_argument("--scale-factor", type=int, default=4, help="Upscaling factor")
    parser.add_argument("--hr-size", type=int, default=128, help="High resolution image size")
    parser.add_argument("--num-res-blocks", type=int, default=16, help="Number of residual blocks in generator")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of CPU threads for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    # Paths
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Path to save models and results")
    
    args = parser.parse_args()
    main(args)