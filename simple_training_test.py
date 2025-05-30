#!/usr/bin/env python3
"""
Minimal test to verify the core SIREN and GAN functionality.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Device setup
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

# Simplified SIREN network
class SimpleSiren(nn.Module):
    def __init__(self, in_features=2, hidden_features=64, hidden_layers=2, out_features=3):
        super().__init__()
        
        layers = []
        # First layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Sigmoid())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Sigmoid())
            
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, coords):
        return self.net(coords)

# Simplified discriminator
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, img):
        return self.net(img)

# Simplified dataset
class SimpleDataset(Dataset):
    def __init__(self, size=64, num_samples=100):
        self.size = size
        self.num_samples = num_samples
        
        # Generate normalized coordinates
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.size),
            torch.linspace(-1, 1, self.size),
            indexing='ij'
        )
        self.coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
        
        # Simple circle image
        dist_from_center = torch.sqrt(x**2 + y**2)
        self.target = (dist_from_center < 0.5).float().unsqueeze(0)
        self.target = torch.cat([self.target, self.target, self.target], dim=0)  # RGB
        self.target = self.target.permute(1, 2, 0).reshape(-1, 3)  # [size*size, 3]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'coords': self.coords,
            'pixels': self.target
        }

def train_simple():
    """Train the simplified models"""
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=64, num_samples=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize models
    generator = SimpleSiren(in_features=2, hidden_features=64, hidden_layers=2, out_features=3).to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # Loss functions
    criterion_pixel = nn.MSELoss()
    criterion_adversarial = nn.BCEWithLogitsLoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    
    for epoch in range(num_epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in dataloader:
                # Get data
                coords = batch['coords'].to(device)
                real_pixels = batch['pixels'].to(device)
                batch_size = coords.shape[0]
                
                # Reshape real images for discriminator
                real_imgs = real_pixels.reshape(batch_size, 64, 64, 3).permute(0, 3, 1, 2)
                
                # ----------- Train Discriminator -----------
                optimizer_D.zero_grad()
                
                # Generate fake images
                fake_pixels = generator(coords)
                fake_imgs = fake_pixels.reshape(batch_size, 64, 64, 3).permute(0, 3, 1, 2)
                
                # Get discriminator outputs
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                
                # Create labels
                valid = torch.ones_like(d_real).to(device)
                fake = torch.zeros_like(d_fake).to(device)
                
                # Calculate discriminator loss
                loss_real = criterion_adversarial(d_real, valid)
                loss_fake = criterion_adversarial(d_fake, fake)
                loss_d = (loss_real + loss_fake) / 2
                
                loss_d.backward()
                optimizer_D.step()
                
                # ----------- Train Generator -----------
                optimizer_G.zero_grad()
                
                # Generate fake images
                fake_pixels = generator(coords)
                fake_imgs = fake_pixels.reshape(batch_size, 64, 64, 3).permute(0, 3, 1, 2)
                
                # Get discriminator output for fake images
                d_fake = discriminator(fake_imgs)
                
                # Calculate generator loss
                loss_adv = criterion_adversarial(d_fake, valid)
                loss_pixel = criterion_pixel(fake_pixels, real_pixels)
                
                # Combine losses
                loss_g = loss_adv + 100 * loss_pixel
                
                loss_g.backward()
                optimizer_G.step()
                
                # Update progress bar
                pbar.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())
                pbar.update()
        
        print(f"Epoch {epoch+1}/{num_epochs} - D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")
    
    print("Training completed!")
    return generator, discriminator

if __name__ == "__main__":
    train_simple() 