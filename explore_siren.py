import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
import cv2
import skimage

# Import SIREN model components
from models.siren import Siren, SineLayer

def get_mgrid_2d(height, width, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if dim != 2:
        raise ValueError("This function currently supports only 2D grids.")

    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    
    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    
    return mgrid

def get_image_tensor(image_path, H, W, device):
    img = Image.open(image_path) 
    transform = Compose([
        Resize((H, W)), 
        ToTensor(),
    ])
    img = transform(img)
    img = img * 2. - 1.  
    
    return img.to(device)

class ImageFitting(Dataset):
    def __init__(self, image_path, H, W, device="cpu"):
        super().__init__()
        self.device = device
        self.H = H
        self.W = W
        
        img = get_image_tensor(image_path, H, W, device)
        self.pixels = img.permute(1, 2, 0).contiguous().view(-1, 3)
        self.coords = get_mgrid_2d(H, W, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels

def save_output(model_output, coords, H, W, save_path):
    with torch.no_grad():
        output_view = model_output.view(H, W, 3)
        output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
        output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
        cv2.imwrite(save_path, output_view[:, :, ::-1])

def test_higher_resolution(model, original_H, original_W, scale_factor, save_path, device):
    target_H = int(original_H * scale_factor)
    target_W = int(original_W * scale_factor)
    
    with torch.no_grad():
        coords = get_mgrid_2d(target_H, target_W, 2).unsqueeze(0).to(device)
        model_output, _ = model(coords)
        output_view = model_output.view(target_H, target_W, 3)
        output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
        output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()
        cv2.imwrite(save_path, output_view[:, :, ::-1])

def train(args):
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
    
    # Initialize dataset
    dataset = ImageFitting(args.image_path, args.H, args.W, device)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    
    # Initialize SIREN model
    model = Siren(in_features=2, out_features=3, hidden_features=args.hidden_features,
                  hidden_layers=args.hidden_layers, outermost_linear=True)
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Lists to store loss values
    losses = []
    
    # Get model input and ground truth
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)
    
    # Training loop
    for step in tqdm(range(args.total_steps)):
        model_output, coords = model(model_input)
        loss = ((model_output - ground_truth)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Save output and print loss every n steps
        if (step + 1) % args.save_interval == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.6f}")
            save_output(model_output, coords, args.H, args.W, 
                       os.path.join(args.output_dir, f"output_step_{step+1}.png"))
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    plt.close()
    
    # Test with higher resolution if requested
    if args.test_scale_factor > 1:
        test_higher_resolution(model, args.H, args.W, args.test_scale_factor,
                             os.path.join(args.output_dir, f"output_scale_{args.test_scale_factor}x.png"),
                             device)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pth"))
    print(f"Training completed. Results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train SIREN on a single image")
    
    # Training parameters
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--H", type=int, default=512, help="Height of the input image")
    parser.add_argument("--W", type=int, default=512, help="Width of the input image")
    parser.add_argument("--hidden-features", type=int, default=256, help="Number of hidden features")
    parser.add_argument("--hidden-layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--total-steps", type=int, default=5000, help="Total number of training steps")
    parser.add_argument("--save-interval", type=int, default=250, help="Interval between saving outputs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-scale-factor", type=float, default=2.0, help="Scale factor for testing higher resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 