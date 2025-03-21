from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm



def normalize_image(image, mean, std):
    normalized_image = (image - mean) / std
    return normalized_image

def get_normalized_coordinates(h, w):
    """Generate normalized coordinate grid (0 to 1) for an image"""
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X, Y], axis=-1)  # [h, w, 2]
    return coords.reshape(-1, 2)  # Flatten to [h*w, 2]


class ImageDataset(Dataset):
    def __init__(self, coordinates, pixel_values):
        self.coordinates = coordinates.astype(np.float32)
        self.pixel_values = pixel_values.astype(np.float32)
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        pixel = self.pixel_values[idx]
        return coord, pixel

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=10):
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
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class INRModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=256, num_layers=4, first_omega_0=10, hidden_omega_0=10, outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=first_omega_0))

        for _ in range(num_layers - 1):
            self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, output_dim)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, np.sqrt(6 / hidden_dim) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_dim, output_dim, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
    

    
    def train_model(self, dataloader, num_epochs = 100, lr = 1e-3, device = 'cpu' if not torch.cuda.is_available() else 'cuda', criterion = nn.MSELoss(), optimizer = None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=lr)
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
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses