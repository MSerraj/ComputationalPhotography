import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Utility function to generate a 2D grid of coordinates
def get_mgrid_2d(height, width, dim=2):
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, dim)

class SingleImagePatchDataset(Dataset):
    def __init__(self, image_path, img_size=(256, 256), patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

        # Load and process image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        self.img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
        self.img_tensor = self.img_tensor.permute(2, 0, 1)  # CxHxW

        # Create coordinates grid
        self.coords = get_mgrid_2d(img_size[1], img_size[0])

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        # Get target pixel
        c, h, w = self.img_tensor.shape
        y = idx // w
        x = idx % w

        # Extract patch with reflection padding
        padded_img = F.pad(self.img_tensor.unsqueeze(0),
                          (self.pad_size,)*4, mode='reflect')[0]

        # Modified patch processing
        patch = padded_img[:, y:y+self.patch_size, x:x+self.patch_size]
        patch = patch.reshape(3, -1)  # [3, 9]

        # Remove center element (index 4 in 0-indexed 3x3 grid)
        patch = torch.cat([patch[:, :4], patch[:, 5:]], dim=1)  # [3, 8]
        patch = patch.flatten()  # [24]

        return {
            'coord': self.coords[idx],
            'patch': patch,
            'target': self.img_tensor[:, y, x]
        }
        
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate



class ConditionedSiren(nn.Module):
    def __init__(self, patch_size=3, hidden_dim=256):
        super().__init__()
        # Correct input dimension calculation (2 coords + 3*(3x3-1) patch values)
        input_dim = 2 + 3*(patch_size**2 - 1)  # Now 26 dimensions (2 + 24)

        self.net = nn.Sequential(
            SineLayer(input_dim, hidden_dim, is_first=True),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, coord, patch):
        x = torch.cat([coord, patch], dim=-1)
        return self.net(x)
