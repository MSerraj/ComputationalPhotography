from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict


def normalize_image(image, mean, std):
    normalized_image = (image - mean) / std
    return normalized_image

def get_normalized_coordinates(h, w):
    """Generate normalized coordinate grid (-1 to 1) for an image"""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X, Y], axis=-1)  # [h, w, 2]
    return coords.reshape(-1, 2)  # Flatten to [h*w, 2]

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int or tuple of ints
    dim: int'''
    if isinstance(sidelen, tuple):
        # If sidelen is a tuple (height, width), create corresponding tensors
        tensors = []
        for i, size in enumerate(sidelen):
            if i < dim:  # Only create up to dim tensors
                tensors.append(torch.linspace(-1, 1, steps=size))
        # If we have fewer dimensions in the tuple than dim, repeat the last one
        while len(tensors) < dim:
            tensors.append(torch.linspace(-1, 1, steps=sidelen[-1]))
        tensors = tuple(tensors)
    else:
        # Original behavior for single integer
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

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


class INRModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=256, num_layers=4, 
                 first_omega_0=30, hidden_omega_0=30., outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(input_dim, hidden_dim,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(num_layers - 1):
            self.net.append(SineLayer(hidden_dim, hidden_dim,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0,
                                              np.sqrt(6 / hidden_dim) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_dim, output_dim,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # Note: We're not cloning coords or requiring gradients here to avoid 
        # unnecessary operations during inference and training
        output = self.net(coords)
        return output
    
    def forward_with_gradients(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
    
    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

    def train_model(self, dataloader, num_epochs=100, lr=1e-3, device='cpu' if not torch.cuda.is_available() else 'cuda', criterion=nn.MSELoss(), optimizer=None):
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

# Differential operators for computing gradients, divergence, and laplacian
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)