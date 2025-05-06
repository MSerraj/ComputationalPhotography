import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2 as cv


class ImageFitting(Dataset):
    def __init__(self, image_path, H, W):
        super().__init__()
        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ])
        img = transform(img) * 2. - 1.  # Normalize to [-1, 1]
        self.pixels = img.permute(1, 2, 0).contiguous().view(-1, 3)
        self.coords = self.get_mgrid(H, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

    @staticmethod
    def get_mgrid(H, W):
        """Generate a 2D grid of normalized coordinates."""
        x = torch.linspace(-1, 1, steps=W)
        y = torch.linspace(-1, 1, steps=H)
        grid = torch.stack(torch.meshgrid(y, x), dim=-1)
        return grid.view(-1, 2)
    

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
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # Allows to take derivative w.r.t. input
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
    
def list_png_files(data_folder):
    """List all PNG files in the specified folder."""
    return [f for f in os.listdir(data_folder) if f.endswith(".png")]
    
def load_image(data_folder, img_file_path):
    """Load and preprocess an image."""
    img_path = os.path.join(data_folder, img_file_path)
    img_original = Image.open(img_path)
    img_np_original = np.array(img_original)

    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np_original)
    plt.axis("off")
    plt.title(f"Original Image: {img_file_path}")
    plt.show()

    # Get image dimensions
    height_target, width_target, channels = img_np_original.shape
    print(f"Image dimensions: {height_target}x{width_target}, {channels} channels")

    return img_np_original, height_target, width_target, channels
    
def pixel_coordinates_normalized(image, downsize_factor): 
    """Generate normalized coordinates and pixel values for a downsampled image."""
    print(f"The original image has shape: {image.shape}")
    x, y = image.shape[:2]

    # Generate high-resolution coordinates for inference
    xs_hr = np.linspace(-1, 1, x)  # x coordinates (-1 to 1)
    ys_hr = np.linspace(-1, 1, y)  # y coordinates (-1 to 1)
    xx_hr, yy_hr = np.meshgrid(xs_hr, ys_hr, indexing="ij")
    high_res_coordinates = np.stack((xx_hr, yy_hr), axis=-1).reshape(-1, 2)

    # Normalize pixel values for the high-resolution image
    high_res_image = image / 255.0
    high_res_pixel_values = high_res_image.reshape(-1, 3)

    # Downsample the image for training
    resized_x = int(x / downsize_factor)
    resized_y = int(y / downsize_factor)
    resized_image = cv.resize(image, (resized_y, resized_x))
    print(f"The downsampled image has shape: {resized_image.shape}")

    # Generate low-resolution coordinates for training
    xs_lr = np.linspace(-1, 1, resized_x)  # x coordinates (-1 to 1)
    ys_lr = np.linspace(-1, 1, resized_y)  # y coordinates (-1 to 1)
    xx_lr, yy_lr = np.meshgrid(xs_lr, ys_lr, indexing="ij")
    low_res_coordinates = np.stack((xx_lr, yy_lr), axis=-1).reshape(-1, 2)

    # Normalize pixel values for the low-resolution image
    low_res_image = resized_image / 255.0
    low_res_pixel_values = low_res_image.reshape(-1, 3)

    return (
        low_res_coordinates, 
        low_res_pixel_values, 
        high_res_coordinates, 
        high_res_pixel_values, 
        (resized_x, resized_y), 
        (x, y)
    )
def plot_image(image, title=None):
    """Plot an image with optional title."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{title} Image: ")
    plt.show()

    # Get image dimensions
    height_target, width_target, channels = image.shape
    print(f"Image dimensions: {height_target}x{width_target}, {channels} channels")
    
    
def train_siren(model, dataloader, total_steps=5000, steps_til_summary=250, lr=1e-4, device=None, H=256, W=256):
    """
    Train the Siren model.

    Args:
        model (Siren): The Siren model.
        dataloader (DataLoader): DataLoader for the dataset.
        total_steps (int): Total number of training steps.
        steps_til_summary (int): Steps between summaries.
        lr (float): Learning rate.
        device (torch.device): Device to use for training.
        H (int): Height of the image.
        W (int): Width of the image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    # Remove batch dimension from ground_truth for reshaping
    ground_truth = ground_truth.squeeze(0)  # Shape: [H*W, 3]

    for step in range(total_steps):
        model_output, coords = model(model_input)  # model_output shape: [1, H*W, 3]
        model_output = model_output.squeeze(0)  # Remove batch dimension, shape: [H*W, 3]

        # Compute loss
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print(f"Step {step}, Total loss {loss.item():0.6f}")
            with torch.no_grad():
                # Reshape the model output to match the image dimensions
                try:
                    output_view = model_output.view(H, W, 3)  # Shape: [H, W, 3]
                except RuntimeError:
                    print(f"Reshaping failed. model_output shape: {model_output.shape}, expected: ({H}, {W}, 3)")
                    continue

                # Normalize and convert to uint8 for visualization
                output_view = torch.clamp(output_view, -1, 1) * 0.5 + 0.5
                output_view = (output_view * 255).to(torch.uint8).cpu().detach().numpy()

                # Plot the reconstructed image
                plt.imshow(output_view)
                plt.axis("off")
                plt.show()

        # Backpropagation and optimization
        optim.zero_grad()
        loss.backward()
        optim.step()