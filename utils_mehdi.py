from torchsr.datasets import Div2K
# from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset
from torch import nn
import torch
import os
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def plot_image(image, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{title} Image: ")
    plt.show()

    # Get image dimensions
    height_target, width_target, channels = image.shape
    print(f"Image dimensions: {height_target}x{width_target}, {channels} channels")

def pixel_coordinates_normalized(image, downsize_factor): 
    print(f"The original image has shape: {image.shape}")
    x, y = image.shape[:2]
    resized_image = cv.resize(image, (y // downsize_factor, x // downsize_factor))
    resized_x, resized_y = resized_image.shape[:2]
    xs = np.linspace(-1, 1, resized_x)  # x coordinates (-1 to 1)
    ys = np.linspace(-1, 1, resized_y)  # y coordinates (-1 to 1)

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    coordinates = np.stack((xx, yy), axis=-1)
    coordinates = coordinates.reshape(-1, 2) 
    resized_image = resized_image/255.0
    #norm_resized_image = (resized_image - np.mean(resized_image)) / np.std(resized_image)
    pixel_values = resized_image.reshape(-1, 3)
    
    return coordinates, pixel_values, resized_image, resized_x, resized_y


