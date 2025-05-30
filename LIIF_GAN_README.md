# LIIF-GAN: Arbitrary-Resolution Image Super-Resolution

This project implements a GAN-based super-resolution model that combines the strengths of:
1. **SIREN (Sinusoidal Representation Networks)** for continuous implicit neural representation
2. **GAN (Generative Adversarial Networks)** for perceptual quality
3. **Patch-based conditioning** for local context awareness
4. **Arbitrary-resolution upscaling** for flexibility

## Features

- **Resolution Independence**: Can upscale images to any arbitrary resolution
- **High-Quality Results**: Uses adversarial training for perceptual quality
- **Efficient Processing**: Patch-based approach for memory efficiency
- **Cross-Platform**: Compatible with CUDA, MPS (Apple Silicon), and CPU

## Architecture

The model consists of:

1. **PatchConditionedSiren**: The generator network that:
   - Takes coordinates and local image patches as input
   - Uses sinusoidal activation functions for continuous representation
   - Outputs RGB values for each coordinate

2. **Discriminator**: A patch-based discriminator that:
   - Provides feedback for perceptual quality
   - Distinguishes between real and generated high-resolution patches

3. **Processing Pipeline**:
   - Divides the task into patches for memory efficiency
   - Uses overlapping patches with weighted blending for seamless results
   - Allows upscaling by arbitrary factors

## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
Pillow>=8.0.0
matplotlib>=3.3.0
tqdm>=4.50.0
scikit-image>=0.18.0
```

## Usage

### Training

To train the model:

```bash
python liif_gan.py
```

### Upscaling Images

To upscale a single image:

```bash
python test_liif_gan.py --input input.png --output upscaled.png --scale 4
```

### Comparing with Other Methods

To compare with standard upscaling methods:

```bash
python test_liif_gan.py --input input.png --compare --output_dir comparison_results
```

### Advanced Options

```bash
python test_liif_gan.py --help
```

## Parameters

- `--input`: Path to the input image
- `--output`: Path to save the upscaled image
- `--model`: Path to the trained model weights
- `--scale`: Upscaling factor (default: 4)
- `--patch_size`: Patch size for processing (default: 128)
- `--overlap`: Overlap between patches (default: 32)
- `--compare`: Enable comparison with other methods
- `--output_dir`: Directory for comparison results

## Implementation Details

### Key Components

1. **Coordinate-Based Representation**:
   - Images are represented as functions mapping coordinates to RGB values
   - Allows querying at arbitrary resolutions

2. **Patch Conditioning**:
   - Low-resolution patches provide local context
   - Helps maintain local consistency in the upscaled image

3. **GAN Training**:
   - Adversarial loss improves perceptual quality
   - Combined with pixel-wise loss for structural correctness

4. **Memory-Efficient Processing**:
   - Overlapping patch-based inference for large images
   - Weighted blending to eliminate boundary artifacts

## Results

The model produces high-quality upscaling with several advantages:
- Better preservation of fine details compared to bicubic interpolation
- Smoother edges and textures compared to nearest-neighbor methods
- Ability to upscale to arbitrary resolutions
- Consistent quality across different image types

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

This implementation draws inspiration from:
- SIREN (Implicit Neural Representations with Periodic Activation Functions)
- LIIF (Learning Continuous Image Representation with Local Implicit Image Function)
- SRGAN (Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network)

# Troubleshooting

## Common Issues

1. **Dimension Mismatch Errors**: The most common issues are related to tensor dimension mismatches, particularly when:
   - Reshaping coordinates and features
   - Combining patches during inference
   - Handling batched inputs

2. **Memory Issues**: Processing large images can cause memory errors. To mitigate this:
   - Use the patch-based approach with appropriate patch sizes
   - Reduce feature dimensions for testing
   - Use the `--simple` flag for quick tests

## Simplified Implementation

For debugging purposes, we've created simplified implementations:

- `simple_training_test.py`: A minimal implementation of the SIREN + GAN approach
- `train_liif_gan_quick.py`: A simplified training script with better stability
- `test_liif_gan_quick.py`: Includes a fallback to simple bicubic upscaling

These scripts are useful for testing the core concepts without dealing with the complexity of the full implementation.

# Approach

## Patch-Conditioned Architecture

The key insight of our approach is to combine:

1. **Coordinate-based representation**: Using a SIREN network that maps coordinates to RGB values
2. **Patch conditioning**: Extracting features from low-resolution patches to condition the network
3. **Adversarial training**: Using a GAN to improve perceptual quality

## Optimizations

For efficiency and scalability, we implemented:

- **Patch-based processing**: To handle arbitrary-sized images
- **Cross-platform compatibility**: Works on CUDA, MPS, and CPU
- **Smooth patch blending**: Weighted combination of overlapping patches 