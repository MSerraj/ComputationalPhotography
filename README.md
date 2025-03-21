# INR-based Super-Resolution

This project implements an Implicit Neural Representation (INR) approach to continuous super-resolution. It trains an INR on a single downsampled image and then can generate upscaled images at arbitrary scales.

## Features

- Training an INR on a single image
- Continuous super-resolution at arbitrary scales
- Sinusoidal positional encoding for better high-frequency detail reconstruction
- Comparison with traditional interpolation methods
- Evaluation using PSNR and SSIM metrics

## Requirements

See the `requirements.txt` file for the necessary dependencies.

## Usage

### Local Execution

To run the script locally:

```bash
python inr_super_resolution.py --image data/your_image.png --output_dir results
```

### Cluster Execution

To run on the EPFL cluster:

1. Make sure you have the necessary files in your directory:
   - `inr_super_resolution.py`
   - `inr_super_resolution.slurm`
   - `requirements.txt`
   - A `data` folder containing your image(s)

2. Submit the job to SLURM:
   ```bash
   sbatch inr_super_resolution.slurm
   ```

3. Check the job status:
   ```bash
   squeue -u $USER
   ```

4. Once the job completes, results will be available in the `results` directory.

## Command-line Arguments

The script supports the following command-line arguments:

- `--image`: Path to the input image (required)
- `--output_dir`: Directory to save results (default: 'results')
- `--downscale_factor`: Factor to downsample the original image (default: 4)
- `--target_scale`: Upscaling factor (default: same as downscale_factor)
- `--hidden_dim`: Hidden dimension of the model (default: 256)
- `--num_layers`: Number of layers in the model (default: 6)
- `--num_epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size for training (default: 4096)
- `--learning_rate`: Initial learning rate (default: 5e-4)
- `--weight_decay`: Weight decay for regularization (default: 1e-6)
- `--no_positional_encoding`: Disable positional encoding
- `--num_frequencies`: Number of frequencies for positional encoding (default: 10)
- `--scheduler_gamma`: Gamma value for learning rate scheduler (default: 0.98)

## Results

The script generates the following outputs in the specified output directory:

- `original.png`: The original input image
- `downsampled.png`: The downsampled image used for training
- `reconstructed.png`: The INR-reconstructed image at the target scale
- `original_resized.png`: The original image resized to the target dimensions (for comparison)
- `training_loss.png`: Plot of the training loss over epochs
- `metrics.txt`: PSNR, SSIM, and final training loss values
- `inr_model.pth`: The trained model weights
- `inr_results.pkl`: Saved metrics and parameters

### Scales Directory

Contains images generated at different scales to demonstrate continuous scaling capability.

### Comparison Directory

Contains images and metrics comparing the INR approach with traditional interpolation methods (Nearest, Bilinear, Bicubic, Lanczos). 