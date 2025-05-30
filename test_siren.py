import os
import argparse
import torch
import cv2
from models.siren import Siren

def get_mgrid_2d(height, width, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if dim != 2:
        raise ValueError("This function currently supports only 2D grids.")

    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    
    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    
    return mgrid

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

def main():
    parser = argparse.ArgumentParser(description="Test trained SIREN model with higher resolution")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--original-H", type=int, required=True, help="Original height of the training image")
    parser.add_argument("--original-W", type=int, required=True, help="Original width of the training image")
    parser.add_argument("--hidden-features", type=int, default=256, help="Number of hidden features in the model")
    parser.add_argument("--hidden-layers", type=int, default=3, help="Number of hidden layers in the model")
    
    # Testing parameters
    parser.add_argument("--scale-factors", type=float, nargs='+', default=[2.0, 4.0], 
                        help="Scale factors to test (space-separated)")
    parser.add_argument("--output-dir", type=str, default="test_results", 
                        help="Directory to save results")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = Siren(in_features=2, out_features=3, 
                  hidden_features=args.hidden_features,
                  hidden_layers=args.hidden_layers, 
                  outermost_linear=True)
    model.to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Test with different scale factors
    for scale_factor in args.scale_factors:
        print(f"Testing with scale factor: {scale_factor}x")
        save_path = os.path.join(args.output_dir, f"output_scale_{scale_factor}x.png")
        test_higher_resolution(model, args.original_H, args.original_W, 
                             scale_factor, save_path, device)
        print(f"Saved result to: {save_path}")
    
    print(f"Testing completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 