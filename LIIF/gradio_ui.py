#!/usr/bin/env python3

"""
Gradio Web Interface for LIIF Super-Resolution
A user-friendly web interface for using the trained LIIF model.
"""

import os
import tempfile
import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from inference_liif import SuperResolutionInference
import time

class GradioSuperResolution:
    def __init__(self):
        self.sr_inference = None
        self.model_loaded = False
        # Store images for zoom functionality
        self.current_original = None
        self.current_sr = None
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_images')
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self):
        """Create temporary directory if it doesn't exist"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
    def _get_temp_path(self, suffix):
        """Generate a temporary file path in our temp directory"""
        import uuid
        filename = f"{uuid.uuid4()}{suffix}"
        return os.path.join(self.temp_dir, filename)
        
    def load_model(self, model_path, patch_size, hidden_dim):
        """Load the super-resolution model"""
        try:
            if not os.path.exists(model_path):
                return f"❌ Error: Model file '{model_path}' not found!", None, None
            
            self.sr_inference = SuperResolutionInference(
                model_path=model_path,
                patch_size=int(patch_size),
                hidden_dim=int(hidden_dim)
            )
            self.model_loaded = True
            return f"✅ Model loaded successfully from {model_path}", None, None
        
        except Exception as e:
            self.model_loaded = False
            return f"❌ Error loading model: {str(e)}", None, None
    
    def super_resolve_image(self, input_image, scale_factor, target_width, target_height, 
                          batch_size, use_custom_size):
        """Perform super-resolution on the uploaded image"""
        
        if not self.model_loaded:
            return "❌ Please load a model first!", None, None, None, None
        
        if input_image is None:
            return "❌ Please upload an image first!", None, None, None, None
        
        try:
            # Save uploaded image to our temporary directory
            temp_input_path = self._get_temp_path('.png')
            input_image.save(temp_input_path)
            
            # Determine target size
            target_size = None
            if use_custom_size and target_width > 0 and target_height > 0:
                target_size = (int(target_width), int(target_height))
                scale_info = f"Custom size: {target_width}×{target_height}"
            else:
                scale_info = f"Scale factor: {scale_factor}×"
            
            # Perform super-resolution
            sr_image, lr_image, original_img = self.sr_inference.super_resolve(
                image_path=temp_input_path,
                target_size=target_size,
                scale_factor=int(scale_factor),
                batch_size=int(batch_size)
            )
            
            # Clean up input temporary file
            os.unlink(temp_input_path)
            
            # Save super-resolved image to temporary directory for download
            temp_sr_path = self._get_temp_path('_super_resolved.png')
            sr_image.save(temp_sr_path)
            
            # Create comparison image
            comparison_img = self.create_comparison_image(original_img, lr_image, sr_image)
            
            # Store images for zoom functionality
            self.current_original = original_img
            self.current_sr = sr_image
            
            # Create error map
            error_map_img = self.create_error_map(original_img, sr_image)
            
            # Prepare info text
            info_text = f"""
✅ Success! Original: {original_img.size[0]}×{original_img.size[1]} → Enhanced: {sr_image.size[0]}×{sr_image.size[1]}
{scale_info} | Batch size: {batch_size}
Saved to: {temp_sr_path}
            """.strip()
            
            return info_text, sr_image, comparison_img, error_map_img, temp_sr_path
        
        except Exception as e:
            return f"❌ Error during super-resolution: {str(e)}", None, None, None, None
    
    def create_comparison_image(self, original_img, lr_img, sr_img, zoom_factor=1.0):
        """Create a side-by-side comparison image with zoom capability"""
        try:
            # For zoomed view, crop center regions to maintain pixel detail
            if zoom_factor > 1.0:
                return self.create_zoomed_comparison(original_img, sr_img, zoom_factor)
            
            # Resize images for comparison (limit height to 400px for display)
            max_height = 400
            
            def resize_for_display(img):
                if img.height > max_height:
                    ratio = max_height / img.height
                    new_width = int(img.width * ratio)
                    return img.resize((new_width, max_height), Image.LANCZOS)
                return img
            
            # Only show original and super-resolved images
            orig_display = resize_for_display(original_img)
            sr_display = resize_for_display(sr_img)
            
            # Create comparison image
            total_width = orig_display.width + sr_display.width + 10  # 10px padding between images
            max_height_display = max(orig_display.height, sr_display.height)
            
            comparison = Image.new('RGB', (total_width, max_height_display + 60), 'white')
            
            # Paste images
            x_offset = 0
            
            # Original image
            y_offset = (max_height_display - orig_display.height) // 2
            comparison.paste(orig_display, (x_offset, y_offset + 30))
            x_offset += orig_display.width + 10
            
            # Super-resolved image
            y_offset = (max_height_display - sr_display.height) // 2
            comparison.paste(sr_display, (x_offset, y_offset + 30))
            
            return comparison
            
        except Exception as e:
            print(f"Error creating comparison: {e}")
            return sr_img
    
    def create_zoomed_comparison(self, original_img, sr_img, zoom_factor):
        """Create a zoomed comparison showing magnified pixels"""
        try:
            # For zoom, we want to crop a smaller region and then magnify it
            # This allows seeing individual pixels clearly
            
            # Calculate the size of the region to crop (smaller region for higher zoom)
            base_crop_size = 200  # Base size for the crop region
            crop_width = int(base_crop_size / zoom_factor)
            crop_height = int(base_crop_size / zoom_factor)
            
            # Make sure crop size is at least a few pixels
            crop_width = max(crop_width, 10)
            crop_height = max(crop_height, 10)
            
            # Calculate center crop coordinates for original image
            left = (original_img.width - crop_width) // 2
            top = (original_img.height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            # Crop center region from original
            orig_cropped = original_img.crop((left, top, right, bottom))
            
            # For super-resolved image, find the corresponding region
            scale_x = sr_img.width / original_img.width
            scale_y = sr_img.height / original_img.height
            
            sr_left = int(left * scale_x)
            sr_top = int(top * scale_y)
            sr_right = int(right * scale_x)
            sr_bottom = int(bottom * scale_y)
            
            # Crop corresponding region from super-resolved image
            sr_cropped = sr_img.crop((sr_left, sr_top, sr_right, sr_bottom))
            
            # Now magnify both crops using nearest neighbor to see pixels clearly
            magnify_factor = int(zoom_factor * 2)  # Additional magnification
            display_size = (crop_width * magnify_factor, crop_height * magnify_factor)
            
            # Magnify original crop (nearest neighbor preserves pixel boundaries)
            orig_magnified = orig_cropped.resize(display_size, Image.NEAREST)
            
            # For super-resolved, we want to show it at its native resolution relative to the original
            # Calculate the display size for SR image
            sr_display_width = int(sr_cropped.width * magnify_factor / scale_x)
            sr_display_height = int(sr_cropped.height * magnify_factor / scale_y)
            sr_magnified = sr_cropped.resize((sr_display_width, sr_display_height), Image.NEAREST)
            
            # Create side-by-side comparison
            max_width = max(orig_magnified.width, sr_magnified.width)
            max_height = max(orig_magnified.height, sr_magnified.height)
            
            total_width = max_width * 2 + 20  # Space for both images plus padding
            
            comparison = Image.new('RGB', (total_width, max_height + 60), 'white')
            
            # Center and paste original (left side)
            orig_x = (max_width - orig_magnified.width) // 2
            orig_y = (max_height - orig_magnified.height) // 2 + 30
            comparison.paste(orig_magnified, (orig_x, orig_y))
            
            # Center and paste super-resolved (right side)
            sr_x = max_width + 20 + (max_width - sr_magnified.width) // 2
            sr_y = (max_height - sr_magnified.height) // 2 + 30
            comparison.paste(sr_magnified, (sr_x, sr_y))
            
            return comparison
            
        except Exception as e:
            print(f"Error creating zoomed comparison: {e}")
            return self.create_comparison_image(original_img, None, sr_img, 1.0)

    def create_error_map(self, original_img, sr_img):
        """Create an error map showing the difference between original and super-resolved images"""
        try:
            # Resize original image to match super-resolved image size for comparison
            original_resized = original_img.resize(sr_img.size, Image.LANCZOS)
            
            # Convert to numpy arrays
            orig_array = np.array(original_resized).astype(np.float32)
            sr_array = np.array(sr_img).astype(np.float32)
            
            # Calculate absolute difference
            diff = np.abs(orig_array - sr_array)
            
            # Convert to grayscale for better visualization
            if len(diff.shape) == 3:
                diff_gray = np.mean(diff, axis=2)
            else:
                diff_gray = diff
            
            # Normalize to 0-255 range
            diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
            
            # Create a colormap using matplotlib
            plt.figure(figsize=(8, 6))
            plt.imshow(diff_normalized, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Difference Intensity')
            plt.title('Error Map: Difference between Original and Super-Resolved')
            plt.axis('off')
            
            # Save to our temporary directory and convert to PIL Image
            temp_error_map_path = self._get_temp_path('_error_map.png')
            plt.savefig(temp_error_map_path, bbox_inches='tight', dpi=150)
            plt.close()
            error_map_img = Image.open(temp_error_map_path)
            os.unlink(temp_error_map_path)  # Clean up after loading
            
            return error_map_img
            
        except Exception as e:
            print(f"Error creating error map: {e}")
            # Return a blank image if error map creation fails
            return Image.new('RGB', (400, 300), 'white')

    def update_zoom(self, zoom_factor):
        """Update the comparison view with new zoom level"""
        if self.current_original is None or self.current_sr is None:
            return None
        
        try:
            # Create new comparison with zoom
            comparison_img = self.create_comparison_image(
                self.current_original, None, self.current_sr, zoom_factor
            )
            return comparison_img
        except Exception as e:
            print(f"Error updating zoom: {e}")
            return None

    def cleanup_old_files(self):
        """Clean up old temporary files"""
        try:
            # Remove files older than 1 hour
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    # If file is older than 1 hour, delete it
                    if current_time - os.path.getmtime(filepath) > 3600:
                        os.unlink(filepath)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

def create_interface():
    """Create and configure the Gradio interface"""
    
    app = GradioSuperResolution()
    
    # Add cleanup on a timer
    def periodic_cleanup():
        while True:
            time.sleep(3600)  # Run every hour
            app.cleanup_old_files()
    
    # Start cleanup thread
    import threading
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="LIIF Super-Resolution",
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.2em;
            font-weight: bold;
            margin: 1rem 0 0.5rem 0;
            color: #2563eb;
        }
        .info-box {
            background: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 🚀 LIIF Super-Resolution
            AI-powered image enhancement and upscaling.
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 Model Configuration", elem_classes=["section-header"])
                
                with gr.Accordion("Model Settings", open=False):
                    model_path = gr.Textbox(
                        label="Model Path (enter the path to your trained LIIF model)",
                        value="models/best_model.pth",
                        placeholder="Path to your trained model (.pth file)"
                    )
                    
                    with gr.Row():
                        patch_size = gr.Number(
                            label="Patch Size (should match training config)",
                            value=5,
                            precision=0
                        )
                        hidden_dim = gr.Number(
                            label="Hidden Dimension (should match training config)",
                            value=512,
                            precision=0
                        )
                    
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="No model loaded",
                        interactive=False
                    )
                
                gr.Markdown("## ⚙️ Super-Resolution Settings", elem_classes=["section-header"])
                
                with gr.Group():
                    use_custom_size = gr.Checkbox(
                        label="Use Custom Output Size (override scale factor with specific dimensions)",
                        value=False
                    )
                    
                    with gr.Row():
                        scale_factor = gr.Slider(
                            label="Scale Factor (how much to upscale the image)",
                            minimum=1.5,
                            maximum=8,
                            step=0.1,
                            value=2
                        )
                        batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=64,
                            maximum=2048,
                            step=64,
                            value=1024,
                            info="Adjust based on GPU memory"
                        )
                    
                    with gr.Row(visible=False) as custom_size_row:
                        target_width = gr.Number(
                            label="Target Width",
                            value=1024,
                            precision=0
                        )
                        target_height = gr.Number(
                            label="Target Height",
                            value=1024,
                            precision=0
                        )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📸 Image Upload", elem_classes=["section-header"])
                
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
        
        # Enhance button in center
        with gr.Row():
            with gr.Column():
                process_btn = gr.Button("✨ Enhance Image", variant="primary", size="lg")
        
        # Results section at the bottom
        gr.Markdown("## 🎯 Results", elem_classes=["section-header"])
        
        result_info = gr.Textbox(
            label="Status",
            value="Ready - upload an image and click 'Enhance Image'",
            interactive=False,
            lines=3
        )
        
        with gr.Row():
            with gr.Column():
                output_image = gr.Image(
                    label="Super-Resolved Image",
                    height=400
                )
            
            with gr.Column():
                comparison_image = gr.Image(
                    label="Comparison View",
                    height=400
                )
                
                zoom_slider = gr.Slider(
                    label="Zoom Level (1x = full view, higher = magnify pixels)",
                    minimum=1.0,
                    maximum=16.0,
                    step=0.5,
                    value=1.0,
                    info="Zoom to see pixel-level details - higher values magnify individual pixels"
                )
        
        with gr.Row():
            with gr.Column():
                error_map_image = gr.Image(
                    label="Error Map (Difference Visualization)",
                    height=300
                )
        
        download_image = gr.File(
            label="📥 Download Enhanced Image",
            visible=False
        )
        
        # Event handlers
        def toggle_custom_size(use_custom):
            return gr.Row(visible=use_custom)
        
        use_custom_size.change(
            toggle_custom_size,
            inputs=[use_custom_size],
            outputs=[custom_size_row]
        )
        
        load_btn.click(
            app.load_model,
            inputs=[model_path, patch_size, hidden_dim],
            outputs=[model_status, output_image, comparison_image]
        )
        
        process_btn.click(
            app.super_resolve_image,
            inputs=[
                input_image, scale_factor, target_width, target_height,
                batch_size, use_custom_size
            ],
            outputs=[result_info, output_image, comparison_image, error_map_image, download_image]
        )
        
        zoom_slider.change(
            app.update_zoom,
            inputs=[zoom_slider],
            outputs=[comparison_image]
        )
        
        # Add examples
        gr.Markdown("## 📚 Tips & Examples")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🎯 Best Practices:
                - Start with scale factor 2x for best results
                - Use higher batch sizes for faster processing (if you have enough GPU memory)
                """)
            
            with gr.Column():
                gr.Markdown("""
                ### ⚡ Performance Tips:
                - GPU processing is much faster than CPU
                - Batch size 1024 works well for most GPUs
                """)
    
    return interface

def main():
    """Launch the Gradio interface"""
    interface = create_interface()
    
    print("🚀 Starting LIIF Super-Resolution Interface...")
    print("📊 The interface will open in your web browser")
    print("🔧 Make sure your model file is accessible from the current directory")
    
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        show_error=True,
        favicon_path=None,
        inbrowser=True          # Automatically open in browser
    )

if __name__ == "__main__":
    main()