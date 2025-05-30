# 🚀 LIIF Super-Resolution Web Interface

A beautiful, user-friendly web interface for your trained LIIF super-resolution model. No command-line knowledge required!

![Interface Preview]()

## ✨ Features

- 📸 **Drag & Drop Upload** - Simply drag your images into the interface
- ⚙️ **Easy Configuration** - Intuitive sliders and controls for all parameters
- 🔄 **Real-time Preview** - See results immediately with side-by-side comparison
- 📊 **Progress Tracking** - Visual progress bars during processing
- 💾 **One-Click Download** - Download enhanced images with a single click

## 🚀 Quick Start

### Option 1: Simple Launch (Recommended)

```bash
python launch_ui.py
```

This will automatically:
- Check and install requirements
- Find your model file
- Launch the web interface
- Open it in your browser

### Option 2: Manual Setup

1. **Install requirements:**
   ```bash
   pip install -r requirements_ui.txt
   ```

2. **Launch interface:**
   ```bash
   python gradio_ui.py
   ```

## 📋 Requirements

The launcher will automatically install these if missing:
- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `pillow>=8.0.0`
- `numpy>=1.20.0`
- `matplotlib>=3.3.0`
- `tqdm>=4.60.0`
- `gradio>=4.0.0`

## 🎯 How to Use

### 1. Load Your Model
- Enter the path to your trained model (defaults to `best_model.pth`)
- Set patch size and hidden dimension (should match your training config)
- Click "🔄 Load Model"

### 2. Upload an Image
- Drag and drop any image file (JPG, PNG, etc.)
- Or click to browse and select a file

### 3. Configure Settings
- **Scale Factor**: Choose any scale factor from 1.5x to 8x
- **Custom Size**: Override scale factor with specific dimensions
- **Batch Size**: Adjust based on your GPU memory (1024 is usually good)

### 4. Enhance!
- Click "✨ Enhance Image"
- Watch the progress bar
- View results in real-time

### 5. Download
- Click the download button to save your enhanced image
- Images are saved in high-quality PNG format

## ⚙️ Configuration Options

### Model Parameters
- **Model Path**: Path to your `.pth` model file
- **Patch Size**: Should match training (usually 5)
- **Hidden Dimension**: Should match training (usually 512)

### Processing Settings
- **Scale Factor**: 1.5x to 8x (2x recommended for best quality)
- **Custom Size**: Specify exact output dimensions
- **Batch Size**: 64-2048 (adjust based on GPU memory)

## 💡 Tips & Best Practices

### 🎯 For Best Results:
- Start with 2x scale factor for optimal quality
- Use images similar to your training data
- Ensure good lighting and reasonable quality input images

### ⚡ Performance Optimization:
- **GPU**: Use CUDA if available (much faster)
- **Batch Size**: 1024 works well for most GPUs
- **Memory**: Reduce batch size if you get out-of-memory errors
- **Large Images**: Use batch size 512 or lower for 4K+ outputs

### 🔧 Troubleshooting:

#### Model Loading Issues
- Verify model file path is correct
- Check patch_size and hidden_dim match training
- Ensure model file isn't corrupted

#### Memory Problems
- Reduce batch size (try 512, 256, 128)
- Use smaller input images
- Close other GPU-intensive applications

#### Poor Quality Results
- Check if input image type matches training data
- Try different scale factors
- Verify model was trained properly

## 🌐 Network Access

The interface runs on `http://localhost:7860` by default.

## 📁 File Structure

```
your-project/
├── gradio_ui.py          # Main UI interface
├── launch_ui.py          # Easy launcher script
├── inference_liif.py     # Inference engine
├── train_liif.py         # Training script
├── requirements_ui.txt   # UI dependencies
├── temp_images           # Temporary image storage
├── models/               # Trained models
├──---- best_model.pth        # Your trained model
└── README.md          # This documentation
```

## 🔄 Advanced Usage

### Batch Processing
For processing multiple images:
1. Use the programmatic interface in `inference_liif.py`
2. Or load images one by one in the web interface

### Integration
To integrate the inference into your own code:

```python
from inference_liif import SuperResolutionInference

# Initialize once
sr = SuperResolutionInference("best_model.pth")

# Process multiple images
for image_path in image_list:
    enhanced, _, _ = sr.super_resolve(image_path, scale_factor=2)
    enhanced.save(f"enhanced_{image_path}")
```

## 🎉 Enjoy!