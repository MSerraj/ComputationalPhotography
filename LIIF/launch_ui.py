#!/usr/bin/env python3

"""
Launcher script for LIIF Super-Resolution UI
Handles setup and launches the Gradio web interface.
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 
        'matplotlib', 'tqdm', 'gradio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("📦 Installing missing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_ui.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_model_file():
    """Check if model file exists"""
    model_files = ["best_model.pth", "final_model.pth"]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model file: {model_file}")
            return True
    
    print("⚠️  Model file not found!")
    print("Expected files: best_model.pth or final_model.pth")
    print("You can still launch the UI and specify the model path manually.")
    return False

def main():
    """Main launcher function"""
    print("🚀 LIIF Super-Resolution UI Launcher")
    print("="*50)
    
    # Check requirements
    print("\n📋 Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        
        install_choice = input("🔧 Would you like to install missing packages? (y/n): ")
        if install_choice.lower() in ['y', 'yes']:
            if not install_requirements():
                print("❌ Failed to install requirements. Please install manually:")
                print("pip install -r requirements_ui.txt")
                return
        else:
            print("❌ Cannot proceed without required packages.")
            print("Please run: pip install -r requirements_ui.txt")
            return
    else:
        print("✅ All requirements satisfied!")
    
    # Check model file
    print("\n🔍 Checking for model files...")
    check_model_file()
    
    # Check inference script
    if not os.path.exists("inference_liif.py"):
        print("❌ Error: inference_liif.py not found!")
        print("Make sure all files are in the same directory.")
        return
    
    print("\n🌐 Starting web interface...")
    print("📱 The interface will open in your browser at: http://localhost:7860")
    print("🔧 You can also access it from other devices on your network")
    print("\n" + "="*50)
    
    # Import and launch
    try:
        from gradio_ui import main as launch_ui
        launch_ui()
    except ImportError as e:
        print(f"❌ Error importing UI module: {e}")
        print("Make sure gradio_ui.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

if __name__ == "__main__":
    main() 