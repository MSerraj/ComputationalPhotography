#!/usr/bin/env python3

import numpy as np
import torch
import os
import subprocess

# Print environment information
print("=" * 50)
print("ENVIRONMENT INFORMATION")
print("=" * 50)
print(f"HOSTNAME: {os.uname().nodename}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not running in SLURM')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Check CUDA availability
print("\n" + "=" * 50)
print("CUDA INFORMATION")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Get GPU memory information
    print("\n" + "=" * 50)
    print("GPU MEMORY INFORMATION")
    print("=" * 50)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Get system memory information
print("\n" + "=" * 50)
print("SYSTEM MEMORY INFORMATION")
print("=" * 50)
try:
    free_output = subprocess.check_output(["free", "-h"], universal_newlines=True)
    print(free_output)
except Exception as e:
    print(f"Error running 'free' command: {e}")

# Test a simple GPU operation if CUDA is available
if torch.cuda.is_available():
    print("\n" + "=" * 50)
    print("RUNNING GPU TEST")
    print("=" * 50)
    
    # Create a tensor on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    
    # Perform matrix multiplication
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    z = torch.matmul(x, y)
    end_time.record()
    
    # Synchronize CUDA events
    torch.cuda.synchronize()
    
    # Calculate elapsed time
    elapsed_time = start_time.elapsed_time(end_time)
    print(f"Matrix multiplication time: {elapsed_time:.2f} ms")
    print(f"Result shape: {z.shape}")
    print("GPU test completed successfully!") 