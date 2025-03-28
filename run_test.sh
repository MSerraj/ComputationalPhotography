#!/bin/bash

echo "Setting up INR Super-Resolution test environment..."

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: 'data' directory not found!"
    echo "Please create a data directory with your test images before proceeding."
    exit 1
fi

# Check if the test image exists
if [ ! -f "data/0002.png" ]; then
    echo "Warning: Test image 'data/0002.png' not found."
    
    # Find any PNG in the data directory
    TEST_IMAGE=$(find data -name "*.png" | head -n 1)
    
    if [ -z "$TEST_IMAGE" ]; then
        echo "Error: No PNG images found in the data directory."
        echo "Please add at least one PNG image to the data directory."
        exit 1
    else
        echo "Using alternative image: $TEST_IMAGE"
        # Update the SLURM script to use this image
        sed -i "s|data/0002.png|$TEST_IMAGE|g" test_inr_super_resolution.slurm
    fi
fi

# Ensure scripts are executable
chmod +x test_inr_super_resolution.slurm
chmod +x inr_super_resolution.slurm

echo "Submitting test job to SLURM..."
sbatch test_inr_super_resolution.slurm

echo "Job submitted. You can check its status with:"
echo "  squeue -u $USER"
echo
echo "Once complete, check the test_results directory for outputs." 