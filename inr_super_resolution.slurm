#!/bin/bash -l
#SBATCH --workdir /scratch/izar/$USER
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --time 3:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs413
#SBATCH --qos cs413
#SBATCH --reservation cs413
#SBATCH --job-name inr_sr
#SBATCH --output=%x_%j.out

echo "Job started on $(hostname) at $(date)"
echo "GPU assigned: $(nvidia-smi -L)"

# Create a directory for the run
RUN_DIR="/scratch/izar/$USER/inr_super_resolution_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RUN_DIR
echo "Created run directory: $RUN_DIR"

# Copy necessary files to the scratch directory
cp -r $SLURM_SUBMIT_DIR/data $RUN_DIR/
cp $SLURM_SUBMIT_DIR/inr_super_resolution.py $RUN_DIR/
cp $SLURM_SUBMIT_DIR/requirements.txt $RUN_DIR/

# Change to the run directory
cd $RUN_DIR
echo "Changed to directory: $RUN_DIR"

# Setup Python environment (you might need to adjust this based on your cluster setup)
module load gcc/9.3.0
module load python/3.9.0
module load cuda/11.3.1

# Create and activate a virtual environment
python -m venv env
source env/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Print Python and torch versions for debugging
python --version
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# Run the script
echo "Starting INR super-resolution training..."
python inr_super_resolution.py \
    --image data/0002.png \
    --output_dir results \
    --downscale_factor 4 \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_epochs 500 \
    --batch_size 8192 \
    --learning_rate 5e-4 \
    --weight_decay 1e-6 \
    --scheduler_gamma 0.98

# Copy results back to the submission directory
mkdir -p $SLURM_SUBMIT_DIR/results
cp -r results/* $SLURM_SUBMIT_DIR/results/

echo "Job finished at $(date)" 