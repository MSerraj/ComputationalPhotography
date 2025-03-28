#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 8G
#SBATCH --time 0:30:00
#SBATCH --gres gpu:1
#SBATCH --account cs413
#SBATCH --qos cs413
#SBATCH --job-name inr_test
#SBATCH --output=%x_%j.out

echo "TEST JOB: Running with minimal settings to verify setup"
echo "Job started on $(hostname) at $(date)"
echo "GPU assigned: $(nvidia-smi -L)"

# Create a temporary directory for the test
TEST_DIR="/scratch/izar/$USER/inr_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEST_DIR
echo "Created test directory: $TEST_DIR"

# Copy necessary files to the scratch directory
cp -r $SLURM_SUBMIT_DIR/data $TEST_DIR/
cp $SLURM_SUBMIT_DIR/inr_super_resolution.py $TEST_DIR/
cp $SLURM_SUBMIT_DIR/requirements.txt $TEST_DIR/

# Change to the test directory
cd $TEST_DIR
echo "Changed to directory: $TEST_DIR"

# Setup Python environment
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

# Run the script with minimal settings for quick testing
echo "Starting TEST run of INR super-resolution..."
python inr_super_resolution.py \
    --image data/0002.png \
    --output_dir test_results \
    --downscale_factor 4 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_epochs 50 \
    --batch_size 4096 \
    --learning_rate 5e-4

# Copy only essential results back to submission directory
mkdir -p $SLURM_SUBMIT_DIR/test_results
cp -r test_results/metrics.txt test_results/reconstructed.png $SLURM_SUBMIT_DIR/test_results/

echo "Test job finished at $(date)"
echo "Check test_results directory for outputs"
echo "If everything works, you can run the full job with: sbatch inr_super_resolution.slurm" 