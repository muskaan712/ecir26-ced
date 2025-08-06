#!/bin/bash
#SBATCH --job-name=AML_simclr             # Job name
#SBATCH --partition=A40short            # Partition
#SBATCH --time=3:00:00                   # Walltime (24 hours)
#SBATCH --gpus=2                          # Number of GPUs
#SBATCH --ntasks=2                        # Number of tasks
#SBATCH --cpus-per-task=10                # CPUs per task for data loading
#SBATCH --output=%x_%j.out                # SLURM output file
#SBATCH --error=%x_%j.err                 # SLURM error file

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Source Conda and activate the environment
source /home/s13mchop/anaconda3/etc/profile.d/conda.sh
conda activate wmt21

# Print Conda environment to verify it's activated
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run your Python script
python /home/s13mchop/LLMs/ecir-ced/2_scaleup_llm/2.4_hybrid/lamma/llama.py
# Deactivate the environment
conda deactivate
