#!/bin/bash
#SBATCH --job-name=LLM_eval             # Job name
#SBATCH --partition=A100medium            # Partition
#SBATCH --time=8:00:00                   # Walltime (24 hours)
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
export HF_TOKEN="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"
export OAPI=""
python /home/s13mchop/LLMs/ced/4_finetuning/4.1_o4_mini/100_perc/inference/inf.py
# Deactivate the environment
conda deactivate
