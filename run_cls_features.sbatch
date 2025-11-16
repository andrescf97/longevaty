#!/bin/bash
#SBATCH --job-name=cls_features
#SBATCH --partition=universe          # change if your GPU partition is different
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/miltank/users/forerove/logs/cls_features_%j.out

# For safety: avoid noisy multithreading
export OMP_NUM_THREADS=4

# Go to project directory
cd /vol/miltank/projects/practical_wise2526/lung_cancer_lft/adlm_lft

# Load uv
module load python/uv

# print some debug info
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not available"

# Run the script with uv
uv run CLS_feature_extractor.py
