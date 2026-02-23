#!/bin/bash
#
#SBATCH --job-name=evaluate_bias
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4                          
#SBATCH --mem=32G        
#SBATCH -t 04:00:00                   
#SBATCH --output=logs/slurm-eval-%j.out         
#SBATCH --error=logs/slurm-eval-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

# Route massive caches to the COMPUTE node's scratch
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
export HF_HUB_CACHE=/scratch/$USER/hf_cache

echo "Creating temporary Conda environment on local scratch..."
eval "$(conda shell.bash hook)"
conda create --prefix /scratch/$USER/env python=3.12 -y
conda activate /scratch/$USER/env

echo "Installing packages..."
python -m pip install --upgrade pip
pip install pandas numpy sentence-transformers

echo "Running evaluation on Llama 8B Transcripts..."
python metrics/main.py \
  --dir data/transcripts/Llama-3.1-8B-Instruct \
  --out results/Llama-8B_evaluation_results.csv

echo "Running evaluation on Llama 70B Transcripts..."
python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --out results/Llama-70B_evaluation_results.csv

echo "Evaluations complete!"