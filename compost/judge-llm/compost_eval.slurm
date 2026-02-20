#!/bin/bash
#SBATCH --job-name=compost_eval
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err 

# Route massive caches to the COMPUTE node's scratch
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
export HF_HUB_CACHE=/scratch/$USER/hf_cache

eval "$(conda shell.bash hook)"
conda create --prefix /scratch/$USER/env python=3.12 -y
conda activate /scratch/$USER/env

python -m pip install --upgrade pip
pip install torch transformers vllm huggingface_hub --ignore-installed packaging

echo "Provisioning local scratch on compute node..."
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4 \

echo "Starting CoMPosT Evaluation with Llama 70B AWQ on 2 GPUs..."

python compost/judge-llm/compost_evaluator.py \
    --data results/pilot_results.json \
    --out results/compost/judge-llm/compost_llama70b.json \
    --judge /scratch/rc5898/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4

echo "Evaluation Complete!"