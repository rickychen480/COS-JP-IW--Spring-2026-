#!/bin/bash
#
#SBATCH --job-name=generate_sims
#SBATCH -N 1
#SBATCH --gres=gpu:4                  
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%j.out         
#SBATCH --error=slurm-%j.err   
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
pip install torch transformers vllm huggingface_hub --ignore-installed packaging

echo "Provisioning local scratch on compute node..."
source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4

mkdir -p results/pilot/

echo "Starting all 3 generation tasks in parallel..."

# Run Target Simulations on GPU 0 (The '&' at the end sends it to the background)
CUDA_VISIBLE_DEVICES=0,1 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
  --model /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quant awq \
  --tp 2 \
  --limit 1000 &

# Run Control Simulations on GPU 1
CUDA_VISIBLE_DEVICES=2,3 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
  --model /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quant awq \
  --tp 2 \
  --limit 1000 &

# Wait for Batch 1 to finish completely
wait
echo "Batch 1 finished! Starting Batch 2 (Default Topics)..."

CUDA_VISIBLE_DEVICES=0,1 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
  --model /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quant awq \
  --tp 2 &

wait

echo "All 3 datasets generated successfully!"
