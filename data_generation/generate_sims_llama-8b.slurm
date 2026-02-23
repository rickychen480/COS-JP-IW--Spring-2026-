#!/bin/bash
#
#SBATCH --job-name=generate_sims
#SBATCH -N 1                          ## Request 1 node
#SBATCH --gres=gpu:3                  ## Request 3 GPUs (one for each script)
#SBATCH -c 12                         ## Request 12 CPUs (4 per script)
#SBATCH --mem=96G                     ## Request 96GB RAM (plenty for 3x 8B models)
#SBATCH -t 12:00:00                   ## Request 12 hours of walltime
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
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /scratch/$USER/hf_models/Llama-3.1-8B-Instruct

mkdir -p results/pilot/

echo "Starting all 3 generation tasks in parallel..."

# Run Target Simulations on GPU 0 (The '&' at the end sends it to the background)
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --model /scratch/rc5898/hf_models/Llama-3.1-8B-Instruct \
  --limit 1000 &

# Run Control Simulations on GPU 1
CUDA_VISIBLE_DEVICES=1 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
  --model /scratch/rc5898/hf_models/Llama-3.1-8B-Instruct \
  --limit 1000 &

# Run Default Topics on GPU 2
CUDA_VISIBLE_DEVICES=2 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
  --model /scratch/rc5898/hf_models/Llama-3.1-8B-Instruct \
  --limit 1000 &

# Don't kill the job until all 3 background tasks finish
wait

echo "All 3 datasets generated successfully!"
