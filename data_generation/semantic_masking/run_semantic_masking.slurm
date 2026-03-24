#!/bin/bash
#SBATCH --job-name=semantic_masking
#SBATCH -N 1                          ## Request 1 node 
#SBATCH --gres=gpu:2                  ## Request 2 GPUs
#SBATCH -c 8                          ## Request 8 CPUs
#SBATCH --mem=64G                     ## Request 64GB RAM
#SBATCH -t 04:00:00                   ## Request 2 hours of walltime
#SBATCH --output=logs/masking_%j.out
#SBATCH --error=logs/masking_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-8B-Instruct"

echo "Provisioning complete. Starting vLLM Semantic Masking..."

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_masking
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_masking
CUDA_VISIBLE_DEVICES=0,1 python data_generation/semantic_masking/semantic_masking.py \
  --input data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --output data/transcripts/Llama-3.1-8B-Instruct/target_simulations_masked.json \
  --model $MODEL_ID

echo "Semantic Masking complete!"