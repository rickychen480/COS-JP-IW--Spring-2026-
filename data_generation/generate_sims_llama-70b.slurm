#!/bin/bash
#
#SBATCH --job-name=generate_sims
#SBATCH -N 1
#SBATCH --gres=gpu:6                 
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --output=logs/slurm-%j.out         
#SBATCH --error=logs/slurm-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"

echo "Provisioning complete. Starting all 3 generation tasks in parallel..."
mkdir -p data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/

# Run Target Simulations on GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --limit 1000 &

# Run Control Simulations on GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --limit 1000 &

# Run Default Topics on GPUs 4 and 5
CUDA_VISIBLE_DEVICES=4,5 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 &

wait

echo "All 3 datasets generated successfully!"