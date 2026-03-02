#!/bin/bash
#
#SBATCH --job-name=generate_sims
#SBATCH -N 1                          ## Request 1 node
#SBATCH --gres=gpu:3                  ## Request 3 GPUs (one for each script)
#SBATCH -c 12                         ## Request 12 CPUs (4 per script)
#SBATCH --mem=96G                     ## Request 96GB RAM (plenty for 3x 8B models)
#SBATCH -t 12:00:00                   ## Request 12 hours of walltime
#SBATCH --output=logs/slurm-%j.out         
#SBATCH --error=logs/slurm-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /n/fs/dynamicbias/models/Llama-3.1-8B-Instruct
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-8B-Instruct"

echo "Provisioning complete. Starting all 3 generation tasks in parallel..."
mkdir -p data/transcripts/Llama-3.1-8B-Instruct/

# Run Target Simulations on GPU 0
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --model $MODEL_ID &

# Run Control Simulations on GPU 1
CUDA_VISIBLE_DEVICES=1 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
  --model $MODEL_ID &

# Run Default Topics on GPU 2
CUDA_VISIBLE_DEVICES=2 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
  --model $MODEL_ID &

# Don't kill the job until all 3 background tasks finish
wait

echo "All 3 datasets generated successfully!"
