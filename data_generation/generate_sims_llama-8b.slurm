#!/bin/bash
#SBATCH --job-name=generate_sims_8b
#SBATCH -N 1                          ## Request 1 node
#SBATCH --gres=gpu:3                  ## Request 3 GPUs (one for each script)
#SBATCH -c 12                         ## Request 12 CPUs (4 per script)
#SBATCH --mem=96G                     ## Request 96GB RAM (plenty for 3x 8B models)
#SBATCH -t 06:00:00                   ## Request 12 hours of walltime
#SBATCH --output=logs/sim_8b_%A_%a.out
#SBATCH --error=logs/sim_8b_%A_%a.err
#SBATCH --array=0-11
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

echo "Provisioning complete."
echo "Running simulation chunk: Array Task ID = $SLURM_ARRAY_TASK_ID"

mkdir -p data/transcripts/Llama-3.1-8B-Instruct/
mkdir -p logs

# Run Target Simulations on GPU 0
CUDA_VISIBLE_DEVICES=0 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/target_simulations_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12

echo "Target chunk $SLURM_ARRAY_TASK_ID complete. Starting control..."

# Run Control Simulations on GPU 1
CUDA_VISIBLE_DEVICES=1 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/control_simulations_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12

echo "Control chunk $SLURM_ARRAY_TASK_ID complete. Starting default topics..."

# Run Default Topics on GPU 2
CUDA_VISIBLE_DEVICES=2 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-8B-Instruct/default_topics_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12

echo "Chunk $SLURM_ARRAY_TASK_ID (all 3 datasets) complete!"
