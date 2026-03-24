#!/bin/bash
#SBATCH --job-name=generate_sims_70b
#SBATCH -N 1
#SBATCH --gres=gpu:6                 
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 16:00:00
#SBATCH --output=logs/sim_70b_%A_%a.out
#SBATCH --error=logs/sim_70b_%A_%a.err
#SBATCH --array=0-11
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"

echo "Provisioning complete."
echo "Running simulation chunk: Array Task ID = $SLURM_ARRAY_TASK_ID"

mkdir -p data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/
mkdir -p logs


# Run Target Simulations on GPUs 0 and 1
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu01
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu01
CUDA_VISIBLE_DEVICES=0,1 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12 &

sleep 300

# Run Control Simulations on GPUs 2 and 3
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu23
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu23
CUDA_VISIBLE_DEVICES=2,3 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12 &

sleep 300

# Run Default Topics on GPUs 4 and 5
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu45
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${SLURM_ARRAY_TASK_ID}_gpu45
CUDA_VISIBLE_DEVICES=4,5 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics_chunk_${SLURM_ARRAY_TASK_ID}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12 &

wait

echo "Chunk $SLURM_ARRAY_TASK_ID (all 3 datasets) complete!"