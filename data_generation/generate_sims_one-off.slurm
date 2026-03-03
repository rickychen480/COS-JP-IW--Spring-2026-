#!/bin/bash
#SBATCH --job-name=generate_sims_70b_chunk5
#SBATCH -N 1
#SBATCH --gres=gpu:6                 
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --output=logs/sim_70b_%j_chunk5.out
#SBATCH --error=logs/sim_70b_%j_chunk5.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

# Define the specific chunk index for this one-off run
CHUNK_INDEX=5

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
echo "Running one-off simulation chunk: Chunk Index = $CHUNK_INDEX"

mkdir -p data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/
mkdir -p logs

# Run Target Simulations on GPUs 0 and 1
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${CHUNK_INDEX}_gpu01
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${CHUNK_INDEX}_gpu01
CUDA_VISIBLE_DEVICES=0,1 python controller.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_chunk_${CHUNK_INDEX}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $CHUNK_INDEX \
  --total_chunks 12 &

sleep 300

# Run Control Simulations on GPUs 2 and 3
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${CHUNK_INDEX}_gpu23
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${CHUNK_INDEX}_gpu23
CUDA_VISIBLE_DEVICES=2,3 python controller.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations_chunk_${CHUNK_INDEX}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $CHUNK_INDEX \
  --total_chunks 12 &

sleep 300

# Run Default Topics on GPUs 4 and 5
export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${CHUNK_INDEX}_gpu45
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${CHUNK_INDEX}_gpu45
CUDA_VISIBLE_DEVICES=4,5 python controller.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics_chunk_${CHUNK_INDEX}.json \
  --model $MODEL_ID \
  --quant awq \
  --tp 2 \
  --chunk_index $CHUNK_INDEX \
  --total_chunks 12 &

wait

echo "Chunk $CHUNK_INDEX (all 3 datasets) complete!"