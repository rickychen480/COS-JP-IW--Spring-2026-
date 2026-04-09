#!/bin/bash
#
#SBATCH --job-name=evaluate_steering_only_one_off
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=logs/steering_eval_chunk_%j.out
#SBATCH --error=logs/steering_eval_chunk_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

set -euo pipefail

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

DATASET_NAME="gpt-4o-mini
CHUNK_INDEX=3
TOTAL_CHUNKS=12
TRANSCRIPT_DIR="data/transcripts/${DATASET_NAME}"
OUT_DIR="results/metrics/${DATASET_NAME}/steering_only"
MASKED_PATH="${TRANSCRIPT_DIR}/target_simulations_masked.json"

echo "Provisioning complete."
echo "Running steering-only one-off evaluation"

mkdir -p "${OUT_DIR}"
mkdir -p logs

rm -rf /dev/shm/* 2>/dev/null || true

export TORCHINDUCTOR_CACHE_DIR="/scratch/rc5898/torch_cache_${DATASET_NAME}_${CHUNK_INDEX}_steering_gpu0"
export TRITON_CACHE_DIR="/scratch/rc5898/triton_cache_${DATASET_NAME}_${CHUNK_INDEX}_steering_gpu0"

CUDA_VISIBLE_DEVICES=0 python metrics/main.py \
  --dir "${TRANSCRIPT_DIR}/" \
  --out "${OUT_DIR}/steering_only_chunk_${CHUNK_INDEX}.csv" \
  --chunk_index "$CHUNK_INDEX" \
  --total_chunks "$TOTAL_CHUNKS" \
  --masked_path "$MASKED_PATH" \
  --embedding_device cuda \
  --embedding_batch_size 256 \
  --steering_only

wait
echo "Steering-only one-off complete!"
