#!/bin/bash
#
#SBATCH --job-name=evaluate_steering_only
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=logs/steering_eval_%A_%a.out
#SBATCH --error=logs/steering_eval_%A_%a.err
#SBATCH --array=0-11
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

set -euo pipefail

# Optional memory allocator tuning for torch/sentence-transformers
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

DATASET_NAME="gpt-4o-mini"
TOTAL_CHUNKS=12
TRANSCRIPT_DIR="data/transcripts/${DATASET_NAME}"
OUT_DIR="results/metrics/${DATASET_NAME}/steering_only"
MASKED_PATH="${TRANSCRIPT_DIR}/target_simulations_masked.json"

echo "Provisioning complete."
echo "Running steering-only chunk: Array Task ID = $SLURM_ARRAY_TASK_ID"

mkdir -p "${OUT_DIR}"
mkdir -p logs

# Clear any orphaned shared memory from previous crashed runs
rm -rf /dev/shm/* 2>/dev/null || true

export TORCHINDUCTOR_CACHE_DIR="/scratch/rc5898/torch_cache_${DATASET_NAME}_${SLURM_ARRAY_TASK_ID}_steering_gpu0"
export TRITON_CACHE_DIR="/scratch/rc5898/triton_cache_${DATASET_NAME}_${SLURM_ARRAY_TASK_ID}_steering_gpu0"

CUDA_VISIBLE_DEVICES=0 python metrics/main.py \
  --dir "${TRANSCRIPT_DIR}/" \
  --out "${OUT_DIR}/steering_only_chunk_${SLURM_ARRAY_TASK_ID}.csv" \
  --chunk_index "$SLURM_ARRAY_TASK_ID" \
  --total_chunks "$TOTAL_CHUNKS" \
  --masked_path "$MASKED_PATH" \
  --embedding_device cuda \
  --embedding_batch_size 256 \
  --steering_only

wait
echo "Chunk $SLURM_ARRAY_TASK_ID complete!"

echo "Steering-only evaluation complete!"
