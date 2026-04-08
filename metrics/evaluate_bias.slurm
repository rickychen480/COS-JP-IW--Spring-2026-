#!/bin/bash
#
#SBATCH --job-name=evaluate_bias
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=logs/bias_eval_%A_%a.out
#SBATCH --error=logs/bias_eval_%A_%a.err
#SBATCH --array=0-11
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

export VLLM_DISABLE_COMPILE_CACHE=1

# Ensure new allocator config and unset deprecated one
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"
if [ ! -d "$MODEL_ID" ]; then
  hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir "$MODEL_ID"
fi

echo "Provisioning complete."
echo "Running evaluation chunk: Array Task ID = $SLURM_ARRAY_TASK_ID"

mkdir -p results/metrics/Llama-3.1-70B-Instruct-AWQ-INT4/
mkdir -p logs

# Clear any orphaned shared memory from previous crashed runs
rm -rf /dev/shm/* 2>/dev/null || true

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${SLURM_ARRAY_TASK_ID}_metrics_gpu01
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${SLURM_ARRAY_TASK_ID}_metrics_gpu01
CUDA_VISIBLE_DEVICES=0,1 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/ \
  --out results/metrics/Llama-3.1-70B-Instruct-AWQ-INT4/dynamic_bias_results_chunk_${SLURM_ARRAY_TASK_ID}.csv \
  --judge_model $MODEL_ID \
  --tensor_parallel_size 2 \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12 \
  --masked_path data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json \
  --embedding_device cuda \
  --embedding_batch_size 256

wait
echo "Chunk $SLURM_ARRAY_TASK_ID complete!"

echo "Evaluations complete!"