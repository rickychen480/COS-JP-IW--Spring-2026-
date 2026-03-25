#!/bin/bash
#
#SBATCH --job-name=evaluate_bias
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH --output=logs/bias_eval_%A_%a.out
#SBATCH --error=logs/bias_eval_%A_%a.err
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
echo "Running evaluation chunk: Array Task ID = $SLURM_ARRAY_TASK_ID"

mkdir -p results/metrics/
mkdir -p logs

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${SLURM_ARRAY_TASK_ID}_metrics_gpu01
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${SLURM_ARRAY_TASK_ID}_metrics_gpu01
CUDA_VISIBLE_DEVICES=0,1 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/
  --out results/metrics/dynamic_bias_results_chunk_${SLURM_ARRAY_TASK_ID}.csv \
  --judge_model $MODEL_ID \
  --tensor_parallel_size 2 \
  --chunk_index $SLURM_ARRAY_TASK_ID \
  --total_chunks 12 \
  --masked_path data/transcripts/Llama-3.1-70B-Instruct/target_simulations_masked.json

wait
echo "Chunk $SLURM_ARRAY_TASK_ID complete!"

echo "Evaluations complete!"