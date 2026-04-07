#!/bin/bash
#
#SBATCH --job-name=evaluate_bias-one_off
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --output=logs/bias_eval_70b_%j_chunk.out
#SBATCH --error=logs/bias_eval_70b_%j_chunk.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

# Define the specific chunk index for this one-off run
CHUNK_INDEX=0

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
echo "Running evaluation chunk: Chunk Index = $CHUNK_INDEX"

mkdir -p results/metrics/
mkdir -p logs

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_${CHUNK_INDEX}_metrics_gpu01
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_${CHUNK_INDEX}_metrics_gpu01
CUDA_VISIBLE_DEVICES=0,1 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/ \
  --out results/metrics/dynamic_bias_results_chunk_${CHUNK_INDEX}.csv \
  --judge_model $MODEL_ID \
  --tensor_parallel_size 2 \
  --chunk_index $CHUNK_INDEX \
  --total_chunks 12 \
  --masked_path data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json \
  --embedding_device cuda \
  --embedding_batch_size 256

wait
echo "Chunk $CHUNK_INDEX complete!"

echo "Evaluation complete!"
