#!/bin/bash
#
#SBATCH --job-name=repr_bias
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --output=logs/repr_bias_eval_%j.out
#SBATCH --error=logs/repr_bias_eval_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

set -euo pipefail

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"

echo "Provisioning complete."
echo "Running representational metrics (single chunk)."

mkdir -p results/metrics/
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/ \
  --out results/metrics/test-repr-bias.csv \
  --judge_model "${MODEL_ID}" \
  --tensor_parallel_size 1 \
  --chunk_index 0 \
  --total_chunks 1 \
  --masked_path data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json

wait
echo "Representational run complete!"

echo "Evaluations complete!"
