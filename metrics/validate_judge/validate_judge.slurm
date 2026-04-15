#!/bin/bash
#
#SBATCH --job-name=validate_judge_1000
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH --output=logs/validate_judge_%j.out
#SBATCH --error=logs/validate_judge_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu

set -euo pipefail

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token

mkdir -p logs
mkdir -p data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4

MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"
INPUT_JSON="data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json"
OUTPUT_JSON="data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/judge_validation_sample_1000.json"

# Optional: download model if missing
if [ ! -d "$MODEL_ID" ]; then
  hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir "$MODEL_ID"
fi

echo "Running validate_judge.py..."
python metrics/validate_judge/validate_judge.py \
  --input "$INPUT_JSON" \
  --output "$OUTPUT_JSON" \
  --model "$MODEL_ID" \
  --num-samples 1000 \
  --seed 42 \
  --batch-size 64 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --judge-max-tokens 512

echo "Done: $OUTPUT_JSON"