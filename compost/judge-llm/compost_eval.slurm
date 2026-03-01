#!/bin/bash
#SBATCH --job-name=compost_eval
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err 
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu   

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"

echo "Provisioning complete. Starting CoMPosT Evaluation with Llama 70B AWQ on 2 GPUs..."

mkdir -p results/compost/judge-llm/
python compost/judge-llm/compost_evaluator.py \
    --data results/pilot_results.json \
    --out results/compost/judge-llm/compost_llama70b.json \
    --judge $MODEL_ID

echo "Evaluation Complete!"