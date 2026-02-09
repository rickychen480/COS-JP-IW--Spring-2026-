#!/bin/bash
#SBATCH --job-name=intersectional-bias-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=12:00:00           # 4 hour limit
#SBATCH --partition=all           # Check specific Neuronic partition names (often 'all' or 'gpu')
#SBATCH --output=logs/%x-%j.out

# 1. Load Modules (Adjust based on 'module avail' on Neuronic)
module load anaconda3
module load cuda/12.1

# 2. Activate Environment
conda activate bias-eval-env

# 3. Start vLLM Server in background (The Target Model)
# This assumes you have access to the model weights on the cluster storage
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 &

# Wait for server to start
sleep 60

# 4. Run the Controller Script
python run_simulation.py