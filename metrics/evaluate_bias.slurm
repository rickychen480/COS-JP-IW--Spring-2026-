#!/bin/bash
#
#SBATCH --job-name=evaluate_bias
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 8                        
#SBATCH --mem=96G        
#SBATCH -t 12:00:00                   
#SBATCH --output=logs/slurm-eval-%j.out         
#SBATCH --error=logs/slurm-eval-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

source ~/.hf_token
hf download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --local-dir /n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4
MODEL_ID="/n/fs/dynamicbias/models/Llama-3.1-70B-Instruct-AWQ-INT4"

echo "Running evaluation on Llama 8B Transcripts..."

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_0
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_0
CUDA_VISIBLE_DEVICES=0,1 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-8B-Instruct \
  --out results/Llama-8B_evaluation_results.csv \
  --judge_model $MODEL_ID \
  --tensor_parallel_size 2 &

sleep 300

echo "Running evaluation on Llama 70B Transcripts..."

export TORCHINDUCTOR_CACHE_DIR=/scratch/rc5898/torch_cache_70b_1
export TRITON_CACHE_DIR=/scratch/rc5898/triton_cache_70b_1
CUDA_VISIBLE_DEVICES=2,3 python metrics/main.py \
  --dir data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4 \
  --out results/Llama-70B_evaluation_results.csv \
  --judge_model $MODEL_ID \
  --tensor_parallel_size 2 &

wait

echo "Evaluations complete!"