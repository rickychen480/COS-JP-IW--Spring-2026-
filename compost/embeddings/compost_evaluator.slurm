#!/bin/bash
#
#SBATCH --job-name=evaluate_compost
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4                          
#SBATCH --mem=32G        
#SBATCH -t 04:00:00                   
#SBATCH --output=logs/slurm-compost-%j.out         
#SBATCH --error=logs/slurm-compost-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

echo "Running CoMPosT on Llama 8B Transcripts..."
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-8B-Instruct/control_simulations.json \
        data/transcripts/Llama-3.1-8B-Instruct/default_topics.json \
        data/transcripts/Llama-3.1-8B-Instruct/target_simulations.json \
  --enable-semantic-masking \
  --cv-strategy GroupKFold \
  --enable-intersectional-eval \
  --output-dir results/compost/embeddings/llama-8b

echo "Running CoMPosT on Llama 70B Transcripts..."
python compost/embeddings/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics.json \
        data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
  --enable-semantic-masking \
  --cv-strategy GroupKFold \
  --enable-intersectional-eval \
  --output-dir results/compost/embeddings/llama-70b

echo "CoMPosT complete!"