#!/bin/bash
#
#SBATCH --job-name=evaluate_compost
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4                          
#SBATCH --mem=32G        
#SBATCH -t 4:00:00                   
#SBATCH --output=logs/slurm-compost-%j.out         
#SBATCH --error=logs/slurm-compost-%j.err   
#SBATCH --mail-type=end
#SBATCH --mail-user=ricky.chen@princeton.edu       

echo "Initializing environment..."

cd ~/COS-JP
eval "$(conda shell.bash hook)"
conda activate ./env

echo "Running CoMPosT on Llama 8B Transcripts..."
python compost/compost_evaluator.py \
  --data data/transcripts/Llama-3.1-8B-Instruct/control_simulations_masked.json \
        data/transcripts/Llama-3.1-8B-Instruct/default_topics_masked.json \
        data/transcripts/Llama-3.1-8B-Instruct/target_simulations_masked.json \
  --output-dir results/compost/Llama-3.1-8B-Instruct/unmarked/

# echo "Running CoMPosT on Llama 70B Transcripts..."
# python compost/compost_evaluator.py \
#   --data data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/control_simulations_masked.json \
#         data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/default_topics_masked.json \
#         data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json \
#   --output-dir results/compost/Llama-3.1-70B-Instruct-AWQ-INT4/

# echo "Running CoMPosT on GPT-4o-mini Transcripts..."
# python compost/compost_evaluator.py \
#   --data data/transcripts/gpt-4o-mini/control_simulations_masked.json \
#         data/transcripts/gpt-4o-mini/default_topics_masked.json \
#         data/transcripts/gpt-4o-mini/target_simulations_masked.json \
#   --output-dir results/compost/gpt-4o-mini/

echo "CoMPosT complete!"
