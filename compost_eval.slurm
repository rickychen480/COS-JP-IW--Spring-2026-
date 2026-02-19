#!/bin/bash
#SBATCH --job-name=compost_eval
#SBATCH -N 1                          ## Request 1 node
#SBATCH --gres=gpu:2                  ## Request 2 GPUs
#SBATCH -c 8                          ## Request 8 CPUs (helps with vLLM data loading)
#SBATCH --mem=64G                     ## Request 64GB of CPU RAM
#SBATCH -t 12:00:00                   ## Request 12 hours of walltime
#SBATCH --output=slurm-%j.out         ## Save standard output to this file
#SBATCH --error=slurm-%j.err          ## Save errors to this file

conda activate env

echo "Starting CoMPosT Evaluation with Llama 70B AWQ on 2 GPUs..."

# Run the python script
python compost_evaluator.py \
    --data pilot_results.json \
    --out compost_llama70b_awq.json \
    --judge /scratch/$USER/hf_models/Llama-3.1-70B-Instruct-AWQ-INT4

echo "Evaluation Complete!"