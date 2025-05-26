#!/bin/bash
#SBATCH --job-name=gemma_trunc_prompt_1_shot
#SBATCH --output=logs/gemma_trunc_prompt_1_shot_%j.out
#SBATCH --error=logs/gemma_trunc_prompt_1_shot_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gpus=3

# === Load modules ===
module load Python
module load CUDA/12.2
module load Python/Miniconda

# === Create Conda ===
eval "$(conda shell.bash hook)"
conda activate final_gemma_env

# === WANDB login ===
export WANDB_API_KEY=a40c76014327526b893781cfa1402f80189458e2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === .py-script ===
python gemma.py
