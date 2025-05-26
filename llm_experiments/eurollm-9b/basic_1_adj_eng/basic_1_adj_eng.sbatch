#!/bin/bash
#SBATCH --job-name=eurollm_trunc_prompt_1_shot_without_start_end
#SBATCH --output=logs/eurollm_trunc_prompt_1_shot_without_start_end_%j.out
#SBATCH --error=logs/eurollm_trunc_prompt_1_shot_without_start_end_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --gpus=1

# === Load modules ===
module load Python
module load Smilei/5.1-cpu
module load GCC/13.2.0
module load CUDA/12.2
module load Python/Miniconda

# === Create Conda ===
eval "$(conda shell.bash hook)"
conda activate eurollm_env_new

# === WANDB login ===
export WANDB_API_KEY=a40c76014327526b893781cfa1402f80189458e2

# === .py-script ===
python basic_1_adj_eng.py
