# Aspect-Based Sentiment Analysis for Restaurant Reviews in Estonian

This repository contains the code, dataset, and evaluation scripts for a term paper project focused on **Aspect-Based Sentiment Analysis (ABSA)** in the **low-resource setting** of the Estonian language. We apply and compare two approaches:  
1. an **unsupervised rule-based model**, and  
2. a **Large Language Model (LLM)**-based method using **EuroLLM-9B**.

The task includes identifying triplets of `(aspect term, opinion term, sentiment polarity)` from Estonian restaurant reviews.

---

## 1. Repository Structure

### `metrics_function.ipynb`
- Contains a **custom evaluation function** that computes **precision**, **recall**, **F1**, and **accuracy** at both:
  - **Aspect-level** (for aspect term identification only)
  - **Triplet-level** (for full aspect-opinion-polarity triplets)
- Includes **post-processing functions** for normalizing and comparing model predictions with gold annotations.

---

### `absa_estonian_restaurant_reviews_dataset/`
Contains all data-related materials: collection, annotation, and gold standard files.

- `restaurant_reviews_crawling.ipynb` — Crawling reviews from the **online-portal DinnerBooking**, followed by filtering and sampling.
- `filtered_final_estonian_restaurant_reviews_dataset.csv` — All raw, unlabelled Estonian restaurant reviews.
- `estonian_restaurant_reviews_val_sample.csv`, `modified_estonian_restaurant_reviews_test_sample.csv` — Unlabelled validation and test samples.
- `estonian_restaurant_reviews_val_sample_annotation.json`, `estonian_restaurant_reviews_test_sample_annotation.json` — Initial annotated reviews (Label Studio format).
- `output_val.json`, `output_corrected.json` — Final **gold-standard validation and test annotations** in simplified format for evaluation.

---

### `unsupervised_rule_based_baseline/`
Implements the rule-based approach.

- `unsupervised_model_experiments_full.ipynb` — Full pipeline:
  - Aspect and opinion term extraction using **Stanza** for dependency parsing
  - Attention score reweighting using **EstBERT**
  - Sentiment polarity assignment
- `model_example_with_heatmap.ipynb` — Demonstration of model output with **attention heatmap visualization**.

---

### `llm_experiments/`
Experiments using **Large Language Models (LLMs)**.

#### Structure:
- Two subdirectories:
  - `eurollm-9b/` — working experiments
  - `gemma-3/` — **not used in paper** due to incompatibility issues on Charisma (library version conflicts and misconfigured `config.json`)
- Each model directory contains subfolders named using this pattern: `basic/annot_N_(adj)_eng/est`,
where:
- `basic` / `annot`: prompt type (basic instruction or full annotation guidelines)
- `N`: number of in-context examples (0–5)
- `adj` (optional): adjusted prompt variation
- `eng` / `est`: prompt language (English or Estonian)

Each such folder contains:
- `.py` — Python script to run the experiment
- `.sbatch` — SLURM file to submit the job to **cHARISMa**
- `.json` — Output file with model **predictions**

---

## 2. How to Run

### Unsupervised Rule-Based Model
To test the rule-based pipeline:
1. Open `unsupervised_rule_based_baseline/unsupervised_model_experiments_full.ipynb`
2. Upload `output_val.json` and `output_corrected.json` from `absa_estonian_restaurant_reviews_dataset/` as gold standard.
3. Run the notebook to extract predictions and evaluate performance using `metrics_function.ipynb`.

---

### LLM Experiments

LLM inference was conducted on the **cHARISMa** supercomputer using **CUDA 12.2**.  

#### Environment Setup:
```bash
module load Python
module load Smilei/5.1-cpu
module load GCC/13.2.0
module load CUDA/12.2

conda create -n eurollm_env_new python=3.10 -y
conda activate eurollm_env_new

pip install torch==2.1.2
pip install xformers==0.0.23.post1
pip install vllm==0.3.2
pip install huggingface_hub transformers regex pandas tqdm wandb
```
See full requirements in `requirements.txt`.

#### Running LLM jobs:
1. Navigate to a specific experiment subfolder in llm_experiments/eurollm-9b/
2. Submit a job with:
```bash
sbatch your_script.sbatch
```
Model predictions will be saved to the corresponding .json file.
To evaluate results, compare model outputs against gold annotations in output_val.json or output_corrected.json using metrics_function.ipynb.

## Credits
Research supervisor — Eduard Klyshinsky, Associate Professor at the School of Linguistics

Research advisor — Anna Aksenova, NLP Researcher
