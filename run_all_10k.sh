#!/bin/bash
# ==========================================================
# Smart Home Finetuning - 10K Dataset Pipeline
# ==========================================================
# Runs: Prepare Data → Train LLaMA → Train Gemma → Evaluate → Push to HF → Push to GitHub
#
# Usage:
#   nohup bash run_all_10k.sh > pipeline_10k.log 2>&1 &
#   tail -f pipeline_10k.log   # to monitor
# ==========================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate smart_home

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "Smart Home Finetuning Pipeline (10K) — START"
log "=========================================="

# Exclude GPU 0 (has stuck memory from zombie process)
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# ---- Step 0: Prepare data from 10K CSV ----
log "[0/5] Preparing train/val/test splits from 10K clean CSV..."
python prepare_data.py
log "  ✓ Data preparation complete"

# ---- Step 1: Finetune LLaMA 3-8B (7-GPU DDP) ----
log "[1/5] Finetuning Meta-Llama-3-8B-Instruct on 10K data..."
accelerate launch --config_file accelerate_config.yaml finetune_llama.py
log "  ✓ LLaMA finetuning complete"

# Clear GPU memory between training runs
log "Clearing GPU memory before Gemma training..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
sleep 5

# ---- Step 2: Finetune Gemma 2-2B (7-GPU DDP) ----
log "[2/5] Finetuning Gemma-2-2B-IT on 10K data..."
accelerate launch --config_file accelerate_config.yaml finetune_gemma.py
log "  ✓ Gemma finetuning complete"

# Clear GPU memory before evaluation
log "Clearing GPU memory before evaluation..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
sleep 5

# ---- Step 3: Evaluate both models ----
log "[3/5] Evaluating both models on test set (500 samples)..."
python evaluate.py --model both --max_samples 500
log "  ✓ Evaluation complete — see results/results.txt"

# ---- Step 4: Push to HuggingFace ----
log "[4/5] Pushing models to HuggingFace (10K repos)..."
python push_to_hf.py --hf_username vishakmsr
log "  ✓ Models pushed to HuggingFace"

# ---- Step 5: Push code to GitHub (10k-data branch) ----
log "[5/5] Pushing code to GitHub (10k-data branch)..."
cd "$PROJECT_DIR"

git add -A
git commit -m "10K experiment: LLaMA-3-8B & Gemma-2-2B LoRA on 10K smart home commands" 2>/dev/null || log "  (nothing new to commit)"
git push -u origin 10k-data 2>/dev/null || log "  ⚠ GitHub push failed"
log "  ✓ GitHub push complete"

log "=========================================="
log "Pipeline (10K) DONE!"
log "Results:      $PROJECT_DIR/results/results.txt"
log "LLaMA model:  $PROJECT_DIR/checkpoints_10k/llama/final/"
log "Gemma model:  $PROJECT_DIR/checkpoints_10k/gemma/final/"
log "HF repos:     vishakmsr/meta-llama-3-8b-smart-home-lora-10k"
log "              vishakmsr/gemma-2-2b-smart-home-lora-10k"
log "=========================================="
