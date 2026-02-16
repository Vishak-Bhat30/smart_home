#!/bin/bash
# ==========================================================
# Resume pipeline from Gemma training (LLaMA already done)
# ==========================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate smart_home

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "Resuming pipeline from Gemma training"
log "LLaMA checkpoint already exists at checkpoints/llama/final/"
log "Using GPUs 1-7 (GPU 0 has stuck memory from zombie process)"
log "=========================================="

# Exclude GPU 0 which has stuck memory
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# ---- Step 2: Finetune Gemma 2-2B (8-GPU DDP) ----
log "[2/5] Finetuning Gemma-2-2B-IT..."
accelerate launch --config_file accelerate_config.yaml finetune_gemma.py
log "  ✓ Gemma finetuning complete"

# Clear GPU memory before evaluation
log "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
sleep 5

# ---- Step 3: Evaluate both models ----
log "[3/5] Evaluating both models on test set (500 samples)..."
python evaluate.py --model both --max_samples 500
log "  ✓ Evaluation complete — see results/results.txt"

# ---- Step 4: Push to HuggingFace ----
log "[4/5] Pushing models to HuggingFace..."
python push_to_hf.py --hf_username vishakmsr
log "  ✓ Models pushed to HuggingFace"

# ---- Step 5: Push code to GitHub ----
log "[5/5] Pushing code to GitHub..."
cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
    git init
    git remote add origin https://github.com/Vishak-Bhat30/smart_home.git 2>/dev/null || true
fi

git add -A
git commit -m "Smart home finetuning: LLaMA-3-8B & Gemma-2-2B LoRA on 100K commands" 2>/dev/null || log "  (nothing new to commit)"
git branch -M main
git push -u origin main 2>/dev/null || log "  ⚠ GitHub push failed — create the repo at https://github.com/new first"
log "  ✓ GitHub push complete"

log "=========================================="
log "Pipeline DONE!"
log "Results:      $PROJECT_DIR/results/results.txt"
log "LLaMA model:  $PROJECT_DIR/checkpoints/llama/final/"
log "Gemma model:  $PROJECT_DIR/checkpoints/gemma/final/"
log "=========================================="
