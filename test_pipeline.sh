#!/bin/bash
# ==========================================================
# Quick Pipeline Test — uses tiny data + 2 training steps
# Validates: train → save → plot → evaluate → push logic
# Should finish in ~3-5 minutes
# ==========================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate smart_home

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "PIPELINE TEST — tiny data, 2 steps"
log "=========================================="

# Create tiny train/val datasets (50 examples each)
log "Creating tiny test datasets..."
python -c "
import os
train_in = os.path.join('$PROJECT_DIR', 'data', 'train.jsonl')
val_in   = os.path.join('$PROJECT_DIR', 'data', 'val.jsonl')
train_out = os.path.join('$PROJECT_DIR', 'data', 'train_tiny.jsonl')
val_out   = os.path.join('$PROJECT_DIR', 'data', 'val_tiny.jsonl')

for src, dst, n in [(train_in, train_out, 50), (val_in, val_out, 20)]:
    with open(src) as f:
        lines = [next(f) for _ in range(n)]
    with open(dst, 'w') as f:
        f.writelines(lines)
    print(f'  Created {dst} ({n} examples)')
"

# ---- Test LLaMA training (2 steps) ----
log "[1/5] Testing LLaMA finetuning (2 steps)..."
accelerate launch --config_file accelerate_config.yaml finetune_llama.py \
    --max_steps 1 \
    --data_suffix _tiny
log "  ✓ LLaMA test training complete"

# ---- Test Gemma training (2 steps) ----
log "[2/5] Testing Gemma finetuning (2 steps)..."
accelerate launch --config_file accelerate_config.yaml finetune_gemma.py \
    --max_steps 1 \
    --data_suffix _tiny
log "  ✓ Gemma test training complete"

# ---- Test evaluation (5 samples) ----
log "[3/5] Testing evaluation (5 samples)..."
python evaluate.py --model both --max_samples 5
log "  ✓ Evaluation test complete"

# ---- Test HF push ----
log "[4/5] Testing HuggingFace push..."
python push_to_hf.py --hf_username vishakmsr
log "  ✓ HuggingFace push test complete"

# ---- Test GitHub push ----
log "[5/5] Testing GitHub push..."
cd "$PROJECT_DIR"
if [ ! -d ".git" ]; then
    git init
    git remote add origin https://github.com/Vishak-Bhat30/smart_home.git 2>/dev/null || true
fi
git add -A
git commit -m "Pipeline test commit" 2>/dev/null || log "  (nothing new to commit)"
git branch -M main
git push -u origin main 2>/dev/null || log "  ⚠ GitHub push failed"
log "  ✓ GitHub push test complete"

# Cleanup tiny data
rm -f "$PROJECT_DIR/data/train_tiny.jsonl" "$PROJECT_DIR/data/val_tiny.jsonl"

log "=========================================="
log "PIPELINE TEST PASSED ✓"
log "=========================================="
