# Smart-Home LLM Evaluation — 40+ Model Comparison

This folder compares **40+ open-source SLMs/LLMs** on the smart-home
natural-language → structured-JSON task defined in
[smart_home/](../smart_home/README.md).

There are **two paths**:

1. **Simulation path** (no GPU needed) — already produced for you here.
2. **Real experiment path** — run on a GPU box to replace the simulated
   numbers with measured ones, model by model.

---

## Files

| File | Purpose |
|------|---------|
| [simulate_evaluation.py](simulate_evaluation.py) | Builds the 40+ model registry and produces simulated metrics (anchored on the two real measurements that already exist in the repo). Outputs CSV, MD, DOCX. |
| [run_experiments.py](run_experiments.py) | Real fine-tune + evaluate runner. Pick which models you want; it does QLoRA fine-tuning, runs eval, and aggregates results into a CSV/MD. |
| [model_comparison.csv](model_comparison.csv) | Master comparison table (simulated). |
| [model_comparison.md](model_comparison.md) | Same as CSV but human-readable, with a focused **Params-vs-Accuracy** table at the top. |
| [evaluation_report.docx](evaluation_report.docx) | Scientific report (8 sections + appendix). |
| [test_sample.jsonl](test_sample.jsonl) | The 200-example test sample used for the simulation. |

---

## 1. Simulation path (no GPU)

The simulation is anchored on the two real measurements already in the
repo:

| Model | Real exact-match | Source |
|---|---|---|
| LLaMA-3-8B-Instruct | **100.00%** | `smart_home/results/results.txt` |
| Gemma-2-2B-IT | **99.40%** | `smart_home/results/results.txt` |

The other 38+ rows are computed from a deterministic, parameter-aware
heuristic (size-curve interpolation + family prior + instruct/zero-shot
penalties). The numbers are **approximate** — they predict ranking,
not absolute accuracy — and every simulated cell is clearly labelled.

```powershell
# Requirements: python-docx (already in your env if Models.docx worked).
python simulate_evaluation.py
```

Outputs:
- `model_comparison.csv`
- `model_comparison.md`  ← contains the **Params vs. Accuracy** focused table
- `evaluation_report.docx`
- `test_sample.jsonl`

---

## 2. Real experiment path (GPU)

The runner in [run_experiments.py](run_experiments.py) is the script
you actually launch on a GPU machine to replace the simulated cells.
The full registry is documented at the top of that file. Reproduced
here for convenience:

### Models the runner can train + evaluate

| Key (pass to `--models`) | HuggingFace repo | Params | Mode |
|---|---|---|---|
| `smollm2_360m`     | `HuggingFaceTB/SmolLM2-360M-Instruct`     | 0.36 B | finetune |
| `qwen25_05b`       | `Qwen/Qwen2.5-0.5B-Instruct`              | 0.5 B  | finetune |
| `llama32_1b`       | `meta-llama/Llama-3.2-1B-Instruct`        | 1.24 B | finetune |
| `qwen25_15b`       | `Qwen/Qwen2.5-1.5B-Instruct`              | 1.5 B  | finetune |
| `gemma2_2b`        | `google/gemma-2-2b-it`                    | 2.6 B  | finetune *(already done)* |
| `qwen25_3b`        | `Qwen/Qwen2.5-3B-Instruct`                | 3 B    | finetune |
| `llama32_3b`       | `meta-llama/Llama-3.2-3B-Instruct`        | 3.2 B  | finetune |
| `phi35_mini`       | `microsoft/Phi-3.5-mini-instruct`         | 3.8 B  | finetune |
| `phi4_mini`        | `microsoft/Phi-4-mini-instruct`           | 3.8 B  | finetune |
| `mistral_7b`       | `mistralai/Mistral-7B-Instruct-v0.3`      | 7.2 B  | finetune |
| `qwen25_7b`        | `Qwen/Qwen2.5-7B-Instruct`                | 7.6 B  | finetune |
| `llama3_8b`        | `NousResearch/Meta-Llama-3-8B-Instruct`   | 8 B    | finetune *(already done)* |
| `qwen25_14b_zs`    | `Qwen/Qwen2.5-14B-Instruct`               | 14 B   | zero-shot baseline |
| `llama31_70b_zs`   | `meta-llama/Llama-3.1-70B-Instruct`       | 70 B   | zero-shot baseline |

Aliases: `all_finetune`, `all_zeroshot`, `everything`.

### Setup (once)

```powershell
# 1. Activate the environment from the smart_home repo:
conda env update -f ..\smart_home\environment.yml -n smart_home
conda activate smart_home

# 2. Authenticate with HuggingFace (needed for Llama / Gemma gated repos):
huggingface-cli login

# 3. Prepare the train/val/test JSONL files:
cd ..\smart_home
python prepare_data.py
cd ..\smart_home_model_evaluation
```

### Run experiments

Pick any subset of models from the registry above:

```powershell
# A: Single fine-tune + eval (small model, single GPU is fine):
python run_experiments.py --models qwen25_15b

# B: Several models in sequence:
python run_experiments.py --models llama32_1b qwen25_15b qwen25_3b phi35_mini

# C: All eight planned fine-tune candidates:
python run_experiments.py --models all_finetune

# D: Both zero-shot baselines (no training, just inference):
python run_experiments.py --models all_zeroshot

# E: Multi-GPU recommended for >= 3B (uses the same accelerate config
#    as the original repo):
accelerate launch --config_file ..\smart_home\accelerate_config.yaml `
    run_experiments.py --models qwen25_7b mistral_7b

# F: Smoke test (cap to 50 training steps, 50 eval samples):
python run_experiments.py --models qwen25_15b --max_steps 50 --max_samples 50

# G: After everything is done, aggregate per-model JSONs into one CSV:
python run_experiments.py --aggregate
```

### Outputs of the real runner

- `checkpoints/<key>/`      — QLoRA adapters per model
- `results_real/<key>.json` — per-model eval metrics
- `model_comparison_real.csv` — aggregated table (created by `--aggregate`)
- `model_comparison_real.md`  — same, human-readable

### What the runner measures

For every model the runner reports:

- **Exact Match %** — strict full-output match after key sorting (the
  primary "accuracy" number).
- **Valid JSON %** — fraction of outputs that parse as JSON.
- **Room-level F1** — precision/recall over predicted room keys.
- **Device-Value Accuracy** — fraction of (room, device, value)
  triples correctly produced.
- **Latency ms / example** — wall-clock per-example inference time.

These are exactly the same metrics computed by `smart_home/evaluate.py`
for the two anchor models, so simulated and measured rows are directly
comparable.

---

## How the table is structured

Both the simulation and the real-experiment runner emphasise the two
columns you care most about:

> **Parameters** (model size in B) and **Accuracy** (exact-match %).

In `model_comparison.md` the very first table after the header is a
focused **Params vs. Accuracy** view of all 40+ models in a single
ranking. The same focused view exists as section 4.1 of the DOCX
report. The remaining columns (valid-JSON, room-F1, device-value,
latency, VRAM) are kept in the full metric tables further down.

---

## Caveats

- Simulated rows are **approximate**. They are anchored on two real
  datapoints, but family-specific surprises are not ruled out.
- Latency and VRAM numbers in the simulation assume a single B200 GPU
  with batch=1, BF16 generation, 4-bit base; on consumer hardware the
  *absolute* numbers will differ but the *ranking* should hold.
- The measured-rows column in the focused table is marked **[measured]**;
  every other row is marked *(simulated, ...)*. Do not quote simulated
  numbers as experimental results.
