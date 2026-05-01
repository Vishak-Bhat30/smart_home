"""
run_experiments.py
==================

Real (not simulated) fine-tuning + evaluation runner for the Smart-Home
NL -> JSON task. This is the script you actually run on a GPU box to
replace the simulated numbers in `model_comparison.csv` with measured
ones.

USAGE
-----

    # 1. Make sure the data is prepared (creates data/train.jsonl etc.)
    cd ../smart_home && python prepare_data.py && cd -

    # 2. Pick which models to run (see MODELS_TO_RUN below) and launch.
    #    Single-GPU:
    python run_experiments.py --models qwen25_15b llama32_1b smollm2_360m

    #    Multi-GPU (recommended for >=3B models):
    accelerate launch --config_file ../smart_home/accelerate_config.yaml \\
        run_experiments.py --models qwen25_7b mistral_7b

    # 3. Aggregate the per-model JSON results back into the master CSV:
    python run_experiments.py --aggregate


===================================================================
LIST OF EXPERIMENTABLE MODELS  ---  pass any of these keys to --models
===================================================================
The keys on the LEFT are what you pass to --models. The HF repo on the
RIGHT is what actually gets pulled. Comment in/out as you wish.

  Sub-1B:
    smollm2_360m       HuggingFaceTB/SmolLM2-360M-Instruct
    qwen25_05b         Qwen/Qwen2.5-0.5B-Instruct

  1B - 3B:
    llama32_1b         meta-llama/Llama-3.2-1B-Instruct
    llama32_3b         meta-llama/Llama-3.2-3B-Instruct
    qwen25_15b         Qwen/Qwen2.5-1.5B-Instruct
    qwen25_3b          Qwen/Qwen2.5-3B-Instruct
    gemma2_2b          google/gemma-2-2b-it             [already done in repo]

  3B - 10B:
    phi35_mini         microsoft/Phi-3.5-mini-instruct
    phi4_mini          microsoft/Phi-4-mini-instruct
    mistral_7b         mistralai/Mistral-7B-Instruct-v0.3
    qwen25_7b          Qwen/Qwen2.5-7B-Instruct
    llama3_8b          NousResearch/Meta-Llama-3-8B-Instruct  [already done]

  Zero-shot baselines (NO fine-tuning, just eval):
    qwen25_14b_zs      Qwen/Qwen2.5-14B-Instruct
    llama31_70b_zs     meta-llama/Llama-3.1-70B-Instruct

A short alias `--models all_finetune` runs the eight planned fine-tune
candidates in sequence; `--models all_zeroshot` runs the two baselines.
===================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Model registry. Only the models we actually want to run.
# ----------------------------------------------------------------------

@dataclass
class ExpModel:
    key: str
    hf_repo: str
    params_b: float
    mode: str               # "finetune" or "zeroshot"
    needs_trust_remote: bool = False


REGISTRY: dict[str, ExpModel] = {
    # --- fine-tune candidates -----------------------------------------
    "smollm2_360m":  ExpModel("smollm2_360m",  "HuggingFaceTB/SmolLM2-360M-Instruct",
                              0.36, "finetune"),
    "qwen25_05b":    ExpModel("qwen25_05b",    "Qwen/Qwen2.5-0.5B-Instruct",
                              0.494, "finetune"),
    "llama32_1b":    ExpModel("llama32_1b",    "meta-llama/Llama-3.2-1B-Instruct",
                              1.24, "finetune"),
    "llama32_3b":    ExpModel("llama32_3b",    "meta-llama/Llama-3.2-3B-Instruct",
                              3.21, "finetune"),
    "qwen25_15b":    ExpModel("qwen25_15b",    "Qwen/Qwen2.5-1.5B-Instruct",
                              1.5, "finetune"),
    "qwen25_3b":     ExpModel("qwen25_3b",     "Qwen/Qwen2.5-3B-Instruct",
                              3.0, "finetune"),
    "gemma2_2b":     ExpModel("gemma2_2b",     "google/gemma-2-2b-it",
                              2.61, "finetune"),
    "phi35_mini":    ExpModel("phi35_mini",    "microsoft/Phi-3.5-mini-instruct",
                              3.82, "finetune", needs_trust_remote=True),
    "phi4_mini":     ExpModel("phi4_mini",     "microsoft/Phi-4-mini-instruct",
                              3.82, "finetune", needs_trust_remote=True),
    "mistral_7b":    ExpModel("mistral_7b",    "mistralai/Mistral-7B-Instruct-v0.3",
                              7.24, "finetune"),
    "qwen25_7b":     ExpModel("qwen25_7b",     "Qwen/Qwen2.5-7B-Instruct",
                              7.62, "finetune"),
    "llama3_8b":     ExpModel("llama3_8b",     "NousResearch/Meta-Llama-3-8B-Instruct",
                              8.03, "finetune"),

    # --- zero-shot baselines ------------------------------------------
    "qwen25_14b_zs": ExpModel("qwen25_14b_zs", "Qwen/Qwen2.5-14B-Instruct",
                              14.0, "zeroshot"),
    "llama31_70b_zs":ExpModel("llama31_70b_zs","meta-llama/Llama-3.1-70B-Instruct",
                              70.0, "zeroshot"),
}

ALIASES = {
    "all_finetune": ["smollm2_360m", "qwen25_05b", "llama32_1b", "llama32_3b",
                     "qwen25_15b", "qwen25_3b", "phi35_mini", "phi4_mini",
                     "mistral_7b", "qwen25_7b"],
    "all_zeroshot": ["qwen25_14b_zs", "llama31_70b_zs"],
    "everything":   list(REGISTRY.keys()),
}


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "smart_home"
DATA_DIR = REPO / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results_real"
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Hyperparameters (mirrors smart_home/finetune_llama.py defaults)
# ----------------------------------------------------------------------

NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


# Per-size batch presets (per-GPU). Adjust to your hardware.
def batch_for(params_b: float) -> tuple[int, int]:
    """Returns (per_device_batch_size, grad_accum_steps)."""
    if params_b <= 1.0:
        return 32, 1
    if params_b <= 3.5:
        return 16, 1
    if params_b <= 8.5:
        return 8, 2
    return 4, 4


# ----------------------------------------------------------------------
# Fine-tune one model
# ----------------------------------------------------------------------

def finetune_one(m: ExpModel, max_steps: int = -1) -> Path:
    """Fine-tune `m` with QLoRA and return the checkpoint dir."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    out_dir = CKPT_DIR / m.key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n[finetune] {m.key} ({m.hf_repo}, {m.params_b}B)\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        m.hf_repo, trust_remote_code=m.needs_trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        m.hf_repo,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=m.needs_trust_remote,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )

    train_ds = load_dataset("json",
                            data_files=str(DATA_DIR / "train.jsonl"),
                            split="train")
    val_ds = load_dataset("json",
                          data_files=str(DATA_DIR / "val.jsonl"),
                          split="train")

    bs, ga = batch_for(m.params_b)
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=0.05,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        max_steps=max_steps if max_steps > 0 else -1,
        optim="paged_adamw_8bit",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=train_ds, eval_dataset=val_ds,
                         peft_config=lora, args=cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[finetune] done. checkpoint: {out_dir}")
    return out_dir


# ----------------------------------------------------------------------
# Evaluate one model (works for both fine-tuned and zero-shot)
# ----------------------------------------------------------------------

def evaluate_one(m: ExpModel, adapter_dir: Optional[Path], max_samples: int = 500) -> dict:
    """Run the standard 4-metric eval and return a result dict."""
    import torch
    import re
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n{'='*60}\n[eval] {m.key}  (max_samples={max_samples})\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(
        m.hf_repo, trust_remote_code=m.needs_trust_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m.hf_repo, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=m.needs_trust_remote,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    test_path = DATA_DIR / "test.jsonl"
    samples: list[dict] = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break

    SYSTEM = ("You are a smart home assistant. Convert the user's command "
              "into a JSON object mapping rooms to devices to states. "
              "Respond with ONLY the JSON object.")

    def render(user_msg: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)

    em = vj = 0
    dv_correct = dv_total = 0
    room_tp = room_fp = room_fn = 0
    t0 = time.time()
    for s in samples:
        prompt = render(s["input"])
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256,
                                 do_sample=False, temperature=0.0,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inp.input_ids.shape[1]:],
                         skip_special_tokens=True).strip()
        # strip code fences
        gen = re.sub(r"^```(json)?|```$", "", gen, flags=re.MULTILINE).strip()
        try:
            pred = json.loads(gen)
            vj += 1
        except Exception:
            pred = {}
        gold = json.loads(s["output"]) if isinstance(s["output"], str) else s["output"]
        if json.dumps(pred, sort_keys=True) == json.dumps(gold, sort_keys=True):
            em += 1
        for room in set(pred) | set(gold):
            if room in pred and room in gold:
                room_tp += 1
            elif room in pred:
                room_fp += 1
            else:
                room_fn += 1
        for room, devs in gold.items():
            for dev, val in devs.items():
                dv_total += 1
                if pred.get(room, {}).get(dev) == val:
                    dv_correct += 1
    n = len(samples)
    elapsed = time.time() - t0
    p = room_tp / max(room_tp + room_fp, 1)
    r = room_tp / max(room_tp + room_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    res = {
        "key": m.key, "hf_repo": m.hf_repo, "params_b": m.params_b,
        "mode": m.mode, "n_test": n,
        "exact_match": round(100 * em / n, 2),
        "valid_json": round(100 * vj / n, 2),
        "room_f1": round(100 * f1, 2),
        "device_value_acc": round(100 * dv_correct / max(dv_total, 1), 2),
        "latency_ms_per_example": round(1000 * elapsed / n, 1),
    }
    out_path = RESULTS_DIR / f"{m.key}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[eval] saved {out_path}")
    print(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------
# Aggregate results back into the master CSV / MD
# ----------------------------------------------------------------------

def aggregate() -> None:
    import csv
    rows: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        rows.append(json.loads(p.read_text()))
    out_csv = ROOT / "model_comparison_real.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "key", "hf_repo", "params_b", "mode", "n_test",
            "exact_match", "valid_json", "room_f1",
            "device_value_acc", "latency_ms_per_example",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {out_csv} ({len(rows)} rows)")

    out_md = ROOT / "model_comparison_real.md"
    rows.sort(key=lambda r: (-r["exact_match"], r["params_b"]))
    lines = [
        "# Smart-Home LLM Comparison — REAL measurements",
        "",
        "| Model | Params (B) | Mode | Exact Match % | Valid JSON % | "
        "Room F1 | Dev-Val % | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['key']} | {r['params_b']:g} | {r['mode']} | "
                     f"{r['exact_match']:.2f} | {r['valid_json']:.2f} | "
                     f"{r['room_f1']:.2f} | {r['device_value_acc']:.2f} | "
                     f"{r['latency_ms_per_example']:.0f} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregate] wrote {out_md}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real fine-tune + eval runner for the Smart-Home task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See top-of-file docstring for the full list of model keys.")
    ap.add_argument("--models", nargs="*", default=[],
                    help="Model keys (or aliases: all_finetune, all_zeroshot, "
                         "everything). See top-of-file docstring.")
    ap.add_argument("--max_samples", type=int, default=500,
                    help="Number of test examples to evaluate on.")
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Cap fine-tuning steps (smoke test only).")
    ap.add_argument("--skip_finetune", action="store_true",
                    help="Skip training; only run eval. Requires existing "
                         "checkpoint or a zero-shot model.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Aggregate per-model JSON results into a CSV/MD.")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return 0

    if not args.models:
        ap.print_help()
        print("\n[error] no --models specified.", file=sys.stderr)
        return 2

    keys: list[str] = []
    for k in args.models:
        if k in ALIASES:
            keys.extend(ALIASES[k])
        elif k in REGISTRY:
            keys.append(k)
        else:
            print(f"[error] unknown model key: {k}", file=sys.stderr)
            return 2

    for k in keys:
        m = REGISTRY[k]
        try:
            adapter = None
            if m.mode == "finetune" and not args.skip_finetune:
                adapter = finetune_one(m, max_steps=args.max_steps)
            elif m.mode == "finetune":
                adapter = CKPT_DIR / m.key
                if not adapter.exists():
                    print(f"[warn] {adapter} missing; running zero-shot eval instead.")
                    adapter = None
            evaluate_one(m, adapter, max_samples=args.max_samples)
        except Exception as e:
            print(f"[error] {k} failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    aggregate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
