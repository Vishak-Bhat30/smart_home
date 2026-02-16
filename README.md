# Smart Home Command Finetuning

Finetuning **Meta-Llama-3-8B-Instruct** and **Gemma 2-2B-IT** to convert natural language smart home commands into structured JSON device states using **QLoRA**.

## Task Description

**Input:** Natural language smart home command
**Output:** JSON object mapping rooms to devices to states

**Example:**
- Input: `"Turn on the lights in the bedroom and set AC to 22C"`
- Output: `{"bedroom": {"lights": "on", "ac": "22C"}}`

## Dataset

- **Source:** `smart_home_100k_2bhk_inspired.csv`
- **Size:** 100,000 examples
- **Split:** 90,000 train / 5,000 validation / 5,000 test
- **Rooms (8):** bathroom, bedroom, balcony, dining_room, hall, kitchen, living_room, study_room
- **Devices (9):** ac, blinds, computer, exhaust, fan, geyser, lights, music_system, tv
- **Avg input length:** ~70 characters | **Avg output length:** ~58 characters

## Method

### Models

| Model | Parameters | Base Model |
|-------|-----------|------------|
| Meta-Llama-3-8B-Instruct | 8.03B | `NousResearch/Meta-Llama-3-8B-Instruct` |
| Gemma 2-2B-IT | 2.61B | `google/gemma-2-2b-it` |

### QLoRA Finetuning

We use **QLoRA (Quantized Low-Rank Adaptation)** for parameter-efficient finetuning. The base model is loaded in 4-bit precision (NF4 quantization with double quantization via bitsandbytes), and small trainable LoRA adapters are injected into the attention and MLP layers.

**LoRA Configuration:**
- Rank: 32 | Alpha: 64 | Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable parameters: ~1-2% of total

**Training Hyperparameters:**
- Epochs: 3
- Effective batch size: 32 (LLaMA: 4x8, Gemma: 8x4)
- Learning rate: 1e-4 (LLaMA 8B) / 2e-4 (Gemma) with cosine schedule
- Warmup: 5% of steps
- Weight decay: 0.01
- Precision: BF16
- Optimizer: AdamW (fused)
- Gradient checkpointing: enabled

**Prompt Template:** Each example uses a chat template with a system prompt describing the smart home assistant role, followed by the user command, with the assistant responding with the JSON output only.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match | Full JSON output matches expected (after key sorting) |
| Valid JSON Rate | Percentage of outputs that are valid JSON |
| Room-level F1 | Precision/recall/F1 of predicted room keys |
| Device-Value Accuracy | Percentage of device-value pairs correctly predicted |

## Results

See `results/results.txt` for detailed evaluation results after training.

## Project Structure

```
smart_home_finetuning/
  prepare_data.py       # Data loading, splitting, chat formatting
  finetune_llama.py     # LLaMA finetuning with QLoRA
  finetune_gemma.py     # Gemma finetuning with QLoRA
  evaluate.py           # Test set evaluation with all metrics
  push_to_hf.py         # Push models to HuggingFace Hub
  run_all.sh            # End-to-end pipeline script
  README.md             # This file
  environment.yml       # Conda environment specification
  data/                 # Train/val/test JSONL files (generated)
  checkpoints/          # Model checkpoints (generated)
  results/              # Evaluation results (generated)
```

## Setup and Usage

### 1. Create Environment

```bash
conda create -n smart_home python=3.11 -y
conda activate smart_home
pip install torch transformers peft trl accelerate datasets bitsandbytes huggingface_hub scikit-learn
```

### 2. Run Full Pipeline

```bash
bash run_all.sh --hf_username YOUR_HF_USERNAME
```

Or run steps individually:

```bash
# Step 1: Prepare data
python prepare_data.py

# Step 2: Finetune LLaMA
python finetune_llama.py

# Step 3: Finetune Gemma
python finetune_gemma.py

# Step 4: Evaluate
python evaluate.py --model both --max_samples 500

# Step 5: Push to HuggingFace
python push_to_hf.py --hf_username YOUR_HF_USERNAME
```

### 3. HuggingFace Login

```bash
huggingface-cli login
# Or set: export HF_TOKEN=your_token
```

## Hardware

- **GPUs:** 8x NVIDIA B200 (183 GB each)
- **Training time:** Estimated ~1-2 hours per model

## License

This project is for research and educational purposes.
