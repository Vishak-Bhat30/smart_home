# Open-Source SLMs & LLMs: Survey and Model Selection Rationale

## Context

**Task:** Convert natural language smart home commands → structured JSON device states  
**Current models:** LLaMA-3-8B-Instruct (QLoRA), Gemma-2-2B-IT (QLoRA)  
**Dataset:** 100K synthetic commands (template-based), 2BHK house, 8 rooms, 9 devices  
**Current results:** LLaMA-3-8B: 100% exact match, Gemma-2-2B: 99.4% exact match (500 test samples)

---

## 1. Complete Model Landscape

### 1.1 Sub-1B Parameter Models (Ultra-Small SLMs)

| Model | Params | Developer | Instruct Variant | License |
|---|---|---|---|---|
| SmolLM2-135M | 135M | HuggingFace | Yes | Apache 2.0 |
| SmolLM2-360M | 360M | HuggingFace | Yes | Apache 2.0 |
| Qwen2.5-0.5B | 0.5B | Alibaba | Yes (Instruct) | Apache 2.0 |
| OLMo-1B | 1B | AI2 | No | Apache 2.0 |
| TinyLlama-1.1B | 1.1B | Zhang et al. | Yes (Chat) | Apache 2.0 |
| Phi-1.5 | 1.3B | Microsoft | No (base only) | MIT |

**Why not considered (as primary):**
- These models have limited instruction-following capacity and struggle with multi-device commands requiring structured JSON output across multiple rooms.
- However, they are ideal for **edge deployment** experiments (smart home hubs, Raspberry Pi). We should test a few (Qwen2.5-0.5B, SmolLM2-360M) to find the *minimum viable model size* for this task.

---

### 1.2 1B–3B Parameter Models (Small Language Models)

| Model | Params | Developer | Instruct Variant | License |
|---|---|---|---|---|
| LLaMA-3.2-1B | 1.24B | Meta | Yes (Instruct) | Llama 3.2 Community |
| LLaMA-3.2-3B | 3.21B | Meta | Yes (Instruct) | Llama 3.2 Community |
| Gemma-2-2B-IT | 2.61B | Google | Yes | Gemma License |
| Qwen2.5-1.5B | 1.5B | Alibaba | Yes (Instruct) | Apache 2.0 |
| Qwen2.5-3B | 3B | Alibaba | Yes (Instruct) | Apache 2.0 |
| Phi-2 | 2.7B | Microsoft | No (base only) | MIT |
| StableLM-2-1.6B | 1.6B | Stability AI | Yes (Zephyr) | StabilityAI Non-Commercial |
| SmolLM2-1.7B | 1.7B | HuggingFace | Yes | Apache 2.0 |
| Danube3-500M/4B | 0.5B/4B | H2O.ai | Yes (Chat) | Apache 2.0 |

**Status:**
- **Gemma-2-2B-IT ✅ already evaluated** — 99.4% exact match.
- **LLaMA-3.2-1B/3B — should evaluate.** Direct smaller siblings of our LLaMA-3-8B; tests how much performance degrades with fewer parameters on the *same architecture*.
- **Qwen2.5-1.5B/3B — should evaluate.** Strong multilingual models; Qwen2.5 family consistently outperforms size peers on structured output tasks.
- **Phi-2 — excluded.** No instruction-tuned variant; would require custom chat template and more data to compensate.
- **StableLM-2-1.6B — excluded.** Non-commercial license (StabilityAI NCNL); cannot deploy in a real smart home product.
- **SmolLM2-1.7B — worth testing** as a HuggingFace-native model with good instruction following for its size.

---

### 1.3 3B–10B Parameter Models (Medium LLMs)

| Model | Params | Developer | Instruct Variant | License |
|---|---|---|---|---|
| LLaMA-3-8B | 8.03B | Meta | Yes (Instruct) | Llama 3 Community |
| LLaMA-3.1-8B | 8.03B | Meta | Yes (Instruct) | Llama 3.1 Community |
| Mistral-7B-v0.3 | 7.24B | Mistral AI | Yes (Instruct) | Apache 2.0 |
| Qwen2.5-7B | 7.62B | Alibaba | Yes (Instruct) | Apache 2.0 |
| Phi-3-mini (3.8B) | 3.82B | Microsoft | Yes (Instruct) | MIT |
| Phi-3.5-mini (3.8B) | 3.82B | Microsoft | Yes (Instruct) | MIT |
| Phi-4-mini (3.8B) | 3.82B | Microsoft | Yes | MIT |
| Yi-1.5-6B / 9B | 6B/9B | 01.AI | Yes (Chat) | Apache 2.0 |
| InternLM2.5-7B | 7B | Shanghai AI Lab | Yes (Chat) | Apache 2.0 |
| Gemma-2-9B-IT | 9.24B | Google | Yes | Gemma License |
| DeepSeek-V2-Lite | 15.7B (2.4B active) | DeepSeek | Yes (Chat) | MIT |
| CodeLlama-7B | 7B | Meta | Yes (Instruct) | Llama 2 Community |

**Status:**
- **LLaMA-3-8B ✅ already evaluated** — 100% exact match.
- **Mistral-7B — should evaluate.** Apache 2.0 licensed, strong on structured generation, sliding window attention is efficient for streaming/edge use.
- **Qwen2.5-7B — should evaluate.** Likely the strongest 7B model for structured JSON output; strong tool-use/function-calling capabilities.
- **Phi-3.5-mini / Phi-4-mini — should evaluate.** At only 3.8B params, they perform like 7B models; MIT licensed; ideal for on-device deployment.
- **Gemma-2-9B-IT — should evaluate.** Same family as our Gemma-2B; measures scaling within one architecture.
- **Yi-1.5, InternLM2.5 — excluded (lower priority).** Strong general models but no specific advantage over Qwen2.5-7B or Mistral-7B for structured output; limited community tooling for QLoRA.
- **CodeLlama-7B — excluded.** Optimized for code generation, not natural language → JSON mapping; would underperform instruction-tuned general models.
- **DeepSeek-V2-Lite — excluded.** MoE architecture requires specialized serving infrastructure; not practical for smart home edge deployment.

---

### 1.4 10B–30B Parameter Models (Large LLMs)

| Model | Params | Developer | Instruct Variant | License |
|---|---|---|---|---|
| LLaMA-3.1-70B | 70.6B | Meta | Yes (Instruct) | Llama 3.1 Community |
| Phi-3-medium (14B) | 14B | Microsoft | Yes (Instruct) | MIT |
| Phi-4 (14B) | 14B | Microsoft | Yes | MIT |
| Qwen2.5-14B / 32B | 14B/32B | Alibaba | Yes (Instruct) | Apache 2.0 |
| Mistral-Small (22B) | 22B | Mistral AI | Yes (Instruct) | Apache 2.0 |
| Yi-1.5-34B | 34B | 01.AI | Yes (Chat) | Apache 2.0 |
| CodeStral (22B) | 22B | Mistral AI | Yes | MNPL |
| Command-R (35B) | 35B | Cohere | Yes | CC-BY-NC-4.0 |

**Why not considered (as primary fine-tuning targets):**
- **Deployment infeasible.** A smart home controller (hub, RPi, edge device) cannot run 14B+ models. Even quantized, 14B models need ≥10GB VRAM.
- **Diminishing returns.** LLaMA-3-8B already achieves 100% on the test set. Scaling up parameters adds cost without measurable accuracy gain on the current benchmark.
- **Fine-tuning cost.** QLoRA on 14B+ models requires significantly more GPU memory and time; not justified when 8B already saturates.
- **However:** We use Qwen2.5-14B and LLaMA-3.1-70B as **zero-shot / few-shot baselines** (no fine-tuning) to establish an upper bound and see how much fine-tuning a small model closes the gap with a large model's zero-shot ability.

---

### 1.5 70B+ Parameter Models (Very Large LLMs)

| Model | Params | Developer | License |
|---|---|---|---|
| LLaMA-3.1-70B / 405B | 70B/405B | Meta | Llama 3.1 Community |
| Qwen2.5-72B | 72B | Alibaba | Apache 2.0 |
| Mixtral-8x7B / 8x22B | 46.7B/141B | Mistral AI | Apache 2.0 |
| DeepSeek-V3 | 671B (37B active) | DeepSeek | MIT |
| DBRX | 132B (36B active) | Databricks | Open |
| Falcon-40B / 180B | 40B/180B | TII | Apache 2.0 |
| Command-R+ (104B) | 104B | Cohere | CC-BY-NC-4.0 |

**Why excluded from fine-tuning:**
- Completely impractical for edge smart home deployment.
- Even with QLoRA, 70B+ models require ≥40GB VRAM for inference.
- **Used only as zero-shot baselines** via API to establish ceiling performance.

---

## 2. Recommended Experiment Plan

### Phase A: Model Scaling Study (Accuracy vs Size)

Fine-tune all models below with QLoRA on the same 100K dataset and evaluate:

| Model | Params | Rationale |
|---|---|---|
| SmolLM2-360M | 360M | Can the smallest models handle this at all? |
| Qwen2.5-0.5B | 0.5B | Minimum deployable on RPi? |
| Qwen2.5-1.5B | 1.5B | Sweet spot for edge? |
| LLaMA-3.2-1B | 1.24B | LLaMA family small end |
| LLaMA-3.2-3B | 3.21B | LLaMA family mid |
| Gemma-2-2B-IT | 2.61B | ✅ Done (99.4%) |
| Phi-3.5-mini | 3.82B | Microsoft's efficient SLM |
| Mistral-7B-v0.3 | 7.24B | Strong open 7B |
| Qwen2.5-7B | 7.62B | Strongest open 7B |
| LLaMA-3-8B | 8.03B | ✅ Done (100%) |

**Goal:** Plot accuracy vs model size → find the *minimum model size* that achieves ≥95% exact match. This is a real research contribution — deployability analysis.

### Phase B: Zero-Shot / Few-Shot Baselines (No Fine-Tuning)

Test without any fine-tuning to measure how much QLoRA helps:

| Model | Params | Mode |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | Zero-shot, 3-shot, 5-shot |
| LLaMA-3.2-3B-Instruct | 3B | Zero-shot, 3-shot, 5-shot |
| Qwen2.5-7B-Instruct | 7B | Zero-shot, 3-shot, 5-shot |
| LLaMA-3-8B-Instruct | 8B | Zero-shot, 3-shot, 5-shot |
| Qwen2.5-14B-Instruct | 14B | Zero-shot, 3-shot |
| LLaMA-3.1-70B-Instruct | 70B | Zero-shot (API) |

**Goal:** Show that fine-tuning a 2B model outperforms zero-shot 14B → justifies the fine-tuning approach.

### Phase C: Informal/Casual Command Robustness (Professor's Request)

Test fine-tuned models on commands that **do not appear in the training distribution**:

| Category | Examples |
|---|---|
| Slang/colloquial | "yo turn off everything", "kill the telly" |
| Implicit intent | "it's too hot in here", "I can't see anything" |
| Typos/misspellings | "trun on teh lights", "bedrrom fan on" |
| Abbreviations | "AC 22 bed", "lights off LR" |
| Multi-turn context | "same in the kitchen" (following a previous command) |
| Ambiguous | "make it cozy", "set it up for guests" |
| Code-switching | "bedroom mein light on karo" (Hindi-English) |
| Negation | "don't turn off the fan", "keep the lights as they are" |

**Goal:** Expose the failure modes of template-trained models. Show that fine-tuned models are brittle to distribution shift → motivates the need for more diverse training data or larger pretrained models.

---

## 3. Summary: Why Each Model Is or Isn't Selected

| Model | Decision | Reason |
|---|---|---|
| SmolLM2-135M | Skip | Too small, unlikely to produce valid JSON |
| SmolLM2-360M | **Evaluate** | Minimum viable SLM experiment |
| Qwen2.5-0.5B | **Evaluate** | Edge deployment candidate |
| TinyLlama-1.1B | Skip | Older architecture, superseded by Qwen2.5-1.5B/LLaMA-3.2-1B |
| Phi-1.5 | Skip | No instruct variant, not chat-tuned |
| OLMo-1B | Skip | Research model, no instruct variant, limited practical tooling |
| Qwen2.5-1.5B | **Evaluate** | Strong SLM, potential edge sweet spot |
| LLaMA-3.2-1B | **Evaluate** | Same-family scaling study |
| LLaMA-3.2-3B | **Evaluate** | Same-family scaling study |
| Gemma-2-2B-IT | ✅ Done | 99.4% exact match |
| SmolLM2-1.7B | Skip | Similar size to Qwen2.5-1.5B which is stronger; avoid redundancy |
| Phi-2 | Skip | No instruct variant |
| StableLM-2-1.6B | Skip | Non-commercial license |
| Phi-3.5-mini | **Evaluate** | 3.8B but performs like 7B; MIT license; edge-friendly |
| Mistral-7B | **Evaluate** | Apache 2.0, strong structured output |
| Qwen2.5-7B | **Evaluate** | Likely strongest open 7B for JSON tasks |
| LLaMA-3-8B | ✅ Done | 100% exact match |
| Gemma-2-9B | Skip | Similar size to LLaMA-8B which already saturates; diminishing returns |
| Yi-1.5-6B/9B | Skip | No advantage over Qwen2.5-7B; less community tooling |
| InternLM2.5-7B | Skip | No advantage over Qwen2.5-7B for English structured output |
| CodeLlama-7B | Skip | Code-optimized, not NL→JSON; instruction following is weaker |
| DeepSeek-V2-Lite | Skip | MoE needs specialized serving; impractical for edge |
| Phi-4 (14B) | Zero-shot only | Too large for edge fine-tuning; useful as baseline |
| Qwen2.5-14B | Zero-shot only | Upper bound reference |
| Mistral-Small (22B) | Skip | Too large, no edge deployment path |
| LLaMA-3.1-70B | Zero-shot only | Ceiling performance reference |
| Qwen2.5-72B | Skip | Redundant with 70B baseline; API costs |
| Mixtral-8x7B | Skip | MoE, 46.7B total params, needs specialized infra |
| DeepSeek-V3 | Skip | 671B params, impractical |
| DBRX | Skip | 132B, MoE, impractical for this use case |
| Falcon-40B/180B | Skip | Older models, superseded by LLaMA-3/Qwen2.5 |
| Command-R/R+ | Skip | Non-commercial license (CC-BY-NC); Cohere-specific tooling |

---

## 4. Key Research Questions This Enables

1. **What is the minimum model size** that achieves ≥95% exact match on structured smart home commands?
2. **Does fine-tuning a 2B model beat zero-shot 14B/70B?** (Justifies the fine-tuning approach over simply calling a bigger API)
3. **How brittle are template-trained models to informal commands?** (Motivates better data augmentation)
4. **Which architecture family (LLaMA vs Qwen vs Phi vs Gemma) is best for structured output at small scale?**
5. **Can sub-1B models handle this task at all?** (Edge deployment feasibility)
