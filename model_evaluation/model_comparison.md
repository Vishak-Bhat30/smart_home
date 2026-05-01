# Smart-Home LLM Comparison Table

_Test sample: n = 200 examples drawn from `smart_home_100k_clean.csv` (seed 20260501)._

_Anchored real measurements: **LLaMA-3-8B-Instruct = 100.00%**, **Gemma-2-2B-IT = 99.40%** (from `smart_home/results/results.txt`, n=500). All other rows are heuristic simulations and are clearly labelled below._


## Params vs. Accuracy (focused view)

All 40+ models in one table, sorted by parameter count. **Accuracy = exact-match %** on the structured-JSON task.

| Model | Parameters | Status | Accuracy (Exact Match %) |
|---|---:|---|---:|
| SmolLM2-135M | 0.135 B | excluded *(simulated, what-if)* | 16.03 |
| SmolLM2-360M | 0.36 B | fine-tune *(simulated, fine-tuned)* | 87.71 |
| Qwen2.5-0.5B-Instruct | 0.494 B | fine-tune *(simulated, fine-tuned)* | 92.67 |
| Danube3-500M | 0.5 B | excluded *(simulated, what-if)* | 24.47 |
| OLMo-1B | 1 B | excluded *(simulated, what-if)* | 12.12 |
| TinyLlama-1.1B-Chat | 1.1 B | excluded *(simulated, what-if)* | 39.01 |
| LLaMA-3.2-1B-Instruct | 1.24 B | fine-tune *(simulated, fine-tuned)* | 97.05 |
| Phi-1.5 | 1.3 B | excluded *(simulated, what-if)* | 18.87 |
| Qwen2.5-1.5B-Instruct | 1.5 B | fine-tune *(simulated, fine-tuned)* | 97.79 |
| StableLM-2-1.6B-Zephyr | 1.6 B | excluded *(simulated, what-if)* | 51.62 |
| SmolLM2-1.7B-Instruct | 1.7 B | excluded *(simulated, what-if)* | 53.12 |
| Gemma-2-2B-IT | 2.61 B | evaluated **[measured]** | 99.40 |
| Phi-2 | 2.7 B | excluded *(simulated, what-if)* | 39.61 |
| Qwen2.5-3B-Instruct | 3 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| LLaMA-3.2-3B-Instruct | 3.21 B | fine-tune *(simulated, fine-tuned)* | 99.21 |
| Phi-3-mini-Instruct | 3.82 B | excluded *(simulated, what-if)* | 75.64 |
| Phi-3.5-mini-Instruct | 3.82 B | fine-tune *(simulated, fine-tuned)* | 99.28 |
| Phi-4-mini | 3.82 B | fine-tune *(simulated, fine-tuned)* | 99.69 |
| Danube3-4B | 4 B | excluded *(simulated, what-if)* | 75.59 |
| Yi-1.5-6B-Chat | 6 B | excluded *(simulated, what-if)* | 81.76 |
| InternLM2.5-7B-Chat | 7 B | excluded *(simulated, what-if)* | 84.16 |
| CodeLlama-7B-Instruct | 7 B | excluded *(simulated, what-if)* | 83.24 |
| Mistral-7B-v0.3-Instruct | 7.24 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| Qwen2.5-7B-Instruct | 7.62 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| LLaMA-3-8B-Instruct | 8.03 B | evaluated **[measured]** | 100.00 |
| LLaMA-3.1-8B-Instruct | 8.03 B | excluded *(simulated, what-if)* | 85.77 |
| Yi-1.5-9B-Chat | 9 B | excluded *(simulated, what-if)* | 86.70 |
| Gemma-2-9B-IT | 9.24 B | excluded *(simulated, what-if)* | 87.31 |
| Phi-3-medium-Instruct | 14 B | excluded *(simulated, what-if)* | 90.76 |
| Phi-4 | 14 B | excluded *(simulated, what-if)* | 91.63 |
| Qwen2.5-14B-Instruct | 14 B | zero-shot *(simulated, zero-shot)* | 91.70 |
| DeepSeek-V2-Lite | 15.7 B | excluded *(simulated, what-if)* | 89.52 |
| Mistral-Small-Instruct | 22 B | excluded *(simulated, what-if)* | 92.73 |
| CodeStral | 22 B | excluded *(simulated, what-if)* | 92.30 |
| Qwen2.5-32B-Instruct | 32 B | excluded *(simulated, what-if)* | 94.40 |
| Yi-1.5-34B-Chat | 34 B | excluded *(simulated, what-if)* | 93.65 |
| Command-R | 35 B | excluded *(simulated, what-if)* | 94.30 |
| Falcon-40B | 40 B | excluded *(simulated, what-if)* | 92.25 |
| Mixtral-8x7B | 46.7 B | excluded *(simulated, what-if)* | 93.15 |
| LLaMA-3.1-70B-Instruct | 70 B | zero-shot *(simulated, zero-shot)* | 95.80 |
| Qwen2.5-72B-Instruct | 72 B | excluded *(simulated, what-if)* | 96.47 |
| Command-R+ | 104 B | excluded *(simulated, what-if)* | 97.06 |
| DBRX | 132 B | excluded *(simulated, what-if)* | 95.15 |
| Mixtral-8x22B | 141 B | excluded *(simulated, what-if)* | 95.66 |
| Falcon-180B | 180 B | excluded *(simulated, what-if)* | 94.87 |
| LLaMA-3.1-405B-Instruct | 405 B | excluded *(simulated, what-if)* | 97.95 |
| DeepSeek-V3 | 671 B | excluded *(simulated, what-if)* | 96.13 |


## Already evaluated (real measurements)

These two rows are real test-set numbers (n=500).

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8.03B | LLaMA-3 | Llama 3 Community | evaluated | 100.00 | 100.00 | 100.00 | 100.00 | 1276 | 6.3 |
| Gemma-2-2B-IT | 2.61B | Gemma-2 | Gemma | evaluated | 99.40 | 100.00 | 100.00 | 99.79 | 481 | 3.1 |

## Planned for fine-tuning (simulated)

Numbers are simulated heuristic estimates of QLoRA fine-tuned performance on this task.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 3B | Qwen2.5 | Apache-2.0 | fine-tune | 99.95 | 99.78 | 100.00 | 100.00 | 530 | 3.3 |
| Mistral-7B-v0.3-Instruct | 7.24B | Mistral | Apache-2.0 | fine-tune | 99.95 | 99.20 | 100.00 | 100.00 | 1158 | 5.8 |
| Qwen2.5-7B-Instruct | 7.62B | Qwen2.5 | Apache-2.0 | fine-tune | 99.95 | 98.94 | 100.00 | 100.00 | 1229 | 6.1 |
| Phi-4-mini | 3.82B | Phi-4 | MIT | fine-tune | 99.69 | 99.76 | 100.00 | 100.00 | 642 | 3.8 |
| Phi-3.5-mini-Instruct | 3.82B | Phi-3.5 | MIT | fine-tune | 99.28 | 98.99 | 100.00 | 99.56 | 654 | 3.8 |
| LLaMA-3.2-3B-Instruct | 3.21B | LLaMA-3.2 | Llama 3.2 Community | fine-tune | 99.21 | 99.13 | 100.00 | 99.58 | 552 | 3.4 |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2.5 | Apache-2.0 | fine-tune | 97.79 | 99.35 | 99.05 | 98.10 | 316 | 2.4 |
| LLaMA-3.2-1B-Instruct | 1.24B | LLaMA-3.2 | Llama 3.2 Community | fine-tune | 97.05 | 96.60 | 98.53 | 97.37 | 263 | 2.2 |
| Qwen2.5-0.5B-Instruct | 0.494B | Qwen2.5 | Apache-2.0 | fine-tune | 92.67 | 91.52 | 93.99 | 93.00 | 154 | 1.8 |
| SmolLM2-360M | 0.36B | SmolLM2 | Apache-2.0 | fine-tune | 87.71 | 91.24 | 89.29 | 88.26 | 150 | 1.7 |

## Zero-shot baselines (simulated)

Zero-shot, no fine-tuning. Estimated penalty applied.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.1-70B-Instruct | 70B | LLaMA-3.1 | Llama 3.1 Community | zero-shot | 95.80 | 99.27 | 97.26 | 96.28 | 10472 | 43.5 |
| Qwen2.5-14B-Instruct | 14B | Qwen2.5 | Apache-2.0 | zero-shot | 91.70 | 99.05 | 93.49 | 92.01 | 2152 | 9.9 |

## Excluded models (what-if simulated)

Numbers are speculative *zero-shot* estimates included only for completeness; these models are excluded from the study for the reasons listed below.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.1-405B-Instruct | 405B | LLaMA-3.1 | Llama 3.1 Community | excluded | 97.95 | 99.74 | 99.72 | 98.37 | 60180 | 244.5 |
| Command-R+ | 104B | Command-R | CC-BY-NC-4.0 | excluded | 97.06 | 99.44 | 98.31 | 97.49 | 15516 | 63.9 |
| Qwen2.5-72B-Instruct | 72B | Qwen2.5 | Apache-2.0 | excluded | 96.47 | 99.16 | 97.90 | 97.00 | 10760 | 44.7 |
| DeepSeek-V3 | 671B | DeepSeek-V3 | MIT | excluded | 96.13 | 99.34 | 97.49 | 96.55 | 1375 | 370.6 |
| Mixtral-8x22B | 141B | Mixtral | Apache-2.0 | excluded | 95.66 | 99.90 | 97.11 | 95.90 | 1461 | 79.1 |
| DBRX | 132B | DBRX | Open | excluded | 95.15 | 99.36 | 96.50 | 95.40 | 1359 | 74.1 |
| Falcon-180B | 180B | Falcon | Apache-2.0 | excluded | 94.87 | 99.49 | 96.26 | 95.17 | 26773 | 109.5 |
| Qwen2.5-32B-Instruct | 32B | Qwen2.5 | Apache-2.0 | excluded | 94.40 | 99.68 | 96.16 | 94.82 | 4820 | 20.7 |
| Command-R | 35B | Command-R | CC-BY-NC-4.0 | excluded | 94.30 | 99.23 | 96.01 | 94.77 | 5290 | 22.5 |
| Yi-1.5-34B-Chat | 34B | Yi-1.5 | Apache-2.0 | excluded | 93.65 | 99.88 | 94.93 | 93.89 | 5149 | 21.9 |
| Mixtral-8x7B | 46.7B | Mixtral | Apache-2.0 | excluded | 93.15 | 98.93 | 94.61 | 93.73 | 516 | 27.2 |
| Mistral-Small-Instruct | 22B | Mistral | Apache-2.0 | excluded | 92.73 | 98.93 | 94.05 | 93.06 | 3337 | 14.7 |
| CodeStral | 22B | CodeStral | MNPL | excluded | 92.30 | 99.18 | 93.73 | 92.66 | 3349 | 14.7 |
| Falcon-40B | 40B | Falcon | Apache-2.0 | excluded | 92.25 | 99.32 | 93.99 | 92.46 | 6006 | 25.5 |
| Phi-4 | 14B | Phi-4 | MIT | excluded | 91.63 | 99.56 | 92.88 | 91.96 | 2158 | 9.9 |
| Phi-3-medium-Instruct | 14B | Phi-3 | MIT | excluded | 90.76 | 99.68 | 92.21 | 91.03 | 2167 | 9.9 |
| DeepSeek-V2-Lite | 15.7B | DeepSeek-V2 | MIT | excluded | 89.52 | 99.69 | 90.95 | 90.04 | 172 | 10.1 |
| Gemma-2-9B-IT | 9.24B | Gemma-2 | Gemma | excluded | 87.31 | 99.54 | 88.95 | 87.78 | 1462 | 7.0 |
| Yi-1.5-9B-Chat | 9B | Yi-1.5 | Apache-2.0 | excluded | 86.70 | 99.54 | 88.03 | 87.24 | 1420 | 6.9 |
| LLaMA-3.1-8B-Instruct | 8.03B | LLaMA-3.1 | Llama 3.1 Community | excluded | 85.77 | 98.87 | 87.39 | 86.14 | 1270 | 6.3 |
| InternLM2.5-7B-Chat | 7B | InternLM2.5 | Apache-2.0 | excluded | 84.16 | 99.64 | 85.74 | 84.62 | 1126 | 5.7 |
| CodeLlama-7B-Instruct | 7B | CodeLlama | Llama 2 Community | excluded | 83.24 | 99.99 | 84.45 | 83.65 | 1143 | 5.7 |
| Yi-1.5-6B-Chat | 6B | Yi-1.5 | Apache-2.0 | excluded | 81.76 | 99.01 | 83.03 | 82.11 | 993 | 5.1 |
| Phi-3-mini-Instruct | 3.82B | Phi-3 | MIT | excluded | 75.64 | 99.64 | 77.40 | 76.13 | 661 | 3.8 |
| Danube3-4B | 4B | Danube3 | Apache-2.0 | excluded | 75.59 | 99.92 | 76.81 | 75.84 | 678 | 3.9 |
| SmolLM2-1.7B-Instruct | 1.7B | SmolLM2 | Apache-2.0 | excluded | 53.12 | 99.42 | 54.77 | 53.44 | 331 | 2.5 |
| StableLM-2-1.6B-Zephyr | 1.6B | StableLM-2 | Non-Commercial | excluded | 51.62 | 98.80 | 52.97 | 52.16 | 344 | 2.5 |
| Phi-2 | 2.7B | Phi | MIT | excluded | 39.61 | 91.25 | 41.06 | 39.83 | 506 | 3.1 |
| TinyLlama-1.1B-Chat | 1.1B | TinyLlama | Apache-2.0 | excluded | 39.01 | 96.70 | 40.64 | 39.48 | 238 | 2.2 |
| Danube3-500M | 0.5B | Danube3 | Apache-2.0 | excluded | 24.47 | 92.31 | 26.04 | 24.89 | 155 | 1.8 |
| Phi-1.5 | 1.3B | Phi | MIT | excluded | 18.87 | 87.98 | 20.22 | 19.28 | 286 | 2.3 |
| SmolLM2-135M | 0.135B | SmolLM2 | Apache-2.0 | excluded | 16.03 | 91.55 | 17.45 | 16.43 | 95 | 1.6 |
| OLMo-1B | 1B | OLMo | Apache-2.0 | excluded | 12.12 | 88.17 | 13.50 | 12.45 | 241 | 2.1 |

## Exclusion rationale (full)

- **SmolLM2-135M** — Too small to reliably emit valid multi-room JSON.
- **OLMo-1B** — No instruction-tuned variant; research-only tooling.
- **TinyLlama-1.1B-Chat** — Older arch superseded by Qwen2.5-1.5B / LLaMA-3.2-1B.
- **Phi-1.5** — Base only; needs custom chat template and more data.
- **Phi-2** — No instruction-tuned variant.
- **StableLM-2-1.6B-Zephyr** — Non-commercial license blocks product deployment.
- **SmolLM2-1.7B-Instruct** — Redundant with stronger Qwen2.5-1.5B.
- **Danube3-500M** — Less established family; limited QLoRA tooling.
- **Danube3-4B** — Less established family; limited QLoRA tooling.
- **LLaMA-3.1-8B-Instruct** — Near-identical to LLaMA-3-8B for this task.
- **Phi-3-mini-Instruct** — Superseded by Phi-3.5-mini and Phi-4-mini.
- **Yi-1.5-6B-Chat** — No clear advantage over Qwen2.5-7B / Mistral-7B.
- **Yi-1.5-9B-Chat** — No clear advantage over Qwen2.5-7B / Mistral-7B.
- **InternLM2.5-7B-Chat** — Optimized for Chinese; no edge over Qwen2.5-7B in EN.
- **Gemma-2-9B-IT** — LLaMA-3-8B already saturates at 100%; no new insight.
- **DeepSeek-V2-Lite** — MoE; needs specialized serving for edge.
- **CodeLlama-7B-Instruct** — Code-tuned, weaker NL instruction following.
- **Phi-3-medium-Instruct** — Too large for edge; no accuracy gain over 8B.
- **Phi-4** — Too large for edge; no accuracy gain over 8B.
- **Qwen2.5-32B-Instruct** — Diminishing returns beyond 14B baseline.
- **Mistral-Small-Instruct** — Too large for edge; no edge over Qwen2.5-14B.
- **Yi-1.5-34B-Chat** — Too large; less established at this scale.
- **CodeStral** — Code-focused; restrictive license.
- **Command-R** — Non-commercial license.
- **LLaMA-3.1-405B-Instruct** — Multi-GPU inference; impractical.
- **Qwen2.5-72B-Instruct** — Redundant with LLaMA-3.1-70B baseline.
- **Mixtral-8x7B** — MoE; full 46.7B must be loaded for inference.
- **Mixtral-8x22B** — MoE at 141B; impractical.
- **DeepSeek-V3** — 671B MoE; custom infra; not edge-relevant.
- **DBRX** — MoE; serving complexity.
- **Falcon-40B** — Older (2023); superseded by LLaMA-3 / Qwen2.5.
- **Falcon-180B** — Older and very large.
- **Command-R+** — Non-commercial; impractical for edge.