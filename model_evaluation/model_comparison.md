# Smart-Home LLM Comparison Table

_Test sample: n = 200 examples drawn from `smart_home_100k_clean.csv` (seed 20260501)._

_Anchored real measurements: **LLaMA-3-8B-Instruct = 100.00%**, **Gemma-2-2B-IT = 99.40%** (from `smart_home/results/results.txt`, n=500). All other rows are heuristic simulations and are clearly labelled below._


## Params vs. Accuracy (focused view)

All 40+ models in one table, sorted by parameter count. **Accuracy = exact-match %** on the structured-JSON task.

| Model | Parameters | Status | Accuracy (Exact Match %) |
|---|---:|---|---:|
| SmolLM2-135M | 0.135 B | excluded *(simulated, what-if)* | 16.22 |
| SmolLM2-360M | 0.36 B | fine-tune *(simulated, fine-tuned)* | 87.56 |
| Qwen2.5-0.5B-Instruct | 0.494 B | fine-tune *(simulated, fine-tuned)* | 92.14 |
| Danube3-500M | 0.5 B | excluded *(simulated, what-if)* | 24.57 |
| OLMo-1B | 1 B | excluded *(simulated, what-if)* | 11.68 |
| TinyLlama-1.1B-Chat | 1.1 B | excluded *(simulated, what-if)* | 38.99 |
| LLaMA-3.2-1B-Instruct | 1.24 B | fine-tune *(simulated, fine-tuned)* | 97.41 |
| Phi-1.5 | 1.3 B | excluded *(simulated, what-if)* | 19.49 |
| Qwen2.5-1.5B-Instruct | 1.5 B | fine-tune *(simulated, fine-tuned)* | 97.78 |
| StableLM-2-1.6B-Zephyr | 1.6 B | excluded *(simulated, what-if)* | 51.74 |
| SmolLM2-1.7B-Instruct | 1.7 B | excluded *(simulated, what-if)* | 53.44 |
| Gemma-2-2B-IT | 2.61 B | evaluated **[measured]** | 99.40 |
| Phi-2 | 2.7 B | excluded *(simulated, what-if)* | 39.38 |
| Qwen2.5-3B-Instruct | 3 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| LLaMA-3.2-3B-Instruct | 3.21 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| Phi-3-mini-Instruct | 3.82 B | excluded *(simulated, what-if)* | 76.39 |
| Phi-3.5-mini-Instruct | 3.82 B | fine-tune *(simulated, fine-tuned)* | 99.35 |
| Phi-4-mini | 3.82 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| Danube3-4B | 4 B | excluded *(simulated, what-if)* | 76.35 |
| Yi-1.5-6B-Chat | 6 B | excluded *(simulated, what-if)* | 81.62 |
| InternLM2.5-7B-Chat | 7 B | excluded *(simulated, what-if)* | 83.76 |
| CodeLlama-7B-Instruct | 7 B | excluded *(simulated, what-if)* | 83.24 |
| Mistral-7B-v0.3-Instruct | 7.24 B | fine-tune *(simulated, fine-tuned)* | 99.81 |
| Qwen2.5-7B-Instruct | 7.62 B | fine-tune *(simulated, fine-tuned)* | 99.95 |
| LLaMA-3-8B-Instruct | 8.03 B | evaluated **[measured]** | 100.00 |
| LLaMA-3.1-8B-Instruct | 8.03 B | excluded *(simulated, what-if)* | 86.48 |
| Yi-1.5-9B-Chat | 9 B | excluded *(simulated, what-if)* | 86.52 |
| Gemma-2-9B-IT | 9.24 B | excluded *(simulated, what-if)* | 86.85 |
| Phi-3-medium-Instruct | 14 B | excluded *(simulated, what-if)* | 91.26 |
| Phi-4 | 14 B | excluded *(simulated, what-if)* | 91.25 |
| Qwen2.5-14B-Instruct | 14 B | zero-shot *(simulated, zero-shot)* | 91.41 |
| DeepSeek-V2-Lite | 15.7 B | excluded *(simulated, what-if)* | 89.28 |
| Mistral-Small-Instruct | 22 B | excluded *(simulated, what-if)* | 93.18 |
| CodeStral | 22 B | excluded *(simulated, what-if)* | 92.27 |
| Qwen2.5-32B-Instruct | 32 B | excluded *(simulated, what-if)* | 94.07 |
| Yi-1.5-34B-Chat | 34 B | excluded *(simulated, what-if)* | 94.05 |
| Command-R | 35 B | excluded *(simulated, what-if)* | 94.59 |
| Falcon-40B | 40 B | excluded *(simulated, what-if)* | 93.15 |
| Mixtral-8x7B | 46.7 B | excluded *(simulated, what-if)* | 93.70 |
| LLaMA-3.1-70B-Instruct | 70 B | zero-shot *(simulated, zero-shot)* | 96.57 |
| Qwen2.5-72B-Instruct | 72 B | excluded *(simulated, what-if)* | 96.16 |
| Command-R+ | 104 B | excluded *(simulated, what-if)* | 96.01 |
| DBRX | 132 B | excluded *(simulated, what-if)* | 94.59 |
| Mixtral-8x22B | 141 B | excluded *(simulated, what-if)* | 95.63 |
| Falcon-180B | 180 B | excluded *(simulated, what-if)* | 95.19 |
| LLaMA-3.1-405B-Instruct | 405 B | excluded *(simulated, what-if)* | 98.11 |
| DeepSeek-V3 | 671 B | excluded *(simulated, what-if)* | 95.90 |


## Already evaluated (real measurements)

These two rows are real test-set numbers (n=500).

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8.03B | LLaMA-3 | Llama 3 Community | evaluated | 100.00 | 100.00 | 100.00 | 100.00 | 1264 | 6.3 |
| Gemma-2-2B-IT | 2.61B | Gemma-2 | Gemma | evaluated | 99.40 | 100.00 | 100.00 | 99.79 | 481 | 3.1 |

## Planned for fine-tuning (simulated)

Numbers are simulated heuristic estimates of QLoRA fine-tuned performance on this task.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 3B | Qwen2.5 | Apache-2.0 | fine-tune | 99.95 | 99.70 | 100.00 | 100.00 | 538 | 3.3 |
| LLaMA-3.2-3B-Instruct | 3.21B | LLaMA-3.2 | Llama 3.2 Community | fine-tune | 99.95 | 99.49 | 100.00 | 100.00 | 559 | 3.4 |
| Phi-4-mini | 3.82B | Phi-4 | MIT | fine-tune | 99.95 | 99.77 | 100.00 | 100.00 | 662 | 3.8 |
| Qwen2.5-7B-Instruct | 7.62B | Qwen2.5 | Apache-2.0 | fine-tune | 99.95 | 98.90 | 100.00 | 100.00 | 1238 | 6.1 |
| Mistral-7B-v0.3-Instruct | 7.24B | Mistral | Apache-2.0 | fine-tune | 99.81 | 99.34 | 100.00 | 100.00 | 1154 | 5.8 |
| Phi-3.5-mini-Instruct | 3.82B | Phi-3.5 | MIT | fine-tune | 99.35 | 99.36 | 100.00 | 99.74 | 673 | 3.8 |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2.5 | Apache-2.0 | fine-tune | 97.78 | 99.87 | 99.02 | 98.13 | 302 | 2.4 |
| LLaMA-3.2-1B-Instruct | 1.24B | LLaMA-3.2 | Llama 3.2 Community | fine-tune | 97.41 | 96.33 | 99.04 | 97.93 | 265 | 2.2 |
| Qwen2.5-0.5B-Instruct | 0.494B | Qwen2.5 | Apache-2.0 | fine-tune | 92.14 | 91.47 | 93.34 | 92.53 | 170 | 1.8 |
| SmolLM2-360M | 0.36B | SmolLM2 | Apache-2.0 | fine-tune | 87.56 | 91.84 | 88.89 | 88.01 | 127 | 1.7 |

## Zero-shot baselines (simulated)

Zero-shot, no fine-tuning. Estimated penalty applied.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.1-70B-Instruct | 70B | LLaMA-3.1 | Llama 3.1 Community | zero-shot | 96.57 | 99.23 | 98.18 | 96.93 | 10476 | 43.5 |
| Qwen2.5-14B-Instruct | 14B | Qwen2.5 | Apache-2.0 | zero-shot | 91.41 | 99.42 | 93.00 | 91.78 | 2158 | 9.9 |

## Excluded models (what-if simulated)

Numbers are speculative *zero-shot* estimates included only for completeness; these models are excluded from the study for the reasons listed below.

| Model | Params | Family | License | Status | EM % | ValidJSON % | Room F1 | Dev-Val % | Latency ms | VRAM 4-bit GB |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.1-405B-Instruct | 405B | LLaMA-3.1 | Llama 3.1 Community | excluded | 98.11 | 98.94 | 99.37 | 98.46 | 60172 | 244.5 |
| Qwen2.5-72B-Instruct | 72B | Qwen2.5 | Apache-2.0 | excluded | 96.16 | 99.65 | 97.45 | 96.73 | 10773 | 44.7 |
| Command-R+ | 104B | Command-R | CC-BY-NC-4.0 | excluded | 96.01 | 99.35 | 97.65 | 96.42 | 15502 | 63.9 |
| DeepSeek-V3 | 671B | DeepSeek-V3 | MIT | excluded | 95.90 | 99.44 | 97.36 | 96.16 | 1375 | 370.6 |
| Mixtral-8x22B | 141B | Mixtral | Apache-2.0 | excluded | 95.63 | 99.83 | 96.90 | 96.02 | 1440 | 79.1 |
| Falcon-180B | 180B | Falcon | Apache-2.0 | excluded | 95.19 | 99.46 | 96.40 | 95.41 | 26795 | 109.5 |
| Command-R | 35B | Command-R | CC-BY-NC-4.0 | excluded | 94.59 | 98.89 | 96.37 | 95.17 | 5277 | 22.5 |
| DBRX | 132B | DBRX | Open | excluded | 94.59 | 99.83 | 96.28 | 94.97 | 1361 | 74.1 |
| Qwen2.5-32B-Instruct | 32B | Qwen2.5 | Apache-2.0 | excluded | 94.07 | 99.60 | 95.56 | 94.47 | 4845 | 20.7 |
| Yi-1.5-34B-Chat | 34B | Yi-1.5 | Apache-2.0 | excluded | 94.05 | 99.57 | 95.39 | 94.48 | 5137 | 21.9 |
| Mixtral-8x7B | 46.7B | Mixtral | Apache-2.0 | excluded | 93.70 | 98.98 | 95.10 | 94.02 | 513 | 27.2 |
| Mistral-Small-Instruct | 22B | Mistral | Apache-2.0 | excluded | 93.18 | 99.44 | 94.43 | 93.48 | 3346 | 14.7 |
| Falcon-40B | 40B | Falcon | Apache-2.0 | excluded | 93.15 | 99.24 | 94.69 | 93.36 | 6008 | 25.5 |
| CodeStral | 22B | CodeStral | MNPL | excluded | 92.27 | 99.82 | 93.97 | 92.69 | 3339 | 14.7 |
| Phi-3-medium-Instruct | 14B | Phi-3 | MIT | excluded | 91.26 | 99.77 | 92.53 | 91.68 | 2167 | 9.9 |
| Phi-4 | 14B | Phi-4 | MIT | excluded | 91.25 | 98.86 | 92.75 | 91.63 | 2151 | 9.9 |
| DeepSeek-V2-Lite | 15.7B | DeepSeek-V2 | MIT | excluded | 89.28 | 98.84 | 90.81 | 89.85 | 182 | 10.1 |
| Gemma-2-9B-IT | 9.24B | Gemma-2 | Gemma | excluded | 86.85 | 99.68 | 88.44 | 87.38 | 1451 | 7.0 |
| Yi-1.5-9B-Chat | 9B | Yi-1.5 | Apache-2.0 | excluded | 86.52 | 99.15 | 88.18 | 86.98 | 1442 | 6.9 |
| LLaMA-3.1-8B-Instruct | 8.03B | LLaMA-3.1 | Llama 3.1 Community | excluded | 86.48 | 99.81 | 88.26 | 87.08 | 1264 | 6.3 |
| InternLM2.5-7B-Chat | 7B | InternLM2.5 | Apache-2.0 | excluded | 83.76 | 99.82 | 85.55 | 84.09 | 1140 | 5.7 |
| CodeLlama-7B-Instruct | 7B | CodeLlama | Llama 2 Community | excluded | 83.24 | 98.93 | 84.55 | 83.69 | 1140 | 5.7 |
| Yi-1.5-6B-Chat | 6B | Yi-1.5 | Apache-2.0 | excluded | 81.62 | 99.89 | 83.08 | 82.01 | 967 | 5.1 |
| Phi-3-mini-Instruct | 3.82B | Phi-3 | MIT | excluded | 76.39 | 99.41 | 78.17 | 76.92 | 667 | 3.8 |
| Danube3-4B | 4B | Danube3 | Apache-2.0 | excluded | 76.35 | 99.33 | 77.94 | 76.90 | 686 | 3.9 |
| SmolLM2-1.7B-Instruct | 1.7B | SmolLM2 | Apache-2.0 | excluded | 53.44 | 99.89 | 55.12 | 54.03 | 337 | 2.5 |
| StableLM-2-1.6B-Zephyr | 1.6B | StableLM-2 | Non-Commercial | excluded | 51.74 | 98.91 | 53.33 | 52.01 | 312 | 2.5 |
| Phi-2 | 2.7B | Phi | MIT | excluded | 39.38 | 90.97 | 40.94 | 39.80 | 479 | 3.1 |
| TinyLlama-1.1B-Chat | 1.1B | TinyLlama | Apache-2.0 | excluded | 38.99 | 96.51 | 40.62 | 39.39 | 263 | 2.2 |
| Danube3-500M | 0.5B | Danube3 | Apache-2.0 | excluded | 24.57 | 91.32 | 25.94 | 24.95 | 164 | 1.8 |
| Phi-1.5 | 1.3B | Phi | MIT | excluded | 19.49 | 88.55 | 20.97 | 19.81 | 280 | 2.3 |
| SmolLM2-135M | 0.135B | SmolLM2 | Apache-2.0 | excluded | 16.22 | 91.78 | 17.97 | 16.71 | 100 | 1.6 |
| OLMo-1B | 1B | OLMo | Apache-2.0 | excluded | 11.68 | 88.23 | 13.13 | 11.91 | 240 | 2.1 |

## Exclusion rationale (full)

- **SmolLM2-135M**: Too small to reliably emit valid multi-room JSON.
- **OLMo-1B**: No instruction-tuned variant; research-only tooling.
- **TinyLlama-1.1B-Chat**: Older arch superseded by Qwen2.5-1.5B / LLaMA-3.2-1B.
- **Phi-1.5**: Base only; needs custom chat template and more data.
- **Phi-2**: No instruction-tuned variant.
- **StableLM-2-1.6B-Zephyr**: Non-commercial license blocks product deployment.
- **SmolLM2-1.7B-Instruct**: Redundant with stronger Qwen2.5-1.5B.
- **Danube3-500M**: Less established family; limited QLoRA tooling.
- **Danube3-4B**: Less established family; limited QLoRA tooling.
- **LLaMA-3.1-8B-Instruct**: Near-identical to LLaMA-3-8B for this task.
- **Phi-3-mini-Instruct**: Superseded by Phi-3.5-mini and Phi-4-mini.
- **Yi-1.5-6B-Chat**: No clear advantage over Qwen2.5-7B / Mistral-7B.
- **Yi-1.5-9B-Chat**: No clear advantage over Qwen2.5-7B / Mistral-7B.
- **InternLM2.5-7B-Chat**: Optimized for Chinese; no edge over Qwen2.5-7B in EN.
- **Gemma-2-9B-IT**: LLaMA-3-8B already saturates at 100%; no new insight.
- **DeepSeek-V2-Lite**: MoE; needs specialized serving for edge.
- **CodeLlama-7B-Instruct**: Code-tuned, weaker NL instruction following.
- **Phi-3-medium-Instruct**: Too large for edge; no accuracy gain over 8B.
- **Phi-4**: Too large for edge; no accuracy gain over 8B.
- **Qwen2.5-32B-Instruct**: Diminishing returns beyond 14B baseline.
- **Mistral-Small-Instruct**: Too large for edge; no edge over Qwen2.5-14B.
- **Yi-1.5-34B-Chat**: Too large; less established at this scale.
- **CodeStral**: Code-focused; restrictive license.
- **Command-R**: Non-commercial license.
- **LLaMA-3.1-405B-Instruct**: Multi-GPU inference; impractical.
- **Qwen2.5-72B-Instruct**: Redundant with LLaMA-3.1-70B baseline.
- **Mixtral-8x7B**: MoE; full 46.7B must be loaded for inference.
- **Mixtral-8x22B**: MoE at 141B; impractical.
- **DeepSeek-V3**: 671B MoE; custom infra; not edge-relevant.
- **DBRX**: MoE; serving complexity.
- **Falcon-40B**: Older (2023); superseded by LLaMA-3 / Qwen2.5.
- **Falcon-180B**: Older and very large.
- **Command-R+**: Non-commercial; impractical for edge.