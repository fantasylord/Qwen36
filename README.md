---
base_model: philbert440/Qwen3.6-27B-Uncensored-Aggressive
license: apache-2.0
tags:
- compressed-tensors
- w4a16
- awq
- vllm
---

# heretic-aggressive-w4a16-g128 — W4A16 AWQ (g128)

4-bit weight-only (W4A16) AWQ quant of [`philbert440/Qwen3.6-27B-Uncensored-Aggressive`](https://huggingface.co/philbert440/Qwen3.6-27B-Uncensored-Aggressive), built for NVIDIA V100 (SM70) under [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).

## Quant
- Format: `compressed-tensors`, pack-quantized
- Method: AWQ (`AWQModifier`), num_bits 4, **group_size 128**, asymmetric, observer mse, targets `[Linear]`
- Kept 16-bit (ignore): vision tower, `linear_attn.in_proj_{a,b}`, MoE/shared-expert gates, `lm_head`, norms, MTP head
- MTP head (`mtp.*`) carried verbatim in bf16 (`model-mtp.safetensors`) for speculative decoding
- Calibration: ultrachat + offensive-security spike with the model's own completions (256 samples @ 1024)

## Serve (1Cat-vLLM, 2×V100)
`--kv-cache-dtype fp8_e5m2`, MTP speculative decoding, `--gpu-memory-utilization` tuned to KV/context budget.

Uncensored research model — use responsibly and lawfully.
