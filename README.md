---
base_model: philbert440/Qwen3.6-27B-Uncensored-Aggressive
license: apache-2.0
tags:
- compressed-tensors
- w4a16
- awq
- vllm
- speculative-decoding
- mtp
---

# heretic-aggressive-w4a16-g128 — W4A16 AWQ (g128) + packed MTP head

Uncensored Qwen3.6-27B (aggressive abliteration), W4A16-AWQ quantized for Volta/SM70 serving
on 1Cat-vLLM, with a **quantized (packed) MTP head** for working speculative decoding on V100.

## Quant
- Format: `compressed-tensors`, pack-quantized
- Method: AWQ (`AWQModifier`), num_bits 4, **group_size 128**, asymmetric, observer mse, targets `[Linear]`
- Kept 16-bit (`ignore`): vision tower, `linear_attn.in_proj_{a,b}`, MoE/shared-expert gates, `lm_head`, all norms, **`mtp.fc`**
- Calibration: ultrachat + offensive-security spike with the model's own completions (256 samples @ 1024)

## MTP head (speculative decoding) — packed W4A16
The MTP head (`mtp.*`, in `model-mtp.safetensors`) is **quantized to W4A16** to match the body, matching
the base Qwen3.6 packed-head layout (36 tensors): each `mtp.layers.0.*_proj` carries
`weight_packed`/`weight_scale`/`weight_shape`/`weight_zero_point`; **`mtp.fc` and the norms are kept fp16**.

> **Gotcha (the fix that makes MTP actually work on SM70):** `mtp.fc` must be **unquantized** —
> list it in `quantization_config.ignore` and do **not** include it in any quant `config_group` target.
> If `mtp.fc` is marked for quantization while its weight is stored plain fp16, vLLM builds a quantized
> `fc` Linear, the plain `fc.weight` fails to load (`Parameter fc.weight not found in params_dict, skip
> loading`), the drafter's fusion projection stays uninitialized, and **MTP accept rate collapses to 0%**.
> With `mtp.fc` in `ignore`, accept is ~80%.
>
> An earlier revision of this repo shipped the MTP head as **bf16** (15 tensors), which crashes SM70
> spec-decode (mixed bf16-head / W4A16-body); this revision replaces it with the correct packed head.

## Serve (1Cat-vLLM 1.2.1, 2×V100, stock — no patches needed)
1Cat-vLLM **1.2.1** serves this on Volta with no local patches (SM70 rotary fallback + MM-enable are
upstream; compressed-tensors `validate_kv_cache_scheme(None)` no longer rejects fp8 KV on SM70).

```
python -m vllm.entrypoints.openai.api_server \
  --model philbert440/Qwen3.6-27B-Uncensored-Aggressive-W4A16-AWQ \
  --trust-remote-code --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.58 \
  --max-model-len 131072 --max-num-seqs 6 --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**Measured on 2×V100-PCIE-32GB (SM70), 1Cat-vLLM 1.2.1:**
- MTP K=2 accept rate **~80%**
- **~55–60 tok/s** single-stream decode (≈1.5× vs no-MTP)
- **fp16 KV** gives a ~213k-token KV pool at `util 0.58` while coexisting with a voice stack — so
  e5m2 KV is **optional** (use `--kv-cache-dtype fp8_e5m2` only if you need >~150k context).
- Env: `VLLM_SM70_QUANT_BACKEND=turbomind NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

`num_speculative_tokens=2` (K=2) is the sweet spot for stock Qwen MTP on V100; K=4 lowers accept.

Uncensored research model — use responsibly and lawfully.
