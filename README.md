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

Uncensored Qwen3.6-27B (aggressive/broad abliteration), W4A16-AWQ quantized for Volta/SM70 serving
on 1Cat-vLLM, with a **packed W4A16 MTP head** for working speculative decoding on V100.

## Quant
- Format: `compressed-tensors`, pack-quantized
- Method: AWQ (`AWQModifier`), num_bits 4, **group_size 128**, asymmetric, observer mse, targets `[Linear]`
- Kept 16-bit (`ignore`): vision tower, `linear_attn.in_proj_{a,b}`, MoE/shared-expert gates, `lm_head`, all norms, **`mtp.fc`**
- Calibration: ultrachat + offensive-security spike with the model's own completions (256 samples @ 1024)

## MTP head (speculative decoding) — packed W4A16
The MTP head (`mtp.*`, in `model-mtp.safetensors`) is **quantized to W4A16** to match the body, matching
the base Qwen3.6 packed-head layout (36 tensors): each `mtp.layers.0.*_proj` carries
`weight_packed`/`weight_scale`/`weight_shape`/`weight_zero_point`; **`mtp.fc` and the norms are kept fp16**.

> **Gotcha (the fix that makes MTP work on SM70):** `mtp.fc` must be **unquantized** — list it in
> `quantization_config.ignore` and do **not** include it in any quant `config_group` target. A quantized
> `mtp.fc` (while stored plain fp16) makes the loader skip `fc.weight`, leaves the drafter uninitialized,
> and **silently drops MTP accept to 0%**. With `mtp.fc` in `ignore`, accept is ~80–84%.
> An earlier revision shipped the MTP head as **bf16** (15 tensors), which crashes SM70 spec-decode; this
> revision replaces it with the correct packed head.

## Serving on V100 / SM70 — 1Cat-vLLM 1.2.1 (recommended)
Run on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) **1.2.1** + the **two SM70 patches** in
**[1Cat-vLLM PR #88](https://github.com/1CatAI/1Cat-vLLM/pull/88)** (P7: `fp8_e5m2` KV on W4A16; and the
fast fp8-KV prefill gather). The complete recipe is in that PR's `docs/v100-sm70-serving.md`.

```
VLLM_SM70_QUANT_BACKEND=turbomind NCCL_P2P_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model philbert440/Qwen3.6-27B-Uncensored-Aggressive-W4A16-AWQ \
  --trust-remote-code --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.58 \
  --max-model-len 262144 --max-num-seqs 6 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**Measured on 2× V100-PCIE-32GB (SM70), 1.2.1 + PR #88:** MTP K=2 accept **~80–84%**, **~57 tok/s**
single-stream decode (≈1.5× vs no-MTP), `fp8_e5m2` KV → **427k-token KV pool** at util 0.58 (full
**262144** context at 1.63× concurrency), coexisting with an ASR+TTS stack on the same 2 GPUs.
`num_speculative_tokens=2` is the V100 sweet spot; K=4 lowers accept. fp16 KV also works without either
patch (~192k usable context) if you don't need the full 262k.

Uncensored research model — use responsibly and lawfully.
