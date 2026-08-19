# Qwen3.8-27B on 16 GB RAM — correct-fit toolkit

Two small, dependency-free tools that compute the **correct** memory budget for
running [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (a dense
27.8 B hybrid model) on a 16 GB CPU-only machine — and detect the one failure
mode a naive recipe cannot.

## Why this exists

An older recipe ("Qwen3-30B-A3B in 16 GB") reasoned that the model fits because
it is a Mixture-of-Experts and only ~3.3 B active params are touched per token,
so `mmap` keeps resident RAM low. That trick does **not** transfer to
Qwen3.8-27B, which is **dense**: every one of its 27.78 B parameters is read on
every token.

But Qwen3.5/3.8 has a different memory superpower the old recipe misses: it is
a **hybrid** architecture. Of its 64 layers, only 16 are full (GQA) attention —
the other 48 are Gated DeltaNet linear-attention layers whose recurrent state is
constant in memory (no per-token KV cache). So the KV cache is ~4× smaller than
a classic dense transformer, and the two scripts here encode that fact instead
of guessing.

## Files

- `qwen_fit.py` — computes the real budget (weights + KV + recurrent state +
  runtime), auto-selects the deepest quant that fits, and emits working
  `llama-server` flags.
- `clipguard.py` — a zero-reference quantization-failure guard. Estimates the
  clip-risk floor of a quant, plus a calibration protocol + server hook for a
  runtime clip-rate drift detector.

## Usage

```bash
# pick the best-fitting quant for 16 GB / 8192 context
python3 qwen_fit.py --ram 16 --ctx 8192

# machine-readable
python3 qwen_fit.py --ram 16 --ctx 8192 --json

# quantization clip-risk for a specific quant
python3 clipguard.py --quant IQ3_XXS

# calibration + runtime-guard protocol
python3 clipguard.py --protocol
```

## Verified facts (not estimates)

| item | value | source |
|------|-------|--------|
| model params (BF16) | 27,781,427,952 | HF safetensors index |
| layers / attention | 64 layers; 16 full-attn, 48 Gated DeltaNet | `config.json` (`full_attention_interval: 4`) |
| KV cache | 64 KiB/token (0.54 GB @ 8k ctx) | 16 layers × 4 KV heads × 256 dim × 2 B |
| GGUF sizes | IQ3_S 12.04 GB, IQ3_XXS 10.93 GB, Q3_K_XL 13.15 GB, Q4_K_M 16.46 GB | `unsloth/Qwen3.8-27B-GGUF` |

## Scope / honest limits

- `qwen_fit.py` is fully functional.
- `clipguard.py` ships the **static** risk layer (runs with no model). The
  **runtime** clip-rate detector needs per-layer activation counters that ggml
  does not yet expose; the calibration protocol and the `ggml_compute_forward`
  hook point are documented, not implemented.

## License

MIT
