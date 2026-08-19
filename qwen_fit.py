#!/usr/bin/env python3
"""
qwen_fit.py — correct memory-budget + launch-flag generator for Qwen3.8-27B
on a 16 GB (CPU-only) machine.

Why this exists: the old "Qwen3-30B-A3B in 16 GB" manual is wrong for the
newest 27B model for one dominant reason — the architecture changed. The
manual's core trick ("only 3.3B active MoE params are touched per token, so
mmap keeps resident RAM low") applies to Qwen3-30B-A3B, a Mixture-of-Experts.
Qwen3.8-27B is DENSE: every one of its 27.78B parameters is read on every
token. So the MoE resident-set argument collapses.

But the newest model has a *different* memory superpower the manual misses:
Qwen3.5/3.8 is a HYBRID. Of its 64 layers only 16 are full (GQA) attention —
the other 48 are Gated DeltaNet linear-attention layers whose recurrent state
is constant in memory (it does not grow a per-token KV cache). So the KV
cache is ~4x smaller than a classic transformer of the same size, and the
depth of the context is nearly free on the RAM side.

This script computes the one correct number (weights + KV + runtime) and
then auto-selects the deepest quant that actually fits, plus the launch flags.
"""

import argparse
import json
import sys

# ---------------------------------------------------------------- architecture
# Qwen/Qwen3.8-27B  (config.json -> text_config), qwen3_5 hybrid
MODEL = {
    "name": "Qwen3.8-27B",
    "params": 27_781_427_952,          # BF16 params (from safetensors index)
    "n_layers": 64,
    "full_attn_interval": 4,           # every 4th layer is full attention
    "n_kv_heads": 4,                   # GQA: 24 Q / 4 KV
    "head_dim": 256,
    "n_linear_value_heads": 48,        # Gated DeltaNet value heads (recurrent)
    "linear_value_head_dim": 128,
    "linear_state_bytes_per_head": 4,  # float32 recurrent state
}

# Real GGUF file sizes (GB, 1e9) from unsloth/Qwen3.8-27B-GGUF. Only these exist.
QUANTS = [
    ("IQ1_S",   6.19),
    ("IQ1_M",   6.73),
    ("IQ2_XXS", 7.27),
    ("IQ2_S",   8.37),
    ("Q2_K_XL", 9.83),
    ("IQ3_XXS", 10.93),
    ("IQ3_S",   12.04),
    ("Q3_K_XL", 13.15),
    ("IQ4_XS",  14.25),
    ("Q4_K_S",  15.36),
    ("Q4_0",    16.06),
    ("Q4_K_M",  16.46),
    ("Q5_K_S",  18.67),
    ("Q5_K_M",  19.77),
    ("Q6_K",    21.98),
    ("Q8_0",    29.05),
]

# expected quality/bytes tradeoff — lower is "deeper" (smaller, lossier).
QUALITY = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_0", "Q4_K_S",
    "IQ4_XS", "Q3_K_XL", "IQ3_S", "IQ3_XXS", "Q2_K_XL", "IQ2_S",
    "IQ2_XXS", "IQ1_M", "IQ1_S",
]


def full_attention_layers(m):
    return m["n_layers"] // m["full_attn_interval"]


def kv_bytes_per_token(m):
    """Only the 16 full-attention layers carry a KV cache. The 48 linear
    (Gated DeltaNet) layers do NOT — they keep a fixed recurrent state."""
    layers = full_attention_layers(m)
    return layers * 2 * m["n_kv_heads"] * m["head_dim"] * 2  # K+V as fp16


def linear_state_bytes(m):
    n_lin = m["n_layers"] - full_attention_layers(m)
    val = n_lin * m["n_linear_value_heads"] * m["linear_value_head_dim"] \
        * m["linear_state_bytes_per_head"]
    return val


def kv_cache_gb(m, ctx):
    return kv_bytes_per_token(m) * ctx / 1e9


def budget(m, weight_gb, ctx, runtime_gb=1.0):
    linear = linear_state_bytes(m) / 1e9
    kv = kv_cache_gb(m, ctx)
    return weight_gb, linear, kv, runtime_gb, weight_gb + linear + kv + runtime_gb


def fit(total_ram_gb, ctx, runtime_gb=1.0, reserve_gb=1.5):
    """Return sorted list of (quant, weight_gb, total_gb, headroom) that fit."""
    results = []
    for name, w in QUANTS:
        _, lin, kv, rt, total = budget(MODEL, w, ctx, runtime_gb)
        headroom = total_ram_gb - total
        if total <= total_ram_gb - reserve_gb:
            results.append((name, w, total, headroom))
    return results


def pick_best(available, ctx):
    """Choose the HIGHEST-quality quant that fits within available GB."""
    fits = [(q, w, t, h) for q, w, t, h in
            fit(available + 0.0, ctx) or []]
    if not fits:
        return None
    best = None
    for q in QUALITY:
        for row in fits:
            if row[0] == q:
                best = row
                break
        if best:
            return best
    return fits[0]


CMD = """llama-server \\
  --model {model} \\
  --host 0.0.0.0 --port 8080 \\
  --ctx-size {ctx} \\
  --threads {threads} \\
  --n-gpu-layers 0 \\
  --parallel 1 \\
  --mmap \\
  --no-mlock \\
  --cache-type-k q8_0 --cache-type-v q8_0 \\
  --no-webui 2>/dev/null || llama-server -m {model} -c {ctx} -t {threads} --mmap"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ram", type=float, default=16.0, help="physical RAM GB")
    ap.add_argument("--ctx", type=int, default=8192, help="context tokens")
    ap.add_argument("--runtime", type=float, default=1.0)
    ap.add_argument("--reserve", type=float, default=1.5,
                    help="GB reserved for OS/headroom")
    ap.add_argument("--model", default="Qwen3.8-27B-UD-{q}.gguf")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    threads = args.threads or __import__("os").cpu_count() or 4

    fits = fit(args.ram, args.ctx, args.runtime, args.reserve)
    best = pick_best(args.ram, args.ctx)

    if args.json:
        print(json.dumps({
            "model": MODEL["name"],
            "ram_gb": args.ram,
            "context": args.ctx,
            "full_attention_layers": full_attention_layers(MODEL),
            "linear_attention_layers": MODEL["n_layers"] - full_attention_layers(MODEL),
            "kv_bytes_per_token": kv_bytes_per_token(MODEL),
            "linear_state_gb": round(linear_state_bytes(MODEL) / 1e9, 4),
            "kv_cache_gb_at_ctx": round(kv_cache_gb(MODEL, args.ctx), 3),
            "fits": [{"quant": q, "weights_gb": w, "total_gb": round(t, 2),
                      "headroom_gb": round(h, 2)} for q, w, t, h in fits],
            "best": None if not best else {
                "quant": best[0],
                "weights_gb": best[1],
                "total_gb": round(best[2], 2),
                "headroom_gb": round(best[3], 2),
            },
        }, indent=2))
        return

    print(f"model: {MODEL['name']}  (dense {MODEL['params']/1e9:.1f}B, hybrid)")
    print(f"layers: {full_attention_layers(MODEL)} full-attn (KV cache) + "
          f"{MODEL['n_layers'] - full_attention_layers(MODEL)} linear (DeltaNet state)")
    print(f"KV cache: {kv_bytes_per_token(MODEL)} bytes/token "
          f"= {kv_cache_gb(MODEL, args.ctx):.2f} GB @ ctx {args.ctx}")
    print(f"linear recurrent state (constant): {linear_state_bytes(MODEL)/1e9:.3f} GB")
    print()
    print(f"fitting into {args.ram} GB RAM (reserving {args.reserve} GB):")
    if not fits:
        print("  NOTHING FITS at this context. Lower --ctx or use IQ1_S.")
    for q, w, t, h in fits:
        print(f"  {q:9s} weights {w:6.2f} GB  total {t:6.2f} GB  headroom {h:5.2f} GB")
    print()
    if best:
        q = best[0]
        model = args.model.format(q=q) if "{q}" in args.model else args.model
        print(f"BEST quant: {q}  (total {best[2]:.2f} GB, headroom {best[3]:.2f} GB)")
        print()
        print("launch:")
        print(CMD.format(model=model, ctx=args.ctx, threads=threads, q=q))
    else:
        print("No quant fits — reduce --ctx below "
              f"{kv_bytes_per_token(MODEL)} bytes/token budget or add swap.")


if __name__ == "__main__":
    main()