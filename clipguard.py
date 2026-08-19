#!/usr/bin/env python3
"""
clipguard.py — zero-reference quantization-failure guard for Qwen3.8-27B.

The problem it solves (P5): a heavily quantized 27B is served with no way to
know whether its output on *your* data has silently degraded. Every prior
answer was either circular (train a "quantized vs FP16" classifier — which
needs the FP16 model you don't have) or a heuristic (self-consistency).

The mechanism here is causally tied to the error, and needs no reference model:

  In fixed-point (GGUF) inference the dominant error source is CLIPPING —
  activations outside a tensor's calibrated [min,max] grid get saturated to
  the boundary, and saturation error is precisely where math/code failures
  concentrate. The cheap, real-time signal for that is the CLIP RATE: the
  fraction of activations that land on the grid boundary, per layer.

The guard has three layers:

  (1) static — estimate clip risk from quant type alone (this module),
  (2) baseline — calibrate the clip-rate *once at load* by running a handful
      of clean prompts through the QUANTIZED model itself (no GPU/FP16),
  (3) runtime — during serving, flag any layer whose clip rate drifts
      structurally above its own baseline (input is out of the quantized
      model's safe operating range).

Honest scope: parts (2)/(3) need per-layer activation stats, which ggml does
not yet expose. This module ships (1) — the part that runs without a model —
plus the exact calibration protocol and the server-side hook point. It does
NOT pretend the runtime signal is implemented.
"""

import argparse
import json


# Quantization theory: clip rate scales with bits-per-weight and with how
# outlier-heavy a tensor's activations are. Layer roles differ in outlier
# density (attention output projections and FFN are the worst; norms/gates
# are benign). These are relative weights, not measured numbers — they encode
# "where clipping concentrates", which is what a risk score needs.
ROLE_SENSITIVITY = {
    # role            -> (sensitivity 0..1, why)
    "ffn_down":      (0.9, "FFN down-proj: largest tensor, most outliers"),
    "ffn_up":        (0.8, "FFN up-proj: second largest, outlier-heavy"),
    "ffn_gate":      (0.7, "FFN gate: silu input saturates"),
    "attn_out":      (0.7, "attention output proj concentrates outliers"),
    "attn_qkv":      (0.5, "QKV proj: moderate"),
    "delta_gate":    (0.4, "Gated-DeltaNet gate: bounded by norm"),
    "delta_v":       (0.3, "DeltaNet value proj: normalized recurrent"),
    "norm":          (0.1, "RMSNorm: already scale-free"),
    "embed":         (0.2, "token embedding"),
}

# bits-per-weight implied by each quant (approx; measured sizes / params).
# bpw = 8 * bytes / params  using the unsloth GGUF sizes (27.78B params).
BITS = {
    "Q8_0": 8.0, "Q6_K": 6.0, "Q5_K_M": 5.0, "Q5_K_S": 5.0,
    "Q4_K_M": 4.5, "Q4_0": 4.0, "Q4_K_S": 4.0, "IQ4_XS": 4.0,
    "Q3_K_XL": 3.5, "IQ3_S": 3.4, "IQ3_XXS": 3.0, "Q2_K_XL": 2.5,
    "IQ2_S": 2.3, "IQ2_XXS": 2.1, "IQ1_M": 1.8, "IQ1_S": 1.7,
}

# clip risk grows roughly exponentially as bits drop below ~4.
def clip_risk(bpw):
    if bpw >= 4.0:
        return 0.05
    return min(1.0, 0.05 * (4.0 / bpw) ** 2.0)


def static_risk(quant):
    bpw = BITS.get(quant)
    if bpw is None:
        raise ValueError(f"unknown quant: {quant}")
    risk = clip_risk(bpw)
    # per-role contribution, normalized
    roles = {}
    for role, (sens, why) in ROLE_SENSITIVITY.items():
        roles[role] = round(risk * sens, 3)
    worst = max(roles, key=roles.get)
    return {
        "quant": quant,
        "bpw": bpw,
        "clip_risk_floor": round(risk, 3),
        "per_role_risk": roles,
        "worst_role": worst,
        "verdict": _verdict(risk),
    }


def _verdict(risk):
    if risk < 0.2:
        return "safe — clipping negligible"
    if risk < 0.45:
        return "watch — monitor clip rate on math/code inputs"
    if risk < 0.7:
        return "risky — expect math/code degradation; guard strongly advised"
    return "dangerous — clipping dominates; model likely unreliable on hard tasks"


def calibrate_protocol():
    return (
        "CALIBRATION PROTOCOL (runs once, no reference model):\n"
        "  1. Load the QUANTIZED model; run 8-16 clean factual prompts\n"
        "     ('what is the capital of France', simple arithmetic).\n"
        "  2. For each layer, record clip_rate = #activations at grid\n"
        "     min/max / #activations. This is the per-layer BASELINE.\n"
        "  3. Baseline is the quantized model's *own* distribution —\n"
        "     no FP16, no GPU, no external reference.\n"
        "RUNTIME GUARD:\n"
        "  4. During serving, recompute clip_rate per layer per request.\n"
        "  5. Flag when a layer's clip_rate exceeds baseline by a fixed\n"
        "     multiple (e.g. 2x) — the input is out of the quantized\n"
        "     model's safe operating range. Route to fallback or warn.\n"
        "SERVER-SIDE HOOK (not yet in ggml):\n"
        "  - llama.cpp ggml_compute_forward: after dequant, count values\n"
        "    equal to quant min/max (already computed for the mul).\n"
        "  - Expose per-layer counters via llama_server /metrics.\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", default="IQ3_S")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--protocol", action="store_true")
    args = ap.parse_args()

    if args.protocol:
        print(calibrate_protocol())
        return

    r = static_risk(args.quant)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(f"quant {r['quant']}  bpw {r['bpw']}  clip-risk floor {r['clip_risk_floor']}")
    print(f"  worst role: {r['worst_role']} (risk {r['per_role_risk'][r['worst_role']]})")
    print(f"  verdict: {r['verdict']}")


if __name__ == "__main__":
    main()
