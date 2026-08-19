#!/usr/bin/env python3
"""
verify_numerical.py — hardware-free verification of the clip-rate mechanism.

Tests the numerical core of H1 WITHOUT a model:

    H1-core: for the heavy-tailed activation distributions that real LLMs
    exhibit (LLM.int8 NeurIPS'22, SmoothQuant ICML'23), the fraction of
    activations clipped to a calibrated grid (CLIP RATE) is a strong
    per-input predictor of the quantization error that clipping introduces.

Method (pure Python, deterministic):
  1. Model one "activation vector" as log-normal magnitude * random sign —
     a heavy-tailed distribution, matching the outlier structure of real
     transformer activations.
  2. Calibrate a robust [q_min, q_max] grid on a calibration draw
     (1st/99th percentile, as calibration libraries do).
  3. For each of many "inputs", vary the tail heaviness (sigma) to simulate
     inputs with different outlier intensity.
  4. For each input: quantize to the grid (round + clip) and record
     (a) clip rate = fraction outside [q_min, q_max],
     (b) normalized MSE between original and quantized values.
  5. Report correlation(clip_rate, error) — the mechanism verdict.

Counterfactual: repeat with LIGHT-tailed (Gaussian) activations. If clip rate
predicts error only under heavy tails, the signal is real and specific, not a
mathematical tautology. If it predicts error in both, it is trivial and the
guard adds nothing.

This verifies the *mechanism*. It does NOT verify end-to-end output
correctness (that needs the model + Phase 1 harness). It decides whether the
Phase-1 run is worth doing at all.
"""

import math
import random
import statistics


def draw_heavy_tailed(n, sigma):
    # log-normal magnitude, random sign => occasional large outliers
    out = []
    for _ in range(n):
        z = random.gauss(0.0, sigma)
        mag = math.exp(z)
        out.append(mag if random.random() < 0.5 else -mag)
    return out


def draw_light_tailed(n, sigma):
    return [random.gauss(0.0, sigma) for _ in range(n)]


def percentile_sorted(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def quantize(xs, q_min, q_max, bits):
    steps = (1 << bits) - 1
    scale = (q_max - q_min) / steps
    n = len(xs)
    clipped = 0
    err_sum = 0.0
    var_sum = 0.0
    for x in xs:
        if x < q_min or x > q_max:
            clipped += 1
        xq = min(max(round((x - q_min) / scale), 0), steps) * scale + q_min
        err_sum += (x - xq) ** 2
        var_sum += x * x
    clip_rate = clipped / n
    nmse = err_sum / var_sum if var_sum > 0 else 0.0
    return clip_rate, nmse


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0] * len(xs)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    ra, rb = rank(a), rank(b)
    n = len(a)
    mean_ra = (n - 1) / 2.0
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_ra) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_ra) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_ra) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def pearson(a, b):
    n = len(a)
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def run(dist_name, draw_fn, n_inputs, n_dims, bits, sigmas):
    # calibration draw (moderate tail) -> grid
    random.seed(1)
    calib = draw_fn(n_dims, sigmas[0])
    s = sorted(calib)
    q_min = percentile_sorted(s, 0.01)
    q_max = percentile_sorted(s, 0.99)

    clip_rates = []
    errors = []
    for sigma in sigmas:
        for _ in range(n_inputs // len(sigmas)):
            xs = draw_fn(n_dims, sigma)
            cr, err = quantize(xs, q_min, q_max, bits)
            clip_rates.append(cr)
            errors.append(err)
    rho = spearman(clip_rates, errors)
    r = pearson(clip_rates, errors)
    return rho, r, q_min, q_max


def run_fixed_sigma(draw_fn, sigma, n_inputs, n_dims, bits):
    """The DECISIVE test: fix the population (sigma), and ask whether
    per-input variation in clip rate (driven only by 'did this specific
    input draw an outlier') predicts per-input error. This is what a real
    guard sees: same user, same style, but some prompts break."""
    random.seed(7)
    calib = draw_fn(n_dims, sigma)
    s = sorted(calib)
    q_min = percentile_sorted(s, 0.01)
    q_max = percentile_sorted(s, 0.99)

    clip_rates = []
    errors = []
    for _ in range(n_inputs):
        xs = draw_fn(n_dims, sigma)
        cr, err = quantize(xs, q_min, q_max, bits)
        clip_rates.append(cr)
        errors.append(err)
    return spearman(clip_rates, errors), pearson(clip_rates, errors)


def main():
    n_inputs = 500
    n_dims = 4096
    bits = 8
    sigmas = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5]

    print("H1-core numerical verification (hardware-free, deterministic)\n")

    # heavy-tailed (real LLM activations)
    rho_h, r_h, qmin_h, qmax_h = run(
        "heavy", draw_heavy_tailed, n_inputs, n_dims, bits, sigmas)
    print(f"[heavy-tailed]  grid [{qmin_h:.3f}, {qmax_h:.3f}] "
          f"(1st/99th pct), {bits}-bit")
    print(f"  Spearman(clip_rate, NMSE) = {rho_h:+.3f}")
    print(f"  Pearson (clip_rate, NMSE) = {r_h:+.3f}")

    # light-tailed counterfactual
    rho_l, r_l, qmin_l, qmax_l = run(
        "light", draw_light_tailed, n_inputs, n_dims, bits, sigmas)
    print(f"\n[light-tailed]  grid [{qmin_l:.3f}, {qmax_l:.3f}], {bits}-bit")
    print(f"  Spearman(clip_rate, NMSE) = {rho_l:+.3f}")
    print(f"  Pearson (clip_rate, NMSE) = {r_l:+.3f}")

    # ---- the decisive, per-input (fixed-population) test ----
    print("\n--- decisive test: FIXED population, per-input variation ---")
    print(f"({n_inputs} inputs, {n_dims} dims, {bits}-bit; variation is only "
          "'did this input draw an outlier')\n")
    for sig in [0.7, 1.3, 2.0]:
        rho, r = run_fixed_sigma(draw_heavy_tailed, sig, n_inputs, n_dims, bits)
        print(f"  heavy sigma={sig}: Spearman {rho:+.3f}  Pearson {r:+.3f}")
    rho_lf, r_lf = run_fixed_sigma(draw_light_tailed, 1.0, n_inputs, n_dims, bits)
    print(f"  light sigma=1.0: Spearman {rho_lf:+.3f}  Pearson {r_lf:+.3f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("  Across-population correlation (top) is confounded: tail heaviness")
    print("  drives both clip rate and error, so it is a tautology.")
    print("  The fixed-population numbers (bottom) are what a real guard sees:")
    print("  per-input clip rate must predict per-input error with high")
    print("  correlation, and must do so under heavy tails but NOT light tails.")


if __name__ == "__main__":
    main()
