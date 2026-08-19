# Verification plan — clip-guard (P5)

## 0. The claim, stated precisely

> **H1** — In inference where **activations** are quantized to a calibrated
> grid (KV-cache `q8_0`/`q4_0`, or W8A8 engines), the fraction of activations
> clipped to the grid boundary (the **clip rate**) is a strong per-input
> predictor of output correctness, measurable at runtime **without a reference
> model**.

## 0.1 What the audit already ruled out

The original framing — "clip rate detects weight-quantization quality" — is
**wrong for the deployment target**. llama.cpp GGUF is weight-only: the
forward pass runs in FP16/FP32, so there is no activation grid and no clip
rate on the weight path. The signal exists only where activations are
quantized. Claim relocated accordingly (above).

## 1. Verification ≠ 100% certainty

The goal is not "100%". It is: *confidence at a pre-registered threshold, on a
held-out set, with the metric fixed before running.* That is the strongest
honest claim available.

**Pre-registered pass criterion:** AUC(clip-rate → wrong-answer) ≥ **0.75**
across the bit sweep, on the held-out split. Anything else is p-hacking.

## 2. The reference-model catch

To verify "clip rate predicts error" you must measure error against ground
truth — which requires the FP16/reference model. Verification needs the
reference; deployment doesn't. Consequences:

- Phase 1 must run on a model small enough that FP16 fits (0.5–3B).
- Phase 2 needs a box that can hold FP8/Q8 alongside the 3-bit target.

## 3. Phases

| Phase | Model | Quantized path | Hardware | Decisive question |
|-------|-------|----------------|----------|-------------------|
| 1 | Qwen2.5-0.5B | activations (W8A8 proxy) | any box, few GB free | does the signal exist at all? |
| 2 | Qwen3.8-27B | KV-cache q8_0/q4_0 | 32 GB (KVM 8) | does it hold at 27B-class? |
| 3 | — | — | — | certify (publish) or kill |

## 4. Phase 1 — exact protocol

1. Load small model in FP32 (reference).
2. Split eval set (≥100 arithmetic/knowledge items with exact-match answers):
   20% calibration, 80% held-out test.
3. For each bit width in {16, 8, 6, 5, 4, 3, 2}:
   - calibrate per-layer activation min/max on the calibration split,
   - on the test split, quantize activations to that grid,
   - record per-example **clip rate** (per layer) and **correctness**.
4. Compute **AUC**(clip-rate → wrong-answer) over all non-reference examples.
5. Pass if AUC ≥ 0.75.

`verify_clipguard.py` implements the harness (currently reports accuracy-per-
bit; the AUC computation is the decisive step and must be added before the run
is trusted).

## 5. Kill criteria

- AUC < 0.75 in Phase 1 → the signal is too weak; the guard has no value;
  **do not deploy, do not spend on Phase 2.**
- AUC ≥ 0.75 but the signal saturates (only detects catastrophic 2-bit
  collapse, not 3-bit edge cases) → value is narrower than claimed; record it
  honestly.
- Any calibration that peeks at the test split → results void, re-run.

## 6. Hardware-free verification result (DONE — NEGATIVE)

`verify_numerical.py` tests H1's numerical core without a model, in pure
Python. Result:

| test | Spearman(clip_rate, error) | meaning |
|------|---------------------------|---------|
| across-population (varying tail) | +0.99 | confounded — tail drives both |
| **fixed-population, per-input** (heavy tail) | **+0.02 … +0.22** | **the real guard signal — weak** |
| fixed-population, per-input (light tail) | +0.39 | counterfactual is *stronger* |

**Conclusion: H1-core fails.** When the population is held fixed (the actual
runtime situation), whether a specific input draws an outlier is essentially
noise — it does NOT predict that input's quantization error at the magnitude a
detector needs. Worse, the signal is *stronger* for light-tailed data, the
opposite of what the mechanism predicts. The claim does not survive its own
mechanism-level test.

What this means:

- The clip-rate guard **does not work as specified** — it cannot distinguish a
  "will fail" input from a "will pass" input at runtime.
- The across-population 0.99 correlation that originally motivated the idea was
  an artifact of varying the tail; it never transferred to per-input use.
- **Verdict: kill.** Do not deploy. Do not spend on Phase 2 (the model run).
  Phase 1 would only re-confirm this on real weights.

This is the value of verification before deployment: the idea cost one evening
of a hardware-free simulation instead of a $26/mo VPS and a model download.

## 7. What still does not exist (blockers, in order)

1. **Per-layer activation counters** — not in ggml. Required for runtime use.
2. **The AUC step** in `verify_clipguard.py` — written as a stub.
3. **Phase-1 execution** on a machine with free RAM.

These are the three concrete things between "idea" and "verified". None are
hardware-magic; all are ordinary work.
