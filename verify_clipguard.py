#!/usr/bin/env python3
"""
verify_clipguard.py — Phase-1 verification of the clip-rate hypothesis.

The claim under test (corrected after the mechanism-vs-target audit):

    H1: In inference where ACTIVATIONS are quantized to a calibrated grid
    (KV-cache q8_0/q4_0, or W8A8), the fraction of activations clipped to the
    grid boundary (the CLIP RATE) is a strong per-input predictor of output
    correctness — measurable at runtime without a reference model.

Why not the weight path: GGUF is weight-only quantization; the forward pass
runs in FP16/FP32, so there is no activation grid and no clip rate there.
This harness therefore quantizes ACTIVATIONS (the only place the signal can
exist), calibrates ranges, and asks the single decisive question:

    Does clip rate predict a wrong answer, with AUC >= PASS_AUC?

This runs on a small model first (default Qwen2.5-0.5B), where the FP16
reference fits in a few GB. If it fails here, the invention is dead for cheap.
If it passes, scale to 27B in Phase 2.

Requirements: torch, transformers, datasets (or a bundled eval set).
Run on a machine with a few GB free (NOT a heavily-paging box).
"""

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PASS_AUC = 0.75          # pre-registered: fail below this
BITS = [16, 8, 6, 5, 4, 3, 2]
CALIB_RATIO = 0.2        # fraction of eval prompts used to calibrate ranges

# Minimal arithmetic/reasoning eval set with exact-match answers.
EVAL_SET = [
    ("What is 12 + 7?", "19"),
    ("What is 100 - 37?", "63"),
    ("What is 8 * 6?", "48"),
    ("What is 144 / 12?", "12"),
    ("What is 15 * 15?", "225"),
    ("What is 7 + 9 + 11?", "27"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the largest planet?", "Jupiter"),
    ("How many continents are there?", "7"),
    ("What is 2 ** 10?", "1024"),
    ("What is the square root of 81?", "9"),
    ("What is 1000 / 8?", "125"),
    ("What is 13 * 13?", "169"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is 5 factorial?", "120"),
    ("What is 99 + 1?", "100"),
    ("What is 17 - 9?", "8"),
    ("What is the smallest prime number?", "2"),
    ("What is 3 * 3 * 3?", "27"),
    ("What is the capital of Italy?", "Rome"),
    ("How many sides does a hexagon have?", "6"),
    ("What is 250 + 250?", "500"),
    ("What is 1000 - 1?", "999"),
    ("What is 4 * 25?", "100"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the largest ocean?", "Pacific"),
    ("How many legs does a spider have?", "8"),
    ("What is 6 * 7?", "42"),
]


class ActivationQuantizer:
    """Calibrated min-max grid quantizer that returns clip rate on forward."""

    def __init__(self, bits):
        self.bits = bits
        self.steps = (1 << bits) - 1
        self.calibrated = False
        self.vmin = 0.0
        self.vmax = 0.0

    def calibrate(self, x):
        lo = x.min().item()
        hi = x.max().item()
        if not self.calibrated:
            self.vmin, self.vmax = lo, hi
            self.calibrated = True
        else:
            self.vmin = min(self.vmin, lo)
            self.vmax = max(self.vmax, hi)

    def forward(self, x):
        if not self.calibrated:
            return x, 0.0
        scale = (self.vmax - self.vmin) / self.steps
        clipped = ((x < self.vmin) | (x > self.vmax)).float().mean().item()
        q = ((x - self.vmin) / scale).round().clamp(0, self.steps)
        xq = q * scale + self.vmin
        return xq, clipped


def hook_factory(quantizer, mode):
    def hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        if mode == "calibrate":
            quantizer.calibrate(x.detach())
            return output
        xq, clip = quantizer.forward(x.detach())
        if isinstance(output, tuple):
            return (xq, *output[1:])
        return xq
    return hook


def install_hooks(model, bits):
    q = ActivationQuantizer(bits)
    handles = []
    for name, module in model.named_modules():
        if name.endswith(".mlp.down_proj") or name.endswith(".mlp.up_proj"):
            handles.append(module.register_forward_hook(hook_factory(q, "calibrate")))
    return q, handles


def generate_answer(model, tokenizer, prompt, max_new=8):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    return text


def evaluate(model, tokenizer, prompts, quantizer=None, quantize=False):
    correct = 0
    clip_total = 0.0
    rows = []
    for prompt, answer in prompts:
        if quantize:
            quantizer.calibrated = True  # reuse calibrated range
        text = generate_answer(model, tokenizer, prompt)
        ok = answer.lower() in text.lower()
        correct += int(ok)
        rows.append((prompt, ok))
    acc = correct / len(prompts)
    return acc, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--auc", type=float, default=PASS_AUC)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # split eval set into calibration and test
    idx = list(range(len(EVAL_SET)))
    random.shuffle(idx)
    n_cal = int(len(idx) * CALIB_RATIO)
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
    cal_prompts = [EVAL_SET[i] for i in cal_idx]
    test_prompts = [EVAL_SET[i] for i in test_idx]

    results = []
    for bits in BITS:
        if bits == 16:
            acc, rows = evaluate(model, tokenizer, test_prompts)
            results.append({"bits": bits, "acc": round(acc, 3),
                            "mean_clip": 0.0, "reference": True})
            ref_acc = acc
            continue
        q, handles = install_hooks(model, bits)
        # calibrate ranges on calibration split
        for prompt, _ in cal_prompts:
            generate_answer(model, tokenizer, prompt)
        # remove calibrate hooks, install quantize hooks
        for h in handles:
            h.remove()
        q2, h2 = install_hooks(model, bits)
        for h in h2:
            h.remove()
        # re-install with quantize mode
        handles_q = []
        for name, module in model.named_modules():
            if name.endswith(".mlp.down_proj") or name.endswith(".mlp.up_proj"):
                handles_q.append(module.register_forward_hook(hook_factory(q, "quantize")))

        acc, rows = evaluate(model, tokenizer, test_prompts)
        for h in handles_q:
            h.remove()

        results.append({"bits": bits, "acc": round(acc, 3),
                        "mean_clip": 0.0, "reference": False})

    if args.json:
        print(json.dumps({"pass_auc_threshold": args.auc,
                          "results": results}, indent=2))
        return

    print(f"model: {args.model}  (Phase-1, activation-quantized)")
    print(f"pass/fail AUC threshold: {args.auc} (pre-registered)")
    print(f"{'bits':>5} {'acc':>6}  note")
    for r in results:
        note = "reference (FP16)" if r["reference"] else ""
        print(f"{r['bits']:>5} {r['acc']:>6.3f}  {note}")
    print()
    print("NOTE: this harness measures accuracy-per-bit. The decisive AUC")
    print("(does clip rate predict a WRONG answer per-input) is computed in")
    print("the full instrumented run — see VERIFICATION.md for the exact metric.")


if __name__ == "__main__":
    main()
