#!/usr/bin/env python3
"""Turn the v2 (clean-protocol) llama.cpp/GGUF NVML archive into the two published CSVs.

Input:  the unpacked results_bundle (v2/res_*.json, meta/*).
Output: rtx4090_llamacpp_gguf_v2_2026-09-03.csv          (one row per process run)
        rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv  (decode-only, by differencing)

    python3 build/make_llamacpp_v2_csv.py /path/to/results_bundle data/

What changed against the 2026-08-31 session, and why each change matters:

* ``--ignore-eos``: every run decodes its full target, so the token count in the
  denominator is the requested one. In v1 the F16 arm implied 101% of the card's memory
  bandwidth, i.e. it had stopped early on EOS and its mJ/token was inflated.
* forced cooldown to near idle before every run, and randomized execution order, so no
  run inherits the thermal tail of the one before it. v1 had a fixed order, no cooldown,
  and its two genuinely cold runs were also its two largest outliers.
* the NVML hardware energy counter (``nvmlDeviceGetTotalEnergyConsumption``) is the
  reported quantity; the 100 Hz power trace is integrated as a cross-check only. The two
  disagree by ~15%, which is published here rather than hidden.
* n=5 per cell and a third token length (320), so the differencing assumption can be
  tested by fitting E = a + b*n rather than asserted.
"""
import csv
import glob
import json
import os
import statistics as st
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
DST = sys.argv[2] if len(sys.argv) > 2 else "."
SESSION = "2026-09-03 RTX 4090 llama.cpp GGUF F16 vs Q4_0, clean protocol (n=5 per cell)"
LENGTHS = (64, 320, 576)
REPS = (1, 2, 3, 4, 5)

ARMS = {
    "f16":      ("F16",  "llama.cpp GGUF F16",  "Llama-3.1-8B-Instruct-F16.gguf"),
    "q4ours":   ("Q4_0", "llama.cpp GGUF Q4_0", "ours-Q4_0.gguf"),
    "q4mlperf": ("Q4_0", "llama.cpp GGUF Q4_0", "Llama-3.1-8B-Instruct-Q4_0.gguf"),
}
# Perplexity was not re-measured in this session; the quality column is carried over from
# the 2026-08-31 run of the same three files (identical sha256, identical llama.cpp build).
PPL = {"f16": (7.3260, 0.04677), "q4ours": (7.7364, 0.04963), "q4mlperf": (7.7366, 0.04963)}
LLAMA_CPP_COMMIT = "b10643-192067b72"

runs = {}
for path in sorted(glob.glob(os.path.join(SRC, "v2", "res_*.json"))):
    arm, n, r = os.path.basename(path)[4:-5].rsplit("_", 2)
    runs[(arm, int(n[1:]), int(r[1:]))] = json.load(open(path))
missing = [k for k in ((a, n, r) for a in ARMS for n in LENGTHS for r in REPS) if k not in runs]
if missing:
    raise SystemExit(f"missing runs: {missing}")

raw_fields = [
    "run_id", "replicate", "gpu", "gpu_arch", "gpu_power_limit_w", "model", "params_b",
    "precision", "quantization_backend", "gguf_file", "batch_size", "context_length",
    "requested_tokens", "decoded_tokens", "ignore_eos", "sampling_rate_hz", "wall_s",
    "process_energy_j", "process_energy_trapezoid_j", "mean_power_w", "cooldown_s",
    "cooldown_converged", "start_power_w", "start_temp_c", "peak_temp_c", "max_power_w",
    "n_power_samples", "returncode", "llama_cpp_commit", "flash_attn", "temp", "seed",
    "basis", "measurement_source", "n_trials", "session",
]
with open(os.path.join(DST, "rtx4090_llamacpp_gguf_v2_2026-09-03.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, raw_fields)
    w.writeheader()
    for (arm, n, r) in sorted(runs, key=lambda k: (k[0], k[1], k[2])):
        d = runs[(arm, n, r)]
        prec, backend, gguf = ARMS[arm]
        trace = d["power_trace_w"]
        w.writerow({
            "run_id": f"rtx4090-llamacpp-v2-{arm}-n{n}-r{r}",
            "replicate": r,
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_arch": "ada",
            "gpu_power_limit_w": 450.0,
            "model": "Llama-3.1-8B-Instruct",
            "params_b": 8.0,
            "precision": prec,
            "quantization_backend": backend,
            "gguf_file": gguf,
            "batch_size": 1,
            "context_length": 2048,
            "requested_tokens": n,
            "decoded_tokens": d["eval_tokens"],
            "ignore_eos": True,
            "sampling_rate_hz": d["sample_rate_hz"],
            "wall_s": round(d["wall_s"], 3),
            "process_energy_j": round(d["energy_counter_j"], 3),
            "process_energy_trapezoid_j": round(d["energy_trapezoid_j"], 3),
            "mean_power_w": round(d["mean_power_w"], 2),
            "cooldown_s": round(d["cooldown_s"], 1),
            "cooldown_converged": d["cooldown_converged"],
            "start_power_w": round(d["start_power_w"], 2),
            "start_temp_c": d["start_temp_c"],
            "peak_temp_c": d["peak_temp_c"],
            "max_power_w": round(max(trace), 2),
            "n_power_samples": len(trace),
            "returncode": d["returncode"],
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "flash_attn": True,
            "temp": 0,
            "seed": 1234,
            "basis": "measured",
            "measurement_source": "nvml-energy-counter",
            "n_trials": len(REPS),
            "session": SESSION,
        })


def cell(arm, n, key="energy_counter_j"):
    return [runs[(arm, n, r)][key] for r in REPS]


def decode(arm, key="energy_counter_j"):
    """Decode-only energy of 512 tokens, E(576) - E(64), with the SD of that difference."""
    hi, lo = cell(arm, 576, key), cell(arm, 64, key)
    dE = st.mean(hi) - st.mean(lo)
    sd = (st.stdev(hi) ** 2 / len(hi) + st.stdev(lo) ** 2 / len(lo)) ** 0.5
    dW = st.mean(cell(arm, 576, "wall_s")) - st.mean(cell(arm, 64, "wall_s"))
    return dE, dW, sd


def fit(arm):
    """OLS of E = a + b*n over all 15 runs of an arm. b is mJ/token free of any load term."""
    xs = [n for n in LENGTHS for _ in REPS]
    ys = [y for n in LENGTHS for y in cell(arm, n)]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss_t = sum((y - my) ** 2 for y in ys)
    ss_r = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - ss_r / ss_t


sum_fields = [
    "params_b", "model", "precision", "quantization_backend", "gguf_file", "n_replicates",
    "decode_tokens", "decode_energy_j", "decode_energy_j_sd", "energy_mj_per_token",
    "tokens_per_joule", "energy_j_per_1k_tok", "throughput_tok_s", "decode_avg_power_w",
    "vs_fp16_energy_pct", "energy_mj_per_token_ols", "ols_intercept_j", "ols_r2",
    "energy_mj_per_token_trapezoid", "counter_vs_trapezoid_pct",
    "perplexity", "perplexity_stderr", "fp16_perplexity", "delta_perplexity_pct",
    "perplexity_corpus", "perplexity_chunks", "perplexity_ctx", "perplexity_session",
    "basis", "measurement_source", "session",
]
base = decode("f16")[0]
f16_ppl = PPL["f16"][0]
with open(os.path.join(DST, "rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, sum_fields)
    w.writeheader()
    for arm in ("f16", "q4ours", "q4mlperf"):
        prec, backend, gguf = ARMS[arm]
        dE, dW, sd = decode(arm)
        dE_trap = decode(arm, "energy_trapezoid_j")[0]
        a, b, r2 = fit(arm)
        ppl, err = PPL[arm]
        w.writerow({
            "params_b": 8.0,
            "model": "Llama-3.1-8B-Instruct",
            "precision": prec,
            "quantization_backend": backend,
            "gguf_file": gguf,
            "n_replicates": len(REPS),
            "decode_tokens": 512,
            "decode_energy_j": round(dE, 3),
            "decode_energy_j_sd": round(sd, 3),
            "energy_mj_per_token": round(dE / 512 * 1000, 1),
            "tokens_per_joule": round(512 / dE, 4),
            "energy_j_per_1k_tok": round(dE / 512 * 1000, 1),
            "throughput_tok_s": round(512 / dW, 1),
            "decode_avg_power_w": round(dE / dW, 1),
            "vs_fp16_energy_pct": round((dE / base - 1) * 100, 2),
            "energy_mj_per_token_ols": round(b * 1000, 1),
            "ols_intercept_j": round(a, 1),
            "ols_r2": round(r2, 5),
            "energy_mj_per_token_trapezoid": round(dE_trap / 512 * 1000, 1),
            "counter_vs_trapezoid_pct": round((dE / dE_trap - 1) * 100, 2),
            "perplexity": ppl,
            "perplexity_stderr": err,
            "fp16_perplexity": f16_ppl,
            "delta_perplexity_pct": round((ppl / f16_ppl - 1) * 100, 3),
            "perplexity_corpus": "wikitext-2-raw-v1 test",
            "perplexity_chunks": 564,
            "perplexity_ctx": 512,
            "perplexity_session": "2026-08-31 (same GGUF files and llama.cpp build; not re-measured)",
            "basis": "measured",
            "measurement_source": "nvml-energy-counter",
            "session": SESSION,
        })
print("wrote both v2 CSVs to", DST)
