#!/usr/bin/env python3
"""Turn the raw llama.cpp/GGUF NVML archive into the two CSVs published beside the site.

Input:  the unpacked measurement archive (res_*.json, ppl_*.log, metadata.json).
Output: rtx4090_llamacpp_gguf_2026-08-31.csv          (one row per process run)
        rtx4090_llamacpp_gguf_2026-08-31.summary.csv  (decode-only, by differencing)

The raw energy of a run covers the whole `llama-cli` process, so it includes weight
loading -- 16 GB for F16 against 4.7 GB for Q4_0, which would flatter Q4_0. Decode-only
energy is therefore obtained by differencing the two token counts: E(576) - E(64) is the
energy of exactly 512 additional decoded tokens, with load and prefill cancelling out.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
DST = sys.argv[2] if len(sys.argv) > 2 else "."
SESSION = "2026-08-31 RTX 4090 llama.cpp GGUF F16 vs Q4_0 (n=3 per cell)"

ARMS = {
    "f16":      ("F16",  "llama.cpp GGUF F16",  "Llama-3.1-8B-Instruct-F16.gguf"),
    "q4ours":   ("Q4_0", "llama.cpp GGUF Q4_0", "ours-Q4_0.gguf"),
    "q4mlperf": ("Q4_0", "llama.cpp GGUF Q4_0", "Llama-3.1-8B-Instruct-Q4_0.gguf"),
}
# A run whose first power sample is near idle started from a cold GPU; every other run
# in the sweep inherited 145-248 W of thermal/power tail from the run before it, because
# the driving script has no cooldown and a fixed arm order. Flagged, not silently dropped.
COLD_START_W = 40.0

meta = json.load(open(os.path.join(SRC, "metadata.json")))
PPL = {}
for arm, log in (("f16", "ppl_f16.log"), ("q4ours", "ppl_q4ours.log"), ("q4mlperf", "ppl_q4mlperf.log")):
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", open(os.path.join(SRC, log)).read())
    PPL[arm] = (float(m.group(1)), float(m.group(2)))

runs = {}
for path in sorted(glob.glob(os.path.join(SRC, "res_*.json"))):
    arm, n, r = os.path.basename(path)[4:-5].rsplit("_", 2)
    runs[(arm, int(n[1:]), int(r[1:]))] = json.load(open(path))

raw_fields = [
    "run_id", "replicate", "gpu", "gpu_arch", "gpu_power_limit_w", "model", "params_b",
    "precision", "quantization_backend", "gguf_file", "batch_size", "context_length",
    "requested_tokens", "sampling_rate_hz", "wall_s", "process_energy_j", "mean_power_w",
    "start_power_w", "max_power_w", "n_power_samples", "cold_start", "returncode",
    "llama_cpp_commit", "flash_attn", "temp", "seed", "basis", "measurement_source",
    "n_trials", "session",
]
with open(os.path.join(DST, "rtx4090_llamacpp_gguf_2026-08-31.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, raw_fields)
    w.writeheader()
    for (arm, n, r) in sorted(runs, key=lambda k: (k[0], k[1], k[2])):
        d = runs[(arm, n, r)]
        prec, backend, gguf = ARMS[arm]
        trace = d["power_trace_w"]
        w.writerow({
            "run_id": f"rtx4090-llamacpp-{arm}-n{n}-r{r}",
            "replicate": r,
            "gpu": meta["gpu"],
            "gpu_arch": "ada",
            "gpu_power_limit_w": meta["gpu_power_limit_w"],
            "model": meta["model"],
            "params_b": 8.0,
            "precision": prec,
            "quantization_backend": backend,
            "gguf_file": gguf,
            "batch_size": 1,
            "context_length": meta["measurement_params"]["ctx_size"],
            "requested_tokens": n,
            "sampling_rate_hz": 10,
            "wall_s": round(d["wall_s"], 3),
            "process_energy_j": round(d["energy_j"], 3),
            "mean_power_w": round(d["mean_power_w"], 2),
            "start_power_w": round(trace[0], 2),
            "max_power_w": round(max(trace), 2),
            "n_power_samples": len(trace),
            "cold_start": "yes" if trace[0] < COLD_START_W else "no",
            "returncode": d["returncode"],
            "llama_cpp_commit": meta["llama_cpp_commit"],
            "flash_attn": meta["measurement_params"]["flash_attn"],
            "temp": meta["measurement_params"]["temp"],
            "seed": meta["measurement_params"]["seed"],
            "basis": "measured",
            "measurement_source": "direct-nvml",
            "n_trials": 3,
            "session": SESSION,
        })


def decode(arm, warm_only):
    lo = [runs[(arm, 64, r)] for r in (1, 2, 3)]
    if warm_only:
        lo = [d for d in lo if d["power_trace_w"][0] >= COLD_START_W]
    hi = [runs[(arm, 576, r)] for r in (1, 2, 3)]
    dE = st.mean(x["energy_j"] for x in hi) - st.mean(x["energy_j"] for x in lo)
    dW = st.mean(x["wall_s"] for x in hi) - st.mean(x["wall_s"] for x in lo)
    # SD of a difference of two independent means, each estimated from its own replicates
    sd = (st.stdev([x["energy_j"] for x in hi]) ** 2 / len(hi)
          + (st.stdev([x["energy_j"] for x in lo]) ** 2 / len(lo) if len(lo) > 1 else 0.0)) ** 0.5
    return dE, dW, sd, len(lo)


sum_fields = [
    "params_b", "model", "precision", "quantization_backend", "gguf_file", "n_replicates",
    "cold_start_runs_excluded", "decode_tokens", "decode_energy_j", "decode_energy_j_sd",
    "energy_mj_per_token", "tokens_per_joule", "energy_j_per_1k_tok", "throughput_tok_s",
    "decode_avg_power_w", "vs_fp16_energy_pct", "vs_fp16_energy_pct_all_runs",
    "perplexity", "perplexity_stderr", "fp16_perplexity", "delta_perplexity_pct",
    "perplexity_corpus", "perplexity_chunks", "perplexity_ctx", "basis", "session",
]
base_warm = decode("f16", True)[0]
base_all = decode("f16", False)[0]
f16_ppl = PPL["f16"][0]
with open(os.path.join(DST, "rtx4090_llamacpp_gguf_2026-08-31.summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, sum_fields)
    w.writeheader()
    for arm in ("f16", "q4ours", "q4mlperf"):
        prec, backend, gguf = ARMS[arm]
        dE, dW, sd, nlo = decode(arm, True)
        dE_all = decode(arm, False)[0]
        ppl, err = PPL[arm]
        w.writerow({
            "params_b": 8.0,
            "model": meta["model"],
            "precision": prec,
            "quantization_backend": backend,
            "gguf_file": gguf,
            "n_replicates": 3,
            "cold_start_runs_excluded": 3 - nlo,
            "decode_tokens": 512,
            "decode_energy_j": round(dE, 3),
            "decode_energy_j_sd": round(sd, 3),
            "energy_mj_per_token": round(dE / 512 * 1000, 1),
            "tokens_per_joule": round(512 / dE, 4),
            "energy_j_per_1k_tok": round(dE / 512 * 1000, 1),
            "throughput_tok_s": round(512 / dW, 1),
            "decode_avg_power_w": round(dE / dW, 1),
            "vs_fp16_energy_pct": round((dE / base_warm - 1) * 100, 2),
            "vs_fp16_energy_pct_all_runs": round((dE_all / base_all - 1) * 100, 2),
            "perplexity": ppl,
            "perplexity_stderr": err,
            "fp16_perplexity": f16_ppl,
            "delta_perplexity_pct": round((ppl / f16_ppl - 1) * 100, 3),
            "perplexity_corpus": "wikitext-2-raw-v1 test",
            "perplexity_chunks": meta["perplexity_params"]["chunks"],
            "perplexity_ctx": meta["perplexity_params"]["ctx"],
            "basis": "measured",
            "session": SESSION,
        })
print("wrote both CSVs to", DST)
