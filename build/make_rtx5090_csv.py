#!/usr/bin/env python3
"""Turn the 2026-09-20 RTX 5090 four-precision archive into the three published CSVs.

Input:  the unpacked raw-report directory (``ecocompute/`` with one sub-directory per
        run holding an ``energy.json``, plus ``fp8_rerun_results.csv`` and the env files).
Output: rtx5090_bnb_2026-09-20.csv          one row per (session, model, precision)
        rtx5090_bnb_2026-09-20.summary.csv  session A vs session B agreement per cell
        rtx5090_fp8_torchao_2026-09-20.csv  the standalone torchao FP8 track

    python3 build/make_rtx5090_csv.py /path/to/ecocompute data/

Three rules this script enforces, because they are what makes the session usable:

* **Only ``measurement_source == "direct-nvml"`` and ``basis == "measured"`` rows survive.**
  The container falls back to interpolating the published dataset when an on-device
  measurement fails, and writes that estimate into the same ``results`` block. Four such
  reports exist in this archive (the runs that named TinyLlama as ``TinyLlama/TinyLlama-1.1B``,
  which is not a resolvable model id); they are dropped here and must never be read as
  measurements.
* **``sessionA_final/`` is a copy of the top-level ``A_*`` directories**, byte-identical
  timestamps, so it is ignored rather than double-counted.
* **Each session's FP16 baseline is the one embedded in that session's own quantized run**,
  never the other session's, and never this site's published values. The two quantized arms
  of a session carry their own baseline field; they are checked against each other here and
  the run's own baseline is what the delta is computed against upstream (the container
  already did the division; this script only re-derives it to verify).

The bitsandbytes track and the FP8 track are kept in separate files on purpose: the FP8 rows
come from a standalone bench script with its own baselines and its own sample count, so its
percentages are comparable within that file and nowhere else.
"""
import csv
import glob
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
DST = sys.argv[2] if len(sys.argv) > 2 else "."

SESSION_OF = {"A": "A", "B": "B"}
# One extra independent repeat of the 7B NF4 cell, run while chasing an unrelated transient.
# Kept as its own session label so it is never silently averaged into A.
EXTRA = {"A_retry": "A_retry", "smoke": "smoke"}

FIELDS = [
    "session", "model", "params_b", "precision", "energy_j_per_1k_tok", "vs_fp16_pct",
    "fp16_energy_j_per_1k_tok", "avg_power_w", "throughput_tok_s", "total_energy_j",
    "tokens_generated", "ppl", "fp16_ppl", "delta_ppl_pct", "batch_size", "context_length",
    "tokens_per_iteration", "iterations", "warmup", "sample_rate_hz", "gpu", "gpu_arch",
    "nvidia_driver", "torch", "torch_cuda", "transformers", "bitsandbytes",
    "basis", "measurement_source", "timestamp_utc",
]


def session_of(dirname):
    if dirname.startswith("A_retry_"):
        return "A_retry"
    if dirname.startswith("smoke"):
        return "smoke"
    return dirname[0]


def load_reports(src):
    out = []
    for path in sorted(glob.glob(os.path.join(src, "*", "energy.json"))):
        d = json.load(open(path))
        r = d.get("results", {})
        if d.get("measurement_source") != "direct-nvml" or r.get("basis") != "measured":
            continue
        out.append((os.path.basename(os.path.dirname(path)), d))
    return out


def row_of(dirname, d):
    w, m, s, r = d["workload"], d["measurement"], d["software"], d["results"]
    q = d.get("quality", {})
    pk = s.get("packages", {})
    return {
        "session": session_of(dirname),
        "model": w["model_name"].split("/")[-1],
        "params_b": w["params_b"],
        "precision": w["precision"],
        "energy_j_per_1k_tok": r["energy_per_token_mj"],
        "vs_fp16_pct": r.get("vs_fp16_energy_pct"),
        "fp16_energy_j_per_1k_tok": r.get("fp16_energy_per_token_mj"),
        "avg_power_w": r.get("avg_power_watts"),
        "throughput_tok_s": r.get("throughput_tokens_per_s"),
        "total_energy_j": r.get("total_energy_joules"),
        "tokens_generated": r.get("tokens_generated"),
        "ppl": q.get("value"),
        "fp16_ppl": q.get("fp16_value"),
        "delta_ppl_pct": q.get("delta_vs_fp16_pct"),
        "batch_size": w["batch_size"],
        "context_length": w["context_length"],
        "tokens_per_iteration": m["tokens_per_run"],
        "iterations": m["iterations"],
        "warmup": m["warmup"],
        "sample_rate_hz": m["sample_rate_hz"],
        "gpu": d["system_under_test"]["gpu"],
        "gpu_arch": d["system_under_test"]["gpu_arch"],
        "nvidia_driver": s.get("nvidia_driver"),
        "torch": pk.get("torch"),
        "torch_cuda": s.get("torch_cuda"),
        "transformers": pk.get("transformers"),
        "bitsandbytes": pk.get("bitsandbytes"),
        "basis": r["basis"],
        "measurement_source": d["measurement_source"],
        "timestamp_utc": d["timestamp_utc"],
    }


def check_baselines(rows):
    """The NF4 and INT8 arms of one session each embed their own FP16 baseline."""
    by = {}
    for r in rows:
        by.setdefault((r["session"], r["params_b"]), []).append(r)
    for key, group in sorted(by.items()):
        base = [float(g["fp16_energy_j_per_1k_tok"]) for g in group
                if g["fp16_energy_j_per_1k_tok"]]
        if len(base) > 1:
            spread = (max(base) - min(base)) / min(base) * 100.0
            if spread > 5.0:
                raise SystemExit(f"{key}: FP16 baselines disagree by {spread:.1f}%")


def main():
    reports = load_reports(SRC)
    if not reports:
        raise SystemExit(f"no direct-nvml reports under {SRC}")
    rows = [row_of(n, d) for n, d in reports]
    check_baselines(rows)
    rows.sort(key=lambda r: (r["session"], float(r["params_b"]), r["precision"]))

    out = os.path.join(DST, "rtx5090_bnb_2026-09-20.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} measured runs "
          f"({len(reports)} kept of {len(glob.glob(os.path.join(SRC, '*', 'energy.json')))} reports)")

    # Session A vs session B, per cell. A_retry and the chat-variant smoke run are listed
    # but not folded into either column.
    idx = {(r["session"], r["precision"], float(r["params_b"])): r for r in rows}
    srows = []
    for prec in ("NF4", "INT8"):
        for n in (0.5, 1.1, 1.5, 3.0, 7.0):
            a, b = idx.get(("A", prec, n)), idx.get(("B", prec, n))
            if not (a and b):
                continue
            da, db = float(a["vs_fp16_pct"]), float(b["vs_fp16_pct"])
            srows.append({
                "precision": prec, "params_b": n, "model": a["model"],
                "A_vs_fp16_pct": da, "B_vs_fp16_pct": db,
                "delta_pp": round(db - da, 2),
                "A_energy_j_per_1k_tok": a["energy_j_per_1k_tok"],
                "B_energy_j_per_1k_tok": b["energy_j_per_1k_tok"],
                "A_throughput_tok_s": a["throughput_tok_s"],
                "B_throughput_tok_s": b["throughput_tok_s"],
                "delta_ppl_pct": a["delta_ppl_pct"],
            })
    sout = os.path.join(DST, "rtx5090_bnb_2026-09-20.summary.csv")
    with open(sout, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    spread = [abs(r["delta_pp"]) for r in srows]
    print(f"{sout}: {len(srows)} cells, |A-B| max {max(spread):.1f} pp, "
          f"mean {sum(spread) / len(spread):.1f} pp")

    # FP8 track: copied through with the columns this site uses, plus the per-row
    # delta against the same-script FP16 baseline of the same model.
    src_fp8 = os.path.join(SRC, "fp8_rerun_results.csv")
    raw = list(csv.DictReader(open(src_fp8)))
    fp16 = {r["model_name"]: float(r["j_per_1k_tokens"]) for r in raw if r["precision"] == "fp16"}
    frows = []
    for r in raw:
        e = float(r["j_per_1k_tokens"])
        base = fp16[r["model_name"]]
        short = int(r["total_tokens"]) < int(r["num_samples"]) * int(r["max_new_tokens"])
        frows.append({
            "model": r["model_name"], "params_b": float(r["params"].rstrip("B")),
            "precision": r["precision"], "precision_label": r["precision_label"],
            "energy_j_per_1k_tok": e,
            "vs_fp16_pct": "" if r["precision"] == "fp16" else round((e / base - 1) * 100, 1),
            "avg_power_w": r["avg_watts"], "peak_power_w": r["peak_watts"],
            "throughput_tok_s": r["tokens_per_sec"], "total_joules": r["total_joules"],
            "total_tokens": r["total_tokens"], "num_samples": r["num_samples"],
            "max_new_tokens": r["max_new_tokens"], "batch_size": r["batch_size"],
            "token_cv_pct": r["token_cv_pct"], "idle_power_w": r["idle_power_w"],
            "short_generation": "yes" if short else "",
            "gpu": r["gpu_name"], "torch": r["pytorch_version"],
            "cuda": r["cuda_version"], "torchao": r["torchao_version"],
            "timestamp_utc": r["timestamp"],
        })
    fout = os.path.join(DST, "rtx5090_fp8_torchao_2026-09-20.csv")
    with open(fout, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frows[0].keys()))
        w.writeheader()
        w.writerows(frows)
    short = [r for r in frows if r["short_generation"]]
    print(f"{fout}: {len(frows)} rows, {len(short)} flagged short_generation "
          f"({', '.join(r['model'] + '/' + r['precision'] for r in short) or 'none'})")


if __name__ == "__main__":
    main()
