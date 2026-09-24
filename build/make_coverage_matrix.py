#!/usr/bin/env python3
"""Build the coverage matrix: what has been measured, and what nobody has measured yet.

Consolidates every energy measurement on the site into one grid of
(quantization method x GPU architecture) x model size, writes it as a CSV and renders
assets/coverage-matrix.png.

Sources, all already published here:
  build/measured.csv                                  29 anchors, five cards
  data/rtx4090_paired_energy_quality_2026-08-19.csv   10 rows with paired perplexity
  data/rtx4090_int8_repeats_2026-08-20.summary.csv     5 rows, n=3 each
  data/replications/*.energy.json                      external contributions
  data/rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv
                                                       llama.cpp, n=5, generation window

Every cell is on the generation window now: the bitsandbytes rows always were
(the container starts NVML sampling after load, quantization and warm-up - the
"whole-process" label this grid once carried was wrong), and the llama.cpp row
is re-cut from its archived 100 Hz power traces by
build/make_window_comparison.py. The llama.cpp block is still drawn separately
because it is a different runtime and workload shape (one 576-token generation
per run vs the container's 10 x 256 tokens with per-iteration prefill) - same
measurement window, different animal, do not read across the line carelessly.

    python3 build/make_coverage_matrix.py
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG = "#0b1020"
PANEL = "#111834"
TXT = "#e6ecff"
MUTED = "#93a0c4"
LINE = "#22305a"
SAVE = "#10b981"     # energy saving
COST = "#f43f5e"     # energy penalty
EMPTY = "#161d3a"

SIZES = [0.5, 1.1, 1.5, 3.0, 7.0, 8.0, 9.0, 14.0]
SIZE_LABELS = ["0.5B", "1.1B", "1.5B", "3B", "7B", "8B", "9B", "14B"]

ARCH_LABEL = {
    "turing": "Turing",
    "ampere": "Ampere",
    "ada": "Ada",
    "blackwell": "Blackwell",
}

# (method, arch) rows, in the order they are drawn
ROWS = [
    ("bnb INT8", "ada"),
    ("bnb INT8", "ampere"),
    ("bnb NF4", "turing"),
    ("bnb NF4", "ampere"),
    ("bnb NF4", "ada"),
    ("bnb NF4", "blackwell"),
    ("llama.cpp Q4_0", "ada"),
]
SPLIT_BEFORE = ("llama.cpp Q4_0", "ada")   # runtime / workload shape changes here


def cells_init():
    return {row: {s: {"deltas": [], "n": 0, "gpus": set(), "dppl": []} for s in SIZES} for row in ROWS}


def add(cells, method, arch, params_b, delta_pct, n, gpu, dppl=None):
    key = (method, arch)
    if key not in cells or params_b not in cells[key]:
        raise KeyError(f"no cell for {key} @ {params_b}B")
    c = cells[key][params_b]
    c["deltas"].append(delta_pct)
    c["n"] += n
    c["gpus"].add(gpu)
    if dppl is not None:
        c["dppl"].append(dppl)


def load(cells):
    # ── 1. build/measured.csv: FP16 and quantized energy per 1k tokens on the same card ──
    for r in csv.DictReader(open(os.path.join(ROOT, "build", "measured.csv"))):
        fp16, quant = float(r["energy_fp16_j1k"]), float(r["energy_quant_j1k"])
        add(cells,
            "bnb INT8" if r["precision"] == "INT8" else "bnb NF4",
            r["arch"], float(r["params_b"]), (quant / fp16 - 1) * 100,
            int(r["n_trials"]), r["gpu"])

    # ── 2. RTX 4090 paired energy+quality, 2026-08-19 ──
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "rtx4090_paired_energy_quality_2026-08-19.csv"))):
        add(cells,
            "bnb INT8" if r["precision"] == "INT8" else "bnb NF4",
            r["gpu_arch"], float(r["params_b"]), float(r["vs_fp16_energy_pct"]),
            int(r["n_trials"]), "RTX 4090",
            float(r["delta_perplexity_pct"]) if r["delta_perplexity_pct"] else None)

    # ── 3. RTX 4090 INT8 repeats, n=3 per cell ──
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "rtx4090_int8_repeats_2026-08-20.summary.csv"))):
        add(cells, "bnb INT8", "ada", float(r["params_b"]),
            float(r["vs_fp16_energy_pct_mean"]), int(r["n_trials"]), "RTX 4090",
            float(r["delta_perplexity_pct"]) if r["delta_perplexity_pct"] else None)

    # ── 4. external replications ──
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replications", "*.json"))):
        rep = json.load(open(path))
        res, wl = rep["results"], rep["workload"]
        if res.get("vs_fp16_energy_pct") is None:
            continue
        add(cells, "bnb INT8" if wl["precision"] == "INT8" else "bnb NF4",
            rep["system_under_test"]["gpu_arch"], float(wl["params_b"]),
            float(res["vs_fp16_energy_pct"]), 1, rep["system_under_test"]["gpu"])

    # ── 5. llama.cpp GGUF, generation window re-cut from the raw traces ──
    #    (build/make_window_comparison.py; supersedes the decode-only figure
    #     this grid used to show. 576 tokens: one long generation per run,
    #     the closest shape to the container's 10 x 256-token iterations.)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv"))):
        if r["n_tokens"] != "576" or r["arm"] == "f16":
            continue
        add(cells, "llama.cpp Q4_0", "ada", 8.0,
            float(r["vs_fp16_pct_gen_th50"]), int(r["n_replicates"]), "RTX 4090", None)


def write_csv(cells, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "gpu_arch", "params_b", "vs_fp16_energy_pct_mean",
                    "vs_fp16_energy_pct_min", "vs_fp16_energy_pct_max",
                    "n_measurements", "gpus", "delta_perplexity_pct_mean", "denominator"])
        for (method, arch) in ROWS:
            denom = ("generation-only (re-cut from the raw trace)"
                     if method.startswith("llama.cpp")
                     else "generation-only (container window)")
            for size in SIZES:
                c = cells[(method, arch)][size]
                if not c["deltas"]:
                    continue
                d = c["deltas"]
                w.writerow([method, arch, size, round(sum(d) / len(d), 1),
                            round(min(d), 1), round(max(d), 1), c["n"],
                            "; ".join(sorted(c["gpus"])),
                            round(sum(c["dppl"]) / len(c["dppl"]), 2) if c["dppl"] else "",
                            denom])


def colour(delta):
    """Sign carries the message; magnitude only modulates intensity."""
    mag = min(abs(delta) / 120.0, 1.0)
    base = SAVE if delta < 0 else COST
    alpha = 0.20 + 0.62 * mag
    return base, alpha


def render(cells, path):
    n_rows, n_cols = len(ROWS), len(SIZES)
    fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    split_index = ROWS.index(SPLIT_BEFORE)
    gap = 0.55   # visual break where the runtime / workload shape changes

    def row_y(ri):
        return n_rows - 1 - ri - (gap if ri >= split_index else 0.0)

    filled = 0
    for ri, row in enumerate(ROWS):
        y = row_y(ri)
        for ci, size in enumerate(SIZES):
            c = cells[row][size]
            if c["deltas"]:
                filled += 1
                mean = sum(c["deltas"]) / len(c["deltas"])
                base, alpha = colour(mean)
                ax.add_patch(Rectangle((ci, y), 0.94, 0.9, facecolor=base, alpha=alpha,
                                       edgecolor=base, lw=1.0))
                lo, hi = min(c["deltas"]), max(c["deltas"])
                spread = hi - lo > 5
                label = f"{mean:+.0f}%"
                ax.text(ci + 0.47, y + (0.62 if spread else 0.56), label,
                        ha="center", va="center", color=TXT,
                        fontsize=11.5 if len(label) <= 4 else 10.2, fontweight="bold")
                ax.text(ci + 0.47, y + (0.38 if spread else 0.26), f"n={c['n']}",
                        ha="center", va="center", color=MUTED, fontsize=7.5)
                if spread:
                    ax.text(ci + 0.47, y + 0.17, f"{lo:+.0f}\u2026{hi:+.0f}", ha="center",
                            va="center", color=MUTED, fontsize=6.4)
            else:
                ax.add_patch(Rectangle((ci, y), 0.94, 0.9, facecolor=EMPTY,
                                       edgecolor=LINE, lw=0.8, ls=(0, (3, 3))))
                ax.text(ci + 0.47, y + 0.45, "?", ha="center", va="center",
                        color="#3d4a75", fontsize=13, fontweight="bold")

    ax.text(-3.6, n_rows + 1.35,
            "Same word, opposite results: where quantization saved energy and where it cost energy",
            ha="left", va="center", color=TXT, fontsize=13, fontweight="bold")
    ax.text(-3.6, n_rows + 0.95,
            "Every published EcoCompute measurement, on one grid. "
            "NVML GPU-package power, batch 1, each cell measured against an FP16 baseline on the same card.",
            ha="left", va="center", color=MUTED, fontsize=9)

    for ci, lab in enumerate(SIZE_LABELS):
        ax.text(ci + 0.47, n_rows + 0.16, lab, ha="center", va="bottom",
                color=TXT, fontsize=10.5, fontweight="bold")
    ax.text(-0.25, n_rows + 0.16, "model size →", ha="right", va="bottom",
            color=MUTED, fontsize=9.5, style="italic")

    for ri, (method, arch) in enumerate(ROWS):
        y = row_y(ri)
        ax.text(-0.25, y + 0.58, method, ha="right", va="center",
                color=TXT, fontsize=10.5, fontweight="bold")
        ax.text(-0.25, y + 0.26, ARCH_LABEL[arch], ha="right", va="center",
                color=MUTED, fontsize=9)

    split_y = row_y(split_index) + 0.9 + gap / 2
    ax.plot([-3.6, n_cols], [split_y, split_y], color=LINE, lw=1.2)
    ax.text(n_cols - 0.06, split_y - 0.08,
            "below the line: llama.cpp — same generation window, different runtime and workload shape (1\u00d7576 tokens vs 10\u00d7256)",
            ha="right", va="top", color=MUTED, fontsize=8, style="italic")

    ax.text(-3.6, -1.17,
            "Change in energy per token vs FP16 on the same card.  "
            "Green = quantization saved energy,  red = it cost energy.  "
            "Cells are the unweighted mean of n measurements, with their full range where it exceeds 5 points.",
            ha="left", va="center", color=MUTED, fontsize=9)
    ax.text(-3.6, -1.57,
            f"{filled} of {n_rows * n_cols} cells measured.  "
            "Every ? is a run nobody has done — a free Colab T4 fills one in about half an hour.",
            ha="left", va="center", color=TXT, fontsize=9.5, fontweight="bold")
    ax.text(n_cols + 0.05, -1.57, "quantenergy.tech · CC BY 4.0", ha="right", va="center",
            color=MUTED, fontsize=8.5)

    ax.set_xlim(-3.7, n_cols + 0.1)
    ax.set_ylim(-1.9, n_rows + 1.7)
    ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    print(f"wrote {path}  ({filled}/{n_rows * n_cols} cells filled)")


def main():
    cells = cells_init()
    load(cells)
    csv_path = os.path.join(ROOT, "data", "coverage_matrix_2026-09-09.csv")
    write_csv(cells, csv_path)
    print(f"wrote {csv_path}")
    render(cells, os.path.join(ROOT, "assets", "coverage-matrix.png"))


if __name__ == "__main__":
    main()
