#!/usr/bin/env python3
"""Generate assets/rtx4090-differencing-holds.png from the v2 summary + per-run CSVs.

Left panel: process energy against requested tokens, 45 runs, with the OLS fit per arm.
A straight line through three token lengths is what licenses reporting E(576) - E(64) as
decode energy; the intercept is the load term that differencing removes.

Right panel: mean decode power against decode throughput. The quantized arms draw more
instantaneous power than F16 and still cost 62% less per token, because they finish
sooner -- the point that neither a speed-only nor a power-only report can make.

    python3 build/make_v2_differencing_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0b1020"
PANEL = "#111834"
TXT = "#e6ecff"
MUTED = "#93a0c4"
LINE = "#22305a"
COLORS = {"F16": "#f43f5e", "ours-Q4_0.gguf": "#10b981", "Llama-3.1-8B-Instruct-Q4_0.gguf": "#38bdf8"}
LABELS = {
    "Llama-3.1-8B-Instruct-F16.gguf": "F16",
    "ours-Q4_0.gguf": "Q4_0 (ours)",
    "Llama-3.1-8B-Instruct-Q4_0.gguf": "Q4_0 (MLPerf Client v2.0 file)",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

runs = list(csv.DictReader(open(os.path.join(ROOT, "data", "rtx4090_llamacpp_gguf_v2_2026-09-03.csv"))))
summary = {r["gguf_file"]: r for r in
           csv.DictReader(open(os.path.join(ROOT, "data", "rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv")))}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=140,
                               gridspec_kw={"width_ratios": [1.35, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=LINE, lw=0.7)
    ax.set_axisbelow(True)

for gguf, label in LABELS.items():
    c = COLORS["F16" if "F16" in gguf else gguf]
    xs = [int(r["requested_tokens"]) for r in runs if r["gguf_file"] == gguf]
    ys = [float(r["process_energy_j"]) for r in runs if r["gguf_file"] == gguf]
    s = summary[gguf]
    a, b, r2 = float(s["ols_intercept_j"]), float(s["energy_mj_per_token_ols"]) / 1000, float(s["ols_r2"])
    ax1.plot([0, 600], [a, a + b * 600], color=c, lw=1.4, alpha=0.85)
    ax1.scatter(xs, ys, s=26, color=c, edgecolor=BG, lw=0.6, zorder=3,
                label=f"{label}   {b*1000:.0f} mJ/tok   R²={r2:.4f}")
    ax1.scatter([0], [a], s=34, facecolor=BG, edgecolor=c, lw=1.4, zorder=4)

ax1.set_xlim(-25, 625)
ax1.set_ylim(0, 3600)
ax1.set_xticks([0, 64, 320, 576])
ax1.set_xlabel("decoded tokens (--ignore-eos, so every run reached its target)", color=MUTED, fontsize=9)
ax1.set_ylabel("process energy, NVML counter (J)", color=MUTED, fontsize=9)
ax1.set_title("Energy is linear in tokens, so differencing is legitimate",
              color=TXT, fontsize=11, fontweight="bold", loc="left", pad=10)
leg = ax1.legend(loc="upper left", fontsize=8, framealpha=0)
for t in leg.get_texts():
    t.set_color(TXT)
ax1.annotate("intercepts (hollow) = weight load,\nremoved by differencing", (6, 275),
             xytext=(300, 470), color=MUTED, fontsize=7.5, va="center",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, alpha=0.7))

order = ["Llama-3.1-8B-Instruct-F16.gguf", "ours-Q4_0.gguf", "Llama-3.1-8B-Instruct-Q4_0.gguf"]
xs = [float(summary[g]["throughput_tok_s"]) for g in order]
ys = [float(summary[g]["decode_avg_power_w"]) for g in order]
OFFSETS = {
    "Llama-3.1-8B-Instruct-F16.gguf": ((0, -36), "center"),
    "ours-Q4_0.gguf": ((14, -2), "left"),
    "Llama-3.1-8B-Instruct-Q4_0.gguf": ((-10, 14), "right"),
}
for g, x, y in zip(order, xs, ys):
    c = COLORS["F16" if "F16" in g else g]
    off, ha = OFFSETS[g]
    ax2.scatter([x], [y], s=90, color=c, edgecolor=BG, lw=1, zorder=3)
    ax2.annotate(f"{LABELS[g].split(' (')[0]}\n{float(summary[g]['energy_mj_per_token']):.0f} mJ/tok",
                 (x, y), textcoords="offset points", xytext=off, ha=ha,
                 color=TXT, fontsize=8.5)
ax2.set_xlim(20, 210)
ax2.set_ylim(240, 345)
ax2.set_xlabel("decode throughput (tok/s)", color=MUTED, fontsize=9)
ax2.set_ylabel("mean decode power (W)", color=MUTED, fontsize=9)
ax2.set_title("More power, far less energy",
              color=TXT, fontsize=11, fontweight="bold", loc="left", pad=10)
ax2.annotate("", xy=(173.8, 318.7), xytext=(54.7, 272.6),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, alpha=0.7))
ax2.text(120, 288, "3.2× throughput\n+17% power", color=MUTED, fontsize=8, ha="center")

fig.text(0.008, 0.015,
         "One RTX 4090 (450 W), Llama-3.1-8B-Instruct, llama.cpp b10643-192067b72, batch 1 · 45 runs, n=5 per cell, randomized order, "
         "forced cooldown to ~25 W before each run\nGPU-package energy from the NVML hardware counter · decode figures difference the 576- and "
         "64-token cells · not an MLPerf result, no MLCommons review or endorsement.",
         color=MUTED, fontsize=6.8, linespacing=1.5)
fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.235, wspace=0.24)

out = os.path.join(ROOT, "assets", "rtx4090-differencing-holds.png")
fig.savefig(out, facecolor=BG)
print("wrote", out)
