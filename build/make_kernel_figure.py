#!/usr/bin/env python3
"""Generate assets/rtx4090-kernel-decides-the-sign.png.

Three weight-only quantizations of a 7-8B model, all on an RTX 4090 (Ada), each compared
against an FP16 baseline measured in its own session. The energy delta spans +106% to
-64% depending only on which kernel executes the quantized weights.

    python3 build/make_kernel_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0b1020"
PANEL = "#111834"
TXT = "#e6ecff"
MUTED = "#93a0c4"
LINE = "#22305a"
GOOD = "#10b981"
BAD = "#f43f5e"
ACCENT = "#38bdf8"

# (label, sublabel, delta energy % vs same-session FP16, delta perplexity %)
ROWS = [
    ("bitsandbytes LLM.int8()", "Qwen2-7B · transformers · 2026-08-19", 105.78, 1.193),
    ("bitsandbytes NF4", "Qwen2-7B · transformers · 2026-08-19", -39.04, 12.057),
    ("llama.cpp GGUF Q4_0", "Llama-3.1-8B-Instruct · 2026-08-31", -63.57, 5.602),
]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.2, 4.3), dpi=140, gridspec_kw={"width_ratios": [1.55, 1]}
)
fig.patch.set_facecolor(BG)
y = range(len(ROWS))[::-1]

for ax in (ax1, ax2):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_yticks(list(y))
    ax.grid(axis="x", color=LINE, lw=0.7)
    ax.set_axisbelow(True)

ax1.set_yticklabels([r[0] for r in ROWS], color=TXT, fontsize=10)
for yy, (_, sub, _, _) in zip(y, ROWS):
    ax1.text(-0.012, yy - 0.30, sub, transform=ax1.get_yaxis_transform(),
             ha="right", va="center", color=MUTED, fontsize=7.5)

de = [r[2] for r in ROWS]
ax1.barh(list(y), de, height=0.46, color=[BAD if v > 0 else GOOD for v in de])
ax1.axvline(0, color=MUTED, lw=1)
for yy, v in zip(y, de):
    ax1.text(v + (5 if v > 0 else -5), yy, f"{v:+.1f}%", va="center",
             ha="left" if v > 0 else "right", color=TXT, fontsize=10, fontweight="bold")
ax1.set_xlim(-112, 132)
ax1.set_xlabel("decode energy per token vs FP16, same session (%)", color=MUTED, fontsize=9)
ax1.set_title("The same idea — 4/8-bit weight-only, one RTX 4090 —\nlands on both sides of zero",
              color=TXT, fontsize=11, fontweight="bold", loc="left", pad=10)

dp = [r[3] for r in ROWS]
ax2.barh(list(y), dp, height=0.46, color=ACCENT)
ax2.set_yticklabels([])
for yy, v in zip(y, dp):
    ax2.text(v + 0.35, yy, f"+{v:.2f}%", va="center", color=TXT, fontsize=10, fontweight="bold")
ax2.set_xlim(0, 19)
ax2.set_xlabel("perplexity vs FP16, same session (%)", color=MUTED, fontsize=9)
ax2.set_title("Quality cost does not order them\nthe same way", color=TXT, fontsize=11,
              fontweight="bold", loc="left", pad=10)

fig.text(0.008, 0.015,
         "GPU-package power via direct NVML · batch 1, single stream · n=1 (bitsandbytes rows) and n=3 (llama.cpp row) · "
         "each delta is against an FP16 baseline from its own session.\nDifferent models, runtimes and token counts: read "
         "the sign and the mechanism, not the exact magnitudes. The llama.cpp row is decode-only by differencing "
         "576- and 64-token runs. Not a certified benchmark result.",
         color=MUTED, fontsize=6.8, linespacing=1.5)
fig.subplots_adjust(left=0.205, right=0.985, top=0.83, bottom=0.235, wspace=0.06)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "assets", "rtx4090-kernel-decides-the-sign.png")
fig.savefig(out, facecolor=BG)
print("wrote", out)
