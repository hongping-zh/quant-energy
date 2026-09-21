#!/usr/bin/env python3
"""Generate the two RTX 5090 figures for the 2026-09-20 session.

    python3 build/make_rtx5090_figures.py

assets/rtx5090-crossover-moved.png
    The same card and the same four models, measured months apart on two software stacks.
    Reads the v1.1.0 anchors from build/measured.csv and the September session from
    data/rtx5090_bnb_2026-09-20.summary.csv, so neither curve is typed in here. The two
    curves are *not* a controlled experiment: driver, CUDA, torch and bitsandbytes all
    moved together, and only the September session records its library versions. What the
    figure shows is that the crossover is not fixed by the card.

assets/rtx5090-fp8-two-paths.png
    The torchao FP8 track from data/rtx5090_fp8_torchao_2026-09-20.csv: the two FP8 code
    paths get worse in opposite directions with model size, and the weight-only path draws
    more power than FP16 while running 7x slower.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG = "#0b1020"
PANEL = "#111834"
TXT = "#e6ecff"
MUTED = "#93a0c4"
LINE = "#22305a"
GOOD = "#10b981"
BAD = "#f43f5e"
ACCENT = "#38bdf8"
OLD = "#f59e0b"


def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=LINE, lw=0.7)
    ax.set_axisbelow(True)


def read(path):
    with open(os.path.join(HERE, path)) as fh:
        return list(csv.DictReader(fh))


def crossing(xs, ys):
    """Linear interpolation of the single sign change, or None."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 > 0 >= y1:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    return None


def crossover_figure():
    old = [(float(r["params_b"]),
            (float(r["energy_quant_j1k"]) / float(r["energy_fp16_j1k"]) - 1) * 100)
           for r in read("build/measured.csv")
           if r["gpu"] == "RTX 5090" and r["precision"] == "NF4"]
    old.sort()
    new = [(float(r["params_b"]),
            (float(r["A_vs_fp16_pct"]) + float(r["B_vs_fp16_pct"])) / 2)
           for r in read("data/rtx5090_bnb_2026-09-20.summary.csv") if r["precision"] == "NF4"]
    new.sort()
    # v1.1.0 has no 0.5B point on this card; tick the sizes both sessions measured, plus it.
    shared = sorted({n for n, _ in old} & {n for n, _ in new})

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=140)
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.axhspan(-100, 0, color=GOOD, alpha=0.05)
    # Largest session-to-session disagreement among the NF4 cells measured twice.
    band = max(abs(float(r["delta_pp"]))
               for r in read("data/rtx5090_bnb_2026-09-20.summary.csv")
               if r["precision"] == "NF4")
    ax.axhspan(-band, band, color=MUTED, alpha=0.13, zorder=1)
    ax.text(8.4, -band - 1.0, f"+/-{band:.1f} pp: largest A-vs-B disagreement in this session",
            color=MUTED, fontsize=8, va="top", ha="right")

    for (xs_ys, color, label, marker) in (
        (old, OLD, "main dataset v1.1.0 · published 2026-04-19", "o"),
        (new, ACCENT, "bitsandbytes 0.50.2 · this session, 2026-09-20 (n=2)", "s"),
    ):
        xs = [x for x, _ in xs_ys]
        ys = [y for _, y in xs_ys]
        ax.plot(xs, ys, color=color, lw=2.2, marker=marker, ms=7, label=label, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 11 if y >= 0 else -17), ha="center",
                        color=color, fontsize=9, fontweight="bold")

    for xs_ys, color in ((old, OLD), (new, ACCENT)):
        c = crossing([x for x, _ in xs_ys], [y for _, y in xs_ys])
        if c:
            ax.plot([c], [0], marker="v", ms=9, color=color, zorder=4)
            ax.annotate(f"crosses zero\nnear {c:.1f} B", (c, 0), textcoords="offset points",
                        xytext=(0, -46), ha="center", color=color, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(shared + [0.5])
    ax.set_xticklabels([f"{x:g}B" for x in shared + [0.5]])
    ax.set_xlim(0.42, 8.6)
    ax.set_ylim(-42, 40)
    ax.set_xlabel("model size (parameters, log scale)", color=MUTED, fontsize=10)
    ax.set_ylabel("NF4 energy per token vs FP16, same session (%)", color=MUTED, fontsize=10)
    ax.set_title("One RTX 5090, the same four models, two software stacks:\n"
                 "the NF4 crossover moved, the card did not",
                 color=TXT, fontsize=13, fontweight="bold", loc="left", pad=12)
    leg = ax.legend(loc="lower left", facecolor=PANEL, edgecolor=LINE, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(TXT)
    fig.text(0.012, 0.015,
             "Not a controlled experiment: driver, CUDA, torch and bitsandbytes moved together "
             "between the two sessions, and only the\nSeptember one records its library versions. "
             "Zero crossings are linear interpolation between adjacent measured points, not a fit. "
             "Whole-process decode\nenergy, batch 1, 256 tokens x 10 iterations, GPU-package power.",
             color=MUTED, fontsize=7.6, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = os.path.join(HERE, "assets", "rtx5090-crossover-moved.png")
    fig.savefig(out, facecolor=BG)
    print(out)


def fp8_figure():
    rows = read("data/rtx5090_fp8_torchao_2026-09-20.csv")
    sizes = sorted({float(r["params_b"]) for r in rows})
    by = {(r["precision"], float(r["params_b"])): r for r in rows}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=140)
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{x:g}B" for x in sizes])
        ax.set_xlabel("model size (parameters, log scale)", color=MUTED, fontsize=9)

    for prec, color, label, dy in (("fp8", BAD, "FP8 weight-only", -17),
                                   ("fp8_dyn", ACCENT, "FP8 dynamic act + weight", 10)):
        ys = [float(by[(prec, n)]["vs_fp16_pct"]) for n in sizes]
        ax1.plot(sizes, ys, color=color, lw=2.2, marker="o", ms=6, label=label)
        for n, y in zip(sizes, ys):
            short = by[(prec, n)]["short_generation"]
            # The two curves cross near 1.5B; nudge those two labels apart.
            off = {("fp8", 1.5): (30, -17), ("fp8_dyn", 1.5): (6, -17)}.get((prec, n), (0, dy))
            ax1.annotate(f"{y:+.0f}%" + ("*" if short else ""), (n, y),
                         textcoords="offset points", xytext=off, ha="center",
                         color=color, fontsize=8.5, fontweight="bold")
    ax1.axhline(0, color=MUTED, lw=1.2)
    ax1.set_ylim(-75, 980)
    ax1.set_ylabel("energy per token vs same-script FP16 (%)", color=MUTED, fontsize=9)
    ax1.set_title("Both FP8 paths cost energy at every size —\nand they get worse in opposite directions",
                  color=TXT, fontsize=11, fontweight="bold", loc="left", pad=10)
    leg = ax1.legend(loc="upper left", facecolor=PANEL, edgecolor=LINE, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TXT)

    for prec, color, label in (("fp16", GOOD, "FP16"), ("fp8", BAD, "FP8 weight-only"),
                               ("fp8_dyn", ACCENT, "FP8 dynamic act + weight")):
        ax2.plot(sizes, [float(by[(prec, n)]["avg_power_w"]) for n in sizes],
                 color=color, lw=2.2, marker="o", ms=6, label=label)
    ax2.set_ylim(80, 480)
    ax2.set_ylabel("mean GPU-package power during generation (W)", color=MUTED, fontsize=9)
    ax2.set_title("Weight-only FP8 draws more power than FP16\nwhile decoding 7x slower",
                  color=TXT, fontsize=11, fontweight="bold", loc="left", pad=10)
    leg = ax2.legend(loc="upper left", facecolor=PANEL, edgecolor=LINE, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TXT)

    fig.text(0.012, 0.015,
             "torchao 0.18.0 on one RTX 5090, batch 1, 256 tokens, 3 samples per cell, one session; "
             "baselines measured by the same script.\n* the 1.1B weight-only run stopped early "
             "(177 of 768 tokens), so its per-token figure covers a shorter generation than the rest.",
             color=MUTED, fontsize=7.6, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = os.path.join(HERE, "assets", "rtx5090-fp8-two-paths.png")
    fig.savefig(out, facecolor=BG)
    print(out)


if __name__ == "__main__":
    crossover_figure()
    fp8_figure()
