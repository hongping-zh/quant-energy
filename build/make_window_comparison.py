#!/usr/bin/env python3
"""Re-integrate the llama.cpp v2 session's raw power traces over three
measurement windows.

Why this exists: the 2026-09-03 llama.cpp session published decode-only energy
(obtained by differencing E(576)-E(64)), while the bitsandbytes container rows
are the container's generation window (sampling starts after load, quantization
and warm-up). The two were never on one definition, so cross-engine cells could
not legitimately be compared. The session's 45 runs each archived a full 100 Hz
power trace; this script re-integrates those traces so every window can be cut
from the same runs:

  whole-process   the NVML hardware energy counter over the whole llama-cli
                  process (the archived aggregate, reproduced here as a check)
  generation      the counter's energy allocated to the generation phase in
                  proportion to the trace: E_gen = counter x (integral of the
                  trace over the generation window / integral over the whole
                  trace). All three windows are therefore on one energy basis
                  (the NVML counter), and the trace only decides WHERE the
                  window boundaries fall.
  decode-only     E(576) - E(64), the published differenced figure (counter)

Phase split: loading a GGUF with -ngl 99 sits on a ~70 W plateau while
generation runs hundreds of watts above it. The generation start is the first
sample above `frac x run-max` that stays above for the following 20 samples;
frac defaults to 0.5, with 0.4/0.55/0.6 written alongside so the choice can be
audited. Caveat the data forces on us: at 64 output tokens the Q4_0 arms peak
near 150 W, so 0.4 x max falls below the ~70 W load plateau and sub-0.45
thresholds are meaningless for those runs - the 64-token sensitivity is
reported for 0.50/0.55 only.

Time axis: the JSONs archive watts only, so sample times are reconstructed as
wall_s / (n_samples - 1) uniformly. The axis jitter this introduces (worst case
~2.2% on the full-trace integral vs the archived energy_trapezoid_j) cancels in
the window-share ratio and is bounded by a per-run sanity check. Window
energies themselves are anchored to the hardware counter, which has no axis at
all.

Reproduction guards (asserted, so the script fails loudly on wrong input):
  * whole-process deltas vs F16: -57.8/-55.2 (64), -60.7/-61.0 (320),
    -61.0/-61.4 (576) — the published summary;
  * decode-only differenced: -61.87/-63.17 — the published headline;
  * generation window at 0.5 x max: -68.6/-67.7, -65.3/-65.2, -62.9/-63.1 —
    the 2026-09-22 window-unification re-analysis (the 576-token "ours"
    value is -62.95 at two decimals and displays as -62.9 at one decimal,
    matching the manuscript's Table 9).

Input:  data/llamacpp_v2_traces/res_{arm}_n{tokens}_r{replicate}.json
        (arm in f16 / q4ours / q4mlperf; the JSONs the rented 4090 wrote,
        vendored into this repo because the Zenodo record 10.5281/zenodo.22295184
        archived only the aggregate CSVs)
Output: data/rtx4090_llamacpp_window_comparison_2026-09-24.csv         per run
        data/rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv per arm

    python3 build/make_window_comparison.py [traces_dir]
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACES = os.path.join(ROOT, "data", "llamacpp_v2_traces")

ARMS = ("f16", "q4ours", "q4mlperf")
ARM_LABEL = {"f16": "F16", "q4ours": "Q4_0 (ours)",
             "q4mlperf": "Q4_0 (MLPerf Client v2.0 file)"}
LENGTHS = (64, 320, 576)
REPS = (1, 2, 3, 4, 5)
FRACS = (0.40, 0.50, 0.55, 0.60)
FRAC_KEY = {0.40: "th40", 0.50: "th50", 0.55: "th55", 0.60: "th60"}
SUSTAIN = 20          # samples the trace must stay above the threshold

# Figures this script must reproduce — guards against wrong input.
EXPECTED_WHOLE_PCT = {64: (-57.8, -55.2), 320: (-60.7, -61.0), 576: (-61.0, -61.4)}
EXPECTED_DECODE_PCT = (-61.87, -63.17)
EXPECTED_GEN_PCT = {64: (-68.6, -67.7), 320: (-65.3, -65.2), 576: (-62.9, -63.1)}
TOL = 0.15
AXIS_TOL = 0.025      # reconstructed-axis vs archived trapezoid, per run


def load_run(traces_dir, arm, n, rep):
    path = os.path.join(traces_dir, "res_%s_n%d_r%d.json" % (arm, n, rep))
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tr = d["power_trace_w"]
    n_s = len(tr)
    if n_s < 2 or d["n_samples"] != n_s:
        raise SystemExit("trace problem in %s: n_samples mismatch" % path)
    dt = d["wall_s"] / (n_s - 1)
    whole = sum((tr[i] + tr[i + 1]) / 2.0 * dt for i in range(n_s - 1))
    if abs(whole - d["energy_trapezoid_j"]) / d["energy_trapezoid_j"] > AXIS_TOL:
        raise SystemExit(
            "time-axis sanity failed for %s: recomputed %.1f J vs archived %.1f J"
            % (path, whole, d["energy_trapezoid_j"]))
    return d, tr, dt, whole


def gen_start_index(tr, frac):
    """First index above frac*max that stays above for SUSTAIN samples."""
    th = frac * max(tr)
    n_s = len(tr)
    for i in range(n_s):
        if tr[i] > th and all(p > th for p in tr[i:i + SUSTAIN]):
            return i
    return n_s - 1


def window_share(tr, dt, start):
    """Fraction of the whole-trace integral that lies at/after `start`."""
    n_s = len(tr)
    win = sum((tr[i] + tr[i + 1]) / 2.0 * dt for i in range(start, n_s - 1))
    whole = sum((tr[i] + tr[i + 1]) / 2.0 * dt for i in range(n_s - 1))
    return win / whole


def analyse(traces_dir):
    rows = []
    for arm in ARMS:
        for n in LENGTHS:
            for rep in REPS:
                d, tr, dt, whole_raw = load_run(traces_dir, arm, n, rep)
                counter = d["energy_counter_j"]
                rec = {
                    "arm": arm, "n_tokens": n, "replicate": rep,
                    "wall_s": round(d["wall_s"], 3),
                    "energy_counter_j": counter,
                    "whole_trapezoid_j": round(d["energy_trapezoid_j"], 2),
                }
                for frac in FRACS:
                    g = gen_start_index(tr, frac)
                    share = window_share(tr, dt, g)
                    key = FRAC_KEY[frac]
                    rec["gen_start_s_" + key] = round(g * dt, 3)
                    rec["gen_energy_j_" + key] = round(counter * share, 2)
                gen05 = rec["gen_energy_j_th50"]
                rec["load_energy_j"] = round(counter - gen05, 2)
                rec["load_share_pct"] = round(100.0 * rec["load_energy_j"] / counter, 1)
                rows.append(rec)
    return rows


def mean(vals):
    return sum(vals) / len(vals)


def mval(rows, arm, n, key):
    return mean([r[key] for r in rows if r["arm"] == arm and r["n_tokens"] == n])


def pct_vs_f16(rows, arm, n, key):
    f = mval(rows, "f16", n, key)
    v = mval(rows, arm, n, key)
    return 100.0 * (v - f) / f


def check_against_published(rows):
    """Assert the pipeline reproduces the published / re-analysed figures."""
    for n in LENGTHS:
        for j, arm in enumerate(("q4ours", "q4mlperf")):
            for key, want_map, label in (
                    ("energy_counter_j", EXPECTED_WHOLE_PCT, "whole-process"),
                    ("gen_energy_j_th50", EXPECTED_GEN_PCT, "generation")):
                got = pct_vs_f16(rows, arm, n, key)
                want = want_map[n][j]
                if abs(got - want) > TOL:
                    raise SystemExit(
                        "%s check failed at n=%d %s: got %.2f, expected %.2f"
                        % (label, n, arm, got, want))
    # decode-only by differencing, counter basis
    def diff(arm):
        return (mval(rows, arm, 576, "energy_counter_j")
                - mval(rows, arm, 64, "energy_counter_j"))
    fd = diff("f16")
    for j, arm in enumerate(("q4ours", "q4mlperf")):
        got = 100.0 * (diff(arm) - fd) / fd
        if abs(got - EXPECTED_DECODE_PCT[j]) > TOL:
            raise SystemExit(
                "decode-only check failed for %s: got %.2f, expected %.2f"
                % (arm, got, EXPECTED_DECODE_PCT[j]))
    print("checks passed: whole-process, generation-window and decode-only all "
          "reproduce their published figures")


def write_per_run(rows, path):
    cols = ["arm", "arm_label", "n_tokens", "replicate", "wall_s",
            "energy_counter_j", "whole_trapezoid_j",
            "gen_start_s_th50", "gen_energy_j_th50",
            "gen_energy_j_th40", "gen_energy_j_th55", "gen_energy_j_th60",
            "load_energy_j", "load_share_pct"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["arm"], ARM_LABEL[r["arm"]], r["n_tokens"], r["replicate"],
                        r["wall_s"], r["energy_counter_j"], r["whole_trapezoid_j"],
                        r["gen_start_s_th50"], r["gen_energy_j_th50"],
                        r["gen_energy_j_th40"], r["gen_energy_j_th55"],
                        r["gen_energy_j_th60"], r["load_energy_j"], r["load_share_pct"]])


def write_summary(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "arm_label", "n_tokens", "n_replicates",
                    "whole_counter_j_mean", "gen_energy_j_th50_mean",
                    "load_energy_j_mean", "load_share_pct_mean",
                    "vs_fp16_pct_whole_counter", "vs_fp16_pct_gen_th50",
                    "vs_fp16_pct_gen_th40", "vs_fp16_pct_gen_th55",
                    "vs_fp16_pct_gen_th60",
                    "vs_fp16_pct_decode_only_diff",
                    "threshold_note"])
        def dmean(arm, n, key):
            return mval(rows, arm, n, key)
        fdiff = dmean("f16", 576, "energy_counter_j") - dmean("f16", 64, "energy_counter_j")
        for arm in ARMS:
            adiff = (dmean(arm, 576, "energy_counter_j")
                     - dmean(arm, 64, "energy_counter_j"))
            for n in LENGTHS:
                sub = [r for r in rows if r["arm"] == arm and r["n_tokens"] == n]
                note = ""
                if n == 64 and arm != "f16":
                    note = ("0.4 x max falls below the ~70 W load plateau for these "
                            "runs (peak ~150 W); the th40 column is not a meaningful "
                            "sensitivity here - use th50/th55")
                w.writerow([
                    arm, ARM_LABEL[arm], n, len(sub),
                    round(dmean(arm, n, "energy_counter_j"), 2),
                    round(dmean(arm, n, "gen_energy_j_th50"), 2),
                    round(dmean(arm, n, "load_energy_j"), 2),
                    round(dmean(arm, n, "load_share_pct"), 1),
                    round(pct_vs_f16(rows, arm, n, "energy_counter_j"), 2),
                    round(pct_vs_f16(rows, arm, n, "gen_energy_j_th50"), 2),
                    round(pct_vs_f16(rows, arm, n, "gen_energy_j_th40"), 2),
                    round(pct_vs_f16(rows, arm, n, "gen_energy_j_th55"), 2),
                    round(pct_vs_f16(rows, arm, n, "gen_energy_j_th60"), 2),
                    round(100.0 * (adiff - fdiff) / fdiff, 2) if arm != "f16" else 0.0,
                    note])


def report(rows):
    print()
    print("Q4_0 vs F16, same 45 runs, three windows (ours / MLPerf file):")
    for label, key in (("whole-process (counter)", "energy_counter_j"),
                       ("generation (0.5xmax)  ", "gen_energy_j_th50")):
        cells = []
        for n in LENGTHS:
            cells.append("%.1f / %.1f" % (pct_vs_f16(rows, "q4ours", n, key),
                                          pct_vs_f16(rows, "q4mlperf", n, key)))
        print("  %s  %s" % (label, "   ".join(cells)))
    def dmean(arm, n):
        return mval(rows, arm, n, "energy_counter_j")
    fd = dmean("f16", 576) - dmean("f16", 64)
    dec = ["%.1f" % (100.0 * ((dmean(a, 576) - dmean(a, 64)) - fd) / fd)
           for a in ("q4ours", "q4mlperf")]
    print("  decode-only (differenced)          %s / %s  (512-token basis)"
          % (dec[0], dec[1]))
    print()
    for n in LENGTHS:
        shares = [r["load_share_pct"] for r in rows if r["n_tokens"] == n]
        loads = [r["load_energy_j"] for r in rows if r["n_tokens"] == n]
        print("  load phase, n=%d: %.0f-%.0f J, %.0f-%.0f%% of the whole-process counter"
              % (n, min(loads), max(loads), min(shares), max(shares)))
    print()
    print("  threshold sensitivity (vs_fp16_pct, ours / mlperf):")
    for key, lab in (("gen_energy_j_th40", "0.40"), ("gen_energy_j_th50", "0.50"),
                     ("gen_energy_j_th55", "0.55"), ("gen_energy_j_th60", "0.60")):
        cells = []
        for n in LENGTHS:
            cells.append("%.1f/%.1f" % (pct_vs_f16(rows, "q4ours", n, key),
                                        pct_vs_f16(rows, "q4mlperf", n, key)))
        print("    frac=%s  %s" % (lab, "  ".join(cells)))


def main():
    traces_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACES
    if not os.path.isdir(traces_dir):
        raise SystemExit("traces dir not found: %s" % traces_dir)
    rows = analyse(traces_dir)
    check_against_published(rows)
    per_run = os.path.join(ROOT, "data", "rtx4090_llamacpp_window_comparison_2026-09-24.csv")
    summary = os.path.join(ROOT, "data",
                           "rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv")
    write_per_run(rows, per_run)
    write_summary(rows, summary)
    print("wrote %s (%d runs)" % (per_run, len(rows)))
    print("wrote %s" % summary)
    report(rows)


if __name__ == "__main__":
    main()
