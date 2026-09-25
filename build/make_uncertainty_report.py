#!/usr/bin/env python3
"""Per-configuration uncertainty report (v1.0-bar skeleton, threshold #5).

Collects every configuration that has more than one independent measurement of
its delta-vs-FP16, from the versioned session archives, and reports cross-session
/ cross-card spread per configuration. Where a configuration has no repeats, the
row is emitted with a 'pending n>=3' placeholder — numbers are never invented.

The v1.0 micro-standard bar asks for published uncertainty at three levels
(session / card / laboratory). This report covers the first two; the
cross-laboratory level stays empty until independent replications exist.

Inputs (fixed manifest, like make_coverage_matrix.py):
  build/measured.csv                              29 anchors (n_trials column)
  data/rtx4090_int8_repeats_2026-08-20.csv        3 within-session replicates per INT8 size
  data/rtx5090_bnb_2026-09-20.csv                 sessions A/B across a restart (1 card)
  data/rtx4090_bnb_2026-09-25.csv                 sessions A/B/C across 2 physical cards

Output:
  data/uncertainty_report_2026-09-25.csv

Guards: the 2026-09-25 NF4 re-test row must reproduce the published summary
(mean +1.6, range 0.5..3.5, 2 cards); the 5090 A/B agreement must match the
published '2.1 points on average'.
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "uncertainty_report_2026-09-25.csv")

# --- inputs ---------------------------------------------------------------
MEASURED = os.path.join(HERE, "build", "measured.csv")
INT8_REPEATS = os.path.join(HERE, "data", "rtx4090_int8_repeats_2026-08-20.csv")
RTX5090 = os.path.join(HERE, "data", "rtx5090_bnb_2026-09-20.csv")
RTX4090_NF4 = os.path.join(HERE, "data", "rtx4090_bnb_2026-09-25.csv")


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stats(vals):
    return {
        "n": len(vals),
        "mean_pct": sum(vals) / len(vals),
        "min_pct": min(vals),
        "max_pct": max(vals),
        "spread_pp": max(vals) - min(vals),
    }


def fmt(x):
    return f"{x:.1f}"


def main():
    rows = []

    # --- layer 1: the 29 anchors ------------------------------------------
    # Cross-session data exists only for the configurations re-measured in the
    # supplementary archives below; everything else is an honest placeholder.
    repeated = {}  # (gpu, model, precision) -> row dict, filled below

    # --- supplementary archive 1: 2026-08-20 INT8 repeats (1 card, 1 session, 3 replicates)
    by_cfg = defaultdict(list)
    for r in read_csv(INT8_REPEATS):
        by_cfg[(r["model"], r["params_b"], r["precision"])].append(float(r["vs_fp16_energy_pct"]))
    for (model, params_b, prec), vals in sorted(by_cfg.items()):
        s = stats(vals)
        rows.append({
            "gpu": "RTX 4090", "model": model, "params_b": params_b, "precision": prec,
            "n_sessions": 1, "n_cards": 1, "scope": "within-session replicates (same day, same stack)",
            **{k: fmt(s[k]) if k != "n" else s[k] for k in ("mean_pct", "min_pct", "max_pct", "spread_pp")},
            "v1_0_level": "pending: 1 session (need >=3 sessions or >=2 cards)",
            "source": "rtx4090_int8_repeats_2026-08-20.csv",
        })
        repeated[("RTX 4090", model, prec)] = True

    # --- supplementary archive 2: 2026-09-20 5090 re-test (2 sessions across restart, 1 card)
    by_cfg = defaultdict(list)
    for r in read_csv(RTX5090):
        by_cfg[(r["model"], r["params_b"], r["precision"])].append(float(r["vs_fp16_pct"]))
    devs = []
    for (model, params_b, prec), vals in sorted(by_cfg.items()):
        s = stats(vals)
        rows.append({
            "gpu": "RTX 5090", "model": model, "params_b": params_b, "precision": prec,
            "n_sessions": s["n"], "n_cards": 1, "scope": "across-restart sessions (same card, current stack)",
            "mean_pct": fmt(s["mean_pct"]), "min_pct": fmt(s["min_pct"]),
            "max_pct": fmt(s["max_pct"]), "spread_pp": fmt(s["spread_pp"]),
            "v1_0_level": "pending: 2 sessions, 1 card (need >=3 sessions or >=2 cards)",
            "source": "rtx5090_bnb_2026-09-20.csv",
        })
        devs.append(s["spread_pp"])
    if devs:
        avg_dev = sum(devs) / len(devs)
        assert 1.5 <= avg_dev <= 2.7, f"5090 A/B agreement {avg_dev:.2f} != published ~2.1 avg"

    # --- supplementary archive 3: 2026-09-25 NF4 two-card re-test (3 sessions, 2 cards)
    by_cfg = defaultdict(list)
    cards = defaultdict(set)
    for r in read_csv(RTX4090_NF4):
        by_cfg[(r["model"], r["params_b"], r["precision"])].append(float(r["vs_fp16_pct"]))
        cards[(r["model"], r["params_b"], r["precision"])].add(r.get("gpu_uuid_short", "card-?"))
    for (model, params_b, prec), vals in sorted(by_cfg.items()):
        s = stats(vals)
        ncards = len(cards[(model, params_b, prec)])
        rows.append({
            "gpu": "RTX 4090", "model": model, "params_b": params_b, "precision": prec,
            "n_sessions": s["n"], "n_cards": ncards,
            "scope": "across-cards sessions (2 physical cards, current stack)",
            "mean_pct": fmt(s["mean_pct"]), "min_pct": fmt(s["min_pct"]),
            "max_pct": fmt(s["max_pct"]), "spread_pp": fmt(s["spread_pp"]),
            "v1_0_level": ("met: n=3 sessions across 2 cards (first configuration at the v1.0 bar)"
                           if s["n"] >= 3 and ncards >= 2 else
                           f"pending: {s['n']} sessions, {ncards} cards"),
            "source": "rtx4090_bnb_2026-09-25.csv",
        })
        if model == "Qwen/Qwen2.5-3B" and prec == "NF4":
            # guard: must reproduce the published summary exactly
            assert s["n"] == 3 and ncards == 2, "NF4 re-test must be n=3 across 2 cards"
            assert abs(s["mean_pct"] - 1.57) < 0.05, f"NF4 mean {s['mean_pct']:.2f} != published 1.6"
            assert abs(s["spread_pp"] - 3.0) < 0.05, f"NF4 spread {s['spread_pp']:.2f} != published 3.0"

    # --- layer 2: every remaining anchor as an explicit placeholder ---------
    for r in read_csv(MEASURED):
        key = (r["gpu"], r["model"], r["precision"])
        if key in repeated:
            continue
        repeated[key] = True
        n = int(r.get("n_trials") or 1)
        if n >= 2:
            scope = "within-session repeated trials (main dataset v1.1.0, CV<2%)"
            level = "pending: n=2 same-session (need >=3 sessions or >=2 cards)"
        else:
            scope = "single trial, no repeats"
            level = "pending: n=1 (need >=3 sessions or >=2 cards)"
        rows.append({
            "gpu": r["gpu"], "model": r["model"], "params_b": r["params_b"], "precision": r["precision"],
            "n_sessions": n, "n_cards": 1, "scope": scope,
            "mean_pct": "", "min_pct": "", "max_pct": "", "spread_pp": "",
            "v1_0_level": level,
            "source": "build/measured.csv (v1.1.0 anchor)",
        })

    fieldnames = ["gpu", "model", "params_b", "precision", "n_sessions", "n_cards", "scope",
                  "mean_pct", "min_pct", "max_pct", "spread_pp", "v1_0_level", "source"]
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    met = sum(1 for r in rows if r["v1_0_level"].startswith("met:"))
    pending = sum(1 for r in rows if r["v1_0_level"].startswith("pending:"))
    print(f"wrote {OUT}")
    print(f"  {len(rows)} configurations: {met} at the v1.0 bar, {pending} pending")
    print(f"  cross-laboratory level: no rows yet (requires independent replications)")


if __name__ == "__main__":
    sys.exit(main())
