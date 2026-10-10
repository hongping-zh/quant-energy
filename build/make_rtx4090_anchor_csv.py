#!/usr/bin/env python3
"""Turn the 2026-09-30 RTX 4090 anchor session into the two published CSVs.

Input:  the unpacked anchor bundle (``results-anchor-{t1,t2}/<Model>/<PREC>/energy.json``
        plus ``env-archive/nvidia-smi-q.txt``) — the layout of
        ``anchor-4090-20260930.tar.gz`` (sha256 ae875a99…640ab).
Output: data/rtx4090_anchor_2026-09-30.csv          one row per run (12)
        data/rtx4090_anchor_2026-09-30.summary.csv  per cell: two trials, how far apart

    python3 build/make_rtx4090_anchor_csv.py <unpacked_bundle_dir> data/

What this session is: four (model x precision) cells that the site already had
anchors for, re-measured on **the same software stack those anchors were measured
on** — torch 2.4.1+cu121 / transformers 4.46.3 / bitsandbytes 0.45.5 / accelerate
1.0.1 / Python 3.8, matching the 2026-07-24 RTX 4090 session whose rows are the
Ada Lovelace anchors in build/measured.csv. Two trials per cell, same card, same
day, September 2026.

Read that carefully, because it is easy to get wrong: every report here carries
``matches_reference_pins: false`` and a ``comparability_note`` saying the numbers
may not be comparable with image runs. Those flags compare this run against the
*container image's default pins* (bitsandbytes 0.43.3), which is what the
2026-08-19/20 sessions used — not against the July rows the curves are fitted on.
Measured against the rows it actually reproduces, this session is a same-stack
repeat, and that is what makes it useful: it is the first cross-session evidence
those four cells have. Against the July anchors (nf4/int8, same versions):
NF4 1.5B +15.11% -> +22.6% and INT8 1.5B +180.73% -> +197.05% move 7.5 and 16.3
points, while NF4 7B −28.46% -> −28.25% and INT8 7B +49.54% -> +50.75% land
within 1.2 points — the small-model cells are where the n=1 risk actually sits.

Rules this script enforces, because they are what makes the session usable:

* **Only ``measurement_source == "direct-nvml"`` and ``basis == "measured"`` rows
  survive** — same rule as make_rtx4090_bnb_csv.py and make_rtx5090_csv.py. The
  container can fall back to interpolating the published dataset and writes that
  into the same block.
* **``vs_fp16_energy_pct`` is re-derived** from the two energies and must match the
  report's own value at the printed precision, then match the values quoted in the
  session notes. Both are transcription guards: if either disagrees, stop.
* **The quality probe must be identical within a model across trials** — same
  vendored corpus, same tokenizer, greedy decode ⇒ perplexity is a software
  checksum; drift means the two trials' arms are not comparable.
* **The FP16 baseline must agree between trials within 2%** (measured here: 0.26%
  at 1.5B, 0.50% at 7B). They are the same model on the same card in one session;
  a bigger spread would mean the environment moved and the delta is not about
  quantization.
* **One software stack across all twelve runs**, checked rather than assumed.

Two honest limits, recorded rather than papered over:

* **Card identity and start temperature come from a single environment snapshot**
  (``env-archive/nvidia-smi-q.txt``), not per-run log lines. They say "this session
  ran on card GPU-a99e7871 at 36 C when sampled"; they are NOT per-run cold-start
  evidence, and they cannot prove trials t1 and t2 ran on the same physical card.
* **No arm-order / window / decoding / thermal records exist in these reports** —
  fields the 2026-09-27 validator hardening requires for *protocol conformance*
  (§4.4.1, §4.6.1, §4.6.2). These twelve reports are therefore **schema-valid but
  not protocol-conformant**, the same status the container's own reports carry and
  the same status published for the 2026-09-25 two-card re-test. The arm order is
  recoverable from file timestamps and is disclosed below as fixed, not randomized.

The coverage-matrix roll-up and the estimator's anchor pool (build/measured.csv)
are deliberately NOT touched by this script: folding these rows into either is a
separate, visible decision, exactly as for the 2026-09-25 file.
"""
import csv
import glob
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
DST = sys.argv[2] if len(sys.argv) > 2 else "."

# Transcription guard: the values quoted in the session notes for this run.
# {(params_b, precision): {trial -> vs_fp16_pct}}
EXPECTED_VS_FP16 = {
    (1.5, "NF4"): {"A": 22.3, "B": 22.9},
    (1.5, "INT8"): {"A": 199.3, "B": 194.8},
    (7.0, "NF4"): {"A": -28.2, "B": -28.3},
    (7.0, "INT8"): {"A": 50.1, "B": 51.4},
}
EXPECTED_MODELS = {"Qwen/Qwen2-1.5B": 1.5, "Qwen/Qwen2-7B": 7.0}
MAX_BASELINE_SPREAD_PCT = 2.0

TRIAL_OF_DIR = {"results-anchor-t1": "A", "results-anchor-t2": "B"}

FIELDS = [
    "session", "model", "params_b", "precision", "energy_j_per_1k_tok", "vs_fp16_pct",
    "fp16_energy_j_per_1k_tok", "avg_power_w", "throughput_tok_s", "total_energy_j",
    "tokens_generated", "ppl", "fp16_ppl", "delta_ppl_pct", "batch_size", "context_length",
    "tokens_per_iteration", "iterations", "warmup", "sample_rate_hz", "gpu", "gpu_arch",
    "nvidia_driver", "torch", "torch_cuda", "transformers", "bitsandbytes",
    "basis", "measurement_source", "timestamp_utc",
    # Session-specific: single card seen once, no per-run start temperature,
    # and a fixed (not randomized) arm order recovered from file mtimes.
    "gpu_uuid", "gpu_uuid_short", "temperature_start_c", "card_evidence",
    "arm_order_in_model", "note",
]


def load_env_snapshot(path):
    """Pass ``env-archive/nvidia-smi-q.txt`` into (uuid, temp_c). One snapshot for the session."""
    uuid = temp = None
    with open(path) as fh:
        for line in fh:
            if "GPU UUID" in line:
                uuid = line.split(":", 1)[1].strip()
            elif "GPU Current Temp" in line:
                temp = int(line.split(":", 1)[1].strip().split()[0])
    if uuid is None:
        raise SystemExit(f"{path}: no GPU UUID found — card identity unknown")
    return uuid, temp


def load_reports(src):
    out = []
    for tdir, session in TRIAL_OF_DIR.items():
        for path in sorted(glob.glob(os.path.join(src, tdir, "Qwen*", "*", "energy.json"))):
            d = json.load(open(path))
            r = d.get("results", {})
            if d.get("measurement_source") != "direct-nvml" or r.get("basis") != "measured":
                continue
            out.append((session, d, path))
    if not out:
        raise SystemExit(f"no direct-nvml reports under {src}")
    return out


def row_of(session, d, path, uuid, temp, arm_positions):
    w, m, s, r = d["workload"], d["measurement"], d["software"], d["results"]
    q = d.get("quality", {})
    pk = s.get("packages", {})
    prec, pb = w["precision"], w["params_b"]
    model_name = w["model_name"]

    # The container already did this division; re-derive to verify, then check
    # against the value quoted in the session notes.
    vs = r.get("vs_fp16_energy_pct")
    fp16_e = r.get("fp16_energy_per_token_mj")
    if prec == "FP16":
        recomputed = None
        if vs is not None or fp16_e is not None:
            raise SystemExit(f"{model_name} {prec} ({session}): FP16 arm should carry no delta")
    else:
        if fp16_e is None or vs is None:
            raise SystemExit(f"{model_name} {prec} ({session}): missing FP16 baseline or delta")
        recomputed = round((float(r["energy_per_token_mj"]) / float(fp16_e) - 1.0) * 100.0, 1)
        if abs(recomputed - float(vs)) > 0.05:
            raise SystemExit(f"{model_name} {prec} ({session}): report says vs_fp16={vs}% "
                             f"but the two energies re-derive to {recomputed}%")
        expected = EXPECTED_VS_FP16.get((pb, prec), {}).get(session)
        if expected is None:
            raise SystemExit(f"{model_name} {prec} ({session}): unexpected cell for this session")
        if recomputed != expected:
            raise SystemExit(f"{model_name} {prec} ({session}): vs_fp16={recomputed}% but the "
                             f"session notes quote {expected}% — transcription drift, fix one")

    # matches_reference_pins compares this run with the container image's default
    # pins (bnb 0.43.3, what the 2026-08 sessions ran) — not with the July rows it
    # reproduces. Say so, or the flag reads as "not comparable with anything here".
    note = ("Same software stack as the 2026-07-24 RTX 4090 anchor session (bnb 0.45.5 / "
            "tf 4.46.3 / torch 2.4.1) — a same-stack repeat, despite the report's "
            "matches_reference_pins: false against the image's own default pins "
            "(0.43.3).") if prec != "FP16" else (
            "FP16 baseline arm of the same model; no delta is defined for it.")
    if s.get("matches_reference_pins") is False:
        note += " Report has no window/decoding/arm-order records: schema-valid, not protocol-conformant."

    return {
        "session": session,
        "model": model_name.split("/")[-1],
        "params_b": pb,
        "precision": prec,
        "energy_j_per_1k_tok": r["energy_per_token_mj"],
        "vs_fp16_pct": vs if vs is not None else "",
        "fp16_energy_j_per_1k_tok": fp16_e if fp16_e is not None else "",
        "avg_power_w": r.get("avg_power_watts"),
        "throughput_tok_s": r.get("throughput_tokens_per_s"),
        "total_energy_j": r.get("total_energy_joules"),
        "tokens_generated": r.get("tokens_generated"),
        "ppl": q.get("value", ""),
        "fp16_ppl": q.get("fp16_value", ""),
        "delta_ppl_pct": q.get("delta_vs_fp16_pct", ""),
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
        "gpu_uuid": uuid,
        "gpu_uuid_short": "card-1",
        "temperature_start_c": temp if temp is not None else "",
        "card_evidence": "session-snapshot",
        "arm_order_in_model": arm_positions.get((session, model_name, prec), ""),
        "note": note,
    }


def arm_order_index(reports):
    """Recover the order the arms ran in, from report timestamps (earliest = 1).

    The reports do not record arm order; the timestamps do. Recording it as a
    derived fact is the honest substitute until the container emits the field.
    """
    per_trial = {}
    for session, d, _p in reports:
        key = (session, d["workload"]["model_name"])
        per_trial.setdefault(key, []).append((d["timestamp_utc"], d["workload"]["precision"]))
    positions = {}
    for (session, model), items in per_trial.items():
        for pos, (_ts, prec) in enumerate(sorted(items), start=1):
            positions[(session, model, prec)] = pos
    # The order must be the same FP16 -> NF4 -> INT8 sweep in every trial and model,
    # or the "fixed order" disclosure below would be wrong.
    orders = set()
    for (session, model), items in per_trial.items():
        orders.add(tuple(p for _ts, p in sorted(items)))
    if len(orders) != 1:
        raise SystemExit(f"arm orders differ across trials/models: {orders} — disclose each")
    order = next(iter(orders))
    return positions, "->".join(order) + " (fixed, not randomized)"


def check_session(reports, rows):
    for session, d, _p in reports:
        if d["workload"]["model_name"] not in EXPECTED_MODELS:
            raise SystemExit(f"trial {session}: unexpected model {d['workload']['model_name']}")
        if d["system_under_test"]["gpu"] != "NVIDIA GeForce RTX 4090":
            raise SystemExit(f"trial {session}: unexpected GPU {d['system_under_test']['gpu']}")
        if EXPECTED_MODELS[d["workload"]["model_name"]] != d["workload"]["params_b"]:
            raise SystemExit(f"trial {session}: params_b disagrees with the model name")
    # Quality probe identical within a model across trials.
    for _session, model in sorted({(s, d["workload"]["model_name"]) for s, d, _p in reports}, key=repr):
        pass
    by_model = {}
    for row in rows:
        by_model.setdefault((row["model"], row["precision"]), []).append(row)
    for (_model, prec), group in by_model.items():
        if prec == "FP16" or len(group) < 2:
            continue
        keys = ("ppl", "fp16_ppl", "delta_ppl_pct")
        base = group[0]
        for row in group[1:]:
            for k in keys:
                if row[k] != base[k]:
                    raise SystemExit(f"{row['model']} {prec}: quality probe drifted on {k} "
                                     f"({row[k]} != {base[k]}) — trials not comparable")
    # FP16 baseline agreement between trials, per model.
    spreads = {}
    for model in sorted({r["model"] for r in rows}):
        bases = [float(r["energy_j_per_1k_tok"]) for r in rows
                 if r["model"] == model and r["precision"] == "FP16"]
        if len(bases) < 2:
            continue
        spread = (max(bases) - min(bases)) / min(bases) * 100.0
        spreads[model] = spread
        if spread > MAX_BASELINE_SPREAD_PCT:
            raise SystemExit(f"{model}: FP16 baselines disagree by {spread:.2f}% across trials")
    stacks = {json.dumps(d["software"]["packages"], sort_keys=True) for _s, d, _p in reports}
    if len(stacks) != 1:
        raise SystemExit("runs used different software stacks — not one session")
    return spreads


def main():
    reports = load_reports(SRC)
    snapshot = os.path.join(SRC, "env-archive", "nvidia-smi-q.txt")
    if not os.path.exists(snapshot):
        raise SystemExit(f"missing {snapshot} (no card evidence for this session)")
    uuid, temp = load_env_snapshot(snapshot)
    positions, order_text = arm_order_index(reports)
    rows = [row_of(s, d, p, uuid, temp, positions) for s, d, p in reports]
    spreads = check_session(reports, rows)
    rows.sort(key=lambda r: (float(r["params_b"]), r["precision"], r["session"]))

    out = os.path.join(DST, "rtx4090_anchor_2026-09-30.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} runs, arm order {order_text}; "
          f"FP16 trial-to-trial spread " +
          ", ".join(f"{m} {s:.2f}%" for m, s in sorted(spreads.items())))

    # Repeatability first: two trials of the same cell, same card, same stack.
    # Then the cross-stack question: where does this third stack land next to the
    # published min-max range for the same cell? The latter is printed, not asserted —
    # whether this stack is "inside the range" is a reading, not a machine check.
    sfields = ["cell", "model", "params_b", "precision", "n_trials", "trials",
               "mean_vs_fp16_pct", "min_vs_fp16_pct", "max_vs_fp16_pct", "range_pp",
               "mean_energy_j_per_1k_tok", "mean_fp16_energy_j_per_1k_tok",
               "mean_avg_power_w", "mean_throughput_tok_s", "delta_ppl_pct"]
    cells = {}
    for r in rows:
        if r["precision"] == "FP16":
            continue
        cells.setdefault((r["model"], r["params_b"], r["precision"]), []).append(r)
    srows = []
    for (model, pb, prec), group in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        pcts = [float(r["vs_fp16_pct"]) for r in group]
        srows.append({
            "cell": f"{model} {prec}",
            "model": model,
            "params_b": pb,
            "precision": prec,
            "n_trials": len(group),
            "trials": "+".join(r["session"] for r in group),
            "mean_vs_fp16_pct": round(sum(pcts) / len(pcts), 2),
            "min_vs_fp16_pct": min(pcts),
            "max_vs_fp16_pct": max(pcts),
            "range_pp": round(max(pcts) - min(pcts), 2),
            "mean_energy_j_per_1k_tok": round(
                sum(float(r["energy_j_per_1k_tok"]) for r in group) / len(group), 3),
            "mean_fp16_energy_j_per_1k_tok": round(
                sum(float(r["fp16_energy_j_per_1k_tok"]) for r in group) / len(group), 3),
            "mean_avg_power_w": round(sum(float(r["avg_power_w"]) for r in group) / len(group), 1),
            "mean_throughput_tok_s": round(
                sum(float(r["throughput_tok_s"]) for r in group) / len(group), 1),
            "delta_ppl_pct": group[0]["delta_ppl_pct"],
        })
    sout = os.path.join(DST, "rtx4090_anchor_2026-09-30.summary.csv")
    with open(sout, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sfields)
        w.writeheader()
        w.writerows(srows)
    worst = max(srows, key=lambda r: r["range_pp"])
    print(f"{sout}: {len(srows)} cells; worst trial-to-trial range {worst['range_pp']} pp "
          f"({worst['cell']})")


if __name__ == "__main__":
    main()
