#!/usr/bin/env python3
"""Turn the 2026-09-25 RTX 4090 Qwen2.5-3B NF4 re-test into the two published CSVs.

Input:  the unpacked trials bundle (``ecocompute/trials/trial-{A,B,C}/energy.json``
        plus the ``trial-X_gpu.txt`` nvidia-smi logs at the top level) — the same
        layout as ``trials-bundle-2026-09-25.tar.gz`` from the AutoDL instance.
Output: rtx4090_bnb_2026-09-25.csv          one row per trial
        rtx4090_bnb_2026-09-25.summary.csv  per-card and overall agreement

    python3 build/make_rtx4090_bnb_csv.py <unpacked_bundle_dir> data/

What this session is: the July curve-fitting dataset had this cell at +0.8% with
n=1, and a later paired-quality session on the same GPU class read -15.1%. This
re-test runs the cell three times on two physical cards (the rental instance
migrated hosts between the trial-A restart and trial B/C, so trial A is card
GPU-7c81f257 and trials B/C are card GPU-ccb89dd1) on the *current* stack — the
same torch 2.14.0 / CUDA 13.0 / bitsandbytes 0.50.2 versions as the RTX 5090
2026-09-20 session, deliberately, so the two cards' stack sensitivity can be
compared like for like.

Rules this script enforces, because they are what makes the session usable:

* **Only ``measurement_source == "direct-nvml"`` and ``basis == "measured"``
  rows survive** (same rule as make_rtx5090_csv.py: the container can fall back
  to interpolating the published dataset and writes that into the same block).
* **``vs_fp16_energy_pct`` is re-derived from the two energies and must match
  the report's own value at the printed precision**, and must equal the values
  quoted in the execution card (+3.5 / +0.7 / +0.5) — a transcription guard,
  not a scientific one.
* **The three quality probes must be identical** (same vendored corpus, greedy
  decode, same stack ⇒ perplexity is a pure software checksum; any drift means
  the arms are not comparable).
* **FP16 baselines across trials must agree within 2%** — they are the same
  model on the same card class; a bigger spread would mean the environment
  moved and the deltas are not about NF4.
* **Card identity comes from the nvidia-smi logs, not from the report** (the
  report has no UUID field); cards are numbered by first appearance in
  timestamp order (card-1 = trial A's host, card-2 = trial B/C's host).

The July pool (build/measured.csv) is deliberately NOT updated from this file:
single-version, no silent pooling. Folding these rows into any aggregate is a
separate, visible decision.
"""
import csv
import glob
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
DST = sys.argv[2] if len(sys.argv) > 2 else "."

# Transcription guard: the values quoted in the execution card and the session
# narrative. The script fails if the reports disagree with them.
EXPECTED_VS_FP16 = {"A": 3.5, "B": 0.7, "C": 0.5}

FIELDS = [
    "session", "model", "params_b", "precision", "energy_j_per_1k_tok", "vs_fp16_pct",
    "fp16_energy_j_per_1k_tok", "avg_power_w", "throughput_tok_s", "total_energy_j",
    "tokens_generated", "ppl", "fp16_ppl", "delta_ppl_pct", "batch_size", "context_length",
    "tokens_per_iteration", "iterations", "warmup", "sample_rate_hz", "gpu", "gpu_arch",
    "nvidia_driver", "torch", "torch_cuda", "transformers", "bitsandbytes",
    "basis", "measurement_source", "timestamp_utc",
    # Session-specific: two physical cards, cold starts, per-trial environment.
    "gpu_uuid", "gpu_uuid_short", "temperature_start_c",
]


def load_gpu_log(path):
    """Parse a trial-X_gpu.txt (nvidia-smi --query-gpu csv) into (uuid, temp)."""
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if len(lines) != 2:
        raise SystemExit(f"{path}: expected header + one data line, got {len(lines)}")
    uuid, _name, temp = [c.strip() for c in lines[1].split(",")]
    return uuid, int(temp)


def load_reports(src):
    out = []
    for path in sorted(glob.glob(os.path.join(src, "ecocompute", "trials", "trial-*", "energy.json"))):
        session = os.path.basename(os.path.dirname(path))[-1]  # trial-A -> A
        d = json.load(open(path))
        r = d.get("results", {})
        if d.get("measurement_source") != "direct-nvml" or r.get("basis") != "measured":
            continue
        gpu_txt = os.path.join(src, f"trial-{session}_gpu.txt")
        if not os.path.exists(gpu_txt):
            raise SystemExit(f"trial {session}: missing {gpu_txt} (card identity unknown)")
        uuid, temp = load_gpu_log(gpu_txt)
        out.append((session, uuid, temp, d))
    if not out:
        raise SystemExit(f"no direct-nvml reports under {src}")
    return out


def row_of(session, uuid, card_short, temp, d):
    w, m, s, r = d["workload"], d["measurement"], d["software"], d["results"]
    q = d.get("quality", {})
    pk = s.get("packages", {})
    # The container already did this division; re-derive to verify.
    e, f = float(r["energy_per_token_mj"]), float(r["fp16_energy_per_token_mj"])
    recomputed = round((e / f - 1.0) * 100.0, 1)
    if abs(recomputed - float(r["vs_fp16_energy_pct"])) > 0.05:
        raise SystemExit(
            f"trial {session}: report says vs_fp16={r['vs_fp16_energy_pct']}% but "
            f"{e}/{f} re-derives to {recomputed}%")
    if recomputed != EXPECTED_VS_FP16[session]:
        raise SystemExit(
            f"trial {session}: vs_fp16={recomputed}% but the execution card "
            f"quotes {EXPECTED_VS_FP16[session]}% — transcription drift, fix one of them")
    return {
        "session": session,
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
        "gpu_uuid": uuid,
        "gpu_uuid_short": card_short,
        "temperature_start_c": temp,
    }


def check_session(reports):
    """Same model, same stack, same quality probe, sane baselines, cold starts."""
    for session, _uuid, temp, d in reports:
        if d["workload"]["model_name"] != "Qwen/Qwen2.5-3B":
            raise SystemExit(f"trial {session}: unexpected model {d['workload']['model_name']}")
        if d["system_under_test"]["gpu"] != "NVIDIA GeForce RTX 4090":
            raise SystemExit(f"trial {session}: unexpected GPU {d['system_under_test']['gpu']}")
        if temp > 40:
            raise SystemExit(f"trial {session}: started at {temp} C (> 40), not a cold start")
    q0 = reports[0][3]["quality"]
    for session, _uuid, _temp, d in reports:
        q = d["quality"]
        for key in ("value", "fp16_value", "delta_vs_fp16_pct", "tokens_evaluated"):
            if q[key] != q0[key]:
                raise SystemExit(
                    f"trial {session}: quality probe drifted on {key} "
                    f"({q[key]} != {q0[key]}) — arms not comparable")
    bases = [float(d["results"]["fp16_energy_per_token_mj"]) for _s, _u, _t, d in reports]
    spread = (max(bases) - min(bases)) / min(bases) * 100.0
    if spread > 2.0:
        raise SystemExit(f"FP16 baselines disagree by {spread:.2f}% across trials")
    stacks = {json.dumps(d["software"]["packages"], sort_keys=True) for _s, _u, _t, d in reports}
    if len(stacks) != 1:
        raise SystemExit("trials ran on different software stacks — not one session")
    return spread


def main():
    reports = load_reports(SRC)
    reports.sort(key=lambda t: t[3]["timestamp_utc"])
    # Card numbering by first appearance in time order (A's host = card-1).
    short_of, seen = {}, 0
    for _s, uuid, _t, _d in reports:
        if uuid not in short_of:
            seen += 1
            short_of[uuid] = f"card-{seen}"
    rows = [row_of(s, u, short_of[u], t, d) for s, u, t, d in reports]
    spread = check_session(reports)
    rows.sort(key=lambda r: r["session"])

    out = os.path.join(DST, "rtx4090_bnb_2026-09-25.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    cards = sorted(set(r["gpu_uuid_short"] for r in rows))
    print(f"{out}: {len(rows)} trials on {len(cards)} physical cards "
          f"({', '.join(cards)}), FP16 baseline spread {spread:.2f}%")

    # Per-card and overall agreement. The design question this answers: how big
    # is same-card session noise (B vs C on card-2) next to the card-to-card
    # gap (A vs B)? An order of magnitude apart is the expected reading.
    sfields = ["group", "n_trials", "trials", "mean_vs_fp16_pct", "min_vs_fp16_pct",
               "max_vs_fp16_pct", "range_pp", "mean_energy_j_per_1k_tok",
               "mean_fp16_energy_j_per_1k_tok", "mean_avg_power_w",
               "mean_throughput_tok_s"]
    srows = []
    for group in cards + ["all"]:
        grp = rows if group == "all" else [r for r in rows if r["gpu_uuid_short"] == group]
        pcts = [float(r["vs_fp16_pct"]) for r in grp]
        srows.append({
            "group": group,
            "n_trials": len(grp),
            "trials": "+".join(r["session"] for r in grp),
            "mean_vs_fp16_pct": round(sum(pcts) / len(pcts), 2),
            "min_vs_fp16_pct": min(pcts),
            "max_vs_fp16_pct": max(pcts),
            "range_pp": round(max(pcts) - min(pcts), 2),
            "mean_energy_j_per_1k_tok": round(
                sum(float(r["energy_j_per_1k_tok"]) for r in grp) / len(grp), 3),
            "mean_fp16_energy_j_per_1k_tok": round(
                sum(float(r["fp16_energy_j_per_1k_tok"]) for r in grp) / len(grp), 3),
            "mean_avg_power_w": round(
                sum(float(r["avg_power_w"]) for r in grp) / len(grp), 1),
            "mean_throughput_tok_s": round(
                sum(float(r["throughput_tok_s"]) for r in grp) / len(grp), 1),
        })
    sout = os.path.join(DST, "rtx4090_bnb_2026-09-25.summary.csv")
    with open(sout, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sfields)
        w.writeheader()
        w.writerows(srows)
    by_group = {r["group"]: r for r in srows}
    same_card = by_group["card-2"]["range_pp"]
    cross = abs(float(rows[0]["vs_fp16_pct"]) - float(rows[1]["vs_fp16_pct"]))
    print(f"{sout}: same-card session range {same_card} pp vs card-to-card gap "
          f"{round(cross, 2)} pp (A vs B)")


if __name__ == "__main__":
    main()
