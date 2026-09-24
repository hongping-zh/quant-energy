#!/usr/bin/env python3
"""Build data/index.json: a machine-readable index of every open data file.

The site publishes its measurement sessions as CSVs under data/ with the prose
context in data/README.md. This script adds the machine-readable layer: one
entry per file with its title, GPU, session date, measurement window, archive
DOI, and live file stats (rows, bytes, sha256) — so a script (or a reviewer)
can discover and compare datasets without parsing the README.

The measurement window is recorded per file because it is load-bearing: the
2026-09 window-unification re-analysis showed the choice of window alone moves
a short-output energy delta by ~13 points. Two rules from that analysis are
encoded here:

  * every entry states its window, and entries with different windows are not
    comparable to each other;
  * the bitsandbytes container rows are the container's *generation* window —
    NVML sampling starts after the model is loaded, quantized and warmed up
    (ecocompute-mlcube entrypoint.py, measure_once). Earlier prose on this
    site called these rows "whole-process"; the container source shows that
    label was wrong.

Fail loudly if a file appears under data/ without curated metadata, so the
index can never silently drift out of date.

    python3 build/make_data_index.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "https://quantenergy.tech"
LICENSE = "CC BY 4.0"

# Files that live under data/ but are not datasets (index/README themselves).
NON_DATA_FILES = {"README.md", "index.json"}

# ---------------------------------------------------------------------------
# Curated metadata. Every *.csv / *.json under data/ (recursively) must appear
# here exactly once, or the build fails. `path` is repo-relative.
# ---------------------------------------------------------------------------
FILES = [
    # --- derived roll-up -----------------------------------------------------
    {
        "path": "data/coverage_matrix_2026-09-09.csv",
        "title": "Coverage matrix: every measurement on one grid",
        "kind": "roll-up",
        "description": (
            "Not a session: a derived roll-up of every other file here plus "
            "build/measured.csv and data/replications/*.json, aggregated into "
            "(quantization method x GPU architecture) x model size. One row per "
            "non-empty cell with unweighted mean, full min-max range, measurement "
            "count, cards involved, and paired mean delta-perplexity where it exists."
        ),
        "gpu": None,
        "gpu_arch": None,
        "session_date": "2026-09-09",
        "measurement_window": "generation-only (all rows)",
        "window_note": (
            "Every cell is on the generation window (the file name keeps its original "
            "date): the bitsandbytes rows are the container window (sampling starts "
            "after load, quantization and warm-up), and since 2026-09-24 the llama.cpp "
            "row is re-cut from its archived 100 Hz power traces (see "
            "rtx4090_llamacpp_window_comparison_2026-09-24). The llama.cpp row is "
            "still listed separately because it is a different runtime and workload "
            "shape (1x576 tokens vs 10x256), not a different denominator."
        ),
        "doi": None,
        "regenerate": "python3 build/make_coverage_matrix.py",
    },
    # --- RTX 4090, llama.cpp window comparison --------------------------------
    {
        "path": "data/rtx4090_llamacpp_window_comparison_2026-09-24.csv",
        "title": "RTX 4090 llama.cpp session, re-cut over three windows (per run)",
        "kind": "session",
        "description": (
            "The 45 runs of the 2026-09-03 v2 session re-integrated over three "
            "measurement windows: whole-process (NVML hardware counter), generation "
            "window (counter energy allocated by the 100 Hz trace, cut where power "
            "leaves the ~70 W load plateau at 0.5x run-max, sustained), and the "
            "published decode-only differenced figure. Thresholds 0.40/0.55/0.60 "
            "carried per run. Built by build/make_window_comparison.py, which "
            "asserts the whole-process and decode-only columns reproduce the "
            "published summary before writing anything."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-09-24",
        "measurement_window": "three windows side by side (whole-process / generation / decode-only)",
        "window_note": (
            "The window choice is worth ~11-13 points at 64 output tokens "
            "(whole-process -57.8/-55.2 vs generation -68.6/-67.7) and ~2 points at "
            "576 (-61.0/-61.4 vs -62.9/-63.1). The excluded load phase is 193-478 J, "
            "57-78% of a 64-token run but 12-21% of a 576-token run. State the "
            "window AND the output length with every energy claim."
        ),
        "doi": "10.5281/zenodo.22295184",
        "companion_of": "data/rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv",
        "regenerate": "python3 build/make_window_comparison.py",
    },
    {
        "path": "data/rtx4090_llamacpp_window_comparison_2026-09-24.summary.csv",
        "title": "RTX 4090 llama.cpp session, window comparison summary",
        "kind": "summary",
        "description": (
            "Per arm and output length: mean energies per window and Q4_0-vs-F16 "
            "percentages for all of them - whole-process counter, generation window "
            "at thresholds 0.40/0.50/0.55/0.60, and the decode-only differenced "
            "figure. The 0.40 column is flagged as not meaningful for the 64-token "
            "Q4_0 arms (their ~150 W peak puts 0.4x max below the load plateau)."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-09-24",
        "measurement_window": "three windows side by side (whole-process / generation / decode-only)",
        "doi": "10.5281/zenodo.22295184",
    },
    # --- RTX 5090, 2026-09-19/20 ---------------------------------------------
    {
        "path": "data/rtx5090_bnb_2026-09-20.csv",
        "title": "RTX 5090 bitsandbytes session, per-run (NF4/INT8, 0.5-7B)",
        "kind": "session",
        "description": (
            "Five models from 0.5B to 7B on one RTX 5090 (Blackwell, 32 GB, driver "
            "580.76.05), NF4 and INT8 against a per-session FP16 baseline, run twice "
            "with a full instance restart between sessions A and B, so every cell is "
            "n=2 across restarts. 256 tokens x 10 iterations, 2 warm-ups, greedy, "
            "NVML at 10 Hz, ecocompute-energy/1.1 reports. The 1.1B model is "
            "TinyLlama-1.1B-intermediate-step-1431k-3T (not the Chat variant)."
        ),
        "gpu": "RTX 5090",
        "gpu_arch": "blackwell",
        "session_date": "2026-09-20",
        "measurement_window": "generation-only (container)",
        "window_note": (
            "NVML sampling starts after the model is loaded, quantized and warmed up; "
            "each of the 10 iterations re-runs the ~10-token prompt prefill; idle "
            "power is not subtracted. Earlier site prose called these rows "
            "whole-process; the container source shows load and quantization are "
            "outside the sampling window."
        ),
        "doi": "10.5281/zenodo.22855133",
        "companion_of": "data/rtx5090_bnb_2026-09-20.summary.csv",
        "regenerate": "python3 build/make_rtx5090_csv.py <unpacked_reports_dir> data",
    },
    {
        "path": "data/rtx5090_bnb_2026-09-20.summary.csv",
        "title": "RTX 5090 bitsandbytes session, A-vs-B summary",
        "kind": "summary",
        "description": (
            "Session A vs session B per cell: vs-FP16 percentages, absolute energies, "
            "throughputs and delta-perplexity. Across the ten cells |A-B| averages "
            "2.1 pp and peaks at 5.2 pp (INT8 3B); NF4 cells agree within 1.9 pp. "
            "That is the restart noise floor to attach to any single cell."
        ),
        "gpu": "RTX 5090",
        "gpu_arch": "blackwell",
        "session_date": "2026-09-20",
        "measurement_window": "generation-only (container)",
        "doi": "10.5281/zenodo.22855133",
    },
    {
        "path": "data/rtx5090_fp8_torchao_2026-09-20.csv",
        "title": "RTX 5090 FP8 (torchao) track, per-cell",
        "kind": "session",
        "description": (
            "Same card, different track: standalone bench_fp8_rtx5090.py with torchao "
            "0.18.0, weight-only and dynamic act+weight FP8, three samples per cell, "
            "own same-run FP16 baseline. Both FP8 paths cost energy at every size "
            "because they decode 2-6x slower than FP16. One 1.1B weight-only cell "
            "generated only 177 of 768 tokens (flagged short_generation)."
        ),
        "gpu": "RTX 5090",
        "gpu_arch": "blackwell",
        "session_date": "2026-09-20",
        "measurement_window": "generation-only (standalone script)",
        "window_note": (
            "Energy measured over generation only, with model load and quantize_() "
            "excluded and idle power not subtracted. Its percentages are comparable "
            "inside this file and nowhere else - including against the bitsandbytes "
            "rows of the same date, whose window shape differs (10x256 tokens with "
            "per-iteration prefill)."
        ),
        "doi": "10.5281/zenodo.22855133",
    },
    # --- RTX 4090, llama.cpp --------------------------------------------------
    {
        "path": "data/rtx4090_llamacpp_gguf_v2_2026-09-03.csv",
        "title": "RTX 4090 llama.cpp GGUF session v2, per-run (the one to quote)",
        "kind": "session",
        "description": (
            "One rented RTX 4090 (Ada, 450 W), llama.cpp b10643, Llama-3.1-8B-Instruct, "
            "three GGUF files (F16, Q4_0 ours, Q4_0 the MLPerf Client v2.0 file) x three "
            "output lengths (64/320/576) x five replicates = 45 runs, randomized order, "
            "forced cooldown to idle before every run. Energy is the NVML hardware "
            "energy counter over the whole llama-cli process; a 100 Hz power trace was "
            "recorded alongside as a cross-check."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-09-03",
        "measurement_window": "decode-only (differenced E(576)-E(64))",
        "window_note": (
            "The summary figures difference out the weight-load term; the per-run CSV "
            "also carries the undifferenced whole-process columns (process_energy_j), "
            "and the raw traces under data/llamacpp_v2_traces/ allow any window to be "
            "re-cut (see the window-comparison file). The load phase is 57-78% of a "
            "64-token run but only 12-21% of a 576-token run: window and output "
            "length must be stated together."
        ),
        "doi": "10.5281/zenodo.22295184",
        "companion_of": "data/rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv",
        "regenerate": "python3 build/make_llamacpp_v2_csv.py <archive_dir> data",
    },
    {
        "path": "data/rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv",
        "title": "RTX 4090 llama.cpp GGUF session v2, summary",
        "kind": "summary",
        "description": (
            "Per-arm decode-only energy, throughput, decode power, delta-vs-F16 and "
            "perplexity. Q4_0 saves 62-63% per token while drawing MORE power than F16 "
            "(296-319 W vs 273 W) - the saving is throughput, not lower power."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-09-03",
        "measurement_window": "decode-only (differenced E(576)-E(64))",
        "doi": "10.5281/zenodo.22295184",
    },
    {
        "path": "data/rtx4090_llamacpp_gguf_2026-08-31.csv",
        "title": "RTX 4090 llama.cpp GGUF first pass, per-run (superseded)",
        "kind": "session",
        "description": (
            "The provisional first pass: 18 runs, fixed F16-ours-MLPerf order, no "
            "cooldown (16 of 18 runs started on 145-248 W of tail from the previous "
            "run), unverified token counts. Superseded by the v2 session above and "
            "kept published so the correction is auditable."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-08-31",
        "measurement_window": "decode-only (differenced E(576)-E(64))",
        "superseded_by": "data/rtx4090_llamacpp_gguf_v2_2026-09-03.csv",
        "doi": None,
    },
    {
        "path": "data/rtx4090_llamacpp_gguf_2026-08-31.summary.csv",
        "title": "RTX 4090 llama.cpp GGUF first pass, summary (superseded)",
        "kind": "summary",
        "description": (
            "Per-arm summary of the provisional session; the v2 rerun moved the "
            "headline by about 1.7 points. Read as a record of the first pass, not "
            "as current numbers."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-08-31",
        "measurement_window": "decode-only (differenced E(576)-E(64))",
        "superseded_by": "data/rtx4090_llamacpp_gguf_v2_2026-09-03.summary.csv",
        "doi": None,
    },
    # --- RTX 4090, bitsandbytes container -------------------------------------
    {
        "path": "data/rtx4090_paired_energy_quality_2026-08-19.csv",
        "title": "RTX 4090 paired energy+quality session (5 sizes x NF4/INT8)",
        "kind": "session",
        "description": (
            "Ten measured configurations from one RTX 4090 session via "
            "ecocompute-mlcube (schema 1.1), each row carrying its own FP16 energy "
            "baseline AND teacher-forcing perplexity for both the quantized model and "
            "the baseline, from the same run. n=1 per configuration; no thermal block "
            "(added later in schema 1.2), so thermal state is unknown - do not "
            "difference against steady-thermal runs."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-08-19",
        "measurement_window": "generation-only (container)",
        "window_note": (
            "NVML sampling starts after load, quantization and warm-up; each of the "
            "10 iterations re-runs the ~10-token prompt prefill; idle not subtracted. "
            "Earlier site prose called this whole-process; the container source shows "
            "otherwise."
        ),
        "doi": "10.5281/zenodo.22037483",
    },
    {
        "path": "data/rtx4090_int8_repeats_2026-08-20.csv",
        "title": "RTX 4090 INT8 repeat session, per-run (n=3 per size)",
        "kind": "session",
        "description": (
            "The same INT8 configurations run three times each on the same instance "
            "the day after the paired session, to size run-to-run noise: CV of "
            "delta-E% is 0.6-3.9%, thirty to fifty times smaller than the "
            "cross-session gap. Absolute joules drifted 12-17% lower overnight while "
            "delta-E% held - compare FP16-normalised deltas, not absolute joules."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-08-20",
        "measurement_window": "generation-only (container)",
        "doi": "10.5281/zenodo.22037483",
        "companion_of": "data/rtx4090_int8_repeats_2026-08-20.summary.csv",
    },
    {
        "path": "data/rtx4090_int8_repeats_2026-08-20.summary.csv",
        "title": "RTX 4090 INT8 repeat session, summary (mean/SD/CV per size)",
        "kind": "summary",
        "description": (
            "Mean, SD and CV per size for INT8 delta-vs-FP16, absolute energy and the "
            "FP16 baseline. The noise floor the site attaches to any single-cell "
            "comparison on this protocol."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-08-20",
        "measurement_window": "generation-only (container)",
        "doi": "10.5281/zenodo.22037483",
    },
    # --- community replications ----------------------------------------------
    {
        "path": "data/replications/2026-08-23-gkgoing-rtx3050ti-tinyllama-nf4.energy.json",
        "title": "Community replication: RTX 3050 Ti laptop, TinyLlama-1.1B, NF4",
        "kind": "replication",
        "description": (
            "First external replication, contributed 2026-08-23: an unmodified "
            "ecocompute-mlcube energy.json from a laptop RTX 3050 Ti (Ampere, 60 W "
            "limit). Published as received; the report's own basis/measurement_source "
            "fields say what it is."
        ),
        "gpu": "RTX 3050 Ti (laptop)",
        "gpu_arch": "ampere",
        "session_date": "2026-08-23",
        "measurement_window": "generation-only (container, per its own report)",
        "doi": None,
    },
]

# Whole directories of same-shaped raw files, indexed as one entry each.
# Every file under these directories must match the declared pattern, and the
# completeness check below accepts exactly the files that exist there.
DIRS = [
    {
        "path": "data/llamacpp_v2_traces",
        "title": "RTX 4090 llama.cpp v2 session, raw per-run power traces",
        "kind": "raw-traces",
        "description": (
            "The 45 per-run JSONs the rented 4090 wrote on 2026-09-03: full 100 Hz "
            "power_trace_w / temp_trace_c for every run (F16, Q4_0 ours, Q4_0 the "
            "MLPerf Client v2.0 file x 64/320/576 tokens x 5 replicates), plus the "
            "NVML counter aggregate and wall time. The Zenodo record "
            "10.5281/zenodo.22295184 archived only the aggregate CSVs, so these "
            "traces are vendored here to make the window re-analysis "
            "(rtx4090_llamacpp_window_comparison_2026-09-24) reproducible from the "
            "repo alone."
        ),
        "gpu": "RTX 4090",
        "gpu_arch": "ada",
        "session_date": "2026-09-03",
        "measurement_window": "whole-process (full-run trace; re-cuttable to any window)",
        "doi": "10.5281/zenodo.22295184",
        "regenerate": None,
    },
]

# Site datasets that live outside data/ but belong in an open-data index.
RELATED = [
    {
        "path": "build/measured.csv",
        "title": "The curve-fitting dataset (2026-07 sessions)",
        "description": (
            "29 anchors, five cards: the FP16 absolute energies and quantization "
            "deltas the site's fitted crossover curves are built from. Not pooled "
            "with the later supplementary sessions - those are cited by their own "
            "DOIs."
        ),
        "url": SITE + "/build/measured.csv",
        "doi": "10.5281/zenodo.19647290",
    },
    {
        "path": "curves.json",
        "title": "Fitted crossover-curve parameters",
        "description": (
            "The dE%(N) = A - S*(x/(1+x)) parameters per architecture and precision, "
            "as used by the in-browser estimator (/estimate)."
        ),
        "url": SITE + "/curves.json",
        "doi": None,
    },
    {
        "path": "schema/quantization-energy-report-schema.yaml",
        "title": "Quantization energy report schema",
        "description": "The YAML schema for quantization-energy experiment reports.",
        "url": SITE + "/schema/quantization-energy-report-schema.yaml",
        "doi": None,
    },
]

ARCHIVES = [
    {
        "name": "EcoCompute measured dataset (main)",
        "concept_doi": "10.5281/zenodo.19647290",
        "version_doi": "10.5281/zenodo.19647290",
        "covers": "build/measured.csv and the fitted curves",
    },
    {
        "name": "RTX 4090 deep dive",
        "concept_doi": "10.5281/zenodo.22019741",
        "version_doi": "10.5281/zenodo.22037483",
        "covers": (
            "rtx4090_paired_energy_quality_2026-08-19 and "
            "rtx4090_int8_repeats_2026-08-20, with the raw energy.json reports"
        ),
    },
    {
        "name": "RTX 4090 llama.cpp GGUF v2 session",
        "concept_doi": "10.5281/zenodo.22295183",
        "version_doi": "10.5281/zenodo.22295184",
        "covers": (
            "rtx4090_llamacpp_gguf_v2_2026-09-03, with raw run records, scripts, "
            "environment snapshot and model hashes"
        ),
    },
    {
        "name": "RTX 5090 session (bnb + FP8 tracks)",
        "concept_doi": "10.5281/zenodo.22855132",
        "version_doi": "10.5281/zenodo.22855133",
        "covers": (
            "rtx5090_bnb_2026-09-20 and rtx5090_fp8_torchao_2026-09-20, with the "
            "raw reports, both scripts and the environment snapshot"
        ),
    },
]

NOTES = [
    (
        "Measurement windows differ across these files and the difference is not "
        "small: re-integrating one llama.cpp session over different windows moved "
        "the 64-token delta-vs-F16 by ~11-13 points while the 576-token figures "
        "agreed within ~2. Every energy claim should name its window AND its "
        "output length."
    ),
    (
        "The bitsandbytes container rows are the container's generation window: "
        "NVML sampling starts after the model is loaded, quantized and warmed up, "
        "and each iteration re-runs the prefill of a ~10-token prompt (idle power "
        "not subtracted). They were earlier described on this site as "
        "whole-process at a fixed output length; the container source "
        "(ecocompute-mlcube entrypoint.py, measure_once) shows that label was "
        "wrong. From container schema 1.3, runs can opt into a power-trace sidecar "
        "so the window can be re-cut post hoc."
    ),
    (
        "All figures are GPU-package joules from NVML software telemetry: no CPU, "
        "DRAM, PSU, wall AC power, PUE or CO2e. Nothing here is an MLPerf result "
        "or a certified benchmark."
    ),
]


def file_stats(path):
    with open(path, "rb") as f:
        raw = f.read()
    rows = 0
    if path.endswith(".csv"):
        # data rows = lines minus header, ignoring a trailing newline
        rows = max(0, raw.count(b"\n") - 1)
        if raw and not raw.endswith(b"\n"):
            rows += 1
    elif path.endswith(".json"):
        try:
            rows = 1  # a single JSON document
        except Exception:
            rows = 0
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows}


def index_entry(meta):
    path = meta["path"]
    full = os.path.join(ROOT, path.replace("/", os.sep))
    if not os.path.exists(full):
        raise SystemExit("indexed file missing on disk: %s" % path)
    entry = {
        "path": path,
        "url": "%s/%s" % (SITE, path),
        "title": meta["title"],
        "kind": meta["kind"],
        "description": meta["description"],
        "gpu": meta.get("gpu"),
        "gpu_arch": meta.get("gpu_arch"),
        "session_date": meta.get("session_date"),
        "measurement_window": meta.get("measurement_window"),
        "license": LICENSE,
    }
    if meta.get("window_note"):
        entry["window_note"] = meta["window_note"]
    if meta.get("doi"):
        entry["doi"] = meta["doi"]
        entry["doi_url"] = "https://doi.org/" + meta["doi"]
    if meta.get("companion_of"):
        entry["companion_of"] = meta["companion_of"]
    if meta.get("superseded_by"):
        entry["superseded_by"] = meta["superseded_by"]
    if meta.get("regenerate"):
        entry["regenerate"] = meta["regenerate"]
    entry.update(file_stats(full))
    return entry


def dir_stats(path):
    names = sorted(os.listdir(path))
    total = 0
    for name in names:
        with open(os.path.join(path, name), "rb") as f:
            total += len(f.read())
    return {"files": len(names), "bytes": total}


def dir_entry(meta):
    path = meta["path"]
    full = os.path.join(ROOT, path.replace("/", os.sep))
    if not os.path.isdir(full):
        raise SystemExit("indexed directory missing on disk: %s" % path)
    entry = {
        "path": path + "/",
        "url": "%s/%s/" % (SITE, path),
        "title": meta["title"],
        "kind": meta["kind"],
        "description": meta["description"],
        "gpu": meta.get("gpu"),
        "gpu_arch": meta.get("gpu_arch"),
        "session_date": meta.get("session_date"),
        "measurement_window": meta.get("measurement_window"),
        "license": LICENSE,
    }
    if meta.get("doi"):
        entry["doi"] = meta["doi"]
        entry["doi_url"] = "https://doi.org/" + meta["doi"]
    entry.update(dir_stats(full))
    return entry


def check_complete_coverage():
    """Fail if anything under data/ is not in the index (or allow-listed)."""
    indexed = {m["path"] for m in FILES}
    dir_covered = set()
    for dmeta in DIRS:
        dpath = os.path.join(ROOT, dmeta["path"].replace("/", os.sep))
        for name in os.listdir(dpath):
            dir_covered.add(
                os.path.relpath(os.path.join(dpath, name), ROOT).replace(os.sep, "/"))
    on_disk = set()
    base = os.path.join(ROOT, "data")
    for dirpath, _, names in os.walk(base):
        for name in names:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/")
            if os.path.basename(rel) in NON_DATA_FILES:
                continue
            on_disk.add(rel)
    missing = sorted(on_disk - indexed - dir_covered)
    stale = sorted(indexed - on_disk)
    if missing:
        raise SystemExit(
            "data/ contains files without index metadata (add them to FILES or DIRS "
            "in build/make_data_index.py):\n  " + "\n  ".join(missing))
    if stale:
        raise SystemExit(
            "index metadata references files that no longer exist:\n  "
            + "\n  ".join(stale))


def main():
    check_complete_coverage()
    import datetime
    index = {
        "schema": "quantenergy/data-index/1.0",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site": SITE,
        "license": LICENSE,
        "files": [index_entry(m) for m in FILES] + [dir_entry(m) for m in DIRS],
        "related": RELATED,
        "archives": ARCHIVES,
        "notes": NOTES,
    }
    out = os.path.join(ROOT, "data", "index.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")
    n_files = len(index["files"])
    n_bytes = sum(e["bytes"] for e in index["files"])
    print("wrote %s: %d files, %d related, %d archives, %.1f kB of data indexed"
          % (out, n_files, len(RELATED), len(ARCHIVES), n_bytes / 1024.0))


if __name__ == "__main__":
    main()
