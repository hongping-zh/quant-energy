# Measured sessions published alongside this site

## `rtx4090_llamacpp_gguf_v2_2026-09-03.csv` (+ `.summary.csv`)

The clean rerun of the session below, and the one to quote. Same card, same build, same three GGUF
files; the protocol is what changed. One rented RTX 4090 (Ada, 450 W limit, driver 550.120), one
llama.cpp build (`b10643-192067b72`, CUDA 12.4), Llama-3.1-8B-Instruct, three GGUF files × three
output lengths (64, 320, 576 tokens) × five replicates = **45 runs**, batch 1,
`-ngl 99 -fa 1 --no-mmap -c 2048 --temp 0 --seed 1234 --ignore-eos`, executed in randomized order,
each preceded by a forced cooldown to idle (all 45 converged, starting at ~25 W and 27–29 °C), all
45 exiting 0. Energy is the **NVML hardware energy counter** over the whole `llama-cli` process; a
100 Hz power trace is recorded and integrated alongside it as a cross-check only. Regenerate with
`python3 build/make_llamacpp_v2_csv.py <archive_dir> data`.

| llama.cpp arm | mJ/token | tokens/J | tok/s | decode power | Δenergy vs F16 | perplexity | Δppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| `F16` | 4980 | 0.201 | 54.7 | 273 W | — | 7.3260 | — |
| `Q4_0`, ours | 1899 | 0.527 | 155.9 | 296 W | −61.9 % | 7.7364 | +5.60 % |
| `Q4_0`, the v2.0 file | 1834 | 0.545 | 173.8 | 319 W | −63.2 % | 7.7366 | +5.61 % |

### Read this before using the numbers

- **Decode-only, by differencing `E(576) − E(64)`, and the third length exists to check that this is
  allowed.** Fitting `E = a + b·n` over all 15 runs of an arm gives R² ≥ 0.9966 with intercepts of
  171–378 J, i.e. the weight-load term differencing is meant to remove; the fitted slope
  (`energy_mj_per_token_ols`) agrees with the differenced value to the printed digit.
- **The quantized arms draw *more* power, not less.** F16 decodes at 273 W, the `Q4_0` arms at
  296–319 W, and they are still 62–63 % cheaper per token because they emit 2.8–3.2× the tokens per
  second. The provisional session below reported all three arms at 296–310 W; with cooldowns and
  randomized order that conclusion inverts.
- **Token counts are inferred, not read back.** llama.cpp's timing line did not parse in the wrapper,
  so `decoded_tokens` is the requested count, credited whenever `--ignore-eos` was set and the
  process exited 0. Sound, given the flag, but an assumption.
- **The counter and the trace disagree by about 15 % per run** (`process_energy_j` against
  `process_energy_trapezoid_j`; range 2.5–31.8 %). The counter integrates in hardware and is what the
  summary reports. After differencing, which cancels the common part, the two accountings agree to
  0.2–4.3 % (`counter_vs_trapezoid_pct`) — relevant to anyone designing an at-home energy tier out of
  sampled telemetry.
- **Perplexity was not re-measured**; the values are carried over from the 2026-08-31 session on the
  same files and build, and `perplexity_session` in the summary says so.
- **The two `Q4_0` arms remain a zero result.** They differ by 3.5 % in energy (1.8 % without a single
  high 64-token run in the shipped-file arm) against a within-cell SD of 1.3–1.8 %, and by 0.0002 in
  perplexity. Provenance evidence, not a ranking.
- **Not an MLPerf result.** No MLCommons review or endorsement; GPU-package joules from software
  telemetry, not the wall AC power MLPerf Client's own methodology measures. n = 5 per cell, one card,
  one session, batch 1, single stream. Not pooled into `build/measured.csv`.
- **Archived at [10.5281/zenodo.22295184](https://doi.org/10.5281/zenodo.22295184)** (CC BY 4.0;
  concept DOI 10.5281/zenodo.22295183) with the raw run records, scripts, environment snapshot and
  model hashes. Its abstract quotes *whole-process* energy per output length — −55 % to −61 %, the two
  `Q4_0` arms within 1.2 % of each other at 320 and 576 tokens — while everything here is decode-only
  after differencing the load out. Same runs, different denominator; both columns are in the per-run
  CSV.

## `rtx4090_llamacpp_gguf_2026-08-31.csv` (+ `.summary.csv`) — superseded

**Superseded by the v2 session above; kept published so the correction is auditable.** Read the
following as a record of the provisional first pass, not as current numbers.

The first session on this site measured with **llama.cpp** rather than transformers +
bitsandbytes, and the only one whose quantized file is the one
[MLPerf Client v2.0](https://mlcommons.org/2026/08/mlperf-client-v2-0/) itself ships. One rented
RTX 4090 (Ada, 450 W limit), one llama.cpp build (`b10643-192067b72`, CUDA 12.x),
Llama-3.1-8B-Instruct, three GGUF files × two output lengths (64 and 576 tokens) × three
replicates = 18 runs, batch 1, `-ngl 99 -fa 1 --no-mmap -c 2048 --temp 0 --seed 1234`, GPU package
power from NVML at 10 Hz around the whole `llama-cli` process. Quality is a separate llama.cpp
`perplexity` pass, WikiText-2 raw test, 564 chunks at context 512, and costs the energy figure
nothing. Both CSVs are regenerated from the raw archive by
`python3 build/make_llamacpp_csv.py <archive_dir> data`; the write-up is
[*It was never the format. It's the kernel.*](https://quantenergy.tech/blog/it-was-never-the-format.html).

| llama.cpp arm | mJ/token | tokens/J | tok/s | decode power | Δenergy vs F16 | perplexity | Δppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| `F16` | 4815 | 0.208 | 63.8 | 307 W | — | 7.3260 | — |
| `Q4_0`, ours | 1754 | 0.570 | 168.6 | 296 W | −63.6 % | 7.7364 | +5.60 % |
| `Q4_0`, the v2.0 file | 1795 | 0.557 | 172.6 | 310 W | −62.7 % | 7.7366 | +5.61 % |

### Read this before using the numbers

- **The rows are decode-only, obtained by differencing, not by direct measurement.** A raw process
  energy total includes loading the weights — 16 GB for F16 against 4.7 GB for `Q4_0` — which hands
  the quantized arm a head start unrelated to decoding. `E(576) − E(64)` leaves the energy of
  exactly 512 additional decoded tokens. The per-run CSV carries the undifferenced
  `process_energy_j` so you can redo this differently.
- **The saving is throughput, not lower power.** All three arms decode at 296–310 W on a 450 W card;
  `Q4_0` produces 2.6–2.7× the tokens per second. This is the same mechanism that makes
  `LLM.int8()` *cost* energy in the other files here, running in the opposite direction — which is
  why an energy claim attached to a bit width, with no runtime named, means nothing.
- **Not an MLPerf result.** No MLCommons review or endorsement; MLPerf Client's own power
  methodology measures wall AC power for the whole system with external instrumentation, these are
  GPU-package joules from software telemetry. The two boundaries are not interchangeable.
- **The two `Q4_0` arms are indistinguishable — publish it as a zero result.** Final perplexity
  7.7364 against 7.7366; over 564 chunks the largest absolute difference is 0.0003 and 472 agree to
  three decimals. The 1.4–2.3 % energy gap is inside the run-to-run spread and its **sign flips**
  depending on whether the cold-start runs are excluded (`vs_fp16_energy_pct` against
  `vs_fp16_energy_pct_all_runs`). Both variants are in the summary for that reason. This is
  provenance evidence, not evidence that either file is more efficient.
- **There is a cold-start artifact.** The driving script runs F16 → ours → MLPerf in fixed order
  with no cooldown, so 16 of 18 runs began at 145–248 W of tail from the previous run. The two that
  began near idle (22 W) are exactly the two low outliers among the short runs. The `cold_start`
  column flags them; the headline excludes them (`cold_start_runs_excluded`).
- **The generated token counts are unverified.** The wrapper collected `stdout` but every field came
  back empty, so llama.cpp's `llama_perf_context_print` — the real `n_eval` — is missing. Against
  the card's ~1008 GB/s, the F16 arm's 63.8 tok/s implies ~101 % of theoretical memory bandwidth,
  which is impossible; F16 most likely stopped early on EOS, in which case its true mJ/token is
  lower and the −63.6 % is an overestimate. The `Q4_0` arms sit at a believable ~80 %. **Treat this
  session as provisional** until a rerun with cooldowns, randomized order and captured `n_eval`.
  That rerun is the v2 session above: the overestimate was real and worth about 1.7 points.
- **n = 3 per cell, one card, one session**, batch 1, single stream, one prompt. SD of the
  differenced energy is 7–52 J (0.8–2.1 %). GPU-package power only: no CPU, DRAM, PSU, PUE or CO₂e.
- **Perplexity absolutes are not comparable with published WikiText numbers** (corpus copy and
  tokenizer dependent); only the within-session delta is meaningful.
- **Not pooled into `build/measured.csv`.** Different runtime, different workload shape and a
  differenced energy definition; the fitted curves stay bitsandbytes-only.

## `rtx4090_paired_energy_quality_2026-08-19.csv`

Ten measured configurations (five model sizes × NF4/INT8) from one RTX 4090 (Ada, 24 GB) session on
2026-08-19, produced by [ecocompute-mlcube](https://github.com/hongping-zh/ecocompute-mlcube) with
report schema `ecocompute-energy/1.1`. Every row is `basis = measured`,
`measurement_source = direct-nvml`, `n_trials = 1`.

What makes this session different from the July 2026 one already summarised in `build/measured.csv`:
each row carries, **from the same run**, its own FP16 energy baseline *and* a teacher-forcing
perplexity for both the quantized model and its FP16 baseline. The quality probe runs after the NVML
sampler stops, so it costs the energy figure nothing.

| N (B) | NF4 Δenergy | NF4 Δppl | INT8 Δenergy | INT8 Δppl |
|------:|------------:|---------:|-------------:|----------:|
| 0.5 | +31.1 % | +9.45 % | +594.6 % | +0.52 % |
| 1.1 | +5.5 % | +5.01 % | +301.6 % | +0.54 % |
| 1.5 | −2.2 % | +6.93 % | +365.1 % | +0.51 % |
| 3.0 | −15.1 % | +27.56 % | +273.2 % | +3.79 % |
| 7.0 | −39.0 % | +12.06 % | +105.8 % | +1.19 % |

### Read this before using the numbers

- **The two axes disagree.** INT8 costs almost no perplexity but 106 %–595 % more energy; NF4 saves
  energy above ≈1.4 B while damaging the language model (3B: −15.1 % energy for +27.6 % perplexity).
  An energy-only recommendation and a quality-aware one point at opposite configurations here. The
  columns are published side by side and are deliberately **not** combined into a single
  "quality-adjusted energy" number, which would hide an arbitrary weighting.
- **Perplexity is a proxy for language-model damage, not downstream-task quality.** Absolute values
  depend on the vendored text (SHA-256 `22ac091a…`) and each model's tokenizer, so only
  `delta_perplexity_pct` within a row is meaningful — never compare the absolute value with
  published WikiText numbers or across models.
- **Do not pool this session with the July 2026 one.** Its INT8 penalty is 2.0–2.5× larger at every
  size (INT8 throughput 0.62–0.65×), while the FP16 baselines agree to −10 %…+19 %. What reproduces
  across sessions is the shape — the NF4 penalty falls monotonically with size and crosses over,
  INT8 never saves energy — not the magnitudes. `rtx4090_int8_repeats_2026-08-20.csv` (below)
  measures how large run-to-run noise actually is, and it is 30–50× smaller than this gap.
- **`n = 1` per configuration.** The 10 decode iterations inside a run are integrated into one energy
  total, not ten independent trials.
- GPU-package power only (NVML at 10 Hz): no CPU, DRAM, PSU, PUE or CO₂e.

### Why the site curves are unchanged

`build/measured.csv` (the curve-fitting input) is untouched by this file. Pooling a session whose
INT8 magnitudes differ by 2× into the same fit would make the published curve — and its error bars —
a mixture of two backends rather than an estimate of either. The fit stays on the July session until
there is a second session that agrees, or an explanation for the divergence.

Internal consistency of this session: each size's FP16 baseline was measured twice (in the NF4 run
and in the INT8 run) and agrees to 0.3 %–2.7 %; the FP16 perplexities agree to the last printed digit.

Raw `energy.json` reports for all ten runs are archived with the Zenodo record for the RTX 4090 deep
dive (concept DOI [10.5281/zenodo.22019741](https://doi.org/10.5281/zenodo.22019741), version [10.5281/zenodo.22037483](https://doi.org/10.5281/zenodo.22037483)).

## `rtx4090_int8_repeats_2026-08-20.csv` (+ `.summary.csv`)

The same INT8 configurations, run **three times each** on the same instance the next day, to answer
the question the single-trial files cannot: how much of the July–August disagreement is just noise?
Fifteen rows, all `measured` / `direct-nvml`, same pins (`torch 2.5.1+cu121`, `bitsandbytes 0.43.3`).
The `.summary.csv` carries mean, SD and CV per size.

| N (B) | Δenergy mean (n=3) | SD | **CV** | energy CV | FP16 baseline CV | 2026-08-19 single trial | July 2026 |
|------:|-------------------:|---:|-------:|----------:|-----------------:|------------------------:|----------:|
| 0.5 | +581.3 % | 12.5 | 2.15 % | 0.78 % | 2.45 % | +594.6 % | +241.9 % |
| 1.1 | +307.1 % | 1.8 | 0.57 % | 2.01 % | 1.97 % | +301.6 % | +146.1 % |
| 1.5 | +347.4 % | 7.6 | 2.19 % | 0.30 % | 1.41 % | +365.1 % | +180.7 % |
| 3.0 | +271.2 % | 7.4 | 2.72 % | 0.50 % | 1.51 % | +273.2 % | +134.8 % |
| 7.0 | +105.9 % | 4.1 | 3.87 % | 1.93 % | 0.12 % | +105.8 % | +49.5 % |

- **Run-to-run noise is 0.6–3.9 % (CV of ΔE%), the cross-session gap is 100–140 percentage points.**
  The disagreement with July is therefore 30–50× the measurement noise — it is a property of the
  session, not of the sampling.
- **The 2026-08-19 single trials were representative**: every n=3 mean lands near them (largest
  deviation 1.5B, −18 points ≈ 2.3 SD).
- **Absolute joules drifted 12–17 % lower overnight on the same host** (INT8 and FP16 together), while
  ΔE% held. Report and compare the **FP16-normalised** delta; the absolute J/1k-token figures in
  these files are not comparable across days, let alone across machines.
- **The perplexity column is bit-identical across all three replicates** (and identical to
  2026-08-19). Teacher forcing is deterministic, so the quality axis has CV = 0 *by construction*:
  this shows the pipeline replays exactly, it is **not** independent evidence that the quality result
  replicates.
- **What causes the July–August gap is still open.** The leading hypothesis is the `LLM.int8()`
  kernel path, which changes with the torch build — and no two sessions here ran the same one: the
  2026-07-24 anchors were recorded with `torch 2.4.1+cu121` (Python 3.8), the August native path got
  `torch 2.5.1+cu121`, and the container image pins `torch 2.13.0` (`matches_reference_pins`
  deliberately excludes torch, which must match the host driver). It is untested; the cheapest
  control is one 1.1B INT8 run pinned back to the July build, since the instance cannot nest Docker
  and so cannot run the image as a control. What any explanation has to account for: at every size
  INT8 energy rose 1.41–1.58× with **unchanged package power** (74–91 W vs July's ~76 W), i.e. it is
  a throughput effect (0.63–0.72×), and at small sizes the August FP16 baseline was also lower
  (0.5B: 0.78×), which is what pushes that size's ratio to 2.0×.
