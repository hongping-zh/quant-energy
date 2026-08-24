# Measured sessions published alongside this site

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
