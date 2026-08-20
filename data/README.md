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
  size (INT8 throughput 0.62–0.65×), while the FP16 baselines agree to −10 %…+19 % and the CUDA build
  was the same (torch 2.5.1+cu121). What reproduces across sessions is the shape — the NF4 penalty
  falls monotonically with size and crosses over, INT8 never saves energy — not the magnitudes.
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
dive (concept DOI [10.5281/zenodo.22019741](https://doi.org/10.5281/zenodo.22019741), version [10.5281/zenodo.22019742](https://doi.org/10.5281/zenodo.22019742)).
