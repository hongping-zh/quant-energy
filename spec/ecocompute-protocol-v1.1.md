# EcoCompute Protocol v1.1

**A Micro-Benchmark Specification for Quantization Energy in LLM Inference**

| | |
|---|---|
| **Status** | Normative specification (stable) |
| **Version** | 1.1 — issued 2026-09-25 |
| **Author** | Hongping Zhang · ORCID [0009-0000-2529-4613](https://orcid.org/0009-0000-2529-4613) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Validation** | `ecocompute-energy/1.2` or later (live: `1.3`) |
| **Canonical** | [DOI 10.5281/zenodo.22958675](https://doi.org/10.5281/zenodo.22958675) (concept: [10.5281/zenodo.22958674](https://doi.org/10.5281/zenodo.22958674)) · rendered at `quantenergy.tech/spec/` |
| **Machine contract** | [`ecocompute-mlcube/schema/energy.schema.json`](https://github.com/hongping-zh/ecocompute-mlcube/blob/main/schema/energy.schema.json) |

---

## Abstract

This document specifies the EcoCompute Protocol: how to measure the effect of weight-only
quantization (e.g. NF4, INT8) on the energy cost of large-language-model inference on a GPU, so
that two measurements taken by different people on different cards can be compared. The protocol
fixes the workload (single-stream decode, batch 1, 256 generated tokens), the baseline (an FP16
run in the same session on the same card), the measurement source (NVML GPU-package power sampled
at ≥ 10 Hz), the measurement window (generation only), and the report contents (validated against
a machine-checkable JSON schema). Requirements are stated in the RFC 2119/8174 style. This
document **is** the protocol, not a description of one: a measurement that satisfies its MUST
requirements and passes its validation entry point is an EcoCompute measurement.

## Status of This Document

This is the normative text of EcoCompute Protocol v1.1. The website `quantenergy.tech` keeps a
human-readable guide with worked examples at `/method/`, the measured evidence at `/measured/`,
data changes at `/changelog/`, and exploratory write-ups at `/blog/` — but the normative text is
**this document**; the web pages are reader's guides. Where a page and this document
disagree, this document governs. The document is archived at
[DOI 10.5281/zenodo.22958675](https://doi.org/10.5281/zenodo.22958675) so that papers can
cite "EcoCompute Protocol v1.1, DOI: 10.5281/zenodo.22958675" rather than a web page that can
change; dataset DOIs and this protocol DOI are separate and correspond through the version table
(§8).

## 1. Introduction

### 1.1 Scope

Every quantization tool documents *how* to quantize; none measures *whether* it saves energy on a
given card. The protocol exists to fill that gap with a measurement that is small enough to run on
a rented or free-tier GPU (a free Colab T4 suffices) and strict enough that its results can be
pooled, compared and falsified.

The measured quantity is the **energy cost of generating tokens**, as a delta between a quantized
configuration and an FP16 baseline measured under identical conditions (§3).

### 1.2 What this protocol is not

- It is **not a certified benchmark** and is not affiliated with MLCommons, any energy-consulting
  body, or any vendor. `scenario` labels such as `SingleStream` are nominal descriptions of the
  workload, not LoadGen-enforced claims.
- It does **not** measure whole-system draw: CPU, DRAM, PSU losses, cooling, PUE and CO₂e are out
  of scope.
- It does **not** measure serving throughput: single-stream decode only (batched serving with
  continuous batching has a different energy profile and is a separate measurement, see §4.8).

## 2. Conventions and Terminology

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**,
**SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** are to be interpreted as described in
RFC 2119 and RFC 8174 when, and only when, they appear in all capitals.

Terms used by this specification:

- **Configuration** — a (model, precision, GPU) triple, e.g. (Qwen2.5-3B, NF4, RTX 4090).
- **Arm** — one measured run of one configuration within a session.
- **Session** — one sitting on one card: arms share the host, driver and library stack.
- **Baseline** — the FP16 arm of the same model in the same session (§4.2).
- **Generation window** — the measured interval: from the first generated token after warm-up to
  the last. Model load, quantization and warm-up are excluded.
- **Thermal mode** — `cold` (cool to idle, fixed warm-up, measure) or `steady` (keep warming until
  three consecutive readings fall within 2 °C, then measure a hot card). Two different
  measurements, not two spellings of one.
- **Anchor** — a distinct (GPU × model × precision) point behind the fitted curves on the site.

## 3. The Measured Quantity

Energy per configuration is obtained by sampling GPU-package power while the model generates
tokens, and integrating power over wall-clock time of the generation window. The reported
quantities are:

- `energy_per_token_mj` — millijoules per generated token, per arm;
- `avg_power_watts` — mean GPU-package power over the window;
- `throughput_tokens_per_s`;
- `vs_fp16_energy_pct` — the quantized arm's energy per token relative to the same-session FP16
  baseline (§4.2), in percent.

Nothing is derived from thermal design power. Values that are **not** measured — sizes between or
beyond the measured anchors, and latency/throughput estimates — are modelled and labelled
`interpolated`, `extrapolated` or `estimated`; measurements and estimates are never mixed
silently, and an estimator output MUST NOT be archived as a measurement.

## 4. Requirements

### 4.1 Power measurement — MUST

1. The power source MUST be NVML on-device GPU-package power sampling.
2. The sampling rate MUST be at least 10 Hz. The report MUST record the **achieved** (not
   requested) sample rate.
3. Energy MUST be the integral of power over wall-clock time of the generation window.
4. Values derived from TDP or vendor typical power MUST NOT be reported as measurements
   (`basis: "measured"` requires `measurement_source: "direct-nvml"`).

### 4.2 Baseline — MUST

1. A `vs_fp16_energy_pct` value MUST be computed against an FP16 run measured **in the same
   session on the same card**.
2. A quantized run without a same-session FP16 baseline MUST NOT yield a vs-FP16 claim.
3. Absolute J/token SHOULD NOT be compared across sessions or days as a proxy for the quantization
   effect: between two consecutive days on the same host, absolute energy per token can drift
   double-digit percent while the FP16-normalised delta moves only a few points.

### 4.3 Workload — MUST

1. Single-stream decode, **batch 1**.
2. **256 generated tokens** per run; 10 decode iterations after a warm-up; greedy decoding.
3. The context length MUST be recorded.
4. Warm-up MUST precede measurement (§4.4 excludes it from the window).

### 4.4 Measurement window — MUST

1. The measurement window MUST be the **generation window**: model load, quantization and warm-up
   excluded.
2. The window definition and the output length MUST be stated with every energy claim —
   re-integrating one session over different windows moves its 64-token figure by up to
   12.5 percentage points.
3. A submission MAY attach whole-run power-trace sidecars (schema `1.3`) with phase markers
   (load → quantization → warm-up → measure → quality) so the window can be re-cut post hoc
   without re-running; this is RECOMMENDED.

### 4.5 Report contents and validation — MUST

1. The report MUST contain the schema-required fields: GPU and architecture, driver,
   runtime/engine and version, model and `params_b`, precision, batch size, context length,
   J/token, tok/s, mean package watts, and the `software` block recording the full version set
   actually used.
2. The report MUST pass validation against `ecocompute-energy/1.2` or any later version (the live
   schema at time of issue is `1.3`; `1.3` validators accept `1.2` reports). See §5.
3. A run on a different stack from the published pins is real but MUST be flagged as not directly
   comparable (the `software` block diffs it; §6.2 governs pooling).

### 4.6 Thermal state — MUST

1. The report MUST record a `thermal` block: the warm-up count and temperature at start, steady
   state, end and peak.
2. Arms MUST be separated by an enforced cooldown; arm order SHOULD be randomised.
3. Comparisons MUST be within one thermal mode, never across `cold` and `steady` (a hot card
   clocks lower and reports more energy per token).
4. A run that never settles MUST report `steady_state_reached: false` rather than pretending it
   did; a card without a temperature sensor MUST report `basis: "unavailable"` rather than
   inventing a value.
5. Sessions recorded before the `thermal` block existed (the 42 seed measurements, v1.1.0 dataset)
   have **unknown** thermal state — not cold. This is why the container default remains
   `--thermal_mode cold` and why their provenance says so.

### 4.7 Replication — SHOULD

1. Each configuration SHOULD be measured at least n = 2 in a reporting dataset (the main dataset's
   repeated trials carry CV < 2%).
2. The replication count MUST travel with the data: `n_trials` in `build/measured.csv`, `n` on
   every anchor in `curves.json`, and n rendered next to every displayed number. A single trial
   MUST be visually marked so no one mistakes one observation for a distribution.
3. Reports SHOULD state `n_session` and, where more than one physical card is involved,
   `n_card` separately (see §6.3).

### 4.8 Additional measurements — MAY

A submitter MAY measure under additional thermal modes (both `cold` and `steady`), batch sizes,
context lengths, token counts, runtimes (vLLM, TensorRT-LLM, SGLang) and quantization formats
(GPTQ, AWQ, FP8, `Q4_K_M`). Such measurements:

1. MUST be labelled separately (the workload fields say what they are), and
2. MUST NOT be pooled with the batch-1 single-stream anchors of the main dataset.

A submitter MAY additionally record the draft `environment` block (schema `1.4-draft`: power
limit, clocks, idle temperature, GPU UUID). This is OPTIONAL at v1.1 and RECOMMENDED for
v1.0-grade submissions (§5, level C): the 2026-09 two-card re-test found that power limit,
clock state and physical card identity were the three context fields whose absence cost the most
when reconstructing a session.

## 5. Validation and Compliance

**A conforming submission MUST pass schema validation AND the semantic checks of the `v1.1-core`
profile — the two are not the same thing.** The machine-readable contract is
[`schema/energy.schema.json`](https://github.com/hongping-zh/ecocompute-mlcube/blob/main/schema/energy.schema.json)
(JSON Schema draft-07) in the container repository, and the semantic checker is
[`tools/validate.py`](https://github.com/hongping-zh/ecocompute-mlcube/blob/main/tools/validate.py)
(`ecocompute validate --profile v1.1-core`). The schema sees structure; the validator sees the
cross-field MUSTs the schema cannot express (same-session FP16 baseline, thermal block present
and honest, pin-mismatch flagging). The reference implementation is the
[EcoCompute energy MLCube](https://github.com/hongping-zh/ecocompute-mlcube), which validates
every report it writes before emitting it, and whose test suite includes deliberately malformed
reports (missing `results`, `basis: "measured"` with a fallback source, mismatched scenario
labels) that must fail. A schema change that lets any of them pass is itself a bug.

"Valid" is not one binary. A report is graded at the highest level it satisfies, and each level
is checked by a different mechanism:

| Level | Name | Criteria | Checked by | May be used for |
|---|---|---|---|---|
| A | Schema-valid | Passes `ecocompute-energy/1.3` validation; required keys present and consistent (version-conditional: at 1.3, `tokens_per_run`/`iterations`/`warmup`/`context_length`/`software` required; `sample_rate_hz` ≥ 10) | the JSON schema | Chart overlay; browser-side comparison; archived as-is |
| B | Protocol-conformant (submittable) | A **plus** every §4 MUST: same-session FP16 baseline, `basis: measured`, `measurement_source: direct-nvml`, batch 1 / 256 tokens / warm-up recorded, full `software` version set, thermal state not violated (or violation disclosed) | the semantic validator (`--profile v1.1-core`) | Publication in `/replications/`, credited; enters the next dataset release after review |
| C | Dataset-eligible (v1.0-grade, draft) | B **plus** power-trace sidecar, achieved sample rate, complete `environment` block, and n ≥ 3 independent sessions or ≥ 2 physical cards for the configuration — cross-session/cross-card spread reported | validator (`--profile dataset-eligible`) **plus dataset-level review**; replication counts live in the build CSV, not in one report | Counted toward the v1.0 micro-standard bar |

Level-C criteria beyond level B are draft and tracked in the container issue tracker; the
`environment` block and the `thermal` block are not yet emitted by the container at schema
`1.3` — until the container emits them, even the maintainer's own 2026-09-25 re-test reports
grade as schema-valid but not protocol-conformant, which the validator states rather than
hides.

## 6. Versioning, Pooling and Comparability

### 6.1 Reports are never rewritten

A `1.2` report stays `1.2`. If its numbers are re-derived (window re-cut, baseline recomputed),
the re-derivation is a **new artifact** with its own provenance entry, and the site changelog
records what moved and why.

### 6.2 What invalidates a comparison

Two reports may both be valid yet not comparable. A pair is not comparable if it differs in:
software stack (quantization kernels move NF4/INT8 results between library versions), measurement
window, thermal mode, locked vs unlocked clocks/power, or any workload field (batch, tokens,
context). Supplementary sessions measured on a different stack MUST be archived separately and
MUST NOT be silently pooled into seed counts or fitted curves; versioning is by session archive
and the changelog says which counts, if any, a session entered.

### 6.3 Card vs session identity

The physical card (GPU UUID) and the session are different units of replication. A dataset that
says n = 3 for one configuration may contain three sessions on two physical cards — for questions
about the card population the effective sample is `n_card` = 2, not 3. Reports SHOULD state both
counts, and SHOULD NOT present a card-vs-session variance decomposition unless each card has at
least two sessions.

## 7. Limitations

1. NVML reports GPU-package power only: CPU, DRAM, PSU losses and cooling are excluded.
2. Single-stream generation only (§1.2).
3. Curves fitted per architecture class pool cards of that class (the Ada class pools RTX 4090
   and RTX 4090D, which differ by roughly 20 percentage points at the same size).
4. Results are backend-specific: NF4 and INT8 here mean bitsandbytes NF4 and `LLM.int8()` at the
   versions recorded in each report. A statement like "INT8 cost energy at every size we tested"
   is a claim about that backend on those cards, not about INT8 as a format.
5. This is a research protocol, not a certified benchmark.

What would falsify the central claims this protocol produces: a measured run, from this container
or an equivalent NVML-based protocol, showing quantization saving energy below the published
crossover for that architecture, or costing energy above it. Such runs are published on the
replications page whether they agree or not.

## 8. Version Correspondence

| Protocol | Report schema | Thermal block | Environment block | Companion datasets (concept DOI) |
|---|---|---|---|---|
| v1.0 (implicit, 2026-07 seed sessions) | `ecocompute-energy/1.2` | absent (thermal state unknown) | absent | main dataset [10.5281/zenodo.19647290](https://doi.org/10.5281/zenodo.19647290) |
| **v1.1 (this document)** | `1.2` accepted, `1.3` live | **required** (§4.6) | draft `1.4-draft`, optional (§4.8) | main dataset [19647290](https://doi.org/10.5281/zenodo.19647290) · RTX 4090 deep dive [22037483](https://doi.org/10.5281/zenodo.22037483) (concept [22019741](https://doi.org/10.5281/zenodo.22019741)) · RTX 5090 re-test [22855133](https://doi.org/10.5281/zenodo.22855133) |

Dataset DOIs and the protocol DOI are separate. Each dataset release states, in its metadata,
the protocol version its sessions were measured under; the table above is the normative
correspondence.

## 9. Acknowledgments

The protocol is only useful if people other than its maintainer can run it. Each independent
replication — an `energy.json` produced by someone else's hardware, electricity and time — is
acknowledged here, in arrival order:

- **@gkgoing** — first independent replication (2026-08): TinyLlama-1.1B in NF4 on an
  RTX 3050 Ti Laptop GPU (Ampere), Windows, NVML package power, with a paired perplexity that
  matches the maintainer's own RTX 4090 run to four decimals.
  [Submission](https://github.com/hongping-zh/ecocompute-mlcube/issues/15) ·
  [gallery entry](https://quantenergy.tech/replications/).

*Reserved:* the first replication on a GPU architecture absent from the maintainer's own set
(dataset v1.1.0 covers Turing, Ada Lovelace, Ampere, Blackwell — e.g. Hopper, or any
architecture not listed there) will be named here in a separate line. That slot is still open;
the first replication above is an Ampere laptop card, and Ampere is in the maintainer's set.

This section is non-normative and grows as replications arrive. The Zenodo snapshot (§ Status
of This Document) carries the version frozen at its publication date; the live count of
independent replications and the current contributor list are maintained at
`quantenergy.tech/replications/`.

## 10. References

- Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels", BCP 14, RFC 2119.
- Leiba, B., "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words", BCP 14, RFC 8174.
- NVIDIA Management Library (NVML), as exposed by `nvidia-ml-py`.
- EcoCompute energy MLCube (reference implementation):
  `github.com/hongping-zh/ecocompute-mlcube` (Apache-2.0).
- Report schema: `ecocompute-mlcube/schema/energy.schema.json` (JSON Schema draft-07).
- Website and evidence: `quantenergy.tech`.
- Preprint: "Weight-Only Quantization Does Not Always Save Energy: An Empirical Study of LLM
  Inference Across NVIDIA GPU Platforms", SSRN #6854700
  ([archived: 10.5281/zenodo.21066652](https://doi.org/10.5281/zenodo.21066652)).

## Appendix A. Relationship to the Website

The site `quantenergy.tech` is organised in four layers, of which this document is the normative
core:

| Layer | URL | Role |
|---|---|---|
| Spec (this document) | `/spec/` + DOI | The protocol itself |
| Spec guide | `/method/` | Human-readable导读: requirements walkthrough, number glossary, contribution flow |
| Machine contract | `/schema/` | The JSON schema, compliance levels, migration rules |
| Evidence | `/measured/` | Every measured number, its n, its stack and its DOI |
| Updates | `/changelog/` | What changed, in what order, and why |
| Findings | `/blog/` | Exploratory write-ups; not counted in the coverage matrix |

*Issued 2026-09-25 · EcoCompute is an independent research project by Hongping Zhang
([ORCID 0009-0000-2529-4613](https://orcid.org/0009-0000-2529-4613)). This document is licensed
CC BY 4.0.*
