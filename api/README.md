# EcoCompute estimator API (`estimate.js`)

A small, **deterministic** "should I quantize?" estimator. Given a model size, a GPU
architecture class and a precision, it returns the expected energy change vs FP16
(`ΔE%`), a 95% range, and an honesty label (`measured` / `interpolated` / `estimated`).

It is the single source of truth behind the demo's **"Any model · estimate"** tab, and
it is intentionally packaged as a reusable tool: import it in the browser, in Node, or
register `estimate()` as a function/tool for an LLM agent. The numbers always come from
this function — never from a model "guessing".

- Core module: [`../estimate.js`](../estimate.js)
- Python reference (1:1 logic, used by validation): [`../build/estimate.py`](../build/estimate.py)
- Calibration data: [`../curves.json`](../curves.json), fit offline by
  [`../build/fit_curves.py`](../build/fit_curves.py) from
  [`../build/measured.csv`](../build/measured.csv) (the `ecocompute-ai` dataset).

## How the estimate is produced

Per `(architecture, precision)` we fit a saturating curve to the **measured** anchors:

```
ΔE%(N) = A − S · g(N / N*),   g(x) = x / (1 + x)
  A  = de-quant penalty floor (>0; large when an INT8 kernel is poor, e.g. Ampere INT8)
  S  = max savings as the model becomes memory-bound (capped: NF4 ≤ 75%, INT8 ≤ 50%)
  N* = crossover scale for that GPU class
```

Small N → `ΔE% ≈ A` (penalty); large N → `ΔE% ≈ A − S` (savings). The sign flip is the
**crossover**. Anything outside the measured size range, or an architecture that has no
data of its own (borrowed shape, e.g. Hopper ← Ampere), is labelled `estimated` with a
wider band and lower confidence. Nothing is fabricated.

## Usage — browser

```html
<script src="estimate.js"></script>
<script>
  const r = estimate(13, "ampere", "NF4");
  // -> { delta_pct:-1, ci:[-6.3,4.3], basis:"interpolated", confidence:"high",
  //      recommendation:"depends", verdict:"...", weight_gb:6.5, weight_gb_fp16:26, ... }

  // also exposed: EcoEstimator.{estimate,parseQuery,weightGB,savingsFactor,CURVES}
</script>
```

## Usage — Node

```js
const { estimate, parseQuery } = require("./estimate.js");

estimate(7, "blackwell", "NF4");            // 7B on RTX 5090, weight-only NF4
estimate(7, "blackwell", "NF4", { batch: 32, ctx: 8192 });   // modelled batch/context
parseQuery("13B on A100, INT8, batch 32, ctx 8k");
// -> { N:13, arch:"ampere", precision:"INT8", batch:32, ctx:8192 }
```

## API

### `estimate(N, arch, precision, opts?) -> result`

| arg | type | notes |
|-----|------|-------|
| `N` | number | model size in **billions** of parameters |
| `arch` | string | `turing` · `ada` · `blackwell` · `ampere` · `hopper` |
| `precision` | string | `NF4` · `INT8` (per-arch availability; see `CURVES`) |
| `opts.batch` | number | default `1`. >1 ⇒ **modelled** adjustment (see below) |
| `opts.ctx` | number | context length in tokens, default `2048` (= measurement baseline) |

`result` fields:

| field | meaning |
|-------|---------|
| `delta_pct` | point estimate of ΔE% vs FP16 (negative = saves energy) |
| `ci` | `[lo, hi]` 95% range |
| `basis` | `measured` · `interpolated` · `estimated` |
| `confidence` | `high` · `medium` · `low` |
| `recommendation` | `quantize` · `do_not_quantize` · `depends` |
| `verdict` | one-line human-readable summary |
| `crossover_b` | crossover size (B) for the arch, or `null` if not pinned down |
| `anchors` | the measured points the curve was fit to |
| `arch_used` | actual arch curve used (differs from `arch` when borrowed) |
| `modelled` | `true` when batch/ctx deviate from the baseline |
| `weight_gb` / `weight_gb_fp16` | approx weight-only memory footprint |
| `notes` | caveats (extrapolation, borrowed shape, modelled batch/ctx) |

### Batch / context (modelled, **not** measured)

`batch=1, ctx≤2048` reproduces the measured curve exactly. When they grow, a transparent,
conservative model shrinks **savings toward zero** (more batched / longer-context work is
more compute-bound, so weight-only quant helps less) — it never invents a new penalty.
Such results are flagged `modelled`, with lower confidence and a wider band.

```
factor = 1/(1 + (batch-1)/8) · 1/(1 + max(0, log2(ctx/2048))·0.12)   // in (0,1]
if delta < 0: delta *= factor                                        // shrink savings only
```

### `parseQuery(text) -> {N?, arch?, precision?, batch?, ctx?}`

Rules-based (no LLM) parser that powers the demo's plain-English box. Recognises sizes
(`13B`, `13 billion`), cards/arch (`A100`/`A800`→ampere, `5090`→blackwell, `T4`→turing,
`4090`→ada, `H100`→hopper), precision (`int8`/`8-bit`, `nf4`/`4-bit`), `batch N` and
`ctx N` (`8k` → 8192). Use it to map free text onto `estimate()` args.

### `weightGB(N, precision)` / `savingsFactor(batch, ctx)`

Helpers exposed for reuse (memory footprint and the batch/ctx factor above).

## REST endpoint (Cloudflare Worker) — live

**Live base URL:** `https://ecocompute-estimator.zhanghongping1982.workers.dev`

Try it:

```bash
curl "https://ecocompute-estimator.zhanghongping1982.workers.dev/v1/estimate?params_b=7&arch=blackwell&precision=nf4"
curl "https://ecocompute-estimator.zhanghongping1982.workers.dev/v1/architectures"
curl "https://ecocompute-estimator.zhanghongping1982.workers.dev/openapi.json"
```

A working Worker that wraps `estimate()` lives in this folder: [`worker.js`](worker.js) +
[`wrangler.toml`](wrangler.toml). It is dependency-free and bundles to ~16 KiB. CORS is
open (`access-control-allow-origin: *`) so browsers and other agents can call it directly.

Endpoints:

```
GET  /v1/estimate?params_b=13&arch=ampere&precision=nf4&batch=32&ctx=8192
GET  /v1/estimate?q=13B%20on%20A100%20int8%20batch%2032   # plain-English query
POST /v1/estimate   {"params_b":13,"arch":"ampere","precision":"NF4","batch":32}
POST /v1/estimate   {"q":"13B on A100 int8 batch 32"}
GET  /v1/architectures        # supported archs, precisions, measured ranges
GET  /openapi.json            # so other agents can auto-discover the tool
```

Example response:

```json
{
  "input": { "params_b": 13, "arch": "ampere", "precision": "INT8", "batch": 32, "ctx": 8192 },
  "delta_pct": 108.8, "ci": [19.7, 197.9], "basis": "interpolated",
  "confidence": "low", "recommendation": "do_not_quantize", "modelled": true,
  "weight_gb": 13, "weight_gb_fp16": 26, "notes": ["Batch/context effect is modelled ..."]
}
```

Deploy:

```bash
cd api
npx wrangler login            # one-time (or set CLOUDFLARE_API_TOKEN)
npx wrangler deploy           # -> https://ecocompute-estimator.zhanghongping1982.workers.dev
npx wrangler deploy --dry-run --outdir /tmp/build   # build-only sanity check
```

Local smoke test of the handler (no Cloudflare account needed):

```bash
node api/test_worker.mjs
```

## MCP server (for IDE / copilot agents)

To let other agents (Claude Desktop, Cursor, internal copilots) call the estimator as a
native tool, see [`../mcp`](../mcp) — a stdio MCP server exposing `should_i_quantize` and
`list_architectures`, reusing the same `estimate.js`. No hosting required.

## Validation

`python3 build/validate.py` checks the fit against the measured data: leave-one-out MAE,
95% band coverage, the S-cap bound, and sign sanity. Current: overall LOO MAE ≈ 6.3 pts,
band coverage 18/18.
