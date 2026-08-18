/* EcoCompute "should-I-quantize?" REST endpoint (Cloudflare Worker).
 *
 * Thin HTTP wrapper around the deterministic estimator in ../estimate.js — the
 * numbers come from measured-anchored curves, never from a model guessing.
 *
 *   GET  /v1/estimate?params_b=13&arch=ampere&precision=nf4&batch=32&ctx=8192
 *   GET  /v1/estimate?q=13B%20on%20A100%20int8%20batch%2032
 *   POST /v1/estimate     { "params_b":13, "arch":"ampere", "precision":"NF4", "batch":32 }
 *   POST /v1/estimate     { "q":"13B on A100 int8 batch 32" }
 *   GET  /v1/architectures
 *   GET  /v1/optimize?params_b=7&arch=ada&objective=min_energy&max_latency_ms=50&max_vram_gb=16
 *   POST /v1/optimize     { "params_b":7, "arch":"ada", "objective":"min_energy", "max_vram_gb":16 }
 *   GET  /openapi.json    (so other agents can auto-discover the tool)
 *
 * Deploy:  cd api && npx wrangler deploy   (see api/README.md)
 */
import estimator from "../estimate.js";
import optimizer from "../optimize.js";
const { estimate, parseQuery, CURVES } = estimator;
const { optimize } = optimizer;

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type",
  "access-control-max-age": "86400",
};
const SAFE = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
};
// The estimator is deterministic: the same query always yields the same answer
// until the curves are refit and the Worker redeployed. Successful GETs are
// therefore cacheable at the edge; anything else is not.
const CACHE_OK = "public, max-age=300, s-maxage=3600";
const CACHE_NO = "no-store";
const json = (obj, status = 200, cache = CACHE_NO) =>
  new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": cache,
      ...SAFE,
      ...CORS,
    },
  });

// Largest POST body we will read. Every documented field is a number or a short
// string; anything larger is a mistake or an attempt to make us allocate.
const MAX_BODY = 8 * 1024;
async function readBody(req) {
  const len = parseInt(req.headers.get("content-length") || "0", 10);
  if (len > MAX_BODY) return { error: json({ error: `body too large (max ${MAX_BODY} bytes)` }, 413) };
  const txt = await req.text();
  if (txt.length > MAX_BODY) return { error: json({ error: `body too large (max ${MAX_BODY} bytes)` }, 413) };
  try {
    const body = JSON.parse(txt || "{}");
    if (!body || typeof body !== "object" || Array.isArray(body)) return { error: json({ error: "body must be a JSON object" }, 400) };
    return { body };
  } catch {
    return { error: json({ error: "invalid JSON body" }, 400) };
  }
}

// Per-IP throttle, only if the ratelimits binding exists (see wrangler.toml).
// Without it the Worker behaves exactly as before rather than failing shut.
async function throttled(req, env) {
  const limiter = env && env.API_RATE_LIMIT;
  if (!limiter || typeof limiter.limit !== "function") return null;
  const key = req.headers.get("cf-connecting-ip") || "anonymous";
  try {
    const { success } = await limiter.limit({ key });
    if (success) return null;
  } catch {
    return null;
  }
  return new Response(
    JSON.stringify({ error: "rate limit exceeded", hint: "the estimator is also a plain JS module: https://github.com/hongping-zh/quant-energy/blob/main/estimate.js" }, null, 2),
    { status: 429, headers: { "content-type": "application/json; charset=utf-8", "retry-after": "60", "cache-control": CACHE_NO, ...SAFE, ...CORS } }
  );
}

function architectures() {
  const out = {};
  for (const [arch, byPrec] of Object.entries(CURVES.curves)) {
    out[arch] = {
      label: CURVES.arch_label[arch] || arch,
      precisions: Object.fromEntries(
        Object.entries(byPrec).map(([p, c]) => [p, { measured_range_b: [c.n_min, c.n_max], crossover_b: c.crossover_b }])
      ),
    };
  }
  for (const [arch, src] of Object.entries(CURVES.borrow || {})) {
    if (!out[arch]) out[arch] = { label: CURVES.arch_label[arch] || arch, borrows_from: src, precisions: {} };
  }
  return out;
}

function resolveParams(p) {
  if (p.q) return parseQuery(String(p.q));
  return {
    N: p.params_b != null ? parseFloat(p.params_b) : (p.N != null ? parseFloat(p.N) : NaN),
    arch: p.arch ? String(p.arch).toLowerCase() : undefined,
    precision: p.precision ? String(p.precision).toUpperCase() : "NF4",
    batch: p.batch != null ? parseInt(p.batch, 10) : 1,
    ctx: p.ctx != null ? parseInt(p.ctx, 10) : 2048,
  };
}

function handleEstimate(p, cache = CACHE_NO) {
  const a = resolveParams(p);
  if (!(a.N > 0)) return json({ error: "params_b (model size in billions) is required and must be > 0" }, 400);
  if (!a.arch) return json({ error: "arch is required", supported: Object.keys(architectures()) }, 400);
  const r = estimate(a.N, a.arch, a.precision || "NF4", { batch: a.batch || 1, ctx: a.ctx || 2048 });
  if (r.error) return json({ ...r, input: a }, 400);
  return json({ input: { params_b: a.N, arch: a.arch, precision: a.precision || "NF4", batch: a.batch || 1, ctx: a.ctx || 2048 }, ...r }, 200, cache);
}

const NUM = ["params_b", "max_latency_ms", "max_vram_gb", "min_throughput", "batch", "context", "ctx"];
const LIST = ["precisions", "batches", "contexts"];
function resolveOptimize(p) {
  const out = {};
  for (const k of Object.keys(p)) {
    if (NUM.includes(k)) out[k] = parseFloat(p[k]);
    else if (LIST.includes(k)) out[k] = Array.isArray(p[k]) ? p[k] : String(p[k]).split(",").map((s) => s.trim()).filter(Boolean);
    else out[k] = p[k];
  }
  if (out.arch) out.arch = String(out.arch).toLowerCase();
  if (Array.isArray(out.batches)) out.batches = out.batches.map(Number);
  if (Array.isArray(out.contexts)) out.contexts = out.contexts.map(Number);
  if (Array.isArray(out.precisions)) out.precisions = out.precisions.map((s) => s.toUpperCase());
  return out;
}

function handleOptimize(p, cache = CACHE_NO) {
  const a = resolveOptimize(p);
  if (!(a.params_b > 0)) return json({ error: "params_b (model size in billions) is required and must be > 0" }, 400);
  if (!a.arch) return json({ error: "arch is required", supported: Object.keys(architectures()) }, 400);
  const r = optimize(a);
  if (r.error) return json({ ...r, input: a }, 400);
  return json(r, 200, cache);
}

const OPENAPI = {
  openapi: "3.0.0",
  info: { title: "EcoCompute should-I-quantize API", version: "1.0.0",
    description: "Deterministic estimate of weight-only quantization energy change (ΔE% vs FP16), calibrated to measured GPU data." },
  paths: {
    "/v1/estimate": {
      get: {
        summary: "Estimate ΔE% for quantizing a model on a GPU architecture",
        parameters: [
          { name: "params_b", in: "query", schema: { type: "number" }, description: "model size in billions of parameters" },
          { name: "arch", in: "query", schema: { type: "string", enum: ["turing", "ada", "blackwell", "ampere", "hopper"] } },
          { name: "precision", in: "query", schema: { type: "string", enum: ["NF4", "INT8"] } },
          { name: "batch", in: "query", schema: { type: "integer", default: 1 } },
          { name: "ctx", in: "query", schema: { type: "integer", default: 2048 } },
          { name: "q", in: "query", schema: { type: "string" }, description: "plain-English query, e.g. '13B on A100 int8 batch 32'" },
        ],
        responses: { "200": { description: "estimate" } },
      },
    },
    "/v1/architectures": { get: { summary: "List supported architectures, precisions and measured ranges", responses: { "200": { description: "ok" } } } },
    "/v1/optimize": {
      get: {
        summary: "Recommend the objective-optimal (precision, batch, context) under latency/VRAM/throughput constraints",
        description: "Exhaustive constrained search. Energy is measured-anchored; latency/throughput are a roofline MODEL; VRAM is computed. Every config carries per-field basis + confidence.",
        parameters: [
          { name: "params_b", in: "query", required: true, schema: { type: "number" }, description: "model size in billions of parameters" },
          { name: "arch", in: "query", required: true, schema: { type: "string", enum: ["turing", "ada", "blackwell", "ampere", "hopper"] } },
          { name: "objective", in: "query", schema: { type: "string", enum: ["min_energy", "min_latency", "max_throughput", "min_vram"], default: "min_energy" } },
          { name: "max_latency_ms", in: "query", schema: { type: "number" }, description: "latency budget, ms per token" },
          { name: "max_vram_gb", in: "query", schema: { type: "number" }, description: "VRAM budget in GB" },
          { name: "min_throughput", in: "query", schema: { type: "number" }, description: "throughput floor, tokens/s" },
          { name: "batch", in: "query", schema: { type: "integer" }, description: "pin batch size (else searched)" },
          { name: "context", in: "query", schema: { type: "integer" }, description: "pin context length (else searched)" },
        ],
        responses: { "200": { description: "recommended config, alternatives and energy/latency Pareto front" } },
      },
    },
  },
};

export default {
  async fetch(req, env) {
    if (req.method === "OPTIONS") return new Response(null, { headers: { ...CORS, ...SAFE } });
    const u = new URL(req.url);
    const path = u.pathname.replace(/\/+$/, "") || "/";

    const limited = await throttled(req, env);
    if (limited) return limited;

    if (path === "/" ) return json({ service: "ecocompute-estimator", endpoints: ["/v1/estimate", "/v1/optimize", "/v1/architectures", "/openapi.json"] }, 200, CACHE_OK);
    if (path === "/openapi.json") return json({ ...OPENAPI, servers: [{ url: u.origin }] }, 200, CACHE_OK);
    if (path === "/v1/architectures") return json(architectures(), 200, CACHE_OK);

    if (path === "/v1/optimize") {
      if (req.method === "GET") return handleOptimize(Object.fromEntries(u.searchParams), CACHE_OK);
      if (req.method === "POST") {
        const { body, error } = await readBody(req);
        if (error) return error;
        return handleOptimize(body);
      }
      return json({ error: "use GET or POST" }, 405);
    }

    if (path === "/v1/estimate") {
      if (req.method === "GET") return handleEstimate(Object.fromEntries(u.searchParams), CACHE_OK);
      if (req.method === "POST") {
        const { body, error } = await readBody(req);
        if (error) return error;
        return handleEstimate(body);
      }
      return json({ error: "use GET or POST" }, 405);
    }
    return json({ error: "not found", endpoints: ["/v1/estimate", "/v1/optimize", "/v1/architectures", "/openapi.json"] }, 404);
  },
};
