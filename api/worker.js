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
 *   GET  /openapi.json    (so other agents can auto-discover the tool)
 *
 * Deploy:  cd api && npx wrangler deploy   (see api/README.md)
 */
import estimator from "../estimate.js";
const { estimate, parseQuery, CURVES } = estimator;

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type",
};
const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...CORS },
  });

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

function handleEstimate(p) {
  const a = resolveParams(p);
  if (!(a.N > 0)) return json({ error: "params_b (model size in billions) is required and must be > 0" }, 400);
  if (!a.arch) return json({ error: "arch is required", supported: Object.keys(architectures()) }, 400);
  const r = estimate(a.N, a.arch, a.precision || "NF4", { batch: a.batch || 1, ctx: a.ctx || 2048 });
  if (r.error) return json({ ...r, input: a }, 400);
  return json({ input: { params_b: a.N, arch: a.arch, precision: a.precision || "NF4", batch: a.batch || 1, ctx: a.ctx || 2048 }, ...r });
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
  },
};

export default {
  async fetch(req) {
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
    const u = new URL(req.url);
    const path = u.pathname.replace(/\/+$/, "") || "/";

    if (path === "/" ) return json({ service: "ecocompute-estimator", endpoints: ["/v1/estimate", "/v1/architectures", "/openapi.json"] });
    if (path === "/openapi.json") return json(OPENAPI);
    if (path === "/v1/architectures") return json(architectures());

    if (path === "/v1/estimate") {
      if (req.method === "GET") return handleEstimate(Object.fromEntries(u.searchParams));
      if (req.method === "POST") {
        let body = {};
        try { body = await req.json(); } catch { return json({ error: "invalid JSON body" }, 400); }
        return handleEstimate(body);
      }
      return json({ error: "use GET or POST" }, 405);
    }
    return json({ error: "not found", endpoints: ["/v1/estimate", "/v1/architectures", "/openapi.json"] }, 404);
  },
};
