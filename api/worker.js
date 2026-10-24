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
 *   POST /mcp             (Model Context Protocol: JSON-RPC 2.0, stateless —
 *                          tools: estimate / crossover / coverage / search,
 *                          every answer carries its evidence boundaries;
 *                          initialize + tools/call feed the funnel KV)
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
/* Keep only small scalar props: funnel events carry counters and labels, never payloads. */
function sanitizeProps(props) {
  const out = {};
  if (!props || typeof props !== "object") return out;
  for (const [k, v] of Object.entries(props).slice(0, 16)) {
    if (typeof v === "number" && isFinite(v)) out[k] = Math.round(v * 1000) / 1000;
    else if (typeof v === "boolean") out[k] = v;
    else if (typeof v === "string") out[k] = v.slice(0, 64);
  }
  return out;
}

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
    "/mcp": { post: { summary: "MCP endpoint (JSON-RPC 2.0, stateless streamable HTTP)", description: "tools: estimate_quantization_energy, get_quantization_crossover, list_data_coverage, search_measurements — every answer carries basis, confidence and evidence boundaries.", responses: { "200": { description: "JSON-RPC response" } } } },
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

/* ---- MCP (Model Context Protocol) endpoint -----------------------------------
 * POST /mcp speaks JSON-RPC 2.0 (streamable HTTP, stateless server): any MCP
 * client — Claude Desktop, Cursor, an agent runtime — can mount the estimator
 * as a tool. Four tools; every answer carries its own evidence boundaries, so
 * an AI assistant relaying the numbers cannot strip the caveats from them. */

const MCP_VERSION = "1.1.0";
const SCOPE_NOTE =
  "Scope: GPU package power measured via NVML — not wall/host power (multiply by ~1.2–1.6 for the whole node). " +
  "RTX 4090 anchors are mostly single-run (n=1). Data: doi.org/10.5281/zenodo.19647290 (main, cross-arch) · " +
  "doi.org/10.5281/zenodo.22037483 (RTX 4090 deep dive). " +
  "Your card not covered? Measure it and submit: github.com/hongping-zh/ecocompute-mlcube (CC BY 4.0).";
const CROSSOVER_NOTE =
  "A crossover belongs to a card + software-stack combination, not to an architecture: a September 2026 re-run on a newer " +
  "stack moved the RTX 5090 NF4 crossover from ≈4.8B to ≈1.8B. Treat these as today's reading, not a constant.";

const mcpText = (text) => ({ type: "text", text });

function toolEstimate(args) {
  const a = resolveParams(args);
  if (!(a.N > 0))
    return { isError: true, content: [mcpText('params_b (model size in billions) is required and > 0. Example: {"params_b": 7, "arch": "ada", "precision": "NF4"}')] };
  if (!a.arch)
    return { isError: true, content: [mcpText("arch is required: turing, ada, blackwell, ampere, hopper.")] };
  const r = estimate(a.N, a.arch, a.precision || "NF4", { batch: a.batch || 1, ctx: a.ctx || 2048 });
  const input = { params_b: a.N, arch: a.arch, precision: a.precision || "NF4", batch: a.batch || 1, ctx: a.ctx || 2048 };
  if (r.error)
    return { isError: true, content: [mcpText("No answer: " + r.error + "\n\n" + SCOPE_NOTE)] };
  const band = Array.isArray(r.ci) ? ` (95% band ${r.ci[0]}% … ${r.ci[1]}%)` : "";
  const text =
    `Quantize ${a.N}B → ${a.precision || "NF4"} on ${CURVES.arch_label[a.arch] || a.arch}:\n` +
    `  task-level ΔE vs FP16: ${r.delta_pct}%${band}\n` +
    `  recommendation: ${r.recommendation}\n` +
    `  basis: ${r.basis} · confidence: ${r.confidence}\n\n` +
    SCOPE_NOTE;
  return { content: [mcpText(text)], structuredContent: { input, ...r } };
}

function toolCrossover(args) {
  const archs = args.arch ? [String(args.arch).toLowerCase()] : Object.keys(CURVES.arch_label);
  const lines = [];
  for (const arch of archs) {
    const label = CURVES.arch_label[arch] || arch;
    const byPrec = CURVES.curves[arch];
    if (!byPrec) {
      lines.push(`${label}: borrows the ${CURVES.borrow[arch]} curves — estimated, not measured on this architecture`);
      continue;
    }
    for (const [p, c] of Object.entries(byPrec)) {
      const x = c.crossover_b == null
        ? "none — quantization costs energy at every measured size"
        : `≈${c.crossover_b}B (below it costs energy, above it saves)`;
      lines.push(`${label} · ${p}: crossover ${x} · measured range ${c.n_min}–${c.n_max}B`);
    }
  }
  const text = "Weight-only quantization crossover by architecture (ΔE vs FP16):\n  " + lines.join("\n  ") + "\n\n" +
    CROSSOVER_NOTE + "\n\n" + SCOPE_NOTE;
  return { content: [mcpText(text)], structuredContent: { crossovers: lines, note: CROSSOVER_NOTE } };
}

function toolCoverage() {
  const measured = [], estimated = [], gaps = [];
  const precs = ["NF4", "INT8"];
  for (const arch of Object.keys(CURVES.arch_label)) {
    const label = CURVES.arch_label[arch];
    for (const p of precs) {
      const c = CURVES.curves[arch] && CURVES.curves[arch][p];
      if (c) measured.push(`${label} · ${p}: ${c.n_min}–${c.n_max}B, ${c.anchors.length} measured anchor${c.anchors.length === 1 ? "" : "s"}`);
      else if (CURVES.borrow[arch]) estimated.push(`${label} · ${p}: no measurement — borrowed from ${CURVES.borrow[arch]} (estimated)`);
      else gaps.push(`${label} · ${p}: no measured data`);
    }
  }
  gaps.push("sizes outside the measured ranges above (e.g. >7B on Ada): the estimator refuses to extrapolate — mark as a gap, invite a measurement");
  const text =
    "What the estimator's numbers rest on:\n" +
    "  measured (direct NVML):\n    " + measured.join("\n    ") + "\n" +
    "  estimated (borrowed curves, lower confidence):\n    " + estimated.join("\n    ") + "\n" +
    "  gaps (no data — say so, do not extrapolate):\n    " + gaps.join("\n    ") + "\n\n" +
    SCOPE_NOTE;
  return { content: [mcpText(text)], structuredContent: { measured, estimated, gaps } };
}

function toolSearch(args) {
  const q = String(args.query || "").toLowerCase().trim();
  if (!q)
    return { isError: true, content: [mcpText('query is required: a model name ("TinyLlama"), a GPU ("4090"), an architecture ("ampere"), or a size ("7").')] };
  const rows = [];
  for (const [arch, byPrec] of Object.entries(CURVES.curves)) {
    for (const [p, c] of Object.entries(byPrec)) {
      for (const a of c.anchors) {
        const hay = `${a.model || ""} ${a.gpu || ""} ${arch} ${p} ${a.N}`.toLowerCase();
        if (hay.includes(q)) rows.push({ arch, arch_label: CURVES.arch_label[arch], precision: p, params_b: a.N, dE_pct_vs_fp16: a.dE, model: a.model, gpu: a.gpu, runs: a.n });
      }
    }
  }
  if (!rows.length)
    return { content: [mcpText(`No measured anchor matches "${args.query}". That is a data gap, not a zero — call list_data_coverage to see the map, and consider contributing a measurement (github.com/hongping-zh/ecocompute-mlcube).`)] , structuredContent: { matches: [] } };
  const text = "Measured anchors matching \"" + args.query + "\" (ΔE = task-level energy change vs FP16):\n" +
    rows.map((r) => `  ${r.params_b}B ${r.precision} on ${r.arch_label}: ${r.dE_pct_vs_fp16}% · ${r.model} @ ${r.gpu} (n=${r.runs})`).join("\n") +
    "\n\n" + SCOPE_NOTE;
  return { content: [mcpText(text)], structuredContent: { matches: rows } };
}

function mcpTools() {
  return [
    {
      name: "estimate_quantization_energy",
      description: "Estimate the task-level energy change (ΔE% vs FP16) of weight-only quantization (NF4 or INT8) for a model of a given size on a GPU architecture class, from measured-anchored curves. The answer carries basis (measured/interpolated/estimated), confidence and evidence boundaries.",
      inputSchema: {
        type: "object",
        properties: {
          params_b: { type: "number", description: "model size in billions of parameters, e.g. 7" },
          arch: { type: "string", enum: ["turing", "ada", "blackwell", "ampere", "hopper"], description: "GPU architecture class" },
          precision: { type: "string", enum: ["NF4", "INT8"], description: "quantization format (default NF4)" },
          batch: { type: "integer", description: "batch size (default 1; >1 shrinks savings — modelled, not measured)" },
          ctx: { type: "integer", description: "context length (default 2048)" },
          q: { type: "string", description: "alternative: plain-English query, e.g. '13B on A100 int8 batch 32'" },
        },
      },
    },
    {
      name: "get_quantization_crossover",
      description: "Get the model-size crossover for NF4/INT8 per architecture: below it quantization costs energy, above it saves. Crossovers belong to a card + software stack, not an architecture — the answer says so.",
      inputSchema: {
        type: "object",
        properties: { arch: { type: "string", enum: ["turing", "ada", "blackwell", "ampere", "hopper"], description: "optional: limit to one architecture" } },
      },
    },
    {
      name: "list_data_coverage",
      description: "List what is actually measured (GPU architecture × precision × size range, NVML direct), what is borrowed/estimated, and where the gaps are. Use this before trusting an estimate at the edges of coverage, and to answer 'is there data for my GPU?' honestly.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "search_measurements",
      description: "Search the measured anchors by model name, GPU name, architecture or size. Returns ΔE% vs FP16 per anchor with run counts — the raw evidence behind the curves.",
      inputSchema: {
        type: "object",
        properties: { query: { type: "string", description: "model ('TinyLlama'), GPU ('4090'), architecture ('ampere') or size ('7')" } },
        required: ["query"],
      },
    },
  ];
}

function mcpCallTool(name, args) {
  switch (name) {
    case "estimate_quantization_energy": return toolEstimate(args);
    case "get_quantization_crossover": return toolCrossover(args);
    case "list_data_coverage": return toolCoverage();
    case "search_measurements": return toolSearch(args);
    default: return null;
  }
}

const MCP_INSTRUCTIONS =
  "EcoCompute estimates the task-level energy change of LLM weight-only quantization (NF4/INT8) vs FP16, from NVML-measured anchors. " +
  "Every answer labels its basis (measured/interpolated/estimated) and confidence; numbers are GPU package power, not wall power. " +
  "When a GPU/precision/size combination has no measurement, the tools say so instead of extrapolating — call list_data_coverage for the map. " +
  "Users with an uncovered GPU are invited to contribute a measurement: github.com/hongping-zh/ecocompute-mlcube (CC BY 4.0).";

function mcpRpc(m) {
  const id = m && m.id !== undefined ? m.id : null;
  if (!m || typeof m.method !== "string") return { jsonrpc: "2.0", id, error: { code: -32600, message: "invalid JSON-RPC request" } };
  const respond = (result) => ({ jsonrpc: "2.0", id, result });
  const fail = (code, message) => ({ jsonrpc: "2.0", id, error: { code, message } });
  switch (m.method) {
    case "initialize":
      return respond({
        protocolVersion: (m.params && m.params.protocolVersion) || "2025-06-18",
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: "ecocompute-estimator", title: "EcoCompute should-I-quantize", version: MCP_VERSION },
        instructions: MCP_INSTRUCTIONS,
      });
    case "ping":
      return respond({});
    case "tools/list":
      return respond({ tools: mcpTools() });
    case "tools/call": {
      const t = mcpCallTool(m.params && m.params.name, (m.params && m.params.arguments) || {});
      if (!t) return fail(-32602, `unknown tool "${m.params && m.params.name}" — call tools/list for the four available tools`);
      return respond(t);
    }
    default:
      if (m.method.startsWith("notifications/")) return null; // notification: no response body
      if (id === null) return null;
      return fail(-32601, `method not found: ${m.method}`);
  }
}

/* MCP is a first-party funnel channel too: initialize and tools/call are
 * recorded server-side into the same FUNNEL KV, same row shape as /v1/track,
 * so distribution through the registry is measured alongside the site.
 * Only whitelisted scalar arguments are stored — free-text fields (q, query)
 * are deliberately left out. */
const MCP_ARG_PROPS = ["params_b", "arch", "precision", "batch", "ctx"];
async function trackMcp(env, event, props) {
  if (!env || !env.FUNNEL) return;
  try {
    const row = { event, path: "/mcp", props: sanitizeProps(props), ts: new Date().toISOString() };
    const key = row.ts.slice(0, 10) + ":" + crypto.randomUUID();
    // 364 days: comfortably under any 1-year TTL ceiling a KV implementation
    // might enforce; events were silently missing with 400 days (2026-10-24).
    await env.FUNNEL.put(key, JSON.stringify(row), { expirationTtl: 86400 * 364 });
  } catch (_) { /* best-effort, same as the beacon */ }
}

async function handleMcp(req, env) {
  if (req.method === "GET")
    return json({ error: "MCP over streamable HTTP: POST JSON-RPC only. This server is stateless (no GET stream)." }, 405);
  if (req.method === "DELETE")
    return json({ error: "stateless server — nothing to terminate" }, 405);
  if (req.method !== "POST") return json({ error: "use POST" }, 405);
  // JSON-RPC 2.0 allows a batch (array) of messages; readBody only accepts
  // objects for the REST endpoints, so MCP reads its own body here.
  const txt = await req.text();
  if (txt.length > MAX_BODY) return json({ error: `body too large (max ${MAX_BODY} bytes)` }, 413);
  let body;
  try { body = JSON.parse(txt || "{}"); } catch { return json({ error: "invalid JSON body" }, 400); }
  if (body === null || typeof body !== "object") return json({ error: "body must be a JSON-RPC object or batch" }, 400);
  const single = !Array.isArray(body);
  if (!single && (body.length === 0 || body.length > 32)) return json({ error: "batch must contain 1–32 messages" }, 400);
  const out = [];
  const tracks = [];
  for (const m of single ? [body] : body) {
    const r = mcpRpc(m);
    if (r) out.push(r);
    if (m && typeof m.method === "string") {
      if (m.method === "initialize") {
        const client = m.params && m.params.clientInfo && m.params.clientInfo.name;
        tracks.push(trackMcp(env, "mcp_initialize", client ? { client: String(client) } : {}));
      } else if (m.method === "tools/call" && m.params) {
        const p = { tool: String(m.params.name || "").slice(0, 64) };
        const args = m.params.arguments || {};
        for (const k of MCP_ARG_PROPS) if (args[k] !== undefined && args[k] !== null) p[k] = args[k];
        const res = r && r.result;
        p.ok = res && !res.isError ? 1 : 0;
        tracks.push(trackMcp(env, "mcp_tool_call", p));
      }
    }
  }
  if (tracks.length) await Promise.all(tracks);
  if (!out.length)
    return new Response(null, { status: 202, headers: { ...CORS, ...SAFE, "cache-control": CACHE_NO } }); // notifications only
  return json(single ? out[0] : out);
}

export default {
  async fetch(req, env) {
    if (req.method === "OPTIONS") return new Response(null, { headers: { ...CORS, ...SAFE } });
    const u = new URL(req.url);
    const path = u.pathname.replace(/\/+$/, "") || "/";

    const limited = await throttled(req, env);
    if (limited) return limited;

    if (path === "/" ) return json({ service: "ecocompute-estimator", endpoints: ["/v1/estimate", "/v1/optimize", "/v1/architectures", "/v1/track", "/mcp", "/openapi.json"] }, 200, CACHE_OK);
    if (path === "/openapi.json") return json({ ...OPENAPI, servers: [{ url: u.origin }] }, 200, CACHE_OK);
    if (path === "/mcp") return handleMcp(req, env);
    if (path === "/v1/architectures") return json(architectures(), 200, CACHE_OK);

    /* First-party funnel beacon: the site track() helper sendBeacon()s here.
     * No cookies, no PII — just { event, props, path }. Stored in the FUNNEL KV
     * namespace when the binding exists (see api/README.md); always accepted (202)
     * so the beacon never blocks or retries. sendBeacon may send text/plain or
     * no content-type at all, so we do not require application/json here. */
    if (path === "/v1/track") {
      if (req.method !== "POST") return json({ error: "use POST" }, 405);
      const { body, error } = await readBody(req);
      if (error) return error;
      const event = String(body.event || "unknown").slice(0, 64);
      if (!/^[a-z0-9_:.-]+$/i.test(event)) return json({ error: "bad event name" }, 400);
      const row = {
        event,
        path: String(body.path || "").slice(0, 128),
        props: sanitizeProps(body.props),
        ts: new Date().toISOString(),
      };
      if (env && env.FUNNEL) {
        try {
          const key = row.ts.slice(0, 10) + ":" + crypto.randomUUID();
          await env.FUNNEL.put(key, JSON.stringify(row), { expirationTtl: 86400 * 364 });
        } catch (_) { /* KV hiccup: swallow, the beacon is best-effort */ }
      }
      return json({ ok: true }, 202);
    }

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
    return json({ error: "not found", endpoints: ["/v1/estimate", "/v1/optimize", "/v1/architectures", "/mcp", "/openapi.json"] }, 404);
  },
};
