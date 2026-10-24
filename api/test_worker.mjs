import worker from "./worker.js";

async function hit(method, url, body, env) {
  const init = { method };
  if (body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
    init.headers = { "content-type": "application/json" };
  }
  const res = await worker.fetch(new Request("https://x" + url, init), env);
  const txt = await res.text();
  let j; try { j = JSON.parse(txt); } catch { j = txt; }
  return { status: res.status, j, cache: res.headers.get("cache-control"), headers: res.headers };
}

const a = await hit("GET", "/v1/estimate?params_b=7&arch=blackwell&precision=nf4");
console.log("GET 7B blackwell:", a.status, a.j.delta_pct, a.j.basis, a.j.confidence);

const b = await hit("GET", "/v1/estimate?q=" + encodeURIComponent("13B on A100 int8 batch 32 ctx 8k"));
console.log("GET q=:", b.status, b.j.input, b.j.delta_pct, b.j.modelled);

const c = await hit("POST", "/v1/estimate", { params_b: 3, arch: "ada", precision: "NF4" });
console.log("POST 3B ada:", c.status, c.j.delta_pct, c.j.basis);

const d = await hit("GET", "/v1/estimate?arch=ampere");
console.log("missing params_b:", d.status, d.j.error);

const e = await hit("GET", "/v1/architectures");
console.log("architectures:", e.status, Object.keys(e.j).join(","));

const f = await hit("GET", "/openapi.json");
console.log("openapi:", f.status, f.j.info && f.j.info.title);

const g = await hit("GET", "/nope");
console.log("404:", g.status);

const o1 = await hit("GET", "/v1/optimize?params_b=7&arch=ada&objective=min_energy&max_latency_ms=50&max_vram_gb=16");
console.log("optimize ada 7B:", o1.status, o1.j.recommended && o1.j.recommended.summary, "| feas:", o1.j.feasible_count);

const o2 = await hit("POST", "/v1/optimize", { params_b: 7, arch: "blackwell", objective: "max_throughput", max_vram_gb: 32 });
console.log("optimize blackwell 7B max_tput:", o2.status, o2.j.recommended && o2.j.recommended.summary, "| pareto:", o2.j.pareto_energy_latency && o2.j.pareto_energy_latency.length);

const o3 = await hit("GET", "/v1/optimize?arch=ada");
console.log("optimize missing params_b:", o3.status, o3.j.error);

const o4 = await hit("GET", "/v1/optimize?params_b=7&arch=ada&max_vram_gb=4");
console.log("optimize infeasible:", o4.status, "recommended=", o4.j.recommended, "| closest:", o4.j.closest_infeasible && o4.j.closest_infeasible.summary);

// Caching: successful GETs are edge-cacheable, everything else is not.
console.log("\ncache GET estimate:", a.cache);
console.log("cache POST estimate:", c.cache);
console.log("cache 400:", d.cache, "| nosniff:", a.headers.get("x-content-type-options"));

// Oversized body is refused before it is parsed.
const big = await hit("POST", "/v1/estimate", '{"q":"' + "x".repeat(9000) + '"}');
console.log("oversized body:", big.status, big.j.error);
const arr = await hit("POST", "/v1/estimate", "[1,2,3]");
console.log("array body:", arr.status, arr.j.error);


// Funnel beacon: accepted without a KV binding (202), stored with one, bad names rejected.
const kv = { put: async (k, v) => { console.log("kv put:", k.slice(0, 10), JSON.parse(v).event); } };
const ok = await hit("POST", "/v1/track", JSON.stringify({ event: "cta_hero_click", props: { view: "estimate" }, path: "/" }), { FUNNEL: kv });
console.log("track ok:", ok.status, ok.j.ok);
const nokv = await hit("POST", "/v1/track", JSON.stringify({ event: "estimate", props: { params_b: 7 } }));
console.log("track no-kv:", nokv.status, nokv.j.ok);
const bad = await hit("POST", "/v1/track", JSON.stringify({ event: "bad name!" }));
console.log("track bad name:", bad.status, bad.j.error);
const get = await hit("GET", "/v1/track");
console.log("track GET:", get.status, get.j.error);

// Rate limiting is optional: absent binding behaves as before, a refusing one 429s.
const noEnv = await hit("GET", "/v1/architectures", undefined, {});
console.log("no limiter bound:", noEnv.status);
const limitedEnv = { API_RATE_LIMIT: { limit: async () => ({ success: false }) } };
const rl = await hit("GET", "/v1/estimate?params_b=7&arch=ada", undefined, limitedEnv);
console.log("limiter refuses:", rl.status, rl.headers.get("retry-after"));
const brokenEnv = { API_RATE_LIMIT: { limit: async () => { throw new Error("down"); } } };
const rb = await hit("GET", "/v1/estimate?params_b=7&arch=ada", undefined, brokenEnv);
console.log("limiter throws -> fail open:", rb.status);

// ---- MCP endpoint: JSON-RPC 2.0, stateless streamable HTTP ----
const m1 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "t", version: "0" } } });
console.log("mcp initialize:", m1.status, m1.j.result.serverInfo.name, m1.j.result.protocolVersion, m1.j.result.instructions ? "instructions-ok" : "NO-instructions");

const m2 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 2, method: "tools/list" });
const toolNames = m2.j.result.tools.map((t) => t.name).join(",");
console.log("mcp tools/list:", m2.status, toolNames, "| schemas:", m2.j.result.tools.every((t) => t.inputSchema && t.description) ? "ok" : "BROKEN");

const m3 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "estimate_quantization_energy", arguments: { params_b: 7, arch: "ada", precision: "NF4" } } });
const t3 = m3.j.result.content[0].text;
console.log("mcp estimate 7B ada:", m3.status, m3.j.result.content[0].type, "| -28.5 in text:", t3.includes("-28"), "| basis:", t3.includes("basis: measured"), "| scope:", t3.includes("package power"), "| structured:", m3.j.result.structuredContent && m3.j.result.structuredContent.delta_pct);

const m3b = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 33, method: "tools/call", params: { name: "estimate_quantization_energy", arguments: { q: "13B on A100 int8 batch 32" } } });
console.log("mcp estimate via q:", m3b.status, m3b.j.result.structuredContent && m3b.j.result.structuredContent.delta_pct, m3b.j.result.structuredContent && m3b.j.result.structuredContent.basis);

const m3c = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 34, method: "tools/call", params: { name: "estimate_quantization_energy", arguments: { arch: "ada" } } });
console.log("mcp estimate missing params_b:", m3c.status, m3c.j.result.isError === true ? "isError-ok" : "NO-error");

const m4 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "get_quantization_crossover", arguments: {} } });
const t4 = m4.j.result.content[0].text;
console.log("mcp crossover:", m4.status, "| ada 3.7:", t4.includes("3.7"), "| INT8 none:", t4.toLowerCase().includes("none"), "| stack note:", t4.includes("software-stack") || t4.includes("card + software"));

const m5 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 5, method: "tools/call", params: { name: "list_data_coverage", arguments: {} } });
const t5 = m5.j.result.content[0].text;
console.log("mcp coverage:", m5.status, "| measured:", t5.includes("measured (direct NVML)"), "| estimated:", t5.includes("borrowed"), "| gaps:", t5.includes("gaps"), "| hopper:", t5.includes("Hopper"));

const m6 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 6, method: "tools/call", params: { name: "search_measurements", arguments: { query: "TinyLlama" } } });
console.log("mcp search TinyLlama:", m6.status, m6.j.result.structuredContent.matches.length, "matches | first:", m6.j.result.structuredContent.matches[0] && m6.j.result.structuredContent.matches[0].dE_pct_vs_fp16);

const m6b = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 61, method: "tools/call", params: { name: "search_measurements", arguments: { query: "A800" } } });
console.log("mcp search A800:", m6b.status, m6b.j.result.structuredContent.matches.length, "matches");

const m7 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 7, method: "ping" });
console.log("mcp ping:", m7.status, JSON.stringify(m7.j.result));

const m8 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 8, method: "no/such/method" });
console.log("mcp bad method:", m8.status, m8.j.error.code);

const m9 = await hit("POST", "/mcp", { jsonrpc: "2.0", method: "notifications/initialized" });
console.log("mcp notification -> 202 no body:", m9.status, m9.j === "" || m9.j === null ? "empty-ok" : m9.j);

const m10 = await hit("GET", "/mcp");
console.log("mcp GET -> 405:", m10.status);

const m11 = await hit("POST", "/mcp", { jsonrpc: "2.0", id: 11, method: "tools/call", params: { name: "no_such_tool", arguments: {} } });
console.log("mcp unknown tool:", m11.status, m11.j.error.code, m11.j.error.message.includes("tools/list") ? "hint-ok" : "no-hint");

const m12 = await hit("POST", "/mcp", [{ jsonrpc: "2.0", id: 12, method: "ping" }, { jsonrpc: "2.0", method: "notifications/initialized" }]);
console.log("mcp batch (ping + notification):", m12.status, Array.isArray(m12.j) && m12.j.length === 1 ? "one-response-ok" : m12.j);

console.log("\nOK");
