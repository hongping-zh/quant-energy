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

// Rate limiting is optional: absent binding behaves as before, a refusing one 429s.
const noEnv = await hit("GET", "/v1/architectures", undefined, {});
console.log("no limiter bound:", noEnv.status);
const limitedEnv = { API_RATE_LIMIT: { limit: async () => ({ success: false }) } };
const rl = await hit("GET", "/v1/estimate?params_b=7&arch=ada", undefined, limitedEnv);
console.log("limiter refuses:", rl.status, rl.headers.get("retry-after"));
const brokenEnv = { API_RATE_LIMIT: { limit: async () => { throw new Error("down"); } } };
const rb = await hit("GET", "/v1/estimate?params_b=7&arch=ada", undefined, brokenEnv);
console.log("limiter throws -> fail open:", rb.status);

console.log("\nOK");
