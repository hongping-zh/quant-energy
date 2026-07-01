/* EcoCompute config optimizer — "give me constraints, get the best config".
 *
 * Turns the deterministic point-estimator (estimate.js) into a constrained
 * optimizer:  user gives {GPU arch, model size, latency/VRAM/throughput budgets,
 * objective} -> we search the discrete config space (precision x batch x context),
 * filter by the constraints, and return the objective-optimal config, honest
 * alternatives, and the energy<->latency Pareto frontier.
 *
 *   optimize({ arch, params_b, objective, max_latency_ms, max_vram_gb,
 *              min_throughput, precisions?, batches?, contexts? }) -> { ... }
 *
 * The config grid is tiny (~3x6x5), so the search is EXHAUSTIVE => globally
 * optimal and exact; no heuristic solver needed.
 *
 * HONESTY (per-field basis, same discipline as estimate.js):
 *   - energy   : MEASURED-anchored. dE% from the fitted curves x the measured
 *                FP16 absolute-energy baseline (curves.json:fp16_energy).
 *   - vram     : COMPUTED. weight bytes + KV-cache (standard formula), approx.
 *   - latency/ : MODELLED (roofline). decode is memory-bandwidth bound:
 *     throughput   t_step ~ (weight_bytes/B + kv_bytes)/(mem_BW*MBU), capped by a
 *                compute roof. Uses public datasheet specs (curves.json:hardware).
 *                NOT measured — flagged as such.
 * Every returned config carries basis {energy,latency,vram} + confidence so a
 * modelled number is never presented as a measured one.
 */
(function (root) {
  "use strict";

  var EST = (typeof require !== "undefined" && typeof module !== "undefined" && module.exports)
    ? require("./estimate.js")
    : (root.EcoEstimator || root);
  var estimate = EST.estimate, CURVES = EST.CURVES;

  // ---- modelling constants (documented, not measured) ----------------------
  var MBU = 0.70;                 // memory-bandwidth utilization during decode (roofline)
  var COMPUTE_UTIL = 0.30;        // tensor-core utilization during decode (compute roof)
  var KV_ELEM_BYTES = 2;          // KV cache stored in fp16
  var BATCH_ENERGY_FIXED = 0.35;  // fraction of per-token energy that is fixed overhead
                                  // (amortized by batching) -> batching saves <=35% e/token
  var CTX_BASE = 2048;            // measurement context baseline
  var BITS = { FP16: 16, INT8: 8, NF4: 4 };

  var DEFAULT_PRECISIONS = ["FP16", "NF4", "INT8"];
  var DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32];
  var DEFAULT_CONTEXTS = [512, 1024, 2048, 4096, 8192];

  var BASIS_RANK = { measured: 3, interpolated: 2, estimated: 1, computed: 3, modelled: 1 };
  function worseBasis(a, b) {
    if (a == null) return b; if (b == null) return a;
    return (BASIS_RANK[a] <= BASIS_RANK[b]) ? a : b;
  }

  // crude params(billions) -> (n_layers, d_model), for KV-cache sizing only (approx).
  function archShape(N) {
    var tbl = [
      { N: 0.5, L: 24, H: 896 }, { N: 1.1, L: 22, H: 2048 }, { N: 1.5, L: 28, H: 1536 },
      { N: 3.0, L: 36, H: 2048 }, { N: 7.0, L: 32, H: 4096 }, { N: 9.0, L: 48, H: 4096 },
      { N: 14.0, L: 48, H: 5120 }, { N: 32.0, L: 64, H: 6656 }, { N: 70.0, L: 80, H: 8192 }
    ];
    if (N <= tbl[0].N) return { L: tbl[0].L, H: tbl[0].H };
    for (var i = 1; i < tbl.length; i++) {
      if (N <= tbl[i].N) {
        var a = tbl[i - 1], b = tbl[i], t = (N - a.N) / (b.N - a.N);
        return { L: Math.round(a.L + (b.L - a.L) * t), H: Math.round(a.H + (b.H - a.H) * t) };
      }
    }
    return { L: tbl[tbl.length - 1].L, H: tbl[tbl.length - 1].H };
  }

  // VRAM (GB): weights + KV cache + framework/activation headroom. Standard calc.
  function vramGB(N, precision, ctx, batch) {
    var bits = BITS[precision] || 16;
    var weights = N * 1e9 * bits / 8;                       // bytes
    var s = archShape(N);
    var kv = 2 * s.L * s.H * ctx * batch * KV_ELEM_BYTES;   // K+V, fp16
    var overhead = 0.8e9 + weights * 0.05;                  // CUDA ctx + activations
    return +((weights + kv + overhead) / 1e9).toFixed(2);
  }

  // Measured FP16 absolute energy baseline (J / 1k tokens) interpolated in N.
  function fp16BaseJ1k(N, arch) {
    var fe = CURVES.fp16_energy && CURVES.fp16_energy[arch];
    if (!fe || !fe.anchors || !fe.anchors.length) return null;
    var a = fe.anchors;
    if (N <= a[0].N) return +(a[0].e_j1k * (N / a[0].N)).toFixed(2); // ~linear below range
    for (var i = 1; i < a.length; i++) {
      if (N <= a[i].N) {
        var p = a[i - 1], q = a[i], t = (N - p.N) / (q.N - p.N);
        return +(p.e_j1k + (q.e_j1k - p.e_j1k) * t).toFixed(2);
      }
    }
    var last = a[a.length - 1], prev = a[a.length - 2];
    var slope = (last.e_j1k - prev.e_j1k) / (last.N - prev.N);
    return +(last.e_j1k + slope * (N - last.N)).toFixed(2);
  }
  function fp16BaseBasis(N, arch) {
    var fe = CURVES.fp16_energy && CURVES.fp16_energy[arch];
    if (!fe) return null;
    for (var i = 0; i < fe.anchors.length; i++) if (Math.abs(fe.anchors[i].N - N) < 1e-6) return "measured";
    return (N >= fe.n_min && N <= fe.n_max) ? "interpolated" : "estimated";
  }

  // Roofline decode latency/throughput (MODELLED). Returns per-sequence ms/token,
  // aggregate tok/s, and which roof binds.
  function roofline(N, precision, ctx, batch, arch) {
    var hw = CURVES.hardware && CURVES.hardware[arch];
    if (!hw) return null;
    var bits = BITS[precision] || 16;
    var weightBytes = N * 1e9 * bits / 8;
    var s = archShape(N);
    var kvPerSeq = 2 * s.L * s.H * ctx * KV_ELEM_BYTES;      // bytes read per token per seq
    var bw = hw.mem_bw_gbps * 1e9 * MBU;                     // effective bytes/s
    var bwStep = (weightBytes + batch * kvPerSeq) / bw;      // s/step, memory-bound
    var flops = hw.fp16_tflops * 1e12 * COMPUTE_UTIL;        // effective FLOP/s
    var computeStep = batch * (2 * N * 1e9) / flops;         // s/step, compute-bound
    var stepTime = Math.max(bwStep, computeStep);
    return {
      latency_ms_per_token: +(stepTime * 1000).toFixed(2),
      throughput_tok_s: +(batch / stepTime).toFixed(1),
      bound: (computeStep > bwStep) ? "compute" : "memory"
    };
  }

  function scoreCandidate(N, arch, precision, batch, ctx) {
    // Pure precision effect at the measurement baseline (batch=1, ctx=base) — the
    // measured signal. Batch/context effects are added by the optimizer's own
    // (modelled) layers below, so we DON'T pass batch/ctx into estimate() here
    // (that would double-count).
    var est;
    if (precision === "FP16") {
      est = { delta_pct: 0, basis: "measured", confidence: "high", modelled: false, notes: [] };
    } else {
      est = estimate(N, arch, precision);
      if (est.error) return { error: est.error };
    }

    var baseJ1k = fp16BaseJ1k(N, arch);
    var baseBasis = fp16BaseBasis(N, arch);
    var deltaPct = est.delta_pct;                                      // vs FP16 @ same batch/ctx

    // absolute energy = measured FP16 baseline x precision factor x modelled batch amortization
    var precFactor = 1 + deltaPct / 100;
    var amort = (1 - BATCH_ENERGY_FIXED) + BATCH_ENERGY_FIXED / batch; // 1 at b=1, applies to all precisions
    var energyJ1k = (baseJ1k != null) ? +(baseJ1k * precFactor * amort).toFixed(1) : null;
    var mjPerTok = (energyJ1k != null) ? +(energyJ1k / 1000).toFixed(3) : null; // J/1k -> mJ/tok

    var vram = vramGB(N, precision, ctx, batch);
    var rl = roofline(N, precision, ctx, batch, arch);

    var energyModelled = (batch > 1);
    var energyBasis = worseBasis(precision === "FP16" ? baseBasis : est.basis, baseBasis);
    var energyConf = est.confidence || "medium";
    if (energyModelled && energyConf === "high") energyConf = "medium";

    // overall confidence = weakest link (latency is always modelled => <= medium)
    var overall = energyConf;
    if (overall === "high") overall = "medium"; // roofline latency is modelled

    return {
      precision: precision, batch: batch, context: ctx,
      energy_j_per_1k_tokens: energyJ1k,
      energy_mj_per_token: mjPerTok,
      energy_rel_fp16: +precFactor.toFixed(3),
      delta_pct_vs_fp16: +deltaPct.toFixed(1),
      latency_ms_per_token: rl ? rl.latency_ms_per_token : null,
      throughput_tok_s: rl ? rl.throughput_tok_s : null,
      bound: rl ? rl.bound : null,
      vram_gb: vram,
      basis: { energy: energyBasis, latency: "modelled (roofline)", vram: "computed" },
      confidence: overall,
      recommendation: est.recommendation || (precision === "FP16" ? "baseline" : undefined)
    };
  }

  function feasible(cand, cons) {
    if (cons.max_vram_gb != null && cand.vram_gb > cons.max_vram_gb) return false;
    if (cons.max_latency_ms != null && cand.latency_ms_per_token != null &&
        cand.latency_ms_per_token > cons.max_latency_ms) return false;
    if (cons.min_throughput != null && cand.throughput_tok_s != null &&
        cand.throughput_tok_s < cons.min_throughput) return false;
    return true;
  }

  function violation(cand, cons) {
    var v = 0;
    if (cons.max_vram_gb != null && cand.vram_gb > cons.max_vram_gb)
      v += (cand.vram_gb - cons.max_vram_gb) / cons.max_vram_gb;
    if (cons.max_latency_ms != null && cand.latency_ms_per_token > cons.max_latency_ms)
      v += (cand.latency_ms_per_token - cons.max_latency_ms) / cons.max_latency_ms;
    if (cons.min_throughput != null && cand.throughput_tok_s < cons.min_throughput)
      v += (cons.min_throughput - cand.throughput_tok_s) / cons.min_throughput;
    return v;
  }

  function objKey(objective) {
    switch (objective) {
      case "min_latency": return function (c) { return c.latency_ms_per_token != null ? c.latency_ms_per_token : Infinity; };
      case "max_throughput": return function (c) { return c.throughput_tok_s != null ? -c.throughput_tok_s : Infinity; };
      case "min_vram": return function (c) { return c.vram_gb; };
      case "min_energy":
      default: return function (c) { return c.energy_j_per_1k_tokens != null ? c.energy_j_per_1k_tokens : c.energy_rel_fp16 * 1e9; };
    }
  }

  // Pareto frontier over (energy, latency): lower is better on both.
  function paretoFront(cands) {
    var pts = cands.filter(function (c) { return c.energy_j_per_1k_tokens != null && c.latency_ms_per_token != null; });
    var front = pts.filter(function (c) {
      return !pts.some(function (o) {
        return o !== c &&
          o.energy_j_per_1k_tokens <= c.energy_j_per_1k_tokens &&
          o.latency_ms_per_token <= c.latency_ms_per_token &&
          (o.energy_j_per_1k_tokens < c.energy_j_per_1k_tokens || o.latency_ms_per_token < c.latency_ms_per_token);
      });
    });
    front.sort(function (a, b) { return a.energy_j_per_1k_tokens - b.energy_j_per_1k_tokens; });
    return front.map(function (c) {
      return { precision: c.precision, batch: c.batch, context: c.context,
        energy_j_per_1k_tokens: c.energy_j_per_1k_tokens, latency_ms_per_token: c.latency_ms_per_token,
        throughput_tok_s: c.throughput_tok_s, vram_gb: c.vram_gb };
    });
  }

  function pct(a, b) { return b ? +(((a - b) / b) * 100).toFixed(1) : null; }

  function optimize(req) {
    req = req || {};
    var N = (req.params_b != null) ? +req.params_b : (req.N != null ? +req.N : NaN);
    var arch = (req.arch || "").toString().toLowerCase();
    if (!(N > 0)) return { error: "params_b (model size in billions) is required and must be > 0" };
    if (!arch) return { error: "arch is required", supported: Object.keys(CURVES.curves) };
    if (!CURVES.curves[arch] && !(CURVES.borrow && CURVES.borrow[arch]))
      return { error: "no data for arch '" + arch + "'", supported: Object.keys(CURVES.curves) };

    var objective = req.objective || "min_energy";
    var precisions = req.precisions || (req.precision ? [String(req.precision).toUpperCase()] : DEFAULT_PRECISIONS);
    // batch/context can be pinned (req.batch/req.context) or searched over.
    var batches = req.batches || (req.batch != null ? [parseInt(req.batch, 10)] : DEFAULT_BATCHES);
    var contexts = req.contexts || (req.context != null ? [parseInt(req.context, 10)] : (req.ctx != null ? [parseInt(req.ctx, 10)] : DEFAULT_CONTEXTS));
    var cons = {
      max_latency_ms: req.max_latency_ms != null ? +req.max_latency_ms : null,
      max_vram_gb: req.max_vram_gb != null ? +req.max_vram_gb : null,
      min_throughput: req.min_throughput != null ? +req.min_throughput : null
    };

    var all = [], errors = {};
    for (var pi = 0; pi < precisions.length; pi++)
      for (var bi = 0; bi < batches.length; bi++)
        for (var ci = 0; ci < contexts.length; ci++) {
          var s = scoreCandidate(N, arch, precisions[pi], batches[bi], contexts[ci]);
          if (s.error) { errors[precisions[pi]] = s.error; continue; }
          all.push(s);
        }

    if (!all.length) return { error: "no candidates could be scored", detail: errors };

    var key = objKey(objective);
    // primary = objective; tie-break toward lower energy, then lower VRAM, then lower latency.
    var byObj = function (a, b) {
      var d = key(a) - key(b); if (Math.abs(d) > 1e-9) return d;
      var e = (a.energy_j_per_1k_tokens || 0) - (b.energy_j_per_1k_tokens || 0); if (Math.abs(e) > 1e-9) return e;
      var v = a.vram_gb - b.vram_gb; if (Math.abs(v) > 1e-9) return v;
      return (a.latency_ms_per_token || 0) - (b.latency_ms_per_token || 0);
    };
    all.sort(byObj);

    var feas = all.filter(function (c) { return feasible(c, cons); });
    var pareto = paretoFront(all);

    var result = {
      input: { params_b: N, arch: arch, objective: objective,
        constraints: cons, grid: { precisions: precisions, batches: batches, contexts: contexts, candidates: all.length } },
      basis: { energy: "measured-anchored (dE% x measured FP16 baseline)",
        latency: "modelled (roofline; datasheet mem-bandwidth/compute)", vram: "computed (weights + KV cache)" },
      feasible_count: feas.length,
      notes: []
    };

    if (!feas.length) {
      var closest = all.slice().sort(function (a, b) { return violation(a, cons) - violation(b, cons); })[0];
      result.recommended = null;
      result.closest_infeasible = decorate(closest, "closest (violates constraints)");
      result.notes.push("No configuration satisfies all constraints. Showing the closest (smallest constraint violation); consider relaxing the binding budget.");
      addGlobalNotes(result, N, arch);
      return result;
    }

    var best = feas[0];
    result.recommended = decorate(best, "objective-optimal under the given constraints");
    result.alternatives = feas.slice(1, 4).map(function (c) {
      var d = decorate(c, "alternative");
      d.vs_recommended = {
        energy_pct: pct(c.energy_j_per_1k_tokens, best.energy_j_per_1k_tokens),
        latency_pct: pct(c.latency_ms_per_token, best.latency_ms_per_token),
        throughput_pct: pct(c.throughput_tok_s, best.throughput_tok_s),
        vram_pct: pct(c.vram_gb, best.vram_gb)
      };
      return d;
    });
    result.pareto_energy_latency = pareto;
    addGlobalNotes(result, N, arch);
    return result;
  }

  function decorate(c, role) {
    var o = {};
    for (var k in c) o[k] = c[k];
    o.role = role;
    o.summary = c.precision + ", batch=" + c.batch + ", context=" + c.context +
      (c.energy_j_per_1k_tokens != null ? " -> " + c.energy_j_per_1k_tokens + " J/1k tok" : "") +
      (c.throughput_tok_s != null ? ", " + c.throughput_tok_s + " tok/s" : "") +
      (c.latency_ms_per_token != null ? ", " + c.latency_ms_per_token + " ms/tok" : "") +
      ", " + c.vram_gb + " GB";
    return o;
  }

  function addGlobalNotes(result, N, arch) {
    result.notes.push("Energy is measured-anchored; latency & throughput are a roofline MODEL (not measured); VRAM is a standard calculation.");
    result.notes.push("Roofline latency assumes memory-bound decode and ignores dequantization compute overhead, so quantized-format latency is a best case (lower bound). Energy — which does capture that overhead — is the measured signal.");
    if (!(CURVES.fp16_energy && CURVES.fp16_energy[arch]))
      result.notes.push("No measured FP16 energy baseline for this architecture — energy is relative only.");
    result.notes.push("Batch>1 energy uses a conservative amortization model (fixed-overhead fraction " + BATCH_ENERGY_FIXED + ").");
  }

  var API = { optimize: optimize, scoreCandidate: scoreCandidate, vramGB: vramGB,
    roofline: roofline, fp16BaseJ1k: fp16BaseJ1k, archShape: archShape, paretoFront: paretoFront,
    DEFAULTS: { precisions: DEFAULT_PRECISIONS, batches: DEFAULT_BATCHES, contexts: DEFAULT_CONTEXTS } };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.EcoOptimizer = API;
  root.optimize = optimize;
})(typeof globalThis !== "undefined" ? globalThis : this);
