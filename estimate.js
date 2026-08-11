/* EcoCompute estimator — deterministic, reproducible.
 *
 * Single source of truth for "should I quantize?" estimates. Drives the demo UI,
 * and is callable as a tool/API by anything (browser, Node, an LLM agent).
 *
 *   estimate(N, arch, precision, opts?) -> { delta_pct, ci, basis, confidence,
 *                                            recommendation, verdict, anchors, ... }
 *
 * CURVES is calibrated OFFLINE by build/fit_curves.py from build/measured.csv
 * (the ecocompute-ai dataset). The base curve ΔE%(N)=A−S·g(N/N*) is measured.
 * The optional batch/context adjustment is a transparent MODEL (clearly labelled),
 * not new measurements: it only shrinks savings toward zero, never invents a penalty.
 */
(function (root) {
  "use strict";

  var CURVES = {
    "s_cap": {"NF4": 75.0, "INT8": 50.0},
    "arch_label": {
      "turing": "Turing (T4)",
      "ada": "Ada (RTX 4090/4090D)",
      "blackwell": "Blackwell (RTX 5090)",
      "ampere": "Ampere (A100/A800)",
      "hopper": "Hopper (H100/H200)"
    },
    "borrow": {"hopper": "ampere"},
    "curves": {
      "ada": {
        "INT8": { "A": 291.8945, "S": 274.5289, "Nstar": 1.4423, "resid_std": 26.5949, "n_min": 0.5, "n_max": 7.0, "crossover_b": null, "loo_mae": 47.455,
          "anchors": [{"N": 0.5, "dE": 241.89, "model": "Qwen2-0.5B", "gpu": "RTX 4090", "n": 1}, {"N": 1.1, "dE": 146.11, "model": "TinyLlama-1.1B", "gpu": "RTX 4090", "n": 1}, {"N": 1.1, "dE": 137.9, "model": "TinyLlama-1.1B", "gpu": "RTX 4090", "n": 1}, {"N": 1.5, "dE": 180.73, "model": "Qwen2-1.5B", "gpu": "RTX 4090", "n": 1}, {"N": 3.0, "dE": 134.82, "model": "Qwen2.5-3B", "gpu": "RTX 4090", "n": 1}, {"N": 7.0, "dE": 49.54, "model": "Qwen2-7B", "gpu": "RTX 4090", "n": 1}] },
        "NF4": { "A": 54.2295, "S": 131.0936, "Nstar": 5.2846, "resid_std": 11.4822, "n_min": 0.5, "n_max": 7.0, "crossover_b": 3.728, "loo_mae": 16.123,
          "anchors": [{"N": 0.5, "dE": 35.63, "model": "Qwen2-0.5B", "gpu": "RTX 4090", "n": 1}, {"N": 0.5, "dE": 56.09, "model": "Qwen2-0.5B", "gpu": "RTX 4090D", "n": 1}, {"N": 1.1, "dE": 16.31, "model": "TinyLlama-1.1B", "gpu": "RTX 4090", "n": 1}, {"N": 1.1, "dE": 33.35, "model": "TinyLlama-1.1B", "gpu": "RTX 4090D", "n": 2}, {"N": 1.5, "dE": 15.11, "model": "Qwen2-1.5B", "gpu": "RTX 4090", "n": 1}, {"N": 1.5, "dE": 38.6, "model": "Qwen2-1.5B", "gpu": "RTX 4090D", "n": 2}, {"N": 3.0, "dE": 0.76, "model": "Qwen2.5-3B", "gpu": "RTX 4090", "n": 1}, {"N": 3.0, "dE": 25.22, "model": "Qwen2.5-3B", "gpu": "RTX 4090D", "n": 2}, {"N": 7.0, "dE": -28.46, "model": "Qwen2-7B", "gpu": "RTX 4090", "n": 1}] }
      },
      "ampere": {
        "INT8": { "A": 180.8252, "S": 127.9389, "Nstar": 10.082, "resid_std": 2.4297, "n_min": 7.0, "n_max": 14.0, "crossover_b": null, "loo_mae": null,
          "anchors": [{"N": 7.0, "dE": 130.83, "model": "Mistral-7B", "gpu": "A800", "n": 2}, {"N": 9.0, "dE": 117.18, "model": "Yi-1.5-9B", "gpu": "A800", "n": 2}, {"N": 14.0, "dE": 107.41, "model": "Qwen2.5-14B", "gpu": "A800", "n": 2}] },
        "NF4": { "A": -1.0293, "S": 0.0, "Nstar": 89.9821, "resid_std": 2.7146, "n_min": 7.0, "n_max": 14.0, "crossover_b": null, "loo_mae": null,
          "anchors": [{"N": 7.0, "dE": -4.09, "model": "Mistral-7B", "gpu": "A800", "n": 2}, {"N": 9.0, "dE": -1.51, "model": "Yi-1.5-9B", "gpu": "A800", "n": 2}, {"N": 14.0, "dE": 2.51, "model": "Qwen2.5-14B", "gpu": "A800", "n": 2}] }
      },
      "blackwell": {
        "NF4": { "A": 45.8272, "S": 104.4224, "Nstar": 6.0749, "resid_std": 2.8022, "n_min": 1.1, "n_max": 7.0, "crossover_b": 4.751, "loo_mae": 4.846,
          "anchors": [{"N": 1.1, "dE": 26.49, "model": "TinyLlama-1.1B", "gpu": "RTX 5090", "n": 2}, {"N": 1.5, "dE": 29.42, "model": "Qwen2-1.5B", "gpu": "RTX 5090", "n": 2}, {"N": 3.0, "dE": 11.74, "model": "Qwen2.5-3B", "gpu": "RTX 5090", "n": 2}, {"N": 7.0, "dE": -11.45, "model": "Qwen2-7B", "gpu": "RTX 5090", "n": 2}] }
      },
      "turing": {
        "NF4": { "A": 7.925, "S": 79.5646, "Nstar": 19.2186, "resid_std": 1.316, "n_min": 1.1, "n_max": 7.0, "crossover_b": 2.126, "loo_mae": 5.262,
          "anchors": [{"N": 1.1, "dE": 4.56, "model": "TinyLlama-1.1B", "gpu": "T4", "n": 2}, {"N": 1.5, "dE": 0.22, "model": "Qwen2-1.5B", "gpu": "T4", "n": 2}, {"N": 3.0, "dE": -1.38, "model": "Qwen2.5-3B", "gpu": "T4", "n": 2}, {"N": 7.0, "dE": -13.75, "model": "Qwen2-7B", "gpu": "T4", "n": 2}] }
      }
    },
    // Measured FP16 absolute decode energy (J / 1k tokens), per arch — anchors the
    // optimizer's absolute-energy numbers. Mirror of curves.json:fp16_energy.
    "fp16_energy": {
      "ada": { "n_min": 0.5, "n_max": 7.0, "anchors": [{ "N": 0.5, "e_j1k": 1525.33 }, { "N": 1.1, "e_j1k": 1585.71 }, { "N": 1.5, "e_j1k": 2213.4 }, { "N": 3.0, "e_j1k": 3368.4 }, { "N": 7.0, "e_j1k": 5467.88 }] },
      "blackwell": { "n_min": 1.1, "n_max": 7.0, "anchors": [{ "N": 1.1, "e_j1k": 1659.0 }, { "N": 1.5, "e_j1k": 2411.09 }, { "N": 3.0, "e_j1k": 3382.64 }, { "N": 7.0, "e_j1k": 5508.56 }] },
      "turing": { "n_min": 1.1, "n_max": 7.0, "anchors": [{ "N": 1.1, "e_j1k": 4251.21 }, { "N": 1.5, "e_j1k": 5731.8 }, { "N": 3.0, "e_j1k": 11267.69 }, { "N": 7.0, "e_j1k": 21722.65 }] },
      "ampere": { "n_min": 7.0, "n_max": 14.0, "anchors": [{ "N": 7.0, "e_j1k": 4402.43 }, { "N": 9.0, "e_j1k": 5445.12 }, { "N": 14.0, "e_j1k": 7359.98 }] }
    },
    // Public datasheet GPU specs — used ONLY by the (modelled) roofline latency
    // layer in optimize.js. Mirror of curves.json:hardware.
    "hardware": {
      "turing": {"gpu": "T4", "mem_bw_gbps": 320.0, "fp16_tflops": 65.0, "source": "NVIDIA T4 datasheet"},
      "ada": {"gpu": "RTX 4090D", "mem_bw_gbps": 1008.0, "fp16_tflops": 330.0, "source": "NVIDIA Ada / RTX 4090 datasheet"},
      "blackwell": {"gpu": "RTX 5090", "mem_bw_gbps": 1792.0, "fp16_tflops": 419.0, "source": "NVIDIA RTX 5090 datasheet"},
      "ampere": {"gpu": "A800", "mem_bw_gbps": 2039.0, "fp16_tflops": 312.0, "source": "NVIDIA A100/A800 80GB datasheet"},
      "hopper": {"gpu": "H100", "mem_bw_gbps": 3350.0, "fp16_tflops": 990.0, "source": "NVIDIA H100 SXM datasheet"}
    }
  };

  var Z = 1.96;
  var BITS = { FP16: 16, INT8: 8, NF4: 4 };   // bits per weight (weight-only quant)
  var CTX_BASE = 2048;                          // measurement context baseline
  var BATCH_HALF = 8;                           // batch where memory-bound savings ~halve

  function fmtNum(x) { return (Math.round(x * 10) / 10).toString(); }
  function g(x) { return x / (1 + x); }
  function modelCurve(N, c) { return c.A - c.S * g(N / c.Nstar); }

  // Measured FP16 absolute decode energy baseline (J / 1k tokens == mJ / token) for a
  // given size+arch, from the NVML-measured fp16_energy anchors. Log-log interpolation
  // inside the measured range ("measured"/"interpolated"), extrapolation outside ("estimated").
  function fp16BaselineJ1k(N, arch) {
    var t = CURVES.fp16_energy[arch];
    if (!t || !t.anchors || !t.anchors.length) return null;
    var a = t.anchors;
    for (var i = 0; i < a.length; i++) { if (Math.abs(a[i].N - N) < 1e-6) return { e_j1k: a[i].e_j1k, basis: "measured" }; }
    var lg = Math.log;
    var sorted = a.slice().sort(function (x, y) { return x.N - y.N; });
    var p0, p1;
    var lo = null, hi = null;
    for (var j = 0; j < sorted.length; j++) { if (sorted[j].N < N) lo = sorted[j]; if (sorted[j].N > N && hi === null) hi = sorted[j]; }
    if (lo && hi) { p0 = lo; p1 = hi; }
    else if (!lo) { p0 = sorted[0]; p1 = sorted[1]; }
    else { p0 = sorted[sorted.length - 2]; p1 = sorted[sorted.length - 1]; }
    var slope = (lg(p1.e_j1k) - lg(p0.e_j1k)) / (lg(p1.N) - lg(p0.N));
    var e = Math.exp(lg(p0.e_j1k) + slope * (lg(N) - lg(p0.N)));
    var basis = (N >= t.n_min && N <= t.n_max) ? "interpolated" : "estimated";
    return { e_j1k: +e.toFixed(1), basis: basis };
  }

  // Approx weight-only memory footprint in GB (not from the dataset; a standard calc).
  function weightGB(N, precision) {
    var bits = BITS[precision] || 16;
    return +(N * bits / 8 * 1.0).toFixed(2); // N(billion) * bytes/param = GB (1.0 = no KV/activation)
  }

  // Conservative, transparent batch/context model. Returns a factor in (0,1] that
  // shrinks SAVINGS toward zero as batch/context grow (more compute-bound / more KV
  // traffic => weight-quant helps less). Never below 0; never turns savings into a
  // penalty without data. factor=1 at batch=1 and ctx<=baseline (reproduces measured).
  function savingsFactor(batch, ctx) {
    var b = (batch && batch > 1) ? batch : 1;
    var fBatch = 1 / (1 + (b - 1) / BATCH_HALF);
    var c = (ctx && ctx > CTX_BASE) ? ctx : CTX_BASE;
    var fCtx = 1 / (1 + Math.max(0, Math.log2(c / CTX_BASE)) * 0.12);
    return fBatch * fCtx;
  }

  function estimate(N, arch, precision, opts) {
    opts = opts || {};
    var curves = CURVES.curves, sCap = (CURVES.s_cap[precision] || 75), borrow = CURVES.borrow || {};
    var srcArch = arch, borrowedFrom = null;
    if (!curves[arch] || !curves[arch][precision]) {
      if (borrow[arch] && curves[borrow[arch]] && curves[borrow[arch]][precision]) { srcArch = borrow[arch]; borrowedFrom = srcArch; }
      else if (curves[arch] && !curves[arch][precision]) { return { error: precision + " not measured for this architecture" }; }
      else { return { error: "no curve for this architecture" }; }
    }
    var c = curves[srcArch][precision];

    var delta = Math.max(modelCurve(N, c), -sCap);
    var exact = [];
    if (!borrowedFrom) { for (var i = 0; i < c.anchors.length; i++) { if (Math.abs(c.anchors[i].N - N) < 1e-6) exact.push(c.anchors[i]); } }
    var spread = 0, basis;
    // a size measured only once (n=1) is an observation, not a distribution: it stays
    // "measured", but it never earns "high" confidence and says so in the notes
    var singleTrial = exact.length > 0 && exact.every(function (a) { return (a.n == null ? 2 : a.n) < 2; });
    if (exact.length) {
      // a class can pool several cards; a size measured on more than one of them gets
      // their mean, and the disagreement is carried into the band instead of hidden
      basis = "measured";
      var sum = 0, mn = exact[0].dE, mx = exact[0].dE;
      for (var k = 0; k < exact.length; k++) { sum += exact[k].dE; mn = Math.min(mn, exact[k].dE); mx = Math.max(mx, exact[k].dE); }
      delta = sum / exact.length; spread = mx - mn;
    }
    else if (borrowedFrom) { basis = "estimated"; }
    else if (N >= c.n_min && N <= c.n_max) { basis = "interpolated"; }
    else { basis = "estimated"; }

    // optional batch/context adjustment (modelled, not measured)
    var factor = savingsFactor(opts.batch, opts.ctx);
    var modelled = factor < 0.999;
    var deltaBase = delta;
    if (modelled && delta < 0) { delta = delta * factor; }   // only shrink savings

    var base = Math.max(c.resid_std, 2);
    var dd = 0; if (N < c.n_min) dd = Math.log(c.n_min / N); else if (N > c.n_max) dd = Math.log(N / c.n_max);
    var extrap = (base + 4) * dd * 1.2, bterm = borrowedFrom ? 10 : 0;
    var mterm = modelled ? Math.abs(deltaBase) * (1 - factor) * 0.5 : 0;
    var sigma = Math.sqrt(base * base + extrap * extrap + bterm * bterm + mterm * mterm);
    if (basis === "measured" && !modelled) sigma = Math.max(base * 0.5, spread / 2);

    var lo = Math.max(delta - Z * sigma, -sCap), hi = delta + Z * sigma, width = hi - lo;

    var confidence;
    // cards inside a class that disagree by more than 10 pts are not a "high" answer
    if (basis === "measured" && !modelled) confidence = (spread > 10 || singleTrial) ? "medium" : "high";
    else if (borrowedFrom) confidence = "low";
    else if (basis === "interpolated") confidence = width < 12 ? "high" : (width < 22 ? "medium" : "low");
    else confidence = (dd < 0.5 && width < 25) ? "medium" : "low";
    if (modelled && confidence === "high") confidence = "medium";
    if (modelled && factor < 0.5 && confidence === "medium") confidence = "low";

    var rec, verdict;
    if (hi < 0) { rec = "quantize"; verdict = precision + " ≈ " + delta.toFixed(0) + "% energy — saves; worth quantizing."; }
    else if (lo > 0) { rec = "do_not_quantize"; verdict = precision + " ≈ +" + delta.toFixed(0) + "% energy — costs more; keep FP16."; }
    else { rec = "depends"; verdict = precision + " ≈ " + (delta >= 0 ? "+" : "") + delta.toFixed(0) + "% energy, but the range crosses zero — near the crossover; verify on your stack."; }

    var notes = [];
    if (singleTrial) {
      var gpus = exact.map(function (a) { return a.gpu; }).filter(function (g, i, all) { return all.indexOf(g) === i; }).sort();
      notes.push("Measured once (n=1) on " + gpus.join(", ") + ": a single trial with no measured spread, so this is one observation rather than a replicated result.");
    }
    if (exact.length > 1) {
      var perCard = exact.map(function (a) { return a.gpu + " " + (a.dE >= 0 ? "+" : "") + a.dE.toFixed(0) + "%"; }).join(", ");
      notes.push(exact.length + " cards in this architecture class measured this size and disagree by " + spread.toFixed(0) + " pts (" + perCard + "); the value shown is their mean and the band covers the spread.");
    }
    if (borrowedFrom) notes.push("No measurements for this architecture; curve shape borrowed from " + (CURVES.arch_label[borrowedFrom] || borrowedFrom) + " — treat as a rough estimate.");
    if (basis === "estimated" && !borrowedFrom) notes.push("Extrapolated beyond the measured range (" + fmtNum(c.n_min) + "–" + fmtNum(c.n_max) + "B); uncertainty is wider.");
    if (modelled) notes.push("Batch/context effect is modelled (batch=" + (opts.batch || 1) + ", context=" + (opts.ctx || CTX_BASE) + ") — not yet measured; treated as savings shrinking toward zero.");
    if (c.crossover_b == null && basis !== "measured") notes.push("Crossover is not pinned down within the measured range for this architecture.");

    return {
      delta_pct: +delta.toFixed(1), ci: [+lo.toFixed(1), +hi.toFixed(1)], basis: basis, confidence: confidence,
      recommendation: rec, verdict: verdict, crossover_b: c.crossover_b, anchors: c.anchors, arch_used: srcArch,
      modelled: modelled, weight_gb: weightGB(N, precision), weight_gb_fp16: weightGB(N, "FP16"), notes: notes
    };
  }

  // ---- lightweight rules-based parser (a no-LLM "agent" entry point) -------------
  var ARCH_ALIASES = [
    ["turing", /\b(turing|t4|tesla\s*t4)\b/i],
    ["ada", /\b(ada|4090d?|rtx\s*40\d0)\b/i],
    ["blackwell", /\b(blackwell|5090|rtx\s*50\d0)\b/i],
    ["ampere", /\b(ampere|a100|a800|a40|a30|a10)\b/i],
    ["hopper", /\b(hopper|h100|h200|h800)\b/i]
  ];
  function parseQuery(text) {
    var t = " " + (text || "") + " ", out = {};
    var pm = t.match(/(\d+(?:\.\d+)?)\s*b\b/i) || t.match(/(\d+(?:\.\d+)?)\s*(?:billion|params?)/i);
    if (pm) out.N = parseFloat(pm[1]);
    for (var i = 0; i < ARCH_ALIASES.length; i++) { if (ARCH_ALIASES[i][1].test(t)) { out.arch = ARCH_ALIASES[i][0]; break; } }
    if (/\bint8\b|\b8[\s-]*bit\b/i.test(t)) out.precision = "INT8";
    else if (/\bnf4\b|\b4[\s-]*bit\b|\bq4\b/i.test(t)) out.precision = "NF4";
    var bm = t.match(/\b(?:batch|bs|b)\s*[=:]?\s*(\d+)\b/i) || t.match(/\bbatch\s*size\s*(\d+)/i);
    if (bm) out.batch = parseInt(bm[1], 10);
    var cm = t.match(/\b(?:ctx|context|seq(?:len)?|sequence)\s*[=:]?\s*(\d+)\s*(k)?\b/i) || t.match(/\b(\d+)\s*k\s*(?:ctx|context|tokens?)\b/i);
    if (cm) out.ctx = parseInt(cm[1], 10) * (cm[2] ? 1024 : 1);
    return out;
  }

  var API = { estimate: estimate, parseQuery: parseQuery, weightGB: weightGB, savingsFactor: savingsFactor, modelCurve: modelCurve, fmtNum: fmtNum, fp16BaselineJ1k: fp16BaselineJ1k, CURVES: CURVES };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.EcoEstimator = API;
  // convenience globals for the demo page
  root.CURVES = CURVES; root.estimate = estimate; root.parseQuery = parseQuery; root.weightGB = weightGB;
  root.modelCurve = modelCurve; root.fmtNum = fmtNum; root.fp16BaselineJ1k = fp16BaselineJ1k;
})(typeof globalThis !== "undefined" ? globalThis : this);
