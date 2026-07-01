#!/usr/bin/env python3
"""
Reference config optimizer (Python). ../optimize.js mirrors this logic 1:1.

optimize(arch, params_b, objective, max_latency_ms, max_vram_gb, min_throughput,
         precisions=None, batches=None, contexts=None, batch=None, context=None)
  -> dict with recommended config, alternatives, and the energy<->latency Pareto front.

The config grid (precision x batch x context) is small, so the search is
EXHAUSTIVE => globally optimal. Honesty (per-field basis) matches optimize.js:
  energy   = MEASURED-anchored (dE% x measured FP16 baseline, curves.json:fp16_energy)
  vram     = COMPUTED (weights + KV cache)
  latency  = MODELLED (roofline; curves.json:hardware datasheet specs)
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES_PATH = os.path.join(os.path.dirname(HERE), "curves.json")

# modelling constants (documented, not measured)
MBU = 0.70
COMPUTE_UTIL = 0.30
KV_ELEM_BYTES = 2
BATCH_ENERGY_FIXED = 0.35
BITS = {"FP16": 16, "INT8": 8, "NF4": 4}

DEFAULT_PRECISIONS = ["FP16", "NF4", "INT8"]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_CONTEXTS = [512, 1024, 2048, 4096, 8192]

_BASIS_RANK = {"measured": 3, "interpolated": 2, "estimated": 1, "computed": 3, "modelled": 1}


def _worse_basis(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if _BASIS_RANK[a] <= _BASIS_RANK[b] else b


# import the point estimator from the same build dir
import importlib.util
_spec = importlib.util.spec_from_file_location("eco_estimate", os.path.join(HERE, "estimate.py"))
_estmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_estmod)
estimate = _estmod.estimate


def _load():
    with open(CURVES_PATH) as f:
        return json.load(f)


def arch_shape(N):
    tbl = [(0.5, 24, 896), (1.1, 22, 2048), (1.5, 28, 1536), (3.0, 36, 2048),
           (7.0, 32, 4096), (9.0, 48, 4096), (14.0, 48, 5120), (32.0, 64, 6656), (70.0, 80, 8192)]
    if N <= tbl[0][0]:
        return tbl[0][1], tbl[0][2]
    for i in range(1, len(tbl)):
        if N <= tbl[i][0]:
            a, b = tbl[i - 1], tbl[i]
            t = (N - a[0]) / (b[0] - a[0])
            return round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t)
    return tbl[-1][1], tbl[-1][2]


def vram_gb(N, precision, ctx, batch):
    bits = BITS.get(precision, 16)
    weights = N * 1e9 * bits / 8
    L, H = arch_shape(N)
    kv = 2 * L * H * ctx * batch * KV_ELEM_BYTES
    overhead = 0.8e9 + weights * 0.05
    return round((weights + kv + overhead) / 1e9, 2)


def fp16_base_j1k(N, arch, data):
    fe = data.get("fp16_energy", {}).get(arch)
    if not fe or not fe.get("anchors"):
        return None
    a = fe["anchors"]
    if N <= a[0]["N"]:
        return round(a[0]["e_j1k"] * (N / a[0]["N"]), 2)
    for i in range(1, len(a)):
        if N <= a[i]["N"]:
            p, q = a[i - 1], a[i]
            t = (N - p["N"]) / (q["N"] - p["N"])
            return round(p["e_j1k"] + (q["e_j1k"] - p["e_j1k"]) * t, 2)
    last, prev = a[-1], a[-2]
    slope = (last["e_j1k"] - prev["e_j1k"]) / (last["N"] - prev["N"])
    return round(last["e_j1k"] + slope * (N - last["N"]), 2)


def fp16_base_basis(N, arch, data):
    fe = data.get("fp16_energy", {}).get(arch)
    if not fe:
        return None
    for an in fe["anchors"]:
        if abs(an["N"] - N) < 1e-6:
            return "measured"
    return "interpolated" if fe["n_min"] <= N <= fe["n_max"] else "estimated"


def roofline(N, precision, ctx, batch, arch, data):
    hw = data.get("hardware", {}).get(arch)
    if not hw:
        return None
    bits = BITS.get(precision, 16)
    weight_bytes = N * 1e9 * bits / 8
    L, H = arch_shape(N)
    kv_per_seq = 2 * L * H * ctx * KV_ELEM_BYTES
    bw = hw["mem_bw_gbps"] * 1e9 * MBU
    bw_step = (weight_bytes + batch * kv_per_seq) / bw
    flops = hw["fp16_tflops"] * 1e12 * COMPUTE_UTIL
    compute_step = batch * (2 * N * 1e9) / flops
    step = max(bw_step, compute_step)
    return {
        "latency_ms_per_token": round(step * 1000, 2),
        "throughput_tok_s": round(batch / step, 1),
        "bound": "compute" if compute_step > bw_step else "memory",
    }


def score_candidate(N, arch, precision, batch, ctx, data):
    if precision == "FP16":
        est = {"delta_pct": 0.0, "basis": "measured", "confidence": "high", "modelled": False}
    else:
        est = estimate(N, arch, precision, data)
        if "error" in est:
            return {"error": est["error"]}

    base_j1k = fp16_base_j1k(N, arch, data)
    base_basis = fp16_base_basis(N, arch, data)
    delta = est["delta_pct"]

    prec_factor = 1 + delta / 100.0
    amort = (1 - BATCH_ENERGY_FIXED) + BATCH_ENERGY_FIXED / batch
    energy_j1k = round(base_j1k * prec_factor * amort, 1) if base_j1k is not None else None
    mj_tok = round(energy_j1k / 1000.0, 3) if energy_j1k is not None else None

    vram = vram_gb(N, precision, ctx, batch)
    rl = roofline(N, precision, ctx, batch, arch, data)

    energy_modelled = batch > 1
    energy_basis = _worse_basis(base_basis if precision == "FP16" else est["basis"], base_basis)
    energy_conf = est.get("confidence", "medium")
    if energy_modelled and energy_conf == "high":
        energy_conf = "medium"
    overall = "medium" if energy_conf == "high" else energy_conf

    return {
        "precision": precision, "batch": batch, "context": ctx,
        "energy_j_per_1k_tokens": energy_j1k,
        "energy_mj_per_token": mj_tok,
        "energy_rel_fp16": round(prec_factor, 3),
        "delta_pct_vs_fp16": round(delta, 1),
        "latency_ms_per_token": rl["latency_ms_per_token"] if rl else None,
        "throughput_tok_s": rl["throughput_tok_s"] if rl else None,
        "bound": rl["bound"] if rl else None,
        "vram_gb": vram,
        "basis": {"energy": energy_basis, "latency": "modelled (roofline)", "vram": "computed"},
        "confidence": overall,
        "recommendation": est.get("recommendation", "baseline" if precision == "FP16" else None),
    }


def _feasible(c, cons):
    if cons["max_vram_gb"] is not None and c["vram_gb"] > cons["max_vram_gb"]:
        return False
    if cons["max_latency_ms"] is not None and c["latency_ms_per_token"] is not None and c["latency_ms_per_token"] > cons["max_latency_ms"]:
        return False
    if cons["min_throughput"] is not None and c["throughput_tok_s"] is not None and c["throughput_tok_s"] < cons["min_throughput"]:
        return False
    return True


def _violation(c, cons):
    v = 0.0
    if cons["max_vram_gb"] is not None and c["vram_gb"] > cons["max_vram_gb"]:
        v += (c["vram_gb"] - cons["max_vram_gb"]) / cons["max_vram_gb"]
    if cons["max_latency_ms"] is not None and c["latency_ms_per_token"] > cons["max_latency_ms"]:
        v += (c["latency_ms_per_token"] - cons["max_latency_ms"]) / cons["max_latency_ms"]
    if cons["min_throughput"] is not None and c["throughput_tok_s"] < cons["min_throughput"]:
        v += (cons["min_throughput"] - c["throughput_tok_s"]) / cons["min_throughput"]
    return v


def _obj_key(objective):
    if objective == "min_latency":
        return lambda c: c["latency_ms_per_token"] if c["latency_ms_per_token"] is not None else math.inf
    if objective == "max_throughput":
        return lambda c: -c["throughput_tok_s"] if c["throughput_tok_s"] is not None else math.inf
    if objective == "min_vram":
        return lambda c: c["vram_gb"]
    return lambda c: c["energy_j_per_1k_tokens"] if c["energy_j_per_1k_tokens"] is not None else c["energy_rel_fp16"] * 1e9


def pareto_front(cands):
    pts = [c for c in cands if c["energy_j_per_1k_tokens"] is not None and c["latency_ms_per_token"] is not None]
    front = []
    for c in pts:
        dominated = any(
            o is not c
            and o["energy_j_per_1k_tokens"] <= c["energy_j_per_1k_tokens"]
            and o["latency_ms_per_token"] <= c["latency_ms_per_token"]
            and (o["energy_j_per_1k_tokens"] < c["energy_j_per_1k_tokens"] or o["latency_ms_per_token"] < c["latency_ms_per_token"])
            for o in pts
        )
        if not dominated:
            front.append(c)
    front.sort(key=lambda c: c["energy_j_per_1k_tokens"])
    return [{"precision": c["precision"], "batch": c["batch"], "context": c["context"],
             "energy_j_per_1k_tokens": c["energy_j_per_1k_tokens"], "latency_ms_per_token": c["latency_ms_per_token"],
             "throughput_tok_s": c["throughput_tok_s"], "vram_gb": c["vram_gb"]} for c in front]


def _pct(a, b):
    return round((a - b) / b * 100, 1) if b else None


def _decorate(c, role):
    o = dict(c)
    o["role"] = role
    parts = [f"{c['precision']}, batch={c['batch']}, context={c['context']}"]
    if c["energy_j_per_1k_tokens"] is not None:
        parts.append(f"{c['energy_j_per_1k_tokens']} J/1k tok")
    if c["throughput_tok_s"] is not None:
        parts.append(f"{c['throughput_tok_s']} tok/s")
    if c["latency_ms_per_token"] is not None:
        parts.append(f"{c['latency_ms_per_token']} ms/tok")
    parts.append(f"{c['vram_gb']} GB")
    o["summary"] = ", ".join(parts)
    return o


def optimize(arch, params_b, objective="min_energy", max_latency_ms=None, max_vram_gb=None,
             min_throughput=None, precisions=None, batches=None, contexts=None,
             batch=None, context=None, data=None):
    data = data or _load()
    N = float(params_b)
    arch = str(arch).lower()
    if not N > 0:
        return {"error": "params_b (billions) required and must be > 0"}
    if arch not in data["curves"] and arch not in data.get("borrow", {}):
        return {"error": f"no data for arch '{arch}'", "supported": list(data["curves"].keys())}

    precisions = precisions or DEFAULT_PRECISIONS
    batches = batches or ([int(batch)] if batch is not None else DEFAULT_BATCHES)
    contexts = contexts or ([int(context)] if context is not None else DEFAULT_CONTEXTS)
    cons = {"max_latency_ms": max_latency_ms, "max_vram_gb": max_vram_gb, "min_throughput": min_throughput}

    all_c, errors = [], {}
    for p in precisions:
        for b in batches:
            for c in contexts:
                s = score_candidate(N, arch, p, b, c, data)
                if "error" in s:
                    errors[p] = s["error"]
                    continue
                all_c.append(s)
    if not all_c:
        return {"error": "no candidates could be scored", "detail": errors}

    key = _obj_key(objective)
    all_c.sort(key=lambda c: (key(c), c["energy_j_per_1k_tokens"] or 0, c["vram_gb"], c["latency_ms_per_token"] or 0))
    feas = [c for c in all_c if _feasible(c, cons)]
    pareto = pareto_front(all_c)

    result = {
        "input": {"params_b": N, "arch": arch, "objective": objective, "constraints": cons,
                  "grid": {"precisions": precisions, "batches": batches, "contexts": contexts, "candidates": len(all_c)}},
        "basis": {"energy": "measured-anchored (dE% x measured FP16 baseline)",
                  "latency": "modelled (roofline; datasheet mem-bandwidth/compute)",
                  "vram": "computed (weights + KV cache)"},
        "feasible_count": len(feas),
        "notes": [],
    }

    def add_notes():
        result["notes"].append("Energy is measured-anchored; latency & throughput are a roofline MODEL (not measured); VRAM is a standard calculation.")
        result["notes"].append("Roofline latency assumes memory-bound decode and ignores dequantization compute overhead, so quantized-format latency is a best case (lower bound). Energy — which does capture that overhead — is the measured signal.")
        if not data.get("fp16_energy", {}).get(arch):
            result["notes"].append("No measured FP16 energy baseline for this architecture — energy is relative only.")
        result["notes"].append(f"Batch>1 energy uses a conservative amortization model (fixed-overhead fraction {BATCH_ENERGY_FIXED}).")

    if not feas:
        closest = sorted(all_c, key=lambda c: _violation(c, cons))[0]
        result["recommended"] = None
        result["closest_infeasible"] = _decorate(closest, "closest (violates constraints)")
        result["notes"].append("No configuration satisfies all constraints. Showing the closest; consider relaxing the binding budget.")
        add_notes()
        return result

    best = feas[0]
    result["recommended"] = _decorate(best, "objective-optimal under the given constraints")
    alts = []
    for c in feas[1:4]:
        d = _decorate(c, "alternative")
        d["vs_recommended"] = {
            "energy_pct": _pct(c["energy_j_per_1k_tokens"], best["energy_j_per_1k_tokens"]) if c["energy_j_per_1k_tokens"] and best["energy_j_per_1k_tokens"] else None,
            "latency_pct": _pct(c["latency_ms_per_token"], best["latency_ms_per_token"]),
            "throughput_pct": _pct(c["throughput_tok_s"], best["throughput_tok_s"]),
            "vram_pct": _pct(c["vram_gb"], best["vram_gb"]),
        }
        alts.append(d)
    result["alternatives"] = alts
    result["pareto_energy_latency"] = pareto
    add_notes()
    return result


if __name__ == "__main__":
    data = _load()
    print("=== RTX 4090D (ada) Qwen2-7B, latency<50ms/tok, vram<16GB, min energy ===")
    r = optimize("ada", 7, "min_energy", max_latency_ms=50, max_vram_gb=16, data=data)
    print(" recommended:", r["recommended"]["summary"])
    print(" basis:", r["recommended"]["basis"], "conf:", r["recommended"]["confidence"])
    for a in r["alternatives"]:
        print("  alt:", a["summary"], "vs_rec:", a["vs_recommended"])
