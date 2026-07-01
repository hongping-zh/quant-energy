#!/usr/bin/env python3
"""Parity + sanity test for the optimizer: optimize.py (reference) vs optimize.js.

Runs the same scenarios through both implementations and asserts the recommended
config, alternatives and Pareto front match numerically (within rounding), plus a
few invariants (feasibility, monotonicity). Exit code 0 == all good.
"""
import json, os, subprocess, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import optimize as opt  # noqa: E402

SCENARIOS = [
    {"arch": "ada", "params_b": 7, "objective": "min_energy", "max_latency_ms": 50, "max_vram_gb": 16},
    {"arch": "blackwell", "params_b": 7, "objective": "min_energy", "max_vram_gb": 32},
    {"arch": "turing", "params_b": 3, "objective": "max_throughput", "max_vram_gb": 16},
    {"arch": "ampere", "params_b": 14, "objective": "min_latency", "max_vram_gb": 80},
    {"arch": "blackwell", "params_b": 1.5, "objective": "min_vram"},
    {"arch": "ada", "params_b": 7, "objective": "min_energy", "max_vram_gb": 4},  # infeasible
    {"arch": "ampere", "params_b": 7, "objective": "min_energy", "context": 4096, "batch": 8},  # pinned
]

JS = r"""
const {optimize}=require(process.argv[1]);
const req=JSON.parse(process.argv[2]);
console.log(JSON.stringify(optimize(req)));
"""


def run_js(req):
    out = subprocess.check_output(
        ["node", "-e", JS, os.path.join(ROOT, "optimize.js"), json.dumps(req)],
        text=True)
    return json.loads(out)


NUM_FIELDS = ["energy_j_per_1k_tokens", "energy_mj_per_token", "energy_rel_fp16",
              "delta_pct_vs_fp16", "latency_ms_per_token", "throughput_tok_s", "vram_gb"]


def close(a, b, tol=0.05):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol + 1e-4 * max(abs(a), abs(b)) * 100


def cmp_cfg(tag, pc, jc, fails):
    for k in ["precision", "batch", "context"]:
        if pc.get(k) != jc.get(k):
            fails.append(f"{tag}: {k} py={pc.get(k)} js={jc.get(k)}")
    for k in NUM_FIELDS:
        if not close(pc.get(k), jc.get(k)):
            fails.append(f"{tag}: {k} py={pc.get(k)} js={jc.get(k)}")
    if pc.get("basis") != jc.get("basis"):
        fails.append(f"{tag}: basis py={pc.get('basis')} js={jc.get('basis')}")
    if pc.get("confidence") != jc.get("confidence"):
        fails.append(f"{tag}: confidence py={pc.get('confidence')} js={jc.get('confidence')}")


def main():
    data = opt._load()
    fails = []
    for i, sc in enumerate(SCENARIOS):
        pr = opt.optimize(data=data, **sc)
        jr = run_js(sc)
        tag = f"[{i}] {sc.get('arch')} {sc.get('params_b')}B {sc.get('objective')}"

        if pr["feasible_count"] != jr["feasible_count"]:
            fails.append(f"{tag}: feasible_count py={pr['feasible_count']} js={jr['feasible_count']}")

        prec, jrec = pr.get("recommended"), jr.get("recommended")
        if (prec is None) != (jrec is None):
            fails.append(f"{tag}: recommended presence mismatch")
        elif prec is not None:
            cmp_cfg(tag + " rec", prec, jrec, fails)
            # invariant: recommended must satisfy constraints
            c = sc
            if c.get("max_vram_gb") is not None and prec["vram_gb"] > c["max_vram_gb"] + 1e-6:
                fails.append(f"{tag}: recommended VRAM {prec['vram_gb']} exceeds budget {c['max_vram_gb']}")
            if c.get("max_latency_ms") is not None and prec["latency_ms_per_token"] > c["max_latency_ms"] + 1e-6:
                fails.append(f"{tag}: recommended latency exceeds budget")
        else:
            cmp_cfg(tag + " closest", pr["closest_infeasible"], jr["closest_infeasible"], fails)

        pa, ja = pr.get("alternatives", []), jr.get("alternatives", [])
        if len(pa) != len(ja):
            fails.append(f"{tag}: alt count py={len(pa)} js={len(ja)}")
        for k, (p, j) in enumerate(zip(pa, ja)):
            cmp_cfg(tag + f" alt{k}", p, j, fails)

        pp, jp = pr.get("pareto_energy_latency", []), jr.get("pareto_energy_latency", [])
        if len(pp) != len(jp):
            fails.append(f"{tag}: pareto len py={len(pp)} js={len(jp)}")
        # invariant: Pareto front strictly improves in latency as energy rises
        for a, b in zip(pp, pp[1:]):
            if not (a["latency_ms_per_token"] >= b["latency_ms_per_token"] - 1e-6):
                fails.append(f"{tag}: pareto not sorted/monotone")
                break
        print(f"{tag}: feas={pr['feasible_count']} rec={prec['summary'] if prec else '(infeasible)'} pareto={len(pp)}")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\nAll optimizer parity + invariant checks passed.")


if __name__ == "__main__":
    main()
