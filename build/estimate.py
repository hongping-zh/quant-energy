#!/usr/bin/env python3
"""
Reference estimator (Python). estimate.js in the demo mirrors this logic 1:1.

estimate(N, arch, precision) -> dict with:
  delta_pct, ci=[lo,hi], basis (measured|interpolated|estimated),
  confidence (high|medium|low), anchors, recommendation, verdict, notes
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(os.path.dirname(HERE), "curves.json")
Z = 1.96  # ~95%


def _load():
    with open(CURVES) as f:
        return json.load(f)


def _g(x):
    return x / (1.0 + x)


def _model(N, c):
    return c["A"] - c["S"] * _g(N / c["Nstar"])


def estimate(N, arch, precision, data=None):
    data = data or _load()
    curves = data["curves"]
    s_cap = data["s_cap"].get(precision, 75.0)
    borrow = data.get("borrow", {})

    borrowed_from = None
    src_arch = arch
    if arch not in curves or precision not in curves.get(arch, {}):
        if arch in borrow and precision in curves.get(borrow[arch], {}):
            src_arch = borrow[arch]
            borrowed_from = src_arch
        elif arch in curves and precision not in curves[arch]:
            return {"error": f"{precision} not measured for {arch}",
                    "supported_precisions": list(curves[arch].keys())}
        else:
            return {"error": f"no curve for arch '{arch}'",
                    "supported_archs": list(curves.keys())}

    c = curves[src_arch][precision]
    delta = _model(N, c)
    delta = max(delta, -s_cap)  # never promise more savings than the weight-byte bound

    # basis
    exact = None
    if borrowed_from is None:
        for a in c["anchors"]:
            if abs(a["N"] - N) < 1e-6:
                exact = a
                break
    if exact is not None:
        basis = "measured"
        delta = exact["dE"]  # return the measured value, not the smoothed curve
    elif borrowed_from is not None:
        basis = "estimated"
    elif c["n_min"] <= N <= c["n_max"]:
        basis = "interpolated"
    else:
        basis = "estimated"

    # uncertainty: residual floor + extrapolation distance (log) + borrowed penalty
    base = max(c["resid_std"], 2.0)
    if N < c["n_min"]:
        dd = math.log(c["n_min"] / N)
    elif N > c["n_max"]:
        dd = math.log(N / c["n_max"])
    else:
        dd = 0.0
    extrap = (base + 4.0) * dd * 1.2
    borrowed_term = 10.0 if borrowed_from is not None else 0.0
    sigma = math.sqrt(base**2 + extrap**2 + borrowed_term**2)
    if basis == "measured":
        sigma = base * 0.5

    lo = max(delta - Z * sigma, -s_cap)
    hi = delta + Z * sigma
    width = hi - lo

    # confidence
    if basis == "measured":
        confidence = "high"
    elif borrowed_from is not None:
        confidence = "low"
    elif basis == "interpolated":
        confidence = "high" if width < 12 else ("medium" if width < 22 else "low")
    else:  # estimated/extrapolated
        confidence = "medium" if (dd < 0.5 and width < 25) else "low"

    # recommendation from the band's relation to zero
    if hi < 0:
        recommendation = "quantize"
        verdict = f"{precision} ~ {delta:.0f}% energy — saves; worth quantizing."
    elif lo > 0:
        recommendation = "do_not_quantize"
        verdict = f"{precision} ~ +{delta:.0f}% energy — costs more; keep FP16."
    else:
        recommendation = "depends"
        verdict = (f"{precision} ~ {delta:+.0f}% energy, but the range crosses zero — "
                   f"near the crossover; verify on your stack.")

    notes = []
    if borrowed_from is not None:
        notes.append(f"No measurements for this architecture; shape borrowed from "
                     f"'{borrowed_from}' — treat as a rough estimate.")
    if basis == "estimated" and borrowed_from is None:
        rng = f"{c['n_min']:g}-{c['n_max']:g}B"
        notes.append(f"Extrapolated beyond the measured range ({rng}); wider uncertainty.")
    if c.get("crossover_b") is None and basis != "measured":
        notes.append("Crossover not pinned down within the measured range for this architecture.")

    return {
        "delta_pct": round(delta, 1),
        "ci": [round(lo, 1), round(hi, 1)],
        "basis": basis,
        "confidence": confidence,
        "recommendation": recommendation,
        "verdict": verdict,
        "crossover_b": c.get("crossover_b"),
        "anchors": c["anchors"],
        "arch_used": src_arch,
        "notes": notes,
    }


if __name__ == "__main__":
    import sys
    data = _load()
    tests = [
        (13, "ampere", "NF4"), (5, "blackwell", "NF4"), (0.7, "blackwell", "NF4"),
        (70, "ada", "NF4"), (3, "ada", "NF4"), (13, "hopper", "NF4"),
        (7, "ampere", "INT8"), (2, "turing", "NF4"), (7, "blackwell", "NF4"),
    ]
    for N, a, p in tests:
        r = estimate(N, a, p, data)
        print(f"\n{N}B {a} {p}:")
        print(f"  delta={r['delta_pct']}%  ci={r['ci']}  basis={r['basis']}  "
              f"conf={r['confidence']}  rec={r['recommendation']}")
        for n in r["notes"]:
            print(f"   - {n}")
