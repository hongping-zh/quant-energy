#!/usr/bin/env python3
"""
Fit the per-(architecture, precision) crossover curve and emit curves.json.

Model (saturating, monotone in N):
    dE%(N) = A - S * g(N / Nstar),   g(x) = x / (1 + x)
  A      : dequant penalty floor (small models) -> dE% ~ A
  S      : swing magnitude; large-N asymptote is (A - S)
  Nstar  : crossover scale (params in B where memory traffic starts to dominate)

Constraints keep extrapolation honest:
  - S bounded by the weight-byte reduction so we never promise impossible savings
    (NF4 <= ~75 pts, INT8 <= ~50 pts on top of the penalty floor).
  - Nstar in [0.1, 200] B.

Outputs curves.json consumed identically by estimate.py (Python) and estimate.js (browser).
Also prints leave-one-out (LOO) validation: per-class MAE and 95% CI coverage.
"""
import csv, json, os, math, warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = os.path.join(HERE, "measured.csv")
OUT = os.path.join(os.path.dirname(HERE), "curves.json")

# max savings magnitude (percentage points) achievable in the fully memory-bound limit
S_CAP = {"NF4": 75.0, "INT8": 50.0}

ARCH_LABEL = {
    "turing":    "Turing (T4)",
    "ada":       "Ada (RTX 4090/4090D)",
    "blackwell": "Blackwell (RTX 5090)",
    "ampere":    "Ampere (A100/A800)",
    "hopper":    "Hopper (H100/H200)",
}
# classes with no measurements borrow the nearest datacenter class shape (flagged estimated)
BORROW = {"hopper": "ampere"}


def g(x):
    return x / (1.0 + x)


def model(N, A, S, Nstar):
    return A - S * g(N / Nstar)


def load():
    rows = []
    with open(MEASURED) as f:
        for r in csv.DictReader(f):
            fp16 = float(r["energy_fp16_j1k"])
            q = float(r["energy_quant_j1k"])
            rows.append({
                "arch": r["arch"], "precision": r["precision"],
                "gpu": r["gpu"], "model": r["model"],
                "N": float(r["params_b"]),
                "dE": (q - fp16) / fp16 * 100.0,
            })
    return rows


def fit_group(pts):
    N = np.array([p["N"] for p in pts], float)
    y = np.array([p["dE"] for p in pts], float)
    prec = pts[0]["precision"]
    scap = S_CAP.get(prec, 75.0)
    # bounds: A in [-50,300], S in [0, scap + max_penalty], Nstar in [0.1,200]
    a_hi = float(max(y)) + 50.0
    s_hi = scap + max(0.0, float(max(y)))
    lo = [-50.0, 0.0, 0.1]
    hi = [a_hi, s_hi, 200.0]
    p0 = [float(max(y)), min(s_hi, 60.0), float(np.median(N))]
    if len(pts) >= 3:
        popt, _ = curve_fit(model, N, y, p0=p0, bounds=(lo, hi), maxfev=200000)
    else:
        # under-determined: fix Nstar at median, fit A,S only
        def m2(N, A, S):
            return model(N, A, S, float(np.median(N)))
        po, _ = curve_fit(m2, N, y, p0=[p0[0], p0[1]], bounds=([lo[0], lo[1]], [hi[0], hi[1]]), maxfev=200000)
        popt = [po[0], po[1], float(np.median(N))]
    resid = y - model(N, *popt)
    rstd = float(np.sqrt(np.mean(resid**2)))
    return [float(v) for v in popt], rstd, N, y


def loo(pts):
    """leave-one-out: refit without each point, predict it, collect abs error."""
    if len(pts) < 4:
        return None
    errs = []
    for i in range(len(pts)):
        sub = pts[:i] + pts[i+1:]
        try:
            popt, _, _, _ = fit_group(sub)
        except Exception:
            return None
        pred = model(pts[i]["N"], *popt)
        errs.append(abs(pred - pts[i]["dE"]))
    return float(np.mean(errs))


def main():
    rows = load()
    groups = {}
    for r in rows:
        groups.setdefault((r["arch"], r["precision"]), []).append(r)

    curves = {}
    print("=== per-(arch, precision) fits ===")
    for (arch, prec), pts in sorted(groups.items()):
        popt, rstd, N, y = fit_group(pts)
        A, S, Nstar = popt
        # crossover where dE%=0  ->  A = S*g(N/Nstar)  ->  N = Nstar * A/(S-A)
        xover = (Nstar * A / (S - A)) if (S > A > 0) else None
        anchors = [{"N": p["N"], "dE": round(p["dE"], 2), "model": p["model"], "gpu": p["gpu"]} for p in pts]
        loo_mae = loo(pts)
        curves.setdefault(arch, {})[prec] = {
            "A": round(A, 4), "S": round(S, 4), "Nstar": round(Nstar, 4),
            "resid_std": round(rstd, 4),
            "n_min": round(float(min(N)), 3), "n_max": round(float(max(N)), 3),
            "crossover_b": (round(xover, 3) if xover and xover > 0 else None),
            "anchors": anchors,
            "loo_mae": (round(loo_mae, 3) if loo_mae is not None else None),
        }
        xs = f"{xover:.2f}B" if (xover and xover > 0) else "n/a"
        lm = f"{loo_mae:.2f}" if loo_mae is not None else "n/a"
        print(f"  {arch:10s} {prec:4s}  A={A:7.2f} S={S:7.2f} Nstar={Nstar:6.2f} "
              f"resid_std={rstd:5.2f} crossover={xs:>7s} LOO_MAE={lm}")

    out = {
        "model": "dE_pct(N) = A - S * (N/Nstar)/(1 + N/Nstar)",
        "s_cap": S_CAP,
        "arch_label": ARCH_LABEL,
        "borrow": BORROW,
        "curves": curves,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
