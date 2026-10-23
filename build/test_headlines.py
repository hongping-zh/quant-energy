#!/usr/bin/env python3
"""Headline-consistency test: what the site claims in prose must be what the
estimator actually answers.

Born from quant-energy PR #5 (July 2026): the site said "7B NF4 saves ~29% on
Ada" while every estimator path answered +20% "do not quantize" — the curves
had been refit but the anchors never reached the paths that answer users. Two
guards, both run here:

  1. estimate.py (the reference) answers the headline claims with the right
     basis, sign and recommendation;
  2. estimate.js (the browser tab / Worker / MCP path) gives the same numbers —
     node runs it and Python asserts the answers agree.

Exit code 0 == consistent.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import estimate as est  # noqa: E402

JS = r"""
const est = require(process.argv[1]);
const q = JSON.parse(process.argv[2]);
const r = est.estimate(q.N, q.arch, q.precision);
console.log(JSON.stringify({delta_pct: r.delta_pct, basis: r.basis,
                            recommendation: r.recommendation}));
"""


def js_estimate(N, arch, precision):
    out = subprocess.check_output(
        ["node", "-e", JS, os.path.join(ROOT, "estimate.js"),
         json.dumps({"N": N, "arch": arch, "precision": precision})],
        text=True)
    return json.loads(out)


# (N, arch, precision, basis, sign (-1 saving / +1 penalty / 0 either),
#  recommendation, why this is a site claim)
CLAIMS = [
    (7.0, "ada", "NF4", "measured", -1, "quantize",
     "headline: NF4 saves ~29% energy at 7B on Ada (July 2026 RTX 4090 run)"),
    (7.0, "ada", "INT8", "measured", +1, "do_not_quantize",
     "claim: LLM.int8() costs energy at 7B on Ada (+50%)"),
    (3.0, "ada", "NF4", "measured", 0, "depends",
     "crossover cell: 3B on Ada is break-even for NF4 "
     "(July anchor stands; 2026-09-25 re-test mean +1.6%, not pooled by design)"),
]

DRIFT_TOL = 0.05  # percentage points between estimate.py and estimate.js


def check(claim):
    N, arch, prec, basis_want, sign_want, rec_want, why = claim
    py = est.estimate(N, arch, prec)
    js = js_estimate(N, arch, prec)
    errs = []
    for name, r in (("estimate.py", py), ("estimate.js", js)):
        if r.get("basis") != basis_want:
            errs.append("%s basis=%r, want %r (%s)" % (name, r.get("basis"), basis_want, why))
        if sign_want == -1 and not r["delta_pct"] < 0:
            errs.append("%s delta_pct=%s but the site claims a SAVING (%s)"
                        % (name, r["delta_pct"], why))
        if sign_want == +1 and not r["delta_pct"] > 0:
            errs.append("%s delta_pct=%s but the site claims a PENALTY (%s)"
                        % (name, r["delta_pct"], why))
        if r.get("recommendation") != rec_want:
            errs.append("%s recommendation=%r, want %r (%s)"
                        % (name, r.get("recommendation"), rec_want, why))
    if abs(py["delta_pct"] - js["delta_pct"]) > DRIFT_TOL:
        errs.append("Python/JS drift: %s vs %s (%s)"
                    % (py["delta_pct"], js["delta_pct"], why))
    return errs


def main():
    bad = []
    for claim in CLAIMS:
        bad += check(claim)
    if bad:
        for e in bad:
            print("FAIL: %s" % e, file=sys.stderr)
        return 1
    print("headline consistency: %d claims OK on estimate.py and estimate.js" % len(CLAIMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
