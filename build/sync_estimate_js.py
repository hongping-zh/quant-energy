#!/usr/bin/env python3
"""
Regenerate the inline CURVES literal in ../estimate.js from curves.json.

estimate.js must stay dependency-free (no fetch, no bundler), so it carries a copy of
the calibration. Run this after every fit_curves.py run; build/test_optimize.py fails on
any Python/JS drift.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CURVES = os.path.join(ROOT, "curves.json")
TARGET = os.path.join(ROOT, "estimate.js")

BEGIN = "  var CURVES = {"
END = "\n  };\n"


def dumps(obj, indent):
    """JSON with the leaf dicts kept on one line, like the hand-written original."""
    return json.dumps(obj, indent=indent).replace('"', '"')


def anchors_line(anchors):
    items = [json.dumps(a, separators=(", ", ": ")) for a in anchors]
    return "[" + ", ".join(items) + "]"


def main():
    c = json.load(open(CURVES))
    L = []
    L.append(BEGIN)
    L.append('    "s_cap": %s,' % json.dumps(c["s_cap"], separators=(", ", ": ")))
    L.append('    "arch_label": %s,' % json.dumps(c["arch_label"], indent=6).replace("\n}", "\n    }"))
    L.append('    "borrow": %s,' % json.dumps(c["borrow"], separators=(", ", ": ")))
    L.append('    "curves": {')
    archs = list(c["curves"])
    for ai, arch in enumerate(archs):
        L.append('      "%s": {' % arch)
        precs = list(c["curves"][arch])
        for pi, prec in enumerate(precs):
            g = dict(c["curves"][arch][prec])
            anchors = g.pop("anchors")
            head = ", ".join('"%s": %s' % (k, json.dumps(v)) for k, v in g.items())
            L.append('        "%s": { %s,' % (prec, head))
            L.append('          "anchors": %s }%s' % (anchors_line(anchors), "" if pi == len(precs) - 1 else ","))
        L.append('      }%s' % ("" if ai == len(archs) - 1 else ","))
    L.append('    },')
    L.append('    // Measured FP16 absolute decode energy (J / 1k tokens), per arch — anchors the')
    L.append("    // optimizer's absolute-energy numbers. Mirror of curves.json:fp16_energy.")
    L.append('    "fp16_energy": {')
    archs = list(c["fp16_energy"])
    for i, arch in enumerate(archs):
        t = c["fp16_energy"][arch]
        pts = ", ".join('{ "N": %s, "e_j1k": %s }' % (json.dumps(a["N"]), json.dumps(a["e_j1k"]))
                        for a in t["anchors"])
        L.append('      "%s": { "n_min": %s, "n_max": %s, "anchors": [%s] }%s'
                 % (arch, json.dumps(t["n_min"]), json.dumps(t["n_max"]), pts,
                    "" if i == len(archs) - 1 else ","))
    L.append('    },')
    L.append('    // Public datasheet GPU specs — used ONLY by the (modelled) roofline latency')
    L.append('    // layer in optimize.js. Mirror of curves.json:hardware.')
    L.append('    "hardware": {')
    archs = list(c["hardware"])
    for i, arch in enumerate(archs):
        L.append('      "%s": %s%s' % (arch, json.dumps(c["hardware"][arch], separators=(", ", ": ")),
                                       "" if i == len(archs) - 1 else ","))
    L.append('    }')
    block = "\n".join(L) + END

    src = open(TARGET).read()
    start = src.index(BEGIN)
    end = src.index(END, start) + len(END)
    open(TARGET, "w").write(src[:start] + block + src[end:])
    print("synced CURVES in %s from %s" % (TARGET, CURVES))


if __name__ == "__main__":
    main()
