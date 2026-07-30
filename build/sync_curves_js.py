#!/usr/bin/env python3
"""
Mirror curves.json into the CURVES literal embedded in ../estimate.js.

estimate.js ships as a single dependency-free file (the browser, the Cloudflare
Worker and the MCP server all import it), so the calibration has to live inside it.
This keeps that copy from drifting away from curves.json after a refit:

    python3 build/fit_curves.py && python3 build/sync_curves_js.py

Run with --check to fail instead of writing (useful in CI).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CURVES = os.path.join(ROOT, "curves.json")
TARGET = os.path.join(ROOT, "estimate.js")

BEGIN = "  var CURVES = "
END = "  };"


def render(data):
    keep = ["s_cap", "arch_label", "borrow", "curves", "fp16_energy", "hardware"]
    body = json.dumps({k: data[k] for k in keep if k in data}, indent=2)
    body = "\n".join(("  " + line) if line else line for line in body.split("\n"))
    return (
        "  // Mirror of curves.json (calibrated by build/fit_curves.py from\n"
        "  // build/measured.csv). Regenerate with build/sync_curves_js.py.\n"
        + BEGIN + body.lstrip() + ";\n"
    )


def main():
    with open(CURVES) as f:
        data = json.load(f)
    with open(TARGET) as f:
        src = f.read()

    start = src.index(BEGIN)
    # swallow the comment lines directly above the literal so re-runs don't stack them
    while True:
        prev = src.rfind("\n", 0, start - 1) + 1
        if prev >= start or not src[prev:start].lstrip().startswith("//"):
            break
        start = prev
    end = src.index("\n" + END, start) + len("\n" + END) + 1
    updated = src[:start] + render(data) + src[end:]

    if "--check" in sys.argv:
        if updated != src:
            print("estimate.js CURVES is out of sync with curves.json "
                  "— run: python3 build/sync_curves_js.py", file=sys.stderr)
            return 1
        print("estimate.js CURVES in sync with curves.json")
        return 0

    if updated == src:
        print("estimate.js already in sync")
        return 0
    with open(TARGET, "w") as f:
        f.write(updated)
    print(f"synced curves.json -> {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
