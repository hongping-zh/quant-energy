#!/usr/bin/env python3
"""Build /variables/ — the bitsandbytes precision x batch-size coverage grid.

A different cut of the same published measurements as make_coverage_matrix.py: that one
asks "where does quantization save energy", this one asks "which combinations has anyone
measured at all". Batch size is the axis because every EcoCompute number so far is
batch 1, and a grid that is 3 cells wide out of 15 says that far more plainly than a
sentence does.

Cells carry only what the data actually records: how many measurements, when the most
recent one was taken, and who ran it. Nothing is placeholder — an empty cell is empty
because nobody has measured it.

    python3 build/make_variable_matrix.py     # writes variables/index.html
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRECISIONS = ["FP16", "NF4", "INT8"]
BATCHES = [1, 4, 8, 16, 32]
SATURATED = 3          # independent measurements before a cell stops asking for more

# Session dates for the CSV sources. The per-row timestamp only exists in the 2026-08-19
# file; the others are single sessions whose date is documented in data/README.md.
MEASURED_CSV_DATE = "2026-07-24"
INT8_REPEATS_DATE = "2026-08-20"

MAINTAINER = "hongping-zh"


def cell_key(precision, batch):
    return (precision, batch)


def add(cells, precision, batch, n, date, who):
    key = cell_key(precision, batch)
    if key not in cells:
        raise KeyError(f"no cell for {precision} @ batch {batch}")
    c = cells[key]
    c["n"] += n
    c["who"].add(who)
    if date > c["last"]:
        c["last"] = date


def load():
    """Every bitsandbytes measurement on the site, bucketed by (precision, batch size).

    Each quantized run also produced an FP16 baseline on the same card in the same
    session, so it contributes a point to the FP16 row too — that is what makes the
    FP16 row dense rather than empty.
    """
    cells = {cell_key(p, b): {"n": 0, "last": "", "who": set()}
             for p in PRECISIONS for b in BATCHES}

    # 1. build/measured.csv — the July 2026 anchors across five cards.
    for r in csv.DictReader(open(os.path.join(ROOT, "build", "measured.csv"))):
        n = int(r["n_trials"])
        add(cells, r["precision"], 1, n, MEASURED_CSV_DATE, MAINTAINER)
        add(cells, "FP16", 1, n, MEASURED_CSV_DATE, MAINTAINER)

    # 2. RTX 4090 paired energy+quality, per-row timestamps.
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "rtx4090_paired_energy_quality_2026-08-19.csv"))):
        n, day = int(r["n_trials"]), r["timestamp_utc"][:10]
        add(cells, r["precision"], int(r["batch_size"]), n, day, MAINTAINER)
        add(cells, "FP16", int(r["batch_size"]), n, day, MAINTAINER)

    # 3. RTX 4090 INT8 repeats, n=3 per size.
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "rtx4090_int8_repeats_2026-08-20.summary.csv"))):
        n = int(r["n_trials"])
        add(cells, "INT8", 1, n, INT8_REPEATS_DATE, MAINTAINER)
        add(cells, "FP16", 1, n, INT8_REPEATS_DATE, MAINTAINER)

    # 4. External replications. The contributor handle is part of the filename the
    #    submission was published under; the report itself carries the timestamp.
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "replications", "*.json"))):
        rep = json.load(open(path))
        wl = rep["workload"]
        if wl["precision"] not in PRECISIONS:
            continue
        who = os.path.basename(path).split("-")[3]
        day = rep["timestamp_utc"][:10]
        batch = int(wl.get("batch_size", 1))
        add(cells, wl["precision"], batch, 1, day, who)
        if rep["results"].get("fp16_energy_per_token_mj") is not None:
            add(cells, "FP16", batch, 1, day, who)

    return cells


def state(n):
    if n == 0:
        return "empty"
    return "full" if n >= SATURATED else "partial"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def render_cell(precision, batch, c):
    cls = state(c["n"])
    if cls == "empty":
        body = ('<div class="count">0</div>'
                '<div class="cmeta">never measured</div>'
                '<div class="action">Measure it \u25b8</div>')
        title = f"{precision} at batch {batch}: no measurement exists"
        href = "/container/"
    else:
        who = ", ".join("@" + w for w in sorted(c["who"]))
        need = SATURATED - c["n"]
        action = ("saturated \u2713" if need <= 0
                  else f"{need} more to saturate \u25b8")
        # Non-breaking hyphen: a handle split across two lines in a 90px cell is unreadable.
        who_html = esc(who).replace("-", "\u2011")
        body = (f'<div class="count">{c["n"]}</div>'
                f'<div class="cmeta"><span class="nw">latest {esc(c["last"])}</span>'
                f'<br>{who_html}</div>'
                f'<div class="action">{action}</div>')
        title = (f"{precision} at batch {batch}: {c['n']} measurements, "
                 f"most recent {c['last']}, by {who}")
        href = "/replications/"
    return (f'<a class="cell {cls}" href="{href}" title="{esc(title)}">'
            f'{body}</a>')


def render(cells):
    grid = ['<div class="grid">', '<div></div>']
    for b in BATCHES:
        grid.append(f'<div class="colh">batch {b}</div>')
    for p in PRECISIONS:
        grid.append(f'<div class="rowh">{p}</div>')
        for b in BATCHES:
            grid.append(render_cell(p, b, cells[cell_key(p, b)]))
    grid.append("</div>")

    total = len(PRECISIONS) * len(BATCHES)
    filled = sum(1 for v in cells.values() if v["n"] > 0)
    points = sum(v["n"] for v in cells.values())
    contributors = sorted({w for v in cells.values() for w in v["who"]})

    return TEMPLATE.format(
        grid="\n    ".join(grid),
        filled=filled,
        total=total,
        empty=total - filled,
        points=points,
        contributors=len(contributors),
        contributor_list=", ".join("@" + c for c in contributors),
        quantized=sum(v["n"] for (p, _), v in cells.items() if p != "FP16"),
        saturated=sum(1 for v in cells.values() if v["n"] >= SATURATED),
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Variable matrix — precision \u00d7 batch size, bitsandbytes \u00b7 EcoCompute</title>
<meta name="description" content="Which quantization precision and batch size combinations anyone has actually measured. Every published EcoCompute number is batch 1, so twelve of fifteen cells are empty — and an empty cell is an invitation, not a gap in the write-up." />
<link rel="canonical" href="https://quantenergy.tech/variables/" />
<meta property="og:title" content="Variable matrix — what has been measured, and what nobody has measured yet" />
<meta property="og:description" content="Precision \u00d7 batch size under the bitsandbytes engine. Cells carry a real measurement count, date and contributor; empty cells are empty because nobody has run them." />
<meta property="og:type" content="website" />
<meta property="og:url" content="https://quantenergy.tech/variables/" />
<meta property="og:image" content="https://quantenergy.tech/preview.png" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:image" content="https://quantenergy.tech/preview.png" />
<meta name="author" content="Hongping Zhang" />
<style>
  :root{{--bg:#0b1020;--panel:#111834;--txt:#e6ecff;--muted:#93a0c4;--accent:#38bdf8;--line:#22305a;--good:#10b981}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--txt);font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.7}}
  .wrap{{max-width:860px;margin:0 auto;padding:32px 20px 72px}}
  a{{color:var(--accent);text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .top{{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:28px}}
  .brand{{font-weight:800;font-size:18px}}
  .brand span{{color:var(--accent)}}
  .nav{{display:flex;gap:14px;flex-wrap:wrap;font-size:14px}}
  h1{{font-size:30px;line-height:1.22;margin:0 0 10px}}
  .sub{{font-size:14px;color:var(--muted);margin-bottom:18px}}
  h2{{font-size:20px;margin:36px 0 6px}}
  p,li{{color:#d3dcf6}}
  code{{background:rgba(56,189,248,.12);padding:2px 6px;border-radius:4px;font-size:13px}}
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}}
  .stat{{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:12px 14px}}
  .stat .v{{font-size:28px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.2}}
  .stat .k{{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}}
  .legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:18px 0 10px}}
  .legend i{{display:inline-block;width:14px;height:14px;border-radius:4px;margin-right:6px;vertical-align:-2px}}
  .grid{{display:grid;grid-template-columns:74px repeat(5,1fr);gap:8px}}
  .colh,.rowh{{font-size:12px;font-weight:700;color:var(--muted);display:flex;align-items:center;justify-content:center;text-align:center}}
  .rowh{{background:var(--panel);border:1px solid var(--line);border-radius:8px}}
  .cell{{border-radius:10px;padding:10px 8px;min-height:104px;display:flex;flex-direction:column;justify-content:space-between;text-align:center;border:1px solid var(--line);color:var(--txt)}}
  .cell:hover{{text-decoration:none;border-color:var(--accent)}}
  .cell .count{{font-size:26px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.1}}
  .cell .cmeta{{font-size:10.5px;color:var(--muted);margin-top:2px;line-height:1.4}}
  .nw{{white-space:nowrap}}
  .cell .action{{font-size:11px;font-weight:600;margin-top:6px}}
  .cell.empty{{background:rgba(255,255,255,.02);border-style:dashed}}
  .cell.empty .count{{color:#3d4a75}}
  .cell.empty .action{{color:var(--accent)}}
  .cell.partial{{background:rgba(56,189,248,.12);border-color:rgba(56,189,248,.45)}}
  .cell.full{{background:rgba(16,185,129,.16);border-color:rgba(16,185,129,.5)}}
  .cell.full .action{{color:var(--good)}}
  .callout{{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:16px 18px;margin:22px 0}}
  .cta{{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 8px;align-items:center}}
  .btn{{display:inline-block;background:var(--accent);color:#06223a;font-weight:700;font-size:14px;border-radius:8px;padding:10px 16px}}
  .btn:hover{{text-decoration:none;filter:brightness(1.08)}}
  .ghost{{display:inline-block;border:1px solid var(--line);color:var(--txt);font-weight:600;font-size:14px;border-radius:8px;padding:10px 16px;background:var(--panel)}}
  .ghost:hover{{text-decoration:none;border-color:var(--accent)}}
  footer{{margin-top:48px;font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}}
  @media(max-width:640px){{.grid{{grid-template-columns:56px repeat(5,1fr);gap:5px}}.cell{{min-height:88px;padding:7px 4px}}.cell .count{{font-size:20px}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">Eco<span>Compute</span> \u00b7 Variable matrix</div>
    <nav class="nav"><a href="/">Tool</a><a href="/container/">Container</a><a href="/replications/">Replications</a><a href="/method/">Methodology</a><a href="/cite/">Cite</a></nav>
  </div>

  <h1>{empty} of {total} cells have never been measured by anyone.</h1>
  <p class="sub">Quantization precision \u00d7 batch size, bitsandbytes engine. Generated from the published data, not hand-maintained.</p>

  <p>Every EcoCompute number so far was taken at <b>batch 1</b>. That is not a rounding detail: batch size changes how
  busy the GPU is between decode steps, and utilisation is the thing that decides whether a quantized kernel pays for
  itself. So the whole right-hand side of this grid is <em>unknown</em> \u2014 including to us.</p>

  <div class="stats">
    <div class="stat"><div class="v">{filled}/{total}</div><div class="k">cells with data</div></div>
    <div class="stat"><div class="v">{saturated}</div><div class="k">saturated (\u2265 3 points)</div></div>
    <div class="stat"><div class="v">{points}</div><div class="k">measurements, baselines included</div></div>
    <div class="stat"><div class="v">{contributors}</div><div class="k">contributors</div></div>
  </div>

  <div class="legend">
    <span><i style="background:rgba(255,255,255,.04);border:1px dashed #22305a"></i>no data</span>
    <span><i style="background:rgba(56,189,248,.35)"></i>1\u20132 points</span>
    <span><i style="background:rgba(16,185,129,.45)"></i>\u2265 3 points (saturated)</span>
  </div>

  <div class="grid">
    {grid}
  </div>

  <p class="sub" style="margin-top:14px">Counts are measurements, not sessions: a cell measured three times on the same
  card reads 3, and a configuration repeated five times contributes 5. FP16 is dense because every quantized run also
  measures its own FP16 baseline on the same card, and each baseline counts in its own right: {quantized} quantized
  measurements plus {quantized} baselines. The \u201c42 rows\u201d quoted elsewhere on this site counts distinct
  <em>configurations</em>, not repeats \u2014 same data, different unit. A quantized number with no same-session baseline
  cannot produce a \u0394E% and is not counted at all. Contributors so far: {contributor_list}.</p>

  <div class="callout">
    <b>Why \u2265 3 and not 1?</b> One measurement is an anecdote about a card on a particular afternoon. Our own
    RTX 4090 INT8 cell moved by 12.5 percentage points across three replicates of the <em>same</em> configuration, which
    is exactly why a single point does not close a cell.
  </div>

  <h2>The fastest cell to fill</h2>
  <p>Any of the four batch-4 cells. They need a card you already have and one parameter change, and they are the
  boundary between "we measured batch 1" and "we know what batch size does" \u2014 which is also the gap between this
  dataset and the large-batch datacentre leaderboards it cannot currently be compared with.</p>

  <div class="cta">
    <a class="btn" href="https://colab.research.google.com/github/hongping-zh/quant-energy/blob/main/notebooks/ecocompute-measure.ipynb" target="_blank" rel="noopener">No GPU? Run it on a free Colab T4 \u2192</a>
    <a class="ghost" href="/container/">Run the container \u2192</a>
  </div>

  <p class="sub">A submission counts once it carries an FP16 baseline from the same session on the same card, and a
  schema-valid <code>energy.json</code>. Disagreements with the published curve are published on the same page as
  confirmations.</p>

  <footer>
    Generated by <code>build/make_variable_matrix.py</code> from the published data \u2014 no cell is hand-entered.
    Methodology: direct NVML GPU-package power sampling, 256 tokens/run, warmup + repeated decode iterations,
    <b>not wall power</b>. Supplemental energy methodology, not a certified MLPerf result; MLCommons, MLPerf and
    MLCube are trademarks of MLCommons Association, used nominatively.
    <br>Other cut of the same data: <a href="/#matrix">quantization method \u00d7 model size</a> \u00b7
    contribute \u2192 <a href="https://github.com/hongping-zh/ecocompute-mlcube" target="_blank" rel="noopener">ecocompute-mlcube</a>
  </footer>
</div>
</body>
</html>
"""


def main():
    cells = load()
    out = os.path.join(ROOT, "variables", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(render(cells))
    filled = sum(1 for v in cells.values() if v["n"] > 0)
    print(f"wrote {out}  ({filled}/{len(cells)} cells with data, "
          f"{sum(v['n'] for v in cells.values())} measurements)")


if __name__ == "__main__":
    main()
