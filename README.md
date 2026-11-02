# quant-energy — EcoCompute official site

Source for **[quantenergy.tech](https://quantenergy.tech)**: the should-I-quantize estimator,
measured GPU energy curves, citation pages, blog, and the MCP endpoint
(`/mcp` — [EcoCompute estimator](https://registry.modelcontextprotocol.io/v0.1/servers?search=ecocompute-estimator)).

Every number on the site states its basis: measured (NVML, GPU-package power) or
interpolated — never dressed up. Data: [Zenodo 10.5281/zenodo.19647290](https://doi.org/10.5281/zenodo.19647290) (CC BY 4.0).

- Site architecture & decisions: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Deployment & local dev: [`USAGE_GUIDE.md`](USAGE_GUIDE.md)

---

## Part of the EcoCompute ecosystem

- **[ecocompute-mlcube](https://github.com/hongping-zh/ecocompute-mlcube)** — Measure energy on YOUR GPU
- **[quant-energy](https://github.com/hongping-zh/quant-energy)** — Interactive curves + API (deploys [quantenergy.tech](https://quantenergy.tech)) *(this repo)*
- **[ecocompute-ai](https://github.com/hongping-zh/ecocompute-ai)** — RTX 5090 benchmark suite
- **[ecocompute-action](https://github.com/hongping-zh/hongping-zh-ecocompute-action)** — CI audit for LLM energy waste

→ **[quantenergy.tech](https://quantenergy.tech)** for the full dataset
