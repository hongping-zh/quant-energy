# EcoCompute MCP server

Exposes the deterministic **"should I quantize?"** estimator as [MCP](https://modelcontextprotocol.io)
tools, so any MCP client — Claude Desktop, Cursor, Windsurf, an internal copilot — can call it.
The numbers come from [`../estimate.js`](../estimate.js) (curves calibrated to real measured GPU
data); the model never guesses energy figures.

## Tools

| tool | what it does |
|------|--------------|
| `should_i_quantize` | ΔE% vs FP16 for `(params_b, arch, precision[, batch, ctx])`, with 95% range, `measured/interpolated/estimated` label, confidence, recommendation. Also accepts a plain-English `query`. |
| `list_architectures` | supported GPU classes, precisions, measured size ranges and crossover points |

## Install & run

```bash
cd mcp
npm install
node server.js      # stdio transport; "running on stdio" prints to stderr
npm run smoke       # optional: drive it with a real MCP client and print results
```

## Register in a client

**Claude Desktop** — add to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "ecocompute": {
      "command": "node",
      "args": ["/absolute/path/to/quant-energy/mcp/server.js"]
    }
  }
}
```

**Cursor** — `~/.cursor/mcp.json` (or *Settings → MCP → Add*), same shape:

```json
{
  "mcpServers": {
    "ecocompute": { "command": "node", "args": ["/absolute/path/to/quant-energy/mcp/server.js"] }
  }
}
```

Restart the client; the two tools appear automatically. Then ask, e.g.
*"Should I quantize a 13B model on an A100 with INT8 at batch 32?"* and the assistant
will call `should_i_quantize` and answer from the returned numbers.

## Example output

```
should_i_quantize { params_b: 7, arch: "blackwell", precision: "NF4" }
-> 7B on Blackwell (RTX 5090), NF4: ΔE% -11.4% (95% range -14.2..-8.7) — measured/high.
   Recommendation: quantize.
   { delta_pct, ci, basis, confidence, recommendation, verdict, weight_gb, notes, ... }
```

Each result is also returned as `structuredContent` for programmatic consumers.

## Notes

- `batch=1, ctx<=2048` reproduce the measured curve. Larger values apply a transparent,
  conservative model (savings shrink toward zero, never invent a penalty) and are flagged
  `modelled` with lower confidence / a wider band.
- Honesty labels are first-class: out-of-range or borrowed-architecture answers come back
  `estimated` with `low` confidence and explanatory `notes`.
- For an HTTP version of the same logic, see [`../api`](../api) (Cloudflare Worker).
