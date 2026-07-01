#!/usr/bin/env node
/* EcoCompute MCP server — exposes the deterministic "should I quantize?" estimator
 * as MCP tools so any MCP client (Claude Desktop, Cursor, an internal copilot) can
 * call it. The numbers come from ../estimate.js (measured-anchored curves), never
 * from the model guessing.
 *
 * Run:  node server.js     (stdio transport)
 * See mcp/README.md for client registration.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { estimate, parseQuery, CURVES } = require("../estimate.js");
const { optimize } = require("../optimize.js");

const ARCHES = ["turing", "ada", "blackwell", "ampere", "hopper"];

function architectures() {
  const out = {};
  for (const [arch, byPrec] of Object.entries(CURVES.curves)) {
    out[arch] = {
      label: CURVES.arch_label[arch] || arch,
      precisions: Object.fromEntries(
        Object.entries(byPrec).map(([p, c]) => [p, { measured_range_b: [c.n_min, c.n_max], crossover_b: c.crossover_b }])
      ),
    };
  }
  for (const [arch, src] of Object.entries(CURVES.borrow || {})) {
    if (!out[arch]) out[arch] = { label: CURVES.arch_label[arch] || arch, borrows_from: src, precisions: {} };
  }
  return out;
}

const server = new McpServer({ name: "ecocompute-estimator", version: "1.0.0" });

server.registerTool(
  "should_i_quantize",
  {
    title: "Should I quantize?",
    description:
      "Estimate the energy change (ΔE% vs FP16) of weight-only quantization (NF4/INT8) for a given model size and GPU architecture class. " +
      "Returns a point estimate, 95% range, an honesty label (measured/interpolated/estimated), confidence, and a recommendation. " +
      "Numbers are calibrated to real measured GPU data, not guessed. Optionally accepts a plain-English query.",
    inputSchema: {
      params_b: z.number().positive().optional().describe("Model size in billions of parameters, e.g. 13"),
      arch: z.enum(ARCHES).optional().describe("GPU architecture class"),
      precision: z.enum(["NF4", "INT8"]).default("NF4").describe("Quantization precision"),
      batch: z.number().int().positive().default(1).describe("Batch size (>1 => modelled, not measured)"),
      ctx: z.number().int().positive().default(2048).describe("Context length in tokens (>2048 => modelled)"),
      query: z.string().optional().describe("Plain-English query, e.g. '13B on A100 int8 batch 32 ctx 8k'. If given, it overrides the structured fields it can parse."),
    },
  },
  async (args) => {
    let { params_b, arch, precision = "NF4", batch = 1, ctx = 2048 } = args;
    if (args.query) {
      const p = parseQuery(args.query);
      if (p.N != null) params_b = p.N;
      if (p.arch) arch = p.arch;
      if (p.precision) precision = p.precision;
      if (p.batch != null) batch = p.batch;
      if (p.ctx != null) ctx = p.ctx;
    }
    if (!(params_b > 0)) return errOut("params_b (model size in billions) is required (or include it in `query`).");
    if (!arch) return errOut("arch is required. Supported: " + ARCHES.join(", "));
    const r = estimate(params_b, arch, precision, { batch, ctx });
    if (r.error) return errOut(r.error);
    const payload = { input: { params_b, arch, precision, batch, ctx }, ...r };
    return {
      content: [{ type: "text", text: summarize(payload) + "\n\n" + JSON.stringify(payload, null, 2) }],
      structuredContent: payload,
    };
  }
);

server.registerTool(
  "list_architectures",
  {
    title: "List supported GPU architectures",
    description: "List the GPU architecture classes, available precisions, measured size ranges and crossover points the estimator knows about.",
    inputSchema: {},
  },
  async () => {
    const a = architectures();
    return { content: [{ type: "text", text: JSON.stringify(a, null, 2) }], structuredContent: a };
  }
);

server.registerTool(
  "recommend_config",
  {
    title: "Recommend the optimal inference config under constraints",
    description:
      "Given a GPU architecture and model size plus optional latency/VRAM/throughput budgets and an objective, " +
      "search the (precision x batch x context) space and return the objective-optimal config, alternatives, and the " +
      "energy<->latency Pareto front. Energy is measured-anchored; latency & throughput are a roofline MODEL (not measured); " +
      "VRAM is a standard calculation. Each config carries per-field basis + confidence.",
    inputSchema: {
      params_b: z.number().positive().describe("Model size in billions of parameters, e.g. 7"),
      arch: z.enum(ARCHES).describe("GPU architecture class"),
      objective: z.enum(["min_energy", "min_latency", "max_throughput", "min_vram"]).default("min_energy").describe("What to optimize"),
      max_latency_ms: z.number().positive().optional().describe("Latency budget, ms per token"),
      max_vram_gb: z.number().positive().optional().describe("VRAM budget, GB"),
      min_throughput: z.number().positive().optional().describe("Throughput floor, tokens/s"),
      batch: z.number().int().positive().optional().describe("Pin batch size (else searched over 1..32)"),
      context: z.number().int().positive().optional().describe("Pin context length (else searched over 512..8192)"),
    },
  },
  async (args) => {
    if (!(args.params_b > 0)) return errOut("params_b (model size in billions) is required.");
    if (!args.arch) return errOut("arch is required. Supported: " + ARCHES.join(", "));
    const r = optimize(args);
    if (r.error) return errOut(r.error);
    const rec = r.recommended || r.closest_infeasible;
    const head = r.recommended
      ? `Recommended (${r.input.objective}): ${rec.summary} — energy basis ${rec.basis.energy}/${rec.confidence}.`
      : `No config satisfies the constraints. Closest: ${rec.summary}.`;
    return {
      content: [{ type: "text", text: head + "\n\n" + JSON.stringify(r, null, 2) }],
      structuredContent: r,
    };
  }
);

function summarize(r) {
  const sign = r.delta_pct >= 0 ? "+" : "";
  return `${r.input.params_b}B on ${CURVES.arch_label[r.input.arch] || r.input.arch}, ${r.input.precision}: ` +
    `ΔE% ${sign}${r.delta_pct}% (95% range ${r.ci[0]}..${r.ci[1]}) — ${r.basis}/${r.confidence}` +
    (r.modelled ? " · modelled" : "") + `. Recommendation: ${r.recommendation}. ${r.verdict}`;
}
function errOut(msg) {
  return { content: [{ type: "text", text: "Error: " + msg }], isError: true };
}

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // stderr only — stdout is the MCP channel
  process.stderr.write("ecocompute-estimator MCP server running on stdio\n");
}
main().catch((e) => { process.stderr.write(String(e && e.stack || e) + "\n"); process.exit(1); });
