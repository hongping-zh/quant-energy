/* Smoke test: spin up server.js over stdio with the real MCP client, list tools,
 * and call should_i_quantize a couple of ways. Run: node smoke.js */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({ command: "node", args: ["server.js"] });
const client = new Client({ name: "smoke", version: "1.0.0" });
await client.connect(transport);

const tools = await client.listTools();
console.log("tools:", tools.tools.map((t) => t.name).join(", "));

const r1 = await client.callTool({ name: "should_i_quantize", arguments: { params_b: 7, arch: "blackwell", precision: "NF4" } });
console.log("\n[7B blackwell NF4]\n" + r1.content[0].text.split("\n")[0]);

const r2 = await client.callTool({ name: "should_i_quantize", arguments: { query: "13B on A100 int8 batch 32 ctx 8k" } });
console.log("\n[query: 13B A100 int8 batch32]\n" + r2.content[0].text.split("\n")[0]);

const r3 = await client.callTool({ name: "should_i_quantize", arguments: { arch: "ampere" } });
console.log("\n[missing params_b]\n" + r3.content[0].text + " isError=" + r3.isError);

const r4 = await client.callTool({ name: "list_architectures", arguments: {} });
console.log("\n[architectures] " + Object.keys(JSON.parse(r4.content[0].text)).join(", "));

await client.close();
console.log("\nOK");
