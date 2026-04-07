# Using Axioma with AI Coding Tools

Axioma exposes its full computation engine as an MCP (Model Context Protocol) server. Any AI coding tool that supports MCP — including Claude Code, VS Code with Copilot, Cursor, Windsurf, and others — can call Axioma's tools directly instead of trying to write Axioma syntax.

## Why tools instead of code generation?

Axioma is a domain-specific language with tensor index conventions, property declarations, and specialised algorithms. LLMs have no training data for Axioma syntax and will hallucinate. The MCP tool path avoids this entirely: the AI calls structured tools with typed parameters and gets structured results back. No syntax involved, no hallucination possible.

## Quick Setup

1. Install the Rust toolchain if not present:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
2. Clone and build:
git clone https://github.com/Manav02012002/axioma.git && cd axioma
cargo build --release -p ax-mcp
3. Add to your AI tool's MCP configuration:
   - **Server name:** `axioma`
   - **Command:** `/path/to/axioma/target/release/axioma-mcp`
   - **Transport:** stdio (default) or `--transport http --port 3000` for HTTP+SSE

### Tool-specific instructions

**Claude Code:** Add to Settings → MCP Servers → Add. Command: path to `axioma-mcp`.

**VS Code + Copilot / Continue:** Add to your MCP server config in `settings.json` or the tool's MCP configuration panel.

**Cursor:** Add to Settings → MCP Servers.

**Generic MCP client:** Connect via stdio (spawn the binary) or HTTP+SSE (run with `--transport http`).

## Available Tools

All tools are prefixed with `axioma_` in the MCP interface.

### Core
| Tool | Description |
| --- | --- |
| `axioma_eval` | Parse and evaluate Axioma code, returning the result |
| `axioma_inspect` | Inspect a stored expression by ID — returns kind, symbols, indices, properties |
| `axioma_suggest` | Suggest applicable algorithms for an expression, optionally given a goal |
| `axioma_render` | Render a stored expression to LaTeX or Unicode |
| `axioma_workflow` | Get an ordered list of tool calls for a given goal (e.g., "compute_curvature") |

### State
| Tool | Description |
| --- | --- |
| `axioma_list_expressions` | List all stored expression IDs with their renderings |
| `axioma_list_metrics` | List all defined metrics with coordinates and dimensions |
| `axioma_list_properties` | List all declared tensor properties grouped by symbol |
| `axioma_list_index_families` | List all declared index families with values and conventions |
| `axioma_get_state_summary` | Compact summary of the full environment state |

### Algebra
| Tool | Description |
| --- | --- |
| `axioma_simplify` | Simplify a stored expression |
| `axioma_differentiate` | Differentiate with respect to a variable |
| `axioma_integrate` | Symbolic integration |
| `axioma_solve` | Solve equations or linear systems |
| `axioma_substitute` | Substitute a sub-expression |
| `axioma_series` | Taylor/Laurent series expansion |

### Tensor algebra
| Tool | Description |
| --- | --- |
| `axioma_canonicalise` | Canonicalise using declared symmetries |
| `axioma_sort_product` | Sort factors in a product using commutativity properties |
| `axioma_eliminate_metric` | Contract metric tensors |
| `axioma_eliminate_kronecker` | Contract Kronecker deltas |
| `axioma_rename_dummies` | Rename dummy indices to canonical names |
| `axioma_meld` | Simplify sums using tensor symmetries |

### General relativity
| Tool | Description |
| --- | --- |
| `axioma_define_metric` | Define a symbolic metric from components and coordinates |
| `axioma_christoffel` | Compute Christoffel symbols |
| `axioma_riemann` | Compute the Riemann tensor |
| `axioma_ricci` | Compute the Ricci tensor |
| `axioma_einstein` | Compute the Einstein tensor |
| `axioma_scalar_curvature` | Compute the Ricci scalar |
| `axioma_kretschner` | Compute the Kretschner scalar |
| `axioma_weyl` | Compute the Weyl tensor |
| `axioma_geodesic` | Compute geodesic equations |

### Diagnostics
| Tool | Description |
| --- | --- |
| `axioma_diff` | Compare two expressions structurally |
| `axioma_check_properties` | Check which symbols have/lack property declarations |
| `axioma_explain` | Explain what an algorithm would do and why it might not change the expression |

## Example Workflow

**User:** "Compute the Ricci tensor for the Schwarzschild metric and verify it's zero."

**AI calls:**
1. `axioma_workflow({"goal": "compute_curvature"})` — gets the step sequence
2. `axioma_define_metric({"name": "g", "components": [...], "coordinates": ["t","r","theta","phi"]})` — returns `metric_id`
3. `axioma_christoffel({"metric_id": "g"})` — returns `christoffel_id`
4. `axioma_riemann({"christoffel_id": "g"})` — returns `riemann_id`
5. `axioma_ricci({"riemann_id": "g"})` — returns `ricci_id` with all components rendered
6. AI reports: "All 16 components of the Ricci tensor are zero, confirming Schwarzschild is vacuum."

## Writing Axioma Code Directly

If the AI needs to generate `.ax` source files (for notebooks, scripts, or batch computations), inject the full language reference from `share/axioma-llm-context.md` into the system prompt or project context file. This 590-line reference covers all syntax, built-in functions, tensor properties, conventions, and common workflows.
