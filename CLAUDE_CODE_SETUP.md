# Using Axioma with Claude Code

## Quick Setup
1. Install Rust toolchain if not present: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Clone axioma: `git clone https://github.com/Manav02012002/axioma.git && cd axioma`
3. Build the MCP server: `cargo build --release -p ax-mcp`
4. Add to Claude Code MCP settings (Settings → MCP Servers → Add):
   - Name: `axioma`
   - Command: `/path/to/axioma/target/release/axioma-mcp`
   - Args: `(none)`

## Available Tools
| Tool | Description |
| --- | --- |
| `axioma_eval` | Parse and evaluate `.ax` code, returning the result. |
| `axioma_parse` | Parse `.ax` code and return diagnostics without evaluation. |
| `axioma_inspect` | Inspect a stored expression by ID. |
| `axioma_suggest` | Suggest applicable algorithms for a stored expression. |
| `axioma_render` | Render a stored expression to LaTeX or Unicode. |
| `axioma_env` | Return the current Axioma environment state. |
| `axioma_simplify` | Simplify a stored expression. |
| `axioma_differentiate` | Differentiate a stored expression with respect to a variable. |
| `axioma_integrate` | Integrate a stored expression, optionally with bounds. |
| `axioma_solve` | Solve a scalar equation or a linear system. |
| `axioma_substitute` | Substitute a target expression inside a stored expression. |
| `axioma_canonicalise` | Canonicalise a stored tensor expression using declared properties. |
| `axioma_define_metric` | Define a symbolic metric from component strings and coordinates. |
| `axioma_christoffel` | Compute Christoffel symbols from a stored metric. |
| `axioma_riemann` | Compute the Riemann tensor from stored Christoffel symbols. |
| `axioma_ricci` | Compute the Ricci tensor from a stored Riemann tensor. |
| `axioma_einstein` | Compute the Einstein tensor directly from a stored metric. |

## Example Conversation
User: "Compute the Ricci tensor for the Schwarzschild metric and verify it's zero"

Claude will call:
1. `axioma_define_metric` with Schwarzschild components
2. `axioma_christoffel`
3. `axioma_riemann`
4. `axioma_ricci`
5. Report that all components are zero

## Injecting Language Context
For best results when asking Claude to write `.ax` files directly, paste the contents of `share/axioma-llm-context.md` into your project's `CLAUDE.md` or system prompt.
