# Axioma

Axioma is a Rust-native symbolic computing system aimed at physics-heavy workflows: tensor algebra, exact symbolic manipulation, convention tracking, code generation, rendering, equation manipulation, perturbation theory, quantum mechanics, QFT utilities, and editor/server tooling around the language.

This repository is not a thin wrapper over SymPy, Mathematica, or a Python runtime. The core stack is implemented in Rust and is organized as a workspace of language, IR, evaluator, tensor, rendering, tooling, and integration crates.

## Table Of Contents

- [What Axioma Is](#what-axioma-is)
- [What Lives In This Repository](#what-lives-in-this-repository)
- [Core Capabilities](#core-capabilities)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation And First Run](#installation-and-first-run)
- [CLI Overview](#cli-overview)
- [REPL Overview](#repl-overview)
- [Project Configuration](#project-configuration)
- [Standard Library](#standard-library)
- [Examples](#examples)
- [Language Pipeline And Architecture](#language-pipeline-and-architecture)
- [Workspace Crates](#workspace-crates)
- [VS Code Extension](#vs-code-extension)
- [MCP, LSP, Notebook, Jupyter, And Plugins](#mcp-lsp-notebook-jupyter-and-plugins)
- [AAS Validation And AI-Oriented Tools](#aas-validation-and-ai-oriented-tools)
- [Development Workflows](#development-workflows)
- [Testing And Verification](#testing-and-verification)
- [License](#license)

## What Axioma Is

Axioma is a scientific computing language and toolchain with a strong focus on:

- symbolic algebra with exact arithmetic
- tensor expressions with abstract indices and variance
- general relativity workflows
- quantum mechanics and QFT helper operations
- perturbation theory and cosmology
- rendering to Unicode and LaTeX
- code generation to Python, Rust, and C++
- developer tooling: CLI, REPL, LSP, MCP, notebook, and VS Code integration

The repository contains both the language implementation and the surrounding tooling. If you are reading this as a user, the main entry point is the `axioma` CLI. If you are reading this as a contributor, the workspace is split into focused crates that roughly follow the flow:

`source text -> syntax/lexer/parser -> core lowering -> IR -> evaluation/rewrite/tensor logic -> render/export/codegen -> integrations`

## What Lives In This Repository

At the top level, the project contains:

- `crates/`: the Rust workspace crates
- `std/`: the Axioma standard library modules
- `examples/`: runnable example `.ax` programs
- `editors/vscode/`: the VS Code extension
- `spec/`: machine-readable specs, currently including the AAS JSON schema
- `share/`: generated reference material such as the LLM language reference
- `plugins/`: example plugin crates, currently including `axp-echo`
- `axioma.toml`: project-root configuration used by context-aware tools

The workspace root `Cargo.toml` declares the canonical workspace license as `Apache-2.0`, the Rust edition as `2021`, and a workspace MSRV field of `1.78`. The active pinned toolchain in this checkout is defined by `rust-toolchain.toml`.

## Core Capabilities

The codebase currently exposes all of the following capability groups.

### Symbolic CAS

- exact integer and rational arithmetic
- symbolic simplification
- differentiation
- integration
- limits
- Taylor series expansion
- substitution and rewrite-rule application
- partial fraction decomposition and factorization helpers

### Tensor Algebra

- indexed tensors with explicit variance markers
- property declarations such as metric, inverse metric, symmetric, antisymmetric, Riemann symmetry, traceless, epsilon tensor, Kronecker delta, derivative operators, and spinor-related properties
- canonicalization and contraction routines
- dummy-index handling
- metric and Kronecker elimination
- epsilon-to-delta style reductions
- convention-aware symbolic workflows

### Physics Domains

- GR metric, Christoffel, Riemann, Ricci, Ricci scalar, Einstein tensor, and geodesic workflows
- perturbation theory for metrics and derived geometric quantities
- cosmology helpers including FRW-oriented perturbative routines
- QM helpers such as kets, bras, brackets, density matrices, Pauli matrices, and operator algebra
- QFT helpers such as gamma chains, gamma traces, normal ordering, Grassmann declarations, superspace, and BRST utilities
- units and dimensional analysis

### Output And Interop

- Unicode rendering
- LaTeX rendering
- session export to HTML and LaTeX
- code generation to Python, Rust, and C++
- SVG plotting

### Tooling

- interactive REPL
- script runner
- parser/render/export/codegen/documentation subcommands
- language server
- MCP server
- notebook server
- VS Code extension
- AAS schema validation
- plugin host and plugin example

## Repository Layout

The highest-signal directories in the repo are:

| Path | Purpose |
|---|---|
| [`Cargo.toml`](Cargo.toml) | Rust workspace manifest, members, workspace license, workspace dependencies |
| [`axioma.toml`](axioma.toml) | Project configuration for spec/build paths and plugin registry |
| [`crates/`](./crates) | All Rust crates implementing the language and tooling |
| [`std/`](./std) | Standard library `.ax` modules imported with `import std.*` |
| [`examples/`](./examples) | Runnable demos covering calculus, GR, QM, perturbation theory, BRST, and SUSY |
| [`spec/`](./spec) | External specifications such as `aas.schema.json` |
| [`share/`](./share) | Generated reference docs, currently including `axioma-llm-context.md` |
| [`editors/vscode/`](./editors/vscode) | VS Code extension package |
| [`plugins/`](./plugins) | Example plugin crates |

## Prerequisites

For the Rust workspace:

- Rust toolchain matching [`rust-toolchain.toml`](./rust-toolchain.toml)
- `cargo`

For the VS Code extension:

- Node.js
- npm
- VS Code

For optional notebook/Jupyter work:

- browser access for the notebook server
- `libzmq` if you intend to build the Jupyter kernel crate

## Installation And First Run

### Install The CLI From Git

```bash
cargo install --git https://github.com/Manav02012002/axioma --bin axioma
```

Then start the REPL:

```bash
axioma repl
```

### Run Locally From This Repository

```bash
cargo run -p axioma -- repl
cargo run -p axioma -- run examples/calculus_demo.ax
cargo run -p axioma -- run examples/gr_tutorial.ax
```

### Initialize A New Axioma Project

The CLI includes an `init` command that creates a project-local `axioma.toml`.

```bash
cargo run -p axioma -- init
```

The default generated shape is:

```toml
[axioma]
version = "0.1.0"

[paths]
spec_dir = "spec"
build_dir = "build"
```

## CLI Overview

The main CLI lives in [`crates/ax-cli/src/main.rs`](./crates/ax-cli/src/main.rs). Its top-level subcommands currently include:

| Command | Purpose |
|---|---|
| `axioma repl` | Start the interactive REPL |
| `axioma run <file>` | Execute an `.ax` script |
| `axioma parse <file>` | Parse/lower a file and emit diagnostics |
| `axioma render <file> --format <latex|unicode...>` | Render a file |
| `axioma export <file>` | Export a file to LaTeX or HTML |
| `axioma codegen <file>` | Generate code from a file |
| `axioma docgen` | Generate `share/axioma-llm-context.md` |
| `axioma install <package>` | Package/install-related plumbing |
| `axioma init` | Create a starter `axioma.toml` |
| `axioma validate <path>` | Validate an AAS JSON document against `spec/aas.schema.json` |
| `axioma paths` | Print resolved root/spec/build paths |
| `axioma fix ...` | Diagnostics/fix workflow |
| `axioma ai fix|pack|apply` | AI-oriented file packet/edit workflow |
| `axioma notebook` | Start notebook server when the `notebook` feature is enabled |
| `axioma plugin list|run` | Plugin registry and execution commands when the `plugins` feature is enabled |

### Validate / Trace Workflow

`validate` and plugin execution write trace reports into the configured build directory unless `--no-trace` is passed. In this checkout, the configured build directory is `build/`, and trace receipts are currently visible under [`build/trace/`](./build/trace).

### Project Path Resolution

Project path discovery is implemented in [`crates/ax-context/src/lib.rs`](./crates/ax-context/src/lib.rs). The system:

- walks upward to find `axioma.toml`
- reads `paths.spec_dir` and `paths.build_dir`
- resolves plugin configuration from `[plugins.<id>]`

## REPL Overview

The REPL lives in [`crates/ax-cli/src/cmd_repl.rs`](./crates/ax-cli/src/cmd_repl.rs).

Recent work in this repository has upgraded the REPL substantially. It now includes:

- startup banner with ANSI color support
- cell-numbered prompts
- LaTeX input toggle mode
- syntax highlighting while typing using Axioma's own lexer
- tab completion for commands, imports, Greek shortcuts, builtins, and env bindings
- history-based ghost-text hints
- colored status and error output

### Prompt Behavior

The prompt distinguishes normal and LaTeX modes:

- normal mode: `ax[1]> `
- LaTeX mode: `tex[1]> `

Continuation lines use a dimmed continuation prompt.

### REPL Commands

The REPL command set exposed by `print_help()` includes:

| Command | Description |
|---|---|
| `:quit`, `:q` | Exit the REPL |
| `:help`, `:h` | Show REPL help |
| `:env` | Show current bindings |
| `:rules` | Show user-defined rewrite rules |
| `:assumptions` | Show active assumptions |
| `:convention` | Show active convention settings |
| `:inspect [expr]` | Structural inspection of an expression or the last result |
| `:suggest [expr]` | Suggest algorithms for an expression |
| `:pool on|off|stats` | Control pooled expression storage |
| `:parallel on|off` | Toggle parallel tensor canonicalization |
| `:codegen python|rust|cpp` | Generate code for the last result |
| `:export latex|html [file]` | Export current session |
| `:reset` | Reset env/rules/session state |
| `:trust` | Show trust level or rewrite trace for the last result |
| `:latex` | Toggle LaTeX input mode |

### Input Features

The REPL currently supports:

- multiline entry using the `is_complete()` loop
- history loading and saving to `~/.axioma_history`
- history-based hints at end-of-line only
- command completion starting with `:`
- import completion after `import `
- Greek shortcuts after backslash prefixes such as `\alp`
- environment-aware identifier completion after successful evaluation

### Evaluation Path

The execution path in the REPL reuses the same lower/eval machinery as file execution:

- input text is optionally routed through `ax_core_ir::latex_to_axioma`
- lowering is performed with `ax_core_ir::lower`
- environment mutations and visible statuses are handled by [`cmd_run::execute_expr`](./crates/ax-cli/src/cmd_run.rs)

`cmd_run::execute_expr` handles:

- imports
- convention updates
- parallel declarations
- graded declarations
- superspace setup
- BRST setup
- property declarations
- coordinate declarations
- index declarations
- binding creation
- rewrite rule registration
- function definitions
- assumption registration

## Project Configuration

The default project config file is [`axioma.toml`](axioma.toml). In this repository it currently looks like:

```toml
[axioma]
version = "0.1.0"

[paths]
spec_dir = "spec"
build_dir = "build"

[plugins.axp-echo]
wasm = "target/wasm32-unknown-unknown/debug/axp_echo.wasm"
allow = []
```

Important fields:

- `paths.spec_dir`: where schema/spec assets live
- `paths.build_dir`: where generated traces and intermediate outputs are written
- `[plugins.<id>]`: plugin registry used by `axioma plugin list` and `axioma plugin run`

## Standard Library

The standard library lives under [`std/`](./std). The search path logic in [`cmd_run.rs`](./crates/ax-cli/src/cmd_run.rs) looks in:

- current working directory
- current file's directory
- `AXIOMA_STD_PATH` if set
- directories near the running executable

### Standard Library Modules Present In This Checkout

| Module Area | Files |
|---|---|
| top-level algebra/calculus/trig | `std/algebra.ax`, `std/calculus.ax`, `std/trig.ax` |
| GR | `std/gr/de_sitter.ax`, `std/gr/frw.ax`, `std/gr/kerr_newman.ax`, `std/gr/minkowski.ax`, `std/gr/schwarzschild.ax` |
| QFT | `std/qft/brst.ax`, `std/qft/dirac.ax`, `std/qft/gamma.ax`, `std/qft/normal_ordering.ax`, `std/qft/scalar_field.ax`, `std/qft/superspace.ax` |
| QM | `std/qm/bell.ax`, `std/qm/harmonic_oscillator.ax`, `std/qm/spin.ax` |
| Tensor helpers | `std/tensor/index.ax`, `std/tensor/symmetry.ax` |
| Units | `std/units/cgs.ax`, `std/units/natural.ax`, `std/units/si.ax` |
| Conventions | `std/conventions/landau.ax`, `std/conventions/mtw.ax`, `std/conventions/particle_physics.ax`, `std/conventions/weinberg.ax` |
| Physics | `std/physics/classical_mechanics.ax`, `std/physics/klein_gordon.ax`, `std/physics/maxwell.ax` |

### Import Style

Imports use dotted module paths:

```ax
import std.gr.schwarzschild
import std.conventions.mtw
import std.units.natural
```

## Examples

The `examples/` directory is the quickest map of what the current evaluator is expected to do in practice.

| File | Focus |
|---|---|
| [`examples/calculus_demo.ax`](./examples/calculus_demo.ax) | differentiation, integration, series, limit, solve |
| [`examples/equation_manipulation.ax`](./examples/equation_manipulation.ax) | equation values and equation-to-rule workflows |
| [`examples/gr_tutorial.ax`](./examples/gr_tutorial.ax) | Schwarzschild metric and curvature pipeline |
| [`examples/schwarzschild.ax`](./examples/schwarzschild.ax) | vacuum Ricci verification |
| [`examples/linearized_gravity.ax`](./examples/linearized_gravity.ax) | perturbative Ricci calculation |
| [`examples/cosmological_perturbations.ax`](./examples/cosmological_perturbations.ax) | FRW/cosmology-oriented perturbation helpers |
| [`examples/qm_tutorial.ax`](./examples/qm_tutorial.ax) | Pauli matrices, commutators, and bra-ket basics |
| [`examples/wess_zumino_model.ax`](./examples/wess_zumino_model.ax) | superspace and chiral/antichiral constructions |
| [`examples/brst_qed.ax`](./examples/brst_qed.ax) | BRST setup and ghost-number-related operations |
| [`examples/test_import.ax`](./examples/test_import.ax) | small import-oriented smoke case |

### Example Commands

```bash
cargo run -p axioma -- run examples/calculus_demo.ax
cargo run -p axioma -- run examples/equation_manipulation.ax
cargo run -p axioma -- run examples/gr_tutorial.ax
cargo run -p axioma -- run examples/qm_tutorial.ax
```

## Language Pipeline And Architecture

The best way to understand the workspace is to follow a source file through the stack.

### 1. Syntax Layer

`ax-syntax` provides:

- token kinds
- lexer output
- parser output
- basic syntax diagnostics

This is also what the REPL now uses for interactive syntax highlighting.

### 2. Lowering Layer

`ax-core-ir` is responsible for lowering source text into the core IR accepted by the evaluator. This layer also handles the LaTeX-to-Axioma translation path used by the REPL.

### 3. Core IR

`ax-ir` defines the expression model used across the workspace. Everything downstream, including evaluation, rendering, rewriting, and code generation, works against this shared IR.

### 4. Evaluation / Semantic Layer

`ax-eval` is the central semantic engine. It contains:

- builtins
- algorithms
- environment mutation helpers
- simplification logic
- calculus routines
- rewrite integration
- registry metadata used by REPL completion and generated docs

### 5. Domain Engines

Domain crates such as `ax-tensor`, `ax-spinor`, `ax-qm`, `ax-perturb`, `ax-graded`, `ax-units`, `ax-forms`, `ax-ode`, `ax-solve`, and `ax-variational` provide specialized symbolic operations that `ax-eval` can expose upward as language functions.

### 6. Output Layer

Rendering and output are split between:

- `ax-render` for Unicode/LaTeX-like rendering
- `ax-codegen` for Python/Rust/C++
- `ax-cli/cmd_export` for HTML and LaTeX document export

### 7. Tooling Layer

The CLI, REPL, MCP server, LSP server, notebook, VS Code extension, plugin host, and AI/AAS helpers sit on top of the same evaluator and IR stack.

## Workspace Crates

This workspace is intentionally decomposed. The table below is a practical reading map for contributors.

| Crate | Role |
|---|---|
| `ax-syntax` | lexer/parser/token kinds and lightweight syntax diagnostics |
| `ax-core-ir` | source lowering and LaTeX-to-Axioma translation |
| `ax-ir` | core expression IR and shared symbolic data structures |
| `ax-eval` | evaluator, registries, builtins, algorithms, and environment logic |
| `ax-render` | Unicode and LaTeX rendering |
| `ax-codegen` | code generation targets |
| `ax-rewrite` | rewrite rule machinery |
| `ax-compare` | semantic comparison/equivalence helpers |
| `ax-equiv` | equivalence-oriented API surface |
| `ax-tensor` | abstract tensor algebra, indices, contractions, canonicalization |
| `ax-perm` | permutation/group utilities used by tensor workflows |
| `ax-young` | Young-tableau related helpers |
| `ax-linalg` | symbolic linear algebra |
| `ax-solve` | solving routines |
| `ax-ode` | ODE/PDE helpers and numerical integration support |
| `ax-forms` | differential forms |
| `ax-units` | units and dimensional analysis |
| `ax-variational` | variational/Euler-Lagrange workflows |
| `ax-graded` | graded algebra, superspace, BRST support |
| `ax-perturb` | perturbation theory and cosmology workflows |
| `ax-qm` | quantum-mechanics utilities |
| `ax-spinor` | spinor-helicity and related symbolic structures |
| `ax-plot` | plotting support |
| `ax-context` | project-root/config discovery |
| `ax-trace` | trace report generation |
| `ax-diagnostics` | diagnostic structures |
| `ax-cli` | main CLI binary, REPL, export/docgen/fix/install tooling |
| `ax-lsp` | language server binary (`axioma-lsp`) |
| `ax-mcp` | MCP server binary (`axioma-mcp`) |
| `ax-notebook` | notebook server implementation |
| `ax-jupyter` | Jupyter kernel integration, excluded from default build |
| `ax-plugin-api` | plugin wire-format types |
| `ax-plugin-host` | plugin host runtime using Wasm |
| `ax-ai-proto` | AI-oriented protocol types |
| `plugins/axp-echo` | sample plugin crate used by integration tests |

## VS Code Extension

The extension lives in [`editors/vscode/`](./editors/vscode).

### What The Extension Contributes

According to [`editors/vscode/package.json`](./editors/vscode/package.json), the extension contributes:

- Axioma language registration for `.ax` and `.axioma`
- TextMate grammar and language configuration
- commands for file/selection/line evaluation
- a compute panel
- an activity-bar view container named `Axioma`
- an MCP-backed workflow picker
- configuration for LSP path, MCP path, timeout, render mode, and auto-evaluate

### Commands Exposed By The Extension

- `Axioma: Evaluate File`
- `Axioma: Evaluate Selection`
- `Axioma: Evaluate Current Line`
- `Axioma: Show Compute Panel`
- `Axioma: Clear Compute Panel`
- `Axioma: Restart Language Server`
- `Axioma: Restart Compute Server`
- `Axioma: Show Available Workflows`

### Keybindings

- `Shift+Enter`: evaluate selection
- `Ctrl+Enter` / `Cmd+Enter`: evaluate current line

### Runtime Behavior

The extension currently:

- starts the LSP client from `axioma-lsp`
- starts an MCP subprocess from `axioma-mcp`
- renders compute results in a webview panel/sidebar
- supports LaTeX rendering in the panel through KaTeX CDN assets
- emits inline result decorations into the editor for evaluated lines

Relevant source files:

- [`editors/vscode/src/extension.ts`](./editors/vscode/src/extension.ts)
- [`editors/vscode/src/lsp.ts`](./editors/vscode/src/lsp.ts)
- [`editors/vscode/src/mcp.ts`](./editors/vscode/src/mcp.ts)
- [`editors/vscode/src/compute.ts`](./editors/vscode/src/compute.ts)

### Building The Extension

```bash
cd editors/vscode
npm install
npm run compile
npm run package
```

## MCP, LSP, Notebook, Jupyter, And Plugins

### LSP

`ax-lsp` builds the `axioma-lsp` binary. The VS Code extension launches it over stdio.

### MCP

`ax-mcp` builds the `axioma-mcp` binary. The VS Code extension spawns it and calls tools over JSON-RPC-like messages on stdio.

### Notebook

`ax-notebook` provides a browser-oriented notebook server. The `axioma` CLI exposes `notebook` when built with the `notebook` feature. In this workspace, `ax-cli` enables `notebook` by default.

Notebook server behavior is session-scoped rather than process-global:

- each browser tab gets its own session identifier
- execution state is isolated per session
- `/reset` resets only the targeted session
- session state expires after inactivity and is cleaned up server-side

Notebook export and browser rendering also use an explicit trust model:

- local interactive notebooks are treated as `trusted_local`
- imported/shared content is treated as `untrusted`
- untrusted HTML and SVG pass through sanitization
- untrusted LaTeX-sensitive content is escaped before export

### Jupyter

`ax-jupyter` is the standalone Jupyter kernel crate. It is excluded from the default workspace build because it requires `libzmq`, but it now has a complete product-facing install path through the `axioma-jupyter` binary.

Build the kernel binary directly:

```bash
cargo build --manifest-path crates/ax-jupyter/Cargo.toml --bin axioma-jupyter
```

Print the generated kernelspec without installing it:

```bash
cargo run --manifest-path crates/ax-jupyter/Cargo.toml --bin axioma-jupyter -- \
  print-kernelspec \
  --binary "$(pwd)/crates/ax-jupyter/target/debug/axioma-jupyter"
```

Install a user kernelspec for a normal Jupyter frontend:

```bash
cargo run --manifest-path crates/ax-jupyter/Cargo.toml --bin axioma-jupyter -- \
  install \
  --user \
  --binary "$(pwd)/crates/ax-jupyter/target/debug/axioma-jupyter"
```

You can also make startup behavior explicit at install time:

```bash
cargo run --manifest-path crates/ax-jupyter/Cargo.toml --bin axioma-jupyter -- \
  install \
  --user \
  --binary "$(pwd)/crates/ax-jupyter/target/debug/axioma-jupyter" \
  --working-dir "$(pwd)" \
  --std-path "$(pwd)/std"
```

That writes a `kernel.json` containing:

- `argv`: the `axioma-jupyter` binary plus the `{connection_file}` placeholder
- `interrupt_mode: "message"` so control-channel interrupts use Jupyter protocol messages rather than OS signals
- `env` entries for `AXIOMA_JUPYTER_WORKDIR` and `AXIOMA_STD_PATH` when provided
- metadata describing the Axioma kernel session/trust model

Kernel startup behavior is explicit:

- If `AXIOMA_JUPYTER_WORKDIR` is set, Axioma uses it as the module-resolution working directory.
- Otherwise, it uses the process current working directory inherited from the frontend.
- Import search paths are then built deterministically from:
  1. `AXIOMA_STD_PATH`, if set
  2. the resolved working directory
  3. executable-relative standard-library locations

At startup the kernel logs:

- connection file path
- bound transport endpoints
- resolved working directory
- effective `AXIOMA_STD_PATH`
- final import search path list

That output is intended to make notebook-launch failures diagnosable without guessing at hidden environment state.

The kernel now implements the core interactive request set mainstream frontends expect:

- `kernel_info_request`
- `execute_request`
- `complete_request`
- `inspect_request`
- `history_request`
- `is_complete_request`
- `interrupt_request`
- `shutdown_request`

Execution behavior follows the normal Jupyter model closely:

- normal evaluation publishes `status(busy) -> execute_input -> stream/display/result/error -> execute_reply -> status(idle)`
- ordinary final values are emitted as `execute_result`, not `display_data`
- explicit side-channel rich displays use `display_data`
- `text/plain` fallback is always present for ordinary expression results
- rich MIME bundles can include `text/latex`, `text/markdown`, `text/html`, `image/svg+xml`, and `application/json`
- `execution_count` advances only for execute requests that count in history
- `silent` execution suppresses visible `Out[n]` behavior
- `store_history = false` keeps execution visible but does not advance stored history

The kernel is now resilient under real frontend traffic:

- malformed frames, bad signatures, invalid JSON, and unsupported message types do not terminate the process
- request-level failures are classified and handled without corrupting kernel state
- parent headers and metadata objects are propagated correctly on replies and IOPub messages
- control-channel interrupts cancel real in-flight evaluation cooperatively rather than faking success

Notebook/kernel integration details:

- Notebook browser tabs use per-tab sessions for `ax-notebook`; Jupyter kernel state is per-kernel-process.
- Notebook and Jupyter now use the same import/module search-path construction and shared import-resolution behavior.
- Exported notebook HTML/LaTeX uses the trust/sanitization model documented in the notebook section and in code comments.
- Jupyter interrupts are cooperative and protocol-driven. A running evaluation is cancelled through the control channel, returns an interrupted error, restores `idle`, and leaves subsequent execution working normally.
- Jupyter shutdown is graceful. The kernel replies to `shutdown_request`, cancels in-flight work if needed, lets the active execute finish its protocol cleanup, and then exits.

Testing the kernel directly:

```bash
cargo test --manifest-path crates/ax-jupyter/Cargo.toml
```

That test suite includes signed protocol/integration coverage for:

- execute success and failure ordering
- completion, inspection, history, and completeness detection
- interrupt and shutdown lifecycle
- malformed-message recovery
- rich MIME output emission
- notebook/kernel semantic parity for shared behaviors such as import resolution

Deliberately out of scope for now:

- Jupyter comms/widgets
- debugger protocol support
- stdin request handling

Those areas are not claimed as supported until they are wired through end to end.

### Plugins

Plugin support is enabled in the CLI through the `plugins` feature, and in this checkout it is enabled by default for `ax-cli`.

The example plugin is [`plugins/axp-echo`](./plugins/axp-echo), a small Wasm plugin used by tests to validate:

- plugin registry resolution from `axioma.toml`
- plugin invocation through `axioma plugin run`
- request/response serialization through `ax-plugin-api`

## AAS Validation And AI-Oriented Tools

The repository contains a schema-driven AAS validation path and a small set of AI-adjacent workflow tools.

### AAS Validation

The schema lives at [`spec/aas.schema.json`](./spec/aas.schema.json). The CLI command:

```bash
axioma validate path/to/file.json
```

does the following:

- loads the schema
- validates the input JSON
- emits structured diagnostics
- writes a trace report unless `--no-trace` is used

### AI Helpers

The CLI also exposes:

- `axioma ai fix`
- `axioma ai pack`
- `axioma ai apply`

Those surfaces are supported by files such as:

- [`crates/ax-cli/src/cmd_ai.rs`](./crates/ax-cli/src/cmd_ai.rs)
- [`crates/ax-cli/src/cmd_ai_pack.rs`](./crates/ax-cli/src/cmd_ai_pack.rs)
- [`crates/ax-cli/src/cmd_ai_apply.rs`](./crates/ax-cli/src/cmd_ai_apply.rs)

The auto-generated language reference consumed by such workflows is:

- [`share/axioma-llm-context.md`](./share/axioma-llm-context.md)

It is produced by:

```bash
cargo run -p axioma -- docgen
```

## Development Workflows

### Common Commands

```bash
cargo check
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

### Targeted Commands

```bash
cargo run -p axioma -- repl
cargo run -p axioma -- run examples/calculus_demo.ax
cargo run -p axioma -- export examples/gr_tutorial.ax --format html
cargo run -p axioma -- export examples/gr_tutorial.ax --format latex
cargo run -p axioma -- codegen examples/calculus_demo.ax --target python
cargo run -p axioma -- paths
cargo run -p axioma -- validate examples/ok.aas.json
```

### VS Code Extension

```bash
cd editors/vscode
npm install
npm run compile
npm run package
```

### Plugin Example

```bash
cargo build -p axp-echo --target wasm32-unknown-unknown
cargo run -p axioma -- plugin list
cargo run -p axioma -- plugin run --plugin axp-echo --op transform --args '{"foo":123}'
```

## Testing And Verification

This repository already has a broad test surface. From the observed code paths and current artifacts, the workspace uses:

- unit tests in individual crates
- integration tests for CLI behavior
- export tests
- plugin registry/plugin run tests
- validation tests
- doc generation support

Examples of high-signal checks:

```bash
cargo test -p axioma
cargo test -p ax-eval
cargo test -p ax-tensor
cargo test -p ax-spinor
cargo test -p ax-perturb
cargo test -p ax-graded
cargo test -p ax-mcp -- --test-threads=1
```

If you touch the REPL specifically, the focused check is:

```bash
cargo check -p axioma
cargo test -p axioma
cargo clippy -p axioma -- -D warnings
```

## License

This repository is standardized on **Apache-2.0**.

That now includes:

- the Rust workspace metadata in [`Cargo.toml`](Cargo.toml)
- the main CLI crate metadata in [`crates/ax-cli/Cargo.toml`](./crates/ax-cli/Cargo.toml)
- the root license file at [`LICENSE`](LICENSE)
- the VS Code extension package metadata in [`editors/vscode/package.json`](./editors/vscode/package.json)
- the VS Code extension license file at [`editors/vscode/LICENSE`](./editors/vscode/LICENSE)

The previous first-party MIT declaration in the VS Code extension has been aligned with the repository-wide Apache-2.0 license so the published metadata and checked-in license files agree.
