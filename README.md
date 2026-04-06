# Axioma

Axioma is a Rust-native symbolic computing system aimed at physics workflows:
tensor algebra, CAS operations, convention tracking, code generation, and
domain-specific tools for GR, QFT, perturbation theory, and quantum mechanics.

The core CLI does not delegate algebra to Python or SymPy. It uses Axioma's own
IR, evaluator, tensor engine, renderers, and standard library.

## Install

```bash
cargo install --git https://github.com/Manav02012002/axioma --bin axioma
```

Then:

```bash
axioma repl
```

For local development:

```bash
cargo run -p axioma -- repl
cargo run -p axioma -- run examples/equation_manipulation.ax
cargo test
```

## Quick Examples

```text
axioma> 1/3 + 1/3 + 1/3
1
  LaTeX: 1

axioma> diff(sin(x^2), x)
2cos(x²)x
  LaTeX: 2\cos\!\left({x}^{2}\right)x

axioma> integrate(x^2, x)
⅓x³
  LaTeX: \frac{1}{3}{x}^{3}

axioma> solve(x^2 - 5*x + 6, x)
[3, 2]

axioma> series(exp(x), x, 0, 4)
1 + x + ½x² + 1/24x⁴ + 1/6x³
```

## Equation Manipulation

Equations are first-class expression objects. A top-level `==` lowers to an
internal equation node and renders naturally.

```text
let eq1 = (2x + 3y == 12)
to_rhs(eq1, y)
// 2x = 12 + -3y

isolate(to_rhs(eq1, y), x)
// x = 1/2 * (12 - 3y)

differentiate_eq(x^2 + y^2 == r^2, x)
// 2x = 0

let rule = eq_to_rule(isolate(to_rhs(eq1, y), x))
rule
// x => 1/2 * (12 - 3y)
```

Available equation helpers include `eq`, `get_lhs`, `get_rhs`, `swap_sides`,
`multiply_through`, `add_through`, `to_rhs`, `to_lhs`, `isolate`,
`eq_to_rule`, `eq_to_subrule`, `differentiate_eq`, `integrate_eq`,
`substitute_eq`, `raise_eq`, and `lower_eq`.

See `examples/equation_manipulation.ax`.

## Tensor Algebra and GR

Axioma supports abstract indexed tensors, property declarations, canonical forms,
dummy-index handling, Kronecker/metric contraction, epsilon-delta conversion,
and GR component pipelines.

```text
indices spacetime [a, b, c, d] dim=4
property R riemann_symmetry

meld(R[a-, b-, c-, d-] + R[a-, c-, d-, b-] + R[a-, d-, b-, c-])
// 0, by the first Bianchi identity

eliminate_kronecker(delta[mu+, nu-] * T[nu+, rho-])
// T[mu+, rho-]

eliminate_metric(g[mu-, nu-] * V[nu+])
// V[mu-]
```

Component GR support includes Christoffel symbols, Riemann, Ricci, Ricci scalar,
Einstein tensor, and geodesic equations. Regression tests cover Minkowski
curvature zero and a Schwarzschild vacuum pipeline through the MCP interface.

Run:

```bash
cargo run -p axioma -- run examples/gr_tutorial.ax
cargo run -p axioma -- run examples/schwarzschild.ax
```

## Perturbation Theory and Cosmology

`ax-perturb` provides perturbative expansion of tensor expressions and derived
geometric quantities:

- metric perturbation expansion `g = g0 + eps*h + eps^2*k + ...`
- inverse metric perturbation
- Christoffel, Riemann, Ricci, and Einstein tensor perturbations
- FRW scalar perturbation equations
- Mukhanov-Sasaki equation, leading power spectrum, spectral index, and tensor-to-scalar ratio
- SVT decomposition, Bardeen variables, Regge-Wheeler and Zerilli equations

Examples:

```bash
cargo run -p axioma -- run examples/linearized_gravity.ax
cargo run -p axioma -- run examples/cosmological_perturbations.ax
```

## Spinor-Helicity and Momentum Twistors

`ax-spinor` implements spinor-helicity expressions and identities:

- angle and square brackets
- Mandelstam expansion/collection
- Schouten identities and momentum conservation
- spinor chains
- Parke-Taylor amplitudes and three-point amplitudes
- BCFW shifts/decomposition
- momentum-twistor four-brackets and Plucker relations

The eval layer exposes functions such as `angle`, `square`, `mandelstam`,
`parke_taylor`, `three_point_mhv`, `expand_chain`, `spinor_simplify`,
`bcfw_shift`, `bcfw_decomposition`, `four_bracket`, and `plucker`.

## Graded Algebra, Superspace, and BRST

`ax-graded` covers Z2/Z-graded algebra with Koszul signs and nilpotent
fermionic symbols. It also includes N=1 superspace and BRST utilities:

- `graded`, `graded_commutator`, `graded_simplify`
- `setup_superspace`, `expand_superfield`, `chiral_superfield`,
  `antichiral_superfield`, `vector_superfield_wz`
- D-algebra operators `d_alpha`, `d_bar`, `d_squared`, `d_bar_squared`
- `superspace_integrate`
- Yang-Mills/abelian BRST setup, ghost number, nilpotency checks, ghost filtering

Examples:

```bash
cargo run -p axioma -- run examples/wess_zumino_model.ax
cargo run -p axioma -- run examples/brst_qed.ax
```

## Quantum Mechanics and QFT

Axioma includes matrix-based quantum mechanics utilities and symbolic QFT
building blocks:

- Pauli matrices, commutators, anticommutators
- braket products, density matrices, partial traces
- gamma traces with symbolic or concrete metrics
- normal ordering and Wick-style operator manipulation
- standard library files for spin, Bell states, harmonic oscillator, gamma matrices, Dirac theory, scalar fields, BRST, and superspace

Run:

```bash
cargo run -p axioma -- run examples/qm_tutorial.ax
```

## CAS, Solvers, and Analysis

The evaluator includes:

- exact rational arithmetic and simplification
- differentiation, integration, limits, and series
- polynomial solving and linear systems
- determinant, inverse, trace, tensor products
- differential forms, wedge product, exterior derivative, Hodge dual
- variational derivatives and Euler-Lagrange equations
- ODE/PDE utilities, including RK4 and PDE classification
- units and dimensional analysis
- SVG plotting

## Rewrites, Conventions, and Trust

Rewrite rules carry trust levels:

```text
rule [exact] pythag: sin(x_)^2 + cos(x_)^2 => 1
rule [heuristic] approximation: f(x_) => g(x_)
```

Convention tracking is built into the IR and eval environment:

```text
import std.conventions.mtw
convention metric_signature mostly_minus
```

Supported convention fields include metric signature, Riemann sign, Ricci
contraction, Levi-Civita normalization, and Fourier sign.

## Rendering and Code Generation

Expressions render to Unicode and LaTeX. The CLI and notebook show both forms
where appropriate.

Code generation targets include Python, Rust, and C++:

```text
axioma> :codegen python
axioma> :codegen rust
axioma> :codegen cpp
```

## MCP, Notebook, LSP, and Jupyter

The workspace includes:

- `axioma` CLI for REPL and script execution
- `axioma-mcp` server for semantic tool access
- notebook support with browser rendering
- LSP server
- Jupyter kernel crate

Jupyter support is separate from the main CLI and may require system `libzmq`.

## REPL Commands

| Command | Description |
|---|---|
| `:help` | Show help |
| `:env` | Show bindings |
| `:rules` | Show user-defined rules |
| `:assumptions` | Show assumptions |
| `:convention` | Show active convention |
| `:codegen python` | Generate Python for the last result |
| `:codegen rust` | Generate Rust for the last result |
| `:codegen cpp` | Generate C++ for the last result |
| `:reset` | Clear environment |
| `:quit` | Exit |

## Examples

Current example scripts include:

```bash
cargo run -p axioma -- run examples/calculus_demo.ax
cargo run -p axioma -- run examples/gr_tutorial.ax
cargo run -p axioma -- run examples/qm_tutorial.ax
cargo run -p axioma -- run examples/schwarzschild.ax
cargo run -p axioma -- run examples/linearized_gravity.ax
cargo run -p axioma -- run examples/cosmological_perturbations.ax
cargo run -p axioma -- run examples/wess_zumino_model.ax
cargo run -p axioma -- run examples/brst_qed.ax
cargo run -p axioma -- run examples/equation_manipulation.ax
```

## Architecture

The workspace currently contains 34 Rust crates:

| Crate | Purpose |
|---|---|
| `ax-ir` | Expression IR, indices, assumptions, tensor properties, conventions |
| `ax-syntax` | Lexer/parser and syntax diagnostics |
| `ax-core-ir` | Source-to-IR lowering and LaTeX input preprocessing |
| `ax-eval` | Evaluator, builtins, registry, property store, equation utilities |
| `ax-rewrite` | Pattern rewriting and rewrite traces |
| `ax-compare` | Structural/tensor-aware pattern matching and substitution |
| `ax-tensor` | Abstract tensors, canonicalisation, component GR, tensor algorithms |
| `ax-perm` | Permutations and canonical forms |
| `ax-young` | Young tableaux and representation tools |
| `ax-render` | Unicode and LaTeX rendering |
| `ax-solve` | Algebraic solving and linear systems |
| `ax-linalg` | Matrix determinant, inverse, trace, tensor product |
| `ax-forms` | Differential forms |
| `ax-qm` | Quantum mechanics and gamma trace utilities |
| `ax-spinor` | Spinor-helicity and momentum twistor algebra |
| `ax-perturb` | GR/cosmology perturbation theory |
| `ax-graded` | Graded algebra, superspace, D-algebra, BRST |
| `ax-ode` | ODE/PDE utilities |
| `ax-variational` | Functional derivatives and Euler-Lagrange equations |
| `ax-equiv` | Semantic equivalence checking |
| `ax-codegen` | Python, Rust, and C++ code generation |
| `ax-units` | Units and dimensional analysis |
| `ax-plot` | SVG plotting |
| `ax-notebook` | Browser notebook support |
| `ax-mcp` | Model Context Protocol server |
| `ax-lsp` | Language Server Protocol server |
| `ax-jupyter` | Jupyter kernel integration |
| `ax-cli` | CLI, REPL, and script runner |
| `ax-ai-proto` | AI integration protocol types |
| `ax-context` | Context/state helpers |
| `ax-diagnostics` | Diagnostic data structures |
| `ax-plugin-api` | Plugin API types |
| `ax-plugin-host` | Plugin host runtime |
| `ax-trace` | Tracing and provenance helpers |

## Standard Library

The `std/` tree currently contains 29 `.ax` files covering:

- `std/gr/` - Minkowski, Schwarzschild, de Sitter, FRW, Kerr-Newman
- `std/qm/` - spin, Bell states, harmonic oscillator
- `std/qft/` - gamma matrices, Dirac, scalar fields, BRST, superspace, normal ordering
- `std/conventions/` - MTW, Weinberg, Landau, particle-physics conventions
- `std/tensor/` - index and symmetry helpers
- `std/units/` - SI, CGS, natural units
- `std/physics/` - classical mechanics, Klein-Gordon, Maxwell
- `std/calculus.ax`, `std/algebra.ax`, `std/trig.ax`

## Development Checks

Useful targeted checks:

```bash
cargo check
cargo test -p ax-eval
cargo test -p ax-tensor
cargo test -p ax-spinor
cargo test -p ax-perturb
cargo test -p ax-graded
cargo test -p ax-mcp -- --test-threads=1
```

## License

Apache-2.0
