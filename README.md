# Axioma

A scientific computing language for physicists. Tensor algebra, symbolic CAS, convention tracking, and code generation - in a single Rust binary with zero system dependencies.

## Install

```bash
cargo install --git https://github.com/manavm12/axioma --bin axioma
```

No Python. No SymPy. No Boost. No cmake. No libzmq. Just Rust.

Then:

```bash
axioma repl
```

## What it does

```
axioma> 1/3 + 1/3 + 1/3
1
  LaTeX: 1

axioma> diff(sin(x^2), x)
2·x·cos(x²)
  LaTeX: 2 x \cos\!\left(x^{2}\right)

axioma> integrate(x^2, x)
⅓·x³
  LaTeX: \frac{x^{3}}{3}

axioma> solve(x^2 - 5*x + 6, x)
[2, 3]

axioma> series(exp(x), x, 0, 4)
1 + x + ½·x² + ⅙·x³ + 1/24·x⁴
```

## General Relativity

Verify the Schwarzschild metric is a vacuum solution in 5 lines:

```
axioma> let g = metric(diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r^2, r^2 * sin(theta)^2))
axioma> let coords = [t, r, theta, phi]
axioma> let Gamma = christoffel(g, coords)
axioma> let R = riemann(Gamma, coords)
axioma> ricci(R)
0   // all components vanish - vacuum solution confirmed in 0.03s
```

## Tensor Algebra

Full abstract index tensor algebra with property declarations, canonicalization, and multi-term symmetry simplification:

```
indices spacetime [a, b, c, d, e, f] dim=4
property g metric
property R riemann_symmetry

// Canonicalize using tensor symmetries
canonicalise(R[a-, b-, c-, d-] + R[a-, c-, d-, b-] + R[a-, d-, b-, c-])
// → 0  (first Bianchi identity)

// Eliminate Kronecker deltas
eliminate_kronecker(delta[mu+, nu-] * T[nu+, rho-])
// → T[mu+, rho-]

// Raise/lower indices with the metric
eliminate_metric(g[mu-, nu-] * V[nu+])
// → V[mu-]
```

## Convention Tracking

Different textbooks use different sign conventions. Axioma tracks them:

```
import std.conventions.mtw
// metric_signature: MostlyPlus, riemann_sign: MTW

convention metric_signature mostly_minus
// Warning: changing convention mid-computation
```

## Trust-Labeled Rewrites

Every rewrite rule carries a trust level. Know exactly how trustworthy your result is:

```
rule [exact] pythag: sin(x_)^2 + cos(x_)^2 => 1
rule [heuristic] my_approx: f(x_) => g(x_)

rewrite(expr)
// trust: exact
// or: trust: heuristic (used rule: my_approx)
```

## Code Generation

Generate Python, Rust, or C++ from any expression:

```
axioma> :codegen python
# math.sin(x)**2 + math.cos(x)**2

axioma> :codegen rust
# x.sin().powi(2) + x.cos().powi(2)

axioma> :codegen cpp
# std::pow(std::sin(x), 2) + std::pow(std::cos(x), 2)
```

## Quantum Mechanics

```
// Pauli matrices with proper complex numbers
let sx = pauli_x()
let sy = pauli_y()
let sz = pauli_z()

// Verify: [σ_x, σ_y] = 2i σ_z
commutator(sx, sy)

// Gamma matrix traces for QFT
gamma_trace([mu, nu])
// → 4·g[mu+, nu+]
```

## What Axioma has that Cadabra doesn't

| Feature | Axioma | Cadabra |
|---|---|---|
| Native CAS (diff, integrate, solve, series, limits) | Yes | No (delegates to SymPy) |
| Convention tracking with mismatch detection | Yes | No |
| Trust-labeled rewrites with audit trails | Yes | No |
| Semantic equivalence checker | Yes | No |
| Code generation (Python, Rust, C++) | Yes | No |
| Units and dimensional analysis | Yes | No |
| ODE solver (symbolic + RK4 numerical) | Yes | No |
| Functional derivatives / Euler-Lagrange | Yes | No |
| Normal ordering / Wick expansion | Yes | No |
| 2D plotting (SVG output) | Yes | No |
| Zero system dependencies | Yes | No (Python, SymPy, Boost, GTK) |
| Single binary install | Yes | No |

## REPL Commands

| Command | Description |
|---|---|
| `:help` | Show help |
| `:env` | Show all bindings |
| `:rules` | Show user-defined rules |
| `:assumptions` | Show assumptions |
| `:convention` | Show active convention |
| `:codegen python` | Generate Python for last result |
| `:codegen rust` | Generate Rust |
| `:codegen cpp` | Generate C++ |
| `:reset` | Clear environment |
| `:quit` | Exit |

## Run Scripts

```bash
axioma run examples/gr_tutorial.ax
axioma run examples/qm_tutorial.ax
axioma run examples/calculus_demo.ax
```

## Browser Notebook

```bash
axioma notebook
# Opens at http://localhost:8888
# KaTeX rendering, cell evaluation, export to LaTeX/HTML
```

## Architecture

~26,000 lines of Rust across 30 crates:

| Crate | Purpose |
|---|---|
| `ax-ir` | Expression IR with canonical constructors, indices, properties |
| `ax-syntax` | Logos lexer + Rowan CST parser |
| `ax-core-ir` | CST to IR lowering, LaTeX input preprocessing |
| `ax-eval` | Tree-walking evaluator with 80+ builtins |
| `ax-rewrite` | Pattern matching, term rewriting, traced rewrites |
| `ax-tensor` | Abstract tensor algebra, canonicalize, meld, evaluate |
| `ax-perm` | Permutation groups, Schreier-Sims, canonical forms |
| `ax-young` | Young tableaux, representation theory |
| `ax-render` | LaTeX and Unicode rendering |
| `ax-solve` | Polynomial solver, linear systems |
| `ax-linalg` | Determinant, inverse, eigenvalues, tensor product |
| `ax-forms` | Differential forms, wedge product, exterior derivative |
| `ax-qm` | Pauli matrices, gamma traces, Fierz, normal ordering |
| `ax-ode` | Symbolic and numerical ODE solver |
| `ax-variational` | Functional derivatives, Euler-Lagrange |
| `ax-equiv` | Semantic equivalence checking |
| `ax-codegen` | Code generation to Python, Rust, C++ |
| `ax-units` | SI and natural units, dimensional analysis |
| `ax-plot` | 2D SVG plotting |
| `ax-notebook` | Browser notebook with KaTeX |
| `ax-cli` | REPL, script runner, all commands |

## Standard Library

27 library files covering:

- `std/gr/` - Minkowski, Schwarzschild, de Sitter, FRW, Kerr-Newman metrics
- `std/qm/` - Spin, Bell states, harmonic oscillator
- `std/qft/` - Gamma matrices, Dirac equation, scalar field theory
- `std/conventions/` - MTW, Weinberg, Landau, particle physics conventions
- `std/units/` - SI, natural units, CGS
- `std/physics/` - Klein-Gordon, Maxwell, classical mechanics

## Optional Features

```bash
# With WASM plugin system (adds wasmtime dependency):
cargo install --git https://github.com/manavm12/axioma --bin axioma --features plugins

# Jupyter kernel (requires libzmq on system):
cd crates/ax-jupyter && cargo build --release
jupyter kernelspec install --user share/jupyter/kernels/axioma
```

## License

Apache-2.0
