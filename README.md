# Axioma

A next-generation scientific computing language for physicists. Type ASCII, get publication-quality LaTeX.

## Features

- **Symbolic by default** — `x + 0` → `x`, not an error. Exact arithmetic: `1/3 + 1/3 + 1/3 = 1`.
- **Tensor index notation** — `T[mu-, nu+]` renders as $T_{\mu}{}^{\nu}$. Einstein summation convention built in.
- **LaTeX rendering** — every expression has a beautiful LaTeX form. Greek letters auto-detected.
- **Symbolic calculus** — differentiation, integration, Taylor series.
- **General Relativity** — compute Christoffel symbols, Riemann tensor, Ricci tensor, Einstein tensor, and Kretschner scalar from any metric.
- **Pattern matching & rewriting** — extensible term rewriting engine.
- **WASM plugin system** — extend Axioma with sandboxed plugins.

## Quick Start
```bash
cargo build --release
./target/release/axioma repl
```
````
axioma> 1/3 + 1/3 + 1/3
1
  LaTeX: 1

axioma> diff(sin(x^2), x)
2·x·cos(x²)
  LaTeX: 2x\cos\!\left(x^{2}\right)

axioma> T[mu-, nu+]
T_{μ}^{ν}
  LaTeX: T_{\mu}{}^{\nu}

axioma> series(exp(x), x, 0, 4)
1 + x + ½·x² + ⅓·x³ + ...
  LaTeX: 1 + x + \frac{x^{2}}{2} + \frac{x^{3}}{6} + \frac{x^{4}}{24}
````

## General Relativity
Verify that the Schwarzschild metric is a vacuum solution:
````
axioma> let g = metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))
axioma> let coords = [t, r, theta, phi]
axioma> let Gamma = christoffel(g, coords)
axioma> let R = riemann(Gamma, coords)
axioma> let Ric = ricci(R)
axioma> Ric
0   (all components vanish — vacuum solution confirmed)
````

## Run Scripts
```bash
./target/release/axioma run std/gr/schwarzschild.ax
```

## Architecture
Axioma is built in Rust as a workspace of focused crates:

| Crate | Purpose |
| --- | --- |
| ax-syntax | Logos lexer + Rowan lossless CST parser |
| ax-ir | Typed expression IR with canonical simplification |
| ax-core-ir | CST → IR lowering |
| ax-eval | Tree-walking symbolic evaluator |
| ax-rewrite | Pattern matching + term rewriting engine |
| ax-tensor | Tensor algebra, Christoffel, Riemann, Ricci |
| ax-render | LaTeX + Unicode rendering |
| ax-cli | REPL, script runner, render command |

## License
Apache-2.0
