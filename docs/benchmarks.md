# Benchmarks

Axioma uses Criterion for the QM benchmark harness.

Run the QM workflow benchmarks without executing them:

```sh
cargo bench -p ax-qm --no-run
cargo bench -p ax-solve --no-run
```

Run the benchmark suites:

```sh
cargo bench -p ax-qm
cargo bench -p ax-solve
```

The current QM benches cover bipartite partial trace, supported small von Neumann
entropy cases, Lindblad RHS construction, and exact finite-dimensional Lindblad
steady-state solving.
