# QM/QFT Audit

This audit is the repository-level guardrail for QM/QFT builtin coverage. It is not a prose checklist. The enforcement lives in `crates/ax-cli/tests/qm_qft_audit.rs`, and the machine-readable inventory lives in `tests/qm_qft_audit_manifest.json`.

The audit currently enforces three things for every public QM/QFT builtin exposed through `ax-eval`'s registry:

- the builtin is present in the manifest and still exists in the registry
- the builtin has registry documentation metadata
- the referenced std/example files and test files still exist

The manifest records, per builtin:

- registry category
- whether registry docs are required
- std/example paths that demonstrate usage
- direct or indirect test paths that cover the builtin
- whether notebook/Jupyter or MCP surface coverage exists where relevant

## Updating The Manifest

Update `tests/qm_qft_audit_manifest.json` whenever one of these happens:

- a QM/QFT builtin is added to `crates/ax-eval/src/registry.rs`
- a builtin is renamed or removed
- example or test files move
- notebook/Jupyter or MCP surface coverage changes

Keep the manifest aligned with the registry-derived public surface. The audit test compares the manifest names against the registry, so missing or stale entries fail CI.

## Running The Audit

Use:

```bash
cargo test -p ax-cli --test qm_qft_audit
cargo test -p ax-cli
```
