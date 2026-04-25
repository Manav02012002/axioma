use ax_ir::Expr;
use ax_plugin_host::{sparse_eigenpairs_via_plugin, WasmPlugin};
use std::{path::PathBuf, process::Command};

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn build_sparse_demo_wasm() -> PathBuf {
    let status = Command::new("cargo")
        .current_dir(repo_root())
        .args([
            "build",
            "-p",
            "axp-sparse-demo",
            "--target",
            "wasm32-unknown-unknown",
        ])
        .status()
        .expect("cargo build axp-sparse-demo wasm");
    assert!(status.success(), "cargo build axp-sparse-demo failed");

    let wasm = repo_root().join("target/wasm32-unknown-unknown/debug/axp_sparse_demo.wasm");
    assert!(wasm.exists(), "missing wasm at {}", wasm.display());
    wasm
}

#[test]
fn sparse_demo_wasm_loads_and_returns_diagonal_eigenpairs() {
    let wasm = build_sparse_demo_wasm();
    let plugin = WasmPlugin::from_file(&wasm).expect("load sparse demo wasm");

    let response = sparse_eigenpairs_via_plugin(
        &plugin,
        "axp-sparse-demo",
        &[
            vec![Expr::Int(1.into()), Expr::Int(0.into())],
            vec![Expr::Int(0.into()), Expr::Int(2.into())],
        ],
        2,
        "SM",
    )
    .expect("sparse eigenpairs via plugin");

    assert!(response.converged);
    let mut eigenvalues = response
        .eigenpairs
        .into_iter()
        .map(|pair| pair.eigenvalue)
        .collect::<Vec<_>>();
    eigenvalues.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(eigenvalues, vec![(1.0, 0.0), (2.0, 0.0)]);
}
