use std::{path::PathBuf, process::Command};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/ax-cli
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // ax-cli
    p.pop(); // crates
    p
}

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ax-cli"));
    c.env("AXIOMA_ROOT", repo_root());
    c
}

fn repo_file(rel: &str) -> String {
    repo_root().join(rel).to_string_lossy().to_string()
}

#[test]
fn ok_script_exits_zero() {
    let ok = repo_file("examples/ok.aas.json");
    let out = bin().args(["validate", &ok]).output().expect("run ax-cli");

    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("ok: AAS is valid"), "stdout was:\n{s}");
}

#[test]
fn bad_script_exits_nonzero_and_emits_diag_code() {
    let bad = repo_file("examples/bad.aas.json");
    let out = bin().args(["validate", &bad]).output().expect("run ax-cli");

    assert!(!out.status.success(), "expected failure");

    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("\"code\": \"AXAAS0001\""), "stdout was:\n{s}");
}
