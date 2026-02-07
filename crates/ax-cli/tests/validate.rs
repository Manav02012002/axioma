use std::{fs, path::PathBuf, process::Command};

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // ax-cli
    p.pop(); // crates
    p
}

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_axioma"));
    c.env("AXIOMA_ROOT", repo_root());
    c
}

fn repo_file(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn run_id(schema_hash_hex: &str, script_hash_hex: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"axioma/");
    h.update(b"0.1.0");
    h.update(b"\n");
    h.update(schema_hash_hex.as_bytes());
    h.update(b"\n");
    h.update(script_hash_hex.as_bytes());
    h.finalize().to_hex().to_string()
}

#[test]
fn ok_script_exits_zero() {
    let ok = repo_file("examples/ok.aas.json");
    let out = bin()
        .args(["validate", ok.to_string_lossy().as_ref(), "--no-trace"])
        .output()
        .expect("run axioma validate");

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
    let out = bin()
        .args(["validate", bad.to_string_lossy().as_ref(), "--no-trace"])
        .output()
        .expect("run axioma validate");

    assert!(!out.status.success(), "expected failure");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("\"code\": \"AXAAS0001\""), "stdout was:\n{s}");
}

#[test]
fn ok_script_writes_expected_trace_file() {
    let schema_bytes = fs::read(repo_file("spec/aas.schema.json")).expect("read schema");
    let script_bytes = fs::read(repo_file("examples/ok.aas.json")).expect("read script");

    let schema_hash = blake3_hex(&schema_bytes);
    let script_hash = blake3_hex(&script_bytes);
    let rid = run_id(&schema_hash, &script_hash);

    let out = bin()
        .args([
            "validate",
            repo_file("examples/ok.aas.json").to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run axioma validate");
    assert!(out.status.success(), "expected success");

    let trace_path = repo_root().join("build/trace").join(format!("{rid}.json"));

    assert!(
        trace_path.exists(),
        "expected trace file to exist at {}",
        trace_path.display()
    );
}

#[test]
fn paths_command_outputs_root_spec_build() {
    let out = bin().args(["paths"]).output().expect("run axioma paths");
    assert!(out.status.success(), "paths should succeed");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("\"spec_dir\""), "stdout was:\n{s}");
    assert!(s.contains("\"build_dir\""), "stdout was:\n{s}");
}
