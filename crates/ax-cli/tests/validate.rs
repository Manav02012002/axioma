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
    h.update(schema_hash_hex.as_bytes());
    h.update(b":");
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
    let schema_json: serde_json::Value = serde_json::from_slice(
        &fs::read(repo_file("spec/aas.schema.json")).expect("read schema"),
    )
    .expect("parse schema json");
    let script_json: serde_json::Value = serde_json::from_slice(
        &fs::read(repo_file("examples/ok.aas.json")).expect("read script"),
    )
    .expect("parse script json");

    let schema_hash =
        blake3_hex(&serde_json::to_vec(&schema_json).expect("serialize schema json"));
    let script_hash =
        blake3_hex(&serde_json::to_vec(&script_json).expect("serialize script json"));
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
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout json");
    let root = v["root"].as_str().expect("root str");
    let spec = v["spec"].as_str().expect("spec str");
    let build = v["build"].as_str().expect("build str");

    assert_eq!(PathBuf::from(root), repo_root(), "root={root}");
    assert_eq!(PathBuf::from(spec), repo_root().join("spec"), "spec={spec}");
    assert_eq!(PathBuf::from(build), repo_root().join("build"), "build={build}");
}
