use ax_ai_proto::{AiEditRequest, Edit, Span};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

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

fn unique_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    repo_root().join("build").join(format!("{name}-{nanos}.ax"))
}

fn write_request(path: &PathBuf, req: &AiEditRequest) {
    let json = serde_json::to_string_pretty(req).expect("serialize request");
    fs::write(path, json).expect("write request");
}

#[test]
fn ai_apply_applies_multiple_original_spans_atomically() {
    let file = unique_path("ai-apply-input");
    let req_file = unique_path("ai-apply-request");
    fs::create_dir_all(file.parent().expect("parent")).expect("mkdir build");

    let src = "abcdef";
    fs::write(&file, src).expect("write input");

    let req = AiEditRequest {
        version: "1".to_string(),
        file_hash_blake3_hex: blake3::hash(src.as_bytes()).to_hex().to_string(),
        edits: vec![
            Edit::Replace {
                span: Span { start: 1, end: 3 },
                replacement: "XYZ".to_string(),
            },
            Edit::Replace {
                span: Span { start: 4, end: 6 },
                replacement: "QQ".to_string(),
            },
        ],
        rationale: None,
    };
    write_request(&req_file, &req);

    let out = bin()
        .args([
            "ai",
            "apply",
            file.to_string_lossy().as_ref(),
            req_file.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run axioma ai apply");

    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(fs::read_to_string(&file).expect("read output"), "aXYZdQQ");
}

#[test]
fn ai_apply_rejects_invalid_batch_without_writing_partial_output() {
    let file = unique_path("ai-apply-invalid-input");
    let req_file = unique_path("ai-apply-invalid-request");
    fs::create_dir_all(file.parent().expect("parent")).expect("mkdir build");

    let src = "abcdef";
    fs::write(&file, src).expect("write input");

    let req = AiEditRequest {
        version: "1".to_string(),
        file_hash_blake3_hex: blake3::hash(src.as_bytes()).to_hex().to_string(),
        edits: vec![
            Edit::Replace {
                span: Span { start: 1, end: 3 },
                replacement: "XYZ".to_string(),
            },
            Edit::Replace {
                span: Span { start: 2, end: 5 },
                replacement: "bad".to_string(),
            },
        ],
        rationale: None,
    };
    write_request(&req_file, &req);

    let out = bin()
        .args([
            "ai",
            "apply",
            file.to_string_lossy().as_ref(),
            req_file.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run axioma ai apply");

    assert!(!out.status.success(), "expected failure");
    assert_eq!(fs::read_to_string(&file).expect("read original"), src);
}
