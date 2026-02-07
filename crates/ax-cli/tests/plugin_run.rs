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

#[test]
fn plugin_run_outputs_json_ok() {
    // Minimal plugin: ignores request and returns a fixed PluginResponse payload.
    // ABI: exports memory, axioma_alloc, axioma_entry, axioma_free.
    let json = r#"{"ok":true,"result":{"hello":"world"},"diagnostics":[]}"#;
    let payload_len = json.len() as u32;
    let len_bytes = payload_len.to_le_bytes();

    // Put [u32 len][json bytes] at offset 1024.
    let mut blob = Vec::new();
    blob.extend_from_slice(&len_bytes);
    blob.extend_from_slice(json.as_bytes());

    let data_bytes = blob
        .iter()
        .map(|b| format!("\\{:02x}", b))
        .collect::<String>();

    let wat_src = format!(
        r#"(module
  (memory (export "memory") 2)
  (func (export "axioma_alloc") (param i32) (result i32)
    (i32.const 8))
  (func (export "axioma_free") (param i32) (param i32)
    nop)
  (data (i32.const 1024) "{data_bytes}")
  (func (export "axioma_entry") (param i32) (param i32) (result i32)
    (i32.const 1024))
)"#
    );

    let wasm = wat::parse_str(&wat_src).expect("wat -> wasm");
    let tmp = repo_root().join("build/tmp_test_plugin.wasm");
    fs::create_dir_all(tmp.parent().unwrap()).unwrap();
    fs::write(&tmp, wasm).unwrap();

    let out = bin()
        .args([
            "plugin",
            "run",
            "--wasm",
            tmp.to_string_lossy().as_ref(),
            "--op",
            "transform",
            "--args",
            r#"{"x":1}"#,
            "--no-trace",
        ])
        .output()
        .expect("run axioma plugin run");

    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout json");
    assert_eq!(v["ok"], true);
    assert_eq!(v["result"]["hello"], "world");
}
