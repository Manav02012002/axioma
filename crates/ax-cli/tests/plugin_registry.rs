use std::process::Command;

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_axioma"));
    c.env_remove("AXIOMA_ROOT");
    c
}

#[test]
fn plugin_list_includes_axp_echo() {
    let out = bin().args(["plugin", "list"]).output().expect("run");
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    let plugins = v["plugins"].as_array().expect("plugins array");
    assert!(plugins.iter().any(|x| x["id"] == "axp-echo"));
}

#[test]
fn plugin_run_by_id_works() {
    // Ensure wasm exists (build step)
    let st = Command::new("cargo")
        .args([
            "build",
            "-p",
            "axp-echo",
            "--target",
            "wasm32-unknown-unknown",
        ])
        .status()
        .expect("cargo build axp-echo");
    assert!(st.success());

    let out = bin()
        .args([
            "plugin",
            "run",
            "--plugin",
            "axp-echo",
            "--op",
            "transform",
            "--args",
            r#"{"foo":123}"#,
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["ok"], true);
    assert_eq!(v["result"]["plugin"], "axp-echo");
    assert_eq!(v["result"]["echo"]["foo"], 123);
}
