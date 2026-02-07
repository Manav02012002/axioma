use ax_plugin_api::PluginRequest;
use ax_plugin_host::WasmPlugin;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // ax-plugin-host
    p.pop(); // crates
    p
}

#[test]
fn wasm_echo_roundtrip() {
    let wasm = repo_root().join("target/wasm32-unknown-unknown/debug/axp_echo.wasm");
    assert!(wasm.exists(), "missing wasm at {}", wasm.display());

    let plugin = WasmPlugin::from_file(&wasm).expect("load wasm");

    let req = PluginRequest {
        plugin: "axp-echo".to_string(),
        op: "transform".to_string(),
        args: serde_json::json!({"x": 1, "y": [2,3]}),
    };

    let resp = plugin.call(&req).expect("call");
    assert!(resp.ok, "plugin returned ok=false: {:?}", resp.diagnostics);
    assert_eq!(resp.result["echo"]["x"], 1);
}
