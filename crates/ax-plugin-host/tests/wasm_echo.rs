use ax_plugin_api::PluginRequest;
use ax_plugin_host::{summarize_symmetry_for_expr, WasmPlugin};
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

#[test]
fn summarize_symmetry_invalid_input_has_context() {
    let err = summarize_symmetry_for_expr("not valid").unwrap_err();
    assert!(
        err.to_string()
            .contains("failed to summarize symmetry for expression"),
        "{err:#}"
    );
}

#[test]
fn plugin_symmetry_summary_roundtrip() {
    let req = PluginRequest {
        plugin: "axp-echo".to_string(),
        op: "symmetry_summary".to_string(),
        args: serde_json::json!({
            "expr": "tableau_symmetry([[2,1]], slots=[[0,1,2]])"
        }),
    };

    let resp = axp_echo::handle_plugin_request(req);
    assert!(resp.ok, "plugin returned ok=false: {:?}", resp.diagnostics);
    assert!(
        resp.result["summary_json"]
            .as_str()
            .unwrap_or("")
            .contains("\"tableaux\""),
        "{:?}",
        resp.result
    );
    assert!(
        resp.result["rendered_ascii"]
            .as_str()
            .unwrap_or("")
            .contains("tableau[0]:"),
        "{:?}",
        resp.result
    );
}

#[test]
fn summarize_symmetry_for_expression_returns_valid_json_and_exact_render() {
    let response =
        summarize_symmetry_for_expr("tableau_symmetry([[2,1]], slots=[[0,1,2]])").expect("summary");
    let summary: ax_ai_proto::TensorSymmetrySummary =
        serde_json::from_str(&response.summary_json).expect("parse summary json");

    assert_eq!(
        summary,
        ax_ai_proto::TensorSymmetrySummary {
            tableaux: vec![ax_ai_proto::TensorSymmetryEntry {
                shape: vec![2, 1],
                slots: vec![0, 1, 2],
                label: None,
                trace_free: false,
                duality: "none".to_string(),
            }],
        }
    );
    assert_eq!(
        response.rendered_ascii,
        "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None"
    );
}
