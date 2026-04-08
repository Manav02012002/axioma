use super::test_harness::{temp_dir, write_module, ProtocolHarness};
use super::*;
use serde_json::json;

fn kernel_plain_text(result: &ProcessResult) -> Option<String> {
    result.outbound.iter().find_map(|outbound| {
        (outbound.message.header["msg_type"] == "execute_result")
            .then(|| outbound.message.content["data"]["text/plain"].as_str().map(str::to_string))
            .flatten()
    })
}

#[test]
fn notebook_and_kernel_share_import_resolution_order() {
    let cwd = temp_dir("parity-import-order");
    write_module(&cwd, "shared.demo", "let imported_value = 77");
    let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
        working_dir: Some(cwd),
        ..ax_context::ImportSearchPathConfig::default()
    });

    let mut notebook_env = ax_eval::Env::new();
    let notebook_interner = ax_ir::Interner::new();
    let notebook = ax_notebook::handle_eval(
        r#"{"source": "import shared.demo\nimported_value"}"#,
        &mut notebook_env,
        &notebook_interner,
        &search_paths,
    );
    assert_eq!(notebook.error, None);
    assert_eq!(notebook.unicode.as_deref(), Some("77"));

    let mut harness = ProtocolHarness::with_search_paths(search_paths);
    let kernel = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "import shared.demo\nimported_value", "store_history": true }),
    );
    assert_eq!(kernel_plain_text(&kernel), Some("77".to_string()));
}

#[test]
fn notebook_and_kernel_render_basic_evaluation_with_same_plain_and_latex() {
    let source = "sin(x)^2 + cos(x)^2";

    let mut notebook_env = ax_eval::Env::new();
    let notebook_interner = ax_ir::Interner::new();
    let notebook = ax_notebook::handle_eval(
        &json!({ "source": source }).to_string(),
        &mut notebook_env,
        &notebook_interner,
        &[],
    );
    assert_eq!(notebook.error, None);

    let mut harness = ProtocolHarness::new();
    let kernel = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": source, "store_history": true }),
    );
    let execute_result = kernel
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
        .expect("kernel execute_result");
    let data = execute_result.message.content["data"]
        .as_object()
        .expect("mime bundle");
    assert_eq!(data["text/plain"], notebook.unicode.clone().expect("notebook unicode"));
    assert_eq!(data["text/latex"], notebook.latex.clone().expect("notebook latex"));
}

#[test]
fn notebook_and_kernel_both_report_import_miss_with_searched_paths_context() {
    let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
        working_dir: Some(temp_dir("parity-import-miss")),
        ..ax_context::ImportSearchPathConfig::default()
    });

    let mut notebook_env = ax_eval::Env::new();
    let notebook_interner = ax_ir::Interner::new();
    let notebook = ax_notebook::handle_eval(
        r#"{"source": "import missing.module"}"#,
        &mut notebook_env,
        &notebook_interner,
        &search_paths,
    );
    let notebook_error = notebook.error.expect("notebook error");
    assert!(notebook_error.contains("searched paths"));

    let mut harness = ProtocolHarness::with_search_paths(search_paths);
    let kernel = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "import missing.module", "store_history": true }),
    );
    let reply = kernel
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "execute_reply")
        .expect("execute reply");
    let kernel_error = reply.message.content["evalue"]
        .as_str()
        .expect("kernel error");
    assert!(kernel_error.contains("searched paths"));
}
