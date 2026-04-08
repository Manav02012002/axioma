use super::test_harness::{
    assert_execute_ok, assert_parent_headers_and_metadata, message_types, output_data,
    reply_content, ProtocolHarness,
};
use super::*;
use serde_json::json;

fn long_addition_code(terms: usize) -> String {
    std::iter::repeat("1").take(terms).collect::<Vec<_>>().join(" + ")
}

#[test]
fn signed_execute_success_preserves_protocol_ordering() {
    let mut harness = ProtocolHarness::new();
    let result = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "1 + 2", "store_history": true }),
    );

    assert_execute_ok(&result, 1, "3");
    assert_parent_headers_and_metadata(&result, "execute_request");
    assert_eq!(
        message_types(&result),
        vec!["status", "execute_input", "execute_result", "execute_reply", "status"]
    );
    let execute_result = result
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
        .expect("execute_result");
    assert_eq!(execute_result.message.content["metadata"], json!({}));
}

#[test]
fn signed_execute_failure_emits_error_reply_and_idle() {
    let mut harness = ProtocolHarness::new();
    let result = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "import does.not.exist", "store_history": true }),
    );

    assert_eq!(
        message_types(&result),
        vec!["status", "execute_input", "error", "execute_reply", "status"]
    );
    let reply = reply_content(&result, "execute_reply");
    assert_eq!(reply["status"], "error");
    assert_eq!(reply["execution_count"], 1);
    let error = result
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "error")
        .expect("error");
    assert_eq!(error.message.content["ename"], "EvalError");
    let statuses: Vec<&str> = result
        .outbound
        .iter()
        .filter(|outbound| outbound.message.header["msg_type"] == "status")
        .map(|outbound| outbound.message.content["execution_state"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(statuses, vec!["busy", "idle"]);
}

#[test]
fn malformed_signed_message_recovers_and_next_valid_request_works() {
    let mut harness = ProtocolHarness::new();
    let mut broken = harness.frames("execute_request", json!({ "code": "1 + 2" }));
    broken.pop();

    let malformed = harness.runtime.process_frames(Channel::Shell, broken, "test-secret");
    assert!(malformed.outbound.is_empty());
    assert!(
        malformed
            .logs
            .iter()
            .any(|line| line.contains("incomplete Jupyter message"))
    );

    let success = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "2 + 3", "store_history": true }),
    );
    assert_execute_ok(&success, 1, "5");
}

#[test]
fn signed_protocol_requests_cover_completion_inspect_history_and_is_complete() {
    let mut harness = ProtocolHarness::new();
    let _ = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "let field_strength = 42", "store_history": true }),
    );

    let complete = harness.send(
        Channel::Shell,
        "complete_request",
        json!({ "code": "field", "cursor_pos": 5 }),
    );
    assert_parent_headers_and_metadata(&complete, "complete_request");
    let matches = reply_content(&complete, "complete_reply")["matches"]
        .as_array()
        .expect("matches");
    assert_eq!(reply_content(&complete, "complete_reply")["status"], "ok");
    assert_eq!(reply_content(&complete, "complete_reply")["cursor_start"], 0);
    assert_eq!(reply_content(&complete, "complete_reply")["cursor_end"], 5);
    assert_eq!(reply_content(&complete, "complete_reply")["metadata"], json!({}));
    assert!(matches.iter().any(|entry| entry == "field_strength"));

    let inspect = harness.send(
        Channel::Shell,
        "inspect_request",
        json!({ "code": "field_strength", "cursor_pos": 5, "detail_level": 1 }),
    );
    assert_eq!(reply_content(&inspect, "inspect_reply")["found"], true);
    assert_eq!(reply_content(&inspect, "inspect_reply")["metadata"], json!({}));

    let history = harness.send(
        Channel::Shell,
        "history_request",
        json!({ "hist_access_type": "tail", "n": 1, "output": true }),
    );
    assert_eq!(reply_content(&history, "history_reply")["status"], "ok");
    assert_eq!(reply_content(&history, "history_reply")["history"][0][1], 1);

    let is_complete = harness.send(
        Channel::Shell,
        "is_complete_request",
        json!({ "code": "f(x," }),
    );
    assert_eq!(reply_content(&is_complete, "is_complete_reply")["status"], "incomplete");
}

#[test]
fn signed_interrupt_and_shutdown_cover_busy_kernel_lifecycle() {
    let mut harness = ProtocolHarness::new();
    let start = harness.send_async(
        Channel::Shell,
        "execute_request",
        json!({ "code": long_addition_code(200_000), "store_history": true }),
    );
    assert_eq!(message_types(&start), vec!["status", "execute_input"]);

    let interrupt = harness.send(Channel::Control, "interrupt_request", json!({}));
    assert_eq!(reply_content(&interrupt, "interrupt_reply")["status"], "ok");

    let finish = harness.wait();
    assert_eq!(reply_content(&finish, "execute_reply")["ename"], "Interrupted");
    let error = finish
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "error")
        .expect("interrupt error");
    assert_eq!(error.message.content["ename"], "Interrupted");
    assert_eq!(
        finish
            .outbound
            .iter()
            .filter(|outbound| outbound.message.header["msg_type"] == "status")
            .count(),
        1
    );

    let next = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "4 + 5", "store_history": true }),
    );
    assert_execute_ok(&next, 2, "9");

    let restart = harness.send(Channel::Control, "shutdown_request", json!({ "restart": false }));
    let reply = reply_content(&restart, "shutdown_reply");
    assert_eq!(reply["status"], "ok");
    assert_eq!(reply["restart"], false);
    assert!(restart.shutdown);
}

#[test]
fn signed_rich_mime_output_keeps_bundle_shape() {
    let parent = DecodedMessage {
        identities: Vec::new(),
        header: json!({
            "msg_id": "test",
            "username": "tester",
            "session": "session-1",
            "date": "0.0",
            "msg_type": "execute_request",
            "version": "5.4"
        }),
        content_bytes: Vec::new(),
    };
    let message = make_output_message(
        &parent,
        &KernelOutput::DisplayData(
            MimeBundle::html("<b>bold</b>")
                .with_markdown("**bold**")
                .with_plain("bold")
                .with_json(json!({ "kind": "html" })),
        ),
        1,
    );
    let data = output_data(&message);
    assert_eq!(message.header["msg_type"], "display_data");
    assert_eq!(data["text/plain"], "bold");
    assert_eq!(data["text/html"], "<b>bold</b>");
    assert_eq!(data["text/markdown"], "**bold**");
    assert_eq!(data["application/json"], json!({ "kind": "html" }));
}

#[test]
fn silent_failure_stays_reply_only_and_does_not_publish_visible_error() {
    let mut harness = ProtocolHarness::new();
    let result = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "import does.not.exist", "silent": true }),
    );

    assert_eq!(message_types(&result), vec!["status", "execute_reply", "status"]);
    let reply = reply_content(&result, "execute_reply");
    assert_eq!(reply["status"], "error");
    assert_eq!(reply["ename"], "EvalError");
}

#[test]
fn store_history_false_keeps_execution_count_stable_but_returns_result() {
    let mut harness = ProtocolHarness::new();
    let counted = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "1 + 2", "store_history": true }),
    );
    assert_execute_ok(&counted, 1, "3");

    let transient = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "2 + 3", "store_history": false }),
    );
    assert_eq!(
        message_types(&transient),
        vec!["status", "execute_input", "execute_result", "execute_reply", "status"]
    );
    let input = transient
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "execute_input")
        .expect("execute_input");
    assert_eq!(input.message.content["execution_count"], 1);
    let reply = reply_content(&transient, "execute_reply");
    assert_eq!(reply["execution_count"], 1);

    let history = harness.send(
        Channel::Shell,
        "history_request",
        json!({ "hist_access_type": "tail", "n": 10, "output": false }),
    );
    let history = reply_content(&history, "history_reply")["history"]
        .as_array()
        .expect("history");
    assert_eq!(history.len(), 1);
}

#[test]
fn empty_execute_produces_no_result_but_still_replies_cleanly() {
    let mut harness = ProtocolHarness::new();
    let result = harness.send(
        Channel::Shell,
        "execute_request",
        json!({ "code": "", "store_history": true }),
    );

    assert_eq!(
        message_types(&result),
        vec!["status", "execute_input", "execute_reply", "status"]
    );
    assert!(
        result
            .outbound
            .iter()
            .all(|outbound| outbound.message.header["msg_type"] != "execute_result")
    );
    let reply = reply_content(&result, "execute_reply");
    assert_eq!(reply["status"], "ok");
    assert_eq!(reply["execution_count"], 1);
}

#[test]
fn repeated_interrupts_and_post_idle_interrupt_reply_ok() {
    let mut harness = ProtocolHarness::new();
    let start = harness.send_async(
        Channel::Shell,
        "execute_request",
        json!({ "code": long_addition_code(200_000), "store_history": true }),
    );
    assert_eq!(message_types(&start), vec!["status", "execute_input"]);

    let first = harness.send(Channel::Control, "interrupt_request", json!({}));
    let second = harness.send(Channel::Control, "interrupt_request", json!({}));
    assert_eq!(reply_content(&first, "interrupt_reply")["status"], "ok");
    assert_eq!(reply_content(&second, "interrupt_reply")["status"], "ok");

    let _ = harness.wait();
    let idle = harness.send(Channel::Control, "interrupt_request", json!({}));
    assert_eq!(reply_content(&idle, "interrupt_reply")["status"], "ok");
}

#[test]
fn shutdown_after_idle_replies_without_extra_iopub_messages() {
    let mut harness = ProtocolHarness::new();
    let shutdown = harness.send(Channel::Control, "shutdown_request", json!({ "restart": false }));

    assert_eq!(message_types(&shutdown), vec!["shutdown_reply"]);
    let reply = reply_content(&shutdown, "shutdown_reply");
    assert_eq!(reply["status"], "ok");
    assert_eq!(reply["restart"], false);
    assert!(shutdown.shutdown);
}
