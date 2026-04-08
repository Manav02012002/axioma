use super::*;
use serde_json::json;
use std::path::{Path, PathBuf};

pub(super) struct ProtocolHarness {
    pub(super) runtime: KernelRuntime,
    key: String,
}

impl ProtocolHarness {
    pub(super) fn new() -> Self {
        Self::with_search_paths(Vec::new())
    }

    pub(super) fn with_search_paths(search_paths: Vec<PathBuf>) -> Self {
        Self {
            runtime: KernelRuntime::new(search_paths),
            key: "test-secret".to_string(),
        }
    }

    pub(super) fn send(&mut self, channel: Channel, msg_type: &str, content: Value) -> ProcessResult {
        self.runtime
            .process_frames(channel, self.frames(msg_type, content), &self.key)
    }

    pub(super) fn send_async(
        &mut self,
        channel: Channel,
        msg_type: &str,
        content: Value,
    ) -> ProcessResult {
        self.runtime
            .process_frames_async(channel, self.frames(msg_type, content), &self.key)
    }

    pub(super) fn wait(&mut self) -> ProcessResult {
        self.runtime.wait_for_execution_result()
    }

    pub(super) fn frames(&self, msg_type: &str, content: Value) -> Vec<Vec<u8>> {
        let header = json!({
            "msg_id": format!("test-{msg_type}"),
            "username": "tester",
            "session": "session-1",
            "date": "0.0",
            "msg_type": msg_type,
            "version": "5.4"
        });
        let parent_header = json!({});
        let metadata = json!({});
        let header_bytes = serde_json::to_vec(&header).expect("header bytes");
        let parent_header_bytes = serde_json::to_vec(&parent_header).expect("parent bytes");
        let metadata_bytes = serde_json::to_vec(&metadata).expect("metadata bytes");
        let content_bytes = serde_json::to_vec(&content).expect("content bytes");
        let signature = compute_hmac(
            &self.key,
            &[
                &header_bytes,
                &parent_header_bytes,
                &metadata_bytes,
                &content_bytes,
            ],
        )
        .expect("signature");

        vec![
            b"client-1".to_vec(),
            b"<IDS|MSG>".to_vec(),
            signature.into_bytes(),
            header_bytes,
            parent_header_bytes,
            metadata_bytes,
            content_bytes,
        ]
    }
}

pub(super) fn message_types(result: &ProcessResult) -> Vec<String> {
    result
        .outbound
        .iter()
        .map(|outbound| {
            outbound.message.header["msg_type"]
                .as_str()
                .unwrap_or("missing")
                .to_string()
        })
        .collect()
}

pub(super) fn reply_content<'a>(result: &'a ProcessResult, msg_type: &str) -> &'a Value {
    &result
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == msg_type)
        .unwrap_or_else(|| panic!("missing {msg_type} in {:?}", message_types(result)))
        .message
        .content
}

pub(super) fn output_data(msg: &JupyterMessage) -> &serde_json::Map<String, Value> {
    msg.content["data"].as_object().expect("mime bundle")
}

pub(super) fn assert_parent_headers_and_metadata(result: &ProcessResult, parent_msg_type: &str) {
    let expected_id = format!("test-{parent_msg_type}");
    for outbound in &result.outbound {
        assert_eq!(outbound.message.parent_header["msg_id"], expected_id);
        assert_eq!(outbound.message.parent_header["msg_type"], parent_msg_type);
        assert!(
            outbound.message.metadata.is_object(),
            "message metadata must be a JSON object: {:?}",
            outbound.message.metadata
        );
    }
}

pub(super) fn assert_execute_ok(result: &ProcessResult, execution_count: u64, expected_text: &str) {
    assert!(
        result.logs.is_empty(),
        "expected no logs, got {:?}",
        result.logs
    );
    let reply = result
        .outbound
        .iter()
        .find(|outbound| matches!(outbound.target, OutboundTarget::Reply(_)))
        .expect("execute reply");
    assert_eq!(reply.message.content["status"], "ok");
    assert_eq!(reply.message.content["execution_count"], execution_count);
    assert!(
        !result
            .outbound
            .iter()
            .any(|outbound| outbound.message.header["msg_type"] == "display_data"),
        "plain execution must not emit display_data: {:?}",
        message_types(result)
    );

    let execute_result = result
        .outbound
        .iter()
        .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
        .expect("execute_result");
    assert_eq!(
        execute_result.message.content["data"]["text/plain"],
        expected_text
    );
}

pub(super) fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "axioma-ax-jupyter-harness-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

pub(super) fn write_module(root: &Path, module: &str, source: &str) -> PathBuf {
    let mut module_path = root.to_path_buf();
    for part in module.split('.') {
        module_path.push(part);
    }
    module_path.set_extension("ax");
    if let Some(parent) = module_path.parent() {
        std::fs::create_dir_all(parent).expect("module parent");
    }
    std::fs::write(&module_path, source).expect("write module");
    module_path
}
