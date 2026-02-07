//! Axioma WASM plugin contract (JSON in/out).
#![forbid(unsafe_code)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginRequest {
    pub plugin: String,
    pub op: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginResponse {
    pub ok: bool,
    pub result: serde_json::Value,
    pub diagnostics: Vec<PluginDiag>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginDiag {
    pub level: String,
    pub message: String,
}
