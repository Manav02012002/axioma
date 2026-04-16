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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SymmetrySummaryRequest {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SymmetrySummaryResponse {
    pub summary_json: String,
    pub rendered_ascii: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetry_request_round_trips() {
        let request = SymmetrySummaryRequest {
            expr: "tableau_symmetry([[2]], slots=[[0,1]])".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SymmetrySummaryRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn symmetry_response_round_trips() {
        let response = SymmetrySummaryResponse {
            summary_json: "{\"tableaux\":[]}".to_string(),
            rendered_ascii: "[][]".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SymmetrySummaryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }
}
