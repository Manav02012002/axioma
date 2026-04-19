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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenRequest {
    pub matrix: Vec<Vec<(f64, f64)>>,
    pub k: usize,
    pub which: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenpair {
    pub eigenvalue: (f64, f64),
    pub eigenvector: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenResponse {
    pub eigenpairs: Vec<SparseEigenpair>,
    pub converged: bool,
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

    #[test]
    fn sparse_eigen_request_round_trips() {
        let request = SparseEigenRequest {
            matrix: vec![vec![(1.0, 0.0), (0.0, 0.0)], vec![(0.0, 0.0), (2.0, 0.0)]],
            k: 1,
            which: "LM".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SparseEigenRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn sparse_eigen_response_round_trips() {
        let response = SparseEigenResponse {
            eigenpairs: vec![SparseEigenpair {
                eigenvalue: (2.0, 0.0),
                eigenvector: vec![(1.0, 0.0), (0.0, 0.0)],
            }],
            converged: true,
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SparseEigenResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }
}
