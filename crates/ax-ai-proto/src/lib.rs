#![forbid(unsafe_code)]

pub mod symmetry;

use serde::{Deserialize, Serialize};

pub use symmetry::{SymmetryExplainResponse, TensorSymmetryEntry, TensorSymmetrySummary};

pub const AI_PACKET_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixIt {
    pub span: Span,
    pub replacement: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub code: String,
    pub severity: String,
    pub message: String,
    pub span: Span,
    pub label: Option<String>,
    pub help: Vec<String>,
    pub notes: Vec<String>,
    pub fixits: Vec<FixIt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageCard {
    pub language: String,
    pub statement_terminator: String,
    pub notes: Vec<String>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFile {
    pub path: String,
    pub text: String,
    pub hash_blake3_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPacket {
    pub version: String,
    pub tool: String,
    pub tool_version: String,
    pub file: SourceFile,
    pub diagnostics: Vec<Diagnostic>,
    pub language_card: LanguageCard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Edit {
    Replace { span: Span, replacement: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiEditRequest {
    pub version: String,
    pub file_hash_blake3_hex: String,
    pub edits: Vec<Edit>,
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiEditResult {
    pub version: String,
    pub applied: usize,
    pub rejected: usize,
    pub output_hash_blake3_hex: String,
    pub output_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CharacterEvaluationResponse {
    pub shape: Vec<usize>,
    pub cycle_type: Vec<usize>,
    pub character: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_evaluation_response_round_trips() {
        let payload = CharacterEvaluationResponse {
            shape: vec![2, 1],
            cycle_type: vec![2, 1],
            character: "0".to_string(),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: CharacterEvaluationResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }
}
