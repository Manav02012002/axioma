#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

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
