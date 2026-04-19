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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CptEquationEntry {
    pub label: String,
    pub unicode: String,
    pub latex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CptDerivationPacket {
    pub background: String,
    pub gauge: String,
    pub matter: String,
    pub equations: Vec<CptEquationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityReportPacket {
    pub suite_name: String,
    pub entries: Vec<ax_perturb::ParityBenchmarkEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HierarchyExportPacket {
    pub species: String,
    pub gauge: String,
    pub closure: String,
    pub payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantumDisplayPacket {
    pub object_kind: String,
    pub unicode: String,
    pub latex: String,
    pub dimension: Option<usize>,
    pub subsystem_dims: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantumMeasurementPacket {
    pub probabilities: Vec<String>,
    pub post_states: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantumChannelPacket {
    pub kraus_count: usize,
    pub dimension: usize,
    pub trace_preserving: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantumDensitySummaryPacket {
    pub dimension: usize,
    pub trace: String,
    pub purity: String,
    pub linear_entropy: String,
    pub is_qubit: bool,
    pub bloch_vector: Option<[String; 3]>,
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

    #[test]
    fn cpt_equation_entry_round_trips() {
        let payload = CptEquationEntry {
            label: "constraint".to_string(),
            unicode: "constraint: x".to_string(),
            latex: "\\text{constraint} &: x".to_string(),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: CptEquationEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn cpt_derivation_packet_round_trips() {
        let payload = CptDerivationPacket {
            background: "FRWBackground(time=conformal, curvature=flat, spatial_dim=3)".to_string(),
            gauge: "Gauge(newtonian)".to_string(),
            matter: "Matter(symbolic)".to_string(),
            equations: vec![CptEquationEntry {
                label: "eq0".to_string(),
                unicode: "eq0: x".to_string(),
                latex: "\\text{eq0} &: x".to_string(),
            }],
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: CptDerivationPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn parity_report_packet_round_trips() {
        let payload = ParityReportPacket {
            suite_name: "suite".to_string(),
            entries: vec![ax_perturb::ParityBenchmarkEntry {
                label: "eq0".to_string(),
                expected: "x".to_string(),
                actual: "x".to_string(),
                matched: true,
            }],
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: ParityReportPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn hierarchy_export_packet_round_trips() {
        let payload = HierarchyExportPacket {
            species: "neutrino".to_string(),
            gauge: "newtonian".to_string(),
            closure: "power_law".to_string(),
            payload: "{\"target\":\"class\"}".to_string(),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: HierarchyExportPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn quantum_display_packet_round_trips() {
        let payload = QuantumDisplayPacket {
            object_kind: "density_matrix".to_string(),
            unicode: "|psi⟩⟨psi|".to_string(),
            latex: "\\left|\\psi\\right\\rangle\\!\\left\\langle \\psi\\right|".to_string(),
            dimension: Some(4),
            subsystem_dims: vec![2, 2],
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: QuantumDisplayPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn quantum_measurement_packet_round_trips() {
        let payload = QuantumMeasurementPacket {
            probabilities: vec!["1/2".to_string(), "1/2".to_string()],
            post_states: vec!["|0⟩".to_string(), "|1⟩".to_string()],
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: QuantumMeasurementPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn quantum_channel_packet_round_trips() {
        let payload = QuantumChannelPacket {
            kraus_count: 2,
            dimension: 2,
            trace_preserving: true,
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: QuantumChannelPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn quantum_density_summary_packet_round_trips() {
        let payload = QuantumDensitySummaryPacket {
            dimension: 2,
            trace: "1".to_string(),
            purity: "1".to_string(),
            linear_entropy: "0".to_string(),
            is_qubit: true,
            bloch_vector: Some(["0".to_string(), "0".to_string(), "1".to_string()]),
        };

        let json = serde_json::to_string(&payload).unwrap();
        let reparsed: QuantumDensitySummaryPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, payload);
    }
}
