use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub level: Level,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
}

impl Diagnostic {
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self {
            level: Level::Error,
            code: code.to_string(),
            message: message.into(),
            action_id: None,
        }
    }

    pub fn with_action_id(mut self, id: impl Into<String>) -> Self {
        self.action_id = Some(id.into());
        self
    }
}

pub fn invalid_tensor_symmetry(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error("invalid_tensor_symmetry", msg)
}

pub fn tableau_projection_annihilated(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error("tableau_projection_annihilated", msg)
}

/// High-value QM/QFT-facing evaluator diagnostic families with normalized wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumDiagnosticKind {
    /// Domain/codomain or Hilbert-space compatibility failure.
    SpaceMismatch,
    /// Unsupported quantum dimension or dimension-dependent convention failure.
    UnsupportedDimension,
    /// Incompatible or missing quantum/QFT convention metadata.
    ConventionMismatch,
    /// Invalid subsystem label or undeclared Hilbert-space reference.
    InvalidSubsystem,
    /// Invalid channel structure or channel dimension mismatch.
    InvalidChannel,
    /// Invalid or incompatible spinor metadata.
    InvalidSpinorMetadata,
}

/// Format a normalized QM/QFT diagnostic message from a kind and detail string.
pub fn quantum_diagnostic_message(kind: QuantumDiagnosticKind, detail: &str) -> String {
    match kind {
        QuantumDiagnosticKind::SpaceMismatch => {
            format!("quantum space mismatch: {detail}")
        }
        QuantumDiagnosticKind::UnsupportedDimension => {
            format!("unsupported quantum dimension: {detail}")
        }
        QuantumDiagnosticKind::ConventionMismatch => {
            format!("quantum convention mismatch: {detail}")
        }
        QuantumDiagnosticKind::InvalidSubsystem => {
            format!("invalid quantum subsystem: {detail}")
        }
        QuantumDiagnosticKind::InvalidChannel => {
            format!("invalid quantum channel: {detail}")
        }
        QuantumDiagnosticKind::InvalidSpinorMetadata => {
            format!("invalid spinor metadata: {detail}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_tensor_symmetry_diagnostic_contains_message() {
        let diagnostic = invalid_tensor_symmetry("bad symmetry");
        assert!(diagnostic.message.contains("bad symmetry"));
    }

    #[test]
    fn tableau_projection_annihilated_contains_message() {
        let diagnostic = tableau_projection_annihilated("projector killed term");
        assert!(diagnostic.message.contains("projector killed term"));
    }

    #[test]
    fn quantum_diagnostic_message_formats_exact_templates() {
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::SpaceMismatch, "A"),
            "quantum space mismatch: A"
        );
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::UnsupportedDimension, "B"),
            "unsupported quantum dimension: B"
        );
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::ConventionMismatch, "C"),
            "quantum convention mismatch: C"
        );
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::InvalidSubsystem, "D"),
            "invalid quantum subsystem: D"
        );
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::InvalidChannel, "E"),
            "invalid quantum channel: E"
        );
        assert_eq!(
            quantum_diagnostic_message(QuantumDiagnosticKind::InvalidSpinorMetadata, "F"),
            "invalid spinor metadata: F"
        );
    }
}
