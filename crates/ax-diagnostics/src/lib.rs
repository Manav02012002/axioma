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
}
