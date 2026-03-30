use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub message: String,
    pub span: Range<usize>,
    pub label: Option<String>,
    pub help: Vec<String>,
    pub notes: Vec<String>,
    pub fixits: Vec<FixIt>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiagnosticCode {
    SyntaxError,
    UnexpectedToken,
    MissingSemicolon,
    UnterminatedComment,
    InvalidNumber,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixIt {
    pub span: Range<usize>,
    pub replacement: String,
    pub message: String,
}

impl Diagnostic {
    pub fn new(
        code: DiagnosticCode,
        severity: Severity,
        message: impl Into<String>,
        span: Range<usize>,
    ) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            span,
            label: None,
            help: Vec::new(),
            notes: Vec::new(),
            fixits: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn help(mut self, msg: impl Into<String>) -> Self {
        self.help.push(msg.into());
        self
    }

    pub fn note(mut self, msg: impl Into<String>) -> Self {
        self.notes.push(msg.into());
        self
    }

    pub fn fix(
        mut self,
        span: Range<usize>,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        self.fixits.push(FixIt {
            span,
            replacement: replacement.into(),
            message: message.into(),
        });
        self
    }
}
