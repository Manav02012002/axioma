pub mod diag;
pub mod kind;
pub mod lexer;
pub mod parser;
pub mod tree;

#[cfg(test)]
mod tests;

pub use diag::{Diagnostic, DiagnosticCode, FixIt, Severity};
pub use kind::{SyntaxKind, T};
pub use lexer::{LexErr, Token};
pub use parser::{Parse, ParseError, Parser};
pub use tree::{AxLanguage, SyntaxNode, SyntaxToken};
