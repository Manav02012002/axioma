use logos::Logos;

use crate::diag::{Diagnostic, DiagnosticCode, Severity};
use crate::kind::SyntaxKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: SyntaxKind,
    pub span: std::ops::Range<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LexErr;

#[derive(Logos, Debug, PartialEq)]
#[logos(error = LexErr)]
enum RawTok {
    #[regex(r"[ \t\r\n]+")]
    Whitespace,

    #[regex(r"//[^\n]*")]
    CommentLine,

    #[regex(r"/\*([^*]|\*[^/])*\*/")]
    CommentBlock,

    #[token("module")]
    KwModule,
    #[token("import")]
    KwImport,
    #[token("let")]
    KwLet,
    #[token("in")]
    KwIn,
    #[token("indexset")]
    KwIndexset,
    #[token("true")]
    KwTrue,
    #[token("false")]
    KwFalse,

    #[regex(r"[A-Za-z][A-Za-z0-9_]*")]
    Ident,

    #[regex(r"[0-9]+\.[0-9]+")]
    Float,

    #[regex(r"[0-9]+")]
    Int,

    #[regex(r#""([^"\\]|\\.)*""#)]
    String,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("^")]
    Caret,

    #[token("=")]
    Eq,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBrack,
    #[token("]")]
    RBrack,

    #[token("_")]
    Underscore,
}

fn map_kind(t: RawTok) -> SyntaxKind {
    match t {
        RawTok::Whitespace => SyntaxKind::Whitespace,
        RawTok::CommentLine => SyntaxKind::CommentLine,
        RawTok::CommentBlock => SyntaxKind::CommentBlock,

        RawTok::KwModule => SyntaxKind::KwModule,
        RawTok::KwImport => SyntaxKind::KwImport,
        RawTok::KwLet => SyntaxKind::KwLet,
        RawTok::KwIn => SyntaxKind::KwIn,
        RawTok::KwIndexset => SyntaxKind::KwIndexset,
        RawTok::KwTrue => SyntaxKind::KwTrue,
        RawTok::KwFalse => SyntaxKind::KwFalse,

        RawTok::Ident => SyntaxKind::Ident,
        RawTok::Int => SyntaxKind::Int,
        RawTok::Float => SyntaxKind::Float,
        RawTok::String => SyntaxKind::String,

        RawTok::Plus => SyntaxKind::Plus,
        RawTok::Minus => SyntaxKind::Minus,
        RawTok::Star => SyntaxKind::Star,
        RawTok::Slash => SyntaxKind::Slash,
        RawTok::Caret => SyntaxKind::Caret,

        RawTok::Eq => SyntaxKind::Eq,
        RawTok::Comma => SyntaxKind::Comma,
        RawTok::Dot => SyntaxKind::Dot,
        RawTok::Colon => SyntaxKind::Colon,
        RawTok::Semi => SyntaxKind::Semi,

        RawTok::LParen => SyntaxKind::LParen,
        RawTok::RParen => SyntaxKind::RParen,
        RawTok::LBrace => SyntaxKind::LBrace,
        RawTok::RBrace => SyntaxKind::RBrace,
        RawTok::LBrack => SyntaxKind::LBrack,
        RawTok::RBrack => SyntaxKind::RBrack,

        RawTok::Underscore => SyntaxKind::Underscore,
    }
}

pub fn lex(input: &str) -> (Vec<(SyntaxKind, std::ops::Range<usize>)>, Vec<Diagnostic>) {
    let mut toks = Vec::new();
    let mut diags = Vec::new();

    let mut lx = RawTok::lexer(input);
    while let Some(rt) = lx.next() {
        let span = lx.span();
        match rt {
            Ok(tok) => {
                toks.push((map_kind(tok), span));
            }
            Err(_e) => {
                toks.push((SyntaxKind::Error, span.clone()));
                diags.push(Diagnostic::new(
                    DiagnosticCode::SyntaxError,
                    Severity::Error,
                    "unrecognised token",
                    span,
                ));
            }
        }
    }

    toks.push((SyntaxKind::Eof, input.len()..input.len()));
    (toks, diags)
}
