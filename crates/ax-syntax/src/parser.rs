use rowan::GreenNodeBuilder;

use crate::diag::{Diagnostic, DiagnosticCode, Severity};
use crate::kind::{is_trivia, SyntaxKind};
use crate::lexer::lex;
use crate::tree::syntax_node_from_green;

#[derive(Debug, Clone)]
pub struct Parse {
    pub green: rowan::GreenNode,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub diag: Diagnostic,
}

pub struct Parser<'a> {
    input: &'a str,
    toks: Vec<(SyntaxKind, std::ops::Range<usize>)>,
    pos: usize,
    diagnostics: Vec<Diagnostic>,
    builder: GreenNodeBuilder<'static>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        let (toks, mut diags) = lex(input);
        let builder = GreenNodeBuilder::new();
        Self {
            input,
            toks,
            pos: 0,
            diagnostics: std::mem::take(&mut diags),
            builder,
        }
    }

    pub fn parse(mut self) -> Parse {
        self.builder.start_node(SyntaxKind::Root.into());
        self.bump_trivia();
        while self.current() != SyntaxKind::Eof {
            if !self.parse_item() {
                self.unexpected_here("expected item");
                self.recover_to_item_boundary();
            }
            self.bump_trivia();
        }
        self.builder.finish_node();

        Parse {
            green: self.builder.finish(),
            diagnostics: self.diagnostics,
        }
    }

    fn parse_item(&mut self) -> bool {
        match self.current() {
            SyntaxKind::KwModule => {
                self.builder.start_node(SyntaxKind::KwModule.into());
                self.bump();
                self.bump_trivia();
                self.expect(SyntaxKind::Ident, "expected module name");
                self.bump_trivia();
                self.expect_semi("module declaration");
                self.builder.finish_node();
                true
            }
            SyntaxKind::KwImport => {
                self.builder.start_node(SyntaxKind::KwImport.into());
                self.bump();
                self.bump_trivia();
                self.parse_path();
                self.bump_trivia();
                self.expect_semi("import");
                self.builder.finish_node();
                true
            }
            _ => {
                self.builder.start_node(SyntaxKind::Error.into());
                self.parse_expr_bp(0);
                self.bump_trivia();
                self.expect_semi("expression");
                self.builder.finish_node();
                true
            }
        }
    }

    fn parse_path(&mut self) {
        self.expect(SyntaxKind::Ident, "expected identifier in import path");
        self.bump_trivia();
        while self.current() == SyntaxKind::Dot {
            self.bump();
            self.bump_trivia();
            self.expect(SyntaxKind::Ident, "expected identifier after '.'");
            self.bump_trivia();
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) {
        self.bump_trivia();
        self.parse_prefix();
        loop {
            self.bump_trivia();
            let (l_bp, r_bp, op) = match self.current() {
                SyntaxKind::Plus => (1, 2, SyntaxKind::Plus),
                SyntaxKind::Minus => (1, 2, SyntaxKind::Minus),
                SyntaxKind::Star => (3, 4, SyntaxKind::Star),
                SyntaxKind::Slash => (3, 4, SyntaxKind::Slash),
                SyntaxKind::Caret => (7, 6, SyntaxKind::Caret),
                _ => break,
            };
            if l_bp < min_bp {
                break;
            }
            self.builder.start_node(op.into());
            self.bump();
            self.parse_expr_bp(r_bp);
            self.builder.finish_node();
        }
    }

    fn parse_prefix(&mut self) {
        self.bump_trivia();
        match self.current() {
            SyntaxKind::Minus => {
                self.builder.start_node(SyntaxKind::Minus.into());
                self.bump();
                self.parse_expr_bp(5);
                self.builder.finish_node();
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) {
        self.bump_trivia();
        match self.current() {
            SyntaxKind::Int | SyntaxKind::Float => {
                self.bump();
            }
            SyntaxKind::Ident => {
                self.bump();
                self.bump_trivia();

                if self.current() == SyntaxKind::LParen {
                    self.parse_call_args();
                } else if self.current() == SyntaxKind::LBrack {
                    self.parse_ascii_indices();
                }
            }
            SyntaxKind::LParen => {
                self.bump();
                self.parse_expr_bp(0);
                self.bump_trivia();
                self.expect(SyntaxKind::RParen, "expected ')'");
            }
            _ => {
                self.unexpected_here("expected expression");
                self.bump();
            }
        }
    }

    fn parse_call_args(&mut self) {
        self.expect(SyntaxKind::LParen, "expected '('");
        self.bump_trivia();
        if self.current() != SyntaxKind::RParen {
            loop {
                self.parse_expr_bp(0);
                self.bump_trivia();
                if self.current() == SyntaxKind::Comma {
                    self.bump();
                    self.bump_trivia();
                    continue;
                }
                break;
            }
        }
        self.bump_trivia();
        self.expect(SyntaxKind::RParen, "expected ')'");
    }

    fn parse_ascii_indices(&mut self) {
        self.expect(SyntaxKind::LBrack, "expected '['");
        self.bump_trivia();
        if self.current() != SyntaxKind::RBrack {
            loop {
                self.expect(SyntaxKind::Ident, "expected index name");
                self.bump_trivia();
                match self.current() {
                    SyntaxKind::Plus | SyntaxKind::Minus => {
                        self.bump();
                    }
                    _ => {
                        self.unexpected_here("expected '+' or '-' for index variance");
                    }
                }
                self.bump_trivia();
                if self.current() == SyntaxKind::Comma {
                    self.bump();
                    self.bump_trivia();
                    continue;
                }
                break;
            }
        }
        self.bump_trivia();
        self.expect(SyntaxKind::RBrack, "expected ']'");
    }

    fn current(&self) -> SyntaxKind {
        self.toks
            .get(self.pos)
            .map(|t| t.0)
            .unwrap_or(SyntaxKind::Eof)
    }

    fn current_span(&self) -> std::ops::Range<usize> {
        self.toks
            .get(self.pos)
            .map(|t| t.1.clone())
            .unwrap_or(self.input.len()..self.input.len())
    }

    fn prev_non_trivia_end_pos(&self) -> usize {
        if self.pos == 0 {
            return 0;
        }
        let mut i = self.pos;
        while i > 0 {
            i -= 1;
            let k = self.toks[i].0;
            if !crate::kind::is_trivia(k) {
                return self.toks[i].1.end;
            }
        }
        0
    }

    fn bump(&mut self) {
        let (k, span) = self.toks[self.pos].clone();
        let text = &self.input[span.clone()];
        self.builder.token(k.into(), text);
        self.pos += 1;
    }

    fn bump_trivia(&mut self) {
        while is_trivia(self.current()) {
            self.bump();
        }
    }

    fn expect(&mut self, k: SyntaxKind, msg: &str) {
        if self.current() == k {
            self.bump();
        } else {
            self.unexpected_here(msg);
        }
    }

    fn expect_semi(&mut self, what: &str) {
        if self.current() == SyntaxKind::Semi {
            self.bump();
            return;
        }
        let span = self.current_span();
        let insert_pos = self.prev_non_trivia_end_pos();
        let insert_at = insert_pos..insert_pos;
        self.diagnostics.push(
            Diagnostic::new(
                DiagnosticCode::MissingSemicolon,
                Severity::Error,
                format!("missing ';' after {what}"),
                span.clone(),
            )
            .with_label("expected ';'")
            .help("add a semicolon to terminate this statement")
            .note("continuing parse at the next statement boundary")
            .fix(insert_at, ";", "insert ';'"),
        );
        self.recover_to_item_boundary();
    }

    fn unexpected_here(&mut self, msg: &str) {
        let span = self.current_span();
        self.diagnostics.push(Diagnostic::new(
            DiagnosticCode::UnexpectedToken,
            Severity::Error,
            msg,
            span,
        ));
    }

    fn recover_to_item_boundary(&mut self) {
        while self.current() != SyntaxKind::Eof && self.current() != SyntaxKind::Semi {
            self.bump();
        }
        if self.current() == SyntaxKind::Semi {
            self.bump();
        }
    }
}

pub fn parse_file(input: &str) -> (crate::tree::SyntaxNode, Vec<Diagnostic>) {
    let p = Parser::new(input).parse();
    (syntax_node_from_green(p.green), p.diagnostics)
}
