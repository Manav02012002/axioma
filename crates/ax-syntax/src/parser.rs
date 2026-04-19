use rowan::GreenNodeBuilder;

use crate::diag::{Diagnostic, DiagnosticCode, Severity};
use crate::kind::{is_trivia, SyntaxKind};
use crate::lexer::lex;
use crate::tree::{syntax_node_from_green, tableau_symmetry_exprs};

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
                self.builder.start_node(SyntaxKind::ExprStmt.into());
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
        let lhs_checkpoint = self.builder.checkpoint();
        self.parse_prefix();
        loop {
            self.bump_trivia();
            match self.current() {
                SyntaxKind::Dagger => {
                    let postfix_bp = 8;
                    if postfix_bp < min_bp {
                        break;
                    }
                    self.builder
                        .start_node_at(lhs_checkpoint, SyntaxKind::DaggerExpr.into());
                    self.bump();
                    self.builder.finish_node();
                    continue;
                }
                SyntaxKind::TensorProduct => {
                    let (l_bp, r_bp) = (3, 4);
                    if l_bp < min_bp {
                        break;
                    }
                    self.builder
                        .start_node_at(lhs_checkpoint, SyntaxKind::TensorProductExpr.into());
                    self.bump();
                    self.parse_expr_bp(r_bp);
                    self.builder.finish_node();
                    continue;
                }
                _ => {}
            }

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
            SyntaxKind::Int | SyntaxKind::Float => self.bump(),
            SyntaxKind::String => {
                self.builder.start_node(SyntaxKind::StringLiteral.into());
                self.bump();
                self.builder.finish_node();
            }
            SyntaxKind::KwTrue | SyntaxKind::KwFalse => {
                self.builder.start_node(SyntaxKind::BoolLiteral.into());
                self.bump();
                self.builder.finish_node();
            }
            SyntaxKind::Ident => {
                let ident_text = self.current_text().to_string();
                if self.nth_non_trivia_kind(1) == Some(SyntaxKind::LParen)
                    && ident_text == "tableau_symmetry"
                {
                    self.parse_tableau_symmetry_expr();
                    return;
                }

                self.bump();
                self.bump_trivia();

                if self.current() == SyntaxKind::LParen {
                    self.parse_call_expr();
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
            SyntaxKind::LBrack => self.parse_list_expr(),
            SyntaxKind::Pipe => self.parse_ket_expr(),
            SyntaxKind::Less => self.parse_bra_or_braket_expr(),
            _ => {
                self.unexpected_here("expected expression");
                self.bump();
            }
        }
    }

    fn parse_ket_expr(&mut self) {
        self.builder.start_node(SyntaxKind::KetExpr.into());
        self.bump();
        self.bump_trivia();
        self.parse_expr_bp(0);
        self.bump_trivia();
        self.expect(SyntaxKind::Greater, "expected '>' to close ket");
        self.builder.finish_node();
    }

    fn parse_bra_or_braket_expr(&mut self) {
        let checkpoint = self.builder.checkpoint();
        self.bump();
        self.bump_trivia();
        self.parse_expr_bp(0);
        self.bump_trivia();

        if self.current() != SyntaxKind::Pipe {
            self.unexpected_here("expected '|' in bra or braket");
            return;
        }

        self.bump();
        self.bump_trivia();

        if self.at_expr_start() {
            self.builder
                .start_node_at(checkpoint, SyntaxKind::BraKetExpr.into());
            self.parse_expr_bp(0);
            self.bump_trivia();
            self.expect(SyntaxKind::Greater, "expected '>' to close braket");
            self.builder.finish_node();
            return;
        }

        self.builder
            .start_node_at(checkpoint, SyntaxKind::BraExpr.into());
        self.builder.finish_node();
    }

    fn parse_call_expr(&mut self) {
        self.builder.start_node(SyntaxKind::CallExpr.into());
        self.parse_call_args();
        self.builder.finish_node();
    }

    fn parse_call_args(&mut self) {
        self.expect(SyntaxKind::LParen, "expected '('");
        self.bump_trivia();
        if self.current() != SyntaxKind::RParen {
            loop {
                if self.current() == SyntaxKind::Ident
                    && self.nth_non_trivia_kind(1) == Some(SyntaxKind::Eq)
                {
                    self.builder.start_node(SyntaxKind::NamedArg.into());
                    self.bump();
                    self.bump_trivia();
                    self.expect(SyntaxKind::Eq, "expected '='");
                    self.bump_trivia();
                    self.parse_expr_bp(0);
                    self.builder.finish_node();
                } else {
                    self.parse_expr_bp(0);
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
        self.expect(SyntaxKind::RParen, "expected ')'");
    }

    fn parse_tableau_symmetry_expr(&mut self) {
        let start_span = self.current_span();
        self.builder
            .start_node(SyntaxKind::TableauSymmetryExpr.into());
        self.bump();
        self.bump_trivia();
        self.expect(SyntaxKind::LParen, "expected '('");
        self.bump_trivia();

        let mut shapes = self.parse_tableau_shape_list();
        let mut slots = Vec::new();
        let mut labels: Option<Vec<String>> = None;
        let mut trace_free: Option<Vec<bool>> = None;
        let mut saw_slots = false;

        self.bump_trivia();
        while self.current() == SyntaxKind::Comma {
            self.bump();
            self.bump_trivia();

            if self.current() == SyntaxKind::Ident
                && self.nth_non_trivia_kind(1) == Some(SyntaxKind::Eq)
            {
                let name = self.current_text().to_string();
                self.bump();
                self.bump_trivia();
                self.expect(SyntaxKind::Eq, "expected '='");
                self.bump_trivia();

                match name.as_str() {
                    "slots" => {
                        slots = self.parse_tableau_slot_map_list();
                        saw_slots = true;
                    }
                    "labels" => {
                        labels = Some(self.parse_tableau_labels());
                    }
                    "trace_free" => {
                        trace_free = Some(self.parse_tableau_trace_free_list());
                    }
                    _ => self.parse_expr_bp(0),
                }
            } else if !saw_slots {
                slots = self.parse_tableau_slot_map_list();
                saw_slots = true;
            } else {
                self.parse_expr_bp(0);
            }
            self.bump_trivia();
        }

        self.bump_trivia();
        self.expect(SyntaxKind::RParen, "expected ')'");
        self.builder.finish_node();

        if shapes.is_empty() {
            self.push_tableau_diag(
                "tableau_symmetry requires at least one tableau shape",
                start_span.clone(),
            );
            shapes = Vec::new();
        }
        if saw_slots && shapes.len() != slots.len() {
            self.push_tableau_diag(
                "tableau_symmetry shapes and slots lists must have the same length",
                start_span.clone(),
            );
        }
        if let Some(entries) = &labels {
            if entries.len() != shapes.len() {
                self.push_tableau_diag(
                    "tableau_symmetry labels list length must match shapes list length",
                    start_span.clone(),
                );
            }
        }
        if let Some(entries) = &trace_free {
            if entries.len() != shapes.len() {
                self.push_tableau_diag(
                    "tableau_symmetry trace_free list length must match shapes list length",
                    start_span,
                );
            }
        }
    }

    fn parse_tableau_shape_list(&mut self) -> Vec<Vec<usize>> {
        self.builder.start_node(SyntaxKind::TableauShapeList.into());
        let result = self.parse_nested_usize_list(true);
        self.builder.finish_node();
        result
    }

    fn parse_tableau_slot_map_list(&mut self) -> Vec<Vec<usize>> {
        self.builder
            .start_node(SyntaxKind::TableauSlotMapList.into());
        let result = self.parse_nested_usize_list(true);
        self.builder.finish_node();
        result
    }

    fn parse_tableau_labels(&mut self) -> Vec<String> {
        self.builder.start_node(SyntaxKind::TableauLabels.into());
        let result = self.parse_string_list();
        self.builder.finish_node();
        result
    }

    fn parse_tableau_trace_free_list(&mut self) -> Vec<bool> {
        self.builder
            .start_node(SyntaxKind::TableauTraceFreeList.into());
        let result = self.parse_bool_list();
        self.builder.finish_node();
        result
    }

    fn parse_nested_usize_list(&mut self, allow_flat_single: bool) -> Vec<Vec<usize>> {
        if self.current() != SyntaxKind::LBrack {
            self.unexpected_here("expected '['");
            return Vec::new();
        }

        self.bump();
        self.bump_trivia();
        if self.current() == SyntaxKind::RBrack {
            self.bump();
            return Vec::new();
        }

        let mut values = Vec::new();
        if allow_flat_single && self.current() != SyntaxKind::LBrack {
            let flat = self.parse_usize_list_body();
            self.bump_trivia();
            self.expect(SyntaxKind::RBrack, "expected ']'");
            return vec![flat];
        }

        loop {
            values.push(self.parse_usize_list());
            self.bump_trivia();
            if self.current() == SyntaxKind::Comma {
                self.bump();
                self.bump_trivia();
                continue;
            }
            break;
        }

        self.bump_trivia();
        self.expect(SyntaxKind::RBrack, "expected ']'");
        values
    }

    fn parse_usize_list(&mut self) -> Vec<usize> {
        self.builder.start_node(SyntaxKind::ListExpr.into());
        if self.current() != SyntaxKind::LBrack {
            self.unexpected_here("expected '['");
            self.builder.finish_node();
            return Vec::new();
        }
        self.bump();
        let values = self.parse_usize_list_body();
        self.bump_trivia();
        self.expect(SyntaxKind::RBrack, "expected ']'");
        self.builder.finish_node();
        values
    }

    fn parse_usize_list_body(&mut self) -> Vec<usize> {
        self.bump_trivia();
        let mut values = Vec::new();
        if self.current() == SyntaxKind::RBrack {
            return values;
        }
        loop {
            self.bump_trivia();
            if self.current() == SyntaxKind::Int {
                if let Ok(value) = self.current_text().parse::<usize>() {
                    values.push(value);
                }
                self.bump();
            } else {
                self.unexpected_here("expected integer");
                if self.current() == SyntaxKind::Eof {
                    break;
                }
                self.bump();
            }
            self.bump_trivia();
            if self.current() == SyntaxKind::Comma {
                self.bump();
                self.bump_trivia();
                continue;
            }
            break;
        }
        values
    }

    fn parse_string_list(&mut self) -> Vec<String> {
        if self.current() != SyntaxKind::LBrack {
            self.unexpected_here("expected '['");
            return Vec::new();
        }
        self.bump();
        self.bump_trivia();
        let mut values = Vec::new();
        if self.current() != SyntaxKind::RBrack {
            loop {
                if self.current() == SyntaxKind::String {
                    values.push(unquote_string(self.current_text()));
                    self.builder.start_node(SyntaxKind::StringLiteral.into());
                    self.bump();
                    self.builder.finish_node();
                } else {
                    self.unexpected_here("expected string literal");
                    if self.current() == SyntaxKind::Eof {
                        break;
                    }
                    self.bump();
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
        values
    }

    fn parse_bool_list(&mut self) -> Vec<bool> {
        if self.current() != SyntaxKind::LBrack {
            self.unexpected_here("expected '['");
            return Vec::new();
        }
        self.bump();
        self.bump_trivia();
        let mut values = Vec::new();
        if self.current() != SyntaxKind::RBrack {
            loop {
                match self.current() {
                    SyntaxKind::KwTrue => {
                        values.push(true);
                        self.builder.start_node(SyntaxKind::BoolLiteral.into());
                        self.bump();
                        self.builder.finish_node();
                    }
                    SyntaxKind::KwFalse => {
                        values.push(false);
                        self.builder.start_node(SyntaxKind::BoolLiteral.into());
                        self.bump();
                        self.builder.finish_node();
                    }
                    _ => {
                        self.unexpected_here("expected boolean literal");
                        if self.current() == SyntaxKind::Eof {
                            break;
                        }
                        self.bump();
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
        values
    }

    fn parse_list_expr(&mut self) {
        self.builder.start_node(SyntaxKind::ListExpr.into());
        self.expect(SyntaxKind::LBrack, "expected '['");
        self.bump_trivia();
        if self.current() != SyntaxKind::RBrack {
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
        self.expect(SyntaxKind::RBrack, "expected ']'");
        self.builder.finish_node();
    }

    fn parse_ascii_indices(&mut self) {
        self.expect(SyntaxKind::LBrack, "expected '['");
        self.bump_trivia();
        if self.current() != SyntaxKind::RBrack {
            loop {
                self.expect(SyntaxKind::Ident, "expected index name");
                self.bump_trivia();
                match self.current() {
                    SyntaxKind::Plus | SyntaxKind::Minus => self.bump(),
                    _ => self.unexpected_here("expected '+' or '-' for index variance"),
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

    fn current_text(&self) -> &str {
        let span = self.current_span();
        &self.input[span]
    }

    fn current_span(&self) -> std::ops::Range<usize> {
        self.toks
            .get(self.pos)
            .map(|t| t.1.clone())
            .unwrap_or(self.input.len()..self.input.len())
    }

    fn nth_non_trivia_kind(&self, mut offset: usize) -> Option<SyntaxKind> {
        let mut idx = self.pos;
        while idx + 1 < self.toks.len() && offset > 0 {
            idx += 1;
            if !is_trivia(self.toks[idx].0) {
                offset -= 1;
            }
        }
        if offset == 0 {
            self.toks.get(idx).map(|token| token.0)
        } else {
            None
        }
    }

    fn prev_non_trivia_end_pos(&self) -> usize {
        if self.pos == 0 {
            return 0;
        }
        let mut i = self.pos;
        while i > 0 {
            i -= 1;
            let kind = self.toks[i].0;
            if !is_trivia(kind) {
                return self.toks[i].1.end;
            }
        }
        0
    }

    fn bump(&mut self) {
        let (kind, span) = self.toks[self.pos].clone();
        let text = &self.input[span.clone()];
        self.builder.token(kind.into(), text);
        self.pos += 1;
    }

    fn bump_trivia(&mut self) {
        while is_trivia(self.current()) {
            self.bump();
        }
    }

    fn expect(&mut self, kind: SyntaxKind, msg: &str) {
        if self.current() == kind {
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

    fn push_tableau_diag(&mut self, message: &str, span: std::ops::Range<usize>) {
        self.diagnostics.push(Diagnostic::new(
            DiagnosticCode::UnexpectedToken,
            Severity::Error,
            message,
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

    fn at_expr_start(&self) -> bool {
        matches!(
            self.current(),
            SyntaxKind::Ident
                | SyntaxKind::Int
                | SyntaxKind::Float
                | SyntaxKind::String
                | SyntaxKind::KwTrue
                | SyntaxKind::KwFalse
                | SyntaxKind::Minus
                | SyntaxKind::LParen
                | SyntaxKind::LBrack
                | SyntaxKind::Pipe
                | SyntaxKind::Less
        )
    }
}

fn unquote_string(text: &str) -> String {
    text.strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(text)
        .to_string()
}

pub fn parse_file(input: &str) -> (crate::tree::SyntaxNode, Vec<Diagnostic>) {
    let parsed = Parser::new(input).parse();
    (syntax_node_from_green(parsed.green), parsed.diagnostics)
}

pub fn parse_tableau_symmetry(input: &str) -> Result<ax_ir::TensorSymmetry, Vec<Diagnostic>> {
    let source = if input.trim_end().ends_with(';') {
        input.to_string()
    } else {
        format!("{input};")
    };
    let (root, diagnostics) = parse_file(&source);
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    if let Some(expr) = tableau_symmetry_exprs(&root).into_iter().next() {
        if let Some(symmetry) = expr.lower_tensor_symmetry() {
            return Ok(symmetry);
        }
    }

    Err(vec![Diagnostic::new(
        DiagnosticCode::UnexpectedToken,
        Severity::Error,
        "expected tableau_symmetry expression",
        0..input.len(),
    )])
}
