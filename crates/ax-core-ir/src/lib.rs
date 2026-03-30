#![forbid(unsafe_code)]

use ax_ir::{Expr, Index, Interner, Variance};
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct LowerResult {
    pub expr: Option<Expr>,
    pub exprs: Vec<Expr>,
    pub errors: Vec<LowerError>,
}

#[derive(Debug, Clone)]
pub struct LowerError {
    pub message: String,
    pub span: Range<usize>,
}

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
    offset: usize,
    interner: &'a Interner,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str, offset: usize, interner: &'a Interner) -> Self {
        Self {
            src,
            pos: 0,
            offset,
            interner,
        }
    }

    fn error(&self, message: impl Into<String>) -> LowerError {
        let start = self.offset + self.pos;
        LowerError {
            message: message.into(),
            span: start..start,
        }
    }

    fn error_at(&self, start: usize, end: usize, message: impl Into<String>) -> LowerError {
        LowerError {
            message: message.into(),
            span: (self.offset + start)..(self.offset + end),
        }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn eat_if(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump_char();
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.bump_char();
            } else {
                break;
            }
        }
    }

    fn consume_keyword(&mut self, kw: &str) -> bool {
        self.skip_ws();
        if !self.rest().starts_with(kw) {
            return false;
        }

        let end = self.pos + kw.len();
        let boundary = self.src[end..].chars().next();
        if matches!(boundary, Some(ch) if ch.is_ascii_alphanumeric() || ch == '_') {
            return false;
        }

        self.pos = end;
        true
    }

    fn parse_ident(&mut self) -> Result<ax_ir::expr::Sym, LowerError> {
        self.skip_ws();
        let start = self.pos;
        let mut chars = self.rest().char_indices();
        match chars.next() {
            Some((_, ch)) if ch.is_ascii_alphabetic() => {
                self.pos += ch.len_utf8();
            }
            _ => {
                return Err(self.error("expected identifier"));
            }
        }

        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.bump_char();
            } else {
                break;
            }
        }

        Ok(self.interner.get_or_intern(&self.src[start..self.pos]))
    }

    fn parse_number(&mut self) -> Result<Expr, LowerError> {
        self.skip_ws();
        let start = self.pos;
        let mut saw_digit = false;

        while matches!(self.peek_char(), Some(ch) if ch.is_ascii_digit()) {
            saw_digit = true;
            self.bump_char();
        }

        if self.eat_if('.') {
            while matches!(self.peek_char(), Some(ch) if ch.is_ascii_digit()) {
                saw_digit = true;
                self.bump_char();
            }

            if !saw_digit {
                return Err(self.error_at(start, self.pos, "invalid float literal"));
            }

            return self.src[start..self.pos]
                .parse::<f64>()
                .map(Expr::Float)
                .map_err(|e| {
                    self.error_at(start, self.pos, format!("invalid float literal: {e}"))
                });
        }

        if !saw_digit {
            return Err(self.error("expected number"));
        }

        self.src[start..self.pos]
            .parse::<i128>()
            .map(|n| Expr::Int(n.into()))
            .map_err(|e| self.error_at(start, self.pos, format!("invalid integer literal: {e}")))
    }

    fn parse_primary(&mut self) -> Result<Expr, LowerError> {
        self.skip_ws();

        if self.consume_keyword("let") {
            let name = self.parse_ident()?;
            self.skip_ws();
            if !self.eat_if('=') {
                return Err(self.error("expected '='"));
            }
            let value = self.parse_expr()?;
            if self.consume_keyword("in") {
                let body = self.parse_expr()?;
                return Ok(Expr::Let(name, Box::new(value), Box::new(body)));
            }
            return Ok(Expr::Let(name, Box::new(value), Box::new(Expr::Sym(name))));
        }

        match self.peek_char() {
            Some(ch) if ch.is_ascii_digit() => self.parse_number(),
            Some(ch) if ch.is_ascii_alphabetic() => {
                let sym = self.parse_ident()?;
                Ok(Expr::Sym(sym))
            }
            Some('(') => {
                self.bump_char();
                let expr = self.parse_expr()?;
                self.skip_ws();
                if !self.eat_if(')') {
                    return Err(self.error("expected ')'"));
                }
                Ok(expr)
            }
            Some('[') => {
                self.bump_char();
                let mut items = Vec::new();
                self.skip_ws();
                if !self.eat_if(']') {
                    loop {
                        items.push(self.parse_expr()?);
                        self.skip_ws();
                        if self.eat_if(',') {
                            self.skip_ws();
                            continue;
                        }
                        if self.eat_if(']') {
                            break;
                        }
                        return Err(self.error("expected ',' or ']' in list"));
                    }
                }
                Ok(Expr::List(items))
            }
            _ => Err(self.error("expected expression")),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, LowerError> {
        let mut expr = self.parse_primary()?;

        loop {
            self.skip_ws();
            match self.peek_char() {
                Some('(') => {
                    let callee = match expr {
                        Expr::Sym(sym) => sym,
                        other => {
                            return Err(self.error_at(
                                self.pos,
                                self.pos + 1,
                                format!("expected function name before call, got {:?}", other),
                            ));
                        }
                    };

                    self.bump_char();
                    let mut args = Vec::new();
                    self.skip_ws();
                    if !self.eat_if(')') {
                        loop {
                            args.push(self.parse_expr()?);
                            self.skip_ws();
                            if self.eat_if(',') {
                                continue;
                            }
                            if self.eat_if(')') {
                                break;
                            }
                            return Err(self.error("expected ',' or ')' in argument list"));
                        }
                    }
                    expr = Expr::Call(callee, args);
                }
                Some('[') => {
                    self.bump_char();
                    let mut indices = Vec::new();
                    self.skip_ws();
                    if !self.eat_if(']') {
                        loop {
                            let name = self.parse_ident()?;
                            self.skip_ws();
                            let variance = if self.eat_if('+') {
                                Variance::Up
                            } else if self.eat_if('-') {
                                Variance::Down
                            } else {
                                return Err(self.error("expected '+' or '-' after index name"));
                            };

                            indices.push(Index { name, variance });
                            self.skip_ws();
                            if self.eat_if(',') {
                                continue;
                            }
                            if self.eat_if(']') {
                                break;
                            }
                            return Err(self.error("expected ',' or ']' in index list"));
                        }
                    }
                    expr = Expr::Indexed(Box::new(expr), indices);
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, LowerError> {
        self.skip_ws();
        if self.eat_if('-') {
            Ok(Expr::neg(self.parse_unary()?))
        } else {
            self.parse_postfix()
        }
    }

    fn parse_pow(&mut self) -> Result<Expr, LowerError> {
        let lhs = self.parse_unary()?;
        self.skip_ws();
        if self.eat_if('^') {
            let rhs = self.parse_pow()?;
            Ok(Expr::pow(lhs, rhs))
        } else {
            Ok(lhs)
        }
    }

    fn parse_mul(&mut self) -> Result<Expr, LowerError> {
        let mut expr = self.parse_pow()?;

        loop {
            self.skip_ws();
            if self.eat_if('*') {
                let rhs = self.parse_pow()?;
                expr = Expr::mul(vec![expr, rhs]);
            } else if self.eat_if('/') {
                let rhs = self.parse_pow()?;
                expr = Expr::mul(vec![expr, Expr::pow(rhs, Expr::Int((-1).into()))]);
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_add(&mut self) -> Result<Expr, LowerError> {
        let mut expr = self.parse_mul()?;

        loop {
            self.skip_ws();
            if self.eat_if('+') {
                let rhs = self.parse_mul()?;
                expr = Expr::add(vec![expr, rhs]);
            } else if self.eat_if('-') {
                let rhs = self.parse_mul()?;
                expr = Expr::add(vec![expr, Expr::neg(rhs)]);
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, LowerError> {
        self.parse_add()
    }

    fn finish(mut self) -> Result<Expr, LowerError> {
        let expr = self.parse_expr()?;
        self.skip_ws();
        let _ = self.eat_if(';');
        self.skip_ws();

        if self.is_eof() {
            Ok(expr)
        } else {
            Err(self.error("unexpected trailing input"))
        }
    }
}

fn push_current(result: &mut Vec<(usize, String)>, current: &mut String, current_offset: usize) {
    if !current.is_empty() {
        result.push((current_offset, std::mem::take(current)));
    }
}

fn append_segment(
    result: &mut Vec<(usize, String)>,
    current: &mut String,
    current_offset: &mut usize,
    segment: &str,
    segment_offset: usize,
    terminated: bool,
) {
    let trimmed = segment.trim();

    if trimmed.is_empty() || trimmed.starts_with("//") {
        if terminated {
            push_current(result, current, *current_offset);
        }
        return;
    }

    if current.is_empty() {
        *current_offset = segment_offset + (segment.len() - segment.trim_start().len());
        current.push_str(trimmed);
    } else {
        current.push(' ');
        current.push_str(trimmed);
    }

    if terminated {
        push_current(result, current, *current_offset);
    }
}

fn split_statements(source: &str) -> Vec<(usize, String)> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_offset = 0;
    let mut offset = 0;

    for chunk in source.split_inclusive('\n') {
        let line = chunk.strip_suffix('\n').unwrap_or(chunk);
        let line_start = offset;

        if !line.contains(';') {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                push_current(&mut result, &mut current, current_offset);
            } else {
                append_segment(
                    &mut result,
                    &mut current,
                    &mut current_offset,
                    line,
                    line_start,
                    !line_continues(trimmed),
                );
            }
        } else {
            let mut segment_start = 0;
            for (idx, ch) in line.char_indices() {
                if ch == ';' {
                    let segment = &line[segment_start..idx];
                    append_segment(
                        &mut result,
                        &mut current,
                        &mut current_offset,
                        segment,
                        line_start + segment_start,
                        true,
                    );
                    segment_start = idx + ch.len_utf8();
                }
            }

            let trailing = &line[segment_start..];
            let trailing_trimmed = trailing.trim();
            if trailing_trimmed.is_empty() || trailing_trimmed.starts_with("//") {
                push_current(&mut result, &mut current, current_offset);
            } else {
                append_segment(
                    &mut result,
                    &mut current,
                    &mut current_offset,
                    trailing,
                    line_start + segment_start,
                    !line_continues(trailing_trimmed),
                );
            }
        }

        offset += chunk.len();
    }

    push_current(&mut result, &mut current, current_offset);
    result
}

fn line_continues(trimmed: &str) -> bool {
    trimmed.ends_with('+')
        || trimmed.ends_with('-')
        || trimmed.ends_with('*')
        || trimmed.ends_with('/')
        || trimmed.ends_with('^')
        || trimmed.ends_with('=')
        || trimmed.ends_with(',')
        || trimmed.ends_with('(')
        || trimmed.ends_with('[')
        || trimmed.ends_with('{')
}

pub fn lower(source: &str, interner: &Interner) -> LowerResult {
    let mut exprs = Vec::new();
    let mut errors = Vec::new();

    for (offset, stmt) in split_statements(source) {
        let trimmed = stmt.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("module ") || trimmed.starts_with("import ") {
            continue;
        }

        match Cursor::new(trimmed, offset, interner).finish() {
            Ok(expr) => exprs.push(expr),
            Err(err) => errors.push(err),
        }
    }

    let expr = exprs.last().cloned();
    LowerResult {
        expr,
        exprs,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_one(src: &str) -> ax_ir::Expr {
        let interner = ax_ir::Interner::new();
        let result = lower(src, &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        result.expr.expect("expected an expression")
    }

    #[test]
    fn lower_integer() {
        let e = lower_one("42;");
        assert_eq!(e, ax_ir::Expr::Int(42.into()));
    }

    #[test]
    fn lower_addition() {
        let e = lower_one("1 + 2;");
        assert_eq!(e, ax_ir::Expr::Int(3.into()));
    }

    #[test]
    fn lower_without_semicolon() {
        let e = lower_one("42");
        assert_eq!(e, ax_ir::Expr::Int(42.into()));
    }

    #[test]
    fn lower_multiline() {
        let interner = ax_ir::Interner::new();
        let result = lower("x + 1\ny + 2", &interner);
        assert_eq!(result.exprs.len(), 2);
    }

    #[test]
    fn lower_continuation() {
        let e = lower_one("1 +\n2");
        assert_eq!(e, ax_ir::Expr::Int(3.into()));
    }

    #[test]
    fn lower_comment_skipped() {
        let interner = ax_ir::Interner::new();
        let result = lower("// this is a comment\n42", &interner);
        assert_eq!(result.exprs.len(), 1);
    }

    #[test]
    fn lower_bare_let() {
        let interner = ax_ir::Interner::new();
        let result = lower("let x = 5", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.unwrap();
        match expr {
            ax_ir::Expr::Let(name, val, body) => {
                assert_eq!(interner.resolve(name), "x");
                assert_eq!(*val, ax_ir::Expr::Int(5.into()));
                assert!(matches!(*body, ax_ir::Expr::Sym(s) if interner.resolve(s) == "x"));
            }
            other => panic!("expected Let, got {:?}", other),
        }
    }

    #[test]
    fn lower_list_literal() {
        let interner = ax_ir::Interner::new();
        let result = lower("[1, 2, 3]", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.unwrap();
        match expr {
            ax_ir::Expr::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], ax_ir::Expr::Int(1.into()));
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn lower_list_of_symbols() {
        let interner = ax_ir::Interner::new();
        let result = lower("[t, r, theta, phi]", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.unwrap();
        match expr {
            ax_ir::Expr::List(items) => assert_eq!(items.len(), 4),
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn lower_function_call_in_power() {
        let interner = ax_ir::Interner::new();
        let result = lower("a(t)^2", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.unwrap();
        match expr {
            ax_ir::Expr::Pow(base, exp) => {
                assert!(matches!(*base, ax_ir::Expr::Call(_, _)));
                assert_eq!(*exp, ax_ir::Expr::Int(2.into()));
            }
            other => panic!("expected Pow(Call, Int), got {:?}", other),
        }
    }

    #[test]
    fn lower_nested_expression() {
        let interner = ax_ir::Interner::new();
        let result = lower("r^2 * sin(theta)^2", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    }

    #[test]
    fn lower_symbol() {
        let interner = ax_ir::Interner::new();
        let result = lower("x;", &interner);
        let e = result.expr.unwrap();
        match e {
            ax_ir::Expr::Sym(s) => assert_eq!(interner.resolve(s), "x"),
            other => panic!("expected Sym, got {:?}", other),
        }
    }

    #[test]
    fn lower_indexed() {
        let interner = ax_ir::Interner::new();
        let result = lower("T[mu-, nu+];", &interner);
        let e = result.expr.unwrap();
        match e {
            ax_ir::Expr::Indexed(base, indices) => {
                let _ = base;
                assert_eq!(indices.len(), 2);
                assert_eq!(indices[0].variance, ax_ir::Variance::Down);
                assert_eq!(indices[1].variance, ax_ir::Variance::Up);
            }
            other => panic!("expected Indexed, got {:?}", other),
        }
    }

    #[test]
    fn lower_function_call() {
        let interner = ax_ir::Interner::new();
        let result = lower("f(1, 2);", &interner);
        let e = result.expr.unwrap();
        match e {
            ax_ir::Expr::Call(f, args) => {
                assert_eq!(interner.resolve(f), "f");
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}
