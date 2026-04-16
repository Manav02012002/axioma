#![forbid(unsafe_code)]

pub mod symmetry;

use ax_ir::{Assumption, Condition, Expr, Index, Interner, Variance};
use std::ops::Range;

pub use symmetry::{
    lower_tensor_symmetry, CoreSymmetryLowerError, CoreTableauAttachment, CoreTensorSymmetry,
};

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
    fn parse_trust_level(&self, name: &str) -> ax_ir::TrustLevel {
        match name {
            "exact" => ax_ir::TrustLevel::Exact,
            "assumptions" => ax_ir::TrustLevel::UnderAssumptions,
            "heuristic" => ax_ir::TrustLevel::Heuristic,
            "numerical" => ax_ir::TrustLevel::NumericallyChecked,
            _ => ax_ir::TrustLevel::Unverified,
        }
    }

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
        if !self.starts_keyword(kw) {
            return false;
        }

        self.pos += kw.len();
        true
    }

    fn starts_keyword(&self, kw: &str) -> bool {
        if !self.rest().starts_with(kw) {
            return false;
        }

        let end = self.pos + kw.len();
        let boundary = self.src[end..].chars().next();
        if matches!(boundary, Some(ch) if ch.is_ascii_alphanumeric() || ch == '_') {
            return false;
        }
        true
    }

    fn consume_arrow(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with("=>") {
            self.pos += 2;
            true
        } else {
            false
        }
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

        if self.consume_keyword("if") {
            let cond = self.parse_condition()?;
            if !self.consume_keyword("then") {
                return Err(self.error("expected 'then'"));
            }
            let then_expr = self.parse_expr()?;
            if !self.consume_keyword("else") {
                return Err(self.error("expected 'else'"));
            }
            let else_expr = self.parse_expr()?;
            return Ok(Expr::Piecewise(vec![
                (then_expr, cond),
                (else_expr, Condition::True),
            ]));
        }

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

        if self.consume_keyword("assume") {
            let var = self.parse_ident()?;
            let mut assumptions = Vec::new();
            loop {
                self.skip_ws();
                if self.consume_keyword("real") {
                    assumptions.push(Assumption::Real);
                } else if self.consume_keyword("positive") {
                    assumptions.push(Assumption::Positive);
                } else if self.consume_keyword("negative") {
                    assumptions.push(Assumption::Negative);
                } else if self.consume_keyword("nonzero") {
                    assumptions.push(Assumption::NonZero);
                } else if self.consume_keyword("integer") {
                    assumptions.push(Assumption::Integer);
                } else if self.consume_keyword("even") {
                    assumptions.push(Assumption::Even);
                } else if self.consume_keyword("odd") {
                    assumptions.push(Assumption::Odd);
                } else {
                    break;
                }
            }
            return Ok(Expr::Assume(var, assumptions));
        }

        if self.consume_keyword("grassmann") {
            let mut vars = Vec::new();
            loop {
                self.skip_ws();
                let saved = self.pos;
                match self.parse_ident() {
                    Ok(sym) => vars.push(sym),
                    Err(_) => {
                        self.pos = saved;
                        break;
                    }
                }
            }
            let grassmann_sym = self.interner.get_or_intern("grassmann");
            return Ok(Expr::Call(
                grassmann_sym,
                vars.into_iter().map(Expr::Sym).collect(),
            ));
        }

        if self.consume_keyword("indices") {
            let family = self.parse_ident()?;
            self.skip_ws();
            if !self.eat_if('[') {
                return Err(self.error("expected '[' after family name"));
            }

            let mut index_names = Vec::new();
            self.skip_ws();
            if !self.eat_if(']') {
                loop {
                    index_names.push(self.parse_ident()?);
                    self.skip_ws();
                    if self.eat_if(',') {
                        self.skip_ws();
                        continue;
                    }
                    if self.eat_if(']') {
                        break;
                    }
                    return Err(self.error("expected ',' or ']'"));
                }
            }

            let mut dim = None;
            let mut values = Vec::new();
            let mut position = String::from("free");

            loop {
                self.skip_ws();
                if self.is_eof() || self.peek_char() == Some(';') {
                    break;
                }
                if self.consume_keyword("dim") {
                    self.skip_ws();
                    let _ = self.eat_if('=');
                    self.skip_ws();
                    let parsed = self.parse_number()?;
                    if let Expr::Int(n) = parsed {
                        dim = Some(n);
                    } else {
                        return Err(self.error("expected integer dimension"));
                    }
                } else if self.consume_keyword("values") {
                    self.skip_ws();
                    let _ = self.eat_if('=');
                    self.skip_ws();
                    if !self.eat_if('[') {
                        return Err(self.error("expected '[' after values="));
                    }
                    self.skip_ws();
                    if !self.eat_if(']') {
                        loop {
                            values.push(self.parse_ident()?);
                            self.skip_ws();
                            if self.eat_if(',') {
                                self.skip_ws();
                                continue;
                            }
                            if self.eat_if(']') {
                                break;
                            }
                            return Err(self.error("expected ',' or ']'"));
                        }
                    }
                } else if self.consume_keyword("position") {
                    self.skip_ws();
                    let _ = self.eat_if('=');
                    self.skip_ws();
                    if self.consume_keyword("fixed") {
                        position = "fixed".to_string();
                    } else {
                        let _ = self.consume_keyword("free");
                        position = "free".to_string();
                    }
                } else {
                    break;
                }
            }

            let declare_sym = self.interner.get_or_intern("__declare_indices");
            let mut args = vec![
                Expr::Sym(family),
                Expr::List(index_names.into_iter().map(Expr::Sym).collect()),
            ];
            if let Some(d) = dim {
                args.push(Expr::Int(d));
            }
            if !values.is_empty() {
                args.push(Expr::List(values.into_iter().map(Expr::Sym).collect()));
            }
            args.push(Expr::Sym(self.interner.get_or_intern(&position)));
            return Ok(Expr::Call(declare_sym, args));
        }

        if self.consume_keyword("coordinates") {
            self.skip_ws();
            if !self.eat_if('[') {
                return Err(self.error("expected '['"));
            }
            let mut coords = Vec::new();
            self.skip_ws();
            if !self.eat_if(']') {
                loop {
                    coords.push(self.parse_ident()?);
                    self.skip_ws();
                    if self.eat_if(',') {
                        self.skip_ws();
                        continue;
                    }
                    if self.eat_if(']') {
                        break;
                    }
                    return Err(self.error("expected ',' or ']'"));
                }
            }
            let sym = self.interner.get_or_intern("__declare_coordinates");
            return Ok(ax_ir::Expr::Call(
                sym,
                coords.into_iter().map(ax_ir::Expr::Sym).collect(),
            ));
        }

        if self.consume_keyword("property") {
            let tensor = self.parse_postfix()?;
            self.skip_ws();
            let prop_name = self.parse_postfix()?;
            let declare_sym = self.interner.get_or_intern("__declare_property");
            return Ok(ax_ir::Expr::Call(declare_sym, vec![tensor, prop_name]));
        }

        if self.consume_keyword("weight") {
            let sym = self.parse_ident()?;
            self.skip_ws();
            // Parse optional leading minus for negative weights
            let neg = self.eat_if('-');
            let num_expr = self.parse_number()?;
            let weight_expr = if neg { Expr::neg(num_expr) } else { num_expr };
            self.skip_ws();
            // Optional label=name
            let mut args = vec![Expr::Sym(sym), weight_expr];
            if self.rest().starts_with("label=") {
                self.pos += "label=".len();
                let label = self.parse_ident()?;
                args.push(Expr::Sym(label));
            }
            let declare_sym = self.interner.get_or_intern("__declare_weight");
            return Ok(Expr::Call(declare_sym, args));
        }

        if self.consume_keyword("depends") {
            let tensor = self.parse_ident()?;
            self.skip_ws();
            let mut deps = Vec::new();
            if self.eat_if('[') {
                self.skip_ws();
                if !self.eat_if(']') {
                    loop {
                        deps.push(self.parse_ident()?);
                        self.skip_ws();
                        if self.eat_if(',') {
                            self.skip_ws();
                            continue;
                        }
                        if self.eat_if(']') {
                            break;
                        }
                        return Err(self.error("expected ',' or ']'"));
                    }
                }
            } else {
                deps.push(self.parse_ident()?);
            }
            let declare_sym = self.interner.get_or_intern("__declare_depends");
            let mut args = vec![Expr::Sym(tensor)];
            args.push(Expr::List(deps.into_iter().map(Expr::Sym).collect()));
            return Ok(Expr::Call(declare_sym, args));
        }

        if self.consume_keyword("convention") {
            let field = self.parse_ident()?;
            self.skip_ws();
            let value = self.parse_ident()?;
            let field_str = self.interner.resolve(field).to_string();
            let value_str = self.interner.resolve(value).to_string();
            return Ok(Expr::SetConvention(field_str, value_str));
        }

        if self.consume_keyword("parallel") {
            self.skip_ws();
            let mode = self.parse_ident()?;
            let set_parallel = self.interner.get_or_intern("__set_parallel");
            return Ok(Expr::Call(set_parallel, vec![Expr::Sym(mode)]));
        }

        if self.consume_keyword("import") {
            let mut path = Vec::new();
            path.push(self.parse_ident()?);
            loop {
                self.skip_ws();
                if self.eat_if('.') {
                    path.push(self.parse_ident()?);
                } else {
                    break;
                }
            }
            return Ok(Expr::Import(path));
        }

        let rule_start = self.pos;
        if self.consume_keyword("rule") {
            if !self.rest().contains("=>") {
                self.pos = rule_start;
            } else {
                self.skip_ws();
                let trust = if self.eat_if('[') {
                    self.skip_ws();
                    let level_name = self.parse_ident()?;
                    self.skip_ws();
                    if !self.eat_if(']') {
                        return Err(self.error("expected ']'"));
                    }
                    let name = self.interner.resolve(level_name);
                    self.parse_trust_level(name)
                } else {
                    ax_ir::TrustLevel::Unverified
                };
                let saved = self.pos;
                if self.parse_ident().is_ok() {
                    self.skip_ws();
                    if !self.eat_if(':') {
                        self.pos = saved;
                    }
                } else {
                    self.pos = saved;
                }
                self.skip_ws();
                let lhs = self.parse_expr()?;
                self.skip_ws();
                if !self.consume_arrow() {
                    return Err(self.error("expected '=>' in rule"));
                }
                self.skip_ws();
                let rhs = self.parse_expr()?;
                return Ok(Expr::Rule(Box::new(lhs), Box::new(rhs), trust));
            }
        }

        match self.peek_char() {
            Some(ch) if ch.is_ascii_digit() => self.parse_number(),
            Some(ch) if ch.is_ascii_alphabetic() => {
                let sym = self.parse_ident()?;
                Ok(Expr::Sym(sym))
            }
            Some('"') => {
                self.bump_char();
                let start = self.pos;
                while let Some(ch) = self.peek_char() {
                    if ch == '"' {
                        let value = &self.src[start..self.pos];
                        self.bump_char();
                        return Ok(Expr::Sym(self.interner.get_or_intern(value)));
                    }
                    self.bump_char();
                }
                Err(self.error("unterminated string literal"))
            }
            Some('(') => {
                self.bump_char();
                let expr = self.parse_expr()?;
                self.skip_ws();
                if !self.eat_if(')') {
                    return Err(self.error("expected ')'"));
                }
                Ok(Expr::group(expr))
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

    fn parse_condition(&mut self) -> Result<Condition, LowerError> {
        self.parse_or_condition()
    }

    fn parse_or_condition(&mut self) -> Result<Condition, LowerError> {
        let mut cond = self.parse_and_condition()?;
        loop {
            self.skip_ws();
            if !self.consume_keyword("or") {
                break;
            }
            let rhs = self.parse_and_condition()?;
            cond = Condition::Or(Box::new(cond), Box::new(rhs));
        }
        Ok(cond)
    }

    fn parse_and_condition(&mut self) -> Result<Condition, LowerError> {
        let mut cond = self.parse_not_condition()?;
        loop {
            self.skip_ws();
            if !self.consume_keyword("and") {
                break;
            }
            let rhs = self.parse_not_condition()?;
            cond = Condition::And(Box::new(cond), Box::new(rhs));
        }
        Ok(cond)
    }

    fn parse_not_condition(&mut self) -> Result<Condition, LowerError> {
        self.skip_ws();

        if self.consume_keyword("not") {
            return Ok(Condition::Not(Box::new(self.parse_not_condition()?)));
        }
        if self.consume_keyword("true") {
            return Ok(Condition::True);
        }
        if self.consume_keyword("false") {
            return Ok(Condition::False);
        }

        self.parse_comparison_condition()
    }

    fn parse_comparison_condition(&mut self) -> Result<Condition, LowerError> {
        let lhs = self.parse_add()?;
        self.skip_ws();

        if self.rest().starts_with(">=") {
            self.pos += 2;
            let rhs = self.parse_add()?;
            return Ok(Condition::Ge(lhs, rhs));
        }
        if self.rest().starts_with("<=") {
            self.pos += 2;
            let rhs = self.parse_add()?;
            return Ok(Condition::Le(lhs, rhs));
        }
        if self.rest().starts_with("==") {
            self.pos += 2;
            let rhs = self.parse_add()?;
            return Ok(Condition::Eq(lhs, rhs));
        }
        if self.rest().starts_with("!=") {
            self.pos += 2;
            let rhs = self.parse_add()?;
            return Ok(Condition::Ne(lhs, rhs));
        }
        if self.eat_if('>') {
            let rhs = self.parse_add()?;
            return Ok(Condition::Gt(lhs, rhs));
        }
        if self.eat_if('<') {
            let rhs = self.parse_add()?;
            return Ok(Condition::Lt(lhs, rhs));
        }

        Err(self.error("expected comparison operator"))
    }

    fn parse_piecewise_args(&mut self) -> Result<Vec<(Expr, Condition)>, LowerError> {
        let mut cases = Vec::new();
        self.skip_ws();
        if self.eat_if(')') {
            return Ok(cases);
        }

        loop {
            let value = self.parse_expr()?;
            self.skip_ws();
            if !self.eat_if(',') {
                return Err(self.error("expected ',' after piecewise value"));
            }
            self.skip_ws();

            if self.consume_keyword("true") {
                self.skip_ws();
                if !self.eat_if(')') {
                    return Err(self.error("expected ')' after piecewise default case"));
                }
                cases.push((value, Condition::True));
                break;
            }
            if self.consume_keyword("false") {
                let condition = Condition::False;
                self.skip_ws();
                if self.eat_if(',') {
                    cases.push((value, condition));
                    continue;
                }
                if self.eat_if(')') {
                    cases.push((value, condition));
                    break;
                }
                return Err(self.error("expected ',' or ')' in piecewise"));
            }

            let condition = self.parse_condition()?;
            cases.push((value, condition));
            self.skip_ws();
            if self.eat_if(',') {
                continue;
            }
            if self.eat_if(')') {
                break;
            }
            return Err(self.error("expected ',' or ')' in piecewise"));
        }

        Ok(cases)
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
                    if self.interner.resolve(callee) == "piecewise" {
                        expr = Expr::Piecewise(self.parse_piecewise_args()?);
                        continue;
                    }

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

                            indices.push(Index {
                                name,
                                variance,
                                index_type: None,
                            });
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
            let before_ws = self.pos;
            self.skip_ws();
            let saw_ws = self.pos != before_ws;
            if self.eat_if('*') {
                let rhs = self.parse_pow()?;
                expr = Expr::mul(vec![expr, rhs]);
            } else if self.eat_if('/') {
                let rhs = self.parse_pow()?;
                expr = Expr::mul(vec![expr, Expr::pow(rhs, Expr::Int((-1).into()))]);
            } else if !saw_ws && self.starts_implicit_mul_rhs() {
                let rhs = self.parse_pow()?;
                expr = Expr::mul(vec![expr, rhs]);
            } else {
                return Ok(expr);
            }
        }
    }

    fn starts_implicit_mul_rhs(&self) -> bool {
        if self.starts_keyword("then") || self.starts_keyword("else") || self.starts_keyword("in") {
            return false;
        }
        matches!(
            self.peek_char(),
            Some(ch) if ch.is_ascii_digit() || ch.is_ascii_alphabetic() || ch == '(' || ch == '[' || ch == '"'
        )
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
        let lhs = self.parse_add()?;
        self.skip_ws();
        if self.rest().starts_with("==") {
            self.pos += 2;
            let rhs = self.parse_add()?;
            let eq_sym = self.interner.get_or_intern("__eq");
            Ok(Expr::Call(eq_sym, vec![lhs, rhs]))
        } else {
            Ok(lhs)
        }
    }

    fn try_rewrite_as_fn_def(&mut self, expr: Expr) -> Result<Expr, LowerError> {
        self.skip_ws();
        if let Expr::Call(name, args) = expr {
            if self.eat_if('=') {
                let params = args
                    .iter()
                    .map(|arg| match arg {
                        Expr::Sym(sym) => Ok(*sym),
                        _ => Err(self.error("function parameters must be identifiers")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let body = self.parse_expr()?;
                return Ok(Expr::FnDef(name, params, Box::new(body)));
            }
            return Ok(Expr::Call(name, args));
        }
        Ok(expr)
    }

    fn finish(mut self) -> Result<Expr, LowerError> {
        let expr = self.parse_expr()?;
        let expr = self.try_rewrite_as_fn_def(expr)?;
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

// ─── LaTeX input translation ──────────────────────────────────────────────────

fn skip_whitespace(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn parse_brace_content(chars: &[char], i: &mut usize) -> String {
    let mut result = String::new();
    if *i < chars.len() && chars[*i] == '{' {
        *i += 1;
        let mut depth = 1;
        while *i < chars.len() && depth > 0 {
            if chars[*i] == '{' {
                depth += 1;
            }
            if chars[*i] == '}' {
                depth -= 1;
            }
            if depth > 0 {
                result.push(chars[*i]);
            }
            *i += 1;
        }
    }
    result
}

fn parse_brace_group(chars: &[char], i: &mut usize) -> String {
    skip_whitespace(chars, i);
    if *i < chars.len() && chars[*i] == '{' {
        parse_brace_content(chars, i)
    } else if *i < chars.len() {
        let c = chars[*i];
        *i += 1;
        c.to_string()
    } else {
        String::new()
    }
}

/// Convert LaTeX-style tensor notation to Axioma notation.
///
/// ```text
/// R_{a b c d}       →  R[a-, b-, c-, d-]
/// T^{a}_{b}         →  T[a+, b-]
/// \partial_{a}      →  partial[a-]
/// \Gamma^{a}_{b c}  →  Gamma[a+, b-, c-]
/// \frac{a}{b}       →  (a) / (b)
/// \sqrt{x}          →  sqrt(x)
/// \mu               →  mu
/// ```
pub fn latex_to_axioma(input: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '\\' {
            // LaTeX command: read alphabetic characters
            let cmd_start = i + 1;
            let mut cmd_end = cmd_start;
            while cmd_end < chars.len() && chars[cmd_end].is_alphabetic() {
                cmd_end += 1;
            }
            let cmd = &input[cmd_start..cmd_end];

            match cmd {
                "frac" => {
                    i = cmd_end;
                    let numer = parse_brace_group(&chars, &mut i);
                    let denom = parse_brace_group(&chars, &mut i);
                    result.push_str(&format!("({}) / ({})", numer, denom));
                }
                "sqrt" => {
                    i = cmd_end;
                    let arg = parse_brace_group(&chars, &mut i);
                    result.push_str(&format!("sqrt({})", arg));
                }
                "bar" => {
                    i = cmd_end;
                    let arg = parse_brace_group(&chars, &mut i);
                    result.push_str(&format!("bar({})", arg));
                }
                "cdot" | "times" => {
                    result.push_str(" * ");
                    i = cmd_end;
                }
                "left" | "right" | "bigl" | "bigr" | "Big" | "big" => {
                    i = cmd_end;
                    if i < chars.len()
                        && (chars[i] == '('
                            || chars[i] == ')'
                            || chars[i] == '['
                            || chars[i] == ']')
                    {
                        result.push(chars[i]);
                        i += 1;
                    }
                }
                "int" => {
                    result.push_str("integrate");
                    i = cmd_end;
                }
                // Greek letters and named differential operators: pass through as-is
                "partial" | "nabla" | "Gamma" | "gamma" | "epsilon" | "delta" | "sigma"
                | "alpha" | "beta" | "mu" | "nu" | "rho" | "lambda" | "theta" | "phi" | "psi"
                | "chi" | "omega" | "pi" | "tau" | "kappa" | "eta" | "zeta" | "xi" | "Pi"
                | "Sigma" | "Omega" | "Delta" | "Lambda" | "Theta" | "Phi" | "Psi" | "Xi" => {
                    result.push_str(cmd);
                    i = cmd_end;
                }
                _ => {
                    // Unknown command: pass through as identifier
                    result.push_str(cmd);
                    i = cmd_end;
                }
            }
        } else if chars[i] == '_' || chars[i] == '^' {
            let variance = if chars[i] == '_' { '-' } else { '+' };
            i += 1;

            if i < chars.len() {
                let index_content = if chars[i] == '{' {
                    parse_brace_content(&chars, &mut i)
                } else {
                    let c = chars[i];
                    i += 1;
                    c.to_string()
                };

                let indices: Vec<&str> = index_content.split_whitespace().collect();
                if !indices.is_empty() {
                    let needs_open = !result.ends_with(']') && !result.ends_with(',');
                    if needs_open && !result.ends_with('[') {
                        result.push('[');
                    } else if result.ends_with(']') {
                        // Extend existing bracket: remove closing `]`, add separator
                        result.pop();
                        result.push_str(", ");
                    }

                    for (j, idx) in indices.iter().enumerate() {
                        if j > 0 {
                            result.push_str(", ");
                        }
                        let clean = latex_to_axioma(idx);
                        result.push_str(&clean);
                        result.push(variance);
                    }
                    result.push(']');
                }
            }
        } else if chars[i] == '{' {
            // Bare brace group outside index context — pass through contents
            i += 1;
            let mut depth = 1;
            while i < chars.len() && depth > 0 {
                if chars[i] == '{' {
                    depth += 1;
                }
                if chars[i] == '}' {
                    depth -= 1;
                }
                if depth > 0 {
                    result.push(chars[i]);
                }
                i += 1;
            }
        } else if chars[i] == '}' {
            i += 1; // skip stray closing braces
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Lower a LaTeX string by first converting to Axioma notation, then parsing.
pub fn lower_latex(input: &str, interner: &Interner) -> LowerResult {
    let axioma_str = latex_to_axioma(input);
    lower(&axioma_str, interner)
}

pub fn lower(source: &str, interner: &Interner) -> LowerResult {
    let mut exprs = Vec::new();
    let mut errors = Vec::new();

    for (offset, stmt) in split_statements(source) {
        let trimmed = stmt.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("module ") {
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

    #[test]
    fn lower_function_definition() {
        let interner = ax_ir::Interner::new();
        let result = lower("f(x, y) = x^2 + y", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match result.expr.unwrap() {
            ax_ir::Expr::FnDef(name, params, body) => {
                assert_eq!(interner.resolve(name), "f");
                assert_eq!(params.len(), 2);
                assert_eq!(interner.resolve(params[0]), "x");
                assert_eq!(interner.resolve(params[1]), "y");
                assert!(matches!(*body, ax_ir::Expr::Add(_)));
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn parse_rule() {
        let interner = ax_ir::Interner::new();
        let result = lower("rule pythag: sin(x_)^2 + cos(x_)^2 => 1", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.unwrap();
        assert!(matches!(expr, ax_ir::Expr::Rule(_, _, _)));
    }

    #[test]
    fn parse_index_declaration() {
        let interner = ax_ir::Interner::new();
        let result = lower("indices spacetime [mu, nu, rho, sigma] dim=4", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn parse_coordinates() {
        let interner = ax_ir::Interner::new();
        let result = lower("coordinates [t, r, theta, phi]", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn parse_property_metric() {
        let interner = ax_ir::Interner::new();
        let result = lower("property g metric", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn parse_property_antisymmetric() {
        let interner = ax_ir::Interner::new();
        let result = lower("property F antisymmetric", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn parse_pattern_property_symmetric_positions() {
        let interner = ax_ir::Interner::new();
        let result = lower("property T[a-, b-] symmetric([0, 1])", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        if let Some(ax_ir::Expr::Call(f, args)) = result.expr {
            assert_eq!(interner.resolve(f), "__declare_property");
            assert!(matches!(args[0], ax_ir::Expr::Indexed(_, _)));
            assert!(matches!(args[1], ax_ir::Expr::Call(_, _)));
        } else {
            panic!("expected __declare_property call");
        }
    }

    #[test]
    fn parse_property_tableau_symmetry_with_args() {
        let interner = ax_ir::Interner::new();
        let result = lower("property T tableau_symmetry([2, 1], [0, 1, 2])", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        if let Some(ax_ir::Expr::Call(f, args)) = result.expr {
            assert_eq!(interner.resolve(f), "__declare_property");
            assert!(matches!(args[0], ax_ir::Expr::Sym(_)));
            assert!(matches!(args[1], ax_ir::Expr::Call(_, _)));
        } else {
            panic!("expected __declare_property call");
        }
    }

    #[test]
    fn parse_rule_with_trust() {
        let interner = ax_ir::Interner::new();
        let result = lower("rule [exact] foo: sin(x_)^2 + cos(x_)^2 => 1", &interner);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        if let ax_ir::Expr::Rule(_, _, trust) = result.expr.unwrap() {
            assert_eq!(trust, ax_ir::TrustLevel::Exact);
        } else {
            panic!("expected Rule");
        }
    }

    #[test]
    fn assume_integer_parse() {
        let interner = ax_ir::Interner::new();
        let result = lower("assume n integer", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(matches!(result.expr.unwrap(), ax_ir::Expr::Assume(_, _)));
    }

    #[test]
    fn parse_import() {
        let interner = ax_ir::Interner::new();
        let result = lower("import std.gr.schwarzschild", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(matches!(result.expr.unwrap(), ax_ir::Expr::Import(_)));
    }

    #[test]
    fn parse_spinor_property() {
        let interner = ax_ir::Interner::new();
        let result = lower("property psi spinor", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    }

    #[test]
    fn parse_depends() {
        let interner = ax_ir::Interner::new();
        let result = lower("depends A [x, t]", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        if let Some(ax_ir::Expr::Call(f, args)) = result.expr {
            assert_eq!(interner.resolve(f), "__declare_depends");
            assert_eq!(args.len(), 2);
        } else {
            panic!("expected __declare_depends call");
        }
    }

    #[test]
    fn parse_depends_single() {
        let interner = ax_ir::Interner::new();
        let result = lower("depends phi x", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        if let Some(ax_ir::Expr::Call(f, _)) = result.expr {
            assert_eq!(interner.resolve(f), "__declare_depends");
        } else {
            panic!("expected __declare_depends call");
        }
    }

    // ── LaTeX translation tests ───────────────────────────────────────────────

    #[test]
    fn latex_simple_tensor() {
        assert_eq!(latex_to_axioma("R_{a b c d}"), "R[a-, b-, c-, d-]");
    }

    #[test]
    fn latex_mixed_indices() {
        let result = latex_to_axioma("T^{a}_{b}");
        assert!(
            result.contains("a+") && result.contains("b-"),
            "got: {}",
            result
        );
    }

    #[test]
    fn latex_frac() {
        assert_eq!(latex_to_axioma("\\frac{a}{b}"), "(a) / (b)");
    }

    #[test]
    fn latex_sqrt() {
        assert_eq!(latex_to_axioma("\\sqrt{x}"), "sqrt(x)");
    }

    #[test]
    fn latex_greek() {
        let result = latex_to_axioma("\\mu");
        assert_eq!(result, "mu");
    }

    #[test]
    fn latex_partial() {
        let result = latex_to_axioma("\\partial_{a}");
        assert!(
            result.contains("partial") && result.contains("a-"),
            "got: {}",
            result
        );
    }

    #[test]
    fn latex_gamma_mixed() {
        // \Gamma^{a}_{b c} → Gamma[a+, b-, c-]
        let result = latex_to_axioma("\\Gamma^{a}_{b c}");
        assert!(result.contains("Gamma"), "got: {}", result);
        assert!(result.contains("a+"), "got: {}", result);
        assert!(result.contains("b-"), "got: {}", result);
        assert!(result.contains("c-"), "got: {}", result);
    }

    #[test]
    fn latex_lower_latex_roundtrip() {
        // lower_latex("g_{\\mu \\nu}") should parse without errors
        let interner = ax_ir::Interner::new();
        let result = lower_latex("g_{a b}", &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.expr.is_some(), "expected an expression");
    }
}
