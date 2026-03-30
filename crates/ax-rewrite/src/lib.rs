#![forbid(unsafe_code)]

use ax_ir::{Expr, Interner};
use std::collections::HashMap;

pub type Bindings = HashMap<lasso::Spur, Expr>;

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Slot(lasso::Spur),
    Wildcard,
    Exact(Expr),
    Add(Vec<Pattern>),
    Mul(Vec<Pattern>),
    Pow(Box<Pattern>, Box<Pattern>),
    Neg(Box<Pattern>),
    Call(lasso::Spur, Vec<Pattern>),
}

#[derive(Clone, Debug)]
pub struct RewriteRule {
    pub name: String,
    pub pattern: Pattern,
    pub replacement: Expr,
    pub condition: Option<fn(&Bindings, &Interner) -> bool>,
}

fn restore_binds(target: &mut Bindings, snapshot: &Bindings) {
    *target = snapshot.clone();
}

fn match_sequence(patterns: &[Pattern], terms: &[Expr], binds: &mut Bindings) -> bool {
    if patterns.is_empty() {
        return terms.is_empty();
    }
    if terms.is_empty() {
        return false;
    }

    fn helper(
        patterns: &[Pattern],
        terms: &[Expr],
        used: &mut [bool],
        binds: &mut Bindings,
    ) -> bool {
        if patterns.is_empty() {
            return used.iter().all(|u| *u);
        }

        for idx in 0..terms.len() {
            if used[idx] {
                continue;
            }
            let snapshot = binds.clone();
            if match_pattern(&patterns[0], &terms[idx], binds) {
                used[idx] = true;
                if helper(&patterns[1..], terms, used, binds) {
                    return true;
                }
                used[idx] = false;
            }
            restore_binds(binds, &snapshot);
        }

        false
    }

    let mut used = vec![false; terms.len()];
    helper(patterns, terms, &mut used, binds)
}

fn commutative_match(patterns: &[Pattern], terms: &[Expr], binds: &mut Bindings, is_add: bool) -> bool {
    if patterns.len() == 1 {
        let whole = if is_add {
            Expr::add(terms.to_vec())
        } else {
            Expr::mul(terms.to_vec())
        };
        return match_pattern(&patterns[0], &whole, binds);
    }

    if patterns.len() == 2 {
        for idx in 0..terms.len() {
            let snapshot = binds.clone();
            if !match_pattern(&patterns[0], &terms[idx], binds) {
                restore_binds(binds, &snapshot);
                continue;
            }

            let remaining = terms
                .iter()
                .enumerate()
                .filter_map(|(j, term)| if j != idx { Some(term.clone()) } else { None })
                .collect::<Vec<_>>();
            let rest_expr = if is_add {
                Expr::add(remaining)
            } else {
                Expr::mul(remaining)
            };

            if match_pattern(&patterns[1], &rest_expr, binds) {
                return true;
            }
            restore_binds(binds, &snapshot);
        }
        return false;
    }

    match_sequence(patterns, terms, binds)
}

pub fn match_pattern(pattern: &Pattern, expr: &Expr, binds: &mut Bindings) -> bool {
    match pattern {
        Pattern::Slot(name) => match binds.get(name) {
            Some(bound) => bound == expr,
            None => {
                binds.insert(*name, expr.clone());
                true
            }
        },
        Pattern::Wildcard => true,
        Pattern::Exact(e) => e == expr,
        Pattern::Neg(inner) => match expr {
            Expr::Neg(e) => match_pattern(inner, e, binds),
            _ => false,
        },
        Pattern::Pow(bp, ep) => match expr {
            Expr::Pow(b, e) => {
                let snapshot = binds.clone();
                if match_pattern(bp, b, binds) && match_pattern(ep, e, binds) {
                    true
                } else {
                    restore_binds(binds, &snapshot);
                    false
                }
            }
            _ => false,
        },
        Pattern::Call(f, pats) => match expr {
            Expr::Call(g, args) if f == g && pats.len() == args.len() => {
                let snapshot = binds.clone();
                for (pat, arg) in pats.iter().zip(args.iter()) {
                    if !match_pattern(pat, arg, binds) {
                        restore_binds(binds, &snapshot);
                        return false;
                    }
                }
                true
            }
            _ => false,
        },
        Pattern::Add(pats) => match expr {
            Expr::Add(terms) => commutative_match(pats, terms, binds, true),
            _ if pats.len() == 1 => match_pattern(&pats[0], expr, binds),
            _ => false,
        },
        Pattern::Mul(pats) => match expr {
            Expr::Mul(terms) => commutative_match(pats, terms, binds, false),
            _ if pats.len() == 1 => match_pattern(&pats[0], expr, binds),
            _ => false,
        },
    }
}

pub fn substitute(template: &Expr, binds: &Bindings) -> Expr {
    match template {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => binds.get(s).cloned().unwrap_or(Expr::Sym(*s)),
        Expr::Add(terms) => Expr::add(terms.iter().map(|t| substitute(t, binds)).collect()),
        Expr::Mul(factors) => Expr::mul(factors.iter().map(|f| substitute(f, binds)).collect()),
        Expr::Pow(base, exp) => Expr::pow(substitute(base, binds), substitute(exp, binds)),
        Expr::Neg(e) => Expr::neg(substitute(e, binds)),
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(|a| substitute(a, binds)).collect()),
        Expr::Indexed(base, indices) => Expr::Indexed(Box::new(substitute(base, binds)), indices.clone()),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(substitute(val, binds)),
            Box::new(substitute(body, binds)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(|i| substitute(i, binds)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|cell| substitute(cell, binds)).collect())
                .collect(),
        ),
    }
}

pub fn apply_rule(rule: &RewriteRule, expr: &Expr, interner: &Interner) -> Option<Expr> {
    let mut binds = Bindings::new();
    if !match_pattern(&rule.pattern, expr, &mut binds) {
        return None;
    }
    if let Some(condition) = rule.condition {
        if !condition(&binds, interner) {
            return None;
        }
    }
    Some(substitute(&rule.replacement, &binds))
}

pub fn rewrite_once(rules: &[RewriteRule], expr: &Expr, interner: &Interner) -> Expr {
    let recursed = match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(
            terms.iter()
                .map(|t| rewrite_once(rules, t, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| rewrite_once(rules, f, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            rewrite_once(rules, base, interner),
            rewrite_once(rules, exp, interner),
        ),
        Expr::Neg(e) => Expr::neg(rewrite_once(rules, e, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| rewrite_once(rules, a, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(rewrite_once(rules, base, interner)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(rewrite_once(rules, val, interner)),
            Box::new(rewrite_once(rules, body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|i| rewrite_once(rules, i, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| rewrite_once(rules, cell, interner))
                        .collect()
                })
                .collect(),
        ),
    };

    for rule in rules {
        if let Some(rewritten) = apply_rule(rule, &recursed, interner) {
            return rewritten;
        }
    }

    recursed
}

pub fn rewrite_fixed_point(
    rules: &[RewriteRule],
    expr: &Expr,
    interner: &Interner,
    max_iter: usize,
) -> Expr {
    let mut current = expr.clone();
    for _ in 0..max_iter {
        let next = rewrite_once(rules, &current, interner);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_interner_and_syms() -> (Interner, lasso::Spur, lasso::Spur, lasso::Spur) {
        let interner = Interner::new();
        let a = interner.get_or_intern("a_");
        let b = interner.get_or_intern("b_");
        let x = interner.get_or_intern("x");
        (interner, a, b, x)
    }

    #[test]
    fn slot_binds_and_checks_consistency() {
        let (_interner, a, _, _) = make_interner_and_syms();
        let pat = Pattern::Slot(a);
        let expr = Expr::Int(42.into());
        let mut binds = Bindings::new();
        assert!(match_pattern(&pat, &expr, &mut binds));
        assert_eq!(binds[&a], Expr::Int(42.into()));
        assert!(match_pattern(&pat, &Expr::Int(42.into()), &mut binds));
        assert!(!match_pattern(&pat, &Expr::Int(99.into()), &mut binds));
    }

    #[test]
    fn exact_matches_literal() {
        let pat = Pattern::Exact(Expr::Int(5.into()));
        let mut binds = Bindings::new();
        assert!(match_pattern(&pat, &Expr::Int(5.into()), &mut binds));
        assert!(!match_pattern(&pat, &Expr::Int(6.into()), &mut binds));
    }

    #[test]
    fn add_commutative_match() {
        let (interner, a, b, x) = make_interner_and_syms();
        let y = interner.get_or_intern("y");
        let pat = Pattern::Add(vec![Pattern::Slot(a), Pattern::Slot(b)]);
        let expr = Expr::add(vec![Expr::Sym(x), Expr::Sym(y)]);
        let mut binds = Bindings::new();
        assert!(match_pattern(&pat, &expr, &mut binds));
        assert!(binds.contains_key(&a));
        assert!(binds.contains_key(&b));
    }

    #[test]
    fn substitute_replaces_slots() {
        let (_interner, a, b, _) = make_interner_and_syms();
        let template = Expr::add(vec![Expr::Sym(a), Expr::Sym(b)]);
        let mut binds = Bindings::new();
        binds.insert(a, Expr::Int(3.into()));
        binds.insert(b, Expr::Int(4.into()));
        let result = substitute(&template, &binds);
        assert_eq!(result, Expr::Int(7.into()));
    }

    #[test]
    fn rewrite_simple_rule() {
        let (interner, a, _, x) = make_interner_and_syms();
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");

        let rule = RewriteRule {
            name: "sin_to_cos".to_string(),
            pattern: Pattern::Call(sin_sym, vec![Pattern::Slot(a)]),
            replacement: Expr::Call(cos_sym, vec![Expr::Sym(a)]),
            condition: None,
        };

        let expr = Expr::Call(sin_sym, vec![Expr::Sym(x)]);
        let result = apply_rule(&rule, &expr, &interner);
        assert!(result.is_some());
        let result = result.unwrap();
        match result {
            Expr::Call(f, args) => {
                assert_eq!(f, cos_sym);
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expr::Sym(x));
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn fixed_point_terminates() {
        let interner = Interner::new();
        let expr = Expr::Int(42.into());
        let result = rewrite_fixed_point(&[], &expr, &interner, 100);
        assert_eq!(result, Expr::Int(42.into()));
    }
}
