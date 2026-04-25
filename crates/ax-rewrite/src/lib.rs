#![forbid(unsafe_code)]

use anyhow::Context;
use ax_ir::{Expr, Interner};
use ax_tensor::PropertyLookup;
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
    pub trust_level: ax_ir::TrustLevel,
}

#[derive(Clone, Debug)]
pub struct RewriteStep {
    pub rule_name: String,
    pub trust_level: ax_ir::TrustLevel,
    pub before: Expr,
    pub after: Expr,
    pub bindings: Bindings,
}

#[derive(Clone, Debug, Default)]
pub struct RewriteTrace {
    pub steps: Vec<RewriteStep>,
}

impl RewriteTrace {
    pub fn overall_trust(&self) -> ax_ir::TrustLevel {
        self.steps
            .iter()
            .map(|s| s.trust_level)
            .min_by_key(|t| match t {
                ax_ir::TrustLevel::Exact => 0,
                ax_ir::TrustLevel::UnderAssumptions => 1,
                ax_ir::TrustLevel::NumericallyChecked => 2,
                ax_ir::TrustLevel::Heuristic => 3,
                ax_ir::TrustLevel::Unverified => 4,
            })
            .unwrap_or(ax_ir::TrustLevel::Exact)
    }
}

fn rewrite_expr_head_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Call(sym, _) => Some(*sym),
        Expr::Indexed(base, _) => rewrite_expr_head_symbol(base),
        _ => None,
    }
}

fn rewrite_mode_metadata_of_factor(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<ax_ir::ModeMetadata> {
    let direct = rewrite_expr_head_symbol(expr).and_then(|sym| {
        props
            .get_properties(sym)
            .into_iter()
            .find_map(|prop| match prop {
                ax_ir::TensorProperty::ModeMeta(metadata) => Some(metadata),
                _ => None,
            })
    });
    if direct.is_some() {
        return direct;
    }

    match expr {
        Expr::Call(_, args) if args.len() == 1 => {
            rewrite_expr_head_symbol(&args[0]).and_then(|sym| {
                props
                    .get_properties(sym)
                    .into_iter()
                    .find_map(|prop| match prop {
                        ax_ir::TensorProperty::ModeMeta(metadata) => Some(metadata),
                        _ => None,
                    })
            })
        }
        _ => None,
    }
}

fn rewrite_has_property_kind(
    expr: &Expr,
    props: &dyn PropertyLookup,
    kind: &ax_ir::TensorProperty,
) -> bool {
    rewrite_expr_head_symbol(expr).is_some_and(|sym| props.has_property_kind(sym, kind))
}

/// Return the graded sign associated with swapping two declared factors.
///
/// The helper is intentionally sign-only: it reports `-1` for metadata-proven
/// odd/fermionic swaps and `+1` otherwise, without inventing commutator,
/// anticommutator, delta, or contraction terms.
pub fn graded_swap_sign(left: &Expr, right: &Expr, props: &dyn PropertyLookup) -> Expr {
    if let (Some(lhs), Some(rhs)) = (
        rewrite_mode_metadata_of_factor(left, props),
        rewrite_mode_metadata_of_factor(right, props),
    ) {
        return if lhs.is_fermionic() && rhs.is_fermionic() {
            Expr::neg(Expr::one())
        } else {
            Expr::one()
        };
    }

    if let (Some(lhs), Some(rhs)) = (
        rewrite_expr_head_symbol(left),
        rewrite_expr_head_symbol(right),
    ) {
        if props.pair_commuting_behaviour(lhs, rhs) == Some(-1) {
            return Expr::neg(Expr::one());
        }
    }

    let left_odd = rewrite_has_property_kind(left, props, &ax_ir::TensorProperty::AntiCommuting)
        || rewrite_has_property_kind(left, props, &ax_ir::TensorProperty::MajoranaSpinor)
        || rewrite_has_property_kind(left, props, &ax_ir::TensorProperty::WeylSpinor);
    let right_odd = rewrite_has_property_kind(right, props, &ax_ir::TensorProperty::AntiCommuting)
        || rewrite_has_property_kind(right, props, &ax_ir::TensorProperty::MajoranaSpinor)
        || rewrite_has_property_kind(right, props, &ax_ir::TensorProperty::WeylSpinor);

    if left_odd && right_odd {
        Expr::neg(Expr::one())
    } else {
        Expr::one()
    }
}

fn empty_props() -> &'static HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> {
    static EMPTY: std::sync::OnceLock<HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>> =
        std::sync::OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

fn empty_index_to_family() -> &'static HashMap<lasso::Spur, lasso::Spur> {
    static EMPTY: std::sync::OnceLock<HashMap<lasso::Spur, lasso::Spur>> =
        std::sync::OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

fn try_apply_rule_compare(
    rule: &RewriteRule,
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Option<(Expr, Bindings)> {
    let pattern = pattern_to_expr_with_wildcard(&rule.pattern, interner.get_or_intern("_"));
    if rule.condition.is_none() {
        let sub_rule =
            ax_compare::SubstitutionRule::new(pattern.clone(), rule.replacement.clone(), None);
        let rewritten = ax_compare::substitute_full(expr, &sub_rule, properties, interner, true)?;
        let bindings = if rewritten != *expr {
            ax_compare::pattern_match(&pattern, expr, properties, index_to_family, interner)
                .map(|m| match_map_to_bindings(&m))
                .unwrap_or_default()
        } else {
            Bindings::new()
        };
        return Some((rewritten, bindings));
    }
    let match_map =
        ax_compare::pattern_match(&pattern, expr, properties, index_to_family, interner)?;
    let bindings = match_map_to_bindings(&match_map);
    if let Some(condition) = rule.condition {
        if !condition(&bindings, interner) {
            return None;
        }
    }
    Some((
        ax_compare::apply_match_map(&rule.replacement, &match_map, interner),
        bindings,
    ))
}

pub fn apply_rule(rule: &RewriteRule, expr: &Expr, interner: &Interner) -> Option<Expr> {
    try_apply_rule_compare(rule, expr, empty_props(), empty_index_to_family(), interner)
        .map(|(rewritten, _)| rewritten)
}

pub fn apply_rule_traced(
    rule: &RewriteRule,
    expr: &Expr,
    interner: &Interner,
    trace: &mut RewriteTrace,
) -> Option<Expr> {
    let (after, binds) =
        try_apply_rule_compare(rule, expr, empty_props(), empty_index_to_family(), interner)?;
    trace.steps.push(RewriteStep {
        rule_name: rule.name.clone(),
        trust_level: rule.trust_level,
        before: expr.clone(),
        after: after.clone(),
        bindings: binds,
    });
    Some(after)
}

pub fn rewrite_once(rules: &[RewriteRule], expr: &Expr, interner: &Interner) -> Expr {
    let recursed = match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rewrite_once(rules, re, interner)),
            Box::new(rewrite_once(rules, im, interner)),
        ),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
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
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(rewrite_once(rules, body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rewrite_once(rules, lhs, interner)),
            Box::new(rewrite_once(rules, rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (rewrite_once(rules, value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(rewrite_once(rules, base, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(rewrite_once(rules, inner, interner)), *rel)
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

pub fn rewrite_once_traced(
    rules: &[RewriteRule],
    expr: &Expr,
    interner: &Interner,
    trace: &mut RewriteTrace,
) -> Expr {
    let recursed = match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rewrite_once_traced(rules, re, interner, trace)),
            Box::new(rewrite_once_traced(rules, im, interner, trace)),
        ),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| rewrite_once_traced(rules, t, interner, trace))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| rewrite_once_traced(rules, f, interner, trace))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            rewrite_once_traced(rules, base, interner, trace),
            rewrite_once_traced(rules, exp, interner, trace),
        ),
        Expr::Neg(e) => Expr::neg(rewrite_once_traced(rules, e, interner, trace)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| rewrite_once_traced(rules, a, interner, trace))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(rewrite_once_traced(rules, body, interner, trace)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rewrite_once_traced(rules, lhs, interner, trace)),
            Box::new(rewrite_once_traced(rules, rhs, interner, trace)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        rewrite_once_traced(rules, value, interner, trace),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(rewrite_once_traced(rules, base, interner, trace)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(rewrite_once_traced(rules, inner, interner, trace)),
            *rel,
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(rewrite_once_traced(rules, val, interner, trace)),
            Box::new(rewrite_once_traced(rules, body, interner, trace)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|i| rewrite_once_traced(rules, i, interner, trace))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| rewrite_once_traced(rules, cell, interner, trace))
                        .collect()
                })
                .collect(),
        ),
    };

    for rule in rules {
        if let Some(rewritten) = apply_rule_traced(rule, &recursed, interner, trace) {
            return rewritten;
        }
    }

    recursed
}

/// Returns whether a rewrite preserves tensor symmetry on a purely syntactic path.
///
/// This only checks whether both expressions are indexed tensor symbols with the
/// same base symbol and the same arity. It is not a semantic proof that the
/// rewrite preserves the full structured symmetry object.
pub fn rewrite_preserves_tensor_symmetry(before: &ax_ir::Expr, after: &ax_ir::Expr) -> bool {
    match (before, after) {
        (Expr::Indexed(before_base, before_indices), Expr::Indexed(after_base, after_indices)) => {
            matches!(
                (before_base.as_ref(), after_base.as_ref()),
                (Expr::Sym(before_sym), Expr::Sym(after_sym))
                    if before_sym == after_sym && before_indices.len() == after_indices.len()
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tensor_symmetry_tests {
    use super::*;

    #[test]
    fn same_indexed_symbol_and_arity_preserves_tensor_symmetry() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let before = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let after = before.clone();
        assert!(rewrite_preserves_tensor_symmetry(&before, &after));
    }

    #[test]
    fn different_symbol_does_not_preserve_tensor_symmetry() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let u = interner.get_or_intern("U");
        let a = interner.get_or_intern("a");
        let before = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![ax_ir::Index {
                name: a,
                variance: ax_ir::Variance::Down,
                index_type: None,
            }],
        );
        let after = Expr::Indexed(
            Box::new(Expr::Sym(u)),
            vec![ax_ir::Index {
                name: a,
                variance: ax_ir::Variance::Down,
                index_type: None,
            }],
        );
        assert!(!rewrite_preserves_tensor_symmetry(&before, &after));
    }
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

pub fn rewrite_fixed_point_traced(
    rules: &[RewriteRule],
    expr: &Expr,
    interner: &Interner,
    max_iter: usize,
    trace: &mut RewriteTrace,
) -> Expr {
    let mut current = expr.clone();
    for _ in 0..max_iter {
        let next = rewrite_once_traced(rules, &current, interner, trace);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

pub fn apply_rule_with_compare(
    rule: &RewriteRule,
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    index_to_family: &std::collections::HashMap<lasso::Spur, lasso::Spur>,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    try_apply_rule_compare(rule, expr, properties, index_to_family, interner)
        .map(|(rewritten, _)| rewritten)
}

pub fn rewrite_once_with_compare(
    rules: &[RewriteRule],
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    index_to_family: &std::collections::HashMap<lasso::Spur, lasso::Spur>,
    interner: &ax_ir::Interner,
) -> Expr {
    let recursed = match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rewrite_once_with_compare(
                rules,
                re,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(rewrite_once_with_compare(
                rules,
                im,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    rewrite_once_with_compare(rules, term, properties, index_to_family, interner)
                })
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    rewrite_once_with_compare(rules, factor, properties, index_to_family, interner)
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            rewrite_once_with_compare(rules, base, properties, index_to_family, interner),
            rewrite_once_with_compare(rules, exp, properties, index_to_family, interner),
        ),
        Expr::Neg(inner) => Expr::neg(rewrite_once_with_compare(
            rules,
            inner,
            properties,
            index_to_family,
            interner,
        )),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| {
                    rewrite_once_with_compare(rules, arg, properties, index_to_family, interner)
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(rewrite_once_with_compare(
                rules,
                body,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rewrite_once_with_compare(
                rules,
                lhs,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(rewrite_once_with_compare(
                rules,
                rhs,
                properties,
                index_to_family,
                interner,
            )),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        rewrite_once_with_compare(
                            rules,
                            value,
                            properties,
                            index_to_family,
                            interner,
                        ),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(rewrite_once_with_compare(
                rules,
                base,
                properties,
                index_to_family,
                interner,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(rewrite_once_with_compare(
                rules,
                inner,
                properties,
                index_to_family,
                interner,
            )),
            *rel,
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(rewrite_once_with_compare(
                rules,
                val,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(rewrite_once_with_compare(
                rules,
                body,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| {
                    rewrite_once_with_compare(rules, item, properties, index_to_family, interner)
                })
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| {
                            rewrite_once_with_compare(
                                rules,
                                cell,
                                properties,
                                index_to_family,
                                interner,
                            )
                        })
                        .collect()
                })
                .collect(),
        ),
    };

    for rule in rules {
        if let Some(rewritten) =
            apply_rule_with_compare(rule, &recursed, properties, index_to_family, interner)
        {
            return rewritten;
        }
    }

    recursed
}

pub fn rewrite_fixed_point_with_compare(
    rules: &[RewriteRule],
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    index_to_family: &std::collections::HashMap<lasso::Spur, lasso::Spur>,
    interner: &ax_ir::Interner,
    max_iter: usize,
) -> Expr {
    let mut current = expr.clone();
    for _ in 0..max_iter {
        let next =
            rewrite_once_with_compare(rules, &current, properties, index_to_family, interner);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

pub fn rewrite_using_tensor_identities(
    expr: &ax_ir::Expr,
    identities: &ax_ir::TensorIdentitySet,
) -> anyhow::Result<Option<ax_ir::Expr>> {
    match ax_tensor::reduce_indexed_factor_modulo_identities(expr, identities) {
        Ok(result) => Ok(result.map(|reduction| reduction.expr)),
        Err(
            ax_tensor::MultitermError::UnsupportedExpr
            | ax_tensor::MultitermError::NoApplicableIdentity,
        ) => Ok(None),
        Err(err) => Err(err).context("failed to rewrite expression using tensor identities"),
    }
}

pub fn pattern_to_expr(pattern: &Pattern) -> Expr {
    pattern_to_expr_with_wildcard(pattern, lasso::Spur::default())
}

fn pattern_to_expr_with_wildcard(pattern: &Pattern, wildcard: lasso::Spur) -> Expr {
    match pattern {
        Pattern::Slot(sym) => Expr::Sym(*sym),
        Pattern::Wildcard => Expr::Sym(wildcard),
        Pattern::Exact(expr) => expr.clone(),
        Pattern::Add(items) => Expr::add(
            items
                .iter()
                .map(|item| pattern_to_expr_with_wildcard(item, wildcard))
                .collect(),
        ),
        Pattern::Mul(items) => Expr::mul(
            items
                .iter()
                .map(|item| pattern_to_expr_with_wildcard(item, wildcard))
                .collect(),
        ),
        Pattern::Pow(base, exp) => Expr::pow(
            pattern_to_expr_with_wildcard(base, wildcard),
            pattern_to_expr_with_wildcard(exp, wildcard),
        ),
        Pattern::Neg(inner) => Expr::neg(pattern_to_expr_with_wildcard(inner, wildcard)),
        Pattern::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| pattern_to_expr_with_wildcard(arg, wildcard))
                .collect(),
        ),
    }
}

fn match_map_to_bindings(map: &ax_compare::MatchMap) -> Bindings {
    let mut out = Bindings::new();
    for (slot, expr) in &map.wildcard_map {
        out.insert(*slot, expr.clone());
    }
    for (pattern_sym, target_sym) in &map.symbol_map {
        out.entry(*pattern_sym).or_insert(Expr::Sym(*target_sym));
    }
    out
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
    fn pattern_to_expr_converts_slots_and_structure() {
        let (interner, a, b, x) = make_interner_and_syms();
        let sin_sym = interner.get_or_intern("sin");
        let pattern = Pattern::Add(vec![
            Pattern::Call(sin_sym, vec![Pattern::Slot(a)]),
            Pattern::Mul(vec![Pattern::Exact(Expr::Sym(x)), Pattern::Slot(b)]),
        ]);
        let expr = pattern_to_expr(&pattern);
        assert_eq!(
            expr,
            Expr::add(vec![
                Expr::Call(sin_sym, vec![Expr::Sym(a)]),
                Expr::mul(vec![Expr::Sym(x), Expr::Sym(b)]),
            ])
        );
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
            trust_level: ax_ir::TrustLevel::Exact,
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

    #[test]
    fn traced_rewrite_records_steps() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a_");
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");
        let x = interner.get_or_intern("x");

        let rule = RewriteRule {
            name: "test".into(),
            pattern: Pattern::Call(sin_sym, vec![Pattern::Slot(a)]),
            replacement: Expr::Call(cos_sym, vec![Expr::Sym(a)]),
            condition: None,
            trust_level: ax_ir::TrustLevel::Heuristic,
        };

        let expr = Expr::Call(sin_sym, vec![Expr::Sym(x)]);
        let mut trace = RewriteTrace::default();
        let result = rewrite_once_traced(&[rule], &expr, &interner, &mut trace);
        assert_eq!(result, Expr::Call(cos_sym, vec![Expr::Sym(x)]));
        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.steps[0].trust_level, ax_ir::TrustLevel::Heuristic);
        assert_eq!(trace.overall_trust(), ax_ir::TrustLevel::Heuristic);
    }

    #[test]
    fn apply_rule_with_compare_matches_partial_products() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let d = interner.get_or_intern("D");
        let rule = RewriteRule {
            name: "partial_mul".into(),
            pattern: Pattern::Mul(vec![
                Pattern::Exact(Expr::Sym(a)),
                Pattern::Exact(Expr::Sym(b)),
            ]),
            replacement: Expr::Sym(c),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        };
        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(d)]);
        let props: std::collections::HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> =
            std::collections::HashMap::new();
        let index_to_family = std::collections::HashMap::new();
        let result = apply_rule_with_compare(&rule, &expr, &props, &index_to_family, &interner);
        assert_eq!(result, Some(Expr::mul(vec![Expr::Sym(c), Expr::Sym(d)])));
    }

    #[test]
    fn rewrite_once_preserves_explicit_grouping() {
        let interner = ax_ir::Interner::new();
        let slot = interner.get_or_intern("a_");
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");
        let x = interner.get_or_intern("x");

        let rule = RewriteRule {
            name: "sin_to_cos".into(),
            pattern: Pattern::Call(sin_sym, vec![Pattern::Slot(slot)]),
            replacement: Expr::Call(cos_sym, vec![Expr::Sym(slot)]),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        };

        let expr = Expr::group(Expr::Call(sin_sym, vec![Expr::Sym(x)]));
        let result = rewrite_once(&[rule], &expr, &interner);
        assert_eq!(result, Expr::group(Expr::Call(cos_sym, vec![Expr::Sym(x)])));
    }

    #[test]
    fn rewrite_using_tensor_identities_returns_some_for_applicable_factor() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let idx = |name| ax_ir::Index {
            name,
            variance: ax_ir::Variance::Down,
            index_type: None,
        };
        let expr = Expr::Indexed(Box::new(Expr::Sym(t)), vec![idx(c), idx(a), idx(b)]);
        let identities = ax_ir::TensorIdentitySet {
            multiterm: vec![ax_ir::TensorMultitermIdentity::CyclicSum {
                slots: vec![0, 1, 2],
            }],
        };
        assert!(rewrite_using_tensor_identities(&expr, &identities)
            .ok()
            .flatten()
            .is_some());
        assert_eq!(
            rewrite_using_tensor_identities(&Expr::Sym(t), &identities)
                .ok()
                .flatten(),
            None
        );
    }

    #[test]
    fn graded_swap_sign_fermionic_mode_metadata_is_negative() {
        let interner = Interner::new();
        let c0 = interner.get_or_intern("c0");
        let c1 = interner.get_or_intern("c1");
        let annihilation = interner.get_or_intern("annihilation");
        let props = HashMap::from([
            (
                c0,
                vec![ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
                    statistics: ax_ir::ModeStatistics::Fermionic,
                    subsystem: None,
                    mode_index: 0,
                    label: None,
                })],
            ),
            (
                c1,
                vec![ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
                    statistics: ax_ir::ModeStatistics::Fermionic,
                    subsystem: None,
                    mode_index: 1,
                    label: None,
                })],
            ),
        ]);

        assert_eq!(
            graded_swap_sign(
                &Expr::Call(annihilation, vec![Expr::Sym(c0)]),
                &Expr::Call(annihilation, vec![Expr::Sym(c1)]),
                &props,
            ),
            Expr::neg(Expr::one())
        );
    }

    #[test]
    fn graded_swap_sign_bosonic_mode_metadata_is_positive() {
        let interner = Interner::new();
        let a0 = interner.get_or_intern("a0");
        let a1 = interner.get_or_intern("a1");
        let creation = interner.get_or_intern("creation");
        let props = HashMap::from([
            (
                a0,
                vec![ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
                    statistics: ax_ir::ModeStatistics::Bosonic,
                    subsystem: None,
                    mode_index: 0,
                    label: None,
                })],
            ),
            (
                a1,
                vec![ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
                    statistics: ax_ir::ModeStatistics::Bosonic,
                    subsystem: None,
                    mode_index: 1,
                    label: None,
                })],
            ),
        ]);

        assert_eq!(
            graded_swap_sign(
                &Expr::Call(creation, vec![Expr::Sym(a0)]),
                &Expr::Call(creation, vec![Expr::Sym(a1)]),
                &props,
            ),
            Expr::one()
        );
    }
}
