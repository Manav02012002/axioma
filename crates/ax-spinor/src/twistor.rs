use crate::{Label, LabelMap, SpinorExpr, SpinorFactor};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::HashMap;

/// Momentum-twistor expressions built from dual-conformal four-brackets.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TwistorExpr {
    FourBracket(Label, Label, Label, Label),
    Product(Vec<TwistorTerm>),
    Sum(Vec<TwistorExpr>),
    Ratio(Box<TwistorExpr>, Box<TwistorExpr>),
    Numeric(BigRational),
    Power(Box<TwistorExpr>, i32),
    Neg(Box<TwistorExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TwistorTerm {
    pub coefficient: BigRational,
    pub factors: Vec<TwistorFactor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TwistorFactor {
    FourBracket(Label, Label, Label, Label),
    /// Fundamental adjacent spinor angle bracket <i i+1>; not expressible by four-brackets alone.
    FundamentalAngle(Label, Label),
    Power(Box<TwistorFactor>, i32),
}

impl TwistorExpr {
    pub fn four_bracket(i: Label, j: Label, k: Label, l: Label) -> Self {
        Self::FourBracket(i, j, k, l)
    }

    pub fn is_zero(&self) -> bool {
        match self {
            TwistorExpr::FourBracket(i, j, k, l) => has_repeated_labels(&[*i, *j, *k, *l]),
            TwistorExpr::Numeric(n) => n.is_zero(),
            TwistorExpr::Neg(inner) => inner.is_zero(),
            TwistorExpr::Product(terms) => terms.iter().any(TwistorTerm::is_zero),
            TwistorExpr::Sum(terms) => terms.iter().all(TwistorExpr::is_zero),
            TwistorExpr::Power(base, n) if *n > 0 => base.is_zero(),
            _ => false,
        }
    }
}

impl TwistorTerm {
    pub fn new(coefficient: BigRational, factors: Vec<TwistorFactor>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero() || self.factors.iter().any(TwistorFactor::is_zero)
    }
}

impl TwistorFactor {
    pub fn is_zero(&self) -> bool {
        match self {
            TwistorFactor::FourBracket(i, j, k, l) => has_repeated_labels(&[*i, *j, *k, *l]),
            TwistorFactor::Power(base, n) if *n > 0 => base.is_zero(),
            _ => false,
        }
    }
}

pub fn canonicalise_four_bracket(expr: &TwistorExpr) -> TwistorExpr {
    match expr {
        TwistorExpr::FourBracket(i, j, k, l) => canonical_four_expr([*i, *j, *k, *l]),
        TwistorExpr::Product(terms) => TwistorExpr::Product(
            terms
                .iter()
                .filter_map(|term| {
                    let mut coefficient = term.coefficient.clone();
                    let mut factors = Vec::new();
                    for factor in &term.factors {
                        let factor = canonical_four_factor(factor, &mut coefficient);
                        if factor.is_zero() {
                            return None;
                        }
                        factors.push(factor);
                    }
                    if coefficient.is_zero() {
                        None
                    } else {
                        Some(TwistorTerm {
                            coefficient,
                            factors,
                        })
                    }
                })
                .collect(),
        ),
        TwistorExpr::Sum(terms) => {
            TwistorExpr::Sum(terms.iter().map(canonicalise_four_bracket).collect())
        }
        TwistorExpr::Ratio(num, den) => TwistorExpr::Ratio(
            Box::new(canonicalise_four_bracket(num)),
            Box::new(canonicalise_four_bracket(den)),
        ),
        TwistorExpr::Power(base, n) => {
            TwistorExpr::Power(Box::new(canonicalise_four_bracket(base)), *n)
        }
        TwistorExpr::Neg(inner) => neg_twistor(canonicalise_four_bracket(inner)),
        _ => expr.clone(),
    }
}

pub fn spinor_to_twistor_mandelstam(i: Label, j: Label) -> TwistorExpr {
    let n = i.0.max(j.0).saturating_add(2).max(4);
    mandelstam_to_four_brackets(&[i, j], n)
}

pub fn mandelstam_to_four_brackets(particles: &[Label], n: u16) -> TwistorExpr {
    match particles {
        [i, j] if are_adjacent(*i, *j, n) => {
            let prev_i = prev_label(*i, n);
            let next_j = next_label(*j, n);
            TwistorExpr::FourBracket(prev_i, *i, *j, next_j)
        }
        [i, j] => TwistorExpr::Ratio(
            Box::new(TwistorExpr::FourBracket(
                prev_label(*i, n),
                *i,
                prev_label(*j, n),
                *j,
            )),
            Box::new(TwistorExpr::Product(vec![TwistorTerm::new(
                BigRational::one(),
                vec![
                    TwistorFactor::FundamentalAngle(prev_label(*i, n), *i),
                    TwistorFactor::FundamentalAngle(prev_label(*j, n), *j),
                ],
            )])),
        ),
        _ => {
            let mut terms = Vec::new();
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    terms.push(mandelstam_to_four_brackets(
                        &[particles[i], particles[j]],
                        n,
                    ));
                }
            }
            TwistorExpr::Sum(terms)
        }
    }
}

pub fn angle_bracket_to_twistor(i: Label, j: Label, n: u16) -> TwistorExpr {
    if are_adjacent(i, j, n) {
        TwistorExpr::Product(vec![TwistorTerm::new(
            BigRational::one(),
            vec![TwistorFactor::FundamentalAngle(i, j)],
        )])
    } else {
        TwistorExpr::Ratio(
            Box::new(TwistorExpr::FourBracket(
                prev_label(i, n),
                i,
                prev_label(j, n),
                j,
            )),
            Box::new(TwistorExpr::Product(vec![TwistorTerm::new(
                BigRational::one(),
                vec![
                    TwistorFactor::FundamentalAngle(prev_label(i, n), i),
                    TwistorFactor::FundamentalAngle(prev_label(j, n), j),
                ],
            )])),
        )
    }
}

pub fn apply_plucker(
    expr: &TwistorExpr,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
    e: Label,
    f: Label,
) -> TwistorExpr {
    match expr {
        TwistorExpr::Product(terms) => {
            let rewritten: Vec<TwistorExpr> = terms
                .iter()
                .map(|term| apply_plucker_to_term(term, a, b, c, d, e, f))
                .collect();
            multiply_twistor_exprs(rewritten)
        }
        TwistorExpr::Sum(terms) => TwistorExpr::Sum(
            terms
                .iter()
                .map(|term| apply_plucker(term, a, b, c, d, e, f))
                .collect(),
        ),
        TwistorExpr::Ratio(num, den) => TwistorExpr::Ratio(
            Box::new(apply_plucker(num, a, b, c, d, e, f)),
            Box::new(apply_plucker(den, a, b, c, d, e, f)),
        ),
        TwistorExpr::Power(base, n) => {
            TwistorExpr::Power(Box::new(apply_plucker(base, a, b, c, d, e, f)), *n)
        }
        TwistorExpr::Neg(inner) => neg_twistor(apply_plucker(inner, a, b, c, d, e, f)),
        _ => expr.clone(),
    }
}

pub fn twistor_simplify(expr: &TwistorExpr) -> TwistorExpr {
    let mut current = expr.clone();
    for _ in 0..5 {
        let next = simplify_twistor_structure(&canonicalise_four_bracket(&current));
        if next == current {
            return next;
        }
        current = next;
    }
    current
}

fn canonical_four_expr(labels: [Label; 4]) -> TwistorExpr {
    let (labels, sign) = canonicalize_labels(labels);
    if sign == 0 {
        TwistorExpr::Numeric(BigRational::zero())
    } else {
        let expr = TwistorExpr::FourBracket(labels[0], labels[1], labels[2], labels[3]);
        if sign < 0 {
            neg_twistor(expr)
        } else {
            expr
        }
    }
}

fn canonical_four_factor(factor: &TwistorFactor, coefficient: &mut BigRational) -> TwistorFactor {
    match factor {
        TwistorFactor::FourBracket(i, j, k, l) => {
            let (labels, sign) = canonicalize_labels([*i, *j, *k, *l]);
            if sign == 0 {
                *coefficient = BigRational::zero();
                factor.clone()
            } else {
                if sign < 0 {
                    *coefficient = -coefficient.clone();
                }
                TwistorFactor::FourBracket(labels[0], labels[1], labels[2], labels[3])
            }
        }
        TwistorFactor::Power(base, n) => {
            let mut inner_coeff = BigRational::one();
            let base = canonical_four_factor(base, &mut inner_coeff);
            if inner_coeff == -BigRational::one() && n % 2 != 0 {
                *coefficient = -coefficient.clone();
            }
            TwistorFactor::Power(Box::new(base), *n)
        }
        _ => factor.clone(),
    }
}

fn canonicalize_labels(mut labels: [Label; 4]) -> ([Label; 4], i32) {
    if has_repeated_labels(&labels) {
        return (labels, 0);
    }
    let mut sign = 1;
    for i in 0..labels.len() {
        for j in 0..labels.len() - 1 - i {
            if labels[j] > labels[j + 1] {
                labels.swap(j, j + 1);
                sign = -sign;
            }
        }
    }
    (labels, sign)
}

fn apply_plucker_to_term(
    term: &TwistorTerm,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
    e: Label,
    f: Label,
) -> TwistorExpr {
    for first in 0..term.factors.len() {
        let Some(sign_first) = match_four_factor(&term.factors[first], [a, b, c, e]) else {
            continue;
        };
        for second in 0..term.factors.len() {
            if first == second {
                continue;
            }
            let Some(sign_second) = match_four_factor(&term.factors[second], [a, b, d, f]) else {
                continue;
            };

            let mut rest = Vec::new();
            for (idx, factor) in term.factors.iter().enumerate() {
                if idx != first && idx != second {
                    rest.push(factor.clone());
                }
            }
            let sign = sign_first * sign_second;
            let coefficient = term.coefficient.clone() * BigRational::from_integer(sign.into());

            let mut first_term = rest.clone();
            first_term.push(TwistorFactor::FourBracket(a, b, c, f));
            first_term.push(TwistorFactor::FourBracket(a, b, d, e));

            let mut second_term = rest;
            second_term.push(TwistorFactor::FourBracket(a, b, c, d));
            second_term.push(TwistorFactor::FourBracket(a, b, e, f));

            return TwistorExpr::Sum(vec![
                TwistorExpr::Product(vec![TwistorTerm::new(coefficient.clone(), first_term)]),
                TwistorExpr::Product(vec![TwistorTerm::new(-coefficient, second_term)]),
            ]);
        }
    }
    TwistorExpr::Product(vec![term.clone()])
}

fn match_four_factor(factor: &TwistorFactor, target: [Label; 4]) -> Option<i32> {
    let TwistorFactor::FourBracket(i, j, k, l) = factor else {
        return None;
    };
    let current = [*i, *j, *k, *l];
    if current.iter().any(|label| !target.contains(label)) {
        return None;
    }
    let (_, current_sign) = canonicalize_labels(current);
    let (_, target_sign) = canonicalize_labels(target);
    if current_sign == 0 || target_sign == 0 {
        None
    } else {
        Some(current_sign * target_sign)
    }
}

fn simplify_twistor_structure(expr: &TwistorExpr) -> TwistorExpr {
    match expr {
        TwistorExpr::Product(terms) => {
            let mut out = Vec::new();
            for term in terms {
                let mut coefficient = term.coefficient.clone();
                let mut factors = Vec::new();
                for factor in &term.factors {
                    push_simplified_factor(factor, &mut coefficient, &mut factors);
                }
                combine_duplicate_factors(&mut factors);
                let term = TwistorTerm {
                    coefficient,
                    factors,
                };
                if !term.is_zero() {
                    out.push(term);
                }
            }
            if out.is_empty() {
                TwistorExpr::Numeric(BigRational::zero())
            } else {
                TwistorExpr::Product(out)
            }
        }
        TwistorExpr::Sum(terms) => combine_like_terms(
            terms
                .iter()
                .flat_map(|term| match simplify_twistor_structure(term) {
                    TwistorExpr::Sum(nested) => nested,
                    TwistorExpr::Numeric(n) if n.is_zero() => Vec::new(),
                    other if other.is_zero() => Vec::new(),
                    other => vec![other],
                })
                .collect(),
        ),
        TwistorExpr::Ratio(num, den) => TwistorExpr::Ratio(
            Box::new(simplify_twistor_structure(num)),
            Box::new(simplify_twistor_structure(den)),
        ),
        TwistorExpr::Power(base, n) => {
            if *n == 0 {
                TwistorExpr::Numeric(BigRational::one())
            } else if *n == 1 {
                simplify_twistor_structure(base)
            } else {
                TwistorExpr::Power(Box::new(simplify_twistor_structure(base)), *n)
            }
        }
        TwistorExpr::Neg(inner) => neg_twistor(simplify_twistor_structure(inner)),
        _ => expr.clone(),
    }
}

fn push_simplified_factor(
    factor: &TwistorFactor,
    coefficient: &mut BigRational,
    out: &mut Vec<TwistorFactor>,
) {
    match factor {
        TwistorFactor::Power(_, 0) => {}
        TwistorFactor::Power(inner, 1) => push_simplified_factor(inner, coefficient, out),
        TwistorFactor::FourBracket(i, j, k, l) if has_repeated_labels(&[*i, *j, *k, *l]) => {
            *coefficient = BigRational::zero();
        }
        _ => out.push(factor.clone()),
    }
}

fn combine_duplicate_factors(factors: &mut Vec<TwistorFactor>) {
    let mut i = 0;
    while i < factors.len() {
        let mut exponent = factor_exponent(&factors[i]);
        let base = factor_base(&factors[i]).clone();
        let mut j = i + 1;
        while j < factors.len() {
            if factor_base(&factors[j]) == &base {
                exponent += factor_exponent(&factors[j]);
                factors.remove(j);
            } else {
                j += 1;
            }
        }
        factors[i] = if exponent == 1 {
            base
        } else {
            TwistorFactor::Power(Box::new(base), exponent)
        };
        i += 1;
    }
}

fn factor_base(factor: &TwistorFactor) -> &TwistorFactor {
    match factor {
        TwistorFactor::Power(base, _) => base,
        _ => factor,
    }
}

fn factor_exponent(factor: &TwistorFactor) -> i32 {
    match factor {
        TwistorFactor::Power(_, n) => *n,
        _ => 1,
    }
}

fn combine_like_terms(terms: Vec<TwistorExpr>) -> TwistorExpr {
    let mut products: Vec<TwistorTerm> = Vec::new();
    let mut others = Vec::new();
    for term in terms {
        match term {
            TwistorExpr::Product(items) if items.len() == 1 => {
                let item = items[0].clone();
                if let Some(existing) = products.iter_mut().find(|p| p.factors == item.factors) {
                    existing.coefficient += item.coefficient;
                } else {
                    products.push(item);
                }
            }
            other => others.push(other),
        }
    }
    let mut out: Vec<TwistorExpr> = products
        .into_iter()
        .filter(|term| !term.coefficient.is_zero() && !term.is_zero())
        .map(|term| TwistorExpr::Product(vec![term]))
        .collect();
    out.extend(others);
    match out.len() {
        0 => TwistorExpr::Numeric(BigRational::zero()),
        1 => out.remove(0),
        _ => TwistorExpr::Sum(out),
    }
}

fn multiply_twistor_exprs(items: Vec<TwistorExpr>) -> TwistorExpr {
    let mut iter = items.into_iter();
    let Some(first) = iter.next() else {
        return TwistorExpr::Product(Vec::new());
    };
    iter.fold(first, multiply_twistor_pair)
}

fn multiply_twistor_pair(left: TwistorExpr, right: TwistorExpr) -> TwistorExpr {
    match (left, right) {
        (TwistorExpr::Sum(terms), rhs) => TwistorExpr::Sum(
            terms
                .into_iter()
                .map(|term| multiply_twistor_pair(term, rhs.clone()))
                .collect(),
        ),
        (lhs, TwistorExpr::Sum(terms)) => TwistorExpr::Sum(
            terms
                .into_iter()
                .map(|term| multiply_twistor_pair(lhs.clone(), term))
                .collect(),
        ),
        (TwistorExpr::Product(mut lhs), TwistorExpr::Product(mut rhs)) => {
            lhs.append(&mut rhs);
            TwistorExpr::Product(lhs)
        }
        (TwistorExpr::Neg(lhs), rhs) => neg_twistor(multiply_twistor_pair(*lhs, rhs)),
        (lhs, TwistorExpr::Neg(rhs)) => neg_twistor(multiply_twistor_pair(lhs, *rhs)),
        (lhs, rhs) => match (expr_to_factor(lhs), expr_to_factor(rhs)) {
            (Some(lhs), Some(rhs)) => {
                TwistorExpr::Product(vec![TwistorTerm::new(BigRational::one(), vec![lhs, rhs])])
            }
            (Some(factor), None) | (None, Some(factor)) => {
                TwistorExpr::Product(vec![TwistorTerm::new(BigRational::one(), vec![factor])])
            }
            (None, None) => TwistorExpr::Product(Vec::new()),
        },
    }
}

fn expr_to_factor(expr: TwistorExpr) -> Option<TwistorFactor> {
    match expr {
        TwistorExpr::FourBracket(i, j, k, l) => Some(TwistorFactor::FourBracket(i, j, k, l)),
        TwistorExpr::Power(base, n) => {
            expr_to_factor(*base).map(|factor| TwistorFactor::Power(Box::new(factor), n))
        }
        _ => None,
    }
}

fn neg_twistor(expr: TwistorExpr) -> TwistorExpr {
    match expr {
        TwistorExpr::Neg(inner) => *inner,
        TwistorExpr::Numeric(n) => TwistorExpr::Numeric(-n),
        TwistorExpr::Product(mut terms) if terms.len() == 1 => {
            terms[0].coefficient = -terms[0].coefficient.clone();
            TwistorExpr::Product(terms)
        }
        other => TwistorExpr::Neg(Box::new(other)),
    }
}

fn has_repeated_labels(labels: &[Label]) -> bool {
    let mut seen = HashMap::new();
    for label in labels {
        if seen.insert(*label, ()).is_some() {
            return true;
        }
    }
    false
}

fn prev_label(label: Label, n: u16) -> Label {
    assert!(n > 0, "number of particles must be positive");
    Label::new((label.0 + n - 1) % n)
}

fn next_label(label: Label, n: u16) -> Label {
    assert!(n > 0, "number of particles must be positive");
    Label::new((label.0 + 1) % n)
}

fn are_adjacent(i: Label, j: Label, n: u16) -> bool {
    next_label(i, n) == j || next_label(j, n) == i
}

#[allow(dead_code)]
fn _uses_requested_imports(_: Option<SpinorExpr>, _: Option<SpinorFactor>, _: Option<LabelMap>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bracket_antisymmetry_canonicalises() {
        let one = Label::new(1);
        let two = Label::new(2);
        let three = Label::new(3);
        let four = Label::new(4);

        assert_eq!(
            canonicalise_four_bracket(&TwistorExpr::FourBracket(two, one, three, four)),
            TwistorExpr::Neg(Box::new(TwistorExpr::FourBracket(one, two, three, four)))
        );
        assert_eq!(
            canonicalise_four_bracket(&TwistorExpr::FourBracket(one, one, three, four)),
            TwistorExpr::Numeric(BigRational::zero())
        );
    }

    #[test]
    fn angle_bracket_adjacent_stays_fundamental() {
        assert_eq!(
            angle_bracket_to_twistor(Label::new(1), Label::new(2), 5),
            TwistorExpr::Product(vec![TwistorTerm::new(
                BigRational::one(),
                vec![TwistorFactor::FundamentalAngle(
                    Label::new(1),
                    Label::new(2)
                )]
            )])
        );
    }

    #[test]
    fn plucker_rewrites_matching_product() {
        let a = Label::new(1);
        let b = Label::new(2);
        let c = Label::new(3);
        let d = Label::new(4);
        let e = Label::new(5);
        let f = Label::new(6);
        let expr = TwistorExpr::Product(vec![TwistorTerm::new(
            BigRational::one(),
            vec![
                TwistorFactor::FourBracket(a, b, c, e),
                TwistorFactor::FourBracket(a, b, d, f),
            ],
        )]);

        assert_eq!(
            apply_plucker(&expr, a, b, c, d, e, f),
            TwistorExpr::Sum(vec![
                TwistorExpr::Product(vec![TwistorTerm::new(
                    BigRational::one(),
                    vec![
                        TwistorFactor::FourBracket(a, b, c, f),
                        TwistorFactor::FourBracket(a, b, d, e),
                    ]
                )]),
                TwistorExpr::Product(vec![TwistorTerm::new(
                    -BigRational::one(),
                    vec![
                        TwistorFactor::FourBracket(a, b, c, d),
                        TwistorFactor::FourBracket(a, b, e, f),
                    ]
                )]),
            ])
        );
    }

    #[test]
    fn simplify_combines_like_four_bracket_terms() {
        let one = Label::new(1);
        let two = Label::new(2);
        let three = Label::new(3);
        let four = Label::new(4);
        let term = TwistorTerm::new(
            BigRational::one(),
            vec![TwistorFactor::FourBracket(one, two, three, four)],
        );
        let expr = TwistorExpr::Sum(vec![
            TwistorExpr::Product(vec![term.clone()]),
            TwistorExpr::Product(vec![term]),
        ]);

        assert_eq!(
            twistor_simplify(&expr),
            TwistorExpr::Product(vec![TwistorTerm::new(
                BigRational::from_integer(2.into()),
                vec![TwistorFactor::FourBracket(one, two, three, four)]
            )])
        );
    }
}
