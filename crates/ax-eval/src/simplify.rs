use crate::{eval, Env};
use ax_ir::Expr;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeSet;

fn square(expr: Expr) -> Expr {
    Expr::pow(expr, Expr::Int(2.into()))
}

fn slot_expr(slot: lasso::Spur) -> Expr {
    Expr::Sym(slot)
}

fn slot_pat(slot: lasso::Spur) -> ax_rewrite::Pattern {
    ax_rewrite::Pattern::Slot(slot)
}

fn unary_pat(f: lasso::Spur, arg: ax_rewrite::Pattern) -> ax_rewrite::Pattern {
    ax_rewrite::Pattern::Call(f, vec![arg])
}

fn pow_pat(base: ax_rewrite::Pattern, n: i64) -> ax_rewrite::Pattern {
    ax_rewrite::Pattern::Pow(
        Box::new(base),
        Box::new(ax_rewrite::Pattern::Exact(Expr::Int(n.into()))),
    )
}

fn add_pat(parts: Vec<ax_rewrite::Pattern>) -> ax_rewrite::Pattern {
    ax_rewrite::Pattern::Add(parts)
}

fn mul_pat(parts: Vec<ax_rewrite::Pattern>) -> ax_rewrite::Pattern {
    ax_rewrite::Pattern::Mul(parts)
}

fn is_nontrivial_slot(
    bindings: &ax_rewrite::Bindings,
    slot: lasso::Spur,
    _interner: &ax_ir::Interner,
) -> bool {
    matches!(
        bindings.get(&slot),
        Some(Expr::Add(_))
            | Some(Expr::Mul(_))
            | Some(Expr::Pow(_, _))
            | Some(Expr::Call(_, _))
            | Some(Expr::Neg(_))
            | Some(Expr::Complex(_, _))
    )
}

fn is_nontrivial_x_slot(
    bindings: &ax_rewrite::Bindings,
    interner: &ax_ir::Interner,
) -> bool {
    let slot = interner.get_or_intern("x_");
    is_nontrivial_slot(bindings, slot, interner)
}

fn build_trig_rules(interner: &ax_ir::Interner) -> Vec<ax_rewrite::RewriteRule> {
    let x_slot = interner.get_or_intern("x_");
    let sin_sym = interner.get_or_intern("sin");
    let cos_sym = interner.get_or_intern("cos");
    let tan_sym = interner.get_or_intern("tan");
    let sec_sym = interner.get_or_intern("sec");

    let sin_x_pat = unary_pat(sin_sym, slot_pat(x_slot));
    let cos_x_pat = unary_pat(cos_sym, slot_pat(x_slot));
    let tan_x_pat = unary_pat(tan_sym, slot_pat(x_slot));
    let sec_x_pat = unary_pat(sec_sym, slot_pat(x_slot));

    let sin_x = Expr::Call(sin_sym, vec![slot_expr(x_slot)]);
    let cos_x = Expr::Call(cos_sym, vec![slot_expr(x_slot)]);
    let tan_x = Expr::Call(tan_sym, vec![slot_expr(x_slot)]);
    let sec_x = Expr::Call(sec_sym, vec![slot_expr(x_slot)]);
    let two_x = Expr::mul(vec![Expr::Int(2.into()), slot_expr(x_slot)]);
    let sin_2x = Expr::Call(sin_sym, vec![two_x.clone()]);
    vec![
        ax_rewrite::RewriteRule {
            name: "pythag_sin_cos".into(),
            pattern: add_pat(vec![pow_pat(sin_x_pat.clone(), 2), pow_pat(cos_x_pat.clone(), 2)]),
            replacement: Expr::one(),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "pythag_1_minus_sin2".into(),
            pattern: add_pat(vec![
                ax_rewrite::Pattern::Exact(Expr::one()),
                ax_rewrite::Pattern::Neg(Box::new(pow_pat(sin_x_pat.clone(), 2))),
            ]),
            replacement: square(cos_x.clone()),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "pythag_1_minus_cos2".into(),
            pattern: add_pat(vec![
                ax_rewrite::Pattern::Exact(Expr::one()),
                ax_rewrite::Pattern::Neg(Box::new(pow_pat(cos_x_pat.clone(), 2))),
            ]),
            replacement: square(sin_x.clone()),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "tan2_plus_1".into(),
            pattern: add_pat(vec![
                pow_pat(tan_x_pat.clone(), 2),
                ax_rewrite::Pattern::Exact(Expr::one()),
            ]),
            replacement: square(sec_x.clone()),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "sec2_minus_1".into(),
            pattern: add_pat(vec![
                pow_pat(sec_x_pat.clone(), 2),
                ax_rewrite::Pattern::Neg(Box::new(ax_rewrite::Pattern::Exact(Expr::one()))),
            ]),
            replacement: square(tan_x.clone()),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "sin_double_expand".into(),
            pattern: unary_pat(
                sin_sym,
                mul_pat(vec![
                    ax_rewrite::Pattern::Exact(Expr::Int(2.into())),
                    slot_pat(x_slot),
                ]),
            ),
            replacement: Expr::mul(vec![Expr::Int(2.into()), sin_x.clone(), cos_x.clone()]),
            condition: Some(is_nontrivial_x_slot),
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "cos_double".into(),
            pattern: unary_pat(
                cos_sym,
                mul_pat(vec![
                    ax_rewrite::Pattern::Exact(Expr::Int(2.into())),
                    slot_pat(x_slot),
                ]),
            ),
            replacement: Expr::add(vec![square(cos_x.clone()), Expr::neg(square(sin_x.clone()))]),
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
        ax_rewrite::RewriteRule {
            name: "sin_double_collect".into(),
            pattern: mul_pat(vec![
                ax_rewrite::Pattern::Exact(Expr::Int(2.into())),
                sin_x_pat.clone(),
                cos_x_pat.clone(),
            ]),
            replacement: sin_2x,
            condition: None,
            trust_level: ax_ir::TrustLevel::Exact,
        },
    ]
}

fn numeric_coeff(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn decompose_term(term: &Expr) -> (BigRational, Expr) {
    match term {
        Expr::Mul(factors) if !factors.is_empty() => {
            if let Some(coeff) = numeric_coeff(&factors[0]) {
                let rest = factors[1..].to_vec();
                let base = if rest.is_empty() {
                    Expr::one()
                } else {
                    Expr::mul(rest)
                };
                (coeff, base)
            } else {
                (BigRational::one(), term.clone())
            }
        }
        Expr::Neg(inner) => (
            BigRational::from_integer((-1).into()),
            inner.as_ref().clone(),
        ),
        Expr::Int(n) => (BigRational::from_integer(n.clone()), Expr::one()),
        Expr::Rational(r) => (r.clone(), Expr::one()),
        _ => (BigRational::one(), term.clone()),
    }
}

fn factor_base_and_exp(expr: &Expr) -> (Expr, BigRational) {
    match expr {
        Expr::Pow(base, exp) => {
            if let Some(n) = numeric_coeff(exp) {
                ((*base.clone()), n)
            } else {
                (expr.clone(), BigRational::one())
            }
        }
        _ => (expr.clone(), BigRational::one()),
    }
}

fn factor_list(expr: &Expr) -> Vec<(Expr, BigRational)> {
    match expr {
        Expr::Mul(factors) => factors.iter().map(factor_base_and_exp).collect(),
        Expr::Int(_) | Expr::Rational(_) => Vec::new(),
        _ => vec![factor_base_and_exp(expr)],
    }
}

fn remove_common_factor(factors: &[(Expr, BigRational)], common: &[(Expr, BigRational)]) -> Expr {
    let mut remaining = factors.to_vec();

    for (common_base, common_exp) in common {
        if let Some((_, exp)) = remaining.iter_mut().find(|(base, _)| *base == *common_base) {
            *exp -= common_exp.clone();
        }
    }

    let rebuilt = remaining
        .into_iter()
        .filter_map(|(base, exp)| {
            if exp.is_zero() {
                None
            } else if exp.is_one() {
                Some(base)
            } else {
                Some(Expr::pow(base, Expr::Rational(exp)))
            }
        })
        .collect::<Vec<_>>();

    Expr::mul(rebuilt)
}

fn extract_common_factor(terms: &[Expr]) -> Option<(Expr, Vec<Expr>)> {
    if terms.len() < 2 {
        return None;
    }

    let mut common = factor_list(&terms[0]);
    if common.is_empty() {
        return None;
    }

    for term in &terms[1..] {
        let factors = factor_list(term);
        common.retain_mut(|(common_base, common_exp)| {
            if let Some((_, exp)) = factors.iter().find(|(base, _)| *base == *common_base) {
                if *exp < *common_exp {
                    *common_exp = exp.clone();
                }
                !common_exp.is_zero()
            } else {
                false
            }
        });

        if common.is_empty() {
            return None;
        }
    }

    if !common.iter().any(|(_, exp)| exp.is_negative()) {
        return None;
    }

    let common_expr = Expr::mul(
        common
            .iter()
            .map(|(base, exp)| {
                if exp.is_one() {
                    base.clone()
                } else {
                    Expr::pow(base.clone(), Expr::Rational(exp.clone()))
                }
            })
            .collect(),
    );

    if common_expr == Expr::one() {
        return None;
    }

    let remainders = terms
        .iter()
        .map(|term| remove_common_factor(&factor_list(term), &common))
        .collect::<Vec<_>>();

    Some((common_expr, remainders))
}

fn collect_flat_add(expr: &Expr) -> Expr {
    let Expr::Add(terms) = expr else {
        return expr.clone();
    };

    let mut groups: Vec<(Expr, BigRational)> = Vec::new();
    for term in terms {
        let (coeff, base) = decompose_term(term);
        if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
            *acc += coeff;
        } else {
            groups.push((base, coeff));
        }
    }

    Expr::add(
        groups
            .into_iter()
            .filter_map(|(base, coeff)| {
                if coeff.is_zero() {
                    return None;
                }

                let coeff_expr = if coeff.is_integer() {
                    Expr::Int(coeff.to_integer())
                } else {
                    Expr::Rational(coeff)
                };

                Some(if base == Expr::one() {
                    coeff_expr
                } else {
                    Expr::mul(vec![coeff_expr, base])
                })
            })
            .collect(),
    )
}

pub fn expand(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let _ = interner;
    match expr {
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand(re, interner)),
            Box::new(expand(im, interner)),
        ),
        Expr::Add(terms) => {
            let expanded = Expr::add(terms.iter().map(|t| expand(t, interner)).collect());
            collect_flat_add(&expanded)
        }
        Expr::Mul(factors) => {
            let expanded_factors = factors
                .iter()
                .map(|f| expand(f, interner))
                .collect::<Vec<_>>();

            if expanded_factors.len() > 6 {
                return Expr::mul(expanded_factors);
            }

            if let Some((idx, terms)) =
                expanded_factors.iter().enumerate().find_map(|(i, factor)| {
                    if let Expr::Add(terms) = factor {
                        Some((i, terms.clone()))
                    } else {
                        None
                    }
                })
            {
                let rest = expanded_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(i, factor)| if i != idx { Some(factor.clone()) } else { None })
                    .collect::<Vec<_>>();

                let distributed = terms
                    .into_iter()
                    .map(|term| {
                        let mut factors = Vec::with_capacity(rest.len() + 1);
                        factors.push(term);
                        factors.extend(rest.clone());
                        Expr::mul(factors)
                    })
                    .collect::<Vec<_>>();

                return expand(&Expr::add(distributed), interner);
            }

            Expr::mul(expanded_factors)
        }
        Expr::Pow(base, exp) => {
            let expanded_base = expand(base, interner);
            let expanded_exp = expand(exp, interner);
            if let (Expr::Add(terms), Expr::Int(n)) = (&expanded_base, &expanded_exp) {
                if *n > 1.into() {
                    if let Some(power) = n.to_u32() {
                        if (2..=8).contains(&power) && terms.len() * (power as usize) <= 12 {
                            let sum = Expr::Add(terms.clone());
                            let repeated = (0..power).map(|_| sum.clone()).collect::<Vec<_>>();
                            return expand(&Expr::Mul(repeated), interner);
                        }
                    }
                }
            }
            Expr::pow(expanded_base, expanded_exp)
        }
        Expr::Neg(e) => {
            let inner = expand(e, interner);
            if let Expr::Add(terms) = inner {
                let expanded = Expr::add(terms.into_iter().map(Expr::neg).collect());
                collect_flat_add(&expanded)
            } else {
                Expr::neg(inner)
            }
        }
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(|a| expand(a, interner)).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(expand(body, interner)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(expand(lhs, interner)),
            Box::new(expand(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (expand(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(expand(base, interner)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(expand(val, interner)),
            Box::new(expand(body, interner)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(|i| expand(i, interner)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|cell| expand(cell, interner)).collect())
                .collect(),
        ),
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => Expr::Sym(*s),
    }
}

pub fn collect_terms(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let normalized_terms = terms
                .iter()
                .map(|term| collect_terms(term, interner))
                .collect::<Vec<_>>();

            let mut groups: Vec<(Expr, BigRational)> = Vec::new();
            for term in &normalized_terms {
                let (coeff, base) = decompose_term(term);
                if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
                    *acc += coeff;
                } else {
                    groups.push((base, coeff));
                }
            }

            let rebuilt = groups
                .into_iter()
                .filter_map(|(base, coeff)| {
                    if coeff.is_zero() {
                        return None;
                    }

                    let coeff_expr = if coeff.is_integer() {
                        Expr::Int(coeff.to_integer())
                    } else {
                        Expr::Rational(coeff)
                    };

                    let term = if base == Expr::one() {
                        coeff_expr
                    } else {
                        Expr::mul(vec![coeff_expr, base])
                    };
                    Some(term)
                })
                .collect::<Vec<_>>();

            let combined = Expr::add(rebuilt);
            if let Expr::Add(combined_terms) = &combined {
                if let Some((common, remainders)) = extract_common_factor(combined_terms) {
                    let inner = collect_terms(&Expr::add(remainders), interner);
                    return Expr::mul(vec![common, inner]);
                }
            }

            combined
        }
        Expr::Mul(factors) => {
            let normalized = factors
                .iter()
                .map(|factor| collect_terms(factor, interner))
                .collect::<Vec<_>>();

            if let Some((idx, terms)) = normalized.iter().enumerate().find_map(|(idx, factor)| {
                if let Expr::Add(terms) = factor {
                    Some((idx, terms.clone()))
                } else {
                    None
                }
            }) {
                if normalized.len() <= 4 {
                    let rest = normalized
                        .iter()
                        .enumerate()
                        .filter_map(
                            |(i, factor)| if i != idx { Some(factor.clone()) } else { None },
                        )
                        .collect::<Vec<_>>();

                    let is_pure_sign_flip = rest.len() == 1
                        && matches!(
                            rest.first(),
                            Some(Expr::Int(n)) if *n == (-1).into()
                        );

                    if is_pure_sign_flip {
                        return Expr::mul(normalized);
                    }

                    let distributed = terms
                        .into_iter()
                        .map(|term| {
                            let mut factors = Vec::with_capacity(rest.len() + 1);
                            factors.push(term);
                            factors.extend(rest.clone());
                            Expr::mul(factors)
                        })
                        .collect::<Vec<_>>();

                    return Expr::add(
                        distributed
                            .iter()
                            .map(|term| collect_terms(term, interner))
                            .collect(),
                    );
                }
            }

            Expr::mul(normalized)
        }
        Expr::Pow(base, exp) => {
            Expr::pow(collect_terms(base, interner), collect_terms(exp, interner))
        }
        Expr::Neg(inner) => Expr::neg(collect_terms(inner, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(collect_terms(re, interner)),
            Box::new(collect_terms(im, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| collect_terms(arg, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(collect_terms(body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(collect_terms(lhs, interner)),
            Box::new(collect_terms(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (collect_terms(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(collect_terms(base, interner)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(collect_terms(val, interner)),
            Box::new(collect_terms(body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| collect_terms(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| collect_terms(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn trim_poly(coeffs: &mut Vec<Expr>) {
    while coeffs.len() > 1
        && coeffs.last().is_some_and(|coeff| {
            matches!(coeff, Expr::Int(n) if n.is_zero())
                || matches!(coeff, Expr::Rational(r) if r.is_zero())
        })
    {
        coeffs.pop();
    }
}

fn poly_degree(coeffs: &[Expr]) -> Option<usize> {
    coeffs
        .iter()
        .rposition(|coeff| !matches!(coeff, Expr::Int(n) if n.is_zero()) && !matches!(coeff, Expr::Rational(r) if r.is_zero()))
}

fn collect_syms(expr: &Expr, out: &mut BTreeSet<lasso::Spur>) {
    match expr {
        Expr::Sym(sym) => {
            out.insert(*sym);
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_syms(term, out);
            }
        }
        Expr::Pow(base, exp) => {
            collect_syms(base, out);
            collect_syms(exp, out);
        }
        Expr::Neg(inner) => collect_syms(inner, out),
        Expr::Call(_, args) => {
            for arg in args {
                collect_syms(arg, out);
            }
        }
        Expr::Complex(re, im) => {
            collect_syms(re, out);
            collect_syms(im, out);
        }
        Expr::FnDef(_, _, body) => collect_syms(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_syms(lhs, out);
            collect_syms(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_syms(value, out);
            }
        }
        Expr::Indexed(base, _) => collect_syms(base, out),
        Expr::Let(_, value, body) => {
            collect_syms(value, out);
            collect_syms(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_syms(cell, out);
                }
            }
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => {}
    }
}

fn poly_from_coeffs(coeffs: &[Expr], var: lasso::Spur) -> Expr {
    let terms = coeffs
        .iter()
        .enumerate()
        .filter_map(|(degree, coeff)| {
            if matches!(coeff, Expr::Int(n) if n.is_zero()) || matches!(coeff, Expr::Rational(r) if r.is_zero()) {
                return None;
            }
            let term = match degree {
                0 => coeff.clone(),
                1 => Expr::mul(vec![coeff.clone(), Expr::Sym(var)]),
                _ => Expr::mul(vec![
                    coeff.clone(),
                    Expr::pow(Expr::Sym(var), Expr::Int((degree as i64).into())),
                ]),
            };
            Some(term)
        })
        .collect::<Vec<_>>();
    if terms.is_empty() {
        Expr::zero()
    } else {
        Expr::add(terms)
    }
}

fn extract_numer_denom(expr: &Expr) -> (Expr, Expr) {
    match expr {
        Expr::Mul(factors) => {
            let mut numer = Vec::new();
            let mut denom = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if n.is_negative()) => {
                        let n = if let Expr::Int(n) = exp.as_ref() { n } else { unreachable!() };
                        denom.push(Expr::pow((**base).clone(), Expr::Int((-n).clone())));
                    }
                    Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Neg(_)) => {
                        if let Expr::Neg(inner) = exp.as_ref() {
                            denom.push(Expr::pow((**base).clone(), (**inner).clone()));
                        }
                    }
                    _ => numer.push(factor.clone()),
                }
            }
            (Expr::mul(numer), Expr::mul(denom))
        }
        Expr::Add(terms) => {
            let extracted = terms.iter().map(extract_numer_denom).collect::<Vec<_>>();
            let common_denom = Expr::mul(extracted.iter().map(|(_, denom)| denom.clone()).collect());
            let combined_numer = Expr::add(
                extracted
                    .iter()
                    .enumerate()
                    .map(|(idx, (numer, _))| {
                        let others = extracted
                            .iter()
                            .enumerate()
                            .filter_map(|(j, (_, denom))| if idx == j { None } else { Some(denom.clone()) })
                            .collect::<Vec<_>>();
                        Expr::mul(
                            std::iter::once(numer.clone())
                                .chain(others)
                                .collect(),
                        )
                    })
                    .collect(),
            );
            (combined_numer, common_denom)
        }
        Expr::Neg(inner) => {
            let (numer, denom) = extract_numer_denom(inner);
            (Expr::neg(numer), denom)
        }
        _ => (expr.clone(), Expr::one()),
    }
}

fn poly_div_rem(
    dividend: &[Expr],
    divisor: &[Expr],
    interner: &ax_ir::Interner,
) -> (Vec<Expr>, Vec<Expr>) {
    let mut remainder = dividend.to_vec();
    trim_poly(&mut remainder);
    let mut quotient = vec![Expr::zero(); remainder.len().saturating_sub(divisor.len()) + 1];
    let Some(divisor_degree) = poly_degree(divisor) else {
        return (Vec::new(), remainder);
    };
    let divisor_lead = divisor[divisor_degree].clone();

    while let (Some(rem_degree), true) = (poly_degree(&remainder), poly_degree(&remainder).unwrap_or(0) >= divisor_degree) {
        let degree_diff = rem_degree - divisor_degree;
        let lead_factor = eval(
            &Expr::mul(vec![
                remainder[rem_degree].clone(),
                Expr::pow(divisor_lead.clone(), Expr::Int((-1).into())),
            ]),
            &Env::new(),
            interner,
        );
        quotient[degree_diff] = Expr::add(vec![quotient[degree_diff].clone(), lead_factor.clone()]);

        let mut subtraction = vec![Expr::zero(); degree_diff + divisor.len()];
        for (idx, coeff) in divisor.iter().enumerate() {
            subtraction[degree_diff + idx] = Expr::mul(vec![lead_factor.clone(), coeff.clone()]);
        }
        if remainder.len() < subtraction.len() {
            remainder.resize(subtraction.len(), Expr::zero());
        }
        for i in 0..subtraction.len() {
            remainder[i] = eval(
                &Expr::add(vec![remainder[i].clone(), Expr::neg(subtraction[i].clone())]),
                &Env::new(),
                interner,
            );
        }
        trim_poly(&mut remainder);
    }

    trim_poly(&mut quotient);
    trim_poly(&mut remainder);
    (quotient, remainder)
}

fn poly_gcd(a: &Expr, b: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    let Some(mut lhs) = ax_solve::extract_polynomial(a, var, interner) else {
        return Expr::one();
    };
    let Some(mut rhs) = ax_solve::extract_polynomial(b, var, interner) else {
        return Expr::one();
    };
    trim_poly(&mut lhs);
    trim_poly(&mut rhs);

    while poly_degree(&rhs)
        .is_some_and(|deg| deg > 0 || !matches!(&rhs[0], Expr::Int(n) if n.is_zero()))
    {
        let (_, mut remainder) = poly_div_rem(&lhs, &rhs, interner);
        trim_poly(&mut remainder);
        if poly_degree(&remainder).is_none() && matches!(remainder.first(), Some(Expr::Int(n)) if n.is_zero()) {
            lhs = rhs;
            break;
        }
        lhs = rhs;
        rhs = remainder;
    }

    if let Some(deg) = poly_degree(&lhs) {
        let lead = lhs[deg].clone();
        let normalized = lhs
            .into_iter()
            .map(|coeff| {
                eval(
                    &Expr::mul(vec![coeff, Expr::pow(lead.clone(), Expr::Int((-1).into()))]),
                    &Env::new(),
                    interner,
                )
            })
            .collect::<Vec<_>>();
        poly_from_coeffs(&normalized, var)
    } else {
        Expr::one()
    }
}

fn cancel_common(numer: &Expr, denom: &Expr, interner: &ax_ir::Interner) -> (Expr, Expr) {
    let mut syms = BTreeSet::new();
    collect_syms(numer, &mut syms);
    collect_syms(denom, &mut syms);

    for sym in syms {
        let gcd = poly_gcd(numer, denom, sym, interner);
        if gcd != Expr::one() {
            let Some(n_coeffs) = ax_solve::extract_polynomial(numer, sym, interner) else {
                continue;
            };
            let Some(d_coeffs) = ax_solve::extract_polynomial(denom, sym, interner) else {
                continue;
            };
            let Some(g_coeffs) = ax_solve::extract_polynomial(&gcd, sym, interner) else {
                continue;
            };
            if poly_degree(&g_coeffs).unwrap_or(0) == 0 {
                continue;
            }
            let (qn, rn) = poly_div_rem(&n_coeffs, &g_coeffs, interner);
            let (qd, rd) = poly_div_rem(&d_coeffs, &g_coeffs, interner);
            let zero_rem = |coeffs: &[Expr]| {
                coeffs.iter().all(|c| matches!(c, Expr::Int(n) if n.is_zero()) || matches!(c, Expr::Rational(r) if r.is_zero()))
            };
            if zero_rem(&rn) && zero_rem(&rd) {
                return (poly_from_coeffs(&qn, sym), poly_from_coeffs(&qd, sym));
            }
        }
    }

    (numer.clone(), denom.clone())
}

pub fn rationalize(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let (numer, denom) = extract_numer_denom(expr);
    let numer = eval(&numer, &Env::new(), interner);
    let denom = eval(&denom, &Env::new(), interner);
    let (numer_s, denom_s) = cancel_common(&numer, &denom, interner);
    let numer_s = eval(&numer_s, &Env::new(), interner);
    let denom_s = eval(&denom_s, &Env::new(), interner);
    if denom_s == Expr::one() {
        numer_s
    } else {
        Expr::mul(vec![numer_s, Expr::pow(denom_s, Expr::Int((-1).into()))])
    }
}

pub fn simplify(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let e1 = expand(expr, interner);
    let e2 = collect_terms(&e1, interner);
    let e3 = rationalize(&e2, interner);
    let e4 = trig_simplify(&e3, interner);
    eval(&e4, &Env::new(), interner)
}

pub fn trig_simplify(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let rules = build_trig_rules(interner);
    let result = ax_rewrite::rewrite_fixed_point(&rules, expr, interner, 20);
    crate::eval(&result, &crate::Env::new(), interner)
}

/// Factor out common factors from terms in a sum.
///
/// a*x + a*y → a*(x + y)
///
/// If `targets` is empty, automatically detect common factors.
pub fn factor_out(
    expr: &Expr,
    targets: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) if terms.len() >= 2 => {
            let targets_to_try: Vec<lasso::Spur> = if targets.is_empty() {
                find_common_symbols(terms)
            } else {
                targets.to_vec()
            };

            for target in &targets_to_try {
                // Simple factor: target appears in every term
                if terms.iter().all(|t| has_factor(t, *target)) {
                    let stripped: Vec<Expr> =
                        terms.iter().map(|t| remove_factor(t, *target)).collect();
                    let inner = Expr::add(stripped);
                    return Expr::mul(vec![
                        Expr::Sym(*target),
                        factor_out(&inner, targets, interner),
                    ]);
                }

                // Power factor: target^min_power
                let min_power = terms
                    .iter()
                    .filter_map(|t| power_of_factor(t, *target))
                    .min();
                if let Some(p) = min_power {
                    if p >= 1 {
                        let stripped: Vec<Expr> = terms
                            .iter()
                            .map(|t| reduce_factor_power(t, *target, p))
                            .collect();
                        let inner = Expr::add(stripped);
                        let factor = if p == 1 {
                            Expr::Sym(*target)
                        } else {
                            Expr::pow(
                                Expr::Sym(*target),
                                Expr::Int(num_bigint::BigInt::from(p)),
                            )
                        };
                        return Expr::mul(vec![factor, factor_out(&inner, targets, interner)]);
                    }
                }
            }

            expr.clone()
        }
        _ => expr.clone(),
    }
}

/// Group terms that share specified pre-factors.
///
/// x*a + x*b + y → x*(a + b) + y
pub fn factor_in(
    expr: &Expr,
    targets: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    factor_out(expr, targets, interner)
}

fn find_common_symbols(terms: &[Expr]) -> Vec<lasso::Spur> {
    if terms.is_empty() {
        return vec![];
    }
    let first_syms = collect_factor_symbols(&terms[0]);
    first_syms
        .into_iter()
        .filter(|s| terms[1..].iter().all(|t| has_factor(t, *s)))
        .collect()
}

fn collect_factor_symbols(expr: &Expr) -> Vec<lasso::Spur> {
    match expr {
        Expr::Sym(s) => vec![*s],
        Expr::Mul(factors) => factors.iter().flat_map(collect_factor_symbols).collect(),
        Expr::Neg(e) => collect_factor_symbols(e),
        Expr::Pow(base, _) => collect_factor_symbols(base),
        _ => vec![],
    }
}

fn has_factor(expr: &Expr, sym: lasso::Spur) -> bool {
    match expr {
        Expr::Sym(s) => *s == sym,
        Expr::Mul(factors) => factors.iter().any(|f| has_factor(f, sym)),
        Expr::Neg(e) => has_factor(e, sym),
        Expr::Pow(base, _) => has_factor(base, sym),
        _ => false,
    }
}

fn remove_factor(expr: &Expr, sym: lasso::Spur) -> Expr {
    match expr {
        Expr::Sym(s) if *s == sym => Expr::one(),
        Expr::Mul(factors) => {
            let mut found = false;
            let new_factors: Vec<Expr> = factors
                .iter()
                .map(|f| {
                    if !found && has_factor(f, sym) {
                        found = true;
                        remove_factor(f, sym)
                    } else {
                        f.clone()
                    }
                })
                .collect();
            Expr::mul(new_factors)
        }
        Expr::Neg(e) => Expr::neg(remove_factor(e, sym)),
        Expr::Pow(base, exp) => {
            if let Expr::Sym(s) = base.as_ref() {
                if *s == sym {
                    if let Expr::Int(n) = exp.as_ref() {
                        let new_exp = n.clone() - num_bigint::BigInt::from(1u32);
                        if new_exp == num_bigint::BigInt::from(0u32) {
                            return Expr::one();
                        }
                        return Expr::pow(base.as_ref().clone(), Expr::Int(new_exp));
                    }
                }
            }
            expr.clone()
        }
        _ => expr.clone(),
    }
}

fn power_of_factor(expr: &Expr, sym: lasso::Spur) -> Option<i64> {
    match expr {
        Expr::Sym(s) if *s == sym => Some(1),
        Expr::Pow(base, exp) => {
            if let Expr::Sym(s) = base.as_ref() {
                if *s == sym {
                    if let Expr::Int(n) = exp.as_ref() {
                        return n.to_i64();
                    }
                }
            }
            None
        }
        Expr::Mul(factors) => factors.iter().filter_map(|f| power_of_factor(f, sym)).next(),
        Expr::Neg(e) => power_of_factor(e, sym),
        _ => Some(0),
    }
}

fn reduce_factor_power(expr: &Expr, sym: lasso::Spur, reduce_by: i64) -> Expr {
    let mut remaining = reduce_by;
    reduce_power_recursive(expr, sym, &mut remaining)
}

fn reduce_power_recursive(expr: &Expr, sym: lasso::Spur, remaining: &mut i64) -> Expr {
    if *remaining <= 0 {
        return expr.clone();
    }
    match expr {
        Expr::Sym(s) if *s == sym => {
            *remaining -= 1;
            Expr::one()
        }
        Expr::Pow(base, exp) if matches!(base.as_ref(), Expr::Sym(s) if *s == sym) => {
            if let Expr::Int(n) = exp.as_ref() {
                let p = n.to_i64().unwrap_or(0);
                let new_p = p - *remaining;
                *remaining = 0;
                if new_p == 0 {
                    Expr::one()
                } else if new_p == 1 {
                    base.as_ref().clone()
                } else {
                    Expr::pow(
                        base.as_ref().clone(),
                        Expr::Int(num_bigint::BigInt::from(new_p)),
                    )
                }
            } else {
                expr.clone()
            }
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| reduce_power_recursive(f, sym, remaining))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(reduce_power_recursive(e, sym, remaining)),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(
            result.errors.is_empty(),
            "lower errors: {:?}",
            result.errors
        );
        let expr = result.expr.expect("expected expression");
        let env = crate::Env::new();
        (crate::eval(&expr, &env, &interner), interner)
    }

    #[test]
    fn expand_distribute() {
        let (e, int) = eval_src("expand(a * (b + c));");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(
            pp.contains("a") && pp.contains("b") && pp.contains("c"),
            "got: {}",
            pp
        );
        assert!(!pp.contains("("), "got: {}", pp);
    }

    #[test]
    fn expand_square() {
        let (e, int) = eval_src("expand((x + 1)^2);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x") && pp.contains("2"), "got: {}", pp);
    }

    #[test]
    fn collect_like_terms() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::add(vec![
            ax_ir::Expr::mul(vec![ax_ir::Expr::Int(2.into()), ax_ir::Expr::Sym(x)]),
            ax_ir::Expr::mul(vec![ax_ir::Expr::Int(3.into()), ax_ir::Expr::Sym(x)]),
        ]);
        let result = collect_terms(&expr, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("5"), "got: {}", pp);
    }

    #[test]
    fn trig_pythag() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");

        let expr = Expr::add(vec![
            Expr::pow(Expr::Call(sin_sym, vec![Expr::Sym(x)]), Expr::Int(2.into())),
            Expr::pow(Expr::Call(cos_sym, vec![Expr::Sym(x)]), Expr::Int(2.into())),
        ]);
        let result = trig_simplify(&expr, &interner);
        assert_eq!(result, Expr::one());
    }

    #[test]
    fn trig_one_minus_sin_sq() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");

        let expr = Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::pow(
                Expr::Call(sin_sym, vec![Expr::Sym(x)]),
                Expr::Int(2.into()),
            )),
        ]);
        let result = trig_simplify(&expr, &interner);
        let expected = Expr::pow(Expr::Call(cos_sym, vec![Expr::Sym(x)]), Expr::Int(2.into()));
        assert_eq!(result, expected);
    }

    #[test]
    fn rationalize_cancels_common_factor() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");

        let numer = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::neg(Expr::one()),
        ]);
        let denom = Expr::add(vec![Expr::Sym(x), Expr::neg(Expr::one())]);
        let expr = Expr::mul(vec![numer, Expr::pow(denom, Expr::Int((-1).into()))]);

        let result = rationalize(&expr, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(!pp.contains("^"), "still has powers: {}", pp);
        assert!(pp.contains("x") && pp.contains("1"), "got: {}", pp);
    }

    #[test]
    fn rationalize_common_denominator() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");

        let expr = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int((-1).into())),
            Expr::pow(Expr::Sym(x), Expr::Int((-2).into())),
        ]);

        let result = rationalize(&expr, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn factor_out_common() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        // a*x + a*y → a*(x + y)
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(x)]),
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(y)]),
        ]);
        let result = factor_out(&expr, &[], &interner);
        if let Expr::Mul(factors) = &result {
            assert!(
                factors.iter().any(|f| *f == Expr::Sym(a)),
                "got {result:?}"
            );
        } else {
            panic!("expected Mul, got {result:?}");
        }
    }

    #[test]
    fn factor_out_explicit_target() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(x)]),
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(y)]),
        ]);
        let result = factor_out(&expr, &[a], &interner);
        if let Expr::Mul(factors) = &result {
            assert!(factors.iter().any(|f| *f == Expr::Sym(a)), "got {result:?}");
        } else {
            panic!("expected Mul, got {result:?}");
        }
    }

    #[test]
    fn factor_out_no_common_unchanged() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        // a*x + y — no common factor
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(x)]),
            Expr::Sym(y),
        ]);
        let result = factor_out(&expr, &[], &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn factor_out_power() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        // a^2*x + a^2*y → a^2*(x + y)
        let a2 = Expr::pow(Expr::Sym(a), Expr::Int(2.into()));
        let expr = Expr::add(vec![
            Expr::mul(vec![a2.clone(), Expr::Sym(x)]),
            Expr::mul(vec![a2.clone(), Expr::Sym(y)]),
        ]);
        let result = factor_out(&expr, &[a], &interner);
        assert!(matches!(result, Expr::Mul(_)), "expected Mul, got {result:?}");
    }
}
