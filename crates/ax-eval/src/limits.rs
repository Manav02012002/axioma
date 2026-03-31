use ax_ir::Expr;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Int(n) => n.is_zero(),
        Expr::Rational(r) => r.is_zero(),
        Expr::Float(f) => *f == 0.0,
        Expr::Complex(re, im) => is_zero(re) && is_zero(im),
        _ => false,
    }
}

fn is_infinity(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Sym(s) => matches!(interner.resolve(*s), "inf" | "infty" | "neg_inf"),
        Expr::Float(f) => f.is_infinite(),
        _ => false,
    }
}

fn is_nan_or_indeterminate(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Float(f) => f.is_nan() || f.is_infinite(),
        Expr::Sym(s) => matches!(interner.resolve(*s), "inf" | "infty" | "neg_inf" | "nan"),
        Expr::Call(f, _) => interner.resolve(*f) == "limit",
        _ => false,
    }
}

fn try_substitution(
    expr: &Expr,
    var: lasso::Spur,
    point: &Expr,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if let Some((_, denom)) = extract_fraction(expr) {
        let mut denom_env = crate::Env::new();
        denom_env.bindings.insert(var, point.clone());
        let denom_value = crate::eval(&denom, &denom_env, interner);
        if is_zero(&denom_value) || is_nan_or_indeterminate(&denom_value, interner) {
            return None;
        }
    }

    let mut env = crate::Env::new();
    env.bindings.insert(var, point.clone());
    let result = crate::eval(expr, &env, interner);
    if is_nan_or_indeterminate(&result, interner) {
        None
    } else {
        Some(result)
    }
}

fn extract_fraction(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::Mul(factors) => {
            let mut numer_parts = Vec::new();
            let mut denom_parts = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Pow(base, exp)
                        if matches!(exp.as_ref(), Expr::Int(n) if *n == (-1).into()) =>
                    {
                        denom_parts.push(base.as_ref().clone());
                    }
                    Expr::Pow(base, exp)
                        if matches!(exp.as_ref(), Expr::Neg(inner) if matches!(inner.as_ref(), Expr::Int(_))) =>
                    {
                        if let Expr::Neg(inner) = exp.as_ref() {
                            denom_parts.push(Expr::pow(base.as_ref().clone(), inner.as_ref().clone()));
                        }
                    }
                    _ => numer_parts.push(factor.clone()),
                }
            }
            if denom_parts.is_empty() {
                None
            } else {
                Some((Expr::mul(numer_parts), Expr::mul(denom_parts)))
            }
        }
        Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == (-1).into()) => {
            Some((Expr::one(), base.as_ref().clone()))
        }
        _ => None,
    }
}

fn as_nonnegative_usize(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Int(n) if !n.is_negative() => n.to_usize(),
        _ => None,
    }
}

fn expr_contains_var(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Sym(s) => *s == var,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| expr_contains_var(term, var))
        }
        Expr::Pow(base, exp) => expr_contains_var(base, var) || expr_contains_var(exp, var),
        Expr::Neg(inner) => expr_contains_var(inner, var),
        Expr::Call(_, args) => args.iter().any(|arg| expr_contains_var(arg, var)),
        Expr::Complex(re, im) => expr_contains_var(re, var) || expr_contains_var(im, var),
        Expr::FnDef(_, _, body) => expr_contains_var(body, var),
        Expr::Rule(lhs, rhs, _) => expr_contains_var(lhs, var) || expr_contains_var(rhs, var),
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, _)| expr_contains_var(value, var)),
        Expr::Let(_, value, body) => expr_contains_var(value, var) || expr_contains_var(body, var),
        Expr::Indexed(base, _) => expr_contains_var(base, var),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| expr_contains_var(cell, var))),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => false,
    }
}

fn degree_of_var(expr: &Expr, var: lasso::Spur) -> Option<usize> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Some(0),
        Expr::Sym(s) => Some(if *s == var { 1 } else { 0 }),
        Expr::Neg(inner) => degree_of_var(inner, var),
        Expr::Add(terms) => terms.iter().map(|term| degree_of_var(term, var)).max().flatten(),
        Expr::Mul(factors) => {
            let mut degree = 0usize;
            for factor in factors {
                degree += degree_of_var(factor, var)?;
            }
            Some(degree)
        }
        Expr::Pow(base, exp) => match (base.as_ref(), as_nonnegative_usize(exp)) {
            (Expr::Sym(s), Some(n)) if *s == var => Some(n),
            (_, Some(0)) => Some(0),
            (base, Some(n)) if !expr_contains_var(base, var) => Some(0usize.saturating_mul(n)),
            _ => None,
        },
        _ => None,
    }
}

fn leading_coefficient(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Option<Expr> {
    let degree = degree_of_var(expr, var)?;
    match expr {
        Expr::Add(terms) => {
            let leading_terms = terms
                .iter()
                .filter(|term| degree_of_var(term, var) == Some(degree))
                .cloned()
                .collect::<Vec<_>>();
            if leading_terms.is_empty() {
                None
            } else {
                Some(crate::eval(
                    &Expr::mul(vec![
                        Expr::add(leading_terms),
                        Expr::pow(Expr::Sym(var), Expr::Int(BigInt::from(-(degree as i64)))),
                    ]),
                    &crate::Env::new(),
                    interner,
                ))
            }
        }
        Expr::Mul(factors) => {
            let coeffs = factors
                .iter()
                .filter(|factor| degree_of_var(factor, var) == Some(0))
                .cloned()
                .collect::<Vec<_>>();
            Some(if coeffs.is_empty() {
                Expr::one()
            } else {
                crate::eval(&Expr::mul(coeffs), &crate::Env::new(), interner)
            })
        }
        Expr::Sym(s) if *s == var => Some(Expr::one()),
        Expr::Pow(base, exp)
            if matches!(base.as_ref(), Expr::Sym(s) if *s == var)
                && matches!(exp.as_ref(), Expr::Int(_)) =>
        {
            Some(Expr::one())
        }
        expr if degree == 0 => Some(expr.clone()),
        _ => None,
    }
}

fn matches_special_limit(
    expr: &Expr,
    var: lasso::Spur,
    point: &Expr,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let sin_sym = interner.get_or_intern("sin");
    let exp_sym = interner.get_or_intern("exp");
    let e_sym = interner.get_or_intern("e");
    let inf_sym = interner.get_or_intern("inf");

    if matches!(point, Expr::Int(n) if n.is_zero()) {
        if let Some((numer, denom)) = extract_fraction(expr) {
            if denom == Expr::Sym(var) {
                match numer {
                    Expr::Call(f, ref args) if f == sin_sym && args.as_slice() == [Expr::Sym(var)] => {
                        return Some(Expr::one());
                    }
                    Expr::Add(ref terms) if terms.len() == 2 => {
                        let mut saw_exp_x = false;
                        let mut saw_neg_one = false;
                        for term in terms {
                            match term {
                                Expr::Call(f, args)
                                    if *f == exp_sym && args.as_slice() == [Expr::Sym(var)] =>
                                {
                                    saw_exp_x = true;
                                }
                                Expr::Int(n) if *n == (-1).into() => saw_neg_one = true,
                                _ => {}
                            }
                        }
                        if saw_exp_x && saw_neg_one {
                            return Some(Expr::one());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if matches!(point, Expr::Sym(s) if *s == inf_sym) {
        if let Expr::Pow(base, exp) = expr {
            if matches!(exp.as_ref(), Expr::Sym(s) if *s == var) {
                if let Expr::Add(terms) = base.as_ref() {
                    let mut saw_one = false;
                    let mut saw_inv_x = false;
                    for term in terms {
                        match term {
                            Expr::Int(n) if n.is_one() => saw_one = true,
                            Expr::Pow(inner, e)
                                if matches!(inner.as_ref(), Expr::Sym(s) if *s == var)
                                    && matches!(e.as_ref(), Expr::Int(n) if *n == (-1).into()) =>
                            {
                                saw_inv_x = true;
                            }
                            _ => {}
                        }
                    }
                    if saw_one && saw_inv_x {
                        return Some(Expr::Sym(e_sym));
                    }
                }
            }
        }

        if let Expr::Mul(factors) = expr {
            let mut polynomial_degree = 0usize;
            let mut saw_exp_neg_x = false;
            for factor in factors {
                match factor {
                    Expr::Pow(base, exp)
                        if matches!(base.as_ref(), Expr::Sym(s) if *s == var)
                            && matches!(exp.as_ref(), Expr::Int(n) if !n.is_negative()) =>
                    {
                        polynomial_degree += as_nonnegative_usize(exp).unwrap_or(0);
                    }
                    Expr::Sym(s) if *s == var => polynomial_degree += 1,
                    Expr::Call(f, args) if *f == exp_sym && args.len() == 1 => {
                        if matches!(&args[0], Expr::Neg(inner) if matches!(inner.as_ref(), Expr::Sym(s) if *s == var))
                        {
                            saw_exp_neg_x = true;
                        }
                    }
                    _ => {}
                }
            }
            if saw_exp_neg_x && polynomial_degree > 0 {
                return Some(Expr::zero());
            }
        }
    }

    None
}

fn rational_limit_at_infinity(
    expr: &Expr,
    var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let inf_sym = interner.get_or_intern("inf");
    let neg_inf_sym = interner.get_or_intern("neg_inf");
    let (numer, denom) = extract_fraction(expr)?;
    let numer_degree = degree_of_var(&numer, var)?;
    let denom_degree = degree_of_var(&denom, var)?;

    if numer_degree < denom_degree {
        return Some(Expr::zero());
    }

    if numer_degree == denom_degree {
        let leading_num = leading_coefficient(&numer, var, interner)?;
        let leading_den = leading_coefficient(&denom, var, interner)?;
        return Some(crate::eval(
            &Expr::mul(vec![
                leading_num,
                Expr::pow(leading_den, Expr::Int((-1).into())),
            ]),
            &crate::Env::new(),
            interner,
        ));
    }

    let leading_num = leading_coefficient(&numer, var, interner)?;
    let leading_den = leading_coefficient(&denom, var, interner)?;
    let sign = crate::to_f64(&crate::eval(
        &Expr::mul(vec![
            leading_num,
            Expr::pow(leading_den, Expr::Int((-1).into())),
        ]),
        &crate::Env::new(),
        interner,
    ))?;

    Some(if sign.is_sign_negative() {
        Expr::Sym(neg_inf_sym)
    } else {
        Expr::Sym(inf_sym)
    })
}

pub fn limit(
    expr: &Expr,
    var: lasso::Spur,
    point: &Expr,
    interner: &ax_ir::Interner,
) -> Expr {
    if let Some(special) = matches_special_limit(expr, var, point, interner) {
        return special;
    }

    if matches!(point, Expr::Sym(s) if matches!(interner.resolve(*s), "inf" | "infty" | "neg_inf")) {
        if let Some(result) = rational_limit_at_infinity(expr, var, interner) {
            return result;
        }
    }

    if let Some(result) = try_substitution(expr, var, point, interner) {
        return result;
    }

    if let Some((mut numer, mut denom)) = extract_fraction(expr) {
        let n_at_point = try_substitution(&numer, var, point, interner);
        let d_at_point = try_substitution(&denom, var, point, interner);

        let is_indeterminate = match (&n_at_point, &d_at_point) {
            (Some(n), Some(d)) if is_zero(n) && is_zero(d) => true,
            (Some(n), Some(d)) if is_infinity(n, interner) && is_infinity(d, interner) => true,
            _ => false,
        };

        if is_indeterminate {
            for _ in 0..5 {
                numer = crate::eval(
                    &crate::differentiate(&numer, var, interner),
                    &crate::Env::new(),
                    interner,
                );
                denom = crate::eval(
                    &crate::differentiate(&denom, var, interner),
                    &crate::Env::new(),
                    interner,
                );

                let next = crate::eval(
                    &Expr::mul(vec![
                        numer.clone(),
                        Expr::pow(denom.clone(), Expr::Int((-1).into())),
                    ]),
                    &crate::Env::new(),
                    interner,
                );

                if let Some(result) = try_substitution(&next, var, point, interner) {
                    return result;
                }

                if let Some((next_numer, next_denom)) = extract_fraction(&next) {
                    numer = next_numer;
                    denom = next_denom;
                } else {
                    return next;
                }
            }
        }
    }

    Expr::Call(
        interner.get_or_intern("limit"),
        vec![expr.clone(), Expr::Sym(var), point.clone()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(result.errors.is_empty(), "lower errors: {:?}", result.errors);
        let expr = result.expr.expect("expected expression");
        let env = crate::Env::new();
        (crate::eval(&expr, &env, &interner), interner)
    }

    #[test]
    fn limit_sinx_over_x() {
        let (e, _) = eval_src("limit(sin(x)/x, x, 0)");
        assert_eq!(e, Expr::Int(1.into()));
    }

    #[test]
    fn limit_polynomial() {
        let (e, _) = eval_src("limit(x^2 + 1, x, 3)");
        assert_eq!(e, Expr::Int(10.into()));
    }

    #[test]
    fn limit_lhopital() {
        let (e, _) = eval_src("limit((exp(x) - 1)/x, x, 0)");
        assert_eq!(e, Expr::Int(1.into()));
    }
}
