use crate::{contains_var, differentiate};
use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;

fn unevaluated(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(
        interner.get_or_intern("integrate"),
        vec![expr.clone(), Expr::Sym(var)],
    )
}

fn call1(name: &str, arg: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern(name), vec![arg])
}

fn as_expr(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Int(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn one_half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn quarter() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 4.into()))
}

fn is_var(expr: &Expr, var: lasso::Spur) -> bool {
    matches!(expr, Expr::Sym(sym) if *sym == var)
}

fn split_constant_variable(expr: &Expr, var: lasso::Spur) -> (Expr, Expr) {
    match expr {
        Expr::Mul(factors) => {
            let (constant, variable): (Vec<_>, Vec<_>) =
                factors.iter().cloned().partition(|f| !contains_var(f, var));
            let c = if constant.is_empty() {
                Expr::one()
            } else {
                Expr::mul(constant)
            };
            let v = if variable.is_empty() {
                Expr::one()
            } else {
                Expr::mul(variable)
            };
            (c, v)
        }
        _ if contains_var(expr, var) => (Expr::one(), expr.clone()),
        _ => (expr.clone(), Expr::one()),
    }
}

fn constant_multiple(lhs: &Expr, rhs: &Expr, var: lasso::Spur) -> Option<Expr> {
    let (lhs_const, lhs_var) = split_constant_variable(lhs, var);
    let (rhs_const, rhs_var) = split_constant_variable(rhs, var);
    if lhs_var == rhs_var {
        Some(Expr::mul(vec![
            lhs_const,
            Expr::pow(rhs_const, Expr::Int((-1).into())),
        ]))
    } else {
        None
    }
}

fn substitute_symbol(expr: &Expr, from: lasso::Spur, to: &Expr) -> Expr {
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(sym) => {
            if *sym == from {
                to.clone()
            } else {
                Expr::Sym(*sym)
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_symbol(term, from, to))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_symbol(factor, from, to))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_symbol(base, from, to),
            substitute_symbol(exp, from, to),
        ),
        Expr::Neg(e) => Expr::neg(substitute_symbol(e, from, to)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| substitute_symbol(arg, from, to))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_symbol(body, from, to)),
        ),
        Expr::Rule(lhs, rhs) => Expr::Rule(
            Box::new(substitute_symbol(lhs, from, to)),
            Box::new(substitute_symbol(rhs, from, to)),
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (substitute_symbol(value, from, to), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(substitute_symbol(base, from, to)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(substitute_symbol(val, from, to)),
            Box::new(substitute_symbol(body, from, to)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_symbol(item, from, to))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_symbol(cell, from, to))
                        .collect()
                })
                .collect(),
        ),
    }
}

fn is_unevaluated_integrate(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Call(f, _) => interner.resolve(*f) == "integrate",
        Expr::Add(terms) => terms.iter().any(|t| is_unevaluated_integrate(t, interner)),
        Expr::Mul(factors) => factors
            .iter()
            .any(|f| is_unevaluated_integrate(f, interner)),
        Expr::Pow(base, exp) => {
            is_unevaluated_integrate(base, interner) || is_unevaluated_integrate(exp, interner)
        }
        Expr::Neg(e) => is_unevaluated_integrate(e, interner),
        _ => false,
    }
}

fn liate_rank_with_interner(expr: &Expr, interner: &ax_ir::Interner) -> usize {
    match expr {
        Expr::Call(f, _) => match interner.resolve(*f) {
            "log" => 0,
            "arcsin" | "arctan" => 1,
            "sin" | "cos" | "tan" | "sec" | "csc" | "cot" => 3,
            "exp" => 4,
            _ => 5,
        },
        Expr::Sym(_) | Expr::Pow(_, _) => 2,
        _ => 5,
    }
}

fn is_named_unary_call(
    expr: &Expr,
    name: &str,
    var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> bool {
    matches!(expr, Expr::Call(f, args) if args.len() == 1 && interner.resolve(*f) == name && is_var(&args[0], var))
}

fn match_var_squared(term: &Expr, var: lasso::Spur) -> bool {
    matches!(term, Expr::Pow(base, exp) if is_var(base, var) && matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into()))
}

fn extract_inv_quadratic(expr: &Expr, var: lasso::Spur) -> Option<Expr> {
    match expr {
        Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == (-1).into()) => {
            let Expr::Add(terms) = base.as_ref() else {
                return None;
            };
            if terms.len() != 2 {
                return None;
            }
            let var_sq = terms.iter().find(|term| match_var_squared(term, var))?;
            let other = terms.iter().find(|term| *term != var_sq)?;
            match other {
                Expr::Pow(inner, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into()) => {
                    if !contains_var(inner, var) {
                        Some((**inner).clone())
                    } else {
                        None
                    }
                }
                Expr::Int(n) if *n == 1.into() => Some(Expr::one()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_one_minus_var_sq(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Add(terms) if terms.len() == 2 => {
            terms
                .iter()
                .any(|term| matches!(term, Expr::Int(n) if *n == 1.into()))
                && terms
                    .iter()
                    .any(|term| matches!(term, Expr::Neg(inner) if match_var_squared(inner, var)))
        }
        _ => false,
    }
}

fn match_one_over_sqrt_one_minus_var_sq(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Rational(r) if *r == BigRational::new((-1).into(), 2.into())) => {
            is_one_minus_var_sq(base, var)
        }
        Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == (-1).into()) => {
            matches!(
                base.as_ref(),
                Expr::Pow(inner, inner_exp)
                    if matches!(inner_exp.as_ref(), Expr::Rational(r) if *r == BigRational::new(1.into(), 2.into()))
                        && is_one_minus_var_sq(inner, var)
            )
        }
        _ => false,
    }
}

fn table_integrate(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Option<Expr> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) if !contains_var(expr, var) => {
            Some(Expr::mul(vec![expr.clone(), Expr::Sym(var)]))
        }
        Expr::Sym(s) => {
            if *s == var {
                Some(Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), Expr::Int(2.into())),
                    one_half(),
                ]))
            } else {
                Some(Expr::mul(vec![Expr::Sym(*s), Expr::Sym(var)]))
            }
        }
        Expr::Neg(e) => Some(Expr::neg(integrate(e, var, interner))),
        Expr::Pow(base, exp) => match (base.as_ref(), exp.as_ref()) {
            (Expr::Sym(s), Expr::Int(n)) if *s == var && *n != (-1).into() => {
                let next = n.clone() + BigInt::from(1);
                Some(Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), Expr::Int(next.clone())),
                    Expr::Rational(BigRational::new(1.into(), next)),
                ]))
            }
            (Expr::Sym(s), Expr::Int(n)) if *s == var && *n == (-1).into() => Some(call1(
                "log",
                call1("abs", Expr::Sym(var), interner),
                interner,
            )),
            (Expr::Sym(s), Expr::Rational(r))
                if *s == var && *r != BigRational::from_integer((-1).into()) =>
            {
                let next = r.clone() + BigRational::one();
                Some(Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), as_expr(next.clone())),
                    as_expr(BigRational::one() / next),
                ]))
            }
            (Expr::Pow(inner, inner_exp), Expr::Int(n))
                if *n == (-1).into()
                    && matches!(inner.as_ref(), Expr::Sym(s) if *s == var)
                    && matches!(inner_exp.as_ref(), Expr::Int(m) if *m > 1.into()) =>
            {
                let Expr::Int(power) = inner_exp.as_ref() else {
                    unreachable!()
                };
                let next = BigInt::from(1) - power.clone();
                Some(Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), Expr::Int(next.clone())),
                    Expr::Rational(BigRational::new(1.into(), next)),
                ]))
            }
            _ => {
                if is_named_unary_call(base, "sin", var, interner)
                    && matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into())
                {
                    Some(Expr::add(vec![
                        Expr::mul(vec![Expr::Sym(var), one_half()]),
                        Expr::neg(Expr::mul(vec![
                            call1(
                                "sin",
                                Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(var)]),
                                interner,
                            ),
                            quarter(),
                        ])),
                    ]))
                } else if is_named_unary_call(base, "cos", var, interner)
                    && matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into())
                {
                    Some(Expr::add(vec![
                        Expr::mul(vec![Expr::Sym(var), one_half()]),
                        Expr::mul(vec![
                            call1(
                                "sin",
                                Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(var)]),
                                interner,
                            ),
                            quarter(),
                        ]),
                    ]))
                } else if is_named_unary_call(base, "sec", var, interner)
                    && matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into())
                {
                    Some(call1("tan", Expr::Sym(var), interner))
                } else if match_one_over_sqrt_one_minus_var_sq(expr, var) {
                    Some(call1("arcsin", Expr::Sym(var), interner))
                } else if let Some(a) = extract_inv_quadratic(expr, var) {
                    Some(Expr::mul(vec![
                        Expr::pow(a.clone(), Expr::Int((-1).into())),
                        call1(
                            "arctan",
                            Expr::mul(vec![Expr::Sym(var), Expr::pow(a, Expr::Int((-1).into()))]),
                            interner,
                        ),
                    ]))
                } else {
                    None
                }
            }
        },
        Expr::Call(f, args) if args.len() == 1 => match (interner.resolve(*f), &args[0]) {
            ("sin", Expr::Sym(s)) if *s == var => {
                Some(Expr::neg(call1("cos", Expr::Sym(var), interner)))
            }
            ("cos", Expr::Sym(s)) if *s == var => Some(call1("sin", Expr::Sym(var), interner)),
            ("exp", Expr::Sym(s)) if *s == var => Some(call1("exp", Expr::Sym(var), interner)),
            ("log", Expr::Sym(s)) if *s == var => Some(Expr::mul(vec![
                Expr::Sym(var),
                Expr::add(vec![
                    call1("log", Expr::Sym(var), interner),
                    Expr::neg(Expr::Int(1.into())),
                ]),
            ])),
            ("tan", Expr::Sym(s)) if *s == var => Some(Expr::neg(call1(
                "log",
                call1("cos", Expr::Sym(var), interner),
                interner,
            ))),
            _ => None,
        },
        Expr::Mul(factors) if factors.len() == 2 => {
            if factors[0] == Expr::Sym(var)
                && is_named_unary_call(&factors[1], "exp", var, interner)
            {
                Some(Expr::mul(vec![
                    Expr::add(vec![Expr::Sym(var), Expr::neg(Expr::one())]),
                    call1("exp", Expr::Sym(var), interner),
                ]))
            } else if factors[1] == Expr::Sym(var)
                && is_named_unary_call(&factors[0], "exp", var, interner)
            {
                Some(Expr::mul(vec![
                    Expr::add(vec![Expr::Sym(var), Expr::neg(Expr::one())]),
                    call1("exp", Expr::Sym(var), interner),
                ]))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn try_u_substitution(
    expr: &Expr,
    var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Mul(factors) = expr else {
        return None;
    };

    for (idx, factor) in factors.iter().enumerate() {
        let (outer_template, inner) = match factor {
            Expr::Call(f, args) if args.len() == 1 && contains_var(&args[0], var) => {
                let u = interner.get_or_intern("u_sub");
                (Expr::Call(*f, vec![Expr::Sym(u)]), args[0].clone())
            }
            Expr::Pow(base, exp) => match base.as_ref() {
                Expr::Call(f, args) if args.len() == 1 && contains_var(&args[0], var) => {
                    let u = interner.get_or_intern("u_sub");
                    (
                        Expr::pow(Expr::Call(*f, vec![Expr::Sym(u)]), exp.as_ref().clone()),
                        args[0].clone(),
                    )
                }
                _ => continue,
            },
            _ => continue,
        };

        let du = differentiate(&inner, var, interner);
        if !contains_var(&du, var) {
            continue;
        }

        let remaining = Expr::mul(
            factors
                .iter()
                .enumerate()
                .filter_map(|(i, term)| if i != idx { Some(term.clone()) } else { None })
                .collect(),
        );

        let Some(scale) = constant_multiple(&remaining, &du, var) else {
            continue;
        };

        let u = interner.get_or_intern("u_sub");
        let integrated_u = integrate(&outer_template, u, interner);
        if is_unevaluated_integrate(&integrated_u, interner) {
            continue;
        }

        return Some(Expr::mul(vec![
            scale,
            substitute_symbol(&integrated_u, u, &inner),
        ]));
    }

    None
}

pub fn try_integration_by_parts(
    expr: &Expr,
    var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Mul(factors) = expr else {
        return None;
    };
    if factors.len() < 2 {
        return None;
    }

    let mut indices = (0..factors.len()).collect::<Vec<_>>();
    indices.sort_by_key(|idx| liate_rank_with_interner(&factors[*idx], interner));

    for idx in indices {
        let u = factors[idx].clone();
        let dv = Expr::mul(
            factors
                .iter()
                .enumerate()
                .filter_map(|(i, term)| if i != idx { Some(term.clone()) } else { None })
                .collect(),
        );

        let v = integrate(&dv, var, interner);
        if is_unevaluated_integrate(&v, interner) {
            continue;
        }

        let du = differentiate(&u, var, interner);
        if du == Expr::zero() {
            continue;
        }

        let remainder = Expr::mul(vec![v.clone(), du]);
        let integrated_remainder = integrate(&remainder, var, interner);
        if is_unevaluated_integrate(&integrated_remainder, interner) {
            continue;
        }

        return Some(Expr::add(vec![
            Expr::mul(vec![u, v]),
            Expr::neg(integrated_remainder),
        ]));
    }

    None
}

pub fn integrate(expr: &ax_ir::Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> ax_ir::Expr {
    if let Some(result) = table_integrate(expr, var, interner) {
        return result;
    }

    if let Expr::Add(terms) = expr {
        return Expr::add(terms.iter().map(|t| integrate(t, var, interner)).collect());
    }

    if let Expr::Mul(factors) = expr {
        let (constant, variable): (Vec<_>, Vec<_>) =
            factors.iter().partition(|f| !contains_var(f, var));
        if !constant.is_empty() && !variable.is_empty() {
            let c = Expr::mul(constant.into_iter().cloned().collect());
            let f = Expr::mul(variable.into_iter().cloned().collect());
            let integrated = integrate(&f, var, interner);
            if !is_unevaluated_integrate(&integrated, interner) {
                return Expr::mul(vec![c, integrated]);
            }
        }
    }

    if let Some(result) = try_u_substitution(expr, var, interner) {
        return result;
    }

    if let Some(result) = try_integration_by_parts(expr, var, interner) {
        return result;
    }

    match expr {
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(integrate(body, var, interner)),
        ),
        Expr::Rule(lhs, rhs) => Expr::Rule(
            Box::new(integrate(lhs, var, interner)),
            Box::new(integrate(rhs, var, interner)),
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (integrate(value, var, interner), condition.clone()))
                .collect(),
        ),
        _ => unevaluated(expr, var, interner),
    }
}

#[cfg(test)]
mod tests {
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
    fn integrate_x_squared() {
        let (e, int) = eval_src("integrate(x^2, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("3"), "got: {}", pp);
    }

    #[test]
    fn integrate_constant() {
        let (e, int) = eval_src("integrate(5, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("5") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn integrate_sin() {
        let (e, int) = eval_src("integrate(sin(x), x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("cos"), "got: {}", pp);
    }

    #[test]
    fn integrate_sum() {
        let (e, int) = eval_src("integrate(x + 1, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn integrate_x_times_exp() {
        let (e, int) = eval_src("integrate(x * exp(x), x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("exp"), "got: {}", pp);
    }

    #[test]
    fn integrate_sin_squared() {
        let (e, int) = eval_src("integrate(sin(x)^2, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn integrate_one_over_one_plus_x_sq() {
        let (e, int) = eval_src("integrate(1/(1 + x^2), x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("arctan") || pp.contains("atan"), "got: {}", pp);
    }
}
