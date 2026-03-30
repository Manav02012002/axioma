use crate::contains_var;
use ax_ir::Expr;
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

pub fn integrate(expr: &ax_ir::Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> ax_ir::Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) if !contains_var(expr, var) => {
            Expr::mul(vec![expr.clone(), Expr::Sym(var)])
        }
        Expr::Sym(s) => {
            if *s == var {
                Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), Expr::Int(2.into())),
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                ])
            } else {
                Expr::mul(vec![Expr::Sym(*s), Expr::Sym(var)])
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| integrate(term, var, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(integrate(e, var, interner)),
        Expr::Mul(factors) => {
            let mut consts = Vec::new();
            let mut vars = Vec::new();
            for factor in factors {
                if contains_var(factor, var) {
                    vars.push(factor.clone());
                } else {
                    consts.push(factor.clone());
                }
            }

            if vars.is_empty() {
                Expr::mul(vec![Expr::mul(factors.clone()), Expr::Sym(var)])
            } else if vars.len() == 1 {
                let constant_part = if consts.is_empty() {
                    Expr::one()
                } else {
                    Expr::mul(consts)
                };
                Expr::mul(vec![constant_part, integrate(&vars[0], var, interner)])
            } else {
                unevaluated(expr, var, interner)
            }
        }
        Expr::Pow(base, exp) => match (base.as_ref(), exp.as_ref()) {
            (Expr::Sym(s), Expr::Int(n)) if *s == var && *n != (-1).into() => {
                let next = n.clone() + num_bigint::BigInt::from(1);
                Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), Expr::Int(next.clone())),
                    Expr::Rational(BigRational::new(1.into(), next)),
                ])
            }
            (Expr::Sym(s), Expr::Int(n)) if *s == var && *n == (-1).into() => {
                call1("log", call1("abs", Expr::Sym(var), interner), interner)
            }
            (Expr::Sym(s), Expr::Rational(r))
                if *s == var && *r != BigRational::from_integer((-1).into()) =>
            {
                let next = r.clone() + BigRational::one();
                Expr::mul(vec![
                    Expr::pow(Expr::Sym(var), as_expr(next.clone())),
                    as_expr(BigRational::one() / next),
                ])
            }
            _ => unevaluated(expr, var, interner),
        },
        Expr::Call(f, args) if args.len() == 1 => match (interner.resolve(*f), &args[0]) {
            ("sin", Expr::Sym(s)) if *s == var => Expr::neg(call1("cos", Expr::Sym(var), interner)),
            ("cos", Expr::Sym(s)) if *s == var => call1("sin", Expr::Sym(var), interner),
            ("exp", Expr::Sym(s)) if *s == var => call1("exp", Expr::Sym(var), interner),
            ("log", Expr::Sym(s)) if *s == var => Expr::mul(vec![
                Expr::Sym(var),
                Expr::add(vec![
                    call1("log", Expr::Sym(var), interner),
                    Expr::neg(Expr::Int(1.into())),
                ]),
            ]),
            _ => unevaluated(expr, var, interner),
        },
        _ => unevaluated(expr, var, interner),
    }
}

#[cfg(test)]
mod tests {
    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(result.errors.is_empty(), "lower errors: {:?}", result.errors);
        let expr = result.expr.expect("expected expression");
        let env = crate::Env::new();
        (crate::eval(&expr, &env, &interner), interner)
    }

    #[test]
    fn integrate_power() {
        let (e, int) = eval_src("integrate(x^2, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("3") && pp.contains("x"), "got: {}", pp);
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
}
