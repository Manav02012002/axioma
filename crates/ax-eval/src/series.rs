use ax_ir::Expr;

fn factorial(n: usize) -> num_bigint::BigInt {
    let mut acc = num_bigint::BigInt::from(1);
    for i in 2..=n {
        acc *= num_bigint::BigInt::from(i);
    }
    acc
}

fn power_base(var: lasso::Spur, point: &Expr) -> Expr {
    if matches!(point, Expr::Int(n) if *n == 0.into()) {
        Expr::Sym(var)
    } else {
        Expr::add(vec![Expr::Sym(var), Expr::neg(point.clone())])
    }
}

pub fn taylor_series(
    expr: &ax_ir::Expr,
    var: lasso::Spur,
    point: &ax_ir::Expr,
    order: usize,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut current = expr.clone();
    let mut terms = Vec::new();

    for n in 0..=order {
        let mut env = crate::Env::new();
        env.bindings.insert(var, point.clone());
        let value_at_point = crate::eval(&current, &env, interner);
        let fact = factorial(n);

        let term = if n == 0 {
            value_at_point
        } else {
            let coeff = Expr::mul(vec![
                value_at_point,
                Expr::Rational(num_rational::BigRational::new(1.into(), fact)),
            ]);

            let base = power_base(var, point);
            let power = if n == 1 {
                base
            } else {
                Expr::pow(base, Expr::Int((n as i64).into()))
            };

            Expr::mul(vec![coeff, power])
        };

        terms.push(term);

        current = crate::differentiate(&current, var, interner);
        current = crate::eval(&current, &crate::Env::new(), interner);
    }

    Expr::add(terms)
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
    fn series_exp_at_zero() {
        let (e, int) = eval_src("series(exp(x), x, 0, 4);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn series_sin_at_zero() {
        let (e, int) = eval_src("series(sin(x), x, 0, 3);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn series_polynomial_is_exact() {
        let (e, int) = eval_src("series(x^2 + 1, x, 0, 3);");
        let direct = crate::eval(
            &ax_core_ir::lower("x^2 + 1;", &int)
                .expr
                .expect("expected expr"),
            &crate::Env::new(),
            &int,
        );
        assert_eq!(
            ax_ir::pretty_print(&e, &int),
            ax_ir::pretty_print(&direct, &int)
        );
    }
}
