#![forbid(unsafe_code)]

use ax_ir::Expr;
use std::collections::HashMap;

fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        Expr::Complex(re, im) => contains_var(re, var) || contains_var(im, var),
        Expr::Sym(sym) => *sym == var,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_var(term, var))
        }
        Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
        Expr::Neg(inner) => contains_var(inner, var),
        Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
        Expr::FnDef(_, _, body) => contains_var(body, var),
        Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => false,
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_var(value, var)),
        Expr::Indexed(base, _) => contains_var(base, var),
        Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_var(cell, var))),
    }
}

fn integrate_call(expr: Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(
        interner.get_or_intern("integrate"),
        vec![expr, Expr::Sym(var)],
    )
}

fn constant_symbol(interner: &ax_ir::Interner) -> Expr {
    Expr::Sym(interner.get_or_intern("C"))
}

fn simplify_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let _ = interner;
    match expr {
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_expr(*re, interner)),
            Box::new(simplify_expr(*im, interner)),
        ),
        Expr::Add(terms) => Expr::add(terms.into_iter().map(|t| simplify_expr(t, interner)).collect()),
        Expr::Mul(factors) => {
            let simplified = factors
                .into_iter()
                .map(|f| simplify_expr(f, interner))
                .collect::<Vec<_>>();
            let mut grouped: Vec<(Expr, usize)> = Vec::new();
            for factor in simplified {
                if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == factor)
                {
                    *count += 1;
                } else {
                    grouped.push((factor, 1));
                }
            }
            Expr::mul(
                grouped
                    .into_iter()
                    .map(|(factor, count)| {
                        if count == 1 {
                            factor
                        } else {
                            Expr::pow(factor, Expr::Int((count as i64).into()))
                        }
                    })
                    .collect(),
            )
        }
        Expr::Pow(base, exp) => Expr::pow(simplify_expr(*base, interner), simplify_expr(*exp, interner)),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner, interner)),
        other => other,
    }
}

fn eval_numeric(expr: &Expr, bindings: &HashMap<lasso::Spur, f64>, interner: &ax_ir::Interner) -> Option<f64> {
    match expr {
        Expr::Int(n) => num_traits::ToPrimitive::to_f64(n),
        Expr::Rational(r) => Some(
            num_traits::ToPrimitive::to_f64(r.numer())?
                / num_traits::ToPrimitive::to_f64(r.denom())?,
        ),
        Expr::Float(f) => Some(*f),
        Expr::Complex(re, im) => {
            let re = eval_numeric(re, bindings, interner)?;
            let im = eval_numeric(im, bindings, interner)?;
            if im == 0.0 { Some(re) } else { None }
        }
        Expr::Sym(sym) => bindings.get(sym).copied(),
        Expr::Add(terms) => {
            let mut acc = 0.0;
            for term in terms {
                acc += eval_numeric(term, bindings, interner)?;
            }
            Some(acc)
        }
        Expr::Mul(factors) => {
            let mut acc = 1.0;
            for factor in factors {
                acc *= eval_numeric(factor, bindings, interner)?;
            }
            Some(acc)
        }
        Expr::Pow(base, exp) => Some(eval_numeric(base, bindings, interner)?.powf(eval_numeric(exp, bindings, interner)?)),
        Expr::Neg(inner) => Some(-eval_numeric(inner, bindings, interner)?),
        Expr::Call(f, args) if args.len() == 1 => {
            let arg = eval_numeric(&args[0], bindings, interner)?;
            match interner.resolve(*f) {
                "exp" => Some(arg.exp()),
                "log" | "ln" => Some(arg.ln()),
                "sin" => Some(arg.sin()),
                "cos" => Some(arg.cos()),
                "tan" => Some(arg.tan()),
                "sqrt" => Some(arg.sqrt()),
                "abs" => Some(arg.abs()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn separable_factors(expr: &Expr, y_sym: lasso::Spur, x_sym: lasso::Spur) -> Option<(Expr, Expr)> {
    let factors = match expr {
        Expr::Mul(factors) => factors.clone(),
        _ => vec![expr.clone()],
    };
    let mut x_terms = Vec::new();
    let mut y_terms = Vec::new();
    for factor in factors {
        let has_x = contains_var(&factor, x_sym);
        let has_y = contains_var(&factor, y_sym);
        match (has_x, has_y) {
            (true, false) => x_terms.push(factor),
            (false, true) => y_terms.push(factor),
            (false, false) => x_terms.push(factor),
            (true, true) => return None,
        }
    }
    if x_terms.is_empty() || y_terms.is_empty() {
        None
    } else {
        Some((Expr::mul(x_terms), Expr::mul(y_terms)))
    }
}

fn solve_separable(
    equation: &Expr,
    y_sym: lasso::Spur,
    x_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if equation == &Expr::Sym(y_sym) {
        return Some(Expr::mul(vec![
            constant_symbol(interner),
            Expr::Call(interner.get_or_intern("exp"), vec![Expr::Sym(x_sym)]),
        ]));
    }

    let (g_x, h_y) = separable_factors(equation, y_sym, x_sym)?;
    if h_y == Expr::Sym(y_sym) {
        return Some(Expr::mul(vec![
            constant_symbol(interner),
            Expr::Call(
                interner.get_or_intern("exp"),
                vec![integrate_call(g_x, x_sym, interner)],
            ),
        ]));
    }

    Some(Expr::add(vec![
        integrate_call(
            Expr::mul(vec![Expr::pow(h_y, Expr::Int((-1).into()))]),
            y_sym,
            interner,
        ),
        Expr::neg(integrate_call(g_x, x_sym, interner)),
        Expr::neg(constant_symbol(interner)),
    ]))
}

fn solve_linear(
    equation: &Expr,
    y_sym: lasso::Spur,
    x_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Add(terms) = equation else {
        return None;
    };
    if terms.len() != 2 {
        return None;
    }

    let (q_term, py_term) = match (&terms[0], &terms[1]) {
        (q, Expr::Neg(inner)) => (q.clone(), inner.as_ref().clone()),
        (Expr::Neg(inner), q) => (q.clone(), inner.as_ref().clone()),
        _ => return None,
    };
    let Expr::Mul(factors) = py_term else {
        return None;
    };
    if factors.len() != 2 {
        return None;
    }
    let p = if factors[0] == Expr::Sym(y_sym) && !contains_var(&factors[1], y_sym) {
        factors[1].clone()
    } else if factors[1] == Expr::Sym(y_sym) && !contains_var(&factors[0], y_sym) {
        factors[0].clone()
    } else {
        return None;
    };
    if contains_var(&q_term, y_sym) || contains_var(&p, y_sym) {
        return None;
    }

    let mu = Expr::Call(
        interner.get_or_intern("exp"),
        vec![integrate_call(p.clone(), x_sym, interner)],
    );
    let solution = Expr::mul(vec![
        Expr::pow(mu.clone(), Expr::Int((-1).into())),
        Expr::add(vec![
            integrate_call(Expr::mul(vec![mu.clone(), q_term]), x_sym, interner),
            constant_symbol(interner),
        ]),
    ]);
    Some(solution)
}

pub fn solve_ode(
    equation: &ax_ir::Expr,
    y_sym: lasso::Spur,
    x_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    if let Some(solution) = solve_separable(equation, y_sym, x_sym, interner) {
        return simplify_expr(solution, interner);
    }
    if let Some(solution) = solve_linear(equation, y_sym, x_sym, interner) {
        return simplify_expr(solution, interner);
    }
    Expr::Call(
        interner.get_or_intern("solve_ode"),
        vec![equation.clone(), Expr::Sym(y_sym), Expr::Sym(x_sym)],
    )
}

pub fn rk4(
    f: &ax_ir::Expr,
    x_sym: lasso::Spur,
    y_sym: lasso::Spur,
    x0: f64,
    y0: f64,
    x_end: f64,
    n_steps: usize,
    interner: &ax_ir::Interner,
) -> Vec<(f64, f64)> {
    if n_steps == 0 {
        return Vec::new();
    }
    let h = (x_end - x0) / n_steps as f64;
    let mut x = x0;
    let mut y = y0;
    let mut out = Vec::with_capacity(n_steps + 1);
    out.push((x, y));

    for _ in 0..n_steps {
        let mut env = HashMap::new();
        env.insert(x_sym, x);
        env.insert(y_sym, y);
        let k1 = match eval_numeric(f, &env, interner) {
            Some(v) => v,
            None => return Vec::new(),
        };

        env.insert(x_sym, x + h / 2.0);
        env.insert(y_sym, y + h * k1 / 2.0);
        let k2 = match eval_numeric(f, &env, interner) {
            Some(v) => v,
            None => return Vec::new(),
        };

        env.insert(y_sym, y + h * k2 / 2.0);
        let k3 = match eval_numeric(f, &env, interner) {
            Some(v) => v,
            None => return Vec::new(),
        };

        env.insert(x_sym, x + h);
        env.insert(y_sym, y + h * k3);
        let k4 = match eval_numeric(f, &env, interner) {
            Some(v) => v,
            None => return Vec::new(),
        };

        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        x += h;
        out.push((x, y));
    }

    out
}

pub fn rk4_system(
    fs: &[ax_ir::Expr],
    x_sym: lasso::Spur,
    y_syms: &[lasso::Spur],
    x0: f64,
    y0s: &[f64],
    x_end: f64,
    n_steps: usize,
    interner: &ax_ir::Interner,
) -> Vec<Vec<f64>> {
    if n_steps == 0 || fs.len() != y_syms.len() || y0s.len() != y_syms.len() {
        return Vec::new();
    }

    let n = y_syms.len();
    let h = (x_end - x0) / n_steps as f64;
    let mut x = x0;
    let mut ys = y0s.to_vec();
    let mut out = Vec::with_capacity(n_steps + 1);
    let mut row = Vec::with_capacity(n + 1);
    row.push(x);
    row.extend_from_slice(&ys);
    out.push(row);

    for _ in 0..n_steps {
        let eval_system = |xv: f64, yv: &[f64]| -> Option<Vec<f64>> {
            let mut bindings = HashMap::new();
            bindings.insert(x_sym, xv);
            for (sym, value) in y_syms.iter().zip(yv.iter()) {
                bindings.insert(*sym, *value);
            }
            fs.iter()
                .map(|expr| eval_numeric(expr, &bindings, interner))
                .collect()
        };

        let k1 = match eval_system(x, &ys) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let yk2 = ys
            .iter()
            .zip(k1.iter())
            .map(|(y, k)| y + h * k / 2.0)
            .collect::<Vec<_>>();
        let k2 = match eval_system(x + h / 2.0, &yk2) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let yk3 = ys
            .iter()
            .zip(k2.iter())
            .map(|(y, k)| y + h * k / 2.0)
            .collect::<Vec<_>>();
        let k3 = match eval_system(x + h / 2.0, &yk3) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let yk4 = ys
            .iter()
            .zip(k3.iter())
            .map(|(y, k)| y + h * k)
            .collect::<Vec<_>>();
        let k4 = match eval_system(x + h, &yk4) {
            Some(v) => v,
            None => return Vec::new(),
        };

        for i in 0..n {
            ys[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        x += h;

        let mut row = Vec::with_capacity(n + 1);
        row.push(x);
        row.extend_from_slice(&ys);
        out.push(row);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_separable_ode() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let equation = Expr::Sym(y);
        let result = solve_ode(&equation, y, x, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("exp"), "got: {}", pp);
    }

    #[test]
    fn rk4_exponential() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let f = Expr::Sym(y);
        let result = rk4(&f, x, y, 0.0, 1.0, 1.0, 100, &interner);
        assert!(!result.is_empty());
        let last = result.last().unwrap();
        assert!((last.1 - std::f64::consts::E).abs() < 0.01, "got y(1) = {}", last.1);
    }
}
