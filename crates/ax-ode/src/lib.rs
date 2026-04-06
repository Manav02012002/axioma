#![forbid(unsafe_code)]

use ax_ir::Expr;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum PdeType {
    Elliptic,
    Parabolic,
    Hyperbolic,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SeparatedSolution {
    pub spatial: ax_ir::Expr,
    pub temporal: ax_ir::Expr,
    pub separation_constant: ax_ir::Expr,
    pub pde_type: PdeType,
}

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
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|t| simplify_expr(t, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let simplified = factors
                .into_iter()
                .map(|f| simplify_expr(f, interner))
                .collect::<Vec<_>>();
            let mut grouped: Vec<(Expr, usize)> = Vec::new();
            for factor in simplified {
                if let Some((_, count)) =
                    grouped.iter_mut().find(|(existing, _)| *existing == factor)
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
        Expr::Pow(base, exp) => Expr::pow(
            simplify_expr(*base, interner),
            simplify_expr(*exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner, interner)),
        other => other,
    }
}

fn expr_to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Neg(inner) => expr_to_rational(inner).map(|r| -r),
        _ => None,
    }
}

fn expr_from_rational(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Int(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn eval_expr_simple(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let _ = interner;
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => expr.clone(),
        Expr::Neg(inner) => {
            let value = eval_expr_simple(inner, interner);
            if let Some(r) = expr_to_rational(&value) {
                expr_from_rational(-r)
            } else {
                Expr::neg(value)
            }
        }
        Expr::Add(terms) => {
            let simplified: Vec<Expr> = terms
                .iter()
                .map(|t| eval_expr_simple(t, interner))
                .collect();
            if simplified.iter().all(|e| expr_to_rational(e).is_some()) {
                let sum = simplified
                    .iter()
                    .filter_map(expr_to_rational)
                    .fold(BigRational::zero(), |acc, r| acc + r);
                expr_from_rational(sum)
            } else {
                Expr::add(simplified)
            }
        }
        Expr::Mul(factors) => {
            let simplified: Vec<Expr> = factors
                .iter()
                .map(|f| eval_expr_simple(f, interner))
                .collect();
            if simplified.iter().all(|e| expr_to_rational(e).is_some()) {
                let product = simplified
                    .iter()
                    .filter_map(expr_to_rational)
                    .fold(BigRational::one(), |acc, r| acc * r);
                expr_from_rational(product)
            } else {
                Expr::mul(simplified)
            }
        }
        Expr::Pow(base, exp) => {
            let base_eval = eval_expr_simple(base, interner);
            let exp_eval = eval_expr_simple(exp, interner);
            match (expr_to_rational(&base_eval), &exp_eval) {
                (Some(b), Expr::Int(n)) => {
                    if n.is_zero() {
                        Expr::one()
                    } else if let Some(pow) = num_traits::ToPrimitive::to_u32(n) {
                        let numer = b.numer().clone().pow(pow);
                        let denom = b.denom().clone().pow(pow);
                        expr_from_rational(BigRational::new(numer, denom))
                    } else if n.is_negative() {
                        let pow = num_traits::ToPrimitive::to_u32(&(-n.clone())).unwrap_or(0);
                        let numer = b.denom().clone().pow(pow);
                        let denom = b.numer().clone().pow(pow);
                        expr_from_rational(BigRational::new(numer, denom))
                    } else {
                        Expr::pow(base_eval, exp_eval)
                    }
                }
                _ => Expr::pow(base_eval, exp_eval),
            }
        }
        _ => expr.clone(),
    }
}

pub fn classify_pde(
    a: &ax_ir::Expr,
    b: &ax_ir::Expr,
    c: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> PdeType {
    let b_sq = ax_ir::Expr::pow(b.clone(), ax_ir::Expr::Int(2.into()));
    let ac = ax_ir::Expr::mul(vec![a.clone(), c.clone()]);
    let disc = ax_ir::Expr::add(vec![b_sq, ax_ir::Expr::neg(ac)]);
    let disc_eval = eval_expr_simple(&disc, interner);

    match &disc_eval {
        ax_ir::Expr::Int(n) => {
            if n.is_negative() {
                PdeType::Elliptic
            } else if n.is_zero() {
                PdeType::Parabolic
            } else {
                PdeType::Hyperbolic
            }
        }
        ax_ir::Expr::Float(v) => {
            if *v < -1e-12 {
                PdeType::Elliptic
            } else if v.abs() < 1e-12 {
                PdeType::Parabolic
            } else {
                PdeType::Hyperbolic
            }
        }
        ax_ir::Expr::Rational(r) => {
            if r.is_negative() {
                PdeType::Elliptic
            } else if r.is_zero() {
                PdeType::Parabolic
            } else {
                PdeType::Hyperbolic
            }
        }
        _ => PdeType::Unknown,
    }
}

pub fn separate_variables(
    pde_type: PdeType,
    spatial_var: lasso::Spur,
    temporal_var: lasso::Spur,
    coefficient: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> SeparatedSolution {
    let k = interner.get_or_intern("k");
    let a_const = interner.get_or_intern("A");
    let b_const = interner.get_or_intern("B");
    let c_const = interner.get_or_intern("C");
    let d_const = interner.get_or_intern("D");

    let sin_sym = interner.get_or_intern("sin");
    let cos_sym = interner.get_or_intern("cos");
    let exp_sym = interner.get_or_intern("exp");

    match pde_type {
        PdeType::Hyperbolic => {
            let spatial = Expr::add(vec![
                Expr::mul(vec![
                    Expr::Sym(a_const),
                    Expr::Call(
                        sin_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
                Expr::mul(vec![
                    Expr::Sym(b_const),
                    Expr::Call(
                        cos_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
            ]);

            let ck = Expr::mul(vec![coefficient.clone(), Expr::Sym(k)]);
            let temporal = Expr::add(vec![
                Expr::mul(vec![
                    Expr::Sym(c_const),
                    Expr::Call(
                        sin_sym,
                        vec![Expr::mul(vec![ck.clone(), Expr::Sym(temporal_var)])],
                    ),
                ]),
                Expr::mul(vec![
                    Expr::Sym(d_const),
                    Expr::Call(cos_sym, vec![Expr::mul(vec![ck, Expr::Sym(temporal_var)])]),
                ]),
            ]);

            SeparatedSolution {
                spatial,
                temporal,
                separation_constant: Expr::pow(Expr::Sym(k), Expr::Int(2.into())),
                pde_type: PdeType::Hyperbolic,
            }
        }
        PdeType::Parabolic => {
            let spatial = Expr::add(vec![
                Expr::mul(vec![
                    Expr::Sym(a_const),
                    Expr::Call(
                        sin_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
                Expr::mul(vec![
                    Expr::Sym(b_const),
                    Expr::Call(
                        cos_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
            ]);

            let decay = Expr::neg(Expr::mul(vec![
                coefficient.clone(),
                Expr::pow(Expr::Sym(k), Expr::Int(2.into())),
                Expr::Sym(temporal_var),
            ]));
            let temporal = Expr::Call(exp_sym, vec![decay]);

            SeparatedSolution {
                spatial,
                temporal,
                separation_constant: Expr::pow(Expr::Sym(k), Expr::Int(2.into())),
                pde_type: PdeType::Parabolic,
            }
        }
        PdeType::Elliptic => {
            let spatial = Expr::add(vec![
                Expr::mul(vec![
                    Expr::Sym(a_const),
                    Expr::Call(
                        sin_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
                Expr::mul(vec![
                    Expr::Sym(b_const),
                    Expr::Call(
                        cos_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(spatial_var)])],
                    ),
                ]),
            ]);

            let sinh_sym = interner.get_or_intern("sinh");
            let cosh_sym = interner.get_or_intern("cosh");
            let temporal = Expr::add(vec![
                Expr::mul(vec![
                    Expr::Sym(c_const),
                    Expr::Call(
                        sinh_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(temporal_var)])],
                    ),
                ]),
                Expr::mul(vec![
                    Expr::Sym(d_const),
                    Expr::Call(
                        cosh_sym,
                        vec![Expr::mul(vec![Expr::Sym(k), Expr::Sym(temporal_var)])],
                    ),
                ]),
            ]);

            SeparatedSolution {
                spatial,
                temporal,
                separation_constant: Expr::pow(Expr::Sym(k), Expr::Int(2.into())),
                pde_type: PdeType::Elliptic,
            }
        }
        PdeType::Unknown => SeparatedSolution {
            spatial: Expr::Sym(interner.get_or_intern("X")),
            temporal: Expr::Sym(interner.get_or_intern("T")),
            separation_constant: Expr::Sym(interner.get_or_intern("lambda")),
            pde_type: PdeType::Unknown,
        },
    }
}

fn eval_numeric(
    expr: &Expr,
    bindings: &HashMap<lasso::Spur, f64>,
    interner: &ax_ir::Interner,
) -> Option<f64> {
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
            if im == 0.0 {
                Some(re)
            } else {
                None
            }
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
        Expr::Pow(base, exp) => Some(
            eval_numeric(base, bindings, interner)?.powf(eval_numeric(exp, bindings, interner)?),
        ),
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

// ─── first_order_form ────────────────────────────────────────────────────────

/// Convert a second-order (or higher-order) ODE into a system of first-order
/// ODEs by introducing auxiliary variables.
///
/// Given `x'' = f(t, x, x')`, introduce `v = x'` and return:
///   `{ x' = v,  v' = f(t, x, v) }`
///
/// The `ode` argument may be either:
///   - The full ODE expression containing `diff(...)` calls, in which case the
///     highest derivative order is detected automatically; or
///   - Just the right-hand side after isolating the highest derivative
///     (no `diff` calls present), in which case a 2nd-order system is assumed.
///
/// Returns a `Vec<(lhs_var, rhs_expr)>` where `lhs_var` is the symbol whose
/// derivative equals `rhs_expr`.
pub fn first_order_form(
    ode: &Expr,
    dependent_var: lasso::Spur,
    independent_var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Vec<(Expr, Expr)> {
    let diff_sym = interner.get_or_intern("diff");
    let detected_order = find_max_derivative_order(ode, dependent_var, independent_var, diff_sym);

    // If no diff() calls are present, treat the expression as the rhs of a
    // 2nd-order ODE (the most common calling convention for this function).
    let max_order = if detected_order == 0 {
        2
    } else {
        detected_order
    };

    if max_order <= 1 {
        // Already first order — nothing to do.
        return vec![];
    }

    // Build auxiliary variable names: v0 = x, v1 = x', v2 = x'', …
    let mut aux_vars: Vec<lasso::Spur> = Vec::new();
    aux_vars.push(dependent_var);
    for i in 1..max_order {
        let name = format!("{}_d{}", interner.resolve(dependent_var), i);
        aux_vars.push(interner.get_or_intern(&name));
    }

    // First n-1 equations: v_{i}' = v_{i+1}
    let mut system: Vec<(Expr, Expr)> = Vec::new();
    for i in 0..max_order - 1 {
        system.push((Expr::Sym(aux_vars[i]), Expr::Sym(aux_vars[i + 1])));
    }

    // Last equation: v_{n-1}' = rhs
    // Substitute each derivative diff^k(x, t) → aux_vars[k] in the ode,
    // from the highest order down so inner substitutions don't mis-match.
    let mut rhs = ode.clone();
    for i in (1..max_order).rev() {
        let deriv_expr = make_nth_derivative(dependent_var, independent_var, i, diff_sym);
        rhs = substitute_expr(&rhs, &deriv_expr, &Expr::Sym(aux_vars[i]));
    }
    // Also substitute x itself (order 0) with aux_vars[0] (same symbol, no-op,
    // but keeps the code uniform if dependent_var appears symbolically).
    // aux_vars[0] == dependent_var so this is a no-op.
    system.push((Expr::Sym(aux_vars[max_order - 1]), rhs));

    system
}

/// Recursively find the maximum derivative order of `var` with respect to
/// `indep` in `expr`, counting nested `diff()` calls.
fn find_max_derivative_order(
    expr: &Expr,
    var: lasso::Spur,
    indep: lasso::Spur,
    diff_sym: lasso::Spur,
) -> usize {
    let mut max_order = 0;
    walk_derivative_order(expr, var, indep, diff_sym, 0, &mut max_order);
    max_order
}

fn walk_derivative_order(
    expr: &Expr,
    var: lasso::Spur,
    indep: lasso::Spur,
    diff_sym: lasso::Spur,
    depth: usize,
    max: &mut usize,
) {
    match expr {
        Expr::Call(f, args) if *f == diff_sym => {
            if args.len() >= 2 {
                // Check that the second argument is the independent variable.
                let wrt_matches = matches!(&args[1], Expr::Sym(s) if *s == indep);
                let new_depth = if wrt_matches { depth + 1 } else { depth };
                // If the inner expression is the dependent variable, record depth.
                if let Expr::Sym(s) = &args[0] {
                    if *s == var && new_depth > *max {
                        *max = new_depth;
                    }
                }
                // Recurse into the inner expression with the updated depth.
                walk_derivative_order(&args[0], var, indep, diff_sym, new_depth, max);
                // Also walk remaining args (e.g. the wrt argument itself) at depth 0.
                for arg in &args[2..] {
                    walk_derivative_order(arg, var, indep, diff_sym, 0, max);
                }
            }
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for t in terms {
                walk_derivative_order(t, var, indep, diff_sym, 0, max);
            }
        }
        Expr::Neg(e) | Expr::Pow(e, _) => {
            walk_derivative_order(e, var, indep, diff_sym, 0, max);
        }
        Expr::Call(_, args) => {
            for arg in args {
                walk_derivative_order(arg, var, indep, diff_sym, 0, max);
            }
        }
        _ => {}
    }
}

/// Build `diff(diff(…diff(var, indep)…, indep), indep)` nested `n` times.
fn make_nth_derivative(
    var: lasso::Spur,
    indep: lasso::Spur,
    n: usize,
    diff_sym: lasso::Spur,
) -> Expr {
    let mut result = Expr::Sym(var);
    for _ in 0..n {
        result = Expr::Call(diff_sym, vec![result, Expr::Sym(indep)]);
    }
    result
}

/// Simple structural substitution: replace every occurrence of `target` in
/// `expr` with `replacement`.  Does not use ax-eval to avoid a circular
/// dependency (ax-eval depends on ax-ode via ax-tensor).
fn substitute_expr(expr: &Expr, target: &Expr, replacement: &Expr) -> Expr {
    if expr == target {
        return replacement.clone();
    }
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| substitute_expr(t, target, replacement))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| substitute_expr(f, target, replacement))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_expr(base, target, replacement),
            substitute_expr(exp, target, replacement),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_expr(inner, target, replacement)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| substitute_expr(a, target, replacement))
                .collect(),
        ),
        other => other.clone(),
    }
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
        assert!(
            (last.1 - std::f64::consts::E).abs() < 0.01,
            "got y(1) = {}",
            last.1
        );
    }

    #[test]
    fn first_order_simple() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");

        // x'' = -x  → system: { x' = x_d1,  x_d1' = -x }
        // The ode is just the rhs (-x), so first_order_form assumes order 2.
        let ode = Expr::neg(Expr::Sym(x));
        let system = first_order_form(&ode, x, t, &interner);
        assert_eq!(system.len(), 2, "expected 2 first-order equations");

        // First equation: x' = x_d1
        let x_d1 = interner.get_or_intern("x_d1");
        assert_eq!(system[0].0, Expr::Sym(x));
        assert_eq!(system[0].1, Expr::Sym(x_d1));

        // Last equation: x_d1' = -x
        assert_eq!(system[1].0, Expr::Sym(x_d1));
        assert_eq!(system[1].1, Expr::neg(Expr::Sym(x)));
    }

    #[test]
    fn first_order_with_diff_calls() {
        // Pass the full ODE: diff(diff(x, t), t) + x = 0,
        // represented as diff(diff(x,t),t) = -x, i.e. ode = diff^2(x,t) - (-x)
        // More simply: pass the expression diff(diff(x,t),t) so that
        // find_max_derivative_order detects order 2.
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");
        let diff_sym = interner.get_or_intern("diff");

        // Represent the full ODE: diff(diff(x,t),t) = -x
        // Pass as the Add form: diff²(x,t) + x = 0  → ode = diff²(x,t) + x
        let d1 = Expr::Call(diff_sym, vec![Expr::Sym(x), Expr::Sym(t)]);
        let d2 = Expr::Call(diff_sym, vec![d1.clone(), Expr::Sym(t)]);
        let ode = Expr::add(vec![d2, Expr::Sym(x)]);

        let system = first_order_form(&ode, x, t, &interner);
        assert_eq!(
            system.len(),
            2,
            "expected 2 first-order equations from diff form"
        );

        let x_d1 = interner.get_or_intern("x_d1");
        assert_eq!(system[0].0, Expr::Sym(x));
        assert_eq!(system[0].1, Expr::Sym(x_d1));
    }

    #[test]
    fn first_order_already_first_order() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");
        let diff_sym = interner.get_or_intern("diff");

        // diff(x, t) = -x  → max_order = 1 → return empty
        let ode = Expr::Call(diff_sym, vec![Expr::Sym(x), Expr::Sym(t)]);
        let system = first_order_form(&ode, x, t, &interner);
        assert!(
            system.is_empty(),
            "first-order ODE should return empty system"
        );
    }

    #[test]
    fn first_order_third_order() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");
        let diff_sym = interner.get_or_intern("diff");

        // x''' = x  → 3rd-order → 3 equations
        let d1 = Expr::Call(diff_sym, vec![Expr::Sym(x), Expr::Sym(t)]);
        let d2 = Expr::Call(diff_sym, vec![d1, Expr::Sym(t)]);
        let d3 = Expr::Call(diff_sym, vec![d2, Expr::Sym(t)]);
        let ode = d3; // full lhs; rhs is x

        let system = first_order_form(&ode, x, t, &interner);
        assert_eq!(system.len(), 3, "3rd-order ODE should give 3 equations");
    }

    #[test]
    fn classify_wave_equation() {
        let interner = ax_ir::Interner::new();
        let result = classify_pde(
            &Expr::Int(1.into()),
            &Expr::Int(0.into()),
            &Expr::Int((-1i64).into()),
            &interner,
        );
        assert_eq!(result, PdeType::Hyperbolic);
    }

    #[test]
    fn classify_heat_equation() {
        let interner = ax_ir::Interner::new();
        let result = classify_pde(
            &Expr::Int(1.into()),
            &Expr::Int(0.into()),
            &Expr::Int(0.into()),
            &interner,
        );
        assert_eq!(result, PdeType::Parabolic);
    }

    #[test]
    fn classify_laplace() {
        let interner = ax_ir::Interner::new();
        let result = classify_pde(
            &Expr::Int(1.into()),
            &Expr::Int(0.into()),
            &Expr::Int(1.into()),
            &interner,
        );
        assert_eq!(result, PdeType::Elliptic);
    }

    #[test]
    fn separate_wave() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");
        let c = interner.get_or_intern("c");
        let sol = separate_variables(PdeType::Hyperbolic, x, t, &Expr::Sym(c), &interner);
        assert_eq!(sol.pde_type, PdeType::Hyperbolic);
        let pp = ax_ir::pretty_print(&sol.spatial, &interner);
        assert!(
            pp.contains("sin") && pp.contains("cos"),
            "spatial should have sin and cos: {}",
            pp
        );
    }
}
