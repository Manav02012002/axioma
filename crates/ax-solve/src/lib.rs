#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LindbladSteadyStateError {
    #[error("Hamiltonian not square: rows={rows}, cols={cols}")]
    HamiltonianNotSquare { rows: usize, cols: usize },
    #[error("jump operator {index} not square: rows={rows}, cols={cols}")]
    JumpOperatorNotSquare {
        index: usize,
        rows: usize,
        cols: usize,
    },
    #[error("dimension mismatch for {which}: expected={expected}, actual={actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        which: &'static str,
    },
    #[error("underdetermined steady state")]
    UnderdeterminedSteadyState,
    #[error("inconsistent steady state system")]
    InconsistentSteadyStateSystem,
}

pub fn can_solve_componentwise_by_symmetry(sym: &ax_ir::TensorSymmetry) -> bool {
    !sym.tableaux.is_empty()
}

fn matrix_shape(matrix: &[Vec<Expr>]) -> Option<(usize, usize)> {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    matrix
        .iter()
        .all(|row| row.len() == cols)
        .then_some((rows, cols))
}

static RHO_SYMBOL_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Group(inner, _) => to_rational(inner),
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

fn rational_to_f64(r: &BigRational) -> f64 {
    r.numer().to_f64().unwrap_or(0.0) / r.denom().to_f64().unwrap_or(1.0)
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}

fn rationalize_f64(x: f64) -> Expr {
    if !x.is_finite() {
        return Expr::Float(x);
    }
    let nearest = x.round();
    if (x - nearest).abs() < 1e-10 {
        return Expr::Int((nearest as i64).into());
    }

    let rounded = (x * 1000.0).round() / 1000.0;
    if (x - rounded).abs() < 1e-10 {
        let numer = (rounded * 1000.0).round() as i64;
        let denom = 1000i64;
        let g = gcd_i64(numer.abs(), denom);
        if denom / g == 1 {
            Expr::Int((numer / g).into())
        } else {
            Expr::Rational(BigRational::new((numer / g).into(), (denom / g).into()))
        }
    } else {
        Expr::Float(x)
    }
}

fn divide_exprs_as_rational(numer: &Expr, denom: &Expr) -> BigRational {
    let n = match numer {
        Expr::Int(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(r) => r.clone(),
        _ => BigRational::zero(),
    };
    let d = match denom {
        Expr::Int(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(r) => r.clone(),
        _ => BigRational::one(),
    };
    n / d
}

fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero()) || matches!(expr, Expr::Rational(r) if r.is_zero())
}

fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        Expr::Complex(re, im) => contains_var(re, var) || contains_var(im, var),
        Expr::Sym(s) => *s == var,
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
            items.iter().any(|item| contains_var(item, var))
        }
        Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
        Expr::Neg(inner) => contains_var(inner, var),
        Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
        Expr::FnDef(_, _, body) => contains_var(body, var),
        Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::Import(_) => false,
        Expr::Assume(_, _) => false,
        Expr::SetConvention(_, _) => false,
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_var(value, var)),
        Expr::Indexed(base, _) => contains_var(base, var),
        Expr::Group(inner, _) => contains_var(inner, var),
        Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_var(cell, var))),
    }
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = to_rational(base)?;
    match exp {
        Expr::Int(n) => {
            if let Some(pow) = n.to_u32() {
                let numer = base_r.numer().clone().pow(pow);
                let denom = base_r.denom().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                Some(expr_from_rational(out))
            } else if n.is_negative() {
                let pow = (-n).to_u32()?;
                let numer = base_r.denom().clone().pow(pow);
                let denom = base_r.numer().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                Some(expr_from_rational(out))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn perfect_square_root(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    let root = n.sqrt();
    if &root * &root == *n {
        Some(root)
    } else {
        None
    }
}

fn simplify_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => expr,
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_expr(*re, interner)),
            Box::new(simplify_expr(*im, interner)),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| simplify_expr(term, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .into_iter()
                .map(|factor| simplify_expr(factor, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => {
            let base = simplify_expr(*base, interner);
            let exp = simplify_expr(*exp, interner);
            if let Some(out) = numeric_pow(&base, &exp) {
                out
            } else {
                Expr::pow(base, exp)
            }
        }
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner, interner)),
        Expr::Group(inner, rel) => Expr::Group(Box::new(simplify_expr(*inner, interner)), rel),
        Expr::Call(f, args) => {
            let args = args
                .into_iter()
                .map(|arg| simplify_expr(arg, interner))
                .collect::<Vec<_>>();
            match (interner.resolve(f), args.as_slice()) {
                ("sqrt", [Expr::Int(n)]) => {
                    if let Some(root) = perfect_square_root(n) {
                        Expr::Int(root)
                    } else {
                        Expr::Call(f, args)
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(name, params, Box::new(simplify_expr(*body, interner)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(simplify_expr(*lhs, interner)),
            Box::new(simplify_expr(*rhs, interner)),
            trust,
        ),
        Expr::Import(path) => Expr::Import(path),
        Expr::Assume(name, assumptions) => Expr::Assume(name, assumptions),
        Expr::SetConvention(field, value) => Expr::SetConvention(field, value),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| (simplify_expr(value, interner), condition))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(simplify_expr(*base, interner)), indices)
        }
        Expr::Let(name, val, body) => Expr::Let(
            name,
            Box::new(simplify_expr(*val, interner)),
            Box::new(simplify_expr(*body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| simplify_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| simplify_expr(cell, interner))
                        .collect()
                })
                .collect(),
        ),
    }
}

fn trim_coeffs(coeffs: &mut Vec<Expr>) {
    while coeffs.len() > 1 && coeffs.last().is_some_and(is_zero) {
        coeffs.pop();
    }
}

fn poly_add(mut lhs: Vec<Expr>, rhs: Vec<Expr>) -> Vec<Expr> {
    if lhs.len() < rhs.len() {
        lhs.resize(rhs.len(), Expr::zero());
    }
    for (idx, coeff) in rhs.into_iter().enumerate() {
        lhs[idx] = Expr::add(vec![lhs[idx].clone(), coeff]);
    }
    trim_coeffs(&mut lhs);
    lhs
}

fn poly_neg(coeffs: Vec<Expr>) -> Vec<Expr> {
    coeffs.into_iter().map(Expr::neg).collect()
}

fn poly_mul(lhs: Vec<Expr>, rhs: Vec<Expr>) -> Vec<Expr> {
    if lhs.is_empty() || rhs.is_empty() {
        return vec![Expr::zero()];
    }
    let mut out = vec![Expr::zero(); lhs.len() + rhs.len() - 1];
    for (i, a) in lhs.iter().enumerate() {
        for (j, b) in rhs.iter().enumerate() {
            out[i + j] = Expr::add(vec![
                out[i + j].clone(),
                Expr::mul(vec![a.clone(), b.clone()]),
            ]);
        }
    }
    trim_coeffs(&mut out);
    out
}

fn poly_pow(base: Vec<Expr>, n: usize) -> Vec<Expr> {
    let mut out = vec![Expr::one()];
    for _ in 0..n {
        out = poly_mul(out, base.clone());
    }
    out
}

pub fn extract_polynomial(
    expr: &ax_ir::Expr,
    var: lasso::Spur,
    _interner: &ax_ir::Interner,
) -> Option<Vec<ax_ir::Expr>> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Some(vec![expr.clone()]),
        Expr::Complex(_, _) => None,
        Expr::Sym(sym) => {
            if *sym == var {
                Some(vec![Expr::zero(), Expr::one()])
            } else {
                Some(vec![expr.clone()])
            }
        }
        Expr::Add(terms) => {
            let mut out = vec![Expr::zero()];
            for term in terms {
                out = poly_add(out, extract_polynomial(term, var, _interner)?);
            }
            Some(out)
        }
        Expr::Neg(inner) => Some(poly_neg(extract_polynomial(inner, var, _interner)?)),
        Expr::Mul(factors) => {
            let mut out = vec![Expr::one()];
            for factor in factors {
                out = poly_mul(out, extract_polynomial(factor, var, _interner)?);
            }
            Some(out)
        }
        Expr::Pow(base, exp) => {
            if !contains_var(expr, var) {
                return Some(vec![expr.clone()]);
            }
            if let Expr::Int(n) = exp.as_ref() {
                if let Some(pow) = n.to_usize() {
                    return Some(poly_pow(extract_polynomial(base, var, _interner)?, pow));
                }
            }
            None
        }
        Expr::Call(_, _)
        | Expr::FnDef(_, _, _)
        | Expr::Rule(_, _, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Piecewise(_)
        | Expr::Indexed(_, _)
        | Expr::Group(_, _)
        | Expr::Let(_, _, _)
        | Expr::List(_)
        | Expr::Matrix(_) => {
            if contains_var(expr, var) {
                None
            } else {
                Some(vec![expr.clone()])
            }
        }
    }
}

fn divisors(n: &BigInt) -> Vec<BigInt> {
    let Some(limit) = n.abs().to_i64() else {
        return Vec::new();
    };
    if limit == 0 {
        return vec![0.into()];
    }
    let mut out = Vec::new();
    for i in 1..=limit {
        if limit % i == 0 {
            out.push(BigInt::from(i));
        }
    }
    out
}

fn eval_poly_at(coeffs: &[Expr], x: &BigRational) -> Option<BigRational> {
    let mut acc = BigRational::zero();
    for coeff in coeffs.iter().rev() {
        let c = to_rational(coeff)?;
        acc = c + x.clone() * acc;
    }
    Some(acc)
}

fn rational_roots(
    coeffs: &[ax_ir::Expr],
    _var: lasso::Spur,
    _interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    if coeffs.len() < 2 {
        return Vec::new();
    }
    let Some(constant) = to_rational(&coeffs[0]) else {
        return Vec::new();
    };
    let Some(leading) = coeffs.last().and_then(to_rational) else {
        return Vec::new();
    };
    if !constant.is_integer() || !leading.is_integer() {
        return Vec::new();
    }

    let p_factors = divisors(&constant.to_integer());
    let q_factors = divisors(&leading.to_integer());
    let mut roots = Vec::new();

    for p in p_factors {
        for q in &q_factors {
            if q.is_zero() {
                continue;
            }
            let candidate = BigRational::new(p.clone(), q.clone());
            for signed in [candidate.clone(), -candidate] {
                if eval_poly_at(coeffs, &signed).is_some_and(|value| value.is_zero()) {
                    let expr = expr_from_rational(signed);
                    if !roots.contains(&expr) {
                        roots.push(expr);
                    }
                }
            }
        }
    }

    roots
}

pub fn poly_divide(
    coeffs: &[ax_ir::Expr],
    root: &ax_ir::Expr,
    _interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let Some(root_r) = to_rational(root) else {
        return Vec::new();
    };
    let mut descending = coeffs
        .iter()
        .map(to_rational)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if descending.is_empty() {
        return Vec::new();
    }
    descending.reverse();

    let mut accum = Vec::with_capacity(descending.len());
    let mut current = descending[0].clone();
    accum.push(current.clone());
    for coeff in descending.iter().skip(1) {
        current = coeff.clone() + root_r.clone() * current;
        accum.push(current.clone());
    }
    accum.pop();
    accum.reverse();
    accum.into_iter().map(expr_from_rational).collect()
}

fn solve_cubic(coeffs: &[Expr], _interner: &ax_ir::Interner) -> Vec<Expr> {
    let a = divide_exprs_as_rational(&coeffs[2], &coeffs[3]);
    let b = divide_exprs_as_rational(&coeffs[1], &coeffs[3]);
    let c = divide_exprs_as_rational(&coeffs[0], &coeffs[3]);

    let p = &b - &(&a * &a) / BigRational::from_integer(3.into());
    let q = &c - &(&a * &b) / BigRational::from_integer(3.into())
        + BigRational::from_integer(2.into()) * &(&a * &a * &a)
            / BigRational::from_integer(27.into());
    let disc = -(BigRational::from_integer(4.into()) * &p * &p * &p
        + BigRational::from_integer(27.into()) * &q * &q);

    let p_f = rational_to_f64(&p);
    let q_f = rational_to_f64(&q);
    let a_f = rational_to_f64(&a);

    if disc > BigRational::zero() {
        let m = (-p_f / 3.0).sqrt();
        let cos_arg = ((-q_f / 2.0) / (m * m * m)).clamp(-1.0, 1.0);
        let theta = cos_arg.acos() / 3.0;
        (0..3)
            .map(|k| {
                let t = 2.0 * m * (theta + 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos();
                let x = t - a_f / 3.0;
                rationalize_f64(x)
            })
            .collect()
    } else {
        let inner = (q_f / 2.0).powi(2) + (p_f / 3.0).powi(3);
        if inner >= 0.0 {
            let sqrt_inner = inner.sqrt();
            let u = (-q_f / 2.0 + sqrt_inner).cbrt();
            let v = (-q_f / 2.0 - sqrt_inner).cbrt();
            let t = u + v;
            let x = t - a_f / 3.0;
            vec![rationalize_f64(x)]
        } else {
            let m = (-p_f / 3.0).sqrt();
            let cos_arg = (-q_f / (2.0 * m.powi(3))).clamp(-1.0, 1.0);
            let theta = cos_arg.acos() / 3.0;
            let t = 2.0 * m * theta.cos();
            let x = t - a_f / 3.0;
            vec![rationalize_f64(x)]
        }
    }
}

fn eval_numeric_at(
    expr: &Expr,
    var: lasso::Spur,
    val: f64,
    interner: &ax_ir::Interner,
) -> Option<f64> {
    match expr {
        Expr::Int(n) => n.to_f64(),
        Expr::Rational(r) => Some(rational_to_f64(r)),
        Expr::Float(v) => Some(*v),
        Expr::Sym(s) => {
            if *s == var {
                Some(val)
            } else if interner.resolve(*s) == "pi" {
                Some(std::f64::consts::PI)
            } else if interner.resolve(*s) == "e" {
                Some(std::f64::consts::E)
            } else {
                None
            }
        }
        Expr::Add(terms) => {
            let mut acc = 0.0;
            for term in terms {
                acc += eval_numeric_at(term, var, val, interner)?;
            }
            Some(acc)
        }
        Expr::Mul(factors) => {
            let mut acc = 1.0;
            for factor in factors {
                acc *= eval_numeric_at(factor, var, val, interner)?;
            }
            Some(acc)
        }
        Expr::Pow(base, exp) => {
            let b = eval_numeric_at(base, var, val, interner)?;
            let e = eval_numeric_at(exp, var, val, interner)?;
            Some(b.powf(e))
        }
        Expr::Neg(inner) => Some(-eval_numeric_at(inner, var, val, interner)?),
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            match (name, args.as_slice()) {
                ("sin", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.sin()),
                ("cos", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.cos()),
                ("tan", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.tan()),
                ("exp", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.exp()),
                ("log", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.ln()),
                ("sqrt", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.sqrt()),
                ("sinh", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.sinh()),
                ("cosh", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.cosh()),
                ("tanh", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.tanh()),
                ("asin" | "arcsin", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.asin())
                }
                ("acos" | "arccos", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.acos())
                }
                ("atan" | "arctan", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.atan())
                }
                ("sec", [arg]) => Some(1.0 / eval_numeric_at(arg, var, val, interner)?.cos()),
                ("csc", [arg]) => Some(1.0 / eval_numeric_at(arg, var, val, interner)?.sin()),
                ("cot", [arg]) => Some(1.0 / eval_numeric_at(arg, var, val, interner)?.tan()),
                ("asinh" | "arcsinh", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.asinh())
                }
                ("acosh" | "arccosh", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.acosh())
                }
                ("atanh" | "arctanh", [arg]) => {
                    Some(eval_numeric_at(arg, var, val, interner)?.atanh())
                }
                ("abs", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.abs()),
                ("sign" | "sgn", [arg]) => Some(eval_numeric_at(arg, var, val, interner)?.signum()),
                ("atan2", [y, x]) => Some(
                    eval_numeric_at(y, var, val, interner)?
                        .atan2(eval_numeric_at(x, var, val, interner)?),
                ),
                _ => None,
            }
        }
        _ => None,
    }
}

fn solve_numerical(
    expr: &Expr,
    var: lasso::Spur,
    interner: &ax_ir::Interner,
    initial_guess: f64,
) -> Option<Expr> {
    let mut x = initial_guess;
    for _ in 0..100 {
        let fx = eval_numeric_at(expr, var, x, interner)?;
        let h = 1e-6_f64.max(x.abs() * 1e-6);
        let dfx = (eval_numeric_at(expr, var, x + h, interner)?
            - eval_numeric_at(expr, var, x - h, interner)?)
            / (2.0 * h);
        if !fx.is_finite() || !dfx.is_finite() || dfx.abs() < 1e-15 {
            return None;
        }
        let x_new = x - fx / dfx;
        if (x_new - x).abs() < 1e-12 {
            return Some(rationalize_f64(x_new));
        }
        x = x_new;
    }
    None
}

fn same_numeric_root(lhs: &Expr, rhs: &Expr) -> bool {
    match (lhs, rhs) {
        (Expr::Int(a), Expr::Int(b)) => a == b,
        (Expr::Rational(a), Expr::Rational(b)) => a == b,
        _ => {
            let l = match lhs {
                Expr::Int(n) => n.to_f64().unwrap_or(f64::NAN),
                Expr::Rational(r) => rational_to_f64(r),
                Expr::Float(v) => *v,
                _ => f64::NAN,
            };
            let r = match rhs {
                Expr::Int(n) => n.to_f64().unwrap_or(f64::NAN),
                Expr::Rational(rr) => rational_to_f64(rr),
                Expr::Float(v) => *v,
                _ => f64::NAN,
            };
            (l - r).abs() < 1e-8
        }
    }
}

fn solve_from_coeffs(
    equation: &Expr,
    coeffs: &[Expr],
    var: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut coeffs = coeffs.to_vec();
    trim_coeffs(&mut coeffs);
    let degree = coeffs.len().saturating_sub(1);

    match degree {
        0 => Expr::List(Vec::new()),
        1 => {
            let a0 = coeffs[0].clone();
            let a1 = coeffs[1].clone();
            let root = simplify_expr(
                Expr::mul(vec![Expr::neg(a0), Expr::pow(a1, Expr::Int((-1).into()))]),
                interner,
            );
            Expr::List(vec![root])
        }
        2 => {
            let a0 = coeffs[0].clone();
            let a1 = coeffs[1].clone();
            let a2 = coeffs[2].clone();
            let disc = simplify_expr(
                Expr::add(vec![
                    Expr::pow(a1.clone(), Expr::Int(2.into())),
                    Expr::neg(Expr::mul(vec![Expr::Int(4.into()), a2.clone(), a0.clone()])),
                ]),
                interner,
            );
            let sqrt_disc = simplify_expr(
                Expr::Call(interner.get_or_intern("sqrt"), vec![disc]),
                interner,
            );
            let two_a2 = simplify_expr(Expr::mul(vec![Expr::Int(2.into()), a2.clone()]), interner);
            let neg_a1 = Expr::neg(a1.clone());

            let x1 = simplify_expr(
                Expr::mul(vec![
                    Expr::add(vec![neg_a1.clone(), sqrt_disc.clone()]),
                    Expr::pow(two_a2.clone(), Expr::Int((-1).into())),
                ]),
                interner,
            );
            let x2 = simplify_expr(
                Expr::mul(vec![
                    Expr::add(vec![neg_a1, Expr::neg(sqrt_disc)]),
                    Expr::pow(two_a2, Expr::Int((-1).into())),
                ]),
                interner,
            );

            if x1 == x2 {
                Expr::List(vec![x1])
            } else {
                Expr::List(vec![x1, x2])
            }
        }
        3 => Expr::List(solve_cubic(&coeffs, interner)),
        _ => {
            let mut current = coeffs.clone();
            let mut roots = Vec::new();

            loop {
                trim_coeffs(&mut current);
                let degree = current.len().saturating_sub(1);
                if degree <= 3 {
                    let rest = solve_from_coeffs(equation, &current, var, interner);
                    if let Expr::List(mut items) = rest {
                        roots.append(&mut items);
                        return Expr::List(roots);
                    }
                    break;
                }

                let candidates = rational_roots(&current, var, interner);
                let Some(root) = candidates.first().cloned() else {
                    break;
                };
                roots.push(root.clone());
                let quotient = poly_divide(&current, &root, interner);
                if quotient.is_empty() {
                    break;
                }
                current = quotient;
            }

            for guess in [-10.0, -1.0, 0.0, 1.0, 10.0] {
                if let Some(root) = solve_numerical(equation, var, interner, guess) {
                    if !roots.iter().any(|r| same_numeric_root(r, &root)) {
                        roots.push(root);
                    }
                }
            }

            Expr::List(roots)
        }
    }
}

pub fn solve(equation: &ax_ir::Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let Some(coeffs) = extract_polynomial(equation, var, interner) else {
        return Expr::Call(
            interner.get_or_intern("solve"),
            vec![equation.clone(), Expr::Sym(var)],
        );
    };
    solve_from_coeffs(equation, &coeffs, var, interner)
}

fn extract_linear_term(expr: &Expr, vars: &[lasso::Spur]) -> Option<(Option<usize>, BigRational)> {
    match expr {
        Expr::Int(n) => Some((None, BigRational::from_integer(n.clone()))),
        Expr::Rational(r) => Some((None, r.clone())),
        Expr::Sym(s) => vars
            .iter()
            .position(|var| var == s)
            .map(|idx| (Some(idx), BigRational::one())),
        Expr::Neg(inner) => {
            let (idx, coeff) = extract_linear_term(inner, vars)?;
            Some((idx, -coeff))
        }
        Expr::Mul(factors) => {
            let mut coeff = BigRational::one();
            let mut var_idx = None;
            for factor in factors {
                match factor {
                    Expr::Int(n) => coeff *= BigRational::from_integer(n.clone()),
                    Expr::Rational(r) => coeff *= r.clone(),
                    Expr::Sym(s) => {
                        let idx = vars.iter().position(|var| var == s)?;
                        if var_idx.replace(idx).is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            Some((var_idx, coeff))
        }
        _ => None,
    }
}

fn extract_linear_equation(
    eq: &Expr,
    vars: &[lasso::Spur],
) -> Option<(Vec<BigRational>, BigRational)> {
    let terms = match eq {
        Expr::Add(terms) => terms.clone(),
        _ => vec![eq.clone()],
    };
    let mut coeffs = vec![BigRational::zero(); vars.len()];
    let mut constant = BigRational::zero();
    for term in terms {
        let (idx, coeff) = extract_linear_term(&term, vars)?;
        if let Some(idx) = idx {
            coeffs[idx] += coeff;
        } else {
            constant += coeff;
        }
    }
    Some((coeffs, -constant))
}

#[derive(Clone, Debug)]
struct ReducedLinearSystem {
    matrix: Vec<Vec<BigRational>>,
    rank: usize,
    inconsistent: bool,
}

fn reduce_linear_system(equations: &[Expr], vars: &[lasso::Spur]) -> Option<ReducedLinearSystem> {
    let rows = equations.len();
    let cols = vars.len();
    if rows == 0 || cols == 0 {
        return Some(ReducedLinearSystem {
            matrix: Vec::new(),
            rank: 0,
            inconsistent: false,
        });
    }

    let mut matrix = equations
        .iter()
        .map(|eq| {
            let (coeffs, rhs) = extract_linear_equation(eq, vars)?;
            let mut row = coeffs;
            row.push(rhs);
            Some(row)
        })
        .collect::<Option<Vec<_>>>()?;

    let mut pivot_row = 0;
    for col in 0..cols {
        let pivot = (pivot_row..rows).find(|&row| !matrix[row][col].is_zero());
        let Some(pivot) = pivot else {
            continue;
        };
        matrix.swap(pivot_row, pivot);
        let pivot_val = matrix[pivot_row][col].clone();
        for j in col..=cols {
            matrix[pivot_row][j] /= pivot_val.clone();
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in col..=cols {
                let pivot_entry = matrix[pivot_row][j].clone();
                matrix[row][j] -= factor.clone() * pivot_entry;
            }
        }
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }

    let inconsistent = matrix
        .iter()
        .any(|row| row[..cols].iter().all(|c| c.is_zero()) && !row[cols].is_zero());

    Some(ReducedLinearSystem {
        matrix,
        rank: pivot_row,
        inconsistent,
    })
}

pub fn solve_linear_system(
    equations: &[ax_ir::Expr],
    vars: &[lasso::Spur],
    _interner: &ax_ir::Interner,
) -> Option<Vec<(lasso::Spur, ax_ir::Expr)>> {
    let cols = vars.len();
    if equations.is_empty() || cols == 0 {
        return Some(Vec::new());
    }

    let reduced = reduce_linear_system(equations, vars)?;
    if reduced.inconsistent || reduced.rank < cols {
        return None;
    }

    let mut out = Vec::with_capacity(cols);
    for i in 0..cols {
        out.push((vars[i], expr_from_rational(reduced.matrix[i][cols].clone())));
    }
    Some(out)
}

/// Compute the symbolic trace of a square matrix by summing its diagonal entries.
pub fn matrix_trace_expr(mat: &[Vec<Expr>]) -> Expr {
    Expr::add(
        mat.iter()
            .enumerate()
            .filter_map(|(i, row)| row.get(i).cloned())
            .collect(),
    )
}

/// Create a fresh symbolic density matrix `rho` together with its flattened symbol list.
pub fn fresh_rho_symbols(
    dim: usize,
    interner: &ax_ir::Interner,
) -> (Vec<lasso::Spur>, Vec<Vec<Expr>>) {
    let nonce = RHO_SYMBOL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut symbols = Vec::with_capacity(dim * dim);
    let matrix = (0..dim)
        .map(|row| {
            (0..dim)
                .map(|col| {
                    let sym = interner.get_or_intern(&format!("rho_ss_{nonce}_{row}_{col}"));
                    symbols.push(sym);
                    Expr::Sym(sym)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (symbols, matrix)
}

/// Solve the finite-dimensional Lindblad steady-state equations `L(rho) = 0` with `Tr(rho) = 1`.
pub fn lindblad_steady_state_linear(
    h: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, LindbladSteadyStateError> {
    let (rows, cols) =
        matrix_shape(h).unwrap_or((h.len(), h.first().map(|row| row.len()).unwrap_or(0)));
    if rows != cols {
        return Err(LindbladSteadyStateError::HamiltonianNotSquare { rows, cols });
    }
    let dim = rows;

    for (index, jump) in jump_ops.iter().enumerate() {
        let (jump_rows, jump_cols) = matrix_shape(jump)
            .unwrap_or((jump.len(), jump.first().map(|row| row.len()).unwrap_or(0)));
        if jump_rows != jump_cols {
            return Err(LindbladSteadyStateError::JumpOperatorNotSquare {
                index,
                rows: jump_rows,
                cols: jump_cols,
            });
        }
        if jump_rows != dim {
            return Err(LindbladSteadyStateError::DimensionMismatch {
                expected: dim,
                actual: jump_rows,
                which: "jump operator",
            });
        }
    }

    let (vars, rho) = fresh_rho_symbols(dim, interner);
    let rhs = ax_qm::lindblad_rhs(h, &rho, jump_ops, interner).map_err(|err| match err {
        ax_qm::LindbladError::HamiltonianNotSquare { rows, cols } => {
            LindbladSteadyStateError::HamiltonianNotSquare { rows, cols }
        }
        ax_qm::LindbladError::StateNotSquare { rows, cols } => {
            LindbladSteadyStateError::DimensionMismatch {
                expected: rows,
                actual: cols,
                which: "state",
            }
        }
        ax_qm::LindbladError::DimensionMismatch {
            expected,
            actual,
            which,
        } => LindbladSteadyStateError::DimensionMismatch {
            expected,
            actual,
            which,
        },
    })?;

    let mut equations = rhs
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect::<Vec<_>>();
    equations.push(Expr::add(vec![
        matrix_trace_expr(&rho),
        Expr::neg(Expr::one()),
    ]));

    let reduced = reduce_linear_system(&equations, &vars)
        .ok_or(LindbladSteadyStateError::InconsistentSteadyStateSystem)?;
    if reduced.inconsistent {
        return Err(LindbladSteadyStateError::InconsistentSteadyStateSystem);
    }
    if reduced.rank < vars.len() {
        return Err(LindbladSteadyStateError::UnderdeterminedSteadyState);
    }

    let solution = solve_linear_system(&equations, &vars, interner)
        .ok_or(LindbladSteadyStateError::InconsistentSteadyStateSystem)?;
    let solution_map = solution
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

    Ok(rho
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| match entry {
                    Expr::Sym(sym) => solution_map.get(&sym).cloned().unwrap_or(Expr::Sym(sym)),
                    other => other,
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::solve;
    use ax_ir::Expr;

    fn solve_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        let expr = result.expr.expect("expected expression");
        let env = ax_eval::Env::new();
        (ax_eval::eval(&expr, &env, &interner), interner)
    }

    #[test]
    fn solve_linear() {
        let (e, _) = solve_src("solve(2*x - 6, x);");
        match e {
            ax_ir::Expr::List(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0], ax_ir::Expr::Int(3.into()));
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn solve_quadratic() {
        let (e, _) = solve_src("solve(x^2 - 5*x + 6, x);");
        match e {
            ax_ir::Expr::List(items) => {
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn solve_quadratic_irrational() {
        let (e, int) = solve_src("solve(x^2 - 2, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("sqrt") || pp.contains("2"), "got: {}", pp);
    }

    #[test]
    fn solve_cubic_simple() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let equation = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(3.into())),
            Expr::neg(Expr::mul(vec![
                Expr::Int(6.into()),
                Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            ])),
            Expr::mul(vec![Expr::Int(11.into()), Expr::Sym(x)]),
            Expr::neg(Expr::Int(6.into())),
        ]);
        let result = solve(&equation, x, &interner);
        if let Expr::List(roots) = &result {
            assert_eq!(
                roots.len(),
                3,
                "x³-6x²+11x-6 should have 3 roots, got {:?}",
                result
            );
        } else {
            panic!("expected List, got {:?}", result);
        }
    }

    #[test]
    fn componentwise_solve_requires_nonempty_tableaux() {
        let empty = ax_ir::TensorSymmetry {
            tableaux: Vec::new(),
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };
        let nonempty = ax_ir::TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape: vec![2],
                slot_map: vec![0, 1],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };

        assert!(!crate::can_solve_componentwise_by_symmetry(&empty));
        assert!(crate::can_solve_componentwise_by_symmetry(&nonempty));
    }
}
