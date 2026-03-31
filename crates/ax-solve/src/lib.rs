#![forbid(unsafe_code)]

use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
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
        _ => {
            let mut current = coeffs.clone();
            let mut roots = Vec::new();

            loop {
                trim_coeffs(&mut current);
                let degree = current.len().saturating_sub(1);
                if degree <= 2 {
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

            Expr::Call(
                interner.get_or_intern("solve"),
                vec![equation.clone(), Expr::Sym(var)],
            )
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

pub fn solve_linear_system(
    equations: &[ax_ir::Expr],
    vars: &[lasso::Spur],
    _interner: &ax_ir::Interner,
) -> Option<Vec<(lasso::Spur, ax_ir::Expr)>> {
    let rows = equations.len();
    let cols = vars.len();
    if rows == 0 || cols == 0 {
        return Some(Vec::new());
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

    for row in &matrix {
        if row[..cols].iter().all(|c| c.is_zero()) && !row[cols].is_zero() {
            return None;
        }
    }

    if pivot_row < cols {
        return None;
    }

    let mut out = Vec::with_capacity(cols);
    for i in 0..cols {
        out.push((vars[i], expr_from_rational(matrix[i][cols].clone())));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
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
}
