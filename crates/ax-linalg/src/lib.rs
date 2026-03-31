#![forbid(unsafe_code)]

use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

fn to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero()) || matches!(expr, Expr::Rational(r) if r.is_zero())
}

fn expr_from_rational(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Int(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = to_rational(base)?;
    match exp {
        Expr::Int(n) => {
            if let Some(pow) = num_traits::ToPrimitive::to_u32(n) {
                let numer = base_r.numer().clone().pow(pow);
                let denom = base_r.denom().clone().pow(pow);
                Some(expr_from_rational(BigRational::new(numer, denom)))
            } else if n.is_negative() {
                let pow = num_traits::ToPrimitive::to_u32(&(-n))?;
                let numer = base_r.denom().clone().pow(pow);
                let denom = base_r.numer().clone().pow(pow);
                Some(expr_from_rational(BigRational::new(numer, denom)))
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
    let mut low = BigInt::zero();
    let mut high = n.clone() + BigInt::from(1);
    while low < high {
        let mid = (&low + &high) / BigInt::from(2);
        let mid_sq = &mid * &mid;
        if mid_sq <= *n {
            low = mid + BigInt::from(1);
        } else {
            high = mid;
        }
    }
    let root = low - BigInt::from(1);
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

pub fn transpose(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut out = vec![vec![Expr::zero(); rows]; cols];
    for (i, row) in matrix.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            out[j][i] = cell.clone();
        }
    }
    out
}

pub fn mat_mul(a: &[Vec<Expr>], b: &[Vec<Expr>], interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let inner = b.len();
    let cols = b[0].len();
    let mut out = vec![vec![Expr::zero(); cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            let mut terms = Vec::with_capacity(inner);
            for (k, brow) in b.iter().enumerate().take(inner) {
                terms.push(Expr::mul(vec![a[i][k].clone(), brow[j].clone()]));
            }
            out[i][j] = simplify_expr(Expr::add(terms), interner);
        }
    }
    out
}

pub fn mat_add(a: &[Vec<Expr>], b: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb.iter())
                .map(|(ca, cb)| Expr::add(vec![ca.clone(), cb.clone()]))
                .collect()
        })
        .collect()
}

pub fn mat_scale(scalar: &Expr, matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| Expr::mul(vec![scalar.clone(), cell.clone()]))
                .collect()
        })
        .collect()
}

pub fn minor(matrix: &[Vec<Expr>], row: usize, col: usize) -> Vec<Vec<Expr>> {
    matrix
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            if i == row {
                None
            } else {
                Some(
                    r.iter()
                        .enumerate()
                        .filter_map(|(j, cell)| if j == col { None } else { Some(cell.clone()) })
                        .collect(),
                )
            }
        })
        .collect()
}

pub fn determinant(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Expr {
    let n = matrix.len();
    match n {
        0 => Expr::one(),
        1 => matrix[0][0].clone(),
        2 => simplify_expr(
            Expr::add(vec![
                Expr::mul(vec![matrix[0][0].clone(), matrix[1][1].clone()]),
                Expr::neg(Expr::mul(vec![matrix[0][1].clone(), matrix[1][0].clone()])),
            ]),
            interner,
        ),
        _ => {
            let mut terms = Vec::with_capacity(n);
            for j in 0..n {
                let sign = if j % 2 == 0 {
                    Expr::one()
                } else {
                    Expr::Int((-1).into())
                };
                let sub = determinant(&minor(matrix, 0, j), interner);
                terms.push(Expr::mul(vec![sign, matrix[0][j].clone(), sub]));
            }
            simplify_expr(Expr::add(terms), interner)
        }
    }
}

pub fn cofactor_matrix(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    let n = matrix.len();
    let mut out = vec![vec![Expr::zero(); n]; n];
    for (i, row) in out.iter_mut().enumerate().take(n) {
        for (j, cell) in row.iter_mut().enumerate().take(n) {
            let sign = if (i + j) % 2 == 0 {
                Expr::one()
            } else {
                Expr::Int((-1).into())
            };
            *cell = simplify_expr(
                Expr::mul(vec![sign, determinant(&minor(matrix, i, j), interner)]),
                interner,
            );
        }
    }
    out
}

pub fn adjugate(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    transpose(&cofactor_matrix(matrix, interner))
}

pub fn inverse(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Option<Vec<Vec<Expr>>> {
    let det = simplify_expr(determinant(matrix, interner), interner);
    if is_zero(&det) {
        return None;
    }
    let inv_det = Expr::pow(det, Expr::Int((-1).into()));
    Some(
        adjugate(matrix, interner)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| simplify_expr(Expr::mul(vec![inv_det.clone(), cell]), interner))
                    .collect()
            })
            .collect(),
    )
}

pub fn trace(matrix: &[Vec<Expr>]) -> Expr {
    Expr::add(
        matrix
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].clone())
            .collect(),
    )
}

pub fn identity(n: usize) -> Vec<Vec<Expr>> {
    let mut out = vec![vec![Expr::zero(); n]; n];
    for (i, row) in out.iter_mut().enumerate().take(n) {
        row[i] = Expr::one();
    }
    out
}

pub fn eigenvalues_2x2(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Vec<Expr> {
    let tr = simplify_expr(trace(matrix), interner);
    let det = simplify_expr(determinant(matrix, interner), interner);
    let disc = simplify_expr(
        Expr::add(vec![
            Expr::pow(tr.clone(), Expr::Int(2.into())),
            Expr::neg(Expr::mul(vec![Expr::Int(4.into()), det.clone()])),
        ]),
        interner,
    );
    let sqrt_disc = simplify_expr(
        Expr::Call(interner.get_or_intern("sqrt"), vec![disc]),
        interner,
    );
    let two = Expr::Int(2.into());

    let x1 = simplify_expr(
        Expr::mul(vec![
            Expr::add(vec![tr.clone(), sqrt_disc.clone()]),
            Expr::pow(two.clone(), Expr::Int((-1).into())),
        ]),
        interner,
    );
    let x2 = simplify_expr(
        Expr::mul(vec![
            Expr::add(vec![tr, Expr::neg(sqrt_disc)]),
            Expr::pow(two, Expr::Int((-1).into())),
        ]),
        interner,
    );
    if x1 == x2 {
        vec![x1]
    } else {
        vec![x1, x2]
    }
}

pub fn eigenvalues_symbolic(matrix: &[Vec<Expr>], interner: &ax_ir::Interner) -> Expr {
    let lambda = interner.get_or_intern("lambda");
    let lambda_sym = Expr::Sym(lambda);
    let scaled_i = mat_scale(&Expr::neg(lambda_sym), &identity(matrix.len()));
    let shifted = mat_add(matrix, &scaled_i);
    simplify_expr(determinant(&shifted, interner), interner)
}

pub fn tensor_product(a: &[Vec<Expr>], b: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();
    let mut out = vec![vec![Expr::zero(); a_cols * b_cols]; a_rows * b_rows];
    for i in 0..a_rows {
        for j in 0..a_cols {
            for k in 0..b_rows {
                for l in 0..b_cols {
                    out[i * b_rows + k][j * b_cols + l] =
                        Expr::mul(vec![a[i][j].clone(), b[k][l].clone()]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
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
    fn det_2x2() {
        let (e, _) = solve_src("det([[1, 2], [3, 4]]);");
        assert_eq!(e, Expr::Int((-2).into()));
    }

    #[test]
    fn det_3x3() {
        let (e, _) = solve_src("det([[1,0,0],[0,1,0],[0,0,1]]);");
        assert_eq!(e, Expr::Int(1.into()));
    }

    #[test]
    fn transpose_test() {
        let (e, _) = solve_src("transpose([[1,2],[3,4]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows[0][1], Expr::Int(3.into()));
                assert_eq!(rows[1][0], Expr::Int(2.into()));
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn trace_test() {
        let (e, _) = solve_src("trace_mat([[1,2],[3,4]]);");
        assert_eq!(e, Expr::Int(5.into()));
    }

    #[test]
    fn inverse_2x2() {
        let (e, _) = solve_src("inv([[1,0],[0,2]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(
                    rows[1][1],
                    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
                );
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }
}
