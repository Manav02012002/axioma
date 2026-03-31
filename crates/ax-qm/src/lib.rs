#![forbid(unsafe_code)]

use ax_ir::Expr;

fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let mut grouped: Vec<(Expr, usize)> = Vec::new();
            for term in terms.into_iter().map(simplify_expr) {
                if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == term)
                {
                    *count += 1;
                } else {
                    grouped.push((term, 1));
                }
            }
            Expr::add(
                grouped
                    .into_iter()
                    .map(|(term, count)| {
                        if count == 1 {
                            term
                        } else {
                            Expr::mul(vec![Expr::Int((count as i64).into()), term])
                        }
                    })
                    .collect(),
            )
        }
        Expr::Mul(factors) => Expr::mul(factors.into_iter().map(simplify_expr).collect()),
        Expr::Pow(base, exp) => Expr::pow(simplify_expr(*base), simplify_expr(*exp)),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner)),
        other => other,
    }
}

fn simplify_matrix(matrix: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    matrix
        .into_iter()
        .map(|row| row.into_iter().map(simplify_expr).collect())
        .collect()
}

pub fn pauli_x(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero()],
    ]
}

pub fn pauli_y(interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    let i = Expr::Sym(interner.get_or_intern("i"));
    let neg_i = Expr::neg(i.clone());
    vec![vec![Expr::zero(), neg_i], vec![i, Expr::zero()]]
}

pub fn pauli_z(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ]
}

pub fn commutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(
        &ab,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &ba),
    ))
}

pub fn anticommutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(&ab, &ba))
}

fn half() -> Expr {
    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
}

pub fn angular_momentum_matrices(
    j: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<(Vec<Vec<ax_ir::Expr>>, Vec<Vec<ax_ir::Expr>>, Vec<Vec<ax_ir::Expr>>)> {
    match j {
        Expr::Rational(r) if *r == num_rational::BigRational::new(1.into(), 2.into()) => {
            let hx = ax_linalg::mat_scale(&half(), &pauli_x(interner));
            let hy = ax_linalg::mat_scale(&half(), &pauli_y(interner));
            let hz = ax_linalg::mat_scale(&half(), &pauli_z(interner));
            Some((hx, hy, hz))
        }
        Expr::Int(n) if *n == 1.into() => {
            let sqrt2 = Expr::Call(interner.get_or_intern("sqrt"), vec![Expr::Int(2.into())]);
            let jp = vec![
                vec![Expr::zero(), sqrt2.clone(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), sqrt2.clone()],
                vec![Expr::zero(), Expr::zero(), Expr::zero()],
            ];
            let jm = ax_linalg::transpose(&jp);
            let jx = ax_linalg::mat_scale(&half(), &ax_linalg::mat_add(&jp, &jm));
            let two_i = Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(interner.get_or_intern("i"))]);
            let jy = ax_linalg::mat_scale(
                &Expr::pow(two_i, Expr::Int((-1).into())),
                &ax_linalg::mat_add(&jp, &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &jm)),
            );
            let jz = vec![
                vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), Expr::Int((-1).into())],
            ];
            Some((jx, jy, jz))
        }
        _ => None,
    }
}

pub fn density_matrix(state: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    let n = state.len();
    let mut rho = vec![vec![Expr::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            rho[i][j] = Expr::mul(vec![state[i].clone(), state[j].clone()]);
        }
    }
    rho
}

pub fn partial_trace(
    rho: &[Vec<ax_ir::Expr>],
    dim_a: usize,
    dim_b: usize,
    trace_over: char,
    _interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    match trace_over {
        'B' => {
            let mut out = vec![vec![Expr::zero(); dim_a]; dim_a];
            for (i, row) in out.iter_mut().enumerate().take(dim_a) {
                for (j, cell) in row.iter_mut().enumerate().take(dim_a) {
                    let terms = (0..dim_b)
                        .map(|k| rho[i * dim_b + k][j * dim_b + k].clone())
                        .collect();
                    *cell = Expr::add(terms);
                }
            }
            out
        }
        'A' => {
            let mut out = vec![vec![Expr::zero(); dim_b]; dim_b];
            for (k, row) in out.iter_mut().enumerate().take(dim_b) {
                for (l, cell) in row.iter_mut().enumerate().take(dim_b) {
                    let terms = (0..dim_a)
                        .map(|i| rho[i * dim_b + k][i * dim_b + l].clone())
                        .collect();
                    *cell = Expr::add(terms);
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

pub fn ket(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    let mut out = vec![Expr::zero(); dim];
    if index < dim {
        out[index] = Expr::one();
    }
    out
}

pub fn bra(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    ket(index, dim)
}

pub fn braket(bra: &[ax_ir::Expr], ket: &[ax_ir::Expr]) -> ax_ir::Expr {
    Expr::add(
        bra.iter()
            .zip(ket.iter())
            .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
            .collect(),
    )
}

pub fn outer(a: &[ax_ir::Expr], b: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    a.iter()
        .map(|ai| {
            b.iter()
                .map(|bj| Expr::mul(vec![ai.clone(), bj.clone()]))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pauli_commutation() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let sy = pauli_y(&interner);
        let comm = commutator(&sx, &sy, &interner);
        let i_sym = interner.get_or_intern("i");
        let expected_00 = Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(i_sym)]);
        let simplified = ax_eval::eval(&comm[0][0], &ax_eval::Env::new(), &interner);
        let expected_simplified = ax_eval::eval(&expected_00, &ax_eval::Env::new(), &interner);
        assert_eq!(simplified, expected_simplified);
    }

    #[test]
    fn anticommutator_pauli() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let anti = anticommutator(&sx, &sx, &interner);
        let simplified_00 = ax_eval::eval(&anti[0][0], &ax_eval::Env::new(), &interner);
        assert_eq!(simplified_00, Expr::Int(2.into()));
    }

    #[test]
    fn ket_basis_vectors() {
        let interner = ax_ir::Interner::new();
        let ket0 = vec![Expr::one(), Expr::zero()];
        let ket1 = vec![Expr::zero(), Expr::one()];
        let inner = Expr::add(
            ket0.iter()
                .zip(ket1.iter())
                .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
                .collect::<Vec<_>>(),
        );
        let result = ax_eval::eval(&inner, &ax_eval::Env::new(), &interner);
        assert_eq!(result, Expr::zero());
    }
}
