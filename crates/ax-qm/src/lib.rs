#![forbid(unsafe_code)]

use ax_ir::Expr;

#[derive(Clone, Debug)]
pub enum GammaEntry {
    Gamma(lasso::Spur),
    Gamma5,
    Identity,
}

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
    let _ = interner;
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));
    let neg_i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())));
    vec![vec![Expr::zero(), neg_i], vec![i, Expr::zero()]]
}

pub fn pauli_z(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ]
}

fn imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
}

fn neg_imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())))
}

pub fn gamma_matrices_dirac(_interner: &ax_ir::Interner) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    let i = imag_unit();
    let neg_i = neg_imag_unit();
    vec![
        vec![
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::neg(Expr::one()), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::neg(Expr::one())],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::neg(Expr::one()), Expr::zero(), Expr::zero()],
            vec![Expr::neg(Expr::one()), Expr::zero(), Expr::zero(), Expr::zero()],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), neg_i.clone()],
            vec![Expr::zero(), Expr::zero(), i.clone(), Expr::zero()],
            vec![Expr::zero(), i, Expr::zero(), Expr::zero()],
            vec![neg_i, Expr::zero(), Expr::zero(), Expr::zero()],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::neg(Expr::one())],
            vec![Expr::neg(Expr::one()), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
        ],
    ]
}

pub fn gamma5(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
    ]
}

pub fn gamma_trace_recursive(
    indices: &[lasso::Spur],
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;
    let n = indices.len();
    if n == 0 {
        return Expr::Int(4.into());
    }
    if n % 2 != 0 {
        return Expr::zero();
    }
    if n == 2 {
        return Expr::mul(vec![
            Expr::Int(4.into()),
            Expr::Indexed(
                Box::new(Expr::Sym(metric_sym)),
                vec![
                    ax_ir::Index {
                        name: indices[0],
                        variance: ax_ir::Variance::Up,
                    },
                    ax_ir::Index {
                        name: indices[1],
                        variance: ax_ir::Variance::Up,
                    },
                ],
            ),
        ]);
    }

    let a1 = indices[0];
    let mut terms = Vec::new();
    for k in 1..n {
        let sign = if (k - 1) % 2 == 0 {
            Expr::one()
        } else {
            Expr::neg(Expr::one())
        };
        let metric_factor = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                ax_ir::Index {
                    name: a1,
                    variance: ax_ir::Variance::Up,
                },
                ax_ir::Index {
                    name: indices[k],
                    variance: ax_ir::Variance::Up,
                },
            ],
        );
        let remaining = indices[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k - 1)
            .map(|(_, sym)| *sym)
            .collect::<Vec<_>>();
        let sub_trace = gamma_trace_recursive(&remaining, metric_sym, interner);
        terms.push(Expr::mul(vec![sign, metric_factor, sub_trace]));
    }
    Expr::add(terms)
}

pub fn gamma_trace(
    indices: &[GammaEntry],
    metric: &ax_tensor::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = metric;
    let metric_sym = interner.get_or_intern("g");
    let epsilon_sym = interner.get_or_intern("epsilon");

    let mut gamma_indices = Vec::new();
    let mut gamma5_count = 0usize;
    for entry in indices {
        match entry {
            GammaEntry::Gamma(sym) => gamma_indices.push(*sym),
            GammaEntry::Gamma5 => gamma5_count += 1,
            GammaEntry::Identity => {}
        }
    }

    if gamma5_count > 1 {
        return Expr::zero();
    }

    if gamma5_count == 1 {
        return match gamma_indices.len() {
            0 | 1 | 2 | 3 => Expr::zero(),
            4 => Expr::mul(vec![
                Expr::Int((-4).into()),
                imag_unit(),
                Expr::Indexed(
                    Box::new(Expr::Sym(epsilon_sym)),
                    gamma_indices
                        .iter()
                        .map(|sym| ax_ir::Index {
                            name: *sym,
                            variance: ax_ir::Variance::Up,
                        })
                        .collect(),
                ),
            ]),
            _ if gamma_indices.len() % 2 != 0 => Expr::zero(),
            _ => Expr::zero(),
        };
    }

    gamma_trace_recursive(&gamma_indices, metric_sym, interner)
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
            let two_i = Expr::mul(vec![
                Expr::Int(2.into()),
                Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one())),
            ]);
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
        let simplified = ax_eval::eval(&comm[0][0], &ax_eval::Env::new(), &interner);
        let expected = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::Int(2.into())));
        assert_eq!(simplified, expected);
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

    #[test]
    fn trace_identity() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let result = gamma_trace_recursive(&[], g, &interner);
        assert_eq!(result, Expr::Int(4.into()));
    }

    #[test]
    fn trace_single_gamma_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let result = gamma_trace_recursive(&[mu], g, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn trace_two_gammas() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let result = gamma_trace_recursive(&[mu, nu], g_sym, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("4") && pp.contains("g"), "got: {}", pp);
    }

    #[test]
    fn trace_odd_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");
        let result = gamma_trace_recursive(&[mu, nu, rho], g, &interner);
        assert_eq!(result, Expr::zero());
    }
}
