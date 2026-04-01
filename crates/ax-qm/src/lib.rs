#![forbid(unsafe_code)]

use ax_ir::{Expr, Index, Variance};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorKind {
    Creation,
    Annihilation,
}

#[derive(Clone, Debug)]
pub enum GammaEntry {
    Gamma(lasso::Spur),
    Gamma5,
    Identity,
}

fn operator_kind(expr: &Expr, operators: &HashMap<lasso::Spur, OperatorKind>) -> Option<OperatorKind> {
    match expr {
        Expr::Sym(sym) => operators.get(sym).copied(),
        _ => None,
    }
}

pub fn normal_order_simple(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;
    match expr {
        Expr::Mul(factors) => {
            let mut other = Vec::new();
            let mut creation = Vec::new();
            let mut annihilation = Vec::new();
            for factor in factors {
                let simplified = normal_order_simple(factor, operators, interner);
                match operator_kind(&simplified, operators) {
                    Some(OperatorKind::Creation) => creation.push(simplified),
                    Some(OperatorKind::Annihilation) => annihilation.push(simplified),
                    None => other.push(simplified),
                }
            }
            other.extend(creation);
            other.extend(annihilation);
            Expr::mul(other)
        }
        Expr::Add(terms) => Expr::add(
            terms.iter()
                .map(|term| normal_order_simple(term, operators, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            normal_order_simple(base, operators, interner),
            normal_order_simple(exp, operators, interner),
        ),
        Expr::Neg(inner) => Expr::neg(normal_order_simple(inner, operators, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normal_order_simple(re, operators, interner)),
            Box::new(normal_order_simple(im, operators, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| normal_order_simple(arg, operators, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(normal_order_simple(body, operators, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(normal_order_simple(lhs, operators, interner)),
            Box::new(normal_order_simple(rhs, operators, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (normal_order_simple(value, operators, interner), condition.clone())
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(normal_order_simple(base, operators, interner)),
            indices.clone(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(normal_order_simple(value, operators, interner)),
            Box::new(normal_order_simple(body, operators, interner)),
        ),
        Expr::List(items) => Expr::List(
            items.iter()
                .map(|item| normal_order_simple(item, operators, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| normal_order_simple(cell, operators, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn normal_order(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    normal_order_simple(expr, operators, interner)
}

pub fn wick_expand_single(
    factors: &[ax_ir::Expr],
    operators: &HashMap<lasso::Spur, OperatorKind>,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut terms = vec![normal_order_simple(&Expr::mul(factors.to_vec()), operators, interner)];

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let (Expr::Sym(lhs), Expr::Sym(rhs)) = (&factors[i], &factors[j]) else {
                continue;
            };
            if let Some(contraction) = contractions.get(&(*lhs, *rhs)) {
                let remaining = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, factor)| {
                        if idx == i || idx == j {
                            None
                        } else {
                            Some(factor.clone())
                        }
                    })
                    .collect::<Vec<_>>();
                let ordered_remaining =
                    normal_order_simple(&Expr::mul(remaining), operators, interner);
                terms.push(Expr::mul(vec![contraction.clone(), ordered_remaining]));
            }
        }
    }

    simplify_expr(Expr::add(terms))
}

pub fn wick_expand(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => wick_expand_single(factors, operators, contractions, interner),
        Expr::Add(terms) => Expr::add(
            terms.iter()
                .map(|term| wick_expand(term, operators, contractions, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            wick_expand(base, operators, contractions, interner),
            wick_expand(exp, operators, contractions, interner),
        ),
        Expr::Neg(inner) => Expr::neg(wick_expand(inner, operators, contractions, interner)),
        _ => normal_order_simple(expr, operators, interner),
    }
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
                        index_type: None,
                    },
                    ax_ir::Index {
                        name: indices[1],
                        variance: ax_ir::Variance::Up,
                        index_type: None,
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
                    index_type: None,
                },
                ax_ir::Index {
                    name: indices[k],
                    variance: ax_ir::Variance::Up,
                    index_type: None,
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
                            index_type: None,
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

/// Join (contract) a product of gamma matrices.
///
/// gamma(a) * gamma(b) → gamma(a, b) + g(a+, b+)
/// gamma(a) * gamma(b, c) → gamma(a, b, c) + g(a+, b+) * gamma(c) - g(a+, c+) * gamma(b)
///
/// Uses the recursive contraction identity.
pub fn join_gamma_pair(
    indices1: &[lasso::Spur],
    indices2: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    if indices1.is_empty() {
        return make_gamma(indices2, gamma_sym);
    }
    if indices2.is_empty() {
        return make_gamma(indices1, gamma_sym);
    }

    let a1 = indices1[0];
    let rest1 = &indices1[1..];

    if rest1.is_empty() {
        join_single_with_multi(a1, indices2, gamma_sym, metric_sym, interner)
    } else {
        let inner = join_gamma_pair(rest1, indices2, gamma_sym, metric_sym, interner);
        join_single_with_expr(a1, &inner, gamma_sym, metric_sym, interner)
    }
}

fn join_single_with_multi(
    a: lasso::Spur,
    bs: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut terms = Vec::new();

    // First term: gamma with all indices combined
    let mut all = vec![a];
    all.extend_from_slice(bs);
    terms.push(make_gamma(&all, gamma_sym));

    // Contraction terms: Σ_k (-1)^k g^{a b_k} γ^{bs \ b_k}
    for k in 0..bs.len() {
        let metric = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                Index { name: a, variance: Variance::Up, index_type: None },
                Index { name: bs[k], variance: Variance::Up, index_type: None },
            ],
        );

        let remaining: Vec<lasso::Spur> = bs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .map(|(_, &b)| b)
            .collect();

        let gamma_part = if remaining.is_empty() {
            Expr::one()
        } else {
            make_gamma(&remaining, gamma_sym)
        };

        let term = Expr::mul(vec![metric, gamma_part]);
        if k % 2 == 0 {
            terms.push(term);
        } else {
            terms.push(Expr::neg(term));
        }
    }

    Expr::add(terms)
}

fn join_single_with_expr(
    a: lasso::Spur,
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_single_with_expr(a, t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            // Find the first gamma factor and join with it
            for (i, factor) in factors.iter().enumerate() {
                if let Expr::Call(f, args) = factor {
                    if *f == gamma_sym {
                        let gamma_indices: Vec<lasso::Spur> = args
                            .iter()
                            .filter_map(|arg| if let Expr::Sym(s) = arg { Some(*s) } else { None })
                            .collect();
                        let joined =
                            join_single_with_multi(a, &gamma_indices, gamma_sym, metric_sym, interner);
                        let mut rest: Vec<Expr> = factors
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, f)| f.clone())
                            .collect();
                        rest.push(joined);
                        return Expr::mul(rest);
                    }
                }
            }
            // No gamma factor found — prepend a single-index gamma
            let mut new_factors = vec![make_gamma(&[a], gamma_sym)];
            new_factors.extend(factors.iter().cloned());
            Expr::mul(new_factors)
        }
        Expr::Neg(e) => {
            Expr::neg(join_single_with_expr(a, e, gamma_sym, metric_sym, interner))
        }
        _ => {
            // Check if expr itself is a gamma call
            if let Expr::Call(f, args) = expr {
                if *f == gamma_sym {
                    let gamma_indices: Vec<lasso::Spur> = args
                        .iter()
                        .filter_map(|arg| if let Expr::Sym(s) = arg { Some(*s) } else { None })
                        .collect();
                    return join_single_with_multi(
                        a,
                        &gamma_indices,
                        gamma_sym,
                        metric_sym,
                        interner,
                    );
                }
            }
            Expr::mul(vec![make_gamma(&[a], gamma_sym), expr.clone()])
        }
    }
}

fn make_gamma(indices: &[lasso::Spur], gamma_sym: lasso::Spur) -> Expr {
    Expr::Call(gamma_sym, indices.iter().map(|&i| Expr::Sym(i)).collect())
}

/// Extract gamma indices from a `gamma(a, b, ...)` Call expression.
fn gamma_indices(args: &[Expr]) -> Vec<lasso::Spur> {
    args.iter()
        .filter_map(|arg| if let Expr::Sym(s) = arg { Some(*s) } else { None })
        .collect()
}

/// Walk an expression and join all adjacent gamma-matrix Call nodes in products.
pub fn join_gammas_in_expr(
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Recursively process each factor first
            let factors: Vec<Expr> = factors
                .iter()
                .map(|f| join_gammas_in_expr(f, gamma_sym, metric_sym, interner))
                .collect();

            // Now fold adjacent gamma pairs left-to-right
            let mut result: Vec<Expr> = Vec::new();
            for factor in factors {
                if let Some(last) = result.last() {
                    if let (Expr::Call(f1, a1), Expr::Call(f2, a2)) = (last, &factor) {
                        if *f1 == gamma_sym && *f2 == gamma_sym {
                            let i1 = gamma_indices(a1);
                            let i2 = gamma_indices(a2);
                            let joined =
                                join_gamma_pair(&i1, &i2, gamma_sym, metric_sym, interner);
                            result.pop();
                            // The joined expression may be an Add — wrap in a group
                            // by pushing the whole joined expression, then distributing
                            // remaining factors over it at the end.
                            result.push(joined);
                            continue;
                        }
                    }
                }
                result.push(factor);
            }

            if result.len() == 1 {
                result.remove(0)
            } else {
                Expr::mul(result)
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_gammas_in_expr(t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => {
            Expr::neg(join_gammas_in_expr(e, gamma_sym, metric_sym, interner))
        }
        _ => expr.clone(),
    }
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

    #[test]
    fn normal_order_puts_creation_first() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2);
            assert_eq!(factors[0], Expr::Sym(a_dag));
            assert_eq!(factors[1], Expr::Sym(a));
        } else {
            panic!("expected Mul");
        }
    }

    #[test]
    fn normal_order_preserves_scalars() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);

        let expr = Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("3"), "got: {}", pp);
    }

    #[test]
    fn join_two_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // γ^a γ^b = γ^{ab} + g^{ab}
        let result = join_gamma_pair(&[a], &[b], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "expected γ^{{ab}} + g^{{ab}}, got {terms:?}");
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_three_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // γ^a γ^{bc} = γ^{abc} + g^{ab} γ^c - g^{ac} γ^b
        let result = join_gamma_pair(&[a], &[b, c], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 3, "expected 3 terms, got {terms:?}");
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_empty_left_is_identity() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");

        // Identity * γ^a = γ^a
        let result = join_gamma_pair(&[], &[a], gamma, g, &interner);
        assert_eq!(result, Expr::Call(gamma, vec![Expr::Sym(a)]));
    }

    #[test]
    fn join_gammas_in_product() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // join_gammas_in_expr(gamma(a) * gamma(b)) → Add(...)
        let expr = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
        ]);
        let result = join_gammas_in_expr(&expr, gamma, g, &interner);
        assert!(matches!(result, Expr::Add(_)), "expected Add, got {result:?}");
    }
}
