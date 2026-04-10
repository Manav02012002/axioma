use crate::{christoffel_from_metric, diff_component, simplify_expr, SymbolicMatrix};
use ax_ir::Expr;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConformalError {
    #[error("conformal transformation requires coordinates length {coords_len} to match metric dimension {metric_dim}")]
    CoordinateMismatch {
        coords_len: usize,
        metric_dim: usize,
    },
    #[error("conformal transformation requires a square metric")]
    InvalidMetric,
    #[error("conformal transformation dimension mismatch: metric dim {metric_dim}, gamma dim {gamma_dim}, riemann dim {riemann_dim}, ricci rows {ricci_rows}, ricci cols {ricci_cols}")]
    DimensionMismatch {
        metric_dim: usize,
        gamma_dim: usize,
        riemann_dim: usize,
        ricci_rows: usize,
        ricci_cols: usize,
    },
    #[error("conformal transformation failed to invert the transformed metric")]
    InverseMetricFailed,
}

fn validate_metric(g: &SymbolicMatrix) -> Result<(), ConformalError> {
    if g.data.len() != g.dim || g.data.iter().any(|row| row.len() != g.dim) {
        return Err(ConformalError::InvalidMetric);
    }
    Ok(())
}

fn validate_gamma_shape(gamma: &[Vec<Vec<Expr>>], n: usize) -> bool {
    gamma.len() == n
        && gamma
            .iter()
            .all(|plane| plane.len() == n && plane.iter().all(|row| row.len() == n))
}

fn ricci_shape(ricci: &[Vec<Expr>]) -> (usize, usize) {
    let rows = ricci.len();
    let cols = ricci.first().map(Vec::len).unwrap_or(0);
    (rows, cols)
}

fn int_expr(value: i64) -> Expr {
    Expr::Int(value.into())
}

fn safe_inverse_metric(
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<SymbolicMatrix, ConformalError> {
    if (0..g.dim).all(|row| (0..g.dim).all(|col| row == col || g.data[row][col] == Expr::zero())) {
        let mut inverse = SymbolicMatrix::new(g.dim);
        for i in 0..g.dim {
            inverse.data[i][i] = simplify_expr(
                Expr::pow(g.data[i][i].clone(), Expr::Int((-1).into())),
                interner,
            );
        }
        return Ok(inverse);
    }
    let inv = ax_linalg::inverse(&g.data, interner).ok_or(ConformalError::InverseMetricFailed)?;
    Ok(SymbolicMatrix {
        dim: g.dim,
        data: inv
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| simplify_expr(cell, interner))
                    .collect()
            })
            .collect(),
    })
}

fn u_covector(omega: &Expr, coords: &[lasso::Spur], interner: &ax_ir::Interner) -> Vec<Expr> {
    coords
        .iter()
        .map(|coord| {
            simplify_expr(
                Expr::mul(vec![
                    diff_component(omega, *coord, interner),
                    Expr::pow(omega.clone(), Expr::Int((-1).into())),
                ]),
                interner,
            )
        })
        .collect()
}

fn raise_covector(
    covector: &[Expr],
    g_inv: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Expr> {
    (0..g_inv.dim)
        .map(|a| {
            let terms = (0..g_inv.dim)
                .filter_map(|b| {
                    if g_inv.get(a, b) == &Expr::zero() || covector[b] == Expr::zero() {
                        None
                    } else {
                        Some(Expr::mul(vec![
                            g_inv.get(a, b).clone(),
                            covector[b].clone(),
                        ]))
                    }
                })
                .collect::<Vec<_>>();
            simplify_expr(Expr::add(terms), interner)
        })
        .collect()
}

fn covariant_derivative_covector(
    covector: &[Expr],
    gamma: &[Vec<Vec<Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Expr>> {
    let n = coords.len();
    let mut out = vec![vec![Expr::zero(); n]; n];
    for a in 0..n {
        for b in 0..n {
            let mut terms = vec![diff_component(&covector[b], coords[a], interner)];
            for m in 0..n {
                if gamma[m][a][b] == Expr::zero() || covector[m] == Expr::zero() {
                    continue;
                }
                terms.push(Expr::neg(Expr::mul(vec![
                    gamma[m][a][b].clone(),
                    covector[m].clone(),
                ])));
            }
            out[a][b] = simplify_expr(Expr::add(terms), interner);
        }
    }
    out
}

fn divergence_vector(
    vector: &[Expr],
    gamma: &[Vec<Vec<Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let n = coords.len();
    let mut terms = Vec::new();
    for a in 0..n {
        terms.push(diff_component(&vector[a], coords[a], interner));
        for b in 0..n {
            if gamma[a][a][b] == Expr::zero() || vector[b] == Expr::zero() {
                continue;
            }
            terms.push(Expr::mul(vec![gamma[a][a][b].clone(), vector[b].clone()]));
        }
    }
    simplify_expr(Expr::add(terms), interner)
}

fn norm_covector(covector: &[Expr], vector: &[Expr], interner: &ax_ir::Interner) -> Expr {
    let terms = covector
        .iter()
        .zip(vector.iter())
        .filter_map(|(down, up)| {
            if *down == Expr::zero() || *up == Expr::zero() {
                None
            } else {
                Some(Expr::mul(vec![down.clone(), up.clone()]))
            }
        })
        .collect::<Vec<_>>();
    simplify_expr(Expr::add(terms), interner)
}

pub fn conformal_transform_metric(
    g: &crate::SymbolicMatrix,
    omega: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> crate::SymbolicMatrix {
    let scale = simplify_expr(Expr::pow(omega.clone(), Expr::Int(2.into())), interner);
    SymbolicMatrix {
        dim: g.dim,
        data: g
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| {
                        simplify_expr(Expr::mul(vec![scale.clone(), entry.clone()]), interner)
                    })
                    .collect()
            })
            .collect(),
    }
}

pub fn conformal_transform_inverse_metric(
    g_inv: &crate::SymbolicMatrix,
    omega: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> crate::SymbolicMatrix {
    let scale = simplify_expr(Expr::pow(omega.clone(), Expr::Int((-2).into())), interner);
    SymbolicMatrix {
        dim: g_inv.dim,
        data: g_inv
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| {
                        simplify_expr(Expr::mul(vec![scale.clone(), entry.clone()]), interner)
                    })
                    .collect()
            })
            .collect(),
    }
}

pub fn conformal_transform_christoffel(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    g: &crate::SymbolicMatrix,
    omega: &ax_ir::Expr,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<ax_ir::Expr>>>, ConformalError> {
    validate_metric(g)?;
    if coords.len() != g.dim {
        return Err(ConformalError::CoordinateMismatch {
            coords_len: coords.len(),
            metric_dim: g.dim,
        });
    }
    if !validate_gamma_shape(gamma, g.dim) {
        let (ricci_rows, ricci_cols) = (0, 0);
        return Err(ConformalError::DimensionMismatch {
            metric_dim: g.dim,
            gamma_dim: gamma.len(),
            riemann_dim: 0,
            ricci_rows,
            ricci_cols,
        });
    }
    let g_inv = safe_inverse_metric(g, interner)?;
    let u_down = u_covector(omega, coords, interner);
    let u_up = raise_covector(&u_down, &g_inv, interner);
    let n = g.dim;
    let mut transformed = vec![vec![vec![Expr::zero(); n]; n]; n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let delta_ab = if a == b { Expr::one() } else { Expr::zero() };
                let delta_ac = if a == c { Expr::one() } else { Expr::zero() };
                transformed[a][b][c] = simplify_expr(
                    Expr::add(vec![
                        gamma[a][b][c].clone(),
                        Expr::mul(vec![delta_ab, u_down[c].clone()]),
                        Expr::mul(vec![delta_ac, u_down[b].clone()]),
                        Expr::neg(Expr::mul(vec![g.get(b, c).clone(), u_up[a].clone()])),
                    ]),
                    interner,
                );
            }
        }
    }
    Ok(transformed)
}

pub fn conformal_transform_ricci(
    ricci: &[Vec<ax_ir::Expr>],
    scalar: &ax_ir::Expr,
    g: &crate::SymbolicMatrix,
    omega: &ax_ir::Expr,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<ax_ir::Expr>>, ConformalError> {
    validate_metric(g)?;
    if coords.len() != g.dim {
        return Err(ConformalError::CoordinateMismatch {
            coords_len: coords.len(),
            metric_dim: g.dim,
        });
    }
    let (ricci_rows, ricci_cols) = ricci_shape(ricci);
    if ricci_rows != g.dim || ricci.iter().any(|row| row.len() != ricci_rows) || ricci_cols != g.dim
    {
        return Err(ConformalError::DimensionMismatch {
            metric_dim: g.dim,
            gamma_dim: 0,
            riemann_dim: 0,
            ricci_rows,
            ricci_cols,
        });
    }
    let _ = scalar;
    let gamma = christoffel_from_metric(g, coords, interner);
    let g_inv = safe_inverse_metric(g, interner)?;
    let n = g.dim;
    let u_down = u_covector(omega, coords, interner);
    let u_up = raise_covector(&u_down, &g_inv, interner);
    let nabla_u = covariant_derivative_covector(&u_down, &gamma, coords, interner);
    let div_u = divergence_vector(&u_up, &gamma, coords, interner);
    let norm_u = norm_covector(&u_down, &u_up, interner);
    let n_minus_two = int_expr((n as i64) - 2);
    let mut transformed = vec![vec![Expr::zero(); n]; n];
    for a in 0..n {
        for b in 0..n {
            transformed[a][b] = simplify_expr(
                Expr::add(vec![
                    ricci[a][b].clone(),
                    Expr::neg(Expr::mul(vec![n_minus_two.clone(), nabla_u[a][b].clone()])),
                    Expr::neg(Expr::mul(vec![g.get(a, b).clone(), div_u.clone()])),
                    Expr::mul(vec![
                        n_minus_two.clone(),
                        Expr::add(vec![
                            Expr::mul(vec![u_down[a].clone(), u_down[b].clone()]),
                            Expr::neg(Expr::mul(vec![g.get(a, b).clone(), norm_u.clone()])),
                        ]),
                    ]),
                ]),
                interner,
            );
        }
    }
    Ok(transformed)
}

pub fn conformal_transform_scalar(
    scalar: &ax_ir::Expr,
    g: &crate::SymbolicMatrix,
    omega: &ax_ir::Expr,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, ConformalError> {
    validate_metric(g)?;
    if coords.len() != g.dim {
        return Err(ConformalError::CoordinateMismatch {
            coords_len: coords.len(),
            metric_dim: g.dim,
        });
    }
    let gamma = christoffel_from_metric(g, coords, interner);
    let g_inv = safe_inverse_metric(g, interner)?;
    let n = g.dim as i64;
    let u_down = u_covector(omega, coords, interner);
    let u_up = raise_covector(&u_down, &g_inv, interner);
    let div_u = divergence_vector(&u_up, &gamma, coords, interner);
    let norm_u = norm_covector(&u_down, &u_up, interner);
    let bracket = Expr::add(vec![
        scalar.clone(),
        Expr::neg(Expr::mul(vec![int_expr(2 * (n - 1)), div_u])),
        Expr::neg(Expr::mul(vec![int_expr((n - 1) * (n - 2)), norm_u])),
    ]);
    Ok(simplify_expr(
        Expr::mul(vec![
            Expr::pow(omega.clone(), Expr::Int((-2).into())),
            bracket,
        ]),
        interner,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        conformal_transform_christoffel, conformal_transform_inverse_metric,
        conformal_transform_metric, conformal_transform_ricci, conformal_transform_scalar,
        int_expr, ConformalError,
    };
    use crate::{
        christoffel_from_metric, ricci_from_riemann, ricci_scalar, riemann_from_christoffel,
        simplify_expr, weyl_from_curvature, SymbolicMatrix,
    };
    use ax_ir::{Convention, Expr, Interner};
    use num_rational::BigRational;

    fn identity_matrix(dim: usize) -> SymbolicMatrix {
        SymbolicMatrix::from_diagonal((0..dim).map(|_| Expr::one()).collect())
    }

    fn minkowski_metric() -> SymbolicMatrix {
        SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ])
    }

    #[test]
    fn constant_omega_scales_metric_and_inverse_metric_correctly() {
        let interner = Interner::new();
        let omega = Expr::Int(3.into());
        let g = minkowski_metric();
        let g_inv = g.symbolic_inverse(&interner);
        let transformed = conformal_transform_metric(&g, &omega, &interner);
        let transformed_inv = conformal_transform_inverse_metric(&g_inv, &omega, &interner);
        assert_eq!(transformed.get(0, 0), &Expr::Int((-9).into()));
        assert_eq!(transformed.get(1, 1), &Expr::Int(9.into()));
        assert_eq!(
            transformed_inv.get(0, 0),
            &Expr::Rational(BigRational::new((-1).into(), 9.into()))
        );
        assert_eq!(
            transformed_inv.get(1, 1),
            &Expr::Rational(BigRational::new(1.into(), 9.into()))
        );
    }

    #[test]
    fn constant_omega_leaves_christoffel_ricci_and_scalar_in_expected_form() {
        let interner = Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let omega = Expr::Int(5.into());
        let g = minkowski_metric();
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let transformed_gamma =
            conformal_transform_christoffel(&gamma, &g, &omega, &coords, &interner).expect("gamma");
        let transformed_ricci =
            conformal_transform_ricci(&ricci, &scalar, &g, &omega, &coords, &interner)
                .expect("ricci");
        let transformed_scalar =
            conformal_transform_scalar(&scalar, &g, &omega, &coords, &interner).expect("scalar");

        assert_eq!(transformed_gamma, gamma);
        assert_eq!(transformed_ricci, ricci);
        assert_eq!(
            transformed_scalar,
            Expr::mul(vec![Expr::pow(omega, Expr::Int((-2).into())), scalar])
        );
    }

    #[test]
    fn minkowski_with_constant_omega_remains_flat() {
        let interner = Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let omega = Expr::Int(7.into());
        let g = minkowski_metric();
        let transformed_metric = conformal_transform_metric(&g, &omega, &interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let transformed_gamma =
            conformal_transform_christoffel(&gamma, &g, &omega, &coords, &interner).expect("gamma");
        let riemann = riemann_from_christoffel(
            &transformed_gamma,
            &coords,
            &interner,
            &Convention::default(),
        );
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = conformal_transform_scalar(&Expr::zero(), &g, &omega, &coords, &interner)
            .expect("scalar");
        assert!(transformed_metric
            .data
            .iter()
            .flatten()
            .any(|entry| *entry != Expr::zero()));
        assert!(ricci.iter().flatten().all(|entry| *entry == Expr::zero()));
        assert_eq!(scalar, Expr::zero());
    }

    #[test]
    fn flat_frw_from_conformal_minkowski_matches_direct_curvature() {
        let interner = Interner::new();
        let eta = interner.get_or_intern("eta");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let a = interner.get_or_intern("a");
        let coords = vec![eta, x, y, z];
        let omega = Expr::Call(a, vec![Expr::Sym(eta)]);
        let g = minkowski_metric();
        let transformed_metric = conformal_transform_metric(&g, &omega, &interner);
        let direct_gamma = christoffel_from_metric(&transformed_metric, &coords, &interner);
        let direct_riemann =
            riemann_from_christoffel(&direct_gamma, &coords, &interner, &Convention::default());
        let direct_ricci =
            ricci_from_riemann(&direct_riemann, g.dim, &interner, &Convention::default());
        let direct_scalar = ricci_scalar(
            &direct_ricci,
            &transformed_metric.symbolic_inverse(&interner),
            &interner,
        );
        let formula_ricci = conformal_transform_ricci(
            &vec![vec![Expr::zero(); 4]; 4],
            &Expr::zero(),
            &g,
            &omega,
            &coords,
            &interner,
        )
        .expect("ricci");
        let formula_scalar =
            conformal_transform_scalar(&Expr::zero(), &g, &omega, &coords, &interner)
                .expect("scalar");

        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(
                    crate::simplify_invariant_expr(
                        Expr::add(vec![
                            formula_ricci[a][b].clone(),
                            Expr::neg(direct_ricci[a][b].clone()),
                        ]),
                        &interner
                    ),
                    Expr::zero()
                );
            }
        }
        assert_eq!(
            crate::simplify_invariant_expr(
                Expr::add(vec![formula_scalar, Expr::neg(direct_scalar)]),
                &interner
            ),
            Expr::zero()
        );
    }

    #[test]
    fn weyl_behavior_under_conformal_scaling_is_consistent() {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| {
                let interner = Interner::new();
                let t = interner.get_or_intern("t");
                let r = interner.get_or_intern("r");
                let theta = interner.get_or_intern("theta");
                let phi = interner.get_or_intern("phi");
                let sin = interner.get_or_intern("sin");
                let coords = vec![t, r, theta, phi];
                let f = Expr::add(vec![
                    Expr::one(),
                    Expr::mul(vec![
                        int_expr(-2),
                        Expr::pow(Expr::Sym(r), Expr::Int((-1).into())),
                    ]),
                ]);
                let g = SymbolicMatrix::from_diagonal(vec![
                    Expr::neg(f.clone()),
                    Expr::pow(f, Expr::Int((-1).into())),
                    Expr::pow(Expr::Sym(r), Expr::Int(2.into())),
                    Expr::mul(vec![
                        Expr::pow(Expr::Sym(r), Expr::Int(2.into())),
                        Expr::pow(Expr::Call(sin, vec![Expr::Sym(theta)]), Expr::Int(2.into())),
                    ]),
                ]);
                let omega = Expr::Int(3.into());
                let gamma = christoffel_from_metric(&g, &coords, &interner);
                let riemann =
                    riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
                let ricci = ricci_from_riemann(&riemann, 4, &interner, &Convention::default());
                let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
                let weyl =
                    weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
                let transformed_metric = conformal_transform_metric(&g, &omega, &interner);
                let omega_sq = Expr::pow(omega, Expr::Int(2.into()));
                for (a, b, c, d) in [
                    (0, 1, 0, 1),
                    (0, 2, 0, 2),
                    (0, 3, 0, 3),
                    (1, 2, 1, 2),
                    (1, 3, 1, 3),
                    (2, 3, 2, 3),
                ] {
                    let lowered = simplify_expr(
                        Expr::mul(vec![g.get(a, a).clone(), weyl[a][b][c][d].clone()]),
                        &interner,
                    );
                    let lowered_transformed = simplify_expr(
                        Expr::mul(vec![
                            transformed_metric.get(a, a).clone(),
                            weyl[a][b][c][d].clone(),
                        ]),
                        &interner,
                    );
                    assert_eq!(
                        simplify_expr(
                            Expr::add(vec![
                                lowered_transformed,
                                Expr::neg(Expr::mul(vec![omega_sq.clone(), lowered])),
                            ]),
                            &interner
                        ),
                        Expr::zero()
                    );
                }
            })
            .expect("spawn conformal Weyl test")
            .join()
            .expect("join conformal Weyl test");
    }

    #[test]
    fn coordinate_mismatch_errors() {
        let interner = Interner::new();
        let coords = vec![interner.get_or_intern("t")];
        let g = identity_matrix(2);
        let gamma = vec![vec![vec![Expr::zero(); 2]; 2]; 2];
        let ricci = vec![vec![Expr::zero(); 2]; 2];
        assert_eq!(
            conformal_transform_christoffel(&gamma, &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::CoordinateMismatch {
                coords_len: 1,
                metric_dim: 2
            })
        );
        assert_eq!(
            conformal_transform_ricci(&ricci, &Expr::zero(), &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::CoordinateMismatch {
                coords_len: 1,
                metric_dim: 2
            })
        );
        assert_eq!(
            conformal_transform_scalar(&Expr::zero(), &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::CoordinateMismatch {
                coords_len: 1,
                metric_dim: 2
            })
        );
    }

    #[test]
    fn invalid_metric_errors() {
        let interner = Interner::new();
        let coords = vec![interner.get_or_intern("t"), interner.get_or_intern("x")];
        let g = SymbolicMatrix {
            dim: 2,
            data: vec![vec![Expr::one(), Expr::zero()]],
        };
        let gamma = vec![vec![vec![Expr::zero(); 2]; 2]; 2];
        let ricci = vec![vec![Expr::zero(); 2]; 2];
        assert_eq!(
            conformal_transform_christoffel(&gamma, &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::InvalidMetric)
        );
        assert_eq!(
            conformal_transform_ricci(&ricci, &Expr::zero(), &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::InvalidMetric)
        );
        assert_eq!(
            conformal_transform_scalar(&Expr::zero(), &g, &Expr::one(), &coords, &interner),
            Err(ConformalError::InvalidMetric)
        );
    }
}
