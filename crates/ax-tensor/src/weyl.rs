use crate::{curvature_decompose, diff_component, simplify_expr};
use ax_ir::Expr;
use num_rational::BigRational;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WeylError {
    #[error("weyl_from_curvature dimension mismatch: metric dim {metric_dim}, riemann dim {riemann_dim}, ricci rows {ricci_rows}, ricci cols {ricci_cols}")]
    DimensionMismatch {
        metric_dim: usize,
        riemann_dim: usize,
        ricci_rows: usize,
        ricci_cols: usize,
    },
    #[error("weyl_from_curvature requires a square Ricci tensor")]
    NonSquareRicci,
    #[error("weyl_from_curvature requires a consistently shaped rank-4 Riemann tensor")]
    InvalidRiemannShape,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConformalCurvatureError {
    #[error("cotton_from_curvature dimension mismatch: metric dim {metric_dim}, gamma dim {gamma_dim}, ricci rows {ricci_rows}, ricci cols {ricci_cols}, coords len {coords_len}")]
    CottonDimensionMismatch {
        metric_dim: usize,
        gamma_dim: usize,
        ricci_rows: usize,
        ricci_cols: usize,
        coords_len: usize,
    },
    #[error("bach_from_curvature dimension mismatch: metric dim {metric_dim}, gamma dim {gamma_dim}, ricci rows {ricci_rows}, ricci cols {ricci_cols}, weyl dim {weyl_dim}, coords len {coords_len}")]
    BachDimensionMismatch {
        metric_dim: usize,
        gamma_dim: usize,
        ricci_rows: usize,
        ricci_cols: usize,
        weyl_dim: usize,
        coords_len: usize,
    },
    #[error("cotton_from_curvature requires a square Ricci tensor")]
    NonSquareRicci,
    #[error("bach_from_curvature requires a consistently shaped rank-4 Weyl tensor")]
    InvalidWeylShape,
    #[error("cotton_from_curvature is only defined here for dimension n >= 3")]
    CottonRequiresAtLeast3D,
    #[error("bach_from_curvature is only defined here for dimension n >= 4")]
    BachRequiresAtLeast4D,
}

fn zero_weyl(dim: usize) -> Vec<Vec<Vec<Vec<Expr>>>> {
    vec![vec![vec![vec![Expr::zero(); dim]; dim]; dim]; dim]
}

fn validate_riemann_shape(riemann: &[Vec<Vec<Vec<Expr>>>]) -> bool {
    let n = riemann.len();
    riemann.iter().all(|cube| {
        cube.len() == n
            && cube
                .iter()
                .all(|plane| plane.len() == n && plane.iter().all(|row| row.len() == n))
    })
}

fn ricci_shape(ricci: &[Vec<Expr>]) -> (usize, usize) {
    let rows = ricci.len();
    let cols = ricci.first().map(Vec::len).unwrap_or(0);
    (rows, cols)
}

fn validate_gamma_shape(gamma: &[Vec<Vec<Expr>>], n: usize) -> bool {
    gamma.len() == n
        && gamma
            .iter()
            .all(|plane| plane.len() == n && plane.iter().all(|row| row.len() == n))
}

fn zero_rank3(dim: usize) -> Vec<Vec<Vec<Expr>>> {
    vec![vec![vec![Expr::zero(); dim]; dim]; dim]
}

fn coefficient_expr(
    terms: &[curvature_decompose::LinearDecompositionTerm],
    kind: &str,
) -> Option<Expr> {
    terms.iter().find(|term| term.kind == kind).map(|term| {
        Expr::Rational(BigRational::new(
            term.coefficient_numer.into(),
            term.coefficient_denom.into(),
        ))
    })
}

fn lower_first_index_of_mixed_rank4(
    tensor: &[Vec<Vec<Vec<Expr>>>],
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Vec<Expr>>>> {
    let n = g.dim;
    let mut lowered = zero_weyl(n);
    for a in 0..n {
        for c in 0..n {
            for b in 0..n {
                for d in 0..n {
                    let terms = (0..n)
                        .map(|m| Expr::mul(vec![g.get(a, m).clone(), tensor[m][c][b][d].clone()]))
                        .collect::<Vec<_>>();
                    lowered[a][c][b][d] =
                        crate::simplify_invariant_expr(Expr::add(terms), interner);
                }
            }
        }
    }
    lowered
}

fn raise_ricci_both_indices(
    ricci: &[Vec<Expr>],
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Vec<Expr>> {
    let n = g.dim;
    let ginv = g.symbolic_inverse(interner);
    let mut raised = vec![vec![Expr::zero(); n]; n];
    for c in 0..n {
        for d in 0..n {
            let mut terms = Vec::with_capacity(n * n);
            for m in 0..n {
                for n_idx in 0..n {
                    terms.push(Expr::mul(vec![
                        ginv.get(c, m).clone(),
                        ginv.get(d, n_idx).clone(),
                        ricci[m][n_idx].clone(),
                    ]));
                }
            }
            raised[c][d] = crate::simplify_invariant_expr(Expr::add(terms), interner);
        }
    }
    raised
}

fn validate_square_ricci(ricci: &[Vec<Expr>]) -> Result<(usize, usize), ConformalCurvatureError> {
    let (rows, cols) = ricci_shape(ricci);
    if ricci.iter().any(|row| row.len() != rows) {
        return Err(ConformalCurvatureError::NonSquareRicci);
    }
    Ok((rows, cols))
}

fn covariant_derivative_tensor4_covariant(
    t: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Vec<ax_ir::Expr>>>> {
    let n = t.len();
    let mut out = zero_weyl(n);
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let mut terms = vec![diff_component(
                        &t[a][b][c][d],
                        coords[coord_index],
                        interner,
                    )];
                    for m in 0..n {
                        terms.push(Expr::neg(Expr::mul(vec![
                            gamma[m][coord_index][a].clone(),
                            t[m][b][c][d].clone(),
                        ])));
                        terms.push(Expr::neg(Expr::mul(vec![
                            gamma[m][coord_index][b].clone(),
                            t[a][m][c][d].clone(),
                        ])));
                        terms.push(Expr::neg(Expr::mul(vec![
                            gamma[m][coord_index][c].clone(),
                            t[a][b][m][d].clone(),
                        ])));
                        terms.push(Expr::neg(Expr::mul(vec![
                            gamma[m][coord_index][d].clone(),
                            t[a][b][c][m].clone(),
                        ])));
                    }
                    out[a][b][c][d] = simplify_expr(Expr::add(terms), interner);
                }
            }
        }
    }
    out
}

fn covariant_derivative_tensor2_covariant(
    t: &[Vec<ax_ir::Expr>],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let n = t.len();
    let mut out = vec![vec![Expr::zero(); n]; n];

    for a in 0..n {
        for b in 0..n {
            let mut terms = vec![diff_component(&t[a][b], coords[coord_index], interner)];
            for m in 0..n {
                terms.push(Expr::neg(Expr::mul(vec![
                    gamma[m][coord_index][a].clone(),
                    t[m][b].clone(),
                ])));
                terms.push(Expr::neg(Expr::mul(vec![
                    gamma[m][coord_index][b].clone(),
                    t[a][m].clone(),
                ])));
            }
            out[a][b] = simplify_expr(Expr::add(terms), interner);
        }
    }

    out
}

/// Compute the Weyl tensor from Riemann, Ricci, scalar curvature, and metric.
///
/// For dimensions n <= 3, return the identically zero Weyl tensor with the correct shape.
/// For n >= 4, use the standard n-dimensional formula.
pub fn weyl_from_curvature(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    ricci: &[Vec<ax_ir::Expr>],
    scalar: &ax_ir::Expr,
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<Vec<ax_ir::Expr>>>>, WeylError> {
    let metric_dim = g.dim;
    let riemann_dim = riemann.len();
    let (ricci_rows, ricci_cols) = ricci_shape(ricci);

    if !validate_riemann_shape(riemann) {
        return Err(WeylError::InvalidRiemannShape);
    }

    if ricci.iter().any(|row| row.len() != ricci_rows) {
        return Err(WeylError::NonSquareRicci);
    }

    if metric_dim != riemann_dim || metric_dim != ricci_rows || metric_dim != ricci_cols {
        return Err(WeylError::DimensionMismatch {
            metric_dim,
            riemann_dim,
            ricci_rows,
            ricci_cols,
        });
    }

    if metric_dim <= 3 {
        return Ok(zero_weyl(metric_dim));
    }

    let scalar = crate::simplify_invariant_expr(scalar.clone(), interner);
    let ricci = ricci
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| crate::simplify_invariant_expr(entry.clone(), interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let ginv = g.symbolic_inverse(interner);
    let ricci_mixed = (0..metric_dim)
        .map(|a| {
            (0..metric_dim)
                .map(|d| {
                    let raised_terms = (0..metric_dim)
                        .map(|m| Expr::mul(vec![ginv.get(a, m).clone(), ricci[m][d].clone()]))
                        .collect::<Vec<_>>();
                    crate::simplify_invariant_expr(Expr::add(raised_terms), interner)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let coefficients = curvature_decompose::riemann_to_weyl_ricci_scalar_coefficients(metric_dim)
        .map_err(|_| WeylError::InvalidRiemannShape)?;
    let Some(ricci_coeff) = coefficient_expr(&coefficients, "metric_ricci_rank4") else {
        return Err(WeylError::InvalidRiemannShape);
    };
    let Some(scalar_coeff) = coefficient_expr(&coefficients, "metric_scalar_rank4") else {
        return Err(WeylError::InvalidRiemannShape);
    };
    let mut weyl = zero_weyl(metric_dim);

    for a in 0..metric_dim {
        for b in 0..metric_dim {
            for c in 0..metric_dim {
                for d in 0..metric_dim {
                    let delta_ac = if a == c { Expr::one() } else { Expr::zero() };
                    let delta_ad = if a == d { Expr::one() } else { Expr::zero() };
                    let ricci_piece = Expr::add(vec![
                        Expr::mul(vec![delta_ac.clone(), ricci[d][b].clone()]),
                        Expr::neg(Expr::mul(vec![delta_ad.clone(), ricci[c][b].clone()])),
                        Expr::neg(Expr::mul(vec![
                            g.get(b, c).clone(),
                            ricci_mixed[a][d].clone(),
                        ])),
                        Expr::mul(vec![g.get(b, d).clone(), ricci_mixed[a][c].clone()]),
                    ]);
                    let scalar_piece = Expr::add(vec![
                        Expr::mul(vec![delta_ac, g.get(d, b).clone()]),
                        Expr::neg(Expr::mul(vec![delta_ad, g.get(c, b).clone()])),
                    ]);
                    weyl[a][b][c][d] = simplify_expr(
                        Expr::add(vec![
                            riemann[a][b][c][d].clone(),
                            Expr::neg(Expr::mul(vec![ricci_coeff.clone(), ricci_piece])),
                            Expr::neg(Expr::mul(vec![
                                scalar_coeff.clone(),
                                scalar.clone(),
                                scalar_piece,
                            ])),
                        ]),
                        interner,
                    );
                }
            }
        }
    }

    Ok(weyl)
}

/// Compute the Cotton tensor from Ricci, scalar curvature, Levi-Civita connection, metric, and coordinates.
///
/// Formula:
/// C_{abc} = ∇_c R_{ab} - ∇_b R_{ac}
///           + 1/(2(n-1)) * ( (∇_b R) g_{ac} - (∇_c R) g_{ab} )
pub fn cotton_from_curvature(
    ricci: &[Vec<ax_ir::Expr>],
    scalar: &ax_ir::Expr,
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    g: &crate::SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<ax_ir::Expr>>>, ConformalCurvatureError> {
    let metric_dim = g.dim;
    let gamma_dim = gamma.len();
    let (ricci_rows, ricci_cols) = validate_square_ricci(ricci)?;

    if metric_dim != gamma_dim
        || metric_dim != ricci_rows
        || metric_dim != ricci_cols
        || metric_dim != coords.len()
        || !validate_gamma_shape(gamma, metric_dim)
    {
        return Err(ConformalCurvatureError::CottonDimensionMismatch {
            metric_dim,
            gamma_dim,
            ricci_rows,
            ricci_cols,
            coords_len: coords.len(),
        });
    }

    if metric_dim < 3 {
        return Err(ConformalCurvatureError::CottonRequiresAtLeast3D);
    }

    let ricci_derivatives = (0..metric_dim)
        .map(|coord_index| {
            covariant_derivative_tensor2_covariant(ricci, gamma, coord_index, coords, interner)
        })
        .collect::<Vec<_>>();
    let scalar_gradient = (0..metric_dim)
        .map(|coord_index| diff_component(scalar, coords[coord_index], interner))
        .collect::<Vec<_>>();
    let coeff = Expr::Rational(BigRational::new(
        1.into(),
        (2 * (metric_dim - 1) as i64).into(),
    ));
    let mut cotton = zero_rank3(metric_dim);

    for a in 0..metric_dim {
        for b in 0..metric_dim {
            for c in 0..metric_dim {
                cotton[a][b][c] = simplify_expr(
                    Expr::add(vec![
                        ricci_derivatives[c][a][b].clone(),
                        Expr::neg(ricci_derivatives[b][a][c].clone()),
                        Expr::mul(vec![
                            coeff.clone(),
                            Expr::add(vec![
                                Expr::mul(vec![scalar_gradient[b].clone(), g.get(a, c).clone()]),
                                Expr::neg(Expr::mul(vec![
                                    scalar_gradient[c].clone(),
                                    g.get(a, b).clone(),
                                ])),
                            ]),
                        ]),
                    ]),
                    interner,
                );
            }
        }
    }

    Ok(cotton)
}

/// Compute the Bach tensor
///
/// B_{ab} = ∇^c ∇^d C_{acbd} + 1/2 R^{cd} C_{acbd}
///
/// where C_{acbd} is the Weyl tensor and ∇^c means raising the derivative index with the inverse metric.
pub fn bach_from_curvature(
    weyl: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    ricci: &[Vec<ax_ir::Expr>],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    g: &crate::SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<ax_ir::Expr>>, ConformalCurvatureError> {
    let metric_dim = g.dim;
    let gamma_dim = gamma.len();
    let weyl_dim = weyl.len();
    let (ricci_rows, ricci_cols) = validate_square_ricci(ricci)?;

    if !validate_riemann_shape(weyl) {
        return Err(ConformalCurvatureError::InvalidWeylShape);
    }

    if metric_dim != gamma_dim
        || metric_dim != ricci_rows
        || metric_dim != ricci_cols
        || metric_dim != weyl_dim
        || metric_dim != coords.len()
        || !validate_gamma_shape(gamma, metric_dim)
    {
        return Err(ConformalCurvatureError::BachDimensionMismatch {
            metric_dim,
            gamma_dim,
            ricci_rows,
            ricci_cols,
            weyl_dim,
            coords_len: coords.len(),
        });
    }

    if metric_dim < 4 {
        return Err(ConformalCurvatureError::BachRequiresAtLeast4D);
    }

    let ricci_is_zero = ricci
        .iter()
        .flatten()
        .all(|entry| crate::simplify_invariant_expr(entry.clone(), interner) == Expr::zero());
    if ricci_is_zero {
        return Ok(vec![vec![Expr::zero(); metric_dim]; metric_dim]);
    }

    let weyl_cov = lower_first_index_of_mixed_rank4(weyl, g, interner);
    let ginv = g.symbolic_inverse(interner);
    let ricci_raised = raise_ricci_both_indices(ricci, g, interner);
    let first_derivatives = (0..metric_dim)
        .map(|d_idx| {
            covariant_derivative_tensor4_covariant(&weyl_cov, gamma, d_idx, coords, interner)
        })
        .collect::<Vec<_>>();
    let second_derivatives = (0..metric_dim)
        .map(|c_deriv| {
            (0..metric_dim)
                .map(|d_deriv| {
                    covariant_derivative_tensor4_covariant(
                        &first_derivatives[d_deriv],
                        gamma,
                        c_deriv,
                        coords,
                        interner,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut bach = vec![vec![Expr::zero(); metric_dim]; metric_dim];
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));

    for a in 0..metric_dim {
        for b in 0..metric_dim {
            let mut divergence_terms = Vec::new();
            for c_slot in 0..metric_dim {
                for d_slot in 0..metric_dim {
                    for c_deriv in 0..metric_dim {
                        for d_deriv in 0..metric_dim {
                            divergence_terms.push(Expr::mul(vec![
                                ginv.get(c_slot, c_deriv).clone(),
                                ginv.get(d_slot, d_deriv).clone(),
                                second_derivatives[c_deriv][d_deriv][a][c_slot][b][d_slot].clone(),
                            ]));
                        }
                    }
                }
            }

            let mut ricci_weyl_terms = Vec::new();
            for c in 0..metric_dim {
                for d in 0..metric_dim {
                    ricci_weyl_terms.push(Expr::mul(vec![
                        half.clone(),
                        ricci_raised[c][d].clone(),
                        weyl_cov[a][c][b][d].clone(),
                    ]));
                }
            }

            bach[a][b] = simplify_expr(
                Expr::add(vec![
                    Expr::add(divergence_terms),
                    Expr::add(ricci_weyl_terms),
                ]),
                interner,
            );
        }
    }

    Ok(bach)
}

#[cfg(test)]
mod tests {
    use super::{
        bach_from_curvature, cotton_from_curvature, weyl_from_curvature, ConformalCurvatureError,
        WeylError,
    };
    use crate::{
        christoffel_from_metric, ricci_from_riemann, ricci_scalar, riemann_from_christoffel,
        simplify_expr, simplify_invariant_expr, SymbolicMatrix,
    };
    use ax_ir::{Convention, Expr, Interner};

    fn assert_rank4_zero(tensor: &[Vec<Vec<Vec<Expr>>>]) {
        assert!(
            tensor
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .all(|entry| *entry == Expr::zero()),
            "expected zero rank-4 tensor, got {:?}",
            tensor
        );
    }

    fn assert_rank3_zero(tensor: &[Vec<Vec<Expr>>]) {
        assert!(
            tensor
                .iter()
                .flatten()
                .flatten()
                .all(|entry| *entry == Expr::zero()),
            "expected zero rank-3 tensor, got {:?}",
            tensor
        );
    }

    fn assert_matrix_zero(matrix: &[Vec<Expr>]) {
        assert!(
            matrix.iter().flatten().all(|entry| *entry == Expr::zero()),
            "expected zero matrix, got {:?}",
            matrix
        );
    }

    fn schwarzschild_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");
        let theta = interner.get_or_intern("theta");
        let phi = interner.get_or_intern("phi");
        let radial = Expr::Sym(r);
        let f = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Int((-2).into()),
                Expr::pow(radial.clone(), Expr::Int((-1).into())),
            ]),
        ]);
        let sin = interner.get_or_intern("sin");

        let mut g = SymbolicMatrix::new(4);
        g.set(0, 0, Expr::neg(f.clone()));
        g.set(1, 1, Expr::pow(f, Expr::Int((-1).into())));
        g.set(2, 2, Expr::pow(radial.clone(), Expr::Int(2.into())));
        g.set(
            3,
            3,
            Expr::mul(vec![
                Expr::pow(radial, Expr::Int(2.into())),
                Expr::pow(Expr::Call(sin, vec![Expr::Sym(theta)]), Expr::Int(2.into())),
            ]),
        );

        (g, vec![t, r, theta, phi])
    }

    fn flat_frw_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let a = interner.get_or_intern("a");
        let scale = Expr::Call(a, vec![Expr::Sym(t)]);
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::pow(scale.clone(), Expr::Int(2.into())),
            Expr::pow(scale.clone(), Expr::Int(2.into())),
            Expr::pow(scale, Expr::Int(2.into())),
        ]);
        (g, vec![t, x, y, z])
    }

    fn three_dimensional_test_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let x_expr = Expr::Sym(x);
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::one(),
            Expr::add(vec![Expr::one(), x_expr.clone()]),
            Expr::add(vec![Expr::Int(2.into()), x_expr]),
        ]);
        (g, vec![x, y, z])
    }

    fn five_dimensional_test_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let w = interner.get_or_intern("w");
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        (g, vec![t, x, y, z, w])
    }

    #[test]
    fn weyl_vanishes_in_low_dimension() {
        let interner = Interner::new();
        let g = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        let riemann = vec![vec![vec![vec![Expr::zero(); 2]; 2]; 2]; 2];
        let ricci = vec![vec![Expr::zero(); 2]; 2];

        let weyl =
            weyl_from_curvature(&riemann, &ricci, &Expr::zero(), &g, &interner).expect("weyl");

        assert_eq!(weyl.len(), 2);
        assert_eq!(weyl[0].len(), 2);
        assert_eq!(weyl[0][0].len(), 2);
        assert_eq!(weyl[0][0][0].len(), 2);
        assert_rank4_zero(&weyl);
    }

    #[test]
    fn weyl_vanishes_in_three_dimensions() {
        let interner = Interner::new();
        let (g, coords) = three_dimensional_test_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
        assert_eq!(weyl.len(), 3);
        assert_eq!(weyl[0].len(), 3);
        assert_eq!(weyl[0][0].len(), 3);
        assert_eq!(weyl[0][0][0].len(), 3);
        for a in 0..g.dim {
            for b in 0..g.dim {
                for c in 0..g.dim {
                    for d in 0..g.dim {
                        assert_eq!(
                            simplify_invariant_expr(weyl[a][b][c][d].clone(), &interner),
                            Expr::zero(),
                            "3D Weyl component [{a}][{b}][{c}][{d}] should vanish"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn minkowski_weyl_is_zero() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
        assert_rank4_zero(&weyl);
    }

    #[test]
    fn schwarzschild_ricci_flat_implies_weyl_equals_riemann() {
        let interner = Interner::new();
        let (g, coords) = schwarzschild_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        for row in &ricci {
            for entry in row {
                assert_eq!(simplify_expr(entry.clone(), &interner), Expr::zero());
            }
        }
        assert_eq!(simplify_expr(scalar.clone(), &interner), Expr::zero());

        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
        for a in 0..g.dim {
            for b in 0..g.dim {
                for c in 0..g.dim {
                    for d in 0..g.dim {
                        assert_eq!(
                            weyl[a][b][c][d], riemann[a][b][c][d],
                            "Weyl-Riemann mismatch at [{a}][{b}][{c}][{d}]"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn flat_frw_is_conformally_flat_so_weyl_is_zero() {
        let interner = Interner::new();
        let (g, coords) = flat_frw_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
        for a in 0..g.dim {
            for b in 0..g.dim {
                for c in 0..g.dim {
                    for d in 0..g.dim {
                        assert_eq!(
                            simplify_invariant_expr(weyl[a][b][c][d].clone(), &interner),
                            Expr::zero(),
                            "FRW Weyl component [{a}][{b}][{c}][{d}] should vanish"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn five_dimensional_weyl_decomposition_has_correct_shape_and_simplifiable_entries() {
        let interner = Interner::new();
        let (g, coords) = five_dimensional_test_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");
        assert_eq!(weyl.len(), 5);
        assert_eq!(weyl[0].len(), 5);
        assert_eq!(weyl[0][0].len(), 5);
        assert_eq!(weyl[0][0][0].len(), 5);

        for a in 0..g.dim {
            for b in 0..g.dim {
                for c in 0..g.dim {
                    for d in 0..g.dim {
                        assert_eq!(
                            simplify_invariant_expr(weyl[a][b][c][d].clone(), &interner),
                            Expr::zero(),
                            "5D flat Weyl component [{a}][{b}][{c}][{d}] should vanish"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cotton_vanishes_for_schwarzschild() {
        let interner = Interner::new();
        let (g, coords) = schwarzschild_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let cotton =
            cotton_from_curvature(&ricci, &scalar, &gamma, &g, &coords, &interner).expect("cotton");
        assert_rank3_zero(&cotton);
    }

    #[test]
    fn cotton_vanishes_for_flat_frw() {
        let interner = Interner::new();
        let (g, coords) = flat_frw_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);

        let cotton =
            cotton_from_curvature(&ricci, &scalar, &gamma, &g, &coords, &interner).expect("cotton");
        for a in 0..g.dim {
            for b in 0..g.dim {
                for c in 0..g.dim {
                    assert_eq!(
                        simplify_invariant_expr(cotton[a][b][c].clone(), &interner),
                        Expr::zero(),
                        "FRW Cotton component [{a}][{b}][{c}] should vanish"
                    );
                }
            }
        }
    }

    #[test]
    fn cotton_is_antisymmetric_in_last_two_slots() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let interner = Interner::new();
                let (g, coords) = three_dimensional_test_metric(&interner);
                let gamma = christoffel_from_metric(&g, &coords, &interner);
                let riemann =
                    riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
                let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
                let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
                let cotton = cotton_from_curvature(&ricci, &scalar, &gamma, &g, &coords, &interner)
                    .expect("cotton");

                let has_nonzero_component =
                    cotton.iter().flatten().flatten().any(|entry| {
                        simplify_invariant_expr(entry.clone(), &interner) != Expr::zero()
                    });
                assert!(
                    has_nonzero_component,
                    "3D test metric should produce a nontrivial Cotton tensor"
                );

                for a in 0..g.dim {
                    for b in 0..g.dim {
                        for c in 0..g.dim {
                            assert_eq!(
                                simplify_invariant_expr(
                                    Expr::add(vec![
                                        cotton[a][b][c].clone(),
                                        cotton[a][c][b].clone()
                                    ]),
                                    &interner,
                                ),
                                Expr::zero(),
                                "Cotton antisymmetry failed at [{a}][{b}][{c}]"
                            );
                        }
                    }
                }
            })
            .expect("spawn cotton antisymmetry worker")
            .join()
            .expect("cotton antisymmetry worker");
    }

    #[test]
    fn bach_vanishes_for_minkowski() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");

        let bach =
            bach_from_curvature(&weyl, &ricci, &gamma, &g, &coords, &interner).expect("bach");
        assert_matrix_zero(&bach);
    }

    #[test]
    fn bach_vanishes_for_schwarzschild() {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                let interner = Interner::new();
                let (g, coords) = schwarzschild_metric(&interner);
                let gamma = christoffel_from_metric(&g, &coords, &interner);
                let riemann =
                    riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
                let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
                let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
                let weyl =
                    weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");

                let bach = bach_from_curvature(&weyl, &ricci, &gamma, &g, &coords, &interner)
                    .expect("bach");
                for a in 0..g.dim {
                    for b in 0..g.dim {
                        assert_eq!(
                            simplify_invariant_expr(bach[a][b].clone(), &interner),
                            Expr::zero(),
                            "Schwarzschild Bach component [{a}][{b}] should vanish"
                        );
                    }
                }
            })
            .expect("spawn bach schwarzschild worker")
            .join()
            .expect("bach schwarzschild worker");
    }

    #[test]
    fn bach_shape_validation_errors() {
        let interner = Interner::new();
        let coords4 = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let g4 = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let gamma4 = vec![vec![vec![Expr::zero(); 4]; 4]; 4];
        let ricci4 = vec![vec![Expr::zero(); 4]; 4];

        let invalid_weyl = vec![vec![vec![vec![Expr::zero(); 4]; 4]; 4]];
        assert_eq!(
            bach_from_curvature(&invalid_weyl, &ricci4, &gamma4, &g4, &coords4, &interner),
            Err(ConformalCurvatureError::InvalidWeylShape)
        );

        let weyl4 = vec![vec![vec![vec![Expr::zero(); 4]; 4]; 4]; 4];
        let g3 = SymbolicMatrix::from_diagonal(vec![Expr::one(); 3]);
        let gamma3 = vec![vec![vec![Expr::zero(); 3]; 3]; 3];
        let ricci3 = vec![vec![Expr::zero(); 3]; 3];
        let coords3 = vec![
            interner.get_or_intern("u"),
            interner.get_or_intern("v"),
            interner.get_or_intern("w"),
        ];
        assert_eq!(
            bach_from_curvature(&weyl4, &ricci3, &gamma3, &g3, &coords3, &interner),
            Err(ConformalCurvatureError::BachDimensionMismatch {
                metric_dim: 3,
                gamma_dim: 3,
                ricci_rows: 3,
                ricci_cols: 3,
                weyl_dim: 4,
                coords_len: 3,
            })
        );

        let weyl3 = vec![vec![vec![vec![Expr::zero(); 3]; 3]; 3]; 3];
        assert_eq!(
            bach_from_curvature(&weyl3, &ricci3, &gamma3, &g3, &coords3, &interner),
            Err(ConformalCurvatureError::BachRequiresAtLeast4D)
        );
    }

    #[test]
    fn weyl_is_traceless() {
        let interner = Interner::new();
        let (g, coords) = schwarzschild_metric(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, &interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, &g, &interner).expect("weyl");

        for b in 0..g.dim {
            for d in 0..g.dim {
                let mut terms = Vec::new();
                for a in 0..g.dim {
                    if weyl[a][b][a][d] == Expr::zero() {
                        continue;
                    }
                    terms.push(weyl[a][b][a][d].clone());
                }
                assert_eq!(
                    simplify_invariant_expr(Expr::add(terms), &interner),
                    Expr::zero(),
                    "trace C^a_bad should vanish for b={b}, d={d}"
                );
            }
        }
    }

    #[test]
    fn weyl_shape_validation_errors() {
        let interner = Interner::new();
        let g = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);

        let invalid_riemann = vec![vec![vec![vec![Expr::zero(); 2]; 2]; 2]];
        let square_ricci = vec![vec![Expr::zero(); 2]; 2];
        assert_eq!(
            weyl_from_curvature(
                &invalid_riemann,
                &square_ricci,
                &Expr::zero(),
                &g,
                &interner
            ),
            Err(WeylError::InvalidRiemannShape)
        );

        let riemann = vec![vec![vec![vec![Expr::zero(); 2]; 2]; 2]; 2];
        let nonsquare_ricci = vec![vec![Expr::zero(); 2], vec![Expr::zero()]];
        assert_eq!(
            weyl_from_curvature(&riemann, &nonsquare_ricci, &Expr::zero(), &g, &interner),
            Err(WeylError::NonSquareRicci)
        );

        let mismatched_g = SymbolicMatrix::from_diagonal(vec![Expr::one(); 3]);
        assert_eq!(
            weyl_from_curvature(
                &riemann,
                &square_ricci,
                &Expr::zero(),
                &mismatched_g,
                &interner
            ),
            Err(WeylError::DimensionMismatch {
                metric_dim: 3,
                riemann_dim: 2,
                ricci_rows: 2,
                ricci_cols: 2,
            })
        );
    }
}
