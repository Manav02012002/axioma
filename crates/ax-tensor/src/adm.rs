use crate::{
    christoffel_from_metric, diff_component, ricci_from_riemann, riemann_from_christoffel,
    simplify_expr, SymbolicMatrix,
};
use ax_ir::{Convention, Expr};
use num_rational::BigRational;

#[derive(Debug, Clone, PartialEq)]
pub struct ADMDecomposition {
    pub lapse: ax_ir::Expr,
    pub shift_covector: Vec<ax_ir::Expr>,
    pub shift_vector: Vec<ax_ir::Expr>,
    pub spatial_metric: crate::SymbolicMatrix,
    pub spatial_inverse_metric: crate::SymbolicMatrix,
    pub extrinsic_curvature: Vec<Vec<ax_ir::Expr>>,
    pub hamiltonian_constraint: ax_ir::Expr,
    pub momentum_constraints: Vec<ax_ir::Expr>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ADMError {
    #[error("adm_decompose requires a square metric of dimension at least 2")]
    InvalidMetric,
    #[error(
        "adm_decompose time coordinate index {time_coord} is out of bounds for dimension {dim}"
    )]
    InvalidTimeCoordinate { time_coord: usize, dim: usize },
    #[error("adm_decompose failed to invert the spatial metric")]
    SpatialInverseFailed,
    #[error("adm_decompose requires coordinates length {coords_len} to match metric dimension {metric_dim}")]
    CoordinateMismatch {
        coords_len: usize,
        metric_dim: usize,
    },
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn sqrt_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("sqrt"), vec![expr])
}

fn simplify_lapse(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let simplified = simplify_expr(expr, interner);
    match simplified {
        Expr::Int(ref n) if *n == 1.into() => Expr::one(),
        Expr::Int(ref n) if *n == 0.into() => Expr::zero(),
        Expr::Rational(ref r) if *r == BigRational::new(1.into(), 1.into()) => Expr::one(),
        Expr::Rational(ref r) if *r == BigRational::new(0.into(), 1.into()) => Expr::zero(),
        other => simplify_expr(sqrt_expr(other, interner), interner),
    }
}

fn simplify_sum(terms: Vec<Expr>, interner: &ax_ir::Interner) -> Expr {
    simplify_expr(Expr::add(terms), interner)
}

fn simplify_matrix_entries(rows: Vec<Vec<Expr>>, interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| simplify_expr(cell, interner))
                .collect()
        })
        .collect()
}

fn invert_matrix(
    matrix: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<SymbolicMatrix, ADMError> {
    let inv_data =
        ax_linalg::inverse(&matrix.data, interner).ok_or(ADMError::SpatialInverseFailed)?;
    Ok(SymbolicMatrix {
        dim: matrix.dim,
        data: simplify_matrix_entries(inv_data, interner),
    })
}

fn covariant_derivative_spatial_covector(
    covector: &[Expr],
    spatial_gamma: &[Vec<Vec<Expr>>],
    coord_index: usize,
    spatial_coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Expr> {
    let n = covector.len();
    (0..n)
        .map(|j| {
            let partial = diff_component(&covector[j], spatial_coords[coord_index], interner);
            let connection_terms = (0..n)
                .filter_map(|k| {
                    if spatial_gamma[k][coord_index][j] == Expr::zero()
                        || covector[k] == Expr::zero()
                    {
                        None
                    } else {
                        Some(Expr::mul(vec![
                            spatial_gamma[k][coord_index][j].clone(),
                            covector[k].clone(),
                        ]))
                    }
                })
                .collect::<Vec<_>>();
            simplify_expr(
                Expr::add(vec![partial, Expr::neg(Expr::add(connection_terms))]),
                interner,
            )
        })
        .collect()
}

fn mixed_trace(
    spatial_inverse: &SymbolicMatrix,
    tensor: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Expr {
    let n = spatial_inverse.dim;
    let mut terms = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if spatial_inverse.get(i, j) == &Expr::zero() || tensor[i][j] == Expr::zero() {
                continue;
            }
            terms.push(Expr::mul(vec![
                spatial_inverse.get(i, j).clone(),
                tensor[i][j].clone(),
            ]));
        }
    }
    simplify_sum(terms, interner)
}

fn raise_first_index(
    spatial_inverse: &SymbolicMatrix,
    tensor: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Expr>> {
    let n = spatial_inverse.dim;
    let mut out = vec![vec![Expr::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut terms = Vec::new();
            for k in 0..n {
                if spatial_inverse.get(i, k) == &Expr::zero() || tensor[k][j] == Expr::zero() {
                    continue;
                }
                terms.push(Expr::mul(vec![
                    spatial_inverse.get(i, k).clone(),
                    tensor[k][j].clone(),
                ]));
            }
            out[i][j] = simplify_sum(terms, interner);
        }
    }
    out
}

fn contract_two_covariant_tensors(
    spatial_inverse: &SymbolicMatrix,
    lhs: &[Vec<Expr>],
    rhs: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Expr {
    let n = spatial_inverse.dim;
    let mut terms = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let term = [
                        spatial_inverse.get(i, k),
                        spatial_inverse.get(j, l),
                        &lhs[i][j],
                        &rhs[k][l],
                    ];
                    if term.iter().any(|expr| **expr == Expr::zero()) {
                        continue;
                    }
                    terms.push(Expr::mul(vec![
                        spatial_inverse.get(i, k).clone(),
                        spatial_inverse.get(j, l).clone(),
                        lhs[i][j].clone(),
                        rhs[k][l].clone(),
                    ]));
                }
            }
        }
    }
    simplify_sum(terms, interner)
}

/// Compute Christoffel symbols for the spatial metric in the given spatial coordinates.
pub fn spatial_christoffel(
    gamma_ij: &crate::SymbolicMatrix,
    spatial_coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    christoffel_from_metric(gamma_ij, spatial_coords, interner)
}

/// Compute the spatial Ricci tensor from the spatial metric alone.
pub fn spatial_ricci_tensor(
    gamma_ij: &crate::SymbolicMatrix,
    spatial_coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let gamma = spatial_christoffel(gamma_ij, spatial_coords, interner);
    let riemann =
        riemann_from_christoffel(&gamma, spatial_coords, interner, &Convention::default());
    ricci_from_riemann(&riemann, gamma_ij.dim, interner, &Convention::default())
}

/// Compute the spatial Ricci scalar from the spatial metric alone.
pub fn spatial_ricci_scalar(
    gamma_ij: &crate::SymbolicMatrix,
    spatial_coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let ricci = spatial_ricci_tensor(gamma_ij, spatial_coords, interner);
    let spatial_inverse =
        invert_matrix(gamma_ij, interner).unwrap_or_else(|_| SymbolicMatrix::new(gamma_ij.dim));
    mixed_trace(&spatial_inverse, &ricci, interner)
}

/// Perform the ADM decomposition of a component metric.
pub fn adm_decompose(
    g: &crate::SymbolicMatrix,
    coords: &[lasso::Spur],
    time_coord: usize,
    interner: &ax_ir::Interner,
) -> Result<ADMDecomposition, ADMError> {
    if g.dim < 2 || g.data.len() != g.dim || g.data.iter().any(|row| row.len() != g.dim) {
        return Err(ADMError::InvalidMetric);
    }
    if coords.len() != g.dim {
        return Err(ADMError::CoordinateMismatch {
            coords_len: coords.len(),
            metric_dim: g.dim,
        });
    }
    if time_coord >= g.dim {
        return Err(ADMError::InvalidTimeCoordinate {
            time_coord,
            dim: g.dim,
        });
    }

    let spatial_slots = (0..g.dim)
        .filter(|slot| *slot != time_coord)
        .collect::<Vec<_>>();
    let spatial_coords = spatial_slots
        .iter()
        .map(|slot| coords[*slot])
        .collect::<Vec<_>>();
    let spatial_dim = spatial_slots.len();

    let mut spatial_metric = SymbolicMatrix::new(spatial_dim);
    for (i, row_slot) in spatial_slots.iter().enumerate() {
        for (j, col_slot) in spatial_slots.iter().enumerate() {
            spatial_metric.set(
                i,
                j,
                simplify_expr(g.get(*row_slot, *col_slot).clone(), interner),
            );
        }
    }
    let spatial_inverse_metric = invert_matrix(&spatial_metric, interner)?;

    let shift_covector = spatial_slots
        .iter()
        .map(|slot| simplify_expr(g.get(time_coord, *slot).clone(), interner))
        .collect::<Vec<_>>();
    let shift_vector = (0..spatial_dim)
        .map(|i| {
            let mut terms = Vec::new();
            for j in 0..spatial_dim {
                if spatial_inverse_metric.get(i, j) == &Expr::zero()
                    || shift_covector[j] == Expr::zero()
                {
                    continue;
                }
                terms.push(Expr::mul(vec![
                    spatial_inverse_metric.get(i, j).clone(),
                    shift_covector[j].clone(),
                ]));
            }
            simplify_sum(terms, interner)
        })
        .collect::<Vec<_>>();

    let shift_norm = simplify_sum(
        (0..spatial_dim)
            .filter_map(|i| {
                if shift_covector[i] == Expr::zero() || shift_vector[i] == Expr::zero() {
                    None
                } else {
                    Some(Expr::mul(vec![
                        shift_covector[i].clone(),
                        shift_vector[i].clone(),
                    ]))
                }
            })
            .collect(),
        interner,
    );
    let lapse_sq = simplify_expr(
        Expr::add(vec![
            Expr::neg(g.get(time_coord, time_coord).clone()),
            shift_norm,
        ]),
        interner,
    );
    let lapse = simplify_lapse(lapse_sq, interner);

    let spatial_gamma = spatial_christoffel(&spatial_metric, &spatial_coords, interner);
    let dt_spatial_metric = (0..spatial_dim)
        .map(|i| {
            (0..spatial_dim)
                .map(|j| diff_component(&spatial_metric.data[i][j], coords[time_coord], interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut extrinsic_curvature = vec![vec![Expr::zero(); spatial_dim]; spatial_dim];
    for i in 0..spatial_dim {
        let d_i_shift = covariant_derivative_spatial_covector(
            &shift_covector,
            &spatial_gamma,
            i,
            &spatial_coords,
            interner,
        );
        for j in 0..spatial_dim {
            let d_j_shift = covariant_derivative_spatial_covector(
                &shift_covector,
                &spatial_gamma,
                j,
                &spatial_coords,
                interner,
            );
            let inner = Expr::add(vec![
                dt_spatial_metric[i][j].clone(),
                Expr::neg(d_i_shift[j].clone()),
                Expr::neg(d_j_shift[i].clone()),
            ]);
            extrinsic_curvature[i][j] = simplify_expr(
                Expr::neg(Expr::mul(vec![
                    half(),
                    Expr::pow(lapse.clone(), Expr::Int((-1).into())),
                    inner,
                ])),
                interner,
            );
        }
    }

    let k_trace = mixed_trace(&spatial_inverse_metric, &extrinsic_curvature, interner);
    let k_squared = simplify_expr(Expr::pow(k_trace.clone(), Expr::Int(2.into())), interner);
    let k_contract = contract_two_covariant_tensors(
        &spatial_inverse_metric,
        &extrinsic_curvature,
        &extrinsic_curvature,
        interner,
    );
    let spatial_ricci = spatial_ricci_tensor(&spatial_metric, &spatial_coords, interner);
    let spatial_scalar = mixed_trace(&spatial_inverse_metric, &spatial_ricci, interner);
    let hamiltonian_constraint = simplify_expr(
        Expr::add(vec![spatial_scalar, k_squared, Expr::neg(k_contract)]),
        interner,
    );

    let k_mixed = raise_first_index(&spatial_inverse_metric, &extrinsic_curvature, interner);
    let a_mixed = (0..spatial_dim)
        .map(|j| {
            (0..spatial_dim)
                .map(|i| {
                    let delta = if i == j { Expr::one() } else { Expr::zero() };
                    simplify_expr(
                        Expr::add(vec![
                            k_mixed[j][i].clone(),
                            Expr::neg(Expr::mul(vec![delta, k_trace.clone()])),
                        ]),
                        interner,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let momentum_constraints = (0..spatial_dim)
        .map(|i| {
            let mut terms = Vec::new();
            for j in 0..spatial_dim {
                let partial = diff_component(&a_mixed[j][i], spatial_coords[j], interner);
                terms.push(partial);
                for m in 0..spatial_dim {
                    if spatial_gamma[j][j][m] != Expr::zero() && a_mixed[m][i] != Expr::zero() {
                        terms.push(Expr::mul(vec![
                            spatial_gamma[j][j][m].clone(),
                            a_mixed[m][i].clone(),
                        ]));
                    }
                    if spatial_gamma[m][j][i] != Expr::zero() && a_mixed[j][m] != Expr::zero() {
                        terms.push(Expr::neg(Expr::mul(vec![
                            spatial_gamma[m][j][i].clone(),
                            a_mixed[j][m].clone(),
                        ])));
                    }
                }
            }
            simplify_sum(terms, interner)
        })
        .collect::<Vec<_>>();

    Ok(ADMDecomposition {
        lapse,
        shift_covector,
        shift_vector,
        spatial_metric,
        spatial_inverse_metric,
        extrinsic_curvature,
        hamiltonian_constraint,
        momentum_constraints,
    })
}

#[cfg(test)]
mod tests {
    use super::{adm_decompose, ADMError};
    use crate::{diff_component, simplify_expr, SymbolicMatrix};
    use ax_ir::Expr;

    #[test]
    fn minkowski_adm_is_trivial() {
        let interner = ax_ir::Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let adm = adm_decompose(&g, &coords, 0, &interner).expect("adm");
        assert_eq!(adm.lapse, Expr::one());
        assert!(adm
            .shift_covector
            .iter()
            .all(|entry| *entry == Expr::zero()));
        assert!(adm.shift_vector.iter().all(|entry| *entry == Expr::zero()));
        assert_eq!(
            adm.spatial_metric,
            SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one(), Expr::one()])
        );
        assert_eq!(
            adm.spatial_inverse_metric,
            SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one(), Expr::one()])
        );
        assert!(adm
            .extrinsic_curvature
            .iter()
            .flatten()
            .all(|entry| *entry == Expr::zero()));
        assert_eq!(adm.hamiltonian_constraint, Expr::zero());
        assert!(adm
            .momentum_constraints
            .iter()
            .all(|entry| *entry == Expr::zero()));
    }

    #[test]
    fn flat_frw_adm_has_zero_shift_and_expected_extrinsic_curvature() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let a = interner.get_or_intern("a");
        let scale = Expr::Call(a, vec![Expr::Sym(t)]);
        let scale_sq = Expr::pow(scale.clone(), Expr::Int(2.into()));
        let coords = vec![t, x, y, z];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            scale_sq.clone(),
            scale_sq.clone(),
            scale_sq.clone(),
        ]);

        let adm = adm_decompose(&g, &coords, 0, &interner).expect("adm");
        assert!(adm
            .shift_covector
            .iter()
            .all(|entry| *entry == Expr::zero()));
        assert!(adm.shift_vector.iter().all(|entry| *entry == Expr::zero()));
        assert_eq!(
            adm.spatial_metric,
            SymbolicMatrix::from_diagonal(vec![scale_sq.clone(), scale_sq.clone(), scale_sq])
        );

        let a_dot = diff_component(&scale, t, &interner);
        let expected_diag =
            simplify_expr(Expr::neg(Expr::mul(vec![scale.clone(), a_dot])), &interner);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    expected_diag.clone()
                } else {
                    Expr::zero()
                };
                assert_eq!(adm.extrinsic_curvature[i][j], expected);
            }
        }
    }

    #[test]
    fn adm_in_one_plus_one_dimensions_is_supported() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let coords = vec![t, x];
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int((-1).into()), Expr::one()]);

        let adm = adm_decompose(&g, &coords, 0, &interner).expect("adm");
        assert_eq!(adm.lapse, Expr::one());
        assert_eq!(adm.shift_covector, vec![Expr::zero()]);
        assert_eq!(adm.shift_vector, vec![Expr::zero()]);
        assert_eq!(
            adm.spatial_metric,
            SymbolicMatrix::from_diagonal(vec![Expr::one()])
        );
        assert_eq!(
            adm.spatial_inverse_metric,
            SymbolicMatrix::from_diagonal(vec![Expr::one()])
        );
        assert_eq!(adm.extrinsic_curvature, vec![vec![Expr::zero()]]);
        assert_eq!(adm.hamiltonian_constraint, Expr::zero());
        assert_eq!(adm.momentum_constraints, vec![Expr::zero()]);
    }

    #[test]
    fn time_coord_out_of_bounds_errors() {
        let interner = ax_ir::Interner::new();
        let coords = vec![interner.get_or_intern("t"), interner.get_or_intern("x")];
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int((-1).into()), Expr::one()]);
        assert_eq!(
            adm_decompose(&g, &coords, 2, &interner),
            Err(ADMError::InvalidTimeCoordinate {
                time_coord: 2,
                dim: 2,
            })
        );
    }

    #[test]
    fn coordinate_mismatch_errors() {
        let interner = ax_ir::Interner::new();
        let coords = vec![interner.get_or_intern("t")];
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int((-1).into()), Expr::one()]);
        assert_eq!(
            adm_decompose(&g, &coords, 0, &interner),
            Err(ADMError::CoordinateMismatch {
                coords_len: 1,
                metric_dim: 2,
            })
        );
    }

    #[test]
    fn spatial_inverse_failure_errors_when_spatial_metric_singular() {
        let interner = ax_ir::Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
        ];
        let g = SymbolicMatrix {
            dim: 3,
            data: vec![
                vec![Expr::Int((-1).into()), Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::one(), Expr::one()],
                vec![Expr::zero(), Expr::one(), Expr::one()],
            ],
        };
        assert_eq!(
            adm_decompose(&g, &coords, 0, &interner),
            Err(ADMError::SpatialInverseFailed)
        );
    }
}
