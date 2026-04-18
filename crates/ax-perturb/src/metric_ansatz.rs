use crate::domain::{FrwBackgroundSpec, TimeCoordinate};
use crate::error::CosmologyError;
use crate::gauge::standard_svt_mode_names;
use ax_ir::{Expr, Variance};

/// Names the three Cartesian spatial coordinates used by the FRW metric ansatz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpatialCoordinateNames {
    /// The `x` coordinate symbol.
    pub x: lasso::Spur,
    /// The `y` coordinate symbol.
    pub y: lasso::Spur,
    /// The `z` coordinate symbol.
    pub z: lasso::Spur,
}

/// Stores the conformal-time plus Cartesian spatial coordinate chart for FRW.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrwCoordinateChart {
    /// The time coordinate symbol.
    pub time: lasso::Spur,
    /// The spatial Cartesian coordinate symbols.
    pub space: SpatialCoordinateNames,
}

impl FrwCoordinateChart {
    /// Returns the chart coordinates ordered as `[time, x, y, z]`.
    pub fn as_vec(&self) -> Vec<lasso::Spur> {
        vec![self.time, self.space.x, self.space.y, self.space.z]
    }
}

/// Names the scalar perturbation fields entering the FRW metric ansatz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarMetricModes {
    /// The scalar lapse perturbation `Phi`.
    pub phi: lasso::Spur,
    /// The scalar curvature perturbation `Psi`.
    pub psi: lasso::Spur,
    /// The scalar shift perturbation `B`.
    pub b: lasso::Spur,
    /// The scalar spatial-shear perturbation `E`.
    pub e: lasso::Spur,
}

/// Names the vector perturbation fields entering the FRW metric ansatz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorMetricModes {
    /// The transverse vector shift perturbation `S`.
    pub s: lasso::Spur,
    /// The transverse vector spatial perturbation `F`.
    pub f: lasso::Spur,
}

/// Names the tensor perturbation fields entering the FRW metric ansatz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMetricModes {
    /// The transverse-traceless tensor perturbation `h_TT`.
    pub h_tt: lasso::Spur,
}

/// Bundles the FRW background, chart, and mode names needed to build metric components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrwMetricAnsatz {
    /// The typed FRW background specification.
    pub background: crate::domain::FrwBackgroundSpec,
    /// The coordinate chart used by the component ansatz.
    pub chart: FrwCoordinateChart,
    /// Scalar perturbation mode names.
    pub scalar_modes: ScalarMetricModes,
    /// Vector perturbation mode names.
    pub vector_modes: VectorMetricModes,
    /// Tensor perturbation mode names.
    pub tensor_modes: TensorMetricModes,
}

/// Builds the default conformal-time Cartesian FRW chart `(eta, x, y, z)`.
pub fn default_frw_chart(
    interner: &ax_ir::Interner,
    bg: &crate::domain::FrwBackgroundSpec,
) -> Result<FrwCoordinateChart, crate::error::CosmologyError> {
    validate_metric_ansatz_background(bg)?;

    Ok(FrwCoordinateChart {
        time: bg.conformal_time,
        space: SpatialCoordinateNames {
            x: interner.get_or_intern("x"),
            y: interner.get_or_intern("y"),
            z: interner.get_or_intern("z"),
        },
    })
}

/// Builds the default FRW metric ansatz using the standard SVT mode names.
pub fn default_frw_metric_ansatz(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<FrwMetricAnsatz, crate::error::CosmologyError> {
    let chart = default_frw_chart(interner, bg)?;
    let names = standard_svt_mode_names(interner);

    Ok(FrwMetricAnsatz {
        background: bg.clone(),
        chart,
        scalar_modes: ScalarMetricModes {
            phi: names.phi,
            psi: names.psi,
            b: names.b,
            e: names.e,
        },
        vector_modes: VectorMetricModes {
            s: names.s,
            f: names.f,
        },
        tensor_modes: TensorMetricModes { h_tt: names.h_tt },
    })
}

/// Builds the background conformal FRW metric matrix.
pub fn background_metric_matrix(
    ansatz: &FrwMetricAnsatz,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let _ = interner;
    validate_metric_ansatz_background(&ansatz.background)?;

    let a_sq = scale_factor_squared(ansatz);
    let mut matrix = ax_tensor::SymbolicMatrix::new(4);
    matrix.set(0, 0, Expr::neg(a_sq.clone()));
    matrix.set(1, 1, a_sq.clone());
    matrix.set(2, 2, a_sq.clone());
    matrix.set(3, 3, a_sq);
    Ok(matrix)
}

/// Builds the first-order scalar-perturbed conformal FRW metric matrix.
pub fn scalar_perturbed_metric_matrix(
    ansatz: &FrwMetricAnsatz,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    validate_metric_ansatz_background(&ansatz.background)?;

    let a_sq = scale_factor_squared(ansatz);
    let phi = Expr::Sym(ansatz.scalar_modes.phi);
    let psi = Expr::Sym(ansatz.scalar_modes.psi);
    let b = Expr::Sym(ansatz.scalar_modes.b);
    let e = Expr::Sym(ansatz.scalar_modes.e);
    let coords = ansatz.chart.as_vec();

    let mut matrix = ax_tensor::SymbolicMatrix::new(4);
    matrix.set(
        0,
        0,
        Expr::mul(vec![
            Expr::neg(a_sq.clone()),
            Expr::add(vec![Expr::one(), Expr::mul(vec![int_expr(2), phi])]),
        ]),
    );

    for (slot, coord) in coords.iter().enumerate().skip(1) {
        let shift = Expr::mul(vec![a_sq.clone(), diff(b.clone(), *coord, interner)]);
        matrix.set(0, slot, shift.clone());
        matrix.set(slot, 0, shift);
    }

    for i in 1..4 {
        for j in 1..4 {
            let delta_term = diag_delta_term(i, j);
            let scalar_piece = if delta_term == Expr::zero() {
                Expr::zero()
            } else {
                Expr::mul(vec![
                    Expr::add(vec![
                        Expr::one(),
                        Expr::neg(Expr::mul(vec![int_expr(2), psi.clone()])),
                    ]),
                    delta_term,
                ])
            };
            let shear_piece = Expr::mul(vec![
                int_expr(2),
                diff(diff(e.clone(), coords[i], interner), coords[j], interner),
            ]);
            matrix.set(
                i,
                j,
                Expr::mul(vec![
                    a_sq.clone(),
                    Expr::add(vec![scalar_piece, shear_piece]),
                ]),
            );
        }
    }

    Ok(matrix)
}

/// Builds component rules for the background FRW metric.
pub fn background_metric_rules(
    metric_symbol: lasso::Spur,
    ansatz: &FrwMetricAnsatz,
    interner: &ax_ir::Interner,
) -> Result<Vec<ax_tensor::ComponentRule>, crate::error::CosmologyError> {
    let matrix = background_metric_matrix(ansatz, interner)?;
    Ok(component_rules_from_matrix(metric_symbol, &matrix, ansatz))
}

/// Builds component rules for the first-order scalar-perturbed FRW metric.
pub fn scalar_perturbed_metric_rules(
    metric_symbol: lasso::Spur,
    ansatz: &FrwMetricAnsatz,
    interner: &ax_ir::Interner,
) -> Result<Vec<ax_tensor::ComponentRule>, crate::error::CosmologyError> {
    let matrix = scalar_perturbed_metric_matrix(ansatz, interner)?;
    Ok(component_rules_from_matrix(metric_symbol, &matrix, ansatz))
}

/// Completes the inverse background metric rules from the background metric rules.
pub fn inverse_background_metric_rules(
    metric_symbol: lasso::Spur,
    inverse_metric_symbol: lasso::Spur,
    ansatz: &FrwMetricAnsatz,
    interner: &ax_ir::Interner,
) -> Result<Vec<ax_tensor::ComponentRule>, crate::error::CosmologyError> {
    let metric_rules = background_metric_rules(metric_symbol, ansatz, interner)?;
    Ok(ax_tensor::complete_inverse_metric(
        &metric_rules,
        metric_symbol,
        inverse_metric_symbol,
        &ansatz.chart.as_vec(),
        interner,
    ))
}

fn validate_metric_ansatz_background(bg: &FrwBackgroundSpec) -> Result<(), CosmologyError> {
    if bg.time_coordinate != TimeCoordinate::Conformal {
        return Err(CosmologyError::IncompatibleTimeCoordinate {
            time_coordinate: bg.time_coordinate,
            operation: "FRW metric ansatz".to_string(),
        });
    }
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }
    Ok(())
}

fn scale_factor_squared(ansatz: &FrwMetricAnsatz) -> Expr {
    Expr::pow(
        Expr::Sym(ansatz.background.scale_factor),
        Expr::Int(2.into()),
    )
}

fn component_rules_from_matrix(
    metric_symbol: lasso::Spur,
    matrix: &ax_tensor::SymbolicMatrix,
    ansatz: &FrwMetricAnsatz,
) -> Vec<ax_tensor::ComponentRule> {
    let coords = ansatz.chart.as_vec();
    let mut rules = Vec::new();

    for i in 0..matrix.dim {
        for j in 0..matrix.dim {
            let value = matrix.get(i, j).clone();
            if value == Expr::zero() {
                continue;
            }
            rules.push(ax_tensor::ComponentRule {
                tensor: metric_symbol,
                indices: vec![(coords[i], Variance::Down), (coords[j], Variance::Down)],
                value,
            });
        }
    }

    rules
}

fn diff(expr: Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn diag_delta_term(i: usize, j: usize) -> Expr {
    if i == j {
        Expr::one()
    } else {
        Expr::zero()
    }
}

fn int_expr(n: i64) -> Expr {
    Expr::Int(n.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ansatz(interner: &ax_ir::Interner) -> FrwMetricAnsatz {
        default_frw_metric_ansatz(
            &FrwBackgroundSpec::default_flat_conformal(interner),
            interner,
        )
        .expect("default FRW metric ansatz")
    }

    fn rule_exists(
        rules: &[ax_tensor::ComponentRule],
        left: lasso::Spur,
        right: lasso::Spur,
    ) -> bool {
        rules
            .iter()
            .any(|rule| rule.indices == vec![(left, Variance::Down), (right, Variance::Down)])
    }

    #[test]
    fn default_frw_chart_uses_eta_x_y_z() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let chart = default_frw_chart(&interner, &bg).expect("default FRW chart");

        assert_eq!(chart.time, bg.conformal_time);
        assert_eq!(interner.resolve(chart.space.x), "x");
        assert_eq!(interner.resolve(chart.space.y), "y");
        assert_eq!(interner.resolve(chart.space.z), "z");
        let coords = chart.as_vec();
        assert_eq!(coords.len(), 4);
        assert_eq!(
            coords,
            vec![
                bg.conformal_time,
                chart.space.x,
                chart.space.y,
                chart.space.z
            ]
        );
    }

    #[test]
    fn default_frw_chart_rejects_cosmic_time_background() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_cosmic(&interner);
        let result = default_frw_chart(&interner, &bg);

        match result {
            Err(CosmologyError::IncompatibleTimeCoordinate {
                operation,
                time_coordinate,
            }) => {
                assert_eq!(operation, "FRW metric ansatz");
                assert_eq!(time_coordinate, TimeCoordinate::Cosmic);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn default_frw_chart_rejects_non_three_spatial_dimensions() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::new(
            interner.get_or_intern("a"),
            interner.get_or_intern("H"),
            interner.get_or_intern("H_cosmic"),
            interner.get_or_intern("eta"),
            interner.get_or_intern("t"),
            2,
            crate::SpatialCurvature::Flat,
            TimeCoordinate::Conformal,
        )
        .expect("background spec");
        let result = default_frw_chart(&interner, &bg);

        assert_eq!(
            result,
            Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions { got: 2 })
        );
    }

    #[test]
    fn background_metric_matrix_has_expected_diagonal_entries() {
        let interner = ax_ir::Interner::new();
        let ansatz = default_ansatz(&interner);
        let matrix = background_metric_matrix(&ansatz, &interner).expect("background matrix");
        let a_sq = Expr::pow(
            Expr::Sym(ansatz.background.scale_factor),
            Expr::Int(2.into()),
        );

        assert_eq!(matrix.dim, 4);
        assert_eq!(matrix.get(0, 0), &Expr::neg(a_sq.clone()));
        assert_eq!(matrix.get(1, 1), &a_sq);
        assert_eq!(
            matrix.get(2, 2),
            &Expr::pow(
                Expr::Sym(ansatz.background.scale_factor),
                Expr::Int(2.into())
            )
        );
        assert_eq!(
            matrix.get(3, 3),
            &Expr::pow(
                Expr::Sym(ansatz.background.scale_factor),
                Expr::Int(2.into())
            )
        );
        assert_eq!(matrix.get(0, 1), &Expr::zero());
    }

    #[test]
    fn scalar_perturbed_metric_matrix_has_expected_representative_components() {
        let interner = ax_ir::Interner::new();
        let ansatz = default_ansatz(&interner);
        let matrix =
            scalar_perturbed_metric_matrix(&ansatz, &interner).expect("scalar perturbed matrix");
        let a_sq = Expr::pow(
            Expr::Sym(ansatz.background.scale_factor),
            Expr::Int(2.into()),
        );
        let phi = Expr::Sym(ansatz.scalar_modes.phi);
        let psi = Expr::Sym(ansatz.scalar_modes.psi);
        let b = Expr::Sym(ansatz.scalar_modes.b);
        let e = Expr::Sym(ansatz.scalar_modes.e);
        let x = ansatz.chart.space.x;
        let y = ansatz.chart.space.y;

        assert_eq!(
            matrix.get(0, 0),
            &Expr::mul(vec![
                Expr::neg(a_sq.clone()),
                Expr::add(vec![Expr::one(), Expr::mul(vec![int_expr(2), phi])]),
            ])
        );
        assert_eq!(
            matrix.get(0, 1),
            &Expr::mul(vec![a_sq.clone(), diff(b.clone(), x, &interner)])
        );
        assert_eq!(matrix.get(1, 0), matrix.get(0, 1));
        assert_eq!(
            matrix.get(1, 1),
            &Expr::mul(vec![
                a_sq.clone(),
                Expr::add(vec![
                    Expr::add(vec![
                        Expr::one(),
                        Expr::neg(Expr::mul(vec![int_expr(2), psi.clone()])),
                    ]),
                    Expr::mul(vec![
                        int_expr(2),
                        diff(diff(e.clone(), x, &interner), x, &interner),
                    ]),
                ]),
            ])
        );
        assert_eq!(
            matrix.get(1, 2),
            &Expr::mul(vec![
                a_sq,
                Expr::mul(vec![int_expr(2), diff(diff(e, x, &interner), y, &interner),]),
            ])
        );
    }

    #[test]
    fn background_metric_rules_skip_zero_components() {
        let interner = ax_ir::Interner::new();
        let ansatz = default_ansatz(&interner);
        let g = interner.get_or_intern("g");
        let rules = background_metric_rules(g, &ansatz, &interner).expect("background rules");

        assert_eq!(rules.len(), 4);
    }

    #[test]
    fn scalar_perturbed_metric_rules_emit_symmetric_off_diagonal_rules() {
        let interner = ax_ir::Interner::new();
        let ansatz = default_ansatz(&interner);
        let g = interner.get_or_intern("g");
        let rules =
            scalar_perturbed_metric_rules(g, &ansatz, &interner).expect("scalar perturbed rules");
        let eta = ansatz.chart.time;
        let x = ansatz.chart.space.x;
        let y = ansatz.chart.space.y;

        assert!(rule_exists(&rules, eta, x));
        assert!(rule_exists(&rules, x, eta));
        assert!(rule_exists(&rules, x, y));
        assert!(rule_exists(&rules, y, x));
    }

    #[test]
    fn inverse_background_metric_rules_match_conformal_frw_inverse() {
        let interner = ax_ir::Interner::new();
        let ansatz = default_ansatz(&interner);
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let rules =
            inverse_background_metric_rules(g, ginv, &ansatz, &interner).expect("inverse rules");
        let a_inv_sq = Expr::pow(
            Expr::Sym(ansatz.background.scale_factor),
            Expr::Int((-2).into()),
        );
        let coords = ansatz.chart.as_vec();

        assert!(rules.iter().any(|rule| {
            rule.indices == vec![(coords[0], Variance::Up), (coords[0], Variance::Up)]
                && rule.value == Expr::neg(a_inv_sq.clone())
        }));
        assert!(rules.iter().any(|rule| {
            rule.indices == vec![(coords[1], Variance::Up), (coords[1], Variance::Up)]
                && rule.value == a_inv_sq.clone()
        }));
        assert!(rules.iter().any(|rule| {
            rule.indices == vec![(coords[2], Variance::Up), (coords[2], Variance::Up)]
                && rule.value == a_inv_sq.clone()
        }));
        assert!(rules.iter().any(|rule| {
            rule.indices == vec![(coords[3], Variance::Up), (coords[3], Variance::Up)]
                && rule.value == a_inv_sq
        }));
    }
}
