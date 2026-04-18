use crate::cosmology::require_conformal_time;
use crate::domain::FrwBackgroundSpec;
use crate::error::CosmologyError;
use crate::gauge_transform::{default_scalar_gauge_generator, normalize_scalar_gauge_expr};
use crate::linearized::{
    count_perturbation_degree, simplify_linearized_expr, strip_common_mixed_gradient,
    strip_common_single_gradient,
};
use crate::metric_ansatz::{
    background_metric_matrix, default_frw_chart, default_frw_metric_ansatz, FrwCoordinateChart,
};
use ax_ir::Expr;
use ax_tensor::SymbolicMatrix;
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondOrderScalarModes {
    pub phi_1: lasso::Spur,
    pub psi_1: lasso::Spur,
    pub b_1: lasso::Spur,
    pub e_1: lasso::Spur,
    pub phi_2: lasso::Spur,
    pub psi_2: lasso::Spur,
    pub b_2: lasso::Spur,
    pub e_2: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondOrderGaugeGenerator {
    pub time_shift_1: lasso::Spur,
    pub space_shift_1: lasso::Spur,
    pub time_shift_2: lasso::Spur,
    pub space_shift_2: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderScalarGaugeVariation {
    pub delta_phi_1: ax_ir::Expr,
    pub delta_psi_1: ax_ir::Expr,
    pub delta_b_1: ax_ir::Expr,
    pub delta_e_1: ax_ir::Expr,
    pub delta_phi_2: ax_ir::Expr,
    pub delta_psi_2: ax_ir::Expr,
    pub delta_b_2: ax_ir::Expr,
    pub delta_e_2: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderEinsteinEquationSplit {
    pub label: String,
    pub full: ax_ir::Expr,
    pub linear_second_order: ax_ir::Expr,
    pub quadratic_source: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderEinsteinSystem {
    pub equations: Vec<SecondOrderEinsteinEquationSplit>,
}

pub fn default_second_order_scalar_modes(interner: &ax_ir::Interner) -> SecondOrderScalarModes {
    SecondOrderScalarModes {
        phi_1: interner.get_or_intern("Phi_1"),
        psi_1: interner.get_or_intern("Psi_1"),
        b_1: interner.get_or_intern("B_1"),
        e_1: interner.get_or_intern("E_1"),
        phi_2: interner.get_or_intern("Phi_2"),
        psi_2: interner.get_or_intern("Psi_2"),
        b_2: interner.get_or_intern("B_2"),
        e_2: interner.get_or_intern("E_2"),
    }
}

pub fn default_second_order_gauge_generator(
    interner: &ax_ir::Interner,
) -> SecondOrderGaugeGenerator {
    SecondOrderGaugeGenerator {
        time_shift_1: interner.get_or_intern("T_1"),
        space_shift_1: interner.get_or_intern("L_1"),
        time_shift_2: interner.get_or_intern("T_2"),
        space_shift_2: interner.get_or_intern("L_2"),
    }
}

pub fn scalar_metric_piece_order_one(
    bg: &crate::domain::FrwBackgroundSpec,
    modes: &SecondOrderScalarModes,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    scalar_metric_piece(
        bg,
        modes.phi_1,
        modes.psi_1,
        Some(modes.b_1),
        Some(modes.e_1),
        interner,
    )
}

pub fn scalar_metric_piece_order_two(
    bg: &crate::domain::FrwBackgroundSpec,
    modes: &SecondOrderScalarModes,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    scalar_metric_piece(
        bg,
        modes.phi_2,
        modes.psi_2,
        Some(modes.b_2),
        Some(modes.e_2),
        interner,
    )
}

pub fn full_metric_with_second_order_parameter(
    bg: &crate::domain::FrwBackgroundSpec,
    modes: &SecondOrderScalarModes,
    epsilon: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let h1 = scalar_metric_piece_order_one(bg, modes, interner)?;
    let h2 = scalar_metric_piece_order_two(bg, modes, interner)?;
    let eps = Expr::Sym(epsilon);
    let eps2_over_2 = Expr::mul(vec![rational(1, 2), Expr::pow(eps.clone(), int(2))]);

    Ok(add_matrices(
        &add_matrices(&g0, &scale_matrix(&h1, eps), interner),
        &scale_matrix(&h2, eps2_over_2),
        interner,
    ))
}

pub fn expand_expr_in_parameter(
    expr: &ax_ir::Expr,
    parameter: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> crate::ExpandedExpression {
    let terms = crate::expand_in_epsilon(expr, parameter, max_order, interner);
    crate::collect_orders(terms, max_order)
}

pub fn expand_matrix_in_parameter(
    matrix: &ax_tensor::SymbolicMatrix,
    parameter: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<ax_tensor::SymbolicMatrix> {
    let mut out = (0..=max_order)
        .map(|_| ax_tensor::SymbolicMatrix::new(matrix.dim))
        .collect::<Vec<_>>();

    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            let expanded =
                expand_expr_in_parameter(matrix.get(row, col), parameter, max_order, interner);
            for order in 0..=max_order {
                let coefficient = expanded
                    .orders
                    .iter()
                    .find(|term| term.order == order)
                    .map(|term| term.expr.clone())
                    .unwrap_or_else(Expr::zero);
                out[order].set(row, col, coefficient);
            }
        }
    }

    out
}

pub fn lie_derivative_covariant_rank2(
    tensor: &ax_tensor::SymbolicMatrix,
    generator_components: &[ax_ir::Expr],
    chart: &crate::metric_ansatz::FrwCoordinateChart,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let coords = chart.as_vec();
    if generator_components.len() != 4 {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "lie_derivative_covariant_rank2".to_string(),
            got: generator_components.len(),
            expected: 4,
        });
    }
    if tensor.dim != coords.len() {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "lie_derivative_covariant_rank2".to_string(),
            got: tensor.dim,
            expected: coords.len(),
        });
    }

    let mut result = SymbolicMatrix::new(tensor.dim);
    for mu in 0..tensor.dim {
        for nu in 0..tensor.dim {
            let mut terms = Vec::new();
            for rho in 0..tensor.dim {
                terms.push(Expr::mul(vec![
                    generator_components[rho].clone(),
                    ax_tensor::diff_component(tensor.get(mu, nu), coords[rho], interner),
                ]));
                terms.push(Expr::mul(vec![
                    tensor.get(rho, nu).clone(),
                    ax_tensor::diff_component(&generator_components[rho], coords[mu], interner),
                ]));
                terms.push(Expr::mul(vec![
                    tensor.get(mu, rho).clone(),
                    ax_tensor::diff_component(&generator_components[rho], coords[nu], interner),
                ]));
            }
            result.set(mu, nu, simplify_linearized_expr(Expr::add(terms), interner));
        }
    }
    Ok(result)
}

pub fn scalar_generator_vector_first_order(
    generator: &SecondOrderGaugeGenerator,
    chart: &crate::metric_ansatz::FrwCoordinateChart,
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    vec![
        Expr::Sym(generator.time_shift_1),
        diff(Expr::Sym(generator.space_shift_1), chart.space.x, interner),
        diff(Expr::Sym(generator.space_shift_1), chart.space.y, interner),
        diff(Expr::Sym(generator.space_shift_1), chart.space.z, interner),
    ]
}

pub fn scalar_generator_vector_second_order(
    generator: &SecondOrderGaugeGenerator,
    chart: &crate::metric_ansatz::FrwCoordinateChart,
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    vec![
        Expr::Sym(generator.time_shift_2),
        diff(Expr::Sym(generator.space_shift_2), chart.space.x, interner),
        diff(Expr::Sym(generator.space_shift_2), chart.space.y, interner),
        diff(Expr::Sym(generator.space_shift_2), chart.space.z, interner),
    ]
}

pub fn extract_scalar_modes_from_metric_piece(
    piece: &ax_tensor::SymbolicMatrix,
    bg: &crate::domain::FrwBackgroundSpec,
    chart: &crate::metric_ansatz::FrwCoordinateChart,
    interner: &ax_ir::Interner,
    operation: &str,
) -> Result<(ax_ir::Expr, ax_ir::Expr, ax_ir::Expr, ax_ir::Expr), crate::error::CosmologyError> {
    if piece.dim != 4 {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: operation.to_string(),
            got: piece.dim,
            expected: 4,
        });
    }

    let a_inv_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(-2));
    let phi = simplify_linearized_expr(
        Expr::mul(vec![
            rational(-1, 2),
            a_inv_sq.clone(),
            piece.get(0, 0).clone(),
        ]),
        interner,
    );

    let b_raw = simplify_linearized_expr(
        Expr::mul(vec![a_inv_sq.clone(), piece.get(0, 1).clone()]),
        interner,
    );
    let b =
        strip_common_single_gradient(&b_raw, chart.space.x, interner, operation).map_err(|_| {
            CosmologyError::ScalarModeExtractionFailure {
                operation: operation.to_string(),
            }
        })?;

    let e_raw = simplify_linearized_expr(
        Expr::mul(vec![
            rational(1, 2),
            a_inv_sq.clone(),
            piece.get(1, 2).clone(),
        ]),
        interner,
    );
    let e = strip_common_mixed_gradient(&e_raw, chart.space.x, chart.space.y, interner, operation)
        .map_err(|_| CosmologyError::ScalarModeExtractionFailure {
            operation: operation.to_string(),
        })?;

    let psi = simplify_linearized_expr(
        Expr::mul(vec![
            rational(-1, 2),
            Expr::add(vec![
                Expr::mul(vec![a_inv_sq.clone(), piece.get(1, 1).clone()]),
                Expr::neg(Expr::mul(vec![
                    int(2),
                    diff(
                        diff(e.clone(), chart.space.x, interner),
                        chart.space.x,
                        interner,
                    ),
                ])),
            ]),
        ]),
        interner,
    );

    Ok((phi, psi, b, e))
}

pub fn second_order_scalar_gauge_variation(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderScalarGaugeVariation, crate::error::CosmologyError> {
    require_conformal_time(bg, "second-order scalar gauge transformation")?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }

    let modes = default_second_order_scalar_modes(interner);
    let generator = default_second_order_gauge_generator(interner);
    let chart = default_frw_chart(interner, bg)?;
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let h1 = scalar_metric_piece_order_one(bg, &modes, interner)?;
    let h2 = scalar_metric_piece_order_two(bg, &modes, interner)?;
    let lifted_symbols = second_order_lifted_symbols(&modes, &generator);
    let lifted_g0 = lift_matrix_for_derivation(&g0, bg, &chart, &lifted_symbols, interner);
    let lifted_h1 = lift_matrix_for_derivation(&h1, bg, &chart, &lifted_symbols, interner);
    let lifted_h2 = lift_matrix_for_derivation(&h2, bg, &chart, &lifted_symbols, interner);
    let xi1 = scalar_generator_vector_first_order(&generator, &chart, interner)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();
    let xi2 = scalar_generator_vector_second_order(&generator, &chart, interner)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();

    let l_xi1_g0 = lie_derivative_covariant_rank2(&lifted_g0, &xi1, &chart, interner)?;
    let l_xi2_g0 = lie_derivative_covariant_rank2(&lifted_g0, &xi2, &chart, interner)?;
    let l_xi1_l_xi1_g0 = lie_derivative_covariant_rank2(&l_xi1_g0, &xi1, &chart, interner)?;
    let l_xi1_h1 = lie_derivative_covariant_rank2(&lifted_h1, &xi1, &chart, interner)?;

    let h1_tilde = strip_lifted_matrix(
        &add_matrices(&lifted_h1, &scale_matrix(&l_xi1_g0, int_expr(-1)), interner),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );
    let h2_tilde = strip_lifted_matrix(
        &add_matrices(
            &add_matrices(
                &add_matrices(&lifted_h2, &scale_matrix(&l_xi2_g0, int_expr(-1)), interner),
                &l_xi1_l_xi1_g0,
                interner,
            ),
            &scale_matrix(&l_xi1_h1, int_expr(-2)),
            interner,
        ),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );

    let _transformed_1 = extract_scalar_modes_from_metric_piece(
        &h1_tilde,
        bg,
        &chart,
        interner,
        "second-order first-order scalar gauge transformation",
    )?;
    let transformed_2 = match extract_scalar_modes_from_metric_piece(
        &h2_tilde,
        bg,
        &chart,
        interner,
        "second-order scalar gauge transformation",
    ) {
        Ok(modes) => modes,
        Err(_) => {
            let zeroed_h2_tilde = substitute_matrix_many(
                &h2_tilde,
                &[
                    (generator.time_shift_1, Expr::zero()),
                    (generator.space_shift_1, Expr::zero()),
                    (modes.phi_1, Expr::zero()),
                    (modes.psi_1, Expr::zero()),
                    (modes.b_1, Expr::zero()),
                    (modes.e_1, Expr::zero()),
                ],
            );
            extract_scalar_modes_from_metric_piece(
                &zeroed_h2_tilde,
                bg,
                &chart,
                interner,
                "second-order scalar gauge transformation",
            )?
        }
    };

    let first_order_generator = crate::gauge_transform::ScalarGaugeGenerator {
        time_shift: generator.time_shift_1,
        spatial_shift: generator.space_shift_1,
    };
    let first_order_variation = crate::gauge_transform::scalar_metric_gauge_variation(
        bg,
        &first_order_generator,
        interner,
    )?;
    let delta_phi_1 = first_order_variation.delta_phi;
    let delta_psi_1 = first_order_variation.delta_psi;
    let delta_b_1 = first_order_variation.delta_b;
    let delta_e_1 = first_order_variation.delta_e;

    let delta_phi_2 = normalize_second_order_expr(
        Expr::add(vec![transformed_2.0, Expr::neg(Expr::Sym(modes.phi_2))]),
        bg,
        &generator,
        interner,
    );
    let delta_psi_2 = normalize_second_order_expr(
        Expr::add(vec![transformed_2.1, Expr::neg(Expr::Sym(modes.psi_2))]),
        bg,
        &generator,
        interner,
    );
    let delta_b_2 = normalize_second_order_expr(
        Expr::add(vec![transformed_2.2, Expr::neg(Expr::Sym(modes.b_2))]),
        bg,
        &generator,
        interner,
    );
    let delta_e_2 = normalize_second_order_expr(
        Expr::add(vec![transformed_2.3, Expr::neg(Expr::Sym(modes.e_2))]),
        bg,
        &generator,
        interner,
    );

    Ok(SecondOrderScalarGaugeVariation {
        delta_phi_1,
        delta_psi_1,
        delta_b_1,
        delta_e_1,
        delta_phi_2,
        delta_psi_2,
        delta_b_2,
        delta_e_2,
    })
}

pub fn second_order_newtonian_metric_with_parameter(
    bg: &crate::domain::FrwBackgroundSpec,
    modes: &SecondOrderScalarModes,
    epsilon: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let h1 = scalar_metric_piece(bg, modes.phi_1, modes.psi_1, None, None, interner)?;
    let h2 = scalar_metric_piece(bg, modes.phi_2, modes.psi_2, None, None, interner)?;
    let eps = Expr::Sym(epsilon);
    let eps2_over_2 = Expr::mul(vec![rational(1, 2), Expr::pow(eps.clone(), int(2))]);

    Ok(add_matrices(
        &add_matrices(&g0, &scale_matrix(&h1, eps), interner),
        &scale_matrix(&h2, eps2_over_2),
        interner,
    ))
}

pub fn classify_second_order_term(
    expr: &ax_ir::Expr,
    second_order_symbols: &[lasso::Spur],
    first_order_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Option<&'static str> {
    let second_degree = count_perturbation_degree(expr, second_order_symbols, interner);
    let first_degree = count_perturbation_degree(expr, first_order_symbols, interner);

    if second_degree == 1 && first_degree == 0 {
        Some("linear_second_order")
    } else if second_degree == 0 && first_degree >= 2 {
        Some("quadratic_source")
    } else {
        None
    }
}

pub fn split_second_order_equation(
    label: &str,
    expr: &ax_ir::Expr,
    second_order_symbols: &[lasso::Spur],
    first_order_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<SecondOrderEinsteinEquationSplit, crate::error::CosmologyError> {
    let full = simplify_linearized_expr(expr.clone(), interner);
    let terms = additive_terms(&full);
    let mut linear_terms = Vec::new();
    let mut quadratic_terms = Vec::new();

    for term in terms {
        match classify_second_order_term(&term, second_order_symbols, first_order_symbols, interner)
        {
            Some("linear_second_order") => linear_terms.push(term),
            Some("quadratic_source") => quadratic_terms.push(term),
            _ => {
                return Err(CosmologyError::UnclassifiedSecondOrderTerm {
                    label: label.to_string(),
                    rendered: ax_ir::pretty_print(&term, interner),
                })
            }
        }
    }

    let linear_second_order = Expr::add(linear_terms);
    let quadratic_source = Expr::add(quadratic_terms);
    Ok(SecondOrderEinsteinEquationSplit {
        label: label.to_string(),
        full: Expr::add(vec![linear_second_order.clone(), quadratic_source.clone()]),
        linear_second_order,
        quadratic_source,
    })
}

pub fn derive_second_order_scalar_einstein_system(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderEinsteinSystem, crate::error::CosmologyError> {
    require_conformal_time(bg, "second-order scalar Einstein derivation")?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }

    let modes = default_second_order_scalar_modes(interner);
    let eps = interner.get_or_intern("eps_cpt");
    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let metric = second_order_newtonian_metric_with_parameter(bg, &modes, eps, interner)?;
    let generator = default_second_order_gauge_generator(interner);
    let lifted_symbols = second_order_lifted_symbols(&modes, &generator);
    let lifted_metric = lift_matrix_for_derivation(&metric, bg, &chart, &lifted_symbols, interner);
    let convention = ax_ir::Convention::default();
    let gamma = ax_tensor::christoffel_from_metric(&lifted_metric, &coords, interner);
    let riemann = ax_tensor::riemann_from_christoffel(&gamma, &coords, interner, &convention);
    let ricci = ax_tensor::ricci_from_riemann(&riemann, coords.len(), interner, &convention);
    let ricci_scalar =
        ax_tensor::ricci_scalar(&ricci, &lifted_metric.symbolic_inverse(interner), interner);
    let einstein = ax_tensor::einstein_tensor(&ricci, &ricci_scalar, &lifted_metric, interner);
    let order_two = expand_matrix_in_parameter(&matrix_from_rank2(&einstein), eps, 2, interner)
        .into_iter()
        .nth(2)
        .map(|matrix| strip_lifted_matrix(&matrix, bg, &chart, &lifted_symbols, interner))
        .map(|matrix| substitute_matrix_many(&matrix, &[(eps, Expr::zero())]))
        .unwrap_or_else(|| SymbolicMatrix::new(4));

    let x = chart.space.x;
    let y = chart.space.y;
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    let g_newton = Expr::Sym(interner.get_or_intern("G"));
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let four_pi_g_a2 = Expr::mul(vec![int(4), pi.clone(), g_newton.clone(), a2.clone()]);
    let eight_pi_g_a2 = Expr::mul(vec![int(8), pi, g_newton, a2.clone()]);
    let rho = Expr::Sym(interner.get_or_intern("rho"));
    let pressure = Expr::Sym(interner.get_or_intern("P"));
    let delta_rho_2 = Expr::Sym(interner.get_or_intern("delta_rho_2"));
    let delta_p_2 = Expr::Sym(interner.get_or_intern("delta_P_2"));
    let v_2 = Expr::Sym(interner.get_or_intern("v_2"));
    let pi_2 = Expr::Sym(interner.get_or_intern("Pi_2"));

    let eq_00 = simplify_linearized_expr(
        Expr::add(vec![
            order_two.get(0, 0).clone(),
            Expr::neg(Expr::mul(vec![four_pi_g_a2.clone(), delta_rho_2.clone()])),
        ]),
        interner,
    );
    let eq_0x_raw = simplify_linearized_expr(
        Expr::add(vec![
            order_two.get(0, 1).clone(),
            Expr::mul(vec![
                four_pi_g_a2.clone(),
                Expr::add(vec![rho.clone(), pressure.clone()]),
                diff(v_2, x, interner),
            ]),
        ]),
        interner,
    );
    let eq_0i =
        strip_common_single_gradient(&eq_0x_raw, x, interner, "second-order scalar 0i projection")?;
    let spatial_trace_average = simplify_linearized_expr(
        Expr::mul(vec![
            rational(1, 3),
            Expr::add(vec![
                order_two.get(1, 1).clone(),
                order_two.get(2, 2).clone(),
                order_two.get(3, 3).clone(),
            ]),
        ]),
        interner,
    );
    let eq_trace = simplify_linearized_expr(
        Expr::add(vec![
            spatial_trace_average,
            Expr::neg(Expr::mul(vec![four_pi_g_a2.clone(), delta_p_2.clone()])),
        ]),
        interner,
    );
    let eq_xy_raw = simplify_linearized_expr(
        Expr::add(vec![
            order_two.get(1, 2).clone(),
            Expr::neg(Expr::mul(vec![
                eight_pi_g_a2,
                diff(diff(pi_2.clone(), x, interner), y, interner),
            ])),
        ]),
        interner,
    );
    let eq_traceless = match strip_common_mixed_gradient(
        &eq_xy_raw,
        x,
        y,
        interner,
        "second-order scalar ij traceless projection",
    ) {
        Ok(expr) => expr,
        Err(_) => eq_xy_raw,
    };

    let second_order_symbols = vec![
        modes.phi_2,
        modes.psi_2,
        modes.b_2,
        modes.e_2,
        interner.get_or_intern("delta_rho_2"),
        interner.get_or_intern("delta_P_2"),
        interner.get_or_intern("v_2"),
        interner.get_or_intern("Pi_2"),
    ];
    let first_order_symbols = vec![modes.phi_1, modes.psi_1, modes.b_1, modes.e_1];

    Ok(SecondOrderEinsteinSystem {
        equations: vec![
            split_second_order_equation(
                "second_order_00_constraint",
                &eq_00,
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_equation(
                "second_order_0i_momentum",
                &eq_0i,
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_equation(
                "second_order_ij_trace",
                &eq_trace,
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_equation(
                "second_order_ij_traceless",
                &eq_traceless,
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
        ],
    })
}

fn scalar_metric_piece(
    bg: &FrwBackgroundSpec,
    phi: lasso::Spur,
    psi: lasso::Spur,
    b: Option<lasso::Spur>,
    e: Option<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> Result<SymbolicMatrix, CosmologyError> {
    require_conformal_time(bg, "second-order scalar metric piece")?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }

    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let phi_expr = Expr::Sym(phi);
    let psi_expr = Expr::Sym(psi);
    let b_expr = b.map(Expr::Sym);
    let e_expr = e.map(Expr::Sym);

    let mut matrix = SymbolicMatrix::new(4);
    matrix.set(
        0,
        0,
        Expr::mul(vec![Expr::neg(a2.clone()), int_expr(2), phi_expr]),
    );

    for (slot, coord) in coords.iter().enumerate().skip(1) {
        let shift = match &b_expr {
            Some(b_expr) => Expr::mul(vec![a2.clone(), diff(b_expr.clone(), *coord, interner)]),
            None => Expr::zero(),
        };
        matrix.set(0, slot, shift.clone());
        matrix.set(slot, 0, shift);
    }

    for i in 1..4 {
        for j in 1..4 {
            let diagonal = if i == j {
                Expr::mul(vec![int_expr(-2), psi_expr.clone()])
            } else {
                Expr::zero()
            };
            let shear = match &e_expr {
                Some(e_expr) => Expr::mul(vec![
                    int_expr(2),
                    diff(
                        diff(e_expr.clone(), coords[i], interner),
                        coords[j],
                        interner,
                    ),
                ]),
                None => Expr::zero(),
            };
            matrix.set(
                i,
                j,
                Expr::mul(vec![a2.clone(), Expr::add(vec![diagonal, shear])]),
            );
        }
    }

    Ok(matrix)
}

fn add_matrices(
    lhs: &SymbolicMatrix,
    rhs: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(lhs.dim);
    for row in 0..lhs.dim {
        for col in 0..lhs.dim {
            out.set(
                row,
                col,
                simplify_linearized_expr(
                    Expr::add(vec![lhs.get(row, col).clone(), rhs.get(row, col).clone()]),
                    interner,
                ),
            );
        }
    }
    out
}

fn scale_matrix(matrix: &SymbolicMatrix, factor: Expr) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(matrix.dim);
    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            out.set(
                row,
                col,
                Expr::mul(vec![factor.clone(), matrix.get(row, col).clone()]),
            );
        }
    }
    out
}

fn matrix_from_rank2(entries: &[Vec<Expr>]) -> SymbolicMatrix {
    let mut matrix = SymbolicMatrix::new(entries.len());
    for (row, values) in entries.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            matrix.set(row, col, value.clone());
        }
    }
    matrix
}

fn additive_terms(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Add(terms) => terms.iter().flat_map(additive_terms).collect(),
        other => vec![other.clone()],
    }
}

fn diff(expr: Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn int_expr(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(BigRational::new(num.into(), den.into()))
}

fn substitute_expr(expr: &Expr, symbol: lasso::Spur, replacement: &Expr) -> Expr {
    match expr {
        Expr::Sym(sym) if *sym == symbol => replacement.clone(),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_expr(term, symbol, replacement))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_expr(factor, symbol, replacement))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_expr(base, symbol, replacement),
            substitute_expr(exp, symbol, replacement),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_expr(inner, symbol, replacement)),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| substitute_expr(arg, symbol, replacement))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_expr(re, symbol, replacement)),
            Box::new(substitute_expr(im, symbol, replacement)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_expr(body, symbol, replacement)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_expr(lhs, symbol, replacement)),
            Box::new(substitute_expr(rhs, symbol, replacement)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_expr(value, symbol, replacement),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_expr(base, symbol, replacement)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_expr(inner, symbol, replacement)), *rel)
        }
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_expr(value, symbol, replacement)),
            Box::new(substitute_expr(body, symbol, replacement)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_expr(item, symbol, replacement))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_expr(cell, symbol, replacement))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn substitute_matrix_many(
    matrix: &SymbolicMatrix,
    replacements: &[(lasso::Spur, Expr)],
) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(matrix.dim);
    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            let expr = replacements
                .iter()
                .fold(matrix.get(row, col).clone(), |acc, (sym, replacement)| {
                    substitute_expr(&acc, *sym, replacement)
                });
            out.set(row, col, expr);
        }
    }
    out
}

fn second_order_lifted_symbols(
    modes: &SecondOrderScalarModes,
    generator: &SecondOrderGaugeGenerator,
) -> Vec<lasso::Spur> {
    vec![
        modes.phi_1,
        modes.psi_1,
        modes.b_1,
        modes.e_1,
        modes.phi_2,
        modes.psi_2,
        modes.b_2,
        modes.e_2,
        generator.time_shift_1,
        generator.space_shift_1,
        generator.time_shift_2,
        generator.space_shift_2,
    ]
}

fn lift_matrix_for_derivation(
    matrix: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    lifted_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(matrix.dim);
    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            out.set(
                row,
                col,
                lift_expr_for_derivation(matrix.get(row, col), bg, chart, lifted_symbols, interner),
            );
        }
    }
    out
}

fn lift_expr_for_derivation(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    lifted_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let scalar_args = chart
        .as_vec()
        .into_iter()
        .map(Expr::Sym)
        .collect::<Vec<_>>();
    match expr {
        Expr::Sym(sym) if *sym == bg.scale_factor => {
            Expr::Call(*sym, vec![Expr::Sym(bg.conformal_time)])
        }
        Expr::Sym(sym) if lifted_symbols.contains(sym) => Expr::Call(*sym, scalar_args),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| lift_expr_for_derivation(term, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| lift_expr_for_derivation(factor, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            lift_expr_for_derivation(base, bg, chart, lifted_symbols, interner),
            lift_expr_for_derivation(exp, bg, chart, lifted_symbols, interner),
        ),
        Expr::Neg(inner) => Expr::neg(lift_expr_for_derivation(
            inner,
            bg,
            chart,
            lifted_symbols,
            interner,
        )),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| lift_expr_for_derivation(arg, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(lift_expr_for_derivation(
                re,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            Box::new(lift_expr_for_derivation(
                im,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(lift_expr_for_derivation(
                body,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(lift_expr_for_derivation(
                lhs,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            Box::new(lift_expr_for_derivation(
                rhs,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        lift_expr_for_derivation(value, bg, chart, lifted_symbols, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(lift_expr_for_derivation(
                base,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(lift_expr_for_derivation(
                inner,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(lift_expr_for_derivation(
                value,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            Box::new(lift_expr_for_derivation(
                body,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| lift_expr_for_derivation(item, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| {
                            lift_expr_for_derivation(cell, bg, chart, lifted_symbols, interner)
                        })
                        .collect()
                })
                .collect(),
        ),
        other => {
            let _ = interner;
            other.clone()
        }
    }
}

fn strip_lifted_matrix(
    matrix: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    lifted_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(matrix.dim);
    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            out.set(
                row,
                col,
                strip_lifted_expr(matrix.get(row, col), bg, chart, lifted_symbols, interner),
            );
        }
    }
    out
}

fn strip_lifted_expr(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    lifted_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let scalar_args = chart
        .as_vec()
        .into_iter()
        .map(Expr::Sym)
        .collect::<Vec<_>>();
    match expr {
        Expr::Call(sym, args)
            if *sym == bg.scale_factor && args == &[Expr::Sym(bg.conformal_time)] =>
        {
            Expr::Sym(*sym)
        }
        Expr::Call(sym, args) if lifted_symbols.contains(sym) && args == &scalar_args => {
            Expr::Sym(*sym)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| strip_lifted_expr(term, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| strip_lifted_expr(factor, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            strip_lifted_expr(base, bg, chart, lifted_symbols, interner),
            strip_lifted_expr(exp, bg, chart, lifted_symbols, interner),
        ),
        Expr::Neg(inner) => Expr::neg(strip_lifted_expr(
            inner,
            bg,
            chart,
            lifted_symbols,
            interner,
        )),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| strip_lifted_expr(arg, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(strip_lifted_expr(re, bg, chart, lifted_symbols, interner)),
            Box::new(strip_lifted_expr(im, bg, chart, lifted_symbols, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(strip_lifted_expr(body, bg, chart, lifted_symbols, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(strip_lifted_expr(lhs, bg, chart, lifted_symbols, interner)),
            Box::new(strip_lifted_expr(rhs, bg, chart, lifted_symbols, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        strip_lifted_expr(value, bg, chart, lifted_symbols, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(strip_lifted_expr(base, bg, chart, lifted_symbols, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(strip_lifted_expr(
                inner,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(strip_lifted_expr(
                value,
                bg,
                chart,
                lifted_symbols,
                interner,
            )),
            Box::new(strip_lifted_expr(body, bg, chart, lifted_symbols, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| strip_lifted_expr(item, bg, chart, lifted_symbols, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| strip_lifted_expr(cell, bg, chart, lifted_symbols, interner))
                        .collect()
                })
                .collect(),
        ),
        other => {
            let _ = interner;
            other.clone()
        }
    }
}

fn normalize_second_order_expr(
    expr: Expr,
    bg: &FrwBackgroundSpec,
    generator: &SecondOrderGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut current = simplify_linearized_expr(expr, interner);
    loop {
        let next = normalize_second_order_expr_once(current.clone(), bg, generator, interner);
        if next == current {
            return next;
        }
        current = next;
    }
}

fn normalize_second_order_expr_once(
    expr: Expr,
    bg: &FrwBackgroundSpec,
    generator: &SecondOrderGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| normalize_second_order_expr(term, bg, generator, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let normalized = factors
                .into_iter()
                .map(|factor| normalize_second_order_expr(factor, bg, generator, interner))
                .collect::<Vec<_>>();
            rewrite_hubble_patterns(normalized, bg, generator, interner)
        }
        Expr::Pow(base, exp) => Expr::pow(
            normalize_second_order_expr(*base, bg, generator, interner),
            normalize_second_order_expr(*exp, bg, generator, interner),
        ),
        Expr::Neg(inner) => Expr::neg(normalize_second_order_expr(*inner, bg, generator, interner)),
        Expr::Call(fun, args) => Expr::Call(
            fun,
            args.into_iter()
                .map(|arg| normalize_second_order_expr(arg, bg, generator, interner))
                .collect(),
        ),
        other => other,
    }
}

fn rewrite_hubble_patterns(
    factors: Vec<Expr>,
    bg: &FrwBackgroundSpec,
    generator: &SecondOrderGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Expr {
    if let Some(rewritten) = replace_mul_pair(
        &factors,
        |expr| matches_scale_factor_inverse(expr, bg),
        |expr| matches_scale_factor_prime(expr, bg, interner),
        || Expr::Sym(bg.conformal_hubble),
    ) {
        return simplify_linearized_expr(rewritten, interner);
    }

    for time_shift in [generator.time_shift_1, generator.time_shift_2] {
        if let Some(rewritten) = replace_mul_pair(
            &factors,
            |expr| matches_scale_factor_inverse(expr, bg),
            |expr| matches_diff_a_times(expr, bg, time_shift, interner),
            || {
                let proxy = default_scalar_gauge_generator(interner);
                let expr = Expr::mul(vec![
                    Expr::pow(Expr::Sym(bg.scale_factor), int(-1)),
                    diff(
                        Expr::mul(vec![Expr::Sym(bg.scale_factor), Expr::Sym(time_shift)]),
                        bg.conformal_time,
                        interner,
                    ),
                ]);
                let mut normalized = normalize_scalar_gauge_expr(expr, bg, &proxy, interner);
                normalized = substitute_expr(&normalized, proxy.time_shift, &Expr::Sym(time_shift));
                normalized
            },
        ) {
            return simplify_linearized_expr(rewritten, interner);
        }
    }

    Expr::mul(factors)
}

fn replace_mul_pair(
    factors: &[Expr],
    left_match: impl Fn(&Expr) -> bool,
    right_match: impl Fn(&Expr) -> bool,
    replacement: impl Fn() -> Expr,
) -> Option<Expr> {
    let left_idx = factors.iter().position(left_match)?;
    let right_idx = factors
        .iter()
        .enumerate()
        .find_map(|(idx, expr)| (idx != left_idx && right_match(expr)).then_some(idx))?;
    let mut rebuilt = Vec::new();
    let mut inserted = false;
    for (idx, factor) in factors.iter().enumerate() {
        if idx == left_idx || idx == right_idx {
            if !inserted {
                rebuilt.push(replacement());
                inserted = true;
            }
            continue;
        }
        rebuilt.push(factor.clone());
    }
    Some(Expr::mul(rebuilt))
}

fn matches_scale_factor_inverse(expr: &Expr, bg: &FrwBackgroundSpec) -> bool {
    *expr == Expr::pow(Expr::Sym(bg.scale_factor), int(-1))
}

fn matches_scale_factor_prime(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> bool {
    *expr == diff(Expr::Sym(bg.scale_factor), bg.conformal_time, interner)
}

fn matches_diff_a_times(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    time_shift: lasso::Spur,
    interner: &ax_ir::Interner,
) -> bool {
    *expr
        == diff(
            Expr::mul(vec![Expr::Sym(bg.scale_factor), Expr::Sym(time_shift)]),
            bg.conformal_time,
            interner,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn substitute_many(expr: &Expr, replacements: &[(lasso::Spur, Expr)]) -> Expr {
        replacements
            .iter()
            .fold(expr.clone(), |acc, (sym, replacement)| {
                substitute_expr(&acc, *sym, replacement)
            })
    }

    #[test]
    fn default_second_order_scalar_modes_use_expected_names() {
        let interner = ax_ir::Interner::new();
        let modes = default_second_order_scalar_modes(&interner);
        assert_eq!(interner.resolve(modes.phi_1), "Phi_1");
        assert_eq!(interner.resolve(modes.psi_1), "Psi_1");
        assert_eq!(interner.resolve(modes.b_1), "B_1");
        assert_eq!(interner.resolve(modes.e_1), "E_1");
        assert_eq!(interner.resolve(modes.phi_2), "Phi_2");
        assert_eq!(interner.resolve(modes.psi_2), "Psi_2");
        assert_eq!(interner.resolve(modes.b_2), "B_2");
        assert_eq!(interner.resolve(modes.e_2), "E_2");
    }

    #[test]
    fn default_second_order_gauge_generator_use_expected_names() {
        let interner = ax_ir::Interner::new();
        let generator = default_second_order_gauge_generator(&interner);
        assert_eq!(interner.resolve(generator.time_shift_1), "T_1");
        assert_eq!(interner.resolve(generator.space_shift_1), "L_1");
        assert_eq!(interner.resolve(generator.time_shift_2), "T_2");
        assert_eq!(interner.resolve(generator.space_shift_2), "L_2");
    }

    #[test]
    fn scalar_generator_vectors_have_expected_components() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let chart = default_frw_chart(&interner, &bg).unwrap();
        let generator = default_second_order_gauge_generator(&interner);
        let first = scalar_generator_vector_first_order(&generator, &chart, &interner);
        let second = scalar_generator_vector_second_order(&generator, &chart, &interner);

        assert_eq!(first.len(), 4);
        assert_eq!(second.len(), 4);
        assert_eq!(
            first[1],
            diff(Expr::Sym(generator.space_shift_1), chart.space.x, &interner)
        );
        assert_eq!(
            first[2],
            diff(Expr::Sym(generator.space_shift_1), chart.space.y, &interner)
        );
        assert_eq!(
            first[3],
            diff(Expr::Sym(generator.space_shift_1), chart.space.z, &interner)
        );
        assert_eq!(
            second[1],
            diff(Expr::Sym(generator.space_shift_2), chart.space.x, &interner)
        );
        assert_eq!(
            second[2],
            diff(Expr::Sym(generator.space_shift_2), chart.space.y, &interner)
        );
        assert_eq!(
            second[3],
            diff(Expr::Sym(generator.space_shift_2), chart.space.z, &interner)
        );
    }

    #[test]
    fn first_order_part_of_second_order_gauge_variation_matches_prompt3_formulas() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let generator = default_second_order_gauge_generator(&interner);
        let variation = second_order_scalar_gauge_variation(&bg, &interner).unwrap();
        let eta = Expr::Sym(bg.conformal_time);
        let h = Expr::Sym(bg.conformal_hubble);
        let t1 = Expr::Sym(generator.time_shift_1);
        let l1 = Expr::Sym(generator.space_shift_1);

        assert_eq!(
            variation.delta_phi_1,
            Expr::neg(Expr::add(vec![
                diff(t1.clone(), bg.conformal_time, &interner),
                Expr::mul(vec![h.clone(), t1.clone()]),
            ]))
        );
        assert_eq!(variation.delta_psi_1, Expr::mul(vec![h, t1.clone()]));
        assert_eq!(
            variation.delta_b_1,
            Expr::add(vec![
                t1,
                Expr::neg(diff(l1.clone(), bg.conformal_time, &interner))
            ])
        );
        assert_eq!(variation.delta_e_1, Expr::neg(l1));
        let _ = eta;
    }

    #[test]
    fn second_order_variation_reduces_to_first_order_form_when_first_order_generator_and_first_order_modes_are_zero(
    ) {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let modes = default_second_order_scalar_modes(&interner);
        let generator = default_second_order_gauge_generator(&interner);
        let variation = second_order_scalar_gauge_variation(&bg, &interner).unwrap();
        let zeroed = [
            (generator.time_shift_1, Expr::zero()),
            (generator.space_shift_1, Expr::zero()),
            (modes.phi_1, Expr::zero()),
            (modes.psi_1, Expr::zero()),
            (modes.b_1, Expr::zero()),
            (modes.e_1, Expr::zero()),
        ];
        let h = Expr::Sym(bg.conformal_hubble);
        let t2 = Expr::Sym(generator.time_shift_2);
        let l2 = Expr::Sym(generator.space_shift_2);

        assert_eq!(
            substitute_many(&variation.delta_phi_2, &zeroed),
            Expr::neg(Expr::add(vec![
                diff(t2.clone(), bg.conformal_time, &interner),
                Expr::mul(vec![h.clone(), t2.clone()]),
            ]))
        );
        assert_eq!(
            substitute_many(&variation.delta_psi_2, &zeroed),
            Expr::mul(vec![h, t2.clone()])
        );
        assert_eq!(
            substitute_many(&variation.delta_b_2, &zeroed),
            Expr::add(vec![
                t2.clone(),
                Expr::neg(diff(l2.clone(), bg.conformal_time, &interner))
            ])
        );
        assert_eq!(
            substitute_many(&variation.delta_e_2, &zeroed),
            Expr::neg(l2)
        );
    }

    #[test]
    fn expand_matrix_in_parameter_extracts_orders_correctly() {
        let interner = ax_ir::Interner::new();
        let eps = interner.get_or_intern("eps_cpt");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let mut matrix = SymbolicMatrix::new(1);
        matrix.set(
            0,
            0,
            Expr::add(vec![
                Expr::one(),
                Expr::mul(vec![Expr::Sym(eps), Expr::Sym(a)]),
                Expr::mul(vec![
                    rational(1, 2),
                    Expr::pow(Expr::Sym(eps), int(2)),
                    Expr::Sym(b),
                ]),
            ]),
        );

        let expanded = expand_matrix_in_parameter(&matrix, eps, 2, &interner);

        assert_eq!(expanded[0].get(0, 0), &Expr::one());
        assert_eq!(expanded[1].get(0, 0), &Expr::Sym(a));
        assert_eq!(
            expanded[2].get(0, 0),
            &Expr::mul(vec![rational(1, 2), Expr::Sym(b)])
        );
    }

    #[test]
    fn split_second_order_equation_separates_linear_and_quadratic_terms() {
        let interner = ax_ir::Interner::new();
        let phi1 = interner.get_or_intern("Phi_1");
        let psi1 = interner.get_or_intern("Psi_1");
        let phi2 = interner.get_or_intern("Phi_2");
        let expr = Expr::add(vec![
            Expr::Sym(phi2),
            Expr::mul(vec![Expr::Sym(phi1), Expr::Sym(psi1)]),
        ]);

        let split =
            split_second_order_equation("test", &expr, &[phi2], &[phi1, psi1], &interner).unwrap();

        assert_eq!(split.linear_second_order, Expr::Sym(phi2));
        assert_eq!(
            split.quadratic_source,
            Expr::mul(vec![Expr::Sym(phi1), Expr::Sym(psi1)])
        );
    }

    #[test]
    fn derive_second_order_scalar_einstein_system_preserves_public_labels() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let system = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();

        assert_eq!(system.equations[0].label, "second_order_00_constraint");
        assert_eq!(system.equations[1].label, "second_order_0i_momentum");
        assert_eq!(system.equations[2].label, "second_order_ij_trace");
        assert_eq!(system.equations[3].label, "second_order_ij_traceless");
    }

    #[test]
    fn quadratic_sources_vanish_when_first_order_modes_are_zero() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let modes = default_second_order_scalar_modes(&interner);
        let system = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();
        let replacements = [(modes.phi_1, Expr::zero()), (modes.psi_1, Expr::zero())];

        for equation in system.equations {
            let simplified = simplify_linearized_expr(
                substitute_many(&equation.quadratic_source, &replacements),
                &interner,
            );
            assert_eq!(simplified, Expr::zero());
        }
    }

    #[test]
    fn public_second_order_equations_no_longer_come_from_hand_built_templates() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let decomp = crate::gauge::svt_decompose_perturbation(3, &interner).unwrap();
        let public =
            crate::cosmology::linearized_einstein_second_order(&bg, &decomp, &interner).unwrap();
        let derived = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();

        assert_eq!(public.len(), derived.equations.len());
        for (lhs, rhs) in public.iter().zip(derived.equations.iter()) {
            assert_eq!(lhs.label, rhs.label);
            assert_eq!(lhs.expr, rhs.full);
        }
    }
}
