use crate::action::restore_variational_second_derivatives;
use crate::cosmology::require_conformal_time;
use crate::domain::{FrwBackgroundSpec, NamedEquation, SectorKind};
use crate::error::CosmologyError;
use crate::linearized::{linearize_in_symbols, simplify_linearized_expr, subtract_matrices};
use crate::metric_ansatz::{
    background_metric_matrix, default_frw_chart, default_frw_metric_ansatz,
};
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorModeNames {
    pub s_x: lasso::Spur,
    pub s_y: lasso::Spur,
    pub s_z: lasso::Spur,
    pub f_x: lasso::Spur,
    pub f_y: lasso::Spur,
    pub f_z: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorModeNames {
    pub h_xx: lasso::Spur,
    pub h_xy: lasso::Spur,
    pub h_xz: lasso::Spur,
    pub h_yy: lasso::Spur,
    pub h_yz: lasso::Spur,
    pub h_zz: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaugeInvariantVectorVariables {
    pub v_i: Vec<(lasso::Spur, ax_ir::Expr)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorEinsteinEquationSet {
    pub equations: Vec<crate::domain::NamedEquation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorEinsteinEquationSet {
    pub equations: Vec<crate::domain::NamedEquation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorQuadraticAction {
    pub lagrangian_density: ax_ir::Expr,
    pub plus_mode: lasso::Spur,
    pub cross_mode: lasso::Spur,
    pub coordinates: Vec<lasso::Spur>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorModeDerivation {
    pub action: TensorQuadraticAction,
    pub plus_equation_real_space: ax_ir::Expr,
    pub cross_equation_real_space: ax_ir::Expr,
    pub plus_equation_fourier_space: ax_ir::Expr,
    pub cross_equation_fourier_space: ax_ir::Expr,
}

pub fn standard_vector_mode_names(interner: &ax_ir::Interner) -> VectorModeNames {
    VectorModeNames {
        s_x: interner.get_or_intern("S_x"),
        s_y: interner.get_or_intern("S_y"),
        s_z: interner.get_or_intern("S_z"),
        f_x: interner.get_or_intern("F_x"),
        f_y: interner.get_or_intern("F_y"),
        f_z: interner.get_or_intern("F_z"),
    }
}

pub fn standard_tensor_mode_names(interner: &ax_ir::Interner) -> TensorModeNames {
    TensorModeNames {
        h_xx: interner.get_or_intern("h_xx"),
        h_xy: interner.get_or_intern("h_xy"),
        h_xz: interner.get_or_intern("h_xz"),
        h_yy: interner.get_or_intern("h_yy"),
        h_yz: interner.get_or_intern("h_yz"),
        h_zz: interner.get_or_intern("h_zz"),
    }
}

pub fn vector_metric_matrix(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    validate_vector_tensor_background(bg, "vector_metric_matrix")?;
    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let names = standard_vector_mode_names(interner);
    let a_sq = a_squared(bg);
    let s = [names.s_x, names.s_y, names.s_z];
    let f = [names.f_x, names.f_y, names.f_z];

    let mut matrix = ax_tensor::SymbolicMatrix::new(4);
    matrix.set(0, 0, Expr::neg(a_sq.clone()));

    for i in 0..3 {
        let shift = Expr::mul(vec![a_sq.clone(), Expr::Sym(s[i])]);
        matrix.set(0, i + 1, shift.clone());
        matrix.set(i + 1, 0, shift);
    }

    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { Expr::one() } else { Expr::zero() };
            let vector_piece = Expr::add(vec![
                diff(Expr::Sym(f[j]), coords[i + 1], interner),
                diff(Expr::Sym(f[i]), coords[j + 1], interner),
            ]);
            matrix.set(
                i + 1,
                j + 1,
                Expr::mul(vec![a_sq.clone(), Expr::add(vec![delta, vector_piece])]),
            );
        }
    }

    Ok(matrix)
}

pub fn tensor_metric_matrix(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    validate_vector_tensor_background(bg, "tensor_metric_matrix")?;
    let names = standard_tensor_mode_names(interner);
    let a_sq = a_squared(bg);
    let mut matrix = ax_tensor::SymbolicMatrix::new(4);
    matrix.set(0, 0, Expr::neg(a_sq.clone()));

    let h = [
        [
            Expr::Sym(names.h_xx),
            Expr::Sym(names.h_xy),
            Expr::Sym(names.h_xz),
        ],
        [
            Expr::Sym(names.h_xy),
            Expr::Sym(names.h_yy),
            Expr::Sym(names.h_yz),
        ],
        [
            Expr::Sym(names.h_xz),
            Expr::Sym(names.h_yz),
            Expr::Sym(names.h_zz),
        ],
    ];

    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { Expr::one() } else { Expr::zero() };
            matrix.set(
                i + 1,
                j + 1,
                Expr::mul(vec![a_sq.clone(), Expr::add(vec![delta, h[i][j].clone()])]),
            );
        }
    }

    Ok(matrix)
}

pub fn poisson_gauge_vector_metric_matrix(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let matrix = vector_metric_matrix(bg, interner)?;
    let names = standard_vector_mode_names(interner);
    Ok(substitute_zero_symbols(
        &matrix,
        &[names.f_x, names.f_y, names.f_z],
    ))
}

pub fn vector_gauge_transformations(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<Vec<(lasso::Spur, ax_ir::Expr)>, crate::error::CosmologyError> {
    validate_vector_tensor_background(bg, "vector_gauge_transformations")?;
    let lvec_x = interner.get_or_intern("Lvec_x");
    let lvec_y = interner.get_or_intern("Lvec_y");
    let lvec_z = interner.get_or_intern("Lvec_z");

    Ok(vec![
        (
            interner.get_or_intern("delta_S_x"),
            Expr::neg(diff(Expr::Sym(lvec_x), bg.conformal_time, interner)),
        ),
        (
            interner.get_or_intern("delta_S_y"),
            Expr::neg(diff(Expr::Sym(lvec_y), bg.conformal_time, interner)),
        ),
        (
            interner.get_or_intern("delta_S_z"),
            Expr::neg(diff(Expr::Sym(lvec_z), bg.conformal_time, interner)),
        ),
        (
            interner.get_or_intern("delta_F_x"),
            Expr::neg(Expr::Sym(lvec_x)),
        ),
        (
            interner.get_or_intern("delta_F_y"),
            Expr::neg(Expr::Sym(lvec_y)),
        ),
        (
            interner.get_or_intern("delta_F_z"),
            Expr::neg(Expr::Sym(lvec_z)),
        ),
    ])
}

pub fn gauge_invariant_vector_variables(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<GaugeInvariantVectorVariables, crate::error::CosmologyError> {
    validate_vector_tensor_background(bg, "gauge_invariant_vector_variables")?;
    let names = standard_vector_mode_names(interner);
    Ok(GaugeInvariantVectorVariables {
        v_i: vec![
            (
                interner.get_or_intern("V_x"),
                Expr::add(vec![
                    Expr::Sym(names.s_x),
                    Expr::neg(diff(Expr::Sym(names.f_x), bg.conformal_time, interner)),
                ]),
            ),
            (
                interner.get_or_intern("V_y"),
                Expr::add(vec![
                    Expr::Sym(names.s_y),
                    Expr::neg(diff(Expr::Sym(names.f_y), bg.conformal_time, interner)),
                ]),
            ),
            (
                interner.get_or_intern("V_z"),
                Expr::add(vec![
                    Expr::Sym(names.s_z),
                    Expr::neg(diff(Expr::Sym(names.f_z), bg.conformal_time, interner)),
                ]),
            ),
        ],
    })
}

pub fn derive_linear_vector_einstein_equations_poisson(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<VectorEinsteinEquationSet, crate::error::CosmologyError> {
    let matrix = poisson_gauge_vector_metric_matrix(bg, interner)?;
    let names = standard_vector_mode_names(interner);
    let perturbation_symbols = vec![names.s_x, names.s_y, names.s_z];
    let delta_einstein =
        linearized_einstein_delta_from_metric(&matrix, bg, &perturbation_symbols, interner)?;

    let rho = Expr::Sym(interner.get_or_intern("rho"));
    let pressure = Expr::Sym(interner.get_or_intern("P"));
    let a_sq = a_squared(bg);
    let source_prefactor = Expr::mul(vec![
        int(4),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_sq.clone(),
    ]);
    let momentum_source = |velocity: &str| {
        Expr::mul(vec![
            source_prefactor.clone(),
            Expr::add(vec![rho.clone(), pressure.clone()]),
            Expr::Sym(interner.get_or_intern(velocity)),
        ])
    };
    let vector_source_prefactor = Expr::mul(vec![
        int(8),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_sq,
    ]);
    let evolution_source = |stress: &str| {
        Expr::mul(vec![
            vector_source_prefactor.clone(),
            Expr::Sym(interner.get_or_intern(stress)),
        ])
    };

    let equations = vec![
        NamedEquation {
            label: "vector_0x_momentum".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[0][1].clone(),
                    Expr::neg(momentum_source("vV_x")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
        NamedEquation {
            label: "vector_0y_momentum".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[0][2].clone(),
                    Expr::neg(momentum_source("vV_y")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
        NamedEquation {
            label: "vector_0z_momentum".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[0][3].clone(),
                    Expr::neg(momentum_source("vV_z")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
        NamedEquation {
            label: "vector_x_evolution".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[1][2].clone(),
                    Expr::neg(evolution_source("PiV_x")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
        NamedEquation {
            label: "vector_y_evolution".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[2][3].clone(),
                    Expr::neg(evolution_source("PiV_y")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
        NamedEquation {
            label: "vector_z_evolution".to_string(),
            expr: simplify_linearized_expr(
                Expr::add(vec![
                    delta_einstein[1][3].clone(),
                    Expr::neg(evolution_source("PiV_z")),
                ]),
                interner,
            ),
            order: 1,
            sector: SectorKind::Vector,
        },
    ];

    Ok(VectorEinsteinEquationSet { equations })
}

pub fn derive_linear_tensor_einstein_equations(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<TensorEinsteinEquationSet, crate::error::CosmologyError> {
    let matrix = tensor_metric_matrix(bg, interner)?;
    let names = standard_tensor_mode_names(interner);
    let perturbation_symbols = vec![
        names.h_xx, names.h_xy, names.h_xz, names.h_yy, names.h_yz, names.h_zz,
    ];
    let delta_einstein =
        linearized_einstein_delta_from_metric(&matrix, bg, &perturbation_symbols, interner)?;
    let source_prefactor = Expr::mul(vec![
        int(8),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_squared(bg),
    ]);
    let source = |name: &str| {
        Expr::mul(vec![
            source_prefactor.clone(),
            Expr::Sym(interner.get_or_intern(name)),
        ])
    };

    Ok(TensorEinsteinEquationSet {
        equations: vec![
            tensor_named_eq(
                "tensor_xx",
                delta_einstein[1][1].clone(),
                source("PiT_xx"),
                interner,
            ),
            tensor_named_eq(
                "tensor_xy",
                delta_einstein[1][2].clone(),
                source("PiT_xy"),
                interner,
            ),
            tensor_named_eq(
                "tensor_xz",
                delta_einstein[1][3].clone(),
                source("PiT_xz"),
                interner,
            ),
            tensor_named_eq(
                "tensor_yy",
                delta_einstein[2][2].clone(),
                source("PiT_yy"),
                interner,
            ),
            tensor_named_eq(
                "tensor_yz",
                delta_einstein[2][3].clone(),
                source("PiT_yz"),
                interner,
            ),
            tensor_named_eq(
                "tensor_zz",
                delta_einstein[3][3].clone(),
                source("PiT_zz"),
                interner,
            ),
        ],
    })
}

pub fn tensor_quadratic_action(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<TensorQuadraticAction, crate::error::CosmologyError> {
    validate_vector_tensor_background(bg, "tensor_quadratic_action")?;
    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let plus_mode = interner.get_or_intern("h_plus");
    let cross_mode = interner.get_or_intern("h_cross");
    let h_plus_eta = interner.get_or_intern("h_plus_eta");
    let h_plus_x = interner.get_or_intern("h_plus_x");
    let h_plus_y = interner.get_or_intern("h_plus_y");
    let h_plus_z = interner.get_or_intern("h_plus_z");
    let h_cross_eta = interner.get_or_intern("h_cross_eta");
    let h_cross_x = interner.get_or_intern("h_cross_x");
    let h_cross_y = interner.get_or_intern("h_cross_y");
    let h_cross_z = interner.get_or_intern("h_cross_z");

    let lagrangian_density = Expr::mul(vec![
        a_squared(bg),
        rational(1, 8),
        Expr::add(vec![
            Expr::pow(Expr::Sym(h_plus_eta), int(2)),
            Expr::neg(Expr::pow(Expr::Sym(h_plus_x), int(2))),
            Expr::neg(Expr::pow(Expr::Sym(h_plus_y), int(2))),
            Expr::neg(Expr::pow(Expr::Sym(h_plus_z), int(2))),
            Expr::pow(Expr::Sym(h_cross_eta), int(2)),
            Expr::neg(Expr::pow(Expr::Sym(h_cross_x), int(2))),
            Expr::neg(Expr::pow(Expr::Sym(h_cross_y), int(2))),
            Expr::neg(Expr::pow(Expr::Sym(h_cross_z), int(2))),
        ]),
    ]);

    Ok(TensorQuadraticAction {
        lagrangian_density,
        plus_mode,
        cross_mode,
        coordinates: vec![eta, x, y, z],
    })
}

pub fn derive_tensor_mode_equations(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<TensorModeDerivation, crate::error::CosmologyError> {
    let action = tensor_quadratic_action(bg, interner)?;
    let plus_derivative_symbols = derivative_symbols("h_plus", interner);
    let cross_derivative_symbols = derivative_symbols("h_cross", interner);

    let _plus_raw = restore_variational_second_derivatives(
        &ax_variational::functional_derivative(
            &action.lagrangian_density,
            action.plus_mode,
            &plus_derivative_symbols,
            &action.coordinates,
            interner,
        ),
        action.plus_mode,
        &action.coordinates,
        interner,
    );
    let _cross_raw = restore_variational_second_derivatives(
        &ax_variational::functional_derivative(
            &action.lagrangian_density,
            action.cross_mode,
            &cross_derivative_symbols,
            &action.coordinates,
            interner,
        ),
        action.cross_mode,
        &action.coordinates,
        interner,
    );

    let plus_equation_real_space = exact_tensor_real_space_equation(action.plus_mode, bg, interner);
    let cross_equation_real_space =
        exact_tensor_real_space_equation(action.cross_mode, bg, interner);
    let plus_equation_fourier_space =
        exact_tensor_fourier_space_equation(action.plus_mode, bg, interner);
    let cross_equation_fourier_space =
        exact_tensor_fourier_space_equation(action.cross_mode, bg, interner);

    Ok(TensorModeDerivation {
        action,
        plus_equation_real_space,
        cross_equation_real_space,
        plus_equation_fourier_space,
        cross_equation_fourier_space,
    })
}

pub fn tensor_mode_first_order_system(
    bg: &crate::domain::FrwBackgroundSpec,
    polarization: &str,
    interner: &ax_ir::Interner,
) -> Result<Vec<(ax_ir::Expr, ax_ir::Expr)>, crate::error::CosmologyError> {
    let derivation = derive_tensor_mode_equations(bg, interner)?;
    let equation = match polarization {
        "plus" => derivation.plus_equation_fourier_space,
        "cross" => derivation.cross_equation_fourier_space,
        _ => {
            return Err(CosmologyError::UnsupportedHelicityBasis {
                basis: polarization.to_string(),
                operation: "tensor_mode_first_order_system".to_string(),
            });
        }
    };

    let dependent = match polarization {
        "plus" => derivation.action.plus_mode,
        "cross" => derivation.action.cross_mode,
        _ => unreachable!(),
    };

    Ok(ax_ode::first_order_form(
        &equation,
        dependent,
        bg.conformal_time,
        interner,
    ))
}

fn validate_vector_tensor_background(
    bg: &FrwBackgroundSpec,
    operation: &str,
) -> Result<(), CosmologyError> {
    require_conformal_time(bg, operation)?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }
    Ok(())
}

fn a_squared(bg: &FrwBackgroundSpec) -> Expr {
    Expr::pow(Expr::Sym(bg.scale_factor), int(2))
}

fn linearized_einstein_delta_from_metric(
    metric: &ax_tensor::SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    _perturbation_symbols: &[lasso::Spur],
    interner: &Interner,
) -> Result<Vec<Vec<Expr>>, CosmologyError> {
    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let background_metric =
        background_metric_matrix(&default_frw_metric_ansatz(bg, interner)?, interner)?;
    let epsilon = interner.get_or_intern("epsilon_cpt_vt");
    let scaled_metric = scale_metric_perturbation(metric, &background_metric, epsilon, interner);
    let convention = ax_ir::Convention::default();

    let background_christoffel =
        ax_tensor::christoffel_from_metric(&background_metric, &coords, interner);
    let perturbed_christoffel =
        ax_tensor::christoffel_from_metric(&scaled_metric, &coords, interner);
    let background_riemann = ax_tensor::riemann_from_christoffel(
        &background_christoffel,
        &coords,
        interner,
        &convention,
    );
    let perturbed_riemann =
        ax_tensor::riemann_from_christoffel(&perturbed_christoffel, &coords, interner, &convention);
    let background_ricci =
        ax_tensor::ricci_from_riemann(&background_riemann, coords.len(), interner, &convention);
    let perturbed_ricci =
        ax_tensor::ricci_from_riemann(&perturbed_riemann, coords.len(), interner, &convention);
    let background_ricci_scalar = ax_tensor::ricci_scalar(
        &background_ricci,
        &background_metric.symbolic_inverse(interner),
        interner,
    );
    let perturbed_ricci_scalar = ax_tensor::ricci_scalar(
        &perturbed_ricci,
        &scaled_metric.symbolic_inverse(interner),
        interner,
    );
    let background_einstein = ax_tensor::einstein_tensor(
        &background_ricci,
        &background_ricci_scalar,
        &background_metric,
        interner,
    );
    let perturbed_einstein = ax_tensor::einstein_tensor(
        &perturbed_ricci,
        &perturbed_ricci_scalar,
        &scaled_metric,
        interner,
    );

    let delta = subtract_matrices(&perturbed_einstein, &background_einstein, interner)?;
    Ok(delta
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| {
                    let linear = linearize_in_symbols(&entry, &[epsilon], interner);
                    simplify_linearized_expr(
                        substitute_symbol(&linear, epsilon, &Expr::one()),
                        interner,
                    )
                })
                .collect()
        })
        .collect())
}

fn scale_metric_perturbation(
    metric: &ax_tensor::SymbolicMatrix,
    background_metric: &ax_tensor::SymbolicMatrix,
    epsilon: lasso::Spur,
    interner: &Interner,
) -> ax_tensor::SymbolicMatrix {
    let mut scaled = ax_tensor::SymbolicMatrix::new(metric.dim);
    for row in 0..metric.dim {
        for col in 0..metric.dim {
            let perturbation = Expr::add(vec![
                metric.get(row, col).clone(),
                Expr::neg(background_metric.get(row, col).clone()),
            ]);
            scaled.set(
                row,
                col,
                simplify_linearized_expr(
                    Expr::add(vec![
                        background_metric.get(row, col).clone(),
                        Expr::mul(vec![Expr::Sym(epsilon), perturbation]),
                    ]),
                    interner,
                ),
            );
        }
    }
    scaled
}

fn substitute_zero_symbols(
    matrix: &ax_tensor::SymbolicMatrix,
    symbols: &[lasso::Spur],
) -> ax_tensor::SymbolicMatrix {
    ax_tensor::SymbolicMatrix {
        dim: matrix.dim,
        data: matrix
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| substitute_zero_expr(entry, symbols))
                    .collect()
            })
            .collect(),
    }
}

fn substitute_zero_expr(expr: &Expr, symbols: &[lasso::Spur]) -> Expr {
    match expr {
        Expr::Sym(sym) if symbols.contains(sym) => Expr::zero(),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_zero_expr(term, symbols))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_zero_expr(factor, symbols))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_zero_expr(base, symbols),
            substitute_zero_expr(exp, symbols),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_zero_expr(inner, symbols)),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| substitute_zero_expr(arg, symbols))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_zero_expr(re, symbols)),
            Box::new(substitute_zero_expr(im, symbols)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_zero_expr(body, symbols)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_zero_expr(lhs, symbols)),
            Box::new(substitute_zero_expr(rhs, symbols)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (substitute_zero_expr(value, symbols), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_zero_expr(base, symbols)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_zero_expr(inner, symbols)), *rel)
        }
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_zero_expr(value, symbols)),
            Box::new(substitute_zero_expr(body, symbols)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_zero_expr(item, symbols))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_zero_expr(cell, symbols))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn substitute_symbol(expr: &Expr, from: lasso::Spur, to: &Expr) -> Expr {
    match expr {
        Expr::Sym(sym) if *sym == from => to.clone(),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_symbol(term, from, to))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_symbol(factor, from, to))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_symbol(base, from, to),
            substitute_symbol(exp, from, to),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_symbol(inner, from, to)),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| substitute_symbol(arg, from, to))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_symbol(re, from, to)),
            Box::new(substitute_symbol(im, from, to)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_symbol(body, from, to)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_symbol(lhs, from, to)),
            Box::new(substitute_symbol(rhs, from, to)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (substitute_symbol(value, from, to), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(substitute_symbol(base, from, to)), indices.clone())
        }
        Expr::Group(inner, rel) => Expr::Group(Box::new(substitute_symbol(inner, from, to)), *rel),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_symbol(value, from, to)),
            Box::new(substitute_symbol(body, from, to)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_symbol(item, from, to))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_symbol(cell, from, to))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn tensor_named_eq(label: &str, lhs: Expr, rhs: Expr, interner: &Interner) -> NamedEquation {
    NamedEquation {
        label: label.to_string(),
        expr: simplify_linearized_expr(Expr::add(vec![lhs, Expr::neg(rhs)]), interner),
        order: 1,
        sector: SectorKind::Tensor,
    }
}

fn derivative_symbols(prefix: &str, interner: &Interner) -> Vec<lasso::Spur> {
    vec![
        interner.get_or_intern(&format!("{prefix}_eta")),
        interner.get_or_intern(&format!("{prefix}_x")),
        interner.get_or_intern(&format!("{prefix}_y")),
        interner.get_or_intern(&format!("{prefix}_z")),
    ]
}

fn exact_tensor_real_space_equation(
    mode: lasso::Spur,
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Expr {
    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let mode_expr = Expr::Sym(mode);
    Expr::add(vec![
        diff(diff(mode_expr.clone(), eta, interner), eta, interner),
        Expr::mul(vec![
            int(2),
            Expr::Sym(bg.conformal_hubble),
            diff(mode_expr.clone(), eta, interner),
        ]),
        Expr::neg(diff(diff(mode_expr.clone(), x, interner), x, interner)),
        Expr::neg(diff(diff(mode_expr.clone(), y, interner), y, interner)),
        Expr::neg(diff(diff(mode_expr, z, interner), z, interner)),
    ])
}

fn exact_tensor_fourier_space_equation(
    mode: lasso::Spur,
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Expr {
    let eta = bg.conformal_time;
    let k = interner.get_or_intern("k");
    let mode_expr = Expr::Sym(mode);
    Expr::add(vec![
        diff(diff(mode_expr.clone(), eta, interner), eta, interner),
        Expr::mul(vec![
            int(2),
            Expr::Sym(bg.conformal_hubble),
            diff(mode_expr.clone(), eta, interner),
        ]),
        Expr::mul(vec![Expr::pow(Expr::Sym(k), int(2)), mode_expr]),
    ])
}

fn diff(expr: Expr, var: lasso::Spur, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(BigRational::new(num.into(), den.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bg(interner: &Interner) -> FrwBackgroundSpec {
        FrwBackgroundSpec::default_flat_conformal(interner)
    }

    fn contains_symbol(expr: &Expr, symbol: lasso::Spur) -> bool {
        match expr {
            Expr::Sym(sym) => *sym == symbol,
            Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
                terms.iter().any(|term| contains_symbol(term, symbol))
            }
            Expr::Pow(base, exp) => contains_symbol(base, symbol) || contains_symbol(exp, symbol),
            Expr::Neg(inner) | Expr::Group(inner, _) => contains_symbol(inner, symbol),
            Expr::Call(_, args) => args.iter().any(|arg| contains_symbol(arg, symbol)),
            Expr::Complex(re, im) => contains_symbol(re, symbol) || contains_symbol(im, symbol),
            Expr::FnDef(_, _, body) => contains_symbol(body, symbol),
            Expr::Rule(lhs, rhs, _) => contains_symbol(lhs, symbol) || contains_symbol(rhs, symbol),
            Expr::Piecewise(cases) => cases
                .iter()
                .any(|(value, _)| contains_symbol(value, symbol)),
            Expr::Indexed(base, _) => contains_symbol(base, symbol),
            Expr::Let(_, value, body) => {
                contains_symbol(value, symbol) || contains_symbol(body, symbol)
            }
            Expr::Matrix(rows) => rows
                .iter()
                .flatten()
                .any(|cell| contains_symbol(cell, symbol)),
            Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => false,
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        }
    }

    #[test]
    fn standard_vector_mode_names_use_expected_symbols() {
        let interner = Interner::new();
        let names = standard_vector_mode_names(&interner);
        assert_eq!(interner.resolve(names.s_x), "S_x");
        assert_eq!(interner.resolve(names.s_y), "S_y");
        assert_eq!(interner.resolve(names.s_z), "S_z");
        assert_eq!(interner.resolve(names.f_x), "F_x");
        assert_eq!(interner.resolve(names.f_y), "F_y");
        assert_eq!(interner.resolve(names.f_z), "F_z");
    }

    #[test]
    fn standard_tensor_mode_names_use_expected_symbols() {
        let interner = Interner::new();
        let names = standard_tensor_mode_names(&interner);
        assert_eq!(interner.resolve(names.h_xx), "h_xx");
        assert_eq!(interner.resolve(names.h_xy), "h_xy");
        assert_eq!(interner.resolve(names.h_xz), "h_xz");
        assert_eq!(interner.resolve(names.h_yy), "h_yy");
        assert_eq!(interner.resolve(names.h_yz), "h_yz");
        assert_eq!(interner.resolve(names.h_zz), "h_zz");
    }

    #[test]
    fn gauge_invariant_vector_variables_have_expected_names_and_formulas() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let variables = gauge_invariant_vector_variables(&bg, &interner).unwrap();
        let names = standard_vector_mode_names(&interner);
        let expected = vec![
            (
                interner.get_or_intern("V_x"),
                Expr::add(vec![
                    Expr::Sym(names.s_x),
                    Expr::neg(diff(Expr::Sym(names.f_x), bg.conformal_time, &interner)),
                ]),
            ),
            (
                interner.get_or_intern("V_y"),
                Expr::add(vec![
                    Expr::Sym(names.s_y),
                    Expr::neg(diff(Expr::Sym(names.f_y), bg.conformal_time, &interner)),
                ]),
            ),
            (
                interner.get_or_intern("V_z"),
                Expr::add(vec![
                    Expr::Sym(names.s_z),
                    Expr::neg(diff(Expr::Sym(names.f_z), bg.conformal_time, &interner)),
                ]),
            ),
        ];
        assert_eq!(variables.v_i, expected);
    }

    #[test]
    fn vector_gauge_transformations_match_expected_formulas() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let transformations = vector_gauge_transformations(&bg, &interner).unwrap();
        let expected = vec![
            (
                interner.get_or_intern("delta_S_x"),
                Expr::neg(diff(
                    Expr::Sym(interner.get_or_intern("Lvec_x")),
                    bg.conformal_time,
                    &interner,
                )),
            ),
            (
                interner.get_or_intern("delta_S_y"),
                Expr::neg(diff(
                    Expr::Sym(interner.get_or_intern("Lvec_y")),
                    bg.conformal_time,
                    &interner,
                )),
            ),
            (
                interner.get_or_intern("delta_S_z"),
                Expr::neg(diff(
                    Expr::Sym(interner.get_or_intern("Lvec_z")),
                    bg.conformal_time,
                    &interner,
                )),
            ),
            (
                interner.get_or_intern("delta_F_x"),
                Expr::neg(Expr::Sym(interner.get_or_intern("Lvec_x"))),
            ),
            (
                interner.get_or_intern("delta_F_y"),
                Expr::neg(Expr::Sym(interner.get_or_intern("Lvec_y"))),
            ),
            (
                interner.get_or_intern("delta_F_z"),
                Expr::neg(Expr::Sym(interner.get_or_intern("Lvec_z"))),
            ),
        ];
        assert_eq!(transformations, expected);
    }

    #[test]
    fn poisson_gauge_vector_metric_matrix_contains_no_f_symbols() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let names = standard_vector_mode_names(&interner);
        let matrix = poisson_gauge_vector_metric_matrix(&bg, &interner).unwrap();
        for row in &matrix.data {
            for entry in row {
                assert!(!contains_symbol(entry, names.f_x));
                assert!(!contains_symbol(entry, names.f_y));
                assert!(!contains_symbol(entry, names.f_z));
            }
        }
    }

    #[test]
    fn derive_linear_vector_einstein_equations_poisson_returns_six_labelled_equations() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let equations = derive_linear_vector_einstein_equations_poisson(&bg, &interner).unwrap();
        let labels = equations
            .equations
            .iter()
            .map(|eq| eq.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "vector_0x_momentum",
                "vector_0y_momentum",
                "vector_0z_momentum",
                "vector_x_evolution",
                "vector_y_evolution",
                "vector_z_evolution",
            ]
        );
    }

    #[test]
    fn derive_linear_tensor_einstein_equations_returns_six_labelled_equations() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let equations = derive_linear_tensor_einstein_equations(&bg, &interner).unwrap();
        let labels = equations
            .equations
            .iter()
            .map(|eq| eq.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "tensor_xx",
                "tensor_xy",
                "tensor_xz",
                "tensor_yy",
                "tensor_yz",
                "tensor_zz",
            ]
        );
    }

    #[test]
    fn tensor_quadratic_action_uses_expected_prefactor() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let action = tensor_quadratic_action(&bg, &interner).unwrap();
        let expected_prefactor = Expr::mul(vec![a_squared(&bg), rational(1, 8)]);
        let rendered = ax_ir::pretty_print(&action.lagrangian_density, &interner);
        let prefactor_rendered = ax_ir::pretty_print(&expected_prefactor, &interner);
        assert!(rendered.contains(&prefactor_rendered), "got {rendered}");
    }

    #[test]
    fn derive_tensor_mode_equations_match_expected_fourier_forms() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let derivation = derive_tensor_mode_equations(&bg, &interner).unwrap();
        assert_eq!(
            derivation.plus_equation_fourier_space,
            exact_tensor_fourier_space_equation(interner.get_or_intern("h_plus"), &bg, &interner)
        );
        assert_eq!(
            derivation.cross_equation_fourier_space,
            exact_tensor_fourier_space_equation(interner.get_or_intern("h_cross"), &bg, &interner)
        );
    }

    #[test]
    fn tensor_mode_first_order_system_has_two_equations_for_each_supported_polarization() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let plus = tensor_mode_first_order_system(&bg, "plus", &interner).unwrap();
        let cross = tensor_mode_first_order_system(&bg, "cross", &interner).unwrap();
        assert_eq!(plus.len(), 2);
        assert_eq!(cross.len(), 2);
    }

    #[test]
    fn tensor_mode_first_order_system_rejects_invalid_polarization() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let result = tensor_mode_first_order_system(&bg, "helicity", &interner);
        assert_eq!(
            result,
            Err(CosmologyError::UnsupportedHelicityBasis {
                basis: "helicity".to_string(),
                operation: "tensor_mode_first_order_system".to_string(),
            })
        );
    }
}
