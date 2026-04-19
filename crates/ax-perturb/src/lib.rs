#![allow(clippy::too_many_arguments)]

pub mod action;
pub mod boltzmann_bridge;
pub mod cosmology;
pub mod cubic;
pub mod domain;
pub mod eft;
pub mod error;
pub mod gauge;
pub mod gauge_transform;
pub mod harmonics;
pub mod hierarchy;
pub mod linearized;
pub mod matter;
pub mod metric_ansatz;
pub mod multifield;
pub mod second_order;
pub mod second_order_vector_tensor;
pub mod vector_tensor;

pub use action::{
    canonical_scalar_real_space_mukhanov_sasaki_equation,
    canonical_scalar_reduced_quadratic_action, derive_mukhanov_sasaki_from_action,
    fourier_reduce_mukhanov_sasaki, mukhanov_sasaki_first_order_system, MukhanovSasakiDerivation,
    ReducedQuadraticAction,
};
pub use boltzmann_bridge::{
    export_boltzmann_bridge_system, standard_boltzmann_species_symbols,
    symbolic_boltzmann_bridge_system, BoltzmannBridgeSystem, BoltzmannSpeciesSymbols,
};
pub use cubic::{
    bispectrum_shape, cubic_fourier_kernel, export_cubic_vertex, reduced_cubic_mixed_action,
    reduced_cubic_scalar_action, reduced_cubic_tensor_action, BispectrumShapeValue,
    CubicInteractionChannel, FourierKernel, InteractionVertexExport, ReducedCubicAction,
};
pub use domain::{
    FrwBackgroundSpec, GaugeGeneratorNames, GaugeKind, HarmonicBasisKind, MatterKind,
    NamedEquation, NamedExpr, SectorKind, SpatialCurvature, SvtModeNames, TimeCoordinate,
};
pub use eft::{
    derive_eft_mode_equations, eft_mode_equations_named, eft_model_name,
    eft_quadratic_sector_named, eft_stability_named, export_eft_mode_rhs,
    extract_stability_conditions, reduced_eft_quadratic_sector, standard_eft_coefficients,
    EftCoefficientSet, EftModeEquations, EftModelKind, EftQuadraticSector, StabilityConditionSet,
};
pub use error::CosmologyError;
pub use gauge_transform::{
    bardeen_variations, default_scalar_gauge_generator, normalize_scalar_gauge_expr,
    scalar_metric_gauge_variation, GaugeInvariantCheck, ScalarGaugeGenerator, ScalarGaugeVariation,
};
pub use harmonics::{
    project_scalar_equations_to_harmonic_space, project_tensor_equations_to_harmonic_space,
    project_vector_equations_to_harmonic_space, render_harmonic_spec_unicode,
    scalar_laplacian_eigenvalue, standard_scalar_harmonic_spec, standard_tensor_harmonic_spec,
    standard_vector_harmonic_spec, tensor_helicity_basis_flat, tensor_laplacian_eigenvalue,
    vector_laplacian_eigenvalue, HarmonicProjectionRule, ProjectedEquationSet, ScalarHarmonicSpec,
    TensorHarmonicSpec, VectorHarmonicSpec,
};
pub use hierarchy::{
    benchmark_report_against_fixture, built_in_parity_reports, default_external_solver_hooks,
    export_hierarchy_system, hierarchy_spec, neutrino_hierarchy_system, photon_hierarchy_system,
    ExternalSolverHook, HierarchyClosure, HierarchyGauge, HierarchySpec, HierarchySystem,
    HierarchyVariable, ParityBenchmarkEntry, ParityBenchmarkReport,
};
pub use linearized::{
    count_perturbation_degree, derive_linearized_einstein_matrices,
    derive_linearized_scalar_equations_newtonian, linearize_in_symbols,
    linearized_scalar_equations_as_named, strip_common_mixed_gradient,
    strip_common_single_gradient, LinearizedEinsteinMatrices, LinearizedScalarEquationSet,
};
pub use matter::{
    perfect_fluid_linear_conservation_equations_newtonian, standard_canonical_scalar_symbols,
    standard_perfect_fluid_symbols, CanonicalScalarSymbols, MatterEquationSet, PerfectFluidSymbols,
};
pub use metric_ansatz::{
    background_metric_matrix, background_metric_rules, default_frw_chart,
    default_frw_metric_ansatz, inverse_background_metric_rules, scalar_perturbed_metric_matrix,
    scalar_perturbed_metric_rules, FrwCoordinateChart, FrwMetricAnsatz, ScalarMetricModes,
    SpatialCoordinateNames, TensorMetricModes, VectorMetricModes,
};
pub use multifield::{
    adiabatic_entropy_basis, derive_multifield_curvature_entropy_equations, multifield_mass_data,
    standard_multifield_symbols, AdiabaticEntropyBasis, MultiFieldEquationSet, MultiFieldMassData,
    MultiFieldSymbols,
};
pub use second_order::{
    default_second_order_gauge_generator, default_second_order_scalar_modes,
    derive_second_order_scalar_einstein_system, expand_expr_in_parameter,
    expand_matrix_in_parameter, lie_derivative_covariant_rank2,
    second_order_scalar_gauge_variation, SecondOrderEinsteinEquationSplit,
    SecondOrderEinsteinSystem, SecondOrderGaugeGenerator, SecondOrderScalarGaugeVariation,
    SecondOrderScalarModes,
};
pub use second_order_vector_tensor::{
    default_second_order_tensor_modes, default_second_order_vector_generator,
    default_second_order_vector_modes, derive_second_order_tensor_system,
    derive_second_order_vector_system, project_second_order_tensor_to_harmonics,
    project_second_order_vector_to_harmonics, second_order_tensor_gauge_variation,
    second_order_vector_gauge_variation, tensor_metric_piece_order_one,
    tensor_metric_piece_order_two, vector_metric_piece_order_one, vector_metric_piece_order_two,
    SecondOrderTensorEquationSplit, SecondOrderTensorGaugeVariation, SecondOrderTensorModes,
    SecondOrderTensorSystem, SecondOrderVectorEquationSplit, SecondOrderVectorGaugeGenerator,
    SecondOrderVectorGaugeVariation, SecondOrderVectorModes, SecondOrderVectorSystem,
};
pub use vector_tensor::{
    derive_linear_tensor_einstein_equations, derive_linear_vector_einstein_equations_poisson,
    derive_tensor_mode_equations, gauge_invariant_vector_variables,
    poisson_gauge_vector_metric_matrix, standard_tensor_mode_names, standard_vector_mode_names,
    tensor_metric_matrix, tensor_mode_first_order_system, tensor_quadratic_action,
    vector_gauge_transformations, vector_metric_matrix, GaugeInvariantVectorVariables,
    TensorEinsteinEquationSet, TensorModeDerivation, TensorModeNames, TensorQuadraticAction,
    VectorEinsteinEquationSet, VectorModeNames,
};

use ax_ir::{Condition, Expr, Index, Variance};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use std::collections::BTreeMap;

pub fn perturbation_tensor_rank_from_symmetry(sym: &ax_ir::TensorSymmetry) -> usize {
    sym.tableaux
        .iter()
        .map(|tableau| tableau.slot_map.len())
        .max()
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct PerturbationSetup {
    pub full_field: lasso::Spur,
    pub background: lasso::Spur,
    pub inverse_background: Option<lasso::Spur>,
    pub perturbations: Vec<PerturbationOrder>,
    pub epsilon: lasso::Spur,
    pub max_order: usize,
}

#[derive(Clone, Debug)]
pub struct PerturbationOrder {
    pub order: usize,
    pub field: lasso::Spur,
}

#[derive(Clone, Debug)]
pub struct ExpandedExpression {
    pub orders: Vec<OrderTerm>,
}

#[derive(Clone, Debug)]
pub struct OrderTerm {
    pub order: usize,
    pub expr: Expr,
}

pub fn perturb_expand(
    expr: &Expr,
    setup: &PerturbationSetup,
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let substituted = substitute_field(expr, setup, interner);
    let terms = expand_in_epsilon(&substituted, setup.epsilon, setup.max_order, interner);
    collect_orders(terms, setup.max_order)
}

pub fn perturb_inverse_metric(
    setup: &PerturbationSetup,
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let mut counter = 0usize;
    let orders = (0..=setup.max_order)
        .filter_map(|order| {
            inverse_metric_order(setup, order, a, b, interner, &mut counter)
                .map(|expr| OrderTerm { order, expr })
        })
        .collect();
    ExpandedExpression { orders }
}

pub fn perturb_christoffel(
    setup: &PerturbationSetup,
    coord_syms: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let a = tensor_label("a", 0, coord_syms, interner);
    let b = tensor_label("b", 1, coord_syms, interner);
    let c = tensor_label("c", 2, coord_syms, interner);
    let mut counter = 0usize;
    let orders = (0..=setup.max_order)
        .map(|order| OrderTerm {
            order,
            expr: christoffel_order_expr(setup, order, a, b, c, interner, &mut counter),
        })
        .collect();
    ExpandedExpression { orders }
}

pub fn perturb_riemann(
    setup: &PerturbationSetup,
    coord_syms: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let a = tensor_label("a", 0, coord_syms, interner);
    let b = tensor_label("b", 1, coord_syms, interner);
    let c = tensor_label("c", 2, coord_syms, interner);
    let d = tensor_label("d", 3, coord_syms, interner);
    let mut counter = 0usize;
    let orders = (0..=setup.max_order)
        .map(|order| OrderTerm {
            order,
            expr: riemann_order_expr(setup, order, a, b, c, d, interner, &mut counter),
        })
        .collect();
    ExpandedExpression { orders }
}

pub fn perturb_ricci(
    setup: &PerturbationSetup,
    coord_syms: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let b = tensor_label("b", 1, coord_syms, interner);
    let d = tensor_label("d", 3, coord_syms, interner);
    let mut counter = 0usize;
    let orders = (0..=setup.max_order)
        .map(|order| OrderTerm {
            order,
            expr: ricci_order_expr(setup, order, b, d, interner, &mut counter),
        })
        .collect();
    ExpandedExpression { orders }
}

pub fn perturb_einstein(
    setup: &PerturbationSetup,
    coord_syms: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ExpandedExpression {
    let a = tensor_label("a", 0, coord_syms, interner);
    let b = tensor_label("b", 1, coord_syms, interner);
    let mut counter = 0usize;
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let orders = (0..=setup.max_order)
        .map(|order| {
            let ricci = ricci_order_expr(setup, order, a, b, interner, &mut counter);
            let scalar_terms = (0..=order)
                .filter_map(|k| {
                    let metric = metric_order_cov(setup, k, a, b)?;
                    let scalar = ricci_scalar_order_expr(setup, order - k, interner, &mut counter);
                    Some(Expr::mul(vec![metric, scalar]))
                })
                .collect::<Vec<_>>();
            let scalar_piece = Expr::mul(vec![Expr::neg(half.clone()), Expr::add(scalar_terms)]);
            OrderTerm {
                order,
                expr: Expr::add(vec![ricci, scalar_piece]),
            }
        })
        .collect();
    ExpandedExpression { orders }
}

fn inverse_metric_order(
    setup: &PerturbationSetup,
    order: usize,
    left: lasso::Spur,
    right: lasso::Spur,
    interner: &ax_ir::Interner,
    counter: &mut usize,
) -> Option<Expr> {
    let inverse_background = inverse_background_symbol(setup, interner);
    if order == 0 {
        return Some(indexed2(
            inverse_background,
            left,
            Variance::Up,
            right,
            Variance::Up,
        ));
    }

    let terms = perturbation_sequences(setup, order)
        .into_iter()
        .map(|sequence| {
            let mut factors = Vec::new();
            let mut current_left = left;
            for perturbation in &sequence {
                let c = generate_dummy(counter, interner);
                let d = generate_dummy(counter, interner);
                factors.push(indexed2(
                    inverse_background,
                    current_left,
                    Variance::Up,
                    c,
                    Variance::Up,
                ));
                factors.push(indexed2(
                    perturbation.field,
                    c,
                    Variance::Down,
                    d,
                    Variance::Down,
                ));
                current_left = d;
            }
            factors.push(indexed2(
                inverse_background,
                current_left,
                Variance::Up,
                right,
                Variance::Up,
            ));

            let term = Expr::mul(factors);
            if sequence.len() % 2 == 1 {
                Expr::Neg(Box::new(term))
            } else {
                term
            }
        })
        .collect::<Vec<_>>();

    match terms.len() {
        0 => None,
        1 => terms.into_iter().next(),
        _ => Some(Expr::add(terms)),
    }
}

fn christoffel_order_expr(
    setup: &PerturbationSetup,
    order: usize,
    up: lasso::Spur,
    down1: lasso::Spur,
    down2: lasso::Spur,
    interner: &ax_ir::Interner,
    counter: &mut usize,
) -> Expr {
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let terms = (0..=order)
        .filter_map(|k| {
            let d = generate_dummy(counter, interner);
            let inverse = inverse_metric_order(setup, k, up, d, interner, counter)?;
            let metric_order = order - k;
            let g_dc = metric_order_cov(setup, metric_order, d, down2)?;
            let g_bd = metric_order_cov(setup, metric_order, down1, d)?;
            let g_bc = metric_order_cov(setup, metric_order, down1, down2)?;
            let derivative_sum = Expr::add(vec![
                partial_derivative(g_dc, down1, interner),
                partial_derivative(g_bd, down2, interner),
                Expr::neg(partial_derivative(g_bc, d, interner)),
            ]);
            Some(Expr::mul(vec![half.clone(), inverse, derivative_sum]))
        })
        .collect::<Vec<_>>();
    Expr::add(terms)
}

fn riemann_order_expr(
    setup: &PerturbationSetup,
    order: usize,
    up: lasso::Spur,
    down1: lasso::Spur,
    down2: lasso::Spur,
    down3: lasso::Spur,
    interner: &ax_ir::Interner,
    counter: &mut usize,
) -> Expr {
    let linear = vec![
        partial_derivative(
            christoffel_order_expr(setup, order, up, down1, down3, interner, counter),
            down2,
            interner,
        ),
        Expr::neg(partial_derivative(
            christoffel_order_expr(setup, order, up, down1, down2, interner, counter),
            down3,
            interner,
        )),
    ];

    let mut terms = linear;
    for k in 0..=order {
        let e = generate_dummy(counter, interner);
        terms.push(Expr::mul(vec![
            christoffel_order_expr(setup, k, up, down2, e, interner, counter),
            christoffel_order_expr(setup, order - k, e, down1, down3, interner, counter),
        ]));
        terms.push(Expr::neg(Expr::mul(vec![
            christoffel_order_expr(setup, k, up, down3, e, interner, counter),
            christoffel_order_expr(setup, order - k, e, down1, down2, interner, counter),
        ])));
    }
    Expr::add(terms)
}

fn ricci_order_expr(
    setup: &PerturbationSetup,
    order: usize,
    down1: lasso::Spur,
    down2: lasso::Spur,
    interner: &ax_ir::Interner,
    counter: &mut usize,
) -> Expr {
    let a = generate_dummy(counter, interner);
    riemann_order_expr(setup, order, a, down1, a, down2, interner, counter)
}

fn ricci_scalar_order_expr(
    setup: &PerturbationSetup,
    order: usize,
    interner: &ax_ir::Interner,
    counter: &mut usize,
) -> Expr {
    let terms = (0..=order)
        .filter_map(|k| {
            let a = generate_dummy(counter, interner);
            let b = generate_dummy(counter, interner);
            let inverse = inverse_metric_order(setup, k, a, b, interner, counter)?;
            let ricci = ricci_order_expr(setup, order - k, a, b, interner, counter);
            Some(Expr::mul(vec![inverse, ricci]))
        })
        .collect::<Vec<_>>();
    Expr::add(terms)
}

fn metric_order_cov(
    setup: &PerturbationSetup,
    order: usize,
    left: lasso::Spur,
    right: lasso::Spur,
) -> Option<Expr> {
    if order == 0 {
        return Some(indexed2(
            setup.background,
            left,
            Variance::Down,
            right,
            Variance::Down,
        ));
    }
    setup
        .perturbations
        .iter()
        .find(|perturbation| perturbation.order == order)
        .map(|perturbation| {
            indexed2(
                perturbation.field,
                left,
                Variance::Down,
                right,
                Variance::Down,
            )
        })
}

fn substitute_field(expr: &Expr, setup: &PerturbationSetup, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Sym(s) if *s == setup.full_field => perturbation_series(None, setup),
        Expr::Indexed(base, indices) => match base.as_ref() {
            Expr::Sym(s) if *s == setup.full_field => {
                perturbation_series(Some(indices.as_slice()), setup)
            }
            _ => Expr::Indexed(
                Box::new(substitute_field(base, setup, interner)),
                indices.clone(),
            ),
        },
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_field(re, setup, interner)),
            Box::new(substitute_field(im, setup, interner)),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_field(term, setup, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_field(factor, setup, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_field(base, setup, interner),
            substitute_field(exp, setup, interner),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_field(inner, setup, interner)),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_field(inner, setup, interner)), *rel)
        }
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| substitute_field(arg, setup, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_field(body, setup, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_field(lhs, setup, interner)),
            Box::new(substitute_field(rhs, setup, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_field(value, setup, interner),
                        substitute_condition(condition, setup, interner),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_field(value, setup, interner)),
            Box::new(substitute_field(body, setup, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_field(item, setup, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_field(cell, setup, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => expr.clone(),
    }
}

pub(crate) fn expand_in_epsilon(
    expr: &Expr,
    epsilon: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<(usize, Expr)> {
    match expr {
        Expr::Sym(s) if *s == epsilon => vec![(1, Expr::one())],
        Expr::Pow(base, exp) if matches!(base.as_ref(), Expr::Sym(s) if *s == epsilon) => {
            if let Expr::Int(n) = exp.as_ref() {
                if let Some(order) = n.to_usize() {
                    if order <= max_order {
                        return vec![(order, Expr::one())];
                    }
                    return Vec::new();
                }
            }
            vec![(0, expr.clone())]
        }
        Expr::Add(terms) => {
            let mut out = Vec::new();
            for term in terms {
                out.extend(expand_in_epsilon(term, epsilon, max_order, interner));
            }
            out
        }
        Expr::Mul(factors) => {
            let mut acc = vec![(0usize, Expr::one())];
            for factor in factors {
                let factor_terms = expand_in_epsilon(factor, epsilon, max_order, interner);
                acc = convolve_terms(acc, factor_terms, max_order);
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }
        Expr::Pow(base, exp) if contains_epsilon(base, epsilon) => {
            if let Expr::Int(n) = exp.as_ref() {
                if let Some(power) = n.to_usize() {
                    return multinomial_expand(base, power, epsilon, max_order, interner);
                }
            }
            vec![(0, expr.clone())]
        }
        Expr::Neg(inner) => expand_in_epsilon(inner, epsilon, max_order, interner)
            .into_iter()
            .map(|(order, term)| (order, Expr::neg(term)))
            .collect(),
        Expr::Complex(re, im) => {
            let re_terms = expand_in_epsilon(re, epsilon, max_order, interner);
            let im_terms = expand_in_epsilon(im, epsilon, max_order, interner);
            let mut grouped: BTreeMap<usize, (Option<Expr>, Option<Expr>)> = BTreeMap::new();
            for (order, term) in re_terms {
                grouped.entry(order).or_default().0 = Some(term);
            }
            for (order, term) in im_terms {
                grouped.entry(order).or_default().1 = Some(term);
            }
            grouped
                .into_iter()
                .map(|(order, (re, im))| {
                    (
                        order,
                        Expr::Complex(
                            Box::new(re.unwrap_or_else(Expr::zero)),
                            Box::new(im.unwrap_or_else(Expr::zero)),
                        ),
                    )
                })
                .collect()
        }
        Expr::Call(f, args) if args.iter().any(|arg| contains_epsilon(arg, epsilon)) => {
            expand_structural_children(expr, *f, args, epsilon, max_order, interner)
        }
        Expr::Indexed(base, indices) if contains_epsilon(base, epsilon) => {
            expand_in_epsilon(base, epsilon, max_order, interner)
                .into_iter()
                .map(|(order, base)| (order, Expr::Indexed(Box::new(base), indices.clone())))
                .collect()
        }
        Expr::List(items) if items.iter().any(|item| contains_epsilon(item, epsilon)) => {
            expand_list(items, epsilon, max_order, interner)
        }
        Expr::Matrix(rows)
            if rows
                .iter()
                .any(|row| row.iter().any(|cell| contains_epsilon(cell, epsilon))) =>
        {
            expand_matrix(rows, epsilon, max_order, interner)
        }
        Expr::Let(name, value, body)
            if contains_epsilon(value, epsilon) || contains_epsilon(body, epsilon) =>
        {
            let value_terms = expand_in_epsilon(value, epsilon, max_order, interner);
            let body_terms = expand_in_epsilon(body, epsilon, max_order, interner);
            convolve_terms(value_terms, body_terms, max_order)
                .into_iter()
                .map(|(order, expr)| match expr {
                    Expr::Mul(mut factors) if factors.len() == 2 => {
                        let body = factors.pop().unwrap();
                        let value = factors.pop().unwrap();
                        (order, Expr::Let(*name, Box::new(value), Box::new(body)))
                    }
                    _ => (order, expr),
                })
                .collect()
        }
        Expr::Piecewise(cases)
            if cases.iter().any(|(value, condition)| {
                contains_epsilon(value, epsilon) || condition_contains_epsilon(condition, epsilon)
            }) =>
        {
            vec![(0, expr.clone())]
        }
        _ => vec![(0, expr.clone())],
    }
}

fn multinomial_expand(
    base: &Expr,
    power: usize,
    epsilon: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<(usize, Expr)> {
    if power == 0 {
        return vec![(0, Expr::one())];
    }

    let base_terms = expand_in_epsilon(base, epsilon, max_order, interner);
    let min_order = base_terms
        .iter()
        .map(|(order, _)| *order)
        .min()
        .unwrap_or(0);
    if min_order.saturating_mul(power) > max_order {
        return Vec::new();
    }

    let mut acc = vec![(0usize, Expr::one())];
    for _ in 0..power {
        acc = convolve_terms(acc, base_terms.clone(), max_order);
        if acc.is_empty() {
            break;
        }
    }
    acc
}

pub(crate) fn collect_orders(terms: Vec<(usize, Expr)>, max_order: usize) -> ExpandedExpression {
    let mut grouped: BTreeMap<usize, Vec<Expr>> = BTreeMap::new();
    for (order, expr) in terms {
        if order <= max_order && !is_zero_expr(&expr) {
            grouped.entry(order).or_default().push(expr);
        }
    }

    let orders = (0..=max_order)
        .filter_map(|order| {
            let terms = grouped.remove(&order)?;
            Some(OrderTerm {
                order,
                expr: Expr::add(terms),
            })
        })
        .collect();
    ExpandedExpression { orders }
}

fn perturbation_series(indices: Option<&[ax_ir::Index]>, setup: &PerturbationSetup) -> Expr {
    let mut terms = Vec::new();
    terms.push(apply_indices(Expr::Sym(setup.background), indices));
    for perturbation in &setup.perturbations {
        if perturbation.order > setup.max_order {
            continue;
        }
        let eps = epsilon_power(setup.epsilon, perturbation.order);
        let field = apply_indices(Expr::Sym(perturbation.field), indices);
        terms.push(Expr::mul(vec![eps, field]));
    }
    Expr::add(terms)
}

fn epsilon_power(epsilon: lasso::Spur, order: usize) -> Expr {
    match order {
        0 => Expr::one(),
        1 => Expr::Sym(epsilon),
        n => Expr::pow(Expr::Sym(epsilon), Expr::Int(BigInt::from(n))),
    }
}

fn apply_indices(base: Expr, indices: Option<&[ax_ir::Index]>) -> Expr {
    match indices {
        Some(indices) => Expr::Indexed(Box::new(base), indices.to_vec()),
        None => base,
    }
}

fn indexed2(
    symbol: lasso::Spur,
    first: lasso::Spur,
    first_variance: Variance,
    second: lasso::Spur,
    second_variance: Variance,
) -> Expr {
    Expr::Indexed(
        Box::new(Expr::Sym(symbol)),
        vec![
            Index {
                name: first,
                variance: first_variance,
                index_type: None,
            },
            Index {
                name: second,
                variance: second_variance,
                index_type: None,
            },
        ],
    )
}

fn partial_derivative(expr: Expr, coordinate: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    let diff = interner.get_or_intern("diff");
    Expr::Call(diff, vec![expr, Expr::Sym(coordinate)])
}

fn tensor_label(
    fallback: &str,
    position: usize,
    coord_syms: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    coord_syms
        .get(position)
        .copied()
        .unwrap_or_else(|| interner.get_or_intern(fallback))
}

fn inverse_background_symbol(setup: &PerturbationSetup, interner: &ax_ir::Interner) -> lasso::Spur {
    setup.inverse_background.unwrap_or_else(|| {
        let name = format!("{}inv", interner.resolve(setup.background));
        interner.get_or_intern(&name)
    })
}

fn generate_dummy(counter: &mut usize, interner: &ax_ir::Interner) -> lasso::Spur {
    let name = format!("_p{}", *counter);
    *counter += 1;
    interner.get_or_intern(&name)
}

fn perturbation_sequences(
    setup: &PerturbationSetup,
    total_order: usize,
) -> Vec<Vec<PerturbationOrder>> {
    fn rec(
        perturbations: &[PerturbationOrder],
        remaining: usize,
        current: &mut Vec<PerturbationOrder>,
        out: &mut Vec<Vec<PerturbationOrder>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        for perturbation in perturbations {
            if perturbation.order == 0 || perturbation.order > remaining {
                continue;
            }
            current.push(perturbation.clone());
            rec(perturbations, remaining - perturbation.order, current, out);
            current.pop();
        }
    }

    let mut perturbations = setup.perturbations.clone();
    perturbations.sort_by_key(|perturbation| perturbation.order);
    let mut out = Vec::new();
    rec(&perturbations, total_order, &mut Vec::new(), &mut out);
    out
}

fn convolve_terms(
    lhs: Vec<(usize, Expr)>,
    rhs: Vec<(usize, Expr)>,
    max_order: usize,
) -> Vec<(usize, Expr)> {
    let mut out = Vec::new();
    for (lhs_order, lhs_expr) in &lhs {
        for (rhs_order, rhs_expr) in &rhs {
            let order = lhs_order + rhs_order;
            if order <= max_order {
                out.push((order, Expr::mul(vec![lhs_expr.clone(), rhs_expr.clone()])));
            }
        }
    }
    out
}

fn contains_epsilon(expr: &Expr, epsilon: lasso::Spur) -> bool {
    match expr {
        Expr::Sym(s) => *s == epsilon,
        Expr::Complex(re, im) => contains_epsilon(re, epsilon) || contains_epsilon(im, epsilon),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_epsilon(term, epsilon))
        }
        Expr::Pow(base, exp) => contains_epsilon(base, epsilon) || contains_epsilon(exp, epsilon),
        Expr::Neg(inner) => contains_epsilon(inner, epsilon),
        Expr::Call(_, args) => args.iter().any(|arg| contains_epsilon(arg, epsilon)),
        Expr::FnDef(_, _, body) => contains_epsilon(body, epsilon),
        Expr::Rule(lhs, rhs, _) => contains_epsilon(lhs, epsilon) || contains_epsilon(rhs, epsilon),
        Expr::Piecewise(cases) => cases.iter().any(|(value, condition)| {
            contains_epsilon(value, epsilon) || condition_contains_epsilon(condition, epsilon)
        }),
        Expr::Indexed(base, _) => contains_epsilon(base, epsilon),
        Expr::Let(_, value, body) => {
            contains_epsilon(value, epsilon) || contains_epsilon(body, epsilon)
        }
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_epsilon(cell, epsilon))),
        _ => false,
    }
}

fn condition_contains_epsilon(condition: &Condition, epsilon: lasso::Spur) -> bool {
    match condition {
        Condition::Gt(a, b)
        | Condition::Lt(a, b)
        | Condition::Ge(a, b)
        | Condition::Le(a, b)
        | Condition::Eq(a, b)
        | Condition::Ne(a, b) => contains_epsilon(a, epsilon) || contains_epsilon(b, epsilon),
        Condition::And(a, b) | Condition::Or(a, b) => {
            condition_contains_epsilon(a, epsilon) || condition_contains_epsilon(b, epsilon)
        }
        Condition::Not(inner) => condition_contains_epsilon(inner, epsilon),
        Condition::True | Condition::False => false,
    }
}

fn substitute_condition(
    condition: &Condition,
    setup: &PerturbationSetup,
    interner: &ax_ir::Interner,
) -> Condition {
    match condition {
        Condition::Gt(a, b) => Condition::Gt(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::Lt(a, b) => Condition::Lt(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::Ge(a, b) => Condition::Ge(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::Le(a, b) => Condition::Le(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::Eq(a, b) => Condition::Eq(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::Ne(a, b) => Condition::Ne(
            substitute_field(a, setup, interner),
            substitute_field(b, setup, interner),
        ),
        Condition::And(a, b) => Condition::And(
            Box::new(substitute_condition(a, setup, interner)),
            Box::new(substitute_condition(b, setup, interner)),
        ),
        Condition::Or(a, b) => Condition::Or(
            Box::new(substitute_condition(a, setup, interner)),
            Box::new(substitute_condition(b, setup, interner)),
        ),
        Condition::Not(inner) => {
            Condition::Not(Box::new(substitute_condition(inner, setup, interner)))
        }
        Condition::True => Condition::True,
        Condition::False => Condition::False,
    }
}

fn expand_structural_children(
    expr: &Expr,
    f: lasso::Spur,
    args: &[Expr],
    epsilon: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<(usize, Expr)> {
    let mut acc = vec![(0usize, Vec::<Expr>::new())];
    for arg in args {
        let arg_terms = expand_in_epsilon(arg, epsilon, max_order, interner);
        let mut next = Vec::new();
        for (acc_order, acc_args) in &acc {
            for (arg_order, arg_expr) in &arg_terms {
                let order = acc_order + arg_order;
                if order <= max_order {
                    let mut args = acc_args.clone();
                    args.push(arg_expr.clone());
                    next.push((order, args));
                }
            }
        }
        acc = next;
    }
    if acc.is_empty() {
        vec![(0, expr.clone())]
    } else {
        acc.into_iter()
            .map(|(order, args)| (order, Expr::Call(f, args)))
            .collect()
    }
}

fn expand_list(
    items: &[Expr],
    epsilon: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<(usize, Expr)> {
    let mut acc = vec![(0usize, Vec::<Expr>::new())];
    for item in items {
        let item_terms = expand_in_epsilon(item, epsilon, max_order, interner);
        let mut next = Vec::new();
        for (acc_order, acc_items) in &acc {
            for (item_order, item_expr) in &item_terms {
                let order = acc_order + item_order;
                if order <= max_order {
                    let mut items = acc_items.clone();
                    items.push(item_expr.clone());
                    next.push((order, items));
                }
            }
        }
        acc = next;
    }
    acc.into_iter()
        .map(|(order, items)| (order, Expr::List(items)))
        .collect()
}

fn expand_matrix(
    rows: &[Vec<Expr>],
    epsilon: lasso::Spur,
    max_order: usize,
    interner: &ax_ir::Interner,
) -> Vec<(usize, Expr)> {
    let flat = rows.iter().flatten().cloned().collect::<Vec<_>>();
    let row_lengths = rows.iter().map(Vec::len).collect::<Vec<_>>();
    expand_list(&flat, epsilon, max_order, interner)
        .into_iter()
        .map(|(order, expr)| {
            let Expr::List(items) = expr else {
                return (order, Expr::Matrix(rows.to_vec()));
            };
            let mut iter = items.into_iter();
            let rebuilt = row_lengths
                .iter()
                .map(|len| iter.by_ref().take(*len).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            (order, Expr::Matrix(rebuilt))
        })
        .collect()
}

fn is_zero_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Int(n) => n.is_zero(),
        Expr::Rational(r) => r.is_zero(),
        Expr::Float(f) => *f == 0.0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_indexed_metric_to_orders() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let g0 = interner.get_or_intern("g0");
        let h = interner.get_or_intern("h");
        let k = interner.get_or_intern("k");
        let eps = interner.get_or_intern("eps");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let indices = vec![
            ax_ir::Index {
                name: a,
                variance: ax_ir::Variance::Down,
                index_type: None,
            },
            ax_ir::Index {
                name: b,
                variance: ax_ir::Variance::Down,
                index_type: None,
            },
        ];
        let expr = Expr::Indexed(Box::new(Expr::Sym(g)), indices.clone());
        let setup = PerturbationSetup {
            full_field: g,
            background: g0,
            inverse_background: None,
            perturbations: vec![
                PerturbationOrder { order: 1, field: h },
                PerturbationOrder { order: 2, field: k },
            ],
            epsilon: eps,
            max_order: 2,
        };

        let expanded = perturb_expand(&expr, &setup, &interner);
        assert_eq!(expanded.orders.len(), 3);
        assert_eq!(expanded.orders[0].order, 0);
        assert_eq!(expanded.orders[1].order, 1);
        assert_eq!(expanded.orders[2].order, 2);
        assert_eq!(
            expanded.orders[0].expr,
            Expr::Indexed(Box::new(Expr::Sym(g0)), indices)
        );
    }

    #[test]
    fn expands_square_to_second_order() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let g0 = interner.get_or_intern("g0");
        let h = interner.get_or_intern("h");
        let eps = interner.get_or_intern("eps");
        let expr = Expr::pow(Expr::Sym(g), Expr::Int(2.into()));
        let setup = PerturbationSetup {
            full_field: g,
            background: g0,
            inverse_background: None,
            perturbations: vec![PerturbationOrder { order: 1, field: h }],
            epsilon: eps,
            max_order: 2,
        };

        let expanded = perturb_expand(&expr, &setup, &interner);
        assert_eq!(expanded.orders.len(), 3);
        assert_eq!(expanded.orders[0].order, 0);
        assert_eq!(expanded.orders[1].order, 1);
        assert_eq!(expanded.orders[2].order, 2);
    }

    #[test]
    fn perturbation_rank_is_inferred_from_longest_slot_map() {
        let symmetry = ax_ir::TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape: vec![2, 1],
                slot_map: vec![0, 1, 2],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };
        assert_eq!(perturbation_tensor_rank_from_symmetry(&symmetry), 3);
    }
}
