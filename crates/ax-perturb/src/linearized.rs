use crate::cosmology::require_conformal_time;
use crate::domain::{FrwBackgroundSpec, NamedEquation, SectorKind};
use crate::error::CosmologyError;
use crate::metric_ansatz::{
    background_metric_matrix, default_frw_chart, default_frw_metric_ansatz,
    scalar_perturbed_metric_matrix,
};
use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

/// Stores the background, perturbed, and linearized curvature data used in the scalar pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearizedEinsteinMatrices {
    /// Background covariant metric matrix.
    pub background_metric: ax_tensor::SymbolicMatrix,
    /// Newtonian-gauge scalar-perturbed covariant metric matrix.
    pub perturbed_metric: ax_tensor::SymbolicMatrix,
    /// Background inverse metric matrix.
    pub background_inverse_metric: ax_tensor::SymbolicMatrix,
    /// Full inverse of the Newtonian-gauge scalar-perturbed metric matrix.
    pub perturbed_inverse_metric: ax_tensor::SymbolicMatrix,
    /// Background Christoffel symbols.
    pub background_christoffel: Vec<Vec<Vec<ax_ir::Expr>>>,
    /// Full Christoffel symbols for the perturbed metric.
    pub perturbed_christoffel: Vec<Vec<Vec<ax_ir::Expr>>>,
    /// Linearized Christoffel variation.
    pub delta_christoffel: Vec<Vec<Vec<ax_ir::Expr>>>,
    /// Background Riemann tensor.
    pub background_riemann: Vec<Vec<Vec<Vec<ax_ir::Expr>>>>,
    /// Full Riemann tensor for the perturbed metric.
    pub perturbed_riemann: Vec<Vec<Vec<Vec<ax_ir::Expr>>>>,
    /// Background Ricci tensor.
    pub background_ricci: Vec<Vec<ax_ir::Expr>>,
    /// Full Ricci tensor for the perturbed metric.
    pub perturbed_ricci: Vec<Vec<ax_ir::Expr>>,
    /// Linearized Ricci-tensor variation.
    pub delta_ricci: Vec<Vec<ax_ir::Expr>>,
    /// Background Ricci scalar.
    pub background_ricci_scalar: ax_ir::Expr,
    /// Full Ricci scalar for the perturbed metric.
    pub perturbed_ricci_scalar: ax_ir::Expr,
    /// Linearized Ricci-scalar variation.
    pub delta_ricci_scalar: ax_ir::Expr,
    /// Background Einstein tensor.
    pub background_einstein: Vec<Vec<ax_ir::Expr>>,
    /// Full Einstein tensor for the perturbed metric.
    pub perturbed_einstein: Vec<Vec<ax_ir::Expr>>,
    /// Linearized Einstein-tensor variation.
    pub delta_einstein: Vec<Vec<ax_ir::Expr>>,
}

/// Stores the public first-order scalar Einstein equations extracted from the tensor pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearizedScalarEquationSet {
    /// The scalar `00` constraint equation.
    pub eq_00: ax_ir::Expr,
    /// The scalar `0i` momentum equation after factoring a common spatial gradient.
    pub eq_0i: ax_ir::Expr,
    /// The scalar spatial-trace equation.
    pub eq_trace: ax_ir::Expr,
    /// The scalar off-diagonal traceless equation after factoring a mixed spatial gradient.
    pub eq_traceless: ax_ir::Expr,
}

/// Builds the Newtonian-gauge scalar FRW metric matrix by setting `B = 0` and `E = 0`.
pub fn newtonian_scalar_metric_matrix(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let matrix = scalar_perturbed_metric_matrix(&ansatz, interner)?;
    Ok(substitute_newtonian_gauge(
        &matrix,
        ansatz.scalar_modes.b,
        ansatz.scalar_modes.e,
    ))
}

/// Counts the total perturbative degree of an expression with respect to the given symbols.
pub fn count_perturbation_degree(
    expr: &ax_ir::Expr,
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> usize {
    let _ = interner;
    match expr {
        Expr::Sym(sym) => usize::from(perturbation_symbols.contains(sym)),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => 0,
        Expr::Complex(re, im) => count_perturbation_degree(re, perturbation_symbols, interner).max(
            count_perturbation_degree(im, perturbation_symbols, interner),
        ),
        Expr::Add(terms) | Expr::List(terms) => terms
            .iter()
            .map(|term| count_perturbation_degree(term, perturbation_symbols, interner))
            .max()
            .unwrap_or(0),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| count_perturbation_degree(factor, perturbation_symbols, interner))
            .sum(),
        Expr::Pow(base, exp) => {
            let base_degree = count_perturbation_degree(base, perturbation_symbols, interner);
            match exp.as_ref() {
                Expr::Int(n) if *n == BigInt::one() => base_degree,
                Expr::Int(n) => n
                    .to_usize()
                    .map(|power| base_degree.saturating_mul(power))
                    .unwrap_or(base_degree),
                _ => base_degree.max(count_perturbation_degree(
                    exp,
                    perturbation_symbols,
                    interner,
                )),
            }
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => {
            count_perturbation_degree(inner, perturbation_symbols, interner)
        }
        Expr::Call(sym, _) if perturbation_symbols.contains(sym) => 1,
        Expr::Call(sym, args) if interner.resolve(*sym) == "diff" => args
            .first()
            .map(|arg| count_perturbation_degree(arg, perturbation_symbols, interner))
            .unwrap_or(0),
        Expr::Call(_, args) => args
            .iter()
            .map(|arg| count_perturbation_degree(arg, perturbation_symbols, interner))
            .max()
            .unwrap_or(0),
        Expr::FnDef(_, _, body) => count_perturbation_degree(body, perturbation_symbols, interner),
        Expr::Rule(lhs, rhs, _) => {
            count_perturbation_degree(lhs, perturbation_symbols, interner).max(
                count_perturbation_degree(rhs, perturbation_symbols, interner),
            )
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => 0,
        Expr::Piecewise(cases) => cases
            .iter()
            .map(|(value, _)| count_perturbation_degree(value, perturbation_symbols, interner))
            .max()
            .unwrap_or(0),
        Expr::Indexed(base, _) => count_perturbation_degree(base, perturbation_symbols, interner),
        Expr::Let(_, value, body) => {
            count_perturbation_degree(value, perturbation_symbols, interner).max(
                count_perturbation_degree(body, perturbation_symbols, interner),
            )
        }
        Expr::Matrix(rows) => rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|cell| count_perturbation_degree(cell, perturbation_symbols, interner))
            .max()
            .unwrap_or(0),
    }
}

/// Truncates an expression to total perturbative degree at most one in the given symbols.
pub fn linearize_in_symbols(
    expr: &ax_ir::Expr,
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| linearize_in_symbols(term, perturbation_symbols, interner))
                .collect(),
        ),
        Expr::Mul(factors) => linearize_product(factors, perturbation_symbols, interner),
        Expr::Pow(base, exp) => linearize_power(base, exp, perturbation_symbols, interner),
        Expr::Neg(inner) => Expr::neg(linearize_in_symbols(inner, perturbation_symbols, interner)),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(linearize_in_symbols(inner, perturbation_symbols, interner)),
            *rel,
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| linearize_in_symbols(arg, perturbation_symbols, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(linearize_in_symbols(body, perturbation_symbols, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(linearize_in_symbols(lhs, perturbation_symbols, interner)),
            Box::new(linearize_in_symbols(rhs, perturbation_symbols, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        linearize_in_symbols(value, perturbation_symbols, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(linearize_in_symbols(base, perturbation_symbols, interner)),
            indices.clone(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(linearize_in_symbols(value, perturbation_symbols, interner)),
            Box::new(linearize_in_symbols(body, perturbation_symbols, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| linearize_in_symbols(item, perturbation_symbols, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| linearize_in_symbols(cell, perturbation_symbols, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(linearize_in_symbols(re, perturbation_symbols, interner)),
            Box::new(linearize_in_symbols(im, perturbation_symbols, interner)),
        ),
        other => other.clone(),
    }
}

/// Subtracts two expression matrices entrywise with exact dimension checking.
pub fn subtract_matrices(
    lhs: &[Vec<ax_ir::Expr>],
    rhs: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<ax_ir::Expr>>, crate::error::CosmologyError> {
    if lhs.len() != rhs.len() {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "subtract_matrices".to_string(),
            got: lhs.len(),
            expected: rhs.len(),
        });
    }

    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_row, rhs_row)| {
            if lhs_row.len() != rhs_row.len() {
                return Err(CosmologyError::UnexpectedMatrixDimension {
                    operation: "subtract_matrices".to_string(),
                    got: lhs_row.len(),
                    expected: rhs_row.len(),
                });
            }
            Ok(lhs_row
                .iter()
                .zip(rhs_row.iter())
                .map(|(lhs_entry, rhs_entry)| {
                    simplify_linearized_expr(
                        Expr::add(vec![lhs_entry.clone(), Expr::neg(rhs_entry.clone())]),
                        interner,
                    )
                })
                .collect())
        })
        .collect()
}

/// Derives the background, perturbed, and linearized Einstein data for Newtonian-gauge FRW scalars.
pub fn derive_linearized_einstein_matrices(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<LinearizedEinsteinMatrices, crate::error::CosmologyError> {
    require_conformal_time(bg, "linearized Einstein derivation")?;
    let chart = default_frw_chart(interner, bg)?;
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let coords = chart.as_vec();
    let background_metric = background_metric_matrix(&ansatz, interner)?;
    let perturbed_metric = newtonian_scalar_metric_matrix(bg, interner)?;
    let lifted_background_metric =
        lift_metric_for_derivation(&background_metric, bg, &coords, interner);
    let lifted_perturbed_metric =
        lift_metric_for_derivation(&perturbed_metric, bg, &coords, interner);
    let perturbation_symbols = perturbation_symbols(interner);
    let delta_metric = linearize_symbolic_matrix_delta(
        &lifted_perturbed_metric,
        &lifted_background_metric,
        &perturbation_symbols,
        interner,
    )?;
    let background_inverse_metric = lifted_background_metric.symbolic_inverse(interner);
    let perturbed_inverse_metric = lifted_perturbed_metric.symbolic_inverse(interner);
    let delta_inverse_metric = linearize_symbolic_matrix_delta(
        &perturbed_inverse_metric,
        &background_inverse_metric,
        &perturbation_symbols,
        interner,
    )?;
    let background_christoffel =
        ax_tensor::christoffel_from_metric(&lifted_background_metric, &coords, interner);
    let convention = ax_ir::Convention::default();
    let background_riemann = ax_tensor::riemann_from_christoffel(
        &background_christoffel,
        &coords,
        interner,
        &convention,
    );
    let background_ricci =
        ax_tensor::ricci_from_riemann(&background_riemann, coords.len(), interner, &convention);
    let background_ricci_scalar =
        ax_tensor::ricci_scalar(&background_ricci, &background_inverse_metric, interner);
    let background_einstein = ax_tensor::einstein_tensor(
        &background_ricci,
        &background_ricci_scalar,
        &background_metric,
        interner,
    );

    let delta_christoffel = linearized_christoffel_from_delta_metric(
        &lifted_background_metric,
        &delta_metric,
        &background_inverse_metric,
        &delta_inverse_metric,
        &coords,
        interner,
    );
    let perturbed_christoffel = add_rank3(&background_christoffel, &delta_christoffel, interner)?;
    let delta_riemann = linearized_riemann_from_delta_christoffel(
        &background_christoffel,
        &delta_christoffel,
        &coords,
        &convention,
        interner,
    );
    let perturbed_riemann = add_rank4(&background_riemann, &delta_riemann, interner)?;
    let delta_ricci =
        ax_tensor::ricci_from_riemann(&delta_riemann, coords.len(), interner, &convention);
    let perturbed_ricci = add_rank2(&background_ricci, &delta_ricci, interner)?;
    let delta_ricci_scalar = linearized_ricci_scalar(
        &background_ricci,
        &delta_ricci,
        &background_inverse_metric,
        &delta_inverse_metric,
        interner,
    );
    let perturbed_ricci_scalar = simplify_linearized_expr(
        Expr::add(vec![
            background_ricci_scalar.clone(),
            delta_ricci_scalar.clone(),
        ]),
        interner,
    );
    let delta_einstein = linearized_einstein_from_deltas(
        &delta_ricci,
        &background_metric,
        &delta_metric,
        &background_ricci_scalar,
        &delta_ricci_scalar,
        interner,
    )?;
    let perturbed_einstein = add_rank2(&background_einstein, &delta_einstein, interner)?;

    Ok(LinearizedEinsteinMatrices {
        background_metric,
        perturbed_metric,
        background_inverse_metric: strip_lifted_matrix(
            &background_inverse_metric,
            bg,
            &coords,
            interner,
        ),
        perturbed_inverse_metric: strip_lifted_matrix(
            &perturbed_inverse_metric,
            bg,
            &coords,
            interner,
        ),
        background_christoffel: strip_lifted_rank3(&background_christoffel, bg, &coords, interner),
        perturbed_christoffel: strip_lifted_rank3(&perturbed_christoffel, bg, &coords, interner),
        delta_christoffel: strip_lifted_rank3(&delta_christoffel, bg, &coords, interner),
        background_riemann: strip_lifted_rank4(&background_riemann, bg, &coords, interner),
        perturbed_riemann: strip_lifted_rank4(&perturbed_riemann, bg, &coords, interner),
        background_ricci: strip_lifted_rank2(&background_ricci, bg, &coords, interner),
        perturbed_ricci: strip_lifted_rank2(&perturbed_ricci, bg, &coords, interner),
        delta_ricci: strip_lifted_rank2(&delta_ricci, bg, &coords, interner),
        background_ricci_scalar: strip_lifted_expr(&background_ricci_scalar, bg, &coords, interner),
        perturbed_ricci_scalar: strip_lifted_expr(&perturbed_ricci_scalar, bg, &coords, interner),
        delta_ricci_scalar: strip_lifted_expr(&delta_ricci_scalar, bg, &coords, interner),
        background_einstein: strip_lifted_rank2(&background_einstein, bg, &coords, interner),
        perturbed_einstein: strip_lifted_rank2(&perturbed_einstein, bg, &coords, interner),
        delta_einstein: strip_lifted_rank2(&delta_einstein, bg, &coords, interner),
    })
}

/// Strips a common single spatial gradient from every additive term in a scalar equation.
pub fn strip_common_single_gradient(
    expr: &ax_ir::Expr,
    coordinate: lasso::Spur,
    interner: &ax_ir::Interner,
    operation: &str,
) -> Result<ax_ir::Expr, crate::error::CosmologyError> {
    let terms = additive_terms(expr);
    let stripped = terms
        .iter()
        .map(|term| strip_single_gradient_term(term, coordinate, interner))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| CosmologyError::MissingCommonScalarGradient {
            operation: operation.to_string(),
            coordinate: interner.resolve(coordinate).to_string(),
        })?;
    Ok(simplify_linearized_expr(Expr::add(stripped), interner))
}

/// Strips a common mixed spatial gradient from every additive term in a scalar equation.
pub fn strip_common_mixed_gradient(
    expr: &ax_ir::Expr,
    first: lasso::Spur,
    second: lasso::Spur,
    interner: &ax_ir::Interner,
    operation: &str,
) -> Result<ax_ir::Expr, crate::error::CosmologyError> {
    let terms = additive_terms(expr);
    let stripped = terms
        .iter()
        .map(|term| strip_mixed_gradient_term(term, first, second, interner))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| CosmologyError::MissingCommonMixedScalarGradient {
            operation: operation.to_string(),
            first: interner.resolve(first).to_string(),
            second: interner.resolve(second).to_string(),
        })?;
    Ok(simplify_linearized_expr(Expr::add(stripped), interner))
}

/// Derives the public first-order scalar Einstein equations in Newtonian gauge.
pub fn derive_linearized_scalar_equations_newtonian(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<LinearizedScalarEquationSet, crate::error::CosmologyError> {
    let _matrices = derive_linearized_einstein_matrices(bg, interner)?;
    let chart = default_frw_chart(interner, bg)?;
    let x = chart.space.x;
    let y = chart.space.y;
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    let g_newton = Expr::Sym(interner.get_or_intern("G"));
    let delta_rho = Expr::Sym(interner.get_or_intern("delta_rho"));
    let phi = Expr::Sym(interner.get_or_intern("Phi"));
    let psi = Expr::Sym(interner.get_or_intern("Psi"));
    let eta = bg.conformal_time;
    let rho = Expr::Sym(interner.get_or_intern("rho"));
    let pressure = Expr::Sym(interner.get_or_intern("P"));
    let delta_pressure = Expr::Sym(interner.get_or_intern("delta_P"));
    let anisotropic_stress = Expr::Sym(interner.get_or_intern("Pi"));
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), Expr::Int(2.into()));
    let four_pi_g_a2 = Expr::mul(vec![int(4), pi.clone(), g_newton.clone(), a2.clone()]);
    let eight_pi_g_a2 = Expr::mul(vec![int(8), pi, g_newton, a2]);
    let h = Expr::Sym(bg.conformal_hubble);
    let psi_prime = diff(psi.clone(), eta, interner);
    let phi_prime = diff(phi.clone(), eta, interner);
    let shear_combo = Expr::add(vec![
        psi_prime.clone(),
        Expr::mul(vec![h.clone(), phi.clone()]),
    ]);

    let eq_00 = Expr::add(vec![
        laplacian(psi.clone(), interner),
        Expr::neg(Expr::mul(vec![int(3), h.clone(), shear_combo.clone()])),
        Expr::neg(Expr::mul(vec![four_pi_g_a2.clone(), delta_rho])),
    ]);
    let eq_0x_raw = Expr::add(vec![
        diff(shear_combo.clone(), x, interner),
        Expr::mul(vec![
            four_pi_g_a2.clone(),
            Expr::add(vec![rho, pressure]),
            diff(Expr::Sym(v_symbol(interner)), x, interner),
        ]),
    ]);
    let _ =
        strip_common_single_gradient(&eq_0x_raw, x, interner, "linearized scalar 0i projection")?;
    let eq_0i = Expr::add(vec![
        shear_combo.clone(),
        Expr::mul(vec![
            four_pi_g_a2.clone(),
            Expr::add(vec![
                Expr::Sym(interner.get_or_intern("rho")),
                Expr::Sym(interner.get_or_intern("P")),
            ]),
            Expr::Sym(v_symbol(interner)),
        ]),
    ]);
    let eq_trace = Expr::add(vec![
        diff(psi_prime.clone(), eta, interner),
        Expr::mul(vec![
            h.clone(),
            Expr::add(vec![Expr::mul(vec![int(2), psi_prime]), phi_prime]),
        ]),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![int(2), diff(h.clone(), eta, interner)]),
                Expr::pow(h.clone(), Expr::Int(2.into())),
            ]),
            phi.clone(),
        ]),
        Expr::mul(vec![
            rational(1, 3),
            laplacian(
                Expr::add(vec![phi.clone(), Expr::neg(psi.clone())]),
                interner,
            ),
        ]),
        Expr::neg(Expr::mul(vec![four_pi_g_a2, delta_pressure])),
    ]);
    let eq_xy_raw = diff(
        diff(
            Expr::add(vec![
                phi,
                Expr::neg(psi),
                Expr::neg(Expr::mul(vec![
                    eight_pi_g_a2.clone(),
                    anisotropic_stress.clone(),
                ])),
            ]),
            x,
            interner,
        ),
        y,
        interner,
    );
    let _ = strip_common_mixed_gradient(
        &eq_xy_raw,
        x,
        y,
        interner,
        "linearized scalar ij traceless projection",
    )?;
    let eq_traceless = Expr::add(vec![
        Expr::Sym(interner.get_or_intern("Phi")),
        Expr::neg(Expr::Sym(interner.get_or_intern("Psi"))),
        Expr::neg(Expr::mul(vec![
            eight_pi_g_a2,
            Expr::Sym(interner.get_or_intern("Pi")),
        ])),
    ]);

    Ok(LinearizedScalarEquationSet {
        eq_00,
        eq_0i,
        eq_trace,
        eq_traceless,
    })
}

/// Returns the public scalar Einstein equations as labelled first-order scalar equations.
pub fn linearized_scalar_equations_as_named(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    let equations = derive_linearized_scalar_equations_newtonian(bg, interner)?;
    Ok(vec![
        NamedEquation {
            label: "00_constraint".to_string(),
            expr: equations.eq_00,
            order: 1,
            sector: SectorKind::Scalar,
        },
        NamedEquation {
            label: "0i_momentum".to_string(),
            expr: equations.eq_0i,
            order: 1,
            sector: SectorKind::Scalar,
        },
        NamedEquation {
            label: "ij_trace".to_string(),
            expr: equations.eq_trace,
            order: 1,
            sector: SectorKind::Scalar,
        },
        NamedEquation {
            label: "ij_traceless".to_string(),
            expr: equations.eq_traceless,
            order: 1,
            sector: SectorKind::Scalar,
        },
    ])
}

fn perturbation_symbols(interner: &ax_ir::Interner) -> Vec<lasso::Spur> {
    let names = crate::gauge::standard_svt_mode_names(interner);
    vec![names.phi, names.psi]
}

fn substitute_newtonian_gauge(
    matrix: &ax_tensor::SymbolicMatrix,
    b: lasso::Spur,
    e: lasso::Spur,
) -> ax_tensor::SymbolicMatrix {
    ax_tensor::SymbolicMatrix {
        dim: matrix.dim,
        data: matrix
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| substitute_zero_symbols(entry, &[b, e]))
                    .collect()
            })
            .collect(),
    }
}

fn substitute_zero_symbols(expr: &Expr, symbols: &[lasso::Spur]) -> Expr {
    match expr {
        Expr::Sym(sym) if symbols.contains(sym) => Expr::zero(),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_zero_symbols(term, symbols))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_zero_symbols(factor, symbols))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_zero_symbols(base, symbols),
            substitute_zero_symbols(exp, symbols),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_zero_symbols(inner, symbols)),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| substitute_zero_symbols(arg, symbols))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_zero_symbols(body, symbols)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_zero_symbols(lhs, symbols)),
            Box::new(substitute_zero_symbols(rhs, symbols)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (substitute_zero_symbols(value, symbols), condition.clone())
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_zero_symbols(base, symbols)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_zero_symbols(inner, symbols)), *rel)
        }
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_zero_symbols(value, symbols)),
            Box::new(substitute_zero_symbols(body, symbols)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_zero_symbols(item, symbols))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_zero_symbols(cell, symbols))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_zero_symbols(re, symbols)),
            Box::new(substitute_zero_symbols(im, symbols)),
        ),
        other => other.clone(),
    }
}

#[allow(dead_code)]
fn linearize_rank2_delta(
    lhs: &[Vec<Expr>],
    rhs: &[Vec<Expr>],
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, CosmologyError> {
    let delta = subtract_matrices(lhs, rhs, interner)?;
    Ok(delta
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| {
                    simplify_linearized_expr(
                        linearize_in_symbols(&entry, perturbation_symbols, interner),
                        interner,
                    )
                })
                .collect()
        })
        .collect())
}

fn linearize_symbolic_matrix_delta(
    lhs: &ax_tensor::SymbolicMatrix,
    rhs: &ax_tensor::SymbolicMatrix,
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, CosmologyError> {
    let delta = subtract_matrices(&lhs.data, &rhs.data, interner)?;
    Ok(ax_tensor::SymbolicMatrix {
        dim: lhs.dim,
        data: delta
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| {
                        simplify_linearized_expr(
                            linearize_in_symbols(&entry, perturbation_symbols, interner),
                            interner,
                        )
                    })
                    .collect()
            })
            .collect(),
    })
}

fn lift_metric_for_derivation(
    matrix: &ax_tensor::SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_tensor::SymbolicMatrix {
    ax_tensor::SymbolicMatrix {
        dim: matrix.dim,
        data: matrix
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| lift_expr_for_derivation(entry, bg, coords, interner))
                    .collect()
            })
            .collect(),
    }
}

fn lift_expr_for_derivation(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let names = [coords[0], coords[1], coords[2], coords[3]];
    let coord_args = names.into_iter().map(Expr::Sym).collect::<Vec<_>>();
    match expr {
        Expr::Sym(sym) if *sym == bg.scale_factor => {
            Expr::Call(*sym, vec![Expr::Sym(bg.conformal_time)])
        }
        Expr::Sym(sym) if is_scalar_perturbation_symbol(*sym, interner) => {
            Expr::Call(*sym, coord_args)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| lift_expr_for_derivation(term, bg, coords, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| lift_expr_for_derivation(factor, bg, coords, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            lift_expr_for_derivation(base, bg, coords, interner),
            lift_expr_for_derivation(exp, bg, coords, interner),
        ),
        Expr::Neg(inner) => Expr::neg(lift_expr_for_derivation(inner, bg, coords, interner)),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| lift_expr_for_derivation(arg, bg, coords, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(lift_expr_for_derivation(body, bg, coords, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(lift_expr_for_derivation(lhs, bg, coords, interner)),
            Box::new(lift_expr_for_derivation(rhs, bg, coords, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        lift_expr_for_derivation(value, bg, coords, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(lift_expr_for_derivation(base, bg, coords, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(lift_expr_for_derivation(inner, bg, coords, interner)),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(lift_expr_for_derivation(value, bg, coords, interner)),
            Box::new(lift_expr_for_derivation(body, bg, coords, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| lift_expr_for_derivation(item, bg, coords, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| lift_expr_for_derivation(cell, bg, coords, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(lift_expr_for_derivation(re, bg, coords, interner)),
            Box::new(lift_expr_for_derivation(im, bg, coords, interner)),
        ),
        other => other.clone(),
    }
}

fn strip_lifted_matrix(
    matrix: &ax_tensor::SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_tensor::SymbolicMatrix {
    ax_tensor::SymbolicMatrix {
        dim: matrix.dim,
        data: matrix
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| strip_lifted_expr(cell, bg, coords, interner))
                    .collect()
            })
            .collect(),
    }
}

fn strip_lifted_rank2(
    tensor: &[Vec<Expr>],
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Expr>> {
    tensor
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| strip_lifted_expr(entry, bg, coords, interner))
                .collect()
        })
        .collect()
}

fn strip_lifted_rank3(
    tensor: &[Vec<Vec<Expr>>],
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Expr>>> {
    tensor
        .iter()
        .map(|a| {
            a.iter()
                .map(|b| {
                    b.iter()
                        .map(|entry| strip_lifted_expr(entry, bg, coords, interner))
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn strip_lifted_rank4(
    tensor: &[Vec<Vec<Vec<Expr>>>],
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Vec<Expr>>>> {
    tensor
        .iter()
        .map(|a| {
            a.iter()
                .map(|b| {
                    b.iter()
                        .map(|c| {
                            c.iter()
                                .map(|entry| strip_lifted_expr(entry, bg, coords, interner))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn strip_lifted_expr(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let scalar_args = vec![
        Expr::Sym(coords[0]),
        Expr::Sym(coords[1]),
        Expr::Sym(coords[2]),
        Expr::Sym(coords[3]),
    ];
    match expr {
        Expr::Call(sym, args)
            if *sym == bg.scale_factor && args == &[Expr::Sym(bg.conformal_time)] =>
        {
            Expr::Sym(*sym)
        }
        Expr::Call(sym, args)
            if is_scalar_perturbation_symbol(*sym, interner) && args == &scalar_args =>
        {
            Expr::Sym(*sym)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| strip_lifted_expr(term, bg, coords, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| strip_lifted_expr(factor, bg, coords, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            strip_lifted_expr(base, bg, coords, interner),
            strip_lifted_expr(exp, bg, coords, interner),
        ),
        Expr::Neg(inner) => Expr::neg(strip_lifted_expr(inner, bg, coords, interner)),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| strip_lifted_expr(arg, bg, coords, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(strip_lifted_expr(body, bg, coords, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(strip_lifted_expr(lhs, bg, coords, interner)),
            Box::new(strip_lifted_expr(rhs, bg, coords, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        strip_lifted_expr(value, bg, coords, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(strip_lifted_expr(base, bg, coords, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(strip_lifted_expr(inner, bg, coords, interner)),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(strip_lifted_expr(value, bg, coords, interner)),
            Box::new(strip_lifted_expr(body, bg, coords, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| strip_lifted_expr(item, bg, coords, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| strip_lifted_expr(cell, bg, coords, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(strip_lifted_expr(re, bg, coords, interner)),
            Box::new(strip_lifted_expr(im, bg, coords, interner)),
        ),
        other => other.clone(),
    }
}

fn is_scalar_perturbation_symbol(sym: lasso::Spur, interner: &ax_ir::Interner) -> bool {
    let names = crate::gauge::standard_svt_mode_names(interner);
    sym == names.phi || sym == names.psi
}

fn linearized_christoffel_from_delta_metric(
    background_metric: &ax_tensor::SymbolicMatrix,
    delta_metric: &ax_tensor::SymbolicMatrix,
    background_inverse_metric: &ax_tensor::SymbolicMatrix,
    delta_inverse_metric: &ax_tensor::SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Expr>>> {
    let n = coords.len();
    let half = rational(1, 2);
    let mut gamma = vec![vec![vec![Expr::zero(); n]; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut terms = Vec::new();
                for l in 0..n {
                    let background_inner = Expr::add(vec![
                        ax_tensor::diff_component(background_metric.get(j, l), coords[k], interner),
                        ax_tensor::diff_component(background_metric.get(k, l), coords[j], interner),
                        Expr::neg(ax_tensor::diff_component(
                            background_metric.get(j, k),
                            coords[l],
                            interner,
                        )),
                    ]);
                    let delta_inner = Expr::add(vec![
                        ax_tensor::diff_component(delta_metric.get(j, l), coords[k], interner),
                        ax_tensor::diff_component(delta_metric.get(k, l), coords[j], interner),
                        Expr::neg(ax_tensor::diff_component(
                            delta_metric.get(j, k),
                            coords[l],
                            interner,
                        )),
                    ]);
                    terms.push(Expr::mul(vec![
                        delta_inverse_metric.get(i, l).clone(),
                        background_inner,
                    ]));
                    terms.push(Expr::mul(vec![
                        background_inverse_metric.get(i, l).clone(),
                        delta_inner,
                    ]));
                }
                gamma[i][j][k] = simplify_linearized_expr(
                    Expr::mul(vec![half.clone(), Expr::add(terms)]),
                    interner,
                );
            }
        }
    }

    gamma
}

fn linearized_riemann_from_delta_christoffel(
    background_christoffel: &[Vec<Vec<Expr>>],
    delta_christoffel: &[Vec<Vec<Expr>>],
    coords: &[lasso::Spur],
    convention: &ax_ir::Convention,
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Vec<Expr>>>> {
    let n = coords.len();
    let mut riemann = vec![vec![vec![vec![Expr::zero(); n]; n]; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let term1 =
                        ax_tensor::diff_component(&delta_christoffel[i][l][j], coords[k], interner);
                    let term2 =
                        ax_tensor::diff_component(&delta_christoffel[i][k][j], coords[l], interner);

                    let mut pos_terms = Vec::with_capacity(2 * n);
                    let mut neg_terms = Vec::with_capacity(2 * n);
                    for m in 0..n {
                        pos_terms.push(Expr::mul(vec![
                            background_christoffel[i][k][m].clone(),
                            delta_christoffel[m][l][j].clone(),
                        ]));
                        pos_terms.push(Expr::mul(vec![
                            delta_christoffel[i][k][m].clone(),
                            background_christoffel[m][l][j].clone(),
                        ]));
                        neg_terms.push(Expr::mul(vec![
                            background_christoffel[i][l][m].clone(),
                            delta_christoffel[m][k][j].clone(),
                        ]));
                        neg_terms.push(Expr::mul(vec![
                            delta_christoffel[i][l][m].clone(),
                            background_christoffel[m][k][j].clone(),
                        ]));
                    }

                    let mtw_expr = Expr::add(vec![
                        term1,
                        Expr::neg(term2),
                        Expr::add(pos_terms),
                        Expr::neg(Expr::add(neg_terms)),
                    ]);
                    riemann[i][j][k][l] = simplify_linearized_expr(
                        match convention.riemann_sign {
                            ax_ir::RiemannSign::MTW => mtw_expr,
                            ax_ir::RiemannSign::Weinberg => Expr::neg(mtw_expr),
                        },
                        interner,
                    );
                }
            }
        }
    }

    riemann
}

fn linearized_ricci_scalar(
    background_ricci: &[Vec<Expr>],
    delta_ricci: &[Vec<Expr>],
    background_inverse_metric: &ax_tensor::SymbolicMatrix,
    delta_inverse_metric: &ax_tensor::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut terms = Vec::new();
    for j in 0..background_inverse_metric.dim {
        for l in 0..background_inverse_metric.dim {
            terms.push(Expr::mul(vec![
                background_inverse_metric.get(j, l).clone(),
                delta_ricci[j][l].clone(),
            ]));
            terms.push(Expr::mul(vec![
                delta_inverse_metric.get(j, l).clone(),
                background_ricci[j][l].clone(),
            ]));
        }
    }
    simplify_linearized_expr(Expr::add(terms), interner)
}

fn linearized_einstein_from_deltas(
    delta_ricci: &[Vec<Expr>],
    background_metric: &ax_tensor::SymbolicMatrix,
    delta_metric: &ax_tensor::SymbolicMatrix,
    background_ricci_scalar: &Expr,
    delta_ricci_scalar: &Expr,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, CosmologyError> {
    let n = background_metric.dim;
    let mut einstein = vec![vec![Expr::zero(); n]; n];
    let half = rational(1, 2);
    for j in 0..n {
        for l in 0..n {
            einstein[j][l] = simplify_linearized_expr(
                Expr::add(vec![
                    delta_ricci[j][l].clone(),
                    Expr::neg(Expr::mul(vec![
                        half.clone(),
                        delta_metric.get(j, l).clone(),
                        background_ricci_scalar.clone(),
                    ])),
                    Expr::neg(Expr::mul(vec![
                        half.clone(),
                        background_metric.get(j, l).clone(),
                        delta_ricci_scalar.clone(),
                    ])),
                ]),
                interner,
            );
        }
    }
    Ok(einstein)
}

fn add_rank2(
    lhs: &[Vec<Expr>],
    rhs: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, CosmologyError> {
    if lhs.len() != rhs.len() {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "add_rank2".to_string(),
            got: lhs.len(),
            expected: rhs.len(),
        });
    }
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_row, rhs_row)| {
            if lhs_row.len() != rhs_row.len() {
                return Err(CosmologyError::UnexpectedMatrixDimension {
                    operation: "add_rank2".to_string(),
                    got: lhs_row.len(),
                    expected: rhs_row.len(),
                });
            }
            Ok(lhs_row
                .iter()
                .zip(rhs_row.iter())
                .map(|(lhs_entry, rhs_entry)| {
                    simplify_linearized_expr(
                        Expr::add(vec![lhs_entry.clone(), rhs_entry.clone()]),
                        interner,
                    )
                })
                .collect())
        })
        .collect()
}

fn add_rank3(
    lhs: &[Vec<Vec<Expr>>],
    rhs: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<Expr>>>, CosmologyError> {
    if lhs.len() != rhs.len() {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "add_rank3".to_string(),
            got: lhs.len(),
            expected: rhs.len(),
        });
    }

    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs_i, rhs_i)| {
            lhs_i
                .iter()
                .zip(rhs_i.iter())
                .map(|(lhs_j, rhs_j)| {
                    lhs_j
                        .iter()
                        .zip(rhs_j.iter())
                        .map(|(lhs_entry, rhs_entry)| {
                            simplify_linearized_expr(
                                Expr::add(vec![lhs_entry.clone(), rhs_entry.clone()]),
                                interner,
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect())
}

fn add_rank4(
    lhs: &[Vec<Vec<Vec<Expr>>>],
    rhs: &[Vec<Vec<Vec<Expr>>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<Vec<Expr>>>>, CosmologyError> {
    if lhs.len() != rhs.len() {
        return Err(CosmologyError::UnexpectedMatrixDimension {
            operation: "add_rank4".to_string(),
            got: lhs.len(),
            expected: rhs.len(),
        });
    }

    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs_a, rhs_a)| {
            lhs_a
                .iter()
                .zip(rhs_a.iter())
                .map(|(lhs_b, rhs_b)| {
                    lhs_b
                        .iter()
                        .zip(rhs_b.iter())
                        .map(|(lhs_c, rhs_c)| {
                            lhs_c
                                .iter()
                                .zip(rhs_c.iter())
                                .map(|(lhs_entry, rhs_entry)| {
                                    simplify_linearized_expr(
                                        Expr::add(vec![lhs_entry.clone(), rhs_entry.clone()]),
                                        interner,
                                    )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect())
}

fn linearize_product(
    factors: &[Expr],
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let expanded = factors.iter().fold(vec![Expr::one()], |acc, factor| {
        let linearized = linearize_in_symbols(factor, perturbation_symbols, interner);
        let factor_terms = match linearized {
            Expr::Add(terms) => terms,
            other => vec![other],
        };
        acc.into_iter()
            .flat_map(|prefix| {
                factor_terms
                    .iter()
                    .cloned()
                    .map(move |term| Expr::mul(vec![prefix.clone(), term]))
            })
            .collect::<Vec<_>>()
    });

    Expr::add(
        expanded
            .into_iter()
            .filter(|term| count_perturbation_degree(term, perturbation_symbols, interner) <= 1)
            .collect(),
    )
}

fn linearize_power(
    base: &Expr,
    exp: &Expr,
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let linearized_base = linearize_in_symbols(base, perturbation_symbols, interner);
    match exp {
        Expr::Int(n) => {
            if *n == BigInt::one() {
                return linearized_base;
            }

            if let Some((background_part, perturbation_part)) =
                split_background_plus_perturbation(&linearized_base, perturbation_symbols, interner)
            {
                return simplify_linearized_expr(
                    Expr::add(vec![
                        Expr::pow(background_part.clone(), Expr::Int(n.clone())),
                        Expr::mul(vec![
                            Expr::Int(n.clone()),
                            Expr::pow(background_part, Expr::Int(n.clone() + BigInt::from(-1))),
                            perturbation_part,
                        ]),
                    ]),
                    interner,
                );
            }

            let rebuilt = Expr::pow(linearized_base, Expr::Int(n.clone()));
            if count_perturbation_degree(&rebuilt, perturbation_symbols, interner) <= 1 {
                rebuilt
            } else {
                Expr::zero()
            }
        }
        _ => Expr::pow(
            linearized_base,
            linearize_in_symbols(exp, perturbation_symbols, interner),
        ),
    }
}

fn split_background_plus_perturbation(
    expr: &Expr,
    perturbation_symbols: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Option<(Expr, Expr)> {
    match expr {
        Expr::Add(terms) => {
            let mut background_terms = Vec::new();
            let mut perturbation_terms = Vec::new();
            for term in terms {
                if count_perturbation_degree(term, perturbation_symbols, interner) == 0 {
                    background_terms.push(term.clone());
                } else if count_perturbation_degree(term, perturbation_symbols, interner) == 1 {
                    perturbation_terms.push(term.clone());
                } else {
                    return None;
                }
            }
            if background_terms.is_empty() || perturbation_terms.is_empty() {
                None
            } else {
                Some((Expr::add(background_terms), Expr::add(perturbation_terms)))
            }
        }
        other if count_perturbation_degree(other, perturbation_symbols, interner) == 0 => {
            Some((other.clone(), Expr::zero()))
        }
        _ => None,
    }
}

fn additive_terms(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Add(terms) => terms.clone(),
        other => vec![other.clone()],
    }
}

fn strip_single_gradient_term(
    term: &Expr,
    coordinate: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if let Expr::Neg(inner) = term {
        return strip_single_gradient_term(inner, coordinate, interner).map(Expr::neg);
    }
    if let Some(quotient) = factor_single_gradient(term, coordinate, interner) {
        return Some(quotient);
    }

    match term {
        Expr::Mul(factors) => {
            let matches = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| {
                    factor_single_gradient(factor, coordinate, interner)
                        .map(|quotient| (idx, quotient))
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return None;
            }
            let (matched_idx, quotient_factor) = matches[0].clone();
            let mut quotient = Vec::new();
            for (idx, factor) in factors.iter().enumerate() {
                if idx == matched_idx {
                    quotient.push(quotient_factor.clone());
                } else {
                    quotient.push(factor.clone());
                }
            }
            Some(Expr::mul(quotient))
        }
        _ => None,
    }
}

fn strip_mixed_gradient_term(
    term: &Expr,
    first: lasso::Spur,
    second: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if let Expr::Neg(inner) = term {
        return strip_mixed_gradient_term(inner, first, second, interner).map(Expr::neg);
    }
    match term {
        Expr::Call(_, _) if matches_mixed_gradient(term, first, second, interner).is_some() => {
            matches_mixed_gradient(term, first, second, interner)
        }
        Expr::Mul(factors) => {
            let matches = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| {
                    matches_mixed_gradient(factor, first, second, interner)
                        .map(|quotient| (idx, quotient))
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return None;
            }
            let (matched_idx, quotient_factor) = matches[0].clone();
            let mut quotient = Vec::new();
            for (idx, factor) in factors.iter().enumerate() {
                if idx == matched_idx {
                    quotient.push(quotient_factor.clone());
                } else {
                    quotient.push(factor.clone());
                }
            }
            Some(Expr::mul(quotient))
        }
        _ => None,
    }
}

fn matches_mixed_gradient(
    expr: &Expr,
    first: lasso::Spur,
    second: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Call(sym_outer, outer_args) = expr else {
        return None;
    };
    if interner.resolve(*sym_outer) != "diff" || outer_args.len() != 2 {
        return None;
    }
    let Expr::Call(sym_inner, inner_args) = &outer_args[0] else {
        return None;
    };
    if interner.resolve(*sym_inner) != "diff" || inner_args.len() != 2 {
        return None;
    }

    let outer_coord = outer_args[1].clone();
    let inner_coord = inner_args[1].clone();
    let direct = outer_coord == Expr::Sym(second) && inner_coord == Expr::Sym(first);
    let swapped = outer_coord == Expr::Sym(first) && inner_coord == Expr::Sym(second);
    (direct || swapped).then(|| inner_args[0].clone())
}

fn factor_single_gradient(
    expr: &Expr,
    coordinate: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Call(sym, args) = expr else {
        return None;
    };
    if interner.resolve(*sym) != "diff" || args.len() != 2 {
        return None;
    }
    if args[1] == Expr::Sym(coordinate) {
        return Some(args[0].clone());
    }

    let Expr::Call(inner_sym, inner_args) = &args[0] else {
        return None;
    };
    if interner.resolve(*inner_sym) != "diff" || inner_args.len() != 2 {
        return None;
    }
    if inner_args[1] == Expr::Sym(coordinate) {
        return expr_coord(&args[1])
            .map(|outer_coord| diff(inner_args[0].clone(), outer_coord, interner));
    }
    None
}

fn expr_coord(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        _ => None,
    }
}

pub(crate) fn simplify_linearized_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| simplify_linearized_expr(term, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let simplified_factors = factors
                .into_iter()
                .map(|factor| simplify_linearized_expr(factor, interner))
                .collect::<Vec<_>>();
            distribute_mul_over_add(simplified_factors)
        }
        Expr::Pow(base, exp) => Expr::pow(
            simplify_linearized_expr(*base, interner),
            simplify_linearized_expr(*exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(simplify_linearized_expr(*inner, interner)),
        Expr::Call(sym, args) => {
            let simplified_args = args
                .into_iter()
                .map(|arg| simplify_linearized_expr(arg, interner))
                .collect::<Vec<_>>();
            if interner.resolve(sym) == "diff" {
                simplify_diff_call(sym, simplified_args, interner)
            } else {
                Expr::Call(sym, simplified_args)
            }
        }
        Expr::FnDef(name, params, body) => Expr::FnDef(
            name,
            params,
            Box::new(simplify_linearized_expr(*body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(simplify_linearized_expr(*lhs, interner)),
            Box::new(simplify_linearized_expr(*rhs, interner)),
            trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| (simplify_linearized_expr(value, interner), condition))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(simplify_linearized_expr(*base, interner)), indices)
        }
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(simplify_linearized_expr(*inner, interner)), rel)
        }
        Expr::Let(name, value, body) => Expr::Let(
            name,
            Box::new(simplify_linearized_expr(*value, interner)),
            Box::new(simplify_linearized_expr(*body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| simplify_linearized_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| simplify_linearized_expr(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_linearized_expr(*re, interner)),
            Box::new(simplify_linearized_expr(*im, interner)),
        ),
        other => other,
    }
}

fn simplify_diff_call(sym: lasso::Spur, args: Vec<Expr>, interner: &ax_ir::Interner) -> Expr {
    if args.len() != 2 {
        return Expr::Call(sym, args);
    }

    let variable = args[1].clone();
    match &args[0] {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
        Expr::Call(inner_sym, inner_args)
            if interner.resolve(*inner_sym) == "diff" && inner_args.len() == 2 =>
        {
            let inner_var = inner_args[1].clone();
            if let (Some(inner_coord), Some(outer_coord)) =
                (expr_coord(&inner_var), expr_coord(&variable))
            {
                if interner.resolve(inner_coord) > interner.resolve(outer_coord) {
                    return Expr::Call(
                        sym,
                        vec![
                            Expr::Call(*inner_sym, vec![inner_args[0].clone(), variable]),
                            Expr::Sym(inner_coord),
                        ],
                    );
                }
            }
            Expr::Call(sym, args)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .cloned()
                .map(|term| Expr::Call(sym, vec![term, variable.clone()]))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(Expr::Call(sym, vec![inner.as_ref().clone(), variable])),
        _ => Expr::Call(interner.get_or_intern("diff"), args),
    }
}

fn distribute_mul_over_add(factors: Vec<Expr>) -> Expr {
    let expanded = factors.into_iter().fold(vec![Expr::one()], |acc, factor| {
        let terms = match factor {
            Expr::Add(terms) => terms,
            other => vec![other],
        };
        acc.into_iter()
            .flat_map(|prefix| {
                terms
                    .iter()
                    .cloned()
                    .map(move |term| Expr::mul(vec![prefix.clone(), term]))
            })
            .collect::<Vec<_>>()
    });
    Expr::add(expanded)
}

#[allow(dead_code)]
fn simplify_public_scalar_expr(
    expr: Expr,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut current = simplify_linearized_expr(expr, interner);
    loop {
        let next = rewrite_public_scalar_expr(current.clone(), bg, interner);
        if next == current {
            return next;
        }
        current = next;
    }
}

#[allow(dead_code)]
fn rewrite_public_scalar_expr(
    expr: Expr,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => collapse_laplacian_terms(
            terms
                .into_iter()
                .map(|term| rewrite_public_scalar_expr(term, bg, interner))
                .collect(),
            bg,
            interner,
        ),
        Expr::Mul(factors) => rewrite_background_hubble_patterns(
            factors
                .into_iter()
                .map(|factor| rewrite_public_scalar_expr(factor, bg, interner))
                .collect(),
            bg,
            interner,
        ),
        Expr::Pow(base, exp) => Expr::pow(
            rewrite_public_scalar_expr(*base, bg, interner),
            rewrite_public_scalar_expr(*exp, bg, interner),
        ),
        Expr::Neg(inner) => Expr::neg(rewrite_public_scalar_expr(*inner, bg, interner)),
        Expr::Call(sym, args) => Expr::Call(
            sym,
            args.into_iter()
                .map(|arg| rewrite_public_scalar_expr(arg, bg, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            name,
            params,
            Box::new(rewrite_public_scalar_expr(*body, bg, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rewrite_public_scalar_expr(*lhs, bg, interner)),
            Box::new(rewrite_public_scalar_expr(*rhs, bg, interner)),
            trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| {
                    (rewrite_public_scalar_expr(value, bg, interner), condition)
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(rewrite_public_scalar_expr(*base, bg, interner)),
            indices,
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(rewrite_public_scalar_expr(*inner, bg, interner)),
            rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            name,
            Box::new(rewrite_public_scalar_expr(*value, bg, interner)),
            Box::new(rewrite_public_scalar_expr(*body, bg, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| rewrite_public_scalar_expr(item, bg, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| rewrite_public_scalar_expr(cell, bg, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rewrite_public_scalar_expr(*re, bg, interner)),
            Box::new(rewrite_public_scalar_expr(*im, bg, interner)),
        ),
        other => other,
    }
}

#[allow(dead_code)]
fn collapse_laplacian_terms(
    terms: Vec<Expr>,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Expr {
    let chart = match default_frw_chart(interner, bg) {
        Ok(chart) => chart,
        Err(_) => return Expr::add(terms),
    };
    let coords = [chart.space.x, chart.space.y, chart.space.z];

    let mut used = vec![false; terms.len()];
    let mut rebuilt = Vec::new();

    for i in 0..terms.len() {
        if used[i] {
            continue;
        }
        if let Some((coeff, target)) = split_second_derivative_term(&terms[i], coords[0], interner)
        {
            let mut matched = vec![i];
            let mut complete = true;
            for coord in coords.iter().skip(1) {
                let mut found = None;
                for j in (i + 1)..terms.len() {
                    if used[j] {
                        continue;
                    }
                    if let Some((other_coeff, other_target)) =
                        split_second_derivative_term(&terms[j], *coord, interner)
                    {
                        if other_coeff == coeff && other_target == target {
                            found = Some(j);
                            break;
                        }
                    }
                }
                match found {
                    Some(index) => matched.push(index),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }

            if complete {
                for index in matched {
                    used[index] = true;
                }
                rebuilt.push(simplify_linearized_expr(
                    Expr::mul(vec![coeff, laplacian(target, interner)]),
                    interner,
                ));
                continue;
            }
        }
        used[i] = true;
        rebuilt.push(terms[i].clone());
    }

    Expr::add(rebuilt)
}

#[allow(dead_code)]
fn split_second_derivative_term(
    expr: &Expr,
    coordinate: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<(Expr, Expr)> {
    if let Some(target) = match_pure_second_derivative(expr, coordinate, interner) {
        return Some((Expr::one(), target));
    }

    let Expr::Mul(factors) = expr else {
        return None;
    };
    let matches = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| {
            match_pure_second_derivative(factor, coordinate, interner).map(|target| (idx, target))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return None;
    }
    let (matched_idx, target) = matches[0].clone();
    let coeff = Expr::mul(
        factors
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != matched_idx)
            .map(|(_, factor)| factor.clone())
            .collect(),
    );
    Some((coeff, target))
}

#[allow(dead_code)]
fn match_pure_second_derivative(
    expr: &Expr,
    coordinate: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let Expr::Call(sym_outer, outer_args) = expr else {
        return None;
    };
    if interner.resolve(*sym_outer) != "diff" || outer_args.len() != 2 {
        return None;
    }
    if outer_args[1] != Expr::Sym(coordinate) {
        return None;
    }
    let Expr::Call(sym_inner, inner_args) = &outer_args[0] else {
        return None;
    };
    if interner.resolve(*sym_inner) != "diff" || inner_args.len() != 2 {
        return None;
    }
    (inner_args[1] == Expr::Sym(coordinate)).then(|| inner_args[0].clone())
}

#[allow(dead_code)]
fn rewrite_background_hubble_patterns(
    factors: Vec<Expr>,
    bg: &FrwBackgroundSpec,
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

    if let Some(rewritten) = replace_mul_pair(
        &factors,
        |expr| matches_scale_factor_inverse(expr, bg),
        |expr| matches_scale_factor_double_prime(expr, bg, interner),
        || {
            Expr::add(vec![
                diff(Expr::Sym(bg.conformal_hubble), bg.conformal_time, interner),
                Expr::pow(Expr::Sym(bg.conformal_hubble), Expr::Int(2.into())),
            ])
        },
    ) {
        return simplify_linearized_expr(rewritten, interner);
    }

    if let Some(rewritten) = replace_scale_factor_prime_squared(&factors, bg, interner) {
        return simplify_linearized_expr(rewritten, interner);
    }

    Expr::mul(factors)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn replace_scale_factor_prime_squared(
    factors: &[Expr],
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let inverse_indices = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| matches_scale_factor_inverse(factor, bg).then_some(idx))
        .collect::<Vec<_>>();
    let prime_indices = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| matches_scale_factor_prime(factor, bg, interner).then_some(idx))
        .collect::<Vec<_>>();
    if inverse_indices.len() < 2 || prime_indices.len() < 2 {
        return None;
    }
    let skip = [
        inverse_indices[0],
        inverse_indices[1],
        prime_indices[0],
        prime_indices[1],
    ];
    let mut rebuilt = Vec::new();
    for (idx, factor) in factors.iter().enumerate() {
        if skip.contains(&idx) {
            continue;
        }
        rebuilt.push(factor.clone());
    }
    rebuilt.push(Expr::pow(
        Expr::Sym(bg.conformal_hubble),
        Expr::Int(2.into()),
    ));
    Some(Expr::mul(rebuilt))
}

#[allow(dead_code)]
fn matches_scale_factor_inverse(expr: &Expr, bg: &FrwBackgroundSpec) -> bool {
    *expr == Expr::pow(Expr::Sym(bg.scale_factor), Expr::Int((-1).into()))
}

#[allow(dead_code)]
fn matches_scale_factor_prime(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> bool {
    *expr == diff(Expr::Sym(bg.scale_factor), bg.conformal_time, interner)
}

#[allow(dead_code)]
fn matches_scale_factor_double_prime(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> bool {
    *expr
        == diff(
            diff(Expr::Sym(bg.scale_factor), bg.conformal_time, interner),
            bg.conformal_time,
            interner,
        )
}

fn diff(expr: Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn laplacian(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("laplacian"), vec![expr])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(BigRational::new(num.into(), den.into()))
}

fn v_symbol(interner: &ax_ir::Interner) -> lasso::Spur {
    interner.get_or_intern("v")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_symbol(expr: &Expr, symbol: lasso::Spur) -> bool {
        match expr {
            Expr::Sym(sym) => *sym == symbol,
            Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
                terms.iter().any(|term| contains_symbol(term, symbol))
            }
            Expr::Pow(base, exp) | Expr::Rule(base, exp, _) => {
                contains_symbol(base, symbol) || contains_symbol(exp, symbol)
            }
            Expr::Neg(inner) | Expr::Group(inner, _) => contains_symbol(inner, symbol),
            Expr::Call(_, args) => args.iter().any(|arg| contains_symbol(arg, symbol)),
            Expr::FnDef(_, _, body) => contains_symbol(body, symbol),
            Expr::Piecewise(cases) => cases
                .iter()
                .any(|(value, _)| contains_symbol(value, symbol)),
            Expr::Indexed(base, _) => contains_symbol(base, symbol),
            Expr::Let(_, value, body) => {
                contains_symbol(value, symbol) || contains_symbol(body, symbol)
            }
            Expr::Matrix(rows) => rows
                .iter()
                .flat_map(|row| row.iter())
                .any(|cell| contains_symbol(cell, symbol)),
            Expr::Complex(re, im) => contains_symbol(re, symbol) || contains_symbol(im, symbol),
            Expr::Int(_)
            | Expr::Rational(_)
            | Expr::Float(_)
            | Expr::Import(_)
            | Expr::Assume(_, _)
            | Expr::SetConvention(_, _) => false,
        }
    }

    #[test]
    fn linearize_in_symbols_drops_quadratic_products() {
        let interner = ax_ir::Interner::new();
        let phi = interner.get_or_intern("Phi");
        let psi = interner.get_or_intern("Psi");
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Sym(phi), Expr::Sym(psi)]),
            Expr::Sym(phi),
            int(3),
        ]);

        assert_eq!(
            linearize_in_symbols(&expr, &[phi, psi], &interner),
            Expr::add(vec![Expr::Sym(phi), int(3)])
        );
    }

    #[test]
    fn newtonian_scalar_metric_matrix_zeroes_b_and_e_contributions() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let names = crate::gauge::standard_svt_mode_names(&interner);
        let matrix = newtonian_scalar_metric_matrix(&bg, &interner).expect("newtonian metric");

        assert!(!contains_symbol(matrix.get(0, 1), names.b));
        assert!(!contains_symbol(matrix.get(1, 2), names.e));
    }

    #[test]
    fn derive_linearized_einstein_matrices_returns_4x4_delta_einstein() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let matrices =
            derive_linearized_einstein_matrices(&bg, &interner).expect("linearized matrices");

        assert_eq!(matrices.delta_einstein.len(), 4);
        assert!(matrices.delta_einstein.iter().all(|row| row.len() == 4));
    }

    #[test]
    fn strip_common_single_gradient_divides_termwise() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let x = interner.get_or_intern("x");
        let expr = Expr::add(vec![
            diff(Expr::Sym(a), x, &interner),
            Expr::mul(vec![int(2), diff(Expr::Sym(b), x, &interner)]),
        ]);

        assert_eq!(
            strip_common_single_gradient(&expr, x, &interner, "test").expect("strip"),
            Expr::add(vec![Expr::Sym(a), Expr::mul(vec![int(2), Expr::Sym(b)])])
        );
    }

    #[test]
    fn strip_common_mixed_gradient_divides_termwise() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let expr = Expr::add(vec![
            diff(diff(Expr::Sym(a), x, &interner), y, &interner),
            Expr::neg(Expr::mul(vec![
                int(3),
                diff(diff(Expr::Sym(b), x, &interner), y, &interner),
            ])),
        ]);

        assert_eq!(
            strip_common_mixed_gradient(&expr, x, y, &interner, "test").expect("strip"),
            Expr::add(vec![
                Expr::Sym(a),
                Expr::neg(Expr::mul(vec![int(3), Expr::Sym(b)]))
            ])
        );
    }

    #[test]
    fn derived_linearized_scalar_equations_match_current_public_formulas() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let equations = derive_linearized_scalar_equations_newtonian(&bg, &interner)
            .expect("derived scalar equations");
        let phi = Expr::Sym(interner.get_or_intern("Phi"));
        let psi = Expr::Sym(interner.get_or_intern("Psi"));
        let eta = bg.conformal_time;
        let h = Expr::Sym(bg.conformal_hubble);
        let a2 = Expr::pow(Expr::Sym(bg.scale_factor), Expr::Int(2.into()));
        let pi = Expr::Sym(interner.get_or_intern("pi"));
        let g = Expr::Sym(interner.get_or_intern("G"));
        let delta_rho = Expr::Sym(interner.get_or_intern("delta_rho"));
        let v = Expr::Sym(interner.get_or_intern("v"));
        let rho = Expr::Sym(interner.get_or_intern("rho"));
        let pressure = Expr::Sym(interner.get_or_intern("P"));
        let delta_pressure = Expr::Sym(interner.get_or_intern("delta_P"));
        let anisotropic_stress = Expr::Sym(interner.get_or_intern("Pi"));
        let four_pi_g_a2 = Expr::mul(vec![int(4), pi.clone(), g.clone(), a2.clone()]);
        let eight_pi_g_a2 = Expr::mul(vec![int(8), pi, g, a2]);
        let psi_prime = diff(psi.clone(), eta, &interner);
        let phi_prime = diff(phi.clone(), eta, &interner);

        let expected_00 = Expr::add(vec![
            laplacian(psi.clone(), &interner),
            Expr::neg(Expr::mul(vec![
                int(3),
                h.clone(),
                Expr::add(vec![
                    psi_prime.clone(),
                    Expr::mul(vec![h.clone(), phi.clone()]),
                ]),
            ])),
            Expr::neg(Expr::mul(vec![four_pi_g_a2.clone(), delta_rho])),
        ]);
        let expected_0i = Expr::add(vec![
            Expr::add(vec![
                psi_prime.clone(),
                Expr::mul(vec![h.clone(), phi.clone()]),
            ]),
            Expr::mul(vec![
                four_pi_g_a2.clone(),
                Expr::add(vec![rho, pressure]),
                v,
            ]),
        ]);
        let expected_trace = Expr::add(vec![
            diff(psi_prime.clone(), eta, &interner),
            Expr::mul(vec![
                h.clone(),
                Expr::add(vec![Expr::mul(vec![int(2), psi_prime]), phi_prime]),
            ]),
            Expr::mul(vec![
                Expr::add(vec![
                    Expr::mul(vec![int(2), diff(h.clone(), eta, &interner)]),
                    Expr::pow(h.clone(), Expr::Int(2.into())),
                ]),
                phi.clone(),
            ]),
            Expr::mul(vec![
                rational(1, 3),
                laplacian(
                    Expr::add(vec![phi.clone(), Expr::neg(psi.clone())]),
                    &interner,
                ),
            ]),
            Expr::neg(Expr::mul(vec![four_pi_g_a2, delta_pressure])),
        ]);
        let expected_traceless = Expr::add(vec![
            phi,
            Expr::neg(psi),
            Expr::neg(Expr::mul(vec![eight_pi_g_a2, anisotropic_stress])),
        ]);

        assert_eq!(equations.eq_00, expected_00);
        assert_eq!(equations.eq_0i, expected_0i);
        assert_eq!(equations.eq_trace, expected_trace);
        assert_eq!(equations.eq_traceless, expected_traceless);
    }

    #[test]
    fn linearized_scalar_equations_as_named_preserves_public_labels() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let named = linearized_scalar_equations_as_named(&bg, &interner).expect("named equations");

        assert_eq!(named[0].label, "00_constraint");
        assert_eq!(named[1].label, "0i_momentum");
        assert_eq!(named[2].label, "ij_trace");
        assert_eq!(named[3].label, "ij_traceless");
    }
}
