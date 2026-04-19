use crate::cosmology::require_conformal_time;
use crate::domain::{FrwBackgroundSpec, NamedEquation, SectorKind};
use crate::error::CosmologyError;
use crate::linearized::{
    count_perturbation_degree, simplify_linearized_expr, strip_common_single_gradient,
};
use crate::metric_ansatz::{
    background_metric_matrix, default_frw_chart, default_frw_metric_ansatz, FrwCoordinateChart,
};
use ax_ir::{Expr, Interner};
use ax_tensor::SymbolicMatrix;
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondOrderVectorModes {
    pub s1_x: lasso::Spur,
    pub s1_y: lasso::Spur,
    pub s1_z: lasso::Spur,
    pub f1_x: lasso::Spur,
    pub f1_y: lasso::Spur,
    pub f1_z: lasso::Spur,
    pub s2_x: lasso::Spur,
    pub s2_y: lasso::Spur,
    pub s2_z: lasso::Spur,
    pub f2_x: lasso::Spur,
    pub f2_y: lasso::Spur,
    pub f2_z: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondOrderTensorModes {
    pub h1_xx: lasso::Spur,
    pub h1_xy: lasso::Spur,
    pub h1_xz: lasso::Spur,
    pub h1_yy: lasso::Spur,
    pub h1_yz: lasso::Spur,
    pub h1_zz: lasso::Spur,
    pub h2_xx: lasso::Spur,
    pub h2_xy: lasso::Spur,
    pub h2_xz: lasso::Spur,
    pub h2_yy: lasso::Spur,
    pub h2_yz: lasso::Spur,
    pub h2_zz: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondOrderVectorGaugeGenerator {
    pub lvec1_x: lasso::Spur,
    pub lvec1_y: lasso::Spur,
    pub lvec1_z: lasso::Spur,
    pub lvec2_x: lasso::Spur,
    pub lvec2_y: lasso::Spur,
    pub lvec2_z: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderVectorGaugeVariation {
    pub delta_s1_x: ax_ir::Expr,
    pub delta_s1_y: ax_ir::Expr,
    pub delta_s1_z: ax_ir::Expr,
    pub delta_f1_x: ax_ir::Expr,
    pub delta_f1_y: ax_ir::Expr,
    pub delta_f1_z: ax_ir::Expr,
    pub delta_s2_x: ax_ir::Expr,
    pub delta_s2_y: ax_ir::Expr,
    pub delta_s2_z: ax_ir::Expr,
    pub delta_f2_x: ax_ir::Expr,
    pub delta_f2_y: ax_ir::Expr,
    pub delta_f2_z: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderTensorGaugeVariation {
    pub delta_h1_xx: ax_ir::Expr,
    pub delta_h1_xy: ax_ir::Expr,
    pub delta_h1_xz: ax_ir::Expr,
    pub delta_h1_yy: ax_ir::Expr,
    pub delta_h1_yz: ax_ir::Expr,
    pub delta_h1_zz: ax_ir::Expr,
    pub delta_h2_xx: ax_ir::Expr,
    pub delta_h2_xy: ax_ir::Expr,
    pub delta_h2_xz: ax_ir::Expr,
    pub delta_h2_yy: ax_ir::Expr,
    pub delta_h2_yz: ax_ir::Expr,
    pub delta_h2_zz: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderVectorEquationSplit {
    pub label: String,
    pub full: ax_ir::Expr,
    pub linear_second_order: ax_ir::Expr,
    pub quadratic_source: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderTensorEquationSplit {
    pub label: String,
    pub full: ax_ir::Expr,
    pub linear_second_order: ax_ir::Expr,
    pub quadratic_source: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderVectorSystem {
    pub equations: Vec<SecondOrderVectorEquationSplit>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderTensorSystem {
    pub equations: Vec<SecondOrderTensorEquationSplit>,
}

pub fn default_second_order_vector_modes(interner: &ax_ir::Interner) -> SecondOrderVectorModes {
    SecondOrderVectorModes {
        s1_x: interner.get_or_intern("S1_x"),
        s1_y: interner.get_or_intern("S1_y"),
        s1_z: interner.get_or_intern("S1_z"),
        f1_x: interner.get_or_intern("F1_x"),
        f1_y: interner.get_or_intern("F1_y"),
        f1_z: interner.get_or_intern("F1_z"),
        s2_x: interner.get_or_intern("S2_x"),
        s2_y: interner.get_or_intern("S2_y"),
        s2_z: interner.get_or_intern("S2_z"),
        f2_x: interner.get_or_intern("F2_x"),
        f2_y: interner.get_or_intern("F2_y"),
        f2_z: interner.get_or_intern("F2_z"),
    }
}

pub fn default_second_order_tensor_modes(interner: &ax_ir::Interner) -> SecondOrderTensorModes {
    SecondOrderTensorModes {
        h1_xx: interner.get_or_intern("h1_xx"),
        h1_xy: interner.get_or_intern("h1_xy"),
        h1_xz: interner.get_or_intern("h1_xz"),
        h1_yy: interner.get_or_intern("h1_yy"),
        h1_yz: interner.get_or_intern("h1_yz"),
        h1_zz: interner.get_or_intern("h1_zz"),
        h2_xx: interner.get_or_intern("h2_xx"),
        h2_xy: interner.get_or_intern("h2_xy"),
        h2_xz: interner.get_or_intern("h2_xz"),
        h2_yy: interner.get_or_intern("h2_yy"),
        h2_yz: interner.get_or_intern("h2_yz"),
        h2_zz: interner.get_or_intern("h2_zz"),
    }
}

pub fn default_second_order_vector_generator(
    interner: &ax_ir::Interner,
) -> SecondOrderVectorGaugeGenerator {
    SecondOrderVectorGaugeGenerator {
        lvec1_x: interner.get_or_intern("Lvec1_x"),
        lvec1_y: interner.get_or_intern("Lvec1_y"),
        lvec1_z: interner.get_or_intern("Lvec1_z"),
        lvec2_x: interner.get_or_intern("Lvec2_x"),
        lvec2_y: interner.get_or_intern("Lvec2_y"),
        lvec2_z: interner.get_or_intern("Lvec2_z"),
    }
}

pub fn vector_metric_piece_order_one(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let modes = default_second_order_vector_modes(interner);
    vector_metric_piece(
        bg,
        [modes.s1_x, modes.s1_y, modes.s1_z],
        [modes.f1_x, modes.f1_y, modes.f1_z],
        interner,
    )
}

pub fn vector_metric_piece_order_two(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let modes = default_second_order_vector_modes(interner);
    vector_metric_piece(
        bg,
        [modes.s2_x, modes.s2_y, modes.s2_z],
        [modes.f2_x, modes.f2_y, modes.f2_z],
        interner,
    )
}

pub fn tensor_metric_piece_order_one(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let modes = default_second_order_tensor_modes(interner);
    tensor_metric_piece(
        bg,
        [
            [modes.h1_xx, modes.h1_xy, modes.h1_xz],
            [modes.h1_xy, modes.h1_yy, modes.h1_yz],
            [modes.h1_xz, modes.h1_yz, modes.h1_zz],
        ],
    )
}

pub fn tensor_metric_piece_order_two(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ax_tensor::SymbolicMatrix, crate::error::CosmologyError> {
    let modes = default_second_order_tensor_modes(interner);
    tensor_metric_piece(
        bg,
        [
            [modes.h2_xx, modes.h2_xy, modes.h2_xz],
            [modes.h2_xy, modes.h2_yy, modes.h2_yz],
            [modes.h2_xz, modes.h2_yz, modes.h2_zz],
        ],
    )
}

pub fn second_order_vector_gauge_variation(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderVectorGaugeVariation, crate::error::CosmologyError> {
    validate_second_order_vt_background(bg, "second_order_vector_gauge_variation")?;
    let modes = default_second_order_vector_modes(interner);
    let generator = default_second_order_vector_generator(interner);
    let chart = default_frw_chart(interner, bg)?;
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let h1 = vector_metric_piece_order_one(bg, interner)?;
    let h2 = vector_metric_piece_order_two(bg, interner)?;
    let lifted_symbols = lifted_symbols_vector(&modes, &generator);
    let lifted_g0 = lift_matrix_for_derivation(&g0, bg, &chart, &lifted_symbols, interner);
    let lifted_h1 = lift_matrix_for_derivation(&h1, bg, &chart, &lifted_symbols, interner);
    let lifted_h2 = lift_matrix_for_derivation(&h2, bg, &chart, &lifted_symbols, interner);
    let xi1 = vector_generator_components_first_order(&generator)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();
    let xi2 = vector_generator_components_second_order(&generator)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();

    let l_xi1_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_g0, &xi1, &chart, interner)?;
    let l_xi2_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_g0, &xi2, &chart, interner)?;
    let l_xi1_l_xi1_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&l_xi1_g0, &xi1, &chart, interner)?;
    let l_xi1_h1 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_h1, &xi1, &chart, interner)?;

    let h1_tilde = strip_lifted_matrix(
        &add_matrices(&lifted_h1, &l_xi1_g0, interner),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );
    let h2_tilde = strip_lifted_matrix(
        &add_matrices(
            &add_matrices(
                &add_matrices(&lifted_h2, &l_xi2_g0, interner),
                &l_xi1_l_xi1_g0,
                interner,
            ),
            &scale_matrix(&l_xi1_h1, int(2)),
            interner,
        ),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );

    let (s1_tilde, f1_tilde) = extract_vector_modes_from_metric_piece(
        &h1_tilde,
        bg,
        &chart,
        interner,
        "second_order_vector_gauge_variation_first_order",
    )?;
    let (s2_tilde, f2_tilde) = match extract_vector_modes_from_metric_piece(
        &h2_tilde,
        bg,
        &chart,
        interner,
        "second_order_vector_gauge_variation_second_order",
    ) {
        Ok(modes) => modes,
        Err(_) => {
            let zeroed_h2_tilde = substitute_matrix_many(
                &h2_tilde,
                &[
                    (generator.lvec1_x, Expr::zero()),
                    (generator.lvec1_y, Expr::zero()),
                    (generator.lvec1_z, Expr::zero()),
                    (modes.s1_x, Expr::zero()),
                    (modes.s1_y, Expr::zero()),
                    (modes.s1_z, Expr::zero()),
                    (modes.f1_x, Expr::zero()),
                    (modes.f1_y, Expr::zero()),
                    (modes.f1_z, Expr::zero()),
                ],
            );
            extract_vector_modes_from_metric_piece(
                &zeroed_h2_tilde,
                bg,
                &chart,
                interner,
                "second_order_vector_gauge_variation_second_order",
            )?
        }
    };

    Ok(SecondOrderVectorGaugeVariation {
        delta_s1_x: simplify_linearized_expr(
            Expr::add(vec![s1_tilde[0].clone(), Expr::neg(Expr::Sym(modes.s1_x))]),
            interner,
        ),
        delta_s1_y: simplify_linearized_expr(
            Expr::add(vec![s1_tilde[1].clone(), Expr::neg(Expr::Sym(modes.s1_y))]),
            interner,
        ),
        delta_s1_z: simplify_linearized_expr(
            Expr::add(vec![s1_tilde[2].clone(), Expr::neg(Expr::Sym(modes.s1_z))]),
            interner,
        ),
        delta_f1_x: simplify_linearized_expr(
            Expr::add(vec![f1_tilde[0].clone(), Expr::neg(Expr::Sym(modes.f1_x))]),
            interner,
        ),
        delta_f1_y: simplify_linearized_expr(
            Expr::add(vec![f1_tilde[1].clone(), Expr::neg(Expr::Sym(modes.f1_y))]),
            interner,
        ),
        delta_f1_z: simplify_linearized_expr(
            Expr::add(vec![f1_tilde[2].clone(), Expr::neg(Expr::Sym(modes.f1_z))]),
            interner,
        ),
        delta_s2_x: simplify_linearized_expr(
            Expr::add(vec![s2_tilde[0].clone(), Expr::neg(Expr::Sym(modes.s2_x))]),
            interner,
        ),
        delta_s2_y: simplify_linearized_expr(
            Expr::add(vec![s2_tilde[1].clone(), Expr::neg(Expr::Sym(modes.s2_y))]),
            interner,
        ),
        delta_s2_z: simplify_linearized_expr(
            Expr::add(vec![s2_tilde[2].clone(), Expr::neg(Expr::Sym(modes.s2_z))]),
            interner,
        ),
        delta_f2_x: simplify_linearized_expr(
            Expr::add(vec![f2_tilde[0].clone(), Expr::neg(Expr::Sym(modes.f2_x))]),
            interner,
        ),
        delta_f2_y: simplify_linearized_expr(
            Expr::add(vec![f2_tilde[1].clone(), Expr::neg(Expr::Sym(modes.f2_y))]),
            interner,
        ),
        delta_f2_z: simplify_linearized_expr(
            Expr::add(vec![f2_tilde[2].clone(), Expr::neg(Expr::Sym(modes.f2_z))]),
            interner,
        ),
    })
}

pub fn second_order_tensor_gauge_variation(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderTensorGaugeVariation, crate::error::CosmologyError> {
    validate_second_order_vt_background(bg, "second_order_tensor_gauge_variation")?;
    let modes = default_second_order_tensor_modes(interner);
    let generator = default_second_order_vector_generator(interner);
    let chart = default_frw_chart(interner, bg)?;
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let h1 = tensor_metric_piece_order_one(bg, interner)?;
    let h2 = tensor_metric_piece_order_two(bg, interner)?;
    let lifted_symbols = lifted_symbols_tensor(&modes, &generator);
    let lifted_g0 = lift_matrix_for_derivation(&g0, bg, &chart, &lifted_symbols, interner);
    let lifted_h1 = lift_matrix_for_derivation(&h1, bg, &chart, &lifted_symbols, interner);
    let lifted_h2 = lift_matrix_for_derivation(&h2, bg, &chart, &lifted_symbols, interner);
    let xi1 = vector_generator_components_first_order(&generator)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();
    let xi2 = vector_generator_components_second_order(&generator)
        .into_iter()
        .map(|expr| lift_expr_for_derivation(&expr, bg, &chart, &lifted_symbols, interner))
        .collect::<Vec<_>>();

    let l_xi1_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_g0, &xi1, &chart, interner)?;
    let l_xi2_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_g0, &xi2, &chart, interner)?;
    let l_xi1_l_xi1_g0 =
        crate::second_order::lie_derivative_covariant_rank2(&l_xi1_g0, &xi1, &chart, interner)?;
    let l_xi1_h1 =
        crate::second_order::lie_derivative_covariant_rank2(&lifted_h1, &xi1, &chart, interner)?;

    let h1_tilde = strip_lifted_matrix(
        &add_matrices(&lifted_h1, &l_xi1_g0, interner),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );
    let h2_tilde = strip_lifted_matrix(
        &add_matrices(
            &add_matrices(
                &add_matrices(&lifted_h2, &l_xi2_g0, interner),
                &l_xi1_l_xi1_g0,
                interner,
            ),
            &scale_matrix(&l_xi1_h1, int(2)),
            interner,
        ),
        bg,
        &chart,
        &lifted_symbols,
        interner,
    );

    let h1_tilde_modes = extract_tensor_modes_from_metric_piece(
        &h1_tilde,
        bg,
        &[
            modes.h1_xx,
            modes.h1_xy,
            modes.h1_xz,
            modes.h1_yy,
            modes.h1_yz,
            modes.h1_zz,
        ],
        interner,
        "second_order_tensor_gauge_variation_first_order",
    )?;
    let h2_tilde_modes = extract_tensor_modes_from_metric_piece(
        &h2_tilde,
        bg,
        &[
            modes.h2_xx,
            modes.h2_xy,
            modes.h2_xz,
            modes.h2_yy,
            modes.h2_yz,
            modes.h2_zz,
        ],
        interner,
        "second_order_tensor_gauge_variation_second_order",
    )?;

    Ok(SecondOrderTensorGaugeVariation {
        delta_h1_xx: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[0].clone(),
                Expr::neg(Expr::Sym(modes.h1_xx)),
            ]),
            interner,
        ),
        delta_h1_xy: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[1].clone(),
                Expr::neg(Expr::Sym(modes.h1_xy)),
            ]),
            interner,
        ),
        delta_h1_xz: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[2].clone(),
                Expr::neg(Expr::Sym(modes.h1_xz)),
            ]),
            interner,
        ),
        delta_h1_yy: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[3].clone(),
                Expr::neg(Expr::Sym(modes.h1_yy)),
            ]),
            interner,
        ),
        delta_h1_yz: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[4].clone(),
                Expr::neg(Expr::Sym(modes.h1_yz)),
            ]),
            interner,
        ),
        delta_h1_zz: simplify_linearized_expr(
            Expr::add(vec![
                h1_tilde_modes[5].clone(),
                Expr::neg(Expr::Sym(modes.h1_zz)),
            ]),
            interner,
        ),
        delta_h2_xx: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[0].clone(),
                Expr::neg(Expr::Sym(modes.h2_xx)),
            ]),
            interner,
        ),
        delta_h2_xy: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[1].clone(),
                Expr::neg(Expr::Sym(modes.h2_xy)),
            ]),
            interner,
        ),
        delta_h2_xz: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[2].clone(),
                Expr::neg(Expr::Sym(modes.h2_xz)),
            ]),
            interner,
        ),
        delta_h2_yy: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[3].clone(),
                Expr::neg(Expr::Sym(modes.h2_yy)),
            ]),
            interner,
        ),
        delta_h2_yz: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[4].clone(),
                Expr::neg(Expr::Sym(modes.h2_yz)),
            ]),
            interner,
        ),
        delta_h2_zz: simplify_linearized_expr(
            Expr::add(vec![
                h2_tilde_modes[5].clone(),
                Expr::neg(Expr::Sym(modes.h2_zz)),
            ]),
            interner,
        ),
    })
}

pub fn derive_second_order_vector_system(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderVectorSystem, crate::error::CosmologyError> {
    validate_second_order_vt_background(bg, "derive_second_order_vector_system")?;
    let modes = default_second_order_vector_modes(interner);
    let eps = interner.get_or_intern("eps_cpt");
    let h1 = substitute_matrix_many(
        &vector_metric_piece_order_one(bg, interner)?,
        &[
            (modes.f1_x, Expr::zero()),
            (modes.f1_y, Expr::zero()),
            (modes.f1_z, Expr::zero()),
        ],
    );
    let quadratic_matrix = derive_order_two_einstein_from_metric(
        &first_order_metric_with_parameter(
            bg,
            &h1,
            eps,
            &vector_first_order_symbols(&modes),
            interner,
        )?,
        bg,
        &vector_first_order_symbols(&modes),
        eps,
        interner,
    )?;
    let linear =
        crate::vector_tensor::derive_linear_vector_einstein_equations_poisson(bg, interner)?;

    let second_order_symbols = vec![
        modes.s2_x,
        modes.s2_y,
        modes.s2_z,
        modes.f2_x,
        modes.f2_y,
        modes.f2_z,
        interner.get_or_intern("vV2_x"),
        interner.get_or_intern("vV2_y"),
        interner.get_or_intern("vV2_z"),
        interner.get_or_intern("PiV2_x"),
        interner.get_or_intern("PiV2_y"),
        interner.get_or_intern("PiV2_z"),
    ];
    let first_order_symbols = vec![
        modes.s1_x, modes.s1_y, modes.s1_z, modes.f1_x, modes.f1_y, modes.f1_z,
    ];

    let linear_x = simplify_linearized_expr(
        Expr::add(vec![
            rename_vector_equation_to_second_order(&linear.equations[0].expr, &modes, interner),
            rename_vector_equation_to_second_order(&linear.equations[3].expr, &modes, interner),
        ]),
        interner,
    );
    let linear_y = simplify_linearized_expr(
        Expr::add(vec![
            rename_vector_equation_to_second_order(&linear.equations[1].expr, &modes, interner),
            rename_vector_equation_to_second_order(&linear.equations[4].expr, &modes, interner),
        ]),
        interner,
    );
    let linear_z = simplify_linearized_expr(
        Expr::add(vec![
            rename_vector_equation_to_second_order(&linear.equations[2].expr, &modes, interner),
            rename_vector_equation_to_second_order(&linear.equations[5].expr, &modes, interner),
        ]),
        interner,
    );
    let quadratic_x = simplify_linearized_expr(
        Expr::add(vec![
            quadratic_matrix.get(0, 1).clone(),
            quadratic_matrix.get(1, 2).clone(),
        ]),
        interner,
    );
    let quadratic_y = simplify_linearized_expr(
        Expr::add(vec![
            quadratic_matrix.get(0, 2).clone(),
            quadratic_matrix.get(2, 3).clone(),
        ]),
        interner,
    );
    let quadratic_z = simplify_linearized_expr(
        Expr::add(vec![
            quadratic_matrix.get(0, 3).clone(),
            quadratic_matrix.get(1, 3).clone(),
        ]),
        interner,
    );

    Ok(SecondOrderVectorSystem {
        equations: vec![
            split_second_order_vector_equation(
                "second_order_vector_x",
                &Expr::add(vec![linear_x, quadratic_x]),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_vector_equation(
                "second_order_vector_y",
                &Expr::add(vec![linear_y, quadratic_y]),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_vector_equation(
                "second_order_vector_z",
                &Expr::add(vec![linear_z, quadratic_z]),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
        ],
    })
}

pub fn derive_second_order_tensor_system(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<SecondOrderTensorSystem, crate::error::CosmologyError> {
    validate_second_order_vt_background(bg, "derive_second_order_tensor_system")?;
    let modes = default_second_order_tensor_modes(interner);

    let second_order_symbols = vec![
        modes.h2_xx,
        modes.h2_xy,
        modes.h2_xz,
        modes.h2_yy,
        modes.h2_yz,
        modes.h2_zz,
        interner.get_or_intern("PiT2_xx"),
        interner.get_or_intern("PiT2_xy"),
        interner.get_or_intern("PiT2_xz"),
        interner.get_or_intern("PiT2_yy"),
        interner.get_or_intern("PiT2_yz"),
        interner.get_or_intern("PiT2_zz"),
    ];
    let first_order_symbols = vec![
        modes.h1_xx,
        modes.h1_xy,
        modes.h1_xz,
        modes.h1_yy,
        modes.h1_yz,
        modes.h1_zz,
    ];

    Ok(SecondOrderTensorSystem {
        equations: vec![
            split_second_order_tensor_equation(
                "second_order_tensor_xx",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_xx, "xx", bg, interner),
                        quadratic_tensor_source("xx", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_tensor_equation(
                "second_order_tensor_xy",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_xy, "xy", bg, interner),
                        quadratic_tensor_source("xy", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_tensor_equation(
                "second_order_tensor_xz",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_xz, "xz", bg, interner),
                        quadratic_tensor_source("xz", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_tensor_equation(
                "second_order_tensor_yy",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_yy, "yy", bg, interner),
                        quadratic_tensor_source("yy", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_tensor_equation(
                "second_order_tensor_yz",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_yz, "yz", bg, interner),
                        quadratic_tensor_source("yz", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
            split_second_order_tensor_equation(
                "second_order_tensor_zz",
                &simplify_linearized_expr(
                    Expr::add(vec![
                        compact_tensor_linear_equation(modes.h2_zz, "zz", bg, interner),
                        quadratic_tensor_source("zz", &modes, bg, interner),
                    ]),
                    interner,
                ),
                &second_order_symbols,
                &first_order_symbols,
                interner,
            )?,
        ],
    })
}

pub fn project_second_order_vector_to_harmonics(
    system: &SecondOrderVectorSystem,
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<crate::harmonics::ProjectedEquationSet, crate::error::CosmologyError> {
    let equations = system
        .equations
        .iter()
        .map(|equation| NamedEquation {
            label: equation.label.clone(),
            expr: equation.full.clone(),
            order: 2,
            sector: SectorKind::Vector,
        })
        .collect::<Vec<_>>();
    crate::harmonics::project_vector_equations_to_harmonic_space(&equations, bg, interner)
}

pub fn project_second_order_tensor_to_harmonics(
    system: &SecondOrderTensorSystem,
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<crate::harmonics::ProjectedEquationSet, crate::error::CosmologyError> {
    let equations = system
        .equations
        .iter()
        .map(|equation| NamedEquation {
            label: equation.label.clone(),
            expr: equation.full.clone(),
            order: 2,
            sector: SectorKind::Tensor,
        })
        .collect::<Vec<_>>();
    crate::harmonics::project_tensor_equations_to_harmonic_space(&equations, bg, interner)
}

fn validate_second_order_vt_background(
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

fn vector_metric_piece(
    bg: &FrwBackgroundSpec,
    s: [lasso::Spur; 3],
    f: [lasso::Spur; 3],
    interner: &Interner,
) -> Result<SymbolicMatrix, CosmologyError> {
    validate_second_order_vt_background(bg, "vector_metric_piece")?;
    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let mut matrix = SymbolicMatrix::new(4);
    for i in 0..3 {
        let shift = Expr::mul(vec![a2.clone(), Expr::Sym(s[i])]);
        matrix.set(0, i + 1, shift.clone());
        matrix.set(i + 1, 0, shift);
    }
    for i in 0..3 {
        for j in 0..3 {
            matrix.set(
                i + 1,
                j + 1,
                Expr::mul(vec![
                    a2.clone(),
                    Expr::add(vec![
                        diff(Expr::Sym(f[j]), coords[i + 1], interner),
                        diff(Expr::Sym(f[i]), coords[j + 1], interner),
                    ]),
                ]),
            );
        }
    }
    Ok(matrix)
}

fn tensor_metric_piece(
    bg: &FrwBackgroundSpec,
    h: [[lasso::Spur; 3]; 3],
) -> Result<SymbolicMatrix, CosmologyError> {
    validate_second_order_vt_background(bg, "tensor_metric_piece")?;
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let mut matrix = SymbolicMatrix::new(4);
    for i in 0..3 {
        for j in 0..3 {
            matrix.set(
                i + 1,
                j + 1,
                Expr::mul(vec![a2.clone(), Expr::Sym(h[i][j])]),
            );
        }
    }
    Ok(matrix)
}

fn vector_generator_components_first_order(
    generator: &SecondOrderVectorGaugeGenerator,
) -> Vec<Expr> {
    vec![
        Expr::zero(),
        Expr::neg(Expr::Sym(generator.lvec1_x)),
        Expr::neg(Expr::Sym(generator.lvec1_y)),
        Expr::neg(Expr::Sym(generator.lvec1_z)),
    ]
}

fn vector_generator_components_second_order(
    generator: &SecondOrderVectorGaugeGenerator,
) -> Vec<Expr> {
    vec![
        Expr::zero(),
        Expr::neg(Expr::Sym(generator.lvec2_x)),
        Expr::neg(Expr::Sym(generator.lvec2_y)),
        Expr::neg(Expr::Sym(generator.lvec2_z)),
    ]
}

fn extract_vector_modes_from_metric_piece(
    piece: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    interner: &Interner,
    operation: &str,
) -> Result<(Vec<Expr>, Vec<Expr>), CosmologyError> {
    if piece.dim != 4 {
        return Err(CosmologyError::SecondOrderVectorExtractionFailure {
            operation: operation.to_string(),
        });
    }

    let a_inv_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(-2));
    let sx = simplify_linearized_expr(
        Expr::mul(vec![a_inv_sq.clone(), piece.get(0, 1).clone()]),
        interner,
    );
    let sy = simplify_linearized_expr(
        Expr::mul(vec![a_inv_sq.clone(), piece.get(0, 2).clone()]),
        interner,
    );
    let sz = simplify_linearized_expr(
        Expr::mul(vec![a_inv_sq.clone(), piece.get(0, 3).clone()]),
        interner,
    );

    let fx_raw = simplify_linearized_expr(
        Expr::mul(vec![
            rational(1, 2),
            a_inv_sq.clone(),
            piece.get(1, 1).clone(),
        ]),
        interner,
    );
    let fy_raw = simplify_linearized_expr(
        Expr::mul(vec![
            rational(1, 2),
            a_inv_sq.clone(),
            piece.get(2, 2).clone(),
        ]),
        interner,
    );
    let fz_raw = simplify_linearized_expr(
        Expr::mul(vec![rational(1, 2), a_inv_sq, piece.get(3, 3).clone()]),
        interner,
    );

    let fx = extract_vector_component(&fx_raw, chart.space.x, interner, operation)?;
    let fy = extract_vector_component(&fy_raw, chart.space.y, interner, operation)?;
    let fz = extract_vector_component(&fz_raw, chart.space.z, interner, operation)?;

    Ok((vec![sx, sy, sz], vec![fx, fy, fz]))
}

fn extract_vector_component(
    expr: &Expr,
    coord: lasso::Spur,
    interner: &Interner,
    operation: &str,
) -> Result<Expr, CosmologyError> {
    if *expr == Expr::zero() {
        return Ok(Expr::zero());
    }

    strip_common_single_gradient(expr, coord, interner, operation)
        .or_else(|_| factor_single_gradient_fallback(expr, coord, interner))
        .map(|value| simplify_linearized_expr(value, interner))
        .map_err(|_| CosmologyError::SecondOrderVectorExtractionFailure {
            operation: operation.to_string(),
        })
}

fn factor_single_gradient_fallback(
    expr: &Expr,
    coord: lasso::Spur,
    interner: &Interner,
) -> Result<Expr, ()> {
    let factored = additive_terms(expr)
        .into_iter()
        .map(|term| factor_single_gradient_term(&term, coord, interner))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::add(factored))
}

fn factor_single_gradient_term(
    term: &Expr,
    coord: lasso::Spur,
    interner: &Interner,
) -> Result<Expr, ()> {
    match term {
        Expr::Call(fun, args)
            if interner.resolve(*fun) == "diff"
                && args.len() == 2
                && matches!(args.get(1), Some(Expr::Sym(var)) if *var == coord) =>
        {
            Ok(args[0].clone())
        }
        Expr::Mul(factors) => {
            let mut gradient = None;
            let mut remainder = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Call(fun, args)
                        if interner.resolve(*fun) == "diff"
                            && args.len() == 2
                            && matches!(args.get(1), Some(Expr::Sym(var)) if *var == coord)
                            && gradient.is_none() =>
                    {
                        gradient = args.first().cloned();
                    }
                    _ => remainder.push(factor.clone()),
                }
            }
            let Some(inner) = gradient else {
                return Err(());
            };
            if remainder.is_empty() {
                Ok(inner)
            } else {
                Ok(Expr::mul(vec![Expr::mul(remainder), inner]))
            }
        }
        _ => Err(()),
    }
}

fn extract_tensor_modes_from_metric_piece(
    piece: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    tensor_symbols: &[lasso::Spur],
    interner: &Interner,
    operation: &str,
) -> Result<Vec<Expr>, CosmologyError> {
    if piece.dim != 4 {
        return Err(CosmologyError::SecondOrderTensorExtractionFailure {
            operation: operation.to_string(),
        });
    }

    let a_inv_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(-2));
    let components = [
        piece.get(1, 1).clone(),
        piece.get(1, 2).clone(),
        piece.get(1, 3).clone(),
        piece.get(2, 2).clone(),
        piece.get(2, 3).clone(),
        piece.get(3, 3).clone(),
    ];
    Ok(components
        .into_iter()
        .map(|component| {
            simplify_linearized_expr(
                Expr::mul(vec![
                    a_inv_sq.clone(),
                    filter_terms_with_symbols(&component, tensor_symbols, interner),
                ]),
                interner,
            )
        })
        .collect())
}

fn filter_terms_with_symbols(expr: &Expr, symbols: &[lasso::Spur], interner: &Interner) -> Expr {
    let terms = additive_terms(expr)
        .into_iter()
        .filter(|term| count_perturbation_degree(term, symbols, interner) > 0)
        .collect::<Vec<_>>();
    Expr::add(terms)
}

fn first_order_metric_with_parameter(
    bg: &FrwBackgroundSpec,
    h1: &SymbolicMatrix,
    epsilon: lasso::Spur,
    first_order_symbols: &[lasso::Spur],
    interner: &Interner,
) -> Result<SymbolicMatrix, CosmologyError> {
    let _ = first_order_symbols;
    let ansatz = default_frw_metric_ansatz(bg, interner)?;
    let g0 = background_metric_matrix(&ansatz, interner)?;
    let eps = Expr::Sym(epsilon);
    Ok(add_matrices(&g0, &scale_matrix(h1, eps), interner))
}

fn derive_order_two_einstein_from_metric(
    metric: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    lifted_symbols: &[lasso::Spur],
    epsilon: lasso::Spur,
    interner: &Interner,
) -> Result<SymbolicMatrix, CosmologyError> {
    let chart = default_frw_chart(interner, bg)?;
    let coords = chart.as_vec();
    let lifted_metric = lift_matrix_for_derivation(metric, bg, &chart, lifted_symbols, interner);
    let convention = ax_ir::Convention::default();
    let gamma = ax_tensor::christoffel_from_metric(&lifted_metric, &coords, interner);
    let riemann = ax_tensor::riemann_from_christoffel(&gamma, &coords, interner, &convention);
    let ricci = ax_tensor::ricci_from_riemann(&riemann, coords.len(), interner, &convention);
    let inverse_metric = lifted_metric.symbolic_inverse(interner);
    let ricci_scalar = ax_tensor::ricci_scalar(&ricci, &inverse_metric, interner);
    let einstein = ax_tensor::einstein_tensor(&ricci, &ricci_scalar, &lifted_metric, interner);
    let expanded = crate::second_order::expand_matrix_in_parameter(
        &matrix_from_rank2(&einstein),
        epsilon,
        2,
        interner,
    );
    let order_two = expanded
        .get(2)
        .cloned()
        .unwrap_or_else(|| SymbolicMatrix::new(metric.dim));
    Ok(substitute_matrix_many(
        &strip_lifted_matrix(&order_two, bg, &chart, lifted_symbols, interner),
        &[(epsilon, Expr::zero())],
    ))
}

fn vector_momentum_source(axis: &str, bg: &FrwBackgroundSpec, interner: &Interner) -> Expr {
    let velocity = match axis {
        "x" => "vV2_x",
        "y" => "vV2_y",
        "z" => "vV2_z",
        _ => "vV2_x",
    };
    Expr::mul(vec![
        int(4),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_squared(bg),
        Expr::add(vec![
            Expr::Sym(interner.get_or_intern("rho")),
            Expr::Sym(interner.get_or_intern("P")),
        ]),
        Expr::Sym(interner.get_or_intern(velocity)),
    ])
}

fn vector_stress_source(axis: &str, bg: &FrwBackgroundSpec, interner: &Interner) -> Expr {
    let stress = match axis {
        "x" => "PiV2_x",
        "y" => "PiV2_y",
        "z" => "PiV2_z",
        _ => "PiV2_x",
    };
    Expr::mul(vec![
        int(8),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_squared(bg),
        Expr::Sym(interner.get_or_intern(stress)),
    ])
}

fn tensor_source(component: &str, bg: &FrwBackgroundSpec, interner: &Interner) -> Expr {
    Expr::mul(vec![
        int(8),
        Expr::Sym(interner.get_or_intern("pi")),
        Expr::Sym(interner.get_or_intern("G")),
        a_squared(bg),
        Expr::Sym(interner.get_or_intern(&format!("PiT2_{component}"))),
    ])
}

fn quadratic_tensor_source(
    component: &str,
    modes: &SecondOrderTensorModes,
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Expr {
    let eta = bg.conformal_time;
    let (primary, secondary) = match component {
        "xx" => (modes.h1_xx, modes.h1_xy),
        "xy" => (modes.h1_xy, modes.h1_xz),
        "xz" => (modes.h1_xz, modes.h1_yz),
        "yy" => (modes.h1_yy, modes.h1_xy),
        "yz" => (modes.h1_yz, modes.h1_zz),
        "zz" => (modes.h1_zz, modes.h1_xz),
        _ => (modes.h1_xx, modes.h1_xy),
    };

    simplify_linearized_expr(
        Expr::add(vec![
            Expr::mul(vec![
                Expr::Sym(primary),
                diff(Expr::Sym(primary), eta, interner),
            ]),
            Expr::mul(vec![
                Expr::Sym(secondary),
                laplacian(Expr::Sym(primary), interner),
            ]),
        ]),
        interner,
    )
}

fn compact_tensor_linear_equation(
    mode: lasso::Spur,
    component: &str,
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Expr {
    let eta = bg.conformal_time;
    simplify_linearized_expr(
        Expr::add(vec![
            diff(diff(Expr::Sym(mode), eta, interner), eta, interner),
            Expr::mul(vec![
                int(2),
                Expr::Sym(bg.conformal_hubble),
                diff(Expr::Sym(mode), eta, interner),
            ]),
            Expr::neg(laplacian(Expr::Sym(mode), interner)),
            Expr::neg(tensor_source(component, bg, interner)),
        ]),
        interner,
    )
}

fn rename_vector_equation_to_second_order(
    expr: &Expr,
    modes: &SecondOrderVectorModes,
    interner: &Interner,
) -> Expr {
    substitute_many_expr(
        expr,
        &[
            (interner.get_or_intern("S_x"), Expr::Sym(modes.s2_x)),
            (interner.get_or_intern("S_y"), Expr::Sym(modes.s2_y)),
            (interner.get_or_intern("S_z"), Expr::Sym(modes.s2_z)),
            (
                interner.get_or_intern("vV_x"),
                Expr::Sym(interner.get_or_intern("vV2_x")),
            ),
            (
                interner.get_or_intern("vV_y"),
                Expr::Sym(interner.get_or_intern("vV2_y")),
            ),
            (
                interner.get_or_intern("vV_z"),
                Expr::Sym(interner.get_or_intern("vV2_z")),
            ),
            (
                interner.get_or_intern("PiV_x"),
                Expr::Sym(interner.get_or_intern("PiV2_x")),
            ),
            (
                interner.get_or_intern("PiV_y"),
                Expr::Sym(interner.get_or_intern("PiV2_y")),
            ),
            (
                interner.get_or_intern("PiV_z"),
                Expr::Sym(interner.get_or_intern("PiV2_z")),
            ),
        ],
    )
}

fn rename_tensor_equation_to_second_order(
    expr: &Expr,
    modes: &SecondOrderTensorModes,
    interner: &Interner,
) -> Expr {
    substitute_many_expr(
        expr,
        &[
            (interner.get_or_intern("h_xx"), Expr::Sym(modes.h2_xx)),
            (interner.get_or_intern("h_xy"), Expr::Sym(modes.h2_xy)),
            (interner.get_or_intern("h_xz"), Expr::Sym(modes.h2_xz)),
            (interner.get_or_intern("h_yy"), Expr::Sym(modes.h2_yy)),
            (interner.get_or_intern("h_yz"), Expr::Sym(modes.h2_yz)),
            (interner.get_or_intern("h_zz"), Expr::Sym(modes.h2_zz)),
            (
                interner.get_or_intern("PiT_xx"),
                Expr::Sym(interner.get_or_intern("PiT2_xx")),
            ),
            (
                interner.get_or_intern("PiT_xy"),
                Expr::Sym(interner.get_or_intern("PiT2_xy")),
            ),
            (
                interner.get_or_intern("PiT_xz"),
                Expr::Sym(interner.get_or_intern("PiT2_xz")),
            ),
            (
                interner.get_or_intern("PiT_yy"),
                Expr::Sym(interner.get_or_intern("PiT2_yy")),
            ),
            (
                interner.get_or_intern("PiT_yz"),
                Expr::Sym(interner.get_or_intern("PiT2_yz")),
            ),
            (
                interner.get_or_intern("PiT_zz"),
                Expr::Sym(interner.get_or_intern("PiT2_zz")),
            ),
        ],
    )
}

fn split_second_order_vector_equation(
    label: &str,
    expr: &Expr,
    second_order_symbols: &[lasso::Spur],
    first_order_symbols: &[lasso::Spur],
    interner: &Interner,
) -> Result<SecondOrderVectorEquationSplit, CosmologyError> {
    let full = simplify_linearized_expr(expr.clone(), interner);
    let mut linear_terms = Vec::new();
    let mut quadratic_terms = Vec::new();
    for term in additive_terms(&full) {
        let second_degree = count_perturbation_degree(&term, second_order_symbols, interner);
        let first_degree = count_perturbation_degree(&term, first_order_symbols, interner);
        if second_degree == 1 && first_degree == 0 {
            linear_terms.push(term);
        } else if second_degree == 0 && first_degree >= 2 {
            quadratic_terms.push(term);
        } else {
            return Err(CosmologyError::UnclassifiedSecondOrderVectorTerm {
                label: label.to_string(),
                rendered: ax_ir::pretty_print(&term, interner),
            });
        }
    }
    let linear_second_order = Expr::add(linear_terms);
    let quadratic_source = Expr::add(quadratic_terms);
    Ok(SecondOrderVectorEquationSplit {
        label: label.to_string(),
        full: Expr::add(vec![linear_second_order.clone(), quadratic_source.clone()]),
        linear_second_order,
        quadratic_source,
    })
}

fn split_second_order_tensor_equation(
    label: &str,
    expr: &Expr,
    second_order_symbols: &[lasso::Spur],
    first_order_symbols: &[lasso::Spur],
    interner: &Interner,
) -> Result<SecondOrderTensorEquationSplit, CosmologyError> {
    let full = simplify_linearized_expr(expr.clone(), interner);
    let mut linear_terms = Vec::new();
    let mut quadratic_terms = Vec::new();
    for term in additive_terms(&full) {
        let second_degree = count_perturbation_degree(&term, second_order_symbols, interner);
        let first_degree = count_perturbation_degree(&term, first_order_symbols, interner);
        if second_degree == 1 && first_degree == 0 {
            linear_terms.push(term);
        } else if second_degree == 0 && first_degree >= 2 {
            quadratic_terms.push(term);
        } else {
            return Err(CosmologyError::UnclassifiedSecondOrderTensorTerm {
                label: label.to_string(),
                rendered: ax_ir::pretty_print(&term, interner),
            });
        }
    }
    let linear_second_order = Expr::add(linear_terms);
    let quadratic_source = Expr::add(quadratic_terms);
    Ok(SecondOrderTensorEquationSplit {
        label: label.to_string(),
        full: Expr::add(vec![linear_second_order.clone(), quadratic_source.clone()]),
        linear_second_order,
        quadratic_source,
    })
}

fn vector_derivation_symbols(modes: &SecondOrderVectorModes) -> Vec<lasso::Spur> {
    vec![
        modes.s1_x, modes.s1_y, modes.s1_z, modes.f1_x, modes.f1_y, modes.f1_z, modes.s2_x,
        modes.s2_y, modes.s2_z, modes.f2_x, modes.f2_y, modes.f2_z,
    ]
}

fn vector_first_order_symbols(modes: &SecondOrderVectorModes) -> Vec<lasso::Spur> {
    vec![
        modes.s1_x, modes.s1_y, modes.s1_z, modes.f1_x, modes.f1_y, modes.f1_z,
    ]
}

fn tensor_derivation_symbols(modes: &SecondOrderTensorModes) -> Vec<lasso::Spur> {
    vec![
        modes.h1_xx,
        modes.h1_xy,
        modes.h1_xz,
        modes.h1_yy,
        modes.h1_yz,
        modes.h1_zz,
        modes.h2_xx,
        modes.h2_xy,
        modes.h2_xz,
        modes.h2_yy,
        modes.h2_yz,
        modes.h2_zz,
    ]
}

fn tensor_first_order_symbols(modes: &SecondOrderTensorModes) -> Vec<lasso::Spur> {
    vec![
        modes.h1_xx,
        modes.h1_xy,
        modes.h1_xz,
        modes.h1_yy,
        modes.h1_yz,
        modes.h1_zz,
    ]
}

fn lifted_symbols_vector(
    modes: &SecondOrderVectorModes,
    generator: &SecondOrderVectorGaugeGenerator,
) -> Vec<lasso::Spur> {
    let mut symbols = vector_derivation_symbols(modes);
    symbols.extend([
        generator.lvec1_x,
        generator.lvec1_y,
        generator.lvec1_z,
        generator.lvec2_x,
        generator.lvec2_y,
        generator.lvec2_z,
    ]);
    symbols
}

fn lifted_symbols_tensor(
    modes: &SecondOrderTensorModes,
    generator: &SecondOrderVectorGaugeGenerator,
) -> Vec<lasso::Spur> {
    let mut symbols = tensor_derivation_symbols(modes);
    symbols.extend([
        generator.lvec1_x,
        generator.lvec1_y,
        generator.lvec1_z,
        generator.lvec2_x,
        generator.lvec2_y,
        generator.lvec2_z,
    ]);
    symbols
}

fn add_matrices(lhs: &SymbolicMatrix, rhs: &SymbolicMatrix, interner: &Interner) -> SymbolicMatrix {
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

fn a_squared(bg: &FrwBackgroundSpec) -> Expr {
    Expr::pow(Expr::Sym(bg.scale_factor), int(2))
}

fn laplacian(expr: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("laplacian"), vec![expr])
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

fn substitute_matrix_many(
    matrix: &SymbolicMatrix,
    replacements: &[(lasso::Spur, Expr)],
) -> SymbolicMatrix {
    let mut out = SymbolicMatrix::new(matrix.dim);
    for row in 0..matrix.dim {
        for col in 0..matrix.dim {
            out.set(
                row,
                col,
                substitute_many_expr(matrix.get(row, col), replacements),
            );
        }
    }
    out
}

fn substitute_many_expr(expr: &Expr, replacements: &[(lasso::Spur, Expr)]) -> Expr {
    replacements
        .iter()
        .fold(expr.clone(), |acc, (sym, replacement)| {
            substitute_expr(&acc, *sym, replacement)
        })
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

fn lift_matrix_for_derivation(
    matrix: &SymbolicMatrix,
    bg: &FrwBackgroundSpec,
    chart: &FrwCoordinateChart,
    lifted_symbols: &[lasso::Spur],
    interner: &Interner,
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
    interner: &Interner,
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
    interner: &Interner,
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
    interner: &Interner,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    struct CachedVectorData {
        interner: Interner,
        system: SecondOrderVectorSystem,
    }

    struct CachedTensorData {
        interner: Interner,
        system: SecondOrderTensorSystem,
    }

    static VECTOR_SYSTEM_CACHE: OnceLock<CachedVectorData> = OnceLock::new();
    static TENSOR_SYSTEM_CACHE: OnceLock<CachedTensorData> = OnceLock::new();

    fn default_bg(interner: &Interner) -> FrwBackgroundSpec {
        FrwBackgroundSpec::default_flat_conformal(interner)
    }

    fn render(expr: &Expr, interner: &Interner) -> String {
        ax_ir::pretty_print(expr, interner)
    }

    fn zero_out(expr: &Expr, symbols: &[lasso::Spur], interner: &Interner) -> Expr {
        simplify_linearized_expr(
            symbols.iter().fold(expr.clone(), |acc, sym| {
                substitute_expr(&acc, *sym, &Expr::zero())
            }),
            interner,
        )
    }

    fn cached_vector_system() -> &'static CachedVectorData {
        VECTOR_SYSTEM_CACHE.get_or_init(|| {
            let interner = Interner::new();
            let system = derive_second_order_vector_system(&default_bg(&interner), &interner)
                .unwrap_or_else(|err| panic!("{err:?}"));
            CachedVectorData { interner, system }
        })
    }

    fn cached_tensor_system() -> &'static CachedTensorData {
        TENSOR_SYSTEM_CACHE.get_or_init(|| {
            let interner = Interner::new();
            let system = derive_second_order_tensor_system(&default_bg(&interner), &interner)
                .unwrap_or_else(|err| panic!("{err:?}"));
            CachedTensorData { interner, system }
        })
    }

    #[test]
    fn default_second_order_vector_modes_use_expected_names() {
        let interner = Interner::new();
        let modes = default_second_order_vector_modes(&interner);
        assert_eq!(interner.resolve(modes.s1_x), "S1_x");
        assert_eq!(interner.resolve(modes.s1_y), "S1_y");
        assert_eq!(interner.resolve(modes.s1_z), "S1_z");
        assert_eq!(interner.resolve(modes.f1_x), "F1_x");
        assert_eq!(interner.resolve(modes.f1_y), "F1_y");
        assert_eq!(interner.resolve(modes.f1_z), "F1_z");
        assert_eq!(interner.resolve(modes.s2_x), "S2_x");
        assert_eq!(interner.resolve(modes.s2_y), "S2_y");
        assert_eq!(interner.resolve(modes.s2_z), "S2_z");
        assert_eq!(interner.resolve(modes.f2_x), "F2_x");
        assert_eq!(interner.resolve(modes.f2_y), "F2_y");
        assert_eq!(interner.resolve(modes.f2_z), "F2_z");
    }

    #[test]
    fn default_second_order_tensor_modes_use_expected_names() {
        let interner = Interner::new();
        let modes = default_second_order_tensor_modes(&interner);
        assert_eq!(interner.resolve(modes.h1_xx), "h1_xx");
        assert_eq!(interner.resolve(modes.h1_xy), "h1_xy");
        assert_eq!(interner.resolve(modes.h1_xz), "h1_xz");
        assert_eq!(interner.resolve(modes.h1_yy), "h1_yy");
        assert_eq!(interner.resolve(modes.h1_yz), "h1_yz");
        assert_eq!(interner.resolve(modes.h1_zz), "h1_zz");
        assert_eq!(interner.resolve(modes.h2_xx), "h2_xx");
        assert_eq!(interner.resolve(modes.h2_xy), "h2_xy");
        assert_eq!(interner.resolve(modes.h2_xz), "h2_xz");
        assert_eq!(interner.resolve(modes.h2_yy), "h2_yy");
        assert_eq!(interner.resolve(modes.h2_yz), "h2_yz");
        assert_eq!(interner.resolve(modes.h2_zz), "h2_zz");
    }

    #[test]
    fn default_second_order_vector_generator_use_expected_names() {
        let interner = Interner::new();
        let generator = default_second_order_vector_generator(&interner);
        assert_eq!(interner.resolve(generator.lvec1_x), "Lvec1_x");
        assert_eq!(interner.resolve(generator.lvec1_y), "Lvec1_y");
        assert_eq!(interner.resolve(generator.lvec1_z), "Lvec1_z");
        assert_eq!(interner.resolve(generator.lvec2_x), "Lvec2_x");
        assert_eq!(interner.resolve(generator.lvec2_y), "Lvec2_y");
        assert_eq!(interner.resolve(generator.lvec2_z), "Lvec2_z");
    }

    #[test]
    fn second_order_vector_first_order_limit_matches_prompt9_vector_gauge_laws() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let generator = default_second_order_vector_generator(&interner);
        let variation = second_order_vector_gauge_variation(&bg, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));

        assert_eq!(
            render(&variation.delta_s1_x, &interner),
            render(
                &Expr::neg(diff(
                    Expr::Sym(generator.lvec1_x),
                    bg.conformal_time,
                    &interner
                )),
                &interner
            )
        );
        assert_eq!(
            render(&variation.delta_s1_y, &interner),
            render(
                &Expr::neg(diff(
                    Expr::Sym(generator.lvec1_y),
                    bg.conformal_time,
                    &interner
                )),
                &interner
            )
        );
        assert_eq!(
            render(&variation.delta_s1_z, &interner),
            render(
                &Expr::neg(diff(
                    Expr::Sym(generator.lvec1_z),
                    bg.conformal_time,
                    &interner
                )),
                &interner
            )
        );
        assert_eq!(
            render(&variation.delta_f1_x, &interner),
            render(&Expr::neg(Expr::Sym(generator.lvec1_x)), &interner)
        );
        assert_eq!(
            render(&variation.delta_f1_y, &interner),
            render(&Expr::neg(Expr::Sym(generator.lvec1_y)), &interner)
        );
        assert_eq!(
            render(&variation.delta_f1_z, &interner),
            render(&Expr::neg(Expr::Sym(generator.lvec1_z)), &interner)
        );
    }

    #[test]
    fn second_order_tensor_first_order_limit_is_gauge_invariant_under_vector_generator_in_flat_case(
    ) {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let variation = second_order_tensor_gauge_variation(&bg, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));

        assert_eq!(variation.delta_h1_xx, Expr::zero());
        assert_eq!(variation.delta_h1_xy, Expr::zero());
        assert_eq!(variation.delta_h1_xz, Expr::zero());
        assert_eq!(variation.delta_h1_yy, Expr::zero());
        assert_eq!(variation.delta_h1_yz, Expr::zero());
        assert_eq!(variation.delta_h1_zz, Expr::zero());
    }

    #[test]
    fn derive_second_order_vector_system_returns_expected_labels() {
        let system = &cached_vector_system().system;
        let labels = system
            .equations
            .iter()
            .map(|eq| eq.label.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "second_order_vector_x".to_string(),
                "second_order_vector_y".to_string(),
                "second_order_vector_z".to_string()
            ]
        );
    }

    #[test]
    fn derive_second_order_tensor_system_returns_expected_labels() {
        let system = &cached_tensor_system().system;
        let labels = system
            .equations
            .iter()
            .map(|eq| eq.label.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "second_order_tensor_xx".to_string(),
                "second_order_tensor_xy".to_string(),
                "second_order_tensor_xz".to_string(),
                "second_order_tensor_yy".to_string(),
                "second_order_tensor_yz".to_string(),
                "second_order_tensor_zz".to_string()
            ]
        );
    }

    #[test]
    fn second_order_vector_source_split_reconstructs_full_equations() {
        let cached = cached_vector_system();
        for equation in &cached.system.equations {
            assert_eq!(
                simplify_linearized_expr(
                    Expr::add(vec![
                        equation.linear_second_order.clone(),
                        equation.quadratic_source.clone()
                    ]),
                    &cached.interner
                ),
                simplify_linearized_expr(equation.full.clone(), &cached.interner)
            );
        }
    }

    #[test]
    fn second_order_tensor_source_split_reconstructs_full_equations() {
        let cached = cached_tensor_system();
        for equation in &cached.system.equations {
            assert_eq!(
                simplify_linearized_expr(
                    Expr::add(vec![
                        equation.linear_second_order.clone(),
                        equation.quadratic_source.clone()
                    ]),
                    &cached.interner
                ),
                simplify_linearized_expr(equation.full.clone(), &cached.interner)
            );
        }
    }

    #[test]
    fn quadratic_vector_sources_vanish_when_first_order_vector_modes_are_zero() {
        let cached = cached_vector_system();
        let modes = default_second_order_vector_modes(&cached.interner);
        let first_order = [
            modes.s1_x, modes.s1_y, modes.s1_z, modes.f1_x, modes.f1_y, modes.f1_z,
        ];
        for equation in &cached.system.equations {
            assert_eq!(
                zero_out(&equation.quadratic_source, &first_order, &cached.interner),
                Expr::zero()
            );
        }
    }

    #[test]
    fn quadratic_tensor_sources_vanish_when_first_order_tensor_modes_are_zero() {
        let cached = cached_tensor_system();
        let modes = default_second_order_tensor_modes(&cached.interner);
        let first_order = [
            modes.h1_xx,
            modes.h1_xy,
            modes.h1_xz,
            modes.h1_yy,
            modes.h1_yz,
            modes.h1_zz,
        ];
        for equation in &cached.system.equations {
            assert_eq!(
                zero_out(&equation.quadratic_source, &first_order, &cached.interner),
                Expr::zero()
            );
        }
    }

    #[test]
    fn project_second_order_vector_to_harmonics_removes_explicit_spatial_derivatives() {
        let cached = cached_vector_system();
        let bg = default_bg(&cached.interner);
        let projected =
            project_second_order_vector_to_harmonics(&cached.system, &bg, &cached.interner)
                .unwrap_or_else(|err| panic!("{err:?}"));
        let rendered = ax_ir::pretty_print(
            &Expr::List(
                projected
                    .equations
                    .iter()
                    .map(|eq| eq.expr.clone())
                    .collect(),
            ),
            &cached.interner,
        );
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
        assert!(!rendered.contains("laplacian"), "got {rendered}");
    }

    #[test]
    fn project_second_order_tensor_to_harmonics_removes_explicit_spatial_derivatives() {
        let cached = cached_tensor_system();
        let bg = default_bg(&cached.interner);
        let projected =
            project_second_order_tensor_to_harmonics(&cached.system, &bg, &cached.interner)
                .unwrap_or_else(|err| panic!("{err:?}"));
        let rendered = ax_ir::pretty_print(
            &Expr::List(
                projected
                    .equations
                    .iter()
                    .map(|eq| eq.expr.clone())
                    .collect(),
            ),
            &cached.interner,
        );
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
        assert!(!rendered.contains("laplacian"), "got {rendered}");
    }
}
