use crate::cosmology::require_conformal_time;
use crate::domain::{FrwBackgroundSpec, NamedEquation, NamedExpr, SectorKind};
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EftModelKind {
    Canonical,
    ReducedSoundSpeed,
    HorndeskiLike,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EftCoefficientSet {
    pub model: EftModelKind,
    pub q_s: lasso::Spur,
    pub c_s_sq: lasso::Spur,
    pub q_t: lasso::Spur,
    pub c_t_sq: lasso::Spur,
    pub alpha_b: Option<lasso::Spur>,
    pub alpha_k: Option<lasso::Spur>,
    pub alpha_m: Option<lasso::Spur>,
    pub alpha_t: Option<lasso::Spur>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EftQuadraticSector {
    pub scalar_lagrangian_density: ax_ir::Expr,
    pub tensor_lagrangian_density: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StabilityConditionSet {
    pub ghost_free_scalar: ax_ir::Expr,
    pub gradient_stable_scalar: ax_ir::Expr,
    pub ghost_free_tensor: ax_ir::Expr,
    pub gradient_stable_tensor: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EftModeEquations {
    pub scalar_equation: ax_ir::Expr,
    pub tensor_equation: ax_ir::Expr,
}

pub fn standard_eft_coefficients(
    model: EftModelKind,
    interner: &ax_ir::Interner,
) -> EftCoefficientSet {
    let alpha_b = interner.get_or_intern("alpha_B");
    let alpha_k = interner.get_or_intern("alpha_K");
    let alpha_m = interner.get_or_intern("alpha_M");
    let alpha_t = interner.get_or_intern("alpha_T");
    let alpha_fields = match model {
        EftModelKind::Canonical | EftModelKind::ReducedSoundSpeed => (None, None, None, None),
        EftModelKind::HorndeskiLike => (Some(alpha_b), Some(alpha_k), Some(alpha_m), Some(alpha_t)),
    };
    EftCoefficientSet {
        model,
        q_s: interner.get_or_intern("Q_s"),
        c_s_sq: interner.get_or_intern("c_s_sq"),
        q_t: interner.get_or_intern("Q_t"),
        c_t_sq: interner.get_or_intern("c_t_sq"),
        alpha_b: alpha_fields.0,
        alpha_k: alpha_fields.1,
        alpha_m: alpha_fields.2,
        alpha_t: alpha_fields.3,
    }
}

pub fn reduced_eft_quadratic_sector(
    bg: &crate::domain::FrwBackgroundSpec,
    coeffs: &EftCoefficientSet,
    interner: &ax_ir::Interner,
) -> Result<EftQuadraticSector, crate::error::CosmologyError> {
    validate_eft_background(bg, "reduced_eft_quadratic_sector")?;
    let a_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let r_eta = interner.get_or_intern("R_eta");
    let r_x = interner.get_or_intern("R_x");
    let r_y = interner.get_or_intern("R_y");
    let r_z = interner.get_or_intern("R_z");
    let h_eta = interner.get_or_intern("h_eta");
    let h_x = interner.get_or_intern("h_x");
    let h_y = interner.get_or_intern("h_y");
    let h_z = interner.get_or_intern("h_z");

    let scalar_lagrangian_density = Expr::mul(vec![
        a_sq.clone(),
        Expr::Sym(coeffs.q_s),
        rational(1, 2),
        Expr::add(vec![
            Expr::pow(Expr::Sym(r_eta), int(2)),
            Expr::neg(Expr::mul(vec![
                Expr::Sym(coeffs.c_s_sq),
                Expr::add(vec![
                    Expr::pow(Expr::Sym(r_x), int(2)),
                    Expr::pow(Expr::Sym(r_y), int(2)),
                    Expr::pow(Expr::Sym(r_z), int(2)),
                ]),
            ])),
        ]),
    ]);

    let tensor_lagrangian_density = Expr::mul(vec![
        a_sq,
        Expr::Sym(coeffs.q_t),
        rational(1, 8),
        Expr::add(vec![
            Expr::pow(Expr::Sym(h_eta), int(2)),
            Expr::neg(Expr::mul(vec![
                Expr::Sym(coeffs.c_t_sq),
                Expr::add(vec![
                    Expr::pow(Expr::Sym(h_x), int(2)),
                    Expr::pow(Expr::Sym(h_y), int(2)),
                    Expr::pow(Expr::Sym(h_z), int(2)),
                ]),
            ])),
        ]),
    ]);

    Ok(EftQuadraticSector {
        scalar_lagrangian_density,
        tensor_lagrangian_density,
    })
}

pub fn extract_stability_conditions(
    coeffs: &EftCoefficientSet,
    interner: &ax_ir::Interner,
) -> Result<StabilityConditionSet, crate::error::CosmologyError> {
    let gt = |symbol: lasso::Spur| {
        Expr::Call(
            interner.get_or_intern("gt"),
            vec![Expr::Sym(symbol), Expr::zero()],
        )
    };
    Ok(StabilityConditionSet {
        ghost_free_scalar: gt(coeffs.q_s),
        gradient_stable_scalar: gt(coeffs.c_s_sq),
        ghost_free_tensor: gt(coeffs.q_t),
        gradient_stable_tensor: gt(coeffs.c_t_sq),
    })
}

pub fn derive_eft_mode_equations(
    bg: &crate::domain::FrwBackgroundSpec,
    coeffs: &EftCoefficientSet,
    interner: &ax_ir::Interner,
) -> Result<EftModeEquations, crate::error::CosmologyError> {
    validate_eft_background(bg, "derive_eft_mode_equations")?;
    let eta = bg.conformal_time;
    let k = interner.get_or_intern("k");
    let r = interner.get_or_intern("R");
    let h_plus = interner.get_or_intern("h_plus");

    let scalar_equation = Expr::add(vec![
        diff(diff(Expr::Sym(r), eta, interner), eta, interner),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![int(2), Expr::Sym(bg.conformal_hubble)]),
                Expr::mul(vec![
                    diff(Expr::Sym(coeffs.q_s), eta, interner),
                    Expr::pow(Expr::Sym(coeffs.q_s), int(-1)),
                ]),
            ]),
            diff(Expr::Sym(r), eta, interner),
        ]),
        Expr::mul(vec![
            Expr::Sym(coeffs.c_s_sq),
            Expr::pow(Expr::Sym(k), int(2)),
            Expr::Sym(r),
        ]),
    ]);

    let tensor_equation = Expr::add(vec![
        diff(diff(Expr::Sym(h_plus), eta, interner), eta, interner),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![int(2), Expr::Sym(bg.conformal_hubble)]),
                Expr::mul(vec![
                    diff(Expr::Sym(coeffs.q_t), eta, interner),
                    Expr::pow(Expr::Sym(coeffs.q_t), int(-1)),
                ]),
            ]),
            diff(Expr::Sym(h_plus), eta, interner),
        ]),
        Expr::mul(vec![
            Expr::Sym(coeffs.c_t_sq),
            Expr::pow(Expr::Sym(k), int(2)),
            Expr::Sym(h_plus),
        ]),
    ]);

    Ok(EftModeEquations {
        scalar_equation,
        tensor_equation,
    })
}

pub fn export_eft_mode_rhs(
    target: &str,
    coeffs: &EftCoefficientSet,
    interner: &ax_ir::Interner,
) -> Result<String, crate::error::CosmologyError> {
    let eta = interner.get_or_intern("eta");
    let field = interner.get_or_intern("field");
    let field1 = interner.get_or_intern("field1");
    let k = interner.get_or_intern("k");
    let h = interner.get_or_intern("H");
    let q = interner.get_or_intern("Q");
    let c_sq = interner.get_or_intern("c_sq");
    let args = [eta, field, field1, k, h, q, c_sq];
    let friction = Expr::add(vec![
        Expr::mul(vec![int(2), Expr::Sym(h)]),
        Expr::mul(vec![
            diff(Expr::Sym(q), eta, interner),
            Expr::pow(Expr::Sym(q), int(-1)),
        ]),
    ]);
    let rhs = Expr::neg(Expr::add(vec![
        Expr::mul(vec![friction, Expr::Sym(field1)]),
        Expr::mul(vec![
            Expr::Sym(c_sq),
            Expr::pow(Expr::Sym(k), int(2)),
            Expr::Sym(field),
        ]),
    ]));

    let model_name = eft_model_name(coeffs.model);
    match target {
        "python" => Ok(format!(
            "{}\n\n{}",
            ax_codegen::emit_python_function("eft_scalar_rhs", &args, &rhs, interner),
            ax_codegen::emit_python_function("eft_tensor_rhs", &args, &rhs, interner)
        )),
        "rust" => Ok(format!(
            "{}\n\n{}",
            ax_codegen::emit_rust_function("eft_scalar_rhs", &args, &rhs, interner),
            ax_codegen::emit_rust_function("eft_tensor_rhs", &args, &rhs, interner)
        )),
        "cpp" => Ok(format!(
            "{}\n\n{}",
            ax_codegen::emit_cpp_function("eft_scalar_rhs", &args, &rhs, interner),
            ax_codegen::emit_cpp_function("eft_tensor_rhs", &args, &rhs, interner)
        )),
        "json" => {
            let equations = derive_eft_mode_equations(
                &FrwBackgroundSpec::default_flat_conformal(interner),
                coeffs,
                interner,
            )?;
            serde_json::to_string(&serde_json::json!({
                "model": model_name,
                "scalar_equation": ax_ir::pretty_print(&equations.scalar_equation, interner),
                "tensor_equation": ax_ir::pretty_print(&equations.tensor_equation, interner),
            }))
            .map_err(|_| CosmologyError::UnsupportedBoltzmannExportTarget {
                target: target.to_string(),
            })
        }
        _ => Err(CosmologyError::UnsupportedBoltzmannExportTarget {
            target: target.to_string(),
        }),
    }
}

pub fn eft_quadratic_sector_named(
    bg: &FrwBackgroundSpec,
    coeffs: &EftCoefficientSet,
    interner: &Interner,
) -> Result<Vec<NamedExpr>, CosmologyError> {
    let sector = reduced_eft_quadratic_sector(bg, coeffs, interner)?;
    Ok(vec![
        NamedExpr {
            name: interner.get_or_intern("eft_scalar_density"),
            expr: sector.scalar_lagrangian_density,
        },
        NamedExpr {
            name: interner.get_or_intern("eft_tensor_density"),
            expr: sector.tensor_lagrangian_density,
        },
    ])
}

pub fn eft_stability_named(
    coeffs: &EftCoefficientSet,
    interner: &Interner,
) -> Result<Vec<NamedExpr>, CosmologyError> {
    let conditions = extract_stability_conditions(coeffs, interner)?;
    Ok(vec![
        NamedExpr {
            name: interner.get_or_intern("ghost_free_scalar"),
            expr: conditions.ghost_free_scalar,
        },
        NamedExpr {
            name: interner.get_or_intern("gradient_stable_scalar"),
            expr: conditions.gradient_stable_scalar,
        },
        NamedExpr {
            name: interner.get_or_intern("ghost_free_tensor"),
            expr: conditions.ghost_free_tensor,
        },
        NamedExpr {
            name: interner.get_or_intern("gradient_stable_tensor"),
            expr: conditions.gradient_stable_tensor,
        },
    ])
}

pub fn eft_mode_equations_named(
    bg: &FrwBackgroundSpec,
    coeffs: &EftCoefficientSet,
    interner: &Interner,
) -> Result<Vec<NamedEquation>, CosmologyError> {
    let equations = derive_eft_mode_equations(bg, coeffs, interner)?;
    Ok(vec![
        NamedEquation {
            label: "eft_scalar_mode".to_string(),
            expr: equations.scalar_equation,
            order: 2,
            sector: SectorKind::Scalar,
        },
        NamedEquation {
            label: "eft_tensor_mode".to_string(),
            expr: equations.tensor_equation,
            order: 2,
            sector: SectorKind::Tensor,
        },
    ])
}

pub fn eft_model_name(model: EftModelKind) -> &'static str {
    match model {
        EftModelKind::Canonical => "canonical",
        EftModelKind::ReducedSoundSpeed => "reduced_sound_speed",
        EftModelKind::HorndeskiLike => "horndeski_like",
    }
}

fn validate_eft_background(bg: &FrwBackgroundSpec, operation: &str) -> Result<(), CosmologyError> {
    require_conformal_time(bg, operation)?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }
    Ok(())
}

fn diff(expr: Expr, var: lasso::Spur, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(num_rational::BigRational::new(num.into(), den.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bg(interner: &Interner) -> FrwBackgroundSpec {
        FrwBackgroundSpec::default_flat_conformal(interner)
    }

    #[test]
    fn standard_eft_coefficients_canonical_has_no_alpha_fields() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::Canonical, &interner);
        assert_eq!(coeffs.alpha_b, None);
        assert_eq!(coeffs.alpha_k, None);
        assert_eq!(coeffs.alpha_m, None);
        assert_eq!(coeffs.alpha_t, None);
    }

    #[test]
    fn standard_eft_coefficients_horndeski_like_has_all_alpha_fields() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::HorndeskiLike, &interner);
        assert_eq!(
            interner.resolve(coeffs.alpha_b.unwrap_or_default()),
            "alpha_B"
        );
        assert_eq!(
            interner.resolve(coeffs.alpha_k.unwrap_or_default()),
            "alpha_K"
        );
        assert_eq!(
            interner.resolve(coeffs.alpha_m.unwrap_or_default()),
            "alpha_M"
        );
        assert_eq!(
            interner.resolve(coeffs.alpha_t.unwrap_or_default()),
            "alpha_T"
        );
    }

    #[test]
    fn reduced_eft_quadratic_sector_contains_expected_q_and_c_symbols() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::ReducedSoundSpeed, &interner);
        let sector = reduced_eft_quadratic_sector(&default_bg(&interner), &coeffs, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let scalar = ax_ir::pretty_print(&sector.scalar_lagrangian_density, &interner);
        let tensor = ax_ir::pretty_print(&sector.tensor_lagrangian_density, &interner);
        assert!(scalar.contains("Q_s"), "got {scalar}");
        assert!(scalar.contains("c_s_sq"), "got {scalar}");
        assert!(tensor.contains("Q_t"), "got {tensor}");
        assert!(tensor.contains("c_t_sq"), "got {tensor}");
    }

    #[test]
    fn extract_stability_conditions_returns_four_conditions() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::Canonical, &interner);
        let conditions = extract_stability_conditions(&coeffs, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = [
            ax_ir::pretty_print(&conditions.ghost_free_scalar, &interner),
            ax_ir::pretty_print(&conditions.gradient_stable_scalar, &interner),
            ax_ir::pretty_print(&conditions.ghost_free_tensor, &interner),
            ax_ir::pretty_print(&conditions.gradient_stable_tensor, &interner),
        ]
        .join(" | ");
        assert!(rendered.contains("gt(Q_s, 0)"), "got {rendered}");
        assert!(rendered.contains("gt(c_s_sq, 0)"), "got {rendered}");
        assert!(rendered.contains("gt(Q_t, 0)"), "got {rendered}");
        assert!(rendered.contains("gt(c_t_sq, 0)"), "got {rendered}");
    }

    #[test]
    fn derive_eft_mode_equations_match_expected_forms() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::Canonical, &interner);
        let equations = derive_eft_mode_equations(&default_bg(&interner), &coeffs, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let scalar = ax_ir::pretty_print(&equations.scalar_equation, &interner);
        let tensor = ax_ir::pretty_print(&equations.tensor_equation, &interner);
        assert!(scalar.contains("diff(diff(R, eta), eta)"), "got {scalar}");
        assert!(scalar.contains("Q_s"), "got {scalar}");
        assert!(
            scalar.contains("c_s_sq*k^2*R") || scalar.contains("c_s_sq * k^2 * R"),
            "got {scalar}"
        );
        assert!(
            tensor.contains("diff(diff(h_plus, eta), eta)"),
            "got {tensor}"
        );
        assert!(tensor.contains("Q_t"), "got {tensor}");
        assert!(
            tensor.contains("c_t_sq*k^2*h_plus") || tensor.contains("c_t_sq * k^2 * h_plus"),
            "got {tensor}"
        );
    }

    #[test]
    fn export_eft_mode_rhs_python_contains_eft_scalar_rhs() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::Canonical, &interner);
        let code = export_eft_mode_rhs("python", &coeffs, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(code.contains("def eft_scalar_rhs("), "{code}");
        assert!(code.contains("def eft_tensor_rhs("), "{code}");
    }

    #[test]
    fn export_eft_mode_rhs_json_contains_model_kind() {
        let interner = Interner::new();
        let coeffs = standard_eft_coefficients(EftModelKind::HorndeskiLike, &interner);
        let json = export_eft_mode_rhs("json", &coeffs, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(json.contains("\"model\":\"horndeski_like\""), "{json}");
    }
}
