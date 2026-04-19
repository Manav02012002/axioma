use crate::cosmology::require_conformal_time;
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoltzmannSpeciesSymbols {
    pub delta_cdm: lasso::Spur,
    pub theta_cdm: lasso::Spur,
    pub delta_b: lasso::Spur,
    pub theta_b: lasso::Spur,
    pub delta_gamma: lasso::Spur,
    pub theta_gamma: lasso::Spur,
    pub sigma_gamma: lasso::Spur,
    pub delta_nu: lasso::Spur,
    pub theta_nu: lasso::Spur,
    pub sigma_nu: lasso::Spur,
    pub phi_metric: lasso::Spur,
    pub psi_metric: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoltzmannBridgeSystem {
    pub variables: Vec<lasso::Spur>,
    pub equations: Vec<(ax_ir::Expr, ax_ir::Expr)>,
}

pub fn standard_boltzmann_species_symbols(interner: &ax_ir::Interner) -> BoltzmannSpeciesSymbols {
    BoltzmannSpeciesSymbols {
        delta_cdm: interner.get_or_intern("delta_cdm"),
        theta_cdm: interner.get_or_intern("theta_cdm"),
        delta_b: interner.get_or_intern("delta_b"),
        theta_b: interner.get_or_intern("theta_b"),
        delta_gamma: interner.get_or_intern("delta_gamma"),
        theta_gamma: interner.get_or_intern("theta_gamma"),
        sigma_gamma: interner.get_or_intern("sigma_gamma"),
        delta_nu: interner.get_or_intern("delta_nu"),
        theta_nu: interner.get_or_intern("theta_nu"),
        sigma_nu: interner.get_or_intern("sigma_nu"),
        phi_metric: interner.get_or_intern("Phi"),
        psi_metric: interner.get_or_intern("Psi"),
    }
}

pub fn symbolic_boltzmann_bridge_system(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<BoltzmannBridgeSystem, crate::error::CosmologyError> {
    require_conformal_time(bg, "symbolic_boltzmann_bridge_system")?;
    let symbols = standard_boltzmann_species_symbols(interner);
    let eta = bg.conformal_time;
    let k = Expr::Sym(interner.get_or_intern("k"));
    let k_sq = Expr::pow(k.clone(), int(2));
    let phi = Expr::Sym(symbols.phi_metric);
    let psi = Expr::Sym(symbols.psi_metric);
    let vars = vec![
        symbols.delta_cdm,
        symbols.theta_cdm,
        symbols.delta_b,
        symbols.theta_b,
        symbols.delta_gamma,
        symbols.theta_gamma,
        symbols.sigma_gamma,
        symbols.delta_nu,
        symbols.theta_nu,
        symbols.sigma_nu,
    ];
    let equations = vec![
        (
            diff(Expr::Sym(symbols.delta_cdm), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::Sym(symbols.theta_cdm)),
                Expr::mul(vec![k.clone(), phi.clone()]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.theta_cdm), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![k.clone(), Expr::Sym(symbols.theta_cdm)])),
                Expr::mul(vec![k_sq.clone(), psi.clone()]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.delta_b), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::Sym(symbols.theta_b)),
                Expr::mul(vec![k.clone(), phi.clone()]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.theta_b), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![k.clone(), Expr::Sym(symbols.theta_b)])),
                Expr::mul(vec![
                    k_sq.clone(),
                    Expr::add(vec![psi.clone(), phi.clone()]),
                ]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.delta_gamma), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![
                    rational(4, 3),
                    Expr::Sym(symbols.theta_gamma),
                ])),
                Expr::mul(vec![k.clone(), phi.clone()]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.theta_gamma), eta, interner),
            Expr::add(vec![
                Expr::mul(vec![
                    k_sq.clone(),
                    Expr::add(vec![
                        Expr::mul(vec![rational(1, 4), Expr::Sym(symbols.delta_gamma)]),
                        psi.clone(),
                    ]),
                ]),
                Expr::neg(Expr::mul(vec![
                    k_sq.clone(),
                    Expr::Sym(symbols.sigma_gamma),
                ])),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.sigma_gamma), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![
                    rational(4, 15),
                    Expr::Sym(symbols.theta_gamma),
                ])),
                Expr::mul(vec![k.clone(), Expr::add(vec![phi.clone(), psi.clone()])]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.delta_nu), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![rational(4, 3), Expr::Sym(symbols.theta_nu)])),
                Expr::mul(vec![k.clone(), phi.clone()]),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.theta_nu), eta, interner),
            Expr::add(vec![
                Expr::mul(vec![
                    k_sq.clone(),
                    Expr::add(vec![
                        Expr::mul(vec![rational(1, 4), Expr::Sym(symbols.delta_nu)]),
                        psi.clone(),
                    ]),
                ]),
                Expr::neg(Expr::mul(vec![k_sq, Expr::Sym(symbols.sigma_nu)])),
            ]),
        ),
        (
            diff(Expr::Sym(symbols.sigma_nu), eta, interner),
            Expr::add(vec![
                Expr::neg(Expr::mul(vec![
                    rational(4, 15),
                    Expr::Sym(symbols.theta_nu),
                ])),
                Expr::mul(vec![k, Expr::add(vec![phi, psi])]),
            ]),
        ),
    ];
    Ok(BoltzmannBridgeSystem {
        variables: vars,
        equations,
    })
}

pub fn export_boltzmann_bridge_system(
    target: &str,
    system: &BoltzmannBridgeSystem,
    interner: &ax_ir::Interner,
) -> Result<String, crate::error::CosmologyError> {
    let species = standard_boltzmann_species_symbols(interner);
    let eta = interner.get_or_intern("eta");
    let k = interner.get_or_intern("k");
    let args = {
        let mut values = vec![eta, k, species.phi_metric, species.psi_metric];
        values.extend(system.variables.iter().copied());
        values
    };
    match target {
        "python" => Ok(system
            .equations
            .iter()
            .enumerate()
            .map(|(idx, (_, rhs))| {
                ax_codegen::emit_python_function(&format!("rhs_{idx}"), &args, rhs, interner)
            })
            .collect::<Vec<_>>()
            .join("\n\n")),
        "rust" => Ok(system
            .equations
            .iter()
            .enumerate()
            .map(|(idx, (_, rhs))| {
                ax_codegen::emit_rust_function(&format!("rhs_{idx}"), &args, rhs, interner)
            })
            .collect::<Vec<_>>()
            .join("\n\n")),
        "cpp" => Ok(system
            .equations
            .iter()
            .enumerate()
            .map(|(idx, (_, rhs))| {
                ax_codegen::emit_cpp_function(&format!("rhs_{idx}"), &args, rhs, interner)
            })
            .collect::<Vec<_>>()
            .join("\n\n")),
        "json" => {
            let payload = serde_json::json!({
                "variables": system
                    .variables
                    .iter()
                    .map(|sym| interner.resolve(*sym).to_string())
                    .collect::<Vec<_>>(),
                "equations": system
                    .equations
                    .iter()
                    .map(|(lhs, rhs)| serde_json::json!({
                        "lhs": ax_ir::pretty_print(lhs, interner),
                        "rhs": ax_ir::pretty_print(rhs, interner),
                    }))
                    .collect::<Vec<_>>(),
            });
            serde_json::to_string_pretty(&payload).map_err(|_| {
                CosmologyError::UnsupportedBoltzmannExportTarget {
                    target: target.to_string(),
                }
            })
        }
        _ => Err(CosmologyError::UnsupportedBoltzmannExportTarget {
            target: target.to_string(),
        }),
    }
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
    use crate::FrwBackgroundSpec;

    #[test]
    fn standard_boltzmann_species_symbols_use_expected_names() {
        let interner = Interner::new();
        let symbols = standard_boltzmann_species_symbols(&interner);
        assert_eq!(interner.resolve(symbols.delta_cdm), "delta_cdm");
        assert_eq!(interner.resolve(symbols.theta_cdm), "theta_cdm");
        assert_eq!(interner.resolve(symbols.delta_b), "delta_b");
        assert_eq!(interner.resolve(symbols.theta_b), "theta_b");
        assert_eq!(interner.resolve(symbols.delta_gamma), "delta_gamma");
        assert_eq!(interner.resolve(symbols.theta_gamma), "theta_gamma");
        assert_eq!(interner.resolve(symbols.sigma_gamma), "sigma_gamma");
        assert_eq!(interner.resolve(symbols.delta_nu), "delta_nu");
        assert_eq!(interner.resolve(symbols.theta_nu), "theta_nu");
        assert_eq!(interner.resolve(symbols.sigma_nu), "sigma_nu");
        assert_eq!(interner.resolve(symbols.phi_metric), "Phi");
        assert_eq!(interner.resolve(symbols.psi_metric), "Psi");
    }

    #[test]
    fn symbolic_boltzmann_bridge_system_has_ten_variables_and_ten_equations() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();
        assert_eq!(system.variables.len(), 10);
        assert_eq!(system.equations.len(), 10);
    }

    #[test]
    fn export_boltzmann_bridge_system_json_contains_variable_names() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();
        let json = export_boltzmann_bridge_system("json", &system, &interner).unwrap();
        assert!(json.contains("\"delta_cdm\""), "{json}");
        assert!(json.contains("\"theta_nu\""), "{json}");
    }

    #[test]
    fn export_boltzmann_bridge_system_python_contains_rhs_0() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();
        let code = export_boltzmann_bridge_system("python", &system, &interner).unwrap();
        assert!(code.contains("def rhs_0("), "{code}");
    }

    #[test]
    fn export_boltzmann_bridge_system_rejects_unknown_target() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();
        let result = export_boltzmann_bridge_system("fortran", &system, &interner);
        match result {
            Err(CosmologyError::UnsupportedBoltzmannExportTarget { target }) => {
                assert_eq!(target, "fortran");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
