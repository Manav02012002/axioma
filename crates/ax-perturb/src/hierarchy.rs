use crate::cosmology::{frw_background, mukhanov_sasaki_equation};
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HierarchyGauge {
    Newtonian,
    Synchronous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HierarchyClosure {
    PowerLaw,
    FreeStreaming,
    UserSymbolic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchySpec {
    pub l_max: usize,
    pub gauge: HierarchyGauge,
    pub closure: HierarchyClosure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchyVariable {
    pub name: lasso::Spur,
    pub ell: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchySystem {
    pub spec: HierarchySpec,
    pub variables: Vec<HierarchyVariable>,
    pub equations: Vec<(ax_ir::Expr, ax_ir::Expr)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityBenchmarkEntry {
    pub label: String,
    pub expected: String,
    pub actual: String,
    pub matched: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityBenchmarkReport {
    pub suite_name: String,
    pub entries: Vec<ParityBenchmarkEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalSolverHook {
    pub target: String,
    pub command_template: String,
    pub input_format: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
struct FixtureEntry {
    label: String,
    expected: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum FixturePayload {
    Entries(Vec<FixtureEntry>),
    Report {
        suite_name: Option<String>,
        entries: Vec<FixtureEntry>,
    },
}

pub fn hierarchy_spec(
    l_max: usize,
    gauge: HierarchyGauge,
    closure: HierarchyClosure,
) -> Result<HierarchySpec, crate::error::CosmologyError> {
    if l_max == 0 {
        return Err(CosmologyError::InvalidHierarchyOrder { got: l_max });
    }
    Ok(HierarchySpec {
        l_max,
        gauge,
        closure,
    })
}

pub fn neutrino_hierarchy_system(
    spec: &HierarchySpec,
    interner: &ax_ir::Interner,
) -> Result<HierarchySystem, crate::error::CosmologyError> {
    let eta = interner.get_or_intern("eta");
    let k = Expr::Sym(interner.get_or_intern("k"));
    let h = Expr::Sym(interner.get_or_intern("H"));
    let metric_symbols = metric_drivers(spec.gauge, interner);
    let variables = (0..=spec.l_max)
        .map(|ell| HierarchyVariable {
            name: interner.get_or_intern(&format!("F_nu_{ell}")),
            ell,
        })
        .collect::<Vec<_>>();

    let equations = variables
        .iter()
        .enumerate()
        .map(|(ell, variable)| {
            let lhs = diff(Expr::Sym(variable.name), eta, interner);
            let rhs = if ell == 0 {
                Expr::add(vec![
                    Expr::neg(Expr::mul(vec![k.clone(), Expr::Sym(variables[1].name)])),
                    metric_symbols.0.clone(),
                    Expr::mul(vec![h.clone(), Expr::Sym(variable.name)]),
                ])
            } else if ell == 1 {
                Expr::add(vec![
                    Expr::mul(vec![
                        rational(1, 3),
                        k.clone(),
                        Expr::add(vec![
                            Expr::Sym(variables[0].name),
                            Expr::neg(Expr::mul(vec![
                                int(2),
                                Expr::Sym(next_name(&variables, ell)),
                            ])),
                        ]),
                    ]),
                    metric_symbols.1.clone(),
                    Expr::neg(Expr::mul(vec![h.clone(), Expr::Sym(variable.name)])),
                ])
            } else if ell < spec.l_max {
                hierarchy_recurrence_rhs(
                    variable.name,
                    ell,
                    variables[ell - 1].name,
                    variables[ell + 1].name,
                    &k,
                    &h,
                )
            } else {
                neutrino_last_rhs(spec, &variables, &k, &h, interner)?
            };
            Ok((lhs, rhs))
        })
        .collect::<Result<Vec<_>, CosmologyError>>()?;

    Ok(HierarchySystem {
        spec: spec.clone(),
        variables,
        equations,
    })
}

pub fn photon_hierarchy_system(
    spec: &HierarchySpec,
    interner: &ax_ir::Interner,
) -> Result<HierarchySystem, crate::error::CosmologyError> {
    let eta = interner.get_or_intern("eta");
    let k = Expr::Sym(interner.get_or_intern("k"));
    let h = Expr::Sym(interner.get_or_intern("H"));
    let tau_c = Expr::Sym(interner.get_or_intern("tau_c"));
    let tau_inv = Expr::pow(tau_c, int(-1));
    let metric_symbols = metric_drivers(spec.gauge, interner);
    let variables = (0..=spec.l_max)
        .map(|ell| HierarchyVariable {
            name: interner.get_or_intern(&format!("F_gamma_{ell}")),
            ell,
        })
        .collect::<Vec<_>>();

    let equations = variables
        .iter()
        .enumerate()
        .map(|(ell, variable)| {
            let lhs = diff(Expr::Sym(variable.name), eta, interner);
            let rhs = if ell == 0 {
                Expr::add(vec![
                    Expr::neg(Expr::mul(vec![k.clone(), Expr::Sym(variables[1].name)])),
                    metric_symbols.0.clone(),
                    Expr::mul(vec![h.clone(), Expr::Sym(variable.name)]),
                    Expr::neg(Expr::mul(vec![tau_inv.clone(), Expr::Sym(variable.name)])),
                ])
            } else if ell == 1 {
                Expr::add(vec![
                    Expr::mul(vec![
                        rational(1, 3),
                        k.clone(),
                        Expr::add(vec![
                            Expr::Sym(variables[0].name),
                            Expr::neg(Expr::mul(vec![
                                int(2),
                                Expr::Sym(next_name(&variables, ell)),
                            ])),
                        ]),
                    ]),
                    metric_symbols.1.clone(),
                    Expr::neg(Expr::mul(vec![h.clone(), Expr::Sym(variable.name)])),
                    Expr::neg(Expr::mul(vec![tau_inv.clone(), Expr::Sym(variable.name)])),
                ])
            } else if ell < spec.l_max {
                Expr::add(vec![
                    hierarchy_recurrence_rhs(
                        variable.name,
                        ell,
                        variables[ell - 1].name,
                        variables[ell + 1].name,
                        &k,
                        &h,
                    ),
                    Expr::neg(Expr::mul(vec![tau_inv.clone(), Expr::Sym(variable.name)])),
                ])
            } else {
                Expr::add(vec![
                    photon_last_rhs(spec, &variables, &k, &h, interner)?,
                    Expr::neg(Expr::mul(vec![tau_inv.clone(), Expr::Sym(variable.name)])),
                ])
            };
            Ok((lhs, rhs))
        })
        .collect::<Result<Vec<_>, CosmologyError>>()?;

    Ok(HierarchySystem {
        spec: spec.clone(),
        variables,
        equations,
    })
}

pub fn export_hierarchy_system(
    target: &str,
    system: &HierarchySystem,
    interner: &ax_ir::Interner,
) -> Result<String, crate::error::CosmologyError> {
    let eta = interner.get_or_intern("eta");
    let k = interner.get_or_intern("k");
    let h = interner.get_or_intern("H");
    let metric = match system.spec.gauge {
        HierarchyGauge::Newtonian => {
            vec![interner.get_or_intern("Phi"), interner.get_or_intern("Psi")]
        }
        HierarchyGauge::Synchronous => vec![
            interner.get_or_intern("h_sync"),
            interner.get_or_intern("eta_sync"),
        ],
    };
    let mut args = vec![eta, k, h];
    args.extend(metric);
    args.extend(system.variables.iter().map(|variable| variable.name));

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
        "json" => export_hierarchy_json("json", system, interner),
        "class_hook" => export_hook_payload("class", system, interner),
        "camb_hook" => export_hook_payload("camb", system, interner),
        _ => Err(CosmologyError::UnsupportedExternalSolverHook {
            target: target.to_string(),
        }),
    }
}

pub fn benchmark_report_against_fixture(
    suite_name: &str,
    labels_and_actuals: &[(String, String)],
    fixture_json: &str,
) -> Result<ParityBenchmarkReport, crate::error::CosmologyError> {
    let payload: FixturePayload = serde_json::from_str(fixture_json).map_err(|_| {
        CosmologyError::ParityFixtureValidationFailure {
            fixture: suite_name.to_string(),
        }
    })?;
    let entries = match payload {
        FixturePayload::Entries(entries) => entries,
        FixturePayload::Report { entries, .. } => entries,
    };

    let actual_map = labels_and_actuals
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeMap::new();
    let mut report_entries = entries
        .into_iter()
        .map(|entry| {
            let actual = actual_map.get(&entry.label).cloned().unwrap_or_default();
            seen.insert(entry.label.clone(), ());
            ParityBenchmarkEntry {
                label: entry.label,
                expected: entry.expected.clone(),
                actual: actual.clone(),
                matched: actual == entry.expected,
            }
        })
        .collect::<Vec<_>>();

    for (label, actual) in labels_and_actuals {
        if !seen.contains_key(label) {
            report_entries.push(ParityBenchmarkEntry {
                label: label.clone(),
                expected: String::new(),
                actual: actual.clone(),
                matched: false,
            });
        }
    }

    Ok(ParityBenchmarkReport {
        suite_name: suite_name.to_string(),
        entries: report_entries,
    })
}

pub fn built_in_parity_reports(
    interner: &ax_ir::Interner,
) -> Result<Vec<ParityBenchmarkReport>, crate::error::CosmologyError> {
    let bg = frw_background(interner);
    let decomp = crate::gauge::svt_decompose_perturbation(3, interner).map_err(|_| {
        CosmologyError::ParityFixtureValidationFailure {
            fixture: "ma_bertschinger_scalar_labels".to_string(),
        }
    })?;
    let scalar_actuals = crate::cosmology::linearized_einstein_scalar(&bg, &decomp, interner)?
        .into_iter()
        .map(|equation| (equation.label.clone(), equation.label))
        .collect::<Vec<_>>();
    let tensor_actuals = crate::cosmology::tensor_mode_equation(&bg, interner)?
        .into_iter()
        .map(|expr| {
            let name = interner.resolve(expr.name).to_string();
            (name.clone(), name)
        })
        .collect::<Vec<_>>();
    let symbols = crate::standard_canonical_scalar_symbols(interner);
    let mukhanov_actual = mukhanov_sasaki_equation(&bg, symbols.slow_roll_epsilon, interner)?;
    let mukhanov_actuals = vec![(
        "mukhanov_sasaki".to_string(),
        ax_ir::pretty_print(&mukhanov_actual, interner),
    )];
    let bridge_actuals = crate::symbolic_boltzmann_bridge_system(&bg, interner)?
        .variables
        .into_iter()
        .map(|sym| {
            let name = interner.resolve(sym).to_string();
            (name.clone(), name)
        })
        .collect::<Vec<_>>();

    Ok(vec![
        benchmark_report_against_fixture(
            "ma_bertschinger_scalar_labels",
            &scalar_actuals,
            include_str!(
                "../../../tests/fixtures/cosmology/parity/ma_bertschinger_scalar_labels.json"
            ),
        )?,
        benchmark_report_against_fixture(
            "tensor_mode_labels",
            &tensor_actuals,
            include_str!("../../../tests/fixtures/cosmology/parity/tensor_mode_labels.json"),
        )?,
        benchmark_report_against_fixture(
            "mukhanov_sasaki_forms",
            &mukhanov_actuals,
            include_str!("../../../tests/fixtures/cosmology/parity/mukhanov_sasaki_forms.json"),
        )?,
        benchmark_report_against_fixture(
            "boltzmann_bridge_labels",
            &bridge_actuals,
            include_str!("../../../tests/fixtures/cosmology/parity/boltzmann_bridge_labels.json"),
        )?,
    ])
}

pub fn default_external_solver_hooks() -> Vec<ExternalSolverHook> {
    vec![
        ExternalSolverHook {
            target: "class".to_string(),
            command_template: "class --input {input_json}".to_string(),
            input_format: "json".to_string(),
        },
        ExternalSolverHook {
            target: "camb".to_string(),
            command_template: "camb {input_json}".to_string(),
            input_format: "json".to_string(),
        },
    ]
}

fn hierarchy_recurrence_rhs(
    name: lasso::Spur,
    ell: usize,
    prev: lasso::Spur,
    next: lasso::Spur,
    k: &Expr,
    h: &Expr,
) -> Expr {
    Expr::add(vec![
        Expr::mul(vec![
            Expr::pow(int(2 * ell as i64 + 1), int(-1)),
            k.clone(),
            Expr::add(vec![
                Expr::mul(vec![int(ell as i64), Expr::Sym(prev)]),
                Expr::neg(Expr::mul(vec![int(ell as i64 + 1), Expr::Sym(next)])),
            ]),
        ]),
        Expr::neg(Expr::mul(vec![h.clone(), Expr::Sym(name)])),
    ])
}

fn neutrino_last_rhs(
    spec: &HierarchySpec,
    variables: &[HierarchyVariable],
    k: &Expr,
    h: &Expr,
    interner: &Interner,
) -> Result<Expr, CosmologyError> {
    let ell = spec.l_max;
    let current = Expr::Sym(variables[ell].name);
    let prev = Expr::Sym(variables[ell - 1].name);
    let closure = match spec.closure {
        HierarchyClosure::PowerLaw => Expr::mul(vec![
            rational(ell as i64, ell as i64 + 1),
            Expr::Sym(variables[ell].name),
            int(spec.l_max as i64),
        ]),
        HierarchyClosure::FreeStreaming => Expr::mul(vec![
            rational(spec.l_max as i64 + 1, 2 * spec.l_max as i64 + 1),
            Expr::Sym(variables[ell].name),
        ]),
        HierarchyClosure::UserSymbolic => Expr::Sym(interner.get_or_intern("Closure_nu")),
    };
    Ok(Expr::add(vec![
        Expr::mul(vec![
            Expr::pow(int(2 * ell as i64 + 1), int(-1)),
            k.clone(),
            Expr::add(vec![
                Expr::mul(vec![int(ell as i64), prev]),
                Expr::neg(Expr::mul(vec![int(ell as i64 + 1), closure])),
            ]),
        ]),
        Expr::neg(Expr::mul(vec![h.clone(), current])),
    ]))
}

fn photon_last_rhs(
    spec: &HierarchySpec,
    variables: &[HierarchyVariable],
    k: &Expr,
    h: &Expr,
    interner: &Interner,
) -> Result<Expr, CosmologyError> {
    let ell = spec.l_max;
    let current = Expr::Sym(variables[ell].name);
    let prev = Expr::Sym(variables[ell - 1].name);
    let closure = match spec.closure {
        HierarchyClosure::PowerLaw => Expr::mul(vec![
            rational(ell as i64, ell as i64 + 1),
            Expr::Sym(variables[ell].name),
            int(spec.l_max as i64),
        ]),
        HierarchyClosure::FreeStreaming => Expr::mul(vec![
            rational(spec.l_max as i64 + 1, 2 * spec.l_max as i64 + 1),
            Expr::Sym(variables[ell].name),
        ]),
        HierarchyClosure::UserSymbolic => Expr::Sym(interner.get_or_intern("Closure_gamma")),
    };
    Ok(Expr::add(vec![
        Expr::mul(vec![
            Expr::pow(int(2 * ell as i64 + 1), int(-1)),
            k.clone(),
            Expr::add(vec![
                Expr::mul(vec![int(ell as i64), prev]),
                Expr::neg(Expr::mul(vec![int(ell as i64 + 1), closure])),
            ]),
        ]),
        Expr::neg(Expr::mul(vec![h.clone(), current])),
    ]))
}

fn export_hierarchy_json(
    kind: &str,
    system: &HierarchySystem,
    interner: &Interner,
) -> Result<String, CosmologyError> {
    let payload = serde_json::json!({
        "kind": kind,
        "gauge": gauge_name(system.spec.gauge),
        "closure": closure_name(system.spec.closure),
        "l_max": system.spec.l_max,
        "variables": system.variables.iter().map(|variable| interner.resolve(variable.name).to_string()).collect::<Vec<_>>(),
        "equations": system.equations.iter().map(|(lhs, rhs)| serde_json::json!({
            "lhs": ax_ir::pretty_print(lhs, interner),
            "rhs": ax_ir::pretty_print(rhs, interner),
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&payload).map_err(|_| {
        CosmologyError::UnsupportedExternalSolverHook {
            target: kind.to_string(),
        }
    })
}

fn export_hook_payload(
    target: &str,
    system: &HierarchySystem,
    interner: &Interner,
) -> Result<String, CosmologyError> {
    let species = system
        .variables
        .first()
        .map(|variable| interner.resolve(variable.name).to_string())
        .unwrap_or_default();
    let payload = serde_json::json!({
        "target": target,
        "species": if species.starts_with("F_gamma") { "photon" } else { "neutrino" },
        "gauge": gauge_name(system.spec.gauge),
        "closure": closure_name(system.spec.closure),
        "variable_order": system.variables.iter().map(|variable| interner.resolve(variable.name).to_string()).collect::<Vec<_>>(),
        "rhs": system.equations.iter().map(|(lhs, rhs)| serde_json::json!({
            "lhs": ax_ir::pretty_print(lhs, interner),
            "rhs": ax_ir::pretty_print(rhs, interner),
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|_| CosmologyError::UnsupportedExternalSolverHook {
        target: target.to_string(),
    })
}

fn next_name(variables: &[HierarchyVariable], ell: usize) -> lasso::Spur {
    variables[ell + 1].name
}

fn metric_drivers(gauge: HierarchyGauge, interner: &Interner) -> (Expr, Expr) {
    match gauge {
        HierarchyGauge::Newtonian => (
            Expr::Sym(interner.get_or_intern("Phi")),
            Expr::Sym(interner.get_or_intern("Psi")),
        ),
        HierarchyGauge::Synchronous => (
            Expr::Sym(interner.get_or_intern("h_sync")),
            Expr::Sym(interner.get_or_intern("eta_sync")),
        ),
    }
}

fn gauge_name(gauge: HierarchyGauge) -> &'static str {
    match gauge {
        HierarchyGauge::Newtonian => "newtonian",
        HierarchyGauge::Synchronous => "synchronous",
    }
}

fn closure_name(closure: HierarchyClosure) -> &'static str {
    match closure {
        HierarchyClosure::PowerLaw => "power_law",
        HierarchyClosure::FreeStreaming => "free_streaming",
        HierarchyClosure::UserSymbolic => "user_symbolic",
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

    #[test]
    fn hierarchy_spec_rejects_zero_lmax() {
        let result = hierarchy_spec(0, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw);
        match result {
            Err(CosmologyError::InvalidHierarchyOrder { got }) => assert_eq!(got, 0),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn neutrino_hierarchy_system_has_lmax_plus_one_variables() {
        let interner = Interner::new();
        let spec = hierarchy_spec(3, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert_eq!(system.variables.len(), 4);
        assert_eq!(interner.resolve(system.variables[0].name), "F_nu_0");
    }

    #[test]
    fn photon_hierarchy_system_has_lmax_plus_one_variables() {
        let interner = Interner::new();
        let spec = hierarchy_spec(4, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = photon_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert_eq!(system.variables.len(), 5);
        assert_eq!(interner.resolve(system.variables[4].name), "F_gamma_4");
    }

    #[test]
    fn synchronous_neutrino_hierarchy_uses_synchronous_metric_symbols() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Synchronous, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = ax_ir::pretty_print(&system.equations[0].1, &interner);
        assert!(rendered.contains("h_sync"), "got {rendered}");
        assert!(
            rendered.contains("eta_sync")
                || ax_ir::pretty_print(&system.equations[1].1, &interner).contains("eta_sync")
        );
    }

    #[test]
    fn newtonian_neutrino_hierarchy_uses_newtonian_metric_symbols() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = ax_ir::pretty_print(&system.equations[0].1, &interner);
        assert!(rendered.contains("Phi"), "got {rendered}");
        assert!(
            rendered.contains("Psi")
                || ax_ir::pretty_print(&system.equations[1].1, &interner).contains("Psi")
        );
    }

    #[test]
    fn power_law_closure_affects_last_equation() {
        let interner = Interner::new();
        let spec = hierarchy_spec(3, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = ax_ir::pretty_print(&system.equations[3].1, &interner);
        assert!(rendered.contains("3"), "got {rendered}");
    }

    #[test]
    fn free_streaming_closure_affects_last_equation() {
        let interner = Interner::new();
        let spec = hierarchy_spec(
            3,
            HierarchyGauge::Newtonian,
            HierarchyClosure::FreeStreaming,
        )
        .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = ax_ir::pretty_print(&system.equations[3].1, &interner);
        assert!(
            rendered.contains("16/7") || rendered.contains("16 / 7"),
            "got {rendered}"
        );
    }

    #[test]
    fn user_symbolic_closure_uses_closure_symbol() {
        let interner = Interner::new();
        let spec = hierarchy_spec(3, HierarchyGauge::Newtonian, HierarchyClosure::UserSymbolic)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let rendered = ax_ir::pretty_print(&system.equations[3].1, &interner);
        assert!(rendered.contains("Closure_nu"), "got {rendered}");
    }

    #[test]
    fn export_hierarchy_system_python_contains_rhs_0() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let code = export_hierarchy_system("python", &system, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(code.contains("def rhs_0("), "{code}");
    }

    #[test]
    fn export_hierarchy_system_json_contains_variable_names() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = photon_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let json = export_hierarchy_system("json", &system, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(json.contains("\"F_gamma_0\""), "{json}");
    }

    #[test]
    fn export_hierarchy_system_class_hook_contains_target_class() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let payload = export_hierarchy_system("class_hook", &system, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(payload.contains("\"target\":\"class\""), "{payload}");
    }

    #[test]
    fn export_hierarchy_system_camb_hook_contains_target_camb() {
        let interner = Interner::new();
        let spec = hierarchy_spec(2, HierarchyGauge::Newtonian, HierarchyClosure::PowerLaw)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let system = neutrino_hierarchy_system(&spec, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        let payload = export_hierarchy_system("camb_hook", &system, &interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(payload.contains("\"target\":\"camb\""), "{payload}");
    }

    #[test]
    fn benchmark_report_against_fixture_matches_expected_entries() {
        let fixture = r#"{"entries":[{"label":"a","expected":"x"},{"label":"b","expected":"y"}]}"#;
        let report = benchmark_report_against_fixture(
            "demo",
            &[
                ("a".to_string(), "x".to_string()),
                ("b".to_string(), "z".to_string()),
            ],
            fixture,
        )
        .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert!(report.entries[0].matched);
        assert!(!report.entries[1].matched);
    }

    #[test]
    fn built_in_parity_reports_returns_four_suites() {
        let interner = Interner::new();
        let reports = built_in_parity_reports(&interner)
            .unwrap_or_else(|err| panic!("unexpected error: {err}"));
        assert_eq!(reports.len(), 4);
        assert_eq!(reports[0].suite_name, "ma_bertschinger_scalar_labels");
        assert_eq!(reports[3].suite_name, "boltzmann_bridge_labels");
    }
}
