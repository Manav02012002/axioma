use ax_eval::property_store::*;
use ax_eval::{callable_entries, Env, EvalState};
use ax_ir::*;
use std::collections::HashMap;

struct TestState {
    interner: Interner,
    env: Env,
    exprs: HashMap<String, Expr>,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            env: Env::new(),
            exprs: HashMap::new(),
        }
    }
}

impl EvalState for TestState {
    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn interner_mut(&mut self) -> &mut Interner {
        &mut self.interner
    }

    fn env(&self) -> &Env {
        &self.env
    }

    fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    fn store_expr(&mut self, expr: Expr) -> String {
        let id = format!("expr{}", self.exprs.len());
        self.exprs.insert(id.clone(), expr);
        id
    }

    fn get_expr(&self, id: &str) -> Option<&Expr> {
        self.exprs.get(id)
    }

    fn parse_code(&mut self, code: &str) -> Result<Expr, String> {
        Err(format!(
            "parse_code is unavailable in TestState for input: {code}"
        ))
    }

    fn render_latex(&self, expr: &Expr) -> String {
        ax_ir::pretty_print(expr, &self.interner)
    }

    fn render_unicode(&self, expr: &Expr) -> String {
        ax_ir::pretty_print(expr, &self.interner)
    }

    fn get_metric(&self, _id: &str) -> Option<&(ax_tensor::SymbolicMatrix, Vec<lasso::Spur>)> {
        None
    }

    fn store_metric(
        &mut self,
        _id: String,
        _metric: ax_tensor::SymbolicMatrix,
        _coords: Vec<lasso::Spur>,
    ) {
    }

    fn get_christoffel(&self, _id: &str) -> Option<&Vec<Vec<Vec<Expr>>>> {
        None
    }

    fn store_christoffel(&mut self, _id: String, _chris: Vec<Vec<Vec<Expr>>>) {}

    fn get_riemann(&self, _id: &str) -> Option<&Vec<Vec<Vec<Vec<Expr>>>>> {
        None
    }

    fn store_riemann(&mut self, _id: String, _riem: Vec<Vec<Vec<Vec<Expr>>>>) {}

    fn get_ricci(&self, _id: &str) -> Option<&Vec<Vec<Expr>>> {
        None
    }

    fn store_ricci(&mut self, _id: String, _ric: Vec<Vec<Expr>>) {}

    fn get_matrix_data(&self, _id: &str) -> Option<Vec<Vec<Expr>>> {
        None
    }
}

fn call_registry(
    state: &mut TestState,
    name: &str,
    args: Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let entry = callable_entries()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("missing callable entry for {name}"));
    (entry.handler)(&args, state)
}

#[test]
fn evaluator_declare_gamma_convention_stores_gamma_convention_meta() {
    let mut state = TestState::default();
    let result = call_registry(
        &mut state,
        "declare_gamma5_convention",
        vec![
            serde_json::json!("gamma"),
            serde_json::json!("mostly_plus"),
            serde_json::json!("plus_two_g"),
            serde_json::json!("levi_civita"),
            serde_json::json!("epsilon"),
            serde_json::json!(4),
        ],
    )
    .expect("gamma convention declaration should succeed");
    assert_eq!(result["status"], "ok");

    let gamma = state.interner.get_or_intern("gamma");
    let epsilon = state.interner.get_or_intern("epsilon");
    let props = state.env.property_store.get_all(gamma);
    assert!(props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::GammaConventionMeta(GammaConventionMetadata {
                signature: MetricSignature::MostlyPlus,
                clifford: CliffordConvention::PlusTwoG,
                gamma5: Some(GammaFiveConvention::LeviCivita),
                epsilon_symbol: Some(sym),
                dimension: Some(4),
            }) if *sym == epsilon
        )
    }));
}

#[test]
fn evaluator_gamma_convention_invalid_signature_returns_exact_error() {
    let mut state = TestState::default();
    let err = call_registry(
        &mut state,
        "declare_gamma_convention",
        vec![
            serde_json::json!("gamma"),
            serde_json::json!("bad_signature"),
            serde_json::json!("plus_two_g"),
            serde_json::json!(4),
        ],
    )
    .expect_err("invalid signature should fail");

    assert_eq!(
        err,
        "gamma convention signature must be one of: mostly_plus, mostly_minus, euclidean"
    );
}

#[test]
fn evaluator_gamma_convention_invalid_clifford_returns_exact_error() {
    let mut state = TestState::default();
    let err = call_registry(
        &mut state,
        "declare_gamma_convention",
        vec![
            serde_json::json!("gamma"),
            serde_json::json!("mostly_plus"),
            serde_json::json!("bad_clifford"),
            serde_json::json!(4),
        ],
    )
    .expect_err("invalid clifford convention should fail");

    assert_eq!(
        err,
        "gamma convention clifford sign must be one of: plus_two_g, minus_two_g"
    );
}

#[test]
fn evaluator_gamma5_convention_invalid_kind_returns_exact_error() {
    let mut state = TestState::default();
    let err = call_registry(
        &mut state,
        "declare_gamma5_convention",
        vec![
            serde_json::json!("gamma"),
            serde_json::json!("mostly_plus"),
            serde_json::json!("plus_two_g"),
            serde_json::json!("bad_gamma5"),
            serde_json::json!("epsilon"),
            serde_json::json!(4),
        ],
    )
    .expect_err("invalid gamma5 convention should fail");

    assert_eq!(
        err,
        "gamma5 convention kind must be one of: levi_civita, abstract_chiral"
    );
}

#[test]
fn evaluator_gamma_convention_invalid_dimension_returns_exact_error() {
    let mut state = TestState::default();
    let err = call_registry(
        &mut state,
        "declare_gamma_convention",
        vec![
            serde_json::json!("gamma"),
            serde_json::json!("mostly_plus"),
            serde_json::json!("plus_two_g"),
            serde_json::json!(0),
        ],
    )
    .expect_err("non-positive dimension should fail");

    assert_eq!(err, "gamma convention dimension must be a positive integer");
}

#[test]
fn simple_declare_and_get() {
    let interner = Interner::new();
    let g = interner.get_or_intern("g");
    let mut store = PropertyStore::new();
    store.declare_simple(g, TensorProperty::Metric);
    store.declare_simple(g, TensorProperty::Symmetric(vec![0, 1]));
    let props = store.get_all(g);
    assert_eq!(props.len(), 2);
}

#[test]
fn pattern_match_with_indices() {
    let interner = Interner::new();
    let r = interner.get_or_intern("R");
    let mu = interner.get_or_intern("mu");
    let nu = interner.get_or_intern("nu");
    let sp = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    families.insert(mu, sp);
    families.insert(nu, sp);
    let mut store = PropertyStore::new();
    store.set_index_to_family(families.clone());
    let pattern = PropertyPattern {
        base_name: r,
        index_slots: vec![
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
        ],
    };
    store.declare(pattern, TensorProperty::RiemannSymmetry);
    let indices = vec![
        Index {
            name: mu,
            variance: Variance::Down,
            index_type: None,
        },
        Index {
            name: nu,
            variance: Variance::Down,
            index_type: None,
        },
        Index {
            name: mu,
            variance: Variance::Down,
            index_type: None,
        },
        Index {
            name: nu,
            variance: Variance::Down,
            index_type: None,
        },
    ];
    let props = store.get(r, &indices, &families);
    assert_eq!(props.len(), 1);
    assert!(matches!(props[0], TensorProperty::RiemannSymmetry));
}

#[test]
fn pattern_mismatch_wrong_variance() {
    let interner = Interner::new();
    let t = interner.get_or_intern("T");
    let a = interner.get_or_intern("a");
    let sp = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    families.insert(a, sp);
    let mut store = PropertyStore::new();
    store.set_index_to_family(families.clone());
    let pattern = PropertyPattern {
        base_name: t,
        index_slots: vec![
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
            SlotSpec {
                variance: Variance::Down,
                family: Some(sp),
            },
        ],
    };
    store.declare(pattern, TensorProperty::Symmetric(vec![0, 1]));
    let indices = vec![
        Index {
            name: a,
            variance: Variance::Up,
            index_type: None,
        },
        Index {
            name: a,
            variance: Variance::Down,
            index_type: None,
        },
    ];
    let props = store.get(t, &indices, &families);
    assert_eq!(props.len(), 0, "up+down should not match down+down pattern");
}

#[test]
fn wildcard_pattern_matches_any_indices() {
    let interner = Interner::new();
    let g = interner.get_or_intern("g");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let sp = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    families.insert(a, sp);
    families.insert(b, sp);
    let mut store = PropertyStore::new();
    store.set_index_to_family(families.clone());
    store.declare_simple(g, TensorProperty::Metric);
    let indices = vec![
        Index {
            name: a,
            variance: Variance::Up,
            index_type: None,
        },
        Index {
            name: b,
            variance: Variance::Down,
            index_type: None,
        },
    ];
    let props = store.get(g, &indices, &families);
    assert_eq!(props.len(), 1);
}

#[test]
fn tensor_identities_are_derived_from_legacy_bianchi_properties() {
    let interner = Interner::new();
    let r = interner.get_or_intern("R");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");
    let mut store = PropertyStore::new();
    store.declare_simple(
        r,
        TensorProperty::SatisfiesBianchi {
            slots: vec![0, 1, 2, 3],
        },
    );

    let identities = store
        .try_get_tensor_identities(
            r,
            &[
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: c,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: d,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
            &HashMap::new(),
        )
        .unwrap();

    assert_eq!(
        identities.multiterm,
        vec![TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [1, 2, 3]
        }]
    );
}

#[test]
fn weight_inheritance_additive() {
    let interner = Interner::new();
    let phi = interner.get_or_intern("phi");
    let psi = interner.get_or_intern("psi");
    let mut store = PropertyStore::new();
    store.add_inheritance(InheritanceRule::WeightInherit {
        label: "field".to_string(),
        combine: WeightCombine::Additive,
    });
    let mut weights = HashMap::new();
    weights.insert((phi, "field".to_string()), 1i64);
    weights.insert((psi, "field".to_string()), 1i64);
    let expr = Expr::mul(vec![Expr::Sym(phi), Expr::Sym(psi)]);
    let w = store.compute_weight(&expr, "field", &weights);
    assert_eq!(w, 2, "weight of phi*psi should be 2, got {}", w);
}

#[test]
fn depends_inheritance() {
    let interner = Interner::new();
    let phi = interner.get_or_intern("phi");
    let chi = interner.get_or_intern("chi");
    let x = interner.get_or_intern("x");
    let t = interner.get_or_intern("t");
    let mut store = PropertyStore::new();
    store.add_inheritance(InheritanceRule::DependsInherit);
    let mut depends: HashMap<lasso::Spur, Vec<lasso::Spur>> = HashMap::new();
    depends.insert(phi, vec![x, t]);
    depends.insert(chi, vec![t]);
    let expr = Expr::mul(vec![Expr::Sym(phi), Expr::Sym(chi)]);
    let deps = store.compute_depends(&expr, &depends);
    assert!(deps.contains(&x));
    assert!(deps.contains(&t));
    assert_eq!(deps.len(), 2);
}

#[test]
fn migrate_from_hashmap() {
    let interner = Interner::new();
    let g = interner.get_or_intern("g");
    let mut old: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    old.insert(
        g,
        vec![
            TensorProperty::Metric,
            TensorProperty::Symmetric(vec![0, 1]),
        ],
    );
    let families: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
    let store = PropertyStore::migrate_from_hashmap(&old, &families);
    assert_eq!(store.get_all(g).len(), 2);
}

#[test]
fn structured_spinor_metadata_is_stored_with_legacy_markers() {
    let interner = Interner::new();
    let psi = interner.get_or_intern("psi");
    let spin = interner.get_or_intern("spin");
    let mut store = PropertyStore::new();

    store.declare_spinor_meta(
        psi,
        SpinorMetadata {
            class: SpinorClass::Majorana,
            dimension: Some(4),
            chirality: None,
            index_family: Some(spin),
        },
    );

    let props = store.get_all(psi);
    assert!(props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(family),
            }) if *family == spin
        )
    }));
    assert!(props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::Spinor)));
    assert!(props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::MajoranaSpinor)));
}

#[test]
fn structured_gamma_and_dirac_bar_metadata_attach_legacy_markers() {
    let interner = Interner::new();
    let gamma = interner.get_or_intern("gamma");
    let eta = interner.get_or_intern("eta");
    let spin = interner.get_or_intern("spin");
    let bar = interner.get_or_intern("psibar");
    let mut store = PropertyStore::new();

    store.declare_gamma_matrix_meta(
        gamma,
        GammaMatrixMetadata {
            dimension: Some(4),
            metric_symbol: Some(eta),
            index_family: Some(spin),
            has_gamma5: true,
        },
    );
    store.declare_dirac_bar_meta(
        bar,
        DiracBarMetadata {
            gamma_symbol: Some(gamma),
            spinor_family: Some(spin),
            reverse_gamma_order: true,
        },
    );

    let gamma_props = store.get_all(gamma);
    assert!(gamma_props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: Some(metric),
                index_family: Some(family),
                has_gamma5: true,
            }) if *metric == eta && *family == spin
        )
    }));
    assert!(gamma_props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::GammaMatrixProp)));

    let bar_props = store.get_all(bar);
    assert!(bar_props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma_symbol),
                spinor_family: Some(family),
                reverse_gamma_order: true,
            }) if *gamma_symbol == gamma && *family == spin
        )
    }));
    assert!(bar_props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::DiracBar)));
}

#[test]
fn structured_trace_space_metadata_is_stored() {
    let interner = Interner::new();
    let tr = interner.get_or_intern("Tr");
    let color = interner.get_or_intern("color");
    let mut store = PropertyStore::new();

    store.declare_trace_space(
        tr,
        TraceSpaceMetadata {
            space_symbol: color,
            cyclic: true,
        },
    );

    let props = store.get_all(tr);
    assert!(props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::TraceSpaceMeta(TraceSpaceMetadata {
                space_symbol,
                cyclic: true,
            }) if *space_symbol == color
        )
    }));
}

#[test]
fn bosonic_mode_declaration_stores_mode_meta_and_noncommuting() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_mode",
        vec![
            serde_json::json!("a0"),
            serde_json::json!("bosonic"),
            serde_json::json!(0),
        ],
    )
    .unwrap();

    let a0 = state.interner.get_or_intern("a0");
    let props = state.env.property_store.get_all(a0);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::ModeMeta(ModeMetadata {
            statistics: ModeStatistics::Bosonic,
            subsystem: None,
            mode_index: 0,
            label: None,
        })
    )));
    assert!(props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::NonCommuting)));
}

#[test]
fn fermionic_mode_declaration_stores_mode_meta_noncommuting_and_anticommuting() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_mode",
        vec![
            serde_json::json!("c1"),
            serde_json::json!("fermionic"),
            serde_json::json!(1),
        ],
    )
    .unwrap();

    let c1 = state.interner.get_or_intern("c1");
    let props = state.env.property_store.get_all(c1);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::ModeMeta(ModeMetadata {
            statistics: ModeStatistics::Fermionic,
            subsystem: None,
            mode_index: 1,
            label: None,
        })
    )));
    assert!(props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::NonCommuting)));
    assert!(props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::AntiCommuting)));
}

#[test]
fn mode_declaration_subsystem_and_label_are_stored_correctly() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_mode_with_label",
        vec![
            serde_json::json!("m0"),
            serde_json::json!("spin"),
            serde_json::json!("reg"),
            serde_json::json!(0),
            serde_json::json!("a"),
        ],
    )
    .unwrap();

    let reg = state.interner.get_or_intern("reg");
    let a = state.interner.get_or_intern("a");
    let m0 = state.interner.get_or_intern("m0");
    let props = state.env.property_store.get_all(m0);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::ModeMeta(ModeMetadata {
            statistics: ModeStatistics::Spin,
            subsystem: Some(subsystem),
            mode_index: 0,
            label: Some(label),
        }) if *subsystem == reg && *label == a
    )));
}

#[test]
fn invalid_mode_statistics_string_returns_exact_error() {
    let mut state = TestState::default();
    let err = call_registry(
        &mut state,
        "declare_mode",
        vec![
            serde_json::json!("x"),
            serde_json::json!("anyon"),
            serde_json::json!(0),
        ],
    )
    .unwrap_err();

    assert_eq!(
        err,
        "declare_mode statistics must be one of: bosonic, fermionic, spin"
    );
}

#[test]
fn bosonic_truncated_two_mode_fock_space_stores_fock_space_meta() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_bosonic_truncated_mode",
        vec![
            serde_json::json!("a0"),
            serde_json::json!(0),
            serde_json::json!(2),
        ],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_bosonic_truncated_mode",
        vec![
            serde_json::json!("a1"),
            serde_json::json!(1),
            serde_json::json!(3),
        ],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_fock_space",
        vec![serde_json::json!("F"), serde_json::json!(["a0", "a1"])],
    )
    .unwrap();

    let f = state.interner.get_or_intern("F");
    let a0 = state.interner.get_or_intern("a0");
    let a1 = state.interner.get_or_intern("a1");
    let props = state.env.property_store.get_all(f);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::FockSpaceMeta(FockSpaceMetadata {
            symbol,
            modes,
            basis_order,
        }) if *symbol == f
            && basis_order == &vec![a0, a1]
            && modes == &vec![
                FockModeFactor {
                    symbol: a0,
                    statistics: ModeStatistics::Bosonic,
                    truncation: Some(2),
                },
                FockModeFactor {
                    symbol: a1,
                    statistics: ModeStatistics::Bosonic,
                    truncation: Some(3),
                },
            ]
    )));
}

#[test]
fn bosonic_fock_basis_builder_enforces_occupation_list_length() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_bosonic_truncated_mode",
        vec![
            serde_json::json!("a0"),
            serde_json::json!(0),
            serde_json::json!(2),
        ],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_bosonic_truncated_mode",
        vec![
            serde_json::json!("a1"),
            serde_json::json!(1),
            serde_json::json!(2),
        ],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_fock_space",
        vec![serde_json::json!("F"), serde_json::json!(["a0", "a1"])],
    )
    .unwrap();

    let err = call_registry(
        &mut state,
        "bosonic_fock_basis_state",
        vec![serde_json::json!("F"), serde_json::json!([1])],
    )
    .unwrap_err();

    assert_eq!(
        err,
        "bosonic_fock_basis_state occupation list does not match the declared Fock space"
    );
}

#[test]
fn fermionic_fock_basis_builder_enforces_binary_occupations() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_fermionic_mode",
        vec![serde_json::json!("c0"), serde_json::json!(0)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_fermionic_mode",
        vec![serde_json::json!("c1"), serde_json::json!(1)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_fock_space",
        vec![serde_json::json!("Ff"), serde_json::json!(["c0", "c1"])],
    )
    .unwrap();

    let err = call_registry(
        &mut state,
        "fermionic_fock_basis_state",
        vec![serde_json::json!("Ff"), serde_json::json!([1, 2])],
    )
    .unwrap_err();

    assert_eq!(
        err,
        "fermionic_fock_basis_state occupation list does not match the declared Fock space"
    );
}

#[test]
fn elementary_hilbert_space_declaration_stores_hilbert_space_meta() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("H"), serde_json::json!(2)],
    )
    .unwrap();

    let h = state.interner.get_or_intern("H");
    let props = state.env.property_store.get_all(h);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::HilbertSpaceMeta(HilbertSpaceMetadata { dimension: 2, factors })
            if factors == &vec![HilbertSpaceFactor { symbol: h, dimension: 2 }]
    )));
}

#[test]
fn composite_hilbert_space_declaration_flattens_factor_order_correctly() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HA"), serde_json::json!(2)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HB"), serde_json::json!(3)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_composite_space",
        vec![serde_json::json!("HAB"), serde_json::json!(["HA", "HB"])],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HC"), serde_json::json!(5)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_composite_space",
        vec![serde_json::json!("HABC"), serde_json::json!(["HAB", "HC"])],
    )
    .unwrap();

    let ha = state.interner.get_or_intern("HA");
    let hb = state.interner.get_or_intern("HB");
    let hc = state.interner.get_or_intern("HC");
    let habc = state.interner.get_or_intern("HABC");
    let meta = state
        .env
        .property_store
        .get_all(habc)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        meta.factors,
        vec![
            HilbertSpaceFactor {
                symbol: ha,
                dimension: 2,
            },
            HilbertSpaceFactor {
                symbol: hb,
                dimension: 3,
            },
            HilbertSpaceFactor {
                symbol: hc,
                dimension: 5,
            },
        ]
    );
}

#[test]
fn composite_hilbert_space_dimension_is_product_of_factor_dimensions() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HA"), serde_json::json!(2)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HB"), serde_json::json!(7)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_composite_space",
        vec![serde_json::json!("HAB"), serde_json::json!(["HA", "HB"])],
    )
    .unwrap();

    let hab = state.interner.get_or_intern("HAB");
    let meta = state
        .env
        .property_store
        .get_all(hab)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(meta.dimension, 14);
}

#[test]
fn quantum_object_declaration_stores_quantum_object_meta() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("H"), serde_json::json!(2)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_quantum_object",
        vec![
            serde_json::json!("psi"),
            serde_json::json!("ket"),
            serde_json::json!("H"),
        ],
    )
    .unwrap();

    let psi = state.interner.get_or_intern("psi");
    let h = state.interner.get_or_intern("H");
    let props = state.env.property_store.get_all(psi);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::QuantumObjectMeta(QuantumObjectMetadata {
            kind: QuantumObjectKind::Ket,
            space_symbol,
        }) if *space_symbol == h
    )));
}

#[test]
fn operator_density_projector_observable_channel_declarations_also_attach_noncommuting() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("H"), serde_json::json!(2)],
    )
    .unwrap();

    for (symbol, kind) in [
        ("A", "operator"),
        ("rho", "density_operator"),
        ("P", "projector"),
        ("Obs", "observable"),
        ("Phi", "channel"),
    ] {
        call_registry(
            &mut state,
            "declare_quantum_object",
            vec![
                serde_json::json!(symbol),
                serde_json::json!(kind),
                serde_json::json!("H"),
            ],
        )
        .unwrap();
        let sym = state.interner.get_or_intern(symbol);
        let props = state.env.property_store.get_all(sym);
        assert!(props
            .iter()
            .any(|prop| matches!(prop, TensorProperty::NonCommuting)));
    }
}

#[test]
fn valid_operator_space_declaration_stores_operator_space_meta() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HA"), serde_json::json!(2)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HB"), serde_json::json!(3)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_operator_space",
        vec![
            serde_json::json!("U"),
            serde_json::json!("HA"),
            serde_json::json!("HB"),
        ],
    )
    .unwrap();

    let ha = state.interner.get_or_intern("HA");
    let hb = state.interner.get_or_intern("HB");
    let u = state.interner.get_or_intern("U");
    let props = state.env.property_store.get_all(u);
    assert!(props.iter().any(|prop| matches!(
        prop,
        TensorProperty::OperatorSpaceMeta(OperatorSpaceMetadata {
            domain_space,
            codomain_space,
        }) if *domain_space == ha && *codomain_space == hb
    )));
}

#[test]
fn composing_compatible_operators_succeeds_and_propagates_metadata() {
    let mut state = TestState::default();
    for (symbol, dim) in [("HA", 2), ("HB", 3), ("HC", 5)] {
        call_registry(
            &mut state,
            "declare_hilbert_space",
            vec![serde_json::json!(symbol), serde_json::json!(dim)],
        )
        .unwrap();
    }
    for (symbol, domain, codomain) in [("U", "HB", "HC"), ("V", "HA", "HB")] {
        call_registry(
            &mut state,
            "declare_operator_space",
            vec![
                serde_json::json!(symbol),
                serde_json::json!(domain),
                serde_json::json!(codomain),
            ],
        )
        .unwrap();
    }

    let u = Expr::Sym(state.interner.get_or_intern("U"));
    let v = Expr::Sym(state.interner.get_or_intern("V"));
    let u_id = state.store_expr(u);
    let v_id = state.store_expr(v);
    let result = call_registry(
        &mut state,
        "compose_operators",
        vec![serde_json::json!(u_id), serde_json::json!(v_id)],
    )
    .expect("compatible operator composition should succeed");
    let expr_id = result["expr_id"]
        .as_str()
        .expect("compose_operators should return expr_id");
    let composed = state
        .get_expr(expr_id)
        .expect("stored composed expression should exist")
        .clone();
    assert_eq!(
        composed,
        Expr::Call(
            state.interner.get_or_intern("compose_operators"),
            vec![
                Expr::Sym(state.interner.get_or_intern("U")),
                Expr::Sym(state.interner.get_or_intern("V")),
            ],
        )
    );

    let ha = state.interner.get_or_intern("HA");
    let hc = state.interner.get_or_intern("HC");
    let meta = ax_eval::operator_space_metadata_of_expr(&mut state.env, &composed, &state.interner)
        .expect("compatible composition should carry propagated metadata");
    assert_eq!(
        meta,
        OperatorSpaceMetadata {
            domain_space: ha,
            codomain_space: hc,
        }
    );
}

#[test]
fn composing_incompatible_operators_returns_exact_error_string() {
    let mut state = TestState::default();
    for (symbol, dim) in [("HA", 2), ("HB", 3), ("HC", 5)] {
        call_registry(
            &mut state,
            "declare_hilbert_space",
            vec![serde_json::json!(symbol), serde_json::json!(dim)],
        )
        .unwrap();
    }
    for (symbol, domain, codomain) in [("U", "HA", "HB"), ("V", "HC", "HC")] {
        call_registry(
            &mut state,
            "declare_operator_space",
            vec![
                serde_json::json!(symbol),
                serde_json::json!(domain),
                serde_json::json!(codomain),
            ],
        )
        .unwrap();
    }

    let u_id = state.store_expr(Expr::Sym(state.interner.get_or_intern("U")));
    let v_id = state.store_expr(Expr::Sym(state.interner.get_or_intern("V")));
    let err = call_registry(
        &mut state,
        "compose_operators",
        vec![serde_json::json!(u_id), serde_json::json!(v_id)],
    )
    .expect_err("incompatible operator composition should fail");

    assert_eq!(
        err,
        "compose_operators requires codomain(right) = domain(left)"
    );
}

#[test]
fn dagger_swaps_domain_and_codomain_metadata() {
    let mut state = TestState::default();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HA"), serde_json::json!(2)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_hilbert_space",
        vec![serde_json::json!("HB"), serde_json::json!(3)],
    )
    .unwrap();
    call_registry(
        &mut state,
        "declare_operator_space",
        vec![
            serde_json::json!("U"),
            serde_json::json!("HA"),
            serde_json::json!("HB"),
        ],
    )
    .unwrap();

    let ha = state.interner.get_or_intern("HA");
    let hb = state.interner.get_or_intern("HB");
    let u = Expr::Sym(state.interner.get_or_intern("U"));
    let dagger = Expr::Call(state.interner.get_or_intern("dagger"), vec![u]);
    let meta = ax_eval::operator_space_metadata_of_expr(&mut state.env, &dagger, &state.interner)
        .expect("dagger should swap operator-space metadata");
    assert_eq!(
        meta,
        OperatorSpaceMetadata {
            domain_space: hb,
            codomain_space: ha,
        }
    );
}
