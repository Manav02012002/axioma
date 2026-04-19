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
