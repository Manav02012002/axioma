use ax_eval::property_store::*;
use ax_ir::*;
use std::collections::HashMap;

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
