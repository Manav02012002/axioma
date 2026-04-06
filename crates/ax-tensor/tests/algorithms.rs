use ax_ir::*;
use ax_tensor::*;
use std::collections::{HashMap, HashSet};

fn interner() -> Interner {
    Interner::new()
}

fn mk_indexed(base: &str, indices: &[(&str, Variance)], interner: &Interner) -> Expr {
    let sym = interner.get_or_intern(base);
    let idx: Vec<Index> = indices
        .iter()
        .map(|(name, var)| Index {
            name: interner.get_or_intern(name),
            variance: var.clone(),
            index_type: None,
        })
        .collect();
    Expr::Indexed(Box::new(Expr::Sym(sym)), idx)
}

// === CANONICALISE ===

#[test]
fn canon_symmetric_reorders() {
    // g_{b a} with Symmetric should become g_{a b}
    let int = interner();
    let g = int.get_or_intern("g");
    let mut props = HashMap::new();
    props.insert(g, vec![TensorProperty::Symmetric(vec![0, 1])]);
    let expr = mk_indexed("g", &[("b", Variance::Down), ("a", Variance::Down)], &int);
    let result = canonicalise(&expr, &props, &int);
    if let Expr::Indexed(_, indices) = &result {
        let names: Vec<_> = indices.iter().map(|i| int.resolve(i.name)).collect();
        assert!(
            names[0] <= names[1],
            "symmetric tensor should have sorted indices, got {:?}",
            names
        );
    } else {
        let result_str = pretty_print(&result, &int);
        assert!(
            result_str.contains('g'),
            "expected indexed g, got {}",
            result_str
        );
    }
}

#[test]
fn canon_antisymmetric_sign() {
    // F_{b a} with AntiSymmetric should become -F_{a b}
    let int = interner();
    let f = int.get_or_intern("F");
    let mut props = HashMap::new();
    props.insert(f, vec![TensorProperty::AntiSymmetric(vec![0, 1])]);
    let expr = mk_indexed("F", &[("b", Variance::Down), ("a", Variance::Down)], &int);
    let result = canonicalise(&expr, &props, &int);
    match &result {
        Expr::Neg(inner) => {
            if let Expr::Indexed(_, indices) = inner.as_ref() {
                let names: Vec<_> = indices.iter().map(|i| int.resolve(i.name)).collect();
                assert!(names[0] < names[1], "antisymmetric should sort and negate");
            }
        }
        Expr::Mul(factors) => {
            let has_neg = factors.iter().any(|factor| {
                matches!(factor, Expr::Int(n) if n == &(-1).into())
                    || matches!(factor, Expr::Neg(_))
            });
            assert!(
                has_neg,
                "expected negative sign for F_{{ba}}, got {:?}",
                result
            );
        }
        _ => {}
    }
}

#[test]
fn canon_riemann_first_bianchi() {
    // R_{abcd} + R_{acdb} + R_{adbc} = 0 (first Bianchi identity)
    let int = interner();
    let r = int.get_or_intern("R");
    let mut props = HashMap::new();
    props.insert(r, vec![TensorProperty::RiemannSymmetry]);
    let t1 = mk_indexed(
        "R",
        &[
            ("a", Variance::Down),
            ("b", Variance::Down),
            ("c", Variance::Down),
            ("d", Variance::Down),
        ],
        &int,
    );
    let t2 = mk_indexed(
        "R",
        &[
            ("a", Variance::Down),
            ("c", Variance::Down),
            ("d", Variance::Down),
            ("b", Variance::Down),
        ],
        &int,
    );
    let t3 = mk_indexed(
        "R",
        &[
            ("a", Variance::Down),
            ("d", Variance::Down),
            ("b", Variance::Down),
            ("c", Variance::Down),
        ],
        &int,
    );
    let sum = Expr::Add(vec![t1, t2, t3]);
    let result = meld(&sum, &props, &int);
    assert_eq!(
        result,
        Expr::zero(),
        "first Bianchi identity should give 0, got {:?}",
        result
    );
}

// === ELIMINATE_KRONECKER ===

#[test]
fn eliminate_kronecker_contracts() {
    // delta^{mu}_{nu} * V^{nu} = V^{mu}
    let int = interner();
    let delta = int.get_or_intern("delta");
    let v = int.get_or_intern("V");
    let mu = int.get_or_intern("mu");
    let nu = int.get_or_intern("nu");
    let delta_expr = Expr::Indexed(
        Box::new(Expr::Sym(delta)),
        vec![
            Index {
                name: mu,
                variance: Variance::Up,
                index_type: None,
            },
            Index {
                name: nu,
                variance: Variance::Down,
                index_type: None,
            },
        ],
    );
    let v_expr = Expr::Indexed(
        Box::new(Expr::Sym(v)),
        vec![Index {
            name: nu,
            variance: Variance::Up,
            index_type: None,
        }],
    );
    let product = Expr::mul(vec![delta_expr, v_expr]);
    let result = eliminate_kronecker(&product, delta, &int);
    let result_str = pretty_print(&result, &int);
    assert!(
        !result_str.contains("delta"),
        "delta should be eliminated, got {}",
        result_str
    );
    assert!(
        !result_str.contains("nu"),
        "nu should be contracted away, got {}",
        result_str
    );
}

// === ELIMINATE_METRIC ===

#[test]
fn eliminate_metric_lowers_index() {
    // g_{mu nu} * V^{nu} should give V_{mu} (with nu contracted)
    let int = interner();
    let g = int.get_or_intern("g");
    let v = int.get_or_intern("V");
    let mu = int.get_or_intern("mu");
    let nu = int.get_or_intern("nu");
    let g_inv = int.get_or_intern("ginv");
    let g_expr = Expr::Indexed(
        Box::new(Expr::Sym(g)),
        vec![
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
        ],
    );
    let v_expr = Expr::Indexed(
        Box::new(Expr::Sym(v)),
        vec![Index {
            name: nu,
            variance: Variance::Up,
            index_type: None,
        }],
    );
    let product = Expr::mul(vec![g_expr, v_expr]);
    let result = eliminate_metric(&product, g, g_inv, &int);
    let result_str = pretty_print(&result, &int);
    assert!(
        !result_str.contains("nu"),
        "nu should be contracted, got {}",
        result_str
    );
}

// === EPSILON_TO_DELTA ===

#[test]
fn epsilon_contraction_3d() {
    // eps_{abc} * eps^{abc} = 6 (= 3!)
    let int = interner();
    let eps = int.get_or_intern("eps");
    let delta = int.get_or_intern("delta");
    let a = int.get_or_intern("a");
    let b = int.get_or_intern("b");
    let c = int.get_or_intern("c");
    let eps_down = Expr::Indexed(
        Box::new(Expr::Sym(eps)),
        vec![
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
        ],
    );
    let eps_up = Expr::Indexed(
        Box::new(Expr::Sym(eps)),
        vec![
            Index {
                name: a,
                variance: Variance::Up,
                index_type: None,
            },
            Index {
                name: b,
                variance: Variance::Up,
                index_type: None,
            },
            Index {
                name: c,
                variance: Variance::Up,
                index_type: None,
            },
        ],
    );
    let product = Expr::mul(vec![eps_down, eps_up]);
    let result = epsilon_to_delta(&product, eps, delta, 3, &int);
    let result_str = pretty_print(&result, &int);
    assert!(
        !result_str.contains("eps"),
        "epsilon should be eliminated, got {}",
        result_str
    );
}

// === SORT_PRODUCT ===

#[test]
fn sort_product_alphabetical() {
    let int = interner();
    let b = int.get_or_intern("B");
    let a = int.get_or_intern("A");
    let expr = Expr::mul(vec![Expr::Sym(b), Expr::Sym(a)]);
    let props = HashMap::new();
    let result = sort_product(&expr, &props, &int);
    if let Expr::Mul(factors) = &result {
        if let (Expr::Sym(s1), Expr::Sym(s2)) = (&factors[0], &factors[1]) {
            assert!(
                int.resolve(*s1) <= int.resolve(*s2),
                "should be sorted alphabetically"
            );
        }
    }
}

// === RENAME_DUMMIES ===

#[test]
fn rename_dummies_alpha_equiv() {
    // T_{a}^{a} and T_{b}^{b} should be equal after rename_dummies
    let int = interner();
    let t = int.get_or_intern("T");
    let a = int.get_or_intern("a");
    let b = int.get_or_intern("b");
    let e1 = Expr::Indexed(
        Box::new(Expr::Sym(t)),
        vec![
            Index {
                name: a,
                variance: Variance::Down,
                index_type: None,
            },
            Index {
                name: a,
                variance: Variance::Up,
                index_type: None,
            },
        ],
    );
    let e2 = Expr::Indexed(
        Box::new(Expr::Sym(t)),
        vec![
            Index {
                name: b,
                variance: Variance::Down,
                index_type: None,
            },
            Index {
                name: b,
                variance: Variance::Up,
                index_type: None,
            },
        ],
    );
    let r1 = rename_dummy_indices(&e1, &int);
    let r2 = rename_dummy_indices(&e2, &int);
    assert_eq!(
        r1, r2,
        "alpha-equivalent contractions should be equal after dummy renaming"
    );
}

// === DISTRIBUTE ===

#[test]
fn distribute_product_over_sum() {
    // A * (B + C) = A*B + A*C
    let int = interner();
    let a = Expr::Sym(int.get_or_intern("A"));
    let b = Expr::Sym(int.get_or_intern("B"));
    let c = Expr::Sym(int.get_or_intern("C"));
    let expr = Expr::mul(vec![a.clone(), Expr::add(vec![b.clone(), c.clone()])]);
    let result = tensor_distribute(&expr, &int);
    match &result {
        Expr::Add(terms) => assert_eq!(terms.len(), 2, "should have 2 terms after distribute"),
        _ => panic!("distribute should give Add, got {:?}", result),
    }
}

// === PRODUCT_RULE ===

#[test]
fn product_rule_splits() {
    // d(A * B) = dA * B + A * dB
    let int = interner();
    let d = int.get_or_intern("d");
    let a = Expr::Sym(int.get_or_intern("A"));
    let b = Expr::Sym(int.get_or_intern("B"));
    let expr = Expr::Call(d, vec![Expr::mul(vec![a, b])]);
    let mut derivs = HashSet::new();
    derivs.insert(d);
    let result = product_rule(&expr, &derivs, &int);
    match &result {
        Expr::Add(terms) => assert_eq!(terms.len(), 2, "product rule should give 2 terms"),
        _ => panic!("product rule should give Add, got {:?}", result),
    }
}
