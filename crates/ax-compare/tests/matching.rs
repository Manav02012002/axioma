use ax_compare::*;
use ax_ir::*;
use std::collections::HashMap;

#[allow(dead_code)]
fn make_interner_and_props() -> (Interner, HashMap<lasso::Spur, Vec<TensorProperty>>) {
    let interner = Interner::new();
    let mut props = HashMap::new();
    let r = interner.get_or_intern("R");
    props.insert(r, vec![TensorProperty::RiemannSymmetry]);
    let g = interner.get_or_intern("g");
    props.insert(
        g,
        vec![
            TensorProperty::Symmetric(vec![0, 1]),
            TensorProperty::Metric,
        ],
    );
    (interner, props)
}

#[test]
fn exact_match_scalar() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let families: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
    let result = pattern_match(&Expr::Sym(x), &Expr::Sym(x), &props, &families, &interner);
    assert!(result.is_some(), "x should match x");
}

#[test]
fn wildcard_match() {
    let interner = Interner::new();
    let x_ = interner.get_or_intern("x_");
    let val = Expr::Int(42.into());
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let families: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
    let result = pattern_match(&Expr::Sym(x_), &val, &props, &families, &interner);
    assert!(result.is_some(), "x_ should match 42");
    let map = result.unwrap();
    assert_eq!(map.wildcard_map.get(&x_), Some(&val));
}

#[test]
fn wildcard_consistency() {
    let interner = Interner::new();
    let x_ = interner.get_or_intern("x_");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let families: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
    let pattern = Expr::add(vec![Expr::Sym(x_), Expr::Sym(x_)]);
    let target_good = Expr::add(vec![Expr::Sym(y), Expr::Sym(y)]);
    let target_bad = Expr::add(vec![Expr::Sym(y), Expr::Sym(z)]);
    assert!(pattern_match(&pattern, &target_good, &props, &families, &interner).is_some());
    assert!(
        pattern_match(&pattern, &target_bad, &props, &families, &interner).is_none(),
        "x_ + x_ should NOT match y + z"
    );
}

#[test]
fn commutative_mul_match() {
    let interner = Interner::new();
    let a = interner.get_or_intern("A");
    let b = interner.get_or_intern("B");
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let families: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
    let pattern = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
    let target = Expr::mul(vec![Expr::Sym(b), Expr::Sym(a)]);
    let result = pattern_match(&pattern, &target, &props, &families, &interner);
    assert!(
        result.is_some(),
        "A*B should match B*A for commuting symbols"
    );
}

#[test]
fn indexed_match_by_family() {
    let interner = Interner::new();
    let t = interner.get_or_intern("T");
    let mu = interner.get_or_intern("mu");
    let nu = interner.get_or_intern("nu");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let spacetime = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    families.insert(mu, spacetime);
    families.insert(nu, spacetime);
    families.insert(a, spacetime);
    families.insert(b, spacetime);
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let pattern = Expr::Indexed(
        Box::new(Expr::Sym(t)),
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
    let target = Expr::Indexed(
        Box::new(Expr::Sym(t)),
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
        ],
    );
    let result = pattern_match(&pattern, &target, &props, &families, &interner);
    assert!(
        result.is_some(),
        "T_{{mu nu}} should match T_{{a b}} in same family"
    );
    let map = result.unwrap();
    assert_eq!(map.index_map.get(&mu), Some(&a));
    assert_eq!(map.index_map.get(&nu), Some(&b));
}

#[test]
fn indexed_mismatch_different_family() {
    let interner = Interner::new();
    let t = interner.get_or_intern("T");
    let mu = interner.get_or_intern("mu");
    let nu = interner.get_or_intern("nu");
    let i = interner.get_or_intern("i");
    let j = interner.get_or_intern("j");
    let spacetime = interner.get_or_intern("spacetime");
    let spatial = interner.get_or_intern("spatial");
    let mut families = HashMap::new();
    families.insert(mu, spacetime);
    families.insert(nu, spacetime);
    families.insert(i, spatial);
    families.insert(j, spatial);
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let pattern = Expr::Indexed(
        Box::new(Expr::Sym(t)),
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
    let target = Expr::Indexed(
        Box::new(Expr::Sym(t)),
        vec![
            Index {
                name: i,
                variance: Variance::Down,
                index_type: None,
            },
            Index {
                name: j,
                variance: Variance::Down,
                index_type: None,
            },
        ],
    );
    let result = pattern_match(&pattern, &target, &props, &families, &interner);
    assert!(
        result.is_none(),
        "spacetime indices should not match spatial indices"
    );
}

#[test]
fn dummy_aware_match() {
    let interner = Interner::new();
    let t_sym = interner.get_or_intern("T");
    let v_sym = interner.get_or_intern("V");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let m = interner.get_or_intern("m");
    let n = interner.get_or_intern("n");
    let sp = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    for idx in [a, b, m, n] {
        families.insert(idx, sp);
    }
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let pattern = Expr::mul(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
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
            ],
        ),
        Expr::Indexed(
            Box::new(Expr::Sym(v_sym)),
            vec![Index {
                name: b,
                variance: Variance::Up,
                index_type: None,
            }],
        ),
    ]);
    let target = Expr::mul(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                Index {
                    name: m,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: n,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        ),
        Expr::Indexed(
            Box::new(Expr::Sym(v_sym)),
            vec![Index {
                name: n,
                variance: Variance::Up,
                index_type: None,
            }],
        ),
    ]);
    let result = pattern_match(&pattern, &target, &props, &families, &interner);
    assert!(
        result.is_some(),
        "T_{{ab}}V^b should match T_{{mn}}V^n with dummy relabelling"
    );
}

#[test]
fn substitution_with_compare() {
    let interner = Interner::new();
    let t_sym = interner.get_or_intern("T");
    let g_sym = interner.get_or_intern("g");
    let v_sym = interner.get_or_intern("V");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let m = interner.get_or_intern("m");
    let n = interner.get_or_intern("n");
    let sp = interner.get_or_intern("spacetime");
    let mut families = HashMap::new();
    for idx in [a, b, m, n] {
        families.insert(idx, sp);
    }
    let props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    let pattern = Expr::Indexed(
        Box::new(Expr::Sym(t_sym)),
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
        ],
    );
    let replacement = Expr::Indexed(
        Box::new(Expr::Sym(g_sym)),
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
        ],
    );
    let expr = Expr::mul(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                Index {
                    name: m,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: n,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        ),
        Expr::Indexed(
            Box::new(Expr::Sym(v_sym)),
            vec![Index {
                name: n,
                variance: Variance::Up,
                index_type: None,
            }],
        ),
    ]);
    let result =
        substitute_with_compare(&expr, &pattern, &replacement, &props, &families, &interner);
    let expected = Expr::mul(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(g_sym)),
            vec![
                Index {
                    name: m,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: n,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        ),
        Expr::Indexed(
            Box::new(Expr::Sym(v_sym)),
            vec![Index {
                name: n,
                variance: Variance::Up,
                index_type: None,
            }],
        ),
    ]);
    assert_eq!(result, expected, "T should be replaced with g");
}
