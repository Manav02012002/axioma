#[test]
fn parallel_canon_matches_sequential_riemann_sum() {
    let interner = ax_ir::Interner::new();
    let r = interner.get_or_intern("R");
    let mut props = std::collections::HashMap::new();
    props.insert(r, vec![ax_ir::TensorProperty::RiemannSymmetry]);
    let index_names: Vec<lasso::Spur> = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "s", "u", "v",
    ]
    .iter()
    .map(|s| interner.get_or_intern(s))
    .collect();
    let mut terms = Vec::new();
    for t in 0..20 {
        let perm = [
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 1, 0, 3],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [3, 1, 0, 2],
            [3, 2, 1, 0],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 3, 2, 1],
            [1, 0, 3, 2],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
            [2, 1, 3, 0],
        ][t];
        let indices = perm
            .iter()
            .map(|&p| ax_ir::Index {
                name: index_names[p],
                variance: ax_ir::Variance::Down,
                index_type: None,
            })
            .collect();
        terms.push(ax_ir::Expr::Indexed(Box::new(ax_ir::Expr::Sym(r)), indices));
    }
    let expr = ax_ir::Expr::Add(terms);
    let seq = ax_tensor::canonicalise(&expr, &props, &interner);
    let par = ax_tensor::canonicalise_parallel(&expr, &props, &interner);
    assert_eq!(seq, par, "parallel canonicalise should match sequential");
}

#[test]
fn parallel_canon_small_expr_uses_sequential() {
    let interner = ax_ir::Interner::new();
    let r = interner.get_or_intern("R");
    let props = std::collections::HashMap::new();
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let mk = |n1, n2| {
        ax_ir::Expr::Indexed(
            Box::new(ax_ir::Expr::Sym(r)),
            vec![
                ax_ir::Index {
                    name: n1,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: n2,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        )
    };
    let expr = ax_ir::Expr::Add(vec![mk(a, b), mk(b, a), mk(a, a)]);
    let seq = ax_tensor::canonicalise(&expr, &props, &interner);
    let par = ax_tensor::canonicalise_parallel(&expr, &props, &interner);
    assert_eq!(seq, par);
}
