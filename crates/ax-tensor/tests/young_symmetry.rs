use ax_ir::{
    DualityKind, Expr, Index, RestrictedSymmetryMode, SymmetrySource, TableauAttachment,
    TensorProperty, TensorSymmetry, Variance,
};
use ax_tensor::{
    decompose_product_irreps, young_engine::apply_realized_tableaux_to_factor,
    young_project_tensor_with_options, YoungProjectTensorOptions,
};
use std::collections::HashMap;

fn index(name: lasso::Spur) -> Index {
    Index {
        name,
        variance: Variance::Down,
        index_type: None,
    }
}

fn tableau_property(shape: Vec<usize>, slot_map: Vec<usize>) -> TensorProperty {
    TensorProperty::TableauSymmetry(TensorSymmetry {
        tableaux: vec![TableauAttachment {
            shape,
            slot_map,
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: DualityKind::None,
            restricted_mode: RestrictedSymmetryMode::FullYoung,
            trace_free: false,
            dimension_guard: None,
            source: SymmetrySource::Declared,
            label: None,
        }],
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    })
}

#[test]
fn symmetric_rank_two_factor_canonicalizes_all_index_orders() {
    let interner = ax_ir::Interner::new();
    let t = interner.get_or_intern("S");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");

    let ab = Expr::Indexed(Box::new(Expr::Sym(t)), vec![index(a), index(b)]);
    let ba = Expr::Indexed(Box::new(Expr::Sym(t)), vec![index(b), index(a)]);

    let props = HashMap::from([(t, vec![tableau_property(vec![2], vec![0, 1])])]);
    let opts = YoungProjectTensorOptions::default();

    let expected = Expr::Indexed(Box::new(Expr::Sym(t)), vec![index(a), index(b)]);
    assert_eq!(young_project_tensor_with_options(&ab, &props, &interner, &opts), expected);
    assert_eq!(young_project_tensor_with_options(&ba, &props, &interner, &opts), expected);
}

#[test]
fn antisymmetric_rank_two_factor_swaps_with_negative_sign() {
    let interner = ax_ir::Interner::new();
    let a_sym = interner.get_or_intern("A");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");

    let ab = Expr::Indexed(Box::new(Expr::Sym(a_sym)), vec![index(a), index(b)]);
    let ba = Expr::Indexed(Box::new(Expr::Sym(a_sym)), vec![index(b), index(a)]);

    let props = HashMap::from([(a_sym, vec![tableau_property(vec![1, 1], vec![0, 1])])]);
    let opts = YoungProjectTensorOptions::default();

    assert_eq!(
        young_project_tensor_with_options(&ab, &props, &interner, &opts),
        ab
    );
    assert_eq!(
        young_project_tensor_with_options(&ba, &props, &interner, &opts),
        Expr::mul(vec![Expr::Int((-1).into()), ab])
    );
}

#[test]
fn rank_three_mixed_symmetry_projection_is_idempotent() {
    let interner = ax_ir::Interner::new();
    let t = interner.get_or_intern("T");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");

    let expr = Expr::Indexed(Box::new(Expr::Sym(t)), vec![index(c), index(a), index(b)]);
    let props = HashMap::from([(t, vec![tableau_property(vec![2, 1], vec![0, 1, 2])])]);
    let opts = YoungProjectTensorOptions::default();

    let projected = young_project_tensor_with_options(&expr, &props, &interner, &opts);
    let projected_twice = young_project_tensor_with_options(&projected, &props, &interner, &opts);
    assert_eq!(projected_twice, projected);
}

#[test]
fn structured_symmetry_takes_precedence_over_legacy_riemann_inference() {
    let interner = ax_ir::Interner::new();
    let r = interner.get_or_intern("R");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    let expr = Expr::Indexed(
        Box::new(Expr::Sym(r)),
        vec![index(b), index(a), index(d), index(c)],
    );
    let props = HashMap::from([(
        r,
        vec![
            tableau_property(vec![2], vec![0, 1]),
            TensorProperty::RiemannSymmetry,
        ],
    )]);
    let opts = YoungProjectTensorOptions {
        modulo_monoterm: true,
        canonicalize_after: false,
        rename_dummies_after: false,
    };

    assert_eq!(
        young_project_tensor_with_options(&expr, &props, &interner, &opts),
        Expr::Indexed(
            Box::new(Expr::Sym(r)),
            vec![index(a), index(b), index(d), index(c)],
        )
    );
}

#[test]
fn factor_without_tableaux_is_unchanged_by_structured_projection_path() {
    let interner = ax_ir::Interner::new();
    let v = interner.get_or_intern("V");
    let a = interner.get_or_intern("a");

    let expr = Expr::Indexed(Box::new(Expr::Sym(v)), vec![index(a)]);
    assert_eq!(
        apply_realized_tableaux_to_factor(&expr, &[], usize::MAX).unwrap(),
        expr
    );
}

#[test]
fn decompose_product_irreps_reports_exact_multiplicity_spaces() {
    let summary = decompose_product_irreps(&[vec![1], vec![1], vec![1]]).unwrap();
    assert_eq!(summary.shapes, vec![vec![1, 1, 1], vec![2, 1], vec![3]]);
    assert_eq!(summary.multiplicities, vec![1, 2, 1]);
    assert_eq!(
        summary.basis_labels,
        vec![
            vec!["m0".to_string()],
            vec!["m0".to_string(), "m1".to_string()],
            vec!["m0".to_string()],
        ]
    );
}
