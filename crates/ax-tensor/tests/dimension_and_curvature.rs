use ax_ir::{
    validate_duality_in_dimension, DualityKind, Expr, Index, RestrictedSymmetryMode,
    SymmetrySource, TableauAttachment, TensorProperty, TensorSymmetry, Variance,
};
use ax_tensor::{
    apply_first_bianchi_if_applicable, reduce_expr_by_dimension, riemann_tensor_symmetry,
    weyl_tensor_symmetry, young_project_tensor_with_options, YoungProjectTensorOptions,
};
use ax_young::induced_form_tableau_duality;
use std::collections::HashMap;

fn index(name: lasso::Spur) -> Index {
    Index {
        name,
        variance: Variance::Down,
        index_type: None,
    }
}

fn indexed(symbol: lasso::Spur, slots: &[lasso::Spur]) -> Expr {
    Expr::Indexed(
        Box::new(Expr::Sym(symbol)),
        slots.iter().copied().map(index).collect(),
    )
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
            label: Some("structured".to_string()),
        }],
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    })
}

#[test]
fn rank_three_column_tensor_reduces_by_dimension_exactly() {
    let interner = ax_ir::Interner::new();
    let f = interner.get_or_intern("F");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let expr = indexed(f, &[a, b, c]);
    let props = vec![tableau_property(vec![1, 1, 1], vec![0, 1, 2])];

    assert_eq!(
        reduce_expr_by_dimension(
            &expr,
            &|symbol| (symbol == f).then_some(props.clone()).unwrap_or_default(),
            Some(2)
        )
        .unwrap(),
        Expr::zero()
    );
    assert_eq!(
        reduce_expr_by_dimension(
            &expr,
            &|symbol| (symbol == f).then_some(props.clone()).unwrap_or_default(),
            Some(3)
        )
        .unwrap(),
        expr
    );
}

#[test]
fn selfdual_rank_two_form_validation_depends_on_dimension() {
    let symmetry = induced_form_tableau_duality(2, 4, DualityKind::SelfDual).unwrap();
    assert_eq!(
        validate_duality_in_dimension(&symmetry.tableaux[0], Some(4)),
        Ok(())
    );
    assert_eq!(
        validate_duality_in_dimension(&symmetry.tableaux[0], Some(3)),
        Err(ax_ir::DualityValidationError::OddDimensionSelfDual { dim: 3 })
    );
}

#[test]
fn curvature_symmetry_builders_are_exact() {
    let riemann = riemann_tensor_symmetry();
    let weyl = weyl_tensor_symmetry();

    assert_eq!(riemann.tableaux[0].shape, vec![2, 2]);
    assert_eq!(weyl.tableaux[0].shape, vec![2, 2]);
    assert!(!riemann.tableaux[0].trace_free);
    assert!(weyl.tableaux[0].trace_free);
}

#[test]
fn first_bianchi_application_on_single_factor_yields_exact_cyclic_sum() {
    let interner = ax_ir::Interner::new();
    let r = interner.get_or_intern("R");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");
    let expr = indexed(r, &[a, b, c, d]);

    let applied = apply_first_bianchi_if_applicable(&expr, &|symbol| {
        if symbol == r {
            vec![TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2, 3],
            }]
        } else {
            vec![]
        }
    })
    .expect("first Bianchi should apply");

    assert_eq!(
        applied,
        Expr::add(vec![
            indexed(r, &[a, b, c, d]),
            indexed(r, &[a, c, d, b]),
            indexed(r, &[a, d, b, c]),
        ])
    );
}

#[test]
fn explicit_structured_symmetry_and_legacy_bianchi_coexist() {
    let interner = ax_ir::Interner::new();
    let t = interner.get_or_intern("T");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let expr = indexed(t, &[b, a, c]);
    let props = HashMap::from([(
        t,
        vec![
            tableau_property(vec![2], vec![0, 1]),
            TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2],
            },
        ],
    )]);

    let projected = young_project_tensor_with_options(
        &expr,
        &props,
        &interner,
        &YoungProjectTensorOptions::default(),
    );
    assert_eq!(
        projected,
        Expr::add(vec![
            Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new((-1).into(), 3.into())),
                indexed(t, &[a, c, b]),
            ]),
            Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new((-1).into(), 3.into())),
                indexed(t, &[b, c, a]),
            ]),
            Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new(2.into(), 3.into())),
                indexed(t, &[a, b, c]),
            ]),
        ])
    );

    let bianchi = apply_first_bianchi_if_applicable(&indexed(t, &[a, b, c]), &|symbol| {
        props.get(&symbol).cloned().unwrap_or_default()
    })
    .expect("legacy Bianchi should still be available");
    assert_eq!(
        bianchi,
        Expr::add(vec![
            indexed(t, &[a, b, c]),
            indexed(t, &[b, c, a]),
            indexed(t, &[c, a, b]),
        ])
    );
}

#[test]
fn curvature_aware_projection_is_deterministic_under_repetition() {
    let interner = ax_ir::Interner::new();
    let r = interner.get_or_intern("R");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");
    let expr = indexed(r, &[b, a, d, c]);
    let props = HashMap::from([(
        r,
        vec![
            TensorProperty::RiemannSymmetry,
            TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2, 3],
            },
        ],
    )]);
    let opts = YoungProjectTensorOptions::default();

    let once = young_project_tensor_with_options(&expr, &props, &interner, &opts);
    let twice = young_project_tensor_with_options(&once, &props, &interner, &opts);
    assert_eq!(twice, once);
}
