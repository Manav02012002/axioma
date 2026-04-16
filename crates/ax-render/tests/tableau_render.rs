use ax_ir::{
    DualityKind, RestrictedSymmetryMode, SymmetrySource, TableauAttachment, TensorSymmetry,
};
use ax_render::{render_tableau_slot_map_ascii, render_tensor_symmetry_summary};

#[test]
fn renders_multi_tableau_summary_with_labels_exactly() {
    let symmetry = TensorSymmetry {
        tableaux: vec![
            TableauAttachment {
                shape: vec![2, 1],
                slot_map: vec![0, 1, 2],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: DualityKind::None,
                restricted_mode: RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: Some("main".to_string()),
            },
            TableauAttachment {
                shape: vec![1, 1],
                slot_map: vec![1, 2],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: DualityKind::None,
                restricted_mode: RestrictedSymmetryMode::FullYoung,
                trace_free: true,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: Some("alt".to_string()),
            },
        ],
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    };

    assert_eq!(
        render_tensor_symmetry_summary(&symmetry),
        concat!(
            "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None, label=\"main\"\n",
            "tableau[1]: shape=[1, 1], slots=[1, 2], trace_free=true, duality=None, label=\"alt\""
        )
    );
}

#[test]
fn renders_tableau_slot_maps_exactly() {
    assert_eq!(
        render_tableau_slot_map_ascii(&[2, 1], &[0, 1, 2]),
        "[0][1]\n[2]"
    );
}

#[test]
fn renders_trace_free_curvature_summary_exactly() {
    let symmetry = TensorSymmetry {
        tableaux: vec![TableauAttachment {
            shape: vec![2, 2],
            slot_map: vec![0, 1, 2, 3],
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: DualityKind::None,
            restricted_mode: RestrictedSymmetryMode::FullYoung,
            trace_free: true,
            dimension_guard: None,
            source: SymmetrySource::Derived,
            label: Some("weyl".to_string()),
        }],
        inherits_under_derivative: true,
        inherits_under_tensor_product: true,
        inherits_under_contraction: true,
        preserves_trace_free_under_projection: true,
    };
    assert_eq!(
        render_tensor_symmetry_summary(&symmetry),
        "tableau[0]: shape=[2, 2], slots=[0, 1, 2, 3], trace_free=true, duality=None, label=\"weyl\""
    );
}
