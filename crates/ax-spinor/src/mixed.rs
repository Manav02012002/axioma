pub fn symmetric_two_undotted_spinors() -> ax_ir::MixedTensorSymmetry {
    ax_ir::MixedTensorSymmetry {
        tableaux: vec![ax_ir::MixedTableauAttachment {
            shape: vec![2],
            slots: vec![
                ax_ir::MixedSlot {
                    index: 0,
                    kind: ax_ir::SlotKind::UndottedSpinor,
                },
                ax_ir::MixedSlot {
                    index: 1,
                    kind: ax_ir::SlotKind::UndottedSpinor,
                },
            ],
            label: None,
        }],
    }
}

pub fn antisymmetric_two_undotted_spinors() -> ax_ir::MixedTensorSymmetry {
    ax_ir::MixedTensorSymmetry {
        tableaux: vec![ax_ir::MixedTableauAttachment {
            shape: vec![1, 1],
            slots: vec![
                ax_ir::MixedSlot {
                    index: 0,
                    kind: ax_ir::SlotKind::UndottedSpinor,
                },
                ax_ir::MixedSlot {
                    index: 1,
                    kind: ax_ir::SlotKind::UndottedSpinor,
                },
            ],
            label: None,
        }],
    }
}

pub fn vector_as_bispinor_symmetry() -> ax_ir::MixedTensorSymmetry {
    ax_ir::MixedTensorSymmetry {
        tableaux: vec![ax_ir::MixedTableauAttachment {
            shape: vec![1, 1],
            slots: vec![
                ax_ir::MixedSlot {
                    index: 0,
                    kind: ax_ir::SlotKind::UndottedSpinor,
                },
                ax_ir::MixedSlot {
                    index: 0,
                    kind: ax_ir::SlotKind::DottedSpinor,
                },
            ],
            label: None,
        }],
    }
}

pub fn symmetric_rank2_tensor_plus_spinor(
    tensor_slots: [usize; 2],
    spinor_slots: [usize; 2],
) -> ax_ir::MixedTensorSymmetry {
    ax_ir::MixedTensorSymmetry {
        tableaux: vec![
            ax_ir::MixedTableauAttachment {
                shape: vec![2],
                slots: tensor_slots
                    .into_iter()
                    .map(|index| ax_ir::MixedSlot {
                        index,
                        kind: ax_ir::SlotKind::Tensor,
                    })
                    .collect(),
                label: Some("tensor".to_string()),
            },
            ax_ir::MixedTableauAttachment {
                shape: vec![2],
                slots: spinor_slots
                    .into_iter()
                    .map(|index| ax_ir::MixedSlot {
                        index,
                        kind: ax_ir::SlotKind::UndottedSpinor,
                    })
                    .collect(),
                label: Some("spinor".to_string()),
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_two_undotted_has_shape_two() {
        assert_eq!(symmetric_two_undotted_spinors().tableaux[0].shape, vec![2]);
    }

    #[test]
    fn antisymmetric_two_undotted_has_column_shape() {
        assert_eq!(
            antisymmetric_two_undotted_spinors().tableaux[0].shape,
            vec![1, 1]
        );
    }

    #[test]
    fn vector_as_bispinor_validates_with_both_chiralities() {
        let symmetry = vector_as_bispinor_symmetry();
        assert_eq!(ax_ir::validate_mixed_tensor_symmetry(&symmetry), Ok(()));
        let slots = &symmetry.tableaux[0].slots;
        assert!(slots
            .iter()
            .any(|slot| slot.kind == ax_ir::SlotKind::UndottedSpinor));
        assert!(slots
            .iter()
            .any(|slot| slot.kind == ax_ir::SlotKind::DottedSpinor));
    }

    #[test]
    fn symmetric_rank2_tensor_plus_spinor_emits_labeled_tableaux() {
        let symmetry = symmetric_rank2_tensor_plus_spinor([0, 1], [0, 1]);
        assert_eq!(symmetry.tableaux.len(), 2);
        assert_eq!(symmetry.tableaux[0].label.as_deref(), Some("tensor"));
        assert_eq!(symmetry.tableaux[1].label.as_deref(), Some("spinor"));
    }
}
