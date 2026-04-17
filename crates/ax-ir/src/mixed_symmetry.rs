use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SlotKind {
    Tensor,
    UndottedSpinor,
    DottedSpinor,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MixedSlot {
    pub index: usize,
    pub kind: SlotKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedTableauAttachment {
    pub shape: Vec<usize>,
    pub slots: Vec<MixedSlot>,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedTensorSymmetry {
    pub tableaux: Vec<MixedTableauAttachment>,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum MixedSymmetryError {
    #[error("mixed tensor symmetry tableau shape cannot be empty")]
    EmptyMixedShape,
    #[error("mixed tensor symmetry shape cell count {shape_cells} does not match mixed slot count {slot_count}")]
    MixedShapeSlotCountMismatch {
        shape_cells: usize,
        slot_count: usize,
    },
    #[error("mixed tensor symmetry contains duplicate slot {index} of kind {kind:?}")]
    DuplicateMixedSlot { index: usize, kind: SlotKind },
    #[error("mixed tensor symmetry rows must be weakly decreasing: {shape:?}")]
    MixedRowsNotWeaklyDecreasing { shape: Vec<usize> },
    #[error("mixed tensor product cannot be decomposed on this path")]
    InvalidMixedTensorProduct,
}

pub fn validate_mixed_tensor_symmetry(
    sym: &MixedTensorSymmetry,
) -> Result<(), MixedSymmetryError> {
    for tableau in &sym.tableaux {
        if tableau.shape.is_empty() {
            return Err(MixedSymmetryError::EmptyMixedShape);
        }
        if tableau.shape.windows(2).any(|rows| rows[0] < rows[1]) {
            return Err(MixedSymmetryError::MixedRowsNotWeaklyDecreasing {
                shape: tableau.shape.clone(),
            });
        }
        let shape_cells = tableau.shape.iter().sum::<usize>();
        if shape_cells != tableau.slots.len() {
            return Err(MixedSymmetryError::MixedShapeSlotCountMismatch {
                shape_cells,
                slot_count: tableau.slots.len(),
            });
        }

        let mut seen = std::collections::HashSet::new();
        for slot in &tableau.slots {
            if !seen.insert((slot.index, slot.kind.clone())) {
                return Err(MixedSymmetryError::DuplicateMixedSlot {
                    index: slot.index,
                    kind: slot.kind.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_undotted_pair_validates() {
        let symmetry = MixedTensorSymmetry {
            tableaux: vec![MixedTableauAttachment {
                shape: vec![2],
                slots: vec![
                    MixedSlot {
                        index: 0,
                        kind: SlotKind::UndottedSpinor,
                    },
                    MixedSlot {
                        index: 1,
                        kind: SlotKind::UndottedSpinor,
                    },
                ],
                label: None,
            }],
        };
        assert_eq!(validate_mixed_tensor_symmetry(&symmetry), Ok(()));
    }

    #[test]
    fn duplicate_index_and_kind_is_rejected() {
        let symmetry = MixedTensorSymmetry {
            tableaux: vec![MixedTableauAttachment {
                shape: vec![2],
                slots: vec![
                    MixedSlot {
                        index: 0,
                        kind: SlotKind::UndottedSpinor,
                    },
                    MixedSlot {
                        index: 0,
                        kind: SlotKind::UndottedSpinor,
                    },
                ],
                label: None,
            }],
        };
        assert_eq!(
            validate_mixed_tensor_symmetry(&symmetry),
            Err(MixedSymmetryError::DuplicateMixedSlot {
                index: 0,
                kind: SlotKind::UndottedSpinor,
            })
        );
    }

    #[test]
    fn same_index_across_spinor_chiralities_is_allowed() {
        let symmetry = MixedTensorSymmetry {
            tableaux: vec![MixedTableauAttachment {
                shape: vec![1, 1],
                slots: vec![
                    MixedSlot {
                        index: 0,
                        kind: SlotKind::UndottedSpinor,
                    },
                    MixedSlot {
                        index: 0,
                        kind: SlotKind::DottedSpinor,
                    },
                ],
                label: None,
            }],
        };
        assert_eq!(validate_mixed_tensor_symmetry(&symmetry), Ok(()));
    }
}
