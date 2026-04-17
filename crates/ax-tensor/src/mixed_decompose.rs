use ax_ir::{MixedSymmetryError, MixedTensorSymmetry, SlotKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedIrrepSummary {
    pub tensor_shapes: Vec<Vec<usize>>,
    pub undotted_spinor_shapes: Vec<Vec<usize>>,
    pub dotted_spinor_shapes: Vec<Vec<usize>>,
}

pub fn summarize_mixed_tensor_symmetry(
    sym: &MixedTensorSymmetry,
) -> Result<MixedIrrepSummary, MixedSymmetryError> {
    ax_ir::validate_mixed_tensor_symmetry(sym)?;

    let mut summary = MixedIrrepSummary {
        tensor_shapes: Vec::new(),
        undotted_spinor_shapes: Vec::new(),
        dotted_spinor_shapes: Vec::new(),
    };

    for tableau in &sym.tableaux {
        let has_tensor = tableau.slots.iter().any(|slot| slot.kind == SlotKind::Tensor);
        let has_undotted = tableau
            .slots
            .iter()
            .any(|slot| slot.kind == SlotKind::UndottedSpinor);
        let has_dotted = tableau
            .slots
            .iter()
            .any(|slot| slot.kind == SlotKind::DottedSpinor);

        if has_tensor && (has_undotted || has_dotted) {
            return Err(MixedSymmetryError::InvalidMixedTensorProduct);
        }
        if has_tensor {
            summary.tensor_shapes.push(tableau.shape.clone());
        } else if has_undotted && has_dotted {
            summary.undotted_spinor_shapes.push(tableau.shape.clone());
            summary.dotted_spinor_shapes.push(tableau.shape.clone());
        } else if has_undotted {
            summary.undotted_spinor_shapes.push(tableau.shape.clone());
        } else if has_dotted {
            summary.dotted_spinor_shapes.push(tableau.shape.clone());
        }
    }

    summary.tensor_shapes.sort();
    summary.undotted_spinor_shapes.sort();
    summary.dotted_spinor_shapes.sort();
    Ok(summary)
}

pub fn decompose_small_mixed_product(
    left: &MixedTensorSymmetry,
    right: &MixedTensorSymmetry,
) -> Result<Vec<MixedIrrepSummary>, MixedSymmetryError> {
    let left_summary = summarize_mixed_tensor_symmetry(left)?;
    let right_summary = summarize_mixed_tensor_symmetry(right)?;

    ensure_supported_regime(left, &left_summary)?;
    ensure_supported_regime(right, &right_summary)?;

    let tensor = decompose_sector(&left_summary.tensor_shapes, &right_summary.tensor_shapes)?;
    let undotted =
        decompose_sector(&left_summary.undotted_spinor_shapes, &right_summary.undotted_spinor_shapes)?;
    let dotted =
        decompose_sector(&left_summary.dotted_spinor_shapes, &right_summary.dotted_spinor_shapes)?;

    let mut out = Vec::new();
    for tensor_shapes in &tensor {
        for undotted_shapes in &undotted {
            for dotted_shapes in &dotted {
                out.push(MixedIrrepSummary {
                    tensor_shapes: tensor_shapes.clone(),
                    undotted_spinor_shapes: undotted_shapes.clone(),
                    dotted_spinor_shapes: dotted_shapes.clone(),
                });
            }
        }
    }

    out.sort_by(|lhs, rhs| {
        lhs.tensor_shapes
            .cmp(&rhs.tensor_shapes)
            .then_with(|| lhs.undotted_spinor_shapes.cmp(&rhs.undotted_spinor_shapes))
            .then_with(|| lhs.dotted_spinor_shapes.cmp(&rhs.dotted_spinor_shapes))
    });
    Ok(out)
}

fn ensure_supported_regime(
    original: &MixedTensorSymmetry,
    summary: &MixedIrrepSummary,
) -> Result<(), MixedSymmetryError> {
    ax_ir::validate_mixed_tensor_symmetry(original)?;

    if summary.tensor_shapes.len() > 1
        || summary.undotted_spinor_shapes.len() > 1
        || summary.dotted_spinor_shapes.len() > 1
    {
        return Err(MixedSymmetryError::InvalidMixedTensorProduct);
    }

    if summary
        .tensor_shapes
        .iter()
        .chain(summary.undotted_spinor_shapes.iter())
        .chain(summary.dotted_spinor_shapes.iter())
        .any(|shape| shape.iter().sum::<usize>() > 2)
    {
        return Err(MixedSymmetryError::InvalidMixedTensorProduct);
    }

    Ok(())
}

fn decompose_sector(
    left_shapes: &[Vec<usize>],
    right_shapes: &[Vec<usize>],
) -> Result<Vec<Vec<Vec<usize>>>, MixedSymmetryError> {
    match (left_shapes.first(), right_shapes.first()) {
        (None, None) => Ok(vec![Vec::new()]),
        (Some(shape), None) | (None, Some(shape)) => Ok(vec![vec![shape.clone()]]),
        (Some(left), Some(right)) => {
            let left_diagram = ax_young::YoungDiagram::try_new(left.clone())
                .map_err(|_| MixedSymmetryError::InvalidMixedTensorProduct)?;
            let right_diagram = ax_young::YoungDiagram::try_new(right.clone())
                .map_err(|_| MixedSymmetryError::InvalidMixedTensorProduct)?;
            let product = ax_young::schur_tensor_product(&left_diagram, &right_diagram)
                .map_err(|_| MixedSymmetryError::InvalidMixedTensorProduct)?;
            let mut out = product
                .support()
                .into_iter()
                .map(|shape| vec![shape.rows])
                .collect::<Vec<_>>();
            out.sort();
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_bispinor_duplicates_shape_into_both_spinor_sectors() {
        let summary = summarize_mixed_tensor_symmetry(&ax_spinor::vector_as_bispinor_symmetry())
            .unwrap();
        assert_eq!(summary.tensor_shapes, Vec::<Vec<usize>>::new());
        assert_eq!(summary.undotted_spinor_shapes, vec![vec![1, 1]]);
        assert_eq!(summary.dotted_spinor_shapes, vec![vec![1, 1]]);
    }

    #[test]
    fn decompose_small_symmetric_undotted_product_uses_exact_lr_shapes() {
        let left = ax_spinor::symmetric_two_undotted_spinors();
        let right = ax_spinor::symmetric_two_undotted_spinors();
        let result = decompose_small_mixed_product(&left, &right).unwrap();
        assert_eq!(
            result,
            vec![
                MixedIrrepSummary {
                    tensor_shapes: vec![],
                    undotted_spinor_shapes: vec![vec![2, 2]],
                    dotted_spinor_shapes: vec![],
                },
                MixedIrrepSummary {
                    tensor_shapes: vec![],
                    undotted_spinor_shapes: vec![vec![3, 1]],
                    dotted_spinor_shapes: vec![],
                },
                MixedIrrepSummary {
                    tensor_shapes: vec![],
                    undotted_spinor_shapes: vec![vec![4]],
                    dotted_spinor_shapes: vec![],
                },
            ]
        );
    }

    #[test]
    fn tensor_spinor_mixed_tableau_is_rejected_on_summary_path() {
        let symmetry = ax_ir::MixedTensorSymmetry {
            tableaux: vec![ax_ir::MixedTableauAttachment {
                shape: vec![2],
                slots: vec![
                    ax_ir::MixedSlot {
                        index: 0,
                        kind: ax_ir::SlotKind::Tensor,
                    },
                    ax_ir::MixedSlot {
                        index: 1,
                        kind: ax_ir::SlotKind::UndottedSpinor,
                    },
                ],
                label: None,
            }],
        };
        assert_eq!(
            summarize_mixed_tensor_symmetry(&symmetry),
            Err(MixedSymmetryError::InvalidMixedTensorProduct)
        );
    }
}
