use crate::partition::YoungDiagram;

pub fn vanishes_in_gl_dimension(shape: &YoungDiagram, dim: usize) -> bool {
    shape.n_rows() > dim
}

pub fn first_nonvanishing_gl_dimension(shape: &YoungDiagram) -> usize {
    shape.n_rows()
}

pub fn schouten_annihilates_antisym_degree(antisym_degree: usize, dim: usize) -> bool {
    antisym_degree > dim
}

pub fn tableau_requires_dimension_annihilation(
    attachment: &ax_ir::TableauAttachment,
    dim: usize,
) -> bool {
    if attachment
        .dimension_guard
        .as_ref()
        .is_some_and(|guard| !guard.allows(Some(dim)))
    {
        return true;
    }
    attachment.shape.len() > dim
}

pub fn tensor_symmetry_annihilates_in_dimension(sym: &ax_ir::TensorSymmetry, dim: usize) -> bool {
    sym.tableaux.iter().any(|attachment| {
        tableau_requires_dimension_annihilation(attachment, dim)
            || ax_ir::validate_duality_in_dimension(attachment, Some(dim)).is_err()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{
        DimensionGuard, DualityKind, RestrictedSymmetryMode, SymmetrySource, TableauAttachment,
        TensorSymmetry,
    };

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    fn attachment(shape: Vec<usize>) -> TableauAttachment {
        TableauAttachment {
            slot_map: (0..shape.iter().sum()).collect(),
            shape,
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: DualityKind::None,
            restricted_mode: RestrictedSymmetryMode::FullYoung,
            trace_free: false,
            dimension_guard: None,
            source: SymmetrySource::Declared,
            label: None,
        }
    }

    #[test]
    fn gl_vanishing_matches_required_cases() {
        assert!(vanishes_in_gl_dimension(&yd(vec![1, 1, 1]), 2));
        assert!(!vanishes_in_gl_dimension(&yd(vec![2, 1]), 2));
        assert_eq!(first_nonvanishing_gl_dimension(&yd(vec![3, 2, 1])), 3);
    }

    #[test]
    fn schouten_antisymmetry_threshold_is_exact() {
        assert!(schouten_annihilates_antisym_degree(5, 4));
        assert!(!schouten_annihilates_antisym_degree(4, 4));
    }

    #[test]
    fn tensor_symmetry_annihilation_uses_guard_and_duality_validation() {
        let mut guarded = attachment(vec![2]);
        guarded.dimension_guard = Some(DimensionGuard {
            min_dimension: None,
            max_dimension: Some(3),
            exact_dimension: None,
        });
        assert!(tableau_requires_dimension_annihilation(&guarded, 4));

        let selfdual = TableauAttachment {
            duality: DualityKind::SelfDual,
            slot_map: vec![0],
            ..attachment(vec![1])
        };
        let sym = TensorSymmetry {
            tableaux: vec![selfdual],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };
        assert!(tensor_symmetry_annihilates_in_dimension(&sym, 4));
    }
}
