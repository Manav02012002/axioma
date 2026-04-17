use crate::{error::YoungError, partition::YoungDiagram};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

pub fn hodge_dual_form_degree(rank: usize, dim: usize) -> usize {
    dim - rank
}

pub fn is_middle_degree(rank: usize, dim: usize) -> bool {
    2 * rank == dim
}

pub fn selfdual_eigenspace_dimension(rank: usize, dim: usize) -> Result<(u64, u64), YoungError> {
    if dim % 2 != 0 || !is_middle_degree(rank, dim) {
        return Err(YoungError::InvalidSelfDualDimension { rank, dim });
    }

    let total = binomial(dim, rank)
        .to_u64()
        .ok_or(YoungError::InvalidSelfDualDimension { rank, dim })?;
    let half = total / 2;
    Ok((half, half))
}

pub fn induced_form_tableau_duality(
    rank: usize,
    dim: usize,
    duality: ax_ir::DualityKind,
) -> Result<ax_ir::TensorSymmetry, YoungError> {
    let shape = YoungDiagram::try_new(vec![1; rank])?;
    let attachment = ax_ir::TableauAttachment {
        shape: shape.rows.clone(),
        slot_map: (0..rank).collect(),
        multiplicity_numer: 1,
        multiplicity_denom: 1,
        duality,
        restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
        trace_free: false,
        dimension_guard: None,
        source: ax_ir::SymmetrySource::Declared,
        label: None,
    };
    ax_ir::validate_duality_in_dimension(&attachment, Some(dim))
        .map_err(|_| YoungError::InvalidSelfDualDimension { rank, dim })?;
    Ok(ax_ir::TensorSymmetry {
        tableaux: vec![attachment],
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    })
}

fn binomial(n: usize, k: usize) -> BigUint {
    let k = k.min(n.saturating_sub(k));
    let mut acc = BigUint::one();
    for step in 0..k {
        acc *= BigUint::from(n - step);
        acc /= BigUint::from(step + 1usize);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hodge_degree_and_middle_degree_match_exact_rules() {
        assert_eq!(hodge_dual_form_degree(2, 4), 2);
        assert!(is_middle_degree(2, 4));
    }

    #[test]
    fn selfdual_eigenspace_dimension_is_exact_for_four_dimensions() {
        assert_eq!(selfdual_eigenspace_dimension(2, 4), Ok((3, 3)));
        assert_eq!(
            selfdual_eigenspace_dimension(1, 4),
            Err(YoungError::InvalidSelfDualDimension { rank: 1, dim: 4 })
        );
    }
}
