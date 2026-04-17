use crate::{symmetry_bridge::realized_tableaux_from_symmetry, young_engine};
use ax_ir::{Expr, Index};
use lasso::Key;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GradedYoungError {
    #[error("graded Young engine failed: {0}")]
    Young(#[from] ax_young::YoungError),
    #[error("graded Young engine supports only indexed tensor factors on this path")]
    UnsupportedExpr,
    #[error("graded Young engine requires slot parity metadata")]
    MissingParityMetadata,
}

pub fn canonicalize_factor_slots_with_graded_projector(
    expr: &ax_ir::Expr,
    projector: &ax_young::group_action::GroupBackedProjector,
    parity: &ax_young::graded::SlotParity,
) -> Result<(ax_ir::Expr, i32), GradedYoungError> {
    let ranking = match expr {
        Expr::Indexed(_, indices) => slot_ranking(indices),
        _ => return Err(GradedYoungError::UnsupportedExpr),
    };
    let (canonical_ranking, sign) = ax_young::graded::canonicalize_slots_under_graded_projector(
        projector,
        &ranking,
        parity,
    )?;

    let mut canonical_slots = Vec::with_capacity(canonical_ranking.len());
    for label in canonical_ranking {
        let source_slot = ranking
            .iter()
            .position(|candidate| *candidate == label)
            .ok_or(GradedYoungError::UnsupportedExpr)?;
        canonical_slots.push(source_slot);
    }

    let rewritten = young_engine::rewrite_indexed_factor_by_slots(expr, &canonical_slots)
        .map_err(|_| GradedYoungError::UnsupportedExpr)?;
    Ok((rewritten, sign))
}

pub fn try_extract_slot_parity_from_properties(
    properties: &[ax_ir::TensorProperty],
) -> Option<ax_young::graded::SlotParity> {
    properties.iter().find_map(|property| match property {
        ax_ir::TensorProperty::GradedParity(values) => {
            ax_young::graded::SlotParity::try_new(values.clone()).ok()
        }
        _ => None,
    })
}

pub(crate) fn first_realized_tableau(
    properties: &[ax_ir::TensorProperty],
) -> Option<crate::symmetry_bridge::RealizedTableau> {
    properties.iter().find_map(|property| match property {
        ax_ir::TensorProperty::TableauSymmetry(symmetry) => {
            realized_tableaux_from_symmetry(symmetry).ok()?.into_iter().next()
        }
        _ => None,
    })
}

fn slot_ranking(indices: &[Index]) -> Vec<usize> {
    let mut keyed = indices
        .iter()
        .enumerate()
        .map(|(slot, index)| {
            (
                slot,
                (
                    index.name.into_usize(),
                    variance_rank(&index.variance),
                    index
                        .index_type
                        .map(|symbol| symbol.into_usize())
                        .unwrap_or(usize::MAX),
                    slot,
                ),
            )
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    let mut ranking = vec![0usize; indices.len()];
    for (label, (slot, _)) in keyed.into_iter().enumerate() {
        ranking[slot] = label;
    }
    ranking
}

fn variance_rank(variance: &ax_ir::Variance) -> usize {
    match variance {
        ax_ir::Variance::Up => 0,
        ax_ir::Variance::Down => 1,
    }
}
