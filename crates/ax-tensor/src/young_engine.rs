use crate::symmetry_bridge::RealizedTableau;
use ax_ir::{Expr, Index};
use ax_perm::{enumerate_subgroup, product, sign};
use lasso::Key;
use num_rational::BigRational;
use num_traits::One;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum YoungEngineError {
    #[error("Young engine bridge failure: {0}")]
    Bridge(#[from] crate::symmetry_bridge::SymmetryBridgeError),
    #[error("Young engine slot count mismatch: expected {expected}, got {actual}")]
    SlotCountMismatch { expected: usize, actual: usize },
    #[error("Young engine only supports indexed tensor factors and multiplicative products on this path")]
    UnsupportedExpr,
    #[error("Young engine exceeded max term limit {max_terms}")]
    ExceededMaxTerms { max_terms: usize },
}

pub fn projector_slots_for_indexed_tensor(expr: &Expr) -> Result<Vec<usize>, YoungEngineError> {
    match expr {
        Expr::Indexed(_, indices) => Ok((0..indices.len()).collect()),
        _ => Err(YoungEngineError::UnsupportedExpr),
    }
}

pub fn canonicalize_factor_slots_with_projector(
    expr: &Expr,
    projector: &ax_young::GroupBackedProjector,
) -> Result<Expr, YoungEngineError> {
    let slots = projector_slots_for_indexed_tensor(expr)?;
    if projector.row_group.degree != slots.len() {
        return Err(YoungEngineError::SlotCountMismatch {
            expected: projector.row_group.degree,
            actual: slots.len(),
        });
    }
    let canonical =
        ax_young::canonicalize_slots_under_both_groups(projector, &slots).map_err(|_| {
            YoungEngineError::SlotCountMismatch {
                expected: projector.row_group.degree,
                actual: slots.len(),
            }
        })?;
    permute_factor_indices(expr, &canonical)
}

pub fn apply_realized_tableaux_to_factor(
    expr: &Expr,
    tableaux: &[RealizedTableau],
    max_terms: usize,
) -> Result<Expr, YoungEngineError> {
    if tableaux.is_empty() {
        return Ok(expr.clone());
    }

    let mut current = expr.clone();
    for tableau in tableaux {
        current = canonicalize_factor_with_realized_tableau(&current, tableau)?;
        if max_terms > 0 && additive_term_count(&current) > max_terms {
            return Err(YoungEngineError::ExceededMaxTerms { max_terms });
        }
    }

    Ok(current)
}

fn canonicalize_factor_with_realized_tableau(
    expr: &Expr,
    realized: &RealizedTableau,
) -> Result<Expr, YoungEngineError> {
    let Expr::Indexed(base, indices) = expr else {
        return Err(YoungEngineError::UnsupportedExpr);
    };
    if realized.projector.row_group.degree != realized.slot_map.len() {
        return Err(YoungEngineError::SlotCountMismatch {
            expected: realized.projector.row_group.degree,
            actual: realized.slot_map.len(),
        });
    }
    if realized.slot_map.iter().any(|slot| *slot >= indices.len()) {
        return Err(YoungEngineError::SlotCountMismatch {
            expected: realized.projector.row_group.degree,
            actual: indices.len(),
        });
    }

    let ranking = slot_ranking(indices, &realized.slot_map);
    let (canonical_ranking, coefficient) =
        canonical_ranking_and_sign(&realized.projector, &ranking).map_err(|_| {
            YoungEngineError::SlotCountMismatch {
                expected: realized.projector.row_group.degree,
                actual: realized.slot_map.len(),
            }
        })?;

    let mut source_slots = Vec::with_capacity(canonical_ranking.len());
    for label in canonical_ranking {
        let source = ranking
            .iter()
            .position(|candidate| *candidate == label)
            .ok_or(YoungEngineError::UnsupportedExpr)?;
        source_slots.push(realized.slot_map[source]);
    }

    let mut rewritten = indices.clone();
    for (dest_slot, source_slot) in realized.slot_map.iter().zip(source_slots.iter()) {
        rewritten[*dest_slot] = indices[*source_slot].clone();
    }

    let rewritten = Expr::Indexed(base.clone(), rewritten);
    if coefficient == BigRational::one() {
        Ok(rewritten)
    } else {
        Ok(crate::multiply_expr_by_rational(rewritten, coefficient))
    }
}

fn permute_factor_indices(
    expr: &Expr,
    canonical_slots: &[usize],
) -> Result<Expr, YoungEngineError> {
    let Expr::Indexed(base, indices) = expr else {
        return Err(YoungEngineError::UnsupportedExpr);
    };
    if canonical_slots.len() != indices.len() {
        return Err(YoungEngineError::SlotCountMismatch {
            expected: canonical_slots.len(),
            actual: indices.len(),
        });
    }
    let rewritten = canonical_slots
        .iter()
        .map(|slot| indices[*slot].clone())
        .collect();
    Ok(Expr::Indexed(base.clone(), rewritten))
}

fn slot_ranking(indices: &[Index], slot_map: &[usize]) -> Vec<usize> {
    let mut keyed = slot_map
        .iter()
        .enumerate()
        .map(|(order, slot)| {
            let index = &indices[*slot];
            (
                order,
                (
                    index.name.into_usize(),
                    variance_rank(&index.variance),
                    index
                        .index_type
                        .map(|sym| sym.into_usize())
                        .unwrap_or(usize::MAX),
                    *slot,
                ),
            )
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    let mut ranking = vec![0usize; slot_map.len()];
    for (label, (original_order, _)) in keyed.into_iter().enumerate() {
        ranking[original_order] = label;
    }
    ranking
}

fn variance_rank(variance: &ax_ir::Variance) -> usize {
    match variance {
        ax_ir::Variance::Up => 0,
        ax_ir::Variance::Down => 1,
    }
}

fn additive_term_count(expr: &Expr) -> usize {
    match expr {
        Expr::Add(terms) => terms.iter().map(additive_term_count).sum(),
        _ => 1,
    }
}

fn canonical_ranking_and_sign(
    projector: &ax_young::GroupBackedProjector,
    ranking: &[usize],
) -> Result<(Vec<usize>, BigRational), ax_young::GroupProjectorError> {
    let row_elements =
        enumerate_subgroup(&projector.row_group.generators, projector.row_group.degree);
    let column_elements = enumerate_subgroup(
        &projector.column_group.generators,
        projector.column_group.degree,
    );

    let mut best = ranking.to_vec();
    let mut coeff = BigRational::one();
    for row in &row_elements {
        for column in &column_elements {
            let composed = product(column, row);
            let candidate = (0..ranking.len())
                .map(|idx| ranking[composed[idx]])
                .collect::<Vec<_>>();
            let candidate_coeff = BigRational::from_integer(sign(column).into());
            if candidate < best || (candidate == best && candidate_coeff < coeff) {
                best = candidate;
                coeff = candidate_coeff;
            }
        }
    }
    Ok((best, coeff))
}
