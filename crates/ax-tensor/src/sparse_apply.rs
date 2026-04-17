use crate::{
    symmetry_bridge::realized_tableaux_from_symmetry,
    young_engine::{
        projector_slots_for_indexed_tensor, rewrite_indexed_factor_by_slots, YoungEngineError,
    },
};
use ax_ir::{Expr, Index, TensorProperty, Variance};
use ax_perm::{enumerate_subgroup, product, sign};
use ax_trace::SparseProjectorTrace;
use ax_young::{
    build_group_backed_projector,
    group_action::GroupBackedProjector,
    sparse_projector::{
        apply_sparse_plan_to_slots, build_sparse_projector_plan, sparse_plan_cache_key,
        SparseProjectorError, SparseProjectorPlan,
    },
    ProjectorNormalization,
};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SparseApplyError {
    #[error("sparse tensor projection failed: {0}")]
    Sparse(#[from] SparseProjectorError),
    #[error("sparse tensor projection supports only single indexed tensor factors on this path")]
    UnsupportedExpr,
}

#[derive(Clone, Debug)]
pub struct SparseProjectionResult {
    pub exprs: Vec<(Expr, BigRational)>,
    pub trace: SparseProjectorTrace,
}

pub fn sparse_project_indexed_factor(
    expr: &Expr,
    projector: &GroupBackedProjector,
    max_terms: usize,
) -> Result<SparseProjectionResult, SparseApplyError> {
    let slots = projector_slots_for_indexed_tensor(expr).map_err(map_engine_error)?;
    let plan = cached_sparse_projector_plan(projector)?;
    let (terms, trace) = apply_sparse_plan_to_slots(&plan, &slots, max_terms)?;

    let mut merged = HashMap::<String, (Expr, BigRational)>::new();
    for (slot_order, coefficient) in terms {
        let permuted =
            rewrite_indexed_factor_by_slots(expr, &slot_order).map_err(map_engine_error)?;
        let (canonical, canonical_coefficient) =
            canonicalize_indexed_expr_under_projector(&permuted, projector)?;
        let total_coefficient = coefficient * canonical_coefficient;
        let key = stable_expr_term_key(&canonical)?;
        let entry = merged
            .entry(key)
            .or_insert_with(|| (canonical.clone(), BigRational::zero()));
        entry.1 += total_coefficient;
    }

    let mut exprs = merged
        .into_values()
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .collect::<Vec<_>>();
    exprs.sort_by(|lhs, rhs| stable_expr_sort_key(&lhs.0).cmp(&stable_expr_sort_key(&rhs.0)));

    Ok(SparseProjectionResult { exprs, trace })
}

pub fn sparse_project_tensor_with_options(
    expr: &Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<TensorProperty>,
    max_terms: usize,
) -> Result<Option<SparseProjectionResult>, SparseApplyError> {
    let Expr::Indexed(base, _) = expr else {
        return Ok(None);
    };
    let Expr::Sym(symbol) = base.as_ref() else {
        return Ok(None);
    };
    let properties = properties_for_symbol(*symbol);
    let Some(realized) = first_structured_realized_tableau(&properties) else {
        return Ok(None);
    };

    sparse_project_indexed_factor(expr, &realized.projector, max_terms).map(Some)
}

fn first_structured_realized_tableau(
    properties: &[TensorProperty],
) -> Option<crate::symmetry_bridge::RealizedTableau> {
    properties.iter().find_map(|property| match property {
        TensorProperty::TableauSymmetry(symmetry) => realized_tableaux_from_symmetry(symmetry)
            .ok()
            .and_then(|mut realized| realized.drain(..1).next())
            .and_then(|mut realized| {
                let normalized = build_group_backed_projector(
                    &realized.projector.tableau,
                    ProjectorNormalization::HookLength,
                )
                .ok()?;
                realized.projector = normalized;
                Some(realized)
            }),
        _ => None,
    })
}

fn cached_sparse_projector_plan(
    projector: &GroupBackedProjector,
) -> Result<SparseProjectorPlan, SparseProjectorError> {
    static CACHE: OnceLock<Mutex<HashMap<String, SparseProjectorPlan>>> = OnceLock::new();

    let key = sparse_plan_cache_key(projector);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(plan) = guard.get(&key) {
            return Ok(plan.clone());
        }
    }

    let plan = build_sparse_projector_plan(projector)?;
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, plan.clone());
    }
    Ok(plan)
}

fn stable_expr_term_key(expr: &Expr) -> Result<String, SparseProjectorError> {
    match expr {
        Expr::Indexed(base, indices) => Ok(format!("base={base:?};indices={indices:?}")),
        _ => Err(SparseProjectorError::TermKeyConstructionFailed),
    }
}

fn stable_expr_sort_key(expr: &Expr) -> String {
    match stable_expr_term_key(expr) {
        Ok(key) => key,
        Err(_) => format!("{expr:?}"),
    }
}

fn map_engine_error(error: YoungEngineError) -> SparseApplyError {
    match error {
        YoungEngineError::UnsupportedExpr => SparseApplyError::UnsupportedExpr,
        YoungEngineError::SlotCountMismatch { expected, actual } => {
            SparseApplyError::Sparse(SparseProjectorError::Group(
                ax_young::GroupProjectorError::DegreeMismatch { expected, actual },
            ))
        }
        YoungEngineError::ExceededMaxTerms { max_terms } => {
            SparseApplyError::Sparse(SparseProjectorError::BudgetExceeded { max_terms })
        }
        YoungEngineError::Bridge(_) => SparseApplyError::UnsupportedExpr,
    }
}

fn canonicalize_indexed_expr_under_projector(
    expr: &Expr,
    projector: &GroupBackedProjector,
) -> Result<(Expr, BigRational), SparseApplyError> {
    let Expr::Indexed(base, indices) = expr else {
        return Err(SparseApplyError::UnsupportedExpr);
    };
    if projector.row_group.degree != indices.len() {
        return Err(SparseApplyError::Sparse(SparseProjectorError::Group(
            ax_young::GroupProjectorError::DegreeMismatch {
                expected: projector.row_group.degree,
                actual: indices.len(),
            },
        )));
    }

    let slot_map = (0..indices.len()).collect::<Vec<_>>();
    let ranking = slot_ranking(indices, &slot_map);
    let (canonical_ranking, coefficient) = canonical_ranking_and_sign(projector, &ranking)?;

    let mut source_slots = Vec::with_capacity(canonical_ranking.len());
    for label in canonical_ranking {
        let Some(source) = ranking.iter().position(|candidate| *candidate == label) else {
            return Err(SparseApplyError::UnsupportedExpr);
        };
        source_slots.push(source);
    }

    let rewritten = source_slots
        .into_iter()
        .map(|source_slot| indices[source_slot].clone())
        .collect::<Vec<_>>();
    Ok((Expr::Indexed(base.clone(), rewritten), coefficient))
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
                    lasso::Key::into_usize(index.name),
                    variance_rank(&index.variance),
                    index
                        .index_type
                        .map(lasso::Key::into_usize)
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

fn variance_rank(variance: &Variance) -> usize {
    match variance {
        Variance::Up => 0,
        Variance::Down => 1,
    }
}

fn canonical_ranking_and_sign(
    projector: &GroupBackedProjector,
    ranking: &[usize],
) -> Result<(Vec<usize>, BigRational), SparseApplyError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Index, TensorSymmetry, Variance};
    use ax_young::{
        build_group_backed_projector, ProjectorNormalization, YoungDiagram, YoungTableau,
    };
    use std::collections::HashMap;

    fn declared_tableau_symmetry(shape: Vec<usize>, slot_map: Vec<usize>) -> TensorProperty {
        TensorProperty::TableauSymmetry(TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape,
                slot_map,
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        })
    }

    fn indexed_factor(symbol: lasso::Spur, names: &[lasso::Spur]) -> Expr {
        Expr::Indexed(
            Box::new(Expr::Sym(symbol)),
            names
                .iter()
                .map(|name| Index {
                    name: *name,
                    variance: Variance::Down,
                    index_type: None,
                })
                .collect(),
        )
    }

    fn canonical_projector(shape: Vec<usize>) -> GroupBackedProjector {
        let diagram = YoungDiagram::try_new(shape).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        build_group_backed_projector(&tableau, ProjectorNormalization::HookLength).unwrap()
    }

    #[test]
    fn sparse_projection_of_symmetric_factor_emits_single_canonical_term() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let expr = indexed_factor(t, &[b, a]);
        let projector = canonical_projector(vec![2]);

        let result = sparse_project_indexed_factor(&expr, &projector, 8).unwrap();

        assert_eq!(result.exprs.len(), 1);
        assert_eq!(result.exprs[0].1, BigRational::from_integer(1.into()));
        let Expr::Indexed(_, indices) = &result.exprs[0].0 else {
            panic!("expected indexed factor");
        };
        assert_eq!(indices[0].name, a);
        assert_eq!(indices[1].name, b);
        assert!(!result.trace.dropped_due_to_budget);
    }

    #[test]
    fn sparse_projection_of_rank_three_shape_is_deterministic() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let expr = indexed_factor(t, &[c, a, b]);
        let projector = canonical_projector(vec![2, 1]);

        let first = sparse_project_indexed_factor(&expr, &projector, 16).unwrap();
        let second = sparse_project_indexed_factor(&expr, &projector, 16).unwrap();

        assert_eq!(first.exprs, second.exprs);
        assert_eq!(
            first.trace.explored_permutation_count,
            second.trace.explored_permutation_count
        );
    }

    #[test]
    fn sparse_projection_returns_none_without_structured_symmetry() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let expr = indexed_factor(t, &[b, a]);

        let result = sparse_project_tensor_with_options(&expr, &|_| vec![], 8).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn sparse_preferred_dispatcher_falls_back_without_structured_symmetry() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let expr = indexed_factor(t, &[b, a]);

        let projected = crate::project_tensor_preferring_sparse(&expr, &|_| vec![], 8).unwrap();

        assert_eq!(projected, expr);
    }

    #[test]
    fn sparse_projection_rejects_non_factor_expressions() {
        let expr = Expr::add(vec![Expr::one(), Expr::one()]);
        let projector = canonical_projector(vec![2]);

        let error = sparse_project_indexed_factor(&expr, &projector, 8).unwrap_err();
        assert!(matches!(error, SparseApplyError::UnsupportedExpr));
    }

    #[test]
    fn sparse_projection_uses_first_structured_tableau() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let expr = indexed_factor(t, &[b, a]);
        let mut props = HashMap::new();
        props.insert(t, vec![declared_tableau_symmetry(vec![2], vec![0, 1])]);

        let result = sparse_project_tensor_with_options(
            &expr,
            &|symbol| props.get(&symbol).cloned().unwrap_or_default(),
            8,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.exprs.len(), 1);
        assert_eq!(result.exprs[0].1, BigRational::from_integer(1.into()));
    }
}
