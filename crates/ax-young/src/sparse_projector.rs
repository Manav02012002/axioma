use crate::{
    group_action::{
        expand_projector_group_algebra, validate_perm, GroupBackedProjector, GroupProjectorError,
        ProjectorNormalization,
    },
    projector::PermutationTerm,
};
use ax_trace::SparseProjectorTrace;
use num_rational::BigRational;
use num_traits::Zero;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseProjectorPlan {
    pub degree: usize,
    pub permutations: Vec<PermutationTerm>,
    pub normalized: bool,
}

#[derive(Error, Debug)]
pub enum SparseProjectorError {
    #[error("sparse projector planning failed: {0}")]
    Group(#[from] GroupProjectorError),
    #[error("sparse projector application exceeded max term budget {max_terms}")]
    BudgetExceeded { max_terms: usize },
    #[error("sparse projector could not construct a stable term key")]
    TermKeyConstructionFailed,
}

pub fn build_sparse_projector_plan(
    projector: &GroupBackedProjector,
) -> Result<SparseProjectorPlan, SparseProjectorError> {
    let mut permutations = expand_projector_group_algebra(projector)?;
    permutations.sort_by(|lhs, rhs| lhs.images.cmp(&rhs.images));
    Ok(SparseProjectorPlan {
        degree: projector.row_group.degree,
        permutations,
        normalized: projector.normalization != ProjectorNormalization::Unnormalized,
    })
}

pub fn apply_sparse_plan_to_slots(
    plan: &SparseProjectorPlan,
    slots: &[usize],
    max_terms: usize,
) -> Result<(Vec<(Vec<usize>, BigRational)>, SparseProjectorTrace), SparseProjectorError> {
    if max_terms == 0 {
        return Err(SparseProjectorError::BudgetExceeded { max_terms });
    }
    if slots.len() != plan.degree {
        return Err(SparseProjectorError::Group(
            GroupProjectorError::DegreeMismatch {
                expected: plan.degree,
                actual: slots.len(),
            },
        ));
    }

    let mut merged = BTreeMap::<Vec<usize>, BigRational>::new();
    for term in &plan.permutations {
        validate_perm(&term.images, plan.degree)?;
        let permuted = term
            .images
            .iter()
            .map(|&image| slots[image])
            .collect::<Vec<_>>();
        if !merged.contains_key(&permuted) && merged.len() >= max_terms {
            return Err(SparseProjectorError::BudgetExceeded { max_terms });
        }
        let entry = merged
            .entry(permuted)
            .or_insert_with(BigRational::zero);
        *entry += term.coefficient.clone();
    }

    let terms = merged
        .into_iter()
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .collect::<Vec<_>>();
    let emitted_term_count = terms.len();
    let trace = SparseProjectorTrace {
        input_term_count: 1,
        explored_permutation_count: plan.permutations.len(),
        emitted_term_count,
        merged_term_count: plan.permutations.len().saturating_sub(emitted_term_count),
        dropped_due_to_budget: false,
    };

    Ok((terms, trace))
}

pub fn sparse_plan_cache_key(projector: &GroupBackedProjector) -> String {
    format!(
        "shape={:?};tableau={:?};norm={:?}",
        projector.diagram.rows, projector.tableau.rows, projector.normalization
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_group_backed_projector, YoungDiagram, YoungTableau};
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn rational(numer: i64, denom: i64) -> BigRational {
        BigRational::new(BigInt::from(numer), BigInt::from(denom))
    }

    #[test]
    fn sparse_plan_key_uses_stable_format() {
        let diagram = YoungDiagram::try_new(vec![2]).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        let projector =
            build_group_backed_projector(&tableau, ProjectorNormalization::Unnormalized).unwrap();

        assert_eq!(
            sparse_plan_cache_key(&projector),
            "shape=[2];tableau=[[0, 1]];norm=Unnormalized"
        );
    }

    #[test]
    fn symmetric_plan_applies_deterministically_to_slots() {
        let diagram = YoungDiagram::try_new(vec![2]).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        let projector =
            build_group_backed_projector(&tableau, ProjectorNormalization::HookLength).unwrap();
        let plan = build_sparse_projector_plan(&projector).unwrap();

        let (terms, trace) = apply_sparse_plan_to_slots(&plan, &[9, 3], 8).unwrap();

        assert_eq!(
            terms,
            vec![
                (vec![3, 9], rational(1, 2)),
                (vec![9, 3], rational(1, 2)),
            ]
        );
        assert_eq!(trace.explored_permutation_count, 2);
        assert_eq!(trace.emitted_term_count, 2);
    }

    #[test]
    fn antisymmetric_plan_applies_deterministically_to_slots() {
        let diagram = YoungDiagram::try_new(vec![1, 1]).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        let projector =
            build_group_backed_projector(&tableau, ProjectorNormalization::HookLength).unwrap();
        let plan = build_sparse_projector_plan(&projector).unwrap();

        let (terms, trace) = apply_sparse_plan_to_slots(&plan, &[9, 3], 8).unwrap();

        assert_eq!(
            terms,
            vec![
                (vec![3, 9], rational(-1, 2)),
                (vec![9, 3], rational(1, 2)),
            ]
        );
        assert_eq!(trace.explored_permutation_count, 2);
        assert_eq!(trace.emitted_term_count, 2);
    }

    #[test]
    fn sparse_plan_respects_budget() {
        let diagram = YoungDiagram::try_new(vec![2]).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        let projector =
            build_group_backed_projector(&tableau, ProjectorNormalization::Unnormalized).unwrap();
        let plan = build_sparse_projector_plan(&projector).unwrap();

        let error = apply_sparse_plan_to_slots(&plan, &[9, 3], 0).unwrap_err();
        assert!(matches!(
            error,
            SparseProjectorError::BudgetExceeded { max_terms: 0 }
        ));
    }
}
