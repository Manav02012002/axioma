use ax_ir::{Expr, Index, Variance};
use lasso::Key;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaPairingTerm {
    pub pairings: Vec<(usize, usize)>,
    pub coefficient: i64,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum EpsilonEngineError {
    #[error("epsilon contraction requires an ambient dimension")]
    DimensionRequired,
    #[error("epsilon contraction rank mismatch: left {left}, right {right}")]
    RankMismatch { left: usize, right: usize },
    #[error("epsilon engine currently supports only pairwise epsilon products on this path")]
    UnsupportedEpsilonProduct,
}

fn permutation_parity(perm: &[usize]) -> i64 {
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

fn next_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let mut pivot = values.len() - 2;
    while values[pivot] >= values[pivot + 1] {
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut swap_with = values.len() - 1;
    while values[swap_with] <= values[pivot] {
        swap_with -= 1;
    }
    values.swap(pivot, swap_with);
    values[pivot + 1..].reverse();
    true
}

fn factorial(n: usize) -> i64 {
    (1..=n).fold(1_i64, |acc, next| acc.saturating_mul(next as i64))
}

pub fn epsilon_epsilon_to_delta_terms(
    rank: usize,
) -> Result<Vec<DeltaPairingTerm>, EpsilonEngineError> {
    if rank == 0 {
        return Ok(vec![DeltaPairingTerm {
            pairings: Vec::new(),
            coefficient: 1,
        }]);
    }

    let mut permutation: Vec<usize> = (0..rank).collect();
    let mut terms = Vec::new();
    loop {
        terms.push(DeltaPairingTerm {
            pairings: (0..rank).map(|slot| (slot, permutation[slot])).collect(),
            coefficient: permutation_parity(&permutation),
        });
        if !next_permutation(&mut permutation) {
            break;
        }
    }
    terms.sort_by(|left, right| left.pairings.cmp(&right.pairings));
    Ok(terms)
}

pub fn partially_contracted_epsilon_epsilon(
    left_rank: usize,
    right_rank: usize,
    contracted: usize,
    dim: Option<usize>,
) -> Result<Vec<DeltaPairingTerm>, EpsilonEngineError> {
    let Some(dimension) = dim else {
        return Err(EpsilonEngineError::DimensionRequired);
    };
    if left_rank != right_rank {
        return Err(EpsilonEngineError::RankMismatch {
            left: left_rank,
            right: right_rank,
        });
    }
    if left_rank != dimension {
        return Err(EpsilonEngineError::RankMismatch {
            left: left_rank,
            right: dimension,
        });
    }
    if contracted > dimension {
        return Err(EpsilonEngineError::UnsupportedEpsilonProduct);
    }

    let free_rank = dimension - contracted;
    let overall_factor = factorial(contracted);
    let mut terms = epsilon_epsilon_to_delta_terms(free_rank)?;
    for term in &mut terms {
        term.coefficient *= overall_factor;
    }
    Ok(terms)
}

pub fn delta_compose(left: &[(usize, usize)], right: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut right_map = std::collections::BTreeMap::new();
    for &(source, target) in right {
        right_map.insert(source, target);
    }

    let mut out = Vec::new();
    let mut consumed_right = std::collections::BTreeSet::new();
    for &(source, target) in left {
        let resolved = right_map.get(&target).copied().unwrap_or(target);
        if resolved != source {
            out.push((source, resolved));
        }
        if right_map.contains_key(&target) {
            consumed_right.insert(target);
        }
    }

    for &(source, target) in right {
        if consumed_right.contains(&source) || source == target {
            continue;
        }
        out.push((source, target));
    }

    out.sort_unstable();
    out.dedup();
    out
}

fn index_key(index: &Index) -> (usize, u8, Option<usize>) {
    (
        index.name.into_usize(),
        match index.variance {
            Variance::Down => 0,
            Variance::Up => 1,
        },
        index.index_type.map(|kind| kind.into_usize()),
    )
}

fn classify_epsilon_pair(
    left: &[Index],
    right: &[Index],
) -> Option<(usize, Vec<Index>, Vec<Index>)> {
    if left.len() != right.len() {
        return None;
    }

    let mut matched_right = vec![false; right.len()];
    let mut contracted = 0usize;
    let mut left_free = Vec::new();
    let mut right_free = Vec::new();

    for left_index in left {
        if let Some((position, _)) = right.iter().enumerate().find(|(position, right_index)| {
            !matched_right[*position]
                && right_index.name == left_index.name
                && right_index.variance != left_index.variance
                && (right_index.index_type == left_index.index_type
                    || right_index.index_type.is_none()
                    || left_index.index_type.is_none())
        }) {
            matched_right[position] = true;
            contracted += 1;
        } else {
            left_free.push(left_index.clone());
        }
    }

    for (position, right_index) in right.iter().enumerate() {
        if !matched_right[position] {
            right_free.push(right_index.clone());
        }
    }

    left_free.sort_by_key(index_key);
    right_free.sort_by_key(index_key);
    Some((contracted, left_free, right_free))
}

pub fn expand_epsilon_pair_product(
    left: &[Index],
    right: &[Index],
    delta_sym: lasso::Spur,
    dim: Option<usize>,
) -> Result<Expr, EpsilonEngineError> {
    let Some((contracted, left_free, right_free)) = classify_epsilon_pair(left, right) else {
        return Err(EpsilonEngineError::UnsupportedEpsilonProduct);
    };

    let terms = partially_contracted_epsilon_epsilon(left.len(), right.len(), contracted, dim)?;
    let mut expanded_terms = Vec::new();
    for term in terms {
        let mut factors = Vec::new();
        if term.coefficient == -1 {
            factors.push(Expr::Int((-1).into()));
        } else if term.coefficient != 1 {
            factors.push(Expr::Int(term.coefficient.into()));
        }
        for (left_slot, right_slot) in &term.pairings {
            let Some(left_index) = left_free.get(*left_slot).cloned() else {
                return Err(EpsilonEngineError::UnsupportedEpsilonProduct);
            };
            let Some(right_index) = right_free.get(*right_slot).cloned() else {
                return Err(EpsilonEngineError::UnsupportedEpsilonProduct);
            };
            factors.push(Expr::Indexed(
                Box::new(Expr::Sym(delta_sym)),
                vec![left_index, right_index],
            ));
        }
        expanded_terms.push(if factors.is_empty() {
            Expr::one()
        } else {
            Expr::mul(factors)
        });
    }

    Ok(Expr::add(expanded_terms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_rank_one_pairing() {
        assert_eq!(
            epsilon_epsilon_to_delta_terms(1).unwrap(),
            vec![DeltaPairingTerm {
                pairings: vec![(0, 0)],
                coefficient: 1,
            }]
        );
    }

    #[test]
    fn epsilon_rank_two_pairing() {
        assert_eq!(
            epsilon_epsilon_to_delta_terms(2).unwrap(),
            vec![
                DeltaPairingTerm {
                    pairings: vec![(0, 0), (1, 1)],
                    coefficient: 1,
                },
                DeltaPairingTerm {
                    pairings: vec![(0, 1), (1, 0)],
                    coefficient: -1,
                },
            ]
        );
    }

    #[test]
    fn partial_contraction_in_three_dimensions() {
        assert_eq!(
            partially_contracted_epsilon_epsilon(3, 3, 2, Some(3)).unwrap(),
            vec![DeltaPairingTerm {
                pairings: vec![(0, 0)],
                coefficient: 2,
            }]
        );
    }

    #[test]
    fn partial_contraction_requires_dimension() {
        assert_eq!(
            partially_contracted_epsilon_epsilon(3, 3, 2, None),
            Err(EpsilonEngineError::DimensionRequired)
        );
    }
}
