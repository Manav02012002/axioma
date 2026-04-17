use crate::{
    group_action::{expand_projector_group_algebra, GroupBackedProjector},
    YoungError,
};
use ax_perm::{enumerate_subgroup, product, sign};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotParity {
    pub values: Vec<u8>,
}

impl SlotParity {
    pub fn try_new(values: Vec<u8>) -> Result<Self, YoungError> {
        for (index, &value) in values.iter().enumerate() {
            if value > 1 {
                return Err(YoungError::InvalidParityValue {
                    index,
                    value: i8::try_from(value).unwrap_or(i8::MAX),
                });
            }
        }
        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<u8> {
        self.values.get(idx).copied()
    }
}

pub fn graded_swap_sign(left_parity: u8, right_parity: u8) -> i32 {
    if left_parity == 1 && right_parity == 1 {
        -1
    } else {
        1
    }
}

pub fn permutation_graded_sign(images: &[usize], parity: &SlotParity) -> Result<i32, YoungError> {
    validate_permutation(images, parity.len())?;

    let mut total = 1;
    for i in 0..images.len() {
        for j in i + 1..images.len() {
            if images[i] > images[j] {
                total *= graded_swap_sign(parity.values[i], parity.values[j]);
            }
        }
    }
    Ok(total)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedPermutationTerm {
    pub images: Vec<usize>,
    pub bosonic_sign: i32,
    pub graded_sign: i32,
    pub total_sign: i32,
    pub coefficient: BigRational,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedProjectorExpansion {
    pub degree: usize,
    pub terms: Vec<GradedPermutationTerm>,
}

pub fn expand_group_backed_projector_graded(
    projector: &GroupBackedProjector,
    parity: &SlotParity,
) -> Result<GradedProjectorExpansion, YoungError> {
    if projector.row_group.degree != parity.len() {
        return Err(YoungError::GradedProjectorDegreeMismatch {
            expected: projector.row_group.degree,
            actual: parity.len(),
        });
    }

    let ungraded = expand_projector_group_algebra(projector)
        .map_err(|_| YoungError::UnsupportedSuperTableauOperation)?;
    let mut bosonic_signs = BTreeMap::new();
    for term in &ungraded {
        bosonic_signs.insert(term.images.clone(), rational_sign(&term.coefficient));
    }

    let terms = ungraded
        .into_iter()
        .map(|term| {
            let bosonic_sign = bosonic_signs.get(&term.images).copied().unwrap_or(1);
            let graded_sign = permutation_graded_sign(&term.images, parity)?;
            let total_sign = bosonic_sign * graded_sign;
            let magnitude = term.coefficient.abs();
            let coefficient = if total_sign < 0 { -magnitude } else { magnitude };
            Ok(GradedPermutationTerm {
                images: term.images,
                bosonic_sign,
                graded_sign,
                total_sign,
                coefficient,
            })
        })
        .collect::<Result<Vec<_>, YoungError>>()?;

    Ok(GradedProjectorExpansion {
        degree: projector.row_group.degree,
        terms,
    })
}

pub fn canonicalize_slots_under_graded_projector(
    projector: &GroupBackedProjector,
    slots: &[usize],
    parity: &SlotParity,
) -> Result<(Vec<usize>, i32), YoungError> {
    if projector.row_group.degree != slots.len() {
        return Err(YoungError::GradedProjectorDegreeMismatch {
            expected: projector.row_group.degree,
            actual: slots.len(),
        });
    }
    if projector.row_group.degree != parity.len() {
        return Err(YoungError::GradedProjectorDegreeMismatch {
            expected: projector.row_group.degree,
            actual: parity.len(),
        });
    }

    let row_elements =
        enumerate_subgroup(&projector.row_group.generators, projector.row_group.degree);
    let column_elements = enumerate_subgroup(
        &projector.column_group.generators,
        projector.column_group.degree,
    );

    let mut actions = row_elements
        .iter()
        .flat_map(|row| {
            column_elements.iter().map(move |column| {
                let images = product(column, row);
                (images, sign(column))
            })
        })
        .collect::<Vec<_>>();
    actions.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let mut best_slots: Option<Vec<usize>> = None;
    let mut best_sign = 1;
    for (images, bosonic_sign) in actions {
        let candidate = apply_perm_to_slots(slots, &images)?;
        let graded_sign = permutation_graded_sign(&images, parity)?;
        let total_sign = bosonic_sign * graded_sign;

        match &best_slots {
            None => {
                best_slots = Some(candidate);
                best_sign = total_sign;
            }
            Some(current) if candidate < *current => {
                best_slots = Some(candidate);
                best_sign = total_sign;
            }
            Some(current) if candidate == *current => {}
            Some(_) => {}
        }
    }

    best_slots
        .map(|canonical| (canonical, best_sign))
        .ok_or(YoungError::UnsupportedSuperTableauOperation)
}

fn validate_permutation(images: &[usize], degree: usize) -> Result<(), YoungError> {
    if images.len() != degree {
        return Err(YoungError::GradedProjectorDegreeMismatch {
            expected: degree,
            actual: images.len(),
        });
    }

    let mut seen = vec![false; degree];
    for &image in images {
        if image >= degree || seen[image] {
            return Err(YoungError::UnsupportedSuperTableauOperation);
        }
        seen[image] = true;
    }
    Ok(())
}

fn apply_perm_to_slots(slots: &[usize], perm: &[usize]) -> Result<Vec<usize>, YoungError> {
    validate_permutation(perm, slots.len())?;
    Ok((0..slots.len()).map(|index| slots[perm[index]]).collect())
}

fn rational_sign(value: &BigRational) -> i32 {
    if value.is_zero() {
        0
    } else if value.is_positive() {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_group_backed_projector, ProjectorNormalization, YoungDiagram, YoungTableau};
    use num_rational::BigRational;
    use num_traits::One;

    fn standard(rows: Vec<usize>) -> GroupBackedProjector {
        let diagram = YoungDiagram::try_new(rows).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        build_group_backed_projector(&tableau, ProjectorNormalization::Unnormalized).unwrap()
    }

    #[test]
    fn slot_parity_accepts_binary_values() {
        assert_eq!(
            SlotParity::try_new(vec![0, 1, 0, 1]),
            Ok(SlotParity {
                values: vec![0, 1, 0, 1],
            })
        );
    }

    #[test]
    fn slot_parity_rejects_invalid_value() {
        assert_eq!(
            SlotParity::try_new(vec![0, 2]),
            Err(YoungError::InvalidParityValue { index: 1, value: 2 })
        );
    }

    #[test]
    fn graded_swap_sign_matches_all_parity_pairs() {
        assert_eq!(graded_swap_sign(0, 0), 1);
        assert_eq!(graded_swap_sign(0, 1), 1);
        assert_eq!(graded_swap_sign(1, 0), 1);
        assert_eq!(graded_swap_sign(1, 1), -1);
    }

    #[test]
    fn permutation_graded_sign_tracks_odd_swap() {
        let parity = SlotParity::try_new(vec![1, 1]).unwrap();
        assert_eq!(permutation_graded_sign(&[1, 0], &parity), Ok(-1));
    }

    #[test]
    fn permutation_graded_sign_tracks_mixed_swap() {
        let parity = SlotParity::try_new(vec![1, 0]).unwrap();
        assert_eq!(permutation_graded_sign(&[1, 0], &parity), Ok(1));
    }

    #[test]
    fn graded_projector_expansion_stores_bosonic_and_graded_metadata() {
        let projector = standard(vec![1, 1]);
        let parity = SlotParity::try_new(vec![1, 1]).unwrap();
        let expansion = expand_group_backed_projector_graded(&projector, &parity).unwrap();

        assert_eq!(expansion.degree, 2);
        assert_eq!(expansion.terms.len(), 2);
        let swap = expansion
            .terms
            .iter()
            .find(|term| term.images == vec![1, 0])
            .unwrap();
        assert_eq!(swap.bosonic_sign, -1);
        assert_eq!(swap.graded_sign, -1);
        assert_eq!(swap.total_sign, 1);
        assert_eq!(swap.coefficient, BigRational::one());
    }

    #[test]
    fn graded_canonicalization_odd_column_swap_cancels_bosonic_sign() {
        let projector = standard(vec![1, 1]);
        let parity = SlotParity::try_new(vec![1, 1]).unwrap();
        assert_eq!(
            canonicalize_slots_under_graded_projector(&projector, &[9, 3], &parity),
            Ok((vec![3, 9], 1))
        );
    }

    #[test]
    fn graded_canonicalization_mixed_column_swap_keeps_bosonic_sign() {
        let projector = standard(vec![1, 1]);
        let parity = SlotParity::try_new(vec![1, 0]).unwrap();
        assert_eq!(
            canonicalize_slots_under_graded_projector(&projector, &[9, 3], &parity),
            Ok((vec![3, 9], -1))
        );
    }
}
