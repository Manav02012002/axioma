use crate::{enumerate_subgroup, identity};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DummySlot {
    pub position: usize,
    pub family: Option<String>,
    pub variance: i8,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DummyGroupError {
    #[error("Dummy-slot variance at position {position} must be +1 or -1, got {variance}")]
    InvalidVariance { position: usize, variance: i8 },
    #[error("Dummy-group input length mismatch: slots {slots}, labels {labels}")]
    MismatchedLengths { slots: usize, labels: usize },
    #[error("Dummy index label {label} appears in incompatible index families")]
    InconsistentDummyFamily { label: String },
    #[error("Dummy index label {label} does not occur once covariant and once contravariant")]
    InconsistentDummyVariance { label: String },
    #[error("Dummy index label {label} occurs {count} times; expected exactly 2 occurrences")]
    OddDummyMultiplicity { label: String, count: usize },
    #[error("Dummy-group permutation degree mismatch: expected {expected}, got {actual}")]
    PermutationDegreeMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DummyRenamingGroup {
    pub degree: usize,
    pub generators: Vec<crate::Perm>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DummyPair {
    family: Option<String>,
    down_position: usize,
    up_position: usize,
}

pub fn validate_dummy_slots(labels: &[String], slots: &[DummySlot]) -> Result<(), DummyGroupError> {
    if labels.len() != slots.len() {
        return Err(DummyGroupError::MismatchedLengths {
            slots: slots.len(),
            labels: labels.len(),
        });
    }

    let mut occurrences: BTreeMap<String, Vec<&DummySlot>> = BTreeMap::new();
    for (label, slot) in labels.iter().zip(slots) {
        if slot.variance != -1 && slot.variance != 1 {
            return Err(DummyGroupError::InvalidVariance {
                position: slot.position,
                variance: slot.variance,
            });
        }
        occurrences.entry(label.clone()).or_default().push(slot);
    }

    for (label, entries) in occurrences {
        match entries.len() {
            1 => {}
            2 => {
                let first = entries[0];
                let second = entries[1];
                if first.family != second.family {
                    return Err(DummyGroupError::InconsistentDummyFamily { label });
                }
                let variances = BTreeSet::from([first.variance, second.variance]);
                if variances != BTreeSet::from([-1, 1]) {
                    return Err(DummyGroupError::InconsistentDummyVariance { label });
                }
            }
            count => {
                return Err(DummyGroupError::OddDummyMultiplicity { label, count });
            }
        }
    }

    Ok(())
}

pub fn build_dummy_renaming_group(
    labels: &[String],
    slots: &[DummySlot],
) -> Result<DummyRenamingGroup, DummyGroupError> {
    validate_dummy_slots(labels, slots)?;

    let mut occurrences: BTreeMap<String, Vec<&DummySlot>> = BTreeMap::new();
    for (label, slot) in labels.iter().zip(slots) {
        occurrences.entry(label.clone()).or_default().push(slot);
    }

    let mut pairs_by_family: BTreeMap<Option<String>, Vec<DummyPair>> = BTreeMap::new();
    for entries in occurrences.values() {
        if entries.len() != 2 {
            continue;
        }
        let down = entries
            .iter()
            .find(|slot| slot.variance == -1)
            .copied()
            .ok_or_else(|| DummyGroupError::InconsistentDummyVariance {
                label: labels[entries[0].position].clone(),
            })?;
        let up = entries
            .iter()
            .find(|slot| slot.variance == 1)
            .copied()
            .ok_or_else(|| DummyGroupError::InconsistentDummyVariance {
                label: labels[entries[0].position].clone(),
            })?;
        pairs_by_family
            .entry(down.family.clone())
            .or_default()
            .push(DummyPair {
                family: down.family.clone(),
                down_position: down.position,
                up_position: up.position,
            });
    }

    let mut generators = Vec::new();
    for pairs in pairs_by_family.values_mut() {
        pairs.sort();
        for window in pairs.windows(2) {
            let left = &window[0];
            let right = &window[1];
            let mut perm = identity(labels.len());
            perm.swap(left.down_position, right.down_position);
            perm.swap(left.up_position, right.up_position);
            generators.push(perm);
        }
    }

    Ok(DummyRenamingGroup {
        degree: labels.len(),
        generators,
    })
}

pub fn apply_permutation_to_labels(
    perm: &crate::Perm,
    labels: &[String],
) -> Result<Vec<String>, DummyGroupError> {
    if perm.len() != labels.len() {
        return Err(DummyGroupError::PermutationDegreeMismatch {
            expected: labels.len(),
            actual: perm.len(),
        });
    }
    Ok((0..labels.len())
        .map(|idx| labels[perm[idx]].clone())
        .collect())
}

pub fn canonicalize_labels_under_dummy_group(
    labels: &[String],
    slots: &[DummySlot],
) -> Result<Vec<String>, DummyGroupError> {
    let group = build_dummy_renaming_group(labels, slots)?;
    let mut best = labels.to_vec();
    for perm in enumerate_subgroup(&group.generators, group.degree) {
        let candidate = apply_permutation_to_labels(&perm, labels)?;
        if candidate < best {
            best = candidate;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(position: usize, variance: i8) -> DummySlot {
        DummySlot {
            position,
            family: None,
            variance,
        }
    }

    #[test]
    fn validate_dummy_slots_accepts_two_pairs() {
        let labels = vec![
            "i".to_string(),
            "j".to_string(),
            "i".to_string(),
            "j".to_string(),
        ];
        let slots = vec![slot(0, -1), slot(1, -1), slot(2, 1), slot(3, 1)];
        assert_eq!(validate_dummy_slots(&labels, &slots), Ok(()));
    }

    #[test]
    fn validate_dummy_slots_rejects_odd_multiplicity() {
        let labels = vec!["i".to_string(), "i".to_string(), "i".to_string()];
        let slots = vec![slot(0, -1), slot(1, 1), slot(2, -1)];
        assert_eq!(
            validate_dummy_slots(&labels, &slots),
            Err(DummyGroupError::OddDummyMultiplicity {
                label: "i".into(),
                count: 3,
            })
        );
    }

    #[test]
    fn validate_dummy_slots_rejects_inconsistent_family() {
        let labels = vec!["i".to_string(), "i".to_string()];
        let slots = vec![
            DummySlot {
                position: 0,
                family: Some("latin".into()),
                variance: -1,
            },
            DummySlot {
                position: 1,
                family: Some("greek".into()),
                variance: 1,
            },
        ];
        assert_eq!(
            validate_dummy_slots(&labels, &slots),
            Err(DummyGroupError::InconsistentDummyFamily { label: "i".into() })
        );
    }

    #[test]
    fn validate_dummy_slots_rejects_inconsistent_variance() {
        let labels = vec!["i".to_string(), "i".to_string()];
        let slots = vec![slot(0, -1), slot(1, -1)];
        assert_eq!(
            validate_dummy_slots(&labels, &slots),
            Err(DummyGroupError::InconsistentDummyVariance { label: "i".into() })
        );
    }

    #[test]
    fn canonicalize_labels_under_dummy_group_lexicographically_minimizes() {
        let labels = vec![
            "j".to_string(),
            "i".to_string(),
            "j".to_string(),
            "i".to_string(),
        ];
        let slots = vec![slot(0, -1), slot(1, -1), slot(2, 1), slot(3, 1)];
        assert_eq!(
            canonicalize_labels_under_dummy_group(&labels, &slots),
            Ok(vec![
                "i".to_string(),
                "j".to_string(),
                "i".to_string(),
                "j".to_string(),
            ])
        );
    }
}
