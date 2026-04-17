use anyhow::{Context, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutedCoefficientTerm {
    pub permutation: Vec<usize>,
    pub coefficient_numer: i64,
    pub coefficient_denom: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorMultitermIdentity {
    FirstBianchi { cyclic_slots: [usize; 3] },
    CyclicSum { slots: Vec<usize> },
    AlternatingSum { slots: Vec<usize> },
    LinearCombination { terms: Vec<PermutedCoefficientTerm> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorIdentitySet {
    pub multiterm: Vec<TensorMultitermIdentity>,
}

impl TensorIdentitySet {
    pub fn empty() -> Self {
        Self {
            multiterm: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.multiterm.is_empty()
    }

    pub fn add(&mut self, identity: TensorMultitermIdentity) {
        self.multiterm.push(identity);
    }

    pub fn len(&self) -> usize {
        self.multiterm.len()
    }
}

pub fn riemann_identity_set() -> TensorIdentitySet {
    TensorIdentitySet {
        multiterm: vec![TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [1, 2, 3],
        }],
    }
}

pub fn identity_permutation_terms(
    identity: &TensorMultitermIdentity,
    arity: usize,
) -> Result<Vec<PermutedCoefficientTerm>> {
    match identity {
        TensorMultitermIdentity::FirstBianchi { cyclic_slots } => {
            expand_cyclic_terms(cyclic_slots.as_slice(), arity)
                .context("failed to expand tensor multiterm identity")
        }
        TensorMultitermIdentity::CyclicSum { slots } => {
            expand_cyclic_terms(slots, arity).context("failed to expand tensor multiterm identity")
        }
        TensorMultitermIdentity::AlternatingSum { slots } => expand_alternating_terms(slots, arity)
            .context("failed to expand tensor multiterm identity"),
        TensorMultitermIdentity::LinearCombination { terms } => validate_linear_terms(terms, arity)
            .context("failed to expand tensor multiterm identity"),
    }
}

fn expand_cyclic_terms(slots: &[usize], arity: usize) -> Result<Vec<PermutedCoefficientTerm>> {
    validate_slots(slots, arity).context("invalid multiterm identity permutation term")?;
    let mut terms = Vec::new();
    for shift in 0..slots.len() {
        let mut permutation: Vec<usize> = (0..arity).collect();
        for (offset, slot) in slots.iter().enumerate() {
            let source = slots[(offset + shift) % slots.len()];
            permutation[*slot] = source;
        }
        terms.push(PermutedCoefficientTerm {
            permutation,
            coefficient_numer: 1,
            coefficient_denom: 1,
        });
    }
    Ok(terms)
}

fn expand_alternating_terms(slots: &[usize], arity: usize) -> Result<Vec<PermutedCoefficientTerm>> {
    validate_slots(slots, arity).context("invalid multiterm identity permutation term")?;
    let mut permutations = Vec::new();
    let mut values = (0..slots.len()).collect::<Vec<_>>();
    permute_all(0, &mut values, &mut permutations);
    permutations.sort();
    permutations.dedup();

    Ok(permutations
        .into_iter()
        .map(|values| {
            let mut permutation: Vec<usize> = (0..arity).collect();
            for (target_position, target_slot) in slots.iter().enumerate() {
                permutation[*target_slot] = slots[values[target_position]];
            }
            let parity = permutation_parity(&values);
            PermutedCoefficientTerm {
                permutation,
                coefficient_numer: if parity % 2 == 0 { 1 } else { -1 },
                coefficient_denom: 1,
            }
        })
        .collect())
}

fn validate_linear_terms(
    terms: &[PermutedCoefficientTerm],
    arity: usize,
) -> Result<Vec<PermutedCoefficientTerm>> {
    let mut validated = Vec::with_capacity(terms.len());
    for term in terms {
        if term.permutation.len() != arity
            || term.coefficient_denom <= 0
            || !is_valid_permutation(&term.permutation)
        {
            anyhow::bail!("invalid multiterm identity permutation term");
        }
        validated.push(term.clone());
    }
    Ok(validated)
}

fn validate_slots(slots: &[usize], arity: usize) -> Result<()> {
    if slots.is_empty() || slots.iter().any(|slot| *slot >= arity) {
        anyhow::bail!("invalid multiterm identity permutation term");
    }
    let mut dedup = slots.to_vec();
    dedup.sort_unstable();
    dedup.dedup();
    if dedup.len() != slots.len() {
        anyhow::bail!("invalid multiterm identity permutation term");
    }
    Ok(())
}

fn is_valid_permutation(permutation: &[usize]) -> bool {
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    sorted.into_iter().eq(0..permutation.len())
}

fn permute_all(start: usize, values: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if start == values.len() {
        out.push(values.clone());
        return;
    }
    for idx in start..values.len() {
        values.swap(start, idx);
        permute_all(start + 1, values, out);
        values.swap(start, idx);
    }
}

fn permutation_parity(values: &[usize]) -> usize {
    let mut inversions = 0usize;
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if values[i] > values[j] {
                inversions += 1;
            }
        }
    }
    inversions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riemann_identity_set_uses_exact_slot_convention() {
        assert_eq!(
            riemann_identity_set(),
            TensorIdentitySet {
                multiterm: vec![TensorMultitermIdentity::FirstBianchi {
                    cyclic_slots: [1, 2, 3]
                }],
            }
        );
    }

    #[test]
    fn first_bianchi_expands_to_exact_three_terms() {
        let terms = identity_permutation_terms(
            &TensorMultitermIdentity::FirstBianchi {
                cyclic_slots: [1, 2, 3],
            },
            4,
        )
        .ok();
        assert_eq!(
            terms,
            Some(vec![
                PermutedCoefficientTerm {
                    permutation: vec![0, 1, 2, 3],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                PermutedCoefficientTerm {
                    permutation: vec![0, 2, 3, 1],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                PermutedCoefficientTerm {
                    permutation: vec![0, 3, 1, 2],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
            ])
        );
    }

    #[test]
    fn cyclic_sum_expands_to_all_rotations() {
        let terms = identity_permutation_terms(
            &TensorMultitermIdentity::CyclicSum {
                slots: vec![0, 1, 2],
            },
            3,
        )
        .ok();
        assert_eq!(
            terms,
            Some(vec![
                PermutedCoefficientTerm {
                    permutation: vec![0, 1, 2],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                PermutedCoefficientTerm {
                    permutation: vec![1, 2, 0],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                PermutedCoefficientTerm {
                    permutation: vec![2, 0, 1],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
            ])
        );
    }

    #[test]
    fn alternating_sum_on_two_slots_has_signed_swap() {
        let terms = identity_permutation_terms(
            &TensorMultitermIdentity::AlternatingSum { slots: vec![0, 1] },
            2,
        )
        .ok();
        assert_eq!(
            terms,
            Some(vec![
                PermutedCoefficientTerm {
                    permutation: vec![0, 1],
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                PermutedCoefficientTerm {
                    permutation: vec![1, 0],
                    coefficient_numer: -1,
                    coefficient_denom: 1,
                },
            ])
        );
    }
}
