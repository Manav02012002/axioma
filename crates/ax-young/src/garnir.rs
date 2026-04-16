use crate::{tableau::Tableaux, YoungError, YoungTableau};
use num_bigint::BigInt;
use num_rational::BigRational;

pub fn standardize_garnir(tableau: &YoungTableau) -> Result<Tableaux<usize>, YoungError> {
    let mut current = Tableaux::new();
    current.add_tableau(tableau.clone());
    loop {
        let mut changed = false;
        let mut next = Tableaux::new();

        for mut tableau in current.storage.drain(..) {
            tableau.sort_within_columns()?;
            let Some((row, col)) = tableau.nonstandard_loc() else {
                next.add_tableau(tableau);
                continue;
            };

            changed = true;
            let left_col = col - 1;
            let right_positions: Vec<(usize, usize)> =
                (row..tableau.column_size(col)).map(|r| (r, col)).collect();
            let left_positions: Vec<(usize, usize)> = (row..tableau.column_size(left_col))
                .map(|r| (r, left_col))
                .collect();

            let right_values = right_positions
                .iter()
                .map(|&(r, c)| {
                    tableau
                        .get(r, c)
                        .copied()
                        .ok_or(YoungError::InvalidCell { row: r, col: c })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let left_values = left_positions
                .iter()
                .map(|&(r, c)| {
                    tableau
                        .get(r, c)
                        .copied()
                        .ok_or(YoungError::InvalidCell { row: r, col: c })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let union = concatenate(&right_values, &left_values);
            for left_choice in ordered_subsets(union.len(), left_positions.len()) {
                let identity_choice: Vec<usize> =
                    (right_values.len()..right_values.len() + left_values.len()).collect();
                if left_choice == identity_choice {
                    continue;
                }

                let left_choice_set = left_choice
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                let left_block: Vec<usize> = left_choice.iter().map(|idx| union[*idx]).collect();
                let right_block: Vec<usize> = union
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !left_choice_set.contains(idx))
                    .map(|(_, value)| *value)
                    .collect();
                let mut candidate = tableau.clone();
                for ((r, c), value) in right_positions.iter().zip(right_block.iter()) {
                    if let Some(cell) = candidate.get_mut(*r, *c) {
                        *cell = *value;
                    } else {
                        return Err(YoungError::InvalidCell { row: *r, col: *c });
                    }
                }
                for ((r, c), value) in left_positions.iter().zip(left_block.iter()) {
                    if let Some(cell) = candidate.get_mut(*r, *c) {
                        *cell = *value;
                    } else {
                        return Err(YoungError::InvalidCell { row: *r, col: *c });
                    }
                }
                candidate.sort_within_columns()?;
                let sign = shuffle_sign(&union, &concatenate(&right_block, &left_block));
                candidate.multiplicity *= BigRational::from_integer(BigInt::from(-sign));
                next.add_tableau(candidate);
            }
        }

        next.remove_nullifying_traces();
        current = next;
        if !changed {
            break;
        }
    }
    Ok(current)
}

fn ordered_subsets(total: usize, choose: usize) -> Vec<Vec<usize>> {
    fn rec(
        total: usize,
        choose: usize,
        next_idx: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == choose {
            out.push(current.clone());
            return;
        }
        for idx in next_idx..total {
            current.push(idx);
            rec(total, choose, idx + 1, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(total, choose, 0, &mut Vec::new(), &mut out);
    out
}

fn concatenate<T: Clone>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    let mut out = lhs.to_vec();
    out.extend_from_slice(rhs);
    out
}

fn decorate_with_occurrence<T: Clone + Ord>(values: &[T]) -> Vec<(T, usize)> {
    let mut counts = std::collections::BTreeMap::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let counter = counts.entry(value.clone()).or_insert(0usize);
        out.push((value.clone(), *counter));
        *counter += 1;
    }
    out
}

fn inversion_parity(perm: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in i + 1..perm.len() {
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

fn shuffle_sign<T: Clone + Ord>(original: &[T], shuffled: &[T]) -> i32 {
    let original_order = decorate_with_occurrence(original);
    let shuffled_order = decorate_with_occurrence(shuffled);
    let mut position_of = std::collections::BTreeMap::new();
    for (idx, item) in shuffled_order.iter().enumerate() {
        position_of.insert(item.clone(), idx);
    }
    let perm = original_order
        .iter()
        .map(|item| position_of.get(item).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    inversion_parity(&perm)
}
