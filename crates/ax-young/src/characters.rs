use crate::{symmetric_functions::PowerSumExpansion, YoungDiagram, YoungError};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub fn is_valid_cycle_type(cycle_type: &[usize]) -> bool {
    !cycle_type.is_empty()
        && cycle_type.iter().all(|part| *part > 0)
        && cycle_type.windows(2).all(|window| window[0] >= window[1])
}

pub fn symmetric_group_character(
    shape: &YoungDiagram,
    cycle_type: &[usize],
) -> Result<BigInt, YoungError> {
    if !is_valid_cycle_type(cycle_type) {
        return Err(YoungError::InvalidCycleType {
            cycle_type: cycle_type.to_vec(),
        });
    }
    let cycle_size = cycle_type.iter().sum::<usize>();
    let shape_size = shape.n_cells();
    if shape_size != cycle_size {
        return Err(YoungError::CharacterSizeMismatch {
            shape_size,
            cycle_size,
        });
    }
    Ok(character_rows(&shape.rows, cycle_type))
}

pub fn frobenius_characteristic(shape: &YoungDiagram) -> Result<PowerSumExpansion, YoungError> {
    let size = shape.n_cells();
    let mut terms = BTreeMap::new();
    for cycle_partition in enumerate_partitions_of_size(size) {
        let character = symmetric_group_character(shape, &cycle_partition.rows)?;
        if character.is_zero() {
            continue;
        }
        let centralizer = cycle_type_centralizer_size(&cycle_partition.rows)?;
        terms.insert(
            cycle_partition,
            BigRational::new(character, centralizer),
        );
    }
    Ok(PowerSumExpansion { terms })
}

pub fn cycle_type_centralizer_size(cycle_type: &[usize]) -> Result<BigInt, YoungError> {
    if !is_valid_cycle_type(cycle_type) {
        return Err(YoungError::InvalidCycleType {
            cycle_type: cycle_type.to_vec(),
        });
    }
    let mut counts = BTreeMap::<usize, usize>::new();
    for &part in cycle_type {
        *counts.entry(part).or_default() += 1;
    }
    let mut out = BigInt::one();
    for (part, multiplicity) in counts {
        out *= BigInt::from(part).pow(multiplicity as u32);
        out *= factorial(multiplicity);
    }
    Ok(out)
}

fn character_rows(shape_rows: &[usize], cycle_type: &[usize]) -> BigInt {
    if cycle_type.is_empty() {
        return if shape_rows.is_empty() {
            BigInt::one()
        } else {
            BigInt::zero()
        };
    }
    let hook_len = cycle_type[0];
    let rest = &cycle_type[1..];
    let mut total = BigInt::zero();
    for (remainder, height) in remove_border_strips(shape_rows, hook_len) {
        let sign = if height % 2 == 0 {
            BigInt::one()
        } else {
            -BigInt::one()
        };
        total += sign * character_rows(&remainder, rest);
    }
    total
}

fn remove_border_strips(shape_rows: &[usize], hook_len: usize) -> Vec<(Vec<usize>, usize)> {
    if hook_len == 0 || hook_len > shape_rows.iter().sum::<usize>() {
        return Vec::new();
    }
    let target_size = shape_rows.iter().sum::<usize>() - hook_len;
    let mut out = Vec::new();
    for inner in enumerate_contained_partitions(shape_rows, target_size) {
        if let Some(height) = border_strip_height(shape_rows, &inner) {
            out.push((inner, height));
        }
    }
    out.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));
    out
}

fn enumerate_contained_partitions(outer: &[usize], target_size: usize) -> Vec<Vec<usize>> {
    fn rec(
        outer: &[usize],
        row: usize,
        prev: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if row == outer.len() {
            if remaining == 0 {
                let mut trimmed = current.clone();
                while trimmed.last().copied() == Some(0) {
                    trimmed.pop();
                }
                out.push(trimmed);
            }
            return;
        }
        let max_part = prev.min(outer[row]).min(remaining);
        for part in (0..=max_part).rev() {
            current.push(part);
            rec(outer, row + 1, part, remaining - part, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(outer, 0, usize::MAX, target_size, &mut Vec::new(), &mut out);
    out.sort();
    out.dedup();
    out
}

fn border_strip_height(outer: &[usize], inner: &[usize]) -> Option<usize> {
    let cells = skew_cells(outer, inner);
    if cells.is_empty() {
        return None;
    }
    if has_two_by_two(&cells) || !is_connected(&cells) {
        return None;
    }
    let row_count = cells.iter().map(|(row, _)| *row).collect::<BTreeSet<_>>().len();
    Some(row_count.saturating_sub(1))
}

fn skew_cells(outer: &[usize], inner: &[usize]) -> BTreeSet<(usize, usize)> {
    let mut cells = BTreeSet::new();
    for (row, &outer_len) in outer.iter().enumerate() {
        let inner_len = inner.get(row).copied().unwrap_or(0);
        for col in inner_len..outer_len {
            cells.insert((row, col));
        }
    }
    cells
}

fn has_two_by_two(cells: &BTreeSet<(usize, usize)>) -> bool {
    cells.iter().any(|&(row, col)| {
        cells.contains(&(row + 1, col))
            && cells.contains(&(row, col + 1))
            && cells.contains(&(row + 1, col + 1))
    })
}

fn is_connected(cells: &BTreeSet<(usize, usize)>) -> bool {
    let Some(&start) = cells.iter().next() else {
        return false;
    };
    let mut queue = VecDeque::from([start]);
    let mut seen = BTreeSet::from([start]);
    while let Some((row, col)) = queue.pop_front() {
        for next in [
            (row.wrapping_sub(1), col),
            (row + 1, col),
            (row, col.wrapping_sub(1)),
            (row, col + 1),
        ] {
            if cells.contains(&next) && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }
    seen.len() == cells.len()
}

fn enumerate_partitions_of_size(total: usize) -> Vec<YoungDiagram> {
    fn rec(
        remaining: usize,
        max_part: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<YoungDiagram>,
    ) {
        if remaining == 0 {
            if !current.is_empty() {
                if let Ok(diagram) = YoungDiagram::try_new(current.clone()) {
                    out.push(diagram);
                }
            }
            return;
        }
        for next in (1..=remaining.min(max_part)).rev() {
            current.push(next);
            rec(remaining - next, next, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(total, total, &mut Vec::new(), &mut out);
    out.sort_by(|lhs, rhs| lhs.rows.cmp(&rhs.rows));
    out
}

fn factorial(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, value| acc * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yd(rows: &[usize]) -> YoungDiagram {
        YoungDiagram::try_new(rows.to_vec()).unwrap()
    }

    #[test]
    fn valid_cycle_type_requires_sorted_positive_parts() {
        assert!(is_valid_cycle_type(&[2, 1]));
        assert!(!is_valid_cycle_type(&[1, 2]));
    }

    #[test]
    fn centralizer_size_matches_small_cycle_types() {
        assert_eq!(
            cycle_type_centralizer_size(&[1, 1, 1]).unwrap(),
            BigInt::from(6usize)
        );
        assert_eq!(
            cycle_type_centralizer_size(&[2, 1]).unwrap(),
            BigInt::from(2usize)
        );
    }

    #[test]
    fn symmetric_group_character_matches_s3_table() {
        assert_eq!(
            symmetric_group_character(&yd(&[3]), &[1, 1, 1]).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            symmetric_group_character(&yd(&[3]), &[2, 1]).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            symmetric_group_character(&yd(&[2, 1]), &[1, 1, 1]).unwrap(),
            BigInt::from(2usize)
        );
        assert_eq!(
            symmetric_group_character(&yd(&[2, 1]), &[2, 1]).unwrap(),
            BigInt::from(0usize)
        );
        assert_eq!(
            symmetric_group_character(&yd(&[2, 1]), &[3]).unwrap(),
            BigInt::from(-1)
        );
        assert_eq!(
            symmetric_group_character(&yd(&[1, 1, 1]), &[3]).unwrap(),
            BigInt::from(1usize)
        );
    }

    #[test]
    fn frobenius_characteristic_of_symmetric_square_is_exact() {
        let frobenius = frobenius_characteristic(&yd(&[2])).unwrap();
        assert_eq!(
            frobenius.terms.get(&yd(&[1, 1])).cloned().unwrap(),
            BigRational::new(BigInt::from(1usize), BigInt::from(2usize))
        );
        assert_eq!(
            frobenius.terms.get(&yd(&[2])).cloned().unwrap(),
            BigRational::new(BigInt::from(1usize), BigInt::from(2usize))
        );
    }

    #[test]
    fn frobenius_characteristic_of_antisymmetric_square_is_exact() {
        let frobenius = frobenius_characteristic(&yd(&[1, 1])).unwrap();
        assert_eq!(
            frobenius.terms.get(&yd(&[1, 1])).cloned().unwrap(),
            BigRational::new(BigInt::from(1usize), BigInt::from(2usize))
        );
        assert_eq!(
            frobenius.terms.get(&yd(&[2])).cloned().unwrap(),
            BigRational::new(BigInt::from(-1), BigInt::from(2usize))
        );
    }
}
