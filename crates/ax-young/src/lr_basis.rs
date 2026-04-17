use crate::{
    enumerate_skew_semistandard_with_content, partition::YoungDiagram, skew::SkewDiagram,
    SemistandardTableau, YoungError,
};
use num_bigint::BigInt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LittlewoodRichardsonBasisEntry {
    pub target: YoungDiagram,
    pub skew: SkewDiagram,
    pub tableau: SemistandardTableau<usize>,
}

pub fn littlewood_richardson_coefficient(
    left: &YoungDiagram,
    right: &YoungDiagram,
    target: &YoungDiagram,
) -> Result<BigInt, YoungError> {
    Ok(BigInt::from(
        littlewood_richardson_basis(left, right, target)?.len(),
    ))
}

pub fn littlewood_richardson_basis(
    left: &YoungDiagram,
    right: &YoungDiagram,
    target: &YoungDiagram,
) -> Result<Vec<LittlewoodRichardsonBasisEntry>, YoungError> {
    validate_lr_target(left, right, target)?;
    let skew = SkewDiagram::try_new(target.clone(), left.clone())?;
    let mut basis = enumerate_skew_semistandard_with_content(&skew, &right.rows)?
        .into_iter()
        .filter(|tableau| tableau.is_lattice_word())
        .map(|tableau| LittlewoodRichardsonBasisEntry {
            target: target.clone(),
            skew: skew.clone(),
            tableau,
        })
        .collect::<Vec<_>>();
    basis.sort_by_key(|entry| {
        entry
            .tableau
            .rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>()
    });
    Ok(basis)
}

pub fn lr_shapes_with_multiplicity(
    left: &YoungDiagram,
    right: &YoungDiagram,
) -> Result<Vec<(YoungDiagram, BigInt)>, YoungError> {
    let total_cells = left.n_cells() + right.n_cells();
    let mut out = enumerate_partitions(total_cells)
        .into_iter()
        .filter(|target| shape_contains(target, left))
        .map(|target| {
            let coeff = littlewood_richardson_coefficient(left, right, &target)?;
            Ok((target, coeff))
        })
        .collect::<Result<Vec<_>, YoungError>>()?;
    out.retain(|(_, coeff)| *coeff != BigInt::from(0usize));
    out.sort_by(|(lhs, _), (rhs, _)| lhs.rows.cmp(&rhs.rows));
    Ok(out)
}

fn validate_lr_target(
    left: &YoungDiagram,
    right: &YoungDiagram,
    target: &YoungDiagram,
) -> Result<(), YoungError> {
    let left_cells = left.n_cells();
    let right_cells = right.n_cells();
    let target_cells = target.n_cells();
    if left_cells + right_cells != target_cells {
        return Err(YoungError::LrShapeSizeMismatch {
            left_cells,
            right_cells,
            target_cells,
        });
    }
    if !shape_contains(target, left) {
        return Err(YoungError::TargetDoesNotContainLeftShape {
            left: left.rows.clone(),
            target: target.rows.clone(),
        });
    }
    Ok(())
}

fn shape_contains(outer: &YoungDiagram, inner: &YoungDiagram) -> bool {
    (0..outer.n_rows().max(inner.n_rows()))
        .all(|row| outer.row_len(row).unwrap_or(0) >= inner.row_len(row).unwrap_or(0))
}

fn enumerate_partitions(total: usize) -> Vec<YoungDiagram> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn lr_coefficients_match_required_small_cases() {
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![1]), &yd(vec![1]), &yd(vec![2])).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![1]), &yd(vec![1]), &yd(vec![1, 1])).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![2]), &yd(vec![1]), &yd(vec![2, 1])).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![2]), &yd(vec![1]), &yd(vec![3])).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![1, 1]), &yd(vec![1]), &yd(vec![2, 1]))
                .unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            littlewood_richardson_coefficient(&yd(vec![1, 1]), &yd(vec![1]), &yd(vec![1, 1, 1]))
                .unwrap(),
            BigInt::from(1usize)
        );
    }

    #[test]
    fn lr_shapes_with_multiplicity_are_exact_and_sorted() {
        assert_eq!(
            lr_shapes_with_multiplicity(&yd(vec![2]), &yd(vec![1])).unwrap(),
            vec![
                (yd(vec![2, 1]), BigInt::from(1usize)),
                (yd(vec![3]), BigInt::from(1usize)),
            ]
        );
    }

    #[test]
    fn lr_basis_lengths_match_required_cases() {
        assert_eq!(
            littlewood_richardson_basis(&yd(vec![2]), &yd(vec![1]), &yd(vec![2, 1]))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            littlewood_richardson_basis(&yd(vec![2]), &yd(vec![1]), &yd(vec![3]))
                .unwrap()
                .len(),
            1
        );
    }
}
