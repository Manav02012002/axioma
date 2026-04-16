use crate::{
    littlewood_richardson_basis, lr_shapes_with_multiplicity, tableau::Tableaux, FilledTableau,
    LittlewoodRichardsonBasisEntry, YoungDiagram, YoungError,
};

pub fn lr_tensor<T: Clone + Ord + Eq>(
    left: &FilledTableau<T>,
    right: &FilledTableau<T>,
) -> Result<Tableaux<T>, YoungError> {
    let left_shape = left.shape()?;
    let right_shape = right.shape()?;
    let mut out = Tableaux::new();

    for (target, _multiplicity) in lr_shapes_with_multiplicity(&left_shape, &right_shape)? {
        let basis = littlewood_richardson_basis(&left_shape, &right_shape, &target)?;
        for entry in &basis {
            let tableau = materialize_lr_tableau(left, right, entry)?;
            out.add_tableau(tableau);
        }
    }

    Ok(out)
}

pub fn lr_shapes(
    left: &YoungDiagram,
    right: &YoungDiagram,
) -> Result<Vec<YoungDiagram>, YoungError> {
    let mut shapes = Vec::new();
    for (shape, multiplicity) in lr_shapes_with_multiplicity(left, right)? {
        let repeats = big_int_to_usize(&multiplicity)?;
        for _ in 0..repeats {
            shapes.push(shape.clone());
        }
    }
    shapes.sort_by(|lhs, rhs| lhs.rows.cmp(&rhs.rows));
    Ok(shapes)
}

fn materialize_lr_tableau<T: Clone + Ord + Eq>(
    left: &FilledTableau<T>,
    right: &FilledTableau<T>,
    entry: &LittlewoodRichardsonBasisEntry,
) -> Result<FilledTableau<T>, YoungError> {
    let mut rows = left.rows.clone();
    rows.resize_with(entry.target.n_rows(), Vec::new);
    for row in 0..entry.target.n_rows() {
        let target_len = entry.target.row_len(row).unwrap_or(0);
        if rows[row].len() > target_len {
            return Err(YoungError::ShapeMismatch);
        }
    }

    let mut buckets = right
        .rows
        .iter()
        .cloned()
        .map(std::collections::VecDeque::from)
        .collect::<Vec<_>>();

    for row in 0..entry.skew.n_rows() {
        let Some((start, end)) = entry.skew.row_interval(row) else {
            continue;
        };
        let mut compact_idx = 0usize;
        if rows[row].len() != start {
            return Err(YoungError::InvalidLrSkewPlacement);
        }
        while rows[row].len() < end {
            let label = entry
                .tableau
                .rows
                .get(row)
                .and_then(|current| current.get(compact_idx))
                .copied()
                .ok_or(YoungError::SkewCellOutOfBounds {
                    row,
                    col: start + compact_idx,
                })?;
            let bucket = buckets.get_mut(label.saturating_sub(1)).ok_or(
                YoungError::SkewCellOutOfBounds {
                    row,
                    col: start + compact_idx,
                },
            )?;
            let value = bucket.pop_front().ok_or(YoungError::ShapeMismatch)?;
            rows[row].push(value);
            compact_idx += 1;
        }
    }

    if buckets.iter().any(|bucket| !bucket.is_empty()) {
        return Err(YoungError::InvalidContentWeight);
    }

    FilledTableau::with_metadata(
        rows,
        left.multiplicity.clone() * right.multiplicity.clone(),
        0,
    )
}

fn big_int_to_usize(value: &num_bigint::BigInt) -> Result<usize, YoungError> {
    use num_traits::{Signed, ToPrimitive};

    if value.is_negative() {
        return Err(YoungError::NegativeMultiplicity);
    }
    value.to_usize().ok_or(YoungError::NegativeMultiplicity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_shapes_for_two_boxes() {
        let left = YoungDiagram::try_new(vec![1]).unwrap();
        let right = YoungDiagram::try_new(vec![1]).unwrap();
        let shapes = lr_shapes(&left, &right).unwrap();
        assert_eq!(
            shapes
                .into_iter()
                .map(|shape| shape.rows)
                .collect::<Vec<_>>(),
            vec![vec![1, 1], vec![2]]
        );
    }

    #[test]
    fn lr_shapes_for_two_and_one() {
        let left = YoungDiagram::try_new(vec![2]).unwrap();
        let right = YoungDiagram::try_new(vec![1]).unwrap();
        let shapes = lr_shapes(&left, &right).unwrap();
        assert_eq!(
            shapes
                .into_iter()
                .map(|shape| shape.rows)
                .collect::<Vec<_>>(),
            vec![vec![2, 1], vec![3]]
        );
    }

    #[test]
    fn lr_shapes_for_one_and_one_preserve_compatibility_order() {
        let left = YoungDiagram::try_new(vec![1]).unwrap();
        let right = YoungDiagram::try_new(vec![1]).unwrap();
        assert_eq!(
            lr_shapes(&left, &right)
                .unwrap()
                .into_iter()
                .map(|shape| shape.rows)
                .collect::<Vec<_>>(),
            vec![vec![1, 1], vec![2]]
        );
    }

    #[test]
    fn lr_tensor_remains_exported_and_materializes_exact_basis() {
        let left = FilledTableau::try_new(vec![vec!['a', 'b']]).unwrap();
        let right = FilledTableau::try_new(vec![vec!['c']]).unwrap();

        let tableaux = lr_tensor(&left, &right).unwrap();
        let shapes = tableaux
            .storage
            .iter()
            .map(|tableau| tableau.shape().unwrap().rows)
            .collect::<Vec<_>>();

        assert_eq!(shapes, vec![vec![2, 1], vec![3]]);
    }
}
