use crate::{partition::YoungDiagram, YoungError};
use std::collections::BTreeSet;

pub fn branch_gl_n_to_gl_n_minus_1(
    shape: &YoungDiagram,
    n: usize,
) -> Result<Vec<YoungDiagram>, YoungError> {
    if shape.n_rows() > n {
        return Err(YoungError::BranchingDimensionTooSmall {
            shape: shape.rows.clone(),
            n,
        });
    }

    fn rec(
        lambda: &[usize],
        row: usize,
        prev: Option<usize>,
        current: &mut Vec<usize>,
        out: &mut Vec<YoungDiagram>,
    ) {
        if row == lambda.len() {
            let rows = trim_trailing_zeros(current.clone());
            if let Ok(diagram) = YoungDiagram::try_new(rows) {
                out.push(diagram);
            }
            return;
        }

        let upper = lambda[row];
        let lower = lambda.get(row + 1).copied().unwrap_or(0);
        let upper = prev.map_or(upper, |prev_row| upper.min(prev_row));
        for value in lower..=upper {
            current.push(value);
            rec(lambda, row + 1, Some(value), current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(&shape.rows, 0, None, &mut Vec::new(), &mut out);
    out.sort_by(|lhs, rhs| lhs.rows.cmp(&rhs.rows));
    out.dedup();
    Ok(out)
}

pub fn branch_s_n_to_s_n_minus_1(shape: &YoungDiagram) -> Result<Vec<YoungDiagram>, YoungError> {
    let mut out = BTreeSet::new();
    for row in 0..shape.n_rows() {
        let current = shape.row_len(row).unwrap_or(0);
        let next = shape.row_len(row + 1).unwrap_or(0);
        if current == 0 || current == next {
            continue;
        }
        let mut rows = shape.rows.clone();
        rows[row] -= 1;
        let rows = trim_trailing_zeros(rows);
        if let Ok(diagram) = YoungDiagram::try_new(rows) {
            out.insert(diagram);
        }
    }
    Ok(out.into_iter().collect())
}

fn trim_trailing_zeros(mut rows: Vec<usize>) -> Vec<usize> {
    while rows.last().copied() == Some(0) {
        rows.pop();
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn gl_branching_interlaces_exactly() {
        assert_eq!(
            branch_gl_n_to_gl_n_minus_1(&yd(vec![2, 1]), 3).unwrap(),
            vec![yd(vec![1]), yd(vec![1, 1]), yd(vec![2]), yd(vec![2, 1])]
        );
    }

    #[test]
    fn gl_branching_rejects_shapes_that_do_not_fit_dimension() {
        assert_eq!(
            branch_gl_n_to_gl_n_minus_1(&yd(vec![1, 1, 1]), 2),
            Err(YoungError::BranchingDimensionTooSmall {
                shape: vec![1, 1, 1],
                n: 2,
            })
        );
    }

    #[test]
    fn symmetric_group_branching_removes_each_removable_corner_once() {
        assert_eq!(
            branch_s_n_to_s_n_minus_1(&yd(vec![2, 1])).unwrap(),
            vec![yd(vec![1, 1]), yd(vec![2])]
        );
        assert_eq!(
            branch_s_n_to_s_n_minus_1(&yd(vec![3])).unwrap(),
            vec![yd(vec![2])]
        );
    }
}
