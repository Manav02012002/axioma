use crate::{SemistandardTableau, SkewDiagram, YoungDiagram, YoungError};
use num_bigint::BigInt;

pub fn enumerate_semistandard_with_content(
    shape: &YoungDiagram,
    content: &[usize],
) -> Result<Vec<SemistandardTableau<usize>>, YoungError> {
    let actual_total = content.iter().sum::<usize>();
    let expected_cells = shape.n_cells();
    if actual_total != expected_cells {
        return Err(YoungError::ContentLengthMismatch {
            expected_cells,
            actual_total,
        });
    }

    let mut state = EnumerationState::new(shape.rows.clone(), vec![0; shape.n_rows()], content);
    enumerate_compact_tableaux(&mut state);
    Ok(state.finish())
}

pub fn enumerate_skew_semistandard_with_content(
    skew: &SkewDiagram,
    content: &[usize],
) -> Result<Vec<SemistandardTableau<usize>>, YoungError> {
    let actual_total = content.iter().sum::<usize>();
    let expected_cells = skew.n_cells();
    if actual_total != expected_cells {
        return Err(YoungError::ContentLengthMismatch {
            expected_cells,
            actual_total,
        });
    }

    let row_lengths = (0..skew.n_rows())
        .map(|row| skew.row_interval(row).map_or(0, |(start, end)| end - start))
        .collect::<Vec<_>>();
    let row_offsets = (0..skew.n_rows())
        .map(|row| skew.inner_row_len(row))
        .collect::<Vec<_>>();
    let mut state = EnumerationState::new(row_lengths, row_offsets, content);
    enumerate_compact_tableaux(&mut state);
    Ok(state.finish())
}

pub fn kostka_number_exact(shape: &YoungDiagram, content: &[usize]) -> Result<BigInt, YoungError> {
    Ok(BigInt::from(
        enumerate_semistandard_with_content(shape, content)?.len(),
    ))
}

#[derive(Clone)]
struct EnumerationState<'a> {
    row_offsets: Vec<usize>,
    rows: Vec<Vec<usize>>,
    remaining: Vec<usize>,
    n_values: usize,
    out: Vec<SemistandardTableau<usize>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> EnumerationState<'a> {
    fn new(row_lengths: Vec<usize>, row_offsets: Vec<usize>, content: &[usize]) -> Self {
        let rows = row_lengths
            .iter()
            .map(|len| vec![0usize; *len])
            .collect::<Vec<_>>();
        Self {
            row_offsets,
            rows,
            remaining: content.to_vec(),
            n_values: content.len(),
            out: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    fn finish(mut self) -> Vec<SemistandardTableau<usize>> {
        self.out.sort_by_key(flatten_row_major);
        self.out
    }

    fn actual_col(&self, row: usize, compact_col: usize) -> usize {
        self.row_offsets[row] + compact_col
    }

    fn left_value(&self, row: usize, compact_col: usize) -> Option<usize> {
        if compact_col > 0 {
            Some(self.rows[row][compact_col - 1])
        } else {
            None
        }
    }

    fn above_value(&self, row: usize, compact_col: usize) -> Option<usize> {
        if row == 0 {
            return None;
        }
        let actual_col = self.actual_col(row, compact_col);
        let prev_offset = self.row_offsets[row - 1];
        if actual_col < prev_offset {
            return None;
        }
        let prev_compact = actual_col - prev_offset;
        self.rows
            .get(row - 1)
            .and_then(|prev_row| prev_row.get(prev_compact))
            .copied()
            .filter(|value| *value != 0)
    }
}

fn enumerate_compact_tableaux(state: &mut EnumerationState<'_>) {
    if let Some((row, col)) = next_unfilled(state) {
        for value_idx in 0..state.n_values {
            if state.remaining[value_idx] == 0 {
                continue;
            }
            let value = value_idx + 1;
            if let Some(left) = state.left_value(row, col) {
                if value < left {
                    continue;
                }
            }
            if let Some(above) = state.above_value(row, col) {
                if value <= above {
                    continue;
                }
            }

            state.rows[row][col] = value;
            state.remaining[value_idx] -= 1;
            enumerate_compact_tableaux(state);
            state.remaining[value_idx] += 1;
            state.rows[row][col] = 0;
        }
        return;
    }

    state.out.push(SemistandardTableau {
        rows: state.rows.clone(),
    });
}

fn next_unfilled(state: &EnumerationState<'_>) -> Option<(usize, usize)> {
    for (row_idx, row) in state.rows.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if *value == 0 {
                return Some((row_idx, col_idx));
            }
        }
    }
    None
}

fn flatten_row_major(tableau: &SemistandardTableau<usize>) -> Vec<usize> {
    tableau
        .rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_shape_two_content_one_one_is_exact() {
        let tableaux =
            enumerate_semistandard_with_content(&YoungDiagram::try_new(vec![2]).unwrap(), &[1, 1])
                .unwrap();
        assert_eq!(tableaux, vec![SemistandardTableau { rows: vec![vec![1, 2]] }]);
    }

    #[test]
    fn enumerate_one_one_content_two_is_empty() {
        let tableaux =
            enumerate_semistandard_with_content(&YoungDiagram::try_new(vec![1, 1]).unwrap(), &[2])
                .unwrap();
        assert!(tableaux.is_empty());
    }

    #[test]
    fn exact_kostka_numbers_match_required_cases() {
        assert_eq!(
            kostka_number_exact(&YoungDiagram::try_new(vec![2, 1]).unwrap(), &[2, 1]).unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            kostka_number_exact(&YoungDiagram::try_new(vec![2, 1]).unwrap(), &[1, 1, 1]).unwrap(),
            BigInt::from(2usize)
        );
    }
}
