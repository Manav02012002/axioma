use crate::error::YoungError;
use num_bigint::BigInt;
use num_traits::One;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct YoungDiagram {
    pub rows: Vec<usize>,
}

impl YoungDiagram {
    pub fn try_new(rows: Vec<usize>) -> Result<Self, YoungError> {
        if rows.is_empty() {
            return Err(YoungError::EmptyDiagram);
        }
        if rows.iter().any(|row| *row == 0) {
            return Err(YoungError::ZeroRowLength { rows });
        }
        if rows.windows(2).any(|window| window[0] < window[1]) {
            return Err(YoungError::NonDecreasingRows { rows });
        }
        Ok(Self { rows })
    }

    pub fn n_cells(&self) -> usize {
        self.rows.iter().sum()
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn n_cols(&self) -> usize {
        self.rows.first().copied().unwrap_or(0)
    }

    pub fn row_len(&self, row: usize) -> Option<usize> {
        self.rows.get(row).copied()
    }

    pub fn column_len(&self, col: usize) -> usize {
        self.rows.iter().filter(|row_len| **row_len > col).count()
    }

    pub fn column_lengths(&self) -> Vec<usize> {
        (0..self.n_cols()).map(|col| self.column_len(col)).collect()
    }

    pub fn conjugate(&self) -> Result<Self, YoungError> {
        YoungDiagram::try_new(self.column_lengths())
    }

    pub fn contains_cell(&self, row: usize, col: usize) -> bool {
        self.rows.get(row).is_some_and(|row_len| col < *row_len)
    }

    pub fn hook_length(&self, row: usize, col: usize) -> Result<usize, YoungError> {
        if !self.contains_cell(row, col) {
            return Err(YoungError::InvalidCell { row, col });
        }
        let arm = self.rows[row] - col - 1;
        let leg = self.column_len(col) - row - 1;
        Ok(arm + leg + 1)
    }

    pub fn hook_length_product(&self) -> Result<BigInt, YoungError> {
        let mut product = BigInt::one();
        for (row, row_len) in self.rows.iter().copied().enumerate() {
            for col in 0..row_len {
                product *= BigInt::from(self.hook_length(row, col)?);
            }
        }
        Ok(product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_diagram_rejected() {
        assert_eq!(YoungDiagram::try_new(vec![]), Err(YoungError::EmptyDiagram));
    }

    #[test]
    fn zero_row_rejected() {
        assert_eq!(
            YoungDiagram::try_new(vec![2, 0]),
            Err(YoungError::ZeroRowLength { rows: vec![2, 0] })
        );
    }

    #[test]
    fn nondecreasing_rows_rejected() {
        assert_eq!(
            YoungDiagram::try_new(vec![1, 2]),
            Err(YoungError::NonDecreasingRows { rows: vec![1, 2] })
        );
    }

    #[test]
    fn valid_diagram_properties() {
        let diagram = YoungDiagram::try_new(vec![3, 2, 1]).unwrap();
        assert_eq!(diagram.n_cells(), 6);
        assert_eq!(diagram.n_rows(), 3);
        assert_eq!(diagram.n_cols(), 3);
        assert_eq!(diagram.column_lengths(), vec![3, 2, 1]);
    }

    #[test]
    fn conjugation_round_trip() {
        let diagram = YoungDiagram::try_new(vec![3, 1]).unwrap();
        let conjugate = diagram.conjugate().unwrap();
        assert_eq!(conjugate.rows, vec![2, 1, 1]);
        assert_eq!(conjugate.conjugate().unwrap(), diagram);
    }

    #[test]
    fn hook_lengths_for_two_one() {
        let diagram = YoungDiagram::try_new(vec![2, 1]).unwrap();
        assert_eq!(diagram.hook_length(0, 0), Ok(3));
        assert_eq!(diagram.hook_length(0, 1), Ok(1));
        assert_eq!(diagram.hook_length(1, 0), Ok(1));
    }
}
