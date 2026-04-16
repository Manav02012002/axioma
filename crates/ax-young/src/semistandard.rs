use crate::{
    kostka_number_exact, partition::YoungDiagram, tableau::FilledTableau, YoungError,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemistandardTableau<T: Clone + Ord + Eq> {
    pub rows: Vec<Vec<T>>,
}

impl<T: Clone + Ord + Eq> SemistandardTableau<T> {
    pub fn try_new(rows: Vec<Vec<T>>) -> Result<Self, YoungError> {
        let tableau = Self { rows };
        match tableau.is_semistandard()? {
            true => Ok(tableau),
            false => {
                for (row_idx, row) in tableau.rows.iter().enumerate() {
                    if row.windows(2).any(|pair| pair[0] > pair[1]) {
                        return Err(YoungError::InvalidSemistandardRow { row: row_idx });
                    }
                }
                let n_cols = tableau.rows.iter().map(Vec::len).max().unwrap_or(0);
                for col in 0..n_cols {
                    let entries: Vec<&T> =
                        tableau.rows.iter().filter_map(|row| row.get(col)).collect();
                    if entries.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(YoungError::InvalidSemistandardColumn { col });
                    }
                }
                Err(YoungError::ShapeMismatch)
            }
        }
    }

    pub fn shape(&self) -> Result<YoungDiagram, YoungError> {
        YoungDiagram::try_new(self.rows.iter().map(Vec::len).collect())
    }

    pub fn is_semistandard(&self) -> Result<bool, YoungError> {
        self.shape()?;
        for row in &self.rows {
            if row.windows(2).any(|pair| pair[0] > pair[1]) {
                return Ok(false);
            }
        }
        let n_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        for col in 0..n_cols {
            let entries: Vec<&T> = self.rows.iter().filter_map(|row| row.get(col)).collect();
            if entries.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn reading_word(&self) -> Vec<T> {
        let mut out = Vec::new();
        for row in &self.rows {
            for value in row.iter().rev() {
                out.push(value.clone());
            }
        }
        out
    }

    pub fn content_multiplicity(&self) -> BTreeMap<T, usize>
    where
        T: Ord,
    {
        let mut map = BTreeMap::new();
        for row in &self.rows {
            for value in row {
                *map.entry(value.clone()).or_default() += 1;
            }
        }
        map
    }

    pub fn content_weight(&self) -> BTreeMap<T, usize>
    where
        T: Clone + Ord,
    {
        self.content_multiplicity()
    }

    pub fn size(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    pub fn to_filled(self) -> FilledTableau<T> {
        FilledTableau {
            rows: self.rows,
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        }
    }
}

impl SemistandardTableau<usize> {
    pub fn is_lattice_word(&self) -> bool {
        let word = self.reading_word();
        let max_label = word.iter().copied().max().unwrap_or(0);
        let mut counts = vec![0usize; max_label + 2];
        for label in word {
            counts[label] += 1;
            for k in 1..=max_label {
                if counts[k] < counts[k + 1] {
                    return false;
                }
            }
        }
        true
    }
}

pub fn kostka_number(shape: &YoungDiagram, content: &[usize]) -> Result<BigInt, YoungError> {
    kostka_number_exact(shape, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semistandard_validity_checks() {
        let valid = SemistandardTableau {
            rows: vec![vec![1, 1], vec![2]],
        };
        assert_eq!(valid.is_semistandard(), Ok(true));

        let invalid_row = SemistandardTableau {
            rows: vec![vec![2, 1], vec![3]],
        };
        assert_eq!(invalid_row.is_semistandard(), Ok(false));

        let invalid_col = SemistandardTableau {
            rows: vec![vec![1, 2], vec![1]],
        };
        assert_eq!(invalid_col.is_semistandard(), Ok(false));
    }

    #[test]
    fn reading_word_is_exact() {
        let tableau = SemistandardTableau {
            rows: vec![vec![1, 2], vec![2]],
        };
        assert_eq!(tableau.reading_word(), vec![2, 1, 2]);
    }

    #[test]
    fn lattice_word_detection_is_exact() {
        assert!(SemistandardTableau {
            rows: vec![vec![1, 1], vec![2]],
        }
        .is_lattice_word());
        assert!(!SemistandardTableau {
            rows: vec![vec![2, 2], vec![1]],
        }
        .is_lattice_word());
    }

    #[test]
    fn kostka_numbers_match_small_examples() {
        let shape = YoungDiagram::try_new(vec![2, 1]).unwrap();
        assert_eq!(
            kostka_number(&shape, &[2, 1]).unwrap(),
            BigInt::from(1usize)
        );

        let shape = YoungDiagram::try_new(vec![2]).unwrap();
        assert_eq!(
            kostka_number(&shape, &[1, 1]).unwrap(),
            BigInt::from(1usize)
        );

        let shape = YoungDiagram::try_new(vec![1, 1]).unwrap();
        assert_eq!(kostka_number(&shape, &[2]).unwrap(), BigInt::from(0usize));
    }
}
