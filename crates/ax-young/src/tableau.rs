use crate::{garnir, partition::YoungDiagram, projector::Symmetriser, YoungError};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilledTableau<T: Clone + Ord + Eq> {
    pub rows: Vec<Vec<T>>,
    pub multiplicity: BigRational,
    pub selfdual_column: i32,
}

pub type YoungTableau = FilledTableau<usize>;

#[derive(Clone, Debug, Default)]
pub struct Tableaux<T: Clone + Ord + Eq> {
    pub storage: Vec<FilledTableau<T>>,
}

impl<T: Clone + Ord + Eq> FilledTableau<T> {
    pub fn try_new(rows: Vec<Vec<T>>) -> Result<Self, YoungError> {
        Self::with_metadata(rows, BigRational::one(), 0)
    }

    pub fn with_metadata(
        rows: Vec<Vec<T>>,
        multiplicity: BigRational,
        selfdual_column: i32,
    ) -> Result<Self, YoungError> {
        validate_row_shape(&rows)?;
        if *multiplicity.denom() <= BigInt::zero() {
            let numer = multiplicity.numer().to_i64().unwrap_or(0);
            let denom = multiplicity.denom().to_i64().unwrap_or(0);
            return Err(YoungError::InvalidMultiplicity { numer, denom });
        }
        if selfdual_column > 0 {
            let diagram = YoungDiagram::try_new(rows.iter().map(Vec::len).collect())?;
            let column = (selfdual_column - 1) as usize;
            let length = diagram.column_len(column);
            if length % 2 != 0 {
                return Err(YoungError::SelfDualInvalidColumn { column, length });
            }
        }
        Ok(Self {
            rows,
            multiplicity,
            selfdual_column,
        })
    }

    pub fn shape(&self) -> Result<YoungDiagram, YoungError> {
        YoungDiagram::try_new(self.rows.iter().map(Vec::len).collect())
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn row_size(&self, row: usize) -> usize {
        self.rows.get(row).map_or(0, Vec::len)
    }

    pub fn column_size(&self, col: usize) -> usize {
        self.rows.iter().filter(|row| row.len() > col).count()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.rows.get(row).and_then(|entries| entries.get(col))
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.rows
            .get_mut(row)
            .and_then(|entries| entries.get_mut(col))
    }

    pub fn add_box(&mut self, row: usize, val: T) -> Result<(), YoungError> {
        if row > self.rows.len() {
            return Err(YoungError::InvalidCell { row, col: 0 });
        }
        if row == self.rows.len() {
            self.rows.push(vec![val]);
        } else {
            self.rows[row].push(val);
        }
        if validate_row_shape(&self.rows).is_err() {
            let _ = self.remove_box(row)?;
            return Err(YoungError::ShapeMismatch);
        }
        Ok(())
    }

    pub fn remove_box(&mut self, row: usize) -> Result<Option<T>, YoungError> {
        let Some(current) = self.rows.get_mut(row) else {
            return Err(YoungError::InvalidCell { row, col: 0 });
        };
        let removed = current.pop();
        while self.rows.last().is_some_and(Vec::is_empty) {
            self.rows.pop();
        }
        if !self.rows.is_empty() && validate_row_shape(&self.rows).is_err() {
            return Err(YoungError::ShapeMismatch);
        }
        Ok(removed)
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.multiplicity = BigRational::one();
        self.selfdual_column = 0;
    }

    pub fn swap_columns(&mut self, c1: usize, c2: usize) -> Result<(), YoungError> {
        let n_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        if c1 >= n_cols {
            return Err(YoungError::InvalidCell { row: 0, col: c1 });
        }
        if c2 >= n_cols {
            return Err(YoungError::InvalidCell { row: 0, col: c2 });
        }
        for row in &mut self.rows {
            if row.len() > c1 && row.len() > c2 {
                row.swap(c1, c2);
            }
        }
        Ok(())
    }

    pub fn find(&self, val: &T) -> Option<(usize, usize)> {
        for (row_idx, row) in self.rows.iter().enumerate() {
            for (col_idx, entry) in row.iter().enumerate() {
                if entry == val {
                    return Some((row_idx, col_idx));
                }
            }
        }
        None
    }

    pub fn column_entries(&self, col: usize) -> Vec<(usize, T)> {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row_idx, row)| row.get(col).cloned().map(|value| (row_idx, value)))
            .collect()
    }

    pub fn set_column_entries(&mut self, col: usize, vals: &[T]) -> Result<(), YoungError> {
        let entries = self.column_entries(col);
        if entries.len() != vals.len() {
            return Err(YoungError::ShapeMismatch);
        }
        for ((row_idx, _), value) in entries.iter().zip(vals.iter()) {
            if let Some(cell) = self.rows.get_mut(*row_idx).and_then(|row| row.get_mut(col)) {
                *cell = value.clone();
            } else {
                return Err(YoungError::InvalidCell { row: *row_idx, col });
            }
        }
        Ok(())
    }

    pub fn has_nullifying_trace(&self) -> bool {
        for row_idx in 0..self.n_rows() {
            for col_idx in 0..self.row_size(row_idx) {
                let Some(value) = self.get(row_idx, col_idx) else {
                    continue;
                };
                for other_row in 0..self.column_size(col_idx) {
                    if other_row != row_idx && self.get(other_row, col_idx) == Some(value) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn compare_without_multiplicity(&self, other: &Self) -> bool {
        self.rows == other.rows && self.selfdual_column == other.selfdual_column
    }

    pub fn sort_within_columns(&mut self) -> Result<(), YoungError> {
        let max_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        for col in 0..max_cols {
            let entries = self.column_entries(col);
            if entries.len() <= 1 {
                continue;
            }
            let original: Vec<T> = entries.iter().map(|(_, value)| value.clone()).collect();
            let mut sorted = original.clone();
            sorted.sort();
            let sign = permutation_sign_between(&original, &sorted);
            self.set_column_entries(col, &sorted)?;
            if sign < 0 {
                self.multiplicity = -self.multiplicity.clone();
            }
        }
        Ok(())
    }

    pub fn sort_columns(&mut self) -> Result<(), YoungError> {
        let max_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for col in 0..max_cols {
            groups.entry(self.column_size(col)).or_default().push(col);
        }
        for cols in groups.values() {
            if cols.len() <= 1 {
                continue;
            }
            let mut ordered_columns: Vec<Vec<T>> = cols
                .iter()
                .map(|&col| {
                    self.column_entries(col)
                        .into_iter()
                        .map(|(_, value)| value)
                        .collect()
                })
                .collect();
            ordered_columns.sort();
            for (target_col, values) in cols.iter().copied().zip(ordered_columns.iter()) {
                self.set_column_entries(target_col, values)?;
            }
        }
        Ok(())
    }

    pub fn canonicalise(&mut self) -> Result<(), YoungError> {
        self.sort_within_columns()?;
        self.sort_columns()
    }

    pub fn nonstandard_loc(&self) -> Option<(usize, usize)> {
        for (row_idx, row) in self.rows.iter().enumerate() {
            for col in 0..row.len().saturating_sub(1) {
                if row[col] > row[col + 1] {
                    return Some((row_idx, col + 1));
                }
            }
        }
        None
    }

    pub fn garnir_set(&self, row: usize, col: usize) -> Result<Vec<T>, YoungError> {
        if col == 0 {
            return Err(YoungError::InvalidCell { row, col });
        }
        if !self
            .rows
            .get(row)
            .is_some_and(|current| col < current.len() && col - 1 < current.len())
        {
            return Err(YoungError::InvalidCell { row, col });
        }
        let mut set = Vec::new();
        for r in row..self.column_size(col) {
            if let Some(value) = self.get(r, col) {
                set.push(value.clone());
            }
        }
        for r in row..self.column_size(col - 1) {
            if let Some(value) = self.get(r, col - 1) {
                set.push(value.clone());
            }
        }
        Ok(set)
    }

    pub fn projector(&self, modulo_monoterm: bool) -> Result<Symmetriser<T>, YoungError> {
        let flat = flatten_rows(&self.rows);
        let mut sym = Symmetriser::from_original(flat);

        let mut offset = 0usize;
        for row in &self.rows {
            if row.len() > 1 {
                let positions: Vec<usize> = (offset..offset + row.len()).collect();
                sym.apply_symmetry(&positions, 1);
            }
            offset += row.len();
        }

        if modulo_monoterm {
            for col in 0..self.rows.first().map_or(0, Vec::len) {
                let size = self.column_size(col);
                if size <= 1 {
                    continue;
                }
                let factor = factorial_bigint(size);
                for (_, coeff) in &mut sym.permutations {
                    *coeff *= BigRational::from_integer(factor.clone());
                }
            }
        } else {
            let max_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
            for col in 0..max_cols {
                let positions = self.column_positions(col);
                if positions.len() > 1 {
                    sym.apply_symmetry(&positions, -1);
                }
            }
        }

        sym.collect();
        Ok(sym)
    }

    pub fn projector_normalisation(&self) -> Result<BigRational, YoungError> {
        Ok(BigRational::new(
            BigInt::one(),
            self.shape()?.hook_length_product()?,
        ))
    }

    fn column_positions(&self, col: usize) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut flat_pos = 0usize;
        for row in &self.rows {
            for (row_col, _) in row.iter().enumerate() {
                if row_col == col {
                    positions.push(flat_pos);
                }
                flat_pos += 1;
            }
        }
        positions
    }
}

impl FilledTableau<usize> {
    pub fn standard(diagram: &YoungDiagram) -> Result<YoungTableau, YoungError> {
        let mut rows = Vec::with_capacity(diagram.rows.len());
        let mut counter = 0usize;
        for row_len in &diagram.rows {
            let mut row = Vec::with_capacity(*row_len);
            for _ in 0..*row_len {
                row.push(counter);
                counter += 1;
            }
            rows.push(row);
        }
        YoungTableau::with_metadata(rows, BigRational::one(), 0)
    }

    pub fn is_standard(&self) -> Result<bool, YoungError> {
        validate_row_shape(&self.rows)?;
        let flattened = flatten_rows(&self.rows);
        let n = flattened.len();
        let unique = flattened.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != n {
            return Ok(false);
        }
        if unique
            .iter()
            .copied()
            .enumerate()
            .any(|(idx, value)| idx != value)
        {
            return Ok(false);
        }
        if self
            .rows
            .iter()
            .any(|row| row.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Ok(false);
        }
        let n_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        for col in 0..n_cols {
            let entries: Vec<usize> = self
                .rows
                .iter()
                .filter_map(|row| row.get(col).copied())
                .collect();
            if entries.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn standardize_garnir(&self) -> Result<Tableaux<usize>, YoungError> {
        garnir::standardize_garnir(self)
    }
}

impl<T: Clone + Ord + Eq> Tableaux<T> {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
        }
    }

    pub fn add_tableau(&mut self, tab: FilledTableau<T>) {
        if tab.multiplicity.is_zero() {
            return;
        }
        if let Some(idx) = self
            .storage
            .iter()
            .position(|current| current.compare_without_multiplicity(&tab))
        {
            self.storage[idx].multiplicity += tab.multiplicity;
            if self.storage[idx].multiplicity.is_zero() {
                self.storage.remove(idx);
            }
            return;
        }
        self.storage.push(tab);
    }

    pub fn remove_nullifying_traces(&mut self) {
        self.storage.retain(|tab| !tab.has_nullifying_trace());
    }

    pub fn total_dimension(&self, dim: usize) -> Result<BigInt, YoungError> {
        self.storage.iter().try_fold(BigInt::zero(), |acc, tab| {
            Ok(acc + crate::dimension_gl(&tab.shape()?, dim)?)
        })
    }
}

impl Tableaux<usize> {
    pub fn standard_form(&mut self) -> Result<bool, YoungError> {
        let original = self.storage.clone();
        let mut next = crate::Tableaux::new();
        for tab in &original {
            let standardised = garnir::standardize_garnir(tab)?;
            for entry in standardised.storage {
                next.add_tableau(entry);
            }
        }
        let already_standard = original == next.storage;
        self.storage = next.storage;
        Ok(already_standard)
    }
}

fn validate_row_shape<T>(rows: &[Vec<T>]) -> Result<(), YoungError> {
    let lengths = rows.iter().map(Vec::len).collect::<Vec<_>>();
    YoungDiagram::try_new(lengths).map(|_| ())
}

fn flatten_rows<T: Clone>(rows: &[Vec<T>]) -> Vec<T> {
    rows.iter().flat_map(|row| row.iter().cloned()).collect()
}

fn factorial_bigint(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, value| acc * BigInt::from(value))
}

fn decorate_with_occurrence<T: Clone + Ord + Eq>(values: &[T]) -> Vec<(T, usize)> {
    let mut counts: BTreeMap<T, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let counter = counts.entry(value.clone()).or_default();
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

fn permutation_sign_between<T: Clone + Ord + Eq>(original: &[T], permuted: &[T]) -> i32 {
    let original_order = decorate_with_occurrence(original);
    let permuted_order = decorate_with_occurrence(permuted);
    let mut position_of = BTreeMap::new();
    for (idx, item) in permuted_order.iter().enumerate() {
        position_of.insert(item.clone(), idx);
    }
    let perm = original_order
        .iter()
        .map(|item| position_of.get(item).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    inversion_parity(&perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_tableau_for_two_one() {
        let diagram = YoungDiagram::try_new(vec![2, 1]).unwrap();
        let tableau = YoungTableau::standard(&diagram).unwrap();
        assert_eq!(tableau.rows, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn standard_detection() {
        let standard =
            YoungTableau::with_metadata(vec![vec![0, 1], vec![2]], BigRational::one(), 0).unwrap();
        assert_eq!(standard.is_standard(), Ok(true));

        let nonstandard =
            YoungTableau::with_metadata(vec![vec![1, 2], vec![0]], BigRational::one(), 0).unwrap();
        assert_eq!(nonstandard.is_standard(), Ok(false));

        let duplicate =
            YoungTableau::with_metadata(vec![vec![0, 0], vec![1]], BigRational::one(), 0).unwrap();
        assert_eq!(duplicate.is_standard(), Ok(false));
    }
}
