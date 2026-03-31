#![forbid(unsafe_code)]

/// A Young diagram/tableau. Stored as a list of row lengths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YoungDiagram {
    pub rows: Vec<usize>,
}

/// A filled Young tableau: each cell contains an index number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YoungTableau {
    pub cells: Vec<Vec<usize>>,
}

impl YoungDiagram {
    pub fn new(rows: Vec<usize>) -> Self {
        let mut rows = rows;
        rows.sort_by(|a, b| b.cmp(a));
        rows.retain(|&r| r > 0);
        Self { rows }
    }

    pub fn n_cells(&self) -> usize {
        self.rows.iter().sum()
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn column_lengths(&self) -> Vec<usize> {
        if self.rows.is_empty() {
            return vec![];
        }
        let max_cols = self.rows[0];
        (0..max_cols)
            .map(|col| self.rows.iter().filter(|&&r| r > col).count())
            .collect()
    }

    /// Conjugate (transpose) diagram.
    pub fn conjugate(&self) -> Self {
        Self::new(self.column_lengths())
    }
}

impl YoungTableau {
    /// Create a standard filling: cells filled 0, 1, 2, ... left to right, top to bottom.
    pub fn standard(diagram: &YoungDiagram) -> Self {
        let mut cells = Vec::new();
        let mut counter = 0usize;
        for &row_len in &diagram.rows {
            let mut row = Vec::new();
            for _ in 0..row_len {
                row.push(counter);
                counter += 1;
            }
            cells.push(row);
        }
        Self { cells }
    }

    pub fn n_rows(&self) -> usize {
        self.cells.len()
    }

    pub fn row_size(&self, row: usize) -> usize {
        self.cells.get(row).map_or(0, |r| r.len())
    }

    pub fn column_size(&self, col: usize) -> usize {
        self.cells.iter().filter(|row| row.len() > col).count()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        self.cells.get(row).and_then(|r| r.get(col)).copied()
    }
}

/// Generate the symmetrizer permutations for a Young tableau.
/// Returns row symmetrizers (symmetric within each row).
pub fn row_symmetrizer_generators(tab: &YoungTableau, n: usize) -> Vec<ax_perm::Perm> {
    let mut gens = Vec::new();
    for row in &tab.cells {
        for i in 0..row.len().saturating_sub(1) {
            let mut p: ax_perm::Perm = (0..n).collect();
            p.swap(row[i], row[i + 1]);
            gens.push(p);
        }
    }
    gens
}

pub fn column_antisymmetrizer_generators(tab: &YoungTableau, n: usize) -> Vec<ax_perm::Perm> {
    let mut gens = Vec::new();
    let n_cols = tab.cells.first().map_or(0, |r| r.len());
    for col in 0..n_cols {
        let col_indices: Vec<usize> = tab
            .cells
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        for i in 0..col_indices.len().saturating_sub(1) {
            let mut p: ax_perm::Perm = (0..n).collect();
            p.swap(col_indices[i], col_indices[i + 1]);
            gens.push(p);
        }
    }
    gens
}

/// Check if a given symmetry is compatible with a Young diagram shape.
/// This returns the dimension of the corresponding irreducible representation of S_k.
pub fn dimension_of_representation(diagram: &YoungDiagram, _n: usize) -> u64 {
    let total = diagram.n_cells();
    if total == 0 {
        return 1;
    }

    let mut hook_product = 1u64;
    for (i, &row_len) in diagram.rows.iter().enumerate() {
        for j in 0..row_len {
            let col_len = diagram.rows.iter().filter(|&&r| r > j).count();
            let hook = (row_len - j) + (col_len - i) - 1;
            hook_product *= hook as u64;
        }
    }

    let mut factorial = 1u64;
    for k in 1..=total {
        factorial *= k as u64;
    }

    factorial / hook_product
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn young_diagram_basics() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        assert_eq!(d.n_cells(), 6);
        assert_eq!(d.n_rows(), 3);
        assert_eq!(d.column_lengths(), vec![3, 2, 1]);
    }

    #[test]
    fn young_conjugate() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        let c = d.conjugate();
        assert_eq!(c.rows, vec![3, 2, 1]);
    }

    #[test]
    fn young_standard_filling() {
        let d = YoungDiagram::new(vec![3, 2]);
        let t = YoungTableau::standard(&d);
        assert_eq!(t.get(0, 0), Some(0));
        assert_eq!(t.get(0, 2), Some(2));
        assert_eq!(t.get(1, 0), Some(3));
        assert_eq!(t.get(1, 1), Some(4));
    }

    #[test]
    fn row_generators() {
        let d = YoungDiagram::new(vec![3]);
        let t = YoungTableau::standard(&d);
        let gens = row_symmetrizer_generators(&t, 3);
        assert_eq!(gens.len(), 2);
    }

    #[test]
    fn dimension_of_trivial() {
        let d = YoungDiagram::new(vec![3]);
        assert_eq!(dimension_of_representation(&d, 3), 1);
    }

    #[test]
    fn dimension_of_alternating() {
        let d = YoungDiagram::new(vec![1, 1, 1]);
        assert_eq!(dimension_of_representation(&d, 3), 1);
    }

    #[test]
    fn dimension_of_standard() {
        let d = YoungDiagram::new(vec![2, 1]);
        assert_eq!(dimension_of_representation(&d, 3), 2);
    }
}
