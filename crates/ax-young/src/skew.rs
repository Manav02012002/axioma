use crate::{YoungDiagram, YoungError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkewDiagram {
    pub outer: YoungDiagram,
    pub inner: YoungDiagram,
}

impl SkewDiagram {
    pub fn try_new(outer: YoungDiagram, inner: YoungDiagram) -> Result<Self, YoungError> {
        let outer_rows = outer.rows.clone();
        let inner_rows = inner.rows.clone();
        for row in 0..outer.n_rows().max(inner.n_rows()) {
            let outer_len = outer.row_len(row).unwrap_or(0);
            let inner_len = inner.row_len(row).unwrap_or(0);
            if inner_len > outer_len {
                return Err(YoungError::InnerDiagramNotContained {
                    outer: outer_rows,
                    inner: inner_rows,
                });
            }
        }
        Ok(Self { outer, inner })
    }

    pub fn n_rows(&self) -> usize {
        self.outer.n_rows()
    }

    pub fn n_cells(&self) -> usize {
        self.outer.n_cells() - self.inner.n_cells()
    }

    pub fn outer_row_len(&self, row: usize) -> usize {
        self.outer.row_len(row).unwrap_or(0)
    }

    pub fn inner_row_len(&self, row: usize) -> usize {
        self.inner.row_len(row).unwrap_or(0)
    }

    pub fn row_interval(&self, row: usize) -> Option<(usize, usize)> {
        let start = self.inner_row_len(row);
        let end = self.outer_row_len(row);
        (start < end).then_some((start, end))
    }

    pub fn contains_cell(&self, row: usize, col: usize) -> bool {
        self.row_interval(row)
            .is_some_and(|(start, end)| start <= col && col < end)
    }

    pub fn cells_row_major(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::with_capacity(self.n_cells());
        for row in 0..self.n_rows() {
            if let Some((start, end)) = self.row_interval(row) {
                for col in start..end {
                    cells.push((row, col));
                }
            }
        }
        cells
    }

    pub fn cells_reading_order(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::with_capacity(self.n_cells());
        for row in 0..self.n_rows() {
            if let Some((start, end)) = self.row_interval(row) {
                for col in (start..end).rev() {
                    cells.push((row, col));
                }
            }
        }
        cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_diagram_properties_for_three_two_over_one() {
        let skew = SkewDiagram::try_new(
            YoungDiagram::try_new(vec![3, 2]).unwrap(),
            YoungDiagram::try_new(vec![1]).unwrap(),
        )
        .unwrap();
        assert_eq!(skew.n_rows(), 2);
        assert_eq!(skew.n_cells(), 4);
        assert_eq!(skew.row_interval(0), Some((1, 3)));
        assert_eq!(skew.row_interval(1), Some((0, 2)));
    }

    #[test]
    fn skew_diagram_rejects_noncontained_inner_shape() {
        assert_eq!(
            SkewDiagram::try_new(
                YoungDiagram::try_new(vec![2]).unwrap(),
                YoungDiagram::try_new(vec![3]).unwrap()
            ),
            Err(YoungError::InnerDiagramNotContained {
                outer: vec![2],
                inner: vec![3],
            })
        );
    }

    #[test]
    fn skew_diagram_reading_order_is_exact() {
        let skew = SkewDiagram::try_new(
            YoungDiagram::try_new(vec![3, 2]).unwrap(),
            YoungDiagram::try_new(vec![1]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            skew.cells_reading_order(),
            vec![(0, 2), (0, 1), (1, 1), (1, 0)]
        );
    }
}
