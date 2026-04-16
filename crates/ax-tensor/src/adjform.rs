#![forbid(unsafe_code)]

use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::BTreeMap;

/// Adjacency-form representation of index contractions in a tensor monomial.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Adjform {
    pub data: Vec<i32>,
}

impl Adjform {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn from_indices(indices: &[ax_ir::Index]) -> Self {
        let n = indices.len();
        let mut data = vec![0i32; n];
        let mut used = vec![false; n];
        let mut dummy_names = std::collections::BTreeSet::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if indices[i].name == indices[j].name && indices[i].variance != indices[j].variance
                {
                    dummy_names.insert(indices[i].name);
                }
            }
        }

        let free_names: std::collections::BTreeSet<lasso::Spur> = indices
            .iter()
            .filter(|idx| !dummy_names.contains(&idx.name))
            .map(|idx| idx.name)
            .collect();
        let mut free_labels: BTreeMap<lasso::Spur, i32> = BTreeMap::new();
        for (offset, name) in free_names.into_iter().enumerate() {
            free_labels.insert(name, -((offset as i32) + 1));
        }

        for i in 0..n {
            if used[i] {
                continue;
            }

            let mut found_pair = false;
            for j in (i + 1)..n {
                if used[j] {
                    continue;
                }
                if indices[i].name == indices[j].name && indices[i].variance != indices[j].variance
                {
                    data[i] = j as i32;
                    data[j] = i as i32;
                    used[i] = true;
                    used[j] = true;
                    found_pair = true;
                    break;
                }
            }

            if !found_pair {
                let label = *free_labels
                    .get(&indices[i].name)
                    .unwrap_or(&(-((i as i32) + 1)));
                data[i] = label;
                used[i] = true;
            }
        }

        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_free(&self, pos: usize) -> bool {
        self.data[pos] < 0
    }

    pub fn is_dummy(&self, pos: usize) -> bool {
        self.data[pos] >= 0
    }

    pub fn n_free(&self) -> usize {
        self.data.iter().filter(|&&v| v < 0).count()
    }

    pub fn n_dummy_pairs(&self) -> usize {
        self.data.iter().filter(|&&v| v >= 0).count() / 2
    }

    /// Apply a permutation to the adjform. This permutes the positions.
    pub fn permute(&self, perm: &[usize]) -> Self {
        let n = self.data.len();
        let mut new_data = vec![0i32; n];
        for i in 0..n {
            let val = self.data[i];
            if val < 0 {
                new_data[perm[i]] = val;
            } else {
                new_data[perm[i]] = perm[val as usize] as i32;
            }
        }
        Self { data: new_data }
    }
}

impl Default for Adjform {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct TableauInfo {
    pub rows: Vec<Vec<usize>>,
    pub columns: Vec<Vec<usize>>,
    pub trace_free: bool,
    pub duality: ax_ir::DualityKind,
}

impl TableauInfo {
    pub fn from_realized(
        realized: &crate::symmetry_bridge::RealizedTableau,
        offset: usize,
    ) -> Self {
        let rows = realized
            .projector
            .tableau
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .filter_map(|label| realized.slot_map.get(*label).map(|slot| offset + *slot))
                    .collect::<Vec<_>>()
            })
            .filter(|row| !row.is_empty())
            .collect::<Vec<_>>();

        let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut columns = Vec::new();
        for col in 0..max_cols {
            let mut column = Vec::new();
            for row in &rows {
                if let Some(slot) = row.get(col) {
                    column.push(*slot);
                }
            }
            if column.len() > 1 {
                columns.push(column);
            }
        }

        Self {
            rows,
            columns,
            trace_free: realized.trace_free,
            duality: realized.duality.clone(),
        }
    }
}

/// A projected adjform: a linear combination of Adjforms with rational coefficients.
#[derive(Clone, Debug)]
pub struct ProjectedAdjform {
    pub terms: std::collections::BTreeMap<Adjform, BigRational>,
}

impl ProjectedAdjform {
    pub fn new() -> Self {
        Self {
            terms: std::collections::BTreeMap::new(),
        }
    }

    pub fn from_adjform(adj: Adjform, coeff: BigRational) -> Self {
        let mut terms = std::collections::BTreeMap::new();
        if coeff != BigRational::from_integer(BigInt::from(0)) {
            terms.insert(adj, coeff);
        }
        Self { terms }
    }

    pub fn add(&mut self, adj: Adjform, coeff: BigRational) {
        if coeff == BigRational::from_integer(BigInt::from(0)) {
            return;
        }
        let new_coeff = self
            .terms
            .get(&adj)
            .cloned()
            .unwrap_or_else(|| BigRational::from_integer(BigInt::from(0)))
            + coeff;
        if new_coeff == BigRational::from_integer(BigInt::from(0)) {
            self.terms.remove(&adj);
        } else {
            self.terms.insert(adj, new_coeff);
        }
    }

    pub fn combine(&mut self, other: &ProjectedAdjform) {
        for (adj, coeff) in &other.terms {
            self.add(adj.clone(), coeff.clone());
        }
    }

    pub fn multiply(&mut self, factor: BigRational) {
        if factor == BigRational::from_integer(BigInt::from(0)) {
            self.terms.clear();
            return;
        }
        for coeff in self.terms.values_mut() {
            *coeff = coeff.clone() * factor.clone();
        }
        self.terms
            .retain(|_, v| *v != BigRational::from_integer(BigInt::from(0)));
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn get_coeff(&self, adj: &Adjform) -> BigRational {
        self.terms
            .get(adj)
            .cloned()
            .unwrap_or_else(|| BigRational::from_integer(BigInt::from(0)))
    }

    /// Apply symmetrization in the given index positions.
    pub fn symmetrize(&mut self, positions: &[usize]) {
        if positions.len() <= 1 || self.terms.is_empty() {
            return;
        }

        let mut new_terms = ProjectedAdjform::new();
        let mut perm_indices: Vec<usize> = positions.to_vec();

        loop {
            let n = self.terms.keys().next().map_or(0, |a| a.len());
            if n == 0 {
                break;
            }
            let mut full_perm: ax_perm::Perm = (0..n).collect();
            for (i, &pos) in positions.iter().enumerate() {
                full_perm[pos] = perm_indices[i];
            }

            for (adj, coeff) in &self.terms {
                let permuted = adj.permute(&full_perm);
                new_terms.add(permuted, coeff.clone());
            }

            if !next_permutation(&mut perm_indices) {
                break;
            }
        }

        *self = new_terms;
    }

    /// Apply antisymmetrization in the given index positions.
    pub fn antisymmetrize(&mut self, positions: &[usize]) {
        if positions.len() <= 1 || self.terms.is_empty() {
            return;
        }

        let mut new_terms = ProjectedAdjform::new();
        let mut perm_indices: Vec<usize> = positions.to_vec();

        loop {
            let n = self.terms.keys().next().map_or(0, |a| a.len());
            if n == 0 {
                break;
            }
            let mut full_perm: ax_perm::Perm = (0..n).collect();
            for (i, &pos) in positions.iter().enumerate() {
                full_perm[pos] = perm_indices[i];
            }
            let s = ax_perm::sign(&full_perm);

            for (adj, coeff) in &self.terms {
                let permuted = adj.permute(&full_perm);
                new_terms.add(
                    permuted,
                    coeff.clone() * BigRational::from_integer(BigInt::from(s)),
                );
            }

            if !next_permutation(&mut perm_indices) {
                break;
            }
        }

        *self = new_terms;
    }

    /// Apply the full Young symmetrizer for a given set of tableaux.
    pub fn young_project(&mut self, tableaux: &[TableauInfo]) {
        for tab in tableaux {
            for col_group in &tab.columns {
                if col_group.len() > 1 {
                    self.antisymmetrize(col_group);
                }
            }
            for row_group in &tab.rows {
                if row_group.len() > 1 {
                    self.symmetrize(row_group);
                }
            }
        }
    }
}

impl Default for ProjectedAdjform {
    fn default() -> Self {
        Self::new()
    }
}

fn next_permutation(arr: &mut [usize]) -> bool {
    let n = arr.len();
    if n <= 1 {
        return false;
    }
    let mut i = n - 2;
    loop {
        if arr[i] < arr[i + 1] {
            break;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    let mut j = n - 1;
    while arr[j] <= arr[i] {
        j -= 1;
    }
    arr.swap(i, j);
    arr[i + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Index, Variance};

    fn make_interner() -> ax_ir::Interner {
        ax_ir::Interner::new()
    }

    #[test]
    fn adjform_from_free_indices() {
        let int = make_interner();
        let mu = int.get_or_intern("mu");
        let nu = int.get_or_intern("nu");
        let indices = vec![
            Index {
                name: mu,
                variance: Variance::Down,
                index_type: None,
            },
            Index {
                name: nu,
                variance: Variance::Down,
                index_type: None,
            },
        ];
        let adj = Adjform::from_indices(&indices);
        assert!(adj.is_free(0));
        assert!(adj.is_free(1));
        assert_eq!(adj.n_free(), 2);
    }

    #[test]
    fn adjform_from_dummy_pair() {
        let int = make_interner();
        let mu = int.get_or_intern("mu");
        let indices = vec![
            Index {
                name: mu,
                variance: Variance::Up,
                index_type: None,
            },
            Index {
                name: mu,
                variance: Variance::Down,
                index_type: None,
            },
        ];
        let adj = Adjform::from_indices(&indices);
        assert!(adj.is_dummy(0));
        assert!(adj.is_dummy(1));
        assert_eq!(adj.data[0], 1);
        assert_eq!(adj.data[1], 0);
    }

    #[test]
    fn projected_adjform_combine() {
        let adj1 = Adjform { data: vec![-1, -2] };
        let adj2 = Adjform { data: vec![-2, -1] };

        let mut pa =
            ProjectedAdjform::from_adjform(adj1.clone(), BigRational::from_integer(1.into()));
        pa.add(adj2.clone(), BigRational::from_integer(1.into()));
        assert_eq!(pa.len(), 2);

        pa.add(adj1, BigRational::from_integer((-1).into()));
        assert_eq!(pa.len(), 1);
    }
}
