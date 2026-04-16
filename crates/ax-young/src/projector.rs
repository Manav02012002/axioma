use crate::{YoungDiagram, YoungError, YoungTableau};
use ax_perm::Perm;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymmetryOperatorKind {
    RowSymmetrizer,
    ColumnAntisymmetrizer,
    YoungProjector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermutationTerm {
    pub images: Vec<usize>,
    pub coefficient: BigRational,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LazyProjector {
    pub diagram: YoungDiagram,
    pub tableau: YoungTableau,
    pub row_generator_set: Vec<Vec<usize>>,
    pub column_generator_set: Vec<Vec<usize>>,
    pub normalized: bool,
}

impl LazyProjector {
    pub fn to_group_backed(
        &self,
    ) -> Result<crate::group_action::GroupBackedProjector, crate::group_action::GroupProjectorError>
    {
        let normalization = if self.normalized {
            crate::group_action::ProjectorNormalization::HookLength
        } else {
            crate::group_action::ProjectorNormalization::Unnormalized
        };
        crate::group_action::build_group_backed_projector(&self.tableau, normalization)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symmetriser<T: Clone + Ord + Eq> {
    pub original: Vec<T>,
    pub permutations: Vec<(Vec<T>, BigRational)>,
}

impl<T: Clone + Ord + Eq> Symmetriser<T> {
    pub fn new() -> Self {
        Self {
            original: Vec::new(),
            permutations: vec![(Vec::new(), BigRational::one())],
        }
    }

    pub fn from_original(original: Vec<T>) -> Self {
        Self {
            original: original.clone(),
            permutations: vec![(original, BigRational::one())],
        }
    }

    pub fn apply_symmetry(&mut self, positions: &[usize], sign: i32) {
        if positions.len() <= 1 {
            return;
        }
        let mut next = Vec::new();
        let perms = permutations_of_indices(positions.len());
        for (perm, parity) in perms {
            for (values, coeff) in &self.permutations {
                let mut candidate = values.clone();
                for (dest_idx, src_idx) in positions.iter().zip(perm.iter()) {
                    candidate[*dest_idx] = values[positions[*src_idx]].clone();
                }
                let factor = if sign < 0 && parity < 0 { -1 } else { 1 };
                next.push((
                    candidate,
                    coeff.clone() * BigRational::from_integer(BigInt::from(factor)),
                ));
            }
        }
        self.permutations = next;
        self.collect();
    }

    pub fn collect(&mut self) {
        let mut combined: BTreeMap<Vec<T>, BigRational> = BTreeMap::new();
        for (values, coeff) in self.permutations.drain(..) {
            *combined.entry(values).or_insert_with(BigRational::zero) += coeff;
        }
        self.permutations = combined
            .into_iter()
            .filter(|(_, coeff)| !coeff.is_zero())
            .collect();
    }

    pub fn size(&self) -> usize {
        self.permutations.len()
    }
}

impl<T: Clone + Ord + Eq> Default for Symmetriser<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn row_symmetrizer_generators(tab: &YoungTableau, n: usize) -> Result<Vec<Perm>, YoungError> {
    validate_permutation_degree(tab, n)?;
    let mut gens = Vec::new();
    for row in &tab.rows {
        for i in 0..row.len().saturating_sub(1) {
            let mut p: Perm = (0..n).collect();
            p.swap(row[i], row[i + 1]);
            gens.push(p);
        }
    }
    Ok(gens)
}

pub fn column_antisymmetrizer_generators(
    tab: &YoungTableau,
    n: usize,
) -> Result<Vec<Perm>, YoungError> {
    validate_permutation_degree(tab, n)?;
    let mut gens = Vec::new();
    let n_cols = tab.rows.iter().map(Vec::len).max().unwrap_or(0);
    for col in 0..n_cols {
        let indices: Vec<usize> = tab
            .rows
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        for i in 0..indices.len().saturating_sub(1) {
            let mut p: Perm = (0..n).collect();
            p.swap(indices[i], indices[i + 1]);
            gens.push(p);
        }
    }
    Ok(gens)
}

pub fn lazy_projector(tab: &YoungTableau) -> Result<LazyProjector, YoungError> {
    let diagram = tab.shape()?;
    Ok(LazyProjector {
        row_generator_set: tab.rows.clone(),
        column_generator_set: (0..diagram.n_cols())
            .map(|col| {
                tab.rows
                    .iter()
                    .filter_map(|row| row.get(col).copied())
                    .collect::<Vec<_>>()
            })
            .filter(|column| !column.is_empty())
            .collect(),
        normalized: false,
        diagram,
        tableau: tab.clone(),
    })
}

pub fn expand_lazy_projector(
    projector: &LazyProjector,
) -> Result<Vec<PermutationTerm>, YoungError> {
    let n = projector.diagram.n_cells();
    let mut terms = vec![PermutationTerm {
        images: (0..n).collect(),
        coefficient: BigRational::one(),
    }];

    for row in &projector.row_generator_set {
        terms = apply_group_terms(&terms, row, false);
    }
    for column in &projector.column_generator_set {
        terms = apply_group_terms(&terms, column, true);
    }

    combine_terms(terms)
}

fn validate_permutation_degree(tab: &YoungTableau, n: usize) -> Result<(), YoungError> {
    let actual = tab
        .rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);
    if actual != n {
        return Err(YoungError::PermutationDegreeMismatch {
            expected: n,
            actual,
        });
    }
    Ok(())
}

fn apply_group_terms(
    current: &[PermutationTerm],
    positions: &[usize],
    antisymmetric: bool,
) -> Vec<PermutationTerm> {
    if positions.len() <= 1 {
        return current.to_vec();
    }
    let permutations = permutations_of_indices(positions.len());
    let mut next = Vec::new();
    for term in current {
        for (perm, parity) in &permutations {
            let mut images = term.images.clone();
            for (dest_idx, src_idx) in positions.iter().zip(perm.iter()) {
                images[*dest_idx] = term.images[positions[*src_idx]];
            }
            let coefficient = if antisymmetric && *parity < 0 {
                -term.coefficient.clone()
            } else {
                term.coefficient.clone()
            };
            next.push(PermutationTerm {
                images,
                coefficient,
            });
        }
    }
    next
}

fn combine_terms(terms: Vec<PermutationTerm>) -> Result<Vec<PermutationTerm>, YoungError> {
    let mut combined: BTreeMap<Vec<usize>, BigRational> = BTreeMap::new();
    for term in terms {
        *combined
            .entry(term.images)
            .or_insert_with(BigRational::zero) += term.coefficient;
    }
    Ok(combined
        .into_iter()
        .filter(|(_, coeff)| !coeff.is_zero())
        .map(|(images, coefficient)| PermutationTerm {
            images,
            coefficient,
        })
        .collect())
}

fn permutations_of_indices(n: usize) -> Vec<(Vec<usize>, i32)> {
    let mut values: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    heap_permute(&mut values, 0, &mut out);
    out.into_iter()
        .map(|perm| {
            let sign = inversion_parity(&perm);
            (perm, sign)
        })
        .collect()
}

fn heap_permute<T: Clone>(values: &mut [T], start: usize, out: &mut Vec<Vec<T>>) {
    if start == values.len() {
        out.push(values.to_vec());
        return;
    }
    for idx in start..values.len() {
        values.swap(start, idx);
        heap_permute(values, start + 1, out);
        values.swap(start, idx);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    #[test]
    fn row_symmetrizer_generators_for_two_boxes() {
        let tab = YoungTableau::with_metadata(vec![vec![0, 1]], BigRational::one(), 0).unwrap();
        let generators = row_symmetrizer_generators(&tab, 2).unwrap();
        assert_eq!(generators.len(), 1);
        assert_eq!(generators[0], vec![1, 0]);
    }

    #[test]
    fn column_antisymmetrizer_generators_for_two_boxes() {
        let tab =
            YoungTableau::with_metadata(vec![vec![0], vec![1]], BigRational::one(), 0).unwrap();
        let generators = column_antisymmetrizer_generators(&tab, 2).unwrap();
        assert_eq!(generators.len(), 1);
        assert_eq!(generators[0], vec![1, 0]);
    }

    #[test]
    fn lazy_projector_preserves_shape_and_tableau() {
        let tab =
            YoungTableau::with_metadata(vec![vec![0, 1], vec![2]], BigRational::one(), 0).unwrap();
        let projector = lazy_projector(&tab).unwrap();
        assert_eq!(projector.diagram.rows, vec![2, 1]);
        assert_eq!(projector.tableau, tab);
    }

    #[test]
    fn expand_lazy_projector_identity_for_single_box() {
        let tab = YoungTableau::with_metadata(vec![vec![0]], BigRational::one(), 0).unwrap();
        let projector = lazy_projector(&tab).unwrap();
        let terms = expand_lazy_projector(&projector).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].images, vec![0]);
        assert_eq!(terms[0].coefficient, BigRational::one());
    }
}
