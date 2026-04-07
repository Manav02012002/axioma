#![forbid(unsafe_code)]

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YoungDiagram {
    pub rows: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilledTableau<T: Clone + Ord + Eq> {
    pub rows: Vec<Vec<T>>,
    pub multiplicity: BigRational,
    pub selfdual_column: i32,
}

pub type YoungTableau = FilledTableau<usize>;

#[derive(Clone, Debug)]
pub struct Symmetriser<T: Clone + Ord + Eq> {
    pub original: Vec<T>,
    pub permutations: Vec<(Vec<T>, BigRational)>,
}

#[derive(Clone, Debug)]
pub struct Tableaux<T: Clone + Ord + Eq> {
    pub storage: Vec<FilledTableau<T>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LRBox<T: Clone + Ord + Eq> {
    value: T,
    source_row: usize,
    source_col: usize,
    added: bool,
}

impl YoungDiagram {
    pub fn new(rows: Vec<usize>) -> Self {
        let mut rows = rows;
        rows.retain(|&row| row > 0);
        for idx in 1..rows.len() {
            assert!(
                rows[idx - 1] >= rows[idx],
                "Young diagram rows must be weakly decreasing"
            );
        }
        Self { rows }
    }

    pub fn n_cells(&self) -> usize {
        self.rows.iter().sum()
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn column_lengths(&self) -> Vec<usize> {
        let max_cols = self.rows.first().copied().unwrap_or(0);
        (0..max_cols)
            .map(|col| self.rows.iter().filter(|&&row| row > col).count())
            .collect()
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.column_lengths())
    }

    pub fn hook_length(&self, row: usize, col: usize) -> usize {
        let arm = self.rows[row] - col - 1;
        let leg = self.column_lengths()[col] - row - 1;
        arm + leg + 1
    }

    pub fn hook_length_product(&self) -> BigInt {
        let mut product = BigInt::one();
        for (row, &row_len) in self.rows.iter().enumerate() {
            for col in 0..row_len {
                product *= BigInt::from(self.hook_length(row, col));
            }
        }
        product
    }

    pub fn dimension(&self, n: usize) -> BigInt {
        if self.rows.is_empty() {
            return BigInt::one();
        }

        let mut numerator = BigInt::one();
        for (row, &row_len) in self.rows.iter().enumerate() {
            for col in 0..row_len {
                numerator *= BigInt::from(n + col - row);
            }
        }
        numerator / self.hook_length_product()
    }
}

pub fn dimension_of_representation(diagram: &YoungDiagram, n: usize) -> u64 {
    diagram.dimension(n).try_into().unwrap_or(u64::MAX)
}

impl<T: Clone + Ord + Eq> FilledTableau<T> {
    pub fn standard(diagram: &YoungDiagram) -> FilledTableau<usize> {
        let mut rows = Vec::new();
        let mut counter = 0usize;
        for &row_len in &diagram.rows {
            let mut row = Vec::with_capacity(row_len);
            for _ in 0..row_len {
                row.push(counter);
                counter += 1;
            }
            rows.push(row);
        }
        FilledTableau {
            rows,
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        }
    }

    pub fn shape(&self) -> YoungDiagram {
        YoungDiagram::new(self.rows.iter().map(Vec::len).collect())
    }

    pub fn number_of_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn n_rows(&self) -> usize {
        self.number_of_rows()
    }

    pub fn row_size(&self, row: usize) -> usize {
        self.rows.get(row).map_or(0, Vec::len)
    }

    pub fn column_size(&self, col: usize) -> usize {
        self.rows.iter().filter(|row| row.len() > col).count()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.rows.get(row).and_then(|current| current.get(col))
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.rows.get_mut(row).and_then(|current| current.get_mut(col))
    }

    pub fn add_box(&mut self, row: usize, val: T) {
        if row >= self.rows.len() {
            self.rows.resize_with(row + 1, Vec::new);
        }
        self.rows[row].push(val);
    }

    pub fn remove_box(&mut self, row: usize) {
        if let Some(current) = self.rows.get_mut(row) {
            current.pop();
        }
        while matches!(self.rows.last(), Some(last) if last.is_empty()) {
            self.rows.pop();
        }
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.multiplicity = BigRational::one();
        self.selfdual_column = 0;
    }

    pub fn swap_columns(&mut self, c1: usize, c2: usize) {
        for row in &mut self.rows {
            if row.len() > c1 && row.len() > c2 {
                row.swap(c1, c2);
            }
        }
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

    pub fn sort_within_columns(&mut self) {
        let max_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        for col in 0..max_cols {
            let entries: Vec<(usize, T)> = self.column_entries(col);
            if entries.len() <= 1 {
                continue;
            }

            let original: Vec<T> = entries.iter().map(|(_, value)| value.clone()).collect();
            let mut sorted = original.clone();
            sorted.sort();
            let sign = permutation_sign_between(&original, &sorted);
            self.set_column_entries(col, &sorted);
            if sign < 0 {
                self.multiplicity = -self.multiplicity.clone();
            }
        }
    }

    pub fn sort_columns(&mut self) {
        let max_cols = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for col in 0..max_cols {
            groups.entry(self.column_size(col)).or_default().push(col);
        }

        for cols in groups.values() {
            if cols.len() <= 1 {
                continue;
            }

            let ordered_columns: Vec<Vec<T>> = {
                let mut columns: Vec<Vec<T>> = cols
                    .iter()
                    .map(|&col| self.column_entries(col).into_iter().map(|(_, value)| value).collect())
                    .collect();
                columns.sort();
                columns
            };

            for (target_col, values) in cols.iter().copied().zip(ordered_columns.iter()) {
                self.set_column_entries(target_col, values);
            }
        }
    }

    pub fn canonicalise(&mut self) {
        self.sort_within_columns();
        self.sort_columns();
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

    pub fn garnir_set(&self, row: usize, col: usize) -> Vec<T> {
        assert!(col > 0, "Garnir set is only defined for col > 0");
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
        set
    }

    pub fn has_nullifying_trace(&self) -> bool {
        for row_idx in 0..self.number_of_rows() {
            for col_idx in 0..self.row_size(row_idx) {
                let Some(value) = self.get(row_idx, col_idx) else {
                    continue;
                };
                for other_row in 0..self.column_size(col_idx) {
                    if other_row == row_idx {
                        continue;
                    }
                    if self.get(other_row, col_idx) == Some(value) {
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

    pub fn projector(&self, modulo_monoterm: bool) -> Symmetriser<T> {
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
                let positions: Vec<usize> = self.column_positions(col);
                if positions.len() > 1 {
                    sym.apply_symmetry(&positions, -1);
                }
            }
        }

        sym.collect();
        sym
    }

    pub fn projector_normalisation(&self) -> BigRational {
        BigRational::new(BigInt::one(), self.shape().hook_length_product())
    }

    fn column_entries(&self, col: usize) -> Vec<(usize, T)> {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row_idx, row)| row.get(col).cloned().map(|value| (row_idx, value)))
            .collect()
    }

    fn set_column_entries(&mut self, col: usize, values: &[T]) {
        for ((row_idx, _), value) in self.column_entries(col).iter().zip(values.iter()) {
            self.rows[*row_idx][col] = value.clone();
        }
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

    pub fn apply_value_symmetry(&mut self, values: &[T], sign: i32) {
        if values.len() <= 1 {
            return;
        }
        let mut used = vec![false; self.original.len()];
        let mut positions = Vec::new();
        for value in values {
            let Some(pos) = self
                .original
                .iter()
                .enumerate()
                .find(|(idx, candidate)| !used[*idx] && *candidate == value)
                .map(|(idx, _)| idx)
            else {
                return;
            };
            used[pos] = true;
            positions.push(pos);
        }
        self.apply_symmetry(&positions, sign);
    }

    pub fn size(&self) -> usize {
        self.permutations.len()
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
}

impl<T: Clone + Ord + Eq> Default for Symmetriser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Ord + Eq> Tableaux<T> {
    pub fn new() -> Self {
        Self { storage: Vec::new() }
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

    pub fn standard_form(&mut self) -> bool {
        let mut already_standard = true;
        loop {
            let mut changed = false;
            let mut next = Tableaux::new();

            for mut tableau in self.storage.drain(..) {
                tableau.sort_within_columns();
                let Some((row, col)) = tableau.nonstandard_loc() else {
                    next.add_tableau(tableau);
                    continue;
                };

                already_standard = false;
                changed = true;

                let left_col = col - 1;
                let right_positions: Vec<(usize, usize)> = (row..tableau.column_size(col))
                    .map(|r| (r, col))
                    .collect();
                let left_positions: Vec<(usize, usize)> = (row..tableau.column_size(left_col))
                    .map(|r| (r, left_col))
                    .collect();

                let right_values: Vec<T> = right_positions
                    .iter()
                    .map(|&(r, c)| tableau.get(r, c).unwrap().clone())
                    .collect();
                let left_values: Vec<T> = left_positions
                    .iter()
                    .map(|&(r, c)| tableau.get(r, c).unwrap().clone())
                    .collect();
                let union = concatenate(&right_values, &left_values);
                for left_choice in ordered_subsets(union.len(), left_positions.len()) {
                    let identity_choice: Vec<usize> =
                        (right_values.len()..right_values.len() + left_values.len()).collect();
                    if left_choice == identity_choice {
                        continue;
                    }

                    let left_choice_set: BTreeMap<usize, ()> =
                        left_choice.iter().copied().map(|idx| (idx, ())).collect();
                    let left_block: Vec<T> = left_choice
                        .iter()
                        .map(|idx| union[*idx].clone())
                        .collect();
                    let right_block: Vec<T> = union
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| !left_choice_set.contains_key(idx))
                        .map(|(_, value)| value.clone())
                        .collect();
                    let mut candidate = tableau.clone();
                    for ((r, c), value) in right_positions.iter().zip(right_block.iter()) {
                        candidate.rows[*r][*c] = value.clone();
                    }
                    for ((r, c), value) in left_positions.iter().zip(left_block.iter()) {
                        candidate.rows[*r][*c] = value.clone();
                    }
                    candidate.sort_within_columns();
                    let sign = shuffle_sign(&union, &concatenate(&right_block, &left_block));
                    candidate.multiplicity *= BigRational::from_integer(BigInt::from(-sign));
                    next.add_tableau(candidate);
                }
            }

            next.remove_nullifying_traces();
            self.storage = next.storage;
            if !changed {
                break;
            }
        }

        already_standard
    }

    pub fn total_dimension(&self, dim: usize) -> BigInt {
        self.storage.iter().fold(BigInt::zero(), |acc, tab| {
            acc + tab.shape().dimension(dim)
        })
    }
}

impl<T: Clone + Ord + Eq> Default for Tableaux<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn lr_tensor<T: Clone + Ord + Eq>(
    tab1: &FilledTableau<T>,
    tab2: &FilledTableau<T>,
    max_rows: usize,
    all_tabs: bool,
) -> Vec<FilledTableau<T>> {
    let lhs = lift_unlabelled(tab1);
    let rhs_labels = labelled_rhs(tab2);
    let mut current = vec![lhs];

    for label in rhs_labels {
        let mut next = Vec::new();
        for tableau in &current {
            for row in 0..=tableau.number_of_rows().min(max_rows.saturating_sub(1)) {
                let Some(candidate) =
                    lr_add_box(tableau, label.clone(), row, max_rows, all_tabs)
                else {
                    continue;
                };
                next.push(candidate);
            }
        }
        current = combine_tableaux(next);
    }

    strip_labels(current)
}

fn lr_add_box<T: Clone + Ord + Eq>(
    base: &FilledTableau<LRBox<T>>,
    value: LRBox<T>,
    row: usize,
    max_rows: usize,
    all_tabs: bool,
) -> Option<FilledTableau<LRBox<T>>> {
    let mut candidate = base.clone();
    if row >= max_rows {
        return None;
    }
    candidate.add_box(row, value.clone());
    if !is_valid_shape(&candidate.rows) {
        return None;
    }
    let col = candidate.row_size(row) - 1;

    if candidate
        .rows
        .iter()
        .enumerate()
        .filter(|(r, current)| *r != row && current.len() > col)
        .any(|(_, current)| {
            let current = &current[col];
            current.added && current.source_row == value.source_row
        })
    {
        return None;
    }

    if all_tabs
        && candidate.rows[row]
            .iter()
            .enumerate()
            .any(|(c, entry)| c != col && entry.added && entry.source_col == value.source_col)
    {
        return None;
    }

    if !lr_lattice_word_ok(&candidate) {
        return None;
    }

    Some(candidate)
}

fn lr_lattice_word_ok<T: Clone + Ord + Eq>(tableau: &FilledTableau<LRBox<T>>) -> bool {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for row in &tableau.rows {
        for entry in row.iter().rev() {
            if !entry.added {
                continue;
            }
            let source_row = entry.source_row;
            *counts.entry(source_row).or_default() += 1;
            if source_row > 0 {
                let current = counts.get(&source_row).copied().unwrap_or(0);
                let previous = counts.get(&(source_row - 1)).copied().unwrap_or(0);
                if current > previous {
                    return false;
                }
            }
        }
    }
    true
}

fn lift_unlabelled<T: Clone + Ord + Eq>(tab: &FilledTableau<T>) -> FilledTableau<LRBox<T>> {
    FilledTableau {
        rows: tab
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| LRBox {
                        value: value.clone(),
                        source_row: usize::MAX,
                        source_col: usize::MAX,
                        added: false,
                    })
                    .collect()
            })
            .collect(),
        multiplicity: tab.multiplicity.clone(),
        selfdual_column: tab.selfdual_column,
    }
}

fn labelled_rhs<T: Clone + Ord + Eq>(tab: &FilledTableau<T>) -> Vec<LRBox<T>> {
    let mut out = Vec::new();
    for (row, current) in tab.rows.iter().enumerate() {
        for (col, value) in current.iter().enumerate() {
            out.push(LRBox {
                value: value.clone(),
                source_row: row,
                source_col: col,
                added: true,
            });
        }
    }
    out
}

fn strip_labels<T: Clone + Ord + Eq>(
    tabs: Vec<FilledTableau<LRBox<T>>>,
) -> Vec<FilledTableau<T>> {
    let mut out = Tableaux::new();
    for tab in tabs {
        out.add_tableau(FilledTableau {
            rows: tab
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(|entry| entry.value).collect())
                .collect(),
            multiplicity: tab.multiplicity,
            selfdual_column: tab.selfdual_column,
        });
    }
    out.storage
}

fn combine_tableaux<T: Clone + Ord + Eq>(tabs: Vec<FilledTableau<T>>) -> Vec<FilledTableau<T>> {
    let mut out = Tableaux::new();
    for tab in tabs {
        out.add_tableau(tab);
    }
    out.storage
}

fn concatenate<T: Clone>(lhs: &[T], rhs: &[T]) -> Vec<T> {
    let mut out = lhs.to_vec();
    out.extend_from_slice(rhs);
    out
}

fn is_valid_shape<T>(rows: &[Vec<T>]) -> bool {
    let lengths: Vec<usize> = rows.iter().map(Vec::len).filter(|len| *len > 0).collect();
    lengths.windows(2).all(|window| window[0] >= window[1])
}

fn flatten_rows<T: Clone>(rows: &[Vec<T>]) -> Vec<T> {
    rows.iter().flat_map(|row| row.iter().cloned()).collect()
}

fn factorial_bigint(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, value| acc * BigInt::from(value))
}

fn permutation_sign_between<T: Clone + Ord + Eq>(original: &[T], permuted: &[T]) -> i32 {
    let original_order = decorate_with_occurrence(original);
    let permuted_order = decorate_with_occurrence(permuted);
    let mut position_of = BTreeMap::new();
    for (idx, item) in permuted_order.iter().enumerate() {
        position_of.insert(item.clone(), idx);
    }
    let perm: Vec<usize> = original_order
        .iter()
        .map(|item| position_of.get(item).copied().unwrap_or(0))
        .collect();
    inversion_parity(&perm)
}

fn permutations_of_indices(n: usize) -> Vec<(Vec<usize>, i32)> {
    let mut values: Vec<usize> = (0..n).collect();
    let mut raw = Vec::new();
    heap_permute(&mut values, 0, &mut raw);
    raw.into_iter()
        .map(|(perm, _)| {
            let sign = inversion_parity(&perm);
            (perm, sign)
        })
        .collect()
}

fn ordered_subsets(total: usize, choose: usize) -> Vec<Vec<usize>> {
    fn rec(
        total: usize,
        choose: usize,
        next_idx: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == choose {
            out.push(current.clone());
            return;
        }
        for idx in next_idx..total {
            current.push(idx);
            rec(total, choose, idx + 1, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(total, choose, 0, &mut Vec::new(), &mut out);
    out
}

fn heap_permute<T: Clone>(values: &mut [T], start: usize, out: &mut Vec<(Vec<T>, i32)>) {
    if start == values.len() {
        out.push((values.to_vec(), 1));
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
    if inversions % 2 == 0 { 1 } else { -1 }
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

fn shuffle_sign<T: Clone + Ord + Eq>(original: &[T], shuffled: &[T]) -> i32 {
    permutation_sign_between(original, shuffled)
}

pub fn row_symmetrizer_generators(tab: &YoungTableau, n: usize) -> Vec<ax_perm::Perm> {
    let mut gens = Vec::new();
    for row in &tab.rows {
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
    let n_cols = tab.rows.iter().map(Vec::len).max().unwrap_or(0);
    for col in 0..n_cols {
        let indices: Vec<usize> = tab
            .rows
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        for i in 0..indices.len().saturating_sub(1) {
            let mut p: ax_perm::Perm = (0..n).collect();
            p.swap(indices[i], indices[i + 1]);
            gens.push(p);
        }
    }
    gens
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn garnir_basic() {
        let tab = FilledTableau {
            rows: vec![vec![0usize, 1usize], vec![2usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        assert_eq!(tab.garnir_set(0, 1), vec![1, 0, 2]);
    }

    #[test]
    fn standard_form_converts_nonstandard() {
        let mut tabs = Tableaux::new();
        tabs.add_tableau(FilledTableau {
            rows: vec![vec![1usize, 0usize], vec![2usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        });
        tabs.standard_form();
        assert!(tabs.storage.iter().any(|tab| {
            tab.rows == vec![vec![0, 1], vec![2]] && tab.multiplicity == BigRational::one()
        }));
        assert!(tabs.storage.iter().any(|tab| {
            tab.rows == vec![vec![0, 2], vec![1]] && tab.multiplicity == -BigRational::one()
        }));
    }

    #[test]
    fn canonicalise_sorts_equal_length_columns_lexicographically() {
        let mut tab = FilledTableau {
            rows: vec![vec![2usize, 0usize], vec![3usize, 1usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        tab.canonicalise();
        assert_eq!(tab.rows, vec![vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn lr_tensor_two_boxes() {
        let tab1 = FilledTableau {
            rows: vec![vec![0usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let tab2 = FilledTableau {
            rows: vec![vec![1usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let results = lr_tensor(&tab1, &tab2, 10, false);
        let shapes: BTreeSet<Vec<usize>> = results
            .into_iter()
            .map(|tab| tab.rows.iter().map(Vec::len).collect())
            .collect();
        assert!(shapes.contains(&vec![2]));
        assert!(shapes.contains(&vec![1, 1]));
    }

    #[test]
    fn lr_tensor_column_times_box() {
        let tab1 = FilledTableau {
            rows: vec![vec![0usize], vec![1usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let tab2 = FilledTableau {
            rows: vec![vec![2usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let results = lr_tensor(&tab1, &tab2, 10, false);
        let shapes: BTreeSet<Vec<usize>> = results
            .into_iter()
            .map(|tab| tab.rows.iter().map(Vec::len).collect())
            .collect();
        assert!(shapes.contains(&vec![2, 1]));
        assert!(shapes.contains(&vec![1, 1, 1]));
    }

    #[test]
    fn projector_for_antisymmetric() {
        let tab = FilledTableau {
            rows: vec![vec![0usize], vec![1usize], vec![2usize]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let projector = tab.projector(false);
        assert_eq!(projector.size(), 6);
        let positive = projector
            .permutations
            .iter()
            .filter(|(_, coeff)| coeff == &BigRational::one())
            .count();
        let negative = projector
            .permutations
            .iter()
            .filter(|(_, coeff)| coeff == &-BigRational::one())
            .count();
        assert_eq!(positive, 3);
        assert_eq!(negative, 3);
    }

    #[test]
    fn hook_length_formula() {
        let d = YoungDiagram::new(vec![3, 2, 1]);
        assert_eq!(d.dimension(6), BigInt::from(896));
    }
}
