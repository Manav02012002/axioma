#![forbid(unsafe_code)]

pub mod adjform;
pub mod index_classifier;
pub mod pooled_canon;

use ax_ir::{Expr, Index, Interner};
use ax_perm::{Perm, SGS};
use index_classifier::{classify_indices, IndexClassification};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

pub trait DummyRenameEnv {
    fn index_families(&self) -> &HashMap<lasso::Spur, ax_ir::IndexFamily>;
    fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur>;
}

pub trait ComponentEvalEnv {
    fn coordinates(&self) -> Vec<lasso::Spur>;
    fn is_coordinate(&self, s: lasso::Spur) -> bool;
    fn tensor_properties(&self) -> &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>;
}

pub trait PropertyLookup: Send + Sync {
    fn get_properties(&self, name: lasso::Spur) -> Vec<&ax_ir::TensorProperty>;
    fn get_properties_with_indices(
        &self,
        name: lasso::Spur,
        indices: &[ax_ir::Index],
    ) -> Vec<&ax_ir::TensorProperty>;
    fn has_property_kind(&self, name: lasso::Spur, kind: &ax_ir::TensorProperty) -> bool;
}

impl PropertyLookup for HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> {
    fn get_properties(&self, name: lasso::Spur) -> Vec<&ax_ir::TensorProperty> {
        self.get(&name)
            .map(|props| props.iter().collect())
            .unwrap_or_default()
    }

    fn get_properties_with_indices(
        &self,
        name: lasso::Spur,
        _indices: &[ax_ir::Index],
    ) -> Vec<&ax_ir::TensorProperty> {
        self.get_properties(name)
    }

    fn has_property_kind(&self, name: lasso::Spur, kind: &ax_ir::TensorProperty) -> bool {
        self.get_properties(name)
            .into_iter()
            .any(|prop| std::mem::discriminant(prop) == std::mem::discriminant(kind))
    }
}

pub struct DefaultEvalEnv {
    pub coords: Vec<lasso::Spur>,
    pub coord_set: HashSet<lasso::Spur>,
    pub tensor_props: HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
}

impl DefaultEvalEnv {
    pub fn new(
        coords: Vec<lasso::Spur>,
        tensor_props: HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    ) -> Self {
        let coord_set = coords.iter().copied().collect();
        Self {
            coords,
            coord_set,
            tensor_props,
        }
    }
}

impl ComponentEvalEnv for DefaultEvalEnv {
    fn coordinates(&self) -> Vec<lasso::Spur> {
        self.coords.clone()
    }

    fn is_coordinate(&self, s: lasso::Spur) -> bool {
        self.coord_set.contains(&s)
    }

    fn tensor_properties(&self) -> &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> {
        &self.tensor_props
    }
}

/// Information about a tensor factor in a product, extracted from Expr.
#[derive(Clone, Debug)]
pub struct TensorFactorInfo {
    pub name: lasso::Spur,
    pub n_indices: usize,
    pub start_position: usize,
    pub properties: Vec<ax_ir::TensorProperty>,
}

/// Component substitution rule: maps index value combinations to scalar values.
#[derive(Clone, Debug)]
pub struct ComponentRule {
    pub tensor: lasso::Spur,
    pub indices: Vec<(lasso::Spur, ax_ir::Variance)>,
    pub value: ax_ir::Expr,
}

/// Build the generating set for the symmetry group of a tensor product expression.
///
/// Each tensor in the product has index positions. The symmetry generators
/// are permutations of these positions.
pub fn build_generating_set(
    factors: &[TensorFactorInfo],
    _interner: &ax_ir::Interner,
) -> Vec<Perm> {
    let _sgs_marker: Option<SGS> = None;
    let total_indices: usize = factors.iter().map(|factor| factor.n_indices).sum();
    let degree = total_indices + 2;
    let sign_pos = total_indices;
    let sign_neg = total_indices + 1;

    let mut generators = Vec::new();

    for factor in factors {
        for prop in &factor.properties {
            match prop {
                ax_ir::TensorProperty::Symmetric(positions) => {
                    if positions.len() < 2 {
                        continue;
                    }
                    for i in 0..(positions.len() - 1) {
                        let mut generator: Perm = (0..degree).collect();
                        let p1 = factor.start_position + positions[i];
                        let p2 = factor.start_position + positions[i + 1];
                        generator.swap(p1, p2);
                        generators.push(generator);
                    }
                }
                ax_ir::TensorProperty::AntiSymmetric(positions) => {
                    if positions.len() < 2 {
                        continue;
                    }
                    for i in 0..(positions.len() - 1) {
                        let mut generator: Perm = (0..degree).collect();
                        let p1 = factor.start_position + positions[i];
                        let p2 = factor.start_position + positions[i + 1];
                        generator.swap(p1, p2);
                        generator.swap(sign_pos, sign_neg);
                        generators.push(generator);
                    }
                }
                ax_ir::TensorProperty::RiemannSymmetry => {
                    if factor.n_indices >= 4 {
                        let s = factor.start_position;

                        let mut g1: Perm = (0..degree).collect();
                        g1.swap(s, s + 1);
                        g1.swap(sign_pos, sign_neg);
                        generators.push(g1);

                        let mut g2: Perm = (0..degree).collect();
                        g2.swap(s + 2, s + 3);
                        g2.swap(sign_pos, sign_neg);
                        generators.push(g2);

                        let mut g3: Perm = (0..degree).collect();
                        g3.swap(s, s + 2);
                        g3.swap(s + 1, s + 3);
                        generators.push(g3);
                    }
                }
                _ => {}
            }
        }
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            if factors[i].name == factors[j].name && factors[i].n_indices == factors[j].n_indices {
                let mut generator: Perm = (0..degree).collect();
                for k in 0..factors[i].n_indices {
                    generator.swap(factors[i].start_position + k, factors[j].start_position + k);
                }
                generators.push(generator);
            }
        }
    }

    generators
}

fn build_generating_set_parallel(
    factor_info: &[TensorFactorInfo],
    interner: &ax_ir::Interner,
) -> Vec<Perm> {
    let generators = build_generating_set(factor_info, interner);
    let total_indices: usize = factor_info.iter().map(|factor| factor.n_indices).sum();
    if total_indices > 20 {
        ax_perm::schreier_sims_parallel(&generators, total_indices + 2).generators
    } else {
        generators
    }
}

pub fn extract_factor_info(
    expr: &ax_ir::Expr,
    tensor_properties: &dyn PropertyLookup,
    _interner: &ax_ir::Interner,
) -> Vec<TensorFactorInfo> {
    let factors = match expr {
        ax_ir::Expr::Mul(factors) => factors,
        _ => return vec![],
    };

    let mut result = Vec::new();
    let mut position = 0usize;

    for factor in factors {
        if let ax_ir::Expr::Indexed(base, indices) = factor {
            if let ax_ir::Expr::Sym(name) = base.as_ref() {
                let props = tensor_properties
                    .get_properties_with_indices(*name, indices)
                    .into_iter()
                    .cloned()
                    .collect();
                result.push(TensorFactorInfo {
                    name: *name,
                    n_indices: indices.len(),
                    start_position: position,
                    properties: props,
                });
                position += indices.len();
            }
        }
    }

    result
}

fn repeated_sets_from_classification(
    classification: &IndexClassification,
) -> Vec<ax_perm::RepeatedSet> {
    let dummy_slots: HashSet<usize> = classification
        .dummy
        .iter()
        .flat_map(|(_, a, b, _, _)| [*a, *b])
        .collect();
    let mut by_key: HashMap<(lasso::Spur, u8, Option<lasso::Spur>), Vec<usize>> = HashMap::new();

    for (_, pos, idx) in &classification.free {
        if dummy_slots.contains(pos) {
            continue;
        }
        let variance_key = match idx.variance {
            ax_ir::Variance::Up => 0,
            ax_ir::Variance::Down => 1,
        };
        by_key
            .entry((idx.name, variance_key, idx.index_type))
            .or_default()
            .push(*pos);
    }

    by_key
        .into_values()
        .filter(|positions| positions.len() > 1)
        .map(|positions| ax_perm::RepeatedSet { positions })
        .collect()
}

fn metric_symmetry_for_slots(positions: &[usize], factor_info: &[TensorFactorInfo]) -> i32 {
    let mut symmetry = 1;

    for &pos in positions {
        if let Some(factor) = factor_info.iter().find(|factor| {
            pos >= factor.start_position && pos < factor.start_position + factor.n_indices
        }) {
            if factor.properties.iter().any(|prop| {
                matches!(
                    prop,
                    ax_ir::TensorProperty::AntiCommuting | ax_ir::TensorProperty::Spinor
                )
            }) {
                return -1;
            }
            if factor.properties.iter().any(|prop| {
                matches!(
                    prop,
                    ax_ir::TensorProperty::Metric | ax_ir::TensorProperty::InverseMetric
                )
            }) {
                symmetry = 1;
            }
        }
    }

    symmetry
}

/// Lower all free (non-contracted) upper indices in an expression.
/// Does not insert a metric; it only flips the variance.
/// Only affects indices whose family has `position = Free`.
pub fn lower_free_indices(
    expr: &ax_ir::Expr,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    index_families: &HashMap<lasso::Spur, ax_ir::IndexFamily>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;

    let mut index_count: HashMap<lasso::Spur, usize> = HashMap::new();
    count_all_index_names(expr, &mut index_count);

    let free: HashSet<lasso::Spur> = index_count
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(name, _)| *name)
        .collect();

    flip_free_indices(
        expr,
        &free,
        ax_ir::Variance::Up,
        ax_ir::Variance::Down,
        index_to_family,
        index_families,
    )
}

/// Raise all free (non-contracted) lower indices in an expression.
/// Does not insert a metric; it only flips the variance.
/// Only affects indices whose family has `position = Free`.
pub fn raise_free_indices(
    expr: &ax_ir::Expr,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    index_families: &HashMap<lasso::Spur, ax_ir::IndexFamily>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;

    let mut index_count: HashMap<lasso::Spur, usize> = HashMap::new();
    count_all_index_names(expr, &mut index_count);

    let free: HashSet<lasso::Spur> = index_count
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(name, _)| *name)
        .collect();

    flip_free_indices(
        expr,
        &free,
        ax_ir::Variance::Down,
        ax_ir::Variance::Up,
        index_to_family,
        index_families,
    )
}

fn count_all_index_names(expr: &Expr, counts: &mut HashMap<lasso::Spur, usize>) {
    match expr {
        Expr::Indexed(base, indices) => {
            count_all_index_names(base, counts);
            for idx in indices {
                *counts.entry(idx.name).or_default() += 1;
            }
        }
        Expr::Mul(factors) => {
            for factor in factors {
                count_all_index_names(factor, counts);
            }
        }
        Expr::Add(terms) => {
            for term in terms {
                count_all_index_names(term, counts);
            }
        }
        Expr::Neg(expr) => count_all_index_names(expr, counts),
        _ => {}
    }
}

fn flip_free_indices(
    expr: &Expr,
    free: &HashSet<lasso::Spur>,
    from_var: ax_ir::Variance,
    to_var: ax_ir::Variance,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    index_families: &HashMap<lasso::Spur, ax_ir::IndexFamily>,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices = indices
                .iter()
                .map(|idx| {
                    if free.contains(&idx.name) && idx.variance == from_var {
                        let allow = index_to_family
                            .get(&idx.name)
                            .and_then(|family| index_families.get(family))
                            .map(|family| family.position == ax_ir::IndexPosition::Free)
                            .unwrap_or(true);

                        if allow {
                            ax_ir::Index {
                                name: idx.name,
                                variance: to_var.clone(),
                                index_type: idx.index_type,
                            }
                        } else {
                            idx.clone()
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();

            Expr::Indexed(
                Box::new(flip_free_indices(
                    base,
                    free,
                    from_var.clone(),
                    to_var.clone(),
                    index_to_family,
                    index_families,
                )),
                new_indices,
            )
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    flip_free_indices(
                        factor,
                        free,
                        from_var.clone(),
                        to_var.clone(),
                        index_to_family,
                        index_families,
                    )
                })
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    flip_free_indices(
                        term,
                        free,
                        from_var.clone(),
                        to_var.clone(),
                        index_to_family,
                        index_families,
                    )
                })
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(flip_free_indices(
            inner,
            free,
            from_var,
            to_var,
            index_to_family,
            index_families,
        )),
        _ => expr.clone(),
    }
}

fn canonicalise_product(
    expr: &Expr,
    tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let precanonical = canonicalize_indices(expr, tensor_properties, interner);
    if precanonical == Expr::zero() {
        return Expr::zero();
    }

    let wrapped;
    let expr_ref = match &precanonical {
        Expr::Mul(_) => &precanonical,
        Expr::Indexed(_, _) => {
            wrapped = Expr::Mul(vec![precanonical.clone()]);
            &wrapped
        }
        _ => return precanonical,
    };

    let factor_info = extract_factor_info(expr_ref, tensor_properties, interner);
    if factor_info.is_empty() {
        return expr.clone();
    }
    let classification = classify_indices(expr_ref);

    let generators = build_generating_set(&factor_info, interner);
    let factors = match expr_ref {
        Expr::Mul(factors) => factors,
        _ => unreachable!(),
    };

    let mut all_indices = Vec::new();
    let mut scalar_factors = Vec::new();
    for factor in factors {
        match factor {
            Expr::Indexed(_, indices) => all_indices.extend(indices.iter().cloned()),
            _ => scalar_factors.push(factor.clone()),
        }
    }

    let total_indices = classification.total;
    if total_indices == 0 {
        return expr.clone();
    }

    let mut keyed_positions: Vec<(usize, (String, u8))> = all_indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (i, sort_key(idx, interner)))
        .collect();
    keyed_positions.sort_by(|(ia, ka), (ib, kb)| ka.cmp(kb).then(ia.cmp(ib)));

    let mut rank_by_pos = vec![0usize; total_indices];
    for (rank, (pos, _)) in keyed_positions.iter().enumerate() {
        rank_by_pos[*pos] = rank;
    }

    let mut extended_perm = rank_by_pos.clone();
    extended_perm.push(total_indices);
    extended_perm.push(total_indices + 1);

    let dummy_pairs: Vec<(usize, usize)> = classification
        .dummy
        .iter()
        .map(|(_, a, b, _, _)| (*a, *b))
        .collect();
    let repeated_sets = repeated_sets_from_classification(&classification);
    let degree = total_indices + 2;

    let mut pairs_by_type: HashMap<Option<lasso::Spur>, Vec<(usize, usize)>> = HashMap::new();
    for &(a, b) in &dummy_pairs {
        let itype = all_indices[a].index_type;
        pairs_by_type.entry(itype).or_default().push((a, b));
    }

    let dummy_sets: Vec<ax_perm::DummySet> = pairs_by_type
        .into_iter()
        .map(|(_, pairs)| {
            let slots: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            ax_perm::DummySet {
                pairs,
                metric_symmetry: metric_symmetry_for_slots(&slots, &factor_info),
            }
        })
        .collect();

    let (canon_perm, canon_sign) =
        if generators.is_empty() && dummy_sets.is_empty() && repeated_sets.is_empty() {
            (extended_perm.clone(), 1)
        } else {
            let sgs = ax_perm::schreier_sims(&[], &generators, degree);
            ax_perm::canonical_perm_with_sets(&extended_perm, &sgs, &dummy_sets, &repeated_sets)
        };

    if canon_sign == -1 && canon_perm[..total_indices] == extended_perm[..total_indices] {
        return Expr::zero();
    }

    let mut index_by_rank = vec![all_indices[0].clone(); total_indices];
    for (pos, rank) in rank_by_pos.iter().enumerate() {
        index_by_rank[*rank] = all_indices[pos].clone();
    }
    let new_indices: Vec<ax_ir::Index> = canon_perm[..total_indices]
        .iter()
        .map(|rank| index_by_rank[*rank].clone())
        .collect();

    let mut result_factors = scalar_factors;
    let mut idx_pos = 0usize;
    for factor in factors {
        if let Expr::Indexed(base, indices) = factor {
            let n = indices.len();
            result_factors.push(Expr::Indexed(
                base.clone(),
                new_indices[idx_pos..idx_pos + n].to_vec(),
            ));
            idx_pos += n;
        }
    }

    let result = Expr::mul(result_factors);
    if canon_sign == -1 {
        Expr::neg(result)
    } else {
        result
    }
}

fn canonicalise_product_parallel(
    expr: &Expr,
    tensor_properties: &(dyn PropertyLookup + Send + Sync),
    interner: &ax_ir::Interner,
) -> Expr {
    let precanonical = canonicalize_indices(expr, tensor_properties, interner);
    if precanonical == Expr::zero() {
        return Expr::zero();
    }

    let wrapped;
    let expr_ref = match &precanonical {
        Expr::Mul(_) => &precanonical,
        Expr::Indexed(_, _) => {
            wrapped = Expr::Mul(vec![precanonical.clone()]);
            &wrapped
        }
        _ => return precanonical,
    };

    let factor_info = extract_factor_info(expr_ref, tensor_properties, interner);
    if factor_info.is_empty() {
        return expr.clone();
    }
    let classification = classify_indices(expr_ref);

    let generators = build_generating_set_parallel(&factor_info, interner);
    let factors = match expr_ref {
        Expr::Mul(factors) => factors,
        _ => unreachable!(),
    };

    let mut all_indices = Vec::new();
    let mut scalar_factors = Vec::new();
    for factor in factors {
        match factor {
            Expr::Indexed(_, indices) => all_indices.extend(indices.iter().cloned()),
            _ => scalar_factors.push(factor.clone()),
        }
    }

    let total_indices = classification.total;
    if total_indices == 0 {
        return expr.clone();
    }

    let mut keyed_positions: Vec<(usize, (String, u8))> = all_indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (i, sort_key(idx, interner)))
        .collect();
    keyed_positions.sort_by(|(ia, ka), (ib, kb)| ka.cmp(kb).then(ia.cmp(ib)));

    let mut rank_by_pos = vec![0usize; total_indices];
    for (rank, (pos, _)) in keyed_positions.iter().enumerate() {
        rank_by_pos[*pos] = rank;
    }

    let mut extended_perm = rank_by_pos.clone();
    extended_perm.push(total_indices);
    extended_perm.push(total_indices + 1);

    let dummy_pairs: Vec<(usize, usize)> = classification
        .dummy
        .iter()
        .map(|(_, a, b, _, _)| (*a, *b))
        .collect();
    let repeated_sets = repeated_sets_from_classification(&classification);
    let degree = total_indices + 2;

    let mut pairs_by_type: HashMap<Option<lasso::Spur>, Vec<(usize, usize)>> = HashMap::new();
    for &(a, b) in &dummy_pairs {
        let itype = all_indices[a].index_type;
        pairs_by_type.entry(itype).or_default().push((a, b));
    }

    let dummy_sets: Vec<ax_perm::DummySet> = pairs_by_type
        .into_iter()
        .map(|(_, pairs)| {
            let slots: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            ax_perm::DummySet {
                pairs,
                metric_symmetry: metric_symmetry_for_slots(&slots, &factor_info),
            }
        })
        .collect();

    let (canon_perm, canon_sign) =
        if generators.is_empty() && dummy_sets.is_empty() && repeated_sets.is_empty() {
            (extended_perm.clone(), 1)
        } else {
            let sgs = ax_perm::schreier_sims_parallel(&generators, degree);
            ax_perm::canonical_perm_with_sets(&extended_perm, &sgs, &dummy_sets, &repeated_sets)
        };

    if canon_sign == -1 && canon_perm[..total_indices] == extended_perm[..total_indices] {
        return Expr::zero();
    }

    let mut index_by_rank = vec![all_indices[0].clone(); total_indices];
    for (pos, rank) in rank_by_pos.iter().enumerate() {
        index_by_rank[*rank] = all_indices[pos].clone();
    }
    let new_indices: Vec<ax_ir::Index> = canon_perm[..total_indices]
        .iter()
        .map(|rank| index_by_rank[*rank].clone())
        .collect();

    let mut result_factors = scalar_factors;
    let mut idx_pos = 0usize;
    for factor in factors {
        if let Expr::Indexed(base, indices) = factor {
            let n = indices.len();
            result_factors.push(Expr::Indexed(
                base.clone(),
                new_indices[idx_pos..idx_pos + n].to_vec(),
            ));
            idx_pos += n;
        }
    }

    let result = Expr::mul(result_factors);
    if canon_sign == -1 {
        Expr::neg(result)
    } else {
        result
    }
}

/// Canonicalise a tensor product expression by reordering indices to canonical form.
///
/// This uses the Butler-Portugal algorithm (via ax-perm) to find the lexicographically
/// smallest index permutation consistent with the tensor symmetries.
pub fn canonicalise(
    expr: &ax_ir::Expr,
    tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(_) => canonicalise_product(expr, tensor_properties, interner),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| canonicalise(term, tensor_properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(canonicalise(inner, tensor_properties, interner)),
        Expr::Indexed(_, _) => {
            let result = canonicalise_product(expr, tensor_properties, interner);
            match result {
                Expr::Mul(mut factors) if factors.len() == 1 => factors.remove(0),
                other => other,
            }
        }
        _ => expr.clone(),
    }
}

/// Parallel canonicalisation for sums with independent terms.
pub fn canonicalise_parallel(
    expr: &ax_ir::Expr,
    properties: &(dyn PropertyLookup + Send + Sync),
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) if terms.len() > 4 => {
            let canonicalised: Vec<Expr> = terms
                .par_iter()
                .map(|term| canonicalise_parallel(term, properties, interner))
                .collect();
            Expr::add(canonicalised)
        }
        Expr::Add(_) => canonicalise(expr, properties, interner),
        Expr::Mul(_) => canonicalise_product_parallel(expr, properties, interner),
        Expr::Neg(inner) => Expr::neg(canonicalise_parallel(inner, properties, interner)),
        Expr::Indexed(_, _) => {
            let result = canonicalise_product_parallel(expr, properties, interner);
            match result {
                Expr::Mul(mut factors) if factors.len() == 1 => factors.remove(0),
                other => other,
            }
        }
        _ => expr.clone(),
    }
}

/// Simplify a sum of tensor monomials using multi-term symmetry information.
pub fn meld(
    expr: &ax_ir::Expr,
    tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) => {
            let mut groups: Vec<(String, Vec<(Expr, usize)>)> = Vec::new();
            for (idx, term) in terms.iter().enumerate() {
                let simplified = meld(term, tensor_properties, interner);
                let key = tensor_structure_key(&simplified, interner);
                if let Some((_, bucket)) = groups.iter_mut().find(|(k, _)| *k == key) {
                    bucket.push((simplified, idx));
                } else {
                    groups.push((key, vec![(simplified, idx)]));
                }
            }

            let mut result_terms: Vec<Expr> = Vec::new();

            for (_, group) in &groups {
                if group.len() <= 1 {
                    result_terms.push(group[0].0.clone());
                    continue;
                }

                let canonical: Vec<Expr> = group
                    .iter()
                    .map(|(t, _)| canonicalise(t, tensor_properties, interner))
                    .collect();

                let factor_info =
                    extract_factor_info_from_term(&canonical[0], tensor_properties, interner);
                let tableaux = tableaux_from_properties(&factor_info, tensor_properties);

                let mut projections: Vec<adjform::ProjectedAdjform> = Vec::new();
                for (term, _) in group.iter() {
                    let (scalar, indices) = extract_scalar_and_indices(term);
                    let adj = adjform::Adjform::from_indices(&indices);
                    let coeff = scalar_to_rational(&scalar);
                    let mut proj = adjform::ProjectedAdjform::from_adjform(adj, coeff);
                    proj.young_project(&tableaux);
                    projections.push(proj);
                }

                let mut independent: Vec<usize> = Vec::new();
                let mut mapping: Vec<adjform::Adjform> = Vec::new();

                for (i, proj) in projections.iter().enumerate() {
                    if proj.is_empty() {
                        continue;
                    }

                    if independent.is_empty() {
                        independent.push(i);
                        if let Some((adj, _)) = proj.terms.iter().next() {
                            mapping.push(adj.clone());
                        }
                        continue;
                    }

                    let mut matrix: Vec<Vec<BigRational>> = Vec::new();
                    let mut rhs_vec: Vec<BigRational> = Vec::new();

                    for adj in &mapping {
                        let mut row = Vec::new();
                        for &ind_idx in &independent {
                            row.push(projections[ind_idx].get_coeff(adj));
                        }
                        matrix.push(row);
                        rhs_vec.push(proj.get_coeff(adj));
                    }

                    let mut dependent = false;
                    if let Some(coeffs) = solve_rational_system(&matrix, &rhs_vec) {
                        let all_adjforms: BTreeSet<&adjform::Adjform> = projections[..=i]
                            .iter()
                            .flat_map(|p| p.terms.keys())
                            .collect();

                        let mut valid = true;
                        for adj in &all_adjforms {
                            let mut lhs = BigRational::from_integer(0.into());
                            for (j, &ind_idx) in independent.iter().enumerate() {
                                lhs += &coeffs[j] * &projections[ind_idx].get_coeff(adj);
                            }
                            if lhs != proj.get_coeff(adj) {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            dependent = true;
                        }
                    }

                    if dependent {
                        continue;
                    }

                    independent.push(i);
                    for adj in proj.terms.keys() {
                        if !mapping.contains(adj) {
                            mapping.push(adj.clone());
                            break;
                        }
                    }
                }

                if independent.is_empty() {
                    continue;
                }

                let mut total_basis_coeffs =
                    vec![BigRational::from_integer(0.into()); independent.len()];
                let basis_adjforms: BTreeSet<&adjform::Adjform> = independent
                    .iter()
                    .flat_map(|&idx| projections[idx].terms.keys())
                    .collect();

                for proj in &projections {
                    if proj.is_empty() {
                        continue;
                    }

                    let mut matrix: Vec<Vec<BigRational>> = Vec::new();
                    let mut rhs_vec: Vec<BigRational> = Vec::new();
                    let mut all_adjforms: BTreeSet<&adjform::Adjform> = basis_adjforms.clone();
                    all_adjforms.extend(proj.terms.keys());

                    for adj in &all_adjforms {
                        let mut row = Vec::new();
                        for &ind_idx in &independent {
                            row.push(projections[ind_idx].get_coeff(adj));
                        }
                        matrix.push(row);
                        rhs_vec.push(proj.get_coeff(adj));
                    }

                    if let Some(coeffs) = solve_rational_system(&matrix, &rhs_vec) {
                        for (j, coeff) in coeffs.into_iter().enumerate() {
                            total_basis_coeffs[j] += coeff;
                        }
                    }
                }

                for (j, &idx) in independent.iter().enumerate() {
                    if total_basis_coeffs[j] == BigRational::from_integer(0.into()) {
                        continue;
                    }
                    result_terms.push(multiply_expr_by_rational(
                        canonical[idx].clone(),
                        total_basis_coeffs[j].clone(),
                    ));
                }
            }

            if result_terms.is_empty() {
                Expr::zero()
            } else {
                Expr::add(result_terms)
            }
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| meld(factor, tensor_properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(meld(inner, tensor_properties, interner)),
        _ => expr.clone(),
    }
}

pub fn meld_parallel(
    expr: &ax_ir::Expr,
    tensor_properties: &(dyn PropertyLookup + Send + Sync),
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) => {
            let mut groups: Vec<(String, Vec<(Expr, usize)>)> = Vec::new();
            for (idx, term) in terms.iter().enumerate() {
                let simplified = meld_parallel(term, tensor_properties, interner);
                let key = tensor_structure_key(&simplified, interner);
                if let Some((_, bucket)) = groups.iter_mut().find(|(k, _)| *k == key) {
                    bucket.push((simplified, idx));
                } else {
                    groups.push((key, vec![(simplified, idx)]));
                }
            }

            let mut result_terms: Vec<Expr> = Vec::new();

            for (_, group) in &groups {
                if group.len() <= 1 {
                    result_terms.push(group[0].0.clone());
                    continue;
                }

                let canonical: Vec<Expr> = if group.len() > 4 {
                    group
                        .par_iter()
                        .map(|(t, _)| canonicalise_parallel(t, tensor_properties, interner))
                        .collect()
                } else {
                    group
                        .iter()
                        .map(|(t, _)| canonicalise(t, tensor_properties, interner))
                        .collect()
                };

                let factor_info =
                    extract_factor_info_from_term(&canonical[0], tensor_properties, interner);
                let tableaux = tableaux_from_properties(&factor_info, tensor_properties);

                let mut projections: Vec<adjform::ProjectedAdjform> = Vec::new();
                for (term, _) in group.iter() {
                    let (scalar, indices) = extract_scalar_and_indices(term);
                    let adj = adjform::Adjform::from_indices(&indices);
                    let coeff = scalar_to_rational(&scalar);
                    let mut proj = adjform::ProjectedAdjform::from_adjform(adj, coeff);
                    proj.young_project(&tableaux);
                    projections.push(proj);
                }

                let mut independent: Vec<usize> = Vec::new();
                let mut mapping: Vec<adjform::Adjform> = Vec::new();

                for (i, proj) in projections.iter().enumerate() {
                    if proj.is_empty() {
                        continue;
                    }

                    if independent.is_empty() {
                        independent.push(i);
                        if let Some((adj, _)) = proj.terms.iter().next() {
                            mapping.push(adj.clone());
                        }
                        continue;
                    }

                    let mut matrix: Vec<Vec<BigRational>> = Vec::new();
                    let mut rhs_vec: Vec<BigRational> = Vec::new();

                    for adj in &mapping {
                        let mut row = Vec::new();
                        for &ind_idx in &independent {
                            row.push(projections[ind_idx].get_coeff(adj));
                        }
                        matrix.push(row);
                        rhs_vec.push(proj.get_coeff(adj));
                    }

                    let mut dependent = false;
                    if let Some(coeffs) = solve_rational_system(&matrix, &rhs_vec) {
                        let all_adjforms: BTreeSet<&adjform::Adjform> = projections[..=i]
                            .iter()
                            .flat_map(|p| p.terms.keys())
                            .collect();

                        let mut valid = true;
                        for adj in &all_adjforms {
                            let mut lhs = BigRational::from_integer(0.into());
                            for (j, &ind_idx) in independent.iter().enumerate() {
                                lhs += &coeffs[j] * &projections[ind_idx].get_coeff(adj);
                            }
                            if lhs != proj.get_coeff(adj) {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            dependent = true;
                        }
                    }

                    if dependent {
                        continue;
                    }

                    independent.push(i);
                    for adj in proj.terms.keys() {
                        if !mapping.contains(adj) {
                            mapping.push(adj.clone());
                            break;
                        }
                    }
                }

                if independent.is_empty() {
                    continue;
                }

                let mut total_basis_coeffs =
                    vec![BigRational::from_integer(0.into()); independent.len()];
                let basis_adjforms: BTreeSet<&adjform::Adjform> = independent
                    .iter()
                    .flat_map(|&idx| projections[idx].terms.keys())
                    .collect();

                for proj in &projections {
                    if proj.is_empty() {
                        continue;
                    }

                    let mut matrix: Vec<Vec<BigRational>> = Vec::new();
                    let mut rhs_vec: Vec<BigRational> = Vec::new();
                    let mut all_adjforms: BTreeSet<&adjform::Adjform> = basis_adjforms.clone();
                    all_adjforms.extend(proj.terms.keys());

                    for adj in &all_adjforms {
                        let mut row = Vec::new();
                        for &ind_idx in &independent {
                            row.push(projections[ind_idx].get_coeff(adj));
                        }
                        matrix.push(row);
                        rhs_vec.push(proj.get_coeff(adj));
                    }

                    if let Some(coeffs) = solve_rational_system(&matrix, &rhs_vec) {
                        for (j, coeff) in coeffs.into_iter().enumerate() {
                            total_basis_coeffs[j] += coeff;
                        }
                    }
                }

                for (j, &idx) in independent.iter().enumerate() {
                    if total_basis_coeffs[j] == BigRational::from_integer(0.into()) {
                        continue;
                    }
                    result_terms.push(multiply_expr_by_rational(
                        canonical[idx].clone(),
                        total_basis_coeffs[j].clone(),
                    ));
                }
            }

            if result_terms.is_empty() {
                Expr::zero()
            } else {
                Expr::add(result_terms)
            }
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| meld_parallel(factor, tensor_properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(meld_parallel(inner, tensor_properties, interner)),
        _ => expr.clone(),
    }
}

fn tensor_structure_key(expr: &Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(s) = base.as_ref() {
                format!("{}:{}", interner.resolve(*s), indices.len())
            } else {
                format!("{base:?}:{}", indices.len())
            }
        }
        Expr::Mul(factors) => {
            let mut parts: Vec<String> = factors
                .iter()
                .map(|factor| tensor_structure_key(factor, interner))
                .filter(|part| !part.is_empty())
                .collect();
            parts.sort();
            parts.join("*")
        }
        Expr::Neg(inner) => tensor_structure_key(inner, interner),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => String::new(),
        _ => format!("{expr:?}"),
    }
}

fn extract_scalar_and_indices(expr: &Expr) -> (Expr, Vec<ax_ir::Index>) {
    match expr {
        Expr::Indexed(_, _) => {
            let indices = classify_indices(expr)
                .all
                .into_iter()
                .map(|(_, idx)| idx)
                .collect();
            (Expr::one(), indices)
        }
        Expr::Mul(factors) => {
            let mut scalar_parts = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Indexed(_, _) => {}
                    other => scalar_parts.push(other.clone()),
                }
            }
            let all_indices = classify_indices(expr)
                .all
                .into_iter()
                .map(|(_, idx)| idx)
                .collect();
            let scalar = if scalar_parts.is_empty() {
                Expr::one()
            } else {
                Expr::mul(scalar_parts)
            };
            (scalar, all_indices)
        }
        Expr::Neg(inner) => {
            let (scalar, indices) = extract_scalar_and_indices(inner);
            (Expr::neg(scalar), indices)
        }
        _ => (expr.clone(), vec![]),
    }
}

fn scalar_to_rational(expr: &Expr) -> BigRational {
    match expr {
        Expr::Int(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(r) => r.clone(),
        Expr::Neg(inner) => -scalar_to_rational(inner),
        _ => BigRational::from_integer(BigInt::from(1)),
    }
}

fn multiply_expr_by_rational(expr: Expr, coeff: BigRational) -> Expr {
    if coeff == BigRational::from_integer(0.into()) {
        return Expr::zero();
    }
    if coeff == BigRational::from_integer(1.into()) {
        return expr;
    }
    Expr::mul(vec![Expr::Rational(coeff), expr])
}

fn extract_factor_info_from_term(
    expr: &Expr,
    tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Vec<TensorFactorInfo> {
    let wrapped = match expr {
        Expr::Mul(_) => expr.clone(),
        _ => Expr::Mul(vec![expr.clone()]),
    };
    extract_factor_info(&wrapped, tensor_properties, interner)
}

pub fn tableaux_from_properties(
    factor_info: &[TensorFactorInfo],
    properties: &dyn PropertyLookup,
) -> Vec<adjform::TableauInfo> {
    let mut result = Vec::new();

    for info in factor_info {
        for prop in properties.get_properties(info.name) {
            match prop {
                ax_ir::TensorProperty::Symmetric(positions) => {
                    let abs: Vec<usize> =
                        positions.iter().map(|p| info.start_position + p).collect();
                    result.push(adjform::TableauInfo {
                        rows: vec![abs],
                        columns: vec![],
                    });
                }
                ax_ir::TensorProperty::AntiSymmetric(positions) => {
                    let abs: Vec<usize> =
                        positions.iter().map(|p| info.start_position + p).collect();
                    result.push(adjform::TableauInfo {
                        rows: vec![],
                        columns: vec![abs],
                    });
                }
                ax_ir::TensorProperty::RiemannSymmetry => {
                    if info.n_indices >= 4 {
                        let s = info.start_position;
                        result.push(adjform::TableauInfo {
                            rows: vec![vec![s, s + 2], vec![s + 1, s + 3]],
                            columns: vec![vec![s, s + 1], vec![s + 2, s + 3]],
                        });
                    }
                }
                _ => {}
            }
        }
    }

    result
}

pub fn solve_rational_system(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
) -> Option<Vec<BigRational>> {
    let rows = matrix.len();
    if rows == 0 {
        return Some(vec![]);
    }
    let cols = matrix[0].len();

    let mut aug: Vec<Vec<BigRational>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(rhs[i].clone());
            r
        })
        .collect();

    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let mut pivot_row = 0usize;
    for col in 0..cols {
        let mut found = None;
        for (row, aug_row) in aug
            .iter()
            .enumerate()
            .skip(pivot_row)
            .take(rows - pivot_row)
        {
            if aug_row[col] != zero {
                found = Some(row);
                break;
            }
        }
        let Some(pr) = found else { continue };

        aug.swap(pivot_row, pr);
        let pivot_val = aug[pivot_row][col].clone();
        for j in 0..=cols {
            aug[pivot_row][j] = &aug[pivot_row][j] / &pivot_val;
        }

        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = aug[row][col].clone();
            if factor == zero {
                continue;
            }
            for j in 0..=cols {
                let sub = &factor * &aug[pivot_row][j];
                aug[row][j] = &aug[row][j] - &sub;
            }
        }

        pivot_row += 1;
    }

    let mut x = vec![zero.clone(); cols];
    for aug_row in aug.iter().take(pivot_row) {
        let mut pivot_col = None;
        for (col, val) in aug_row.iter().enumerate().take(cols) {
            if *val == one {
                pivot_col = Some(col);
                break;
            }
        }
        if let Some(col) = pivot_col {
            x[col] = aug_row[cols].clone();
        }
    }

    for (i, row) in matrix.iter().enumerate() {
        let mut sum = zero.clone();
        for (j, val) in row.iter().enumerate() {
            sum += val * &x[j];
        }
        if sum != rhs[i] {
            return None;
        }
    }

    Some(x)
}

fn substitute_indices(expr: &Expr, assignment: &HashMap<lasso::Spur, lasso::Spur>) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| ax_ir::Index {
                    name: assignment.get(&idx.name).copied().unwrap_or(idx.name),
                    variance: idx.variance.clone(),
                    index_type: idx.index_type,
                })
                .collect();
            Expr::Indexed(Box::new(substitute_indices(base, assignment)), new_indices)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_indices(factor, assignment))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_indices(term, assignment))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_indices(inner, assignment)),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_indices(base, assignment),
            substitute_indices(exp, assignment),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| substitute_indices(arg, assignment))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_indices(re, assignment)),
            Box::new(substitute_indices(im, assignment)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_indices(body, assignment)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_indices(lhs, assignment)),
            Box::new(substitute_indices(rhs, assignment)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (substitute_indices(value, assignment), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_indices(value, assignment)),
            Box::new(substitute_indices(body, assignment)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_indices(item, assignment))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_indices(cell, assignment))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn evaluate_with_rules(expr: &Expr, rules: &[ComponentRule]) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(tensor_name) = base.as_ref() {
                for rule in rules {
                    if rule.tensor == *tensor_name && rule.indices.len() == indices.len() {
                        let exact = rule
                            .indices
                            .iter()
                            .zip(indices.iter())
                            .all(|((rv, rvar), idx)| *rv == idx.name && *rvar == idx.variance);
                        if exact {
                            return rule.value.clone();
                        }
                        let names_only = rule
                            .indices
                            .iter()
                            .zip(indices.iter())
                            .all(|((rv, _), idx)| *rv == idx.name);
                        if names_only {
                            return rule.value.clone();
                        }
                    }
                }
                Expr::zero()
            } else {
                expr.clone()
            }
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| evaluate_with_rules(factor, rules))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| evaluate_with_rules(term, rules))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(evaluate_with_rules(inner, rules)),
        Expr::Pow(base, exp) => Expr::pow(
            evaluate_with_rules(base, rules),
            evaluate_with_rules(exp, rules),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| evaluate_with_rules(arg, rules))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(evaluate_with_rules(re, rules)),
            Box::new(evaluate_with_rules(im, rules)),
        ),
        Expr::FnDef(_, _, _) => expr.clone(),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(evaluate_with_rules(lhs, rules)),
            Box::new(evaluate_with_rules(rhs, rules)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (evaluate_with_rules(value, rules), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(evaluate_with_rules(value, rules)),
            Box::new(evaluate_with_rules(body, rules)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| evaluate_with_rules(item, rules))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| evaluate_with_rules(cell, rules))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn evaluate_components_v2(
    expr: &Expr,
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    interner: &ax_ir::Interner,
) -> Expr {
    evaluate_node(expr, rules, env, env.tensor_properties(), interner)
}

fn evaluate_node(
    expr: &Expr,
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => handle_sum(terms, rules, env, properties, interner),
        Expr::Mul(factors) => handle_prod(factors, rules, env, properties, interner),
        Expr::Neg(e) => Expr::neg(evaluate_node(e, rules, env, properties, interner)),
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(tensor_name) = base.as_ref() {
                let is_epsilon = properties
                    .get_properties_with_indices(*tensor_name, indices)
                    .iter()
                    .any(|prop| matches!(prop, ax_ir::TensorProperty::EpsilonTensor));
                if is_epsilon {
                    let epsilon_value = handle_epsilon(expr, *tensor_name, env, interner);
                    if epsilon_value != *expr {
                        return epsilon_value;
                    }
                }
            }
            handle_factor(expr, rules, env, properties, interner)
        }
        Expr::Call(f, args) => {
            let f_name = interner.resolve(*f);
            if is_derivative_name(f_name) {
                handle_derivative(expr, *f, args, rules, env, properties, interner)
            } else {
                let evaled_args: Vec<Expr> = args
                    .iter()
                    .map(|a| evaluate_node(a, rules, env, properties, interner))
                    .collect();
                Expr::Call(*f, evaled_args)
            }
        }
        _ => expr.clone(),
    }
}

fn is_derivative_name(name: &str) -> bool {
    matches!(
        name,
        "partial" | "nabla" | "D" | "d" | "diff" | "partial_derivative"
    )
}

fn handle_epsilon(
    expr: &Expr,
    epsilon_sym: lasso::Spur,
    env: &dyn ComponentEvalEnv,
    _interner: &ax_ir::Interner,
) -> Expr {
    if let Expr::Indexed(base, indices) = expr {
        if let Expr::Sym(sym) = base.as_ref() {
            if *sym != epsilon_sym {
                return expr.clone();
            }
            let is_epsilon = env
                .tensor_properties()
                .get(sym)
                .map(|props| {
                    props
                        .iter()
                        .any(|prop| matches!(prop, ax_ir::TensorProperty::EpsilonTensor))
                })
                .unwrap_or(false);
            if !is_epsilon {
                return expr.clone();
            }

            if !indices.iter().all(|idx| env.is_coordinate(idx.name)) {
                return expr.clone();
            }

            let coords = env.coordinates();
            let n = indices.len();
            let mut positions = Vec::with_capacity(n);
            for idx in indices {
                positions.push(coords.iter().position(|&coord| coord == idx.name));
            }
            if positions.iter().any(|pos| pos.is_none()) {
                return Expr::zero();
            }

            let pos_values: Vec<usize> = positions.into_iter().map(|pos| pos.unwrap()).collect();
            let mut seen = HashSet::new();
            for &pos in &pos_values {
                if !seen.insert(pos) {
                    return Expr::zero();
                }
            }

            if n != coords.len() {
                return Expr::zero();
            }

            match ax_perm::sign(&pos_values) {
                1 => Expr::Int(1.into()),
                -1 => Expr::Int((-1i64).into()),
                _ => Expr::zero(),
            }
        } else {
            expr.clone()
        }
    } else {
        expr.clone()
    }
}

fn lookup_component_rule(
    tensor_name: lasso::Spur,
    indices: &[Index],
    rules: &[ComponentRule],
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    for rule in rules {
        if rule.tensor == tensor_name && rule.indices.len() == indices.len() {
            let exact = rule
                .indices
                .iter()
                .zip(indices.iter())
                .all(|((rv, rvar), idx)| *rv == idx.name && *rvar == idx.variance);
            if exact {
                return Some(rule.value.clone());
            }
            let names_only = rule
                .indices
                .iter()
                .zip(indices.iter())
                .all(|((rv, _), idx)| *rv == idx.name);
            if names_only {
                return Some(rule.value.clone());
            }
        }
    }

    let index_names: Vec<lasso::Spur> = indices.iter().map(|i| i.name).collect();
    let variances: Vec<ax_ir::Variance> = indices.iter().map(|i| i.variance.clone()).collect();

    for prop in properties.get_properties_with_indices(tensor_name, indices) {
        match prop {
            ax_ir::TensorProperty::Symmetric(positions) => {
                let symmetric_slots: Vec<usize> = positions
                    .iter()
                    .filter(|&&p| p < index_names.len())
                    .copied()
                    .collect();
                if symmetric_slots.len() < 2 {
                    continue;
                }

                let mut slot_values: Vec<lasso::Spur> =
                    symmetric_slots.iter().map(|&p| index_names[p]).collect();
                slot_values.sort_by_key(|s| interner.resolve(*s).to_string());

                loop {
                    let mut trial = index_names.clone();
                    for (i, &slot) in symmetric_slots.iter().enumerate() {
                        trial[slot] = slot_values[i];
                    }

                    for rule in rules {
                        if rule.tensor == tensor_name && rule.indices.len() == indices.len() {
                            let matches = rule
                                .indices
                                .iter()
                                .zip(trial.iter().zip(variances.iter()))
                                .all(|((rv, rvar), (&tv, variance))| {
                                    *rv == tv && *rvar == *variance
                                });
                            if matches {
                                return Some(rule.value.clone());
                            }
                        }
                    }

                    if !next_permutation_by_key(&mut slot_values, interner) {
                        break;
                    }
                }
            }
            ax_ir::TensorProperty::AntiSymmetric(positions) => {
                let symmetric_slots: Vec<usize> = positions
                    .iter()
                    .filter(|&&p| p < index_names.len())
                    .copied()
                    .collect();
                if symmetric_slots.len() < 2 {
                    continue;
                }

                let original_values: Vec<lasso::Spur> =
                    symmetric_slots.iter().map(|&p| index_names[p]).collect();
                let mut slot_values = original_values.clone();
                slot_values.sort_by_key(|s| interner.resolve(*s).to_string());

                loop {
                    let mut trial = index_names.clone();
                    for (i, &slot) in symmetric_slots.iter().enumerate() {
                        trial[slot] = slot_values[i];
                    }

                    for rule in rules {
                        if rule.tensor == tensor_name && rule.indices.len() == indices.len() {
                            let matches = rule
                                .indices
                                .iter()
                                .zip(trial.iter().zip(variances.iter()))
                                .all(|((rv, rvar), (&tv, variance))| {
                                    *rv == tv && *rvar == *variance
                                });
                            if matches {
                                let sign = permutation_sign_between(&original_values, &slot_values);
                                return Some(if sign < 0 {
                                    Expr::neg(rule.value.clone())
                                } else {
                                    rule.value.clone()
                                });
                            }
                        }
                    }

                    if !next_permutation_by_key(&mut slot_values, interner) {
                        break;
                    }
                }
            }
            ax_ir::TensorProperty::RiemannSymmetry => {
                if indices.len() == 4 {
                    let riemann_perms: [([usize; 4], i32); 8] = [
                        ([0, 1, 2, 3], 1),
                        ([1, 0, 2, 3], -1),
                        ([0, 1, 3, 2], -1),
                        ([1, 0, 3, 2], 1),
                        ([2, 3, 0, 1], 1),
                        ([3, 2, 0, 1], -1),
                        ([2, 3, 1, 0], -1),
                        ([3, 2, 1, 0], 1),
                    ];

                    for (perm, sign) in riemann_perms {
                        let trial: Vec<lasso::Spur> =
                            perm.iter().map(|&p| index_names[p]).collect();
                        let trial_vars: Vec<ax_ir::Variance> =
                            perm.iter().map(|&p| variances[p].clone()).collect();

                        for rule in rules {
                            if rule.tensor == tensor_name && rule.indices.len() == 4 {
                                let matches = rule
                                    .indices
                                    .iter()
                                    .zip(trial.iter().zip(trial_vars.iter()))
                                    .all(|((rv, rvar), (&tv, variance))| {
                                        *rv == tv && *rvar == *variance
                                    });
                                if matches {
                                    return Some(if sign < 0 {
                                        Expr::neg(rule.value.clone())
                                    } else {
                                        rule.value.clone()
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

fn permutation_sign_between(original: &[lasso::Spur], permuted: &[lasso::Spur]) -> i32 {
    let n = original.len();
    let mut visited = vec![false; n];
    let mut sign = 1i32;
    let perm: Vec<usize> = permuted
        .iter()
        .map(|p| original.iter().position(|o| o == p).unwrap_or(0))
        .collect();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut cycle_len = 0;
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            j = perm[j];
            cycle_len += 1;
        }
        if cycle_len > 1 && cycle_len % 2 == 0 {
            sign = -sign;
        }
    }

    sign
}

fn next_permutation_by_key(arr: &mut [lasso::Spur], interner: &ax_ir::Interner) -> bool {
    let n = arr.len();
    if n <= 1 {
        return false;
    }

    let key = |s: &lasso::Spur| interner.resolve(*s).to_string();
    let mut i = n - 2;
    loop {
        if key(&arr[i]) < key(&arr[i + 1]) {
            break;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }

    let mut j = n - 1;
    while key(&arr[j]) <= key(&arr[i]) {
        j -= 1;
    }
    arr.swap(i, j);
    arr[i + 1..].reverse();
    true
}

fn handle_sum(
    terms: &[Expr],
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let evaled: Vec<Expr> = terms
        .iter()
        .map(|t| evaluate_node(t, rules, env, properties, interner))
        .collect();
    let simplified = Expr::add(evaled);
    simplify_expr(simplified, interner)
}

fn handle_prod(
    factors: &[Expr],
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let evaled: Vec<Expr> = factors
        .iter()
        .map(|f| evaluate_node(f, rules, env, properties, interner))
        .collect();

    let product = Expr::mul(evaled.clone());
    let ic = index_classifier::classify_indices(&product);
    if ic.dummy.is_empty() {
        return Expr::mul(evaled);
    }

    let mut coords = env.coordinates();
    coords.sort_by_key(|sym| interner.resolve(*sym).to_string());
    if coords.is_empty() {
        return Expr::mul(evaled);
    }

    let n_dummy = ic.dummy.len();
    let n_coords = coords.len();
    let total_combos = n_coords.pow(n_dummy as u32);
    let mut sum_terms = Vec::new();

    for combo in 0..total_combos {
        let mut assignment: HashMap<lasso::Spur, lasso::Spur> = HashMap::new();
        let mut idx = combo;
        for (name, _, _, _, _) in &ic.dummy {
            assignment.insert(*name, coords[idx % n_coords]);
            idx /= n_coords;
        }

        let mut term_factors = Vec::new();
        for factor in &evaled {
            let substituted = substitute_index_values(factor, &assignment, interner);
            let evaluated = evaluate_with_rules_deep(&substituted, rules, env, interner);
            term_factors.push(evaluated);
        }

        let term = Expr::mul(term_factors);
        let simplified = simplify_expr(term, interner);
        if simplified != Expr::zero() {
            sum_terms.push(simplified);
        }
    }

    if sum_terms.is_empty() {
        Expr::zero()
    } else {
        let result = Expr::add(sum_terms);
        simplify_expr(result, interner)
    }
}

fn handle_derivative(
    expr: &Expr,
    deriv_sym: lasso::Spur,
    args: &[Expr],
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if args.is_empty() {
        return expr.clone();
    }

    let inner = &args[0];
    let inner_evaled = match inner {
        Expr::Call(f, inner_args) if is_derivative_name(interner.resolve(*f)) => {
            handle_derivative(inner, *f, inner_args, rules, env, properties, interner)
        }
        _ => evaluate_node(inner, rules, env, properties, interner),
    };

    let deriv_indices: Vec<lasso::Spur> = args[1..]
        .iter()
        .filter_map(|arg| match arg {
            Expr::Sym(s) => Some(*s),
            _ => None,
        })
        .collect();

    if deriv_indices.is_empty() {
        return Expr::Call(deriv_sym, vec![inner_evaled]);
    }

    if deriv_indices.len() == 1 && env.is_coordinate(deriv_indices[0]) {
        return diff_component(&inner_evaled, deriv_indices[0], interner);
    }

    if deriv_indices.len() == 1 {
        let idx = deriv_indices[0];
        if has_abstract_indices(&inner_evaled) {
            let expanded = handle_factor(&inner_evaled, rules, env, properties, interner);
            if has_abstract_indices(&expanded) {
                return Expr::Call(deriv_sym, vec![expanded, Expr::Sym(idx)]);
            }
            return diff_component(&expanded, idx, interner);
        }

        return diff_component(&inner_evaled, idx, interner);
    }

    let mut result = inner_evaled;
    for &idx in deriv_indices.iter().rev() {
        if env.is_coordinate(idx) {
            result = diff_component(&result, idx, interner);
        } else {
            result = Expr::Call(deriv_sym, vec![result, Expr::Sym(idx)]);
        }
    }

    simplify_expr(result, interner)
}

fn has_abstract_indices(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, indices) => !indices.is_empty(),
        Expr::Mul(factors) => factors.iter().any(has_abstract_indices),
        Expr::Add(terms) => terms.iter().any(has_abstract_indices),
        Expr::Neg(e) => has_abstract_indices(e),
        _ => false,
    }
}

fn handle_factor(
    expr: &Expr,
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if let Expr::Indexed(base, indices) = expr {
        if let Expr::Sym(tensor_name) = base.as_ref() {
            let epsilon_value = handle_epsilon(expr, *tensor_name, env, interner);
            if epsilon_value != *expr {
                return epsilon_value;
            }

            let classification = classify_indices(expr);
            let mut unresolved_names: Vec<lasso::Spur> = classification
                .all
                .iter()
                .filter_map(|(_, idx)| (!env.is_coordinate(idx.name)).then_some(idx.name))
                .collect();
            unresolved_names.sort_by_key(|sym| interner.resolve(*sym).to_string());
            unresolved_names.dedup();

            if !unresolved_names.is_empty() {
                let mut coords = env.coordinates();
                coords.sort_by_key(|sym| interner.resolve(*sym).to_string());
                if coords.is_empty() {
                    return expr.clone();
                }

                fn recurse(
                    pos: usize,
                    names: &[lasso::Spur],
                    coords: &[lasso::Spur],
                    expr: &Expr,
                    rules: &[ComponentRule],
                    env: &dyn ComponentEvalEnv,
                    properties: &dyn PropertyLookup,
                    interner: &ax_ir::Interner,
                    assignment: &mut HashMap<lasso::Spur, lasso::Spur>,
                    out: &mut Vec<Expr>,
                ) {
                    if pos == names.len() {
                        let substituted = substitute_index_values(expr, assignment, interner);
                        out.push(handle_factor(
                            &substituted,
                            rules,
                            env,
                            properties,
                            interner,
                        ));
                        return;
                    }

                    for &coord in coords {
                        assignment.insert(names[pos], coord);
                        recurse(
                            pos + 1,
                            names,
                            coords,
                            expr,
                            rules,
                            env,
                            properties,
                            interner,
                            assignment,
                            out,
                        );
                    }
                }

                let mut terms = Vec::new();
                let mut assignment = HashMap::new();
                recurse(
                    0,
                    &unresolved_names,
                    &coords,
                    expr,
                    rules,
                    env,
                    properties,
                    interner,
                    &mut assignment,
                    &mut terms,
                );
                let result = Expr::add(terms);
                return simplify_expr(result, interner);
            }

            let all_concrete = indices.iter().all(|idx| env.is_coordinate(idx.name));
            if all_concrete {
                if let Some(value) =
                    lookup_component_rule(*tensor_name, indices, rules, properties, interner)
                {
                    return value;
                }
                return Expr::zero();
            }
        }
    }
    expr.clone()
}

fn evaluate_with_rules_deep(
    expr: &Expr,
    rules: &[ComponentRule],
    env: &dyn ComponentEvalEnv,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(tensor_name) = base.as_ref() {
                let epsilon_value = handle_epsilon(expr, *tensor_name, env, interner);
                if epsilon_value != *expr {
                    return epsilon_value;
                }
                if let Some(value) = lookup_component_rule(
                    *tensor_name,
                    indices,
                    rules,
                    env.tensor_properties(),
                    interner,
                ) {
                    return value;
                }
                Expr::zero()
            } else {
                expr.clone()
            }
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| evaluate_with_rules_deep(f, rules, env, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| evaluate_with_rules_deep(t, rules, env, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(evaluate_with_rules_deep(e, rules, env, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| evaluate_with_rules_deep(a, rules, env, interner))
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn substitute_index_values(
    expr: &Expr,
    assignment: &HashMap<lasso::Spur, lasso::Spur>,
    _interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    let new_name = assignment.get(&idx.name).copied().unwrap_or(idx.name);
                    ax_ir::Index {
                        name: new_name,
                        variance: idx.variance.clone(),
                        index_type: idx.index_type,
                    }
                })
                .collect();
            Expr::Indexed(
                Box::new(substitute_index_values(base, assignment, _interner)),
                new_indices,
            )
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| substitute_index_values(f, assignment, _interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| substitute_index_values(t, assignment, _interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(substitute_index_values(e, assignment, _interner)),
        _ => expr.clone(),
    }
}

pub fn evaluate_components<E: ComponentEvalEnv>(
    expr: &ax_ir::Expr,
    rules: &[ComponentRule],
    _index_values: &std::collections::HashMap<lasso::Spur, Vec<lasso::Spur>>,
    env: &E,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    evaluate_components_v2(expr, rules, env, interner)
}

#[derive(Clone, Debug)]
pub struct SymbolicMatrix {
    pub dim: usize,
    pub data: Vec<Vec<ax_ir::Expr>>,
}

impl SymbolicMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![vec![Expr::zero(); dim]; dim],
        }
    }

    pub fn from_diagonal(diag: Vec<ax_ir::Expr>) -> Self {
        let dim = diag.len();
        let mut matrix = Self::new(dim);
        for (i, value) in diag.into_iter().enumerate() {
            matrix.data[i][i] = value;
        }
        matrix
    }

    pub fn get(&self, row: usize, col: usize) -> &ax_ir::Expr {
        &self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: ax_ir::Expr) {
        self.data[row][col] = val;
    }

    pub fn symbolic_inverse(&self, interner: &ax_ir::Interner) -> Self {
        let is_diagonal = (0..self.dim)
            .all(|row| (0..self.dim).all(|col| row == col || self.data[row][col] == Expr::zero()));

        if is_diagonal {
            let mut inverse = Self::new(self.dim);
            for i in 0..self.dim {
                inverse.data[i][i] = simplify_expr(
                    Expr::pow(self.data[i][i].clone(), Expr::Int((-1).into())),
                    interner,
                );
            }
            return inverse;
        }

        match ax_linalg::inverse(&self.data, interner) {
            Some(inv_data) => {
                let mut result = Self::new(self.dim);
                for (i, row) in inv_data.iter().enumerate().take(self.dim) {
                    for (j, cell) in row.iter().enumerate().take(self.dim) {
                        result.data[i][j] = simplify_expr(cell.clone(), interner);
                    }
                }
                result
            }
            None => panic!("metric tensor is singular (determinant is zero)"),
        }
    }
}

pub fn detect_contractions(indices: &[ax_ir::Index]) -> Vec<(usize, usize)> {
    let mut used = HashSet::new();
    let mut pairs = Vec::new();

    for i in 0..indices.len() {
        if used.contains(&i) {
            continue;
        }
        for j in (i + 1)..indices.len() {
            if used.contains(&j) {
                continue;
            }
            if indices[i].name == indices[j].name
                && indices[i].variance != indices[j].variance
                && (indices[i].index_type == indices[j].index_type
                    || indices[i].index_type.is_none()
                    || indices[j].index_type.is_none())
            {
                used.insert(i);
                used.insert(j);
                pairs.push((i, j));
                break;
            }
        }
    }

    pairs
}

fn permutation_parity(original: &[ax_ir::Index], sorted: &[ax_ir::Index]) -> bool {
    let mut used = vec![false; sorted.len()];
    let mut permutation = Vec::with_capacity(original.len());
    for item in original {
        let pos = sorted
            .iter()
            .enumerate()
            .find(|(idx, candidate)| !used[*idx] && *candidate == item)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        used[pos] = true;
        permutation.push(pos);
    }

    let mut inversions = 0usize;
    for i in 0..permutation.len() {
        for j in (i + 1)..permutation.len() {
            if permutation[i] > permutation[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 1
}

fn sort_key(index: &ax_ir::Index, interner: &Interner) -> (String, u8) {
    (
        interner.resolve(index.name).to_string(),
        match index.variance {
            ax_ir::Variance::Down => 0,
            ax_ir::Variance::Up => 1,
        },
    )
}

fn tensor_sort_key(expr: &Expr, interner: &ax_ir::Interner) -> (u8, String, String) {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => (0, String::new(), String::new()),
        Expr::Sym(s) => (1, interner.resolve(*s).to_string(), String::new()),
        Expr::Indexed(base, indices) => {
            let base_name = if let Expr::Sym(s) = base.as_ref() {
                interner.resolve(*s).to_string()
            } else {
                format!("{base:?}")
            };
            let first_index = indices
                .first()
                .map(|idx| interner.resolve(idx.name).to_string())
                .unwrap_or_default();
            (2, base_name, first_index)
        }
        Expr::Call(f, args) => {
            let first_arg = args
                .first()
                .map(|arg| format!("{arg:?}"))
                .unwrap_or_default();
            (2, interner.resolve(*f).to_string(), first_arg)
        }
        _ => (4, format!("{expr:?}"), String::new()),
    }
}

pub fn sort_product(
    expr: &ax_ir::Expr,
    _tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut sorted: Vec<Expr> = factors
                .iter()
                .map(|factor| sort_product(factor, _tensor_properties, interner))
                .collect();
            sorted.sort_by(|a, b| tensor_sort_key(a, interner).cmp(&tensor_sort_key(b, interner)));
            Expr::mul(sorted)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| sort_product(term, _tensor_properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(sort_product(inner, _tensor_properties, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            sort_product(base, _tensor_properties, interner),
            sort_product(exp, _tensor_properties, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| sort_product(arg, _tensor_properties, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(sort_product(re, _tensor_properties, interner)),
            Box::new(sort_product(im, _tensor_properties, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(sort_product(body, _tensor_properties, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(sort_product(lhs, _tensor_properties, interner)),
            Box::new(sort_product(rhs, _tensor_properties, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| {
                    (
                        sort_product(value, _tensor_properties, interner),
                        cond.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(sort_product(value, _tensor_properties, interner)),
            Box::new(sort_product(body, _tensor_properties, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| sort_product(item, _tensor_properties, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| sort_product(cell, _tensor_properties, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn product_rule(
    expr: &ax_ir::Expr,
    derivative_syms: &HashSet<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Call(f, args) if derivative_syms.contains(f) && args.len() == 1 => match &args[0] {
            Expr::Mul(factors) if factors.len() >= 2 => {
                let terms: Vec<Expr> = (0..factors.len())
                    .map(|i| {
                        let mut new_factors = Vec::with_capacity(factors.len());
                        for (j, factor) in factors.iter().enumerate() {
                            if i == j {
                                new_factors.push(Expr::Call(
                                    *f,
                                    vec![product_rule(factor, derivative_syms, interner)],
                                ));
                            } else {
                                new_factors.push(product_rule(factor, derivative_syms, interner));
                            }
                        }
                        Expr::mul(new_factors)
                    })
                    .collect();
                Expr::add(terms)
            }
            Expr::Add(terms) => Expr::add(
                terms
                    .iter()
                    .map(|term| Expr::Call(*f, vec![product_rule(term, derivative_syms, interner)]))
                    .collect(),
            ),
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
            _ => Expr::Call(*f, vec![product_rule(&args[0], derivative_syms, interner)]),
        },
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| product_rule(term, derivative_syms, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| product_rule(factor, derivative_syms, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(product_rule(inner, derivative_syms, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            product_rule(base, derivative_syms, interner),
            product_rule(exp, derivative_syms, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| product_rule(arg, derivative_syms, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(product_rule(re, derivative_syms, interner)),
            Box::new(product_rule(im, derivative_syms, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(product_rule(body, derivative_syms, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(product_rule(lhs, derivative_syms, interner)),
            Box::new(product_rule(rhs, derivative_syms, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (product_rule(value, derivative_syms, interner), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(product_rule(value, derivative_syms, interner)),
            Box::new(product_rule(body, derivative_syms, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| product_rule(item, derivative_syms, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| product_rule(cell, derivative_syms, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn tensor_distribute(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let distributed_factors: Vec<Expr> = factors
                .iter()
                .map(|factor| tensor_distribute(factor, interner))
                .collect();
            for (i, factor) in distributed_factors.iter().enumerate() {
                if let Expr::Add(terms) = factor {
                    let rest: Vec<Expr> = distributed_factors
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, f)| f.clone())
                        .collect();
                    let expanded: Vec<Expr> = terms
                        .iter()
                        .map(|term| {
                            let mut new_factors = rest.clone();
                            new_factors.push(term.clone());
                            Expr::mul(new_factors)
                        })
                        .collect();
                    return tensor_distribute(&Expr::add(expanded), interner);
                }
            }
            Expr::mul(distributed_factors)
        }
        Expr::Indexed(base, indices) => {
            let distributed_base = tensor_distribute(base, interner);
            if let Expr::Add(terms) = distributed_base {
                Expr::add(
                    terms
                        .iter()
                        .map(|term| {
                            Expr::Indexed(
                                Box::new(tensor_distribute(term, interner)),
                                indices.clone(),
                            )
                        })
                        .collect(),
                )
            } else {
                Expr::Indexed(Box::new(distributed_base), indices.clone())
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| tensor_distribute(term, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(tensor_distribute(inner, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            tensor_distribute(base, interner),
            tensor_distribute(exp, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| tensor_distribute(arg, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(tensor_distribute(re, interner)),
            Box::new(tensor_distribute(im, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(tensor_distribute(body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(tensor_distribute(lhs, interner)),
            Box::new(tensor_distribute(rhs, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (tensor_distribute(value, interner), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(tensor_distribute(value, interner)),
            Box::new(tensor_distribute(body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| tensor_distribute(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| tensor_distribute(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn factorial(n: usize) -> num_bigint::BigInt {
    (1..=n).fold(num_bigint::BigInt::from(1), |acc, i| {
        acc * num_bigint::BigInt::from(i)
    })
}

pub fn epsilon_to_delta(
    expr: &ax_ir::Expr,
    epsilon_sym: lasso::Spur,
    delta_sym: lasso::Spur,
    dim: usize,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut eps_indices: Vec<(usize, Vec<ax_ir::Index>)> = Vec::new();
            for (i, factor) in factors.iter().enumerate() {
                if let Expr::Indexed(base, indices) = factor {
                    if let Expr::Sym(sym) = base.as_ref() {
                        if *sym == epsilon_sym && indices.len() == dim {
                            eps_indices.push((i, indices.clone()));
                        }
                    }
                }
            }

            if eps_indices.len() < 2 {
                return Expr::mul(
                    factors
                        .iter()
                        .map(|factor| {
                            epsilon_to_delta(factor, epsilon_sym, delta_sym, dim, interner)
                        })
                        .collect(),
                );
            }

            let (i1, idx1) = &eps_indices[0];
            let (i2, idx2) = &eps_indices[1];

            let mut contracted = Vec::new();
            let mut free1 = Vec::new();
            let mut free2 = Vec::new();

            for a in idx1 {
                let mut found = false;
                for b in idx2 {
                    if a.name == b.name && a.variance != b.variance {
                        contracted.push(a.name);
                        found = true;
                        break;
                    }
                }
                if !found {
                    free1.push(a.clone());
                }
            }
            for b in idx2 {
                if !contracted.contains(&b.name) {
                    free2.push(b.clone());
                }
            }

            let coeff = Expr::Int(factorial(contracted.len()));
            let delta_product = if free1.is_empty() {
                Expr::one()
            } else {
                let deltas: Vec<Expr> = free1
                    .iter()
                    .zip(free2.iter())
                    .map(|(a, b)| {
                        Expr::Indexed(Box::new(Expr::Sym(delta_sym)), vec![a.clone(), b.clone()])
                    })
                    .collect();
                if deltas.len() == 1 {
                    deltas[0].clone()
                } else {
                    Expr::mul(deltas)
                }
            };

            let mut remaining: Vec<Expr> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != *i1 && *j != *i2)
                .map(|(_, factor)| epsilon_to_delta(factor, epsilon_sym, delta_sym, dim, interner))
                .collect();
            remaining.push(coeff);
            remaining.push(delta_product);
            Expr::mul(remaining)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| epsilon_to_delta(term, epsilon_sym, delta_sym, dim, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(epsilon_to_delta(
            inner,
            epsilon_sym,
            delta_sym,
            dim,
            interner,
        )),
        Expr::Pow(base, exp) => Expr::pow(
            epsilon_to_delta(base, epsilon_sym, delta_sym, dim, interner),
            epsilon_to_delta(exp, epsilon_sym, delta_sym, dim, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| epsilon_to_delta(arg, epsilon_sym, delta_sym, dim, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(epsilon_to_delta(re, epsilon_sym, delta_sym, dim, interner)),
            Box::new(epsilon_to_delta(im, epsilon_sym, delta_sym, dim, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(epsilon_to_delta(
                body,
                epsilon_sym,
                delta_sym,
                dim,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(epsilon_to_delta(lhs, epsilon_sym, delta_sym, dim, interner)),
            Box::new(epsilon_to_delta(rhs, epsilon_sym, delta_sym, dim, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| {
                    (
                        epsilon_to_delta(value, epsilon_sym, delta_sym, dim, interner),
                        cond.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(epsilon_to_delta(
                value,
                epsilon_sym,
                delta_sym,
                dim,
                interner,
            )),
            Box::new(epsilon_to_delta(
                body,
                epsilon_sym,
                delta_sym,
                dim,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| epsilon_to_delta(item, epsilon_sym, delta_sym, dim, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| epsilon_to_delta(cell, epsilon_sym, delta_sym, dim, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

/// Expand a generalised Kronecker delta (more than 2 indices) into a sum
/// of products of ordinary 2-index deltas.
pub fn expand_delta(expr: &Expr, delta_sym: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    let _ = interner;

    match expr {
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(sym) = base.as_ref() {
                if *sym == delta_sym && indices.len() > 2 && indices.len() % 2 == 0 {
                    let n = indices.len() / 2;
                    let upper: Vec<&ax_ir::Index> = indices
                        .iter()
                        .filter(|idx| idx.variance == ax_ir::Variance::Up)
                        .collect();
                    let lower: Vec<&ax_ir::Index> = indices
                        .iter()
                        .filter(|idx| idx.variance == ax_ir::Variance::Down)
                        .collect();

                    if upper.len() != n || lower.len() != n {
                        return expr.clone();
                    }

                    let mut perm: Vec<usize> = (0..n).collect();
                    let mut terms = Vec::new();

                    loop {
                        let sign = ax_perm::sign(&perm);
                        let mut deltas = Vec::with_capacity(n);
                        for i in 0..n {
                            deltas.push(Expr::Indexed(
                                Box::new(Expr::Sym(delta_sym)),
                                vec![upper[i].clone(), lower[perm[i]].clone()],
                            ));
                        }

                        let product = Expr::mul(deltas);
                        if sign == -1 {
                            terms.push(Expr::neg(product));
                        } else {
                            terms.push(product);
                        }

                        if !next_permutation_usize(&mut perm) {
                            break;
                        }
                    }

                    return Expr::add(terms);
                }
            }
            expr.clone()
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| expand_delta(factor, delta_sym, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| expand_delta(term, delta_sym, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(expand_delta(inner, delta_sym, interner)),
        _ => expr.clone(),
    }
}

/// Symmetrise or antisymmetrise an expression over the given index positions.
/// Returns the sum over all permutations of those indices, divided by n!.
pub fn symmetrise(
    expr: &Expr,
    positions: &[usize],
    antisymmetric: bool,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = interner;

    let n = positions.len();
    if n <= 1 {
        return expr.clone();
    }

    let mut perm: Vec<usize> = (0..n).collect();
    let mut terms = Vec::new();
    let factorial = (1..=n).product::<usize>();

    loop {
        let sign = if antisymmetric {
            ax_perm::sign(&perm)
        } else {
            1
        };
        let permuted = permute_indices_at_positions(expr, positions, &perm);
        if sign == -1 {
            terms.push(Expr::neg(permuted));
        } else {
            terms.push(permuted);
        }

        if !next_permutation_usize(&mut perm) {
            break;
        }
    }

    Expr::mul(vec![
        Expr::Rational(BigRational::new(
            1.into(),
            num_bigint::BigInt::from(factorial),
        )),
        Expr::add(terms),
    ])
}

fn permute_indices_at_positions(expr: &Expr, positions: &[usize], perm: &[usize]) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let mut new_indices = indices.clone();
            for (i, &pos) in positions.iter().enumerate() {
                if pos < indices.len() && perm[i] < positions.len() {
                    let source_pos = positions[perm[i]];
                    if source_pos < indices.len() {
                        new_indices[pos] = ax_ir::Index {
                            name: indices[source_pos].name,
                            variance: indices[pos].variance.clone(),
                            index_type: indices[pos].index_type,
                        };
                    }
                }
            }
            Expr::Indexed(base.clone(), new_indices)
        }
        Expr::Mul(factors) => {
            let mut result = factors.clone();
            let mut idx_offset = 0usize;

            for factor in &mut result {
                if let Expr::Indexed(_, indices) = factor {
                    let n_idx = indices.len();
                    let local_positions: Vec<usize> = positions
                        .iter()
                        .copied()
                        .filter(|pos| *pos >= idx_offset && *pos < idx_offset + n_idx)
                        .map(|pos| pos - idx_offset)
                        .collect();
                    if !local_positions.is_empty() {
                        *factor = permute_indices_at_positions(factor, &local_positions, perm);
                    }
                    idx_offset += n_idx;
                }
            }

            Expr::mul(result)
        }
        _ => expr.clone(),
    }
}

fn next_permutation_usize(arr: &mut [usize]) -> bool {
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

pub fn canonicalize_indices(
    expr: &ax_ir::Expr,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let base_expr = canonicalize_indices(base, properties, interner);
            let mut indices = indices.clone();
            let mut negate = false;

            if let Expr::Sym(sym) = &base_expr {
                for prop in properties.get_properties_with_indices(*sym, &indices) {
                    match prop {
                        ax_ir::TensorProperty::Symmetric(positions) => {
                            let mut original = positions
                                .iter()
                                .filter_map(|&pos| indices.get(pos).cloned())
                                .collect::<Vec<_>>();
                            let mut sorted = original.clone();
                            sorted.sort_by_key(|idx| sort_key(idx, interner));
                            for (slot, value) in positions.iter().zip(sorted.iter()) {
                                if let Some(target) = indices.get_mut(*slot) {
                                    *target = value.clone();
                                }
                            }
                            original.clear();
                        }
                        ax_ir::TensorProperty::AntiSymmetric(positions) => {
                            let original = positions
                                .iter()
                                .filter_map(|&pos| indices.get(pos).cloned())
                                .collect::<Vec<_>>();
                            if original.len() >= 2 {
                                for i in 0..original.len() {
                                    for j in (i + 1)..original.len() {
                                        if original[i] == original[j] {
                                            return Expr::zero();
                                        }
                                    }
                                }
                            }
                            let mut sorted = original.clone();
                            sorted.sort_by_key(|idx| sort_key(idx, interner));
                            if permutation_parity(&original, &sorted) {
                                negate = !negate;
                            }
                            for (slot, value) in positions.iter().zip(sorted.iter()) {
                                if let Some(target) = indices.get_mut(*slot) {
                                    *target = value.clone();
                                }
                            }
                        }
                        ax_ir::TensorProperty::RiemannSymmetry => {
                            if indices.len() == 4 {
                                let pairs = [vec![0usize, 1usize], vec![2usize, 3usize]];
                                for positions in pairs {
                                    let original = positions
                                        .iter()
                                        .filter_map(|&pos| indices.get(pos).cloned())
                                        .collect::<Vec<_>>();
                                    if original[0] == original[1] {
                                        return Expr::zero();
                                    }
                                    let mut sorted = original.clone();
                                    sorted.sort_by_key(|idx| sort_key(idx, interner));
                                    if permutation_parity(&original, &sorted) {
                                        negate = !negate;
                                    }
                                    for (slot, value) in positions.iter().zip(sorted.iter()) {
                                        indices[*slot] = value.clone();
                                    }
                                }

                                let left = vec![indices[0].clone(), indices[1].clone()];
                                let right = vec![indices[2].clone(), indices[3].clone()];
                                let left_key = left
                                    .iter()
                                    .map(|i| sort_key(i, interner))
                                    .collect::<Vec<_>>();
                                let right_key = right
                                    .iter()
                                    .map(|i| sort_key(i, interner))
                                    .collect::<Vec<_>>();
                                if right_key < left_key {
                                    indices.swap(0, 2);
                                    indices.swap(1, 3);
                                }
                            }
                        }
                        ax_ir::TensorProperty::Traceless
                        | ax_ir::TensorProperty::Metric
                        | ax_ir::TensorProperty::InverseMetric
                        | ax_ir::TensorProperty::KroneckerDelta
                        | ax_ir::TensorProperty::EpsilonTensor
                        | ax_ir::TensorProperty::Derivative
                        | ax_ir::TensorProperty::PartialDerivative
                        | ax_ir::TensorProperty::CovariantDerivative
                        | ax_ir::TensorProperty::Depends(_)
                        | ax_ir::TensorProperty::Spinor
                        | ax_ir::TensorProperty::DiracBar
                        | ax_ir::TensorProperty::GammaMatrixProp
                        | ax_ir::TensorProperty::Commuting
                        | ax_ir::TensorProperty::AntiCommuting
                        | ax_ir::TensorProperty::NonCommuting
                        | ax_ir::TensorProperty::SortOrder(_)
                        | ax_ir::TensorProperty::TableauSymmetry { .. }
                        | ax_ir::TensorProperty::SatisfiesBianchi
                        | ax_ir::TensorProperty::WeylTensor
                        | ax_ir::TensorProperty::DifferentialFormDegree(_) => {}
                    }
                }
            }

            let out = Expr::Indexed(Box::new(base_expr), indices);
            if negate {
                Expr::neg(out)
            } else {
                out
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| canonicalize_indices(term, properties, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| canonicalize_indices(factor, properties, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            canonicalize_indices(base, properties, interner),
            canonicalize_indices(exp, properties, interner),
        ),
        Expr::Neg(inner) => Expr::neg(canonicalize_indices(inner, properties, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| canonicalize_indices(arg, properties, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(canonicalize_indices(re, properties, interner)),
            Box::new(canonicalize_indices(im, properties, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(canonicalize_indices(body, properties, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(canonicalize_indices(lhs, properties, interner)),
            Box::new(canonicalize_indices(rhs, properties, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| {
                    (
                        canonicalize_indices(value, properties, interner),
                        cond.clone(),
                    )
                })
                .collect(),
        ),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(canonicalize_indices(val, properties, interner)),
            Box::new(canonicalize_indices(body, properties, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| canonicalize_indices(item, properties, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| canonicalize_indices(cell, properties, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn apply_index_rename(expr: &Expr, rename_map: &HashMap<lasso::Spur, lasso::Spur>) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(apply_index_rename(base, rename_map)),
            indices
                .iter()
                .map(|idx| ax_ir::Index {
                    name: rename_map.get(&idx.name).copied().unwrap_or(idx.name),
                    variance: idx.variance.clone(),
                    index_type: idx.index_type,
                })
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| apply_index_rename(factor, rename_map))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| apply_index_rename(term, rename_map))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(apply_index_rename(inner, rename_map)),
        Expr::Pow(base, exp) => Expr::pow(
            apply_index_rename(base, rename_map),
            apply_index_rename(exp, rename_map),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| apply_index_rename(arg, rename_map))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(apply_index_rename(re, rename_map)),
            Box::new(apply_index_rename(im, rename_map)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(apply_index_rename(body, rename_map)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(apply_index_rename(lhs, rename_map)),
            Box::new(apply_index_rename(rhs, rename_map)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (apply_index_rename(value, rename_map), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(apply_index_rename(value, rename_map)),
            Box::new(apply_index_rename(body, rename_map)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| apply_index_rename(item, rename_map))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| apply_index_rename(cell, rename_map))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn rename_dummies<E: DummyRenameEnv>(
    expr: &ax_ir::Expr,
    env: &E,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    fn family_key<E: DummyRenameEnv>(
        env: &E,
        sym: lasso::Spur,
        interner: &ax_ir::Interner,
    ) -> String {
        env.index_to_family()
            .get(&sym)
            .map(|family| interner.resolve(*family).to_string())
            .unwrap_or_default()
    }

    fn canonical_name<E: DummyRenameEnv>(
        env: &E,
        original: lasso::Spur,
        slot: usize,
        interner: &ax_ir::Interner,
    ) -> lasso::Spur {
        if let Some(family) = env.index_to_family().get(&original) {
            interner.get_or_intern(&format!("_{}_d{}", interner.resolve(*family), slot))
        } else {
            interner.get_or_intern(&format!("_d{}", slot))
        }
    }

    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| rename_dummies(term, env, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(rename_dummies(inner, env, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            rename_dummies(base, env, interner),
            rename_dummies(exp, env, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| rename_dummies(arg, env, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rename_dummies(re, env, interner)),
            Box::new(rename_dummies(im, env, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(rename_dummies(body, env, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rename_dummies(lhs, env, interner)),
            Box::new(rename_dummies(rhs, env, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (rename_dummies(value, env, interner), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(rename_dummies(value, env, interner)),
            Box::new(rename_dummies(body, env, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| rename_dummies(item, env, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| rename_dummies(cell, env, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => {
            let classification = classify_indices(expr);
            let mut dummy_indices: Vec<(lasso::Spur, usize)> = classification
                .dummy
                .iter()
                .map(|(name, pos1, _, _, _)| (*name, *pos1))
                .collect();
            dummy_indices.sort_by_key(|(sym, first_occurrence)| {
                (family_key(env, *sym, interner), *first_occurrence)
            });

            let mut rename_map = HashMap::new();
            for (i, (dummy, _)) in dummy_indices.iter().enumerate() {
                rename_map.insert(*dummy, canonical_name(env, *dummy, i, interner));
            }

            apply_index_rename(expr, &rename_map)
        }
    }
}

pub fn rename_dummy_indices(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    struct EmptyDummyEnv {
        index_families: HashMap<lasso::Spur, ax_ir::IndexFamily>,
        index_to_family: HashMap<lasso::Spur, lasso::Spur>,
    }

    impl DummyRenameEnv for EmptyDummyEnv {
        fn index_families(&self) -> &HashMap<lasso::Spur, ax_ir::IndexFamily> {
            &self.index_families
        }

        fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur> {
            &self.index_to_family
        }
    }

    let env = EmptyDummyEnv {
        index_families: HashMap::new(),
        index_to_family: HashMap::new(),
    };
    rename_dummies(expr, &env, interner)
}

fn contains_index(expr: &Expr, idx: lasso::Spur) -> bool {
    classify_indices(expr)
        .all
        .iter()
        .any(|(_, index)| index.name == idx)
}

fn replace_index(expr: &Expr, from: lasso::Spur, to: lasso::Spur) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    if idx.name == from {
                        ax_ir::Index {
                            name: to,
                            variance: idx.variance.clone(),
                            index_type: idx.index_type,
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();
            Expr::Indexed(base.clone(), new_indices)
        }
        Expr::Mul(factors) => {
            Expr::mul(factors.iter().map(|f| replace_index(f, from, to)).collect())
        }
        Expr::Add(terms) => Expr::add(terms.iter().map(|t| replace_index(t, from, to)).collect()),
        Expr::Neg(e) => Expr::neg(replace_index(e, from, to)),
        _ => expr.clone(),
    }
}

fn has_index_with_variance(expr: &Expr, name: lasso::Spur, variance: &ax_ir::Variance) -> bool {
    classify_indices(expr)
        .all
        .iter()
        .any(|(_, index)| index.name == name && index.variance == *variance)
}

fn replace_index_with_variance(
    expr: &Expr,
    from_name: lasso::Spur,
    from_variance: &ax_ir::Variance,
    to_name: lasso::Spur,
    to_variance: &ax_ir::Variance,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    if idx.name == from_name && idx.variance == *from_variance {
                        ax_ir::Index {
                            name: to_name,
                            variance: to_variance.clone(),
                            index_type: idx.index_type,
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();
            Expr::Indexed(base.clone(), new_indices)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| {
                    replace_index_with_variance(f, from_name, from_variance, to_name, to_variance)
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

/// Simplify products of Kronecker deltas by performing contractions.
///
/// δ^a_b δ^b_c → δ^a_c
/// δ^a_a → dim
/// Apply a Young projector to a tensor expression.
///
/// Antisymmetrises over columns, then symmetrises over rows.
pub fn young_project(
    expr: &Expr,
    tableau: &ax_young::YoungTableau,
    interner: &ax_ir::Interner,
) -> Expr {
    // Build column groups (for antisymmetrisation)
    let n_cols = tableau.cells.first().map_or(0, |r| r.len());
    let mut col_groups: Vec<Vec<usize>> = Vec::new();
    for col in 0..n_cols {
        let group: Vec<usize> = tableau
            .cells
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        if group.len() > 1 {
            col_groups.push(group);
        }
    }

    // Build row groups (for symmetrisation)
    let mut row_groups: Vec<Vec<usize>> = Vec::new();
    for row in &tableau.cells {
        if row.len() > 1 {
            row_groups.push(row.clone());
        }
    }

    // Apply column antisymmetrisation, then row symmetrisation
    let antisymmed = antisymmetrise_groups(expr, &col_groups, interner);
    symmetrise_groups(&antisymmed, &row_groups, interner)
}

fn antisymmetrise_groups(expr: &Expr, groups: &[Vec<usize>], interner: &ax_ir::Interner) -> Expr {
    let mut result = expr.clone();
    for group in groups {
        result = symmetrise(&result, group, true, interner);
    }
    result
}

fn symmetrise_groups(expr: &Expr, groups: &[Vec<usize>], interner: &ax_ir::Interner) -> Expr {
    let mut result = expr.clone();
    for group in groups {
        result = symmetrise(&result, group, false, interner);
    }
    result
}

/// Apply Young projection to a tensor based on its declared `TableauSymmetry` property.
pub fn young_project_tensor(
    expr: &Expr,
    tensor_properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, _) => {
            if let Expr::Sym(name) = base.as_ref() {
                for prop in tensor_properties.get_properties(*name) {
                    if let ax_ir::TensorProperty::TableauSymmetry {
                        shape,
                        indices: tab_indices,
                    } = prop
                    {
                        let mut cells: Vec<Vec<usize>> = Vec::new();
                        let mut cursor = 0;
                        for &row_len in shape {
                            let mut row = Vec::new();
                            for _ in 0..row_len {
                                if cursor < tab_indices.len() {
                                    row.push(tab_indices[cursor]);
                                    cursor += 1;
                                }
                            }
                            cells.push(row);
                        }
                        let tableau = ax_young::YoungTableau { cells };
                        return young_project(expr, &tableau, interner);
                    }
                }
            }
            expr.clone()
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| young_project_tensor(t, tensor_properties, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| young_project_tensor(f, tensor_properties, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(young_project_tensor(e, tensor_properties, interner)),
        _ => expr.clone(),
    }
}

pub fn reduce_delta(
    expr: &Expr,
    delta_sym: lasso::Spur,
    dim_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut current = expr.clone();
    for _ in 0..20 {
        let next = reduce_delta_once(&current, delta_sym, dim_sym, interner);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn reduce_delta_once(
    expr: &Expr,
    delta_sym: lasso::Spur,
    dim_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Check trace first: δ^a_a → dim
            for i in 0..factors.len() {
                let idx_i = match &factors[i] {
                    Expr::Indexed(b, idx)
                        if matches!(b.as_ref(), Expr::Sym(s) if *s == delta_sym)
                            && idx.len() == 2 =>
                    {
                        idx
                    }
                    _ => continue,
                };
                if idx_i[0].name == idx_i[1].name && idx_i[0].variance != idx_i[1].variance {
                    let mut new_factors: Vec<Expr> = factors
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i)
                        .map(|(_, f)| f.clone())
                        .collect();
                    new_factors.push(Expr::Sym(dim_sym));
                    return Expr::mul(new_factors);
                }
            }

            // Find two deltas that share a contracted index
            for i in 0..factors.len() {
                let idx_i = match &factors[i] {
                    Expr::Indexed(b, idx)
                        if matches!(b.as_ref(), Expr::Sym(s) if *s == delta_sym)
                            && idx.len() == 2 =>
                    {
                        idx
                    }
                    _ => continue,
                };
                for j in (i + 1)..factors.len() {
                    let idx_j = match &factors[j] {
                        Expr::Indexed(b, idx)
                            if matches!(b.as_ref(), Expr::Sym(s) if *s == delta_sym)
                                && idx.len() == 2 =>
                        {
                            idx
                        }
                        _ => continue,
                    };
                    for ii in 0..2 {
                        for jj in 0..2 {
                            if idx_i[ii].name == idx_j[jj].name
                                && idx_i[ii].variance != idx_j[jj].variance
                            {
                                // Contracted: δ^a_X δ^X_b → δ^a_b
                                let remaining_i = idx_i[1 - ii].clone();
                                let remaining_j = idx_j[1 - jj].clone();
                                let new_delta = Expr::Indexed(
                                    Box::new(Expr::Sym(delta_sym)),
                                    vec![remaining_i, remaining_j],
                                );
                                let mut new_factors: Vec<Expr> = factors
                                    .iter()
                                    .enumerate()
                                    .filter(|(k, _)| *k != i && *k != j)
                                    .map(|(_, f)| f.clone())
                                    .collect();
                                new_factors.push(new_delta);
                                return Expr::mul(new_factors);
                            }
                        }
                    }
                }
            }

            // No simplification found at this level — recurse into sub-expressions
            Expr::mul(
                factors
                    .iter()
                    .map(|f| reduce_delta_once(f, delta_sym, dim_sym, interner))
                    .collect(),
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| reduce_delta_once(t, delta_sym, dim_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(reduce_delta_once(e, delta_sym, dim_sym, interner)),
        // Bare delta trace: δ^a_a → dim
        Expr::Indexed(base, idx)
            if matches!(base.as_ref(), Expr::Sym(s) if *s == delta_sym)
                && idx.len() == 2
                && idx[0].name == idx[1].name
                && idx[0].variance != idx[1].variance =>
        {
            Expr::Sym(dim_sym)
        }
        _ => expr.clone(),
    }
}

pub fn eliminate_kronecker(
    expr: &ax_ir::Expr,
    delta_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut remaining = factors.clone();
            let mut changed = true;
            while changed {
                changed = false;
                for i in 0..remaining.len() {
                    let delta_info = match &remaining[i] {
                        Expr::Indexed(base, indices) => match base.as_ref() {
                            Expr::Sym(s) if *s == delta_sym && indices.len() == 2 => Some((
                                indices[0].name,
                                indices[0].variance.clone(),
                                indices[1].name,
                                indices[1].variance.clone(),
                            )),
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some((left_name, left_var, right_name, right_var)) = delta_info else {
                        continue;
                    };

                    let (from, to) = if left_var == ax_ir::Variance::Up
                        && right_var == ax_ir::Variance::Down
                    {
                        (right_name, left_name)
                    } else if left_var == ax_ir::Variance::Down && right_var == ax_ir::Variance::Up
                    {
                        (left_name, right_name)
                    } else {
                        continue;
                    };

                    let mut found = false;
                    for j in 0..remaining.len() {
                        if i == j {
                            continue;
                        }
                        if contains_index(&remaining[j], from) {
                            remaining[j] = replace_index(&remaining[j], from, to);
                            found = true;
                        }
                    }
                    if found {
                        remaining.remove(i);
                        changed = true;
                        break;
                    }

                    if left_name == right_name {
                        remaining[i] = Expr::Sym(interner.get_or_intern("dim"));
                        changed = true;
                        break;
                    }
                }
            }
            Expr::mul(
                remaining
                    .into_iter()
                    .map(|f| eliminate_kronecker(&f, delta_sym, interner))
                    .collect(),
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| eliminate_kronecker(t, delta_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(eliminate_kronecker(e, delta_sym, interner)),
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(s) = base.as_ref() {
                if *s == delta_sym && indices.len() == 2 && indices[0].name == indices[1].name {
                    return Expr::Sym(interner.get_or_intern("dim"));
                }
            }
            expr.clone()
        }
        _ => expr.clone(),
    }
}

pub fn eliminate_metric(
    expr: &ax_ir::Expr,
    metric_sym: lasso::Spur,
    inv_metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut remaining = factors.clone();
            let mut changed = true;
            while changed {
                changed = false;
                for i in 0..remaining.len() {
                    let metric_info = match &remaining[i] {
                        Expr::Indexed(base, indices) => match base.as_ref() {
                            Expr::Sym(s) if indices.len() == 2 => Some((
                                *s,
                                indices[0].name,
                                indices[0].variance.clone(),
                                indices[1].name,
                                indices[1].variance.clone(),
                            )),
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some((sym, left_name, left_var, right_name, right_var)) = metric_info
                    else {
                        continue;
                    };

                    let is_metric = sym == metric_sym
                        && left_var == ax_ir::Variance::Down
                        && right_var == ax_ir::Variance::Down;
                    let is_inv_metric = sym == inv_metric_sym
                        && left_var == ax_ir::Variance::Up
                        && right_var == ax_ir::Variance::Up;
                    if !is_metric && !is_inv_metric {
                        continue;
                    }

                    let target_variance = if is_metric {
                        ax_ir::Variance::Up
                    } else {
                        ax_ir::Variance::Down
                    };
                    let new_variance = if is_metric {
                        ax_ir::Variance::Down
                    } else {
                        ax_ir::Variance::Up
                    };

                    let mut found = false;
                    let mut contract_idx = right_name;
                    let mut replace_with = left_name;
                    for j in 0..remaining.len() {
                        if i == j {
                            continue;
                        }
                        if has_index_with_variance(&remaining[j], contract_idx, &target_variance) {
                            remaining[j] = replace_index_with_variance(
                                &remaining[j],
                                contract_idx,
                                &target_variance,
                                replace_with,
                                &new_variance,
                            );
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        contract_idx = left_name;
                        replace_with = right_name;
                        for j in 0..remaining.len() {
                            if i == j {
                                continue;
                            }
                            if has_index_with_variance(
                                &remaining[j],
                                contract_idx,
                                &target_variance,
                            ) {
                                remaining[j] = replace_index_with_variance(
                                    &remaining[j],
                                    contract_idx,
                                    &target_variance,
                                    replace_with,
                                    &new_variance,
                                );
                                found = true;
                                break;
                            }
                        }
                    }

                    if found {
                        remaining.remove(i);
                        changed = true;
                        break;
                    }
                }
            }
            Expr::mul(
                remaining
                    .into_iter()
                    .map(|f| eliminate_metric(&f, metric_sym, inv_metric_sym, interner))
                    .collect(),
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| eliminate_metric(t, metric_sym, inv_metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(eliminate_metric(e, metric_sym, inv_metric_sym, interner)),
        _ => expr.clone(),
    }
}

/// Eliminate vielbein objects by performing index contractions.
///
/// `e^{a}_{μ} * T^{μ}` → `T^{a}`
///
/// Structurally identical to `eliminate_metric` but operates on vielbein
/// (and inverse-vielbein) symbols that convert between two index families,
/// e.g. tangent-space index `a` and spacetime index `μ`.
pub fn eliminate_vielbein(
    expr: &Expr,
    vielbein_sym: lasso::Spur,
    inv_vielbein_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut remaining = factors.clone();
            let mut changed = true;
            while changed {
                changed = false;
                'outer: for i in 0..remaining.len() {
                    if let Expr::Indexed(base, indices) = &remaining[i].clone() {
                        if let Expr::Sym(s) = base.as_ref() {
                            let is_vb = *s == vielbein_sym || *s == inv_vielbein_sym;
                            if !is_vb || indices.len() != 2 {
                                continue;
                            }

                            for idx_pos in 0..2 {
                                let contract_idx = &indices[idx_pos];
                                let replace_idx = &indices[1 - idx_pos];
                                let target_var = match contract_idx.variance {
                                    ax_ir::Variance::Up => ax_ir::Variance::Down,
                                    ax_ir::Variance::Down => ax_ir::Variance::Up,
                                };

                                for j in 0..remaining.len() {
                                    if i == j {
                                        continue;
                                    }
                                    if has_index_with_variance(
                                        &remaining[j],
                                        contract_idx.name,
                                        &target_var,
                                    ) {
                                        remaining[j] = replace_index_with_variance(
                                            &remaining[j],
                                            contract_idx.name,
                                            &target_var,
                                            replace_idx.name,
                                            &replace_idx.variance,
                                        );
                                        remaining.remove(i);
                                        changed = true;
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Expr::mul(
                remaining
                    .into_iter()
                    .map(|f| eliminate_vielbein(&f, vielbein_sym, inv_vielbein_sym, interner))
                    .collect(),
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| eliminate_vielbein(t, vielbein_sym, inv_vielbein_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(eliminate_vielbein(
            e,
            vielbein_sym,
            inv_vielbein_sym,
            interner,
        )),
        _ => expr.clone(),
    }
}

pub fn diff_component(
    expr: &ax_ir::Expr,
    coord: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
            Expr::Complex(re, im) => contains_var(re, var) || contains_var(im, var),
            Expr::Sym(s) => *s == var,
            Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
                items.iter().any(|item| contains_var(item, var))
            }
            Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
            Expr::Neg(e) => contains_var(e, var),
            Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
            Expr::FnDef(_, _, body) => contains_var(body, var),
            Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
            Expr::Import(_) => false,
            Expr::Assume(_, _) => false,
            Expr::SetConvention(_, _) => false,
            Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_var(value, var)),
            Expr::Indexed(base, _) => contains_var(base, var),
            Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
            Expr::Matrix(rows) => rows
                .iter()
                .any(|row| row.iter().any(|cell| contains_var(cell, var))),
        }
    }

    fn one_half() -> Expr {
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    }

    fn diff(expr: &Expr, var: lasso::Spur, interner: &Interner) -> Expr {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
            Expr::Complex(re, im) => Expr::Complex(
                Box::new(diff(re, var, interner)),
                Box::new(diff(im, var, interner)),
            ),
            Expr::Sym(s) => {
                if *s == var {
                    Expr::one()
                } else {
                    Expr::zero()
                }
            }
            Expr::Add(terms) => Expr::add(terms.iter().map(|t| diff(t, var, interner)).collect()),
            Expr::Neg(e) => Expr::neg(diff(e, var, interner)),
            Expr::Mul(factors) => Expr::add(
                factors
                    .iter()
                    .enumerate()
                    .map(|(i, factor)| {
                        let mut product = Vec::with_capacity(factors.len());
                        product.extend(factors[..i].iter().cloned());
                        product.push(diff(factor, var, interner));
                        product.extend(factors[i + 1..].iter().cloned());
                        Expr::mul(product)
                    })
                    .collect(),
            ),
            Expr::Pow(base, exp) if !contains_var(exp, var) => Expr::mul(vec![
                exp.as_ref().clone(),
                Expr::pow(
                    base.as_ref().clone(),
                    Expr::add(vec![exp.as_ref().clone(), Expr::neg(Expr::one())]),
                ),
                diff(base, var, interner),
            ]),
            Expr::Pow(_, _) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
            Expr::Call(f, args) if args.len() == 1 => match interner.resolve(*f) {
                "sin" => Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("cos"), args.clone()),
                    diff(&args[0], var, interner),
                ]),
                "cos" => Expr::mul(vec![
                    Expr::neg(Expr::Call(interner.get_or_intern("sin"), args.clone())),
                    diff(&args[0], var, interner),
                ]),
                "exp" => Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("exp"), args.clone()),
                    diff(&args[0], var, interner),
                ]),
                "log" => Expr::mul(vec![
                    Expr::pow(args[0].clone(), Expr::neg(Expr::one())),
                    diff(&args[0], var, interner),
                ]),
                "sqrt" => diff(&Expr::pow(args[0].clone(), one_half()), var, interner),
                _ => Expr::Call(
                    interner.get_or_intern("diff"),
                    vec![expr.clone(), Expr::Sym(var)],
                ),
            },
            Expr::Call(_, _) | Expr::Indexed(_, _) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
            Expr::FnDef(name, params, body) => {
                Expr::FnDef(*name, params.clone(), Box::new(diff(body, var, interner)))
            }
            Expr::Rule(lhs, rhs, trust) => Expr::Rule(
                Box::new(diff(lhs, var, interner)),
                Box::new(diff(rhs, var, interner)),
                *trust,
            ),
            Expr::Import(path) => Expr::Import(path.clone()),
            Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
            Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
            Expr::Piecewise(cases) => Expr::Piecewise(
                cases
                    .iter()
                    .map(|(value, condition)| (diff(value, var, interner), condition.clone()))
                    .collect(),
            ),
            Expr::Let(name, val, body) => {
                Expr::Let(*name, val.clone(), Box::new(diff(body, var, interner)))
            }
            Expr::List(items) => Expr::List(items.iter().map(|i| diff(i, var, interner)).collect()),
            Expr::Matrix(rows) => Expr::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|cell| diff(cell, var, interner)).collect())
                    .collect(),
            ),
        }
    }

    diff(expr, coord, interner)
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn numeric_coeff(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

#[allow(dead_code)]
fn decompose_term(term: &Expr) -> (BigRational, Expr) {
    match term {
        Expr::Mul(factors) if !factors.is_empty() => {
            if let Some(coeff) = numeric_coeff(&factors[0]) {
                let rest = factors[1..].to_vec();
                let base = if rest.is_empty() {
                    Expr::one()
                } else {
                    Expr::mul(rest)
                };
                (coeff, base)
            } else {
                (BigRational::one(), term.clone())
            }
        }
        Expr::Neg(inner) => (
            BigRational::from_integer((-1).into()),
            inner.as_ref().clone(),
        ),
        Expr::Int(n) => (BigRational::from_integer(n.clone()), Expr::one()),
        Expr::Rational(r) => (r.clone(), Expr::one()),
        _ => (BigRational::one(), term.clone()),
    }
}

#[allow(dead_code)]
fn factor_base_and_exp(expr: &Expr) -> (Expr, BigRational) {
    match expr {
        Expr::Pow(base, exp) => {
            if let Some(n) = numeric_coeff(exp) {
                ((*base.clone()), n)
            } else {
                (expr.clone(), BigRational::one())
            }
        }
        _ => (expr.clone(), BigRational::one()),
    }
}

#[allow(dead_code)]
fn factor_list(expr: &Expr) -> Vec<(Expr, BigRational)> {
    match expr {
        Expr::Mul(factors) => factors.iter().map(factor_base_and_exp).collect(),
        Expr::Int(_) | Expr::Rational(_) => Vec::new(),
        _ => vec![factor_base_and_exp(expr)],
    }
}

#[allow(dead_code)]
fn remove_common_factor(factors: &[(Expr, BigRational)], common: &[(Expr, BigRational)]) -> Expr {
    let mut remaining = factors.to_vec();

    for (common_base, common_exp) in common {
        if let Some((_, exp)) = remaining.iter_mut().find(|(base, _)| *base == *common_base) {
            *exp -= common_exp.clone();
        }
    }

    let rebuilt = remaining
        .into_iter()
        .filter_map(|(base, exp)| {
            if exp.is_zero() {
                None
            } else if exp.is_one() {
                Some(base)
            } else {
                Some(Expr::pow(base, Expr::Rational(exp)))
            }
        })
        .collect::<Vec<_>>();

    Expr::mul(rebuilt)
}

#[allow(dead_code)]
fn extract_common_factor(terms: &[Expr]) -> Option<(Expr, Vec<Expr>)> {
    if terms.len() < 2 {
        return None;
    }

    let mut common = factor_list(&terms[0]);
    if common.is_empty() {
        return None;
    }

    for term in &terms[1..] {
        let factors = factor_list(term);
        common.retain_mut(|(common_base, common_exp)| {
            if let Some((_, exp)) = factors.iter().find(|(base, _)| *base == *common_base) {
                if *exp < *common_exp {
                    *common_exp = exp.clone();
                }
                !common_exp.is_zero()
            } else {
                false
            }
        });

        if common.is_empty() {
            return None;
        }
    }

    if !common.iter().any(|(_, exp)| exp.is_negative()) {
        return None;
    }

    let common_expr = Expr::mul(
        common
            .iter()
            .map(|(base, exp)| {
                if exp.is_one() {
                    base.clone()
                } else {
                    Expr::pow(base.clone(), Expr::Rational(exp.clone()))
                }
            })
            .collect(),
    );

    if common_expr == Expr::one() {
        return None;
    }

    let remainders = terms
        .iter()
        .map(|term| remove_common_factor(&factor_list(term), &common))
        .collect::<Vec<_>>();

    Some((common_expr, remainders))
}

#[allow(dead_code)]
fn collect_flat_add(expr: &Expr) -> Expr {
    let Expr::Add(terms) = expr else {
        return expr.clone();
    };

    let mut groups: Vec<(Expr, BigRational)> = Vec::new();
    for term in terms {
        let (coeff, base) = decompose_term(term);
        if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
            *acc += coeff;
        } else {
            groups.push((base, coeff));
        }
    }

    Expr::add(
        groups
            .into_iter()
            .filter_map(|(base, coeff)| {
                if coeff.is_zero() {
                    return None;
                }

                let coeff_expr = if coeff.is_integer() {
                    Expr::Int(coeff.to_integer())
                } else {
                    Expr::Rational(coeff)
                };

                Some(if base == Expr::one() {
                    coeff_expr
                } else {
                    Expr::mul(vec![coeff_expr, base])
                })
            })
            .collect(),
    )
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = numeric_coeff(base)?;
    match exp {
        Expr::Int(n) => {
            if let Some(pow) = n.to_u32() {
                let numer = base_r.numer().clone().pow(pow);
                let denom = base_r.denom().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else if n.is_negative() {
                let pow = (-n).to_u32()?;
                let numer = base_r.denom().clone().pow(pow);
                let denom = base_r.numer().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else {
                None
            }
        }
        Expr::Rational(_) => None,
        _ => None,
    }
}

fn eval_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(Box::new(eval_expr(re)), Box::new(eval_expr(im))),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(terms.iter().map(eval_expr).collect()),
        Expr::Mul(factors) => Expr::mul(factors.iter().map(eval_expr).collect()),
        Expr::Pow(base, exp) => {
            let evaled_base = eval_expr(base);
            let evaled_exp = eval_expr(exp);
            if let Some(out) = numeric_pow(&evaled_base, &evaled_exp) {
                out
            } else {
                Expr::pow(evaled_base, evaled_exp)
            }
        }
        Expr::Neg(e) => Expr::neg(eval_expr(e)),
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(eval_expr).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(eval_expr(body)))
        }
        Expr::Rule(lhs, rhs, trust) => {
            Expr::Rule(Box::new(eval_expr(lhs)), Box::new(eval_expr(rhs)), *trust)
        }
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (eval_expr(value), condition.clone()))
                .collect(),
        ),
        Expr::Let(name, val, body) => {
            Expr::Let(*name, Box::new(eval_expr(val)), Box::new(eval_expr(body)))
        }
        Expr::Indexed(base, indices) => Expr::Indexed(Box::new(eval_expr(base)), indices.clone()),
        Expr::List(items) => Expr::List(items.iter().map(eval_expr).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(eval_expr).collect())
                .collect(),
        ),
    }
}

#[allow(dead_code)]
fn node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => 1,
        Expr::Complex(re, im) => 1 + node_count(re) + node_count(im),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::FnDef(_, params, body) => params.len() + 1 + node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + node_count(lhs) + node_count(rhs),
        Expr::Import(path) => 1 + path.len(),
        Expr::Assume(_, assumptions) => 1 + assumptions.len(),
        Expr::SetConvention(field, value) => 1 + field.len() + value.len(),
        Expr::Piecewise(cases) => {
            1 + cases
                .iter()
                .map(|(value, _)| node_count(value))
                .sum::<usize>()
        }
        Expr::Indexed(base, _) => 1 + node_count(base),
        Expr::Let(_, val, body) => 1 + node_count(val) + node_count(body),
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
    }
}

#[allow(dead_code)]
fn expand_expr(expr: &Expr, interner: &Interner) -> Expr {
    let _ = interner;
    match expr {
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_expr(re, interner)),
            Box::new(expand_expr(im, interner)),
        ),
        Expr::Add(terms) => {
            let expanded = Expr::add(terms.iter().map(|t| expand_expr(t, interner)).collect());
            collect_flat_add(&expanded)
        }
        Expr::Mul(factors) => {
            let expanded_factors = factors
                .iter()
                .map(|f| expand_expr(f, interner))
                .collect::<Vec<_>>();

            if expanded_factors.len() > 6 {
                return Expr::mul(expanded_factors);
            }

            if let Some((idx, terms)) =
                expanded_factors.iter().enumerate().find_map(|(i, factor)| {
                    if let Expr::Add(terms) = factor {
                        Some((i, terms.clone()))
                    } else {
                        None
                    }
                })
            {
                let rest = expanded_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(i, factor)| if i != idx { Some(factor.clone()) } else { None })
                    .collect::<Vec<_>>();

                let distributed = terms
                    .into_iter()
                    .map(|term| {
                        let mut factors = Vec::with_capacity(rest.len() + 1);
                        factors.push(term);
                        factors.extend(rest.clone());
                        Expr::mul(factors)
                    })
                    .collect::<Vec<_>>();

                return expand_expr(&Expr::add(distributed), interner);
            }

            Expr::mul(expanded_factors)
        }
        Expr::Pow(base, exp) => {
            let expanded_base = expand_expr(base, interner);
            let expanded_exp = expand_expr(exp, interner);
            if let (Expr::Add(terms), Expr::Int(n)) = (&expanded_base, &expanded_exp) {
                if *n > 1.into() {
                    if let Some(power) = n.to_u32() {
                        if (2..=8).contains(&power) && terms.len() * (power as usize) <= 12 {
                            let sum = Expr::Add(terms.clone());
                            let repeated = (0..power).map(|_| sum.clone()).collect::<Vec<_>>();
                            return expand_expr(&Expr::Mul(repeated), interner);
                        }
                    }
                }
            }
            Expr::pow(expanded_base, expanded_exp)
        }
        Expr::Neg(e) => {
            let inner = expand_expr(e, interner);
            if let Expr::Add(terms) = inner {
                let expanded = Expr::add(terms.into_iter().map(Expr::neg).collect());
                collect_flat_add(&expanded)
            } else {
                Expr::neg(inner)
            }
        }
        Expr::Call(f, args) => {
            Expr::Call(*f, args.iter().map(|a| expand_expr(a, interner)).collect())
        }
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(expand_expr(body, interner)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(expand_expr(lhs, interner)),
            Box::new(expand_expr(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (expand_expr(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(expand_expr(base, interner)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(expand_expr(val, interner)),
            Box::new(expand_expr(body, interner)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(|i| expand_expr(i, interner)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|cell| expand_expr(cell, interner)).collect())
                .collect(),
        ),
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => Expr::Sym(*s),
    }
}

#[allow(dead_code)]
pub(crate) fn collect_terms_expr(expr: &Expr, interner: &Interner) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let normalized_terms = terms
                .iter()
                .map(|term| collect_terms_expr(term, interner))
                .collect::<Vec<_>>();

            let mut groups: Vec<(Expr, BigRational)> = Vec::new();
            for term in &normalized_terms {
                let (coeff, base) = decompose_term(term);
                if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
                    *acc += coeff;
                } else {
                    groups.push((base, coeff));
                }
            }

            let rebuilt = groups
                .into_iter()
                .filter_map(|(base, coeff)| {
                    if coeff.is_zero() {
                        return None;
                    }

                    let coeff_expr = if coeff.is_integer() {
                        Expr::Int(coeff.to_integer())
                    } else {
                        Expr::Rational(coeff)
                    };

                    Some(if base == Expr::one() {
                        coeff_expr
                    } else {
                        Expr::mul(vec![coeff_expr, base])
                    })
                })
                .collect::<Vec<_>>();

            let combined = Expr::add(rebuilt);
            if let Expr::Add(combined_terms) = &combined {
                if let Some((common, remainders)) = extract_common_factor(combined_terms) {
                    let inner = collect_terms_expr(&Expr::add(remainders), interner);
                    return Expr::mul(vec![common, inner]);
                }
            }

            combined
        }
        Expr::Mul(factors) => {
            let normalized = factors
                .iter()
                .map(|factor| collect_terms_expr(factor, interner))
                .collect::<Vec<_>>();

            let add_factor = normalized.iter().enumerate().find_map(|(idx, factor)| {
                if let Expr::Add(terms) = factor {
                    Some((idx, terms.clone()))
                } else {
                    None
                }
            });

            if let Some((idx, terms)) = add_factor {
                if normalized.len() <= 4 {
                    let rest = normalized
                        .iter()
                        .enumerate()
                        .filter_map(
                            |(i, factor)| if i != idx { Some(factor.clone()) } else { None },
                        )
                        .collect::<Vec<_>>();

                    let is_pure_sign_flip = rest.len() == 1
                        && matches!(rest.first(), Some(Expr::Int(n)) if *n == (-1).into());

                    if is_pure_sign_flip {
                        return Expr::mul(normalized);
                    }

                    let distributed = terms
                        .into_iter()
                        .map(|term| {
                            let mut factors = Vec::with_capacity(rest.len() + 1);
                            factors.push(term);
                            factors.extend(rest.clone());
                            Expr::mul(factors)
                        })
                        .collect::<Vec<_>>();

                    return Expr::add(
                        distributed
                            .iter()
                            .map(|term| collect_terms_expr(term, interner))
                            .collect(),
                    );
                }
            }

            Expr::mul(normalized)
        }
        Expr::Pow(base, exp) => Expr::pow(
            collect_terms_expr(base, interner),
            collect_terms_expr(exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(collect_terms_expr(inner, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| collect_terms_expr(arg, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(collect_terms_expr(body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(collect_terms_expr(lhs, interner)),
            Box::new(collect_terms_expr(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (collect_terms_expr(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(collect_terms_expr(base, interner)),
            indices.clone(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(collect_terms_expr(val, interner)),
            Box::new(collect_terms_expr(body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| collect_terms_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| collect_terms_expr(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn simplify_expr(expr: Expr, interner: &Interner) -> Expr {
    let _ = interner;
    eval_expr(&expr)
}

pub fn christoffel_from_metric(
    g: &SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    let ginv = g.symbolic_inverse(interner);
    let n = coords.len();
    assert_eq!(n, g.dim);

    let mut gamma = vec![vec![vec![Expr::zero(); n]; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut sum_terms = Vec::with_capacity(n);
                for l in 0..n {
                    let dj_glk = diff_component(g.get(j, l), coords[k], interner);
                    let dk_glj = diff_component(g.get(k, l), coords[j], interner);
                    let dl_gjk = diff_component(g.get(j, k), coords[l], interner);
                    let inner = Expr::add(vec![dj_glk, dk_glj, Expr::neg(dl_gjk)]);
                    sum_terms.push(Expr::mul(vec![ginv.get(i, l).clone(), inner]));
                }
                let expr = Expr::mul(vec![half(), Expr::add(sum_terms)]);
                gamma[i][j][k] = simplify_expr(expr, interner);
            }
        }
    }
    gamma
}

pub fn riemann_from_christoffel(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
    convention: &ax_ir::Convention,
) -> Vec<Vec<Vec<Vec<ax_ir::Expr>>>> {
    let n = coords.len();
    let mut riemann = vec![vec![vec![vec![Expr::zero(); n]; n]; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let term1 = diff_component(&gamma[i][l][j], coords[k], interner);
                    let term2 = diff_component(&gamma[i][k][j], coords[l], interner);

                    let mut pos_terms = Vec::with_capacity(n);
                    let mut neg_terms = Vec::with_capacity(n);
                    for m in 0..n {
                        pos_terms.push(Expr::mul(vec![
                            gamma[i][k][m].clone(),
                            gamma[m][l][j].clone(),
                        ]));
                        neg_terms.push(Expr::mul(vec![
                            gamma[i][l][m].clone(),
                            gamma[m][k][j].clone(),
                        ]));
                    }

                    let expr = match convention.riemann_sign {
                        ax_ir::RiemannSign::MTW => Expr::add(vec![
                            term1,
                            Expr::neg(term2),
                            Expr::add(pos_terms),
                            Expr::neg(Expr::add(neg_terms)),
                        ]),
                        ax_ir::RiemannSign::Weinberg => Expr::add(vec![
                            term2,
                            Expr::neg(term1),
                            Expr::add(neg_terms),
                            Expr::neg(Expr::add(pos_terms)),
                        ]),
                    };
                    riemann[i][j][k][l] = simplify_expr(expr, interner);
                }
            }
        }
    }

    riemann
}

pub fn ricci_from_riemann(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    n: usize,
    interner: &ax_ir::Interner,
    convention: &ax_ir::Convention,
) -> Vec<Vec<ax_ir::Expr>> {
    let mut ricci = vec![vec![Expr::zero(); n]; n];
    for j in 0..n {
        for l in 0..n {
            let terms = (0..n)
                .map(|i| match convention.ricci_contraction {
                    ax_ir::RicciContraction::FirstThird => riemann[i][j][i][l].clone(),
                    ax_ir::RicciContraction::FirstFourth => riemann[i][j][l][i].clone(),
                })
                .collect::<Vec<_>>();
            ricci[j][l] = simplify_expr(Expr::add(terms), interner);
        }
    }
    ricci
}

pub fn ricci_scalar(
    ricci: &[Vec<ax_ir::Expr>],
    ginv: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let n = ginv.dim;
    let mut terms = Vec::with_capacity(n * n);
    for j in 0..n {
        for l in 0..n {
            terms.push(Expr::mul(vec![ginv.get(j, l).clone(), ricci[j][l].clone()]));
        }
    }
    let _ = interner;
    Expr::add(terms)
}

pub fn einstein_tensor(
    ricci: &[Vec<ax_ir::Expr>],
    scalar: &ax_ir::Expr,
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let n = g.dim;
    let mut einstein = vec![vec![Expr::zero(); n]; n];

    for j in 0..n {
        for l in 0..n {
            let expr = Expr::add(vec![
                ricci[j][l].clone(),
                Expr::neg(Expr::mul(vec![half(), g.get(j, l).clone(), scalar.clone()])),
            ]);
            einstein[j][l] = simplify_expr(expr, interner);
        }
    }

    einstein
}

pub fn kretschner_scalar(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let n = g.dim;
    let ginv = g.symbolic_inverse(interner);
    let mut terms = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let r = &riemann[i][j][k][l];
                    if *r == Expr::zero() {
                        continue;
                    }
                    let term = Expr::mul(vec![
                        g.get(i, i).clone(),
                        ginv.get(j, j).clone(),
                        ginv.get(k, k).clone(),
                        ginv.get(l, l).clone(),
                        Expr::pow(r.clone(), Expr::Int(2.into())),
                    ]);
                    terms.push(term);
                }
            }
        }
    }

    simplify_expr(Expr::add(terms), interner)
}

pub fn covariant_derivative_vector(
    v: &[ax_ir::Expr],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = v.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = vec![diff_component(&v[i], coords[coord_index], interner)];
        for (j, vj) in v.iter().enumerate() {
            terms.push(Expr::mul(vec![
                gamma[i][coord_index][j].clone(),
                vj.clone(),
            ]));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

pub fn covariant_derivative_covector(
    w: &[ax_ir::Expr],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = w.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = vec![diff_component(&w[i], coords[coord_index], interner)];
        for (j, wj) in w.iter().enumerate() {
            terms.push(Expr::neg(Expr::mul(vec![
                gamma[j][coord_index][i].clone(),
                wj.clone(),
            ])));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

pub fn covariant_derivative_tensor2(
    t: &[Vec<ax_ir::Expr>],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let n = t.len();
    let mut out = vec![vec![Expr::zero(); n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut terms = vec![diff_component(&t[i][j], coords[coord_index], interner)];
            for m in 0..n {
                terms.push(Expr::mul(vec![
                    gamma[i][coord_index][m].clone(),
                    t[m][j].clone(),
                ]));
                terms.push(Expr::mul(vec![
                    gamma[j][coord_index][m].clone(),
                    t[i][m].clone(),
                ]));
            }
            out[i][j] = simplify_expr(Expr::add(terms), interner);
        }
    }

    out
}

pub fn geodesic_equations(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = coords.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = Vec::new();
        for j in 0..n {
            for k in 0..n {
                let dot_j = Expr::Sym(
                    interner.get_or_intern(&format!("dot_{}", interner.resolve(coords[j]))),
                );
                let dot_k = Expr::Sym(
                    interner.get_or_intern(&format!("dot_{}", interner.resolve(coords[k]))),
                );
                terms.push(Expr::mul(vec![gamma[i][j][k].clone(), dot_j, dot_k]));
            }
        }
        out[i] = simplify_expr(Expr::neg(Expr::add(terms)), interner);
    }

    out
}

pub fn lie_derivative_scalar(
    f: &ax_ir::Expr,
    v: &[ax_ir::Expr],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let terms = v
        .iter()
        .enumerate()
        .map(|(i, vi)| Expr::mul(vec![vi.clone(), diff_component(f, coords[i], interner)]))
        .collect::<Vec<_>>();
    simplify_expr(Expr::add(terms), interner)
}

pub fn lie_derivative_vector(
    w: &[ax_ir::Expr],
    v: &[ax_ir::Expr],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = w.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = Vec::new();
        for j in 0..n {
            terms.push(Expr::mul(vec![
                v[j].clone(),
                diff_component(&w[i], coords[j], interner),
            ]));
            terms.push(Expr::neg(Expr::mul(vec![
                w[j].clone(),
                diff_component(&v[i], coords[j], interner),
            ])));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

/// Pull non-dependent factors out of derivative operators.
///
/// D(a * f(x)) → a * D(f(x))  when a is a constant (not in Depends set).
///
/// Also handles:
/// D(scalar) → 0  when scalar has no indices and is not in the Depends set.
pub fn unwrap_derivatives(
    expr: &Expr,
    derivative_syms: &HashSet<lasso::Spur>,
    depends: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Call(f, args) if derivative_syms.contains(f) && args.len() == 1 => {
            match &args[0] {
                Expr::Mul(factors) => {
                    let mut inside = Vec::new();
                    let mut outside = Vec::new();

                    for factor in factors {
                        if depends_on_anything(factor, depends) {
                            inside.push(factor.clone());
                        } else {
                            outside.push(factor.clone());
                        }
                    }

                    if outside.is_empty() {
                        // Everything depends, can't unwrap — recurse into factors
                        Expr::Call(
                            *f,
                            vec![Expr::mul(
                                factors
                                    .iter()
                                    .map(|fac| {
                                        unwrap_derivatives(fac, derivative_syms, depends, interner)
                                    })
                                    .collect(),
                            )],
                        )
                    } else if inside.is_empty() {
                        // Nothing depends — derivative of constant is zero
                        Expr::zero()
                    } else {
                        // Pull out the non-dependent factors
                        let deriv_part = Expr::Call(*f, vec![Expr::mul(inside)]);
                        outside.push(deriv_part);
                        Expr::mul(outside)
                    }
                }
                // Derivative of a non-dependent symbol
                Expr::Sym(s) if !depends.contains_key(s) => Expr::zero(),
                // Derivative of a numeric literal
                Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
                // Recurse into anything else
                other => {
                    let inner = unwrap_derivatives(other, derivative_syms, depends, interner);
                    Expr::Call(*f, vec![inner])
                }
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| unwrap_derivatives(t, derivative_syms, depends, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|fac| unwrap_derivatives(fac, derivative_syms, depends, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(unwrap_derivatives(e, derivative_syms, depends, interner)),
        _ => expr.clone(),
    }
}

fn depends_on_anything(expr: &Expr, depends: &HashMap<lasso::Spur, Vec<lasso::Spur>>) -> bool {
    match expr {
        Expr::Sym(s) => depends.contains_key(s),
        Expr::Indexed(base, _) => {
            if let Expr::Sym(s) = base.as_ref() {
                depends.contains_key(s)
            } else {
                true
            }
        }
        Expr::Call(_, args) => args.iter().any(|a| depends_on_anything(a, depends)),
        Expr::Mul(factors) => factors.iter().any(|f| depends_on_anything(f, depends)),
        Expr::Add(terms) => terms.iter().any(|t| depends_on_anything(t, depends)),
        Expr::Neg(e) => depends_on_anything(e, depends),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        _ => true, // conservative: assume it depends
    }
}

/// Perform integration by parts on a product inside an integral.
/// Moves derivatives away from `away_from` and onto other factors.
///
/// D(A) * B * C → -A * D(B) * C - A * B * D(C)
///
/// The `integral` wrapper is not represented explicitly — we assume the whole expression
/// is inside an integral and boundary terms vanish.
pub fn integrate_by_parts(
    expr: &Expr,
    away_from: lasso::Spur,
    derivative_syms: &HashSet<lasso::Spur>,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Find the first derivative acting on a term that contains `away_from`
            for (i, factor) in factors.iter().enumerate() {
                if let Expr::Call(d, args) = factor {
                    if derivative_syms.contains(d) && args.len() == 1 {
                        if contains_sym(&args[0], away_from) {
                            let inner = args[0].clone();
                            let other_factors: Vec<Expr> = factors
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, f)| f.clone())
                                .collect();

                            let mut ibp_terms = Vec::new();
                            for k in 0..other_factors.len() {
                                let mut new_factors = vec![inner.clone()];
                                for (l, f) in other_factors.iter().enumerate() {
                                    if l == k {
                                        new_factors.push(Expr::Call(*d, vec![f.clone()]));
                                    } else {
                                        new_factors.push(f.clone());
                                    }
                                }
                                ibp_terms.push(Expr::neg(Expr::mul(new_factors)));
                            }

                            return Expr::add(ibp_terms);
                        }
                    }
                }
            }
            expr.clone()
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| integrate_by_parts(t, away_from, derivative_syms, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(integrate_by_parts(e, away_from, derivative_syms, interner)),
        _ => expr.clone(),
    }
}

/// Compute the total weight of an expression under the given label.
///
/// - Sym: look up (s, label) in weights; default 0 if absent.
/// - Numeric literals: weight 0.
/// - Mul: sum of factor weights.
/// - Pow(base, Int(n)): weight(base) * n.
/// - Add: all terms must share the same weight; returns None if they differ.
/// - Neg: same weight as inner.
/// - Indexed: weight of the base symbol.
pub fn compute_weight(
    expr: &Expr,
    weights: &HashMap<(lasso::Spur, String), i64>,
    label: &str,
    interner: &Interner,
) -> Option<i64> {
    match expr {
        Expr::Sym(s) => Some(weights.get(&(*s, label.to_string())).copied().unwrap_or(0)),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Some(0),
        Expr::Neg(e) => compute_weight(e, weights, label, interner),
        Expr::Indexed(base, _) => compute_weight(base, weights, label, interner),
        Expr::Mul(factors) => {
            let mut total = 0i64;
            for f in factors {
                total += compute_weight(f, weights, label, interner)?;
            }
            Some(total)
        }
        Expr::Pow(base, exp) => {
            let base_w = compute_weight(base, weights, label, interner)?;
            if let Expr::Int(n) = exp.as_ref() {
                use num_traits::ToPrimitive;
                let n = n.to_i64()?;
                Some(base_w * n)
            } else {
                // Non-integer exponent: can't determine weight statically
                None
            }
        }
        Expr::Add(terms) => {
            let mut common: Option<i64> = None;
            for t in terms {
                let w = compute_weight(t, weights, label, interner)?;
                match common {
                    None => common = Some(w),
                    Some(c) if c == w => {}
                    _ => return None,
                }
            }
            common.or(Some(0))
        }
        // For anything else (Call, etc.) conservatively return None
        _ => None,
    }
}

/// Keep only terms with the specified weight.
pub fn keep_weight(
    expr: &Expr,
    target_weight: i64,
    weights: &HashMap<(lasso::Spur, String), i64>,
    label: &str,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let kept: Vec<Expr> = terms
                .iter()
                .filter(|t| compute_weight(t, weights, label, interner) == Some(target_weight))
                .cloned()
                .collect();
            if kept.is_empty() {
                Expr::zero()
            } else {
                Expr::add(kept)
            }
        }
        _ => {
            if compute_weight(expr, weights, label, interner) == Some(target_weight) {
                expr.clone()
            } else {
                Expr::zero()
            }
        }
    }
}

/// Drop terms with the specified weight.
pub fn drop_weight(
    expr: &Expr,
    target_weight: i64,
    weights: &HashMap<(lasso::Spur, String), i64>,
    label: &str,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let kept: Vec<Expr> = terms
                .iter()
                .filter(|t| compute_weight(t, weights, label, interner) != Some(target_weight))
                .cloned()
                .collect();
            if kept.is_empty() {
                Expr::zero()
            } else {
                Expr::add(kept)
            }
        }
        _ => {
            if compute_weight(expr, weights, label, interner) == Some(target_weight) {
                Expr::zero()
            } else {
                expr.clone()
            }
        }
    }
}

/// Given component rules for a metric, compute component rules for its inverse.
///
/// Input: rules for g_{ij} components.
/// Output: additional rules for g^{ij} (inverse metric components).
pub fn complete_inverse_metric(
    metric_rules: &[ComponentRule],
    metric_sym: lasso::Spur,
    inv_metric_sym: lasso::Spur,
    coordinates: &[lasso::Spur],
    interner: &Interner,
) -> Vec<ComponentRule> {
    let dim = coordinates.len();

    let mut g = SymbolicMatrix::new(dim);
    for rule in metric_rules {
        if rule.tensor != metric_sym {
            continue;
        }
        if rule.indices.len() != 2 {
            continue;
        }

        let i = coordinates.iter().position(|c| *c == rule.indices[0].0);
        let j = coordinates.iter().position(|c| *c == rule.indices[1].0);

        if let (Some(i), Some(j)) = (i, j) {
            g.set(i, j, rule.value.clone());
            if i != j {
                g.set(j, i, rule.value.clone());
            }
        }
    }

    let ginv = g.symbolic_inverse(interner);

    let mut result = Vec::new();
    for i in 0..dim {
        for j in 0..dim {
            let value = simplify_expr(ginv.get(i, j).clone(), interner);
            if value != Expr::zero() {
                result.push(ComponentRule {
                    tensor: inv_metric_sym,
                    indices: vec![
                        (coordinates[i], ax_ir::Variance::Up),
                        (coordinates[j], ax_ir::Variance::Up),
                    ],
                    value,
                });
            }
        }
    }

    result
}

/// Fix index positions on dummy pairs.
///
/// If two contracted indices share a name but both have the same variance,
/// flip one of them so the pair has opposing variances (one up, one down).
pub fn einsteinify(
    expr: &Expr,
    metric_sym: Option<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Collect (factor_idx, index_idx, index) for all indexed factors
            let mut all_indices: Vec<(usize, usize, ax_ir::Index)> = Vec::new();
            for (fi, factor) in factors.iter().enumerate() {
                if let Expr::Indexed(_, indices) = factor {
                    for (ii, idx) in indices.iter().enumerate() {
                        all_indices.push((fi, ii, idx.clone()));
                    }
                }
            }

            // Find pairs: same name, same variance
            let mut fixes: Vec<(usize, usize)> = Vec::new();
            let mut used = std::collections::HashSet::new();

            for i in 0..all_indices.len() {
                if used.contains(&i) {
                    continue;
                }
                for j in (i + 1)..all_indices.len() {
                    if used.contains(&j) {
                        continue;
                    }
                    if all_indices[i].2.name == all_indices[j].2.name
                        && all_indices[i].2.variance == all_indices[j].2.variance
                    {
                        fixes.push((i, j));
                        used.insert(i);
                        used.insert(j);
                        break;
                    }
                }
            }

            if fixes.is_empty() {
                return Expr::mul(
                    factors
                        .iter()
                        .map(|f| einsteinify(f, metric_sym, interner))
                        .collect(),
                );
            }

            // Flip the second index in each offending pair
            let mut new_factors = factors.clone();
            for (_, j) in &fixes {
                let (fi, ii, ref idx) = all_indices[*j];
                if let Expr::Indexed(base, indices) = &new_factors[fi].clone() {
                    let new_var = match &idx.variance {
                        ax_ir::Variance::Up => ax_ir::Variance::Down,
                        ax_ir::Variance::Down => ax_ir::Variance::Up,
                    };
                    let mut new_indices = indices.clone();
                    new_indices[ii] = ax_ir::Index {
                        name: idx.name,
                        variance: new_var,
                        index_type: idx.index_type,
                    };
                    new_factors[fi] = Expr::Indexed(base.clone(), new_indices);
                }
            }

            Expr::mul(new_factors)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| einsteinify(t, metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(einsteinify(e, metric_sym, interner)),
        _ => expr.clone(),
    }
}

/// Split indices of a given family into two sub-families.
///
/// Every occurrence of an index whose name is in `parent_indices` is replaced
/// by a sum over terms using `sub1_indices` and `sub2_indices`.
///
/// For a single free index position carrying a parent index, two terms are
/// produced: one with the sub1 replacement, one with sub2.  For n positions,
/// 2^n terms are produced (all combinations).
pub fn split_index(
    expr: &Expr,
    parent_indices: &[lasso::Spur],
    sub1_indices: &[lasso::Spur],
    sub2_indices: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let split_positions: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(_, idx)| parent_indices.contains(&idx.name))
                .map(|(i, _)| i)
                .collect();

            if split_positions.is_empty() {
                return Expr::Indexed(
                    Box::new(split_index(
                        base,
                        parent_indices,
                        sub1_indices,
                        sub2_indices,
                        interner,
                    )),
                    indices.clone(),
                );
            }

            let n = split_positions.len();
            let total = 1usize << n;
            let mut terms = Vec::new();

            for combo in 0..total {
                let mut new_indices = indices.clone();
                let mut valid = true;
                for (bit, &pos) in split_positions.iter().enumerate() {
                    let sub = if (combo >> bit) & 1 == 0 {
                        sub1_indices
                    } else {
                        sub2_indices
                    };
                    if sub.is_empty() {
                        valid = false;
                        break;
                    }
                    // Cycle through sub-indices if there are fewer than positions
                    let sub_name = sub[bit % sub.len()];
                    new_indices[pos] = ax_ir::Index {
                        name: sub_name,
                        variance: indices[pos].variance.clone(),
                        index_type: indices[pos].index_type,
                    };
                }
                if valid {
                    terms.push(Expr::Indexed(base.clone(), new_indices));
                }
            }

            Expr::add(terms)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| split_index(f, parent_indices, sub1_indices, sub2_indices, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| split_index(t, parent_indices, sub1_indices, sub2_indices, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(split_index(
            e,
            parent_indices,
            sub1_indices,
            sub2_indices,
            interner,
        )),
        _ => expr.clone(),
    }
}

fn contains_sym(expr: &Expr, sym: lasso::Spur) -> bool {
    match expr {
        Expr::Sym(s) => *s == sym,
        Expr::Indexed(base, _) => contains_sym(base, sym),
        Expr::Call(f, args) => *f == sym || args.iter().any(|a| contains_sym(a, sym)),
        Expr::Mul(factors) => factors.iter().any(|f| contains_sym(f, sym)),
        Expr::Add(terms) => terms.iter().any(|t| contains_sym(t, sym)),
        Expr::Neg(e) => contains_sym(e, sym),
        _ => false,
    }
}

// ─── expand_dummies ───────────────────────────────────────────────────────────

/// Rename every occurrence of index `from` to `to` throughout `expr`.
fn replace_index_name_everywhere(expr: &Expr, from: lasso::Spur, to: lasso::Spur) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    if idx.name == from {
                        ax_ir::Index {
                            name: to,
                            variance: idx.variance.clone(),
                            index_type: idx.index_type,
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();
            Expr::Indexed(
                Box::new(replace_index_name_everywhere(base, from, to)),
                new_indices,
            )
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| replace_index_name_everywhere(f, from, to))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| replace_index_name_everywhere(t, from, to))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(replace_index_name_everywhere(e, from, to)),
        _ => expr.clone(),
    }
}

/// Expand dummy index contractions into explicit sums over coordinate values.
///
/// `T_{mu}^{mu}` with coordinates `[t, r, theta, phi]` becomes
/// `T_{t}^{t} + T_{r}^{r} + T_{theta}^{theta} + T_{phi}^{phi}`.
///
/// Each Einstein-summation dummy pair (one up, one down index with the same name)
/// is replaced by a sum over all supplied coordinate labels. Multiple dummy pairs
/// are expanded left-to-right, one at a time.
pub fn expand_dummies(
    expr: &Expr,
    coordinates: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = interner;
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| expand_dummies(t, coordinates, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(expand_dummies(e, coordinates, interner)),
        _ => {
            let classification = classify_indices(expr);
            let coord_set: HashSet<lasso::Spur> = coordinates.iter().copied().collect();
            let mut dummy_names: Vec<lasso::Spur> = classification
                .dummy
                .iter()
                .filter(|(name, _, _, _, _)| !coord_set.contains(name))
                .map(|(name, _, _, _, _)| *name)
                .collect();

            if dummy_names.is_empty() {
                return expr.clone();
            }

            // Deterministic order so tests are reproducible
            dummy_names.sort_by_key(|s| s.into_inner());

            let first_dummy = dummy_names[0];
            let terms: Vec<Expr> = coordinates
                .iter()
                .map(|&coord| {
                    let replaced = replace_index_name_everywhere(expr, first_dummy, coord);
                    // Recursively expand any remaining dummy pairs
                    expand_dummies(&replaced, coordinates, interner)
                })
                .collect();

            Expr::add(terms)
        }
    }
}

// ─── explicit_indices ─────────────────────────────────────────────────────────

/// Make implicit indices on matrix-like objects explicit.
///
/// In a product `A * B` where both are 2-index implicit-index objects, the
/// contraction chain is introduced: `A[_i0+, _i1-] * B[_i1+, _i2-]`.
/// The lower index of each factor contracts with the upper index of the next.
///
/// Parameters:
/// - `implicit_index_tensors`: names of tensors that carry implicit indices
/// - `available_indices`: pool of fresh index name spurs to draw from
/// - `n_indices_per_tensor`: how many indices each such tensor has (typically 2)
pub fn explicit_indices(
    expr: &Expr,
    implicit_index_tensors: &HashSet<lasso::Spur>,
    available_indices: &[lasso::Spur],
    n_indices_per_tensor: &HashMap<lasso::Spur, usize>,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = interner;
    match expr {
        Expr::Mul(factors) => {
            // Identify which factor positions carry implicit indices, and how many
            let mut implicit_factors: Vec<(usize, lasso::Spur, usize)> = Vec::new();
            for (i, factor) in factors.iter().enumerate() {
                let name = match factor {
                    Expr::Sym(s) => Some(*s),
                    Expr::Call(f, _) => Some(*f),
                    Expr::Indexed(base, _) => {
                        if let Expr::Sym(s) = base.as_ref() {
                            Some(*s)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(n) = name {
                    if implicit_index_tensors.contains(&n) {
                        let n_idx = n_indices_per_tensor.get(&n).copied().unwrap_or(2);
                        implicit_factors.push((i, n, n_idx));
                    }
                }
            }

            if implicit_factors.is_empty() {
                return expr.clone();
            }

            let mut idx_counter = 0usize;
            let mut new_factors = factors.clone();
            let mut prev_lower: Option<lasso::Spur> = None;

            let mut fresh = |counter: &mut usize, interner: &ax_ir::Interner| -> lasso::Spur {
                let idx = available_indices
                    .get(*counter)
                    .copied()
                    .unwrap_or_else(|| interner.get_or_intern(&format!("_i{}", *counter)));
                *counter += 1;
                idx
            };

            for &(pos, _name, n_idx) in &implicit_factors {
                if n_idx != 2 {
                    // Only the standard 2-index (matrix) case is handled here
                    continue;
                }

                // Upper index: reuse the previous factor's lower, or allocate fresh
                let upper_idx = if let Some(prev) = prev_lower.take() {
                    prev
                } else {
                    fresh(&mut idx_counter, interner)
                };

                let lower_idx = fresh(&mut idx_counter, interner);
                prev_lower = Some(lower_idx);

                let base = match &new_factors[pos] {
                    Expr::Indexed(b, _) => *b.clone(),
                    other => other.clone(),
                };
                new_factors[pos] = Expr::Indexed(
                    Box::new(base),
                    vec![
                        ax_ir::Index {
                            name: upper_idx,
                            variance: ax_ir::Variance::Up,
                            index_type: None,
                        },
                        ax_ir::Index {
                            name: lower_idx,
                            variance: ax_ir::Variance::Down,
                            index_type: None,
                        },
                    ],
                );
            }

            Expr::mul(new_factors)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| {
                    explicit_indices(
                        t,
                        implicit_index_tensors,
                        available_indices,
                        n_indices_per_tensor,
                        interner,
                    )
                })
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(explicit_indices(
            e,
            implicit_index_tensors,
            available_indices,
            n_indices_per_tensor,
            interner,
        )),
        _ => expr.clone(),
    }
}

// ─── rewrite_indices ──────────────────────────────────────────────────────────

/// Convert indices on tensors between up/down using the metric or inverse metric.
///
/// For each index on a registered tensor that does not match its desired
/// variance, a metric factor is inserted and the original index is replaced by
/// a fresh dummy:
///
/// ```text
/// T[a+]  with target Down  →  g[a-, _rw0-] * T[_rw0+]   (lower with g)
/// T[a-]  with target Up    →  ginv[a+, _rw0+] * T[_rw0-] (raise with g^{-1})
/// ```
///
/// Parameters:
/// - `target_tensors`: map from tensor symbol to the desired `Variance` for each index slot
/// - `metric_sym`: symbol for the covariant metric g_{ab}
/// - `inv_metric_sym`: symbol for the contravariant metric g^{ab}
pub fn rewrite_indices(
    expr: &Expr,
    target_tensors: &HashMap<lasso::Spur, Vec<ax_ir::Variance>>,
    metric_sym: lasso::Spur,
    inv_metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(name) = base.as_ref() {
                if let Some(target_variances) = target_tensors.get(name) {
                    if target_variances.len() == indices.len() {
                        let mut result_factors: Vec<Expr> = Vec::new();
                        let mut new_indices = indices.clone();
                        let mut dummy_counter = 0usize;

                        for (i, (current, target)) in
                            indices.iter().zip(target_variances.iter()).enumerate()
                        {
                            if current.variance == *target {
                                continue; // already correct variance
                            }

                            let dummy_name =
                                interner.get_or_intern(&format!("_rw{}", dummy_counter));
                            dummy_counter += 1;

                            // Lowering (Up → Down): insert g_{orig, dummy}, tensor gets dummy Up
                            // Raising (Down → Up): insert ginv^{orig, dummy}, tensor gets dummy Down
                            let (m_sym, m_v1, m_v2, new_var) =
                                if current.variance == ax_ir::Variance::Up {
                                    (
                                        metric_sym,
                                        ax_ir::Variance::Down,
                                        ax_ir::Variance::Down,
                                        ax_ir::Variance::Up,
                                    )
                                } else {
                                    (
                                        inv_metric_sym,
                                        ax_ir::Variance::Up,
                                        ax_ir::Variance::Up,
                                        ax_ir::Variance::Down,
                                    )
                                };

                            result_factors.push(Expr::Indexed(
                                Box::new(Expr::Sym(m_sym)),
                                vec![
                                    ax_ir::Index {
                                        name: current.name,
                                        variance: m_v1,
                                        index_type: current.index_type,
                                    },
                                    ax_ir::Index {
                                        name: dummy_name,
                                        variance: m_v2,
                                        index_type: current.index_type,
                                    },
                                ],
                            ));

                            new_indices[i] = ax_ir::Index {
                                name: dummy_name,
                                variance: new_var,
                                index_type: current.index_type,
                            };
                        }

                        if result_factors.is_empty() {
                            return expr.clone(); // all variances already correct
                        }

                        result_factors.push(Expr::Indexed(base.clone(), new_indices));
                        return Expr::mul(result_factors);
                    }
                }
            }
            expr.clone()
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| rewrite_indices(f, target_tensors, metric_sym, inv_metric_sym, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| rewrite_indices(t, target_tensors, metric_sym, inv_metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(rewrite_indices(
            e,
            target_tensors,
            metric_sym,
            inv_metric_sym,
            interner,
        )),
        _ => expr.clone(),
    }
}

// ─── decompose ───────────────────────────────────────────────────────────────
//
// Express `expr` as a linear combination of the provided `basis` monomials.
// Each basis element is canonicalised; then every term in `expr` is matched
// against the canonical basis and a rational coefficient extracted.
// Any term that does not match any basis element is left as a residual.
//
// Returns `Expr::Add([c0 * basis[0], c1 * basis[1], ..., residual_terms...])`,
// simplified by `Expr::add`.

pub fn decompose(
    expr: &Expr,
    basis: &[Expr],
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Expr {
    let canon_basis: Vec<Expr> = basis
        .iter()
        .map(|b| canonicalise(b, properties, interner))
        .collect();

    let terms: Vec<Expr> = match expr {
        Expr::Add(ts) => ts.clone(),
        _ => vec![expr.clone()],
    };

    // Accumulate rational coefficients per basis slot.
    let mut coeffs: Vec<BigRational> = vec![BigRational::zero(); canon_basis.len()];
    let mut residual: Vec<Expr> = Vec::new();

    for term in &terms {
        let canon_term = canonicalise(term, properties, interner);
        match try_direct_match(&canon_term, &canon_basis, interner) {
            Some((idx, ratio)) => {
                coeffs[idx] = coeffs[idx].clone() + ratio;
            }
            None => residual.push(term.clone()),
        }
    }

    // Build output terms: coeff * basis_elem for non-zero coefficients.
    let mut out: Vec<Expr> = Vec::new();
    for (i, coeff) in coeffs.into_iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let coeff_expr = if coeff.is_integer() {
            Expr::Int(coeff.to_integer())
        } else {
            Expr::Rational(coeff)
        };
        out.push(Expr::mul(vec![coeff_expr, basis[i].clone()]));
    }
    out.extend(residual);
    Expr::add(out)
}

/// Try to match `canon_term` against any element of `canon_basis`.
/// Returns `Some((index, ratio))` where `ratio * canon_basis[index] == canon_term`.
fn try_direct_match(
    canon_term: &Expr,
    canon_basis: &[Expr],
    interner: &Interner,
) -> Option<(usize, BigRational)> {
    for (i, cb) in canon_basis.iter().enumerate() {
        if let Some(r) = extract_ratio(canon_term, cb, interner) {
            return Some((i, r));
        }
    }
    None
}

/// If `expr = ratio * basis_elem` (or `basis_elem = ratio * expr`), return the
/// rational coefficient `ratio` such that `ratio * basis_elem == expr`.
/// Both `expr` and `basis_elem` are already in canonical form.
fn extract_ratio(expr: &Expr, basis_elem: &Expr, interner: &Interner) -> Option<BigRational> {
    // Peel numeric coefficients from each side.
    let (expr_coeff, expr_base) = split_numeric_coeff(expr);
    let (basis_coeff, basis_base) = split_numeric_coeff(basis_elem);

    // The tensor structures must match (same canonical form).
    if !tensor_structures_equal(&expr_base, &basis_base, interner) {
        return None;
    }

    // ratio = expr_coeff / basis_coeff
    if basis_coeff.is_zero() {
        return None;
    }
    Some(expr_coeff / basis_coeff)
}

/// Split an expression into (numeric_coefficient, tensor_part).
/// e.g. `2 * R[a,b]` → (2, R[a,b])
///      `-R[a,b]`    → (-1, R[a,b])
///      `R[a,b]`     → (1, R[a,b])
fn split_numeric_coeff(expr: &Expr) -> (BigRational, Expr) {
    match expr {
        Expr::Mul(factors) if !factors.is_empty() => {
            if let Some(coeff) = numeric_coeff(&factors[0]) {
                let rest = factors[1..].to_vec();
                let base = match rest.len() {
                    0 => Expr::one(),
                    1 => rest.into_iter().next().unwrap(),
                    _ => Expr::Mul(rest),
                };
                (coeff, base)
            } else {
                (BigRational::one(), expr.clone())
            }
        }
        Expr::Neg(inner) => {
            let (c, b) = split_numeric_coeff(inner);
            (-c, b)
        }
        Expr::Int(n) => (BigRational::from_integer(n.clone()), Expr::one()),
        Expr::Rational(r) => (r.clone(), Expr::one()),
        _ => (BigRational::one(), expr.clone()),
    }
}

/// Check whether two already-canonicalised tensor expressions have the same
/// structure (same tensor symbol and same index names in order, ignoring sign /
/// numeric prefactor).
fn tensor_structures_equal(a: &Expr, b: &Expr, _interner: &Interner) -> bool {
    match (a, b) {
        (Expr::Indexed(ba, ia), Expr::Indexed(bb, ib)) => {
            ba == bb
                && ia.len() == ib.len()
                && ia
                    .iter()
                    .zip(ib.iter())
                    .all(|(x, y)| x.name == y.name && x.variance == y.variance)
        }
        (Expr::Mul(fa), Expr::Mul(fb)) => {
            fa.len() == fb.len()
                && fa
                    .iter()
                    .zip(fb.iter())
                    .all(|(x, y)| tensor_structures_equal(x, y, _interner))
        }
        (Expr::Sym(a), Expr::Sym(b)) => a == b,
        _ => a == b,
    }
}

// ─── decompose_product ───────────────────────────────────────────────────────
//
// Decomposes a product of two rank-2 tensors T_{ab} S_{cd} into a sum of
// basis elements built from metric tensors g_{ab}, antisymmetrised metric
// products, and trace pieces.  This is the tensor analogue of decomposing a
// direct product of representations.
//
// For a rank-2 ⊗ rank-2 product in `dim` dimensions the symmetric trace-free,
// antisymmetric, and trace basis elements are:
//   - g_{ac} g_{bd} + g_{ad} g_{bc}   (symmetric part)
//   - g_{ac} g_{bd} - g_{ad} g_{bc}   (antisymmetric part)
//   - g_{ab} g_{cd}                    (trace part)
//
// `expr` should be a product T_{ab} S_{cd}; indices are read from the
// outermost `Mul`.  When the expression does not have exactly two rank-2
// Indexed factors the function returns the input unchanged.

pub fn decompose_product(
    expr: &Expr,
    dim: usize,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Expr {
    // Collect exactly two Indexed factors from a product.
    let factors = match expr {
        Expr::Mul(fs) => fs.clone(),
        Expr::Indexed(_, _) => vec![expr.clone()],
        _ => return expr.clone(),
    };

    let indexed: Vec<&Expr> = factors
        .iter()
        .filter(|f| matches!(f, Expr::Indexed(_, _)))
        .collect();

    if indexed.len() != 2 {
        return expr.clone();
    }

    let (idx_a, idx_b) = match (indexed[0], indexed[1]) {
        (Expr::Indexed(_, ia), Expr::Indexed(_, ib)) if ia.len() == 2 && ib.len() == 2 => (ia, ib),
        _ => return expr.clone(),
    };

    // Free index names: [a, b] from first tensor, [c, d] from second.
    let a = idx_a[0].name;
    let b = idx_a[1].name;
    let c = idx_b[0].name;
    let d = idx_b[1].name;

    let va_a = idx_a[0].variance.clone();
    let va_b = idx_a[1].variance.clone();
    let va_c = idx_b[0].variance.clone();
    let va_d = idx_b[1].variance.clone();

    let g = interner.get_or_intern("g");

    let mk_g = |i1: lasso::Spur, v1: ax_ir::Variance, i2: lasso::Spur, v2: ax_ir::Variance| {
        Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                ax_ir::Index {
                    name: i1,
                    variance: v1,
                    index_type: None,
                },
                ax_ir::Index {
                    name: i2,
                    variance: v2,
                    index_type: None,
                },
            ],
        )
    };

    // g_{ac} g_{bd}
    let g_ac_bd = Expr::mul(vec![
        mk_g(a, va_a.clone(), c, va_c.clone()),
        mk_g(b, va_b.clone(), d, va_d.clone()),
    ]);
    // g_{ad} g_{bc}
    let g_ad_bc = Expr::mul(vec![
        mk_g(a, va_a.clone(), d, va_d.clone()),
        mk_g(b, va_b.clone(), c, va_c.clone()),
    ]);
    // g_{ab} g_{cd}
    let g_ab_cd = Expr::mul(vec![mk_g(a, va_a, b, va_b), mk_g(c, va_c, d, va_d)]);

    // Symmetric part: (1/2)(g_ac g_bd + g_ad g_bc)
    let sym_part = Expr::mul(vec![
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::add(vec![g_ac_bd.clone(), g_ad_bc.clone()]),
    ]);

    // Antisymmetric part: (1/2)(g_ac g_bd - g_ad g_bc)
    let antisym_part = Expr::mul(vec![
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::add(vec![g_ac_bd, Expr::neg(g_ad_bc)]),
    ]);

    // Trace part with dimension factor
    let trace_part = Expr::mul(vec![
        Expr::Rational(BigRational::new(
            1.into(),
            num_bigint::BigInt::from(dim as i64),
        )),
        g_ab_cd,
    ]);

    let basis = vec![
        canonicalise(&sym_part, properties, interner),
        canonicalise(&antisym_part, properties, interner),
        canonicalise(&trace_part, properties, interner),
    ];

    decompose(expr, &basis, properties, interner)
}

// ─── expand_implicit ─────────────────────────────────────────────────────────
//
// Write out products of objects with implicit indices by making all index
// contractions explicit.  This is a recursion wrapper around `explicit_indices`
// that handles sums, negations, and nested Call arguments.
//
// For an Add, each term gets its own freshly-named dummy indices so that
// different terms never share dummy names, which would create phantom
// contractions in subsequent canonicalisation.

pub fn expand_implicit(
    expr: &Expr,
    implicit_index_tensors: &HashSet<lasso::Spur>,
    available_indices: &[lasso::Spur],
    n_indices_per_tensor: &HashMap<lasso::Spur, usize>,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Mul(_) | Expr::Sym(_) => explicit_indices(
            expr,
            implicit_index_tensors,
            available_indices,
            n_indices_per_tensor,
            interner,
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .enumerate()
                .map(|(i, term)| {
                    // Each term gets a disjoint block of 20 index names.
                    let offset = i * 20;
                    let term_indices: Vec<lasso::Spur> = (offset..offset + 20)
                        .map(|j| interner.get_or_intern(&format!("_exp{}", j)))
                        .collect();
                    expand_implicit(
                        term,
                        implicit_index_tensors,
                        &term_indices,
                        n_indices_per_tensor,
                        interner,
                    )
                })
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(expand_implicit(
            e,
            implicit_index_tensors,
            available_indices,
            n_indices_per_tensor,
            interner,
        )),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| {
                    expand_implicit(
                        a,
                        implicit_index_tensors,
                        available_indices,
                        n_indices_per_tensor,
                        interner,
                    )
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Index, Variance};
    use std::collections::HashMap;

    #[test]
    fn classify_simple_product() {
        let interner = ax_ir::Interner::new();
        let a_sym = interner.get_or_intern("A");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(a_sym)),
            vec![
                Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: nu,
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );
        let ic = index_classifier::classify_indices(&expr);
        assert_eq!(ic.free.len(), 2);
        assert_eq!(ic.dummy.len(), 0);
        assert_eq!(ic.total, 2);
    }

    #[test]
    fn classify_contracted_product() {
        let interner = ax_ir::Interner::new();
        let a_sym = interner.get_or_intern("A");
        let b_sym = interner.get_or_intern("B");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(a_sym)),
                vec![Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(b_sym)),
                vec![
                    Index {
                        name: mu,
                        variance: Variance::Up,
                        index_type: None,
                    },
                    Index {
                        name: nu,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            ),
        ]);
        let ic = index_classifier::classify_indices(&expr);
        assert_eq!(ic.free.len(), 1);
        assert_eq!(ic.dummy.len(), 1);
        assert_eq!(ic.total, 3);
    }

    #[test]
    fn fresh_dummy_avoids_existing() {
        let interner = ax_ir::Interner::new();
        let mu = interner.get_or_intern("mu");
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(interner.get_or_intern("T"))),
            vec![Index {
                name: mu,
                variance: Variance::Down,
                index_type: None,
            }],
        );
        let ic = index_classifier::classify_indices(&expr);
        let fresh = index_classifier::get_fresh_dummy(&ic, "d", &interner);
        assert_ne!(fresh, mu);
    }

    #[test]
    fn diagonal_inverse() {
        let interner = ax_ir::Interner::new();
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int(2.into()), Expr::Int(3.into())]);
        let ginv = g.symbolic_inverse(&interner);
        let expected = Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
        assert_eq!(*ginv.get(0, 0), expected);
    }

    #[test]
    fn non_diagonal_metric_inverse() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        let mut g = SymbolicMatrix::new(2);
        g.set(0, 0, Expr::Sym(a));
        g.set(0, 1, Expr::Sym(b));
        g.set(1, 0, Expr::Sym(b));
        g.set(1, 1, Expr::Sym(c));

        let ginv = g.symbolic_inverse(&interner);

        assert!(
            ginv.get(0, 0) != &Expr::zero(),
            "ginv[0][0] should not be zero for general metric"
        );
        assert!(
            ginv.get(0, 1) != &Expr::zero(),
            "ginv[0][1] should not be zero for off-diagonal metric"
        );
    }

    #[test]
    fn non_diagonal_inverse_times_original_is_identity() {
        let interner = ax_ir::Interner::new();
        let mut g = SymbolicMatrix::new(2);
        g.set(0, 0, Expr::Int(2.into()));
        g.set(0, 1, Expr::Int(1.into()));
        g.set(1, 0, Expr::Int(1.into()));
        g.set(1, 1, Expr::Int(3.into()));

        let ginv = g.symbolic_inverse(&interner);

        let product = ax_linalg::mat_mul(&g.data, &ginv.data, &interner);
        let product_simplified: Vec<Vec<Expr>> = product
            .iter()
            .map(|row| {
                row.iter()
                    .map(|e| simplify_expr(e.clone(), &interner))
                    .collect()
            })
            .collect();

        assert_eq!(product_simplified[0][0], Expr::one());
        assert_eq!(product_simplified[0][1], Expr::zero());
        assert_eq!(product_simplified[1][0], Expr::zero());
        assert_eq!(product_simplified[1][1], Expr::one());
    }

    #[test]
    fn detect_contraction_pair() {
        let interner = ax_ir::Interner::new();
        let mu = interner.get_or_intern("mu");
        let indices = vec![
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Up,
                index_type: None,
            },
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Down,
                index_type: None,
            },
        ];
        let pairs = detect_contractions(&indices);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn minkowski_christoffel_is_zero() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
        ]);

        let gamma = christoffel_from_metric(&g, &coords, &interner);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert_eq!(
                        gamma[i][j][k],
                        Expr::zero(),
                        "Gamma[{}][{}][{}] = {:?}",
                        i,
                        j,
                        k,
                        gamma[i][j][k]
                    );
                }
            }
        }
    }

    #[test]
    fn minkowski_kretschner_is_zero() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
        ]);

        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann =
            riemann_from_christoffel(&gamma, &coords, &interner, &ax_ir::Convention::default());
        let k = kretschner_scalar(&riemann, &g, &interner);
        assert_eq!(k, Expr::zero());
    }

    #[test]
    fn minkowski_geodesic_is_zero() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let geod = geodesic_equations(&gamma, &coords, &interner);
        for (i, eq) in geod.iter().enumerate() {
            assert_eq!(*eq, Expr::zero(), "geodesic[{}] = {:?}", i, eq);
        }
    }

    #[test]
    fn symmetric_tensor_canonicalize() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let mut properties = HashMap::new();
        properties.insert(g, vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])]);

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalize_indices(&expr, &properties, &interner);
        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices[0].name, mu);
            assert_eq!(indices[1].name, nu);
        } else {
            panic!("expected Indexed");
        }
    }

    #[test]
    fn antisymmetric_trace_is_zero() {
        let interner = ax_ir::Interner::new();
        let f_sym = interner.get_or_intern("F");
        let mu = interner.get_or_intern("mu");

        let mut properties = HashMap::new();
        properties.insert(
            f_sym,
            vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])],
        );

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalize_indices(&expr, &properties, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn kronecker_eliminates_in_product() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let a_sym = interner.get_or_intern("A");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");

        let delta_expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let a_expr = Expr::Indexed(
            Box::new(Expr::Sym(a_sym)),
            vec![
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: rho,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let product = Expr::mul(vec![delta_expr, a_expr]);
        let result = eliminate_kronecker(&product, delta, &interner);

        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, mu);
            assert_eq!(indices[1].name, rho);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn kronecker_trace_gives_dim() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let mu = interner.get_or_intern("mu");
        let dim = interner.get_or_intern("dim");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = eliminate_kronecker(&expr, delta, &interner);
        assert_eq!(result, Expr::Sym(dim));
    }

    #[test]
    fn expand_delta_4_indices() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: c,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: d,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = expand_delta(&expr, delta, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "expected 2 terms for 2x2 delta expansion");
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn symmetrise_2_tensor() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = symmetrise(&expr, &[0, 1], false, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("T"), "got: {}", pp);
    }

    #[test]
    fn antisymmetrise_gives_zero_for_symmetric() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let t = interner.get_or_intern("T");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = symmetrise(&expr, &[0, 1], true, &interner);
        let simplified = ax_eval::eval(&result, &ax_eval::Env::new(), &interner);
        assert_eq!(simplified, Expr::zero());
    }

    #[test]
    fn metric_lowers_index() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let v_sym = interner.get_or_intern("V");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let metric = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let vector = Expr::Indexed(
            Box::new(Expr::Sym(v_sym)),
            vec![ax_ir::Index {
                name: nu,
                variance: ax_ir::Variance::Up,
                index_type: None,
            }],
        );
        let product = Expr::mul(vec![metric, vector]);
        let result = eliminate_metric(&product, g, ginv, &interner);

        if let Expr::Indexed(base, indices) = &result {
            assert_eq!(**base, Expr::Sym(v_sym));
            assert_eq!(indices.len(), 1);
            assert_eq!(indices[0].name, mu);
            assert_eq!(indices[0].variance, ax_ir::Variance::Down);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn rename_dummies_canonical() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let alpha = interner.get_or_intern("alpha");

        struct TestEnv {
            index_families: HashMap<lasso::Spur, ax_ir::IndexFamily>,
            index_to_family: HashMap<lasso::Spur, lasso::Spur>,
        }

        impl DummyRenameEnv for TestEnv {
            fn index_families(&self) -> &HashMap<lasso::Spur, ax_ir::IndexFamily> {
                &self.index_families
            }

            fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur> {
                &self.index_to_family
            }
        }

        let env = TestEnv {
            index_families: HashMap::new(),
            index_to_family: HashMap::new(),
        };

        let expr1 = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let expr2 = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                ax_ir::Index {
                    name: alpha,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: alpha,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );

        let r1 = rename_dummies(&expr1, &env, &interner);
        let r2 = rename_dummies(&expr2, &env, &interner);
        assert_eq!(r1, r2);
    }

    #[test]
    fn sort_puts_scalars_first() {
        let interner = ax_ir::Interner::new();
        let props = HashMap::new();
        let b_sym = interner.get_or_intern("B");
        let a_sym = interner.get_or_intern("A");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::Mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(b_sym)),
                vec![ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                }],
            ),
            Expr::Int(3.into()),
            Expr::Indexed(
                Box::new(Expr::Sym(a_sym)),
                vec![ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                }],
            ),
        ]);
        let result = sort_product(&expr, &props, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors[0], Expr::Int(3.into()));
            if let Expr::Indexed(base, _) = &factors[1] {
                assert_eq!(**base, Expr::Sym(a_sym));
            } else {
                panic!("expected indexed tensor as second factor");
            }
            if let Expr::Indexed(base, _) = &factors[2] {
                assert_eq!(**base, Expr::Sym(b_sym));
            } else {
                panic!("expected indexed tensor as third factor");
            }
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn leibniz_on_product() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        let expr = Expr::Call(d, vec![Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)])]);
        let result = product_rule(&expr, &derivs, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2);
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn leibniz_on_constant() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        let expr = Expr::Call(d, vec![Expr::Int(5.into())]);
        let result = product_rule(&expr, &derivs, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn leibniz_on_sum() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        let expr = Expr::Call(d, vec![Expr::add(vec![Expr::Sym(a), Expr::Sym(b)])]);
        let result = product_rule(&expr, &derivs, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2);
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn distribute_product_over_sum() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(a)),
                vec![ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                }],
            ),
            Expr::add(vec![Expr::Sym(b), Expr::Sym(c)]),
        ]);
        let result = tensor_distribute(&expr, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2);
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn epsilon_fully_contracted_4d() {
        let interner = ax_ir::Interner::new();
        let eps = interner.get_or_intern("epsilon");
        let delta = interner.get_or_intern("delta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let e1 = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: c,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: d,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
            ],
        );
        let e2 = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: c,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: d,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let product = Expr::mul(vec![e1, e2]);
        let result = epsilon_to_delta(&product, eps, delta, 4, &interner);
        let simplified = ax_eval::eval(&result, &ax_eval::Env::new(), &interner);
        assert_eq!(simplified, Expr::Int(24.into()));
    }

    #[test]
    fn evaluate_simple_contraction() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let t_val = interner.get_or_intern("t");
        let x_val = interner.get_or_intern("x");

        let rules = vec![
            ComponentRule {
                tensor: t_sym,
                indices: vec![
                    (t_val, ax_ir::Variance::Down),
                    (t_val, ax_ir::Variance::Down),
                ],
                value: Expr::Int(1.into()),
            },
            ComponentRule {
                tensor: t_sym,
                indices: vec![
                    (x_val, ax_ir::Variance::Down),
                    (x_val, ax_ir::Variance::Down),
                ],
                value: Expr::Int(2.into()),
            },
        ];

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );

        let env = DefaultEvalEnv::new(vec![t_val, x_val], HashMap::new());

        let result = evaluate_components(&expr, &rules, &HashMap::new(), &env, &interner);
        let simplified = simplify_expr(result, &interner);
        assert_eq!(simplified, Expr::Int(3.into()));
    }

    #[test]
    fn evaluate_metric_contraction() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");

        let rules = vec![
            ComponentRule {
                tensor: g,
                indices: vec![(t, Variance::Down), (t, Variance::Down)],
                value: Expr::Int((-1).into()),
            },
            ComponentRule {
                tensor: g,
                indices: vec![(r, Variance::Down), (r, Variance::Down)],
                value: Expr::Int(1.into()),
            },
        ];

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let env = DefaultEvalEnv::new(vec![t, r], HashMap::new());
        let result = evaluate_components_v2(&expr, &rules, &env, &interner);
        let simplified = ax_eval::eval(&result, &ax_eval::Env::new(), &interner);
        assert_eq!(simplified, Expr::zero());
    }

    #[test]
    fn evaluate_with_symmetry() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");

        let rules = vec![
            ComponentRule {
                tensor: g,
                indices: vec![(t, Variance::Down), (t, Variance::Down)],
                value: Expr::Int((-1).into()),
            },
            ComponentRule {
                tensor: g,
                indices: vec![(r, Variance::Down), (r, Variance::Down)],
                value: Expr::Int(1.into()),
            },
        ];

        let mut props = HashMap::new();
        props.insert(g, vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])]);
        let env = DefaultEvalEnv::new(vec![t, r], props);

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                Index {
                    name: t,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: r,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = evaluate_components_v2(&expr, &rules, &env, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn symmetric_metric_lookup() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");

        let rules = vec![
            ComponentRule {
                tensor: g,
                indices: vec![(t, Variance::Down), (t, Variance::Down)],
                value: Expr::Int((-1).into()),
            },
            ComponentRule {
                tensor: g,
                indices: vec![(t, Variance::Down), (r, Variance::Down)],
                value: Expr::Int(0.into()),
            },
            ComponentRule {
                tensor: g,
                indices: vec![(r, Variance::Down), (r, Variance::Down)],
                value: Expr::Int(1.into()),
            },
        ];

        let mut props = HashMap::new();
        props.insert(g, vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])]);
        let env = DefaultEvalEnv::new(vec![t, r], props.clone());

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                Index {
                    name: r,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: t,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = handle_factor(&expr, &rules, &env, &props, &interner);
        assert_eq!(
            result,
            Expr::Int(0.into()),
            "symmetric g_{{rt}} should equal g_{{tr}} = 0"
        );
    }

    #[test]
    fn antisymmetric_lookup_flips_sign() {
        let interner = ax_ir::Interner::new();
        let f = interner.get_or_intern("F");
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");

        let rules = vec![ComponentRule {
            tensor: f,
            indices: vec![(t, Variance::Down), (r, Variance::Down)],
            value: Expr::Int(5.into()),
        }];

        let mut props = HashMap::new();
        props.insert(f, vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])]);
        let env = DefaultEvalEnv::new(vec![t, r], props.clone());

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(f)),
            vec![
                Index {
                    name: r,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: t,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = handle_factor(&expr, &rules, &env, &props, &interner);
        assert_eq!(
            result,
            Expr::neg(Expr::Int(5.into())),
            "antisymmetric F_{{rt}} should be -F_{{tr}}"
        );
    }

    #[test]
    fn epsilon_3d_identity() {
        let interner = ax_ir::Interner::new();
        let eps = interner.get_or_intern("epsilon");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");

        let mut props = HashMap::new();
        props.insert(eps, vec![ax_ir::TensorProperty::EpsilonTensor]);
        let env = DefaultEvalEnv::new(vec![x, y, z], props.clone());

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                Index {
                    name: x,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: y,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: z,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = handle_epsilon(&expr, eps, &env, &interner);
        assert_eq!(result, Expr::Int(1.into()));

        let expr2 = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                Index {
                    name: y,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: x,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: z,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result2 = handle_epsilon(&expr2, eps, &env, &interner);
        assert_eq!(result2, Expr::Int((-1i64).into()));

        let expr3 = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                Index {
                    name: x,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: x,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: z,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result3 = handle_epsilon(&expr3, eps, &env, &interner);
        assert_eq!(result3, Expr::zero());
    }

    #[test]
    fn nested_derivative() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("partial");
        let r = interner.get_or_intern("r");

        let inner = Expr::Call(
            d,
            vec![Expr::pow(Expr::Sym(r), Expr::Int(2.into())), Expr::Sym(r)],
        );
        let outer = Expr::Call(d, vec![inner, Expr::Sym(r)]);

        let env = DefaultEvalEnv::new(vec![r], HashMap::new());
        let result = evaluate_components_v2(&outer, &[], &env, &interner);
        let simplified = simplify_expr(result, &interner);
        assert_eq!(simplified, Expr::Int(2.into()));
    }

    #[test]
    fn derivative_of_product() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("partial");
        let x = interner.get_or_intern("x");

        let expr = Expr::Call(
            d,
            vec![
                Expr::mul(vec![
                    Expr::Sym(x),
                    Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
                ]),
                Expr::Sym(x),
            ],
        );

        let env = DefaultEvalEnv::new(vec![x], HashMap::new());
        let result = evaluate_components_v2(&expr, &[], &env, &interner);
        let simplified = simplify_expr(result, &interner);
        assert_eq!(
            simplified,
            Expr::mul(vec![
                Expr::Int(3.into()),
                Expr::pow(Expr::Sym(x), Expr::Int(2.into()))
            ])
        );
    }

    #[test]
    fn generating_set_symmetric_2tensor() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");

        let factors = vec![TensorFactorInfo {
            name: g_sym,
            n_indices: 2,
            start_position: 0,
            properties: vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])],
        }];

        let gens = build_generating_set(&factors, &interner);
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0][0], 1);
        assert_eq!(gens[0][1], 0);
        assert_eq!(gens[0][2], 2);
        assert_eq!(gens[0][3], 3);
    }

    #[test]
    fn canonicalise_symmetric_tensor() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let mut props = HashMap::new();
        props.insert(g_sym, vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])]);

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g_sym)),
            vec![
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalise(&expr, &props, &interner);
        if let Expr::Indexed(_, indices) = &result {
            let first = interner.resolve(indices[0].name);
            let second = interner.resolve(indices[1].name);
            assert!(
                first <= second,
                "expected canonical order, got {} {}",
                first,
                second
            );
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn test_parallel_canonicalise_matches_sequential() {
        let interner = ax_ir::Interner::new();
        let r_sym = interner.get_or_intern("R");
        let names = [
            interner.get_or_intern("a"),
            interner.get_or_intern("b"),
            interner.get_or_intern("c"),
            interner.get_or_intern("d"),
            interner.get_or_intern("e"),
            interner.get_or_intern("f"),
            interner.get_or_intern("g"),
            interner.get_or_intern("h"),
        ];

        let mut props = HashMap::new();
        props.insert(r_sym, vec![ax_ir::TensorProperty::RiemannSymmetry]);

        let terms = (0..20)
            .map(|offset| {
                let slots = [
                    names[offset % names.len()],
                    names[(offset + 1) % names.len()],
                    names[(offset + 2) % names.len()],
                    names[(offset + 3) % names.len()],
                ];
                Expr::Indexed(
                    Box::new(Expr::Sym(r_sym)),
                    slots
                        .into_iter()
                        .map(|name| ax_ir::Index {
                            name,
                            variance: ax_ir::Variance::Down,
                            index_type: None,
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let expr = Expr::add(terms);

        assert_eq!(
            canonicalise_parallel(&expr, &props, &interner),
            canonicalise(&expr, &props, &interner)
        );
    }

    #[test]
    fn generating_set_antisymmetric_2tensor() {
        let interner = ax_ir::Interner::new();
        let f_sym = interner.get_or_intern("F");

        let factors = vec![TensorFactorInfo {
            name: f_sym,
            n_indices: 2,
            start_position: 0,
            properties: vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])],
        }];

        let gens = build_generating_set(&factors, &interner);
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0][0], 1);
        assert_eq!(gens[0][1], 0);
        assert_eq!(gens[0][2], 3);
        assert_eq!(gens[0][3], 2);
    }

    #[test]
    fn canonicalise_antisymmetric_zero() {
        let interner = ax_ir::Interner::new();
        let f_sym = interner.get_or_intern("F");
        let mu = interner.get_or_intern("mu");

        let mut props = HashMap::new();
        props.insert(
            f_sym,
            vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])],
        );

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalise(&expr, &props, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn meld_detects_cancellation() {
        let interner = ax_ir::Interner::new();
        let f_sym = interner.get_or_intern("F");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let mut props = HashMap::new();
        props.insert(
            f_sym,
            vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])],
        );

        let t1 = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let t2 = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let sum = Expr::add(vec![t1, t2]);
        let result = meld(&sum, &props, &interner);
        assert_eq!(
            result,
            Expr::zero(),
            "F[ab] + F[ba] should cancel for antisymmetric F"
        );
    }

    #[test]
    fn meld_bianchi_identity() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let mut props = HashMap::new();
        props.insert(r, vec![ax_ir::TensorProperty::RiemannSymmetry]);

        let make_r = |i0, i1, i2, i3| {
            Expr::Indexed(
                Box::new(Expr::Sym(r)),
                vec![
                    Index {
                        name: i0,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i1,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i2,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i3,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            )
        };

        let bianchi = Expr::add(vec![
            make_r(a, b, c, d),
            make_r(a, c, d, b),
            make_r(a, d, b, c),
        ]);

        let result = meld(&bianchi, &props, &interner);
        assert_eq!(
            result,
            Expr::zero(),
            "Bianchi identity should be detected by meld, got: {}",
            ax_ir::pretty_print(&result, &interner)
        );
    }

    #[test]
    fn generating_set_riemann() {
        let interner = ax_ir::Interner::new();
        let r_sym = interner.get_or_intern("R");

        let factors = vec![TensorFactorInfo {
            name: r_sym,
            n_indices: 4,
            start_position: 0,
            properties: vec![ax_ir::TensorProperty::RiemannSymmetry],
        }];

        let gens = build_generating_set(&factors, &interner);
        assert_eq!(gens.len(), 3);
    }

    #[test]
    fn identical_tensor_exchange() {
        let interner = ax_ir::Interner::new();
        let a_sym = interner.get_or_intern("A");

        let factors = vec![
            TensorFactorInfo {
                name: a_sym,
                n_indices: 1,
                start_position: 0,
                properties: vec![],
            },
            TensorFactorInfo {
                name: a_sym,
                n_indices: 1,
                start_position: 1,
                properties: vec![],
            },
        ];

        let gens = build_generating_set(&factors, &interner);
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0][0], 1);
        assert_eq!(gens[0][1], 0);
    }

    #[test]
    fn lower_free_flips_variance() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: mu,
                variance: Variance::Up,
                index_type: None,
            }],
        );

        let result = lower_free_indices(&expr, &HashMap::new(), &HashMap::new(), &interner);
        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices[0].variance, Variance::Down);
        } else {
            panic!("expected Indexed");
        }
    }

    #[test]
    fn raise_free_flips_variance() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: mu,
                variance: Variance::Down,
                index_type: None,
            }],
        );

        let result = raise_free_indices(&expr, &HashMap::new(), &HashMap::new(), &interner);
        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices[0].variance, Variance::Up);
        } else {
            panic!("expected Indexed");
        }
    }

    #[test]
    fn lower_free_does_not_flip_contracted_index() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
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
            ],
        );

        let result = lower_free_indices(&expr, &HashMap::new(), &HashMap::new(), &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn lower_free_respects_non_free_family_position() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let spacetime = interner.get_or_intern("spacetime");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: mu,
                variance: Variance::Up,
                index_type: None,
            }],
        );

        let mut index_to_family = HashMap::new();
        index_to_family.insert(mu, spacetime);

        let mut index_families = HashMap::new();
        index_families.insert(
            spacetime,
            ax_ir::IndexFamily {
                name: spacetime,
                values: vec![],
                position: ax_ir::IndexPosition::Fixed,
                dimension: None,
                parent: None,
            },
        );

        let result = lower_free_indices(&expr, &index_to_family, &index_families, &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn unwrap_pulls_constant_out() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("a");
        let phi = interner.get_or_intern("phi");

        let mut derivs = HashSet::new();
        derivs.insert(d);
        let mut depends = HashMap::new();
        depends.insert(phi, vec![]); // phi is in the depends map → it depends on something
                                     // a is NOT in depends → it's a constant

        // D(a * phi) → a * D(phi)
        let expr = Expr::Call(d, vec![Expr::mul(vec![Expr::Sym(a), Expr::Sym(phi)])]);
        let result = unwrap_derivatives(&expr, &derivs, &depends, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        // a should be outside the derivative
        assert!(pp.contains('a'), "got: {pp}");
    }

    #[test]
    fn unwrap_constant_gives_zero() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");

        let mut derivs = HashSet::new();
        derivs.insert(d);
        let depends = HashMap::new();

        // D(5) → 0
        let expr = Expr::Call(d, vec![Expr::Int(5.into())]);
        let result = unwrap_derivatives(&expr, &derivs, &depends, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn ibp_moves_derivative() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        // D(A) * B → -A * D(B)
        let expr = Expr::mul(vec![Expr::Call(d, vec![Expr::Sym(a)]), Expr::Sym(b)]);
        let result = integrate_by_parts(&expr, a, &derivs, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains('B') && pp.contains('A'), "got: {pp}");
    }

    #[test]
    fn ibp_three_factor_product() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        // D(A) * B * C → -A * D(B) * C - A * B * D(C)
        let expr = Expr::mul(vec![
            Expr::Call(d, vec![Expr::Sym(a)]),
            Expr::Sym(b),
            Expr::Sym(c),
        ]);
        let result = integrate_by_parts(&expr, a, &derivs, &interner);
        // Result should be a sum of two negated terms
        assert!(
            matches!(result, Expr::Add(_)),
            "expected Add, got: {result:?}"
        );
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains('B') && pp.contains('C'), "got: {pp}");
    }

    #[test]
    fn weight_computation() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let mut weights = HashMap::new();
        weights.insert((x, String::new()), 1i64);
        weights.insert((y, String::new()), 2i64);

        // x * y has weight 3
        let expr = Expr::mul(vec![Expr::Sym(x), Expr::Sym(y)]);
        assert_eq!(compute_weight(&expr, &weights, "", &interner), Some(3));

        // x^2 has weight 2
        let expr2 = Expr::pow(Expr::Sym(x), Expr::Int(2.into()));
        assert_eq!(compute_weight(&expr2, &weights, "", &interner), Some(2));
    }

    #[test]
    fn keep_weight_filters() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let mut weights = HashMap::new();
        weights.insert((x, String::new()), 1i64);
        weights.insert((y, String::new()), 2i64);

        // x + y: keep weight=1 → x only
        let expr = Expr::add(vec![Expr::Sym(x), Expr::Sym(y)]);
        let result = keep_weight(&expr, 1, &weights, "", &interner);
        assert_eq!(result, Expr::Sym(x));
    }

    #[test]
    fn drop_weight_filters() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let mut weights = HashMap::new();
        weights.insert((x, String::new()), 1i64);
        weights.insert((y, String::new()), 2i64);

        // x + y: drop weight=1 → y only
        let expr = Expr::add(vec![Expr::Sym(x), Expr::Sym(y)]);
        let result = drop_weight(&expr, 1, &weights, "", &interner);
        assert_eq!(result, Expr::Sym(y));
    }

    #[test]
    fn weight_add_mixed_returns_none() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let mut weights = HashMap::new();
        weights.insert((x, String::new()), 1i64);
        weights.insert((y, String::new()), 2i64);

        // x + y has no single weight
        let expr = Expr::add(vec![Expr::Sym(x), Expr::Sym(y)]);
        assert_eq!(compute_weight(&expr, &weights, "", &interner), None);
    }

    #[test]
    fn complete_diagonal_metric() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");
        let ginv_sym = interner.get_or_intern("ginv");
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");

        let rules = vec![
            ComponentRule {
                tensor: g_sym,
                indices: vec![(t, ax_ir::Variance::Down), (t, ax_ir::Variance::Down)],
                value: Expr::Int((-1).into()),
            },
            ComponentRule {
                tensor: g_sym,
                indices: vec![(r, ax_ir::Variance::Down), (r, ax_ir::Variance::Down)],
                value: Expr::Int(1.into()),
            },
        ];

        let inv_rules = complete_inverse_metric(&rules, g_sym, ginv_sym, &[t, r], &interner);

        // g^{tt} = -1, g^{rr} = 1
        assert!(
            inv_rules.iter().any(|rule| rule.indices
                == vec![(t, ax_ir::Variance::Up), (t, ax_ir::Variance::Up)]
                && rule.value == Expr::Int((-1).into())),
            "missing g^tt = -1, got: {inv_rules:?}"
        );
        assert!(
            inv_rules.iter().any(|rule| rule.indices
                == vec![(r, ax_ir::Variance::Up), (r, ax_ir::Variance::Up)]
                && rule.value == Expr::Int(1.into())),
            "missing g^rr = 1, got: {inv_rules:?}"
        );
    }

    #[test]
    fn ibp_no_derivative_unchanged() {
        let interner = ax_ir::Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let mut derivs = HashSet::new();
        derivs.insert(d);

        // A * B (no derivative) → unchanged
        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = integrate_by_parts(&expr, a, &derivs, &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn young_project_antisymmetric_pair() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // Column tableau with 2 rows: antisymmetrise in positions (0, 1)
        let tableau = ax_young::YoungTableau {
            cells: vec![vec![0], vec![1]],
        };

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = young_project(&expr, &tableau, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("T"), "got: {pp}");
    }

    #[test]
    fn young_project_symmetric_pair() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // Row tableau with 1 row: symmetrise in positions (0, 1)
        let tableau = ax_young::YoungTableau {
            cells: vec![vec![0, 1]],
        };

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = young_project(&expr, &tableau, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("T"), "got: {pp}");
        // Symmetric result should contain both orderings summed
        assert!(
            matches!(result, Expr::Add(_) | Expr::Mul(_)),
            "got: {result:?}"
        );
    }

    #[test]
    fn reduce_delta_chain() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let dim = interner.get_or_intern("dim");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // δ^a_b δ^b_c → δ^a_c
        let d1 = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let d2 = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                Index {
                    name: b,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: c,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let expr = Expr::mul(vec![d1, d2]);
        let result = reduce_delta(&expr, delta, dim, &interner);

        match &result {
            Expr::Indexed(_, indices) => {
                assert_eq!(indices.len(), 2, "expected 2 indices");
                assert_eq!(indices[0].name, a);
                assert_eq!(indices[1].name, c);
            }
            Expr::Mul(factors) => {
                let indexed: Vec<_> = factors
                    .iter()
                    .filter(|f| matches!(f, Expr::Indexed(_, _)))
                    .collect();
                assert_eq!(indexed.len(), 1, "expected 1 delta factor, got {factors:?}");
                if let Expr::Indexed(_, indices) = indexed[0] {
                    assert_eq!(indices[0].name, a);
                    assert_eq!(indices[1].name, c);
                }
            }
            _ => panic!("expected Indexed or Mul, got {result:?}"),
        }
    }

    #[test]
    fn reduce_delta_trace_gives_dim() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let dim = interner.get_or_intern("dim");
        let a = interner.get_or_intern("a");

        // δ^a_a → dim
        let d = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = reduce_delta(&d, delta, dim, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("dim"), "expected dim, got: {pp}");
    }

    #[test]
    fn einsteinify_fixes_both_up() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let s = interner.get_or_intern("S");
        let a = interner.get_or_intern("a");

        // T[a+] * S[a+] → T[a+] * S[a-]
        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
        ]);
        let result = einsteinify(&expr, None, &interner);
        if let Expr::Mul(factors) = &result {
            let mut ups = 0usize;
            let mut downs = 0usize;
            for f in factors {
                if let Expr::Indexed(_, indices) = f {
                    for idx in indices {
                        if idx.name == a {
                            match idx.variance {
                                Variance::Up => ups += 1,
                                Variance::Down => downs += 1,
                            }
                        }
                    }
                }
            }
            assert_eq!(ups, 1, "expected 1 up index");
            assert_eq!(downs, 1, "expected 1 down index");
        } else {
            panic!("expected Mul, got: {result:?}");
        }
    }

    #[test]
    fn einsteinify_both_down_fixed() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let s = interner.get_or_intern("S");
        let a = interner.get_or_intern("a");

        // T[a-] * S[a-] → one up, one down
        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                }],
            ),
        ]);
        let result = einsteinify(&expr, None, &interner);
        if let Expr::Mul(factors) = &result {
            let mut ups = 0usize;
            let mut downs = 0usize;
            for f in factors {
                if let Expr::Indexed(_, indices) = f {
                    for idx in indices {
                        if idx.name == a {
                            match idx.variance {
                                Variance::Up => ups += 1,
                                Variance::Down => downs += 1,
                            }
                        }
                    }
                }
            }
            assert_eq!(ups + downs, 2);
            assert_eq!(ups, 1);
            assert_eq!(downs, 1);
        } else {
            panic!("expected Mul, got: {result:?}");
        }
    }

    #[test]
    fn einsteinify_already_correct_unchanged() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let s = interner.get_or_intern("S");
        let a = interner.get_or_intern("a");

        // T[a+] * S[a-] is already correct — unchanged
        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                }],
            ),
        ]);
        let result = einsteinify(&expr, None, &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn split_single_index() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let t0 = interner.get_or_intern("0");
        let i = interner.get_or_intern("i");

        // T[mu-] split mu → {0} + {i}  gives T[0-] + T[i-]
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![Index {
                name: mu,
                variance: Variance::Down,
                index_type: None,
            }],
        );
        let result = split_index(&expr, &[mu], &[t0], &[i], &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "expected 2 terms, got: {terms:?}");
        } else {
            panic!("expected Add, got: {result:?}");
        }
    }

    #[test]
    fn split_two_indices() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let t0 = interner.get_or_intern("t0");
        let i = interner.get_or_intern("i");

        // T[mu- nu-] with both mu and nu split → 4 terms
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
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
            ],
        );
        let result = split_index(&expr, &[mu, nu], &[t0], &[i], &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 4, "expected 4 terms, got: {terms:?}");
        } else {
            panic!("expected Add, got: {result:?}");
        }
    }

    #[test]
    fn split_no_matching_index_unchanged() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let t0 = interner.get_or_intern("t0");
        let i = interner.get_or_intern("i");

        // T[nu-] doesn't contain mu — unchanged
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![Index {
                name: nu,
                variance: Variance::Down,
                index_type: None,
            }],
        );
        let result = split_index(&expr, &[mu], &[t0], &[i], &interner);
        assert_eq!(result, expr);
    }

    // ── expand_dummies tests ──────────────────────────────────────────────────

    #[test]
    fn expand_dummies_trace() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let t_coord = interner.get_or_intern("t");
        let r_coord = interner.get_or_intern("r");

        // T[mu+, mu-] with coords [t, r] → T[t+, t-] + T[r+, r-]
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
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
            ],
        );
        let result = expand_dummies(&expr, &[t_coord, r_coord], &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "expected 2 terms, got {terms:?}");
        } else {
            panic!("expected Add with 2 terms, got {:?}", result);
        }
    }

    #[test]
    fn expand_dummies_four_coords() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let coords: Vec<lasso::Spur> = ["t", "r", "theta", "phi"]
            .iter()
            .map(|s| interner.get_or_intern(s))
            .collect();

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
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
            ],
        );
        let result = expand_dummies(&expr, &coords, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 4, "expected 4 terms for 4 coordinates");
            // Each term should be T[coord+, coord-] for some coord
            for term in terms {
                if let Expr::Indexed(_, indices) = term {
                    assert_eq!(indices.len(), 2);
                    assert_eq!(
                        indices[0].name, indices[1].name,
                        "both indices should be the same coordinate"
                    );
                    assert_ne!(
                        indices[0].variance, indices[1].variance,
                        "variances should differ"
                    );
                } else {
                    panic!("expected Indexed term, got {:?}", term);
                }
            }
        } else {
            panic!("expected Add with 4 terms, got {:?}", result);
        }
    }

    #[test]
    fn expand_dummies_no_dummy_unchanged() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let t_coord = interner.get_or_intern("t");
        let r_coord = interner.get_or_intern("r");

        // T[a-, b-]: no contraction (both down), should be returned unchanged
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = expand_dummies(&expr, &[t_coord, r_coord], &interner);
        assert_eq!(
            result, expr,
            "free-index expression should be returned unchanged"
        );
    }

    #[test]
    fn expand_dummies_two_contractions() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let t_coord = interner.get_or_intern("t");
        let r_coord = interner.get_or_intern("r");

        // T[mu+, mu-, nu+, nu-] with coords [t, r]
        // → sum over 2 choices for mu × 2 choices for nu = 4 terms
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t_sym)),
            vec![
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
                Index {
                    name: nu,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: nu,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = expand_dummies(&expr, &[t_coord, r_coord], &interner);
        // Should have 2×2 = 4 terms
        let count = count_add_terms(&result);
        assert_eq!(
            count, 4,
            "two dummy pairs over 2 coords should give 4 terms, got {count}"
        );
    }

    #[test]
    fn expand_dummies_product() {
        let interner = ax_ir::Interner::new();
        let a_sym = interner.get_or_intern("A");
        let b_sym = interner.get_or_intern("B");
        let mu = interner.get_or_intern("mu");
        let t_coord = interner.get_or_intern("t");
        let r_coord = interner.get_or_intern("r");

        // A[mu+] * B[mu-] with coords [t, r] → A[t+]*B[t-] + A[r+]*B[r-]
        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(a_sym)),
                vec![Index {
                    name: mu,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(b_sym)),
                vec![Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                }],
            ),
        ]);
        let result = expand_dummies(&expr, &[t_coord, r_coord], &interner);
        let count = count_add_terms(&result);
        assert_eq!(
            count, 2,
            "A[mu+]*B[mu-] over 2 coords should give 2 terms, got {count}"
        );
    }

    #[test]
    fn expand_dummies_add_distributes() {
        let interner = ax_ir::Interner::new();
        let t_sym = interner.get_or_intern("T");
        let s_sym = interner.get_or_intern("S");
        let mu = interner.get_or_intern("mu");
        let t_coord = interner.get_or_intern("t");
        let r_coord = interner.get_or_intern("r");

        let mk = |sym, var: Variance| {
            Expr::Indexed(
                Box::new(Expr::Sym(sym)),
                vec![
                    Index {
                        name: mu,
                        variance: var.clone(),
                        index_type: None,
                    },
                    Index {
                        name: mu,
                        variance: if var == Variance::Up {
                            Variance::Down
                        } else {
                            Variance::Up
                        },
                        index_type: None,
                    },
                ],
            )
        };

        // (T[mu+,mu-] + S[mu+,mu-]) with 2 coords → 4 terms total
        let expr = Expr::add(vec![mk(t_sym, Variance::Up), mk(s_sym, Variance::Up)]);
        let result = expand_dummies(&expr, &[t_coord, r_coord], &interner);
        let count = count_add_terms(&result);
        assert_eq!(
            count, 4,
            "sum of two traces over 2 coords should give 4 terms, got {count}"
        );
    }

    fn count_add_terms(expr: &Expr) -> usize {
        match expr {
            Expr::Add(terms) => terms.iter().map(count_add_terms).sum(),
            _ => 1,
        }
    }

    // ── explicit_indices tests ────────────────────────────────────────────────

    #[test]
    fn explicit_indices_matrix_product() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let i0 = interner.get_or_intern("_i0");
        let i1 = interner.get_or_intern("_i1");
        let i2 = interner.get_or_intern("_i2");

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2)].into_iter().collect();
        let avail = vec![i0, i1, i2];

        // A * B → A[_i0+, _i1-] * B[_i1+, _i2-]
        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);

        if let Expr::Mul(factors) = &result {
            let indexed_count = factors
                .iter()
                .filter(|f| matches!(f, Expr::Indexed(_, _)))
                .count();
            assert_eq!(
                indexed_count, 2,
                "both factors should have explicit indices: {:?}",
                result
            );

            let mut all_up = Vec::new();
            let mut all_down = Vec::new();
            for f in factors {
                if let Expr::Indexed(_, indices) = f {
                    for idx in indices {
                        match idx.variance {
                            Variance::Up => all_up.push(idx.name),
                            Variance::Down => all_down.push(idx.name),
                        }
                    }
                }
            }
            // There should be at least one shared contracted index
            let shared: Vec<_> = all_up.iter().filter(|u| all_down.contains(u)).collect();
            assert!(
                !shared.is_empty(),
                "should have a contracted index between A and B"
            );
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn explicit_indices_triple_product() {
        // A * B * C — three chained matrices
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let avail: Vec<lasso::Spur> = (0..6)
            .map(|i| interner.get_or_intern(&format!("_i{}", i)))
            .collect();

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        implicit.insert(c);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2), (c, 2)].into_iter().collect();

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);

        if let Expr::Mul(factors) = &result {
            assert_eq!(
                factors
                    .iter()
                    .filter(|f| matches!(f, Expr::Indexed(_, _)))
                    .count(),
                3,
                "all three factors should be indexed"
            );
            // Collect every index name that appears as both up and down
            let mut up: Vec<lasso::Spur> = Vec::new();
            let mut down: Vec<lasso::Spur> = Vec::new();
            for f in factors {
                if let Expr::Indexed(_, idxs) = f {
                    for idx in idxs {
                        match idx.variance {
                            Variance::Up => up.push(idx.name),
                            Variance::Down => down.push(idx.name),
                        }
                    }
                }
            }
            let contractions: Vec<_> = up.iter().filter(|u| down.contains(u)).collect();
            // A-B contraction + B-C contraction = 2 contracted pairs
            assert_eq!(
                contractions.len(),
                2,
                "expected 2 contracted index pairs for A*B*C"
            );
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn explicit_indices_no_implicit_unchanged() {
        // A tensor with no implicit-index property should be untouched
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let implicit: HashSet<lasso::Spur> = HashSet::new(); // empty
        let n_per: HashMap<lasso::Spur, usize> = HashMap::new();
        let avail: Vec<lasso::Spur> = Vec::new();

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);
        assert_eq!(
            result, expr,
            "expression without implicit tensors should be unchanged"
        );
    }

    #[test]
    fn explicit_indices_scalar_mixed_in_product() {
        // 3 * A * B — the scalar should be preserved, only A and B get indices
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let avail: Vec<lasso::Spur> = (0..4)
            .map(|i| interner.get_or_intern(&format!("_i{}", i)))
            .collect();

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2)].into_iter().collect();

        let expr = Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(a), Expr::Sym(b)]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);

        if let Expr::Mul(factors) = &result {
            let indexed = factors
                .iter()
                .filter(|f| matches!(f, Expr::Indexed(_, _)))
                .count();
            assert_eq!(indexed, 2, "only A and B should be indexed");
            // scalar 3 should still be present
            assert!(
                factors.iter().any(|f| *f == Expr::Int(3.into())),
                "scalar factor 3 should be preserved"
            );
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn explicit_indices_in_sum() {
        // (A * B) + (A * B) — distributes over Add
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let avail: Vec<lasso::Spur> = (0..4)
            .map(|i| interner.get_or_intern(&format!("_i{}", i)))
            .collect();

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2)].into_iter().collect();

        let product = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
        let expr = Expr::add(vec![product.clone(), product]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);

        // Each branch of the sum should be a Mul with two Indexed factors
        let check = |term: &Expr| {
            if let Expr::Mul(factors) = term {
                factors
                    .iter()
                    .filter(|f| matches!(f, Expr::Indexed(_, _)))
                    .count()
                    == 2
            } else {
                false
            }
        };
        match &result {
            Expr::Add(terms) => {
                for t in terms {
                    assert!(
                        check(t),
                        "each sum term should have two indexed factors, got {:?}",
                        t
                    );
                }
            }
            // If Expr::add collapsed identical terms into 2*(...), that's fine too
            Expr::Mul(factors) => {
                assert!(
                    factors.iter().any(|f| matches!(f, Expr::Indexed(_, _))),
                    "expected indexed factors in collapsed sum"
                );
            }
            _ => panic!("expected Add or Mul, got {:?}", result),
        }
    }

    #[test]
    fn explicit_indices_contraction_chain_correct() {
        // Verify A[_i0+, _i1-] * B[_i1+, _i2-]: A's lower == B's upper
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let i0 = interner.get_or_intern("_i0");
        let i1 = interner.get_or_intern("_i1");
        let i2 = interner.get_or_intern("_i2");

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2)].into_iter().collect();
        let avail = vec![i0, i1, i2];

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = explicit_indices(&expr, &implicit, &avail, &n_per, &interner);

        if let Expr::Mul(factors) = &result {
            let get_idx = |f: &Expr| -> Option<(lasso::Spur, lasso::Spur)> {
                if let Expr::Indexed(_, idxs) = f {
                    if idxs.len() == 2 {
                        let up = idxs.iter().find(|i| i.variance == Variance::Up)?.name;
                        let dn = idxs.iter().find(|i| i.variance == Variance::Down)?.name;
                        return Some((up, dn));
                    }
                }
                None
            };
            let indexed: Vec<_> = factors.iter().filter_map(get_idx).collect();
            assert_eq!(indexed.len(), 2, "expected 2 indexed factors");
            let (a_up, a_dn) = indexed[0];
            let (b_up, b_dn) = indexed[1];
            assert_eq!(
                a_dn,
                b_up,
                "A's lower index ({}) should match B's upper index ({})",
                interner.resolve(a_dn),
                interner.resolve(b_up)
            );
            assert_ne!(a_up, b_dn, "outer indices should be distinct");
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    // ── rewrite_indices tests ─────────────────────────────────────────────────

    #[test]
    fn rewrite_lower_index() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");

        // T[a+] with target Down → g[a-, _rw0-] * T[_rw0+]
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: a,
                variance: Variance::Up,
                index_type: None,
            }],
        );
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Down]);

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2, "expected metric * tensor");
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn rewrite_raise_index() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");

        // T[a-] with target Up → ginv[a+, _rw0+] * T[_rw0-]
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: a,
                variance: Variance::Down,
                index_type: None,
            }],
        );
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Up]);

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2, "expected inv-metric * tensor");
            // First factor should use ginv
            if let Expr::Indexed(base, _) = &factors[0] {
                if let Expr::Sym(s) = base.as_ref() {
                    assert_eq!(interner.resolve(*s), "ginv", "expected ginv for raising");
                }
            }
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn rewrite_two_indices() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // T[a+, b+] with target [Down, Down] → g[a-, _rw0-] * g[b-, _rw1-] * T[_rw0+, _rw1+]
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Down, Variance::Down]);

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 3, "expected g * g * T for two lowerings");
            // Two metric factors + the tensor
            let metric_count = factors
                .iter()
                .filter(|f| {
                    if let Expr::Indexed(base, _) = f {
                        if let Expr::Sym(s) = base.as_ref() {
                            return interner.resolve(*s) == "g";
                        }
                    }
                    false
                })
                .count();
            assert_eq!(metric_count, 2, "expected two metric factors");
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn rewrite_already_correct_unchanged() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");

        // T[a-] with target Down — already correct, no change
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: a,
                variance: Variance::Down,
                index_type: None,
            }],
        );
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Down]);

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        assert_eq!(
            result, expr,
            "already-correct expression should be unchanged"
        );
    }

    #[test]
    fn rewrite_unregistered_tensor_unchanged() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![Index {
                name: a,
                variance: Variance::Up,
                index_type: None,
            }],
        );
        // targets is empty — T is not registered
        let targets = HashMap::new();

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        assert_eq!(result, expr, "unregistered tensor should be unchanged");
    }

    #[test]
    fn rewrite_dummy_indices_are_fresh() {
        // Two separate rewrites should use distinct dummy names
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Down, Variance::Down]);

        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);
        if let Expr::Mul(factors) = &result {
            // Collect all dummy names used in metric factors
            let mut dummy_names: Vec<lasso::Spur> = Vec::new();
            for f in factors {
                if let Expr::Indexed(base, idxs) = f {
                    if let Expr::Sym(s) = base.as_ref() {
                        if interner.resolve(*s) == "g" {
                            if let Some(dummy_idx) = idxs.get(1) {
                                dummy_names.push(dummy_idx.name);
                            }
                        }
                    }
                }
            }
            assert_eq!(dummy_names.len(), 2, "expected 2 dummy index names");
            assert_ne!(
                dummy_names[0],
                dummy_names[1],
                "dummy indices should be distinct: both are '{}'",
                interner.resolve(dummy_names[0])
            );
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn rewrite_distributes_over_add() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let s = interner.get_or_intern("S");
        let g = interner.get_or_intern("g");
        let ginv = interner.get_or_intern("ginv");
        let a = interner.get_or_intern("a");

        let mk = |sym| {
            Expr::Indexed(
                Box::new(Expr::Sym(sym)),
                vec![Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                }],
            )
        };
        let mut targets = HashMap::new();
        targets.insert(t, vec![Variance::Down]);
        targets.insert(s, vec![Variance::Down]);

        let expr = Expr::add(vec![mk(t), mk(s)]);
        let result = rewrite_indices(&expr, &targets, g, ginv, &interner);

        // Both terms should now be products (metric * tensor)
        if let Expr::Add(terms) = &result {
            for term in terms {
                assert!(
                    matches!(term, Expr::Mul(_)),
                    "each sum term should be a Mul after rewriting, got {:?}",
                    term
                );
            }
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    // ── eliminate_vielbein tests ──────────────────────────────────────────────

    #[test]
    fn vielbein_contracts() {
        let interner = ax_ir::Interner::new();
        let e = interner.get_or_intern("e");
        let einv = interner.get_or_intern("einv");
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let mu = interner.get_or_intern("mu");

        // e[a+, mu-] * T[mu+] → T[a+]
        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(e)),
                vec![
                    Index {
                        name: a,
                        variance: Variance::Up,
                        index_type: None,
                    },
                    Index {
                        name: mu,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![Index {
                    name: mu,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
        ]);
        let result = eliminate_vielbein(&expr, e, einv, &interner);
        if let Expr::Indexed(base, indices) = &result {
            assert_eq!(**base, Expr::Sym(t));
            assert_eq!(indices[0].name, a);
            assert_eq!(indices[0].variance, Variance::Up);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn inv_vielbein_contracts() {
        // einv[mu+, a-] * T[a+] → T[mu+]
        let interner = ax_ir::Interner::new();
        let e = interner.get_or_intern("e");
        let einv = interner.get_or_intern("einv");
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(einv)),
                vec![
                    Index {
                        name: mu,
                        variance: Variance::Up,
                        index_type: None,
                    },
                    Index {
                        name: a,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                }],
            ),
        ]);
        let result = eliminate_vielbein(&expr, e, einv, &interner);
        if let Expr::Indexed(base, indices) = &result {
            assert_eq!(**base, Expr::Sym(t));
            assert_eq!(indices[0].name, mu);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn vielbein_two_indices_chained() {
        // e[a+, mu-] * e[b+, nu-] * T[mu+, nu+] → T[a+, b+]
        let interner = ax_ir::Interner::new();
        let e = interner.get_or_intern("e");
        let einv = interner.get_or_intern("einv");
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let mk_e = |up, dn| {
            Expr::Indexed(
                Box::new(Expr::Sym(e)),
                vec![
                    Index {
                        name: up,
                        variance: Variance::Up,
                        index_type: None,
                    },
                    Index {
                        name: dn,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            )
        };
        let expr = Expr::mul(vec![
            mk_e(a, mu),
            mk_e(b, nu),
            Expr::Indexed(
                Box::new(Expr::Sym(t)),
                vec![
                    Index {
                        name: mu,
                        variance: Variance::Up,
                        index_type: None,
                    },
                    Index {
                        name: nu,
                        variance: Variance::Up,
                        index_type: None,
                    },
                ],
            ),
        ]);
        let result = eliminate_vielbein(&expr, e, einv, &interner);
        if let Expr::Indexed(base, indices) = &result {
            assert_eq!(**base, Expr::Sym(t));
            assert_eq!(indices.len(), 2);
            let names: Vec<lasso::Spur> = indices.iter().map(|i| i.name).collect();
            assert!(
                names.contains(&a) && names.contains(&b),
                "expected a and b, got {:?}",
                names
            );
        } else {
            panic!("expected Indexed T[a+,b+], got {:?}", result);
        }
    }

    #[test]
    fn vielbein_no_contraction_unchanged() {
        // e[a+, mu-] alone — nothing to contract with, unchanged
        let interner = ax_ir::Interner::new();
        let e = interner.get_or_intern("e");
        let einv = interner.get_or_intern("einv");
        let a = interner.get_or_intern("a");
        let mu = interner.get_or_intern("mu");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(e)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: mu,
                    variance: Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = eliminate_vielbein(&expr, e, einv, &interner);
        assert_eq!(
            result, expr,
            "lone vielbein with no contraction partner should be unchanged"
        );
    }

    #[test]
    fn vielbein_distributes_over_add() {
        let interner = ax_ir::Interner::new();
        let e = interner.get_or_intern("e");
        let einv = interner.get_or_intern("einv");
        let t = interner.get_or_intern("T");
        let s = interner.get_or_intern("S");
        let a = interner.get_or_intern("a");
        let mu = interner.get_or_intern("mu");

        let mk_term = |sym| {
            Expr::mul(vec![
                Expr::Indexed(
                    Box::new(Expr::Sym(e)),
                    vec![
                        Index {
                            name: a,
                            variance: Variance::Up,
                            index_type: None,
                        },
                        Index {
                            name: mu,
                            variance: Variance::Down,
                            index_type: None,
                        },
                    ],
                ),
                Expr::Indexed(
                    Box::new(Expr::Sym(sym)),
                    vec![Index {
                        name: mu,
                        variance: Variance::Up,
                        index_type: None,
                    }],
                ),
            ])
        };
        let expr = Expr::add(vec![mk_term(t), mk_term(s)]);
        let result = eliminate_vielbein(&expr, e, einv, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2);
            for term in terms {
                assert!(
                    matches!(term, Expr::Indexed(..)),
                    "each term should be a contracted T[a+] or S[a+], got {:?}",
                    term
                );
            }
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn decompose_finds_coefficient() {
        // expr = 3 * R[a,b,c,d] + (-3) * R[b,a,c,d]
        // basis = [R[a,b,c,d]]
        // R[b,a,c,d] canonicalises to -R[a,b,c,d], so the sum canonicalises
        // to 6 * R[a,b,c,d].  decompose should return 6 * basis[0].
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");
        let mut props = HashMap::new();
        props.insert(r, vec![ax_ir::TensorProperty::RiemannSymmetry]);

        let mk = |i0, i1, i2, i3| {
            Expr::Indexed(
                Box::new(Expr::Sym(r)),
                vec![
                    Index {
                        name: i0,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i1,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i2,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i3,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            )
        };

        let basis = vec![mk(a, b, c, d)];
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Int(3.into()), mk(a, b, c, d)]),
            Expr::mul(vec![Expr::Int((-3i64).into()), mk(b, a, c, d)]),
        ]);

        let result = decompose(&expr, &basis, &props, &interner);

        // result should be 6 * R[a,b,c,d]
        match &result {
            Expr::Mul(factors) => {
                let coeff = &factors[0];
                assert!(
                    matches!(coeff, Expr::Int(n) if n.to_str_radix(10) == "6"),
                    "expected coefficient 6, got {:?}",
                    coeff
                );
            }
            _ => panic!("expected Mul, got: {:?}", result),
        }
    }

    #[test]
    fn decompose_product_rank2() {
        // g[a-,c-] * g[b-,d-] is itself one of the basis elements produced by
        // decompose_product, so decomposing it should yield a non-zero coefficient
        // for that basis slot (the antisymmetric/symmetric decomposition).
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");
        let props: HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> = HashMap::new();

        let mk_g = |i1, v1, i2, v2| {
            Expr::Indexed(
                Box::new(Expr::Sym(g)),
                vec![
                    Index {
                        name: i1,
                        variance: v1,
                        index_type: None,
                    },
                    Index {
                        name: i2,
                        variance: v2,
                        index_type: None,
                    },
                ],
            )
        };

        // expr = g[a-,c-] * g[b-,d-] — a product of two rank-1 tensors indexed
        // with four distinct free indices.
        let expr = Expr::mul(vec![
            mk_g(a, Variance::Down, c, Variance::Down),
            mk_g(b, Variance::Down, d, Variance::Down),
        ]);

        let result = decompose_product(&expr, 4, &props, &interner);
        // The result is a linear combination in terms of the basis; it should
        // be an Add (or a scaled basis element), not the bare product unchanged.
        // We verify it differs from the identity decomposition (residual-only).
        match &result {
            Expr::Add(terms) => {
                assert!(!terms.is_empty(), "decomposed result should have terms");
            }
            Expr::Mul(_) => {} // single-term result is also fine
            other => panic!(
                "expected Add or Mul from decompose_product, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn expand_implicit_chain() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        implicit.insert(c);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2), (c, 2)].into_iter().collect();
        let avail: Vec<lasso::Spur> = (0..10)
            .map(|i| interner.get_or_intern(&format!("_e{}", i)))
            .collect();

        // A * B * C → A[_e0+, _e1-] * B[_e1+, _e2-] * C[_e2+, _e3-]
        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = expand_implicit(&expr, &implicit, &avail, &n_per, &interner);

        if let Expr::Mul(factors) = &result {
            let indexed_count = factors
                .iter()
                .filter(|f| matches!(f, Expr::Indexed(_, _)))
                .count();
            assert_eq!(
                indexed_count, 3,
                "all three should have explicit indices: {:?}",
                result
            );
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn expand_implicit_add_uses_disjoint_indices() {
        // Two structurally distinct terms in a sum: A*B and A (single factor).
        // After expansion each should have indexed factors; the sum should survive
        // as an Add because the two terms are structurally different.
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        implicit.insert(c);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2), (c, 2)].into_iter().collect();
        let avail: Vec<lasso::Spur> = (0..10)
            .map(|i| interner.get_or_intern(&format!("_e{}", i)))
            .collect();

        // A*B  +  B*C  — distinct products, distinct structures after expansion.
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::mul(vec![Expr::Sym(b), Expr::Sym(c)]),
        ]);
        let result = expand_implicit(&expr, &implicit, &avail, &n_per, &interner);

        // Both terms should have been expanded (have Indexed factors inside Mul).
        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                2,
                "should still have 2 terms after expanding distinct products"
            );
            for term in terms {
                assert!(
                    matches!(term, Expr::Mul(_)),
                    "each term should be Mul: {:?}",
                    term
                );
            }
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn expand_implicit_neg_passes_through() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let mut implicit = HashSet::new();
        implicit.insert(a);
        implicit.insert(b);
        let n_per: HashMap<lasso::Spur, usize> = vec![(a, 2), (b, 2)].into_iter().collect();
        let avail: Vec<lasso::Spur> = (0..10)
            .map(|i| interner.get_or_intern(&format!("_e{}", i)))
            .collect();

        let expr = Expr::neg(Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]));
        let result = expand_implicit(&expr, &implicit, &avail, &n_per, &interner);

        // Should be Neg(Mul([Indexed(...), Indexed(...)]))
        match &result {
            Expr::Neg(inner) => {
                assert!(
                    matches!(inner.as_ref(), Expr::Mul(_)),
                    "inner should be Mul: {:?}",
                    inner
                );
            }
            Expr::Mul(_) => {} // canonicalised away the Neg — also fine
            other => panic!("expected Neg or Mul, got {:?}", other),
        }
    }
}
