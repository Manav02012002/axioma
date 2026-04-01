#![forbid(unsafe_code)]

pub mod adjform;

use ax_perm::{Perm, SGS};
use ax_ir::{Expr, Interner};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};

pub trait DummyRenameEnv {
    fn index_families(&self) -> &HashMap<lasso::Spur, ax_ir::IndexFamily>;
    fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur>;
}

pub trait ComponentEvalEnv {
    fn coordinates(&self) -> &HashSet<lasso::Spur>;
    fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur>;
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
                    generator.swap(
                        factors[i].start_position + k,
                        factors[j].start_position + k,
                    );
                }
                generators.push(generator);
            }
        }
    }

    generators
}

pub fn extract_factor_info(
    expr: &ax_ir::Expr,
    tensor_properties: &std::collections::HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
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
                let props = tensor_properties.get(name).cloned().unwrap_or_default();
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

fn detect_dummy_pairs(indices: &[ax_ir::Index]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut used = HashSet::new();

    for i in 0..indices.len() {
        if used.contains(&i) {
            continue;
        }
        for j in (i + 1)..indices.len() {
            if used.contains(&j) {
                continue;
            }
            if indices[i].name == indices[j].name && indices[i].variance != indices[j].variance {
                pairs.push((i, j));
                used.insert(i);
                used.insert(j);
                break;
            }
        }
    }

    pairs
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
    tensor_properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
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

    let total_indices = all_indices.len();
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

    let dummy_pairs = detect_dummy_pairs(&all_indices);
    let degree = total_indices + 2;

    let (canon_perm, canon_sign) = if generators.is_empty() {
        (extended_perm.clone(), 1)
    } else {
        let sgs = ax_perm::schreier_sims(&[], &generators, degree);
        ax_perm::canonical_perm(&extended_perm, &sgs, &dummy_pairs)
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
    tensor_properties: &std::collections::HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
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

/// Simplify a sum of tensor monomials using multi-term symmetry information.
pub fn meld(
    expr: &ax_ir::Expr,
    tensor_properties: &std::collections::HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) => {
            let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
            for term in terms {
                let simplified = meld(term, tensor_properties, interner);
                let structure = tensor_structure_key(&simplified, interner);
                if let Some((_, bucket)) = groups.iter_mut().find(|(key, _)| *key == structure) {
                    bucket.push(simplified);
                } else {
                    groups.push((structure, vec![simplified]));
                }
            }

            let mut simplified_terms = Vec::new();
            for (_, group_terms) in groups {
                if group_terms.len() == 1 {
                    simplified_terms.push(group_terms.into_iter().next().unwrap());
                    continue;
                }

                let factor_info =
                    extract_factor_info_from_term(&group_terms[0], tensor_properties, interner);
                let mut projected = adjform::ProjectedAdjform::new();
                let mut canonical_terms: Vec<(Expr, Expr)> = Vec::new();

                for term in group_terms {
                    let canonical = canonicalise(&term, tensor_properties, interner);
                    let (scalar, indices) = extract_scalar_and_indices(&canonical);
                    let adj = adjform::Adjform::from_indices(&indices);
                    projected.add(adj, scalar_to_i32(&scalar));
                    canonical_terms.push((canonical.clone(), scalar_free_tensor_part(&canonical)));
                }

                for info in &factor_info {
                    for prop in &info.properties {
                        match prop {
                            ax_ir::TensorProperty::Symmetric(positions) => {
                                let abs_positions: Vec<usize> =
                                    positions.iter().map(|p| info.start_position + p).collect();
                                projected.symmetrize(&abs_positions);
                            }
                            ax_ir::TensorProperty::AntiSymmetric(positions) => {
                                let abs_positions: Vec<usize> =
                                    positions.iter().map(|p| info.start_position + p).collect();
                                projected.antisymmetrize(&abs_positions);
                            }
                            _ => {}
                        }
                    }
                }

                let mut combined: Vec<(Expr, i32)> = Vec::new();
                for (canonical, tensor_part) in canonical_terms {
                    let coeff = scalar_to_i32(&extract_scalar_and_indices(&canonical).0);
                    if let Some((_, acc)) =
                        combined.iter_mut().find(|(existing, _)| *existing == tensor_part)
                    {
                        *acc += coeff;
                    } else {
                        combined.push((tensor_part, coeff));
                    }
                }

                if combined.iter().all(|(_, coeff)| *coeff == 0) || projected.is_empty() {
                    continue;
                }

                for (tensor_part, coeff) in combined {
                    if coeff == 0 {
                        continue;
                    }
                    let term = match coeff {
                        1 => tensor_part,
                        -1 => Expr::neg(tensor_part),
                        _ => Expr::mul(vec![Expr::Int(coeff.into()), tensor_part]),
                    };
                    simplified_terms.push(term);
                }
            }

            if simplified_terms.is_empty() {
                Expr::zero()
            } else {
                Expr::add(simplified_terms)
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
        Expr::Indexed(_, indices) => (Expr::one(), indices.clone()),
        Expr::Mul(factors) => {
            let mut scalar_parts = Vec::new();
            let mut all_indices = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Indexed(_, indices) => all_indices.extend(indices.iter().cloned()),
                    other => scalar_parts.push(other.clone()),
                }
            }
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

fn scalar_to_i32(expr: &Expr) -> i32 {
    match expr {
        Expr::Int(n) => n.to_str_radix(10).parse().unwrap_or(1),
        Expr::Neg(inner) => -scalar_to_i32(inner),
        _ => 1,
    }
}

fn scalar_free_tensor_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let tensor_factors: Vec<Expr> = factors
                .iter()
                .filter(|factor| !matches!(factor, Expr::Int(_) | Expr::Rational(_) | Expr::Float(_)))
                .cloned()
                .collect();
            match tensor_factors.len() {
                0 => Expr::one(),
                1 => tensor_factors.into_iter().next().unwrap(),
                _ => Expr::mul(tensor_factors),
            }
        }
        Expr::Neg(inner) => scalar_free_tensor_part(inner),
        _ => expr.clone(),
    }
}

fn extract_factor_info_from_term(
    expr: &Expr,
    tensor_properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    interner: &ax_ir::Interner,
) -> Vec<TensorFactorInfo> {
    let wrapped = match expr {
        Expr::Mul(_) => expr.clone(),
        _ => Expr::Mul(vec![expr.clone()]),
    };
    extract_factor_info(&wrapped, tensor_properties, interner)
}

fn count_index_occurrences(expr: &Expr, counts: &mut HashMap<lasso::Spur, usize>) {
    match expr {
        Expr::Indexed(base, indices) => {
            count_index_occurrences(base, counts);
            for idx in indices {
                *counts.entry(idx.name).or_default() += 1;
            }
        }
        Expr::Mul(factors) | Expr::Add(factors) | Expr::List(factors) => {
            for factor in factors {
                count_index_occurrences(factor, counts);
            }
        }
        Expr::Neg(inner) => count_index_occurrences(inner, counts),
        Expr::Pow(base, exp) => {
            count_index_occurrences(base, counts);
            count_index_occurrences(exp, counts);
        }
        Expr::Call(_, args) => {
            for arg in args {
                count_index_occurrences(arg, counts);
            }
        }
        Expr::Complex(re, im) => {
            count_index_occurrences(re, counts);
            count_index_occurrences(im, counts);
        }
        Expr::FnDef(_, _, body) => count_index_occurrences(body, counts),
        Expr::Rule(lhs, rhs, _) => {
            count_index_occurrences(lhs, counts);
            count_index_occurrences(rhs, counts);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                count_index_occurrences(value, counts);
            }
        }
        Expr::Let(_, value, body) => {
            count_index_occurrences(value, counts);
            count_index_occurrences(body, counts);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    count_index_occurrences(cell, counts);
                }
            }
        }
        _ => {}
    }
}

fn substitute_indices(
    expr: &Expr,
    assignment: &HashMap<lasso::Spur, lasso::Spur>,
) -> Expr {
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
                .map(|row| row.iter().map(|cell| substitute_indices(cell, assignment)).collect())
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn evaluate_with_rules(
    expr: &Expr,
    rules: &[ComponentRule],
) -> Expr {
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
        Expr::Pow(base, exp) => Expr::pow(evaluate_with_rules(base, rules), evaluate_with_rules(exp, rules)),
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
        Expr::List(items) => Expr::List(items.iter().map(|item| evaluate_with_rules(item, rules)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|cell| evaluate_with_rules(cell, rules)).collect())
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn values_for_index<E: ComponentEvalEnv>(
    idx: lasso::Spur,
    index_values: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
    env: &E,
    interner: &ax_ir::Interner,
) -> Vec<lasso::Spur> {
    if let Some(family) = env.index_to_family().get(&idx) {
        if let Some(values) = index_values.get(family) {
            return values.clone();
        }
    }
    let mut defaults: Vec<lasso::Spur> = env.coordinates().iter().copied().collect();
    defaults.sort_by_key(|sym| interner.resolve(*sym).to_string());
    defaults
}

fn sum_over_dummies<E: ComponentEvalEnv>(
    expr: &Expr,
    dummy_indices: &[lasso::Spur],
    rules: &[ComponentRule],
    index_values: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
    env: &E,
    interner: &ax_ir::Interner,
) -> Expr {
    if dummy_indices.is_empty() {
        return evaluate_with_rules(expr, rules);
    }

    fn recurse<E: ComponentEvalEnv>(
        pos: usize,
        expr: &Expr,
        dummy_indices: &[lasso::Spur],
        rules: &[ComponentRule],
        index_values: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
        env: &E,
        interner: &ax_ir::Interner,
        assignment: &mut HashMap<lasso::Spur, lasso::Spur>,
        acc: &mut Vec<Expr>,
    ) {
        if pos == dummy_indices.len() {
            let specialized = substitute_indices(expr, assignment);
            acc.push(simplify_expr(
                evaluate_with_rules(&specialized, rules),
                interner,
            ));
            return;
        }

        let idx = dummy_indices[pos];
        for value in values_for_index(idx, index_values, env, interner) {
            assignment.insert(idx, value);
            recurse(
                pos + 1,
                expr,
                dummy_indices,
                rules,
                index_values,
                env,
                interner,
                assignment,
                acc,
            );
        }
    }

    let mut terms = Vec::new();
    let mut assignment = HashMap::new();
    recurse(
        0,
        expr,
        dummy_indices,
        rules,
        index_values,
        env,
        interner,
        &mut assignment,
        &mut terms,
    );
    simplify_expr(Expr::add(terms), interner)
}

pub fn evaluate_components<E: ComponentEvalEnv>(
    expr: &ax_ir::Expr,
    rules: &[ComponentRule],
    index_values: &std::collections::HashMap<lasso::Spur, Vec<lasso::Spur>>,
    env: &E,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut index_count = HashMap::new();
    count_index_occurrences(expr, &mut index_count);

    let mut free_indices: Vec<lasso::Spur> = index_count
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(name, _)| *name)
        .collect();
    free_indices.sort_by_key(|sym| interner.resolve(*sym).to_string());

    let mut dummy_indices: Vec<lasso::Spur> = index_count
        .iter()
        .filter(|(_, count)| **count == 2)
        .map(|(name, _)| *name)
        .collect();
    dummy_indices.sort_by_key(|sym| interner.resolve(*sym).to_string());

    if free_indices.is_empty() {
        return sum_over_dummies(expr, &dummy_indices, rules, index_values, env, interner);
    }

    fn recurse_free<E: ComponentEvalEnv>(
        pos: usize,
        expr: &Expr,
        free_indices: &[lasso::Spur],
        dummy_indices: &[lasso::Spur],
        rules: &[ComponentRule],
        index_values: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
        env: &E,
        interner: &ax_ir::Interner,
        assignment: &mut HashMap<lasso::Spur, lasso::Spur>,
        rows: &mut Vec<Expr>,
    ) {
        if pos == free_indices.len() {
            let specialized = substitute_indices(expr, assignment);
            let value = sum_over_dummies(
                &specialized,
                dummy_indices,
                rules,
                index_values,
                env,
                interner,
            );
            let simplified = simplify_expr(value, interner);

            let mut row = Vec::new();
            for fi in free_indices {
                row.push(Expr::Sym(*assignment.get(fi).unwrap()));
            }
            row.push(simplified);
            rows.push(Expr::List(row));
            return;
        }

        let idx = free_indices[pos];
        for value in values_for_index(idx, index_values, env, interner) {
            assignment.insert(idx, value);
            recurse_free(
                pos + 1,
                expr,
                free_indices,
                dummy_indices,
                rules,
                index_values,
                env,
                interner,
                assignment,
                rows,
            );
        }
    }

    let mut rows = Vec::new();
    let mut assignment = HashMap::new();
    recurse_free(
        0,
        expr,
        &free_indices,
        &dummy_indices,
        rules,
        index_values,
        env,
        interner,
        &mut assignment,
        &mut rows,
    );
    Expr::List(rows)
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
        let is_diagonal = (0..self.dim).all(|row| {
            (0..self.dim).all(|col| row == col || self.data[row][col] == Expr::zero())
        });

        if is_diagonal {
            let mut inverse = Self::new(self.dim);
            for i in 0..self.dim {
                inverse.data[i][i] =
                    simplify_expr(Expr::pow(self.data[i][i].clone(), Expr::Int((-1).into())), interner);
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
            let first_arg = args.first().map(|arg| format!("{arg:?}")).unwrap_or_default();
            (2, interner.resolve(*f).to_string(), first_arg)
        }
        _ => (4, format!("{expr:?}"), String::new()),
    }
}

pub fn sort_product(
    expr: &ax_ir::Expr,
    _tensor_properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
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
                .map(|(value, cond)| (sort_product(value, _tensor_properties, interner), cond.clone()))
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
            _ => Expr::Call(
                *f,
                vec![product_rule(&args[0], derivative_syms, interner)],
            ),
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

pub fn tensor_distribute(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
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
                        .map(|factor| epsilon_to_delta(factor, epsilon_sym, delta_sym, dim, interner))
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
                .map(|(_, factor)| {
                    epsilon_to_delta(factor, epsilon_sym, delta_sym, dim, interner)
                })
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
            Box::new(epsilon_to_delta(body, epsilon_sym, delta_sym, dim, interner)),
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
                    (epsilon_to_delta(value, epsilon_sym, delta_sym, dim, interner), cond.clone())
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(epsilon_to_delta(value, epsilon_sym, delta_sym, dim, interner)),
            Box::new(epsilon_to_delta(body, epsilon_sym, delta_sym, dim, interner)),
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
    properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let base_expr = canonicalize_indices(base, properties, interner);
            let mut indices = indices.clone();
            let mut negate = false;

            if let Expr::Sym(sym) = &base_expr {
                if let Some(props) = properties.get(sym) {
                    for prop in props {
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
                                    let left_key = left.iter().map(|i| sort_key(i, interner)).collect::<Vec<_>>();
                                    let right_key = right.iter().map(|i| sort_key(i, interner)).collect::<Vec<_>>();
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
            }

            let out = Expr::Indexed(Box::new(base_expr), indices);
            if negate { Expr::neg(out) } else { out }
        }
        Expr::Add(terms) => Expr::add(
            terms.iter()
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
            cases.iter()
                .map(|(value, cond)| (canonicalize_indices(value, properties, interner), cond.clone()))
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

fn collect_all_indices(
    expr: &Expr,
    counts: &mut HashMap<lasso::Spur, Vec<(ax_ir::Variance, usize)>>,
    id: &mut usize,
) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_all_indices(base, counts, id);
            for idx in indices {
                counts
                    .entry(idx.name)
                    .or_default()
                    .push((idx.variance.clone(), *id));
                *id += 1;
            }
        }
        Expr::Mul(factors) | Expr::List(factors) => {
            for factor in factors {
                collect_all_indices(factor, counts, id);
            }
        }
        Expr::Add(terms) => {
            for term in terms {
                collect_all_indices(term, counts, id);
            }
        }
        Expr::Neg(inner) => collect_all_indices(inner, counts, id),
        Expr::Pow(base, exp) => {
            collect_all_indices(base, counts, id);
            collect_all_indices(exp, counts, id);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_all_indices(arg, counts, id);
            }
        }
        Expr::Complex(re, im) => {
            collect_all_indices(re, counts, id);
            collect_all_indices(im, counts, id);
        }
        Expr::FnDef(_, _, body) => collect_all_indices(body, counts, id),
        Expr::Rule(lhs, rhs, _) => {
            collect_all_indices(lhs, counts, id);
            collect_all_indices(rhs, counts, id);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_all_indices(value, counts, id);
            }
        }
        Expr::Let(_, value, body) => {
            collect_all_indices(value, counts, id);
            collect_all_indices(body, counts, id);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_all_indices(cell, counts, id);
                }
            }
        }
        Expr::SetConvention(_, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_) => {}
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
            terms.iter()
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
            let mut index_count: HashMap<lasso::Spur, Vec<(ax_ir::Variance, usize)>> = HashMap::new();
            let mut occurrence_id = 0usize;
            collect_all_indices(expr, &mut index_count, &mut occurrence_id);

            let mut dummy_indices = Vec::new();
            for (name, occurrences) in &index_count {
                if occurrences.len() == 2 && occurrences[0].0 != occurrences[1].0 {
                    dummy_indices.push(*name);
                }
            }
            dummy_indices.sort_by_key(|sym| {
                let first_occurrence = index_count
                    .get(sym)
                    .and_then(|items| items.iter().map(|(_, id)| *id).min())
                    .unwrap_or(usize::MAX);
                (family_key(env, *sym, interner), first_occurrence)
            });

            let mut rename_map = HashMap::new();
            for (i, dummy) in dummy_indices.iter().enumerate() {
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
    match expr {
        Expr::Indexed(_, indices) => indices.iter().any(|i| i.name == idx),
        Expr::Mul(factors) => factors.iter().any(|f| contains_index(f, idx)),
        Expr::Add(terms) => terms.iter().any(|t| contains_index(t, idx)),
        Expr::Neg(e) => contains_index(e, idx),
        _ => false,
    }
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
    match expr {
        Expr::Indexed(_, indices) => indices
            .iter()
            .any(|index| index.name == name && index.variance == *variance),
        Expr::Mul(factors) => factors
            .iter()
            .any(|factor| has_index_with_variance(factor, name, variance)),
        _ => false,
    }
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
    tensor_properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Indexed(base, _) => {
            if let Expr::Sym(name) = base.as_ref() {
                if let Some(props) = tensor_properties.get(name) {
                    for prop in props {
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
                    } else if left_var == ax_ir::Variance::Down
                        && right_var == ax_ir::Variance::Up
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
                    let Some((sym, left_name, left_var, right_name, right_var)) = metric_info else {
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
            Expr::Complex(re, im) => {
                Expr::Complex(Box::new(diff(re, var, interner)), Box::new(diff(im, var, interner)))
            }
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
                cases.iter()
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
            cases.iter()
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
            1 + cases.iter().map(|(value, _)| node_count(value)).sum::<usize>()
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
            cases.iter()
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
fn collect_terms_expr(expr: &Expr, interner: &Interner) -> Expr {
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
            cases.iter()
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
                terms.push(Expr::mul(vec![
                    gamma[i][j][k].clone(),
                    dot_j,
                    dot_k,
                ]));
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
                    let inner =
                        unwrap_derivatives(other, derivative_syms, depends, interner);
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
        Expr::Neg(e) => {
            Expr::neg(unwrap_derivatives(e, derivative_syms, depends, interner))
        }
        _ => expr.clone(),
    }
}

fn depends_on_anything(
    expr: &Expr,
    depends: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
) -> bool {
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
        Expr::Neg(e) => {
            Expr::neg(integrate_by_parts(e, away_from, derivative_syms, interner))
        }
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
                .filter(|t| {
                    compute_weight(t, weights, label, interner) == Some(target_weight)
                })
                .cloned()
                .collect();
            if kept.is_empty() { Expr::zero() } else { Expr::add(kept) }
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
                .filter(|t| {
                    compute_weight(t, weights, label, interner) != Some(target_weight)
                })
                .cloned()
                .collect();
            if kept.is_empty() { Expr::zero() } else { Expr::add(kept) }
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
        if rule.tensor != metric_sym { continue; }
        if rule.indices.len() != 2 { continue; }

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
                if used.contains(&i) { continue; }
                for j in (i + 1)..all_indices.len() {
                    if used.contains(&j) { continue; }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Index, Variance};
    use std::collections::HashMap;

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
        let riemann = riemann_from_christoffel(
            &gamma,
            &coords,
            &interner,
            &ax_ir::Convention::default(),
        );
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
                ax_ir::Index { name: a, variance: ax_ir::Variance::Up, index_type: None },
                ax_ir::Index { name: b, variance: ax_ir::Variance::Up, index_type: None },
                ax_ir::Index { name: c, variance: ax_ir::Variance::Up, index_type: None },
                ax_ir::Index { name: d, variance: ax_ir::Variance::Up, index_type: None },
            ],
        );
        let e2 = Expr::Indexed(
            Box::new(Expr::Sym(eps)),
            vec![
                ax_ir::Index { name: a, variance: ax_ir::Variance::Down, index_type: None },
                ax_ir::Index { name: b, variance: ax_ir::Variance::Down, index_type: None },
                ax_ir::Index { name: c, variance: ax_ir::Variance::Down, index_type: None },
                ax_ir::Index { name: d, variance: ax_ir::Variance::Down, index_type: None },
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

        struct TestEnv {
            coordinates: HashSet<lasso::Spur>,
            index_to_family: HashMap<lasso::Spur, lasso::Spur>,
        }

        impl ComponentEvalEnv for TestEnv {
            fn coordinates(&self) -> &HashSet<lasso::Spur> {
                &self.coordinates
            }

            fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur> {
                &self.index_to_family
            }
        }

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
                ax_ir::Index { name: mu, variance: ax_ir::Variance::Up, index_type: None },
                ax_ir::Index { name: mu, variance: ax_ir::Variance::Down, index_type: None },
            ],
        );

        let mut env = TestEnv {
            coordinates: HashSet::new(),
            index_to_family: HashMap::new(),
        };
        env.coordinates.insert(t_val);
        env.coordinates.insert(x_val);

        let result = evaluate_components(&expr, &rules, &HashMap::new(), &env, &interner);
        let simplified = simplify_expr(result, &interner);
        assert_eq!(simplified, Expr::Int(3.into()));
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
                ax_ir::Index { name: nu, variance: ax_ir::Variance::Down, index_type: None },
                ax_ir::Index { name: mu, variance: ax_ir::Variance::Down, index_type: None },
            ],
        );
        let result = canonicalise(&expr, &props, &interner);
        if let Expr::Indexed(_, indices) = &result {
            let first = interner.resolve(indices[0].name);
            let second = interner.resolve(indices[1].name);
            assert!(first <= second, "expected canonical order, got {} {}", first, second);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
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
        props.insert(f_sym, vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])]);

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                ax_ir::Index { name: mu, variance: ax_ir::Variance::Down, index_type: None },
                ax_ir::Index { name: mu, variance: ax_ir::Variance::Down, index_type: None },
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
        props.insert(f_sym, vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])]);

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
        let expr = Expr::Call(
            d,
            vec![Expr::mul(vec![Expr::Sym(a), Expr::Sym(phi)])],
        );
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
        let expr = Expr::mul(vec![
            Expr::Call(d, vec![Expr::Sym(a)]),
            Expr::Sym(b),
        ]);
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
        assert!(matches!(result, Expr::Add(_)), "expected Add, got: {result:?}");
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
        let tableau = ax_young::YoungTableau { cells: vec![vec![0], vec![1]] };

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index { name: a, variance: Variance::Down, index_type: None },
                Index { name: b, variance: Variance::Down, index_type: None },
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
        let tableau = ax_young::YoungTableau { cells: vec![vec![0, 1]] };

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                Index { name: a, variance: Variance::Down, index_type: None },
                Index { name: b, variance: Variance::Down, index_type: None },
            ],
        );

        let result = young_project(&expr, &tableau, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("T"), "got: {pp}");
        // Symmetric result should contain both orderings summed
        assert!(matches!(result, Expr::Add(_) | Expr::Mul(_)), "got: {result:?}");
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
                Index { name: a, variance: Variance::Up, index_type: None },
                Index { name: b, variance: Variance::Down, index_type: None },
            ],
        );
        let d2 = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                Index { name: b, variance: Variance::Up, index_type: None },
                Index { name: c, variance: Variance::Down, index_type: None },
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
                Index { name: a, variance: Variance::Up, index_type: None },
                Index { name: a, variance: Variance::Down, index_type: None },
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
                vec![Index { name: a, variance: Variance::Up, index_type: None }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index { name: a, variance: Variance::Up, index_type: None }],
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
                vec![Index { name: a, variance: Variance::Down, index_type: None }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index { name: a, variance: Variance::Down, index_type: None }],
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
                vec![Index { name: a, variance: Variance::Up, index_type: None }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(s)),
                vec![Index { name: a, variance: Variance::Down, index_type: None }],
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
            vec![Index { name: mu, variance: Variance::Down, index_type: None }],
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
                Index { name: mu, variance: Variance::Down, index_type: None },
                Index { name: nu, variance: Variance::Down, index_type: None },
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
            vec![Index { name: nu, variance: Variance::Down, index_type: None }],
        );
        let result = split_index(&expr, &[mu], &[t0], &[i], &interner);
        assert_eq!(result, expr);
    }
}
