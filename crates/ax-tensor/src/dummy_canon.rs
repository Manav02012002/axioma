use crate::symmetry_bridge::{
    realized_tableaux_from_properties, realized_tableaux_from_symmetry, RealizedTableau,
};
use crate::young_engine::YoungEngineError;
use ax_ir::{Expr, Index, TensorProperty, Variance};
use ax_perm::dummy_group::{
    apply_permutation_to_labels, build_dummy_renaming_group, canonicalize_labels_under_dummy_group,
    DummySlot,
};
use ax_perm::{enumerate_subgroup, product, sign};
use lasso::Key;
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DummyCanonError {
    #[error("dummy canonicalization failed: {0}")]
    Dummy(#[from] ax_perm::dummy_group::DummyGroupError),
    #[error("dummy canonicalization Young phase failed: {0}")]
    Young(#[from] crate::young_engine::YoungEngineError),
    #[error("dummy canonicalization symmetry bridge failed: {0}")]
    Bridge(#[from] crate::symmetry_bridge::SymmetryBridgeError),
    #[error("dummy canonicalization supports only indexed tensor factors and multiplicative products on this path")]
    UnsupportedExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DummyCanonicalizationResult {
    pub expr: ax_ir::Expr,
    pub sign: i32,
    pub trace: ax_trace::DummyCanonicalizationTrace,
}

pub fn extract_slot_labels_and_metadata(
    expr: &ax_ir::Expr,
) -> Result<(Vec<String>, Vec<ax_perm::dummy_group::DummySlot>), DummyCanonError> {
    let Expr::Indexed(_, indices) = expr else {
        return Err(DummyCanonError::UnsupportedExpr);
    };
    Ok(factor_labels_and_slots(indices))
}

pub fn canonicalize_indexed_factor_modulo_symmetry_and_dummies(
    expr: &ax_ir::Expr,
    properties: &[ax_ir::TensorProperty],
) -> Result<DummyCanonicalizationResult, DummyCanonError> {
    let Expr::Indexed(_, original_indices) = expr else {
        return Err(DummyCanonError::UnsupportedExpr);
    };

    let (original_slot_labels, _) = extract_slot_labels_and_metadata(expr)?;
    let mut current = expr.clone();
    let mut current_permutation = (0..original_indices.len()).collect::<Vec<_>>();
    let mut total_sign = 1i32;
    let mut symmetry_orbit_count = 1usize;

    let normalized_properties = normalized_properties(properties);
    let tableaux = realized_tableaux(&normalized_properties)?;
    for tableau in tableaux {
        let step = canonicalize_with_realized_tableau(&current, &tableau)?;
        current = step.expr;
        total_sign *= step.sign;
        symmetry_orbit_count = symmetry_orbit_count.saturating_mul(step.candidate_count.max(1));
        current_permutation = step
            .permutation
            .iter()
            .map(|idx| current_permutation[*idx])
            .collect();
    }

    let Expr::Indexed(base, current_indices) = &current else {
        return Err(DummyCanonError::UnsupportedExpr);
    };
    let (current_labels, current_slots) = factor_labels_and_slots(current_indices);
    let canonical_slot_labels =
        canonicalize_labels_under_dummy_group(&current_labels, &current_slots)?;
    let dummy_orbit_count = dummy_orbit_count(&current_labels, &current_slots)?;
    let renamed_indices = rename_indices_by_labels(current_indices, &canonical_slot_labels);

    Ok(DummyCanonicalizationResult {
        expr: Expr::Indexed(base.clone(), renamed_indices),
        sign: total_sign,
        trace: ax_trace::DummyCanonicalizationTrace {
            original_slot_labels,
            canonical_slot_labels,
            original_slot_permutation: (0..original_indices.len()).collect(),
            canonical_slot_permutation: current_permutation,
            dummy_orbit_count,
            symmetry_orbit_count,
            sign: total_sign,
        },
    })
}

pub fn canonicalize_product_modulo_symmetry_and_dummies(
    expr: &ax_ir::Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<ax_ir::TensorProperty>,
) -> Result<DummyCanonicalizationResult, DummyCanonError> {
    let Expr::Mul(factors) = expr else {
        return Err(DummyCanonError::UnsupportedExpr);
    };
    if !factors
        .iter()
        .all(|factor| matches!(factor, Expr::Indexed(_, _)))
    {
        return Err(DummyCanonError::UnsupportedExpr);
    }

    let mut original_slot_labels = Vec::new();
    let mut canonical_slot_permutation = Vec::new();
    let mut symmetry_orbit_count = 1usize;
    let mut total_sign = 1i32;
    let mut rewritten_factors = Vec::with_capacity(factors.len());
    let mut slot_offset = 0usize;

    for factor in factors {
        let Expr::Indexed(base, _) = factor else {
            return Err(DummyCanonError::UnsupportedExpr);
        };
        let Expr::Sym(symbol) = base.as_ref() else {
            return Err(DummyCanonError::UnsupportedExpr);
        };
        let props = properties_for_symbol(*symbol);
        let factor_result =
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(factor, &props)?;
        original_slot_labels.extend(factor_result.trace.original_slot_labels.clone());
        canonical_slot_permutation.extend(
            factor_result
                .trace
                .canonical_slot_permutation
                .iter()
                .map(|slot| slot + slot_offset),
        );
        slot_offset += factor_result.trace.canonical_slot_permutation.len();
        symmetry_orbit_count =
            symmetry_orbit_count.saturating_mul(factor_result.trace.symmetry_orbit_count.max(1));
        total_sign *= factor_result.sign;
        rewritten_factors.push(factor_result.expr);
    }

    let product_indices = flattened_indices(&rewritten_factors)?;
    let (current_labels, current_slots) = factor_labels_and_slots(&product_indices);
    let canonical_slot_labels =
        canonicalize_labels_under_dummy_group(&current_labels, &current_slots)?;
    let dummy_orbit_count = dummy_orbit_count(&current_labels, &current_slots)?;
    let renamed_factors =
        rename_product_indices_by_labels(&rewritten_factors, &canonical_slot_labels)?;

    Ok(DummyCanonicalizationResult {
        expr: Expr::mul(renamed_factors),
        sign: total_sign,
        trace: ax_trace::DummyCanonicalizationTrace {
            original_slot_labels,
            canonical_slot_labels,
            original_slot_permutation: (0..current_labels.len()).collect(),
            canonical_slot_permutation,
            dummy_orbit_count,
            symmetry_orbit_count,
            sign: total_sign,
        },
    })
}

pub fn alpha_equivalent_modulo_dummies_and_symmetry(
    left: &ax_ir::Expr,
    right: &ax_ir::Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<ax_ir::TensorProperty>,
) -> Result<bool, DummyCanonError> {
    let lhs = canonicalize_expr(left, properties_for_symbol)?;
    let rhs = canonicalize_expr(right, properties_for_symbol)?;
    Ok(lhs.sign == rhs.sign
        && dummy_abstract_signature(&lhs.expr)? == dummy_abstract_signature(&rhs.expr)?)
}

fn canonicalize_expr(
    expr: &Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<TensorProperty>,
) -> Result<DummyCanonicalizationResult, DummyCanonError> {
    match expr {
        Expr::Indexed(base, _) => {
            let Expr::Sym(symbol) = base.as_ref() else {
                return Err(DummyCanonError::UnsupportedExpr);
            };
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(
                expr,
                &properties_for_symbol(*symbol),
            )
        }
        Expr::Mul(_) => {
            canonicalize_product_modulo_symmetry_and_dummies(expr, properties_for_symbol)
        }
        _ => Err(DummyCanonError::UnsupportedExpr),
    }
}

fn factor_labels_and_slots(indices: &[Index]) -> (Vec<String>, Vec<DummySlot>) {
    let labels = indices
        .iter()
        .map(|index| index.name.into_usize().to_string())
        .collect::<Vec<_>>();
    let slots = indices
        .iter()
        .enumerate()
        .map(|(position, index)| DummySlot {
            position,
            family: index
                .index_type
                .map(|family| family.into_usize().to_string()),
            variance: match index.variance {
                Variance::Up => 1,
                Variance::Down => -1,
            },
        })
        .collect::<Vec<_>>();
    (labels, slots)
}

fn rename_indices_by_labels(indices: &[Index], labels: &[String]) -> Vec<Index> {
    let name_lookup = indices
        .iter()
        .map(|index| (index.name.into_usize().to_string(), index.name))
        .collect::<HashMap<_, _>>();
    indices
        .iter()
        .zip(labels)
        .map(|(index, label)| Index {
            name: name_lookup.get(label).copied().unwrap_or(index.name),
            variance: index.variance.clone(),
            index_type: index.index_type,
        })
        .collect()
}

fn rename_product_indices_by_labels(
    factors: &[Expr],
    labels: &[String],
) -> Result<Vec<Expr>, DummyCanonError> {
    let indices = flattened_indices(factors)?;
    let name_lookup = indices
        .iter()
        .map(|index| (index.name.into_usize().to_string(), index.name))
        .collect::<HashMap<_, _>>();
    let mut label_iter = labels.iter();
    let mut renamed = Vec::with_capacity(factors.len());
    for factor in factors {
        let Expr::Indexed(base, idxs) = factor else {
            return Err(DummyCanonError::UnsupportedExpr);
        };
        let rewritten = idxs
            .iter()
            .map(|index| {
                let label = label_iter.next().cloned().unwrap_or_default();
                Index {
                    name: name_lookup.get(&label).copied().unwrap_or(index.name),
                    variance: index.variance.clone(),
                    index_type: index.index_type,
                }
            })
            .collect();
        renamed.push(Expr::Indexed(base.clone(), rewritten));
    }
    Ok(renamed)
}

fn flattened_indices(factors: &[Expr]) -> Result<Vec<Index>, DummyCanonError> {
    let mut indices = Vec::new();
    for factor in factors {
        let Expr::Indexed(_, factor_indices) = factor else {
            return Err(DummyCanonError::UnsupportedExpr);
        };
        indices.extend(factor_indices.iter().cloned());
    }
    Ok(indices)
}

fn dummy_abstract_signature(expr: &Expr) -> Result<String, DummyCanonError> {
    let indices = match expr {
        Expr::Indexed(_, factor_indices) => factor_indices.clone(),
        Expr::Mul(factors) => flattened_indices(factors)?,
        _ => return Err(DummyCanonError::UnsupportedExpr),
    };
    let mut counts: HashMap<lasso::Spur, usize> = HashMap::new();
    for index in &indices {
        *counts.entry(index.name).or_default() += 1;
    }

    let mut dummy_names = Vec::new();
    let mut seen = BTreeSet::new();
    for index in &indices {
        if counts.get(&index.name).copied().unwrap_or_default() == 2 && seen.insert(index.name) {
            dummy_names.push(index.name);
        }
    }
    let dummy_map = dummy_names
        .into_iter()
        .enumerate()
        .map(|(idx, name)| (name, format!("d{idx}")))
        .collect::<HashMap<_, _>>();

    render_expr_with_dummy_map(expr, &dummy_map)
}

fn render_expr_with_dummy_map(
    expr: &Expr,
    dummy_map: &HashMap<lasso::Spur, String>,
) -> Result<String, DummyCanonError> {
    let render_index = |index: &Index| {
        let label = dummy_map
            .get(&index.name)
            .cloned()
            .unwrap_or_else(|| format!("f{}", index.name.into_usize()));
        let variance = match index.variance {
            Variance::Up => "up",
            Variance::Down => "down",
        };
        let family = index
            .index_type
            .map(|family| family.into_usize().to_string())
            .unwrap_or_else(|| "_".into());
        format!("{label}:{variance}:{family}")
    };

    Ok(match expr {
        Expr::Indexed(base, factor_indices) => {
            let Expr::Sym(symbol) = base.as_ref() else {
                return Err(DummyCanonError::UnsupportedExpr);
            };
            format!(
                "I{}[{}]",
                symbol.into_usize(),
                factor_indices
                    .iter()
                    .map(render_index)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| render_expr_with_dummy_map(factor, dummy_map))
            .collect::<Result<Vec<_>, _>>()?
            .join("*"),
        _ => return Err(DummyCanonError::UnsupportedExpr),
    })
}

fn dummy_orbit_count(labels: &[String], slots: &[DummySlot]) -> Result<usize, DummyCanonError> {
    let group = build_dummy_renaming_group(labels, slots)?;
    let mut orbit = BTreeSet::new();
    for perm in enumerate_subgroup(&group.generators, group.degree) {
        orbit.insert(apply_permutation_to_labels(&perm, labels)?);
    }
    Ok(orbit.len())
}

fn normalized_properties(properties: &[TensorProperty]) -> Vec<TensorProperty> {
    let mut out = properties.to_vec();
    let (symmetry, identities) = crate::structured_curvature_properties_from_legacy(properties);
    if let Some(symmetry) = symmetry {
        if !out
            .iter()
            .any(|prop| matches!(prop, TensorProperty::TableauSymmetry(_)))
        {
            out.push(TensorProperty::TableauSymmetry(symmetry));
        }
    }
    if !identities.is_empty() {
        out.push(TensorProperty::TensorIdentities(identities));
    }
    out
}

fn realized_tableaux(
    properties: &[TensorProperty],
) -> Result<Vec<RealizedTableau>, DummyCanonError> {
    for property in properties {
        if let TensorProperty::TableauSymmetry(symmetry) = property {
            return Ok(realized_tableaux_from_symmetry(symmetry)?);
        }
    }
    match realized_tableaux_from_properties(properties) {
        Ok(tableaux) => Ok(tableaux),
        Err(crate::symmetry_bridge::SymmetryBridgeError::MissingTableaux) => Ok(Vec::new()),
        Err(err) => Err(err.into()),
    }
}

struct TableauStep {
    expr: Expr,
    sign: i32,
    permutation: Vec<usize>,
    candidate_count: usize,
}

fn canonicalize_with_realized_tableau(
    expr: &Expr,
    realized: &RealizedTableau,
) -> Result<TableauStep, DummyCanonError> {
    let Expr::Indexed(_base, indices) = expr else {
        return Err(DummyCanonError::UnsupportedExpr);
    };
    if realized.projector.row_group.degree != realized.slot_map.len()
        || realized.slot_map.iter().any(|slot| *slot >= indices.len())
    {
        return Err(YoungEngineError::SlotCountMismatch {
            expected: realized.projector.row_group.degree,
            actual: indices.len(),
        }
        .into());
    }

    let ranking = slot_ranking(indices, &realized.slot_map);
    let row_elements = enumerate_subgroup(
        &realized.projector.row_group.generators,
        realized.projector.row_group.degree,
    );
    let column_elements = enumerate_subgroup(
        &realized.projector.column_group.generators,
        realized.projector.column_group.degree,
    );

    let mut best_ranking = ranking.clone();
    let mut best_perm = (0..realized.slot_map.len()).collect::<Vec<_>>();
    let mut best_coeff = BigRational::one();

    for row in &row_elements {
        for column in &column_elements {
            let composed = product(column, row);
            let candidate = (0..ranking.len())
                .map(|idx| ranking[composed[idx]])
                .collect::<Vec<_>>();
            let candidate_coeff = BigRational::from_integer(sign(column).into());
            if candidate < best_ranking
                || (candidate == best_ranking && candidate_coeff < best_coeff)
            {
                best_ranking = candidate;
                best_perm = composed;
                best_coeff = candidate_coeff;
            }
        }
    }

    let mut full_permutation = (0..indices.len()).collect::<Vec<_>>();
    for (dest_local, source_local) in best_perm.iter().enumerate() {
        full_permutation[realized.slot_map[dest_local]] = realized.slot_map[*source_local];
    }

    let rewritten = crate::young_engine::rewrite_indexed_factor_by_slots(expr, &full_permutation)?;
    Ok(TableauStep {
        expr: rewritten,
        sign: if best_coeff.is_negative() { -1 } else { 1 },
        permutation: full_permutation,
        candidate_count: row_elements.len() * column_elements.len(),
    })
}

fn slot_ranking(indices: &[Index], slot_map: &[usize]) -> Vec<usize> {
    let mut keyed = slot_map
        .iter()
        .enumerate()
        .map(|(order, slot)| {
            let index = &indices[*slot];
            (
                order,
                (
                    index.name.into_usize(),
                    match index.variance {
                        Variance::Up => 0usize,
                        Variance::Down => 1usize,
                    },
                    index
                        .index_type
                        .map(|family| family.into_usize())
                        .unwrap_or(usize::MAX),
                    *slot,
                ),
            )
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    let mut ranking = vec![0usize; slot_map.len()];
    for (label, (original_order, _)) in keyed.into_iter().enumerate() {
        ranking[original_order] = label;
    }
    ranking
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Interner, TensorSymmetry};
    use std::collections::HashMap;

    fn down(name: lasso::Spur) -> Index {
        Index {
            name,
            variance: Variance::Down,
            index_type: None,
        }
    }

    fn up(name: lasso::Spur) -> Index {
        Index {
            name,
            variance: Variance::Up,
            index_type: None,
        }
    }

    fn row_symmetry() -> TensorProperty {
        TensorProperty::TableauSymmetry(TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape: vec![2],
                slot_map: vec![0, 1],
                duality: ax_ir::DualityKind::None,
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        })
    }

    fn col_symmetry() -> TensorProperty {
        TensorProperty::TableauSymmetry(TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape: vec![1, 1],
                slot_map: vec![0, 1],
                trace_free: false,
                duality: ax_ir::DualityKind::None,
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        })
    }

    #[test]
    fn symmetric_factor_slots_canonicalize_to_same_result() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let ab = Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(a), down(b)]);
        let ba = Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(b), down(a)]);
        let props = vec![row_symmetry()];

        let lhs = canonicalize_indexed_factor_modulo_symmetry_and_dummies(&ab, &props)
            .ok()
            .map(|result| result.expr);
        let rhs = canonicalize_indexed_factor_modulo_symmetry_and_dummies(&ba, &props)
            .ok()
            .map(|result| result.expr);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn antisymmetric_factor_slots_track_negative_sign() {
        let interner = Interner::new();
        let f = interner.get_or_intern("F");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let ab = Expr::Indexed(Box::new(Expr::Sym(f)), vec![down(a), down(b)]);
        let ba = Expr::Indexed(Box::new(Expr::Sym(f)), vec![down(b), down(a)]);
        let props = vec![col_symmetry()];

        let canonical_ab = canonicalize_indexed_factor_modulo_symmetry_and_dummies(&ab, &props)
            .ok()
            .map(|result| result.expr);
        let canonical_ba = canonicalize_indexed_factor_modulo_symmetry_and_dummies(&ba, &props);
        assert_eq!(canonical_ba.ok().map(|result| result.expr), canonical_ab);
        assert_eq!(
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(&ba, &props)
                .ok()
                .map(|result| result.sign),
            Some(-1)
        );
    }

    #[test]
    fn dummy_labels_canonicalize_deterministically() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let i = interner.get_or_intern("i");
        let j = interner.get_or_intern("j");
        let lhs = Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(j), down(i), up(j), up(i)]);
        let rhs = Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(i), down(j), up(i), up(j)]);

        assert_eq!(
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(&lhs, &[])
                .ok()
                .map(|result| result.expr),
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(&rhs, &[])
                .ok()
                .map(|result| result.expr)
        );
    }

    #[test]
    fn alpha_equivalent_products_match_after_dummy_renaming() {
        let interner = Interner::new();
        let f = interner.get_or_intern("F");
        let g = interner.get_or_intern("G");
        let i = interner.get_or_intern("i");
        let j = interner.get_or_intern("j");
        let a = interner.get_or_intern("a");
        let left = Expr::mul(vec![
            Expr::Indexed(Box::new(Expr::Sym(f)), vec![down(i), down(a)]),
            Expr::Indexed(Box::new(Expr::Sym(g)), vec![up(i)]),
        ]);
        let right = Expr::mul(vec![
            Expr::Indexed(Box::new(Expr::Sym(f)), vec![down(j), down(a)]),
            Expr::Indexed(Box::new(Expr::Sym(g)), vec![up(j)]),
        ]);

        let props = |_sym| Vec::<TensorProperty>::new();
        assert_eq!(
            alpha_equivalent_modulo_dummies_and_symmetry(&left, &right, &props)
                .ok()
                .as_ref(),
            Some(&true)
        );
    }

    #[test]
    fn direct_factor_path_rejects_sum_expression() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let expr = Expr::add(vec![
            Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(a)]),
            Expr::Indexed(Box::new(Expr::Sym(t)), vec![up(a)]),
        ]);
        assert!(matches!(
            canonicalize_indexed_factor_modulo_symmetry_and_dummies(&expr, &[]),
            Err(DummyCanonError::UnsupportedExpr)
        ));
    }

    #[test]
    fn product_canonicalization_is_available_from_public_lookup_closure() {
        let interner = Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let expr = Expr::mul(vec![
            Expr::Indexed(Box::new(Expr::Sym(t)), vec![down(b), down(a)]),
            Expr::Indexed(Box::new(Expr::Sym(t)), vec![up(a), up(b)]),
        ]);
        let mut props = HashMap::new();
        props.insert(t, vec![row_symmetry()]);
        let lookup = |sym| props.get(&sym).cloned().unwrap_or_default();

        assert!(canonicalize_product_modulo_symmetry_and_dummies(&expr, &lookup).is_ok());
    }
}
