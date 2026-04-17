use ax_ir::{Expr, PermutedCoefficientTerm, TensorMultitermIdentity};
use ax_perm::{enumerate_subgroup, product};
use lasso::Key;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MultitermError {
    #[error("multiterm reduction supports only single indexed tensor factors on this path")]
    UnsupportedExpr,
    #[error("multiterm identity expansion failed: {0}")]
    IdentityExpansion(#[from] anyhow::Error),
    #[error("no applicable multiterm identity found for tensor factor")]
    NoApplicableIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultitermReductionResult {
    pub expr: ax_ir::Expr,
    pub trace: ax_trace::MultitermReductionTrace,
}

/// Reduce a single indexed factor by orienting one transformed identity instance so the
/// lexicographically largest slot arrangement in the generated orbit is the pivot term.
///
/// The reduction only fires when `expr` is already that pivot representative. This makes the
/// normal form deterministic: every successful rewrite strictly replaces the orbit maximum by a
/// lexicographically smaller linear combination, and non-pivot orbit elements are left unchanged.
pub fn reduce_indexed_factor_modulo_identity(
    expr: &ax_ir::Expr,
    identity: &ax_ir::TensorMultitermIdentity,
) -> Result<MultitermReductionResult, MultitermError> {
    let Expr::Indexed(base, slots) = expr else {
        return Err(MultitermError::UnsupportedExpr);
    };
    if !identity_applies_to_arity(identity, slots.len()) {
        return Err(MultitermError::NoApplicableIdentity);
    }

    let terms = ax_ir::identity_permutation_terms(identity, slots.len())?;
    let subgroup = enumerate_subgroup(
        &terms
            .iter()
            .map(|term| term.permutation.clone())
            .collect::<Vec<_>>(),
        slots.len(),
    );
    let orbit = subgroup
        .iter()
        .map(|perm| apply_permutation_to_slots(perm, slots))
        .collect::<Vec<_>>();
    let Some(pivot_slots) = orbit
        .iter()
        .max_by_key(|slot_vector| slots_sort_key(slot_vector))
        .cloned()
    else {
        return Err(MultitermError::NoApplicableIdentity);
    };
    if *slots != pivot_slots {
        return Err(MultitermError::NoApplicableIdentity);
    }

    let mut relations = subgroup
        .iter()
        .map(|group_perm| relation_instance(&terms, group_perm, slots))
        .collect::<Vec<_>>();
    relations.sort_by(|lhs, rhs| lhs.support_keys.cmp(&rhs.support_keys));

    let pivot_key = slots_sort_key(&pivot_slots);
    let relation = relations
        .into_iter()
        .find(|candidate| candidate.coefficients.contains_key(&pivot_key))
        .ok_or(MultitermError::NoApplicableIdentity)?;
    let pivot_coeff = relation
        .coefficients
        .get(&pivot_key)
        .map(|term| term.coefficient.clone())
        .filter(|coeff| !coeff.is_zero())
        .ok_or(MultitermError::NoApplicableIdentity)?;

    let mut reduced_terms = relation
        .coefficients
        .into_iter()
        .filter(|(key, _)| *key != pivot_key)
        .map(|(_, term)| (term.slots, -term.coefficient / pivot_coeff.clone()))
        .filter(|(_, coeff)| !coeff.is_zero())
        .collect::<Vec<_>>();
    reduced_terms.sort_by_key(|(slot_vector, _)| slots_sort_key(slot_vector));

    let expr = Expr::add(
        reduced_terms
            .iter()
            .map(|(slot_vector, coeff)| {
                multiply_expr_by_rational(
                    Expr::Indexed(base.clone(), slot_vector.clone()),
                    coeff.clone(),
                )
            })
            .collect(),
    );

    Ok(MultitermReductionResult {
        expr,
        trace: ax_trace::MultitermReductionTrace {
            original_slots: slot_strings(slots),
            pivot_slots: slot_strings(&pivot_slots),
            reduced_term_count: reduced_terms.len(),
            identity_kind: identity_kind(identity).into(),
        },
    })
}

pub fn reduce_indexed_factor_modulo_identities(
    expr: &ax_ir::Expr,
    identities: &ax_ir::TensorIdentitySet,
) -> Result<Option<MultitermReductionResult>, MultitermError> {
    let arity = match expr {
        Expr::Indexed(_, slots) => slots.len(),
        _ => return Err(MultitermError::UnsupportedExpr),
    };
    for identity in &identities.multiterm {
        if !identity_applies_to_arity(identity, arity) {
            continue;
        }
        if let Ok(result) = reduce_indexed_factor_modulo_identity(expr, identity) {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

pub fn identity_applies_to_arity(identity: &ax_ir::TensorMultitermIdentity, arity: usize) -> bool {
    match identity {
        TensorMultitermIdentity::FirstBianchi { cyclic_slots } => {
            cyclic_slots.iter().all(|slot| *slot < arity)
        }
        TensorMultitermIdentity::CyclicSum { slots }
        | TensorMultitermIdentity::AlternatingSum { slots } => {
            slots.iter().all(|slot| *slot < arity)
        }
        TensorMultitermIdentity::LinearCombination { terms } => {
            terms.iter().all(|term| term.permutation.len() == arity)
        }
    }
}

struct RelationInstance {
    support_keys: Vec<Vec<SlotKey>>,
    coefficients: BTreeMap<Vec<SlotKey>, RelationTerm>,
}

#[derive(Clone)]
struct RelationTerm {
    slots: Vec<ax_ir::Index>,
    coefficient: BigRational,
}

fn relation_instance(
    terms: &[PermutedCoefficientTerm],
    group_perm: &[usize],
    slots: &[ax_ir::Index],
) -> RelationInstance {
    let mut coefficients: BTreeMap<Vec<SlotKey>, RelationTerm> = BTreeMap::new();
    for term in terms {
        let composed = product(&term.permutation, group_perm);
        let slot_vector = apply_permutation_to_slots(&composed, slots);
        let key = slots_sort_key(&slot_vector);
        let coeff = BigRational::new(
            BigInt::from(term.coefficient_numer),
            BigInt::from(term.coefficient_denom),
        );
        if let Some(existing) = coefficients.get_mut(&key) {
            existing.coefficient += coeff;
        } else {
            coefficients.insert(
                key,
                RelationTerm {
                    slots: slot_vector,
                    coefficient: coeff,
                },
            );
        }
    }
    let support_keys = coefficients.keys().cloned().collect();
    RelationInstance {
        support_keys,
        coefficients,
    }
}

fn apply_permutation_to_slots(perm: &[usize], slots: &[ax_ir::Index]) -> Vec<ax_ir::Index> {
    perm.iter().map(|idx| slots[*idx].clone()).collect()
}

fn slot_strings(slots: &[ax_ir::Index]) -> Vec<String> {
    slots
        .iter()
        .map(|slot| slot.name.into_usize().to_string())
        .collect()
}

type SlotKey = (usize, u8, Option<usize>);

fn slots_sort_key(slots: &[ax_ir::Index]) -> Vec<SlotKey> {
    slots.iter().map(slot_sort_key).collect()
}

fn slot_sort_key(slot: &ax_ir::Index) -> SlotKey {
    (
        slot.name.into_usize(),
        match slot.variance {
            ax_ir::Variance::Up => 0,
            ax_ir::Variance::Down => 1,
        },
        slot.index_type.map(|family| family.into_usize()),
    )
}

fn multiply_expr_by_rational(expr: Expr, coeff: BigRational) -> Expr {
    if coeff.is_zero() {
        Expr::zero()
    } else if coeff.is_one() {
        expr
    } else {
        Expr::mul(vec![Expr::Rational(coeff), expr])
    }
}

fn identity_kind(identity: &TensorMultitermIdentity) -> &'static str {
    match identity {
        TensorMultitermIdentity::FirstBianchi { .. } => "FirstBianchi",
        TensorMultitermIdentity::CyclicSum { .. } => "CyclicSum",
        TensorMultitermIdentity::AlternatingSum { .. } => "AlternatingSum",
        TensorMultitermIdentity::LinearCombination { .. } => "LinearCombination",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::TensorIdentitySet;

    fn idx(name: lasso::Spur) -> ax_ir::Index {
        ax_ir::Index {
            name,
            variance: ax_ir::Variance::Down,
            index_type: None,
        }
    }

    fn indexed(symbol: lasso::Spur, slots: &[lasso::Spur]) -> Expr {
        Expr::Indexed(
            Box::new(Expr::Sym(symbol)),
            slots.iter().copied().map(idx).collect(),
        )
    }

    #[test]
    fn first_bianchi_reduction_solves_for_lexicographically_largest_term() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");
        let expr = indexed(r, &[a, d, b, c]);

        let reduced = reduce_indexed_factor_modulo_identity(
            &expr,
            &TensorMultitermIdentity::FirstBianchi {
                cyclic_slots: [1, 2, 3],
            },
        )
        .ok();

        assert_eq!(
            reduced.as_ref().map(|result| result.expr.clone()),
            Some(Expr::add(vec![
                Expr::neg(indexed(r, &[a, b, c, d])),
                Expr::neg(indexed(r, &[a, c, d, b])),
            ]))
        );
        assert_eq!(
            reduced
                .as_ref()
                .map(|result| result.trace.reduced_term_count),
            Some(2)
        );
    }

    #[test]
    fn cyclic_sum_reduction_solves_for_lexicographically_largest_term() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let expr = indexed(t, &[c, a, b]);

        let reduced = reduce_indexed_factor_modulo_identity(
            &expr,
            &TensorMultitermIdentity::CyclicSum {
                slots: vec![0, 1, 2],
            },
        )
        .ok();

        assert_eq!(
            reduced.as_ref().map(|result| result.expr.clone()),
            Some(Expr::add(vec![
                Expr::neg(indexed(t, &[a, b, c])),
                Expr::neg(indexed(t, &[b, c, a])),
            ]))
        );
    }

    #[test]
    fn identity_reduction_returns_none_when_arity_does_not_apply() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let expr = indexed(t, &[a, b, c]);
        let identities = TensorIdentitySet {
            multiterm: vec![TensorMultitermIdentity::FirstBianchi {
                cyclic_slots: [1, 2, 3],
            }],
        };

        assert_eq!(
            reduce_indexed_factor_modulo_identities(&expr, &identities).ok(),
            Some(None)
        );
    }

    #[test]
    fn repeated_identity_reduction_is_deterministic() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let expr = indexed(t, &[c, a, b]);
        let properties = |symbol| {
            (symbol == t)
                .then_some(vec![ax_ir::TensorProperty::TensorIdentities(
                    TensorIdentitySet {
                        multiterm: vec![TensorMultitermIdentity::CyclicSum {
                            slots: vec![0, 1, 2],
                        }],
                    },
                )])
                .unwrap_or_default()
        };

        let once = crate::reduce_modulo_tensor_identities(&expr, &properties).ok();
        let twice = once
            .as_ref()
            .and_then(|reduced| crate::reduce_modulo_tensor_identities(reduced, &properties).ok());

        assert_eq!(once, twice);
    }
}
