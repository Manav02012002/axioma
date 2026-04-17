use ax_ir::{Expr, TensorProperty, TensorSymmetry, Variance};

use crate::detect_contractions;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum TraceEngineError {
    #[error(
        "trace engine supports only indexed tensor factors and simple contractions on this path"
    )]
    UnsupportedExpr,
    #[error("trace engine requires tensor properties for contraction analysis")]
    MissingProperties,
}

fn structured_symmetry(properties: &[TensorProperty]) -> Option<&TensorSymmetry> {
    properties.iter().find_map(|property| match property {
        TensorProperty::TableauSymmetry(symmetry) => Some(symmetry),
        _ => None,
    })
}

fn trace_free_from_properties(properties: &[TensorProperty]) -> bool {
    if let Some(symmetry) = structured_symmetry(properties) {
        return symmetry.any_trace_free();
    }
    properties
        .iter()
        .any(|property| matches!(property, TensorProperty::Traceless))
}

pub fn contraction_annihilated_by_trace_free_symmetry(
    symmetry: &TensorSymmetry,
    contracted_slots: &[(usize, usize)],
) -> bool {
    if !symmetry.any_trace_free() {
        return false;
    }
    let slot_count = symmetry.total_slots();
    contracted_slots
        .iter()
        .any(|(left, right)| *left < slot_count && *right < slot_count && left != right)
}

pub fn reduce_trace_free_factor_if_applicable(
    expr: &Expr,
    properties: &[TensorProperty],
) -> Result<Option<Expr>, TraceEngineError> {
    let Expr::Indexed(_, indices) = expr else {
        return Err(TraceEngineError::UnsupportedExpr);
    };
    if !trace_free_from_properties(properties) {
        return Ok(None);
    }
    if detect_contractions(indices).is_empty() {
        return Ok(None);
    }
    Ok(Some(Expr::zero()))
}

fn replace_index_name(expr: &Expr, from: lasso::Spur, to: lasso::Spur) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => Expr::Indexed(
            base.clone(),
            indices
                .iter()
                .map(|index| {
                    if index.name == from {
                        ax_ir::Index {
                            name: to,
                            variance: index.variance.clone(),
                            index_type: index.index_type,
                        }
                    } else {
                        index.clone()
                    }
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn eliminate_kronecker_on_factor(expr: &Expr) -> Result<Expr, TraceEngineError> {
    let Expr::Mul(factors) = expr else {
        return Err(TraceEngineError::UnsupportedExpr);
    };

    let indexed_positions = factors
        .iter()
        .enumerate()
        .filter(|(_, factor)| matches!(factor, Expr::Indexed(_, _)))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&tensor_position) = indexed_positions.last() else {
        return Ok(expr.clone());
    };

    let mut tensor_factor = None;
    let mut delta_factors = Vec::new();
    let mut passthrough = Vec::new();

    for (position, factor) in factors.iter().enumerate() {
        match factor {
            Expr::Indexed(_, _) if position == tensor_position => {
                tensor_factor = Some(factor.clone());
            }
            Expr::Indexed(_, indices)
                if indices.len() == 2 && indices[0].variance != indices[1].variance =>
            {
                delta_factors.push(factor.clone());
            }
            Expr::Indexed(_, _) => return Err(TraceEngineError::UnsupportedExpr),
            _ => {
                passthrough.push(factor.clone());
            }
        }
    }

    let Some(mut reduced_factor) = tensor_factor else {
        return Ok(expr.clone());
    };
    if delta_factors.is_empty() {
        return Ok(expr.clone());
    }

    let mut changed = false;
    for delta in &delta_factors {
        let Expr::Indexed(_, indices) = delta else {
            continue;
        };
        let (from, to) = match (&indices[0].variance, &indices[1].variance) {
            (Variance::Up, Variance::Down) => (indices[1].name, indices[0].name),
            (Variance::Down, Variance::Up) => (indices[0].name, indices[1].name),
            _ => continue,
        };

        let updated = replace_index_name(&reduced_factor, from, to);
        if updated != reduced_factor {
            reduced_factor = updated;
            changed = true;
        }
    }

    if !changed {
        return Ok(expr.clone());
    }

    passthrough.push(reduced_factor);
    Ok(Expr::mul(passthrough))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{
        DualityKind, Index, RestrictedSymmetryMode, SymmetrySource, TableauAttachment,
        TensorProperty, TensorSymmetry,
    };

    fn trace_free_symmetry(trace_free: bool) -> TensorSymmetry {
        TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![2, 2],
                slot_map: vec![0, 1, 2, 3],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: DualityKind::None,
                restricted_mode: RestrictedSymmetryMode::FullYoung,
                trace_free,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: Some("weyl".to_string()),
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        }
    }

    #[test]
    fn trace_free_symmetry_detects_annihilating_contraction() {
        assert!(contraction_annihilated_by_trace_free_symmetry(
            &trace_free_symmetry(true),
            &[(0, 2)]
        ));
    }

    #[test]
    fn non_trace_free_symmetry_does_not_annihilate_contraction() {
        assert!(!contraction_annihilated_by_trace_free_symmetry(
            &trace_free_symmetry(false),
            &[(0, 2)]
        ));
    }

    #[test]
    fn explicit_trace_on_trace_free_factor_reduces_to_zero() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let factor = Expr::Indexed(
            Box::new(Expr::Sym(interner.get_or_intern("W"))),
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
                Index {
                    name: a,
                    variance: Variance::Down,
                    index_type: None,
                },
                Index {
                    name: b,
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );
        let properties = vec![TensorProperty::TableauSymmetry(trace_free_symmetry(true))];

        assert_eq!(
            reduce_trace_free_factor_if_applicable(&factor, &properties).unwrap(),
            Some(Expr::zero())
        );
    }
}
