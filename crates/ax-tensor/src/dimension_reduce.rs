use ax_ir::{Expr, TensorProperty};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DimensionReduceError {
    #[error("dimension-aware reduction requires an ambient dimension")]
    MissingAmbientDimension,
    #[error("dimension-aware reduction currently supports only indexed tensor factors and additive/multiplicative combinations on this path")]
    UnsupportedExpr,
}

pub fn tensor_annihilates_from_properties(properties: &[TensorProperty], dim: usize) -> bool {
    let structured = properties
        .iter()
        .filter_map(|property| match property {
            TensorProperty::TableauSymmetry(symmetry) => Some(symmetry),
            _ => None,
        })
        .collect::<Vec<_>>();

    if structured.is_empty() {
        return false;
    }

    structured
        .into_iter()
        .any(|symmetry| ax_young::tensor_symmetry_annihilates_in_dimension(symmetry, dim))
}

pub fn reduce_expr_by_dimension(
    expr: &Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<TensorProperty>,
    dim: Option<usize>,
) -> Result<Expr, DimensionReduceError> {
    let dim = dim.ok_or(DimensionReduceError::MissingAmbientDimension)?;
    reduce(expr, properties_for_symbol, dim)
}

fn reduce(
    expr: &Expr,
    properties_for_symbol: &dyn Fn(lasso::Spur) -> Vec<TensorProperty>,
    dim: usize,
) -> Result<Expr, DimensionReduceError> {
    match expr {
        Expr::Add(terms) => Ok(Expr::add(
            terms.iter()
                .map(|term| reduce(term, properties_for_symbol, dim))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Expr::Mul(factors) => Ok(Expr::mul(
            factors
                .iter()
                .map(|factor| reduce(factor, properties_for_symbol, dim))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Expr::Neg(inner) => Ok(Expr::neg(reduce(inner, properties_for_symbol, dim)?)),
        Expr::Indexed(base, _) => match base.as_ref() {
            Expr::Sym(symbol) => {
                if tensor_annihilates_from_properties(&properties_for_symbol(*symbol), dim) {
                    Ok(Expr::zero())
                } else {
                    Ok(expr.clone())
                }
            }
            _ => Err(DimensionReduceError::UnsupportedExpr),
        },
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => Ok(expr.clone()),
        _ => Err(DimensionReduceError::UnsupportedExpr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{
        DualityKind, Index, RestrictedSymmetryMode, SymmetrySource, TableauAttachment,
        TensorSymmetry, Variance,
    };
    use lasso::Key;

    fn indexed_expr(symbol: lasso::Spur, count: usize) -> Expr {
        Expr::Indexed(
            Box::new(Expr::Sym(symbol)),
            (0..count)
                .map(|index| Index {
                    name: lasso::Spur::try_from_usize(index).unwrap(),
                    variance: Variance::Down,
                    index_type: None,
                })
                .collect(),
        )
    }

    fn form_property(rank: usize) -> TensorProperty {
        TensorProperty::TableauSymmetry(TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![1; rank],
                slot_map: (0..rank).collect(),
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: DualityKind::None,
                restricted_mode: RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        })
    }

    #[test]
    fn rank_three_column_form_annihilates_in_dimension_two() {
        let symbol = lasso::Spur::try_from_usize(100).unwrap();
        let expr = indexed_expr(symbol, 3);
        let reduced = reduce_expr_by_dimension(
            &expr,
            &|name| {
                if name == symbol {
                    vec![form_property(3)]
                } else {
                    vec![]
                }
            },
            Some(2),
        )
        .unwrap();
        assert_eq!(reduced, Expr::zero());
    }

    #[test]
    fn rank_three_column_form_survives_in_dimension_three() {
        let symbol = lasso::Spur::try_from_usize(101).unwrap();
        let expr = indexed_expr(symbol, 3);
        let reduced = reduce_expr_by_dimension(
            &expr,
            &|name| {
                if name == symbol {
                    vec![form_property(3)]
                } else {
                    vec![]
                }
            },
            Some(3),
        )
        .unwrap();
        assert_eq!(reduced, expr);
    }

    #[test]
    fn missing_dimension_is_an_error() {
        let expr = Expr::zero();
        assert!(matches!(
            reduce_expr_by_dimension(&expr, &|_| vec![], None),
            Err(DimensionReduceError::MissingAmbientDimension)
        ));
    }
}
