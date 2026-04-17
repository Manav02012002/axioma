use ax_trace::CurvatureDecompositionTrace;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum CurvatureDecomposeError {
    #[error("curvature decomposition requires dimension at least 3, got {dim}")]
    DimensionTooSmall { dim: usize },
    #[error(
        "curvature decomposition supports only single abstract curvature tensors on this path"
    )]
    UnsupportedExpr,
    #[error("curvature decomposition coefficient overflowed i64 representation")]
    CoefficientOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearDecompositionTerm {
    pub kind: String,
    pub coefficient_numer: i64,
    pub coefficient_denom: i64,
}

fn as_i64(value: usize) -> Result<i64, CurvatureDecomposeError> {
    i64::try_from(value).map_err(|_| CurvatureDecomposeError::CoefficientOverflow)
}

fn checked_mul_i64(left: i64, right: i64) -> Result<i64, CurvatureDecomposeError> {
    left.checked_mul(right)
        .ok_or(CurvatureDecomposeError::CoefficientOverflow)
}

pub fn symmetric_rank2_trace_decomposition(
    dim: usize,
) -> Result<Vec<LinearDecompositionTerm>, CurvatureDecomposeError> {
    Ok(vec![
        LinearDecompositionTerm {
            kind: "traceless_rank2".to_string(),
            coefficient_numer: 1,
            coefficient_denom: 1,
        },
        LinearDecompositionTerm {
            kind: "metric_trace_rank2".to_string(),
            coefficient_numer: 1,
            coefficient_denom: as_i64(dim)?,
        },
    ])
}

pub fn schouten_from_ricci_coefficients(
    dim: usize,
) -> Result<Vec<LinearDecompositionTerm>, CurvatureDecomposeError> {
    if dim < 3 {
        return Err(CurvatureDecomposeError::DimensionTooSmall { dim });
    }
    let dim_minus_two = as_i64(dim - 2)?;
    let dim_minus_one = as_i64(dim - 1)?;
    let scalar_denom = checked_mul_i64(2, checked_mul_i64(dim_minus_one, dim_minus_two)?)?;

    Ok(vec![
        LinearDecompositionTerm {
            kind: "ricci_rank2".to_string(),
            coefficient_numer: 1,
            coefficient_denom: dim_minus_two,
        },
        LinearDecompositionTerm {
            kind: "scalar_metric_rank2".to_string(),
            coefficient_numer: -1,
            coefficient_denom: scalar_denom,
        },
    ])
}

pub fn riemann_to_weyl_ricci_scalar_coefficients(
    dim: usize,
) -> Result<Vec<LinearDecompositionTerm>, CurvatureDecomposeError> {
    if dim < 3 {
        return Err(CurvatureDecomposeError::DimensionTooSmall { dim });
    }
    let dim_minus_two = as_i64(dim - 2)?;
    let dim_minus_one = as_i64(dim - 1)?;
    let scalar_denom = checked_mul_i64(dim_minus_one, dim_minus_two)?;

    Ok(vec![
        LinearDecompositionTerm {
            kind: "weyl_rank4".to_string(),
            coefficient_numer: 1,
            coefficient_denom: 1,
        },
        LinearDecompositionTerm {
            kind: "metric_ricci_rank4".to_string(),
            coefficient_numer: 1,
            coefficient_denom: dim_minus_two,
        },
        LinearDecompositionTerm {
            kind: "metric_scalar_rank4".to_string(),
            coefficient_numer: -1,
            coefficient_denom: scalar_denom,
        },
    ])
}

pub fn curvature_decomposition_trace(
    dim: usize,
    input_kind: &str,
) -> Result<CurvatureDecompositionTrace, CurvatureDecomposeError> {
    let terms = decompose_curvature_symbolically(input_kind, dim)?;
    Ok(CurvatureDecompositionTrace {
        dimension: dim,
        input_kind: input_kind.to_string(),
        output_kinds: terms.iter().map(|term| term.kind.clone()).collect(),
        coefficient_numerators: terms.iter().map(|term| term.coefficient_numer).collect(),
        coefficient_denominators: terms.iter().map(|term| term.coefficient_denom).collect(),
    })
}

pub fn decompose_curvature_symbolically(
    input_kind: &str,
    dim: usize,
) -> Result<Vec<LinearDecompositionTerm>, CurvatureDecomposeError> {
    match input_kind {
        "symmetric_rank2" => symmetric_rank2_trace_decomposition(dim),
        "schouten_from_ricci" => schouten_from_ricci_coefficients(dim),
        "riemann_rank4" => riemann_to_weyl_ricci_scalar_coefficients(dim),
        _ => Err(CurvatureDecomposeError::UnsupportedExpr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_rank2_decomposition_is_exact() {
        assert_eq!(
            symmetric_rank2_trace_decomposition(4).unwrap(),
            vec![
                LinearDecompositionTerm {
                    kind: "traceless_rank2".to_string(),
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                LinearDecompositionTerm {
                    kind: "metric_trace_rank2".to_string(),
                    coefficient_numer: 1,
                    coefficient_denom: 4,
                },
            ]
        );
    }

    #[test]
    fn schouten_coefficients_are_exact() {
        assert_eq!(
            schouten_from_ricci_coefficients(4).unwrap(),
            vec![
                LinearDecompositionTerm {
                    kind: "ricci_rank2".to_string(),
                    coefficient_numer: 1,
                    coefficient_denom: 2,
                },
                LinearDecompositionTerm {
                    kind: "scalar_metric_rank2".to_string(),
                    coefficient_numer: -1,
                    coefficient_denom: 12,
                },
            ]
        );
    }

    #[test]
    fn riemann_decomposition_is_exact() {
        assert_eq!(
            riemann_to_weyl_ricci_scalar_coefficients(4).unwrap(),
            vec![
                LinearDecompositionTerm {
                    kind: "weyl_rank4".to_string(),
                    coefficient_numer: 1,
                    coefficient_denom: 1,
                },
                LinearDecompositionTerm {
                    kind: "metric_ricci_rank4".to_string(),
                    coefficient_numer: 1,
                    coefficient_denom: 2,
                },
                LinearDecompositionTerm {
                    kind: "metric_scalar_rank4".to_string(),
                    coefficient_numer: -1,
                    coefficient_denom: 6,
                },
            ]
        );
    }

    #[test]
    fn riemann_decomposition_rejects_dimension_two() {
        assert_eq!(
            riemann_to_weyl_ricci_scalar_coefficients(2),
            Err(CurvatureDecomposeError::DimensionTooSmall { dim: 2 })
        );
    }

    #[test]
    fn curvature_trace_records_riemann_outputs() {
        let trace = curvature_decomposition_trace(4, "riemann_rank4").unwrap();
        assert_eq!(trace.dimension, 4);
        assert_eq!(trace.input_kind, "riemann_rank4");
        assert_eq!(
            trace.output_kinds,
            vec![
                "weyl_rank4".to_string(),
                "metric_ricci_rank4".to_string(),
                "metric_scalar_rank4".to_string(),
            ]
        );
    }
}
