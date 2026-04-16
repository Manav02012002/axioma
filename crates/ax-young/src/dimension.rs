use crate::{
    partition::YoungDiagram,
    rep_ring::{RepExpansion, SchurExpansion},
    YoungError,
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};

pub fn hook_content_factors(
    diagram: &YoungDiagram,
    n: usize,
) -> Result<Vec<(usize, usize, BigInt)>, YoungError> {
    let mut factors = Vec::new();
    for (row, row_len) in diagram.rows.iter().copied().enumerate() {
        for col in 0..row_len {
            let content = (n as isize) + (col as isize) - (row as isize);
            factors.push((row, col, BigInt::from(content)));
        }
    }
    Ok(factors)
}

pub fn dimension_gl(diagram: &YoungDiagram, n: usize) -> Result<BigInt, YoungError> {
    let mut numerator = BigInt::one();
    for (_, _, factor) in hook_content_factors(diagram, n)? {
        numerator *= factor;
    }
    let hook_product = diagram.hook_length_product()?;
    if hook_product.is_zero() {
        return Ok(BigInt::zero());
    }
    Ok(numerator / hook_product)
}

pub fn dimension_gl_u64_saturating(diagram: &YoungDiagram, n: usize) -> Result<u64, YoungError> {
    let dimension = dimension_gl(diagram, n)?;
    Ok(dimension.to_u64().unwrap_or(u64::MAX))
}

pub fn dimension_of_representation(diagram: &YoungDiagram, n: usize) -> u64 {
    dimension_gl_u64_saturating(diagram, n).unwrap_or(0)
}

pub fn dimension_of_schur_expansion(
    expansion: &SchurExpansion,
    n: usize,
) -> Result<BigInt, YoungError> {
    let mut total = BigInt::zero();
    for (shape, coeff) in &expansion.terms {
        total += coeff * dimension_gl(shape, n)?;
    }
    Ok(total)
}

pub fn dimension_of_rep_expansion(
    expansion: &RepExpansion,
    n: usize,
) -> Result<BigInt, YoungError> {
    match expansion {
        RepExpansion::Scalar(value) => Ok(value.clone()),
        RepExpansion::Schur(expansion) => dimension_of_schur_expansion(expansion, n),
        RepExpansion::DirectSum { scalar, schur } => {
            Ok(scalar + dimension_of_schur_expansion(schur, n)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rep_ring::{RepExpansion, SchurExpansion};

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn schur_expansion_dimension_sums_exactly() {
        let expansion = SchurExpansion {
            terms: [
                (yd(vec![1, 1]), BigInt::from(2usize)),
                (yd(vec![2]), BigInt::from(1usize)),
            ]
            .into_iter()
            .collect(),
        };

        assert_eq!(
            dimension_of_schur_expansion(&expansion, 4).unwrap(),
            BigInt::from(22usize)
        );
    }

    #[test]
    fn rep_expansion_dimension_handles_scalar_and_direct_sum() {
        assert_eq!(
            dimension_of_rep_expansion(&RepExpansion::Scalar(BigInt::from(7usize)), 4).unwrap(),
            BigInt::from(7usize)
        );

        assert_eq!(
            dimension_of_rep_expansion(
                &RepExpansion::DirectSum {
                    scalar: BigInt::from(1usize),
                    schur: SchurExpansion::from_shape(yd(vec![1])),
                },
                4,
            )
            .unwrap(),
            BigInt::from(5usize)
        );
    }
}
