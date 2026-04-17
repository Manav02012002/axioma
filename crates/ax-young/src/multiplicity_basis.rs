use crate::{
    partition::YoungDiagram,
    rep_ring::{multiply_schur_expansions, SchurExpansion},
    YoungError,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use num_traits::ToPrimitive;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicityBasisVector {
    pub label: String,
    pub path: Vec<YoungDiagram>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicityBasis {
    pub target: YoungDiagram,
    pub vectors: Vec<MultiplicityBasisVector>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssociationConvention {
    LeftAssociated,
    RightAssociated,
}

pub fn canonical_multiplicity_basis(
    factors: &[YoungDiagram],
    target: &YoungDiagram,
    convention: AssociationConvention,
) -> Result<MultiplicityBasis, YoungError> {
    let multiplicity = target_multiplicity(factors, target)?;
    if multiplicity == 0 {
        return Ok(MultiplicityBasis {
            target: target.clone(),
            vectors: Vec::new(),
        });
    }
    ensure_supported_regime(factors, target, multiplicity)?;

    let mut paths = match convention {
        AssociationConvention::LeftAssociated => left_associated_paths(target),
        AssociationConvention::RightAssociated => right_associated_paths(target),
    };
    paths.sort_by(|lhs, rhs| {
        lhs.iter()
            .map(|shape| shape.rows.clone())
            .collect::<Vec<_>>()
            .cmp(&rhs.iter().map(|shape| shape.rows.clone()).collect::<Vec<_>>())
    });

    Ok(MultiplicityBasis {
        target: target.clone(),
        vectors: paths
            .into_iter()
            .enumerate()
            .map(|(index, path)| MultiplicityBasisVector {
                label: format!("m{index}"),
                path,
            })
            .collect(),
    })
}

pub fn basis_change_matrix_between_associations(
    factors: &[YoungDiagram],
    target: &YoungDiagram,
) -> Result<Vec<Vec<BigRational>>, YoungError> {
    let multiplicity = target_multiplicity(factors, target)?;
    if multiplicity == 0 {
        return Ok(Vec::new());
    }
    if multiplicity == 1 {
        ensure_supported_regime(factors, target, multiplicity)?;
        return Ok(vec![vec![BigRational::one()]]);
    }
    ensure_supported_regime(factors, target, multiplicity)?;

    let left = canonical_multiplicity_basis(factors, target, AssociationConvention::LeftAssociated)?;
    let right =
        canonical_multiplicity_basis(factors, target, AssociationConvention::RightAssociated)?;
    validate_same_vector_count(left.vectors.len(), right.vectors.len())?;

    // In the supported [1]⊗[1]⊗[1] -> [2,1] regime, the multiplicity space is the
    // standard Specht module of S3. The left-associated basis diagonalizes s1=(12),
    // the right-associated basis diagonalizes s2=(23). In a common rational ambient
    // basis, these canonical coupling bases are:
    // left:  v_{[1,1]}=(1,0), v_{[2]}=(1,2)
    // right: w_{[1,1]}=(0,1), w_{[2]}=(2,1)
    // Sorting by path gives [ [1,1],[2,1] ], [ [2],[2,1] ] on both sides.
    Ok(vec![
        vec![
            BigRational::new(BigInt::from(-1), BigInt::from(2usize)),
            BigRational::new(BigInt::from(1usize), BigInt::from(2usize)),
        ],
        vec![
            BigRational::new(BigInt::from(1usize), BigInt::from(2usize)),
            BigRational::new(BigInt::from(3usize), BigInt::from(2usize)),
        ],
    ])
}

pub fn multiplicity_basis_trace(
    factors: &[YoungDiagram],
    target: &YoungDiagram,
) -> Result<ax_trace::MultiplicityBasisTrace, YoungError> {
    let left = canonical_multiplicity_basis(factors, target, AssociationConvention::LeftAssociated)?;
    let right =
        canonical_multiplicity_basis(factors, target, AssociationConvention::RightAssociated)?;
    validate_same_vector_count(left.vectors.len(), right.vectors.len())?;
    let matrix = basis_change_matrix_between_associations(factors, target)?;
    validate_square_matrix(&matrix)?;

    Ok(ax_trace::MultiplicityBasisTrace {
        factors: factors.iter().map(|shape| shape.rows.clone()).collect(),
        target: target.rows.clone(),
        left_associated_basis: left.vectors.into_iter().map(|vector| vector.label).collect(),
        right_associated_basis: right.vectors.into_iter().map(|vector| vector.label).collect(),
        change_of_basis_matrix: matrix,
    })
}

fn ensure_supported_regime(
    factors: &[YoungDiagram],
    target: &YoungDiagram,
    multiplicity: usize,
) -> Result<(), YoungError> {
    let supported_factors = factors.len() == 3 && factors.iter().all(|shape| shape.rows == vec![1]);
    let supported_target =
        target.rows == vec![3] || target.rows == vec![2, 1] || target.rows == vec![1, 1, 1];
    if supported_factors && supported_target && multiplicity <= 2 {
        return Ok(());
    }
    Err(YoungError::MultiplicityBasisUnsupported {
        factors: factors.iter().map(|shape| shape.rows.clone()).collect(),
        target: target.rows.clone(),
    })
}

fn target_multiplicity(factors: &[YoungDiagram], target: &YoungDiagram) -> Result<usize, YoungError> {
    if factors.is_empty() {
        return Ok(0);
    }
    let mut current = SchurExpansion::from_shape(factors[0].clone());
    for factor in &factors[1..] {
        current = multiply_schur_expansions(&current, &SchurExpansion::from_shape(factor.clone()))?;
    }
    Ok(current
        .coefficient(target)
        .to_usize()
        .unwrap_or(0usize))
}

fn left_associated_paths(target: &YoungDiagram) -> Vec<Vec<YoungDiagram>> {
    match target.rows.as_slice() {
        [3] => vec![vec![
            YoungDiagram::try_new(vec![2]).ok().unwrap_or_else(|| unreachable!()),
            target.clone(),
        ]],
        [2, 1] => vec![
            vec![
                YoungDiagram::try_new(vec![1, 1]).ok().unwrap_or_else(|| unreachable!()),
                target.clone(),
            ],
            vec![
                YoungDiagram::try_new(vec![2]).ok().unwrap_or_else(|| unreachable!()),
                target.clone(),
            ],
        ],
        [1, 1, 1] => vec![vec![
            YoungDiagram::try_new(vec![1, 1]).ok().unwrap_or_else(|| unreachable!()),
            target.clone(),
        ]],
        _ => Vec::new(),
    }
}

fn right_associated_paths(target: &YoungDiagram) -> Vec<Vec<YoungDiagram>> {
    // For [1]⊗[1]⊗[1], the intermediate irreps for the last two factors are the same set.
    left_associated_paths(target)
}

fn validate_square_matrix(matrix: &[Vec<BigRational>]) -> Result<(), YoungError> {
    if matrix.is_empty() {
        return Ok(());
    }
    let cols = matrix[0].len();
    if matrix.iter().any(|row| row.len() != cols) || matrix.len() != cols {
        return Err(YoungError::BasisMatrixNotSquare {
            rows: matrix.len(),
            cols,
        });
    }
    Ok(())
}

fn validate_same_vector_count(expected: usize, actual: usize) -> Result<(), YoungError> {
    if expected != actual {
        return Err(YoungError::BasisVectorCountMismatch { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    fn yd(rows: &[usize]) -> YoungDiagram {
        YoungDiagram::try_new(rows.to_vec()).unwrap()
    }

    #[test]
    fn left_associated_basis_for_three_vectors_and_two_one_is_exact() {
        let basis = canonical_multiplicity_basis(
            &[yd(&[1]), yd(&[1]), yd(&[1])],
            &yd(&[2, 1]),
            AssociationConvention::LeftAssociated,
        )
        .unwrap();

        assert_eq!(
            basis.vectors
                .iter()
                .map(|vector| vector.label.clone())
                .collect::<Vec<_>>(),
            vec!["m0".to_string(), "m1".to_string()]
        );
    }

    #[test]
    fn right_associated_basis_for_three_vectors_and_two_one_is_exact() {
        let basis = canonical_multiplicity_basis(
            &[yd(&[1]), yd(&[1]), yd(&[1])],
            &yd(&[2, 1]),
            AssociationConvention::RightAssociated,
        )
        .unwrap();

        assert_eq!(
            basis.vectors
                .iter()
                .map(|vector| vector.label.clone())
                .collect::<Vec<_>>(),
            vec!["m0".to_string(), "m1".to_string()]
        );
    }

    #[test]
    fn change_of_basis_matrix_for_three_vectors_and_two_one_is_nontrivial() {
        let matrix =
            basis_change_matrix_between_associations(&[yd(&[1]), yd(&[1]), yd(&[1])], &yd(&[2, 1]))
                .unwrap();

        assert_eq!(matrix.len(), 2);
        assert!(matrix.iter().all(|row| row.len() == 2));
        assert!(matrix.iter().flatten().any(|entry| !entry.is_zero()));
        assert_ne!(matrix, vec![vec![BigRational::one(), BigRational::zero()], vec![BigRational::zero(), BigRational::one()]]);
    }

    #[test]
    fn unsupported_four_factor_regime_reports_exact_error() {
        assert_eq!(
            canonical_multiplicity_basis(
                &[yd(&[1]), yd(&[1]), yd(&[1]), yd(&[1])],
                &yd(&[2, 1, 1]),
                AssociationConvention::LeftAssociated,
            ),
            Err(YoungError::MultiplicityBasisUnsupported {
                factors: vec![vec![1], vec![1], vec![1], vec![1]],
                target: vec![2, 1, 1],
            })
        );
    }
}
