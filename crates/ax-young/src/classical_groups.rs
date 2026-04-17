use crate::{
    littlewood_richardson_coefficient,
    partition::YoungDiagram,
    YoungError,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassicalGroupFamily {
    OrthogonalOdd,
    OrthogonalEven,
    Symplectic,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClassicalBranchTarget {
    Scalar,
    Irrep(YoungDiagram),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassicalIrrepSummary {
    pub family: ClassicalGroupFamily,
    pub rank: usize,
    pub highest_weight: YoungDiagram,
    pub dimension: BigInt,
}

pub fn validate_orthogonal_highest_weight(
    shape: &YoungDiagram,
    rank: usize,
) -> Result<(), YoungError> {
    if rank == 0 {
        return Err(YoungError::InvalidOrthogonalRank { n: rank });
    }
    if shape.n_rows() > rank {
        return Err(YoungError::HighestWeightTooLongForOrthogonal {
            shape: shape.rows.clone(),
            rank,
        });
    }
    Ok(())
}

pub fn validate_symplectic_highest_weight(
    shape: &YoungDiagram,
    rank: usize,
) -> Result<(), YoungError> {
    if rank == 0 {
        return Err(YoungError::InvalidSymplecticRank { n: rank });
    }
    if shape.n_rows() > rank {
        return Err(YoungError::HighestWeightTooLongForSymplectic {
            shape: shape.rows.clone(),
            rank,
        });
    }
    Ok(())
}

pub fn dimension_so_odd(shape: &YoungDiagram, rank: usize) -> Result<BigInt, YoungError> {
    validate_orthogonal_highest_weight(shape, rank)?;
    let rho = (0..rank)
        .map(|index| RationalRoot::half_integer((2 * (rank - index) - 1) as i64))
        .collect::<Vec<_>>();
    let shifted = (0..rank)
        .map(|index| {
            RationalRoot::integer(i64::try_from(shape.rows.get(index).copied().unwrap_or(0)).unwrap_or(0))
                .add(&rho[index])
        })
        .collect::<Vec<_>>();
    let mut dimension = BigRational::one();

    for index in 0..rank {
        dimension *= shifted[index].to_rational() / rho[index].to_rational();
    }
    for i in 0..rank {
        for j in i + 1..rank {
            let left = shifted[i].sub(&shifted[j]).to_rational();
            let right = rho[i].sub(&rho[j]).to_rational();
            dimension *= left / right;

            let left = shifted[i].add(&shifted[j]).to_rational();
            let right = rho[i].add(&rho[j]).to_rational();
            dimension *= left / right;
        }
    }

    rational_dimension_to_bigint(dimension, "so_odd", shape, rank)
}

pub fn dimension_so_even(shape: &YoungDiagram, rank: usize) -> Result<BigInt, YoungError> {
    validate_orthogonal_highest_weight(shape, rank)?;
    if rank < 2 {
        return Err(YoungError::ClassicalGroupDimensionUnsupported {
            family: "so_even",
            shape: shape.rows.clone(),
            rank,
        });
    }
    let rho = (0..rank)
        .map(|index| RationalRoot::integer((rank - index) as i64 - 1))
        .collect::<Vec<_>>();
    let shifted = (0..rank)
        .map(|index| {
            RationalRoot::integer(i64::try_from(shape.rows.get(index).copied().unwrap_or(0)).unwrap_or(0))
                .add(&rho[index])
        })
        .collect::<Vec<_>>();
    let mut dimension = BigRational::one();

    for i in 0..rank {
        for j in i + 1..rank {
            let left = shifted[i].sub(&shifted[j]).to_rational();
            let right = rho[i].sub(&rho[j]).to_rational();
            dimension *= left / right;

            let left = shifted[i].add(&shifted[j]).to_rational();
            let right = rho[i].add(&rho[j]).to_rational();
            dimension *= left / right;
        }
    }

    rational_dimension_to_bigint(dimension, "so_even", shape, rank)
}

pub fn dimension_sp(shape: &YoungDiagram, rank: usize) -> Result<BigInt, YoungError> {
    validate_symplectic_highest_weight(shape, rank)?;
    let rho = (0..rank)
        .map(|index| RationalRoot::integer((rank - index) as i64))
        .collect::<Vec<_>>();
    let shifted = (0..rank)
        .map(|index| {
            RationalRoot::integer(i64::try_from(shape.rows.get(index).copied().unwrap_or(0)).unwrap_or(0))
                .add(&rho[index])
        })
        .collect::<Vec<_>>();
    let mut dimension = BigRational::one();

    for index in 0..rank {
        dimension *= shifted[index].to_rational() / rho[index].to_rational();
    }
    for i in 0..rank {
        for j in i + 1..rank {
            let left = shifted[i].sub(&shifted[j]).to_rational();
            let right = rho[i].sub(&rho[j]).to_rational();
            dimension *= left / right;

            let left = shifted[i].add(&shifted[j]).to_rational();
            let right = rho[i].add(&rho[j]).to_rational();
            dimension *= left / right;
        }
    }

    rational_dimension_to_bigint(dimension, "sp", shape, rank)
}

pub fn branch_gl_to_so(
    shape: &YoungDiagram,
    rank: usize,
    even: bool,
) -> Result<Vec<(ClassicalBranchTarget, BigInt)>, YoungError> {
    if rank < 2 || shape.n_rows() > 2 {
        return Err(YoungError::ClassicalBranchingUnsupported {
            family: if even { "so_even" } else { "so_odd" },
            shape: shape.rows.clone(),
            rank,
        });
    }
    validate_orthogonal_highest_weight(shape, rank)?;
    stable_even_row_branching(shape, rank, if even { "so_even" } else { "so_odd" })
}

pub fn branch_gl_to_sp(
    shape: &YoungDiagram,
    rank: usize,
) -> Result<Vec<(ClassicalBranchTarget, BigInt)>, YoungError> {
    if rank < 2 || shape.n_rows() > 2 {
        return Err(YoungError::ClassicalBranchingUnsupported {
            family: "sp",
            shape: shape.rows.clone(),
            rank,
        });
    }
    validate_symplectic_highest_weight(shape, rank)?;
    stable_even_row_branching(shape, rank, "sp")
}

pub fn summarize_classical_irrep(
    family: ClassicalGroupFamily,
    rank: usize,
    shape: &YoungDiagram,
) -> Result<ClassicalIrrepSummary, YoungError> {
    let dimension = match family {
        ClassicalGroupFamily::OrthogonalOdd => dimension_so_odd(shape, rank)?,
        ClassicalGroupFamily::OrthogonalEven => dimension_so_even(shape, rank)?,
        ClassicalGroupFamily::Symplectic => dimension_sp(shape, rank)?,
    };
    Ok(ClassicalIrrepSummary {
        family,
        rank,
        highest_weight: shape.clone(),
        dimension,
    })
}

fn stable_even_row_branching(
    shape: &YoungDiagram,
    rank: usize,
    family: &'static str,
) -> Result<Vec<(ClassicalBranchTarget, BigInt)>, YoungError> {
    let mut out = BTreeMap::<ClassicalBranchTarget, BigInt>::new();
    for target in candidate_targets(shape.n_cells()) {
        let target_size = target.as_shape().map_or(0, YoungDiagram::n_cells);
        if target_size > shape.n_cells() {
            continue;
        }
        let remainder = shape.n_cells() - target_size;
        if remainder % 2 != 0 {
            continue;
        }

        let mut multiplicity = BigInt::zero();
        if remainder == 0 {
            if matches!(target.as_shape(), Some(irrep) if irrep == shape) {
                multiplicity += BigInt::one();
            }
        }
        for even_shape in even_row_diagrams(remainder)? {
            let coeff = match target.as_shape() {
                Some(irrep) => littlewood_richardson_coefficient(irrep, &even_shape, shape)?,
                None => {
                    if even_shape == *shape {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    }
                }
            };
            multiplicity += coeff;
        }

        if !multiplicity.is_zero() {
            out.insert(target, multiplicity);
        }
    }

    let mut items = out.into_iter().collect::<Vec<_>>();
    items.sort_by(compare_branch_targets);
    if items.is_empty() {
        return Err(YoungError::ClassicalBranchingUnsupported {
            family,
            shape: shape.rows.clone(),
            rank,
        });
    }
    Ok(items)
}

fn candidate_targets(total_cells: usize) -> Vec<ClassicalBranchTarget> {
    let mut out = vec![ClassicalBranchTarget::Scalar];
    for size in 1..=total_cells {
        for first in (size.div_ceil(2)..=size).rev() {
            let second = size - first;
            if second > first {
                continue;
            }
            let rows = if second == 0 {
                vec![first]
            } else {
                vec![first, second]
            };
            if let Ok(diagram) = YoungDiagram::try_new(rows) {
                out.push(ClassicalBranchTarget::Irrep(diagram));
            }
        }
    }
    out
}

fn even_row_diagrams(total_cells: usize) -> Result<Vec<YoungDiagram>, YoungError> {
    if total_cells == 0 {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for first in (0..=total_cells / 2).rev() {
        let row0 = 2 * first;
        let remaining = total_cells - row0;
        if remaining % 2 != 0 {
            continue;
        }
        let row1 = remaining;
        if row0 == 0 {
            continue;
        }
        if row1 > row0 {
            continue;
        }
        let rows = if row1 == 0 { vec![row0] } else { vec![row0, row1] };
        out.push(YoungDiagram::try_new(rows)?);
    }
    out.sort_by(|lhs, rhs| lhs.rows.cmp(&rhs.rows));
    out.dedup();
    Ok(out)
}

fn rational_dimension_to_bigint(
    value: BigRational,
    family: &'static str,
    shape: &YoungDiagram,
    rank: usize,
) -> Result<BigInt, YoungError> {
    if value.is_integer() {
        Ok(value.to_integer())
    } else {
        Err(YoungError::ClassicalGroupDimensionUnsupported {
            family,
            shape: shape.rows.clone(),
            rank,
        })
    }
}

fn compare_branch_targets(
    left: &(ClassicalBranchTarget, BigInt),
    right: &(ClassicalBranchTarget, BigInt),
) -> Ordering {
    match (&left.0, &right.0) {
        (ClassicalBranchTarget::Scalar, ClassicalBranchTarget::Scalar) => Ordering::Equal,
        (ClassicalBranchTarget::Scalar, _) => Ordering::Less,
        (_, ClassicalBranchTarget::Scalar) => Ordering::Greater,
        (ClassicalBranchTarget::Irrep(lhs), ClassicalBranchTarget::Irrep(rhs)) => {
            lhs.rows.cmp(&rhs.rows)
        }
    }
}

#[derive(Clone, Debug)]
struct RationalRoot {
    numer: i64,
    denom: i64,
}

impl RationalRoot {
    fn integer(value: i64) -> Self {
        Self {
            numer: value,
            denom: 1,
        }
    }

    fn half_integer(odd_numer: i64) -> Self {
        Self {
            numer: odd_numer,
            denom: 2,
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            numer: self.numer * other.denom + other.numer * self.denom,
            denom: self.denom * other.denom,
        }
        .normalized()
    }

    fn sub(&self, other: &Self) -> Self {
        Self {
            numer: self.numer * other.denom - other.numer * self.denom,
            denom: self.denom * other.denom,
        }
        .normalized()
    }

    fn to_rational(&self) -> BigRational {
        BigRational::new(BigInt::from(self.numer), BigInt::from(self.denom))
    }

    fn normalized(self) -> Self {
        let gcd = gcd_i64(self.numer.abs(), self.denom.abs()).max(1);
        let mut numer = self.numer / gcd;
        let mut denom = self.denom / gcd;
        if denom < 0 {
            numer = -numer;
            denom = -denom;
        }
        Self { numer, denom }
    }
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.abs()
}

impl ClassicalBranchTarget {
    fn as_shape(&self) -> Option<&YoungDiagram> {
        match self {
            Self::Scalar => None,
            Self::Irrep(shape) => Some(shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yd(rows: &[usize]) -> YoungDiagram {
        YoungDiagram::try_new(rows.to_vec()).unwrap()
    }

    #[test]
    fn orthogonal_highest_weight_validation_accepts_rank_compatible_shape() {
        assert_eq!(validate_orthogonal_highest_weight(&yd(&[2, 1]), 2), Ok(()));
    }

    #[test]
    fn orthogonal_highest_weight_validation_rejects_too_many_rows() {
        assert_eq!(
            validate_orthogonal_highest_weight(&yd(&[1, 1, 1]), 2),
            Err(YoungError::HighestWeightTooLongForOrthogonal {
                shape: vec![1, 1, 1],
                rank: 2,
            })
        );
    }

    #[test]
    fn symplectic_highest_weight_validation_accepts_rank_compatible_shape() {
        assert_eq!(validate_symplectic_highest_weight(&yd(&[2, 1]), 2), Ok(()));
    }

    #[test]
    fn symplectic_highest_weight_validation_rejects_too_many_rows() {
        assert_eq!(
            validate_symplectic_highest_weight(&yd(&[1, 1, 1]), 2),
            Err(YoungError::HighestWeightTooLongForSymplectic {
                shape: vec![1, 1, 1],
                rank: 2,
            })
        );
    }

    #[test]
    fn so_odd_vector_dimension_is_exact() {
        assert_eq!(dimension_so_odd(&yd(&[1]), 2).unwrap(), BigInt::from(5usize));
    }

    #[test]
    fn so_even_vector_dimension_is_exact() {
        assert_eq!(dimension_so_even(&yd(&[1]), 3).unwrap(), BigInt::from(6usize));
    }

    #[test]
    fn sp_fundamental_dimension_is_exact() {
        assert_eq!(dimension_sp(&yd(&[1]), 2).unwrap(), BigInt::from(4usize));
    }

    #[test]
    fn sp_symmetric_square_dimension_is_exact() {
        assert_eq!(dimension_sp(&yd(&[2]), 2).unwrap(), BigInt::from(10usize));
    }

    #[test]
    fn sp_two_form_dimension_is_exact() {
        assert_eq!(dimension_sp(&yd(&[1, 1]), 2).unwrap(), BigInt::from(5usize));
    }

    #[test]
    fn gl_to_sp_branching_keeps_fundamental_irrep() {
        assert_eq!(
            branch_gl_to_sp(&yd(&[1]), 2).unwrap(),
            vec![(ClassicalBranchTarget::Irrep(yd(&[1])), BigInt::one())]
        );
    }

    #[test]
    fn gl_to_sp_branching_of_two_box_row_is_scalar_plus_irrep() {
        assert_eq!(
            branch_gl_to_sp(&yd(&[2]), 2).unwrap(),
            vec![
                (ClassicalBranchTarget::Scalar, BigInt::one()),
                (ClassicalBranchTarget::Irrep(yd(&[2])), BigInt::one()),
            ]
        );
    }

    #[test]
    fn gl_to_so_branching_of_two_box_row_is_scalar_plus_irrep() {
        assert_eq!(
            branch_gl_to_so(&yd(&[2]), 2, false).unwrap(),
            vec![
                (ClassicalBranchTarget::Scalar, BigInt::one()),
                (ClassicalBranchTarget::Irrep(yd(&[2])), BigInt::one()),
            ]
        );
    }

    #[test]
    fn unsupported_branching_regime_reports_exact_error() {
        assert_eq!(
            branch_gl_to_sp(&yd(&[3, 2, 1]), 2),
            Err(YoungError::ClassicalBranchingUnsupported {
                family: "sp",
                shape: vec![3, 2, 1],
                rank: 2,
            })
        );
    }
}
