use crate::{
    characters::frobenius_characteristic,
    kostka_number_exact,
    rep_ring::SchurExpansion,
    YoungDiagram, YoungError,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};

pub type Partition = YoungDiagram;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerSumExpansion {
    pub terms: BTreeMap<Partition, BigRational>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonomialExpansion {
    pub terms: BTreeMap<Partition, BigInt>,
}

pub fn partition_from_parts(parts: Vec<usize>) -> Result<Partition, YoungError> {
    if parts.is_empty()
        || parts.iter().any(|part| *part == 0)
        || parts.windows(2).any(|window| window[0] < window[1])
    {
        return Err(YoungError::InvalidPartitionContent { parts });
    }
    YoungDiagram::try_new(parts)
}

pub fn schur_to_monomial(schur: &SchurExpansion) -> Result<MonomialExpansion, YoungError> {
    let mut out = BTreeMap::<Partition, BigInt>::new();
    for (shape, schur_coeff) in &schur.terms {
        for content in enumerate_partitions_of_size(shape.n_cells()) {
            let kostka = kostka_number_exact(shape, &content.rows)?;
            if kostka.is_zero() {
                continue;
            }
            *out.entry(content).or_insert_with(BigInt::zero) += schur_coeff * kostka;
        }
    }
    out.retain(|_, coeff| !coeff.is_zero());
    Ok(MonomialExpansion { terms: out })
}

pub fn schur_to_power_sum(schur: &SchurExpansion) -> Result<PowerSumExpansion, YoungError> {
    let mut out = BTreeMap::<Partition, BigRational>::new();
    for (shape, coeff) in &schur.terms {
        let frobenius = frobenius_characteristic(shape)?;
        for (partition, term_coeff) in frobenius.terms {
            *out.entry(partition).or_insert_with(BigRational::zero) +=
                term_coeff * BigRational::from_integer(coeff.clone());
        }
    }
    out.retain(|_, coeff| !coeff.is_zero());
    Ok(PowerSumExpansion { terms: out })
}

pub fn multiply_power_sum(
    left: &PowerSumExpansion,
    right: &PowerSumExpansion,
) -> Result<PowerSumExpansion, YoungError> {
    let mut out = BTreeMap::<Partition, BigRational>::new();
    for (left_partition, left_coeff) in &left.terms {
        for (right_partition, right_coeff) in &right.terms {
            let mut parts = left_partition.rows.clone();
            parts.extend(right_partition.rows.iter().copied());
            parts.sort_by(|lhs, rhs| rhs.cmp(lhs));
            let partition = partition_from_parts(parts)?;
            *out.entry(partition).or_insert_with(BigRational::zero) += left_coeff * right_coeff;
        }
    }
    out.retain(|_, coeff| !coeff.is_zero());
    Ok(PowerSumExpansion { terms: out })
}

pub fn multiply_monomial(
    left: &MonomialExpansion,
    right: &MonomialExpansion,
) -> Result<MonomialExpansion, YoungError> {
    let total_degree = left
        .terms
        .iter()
        .map(|(partition, _)| partition.n_cells())
        .max()
        .unwrap_or(0)
        + right
            .terms
            .iter()
            .map(|(partition, _)| partition.n_cells())
            .max()
            .unwrap_or(0);
    if total_degree > 4 {
        return Err(YoungError::MonomialMultiplicationUnsupported { total_degree });
    }
    let n_vars = total_degree.max(1);
    let mut polynomial = BTreeMap::<Vec<usize>, BigInt>::new();
    for (left_partition, left_coeff) in &left.terms {
        let left_poly = monomial_symmetric_polynomial(left_partition, n_vars);
        for (right_partition, right_coeff) in &right.terms {
            let right_poly = monomial_symmetric_polynomial(right_partition, n_vars);
            for (left_exp, left_term_coeff) in &left_poly {
                for (right_exp, right_term_coeff) in &right_poly {
                    let product_exp = left_exp
                        .iter()
                        .zip(right_exp.iter())
                        .map(|(lhs, rhs)| lhs + rhs)
                        .collect::<Vec<_>>();
                    *polynomial.entry(product_exp).or_insert_with(BigInt::zero) +=
                        left_coeff * right_coeff * left_term_coeff * right_term_coeff;
                }
            }
        }
    }

    let mut aggregated = BTreeMap::<Partition, BigInt>::new();
    let mut orbit_totals = BTreeMap::<Partition, BigInt>::new();
    for (exp, coeff) in polynomial {
        let partition = partition_from_exponents(&exp)?;
        *orbit_totals.entry(partition).or_insert_with(BigInt::zero) += coeff;
    }
    for (partition, total) in orbit_totals {
        let orbit_size = BigInt::from(distinct_padded_permutations(&partition.rows, n_vars).len());
        aggregated.insert(partition, total / orbit_size);
    }
    aggregated.retain(|_, coeff| !coeff.is_zero());
    Ok(MonomialExpansion { terms: aggregated })
}

fn partition_from_exponents(exponents: &[usize]) -> Result<Partition, YoungError> {
    let mut parts = exponents
        .iter()
        .copied()
        .filter(|part| *part > 0)
        .collect::<Vec<_>>();
    parts.sort_by(|lhs, rhs| rhs.cmp(lhs));
    partition_from_parts(parts)
}

fn monomial_symmetric_polynomial(partition: &Partition, n_vars: usize) -> BTreeMap<Vec<usize>, BigInt> {
    distinct_padded_permutations(&partition.rows, n_vars)
        .into_iter()
        .map(|exp| (exp, BigInt::one()))
        .collect()
}

fn distinct_padded_permutations(parts: &[usize], n_vars: usize) -> BTreeSet<Vec<usize>> {
    let mut padded = parts.to_vec();
    padded.resize(n_vars, 0);

    fn rec(current: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut BTreeSet<Vec<usize>>) {
        if remaining.is_empty() {
            out.insert(current.clone());
            return;
        }
        let mut used = BTreeSet::new();
        for index in 0..remaining.len() {
            if !used.insert(remaining[index]) {
                continue;
            }
            let value = remaining.remove(index);
            current.push(value);
            rec(current, remaining, out);
            current.pop();
            remaining.insert(index, value);
        }
    }

    let mut out = BTreeSet::new();
    rec(&mut Vec::new(), &mut padded, &mut out);
    out
}

fn enumerate_partitions_of_size(total: usize) -> Vec<Partition> {
    fn rec(
        remaining: usize,
        max_part: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Partition>,
    ) {
        if remaining == 0 {
            if let Ok(partition) = partition_from_parts(current.clone()) {
                out.push(partition);
            }
            return;
        }
        for next in (1..=remaining.min(max_part)).rev() {
            current.push(next);
            rec(remaining - next, next, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(total, total, &mut Vec::new(), &mut out);
    out.sort_by(|lhs, rhs| lhs.rows.cmp(&rhs.rows));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yd(rows: &[usize]) -> YoungDiagram {
        YoungDiagram::try_new(rows.to_vec()).unwrap()
    }

    #[test]
    fn schur_to_monomial_uses_exact_kostka_expansion() {
        let monomial = schur_to_monomial(&SchurExpansion::from_shape(yd(&[2, 1]))).unwrap();

        assert_eq!(
            monomial.terms.get(&yd(&[2, 1])).cloned().unwrap(),
            BigInt::from(1usize)
        );
        assert_eq!(
            monomial.terms.get(&yd(&[1, 1, 1])).cloned().unwrap(),
            BigInt::from(2usize)
        );
    }

    #[test]
    fn multiply_power_sum_concatenates_cycle_partitions() {
        let left = PowerSumExpansion {
            terms: BTreeMap::from([(yd(&[1]), BigRational::one())]),
        };
        let right = PowerSumExpansion {
            terms: BTreeMap::from([(yd(&[2]), BigRational::one())]),
        };
        let product = multiply_power_sum(&left, &right).unwrap();

        assert_eq!(
            product.terms,
            BTreeMap::from([(yd(&[2, 1]), BigRational::one())])
        );
    }

    #[test]
    fn multiply_monomial_is_exact_in_supported_degree() {
        let left = MonomialExpansion {
            terms: BTreeMap::from([(yd(&[1]), BigInt::one())]),
        };
        let right = MonomialExpansion {
            terms: BTreeMap::from([(yd(&[1]), BigInt::one())]),
        };
        let product = multiply_monomial(&left, &right).unwrap();

        assert_eq!(product.terms.get(&yd(&[2])).cloned().unwrap(), BigInt::one());
        assert_eq!(
            product.terms.get(&yd(&[1, 1])).cloned().unwrap(),
            BigInt::from(2usize)
        );
    }
}
