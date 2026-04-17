use crate::{
    kostka_number_exact,
    rep_ring::{RepExpansion, SchurExpansion},
    YoungDiagram, YoungError,
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

type Poly = BTreeMap<Vec<usize>, BigInt>;

pub fn plethysm_schur_by_shape(
    outer: &YoungDiagram,
    inner: &YoungDiagram,
) -> Result<SchurExpansion, YoungError> {
    let total_degree = outer.n_cells() * inner.n_cells();
    let var_count = total_degree.max(1);
    let inner_poly = schur_polynomial_monomials(inner, var_count)?;
    let max_h = outer
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| row + outer.n_rows().saturating_sub(index + 1))
        .max()
        .unwrap_or(0);
    let complete = complete_symmetric_plethysm(&inner_poly, var_count, max_h);
    let plethysm_poly = jacobi_trudi_plethysm(outer, &complete, var_count);
    schur_from_symmetric_polynomial(&plethysm_poly, total_degree, var_count)
}

pub fn plethysm_rep_expansion(
    outer: &RepExpansion,
    inner: &RepExpansion,
) -> Result<RepExpansion, YoungError> {
    let outer = match outer {
        RepExpansion::Schur(expansion) => expansion,
        _ => return Err(YoungError::InvalidPlethysmOuter),
    };
    let inner = match inner {
        RepExpansion::Schur(expansion) => expansion,
        _ => return Err(YoungError::InvalidPlethysmInner),
    };
    if outer.is_zero() || inner.is_zero() {
        return Err(YoungError::EmptySchurExpansion);
    }

    let mut out = SchurExpansion::zero();
    for (outer_shape, outer_coeff) in &outer.terms {
        for (inner_shape, inner_coeff) in &inner.terms {
            let plethysm = plethysm_schur_by_shape(outer_shape, inner_shape)?;
            for (shape, coeff) in plethysm.terms {
                out.add_term(shape, outer_coeff * inner_coeff * coeff);
            }
        }
    }
    Ok(RepExpansion::Schur(out.normalized()))
}

fn schur_polynomial_monomials(shape: &YoungDiagram, var_count: usize) -> Result<Poly, YoungError> {
    let mut out = BTreeMap::new();
    for composition in compositions_of(shape.n_cells(), var_count) {
        let coeff = kostka_number_exact(shape, &composition)?;
        if !coeff.is_zero() {
            out.insert(composition, coeff);
        }
    }
    Ok(out)
}

fn complete_symmetric_plethysm(
    inner_poly: &Poly,
    var_count: usize,
    max_degree: usize,
) -> Vec<Poly> {
    let mut complete = vec![BTreeMap::new(); max_degree + 1];
    complete[0].insert(vec![0; var_count], BigInt::one());

    for (monomial, multiplicity) in inner_poly {
        if multiplicity.is_zero() {
            continue;
        }
        let mut next = vec![BTreeMap::new(); max_degree + 1];
        for current_degree in 0..=max_degree {
            for repeat in 0..=max_degree - current_degree {
                let factor = multiset_binomial(multiplicity, repeat);
                if factor.is_zero() {
                    continue;
                }
                let shift = scale_monomial(monomial, repeat);
                accumulate_shifted(
                    &mut next[current_degree + repeat],
                    &complete[current_degree],
                    &shift,
                    &factor,
                );
            }
        }
        complete = next;
    }

    complete
}

fn jacobi_trudi_plethysm(shape: &YoungDiagram, complete: &[Poly], var_count: usize) -> Poly {
    let size = shape.n_rows();
    let mut matrix = vec![vec![BTreeMap::new(); size]; size];
    for row in 0..size {
        for col in 0..size {
            let index = shape.rows[row] as isize + col as isize - row as isize;
            matrix[row][col] = match index.cmp(&0) {
                Ordering::Less => BTreeMap::new(),
                Ordering::Equal => unit_poly(var_count),
                Ordering::Greater => complete[index as usize].clone(),
            };
        }
    }
    determinant_poly(&matrix, var_count)
}

fn schur_from_symmetric_polynomial(
    polynomial: &Poly,
    total_degree: usize,
    var_count: usize,
) -> Result<SchurExpansion, YoungError> {
    let monomial_coeffs = monomial_symmetric_coefficients(polynomial, var_count);
    let mut partitions = enumerate_partitions(total_degree, total_degree, &mut Vec::new());
    partitions.retain(|shape| shape.n_rows() <= var_count);
    partitions.sort_by(dominance_then_lex);

    let mut coeffs = BTreeMap::new();
    for current in &partitions {
        let mut coeff = monomial_coeffs
            .get(current)
            .cloned()
            .unwrap_or_else(BigInt::zero);
        for (larger, larger_coeff) in &coeffs {
            let kostka = kostka_number_exact(larger, &partition_as_content(current))?;
            coeff -= larger_coeff * kostka;
        }
        if !coeff.is_zero() {
            coeffs.insert(current.clone(), coeff);
        }
    }

    Ok(SchurExpansion { terms: coeffs }.normalized())
}

fn monomial_symmetric_coefficients(
    polynomial: &Poly,
    var_count: usize,
) -> BTreeMap<YoungDiagram, BigInt> {
    let mut totals: BTreeMap<YoungDiagram, BigInt> = BTreeMap::new();
    for (monomial, coeff) in polynomial {
        let mut parts = monomial.clone();
        parts.sort_by(|lhs, rhs| rhs.cmp(lhs));
        while parts.last().copied() == Some(0) {
            parts.pop();
        }
        if let Ok(shape) = YoungDiagram::try_new(parts) {
            *totals.entry(shape).or_insert_with(BigInt::zero) += coeff.clone();
        }
    }

    totals
        .into_iter()
        .map(|(shape, total)| {
            let orbit = monomial_orbit_size(&shape.rows, var_count);
            (shape, total / orbit)
        })
        .collect()
}

fn monomial_orbit_size(parts: &[usize], var_count: usize) -> BigInt {
    let mut padded = parts.to_vec();
    padded.resize(var_count, 0);
    let mut counts = BTreeMap::new();
    for value in padded {
        *counts.entry(value).or_insert(0usize) += 1;
    }
    let mut orbit = factorial(var_count);
    for multiplicity in counts.into_values() {
        orbit /= factorial(multiplicity);
    }
    orbit
}

fn determinant_poly(matrix: &[Vec<Poly>], var_count: usize) -> Poly {
    fn rec(
        matrix: &[Vec<Poly>],
        row: usize,
        used_cols: &mut [bool],
        current: &Poly,
        sign: i32,
        out: &mut Poly,
        var_count: usize,
    ) {
        if row == matrix.len() {
            add_scaled_poly(out, current, &BigInt::from(sign));
            return;
        }
        for col in 0..matrix.len() {
            if used_cols[col] {
                continue;
            }
            used_cols[col] = true;
            let next = multiply_poly(current, &matrix[row][col], var_count);
            let flips = used_cols[..col].iter().filter(|used| !**used).count();
            let next_sign = if flips % 2 == 0 { sign } else { -sign };
            rec(matrix, row + 1, used_cols, &next, next_sign, out, var_count);
            used_cols[col] = false;
        }
    }

    let mut out = BTreeMap::new();
    let mut used = vec![false; matrix.len()];
    rec(
        matrix,
        0,
        &mut used,
        &unit_poly(var_count),
        1,
        &mut out,
        var_count,
    );
    normalize_poly(&mut out);
    out
}

fn multiply_poly(left: &Poly, right: &Poly, var_count: usize) -> Poly {
    let mut out = BTreeMap::new();
    for (left_monomial, left_coeff) in left {
        for (right_monomial, right_coeff) in right {
            let mut monomial = vec![0; var_count];
            for index in 0..var_count {
                monomial[index] = left_monomial[index] + right_monomial[index];
            }
            *out.entry(monomial).or_insert_with(BigInt::zero) += left_coeff * right_coeff;
        }
    }
    normalize_poly(&mut out);
    out
}

fn add_scaled_poly(target: &mut Poly, source: &Poly, scale: &BigInt) {
    if scale.is_zero() {
        return;
    }
    for (monomial, coeff) in source {
        *target.entry(monomial.clone()).or_insert_with(BigInt::zero) += coeff * scale;
    }
    normalize_poly(target);
}

fn accumulate_shifted(target: &mut Poly, source: &Poly, shift: &[usize], scale: &BigInt) {
    if scale.is_zero() {
        return;
    }
    for (monomial, coeff) in source {
        let shifted = monomial
            .iter()
            .zip(shift.iter())
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        *target.entry(shifted).or_insert_with(BigInt::zero) += coeff * scale;
    }
    normalize_poly(target);
}

fn normalize_poly(poly: &mut Poly) {
    poly.retain(|_, coeff| !coeff.is_zero());
}

fn unit_poly(var_count: usize) -> Poly {
    BTreeMap::from([(vec![0; var_count], BigInt::one())])
}

fn scale_monomial(monomial: &[usize], factor: usize) -> Vec<usize> {
    monomial.iter().map(|value| value * factor).collect()
}

fn partition_as_content(shape: &YoungDiagram) -> Vec<usize> {
    shape.rows.clone()
}

fn multiset_binomial(multiplicity: &BigInt, repeat: usize) -> BigInt {
    if repeat == 0 {
        return BigInt::one();
    }
    let mut out = BigInt::one();
    for step in 0..repeat {
        out *= multiplicity + BigInt::from(step);
        out /= BigInt::from(step + 1);
    }
    out
}

fn factorial(value: usize) -> BigInt {
    (1..=value).fold(BigInt::one(), |acc, current| acc * current)
}

fn dominance_then_lex(lhs: &YoungDiagram, rhs: &YoungDiagram) -> Ordering {
    let lhs_dominates = dominates(lhs, rhs);
    let rhs_dominates = dominates(rhs, lhs);
    match (lhs_dominates, rhs_dominates) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => rhs.rows.cmp(&lhs.rows),
    }
}

fn dominates(lhs: &YoungDiagram, rhs: &YoungDiagram) -> bool {
    let mut lhs_sum = 0usize;
    let mut rhs_sum = 0usize;
    let max_len = lhs.n_rows().max(rhs.n_rows());
    for index in 0..max_len {
        lhs_sum += lhs.row_len(index).unwrap_or(0);
        rhs_sum += rhs.row_len(index).unwrap_or(0);
        if lhs_sum < rhs_sum {
            return false;
        }
    }
    true
}

fn compositions_of(total: usize, len: usize) -> Vec<Vec<usize>> {
    fn rec(remaining: usize, len: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() + 1 == len {
            current.push(remaining);
            out.push(current.clone());
            current.pop();
            return;
        }
        for value in 0..=remaining {
            current.push(value);
            rec(remaining - value, len, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(total, len, &mut Vec::new(), &mut out);
    out
}

fn enumerate_partitions(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
) -> Vec<YoungDiagram> {
    if remaining == 0 {
        return YoungDiagram::try_new(current.clone()).into_iter().collect();
    }

    let mut out = Vec::new();
    for next in (1..=remaining.min(max_part)).rev() {
        current.push(next);
        out.extend(enumerate_partitions(remaining - next, next, current));
        current.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rep_ring::SchurExpansion;

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn plethysm_of_vector_by_shape_is_identity_on_inner_shape() {
        assert_eq!(
            plethysm_schur_by_shape(&yd(vec![1]), &yd(vec![2])).unwrap(),
            SchurExpansion::from_shape(yd(vec![2]))
        );
    }

    #[test]
    fn plethysm_of_symmetric_square_of_vector_is_symmetric_square() {
        assert_eq!(
            plethysm_schur_by_shape(&yd(vec![2]), &yd(vec![1])).unwrap(),
            SchurExpansion::from_shape(yd(vec![2]))
        );
    }

    #[test]
    fn plethysm_of_exterior_square_of_vector_is_exterior_square() {
        assert_eq!(
            plethysm_schur_by_shape(&yd(vec![1, 1]), &yd(vec![1])).unwrap(),
            SchurExpansion::from_shape(yd(vec![1, 1]))
        );
    }

    #[test]
    fn plethysm_of_symmetric_square_of_symmetric_square_is_exact() {
        let result = plethysm_schur_by_shape(&yd(vec![2]), &yd(vec![2])).unwrap();
        assert_eq!(
            result,
            SchurExpansion {
                terms: BTreeMap::from([
                    (yd(vec![2, 2]), BigInt::from(1usize)),
                    (yd(vec![4]), BigInt::from(1usize)),
                ]),
            }
        );
    }

    #[test]
    fn plethysm_of_exterior_square_of_symmetric_square_is_exact() {
        let result = plethysm_schur_by_shape(&yd(vec![1, 1]), &yd(vec![2])).unwrap();
        assert_eq!(
            result,
            SchurExpansion {
                terms: BTreeMap::from([(yd(vec![3, 1]), BigInt::from(1usize))]),
            }
        );
    }
}
