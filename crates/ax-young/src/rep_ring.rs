use crate::{lr_shapes_with_multiplicity, partition::YoungDiagram, YoungError};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchurExpansion {
    pub terms: BTreeMap<YoungDiagram, BigInt>,
}

impl SchurExpansion {
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    pub fn from_shape(shape: YoungDiagram) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(shape, BigInt::one());
        Self { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn normalize(&mut self) {
        self.terms.retain(|_, coeff| !coeff.is_zero());
    }

    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn add_term(&mut self, shape: YoungDiagram, coeff: BigInt) {
        *self.terms.entry(shape).or_insert_with(BigInt::zero) += coeff;
        self.normalize();
    }

    pub fn coefficient(&self, shape: &YoungDiagram) -> BigInt {
        self.terms.get(shape).cloned().unwrap_or_else(BigInt::zero)
    }

    pub fn support(&self) -> Vec<YoungDiagram> {
        self.terms.keys().cloned().collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepExpansion {
    Scalar(BigInt),
    Schur(SchurExpansion),
    DirectSum {
        scalar: BigInt,
        schur: SchurExpansion,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicitySpace {
    pub shape: YoungDiagram,
    pub multiplicity: usize,
    pub basis_labels: Vec<String>,
}

impl MultiplicitySpace {
    pub fn basis_label(&self, index: usize) -> Result<&str, YoungError> {
        self.basis_labels
            .get(index)
            .map(|label| label.as_str())
            .ok_or(YoungError::MultiplicityBasisIndexOutOfRange {
                index,
                multiplicity: self.multiplicity,
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorProductDecomposition {
    pub factors: Vec<YoungDiagram>,
    pub irreps: Vec<MultiplicitySpace>,
}

pub fn add_rep_expansions(a: &RepExpansion, b: &RepExpansion) -> RepExpansion {
    let (scalar_a, schur_a) = split_rep_expansion(a);
    let (scalar_b, schur_b) = split_rep_expansion(b);
    make_rep_expansion(
        scalar_a + scalar_b,
        add_schur_expansions(&schur_a, &schur_b),
    )
}

pub fn multiply_rep_expansions(
    a: &RepExpansion,
    b: &RepExpansion,
) -> Result<RepExpansion, YoungError> {
    let (scalar_a, schur_a) = split_rep_expansion(a);
    let (scalar_b, schur_b) = split_rep_expansion(b);

    let scalar_part = &scalar_a * &scalar_b;
    let mut schur_part = SchurExpansion::zero();

    if !scalar_a.is_zero() {
        schur_part = add_schur_expansions(&schur_part, &scale_schur_expansion(&schur_b, &scalar_a));
    }
    if !scalar_b.is_zero() {
        schur_part = add_schur_expansions(&schur_part, &scale_schur_expansion(&schur_a, &scalar_b));
    }
    if !schur_a.is_zero() && !schur_b.is_zero() {
        schur_part =
            add_schur_expansions(&schur_part, &multiply_schur_expansions(&schur_a, &schur_b)?);
    }

    Ok(make_rep_expansion(scalar_part, schur_part))
}

pub fn tensor_product_decomposition(
    factors: &[YoungDiagram],
) -> Result<TensorProductDecomposition, YoungError> {
    if factors.is_empty() {
        return Ok(TensorProductDecomposition {
            factors: Vec::new(),
            irreps: Vec::new(),
        });
    }

    let mut current = SchurExpansion::from_shape(factors[0].clone());
    for factor in &factors[1..] {
        current = multiply_schur_expansions(&current, &SchurExpansion::from_shape(factor.clone()))?;
    }

    let irreps = current
        .support()
        .into_iter()
        .map(|shape| {
            let multiplicity = current
                .coefficient(&shape)
                .to_usize()
                .ok_or(YoungError::NegativeMultiplicity)?;
            let basis_labels = if factors.len() == 3 {
                match crate::multiplicity_basis::canonical_multiplicity_basis(
                    factors,
                    &shape,
                    crate::multiplicity_basis::AssociationConvention::LeftAssociated,
                ) {
                    Ok(basis) => basis
                        .vectors
                        .into_iter()
                        .map(|vector| vector.label)
                        .collect::<Vec<_>>(),
                    Err(YoungError::MultiplicityBasisUnsupported { .. }) => (0..multiplicity)
                        .map(|index| format!("m{index}"))
                        .collect::<Vec<_>>(),
                    Err(error) => return Err(error),
                }
            } else {
                (0..multiplicity)
                    .map(|index| format!("m{index}"))
                    .collect::<Vec<_>>()
            };
            Ok(MultiplicitySpace {
                shape,
                multiplicity,
                basis_labels,
            })
        })
        .collect::<Result<Vec<_>, YoungError>>()?;

    Ok(TensorProductDecomposition {
        factors: factors.to_vec(),
        irreps,
    })
}

pub(crate) fn add_schur_expansions(a: &SchurExpansion, b: &SchurExpansion) -> SchurExpansion {
    let mut terms = a.terms.clone();
    for (shape, coeff) in &b.terms {
        *terms.entry(shape.clone()).or_insert_with(BigInt::zero) += coeff.clone();
    }
    SchurExpansion { terms }.normalized()
}

pub(crate) fn scale_schur_expansion(expansion: &SchurExpansion, scalar: &BigInt) -> SchurExpansion {
    if scalar.is_zero() || expansion.is_zero() {
        return SchurExpansion::zero();
    }
    let mut terms = BTreeMap::new();
    for (shape, coeff) in &expansion.terms {
        terms.insert(shape.clone(), coeff * scalar);
    }
    SchurExpansion { terms }.normalized()
}

pub(crate) fn multiply_schur_expansions(
    left: &SchurExpansion,
    right: &SchurExpansion,
) -> Result<SchurExpansion, YoungError> {
    let mut out = SchurExpansion::zero();
    for (left_shape, left_coeff) in &left.terms {
        for (right_shape, right_coeff) in &right.terms {
            for (target, multiplicity) in lr_shapes_with_multiplicity(left_shape, right_shape)? {
                let coeff = left_coeff * right_coeff * multiplicity;
                out.add_term(target, coeff);
            }
        }
    }
    Ok(out.normalized())
}

fn split_rep_expansion(expansion: &RepExpansion) -> (BigInt, SchurExpansion) {
    match expansion {
        RepExpansion::Scalar(value) => (value.clone(), SchurExpansion::zero()),
        RepExpansion::Schur(expansion) => (BigInt::zero(), expansion.clone()),
        RepExpansion::DirectSum { scalar, schur } => (scalar.clone(), schur.clone()),
    }
}

fn make_rep_expansion(scalar: BigInt, schur: SchurExpansion) -> RepExpansion {
    let schur = schur.normalized();
    if schur.is_zero() {
        RepExpansion::Scalar(scalar)
    } else if scalar.is_zero() {
        RepExpansion::Schur(schur)
    } else {
        RepExpansion::DirectSum { scalar, schur }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schur_tensor_product;

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn tensor_product_decomposition_tracks_explicit_multiplicity_spaces() {
        let decomposition =
            tensor_product_decomposition(&[yd(vec![1]), yd(vec![1]), yd(vec![1])]).unwrap();

        assert_eq!(
            decomposition.irreps,
            vec![
                MultiplicitySpace {
                    shape: yd(vec![1, 1, 1]),
                    multiplicity: 1,
                    basis_labels: vec!["m0".to_string()],
                },
                MultiplicitySpace {
                    shape: yd(vec![2, 1]),
                    multiplicity: 2,
                    basis_labels: vec!["m0".to_string(), "m1".to_string()],
                },
                MultiplicitySpace {
                    shape: yd(vec![3]),
                    multiplicity: 1,
                    basis_labels: vec!["m0".to_string()],
                },
            ]
        );
    }

    #[test]
    fn basis_label_reports_out_of_range_index() {
        let space = MultiplicitySpace {
            shape: yd(vec![2, 1]),
            multiplicity: 2,
            basis_labels: vec!["m0".to_string(), "m1".to_string()],
        };
        assert_eq!(
            space.basis_label(2),
            Err(YoungError::MultiplicityBasisIndexOutOfRange {
                index: 2,
                multiplicity: 2,
            })
        );
    }

    #[test]
    fn schur_schur_product_is_exact_lr_multiplication() {
        let product = multiply_schur_expansions(
            &SchurExpansion::from_shape(yd(vec![1])),
            &SchurExpansion::from_shape(yd(vec![1])),
        )
        .unwrap();

        assert_eq!(product.coefficient(&yd(vec![1, 1])), BigInt::from(1usize));
        assert_eq!(product.coefficient(&yd(vec![2])), BigInt::from(1usize));
    }

    #[test]
    fn schur_tensor_product_wrapper_matches_ring_product() {
        assert_eq!(
            schur_tensor_product(&yd(vec![1]), &yd(vec![1])).unwrap(),
            multiply_schur_expansions(
                &SchurExpansion::from_shape(yd(vec![1])),
                &SchurExpansion::from_shape(yd(vec![1])),
            )
            .unwrap()
        );
    }

    #[test]
    fn scalar_schur_multiplication_scales_coefficients_exactly() {
        let product = multiply_rep_expansions(
            &RepExpansion::Scalar(BigInt::from(3usize)),
            &RepExpansion::Schur(SchurExpansion::from_shape(yd(vec![2]))),
        )
        .unwrap();

        assert_eq!(
            product,
            RepExpansion::Schur(SchurExpansion {
                terms: BTreeMap::from([(yd(vec![2]), BigInt::from(3usize))]),
            })
        );
    }

    #[test]
    fn mixed_addition_preserves_scalar_and_schur_parts() {
        let sum = add_rep_expansions(
            &RepExpansion::Scalar(BigInt::from(2usize)),
            &RepExpansion::Schur(SchurExpansion::from_shape(yd(vec![1, 1]))),
        );

        assert_eq!(
            sum,
            RepExpansion::DirectSum {
                scalar: BigInt::from(2usize),
                schur: SchurExpansion::from_shape(yd(vec![1, 1])),
            }
        );
    }
}
