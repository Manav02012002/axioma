use crate::{lr_shapes_with_multiplicity, rep_ring::SchurExpansion, YoungDiagram, YoungError};

pub fn schur_basis_shape(shape: &YoungDiagram) -> SchurExpansion {
    SchurExpansion::from_shape(shape.clone())
}

pub fn schur_tensor_product(
    left: &YoungDiagram,
    right: &YoungDiagram,
) -> Result<SchurExpansion, YoungError> {
    let mut expansion = SchurExpansion::zero();
    for (shape, coeff) in lr_shapes_with_multiplicity(left, right)? {
        expansion.add_term(shape, coeff);
    }
    Ok(expansion.normalized())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn yd(rows: Vec<usize>) -> YoungDiagram {
        YoungDiagram::try_new(rows).unwrap()
    }

    #[test]
    fn schur_tensor_product_of_two_vectors_is_symmetric_plus_antisymmetric() {
        let product = schur_tensor_product(&yd(vec![1]), &yd(vec![1])).unwrap();
        assert_eq!(product.coefficient(&yd(vec![2])), BigInt::from(1usize));
        assert_eq!(product.coefficient(&yd(vec![1, 1])), BigInt::from(1usize));
    }
}
