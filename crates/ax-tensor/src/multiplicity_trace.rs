use anyhow::Context;

pub fn tensor_product_multiplicity_trace(
    shapes: &[Vec<usize>],
    target: &[usize],
) -> anyhow::Result<ax_trace::MultiplicityBasisTrace> {
    let diagrams = shapes
        .iter()
        .map(|shape| ax_young::YoungDiagram::try_new(shape.clone()))
        .collect::<Result<Vec<_>, _>>()
        .context("failed to compute tensor-product multiplicity trace")?;
    let target = ax_young::YoungDiagram::try_new(target.to_vec())
        .context("failed to compute tensor-product multiplicity trace")?;

    ax_young::multiplicity_basis_trace(&diagrams, &target)
        .context("failed to compute tensor-product multiplicity trace")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_product_multiplicity_trace_succeeds_for_three_vectors_and_two_one() {
        let trace = tensor_product_multiplicity_trace(&[vec![1], vec![1], vec![1]], &[2, 1]).unwrap();
        assert_eq!(trace.target, vec![2, 1]);
    }

    #[test]
    fn tensor_product_multiplicity_trace_contextualizes_invalid_input() {
        let error = tensor_product_multiplicity_trace(&[vec![], vec![1], vec![1]], &[2, 1])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("failed to compute tensor-product multiplicity trace"));
    }

    #[test]
    fn decompose_product_with_basis_trace_reports_two_one_trace() {
        let (summary, traces) =
            crate::decompose_product_with_basis_trace(&[vec![1], vec![1], vec![1]]).unwrap();
        assert_eq!(summary.shapes, vec![vec![1, 1, 1], vec![2, 1], vec![3]]);
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].target, vec![2, 1]);
    }
}
