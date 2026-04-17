use anyhow::{anyhow, Context};
use num_traits::ToPrimitive;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassicalDecompositionSummary {
    pub family: String,
    pub rank: usize,
    pub shapes: Vec<Vec<usize>>,
    pub multiplicities: Vec<usize>,
    pub scalar_multiplicity: usize,
}

pub fn decompose_gl_shape_to_classical(
    shape: &[usize],
    family: &str,
    rank: usize,
) -> anyhow::Result<ClassicalDecompositionSummary> {
    let diagram = ax_young::YoungDiagram::try_new(shape.to_vec())
        .context("failed to decompose GL shape into classical-group irreps")?;
    let decomposition = match family {
        "so_odd" => ax_young::classical_groups::branch_gl_to_so(&diagram, rank, false),
        "so_even" => ax_young::classical_groups::branch_gl_to_so(&diagram, rank, true),
        "sp" => ax_young::classical_groups::branch_gl_to_sp(&diagram, rank),
        _ => Err(ax_young::YoungError::ClassicalBranchingUnsupported {
            family: "unknown",
            shape: shape.to_vec(),
            rank,
        }),
    }
    .context("failed to decompose GL shape into classical-group irreps")?;

    let mut shapes = Vec::new();
    let mut multiplicities = Vec::new();
    let mut scalar_multiplicity = 0usize;
    for (target, multiplicity) in decomposition {
        let multiplicity = multiplicity
            .to_usize()
            .ok_or_else(|| anyhow!("classical branching multiplicity does not fit in usize"))?;
        match target {
            ax_young::classical_groups::ClassicalBranchTarget::Scalar => {
                scalar_multiplicity = multiplicity;
            }
            ax_young::classical_groups::ClassicalBranchTarget::Irrep(diagram) => {
                shapes.push(diagram.rows);
                multiplicities.push(multiplicity);
            }
        }
    }

    Ok(ClassicalDecompositionSummary {
        family: family.to_string(),
        rank,
        shapes,
        multiplicities,
        scalar_multiplicity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_gl_shape_to_sp_reports_scalar_and_irrep_parts() {
        let summary = decompose_gl_shape_to_classical(&[2], "sp", 2).unwrap();
        assert_eq!(summary.family, "sp");
        assert_eq!(summary.rank, 2);
        assert_eq!(summary.shapes, vec![vec![2]]);
        assert_eq!(summary.multiplicities, vec![1]);
        assert_eq!(summary.scalar_multiplicity, 1);
    }

    #[test]
    fn invalid_family_string_is_contextualized() {
        let error = decompose_gl_shape_to_classical(&[2], "bad", 2).unwrap_err();
        assert!(error
            .to_string()
            .contains("failed to decompose GL shape into classical-group irreps"));
    }
}
