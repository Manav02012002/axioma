use crate::{
    projector::{column_antisymmetrizer_generators, row_symmetrizer_generators, PermutationTerm},
    YoungDiagram, YoungError, YoungTableau,
};
use ax_perm::{all_orbits, enumerate_subgroup, identity, product, schreier_sims, sign, Perm, SGS};
use ax_trace::{CanonicalizationTrace, ProjectorBuildTrace};
use num_rational::BigRational;
use num_traits::{One, Zero};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizerGroup {
    pub degree: usize,
    pub generators: Vec<Perm>,
    pub sgs: SGS,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectorNormalization {
    Unnormalized,
    HookLength,
    Idempotent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupBackedProjector {
    pub diagram: YoungDiagram,
    pub tableau: YoungTableau,
    pub row_group: StabilizerGroup,
    pub column_group: StabilizerGroup,
    pub normalization: ProjectorNormalization,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum GroupProjectorError {
    #[error("Young projector construction failed: {0}")]
    Young(#[from] YoungError),
    #[error("Projector group degree mismatch: expected {expected}, got {actual}")]
    DegreeMismatch { expected: usize, actual: usize },
    #[error("Invalid permutation image {image} at position {index} for degree {degree}")]
    InvalidPermutation {
        index: usize,
        image: usize,
        degree: usize,
    },
    #[error("Projector stabilizer group cannot be constructed from an empty generator set")]
    EmptyGeneratorSet,
}

pub fn validate_perm(perm: &[usize], degree: usize) -> Result<(), GroupProjectorError> {
    if perm.len() != degree {
        return Err(GroupProjectorError::DegreeMismatch {
            expected: degree,
            actual: perm.len(),
        });
    }

    let mut seen = vec![false; degree];
    for (index, &image) in perm.iter().enumerate() {
        if image >= degree || seen[image] {
            return Err(GroupProjectorError::InvalidPermutation {
                index,
                image,
                degree,
            });
        }
        seen[image] = true;
    }

    Ok(())
}

pub fn build_stabilizer_group(
    degree: usize,
    generators: Vec<Perm>,
) -> Result<StabilizerGroup, GroupProjectorError> {
    if generators.is_empty() {
        return Err(GroupProjectorError::EmptyGeneratorSet);
    }

    for generator in &generators {
        validate_perm(generator, degree)?;
    }

    let sgs = schreier_sims(&[], &generators, degree);
    Ok(StabilizerGroup {
        degree,
        generators,
        sgs,
    })
}

pub fn build_group_backed_projector(
    tableau: &YoungTableau,
    normalization: ProjectorNormalization,
) -> Result<GroupBackedProjector, GroupProjectorError> {
    let diagram = tableau.shape()?;
    let degree = diagram.n_cells();

    let mut row_generators = row_symmetrizer_generators(tableau, degree)?;
    let mut column_generators = column_antisymmetrizer_generators(tableau, degree)?;
    if row_generators.is_empty() {
        row_generators.push(identity(degree));
    }
    if column_generators.is_empty() {
        column_generators.push(identity(degree));
    }

    Ok(GroupBackedProjector {
        diagram,
        tableau: tableau.clone(),
        row_group: build_stabilizer_group(degree, row_generators)?,
        column_group: build_stabilizer_group(degree, column_generators)?,
        normalization,
    })
}

pub fn build_projector_with_trace(
    tableau: &YoungTableau,
    normalization: ProjectorNormalization,
) -> Result<(GroupBackedProjector, ProjectorBuildTrace), GroupProjectorError> {
    let projector = build_group_backed_projector(tableau, normalization)?;
    let expanded_term_count = expand_projector_group_algebra(&projector)?.len();
    let trace = ProjectorBuildTrace {
        shape: projector.diagram.rows.clone(),
        degree: projector.row_group.degree,
        row_generator_count: projector.row_group.generators.len(),
        column_generator_count: projector.column_group.generators.len(),
        expanded_term_count,
    };
    Ok((projector, trace))
}

pub fn row_group_orbits(projector: &GroupBackedProjector) -> Vec<usize> {
    all_orbits(&projector.row_group.generators, projector.row_group.degree)
}

pub fn column_group_orbits(projector: &GroupBackedProjector) -> Vec<usize> {
    all_orbits(
        &projector.column_group.generators,
        projector.column_group.degree,
    )
}

pub fn expand_projector_group_algebra(
    projector: &GroupBackedProjector,
) -> Result<Vec<PermutationTerm>, GroupProjectorError> {
    let row_elements =
        enumerate_subgroup(&projector.row_group.generators, projector.row_group.degree);
    let column_elements = enumerate_subgroup(
        &projector.column_group.generators,
        projector.column_group.degree,
    );

    let normalisation = projector_normalisation(projector)?;
    let mut combined = std::collections::BTreeMap::<Vec<usize>, BigRational>::new();
    for row in &row_elements {
        for column in &column_elements {
            let images = product(column, row);
            let coefficient =
                BigRational::from_integer(sign(column).into()) * normalisation.clone();
            *combined.entry(images).or_insert_with(BigRational::zero) += coefficient;
        }
    }

    Ok(combined
        .into_iter()
        .filter_map(|(images, coefficient)| {
            (!coefficient.is_zero()).then_some(PermutationTerm {
                images,
                coefficient,
            })
        })
        .collect())
}

pub fn canonicalize_slots_under_row_group(
    projector: &GroupBackedProjector,
    slots: &[usize],
) -> Result<Vec<usize>, GroupProjectorError> {
    if slots.len() != projector.row_group.degree {
        return Err(GroupProjectorError::DegreeMismatch {
            expected: projector.row_group.degree,
            actual: slots.len(),
        });
    }

    let row_elements =
        enumerate_subgroup(&projector.row_group.generators, projector.row_group.degree);
    let mut best = slots.to_vec();
    for element in row_elements {
        let candidate = apply_perm_to_slots(slots, &element)?;
        if candidate < best {
            best = candidate;
        }
    }
    Ok(best)
}

pub fn canonicalize_slots_under_both_groups(
    projector: &GroupBackedProjector,
    slots: &[usize],
) -> Result<Vec<usize>, GroupProjectorError> {
    if slots.len() != projector.row_group.degree {
        return Err(GroupProjectorError::DegreeMismatch {
            expected: projector.row_group.degree,
            actual: slots.len(),
        });
    }

    let mut generators = projector.row_group.generators.clone();
    generators.extend(projector.column_group.generators.iter().cloned());
    let elements = enumerate_subgroup(&generators, projector.row_group.degree);
    let mut best = slots.to_vec();
    for element in elements {
        let candidate = apply_perm_to_slots(slots, &element)?;
        if candidate < best {
            best = candidate;
        }
    }
    Ok(best)
}

pub fn canonicalize_slots_with_trace(
    projector: &GroupBackedProjector,
    slots: &[usize],
) -> Result<(Vec<usize>, CanonicalizationTrace), GroupProjectorError> {
    if slots.len() != projector.row_group.degree {
        return Err(GroupProjectorError::DegreeMismatch {
            expected: projector.row_group.degree,
            actual: slots.len(),
        });
    }

    let mut generators = projector.row_group.generators.clone();
    generators.extend(projector.column_group.generators.iter().cloned());
    let elements = enumerate_subgroup(&generators, projector.row_group.degree);
    let candidate_count = elements.len();

    let mut best = slots.to_vec();
    for element in elements {
        let candidate = apply_perm_to_slots(slots, &element)?;
        if candidate < best {
            best = candidate;
        }
    }

    let trace = CanonicalizationTrace {
        input_slots: slots.to_vec(),
        candidate_count,
        canonical_slots: best.clone(),
    };
    Ok((best, trace))
}

fn apply_perm_to_slots(slots: &[usize], perm: &[usize]) -> Result<Vec<usize>, GroupProjectorError> {
    validate_perm(perm, slots.len())?;
    Ok((0..slots.len()).map(|index| slots[perm[index]]).collect())
}

fn projector_normalisation(
    projector: &GroupBackedProjector,
) -> Result<BigRational, GroupProjectorError> {
    match projector.normalization {
        ProjectorNormalization::Unnormalized => Ok(BigRational::one()),
        ProjectorNormalization::HookLength | ProjectorNormalization::Idempotent => Ok(
            BigRational::new(1.into(), projector.diagram.hook_length_product()?),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YoungDiagram;

    fn standard(rows: Vec<usize>) -> YoungTableau {
        let diagram = YoungDiagram::try_new(rows).unwrap();
        YoungTableau::standard(&diagram).unwrap()
    }

    #[test]
    fn validate_perm_accepts_transposition() {
        assert_eq!(validate_perm(&[1, 0], 2), Ok(()));
    }

    #[test]
    fn validate_perm_rejects_duplicate_image() {
        assert_eq!(
            validate_perm(&[0, 0], 2),
            Err(GroupProjectorError::InvalidPermutation {
                index: 1,
                image: 0,
                degree: 2,
            })
        );
    }

    #[test]
    fn build_stabilizer_group_rejects_empty_generators() {
        assert_eq!(
            build_stabilizer_group(2, vec![]),
            Err(GroupProjectorError::EmptyGeneratorSet)
        );
    }

    #[test]
    fn group_backed_projector_substitutes_identity_for_empty_column_group() {
        let projector =
            build_group_backed_projector(&standard(vec![2]), ProjectorNormalization::Unnormalized)
                .unwrap();
        assert_eq!(projector.row_group.degree, 2);
        assert_eq!(projector.column_group.generators, vec![identity(2)]);
    }

    #[test]
    fn group_backed_projector_substitutes_identity_for_empty_row_group() {
        let projector = build_group_backed_projector(
            &standard(vec![1, 1]),
            ProjectorNormalization::Unnormalized,
        )
        .unwrap();
        assert_eq!(projector.column_group.degree, 2);
        assert_eq!(projector.row_group.generators, vec![identity(2)]);
    }

    #[test]
    fn row_group_orbits_merge_two_boxes_in_one_row() {
        let projector =
            build_group_backed_projector(&standard(vec![2]), ProjectorNormalization::Unnormalized)
                .unwrap();
        let labels = row_group_orbits(&projector);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn column_group_orbits_merge_two_boxes_in_one_column() {
        let projector = build_group_backed_projector(
            &standard(vec![1, 1]),
            ProjectorNormalization::Unnormalized,
        )
        .unwrap();
        let labels = column_group_orbits(&projector);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn expand_projector_group_algebra_identity_for_single_box() {
        let projector =
            build_group_backed_projector(&standard(vec![1]), ProjectorNormalization::Unnormalized)
                .unwrap();
        let terms = expand_projector_group_algebra(&projector).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].images, vec![0]);
        assert_eq!(terms[0].coefficient, BigRational::one());
    }

    #[test]
    fn canonicalize_slots_under_row_group_sorts_two_row_slots() {
        let projector =
            build_group_backed_projector(&standard(vec![2]), ProjectorNormalization::Unnormalized)
                .unwrap();
        assert_eq!(
            canonicalize_slots_under_row_group(&projector, &[9, 3]).unwrap(),
            vec![3, 9]
        );
    }

    #[test]
    fn canonicalize_slots_under_both_groups_sorts_two_column_slots() {
        let projector = build_group_backed_projector(
            &standard(vec![1, 1]),
            ProjectorNormalization::Unnormalized,
        )
        .unwrap();
        assert_eq!(
            canonicalize_slots_under_both_groups(&projector, &[9, 3]).unwrap(),
            vec![3, 9]
        );
    }

    #[test]
    fn shape_two_one_projector_expands_deterministically_without_duplicates() {
        let projector = build_group_backed_projector(
            &standard(vec![2, 1]),
            ProjectorNormalization::Unnormalized,
        )
        .unwrap();
        let terms = expand_projector_group_algebra(&projector).unwrap();
        assert!(!terms.is_empty());
        let mut sorted = terms
            .iter()
            .map(|term| term.images.clone())
            .collect::<Vec<_>>();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted, deduped);
        sorted.sort();
        assert_eq!(
            terms
                .iter()
                .map(|term| term.images.clone())
                .collect::<Vec<_>>(),
            sorted
        );
    }

    #[test]
    fn projector_build_trace_reports_real_counts_for_two_one() {
        let (projector, trace) =
            build_projector_with_trace(&standard(vec![2, 1]), ProjectorNormalization::Unnormalized)
                .unwrap();
        assert_eq!(trace.shape, vec![2, 1]);
        assert_eq!(trace.degree, 3);
        assert_eq!(
            trace.row_generator_count,
            projector.row_group.generators.len()
        );
        assert_eq!(
            trace.column_generator_count,
            projector.column_group.generators.len()
        );
        assert!(trace.row_generator_count >= 1);
        assert!(trace.column_generator_count >= 1);
    }

    #[test]
    fn canonicalization_trace_reports_candidates_and_canonical_slots() {
        let projector =
            build_group_backed_projector(&standard(vec![2]), ProjectorNormalization::Unnormalized)
                .unwrap();
        let (canonical_slots, trace) = canonicalize_slots_with_trace(&projector, &[9, 3]).unwrap();
        assert_eq!(canonical_slots, vec![3, 9]);
        assert_eq!(trace.input_slots, vec![9, 3]);
        assert_eq!(trace.canonical_slots, vec![3, 9]);
        assert!(trace.candidate_count >= 1);
    }
}
