use ax_young::{
    branch_gl_n_to_gl_n_minus_1, branch_s_n_to_s_n_minus_1, enumerate_semistandard_with_content,
    kostka_number_exact, littlewood_richardson_coefficient, plethysm_schur_by_shape,
    schouten_annihilates_antisym_degree, selfdual_eigenspace_dimension,
    tensor_product_decomposition, vanishes_in_gl_dimension, YoungDiagram,
};
use num_bigint::BigInt;

fn yd(rows: &[usize]) -> YoungDiagram {
    YoungDiagram::try_new(rows.to_vec()).expect("valid diagram")
}

fn schur_terms(expansion: &ax_young::SchurExpansion) -> Vec<(Vec<usize>, BigInt)> {
    expansion
        .terms
        .iter()
        .map(|(shape, coeff)| (shape.rows.clone(), coeff.clone()))
        .collect()
}

#[test]
fn exact_kostka_regressions_hold() {
    assert_eq!(
        kostka_number_exact(&yd(&[3]), &[1, 1, 1]).unwrap(),
        BigInt::from(1usize)
    );
    assert_eq!(
        kostka_number_exact(&yd(&[2, 1]), &[1, 1, 1]).unwrap(),
        BigInt::from(2usize)
    );
    assert_eq!(
        kostka_number_exact(&yd(&[1, 1, 1]), &[3]).unwrap(),
        BigInt::from(0usize)
    );
}

#[test]
fn semistandard_enumeration_is_exact_and_deterministic() {
    let tableaux = enumerate_semistandard_with_content(&yd(&[2, 1]), &[1, 1, 1]).unwrap();
    assert_eq!(
        tableaux
            .into_iter()
            .map(|tableau| tableau.rows)
            .collect::<Vec<_>>(),
        vec![vec![vec![1, 2], vec![3]], vec![vec![1, 3], vec![2]]]
    );
}

#[test]
fn exact_lr_coefficients_match_required_regressions() {
    let cases = [
        (vec![1], vec![1], vec![2]),
        (vec![1], vec![1], vec![1, 1]),
        (vec![2], vec![1], vec![3]),
        (vec![2], vec![1], vec![2, 1]),
        (vec![1, 1], vec![1], vec![2, 1]),
        (vec![1, 1], vec![1], vec![1, 1, 1]),
        (vec![2, 1], vec![1], vec![3, 1]),
        (vec![2, 1], vec![1], vec![2, 2]),
    ];

    for (left, right, target) in cases {
        assert_eq!(
            littlewood_richardson_coefficient(&yd(&left), &yd(&right), &yd(&target)).unwrap(),
            BigInt::from(1usize),
            "unexpected LR coefficient for {:?} x {:?} -> {:?}",
            left,
            right,
            target
        );
    }
}

#[test]
fn tensor_product_multiplicities_are_exact() {
    let decomposition = tensor_product_decomposition(&[yd(&[1]), yd(&[1]), yd(&[1])]).unwrap();
    assert_eq!(
        decomposition
            .irreps
            .iter()
            .map(|space| (space.shape.rows.clone(), space.multiplicity))
            .collect::<Vec<_>>(),
        vec![
            (vec![1, 1, 1], 1usize),
            (vec![2, 1], 2usize),
            (vec![3], 1usize)
        ]
    );

    let decomposition = tensor_product_decomposition(&[yd(&[2]), yd(&[2])]).unwrap();
    assert_eq!(
        decomposition
            .irreps
            .iter()
            .map(|space| (space.shape.rows.clone(), space.multiplicity))
            .collect::<Vec<_>>(),
        vec![
            (vec![2, 2], 1usize),
            (vec![3, 1], 1usize),
            (vec![4], 1usize)
        ]
    );
}

#[test]
fn plethysm_expansions_are_exact() {
    assert_eq!(
        schur_terms(&plethysm_schur_by_shape(&yd(&[1]), &yd(&[2])).unwrap()),
        vec![(vec![2], BigInt::from(1usize))]
    );
    assert_eq!(
        schur_terms(&plethysm_schur_by_shape(&yd(&[2]), &yd(&[1])).unwrap()),
        vec![(vec![2], BigInt::from(1usize))]
    );
    assert_eq!(
        schur_terms(&plethysm_schur_by_shape(&yd(&[1, 1]), &yd(&[1])).unwrap()),
        vec![(vec![1, 1], BigInt::from(1usize))]
    );
    assert_eq!(
        schur_terms(&plethysm_schur_by_shape(&yd(&[2]), &yd(&[2])).unwrap()),
        vec![
            (vec![2, 2], BigInt::from(1usize)),
            (vec![4], BigInt::from(1usize)),
        ]
    );
    assert_eq!(
        schur_terms(&plethysm_schur_by_shape(&yd(&[1, 1]), &yd(&[2])).unwrap()),
        vec![(vec![3, 1], BigInt::from(1usize))]
    );
}

#[test]
fn branching_rules_are_exact() {
    assert_eq!(
        branch_gl_n_to_gl_n_minus_1(&yd(&[2, 1]), 3)
            .unwrap()
            .into_iter()
            .map(|shape| shape.rows)
            .collect::<Vec<_>>(),
        vec![vec![1], vec![1, 1], vec![2], vec![2, 1]]
    );
    assert_eq!(
        branch_s_n_to_s_n_minus_1(&yd(&[2, 1]))
            .unwrap()
            .into_iter()
            .map(|shape| shape.rows)
            .collect::<Vec<_>>(),
        vec![vec![1, 1], vec![2]]
    );
    assert_eq!(
        branch_s_n_to_s_n_minus_1(&yd(&[3]))
            .unwrap()
            .into_iter()
            .map(|shape| shape.rows)
            .collect::<Vec<_>>(),
        vec![vec![2]]
    );
}

#[test]
fn dimension_identity_regressions_are_exact() {
    assert!(vanishes_in_gl_dimension(&yd(&[1, 1, 1]), 2));
    assert!(!vanishes_in_gl_dimension(&yd(&[2, 1]), 2));
    assert!(schouten_annihilates_antisym_degree(5, 4));
    assert_eq!(selfdual_eigenspace_dimension(2, 4).unwrap(), (3, 3));
}
