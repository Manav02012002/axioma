use ax_young::{
    build_group_backed_projector, canonicalize_slots_under_both_groups, dimension_gl,
    expand_projector_group_algebra, kostka_number, lr_shapes, ProjectorNormalization, YoungDiagram,
    YoungTableau,
};
use num_bigint::BigInt;

fn diagram(rows: &[usize]) -> YoungDiagram {
    YoungDiagram::try_new(rows.to_vec()).expect("valid diagram")
}

fn standard_tableau(rows: &[usize]) -> YoungTableau {
    YoungTableau::standard(&diagram(rows)).expect("standard tableau")
}

#[test]
fn partition_conjugation_round_trips_for_regression_shapes() {
    for rows in [vec![1], vec![2], vec![2, 1], vec![3, 2, 1]] {
        let diagram = diagram(&rows);
        assert_eq!(diagram.conjugate().unwrap().conjugate().unwrap(), diagram);
    }
}

#[test]
fn hook_content_dimensions_match_known_small_values() {
    let cases = [
        (vec![1], 4usize, BigInt::from(4usize)),
        (vec![2], 4usize, BigInt::from(10usize)),
        (vec![1, 1], 4usize, BigInt::from(6usize)),
        (vec![2, 1], 4usize, BigInt::from(20usize)),
    ];

    for (rows, n, expected) in cases {
        assert_eq!(dimension_gl(&diagram(&rows), n).unwrap(), expected);
    }
}

#[test]
fn littlewood_richardson_products_match_exact_expected_shapes() {
    let cases = [
        (vec![1], vec![1], vec![vec![1, 1], vec![2]]),
        (vec![2], vec![1], vec![vec![2, 1], vec![3]]),
        (vec![1, 1], vec![1], vec![vec![1, 1, 1], vec![2, 1]]),
    ];

    for (left, right, expected) in cases {
        let shapes = lr_shapes(&diagram(&left), &diagram(&right))
            .unwrap()
            .into_iter()
            .map(|shape| shape.rows)
            .collect::<Vec<_>>();
        assert_eq!(shapes, expected);
    }
}

#[test]
fn kostka_numbers_match_required_regressions() {
    assert_eq!(
        kostka_number(&diagram(&[2, 1]), &[2, 1]).unwrap(),
        BigInt::from(1usize)
    );
    assert_eq!(
        kostka_number(&diagram(&[2]), &[1, 1]).unwrap(),
        BigInt::from(1usize)
    );
    assert_eq!(
        kostka_number(&diagram(&[1, 1]), &[2]).unwrap(),
        BigInt::from(0usize)
    );
}

#[test]
fn group_backed_projector_expansion_is_deterministic() {
    let projector = build_group_backed_projector(
        &standard_tableau(&[2, 1]),
        ProjectorNormalization::Unnormalized,
    )
    .unwrap();

    let first = expand_projector_group_algebra(&projector).unwrap();
    let second = expand_projector_group_algebra(&projector).unwrap();
    assert_eq!(first, second);
}

#[test]
fn canonicalize_slots_under_both_groups_collapses_permutations_to_one_vector() {
    let projector = build_group_backed_projector(
        &standard_tableau(&[2, 1]),
        ProjectorNormalization::Unnormalized,
    )
    .unwrap();

    for slots in [vec![2, 0, 1], vec![1, 2, 0], vec![0, 2, 1]] {
        assert_eq!(
            canonicalize_slots_under_both_groups(&projector, &slots).unwrap(),
            vec![0, 1, 2]
        );
    }
}
