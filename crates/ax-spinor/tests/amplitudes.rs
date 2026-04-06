use ax_spinor::{
    bcfw_decomposition, parke_taylor, parke_taylor_conjugate, three_point_mhv, BCFWShift, Label,
};

#[test]
fn parke_taylor_4pt_correct_structure() {
    let amp = parke_taylor(4, Label::new(1), Label::new(2));
    assert_eq!(
        amp.mass_dimension(),
        0,
        "4-pt amplitude should have mass dimension 0"
    );
}

#[test]
fn parke_taylor_5pt_mass_dimension() {
    let amp = parke_taylor(5, Label::new(1), Label::new(2));
    assert_eq!(
        amp.mass_dimension(),
        -1,
        "5-pt amplitude should have mass dimension -1"
    );
}

#[test]
fn parke_taylor_little_group() {
    let amp = parke_taylor(4, Label::new(1), Label::new(2));
    assert_eq!(
        amp.little_group_weight(Label::new(1)),
        2,
        "particle 1 (h=-1) should have weight +2"
    );
    assert_eq!(
        amp.little_group_weight(Label::new(3)),
        -2,
        "particle 3 (h=+1) should have weight -2"
    );
}

#[test]
fn parke_taylor_conjugate_structure() {
    let amp = parke_taylor_conjugate(4, Label::new(3), Label::new(4));
    assert_eq!(amp.mass_dimension(), 0);
}

#[test]
fn three_point_mhv_mass_dimension() {
    let amp = three_point_mhv([Label::new(1), Label::new(2), Label::new(3)]);
    assert_eq!(amp.mass_dimension(), 1, "3-pt MHV mass dim wrong");
}

#[test]
fn bcfw_decomposition_4pt_channels() {
    let shift = BCFWShift {
        shifted_angle: Label::new(1),
        shifted_square: Label::new(2),
    };
    let helicities = vec![-1, -1, 1, 1];
    let terms = bcfw_decomposition(4, &shift, &helicities);
    assert_eq!(
        terms.len(),
        2,
        "4-pt BCFW should have 2 terms (one channel, two helicities)"
    );
    for term in &terms {
        assert!(
            term.left_particles.contains(&Label::new(1))
                || term.right_particles.contains(&Label::new(1))
        );
        assert!(
            term.left_particles.contains(&Label::new(2))
                || term.right_particles.contains(&Label::new(2))
        );
    }
}
