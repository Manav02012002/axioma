use ax_ir::{Expr, Interner};
use ax_perturb::{
    derive_linear_tensor_einstein_equations, derive_linear_vector_einstein_equations_poisson,
    derive_tensor_mode_equations, project_tensor_equations_to_harmonic_space, FrwBackgroundSpec,
};

fn diff(expr: Expr, var: lasso::Spur, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn int(n: i64) -> Expr {
    Expr::Int(n.into())
}

#[test]
fn linear_vector_equations_have_stable_labels() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let equations = derive_linear_vector_einstein_equations_poisson(&bg, &interner)
        .unwrap()
        .equations;
    let labels = equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec![
            "vector_0x_momentum",
            "vector_0y_momentum",
            "vector_0z_momentum",
            "vector_x_evolution",
            "vector_y_evolution",
            "vector_z_evolution",
        ]
    );
}

#[test]
fn linear_tensor_equations_have_stable_labels() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let equations = derive_linear_tensor_einstein_equations(&bg, &interner)
        .unwrap()
        .equations;
    let labels = equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec![
            "tensor_xx",
            "tensor_xy",
            "tensor_xz",
            "tensor_yy",
            "tensor_yz",
            "tensor_zz",
        ]
    );
}

#[test]
fn tensor_mode_equations_from_action_have_expected_plus_cross_forms() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let derivation = derive_tensor_mode_equations(&bg, &interner).unwrap();
    let eta = bg.conformal_time;
    let hubble = Expr::Sym(bg.conformal_hubble);
    let k = Expr::Sym(interner.get_or_intern("k"));
    let k_sq = Expr::pow(k, int(2));
    let h_plus = Expr::Sym(derivation.action.plus_mode);
    let h_cross = Expr::Sym(derivation.action.cross_mode);

    let expected_plus = Expr::add(vec![
        diff(diff(h_plus.clone(), eta, &interner), eta, &interner),
        Expr::mul(vec![int(2), hubble.clone(), diff(h_plus, eta, &interner)]),
        Expr::mul(vec![k_sq.clone(), Expr::Sym(derivation.action.plus_mode)]),
    ]);
    let expected_cross = Expr::add(vec![
        diff(diff(h_cross.clone(), eta, &interner), eta, &interner),
        Expr::mul(vec![int(2), hubble, diff(h_cross, eta, &interner)]),
        Expr::mul(vec![k_sq, Expr::Sym(derivation.action.cross_mode)]),
    ]);

    assert_eq!(derivation.plus_equation_fourier_space, expected_plus);
    assert_eq!(derivation.cross_equation_fourier_space, expected_cross);
}

#[test]
fn tensor_harmonic_projection_removes_explicit_spatial_derivatives() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let equations = derive_linear_tensor_einstein_equations(&bg, &interner)
        .unwrap()
        .equations;
    let projected = project_tensor_equations_to_harmonic_space(&equations, &bg, &interner).unwrap();

    for equation in &projected.equations {
        let rendered = ax_ir::pretty_print(&equation.expr, &interner);
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
    }
}
