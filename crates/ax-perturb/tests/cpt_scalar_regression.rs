use ax_ir::{Expr, Interner};
use ax_perturb::{
    bardeen_variations, default_scalar_gauge_generator, derive_mukhanov_sasaki_from_action,
    gauge::svt_decompose_perturbation, project_scalar_equations_to_harmonic_space,
    standard_canonical_scalar_symbols, FrwBackgroundSpec,
};

fn contains_symbol(expr: &Expr, symbol: lasso::Spur) -> bool {
    match expr {
        Expr::Sym(sym) => *sym == symbol,
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
            items.iter().any(|item| contains_symbol(item, symbol))
        }
        Expr::Pow(base, exp) => contains_symbol(base, symbol) || contains_symbol(exp, symbol),
        Expr::Neg(inner) | Expr::Group(inner, _) => contains_symbol(inner, symbol),
        Expr::Call(_, args) => args.iter().any(|arg| contains_symbol(arg, symbol)),
        Expr::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|item| contains_symbol(item, symbol)),
        _ => false,
    }
}

fn assert_no_spatial_derivatives(rendered: &str) {
    assert!(!rendered.contains(", x)"), "got {rendered}");
    assert!(!rendered.contains(", y)"), "got {rendered}");
    assert!(!rendered.contains(", z)"), "got {rendered}");
}

#[test]
fn bardeen_invariance_regression() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let generator = default_scalar_gauge_generator(&interner);
    let variations = bardeen_variations(&bg, &generator, &interner).unwrap();

    assert_eq!(variations.len(), 2);
    assert_eq!(variations[0].variation, Expr::zero());
    assert_eq!(variations[1].variation, Expr::zero());
}

#[test]
fn linear_scalar_equations_have_stable_labels_and_count() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let decomp = svt_decompose_perturbation(3, &interner).unwrap();
    let equations =
        ax_perturb::cosmology::linearized_einstein_scalar(&bg, &decomp, &interner).unwrap();
    let labels = equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec!["00_constraint", "0i_momentum", "ij_trace", "ij_traceless"]
    );
}

#[test]
fn linear_scalar_equations_project_consistently_to_harmonic_space() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let decomp = svt_decompose_perturbation(3, &interner).unwrap();
    let equations =
        ax_perturb::cosmology::linearized_einstein_scalar(&bg, &decomp, &interner).unwrap();
    let projected = project_scalar_equations_to_harmonic_space(&equations, &bg, &interner).unwrap();
    let k = interner.get_or_intern("k");

    for equation in &projected.equations {
        assert!(
            contains_symbol(&equation.expr, k),
            "missing k in {:?}",
            equation.expr
        );
        let rendered = ax_ir::pretty_print(&equation.expr, &interner);
        assert_no_spatial_derivatives(&rendered);
    }
}

#[test]
fn fluid_equations_have_stable_labels() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let equations =
        ax_perturb::cosmology::perfect_fluid_linear_conservation(&bg, &interner).unwrap();
    let labels = equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(labels, vec!["fluid_continuity", "fluid_euler"]);
}

#[test]
fn mukhanov_sasaki_action_and_public_api_match() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_canonical_scalar_symbols(&interner);
    let from_action = derive_mukhanov_sasaki_from_action(&bg, &symbols, &interner).unwrap();
    let from_api =
        ax_perturb::cosmology::mukhanov_sasaki_equation(&bg, symbols.slow_roll_epsilon, &interner)
            .unwrap();

    assert_eq!(from_action.fourier_space_equation, from_api);
}
