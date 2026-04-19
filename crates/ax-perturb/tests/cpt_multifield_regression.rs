use ax_ir::Interner;
use ax_perturb::{
    derive_multifield_curvature_entropy_equations, standard_multifield_symbols,
    symbolic_boltzmann_bridge_system, FrwBackgroundSpec,
};

#[test]
fn twofield_multifield_system_returns_curvature_and_entropy_equations() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_multifield_symbols(2, &interner).unwrap();
    let system = derive_multifield_curvature_entropy_equations(&bg, &symbols, &interner).unwrap();

    assert_eq!(system.equations.len(), 2);
}

#[test]
fn threefield_multifield_system_returns_curvature_and_two_entropy_equations() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_multifield_symbols(3, &interner).unwrap();
    let system = derive_multifield_curvature_entropy_equations(&bg, &symbols, &interner).unwrap();

    assert_eq!(system.equations.len(), 3);
}

#[test]
fn multifield_equation_labels_are_stable() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_multifield_symbols(3, &interner).unwrap();
    let system = derive_multifield_curvature_entropy_equations(&bg, &symbols, &interner).unwrap();
    let labels = system
        .equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec![
            "multifield_curvature",
            "multifield_entropy_1",
            "multifield_entropy_2"
        ]
    );
}

#[test]
fn boltzmann_bridge_has_ten_variables_and_ten_rhs_entries() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();

    assert_eq!(system.variables.len(), 10);
    assert_eq!(system.equations.len(), 10);
}
