use ax_ir::Interner;
use ax_perturb::{
    derive_mukhanov_sasaki_from_action, mukhanov_sasaki_first_order_system,
    standard_canonical_scalar_symbols, tensor_mode_first_order_system, tensor_quadratic_action,
    FrwBackgroundSpec,
};

#[test]
fn single_field_reduced_action_contains_z_double_prime_over_z() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_canonical_scalar_symbols(&interner);
    let derivation = derive_mukhanov_sasaki_from_action(&bg, &symbols, &interner).unwrap();
    let rendered = ax_ir::pretty_print(&derivation.action.lagrangian_density, &interner);

    assert!(
        rendered.contains("diff(diff(z, eta), eta)"),
        "got {rendered}"
    );
    assert!(rendered.contains("z^-1"), "got {rendered}");
}

#[test]
fn tensor_reduced_action_contains_a_squared_over_eight() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let action = tensor_quadratic_action(&bg, &interner).unwrap();
    let rendered = ax_ir::pretty_print(&action.lagrangian_density, &interner);

    assert!(rendered.contains("1/8"), "got {rendered}");
    assert!(rendered.contains("a^2"), "got {rendered}");
}

#[test]
fn first_order_system_exports_have_two_equations() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let scalar_symbols = standard_canonical_scalar_symbols(&interner);
    let scalar = mukhanov_sasaki_first_order_system(&bg, &scalar_symbols, &interner).unwrap();
    let plus = tensor_mode_first_order_system(&bg, "plus", &interner).unwrap();
    let cross = tensor_mode_first_order_system(&bg, "cross", &interner).unwrap();

    assert_eq!(scalar.len(), 2);
    assert_eq!(plus.len(), 2);
    assert_eq!(cross.len(), 2);
}
