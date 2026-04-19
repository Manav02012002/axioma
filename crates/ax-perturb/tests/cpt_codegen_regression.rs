use ax_codegen::{emit_cpp_function, emit_python_function, emit_rust_function};
use ax_ir::{Expr, Interner};
use ax_perturb::{
    export_boltzmann_bridge_system, mukhanov_sasaki_first_order_system,
    standard_canonical_scalar_symbols, symbolic_boltzmann_bridge_system, FrwBackgroundSpec,
};

fn substitute_symbol(expr: &Expr, target: lasso::Spur, replacement: &Expr) -> Expr {
    match expr {
        Expr::Sym(sym) if *sym == target => replacement.clone(),
        Expr::Add(items) => Expr::add(
            items
                .iter()
                .map(|item| substitute_symbol(item, target, replacement))
                .collect(),
        ),
        Expr::Mul(items) => Expr::mul(
            items
                .iter()
                .map(|item| substitute_symbol(item, target, replacement))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_symbol(base, target, replacement),
            substitute_symbol(exp, target, replacement),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_symbol(inner, target, replacement)),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| substitute_symbol(arg, target, replacement))
                .collect(),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_symbol(item, target, replacement))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| substitute_symbol(item, target, replacement))
                        .collect()
                })
                .collect(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(substitute_symbol(inner, target, replacement)),
            *rel,
        ),
        other => other.clone(),
    }
}

#[test]
fn scalar_mode_rhs_codegen_is_deterministic_across_repeated_calls() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let symbols = standard_canonical_scalar_symbols(&interner);
    let system = mukhanov_sasaki_first_order_system(&bg, &symbols, &interner).unwrap();
    let Expr::Sym(lhs_second) = &system[1].0 else {
        panic!("expected symbol lhs");
    };
    let v1 = interner.get_or_intern("v1");
    let rhs = substitute_symbol(&system[1].1, *lhs_second, &Expr::Sym(v1));
    let args = vec![
        interner.get_or_intern("eta"),
        interner.get_or_intern("v"),
        v1,
        interner.get_or_intern("k"),
        interner.get_or_intern("c_s"),
        interner.get_or_intern("a"),
        interner.get_or_intern("epsilon"),
    ];

    let python_a = emit_python_function("ms_rhs", &args, &rhs, &interner);
    let python_b = emit_python_function("ms_rhs", &args, &rhs, &interner);
    let rust_a = emit_rust_function("ms_rhs", &args, &rhs, &interner);
    let rust_b = emit_rust_function("ms_rhs", &args, &rhs, &interner);
    let cpp_a = emit_cpp_function("ms_rhs", &args, &rhs, &interner);
    let cpp_b = emit_cpp_function("ms_rhs", &args, &rhs, &interner);

    assert_eq!(python_a, python_b);
    assert_eq!(rust_a, rust_b);
    assert_eq!(cpp_a, cpp_b);
}

#[test]
fn boltzmann_bridge_codegen_is_deterministic_across_repeated_calls() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();

    for target in ["python", "rust", "cpp"] {
        let first = export_boltzmann_bridge_system(target, &system, &interner).unwrap();
        let second = export_boltzmann_bridge_system(target, &system, &interner).unwrap();
        assert_eq!(first, second, "target={target}");
    }
}

#[test]
fn json_export_is_stable_and_contains_expected_keys() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let system = symbolic_boltzmann_bridge_system(&bg, &interner).unwrap();
    let first = export_boltzmann_bridge_system("json", &system, &interner).unwrap();
    let second = export_boltzmann_bridge_system("json", &system, &interner).unwrap();

    assert_eq!(first, second);
    assert!(first.contains("\"variables\""), "{first}");
    assert!(first.contains("\"equations\""), "{first}");
}
