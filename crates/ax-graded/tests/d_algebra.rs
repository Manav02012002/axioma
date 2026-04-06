use ax_graded::d_algebra::*;
use ax_graded::superspace::*;
use ax_graded::*;
use ax_ir::Expr;

#[test]
fn d_alpha_on_theta_same_index() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let theta0_expr = Expr::Sym(setup.theta[0]);
    let result = apply_d_alpha(&theta0_expr, 0, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::one(),
        "D_0(theta_0) should be 1, got {:?}",
        simplified
    );
}

#[test]
fn d_alpha_on_theta_different_index() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let theta1_expr = Expr::Sym(setup.theta[1]);
    let result = apply_d_alpha(&theta1_expr, 0, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "D_0(theta_1) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn d_alpha_on_bosonic_symbol() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let c = interner.get_or_intern("c");
    let result = apply_d_alpha(&Expr::Sym(c), 0, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "D_0(constant) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn d_squared_on_theta_theta() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let theta_theta = Expr::mul(vec![Expr::Sym(setup.theta[0]), Expr::Sym(setup.theta[1])]);
    let result = d_squared(&theta_theta, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::Int((-2).into()),
        "D^2(theta^0 theta^1) should be -2, got {:?}",
        simplified
    );
}

#[test]
fn d_bar_d_anticommutator() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let tbar0 = Expr::Sym(setup.theta_bar[0]);
    let d_result = apply_d_alpha(&tbar0, 0, &setup, &table, &interner);
    let simplified = graded_simplify(&d_result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "D_0(theta_bar_0) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn chiral_condition() {
    let interner = ax_ir::Interner::new();
    let (setup, table) = setup_n1_superspace(&interner);
    let phi_name = interner.get_or_intern("Phi");
    let expansion = expand_superfield(phi_name, &setup, &interner);
    let chiral = chiral_constraint(&expansion, &setup, &interner);
    let chiral_expr = superfield_to_expr(&chiral, &setup, &interner);
    let d_bar_0 = apply_d_bar_alpha_dot(&chiral_expr, 0, &setup, &table, &interner);
    let result = graded_simplify(&d_bar_0, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "D_bar_0 on chiral superfield should be 0, got {:?}",
        result
    );
}
