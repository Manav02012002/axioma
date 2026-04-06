use ax_eval::equation::*;
use ax_eval::*;
use ax_ir::*;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn make_and_check_equation() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = make_equation(Expr::Sym(x), Expr::Int(5.into()), &interner);
    assert!(is_equation(&eq, &interner));
    assert_eq!(get_lhs(&eq, &interner), Some(Expr::Sym(x)));
    assert_eq!(get_rhs(&eq, &interner), Some(Expr::Int(5.into())));
}

#[test]
fn swap_sides_works() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = make_equation(Expr::Sym(x), Expr::Int(5.into()), &interner);
    let swapped = swap_sides(&eq, &interner);
    assert_eq!(get_lhs(&swapped, &interner), Some(Expr::Int(5.into())));
    assert_eq!(get_rhs(&swapped, &interner), Some(Expr::Sym(x)));
}

#[test]
fn multiply_through_both_sides() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = make_equation(Expr::Sym(x), Expr::Int(3.into()), &interner);
    let result = multiply_through(&eq, &Expr::Int(2.into()), &interner);
    let lhs = get_lhs(&result, &interner).unwrap();
    let rhs = get_rhs(&result, &interner).unwrap();
    let mut env = Env::new();
    env.bindings.insert(x, Expr::Int(5.into()));
    let lhs_val = eval(&lhs, &env, &interner);
    let rhs_val = eval(&rhs, &env, &interner);
    assert_eq!(lhs_val, Expr::Int(10.into()), "2*x at x=5 should be 10");
    assert_eq!(rhs_val, Expr::Int(6.into()), "2*3 should be 6");
}

#[test]
fn to_rhs_moves_term() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let lhs = Expr::add(vec![Expr::Sym(x), Expr::Sym(y)]);
    let eq = make_equation(lhs, Expr::Int(5.into()), &interner);
    let result = to_rhs(&eq, &Expr::Sym(y), &interner);
    let new_lhs = get_lhs(&result, &interner).unwrap();
    let new_rhs = get_rhs(&result, &interner).unwrap();
    assert!(
        !expr_contains(&new_lhs, &Expr::Sym(y)),
        "y should be moved from LHS"
    );
    assert!(
        expr_contains(&new_rhs, &Expr::Sym(y)),
        "y should appear in RHS"
    );
}

#[test]
fn isolate_simple() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let lhs = Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(x)]);
    let eq = make_equation(lhs, Expr::Int(6.into()), &interner);
    let result = isolate(&eq, &Expr::Sym(x), &interner);
    let isolated_lhs = get_lhs(&result, &interner).unwrap();
    let isolated_rhs = get_rhs(&result, &interner).unwrap();
    assert_eq!(isolated_lhs, Expr::Sym(x), "LHS should be x");
    let rhs_val = eval(&isolated_rhs, &Env::new(), &interner);
    assert_eq!(
        rhs_val,
        Expr::Int(3.into()),
        "RHS should evaluate to 3, got {:?}",
        rhs_val
    );
}

#[test]
fn isolate_with_additive_terms() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let lhs = Expr::add(vec![Expr::Sym(x), Expr::Int(3.into())]);
    let eq = make_equation(lhs, Expr::Int(7.into()), &interner);
    let result = isolate(&eq, &Expr::Sym(x), &interner);
    let isolated_rhs = get_rhs(&result, &interner).unwrap();
    let rhs_val = eval(&isolated_rhs, &Env::new(), &interner);
    assert_eq!(
        rhs_val,
        Expr::Int(4.into()),
        "RHS should be 4, got {:?}",
        rhs_val
    );
}

#[test]
fn isolate_multiplicative_and_additive() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let lhs = Expr::add(vec![
        Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(x)]),
        Expr::Int((-9).into()),
    ]);
    let eq = make_equation(lhs, Expr::zero(), &interner);
    let result = isolate(&eq, &Expr::Sym(x), &interner);
    let isolated_rhs = get_rhs(&result, &interner).unwrap();
    let rhs_val = eval(&isolated_rhs, &Env::new(), &interner);
    assert_eq!(
        rhs_val,
        Expr::Int(3.into()),
        "x should be 3, got {:?}",
        rhs_val
    );
}

#[test]
fn eq_to_rule_converts() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = make_equation(Expr::Sym(x), Expr::Int(5.into()), &interner);
    let rule = equation_to_rule(&eq, &interner);
    match rule {
        Expr::Rule(lhs, rhs, trust) => {
            assert_eq!(*lhs, Expr::Sym(x));
            assert_eq!(*rhs, Expr::Int(5.into()));
            assert_eq!(trust, TrustLevel::Exact);
        }
        _ => panic!("should produce a Rule, got {:?}", rule),
    }
}

#[test]
fn differentiate_equation_both_sides() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let lhs = Expr::pow(Expr::Sym(x), Expr::Int(2.into()));
    let eq = make_equation(lhs, Expr::Int(4.into()), &interner);
    let result = differentiate_equation(&eq, x, &interner);
    let new_rhs = get_rhs(&result, &interner).unwrap();
    assert_eq!(new_rhs, Expr::zero(), "d/dx(4) should be 0");
    let new_lhs = get_lhs(&result, &interner).unwrap();
    let mut env = Env::new();
    env.bindings.insert(x, Expr::Int(3.into()));
    let lhs_val = eval(&new_lhs, &env, &interner);
    assert_eq!(
        lhs_val,
        Expr::Int(6.into()),
        "d/dx(x^2) at x=3 should be 6, got {:?}",
        lhs_val
    );
}

#[test]
fn not_equation_passthrough() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let expr = Expr::Sym(x);
    assert!(!is_equation(&expr, &interner));
    assert_eq!(get_lhs(&expr, &interner), None);
    assert_eq!(swap_sides(&expr, &interner), expr);
}

#[test]
fn render_equation_latex() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = make_equation(Expr::Sym(x), Expr::Int(5.into()), &interner);
    let latex = ax_render::to_latex(&eq, &interner);
    assert!(
        latex.contains("="),
        "equation should render with = sign, got {}",
        latex
    );
    assert!(latex.contains("x"), "should contain x");
    assert!(latex.contains("5"), "should contain 5");
}
