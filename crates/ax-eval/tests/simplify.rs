use ax_eval::*;
use ax_ir::*;

#[allow(dead_code)]
fn simplify_str(code: &str) -> String {
    let interner = Interner::new();
    let env = Env::new();
    let result = ax_core_ir::lower(code, &interner);
    assert!(
        result.errors.is_empty(),
        "lowering failed for {code:?}: {:?}",
        result.errors
    );
    let expr = result.expr.unwrap_or_else(Expr::zero);
    let evaled = eval(&expr, &env, &interner);
    ax_ir::pretty_print(&evaled, &interner)
}

fn simplify_expr(code: &str) -> (Expr, Interner) {
    let interner = Interner::new();
    let env = Env::new();
    let result = ax_core_ir::lower(code, &interner);
    assert!(
        result.errors.is_empty(),
        "lowering failed for {code:?}: {:?}",
        result.errors
    );
    let expr = result.expr.unwrap_or_else(Expr::zero);
    let evaled = eval(&expr, &env, &interner);
    (evaled, interner)
}

#[test]
fn simplify_pythag() {
    // sin(x)^2 + cos(x)^2 = 1
    let (result, _) = simplify_expr("simplify(sin(x)^2 + cos(x)^2)");
    assert_eq!(
        result,
        Expr::one(),
        "sin²+cos² should be 1, got {:?}",
        result
    );
}

#[test]
fn simplify_double_neg() {
    let (result, interner) = simplify_expr("--x");
    let x = interner.get_or_intern("x");
    assert_eq!(result, Expr::Sym(x), "--x should be x");
}

#[test]
fn simplify_zero_add() {
    let (result, interner) = simplify_expr("x + 0");
    let x = interner.get_or_intern("x");
    assert_eq!(result, Expr::Sym(x), "x+0 should be x");
}

#[test]
fn simplify_zero_mul() {
    let (result, _) = simplify_expr("x * 0");
    assert_eq!(result, Expr::zero(), "x*0 should be 0");
}

#[test]
fn simplify_one_mul() {
    let (result, interner) = simplify_expr("x * 1");
    let x = interner.get_or_intern("x");
    assert_eq!(result, Expr::Sym(x), "x*1 should be x");
}

#[test]
fn simplify_power_zero() {
    let (result, _) = simplify_expr("x^0");
    assert_eq!(result, Expr::one(), "x^0 should be 1");
}

#[test]
fn simplify_power_one() {
    let (result, interner) = simplify_expr("x^1");
    let x = interner.get_or_intern("x");
    assert_eq!(result, Expr::Sym(x), "x^1 should be x");
}

#[test]
fn simplify_collect_terms() {
    // 3x + 5x = 8x
    let (result, interner) = simplify_expr("3*x + 5*x");
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(7.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::Int(56.into()),
        "3x+5x at x=7 should be 56, got {:?}",
        val
    );
}

#[test]
fn simplify_expand_square() {
    // (x+1)^2 expanded should be x^2 + 2x + 1
    let (result, interner) = simplify_expr("expand((x + 1)^2)");
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(3.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::Int(16.into()),
        "(3+1)^2 should be 16, got {:?}",
        val
    );
}

#[test]
fn simplify_rational_arithmetic() {
    let (result, _) = simplify_expr("1/3 + 1/6");
    let expected = Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
    assert_eq!(
        result, expected,
        "1/3 + 1/6 should be 1/2, got {:?}",
        result
    );
}

#[test]
fn simplify_nested_fractions() {
    let (result, _) = simplify_expr("(1/2) / (3/4)");
    let expected = Expr::Rational(num_rational::BigRational::new(2.into(), 3.into()));
    assert_eq!(
        result, expected,
        "(1/2)/(3/4) should be 2/3, got {:?}",
        result
    );
}

#[test]
fn simplify_sqrt_perfect() {
    let (result, _) = simplify_expr("sqrt(9)");
    assert_eq!(
        result,
        Expr::Int(3.into()),
        "sqrt(9) should be 3, got {:?}",
        result
    );
}

#[test]
fn simplify_abs_negative() {
    let (result, _) = simplify_expr("abs(-5)");
    assert_eq!(
        result,
        Expr::Int(5.into()),
        "abs(-5) should be 5, got {:?}",
        result
    );
}
