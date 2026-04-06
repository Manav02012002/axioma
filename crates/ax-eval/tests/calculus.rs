use ax_eval::*;
use ax_ir::*;

fn eval_str(code: &str) -> (Expr, Interner) {
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

#[allow(dead_code)]
fn eval_to_string(code: &str) -> String {
    let (expr, interner) = eval_str(code);
    ax_ir::pretty_print(&expr, &interner)
}

// === DIFFERENTIATION ===

#[test]
fn diff_power_rule() {
    // d/dx(x^5) = 5x^4
    let (result, interner) = eval_str("diff(x^5, x)");
    // Evaluate at x=2: should give 5*16 = 80
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(2.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::Int(80.into()),
        "d/dx(x^5) at x=2 should be 80, got {:?}",
        val
    );
}

#[test]
fn diff_chain_rule() {
    // d/dx(sin(x^2)) = 2x*cos(x^2)
    let (result, interner) = eval_str("diff(sin(x^2), x)");
    // Evaluate at x=0: 2*0*cos(0) = 0
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(0.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::zero(),
        "d/dx(sin(x^2)) at x=0 should be 0, got {:?}",
        val
    );
}

#[test]
fn diff_product_rule() {
    // d/dx(x * sin(x)) = sin(x) + x*cos(x)
    let (result, interner) = eval_str("diff(x * sin(x), x)");
    // At x=0: sin(0) + 0*cos(0) = 0
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(0.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::zero(),
        "d/dx(x*sin(x)) at x=0 should be 0, got {:?}",
        val
    );
}

#[test]
fn diff_quotient_rule() {
    // d/dx(1/x) = -1/x^2
    let (result, interner) = eval_str("diff(x^(-1), x)");
    // At x=3: -1/9
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(3.into()));
    let val = eval(&result, &env, &interner);
    let expected = Expr::Rational(num_rational::BigRational::new((-1).into(), 9.into()));
    assert_eq!(
        val, expected,
        "d/dx(1/x) at x=3 should be -1/9, got {:?}",
        val
    );
}

#[test]
fn diff_exp() {
    // d/dx(exp(x)) = exp(x)
    let (result, interner) = eval_str("diff(exp(x), x)");
    let result_str = ax_ir::pretty_print(&result, &interner);
    assert!(
        result_str.contains("exp"),
        "d/dx(exp(x)) should contain exp, got {}",
        result_str
    );
}

#[test]
fn diff_log() {
    // d/dx(log(x)) = 1/x = x^(-1)
    let (result, interner) = eval_str("diff(log(x), x)");
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(5.into()));
    let val = eval(&result, &env, &interner);
    let expected = Expr::Rational(num_rational::BigRational::new(1.into(), 5.into()));
    assert_eq!(
        val, expected,
        "d/dx(log(x)) at x=5 should be 1/5, got {:?}",
        val
    );
}

#[test]
fn diff_constant() {
    let (result, _) = eval_str("diff(7, x)");
    assert_eq!(result, Expr::zero(), "d/dx(7) should be 0");
}

#[test]
fn diff_second_derivative() {
    // d²/dx²(x^3) = 6x
    let (result, interner) = eval_str("diff(diff(x^3, x), x)");
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(4.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::Int(24.into()),
        "d²/dx²(x^3) at x=4 should be 24, got {:?}",
        val
    );
}

// === INTEGRATION ===

#[test]
fn integrate_power() {
    // ∫x^2 dx = x^3/3
    let (result, interner) = eval_str("integrate(x^2, x)");
    // Differentiate the result and check we get x^2 back
    let check = differentiate(&result, interner.get_or_intern("x"), &interner);
    let simplified = simplify::simplify(&check, &interner);
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(3.into()));
    let val = eval(&simplified, &env, &interner);
    assert_eq!(
        val,
        Expr::Int(9.into()),
        "d/dx(∫x^2 dx) at x=3 should be 9, got {:?}",
        val
    );
}

#[test]
fn integrate_sin() {
    // ∫sin(x) dx = -cos(x)
    let (result, interner) = eval_str("integrate(sin(x), x)");
    let result_str = ax_ir::pretty_print(&result, &interner);
    assert!(
        result_str.contains("cos"),
        "∫sin(x) should contain cos, got {}",
        result_str
    );
}

// === LIMITS ===

#[test]
fn limit_sinc() {
    // lim_{x→0} sin(x)/x = 1
    let (result, _) = eval_str("limit(sin(x)/x, x, 0)");
    assert_eq!(
        result,
        Expr::one(),
        "lim sin(x)/x as x→0 should be 1, got {:?}",
        result
    );
}

// === SERIES ===

#[test]
fn series_exp() {
    // exp(x) around 0 to order 3 = 1 + x + x^2/2 + x^3/6
    let (result, interner) = eval_str("series(exp(x), x, 0, 3)");
    // Evaluate at x=0: should give 1
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(0.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::one(),
        "exp(x) series at x=0 should be 1, got {:?}",
        val
    );
}

#[test]
fn series_sin() {
    // sin(x) around 0 to order 3 = x - x^3/6
    let (result, interner) = eval_str("series(sin(x), x, 0, 3)");
    let mut env = Env::new();
    env.bindings
        .insert(interner.get_or_intern("x"), Expr::Int(0.into()));
    let val = eval(&result, &env, &interner);
    assert_eq!(
        val,
        Expr::zero(),
        "sin(x) series at x=0 should be 0, got {:?}",
        val
    );
}
