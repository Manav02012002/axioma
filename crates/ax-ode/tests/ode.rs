use ax_ir::*;
use ax_ode::*;
use num_rational::BigRational;

fn int() -> Interner {
    Interner::new()
}

fn one_tenth() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 10.into()))
}

#[test]
fn rk4_exponential() {
    // y' = y, y(0) = 1 → y = e^x
    let interner = int();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let rhs = Expr::Sym(y);
    let points = rk4(&rhs, x, y, 0.0, 1.0, 1.0, 1000, &interner);
    let last = points.last().unwrap();
    assert!(
        (last.1 - std::f64::consts::E).abs() < 0.01,
        "y(1) should be ≈ 2.718, got {}",
        last.1
    );
}

#[test]
fn rk4_linear() {
    // y' = 1, y(0) = 0 → y = x
    let interner = int();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let rhs = Expr::one();
    let points = rk4(&rhs, x, y, 0.0, 0.0, 5.0, 100, &interner);
    let last = points.last().unwrap();
    assert!(
        (last.1 - 5.0).abs() < 0.01,
        "y(5) should be 5, got {}",
        last.1
    );
}

#[test]
fn classify_pde_wave() {
    // Wave equation: A=1, B=0, C=-1. Discriminant B²-AC = 1 > 0 → hyperbolic
    let interner = int();
    let result = classify_pde(
        &Expr::one(),
        &Expr::zero(),
        &Expr::Int((-1).into()),
        &interner,
    );
    assert!(
        matches!(result, PdeType::Hyperbolic),
        "wave equation should be hyperbolic, got {:?}",
        result
    );
}

#[test]
fn classify_pde_heat() {
    // Heat equation: A=1, B=0, C=0. Discriminant = 0 → parabolic
    let interner = int();
    let result = classify_pde(&Expr::one(), &Expr::zero(), &Expr::zero(), &interner);
    assert!(
        matches!(result, PdeType::Parabolic),
        "heat equation should be parabolic, got {:?}",
        result
    );
}

#[test]
fn classify_pde_laplace() {
    // Laplace equation: A=1, B=0, C=1. Discriminant = -1 < 0 → elliptic
    let interner = int();
    let result = classify_pde(&Expr::one(), &Expr::zero(), &Expr::one(), &interner);
    assert!(
        matches!(result, PdeType::Elliptic),
        "Laplace equation should be elliptic, got {:?}",
        result
    );
}

#[test]
fn lindblad_euler_zero_rhs_leaves_state_unchanged() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let jump_ops: Vec<Vec<Vec<Expr>>> = Vec::new();
    let dt = one_tenth();
    let next = lindblad_euler_step(&h, &rho, &jump_ops, &dt, &interner).unwrap();
    assert_eq!(next, rho);
}

#[test]
fn lindblad_rk4_zero_rhs_leaves_state_unchanged() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let jump_ops: Vec<Vec<Vec<Expr>>> = Vec::new();
    let dt = one_tenth();
    let next = lindblad_rk4_step(&h, &rho, &jump_ops, &dt, &interner).unwrap();
    assert_eq!(next, rho);
}

#[test]
fn lindblad_step_rejects_zero_dt() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let jump_ops: Vec<Vec<Vec<Expr>>> = Vec::new();

    assert_eq!(
        lindblad_euler_step(&h, &rho, &jump_ops, &Expr::zero(), &interner),
        Err(QuantumOdeError::ZeroTimeStep)
    );
    assert_eq!(
        lindblad_rk4_step(&h, &rho, &jump_ops, &Expr::zero(), &interner),
        Err(QuantumOdeError::ZeroTimeStep)
    );
}
