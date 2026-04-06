use ax_ir::*;
use ax_variational::*;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn euler_lagrange_free_particle() {
    // L = 1/2 * m * v^2 where v = dx/dt
    let interner = int();
    let m = interner.get_or_intern("m");
    let v = interner.get_or_intern("v");
    let x = interner.get_or_intern("x");
    let t = interner.get_or_intern("t");
    let lagrangian = Expr::mul(vec![
        Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
        Expr::Sym(m),
        Expr::pow(Expr::Sym(v), Expr::Int(2.into())),
    ]);
    let result = functional_derivative(&lagrangian, x, &[v], &[t], &interner);
    assert_ne!(
        result,
        Expr::zero(),
        "E-L of free particle should give nonzero EOM expression"
    );
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains('m') || result_str.contains('v'),
        "E-L should contain m or v, got {}",
        result_str
    );
}

#[test]
fn euler_lagrange_harmonic_oscillator() {
    // L = 1/2 * m * v^2 - 1/2 * k * x^2
    let interner = int();
    let m = interner.get_or_intern("m");
    let k = interner.get_or_intern("k");
    let v = interner.get_or_intern("v");
    let x = interner.get_or_intern("x");
    let t = interner.get_or_intern("t");
    let lagrangian = Expr::add(vec![
        Expr::mul(vec![
            Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
            Expr::Sym(m),
            Expr::pow(Expr::Sym(v), Expr::Int(2.into())),
        ]),
        Expr::neg(Expr::mul(vec![
            Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
            Expr::Sym(k),
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
        ])),
    ]);
    let result = functional_derivative(&lagrangian, x, &[v], &[t], &interner);
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains('k'),
        "E-L of HO should contain k, got {}",
        result_str
    );
}

#[test]
fn vary_action_produces_variation() {
    let interner = int();
    let phi = interner.get_or_intern("phi");
    let dphi = interner.get_or_intern("dphi");
    let delta_phi = interner.get_or_intern("delta_phi");
    let delta_dphi = interner.get_or_intern("delta_dphi");
    let lagrangian = Expr::mul(vec![
        Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
        Expr::pow(Expr::Sym(dphi), Expr::Int(2.into())),
    ]);
    let result = vary_action(
        &lagrangian,
        phi,
        delta_phi,
        &[dphi],
        &[delta_dphi],
        &interner,
    );
    assert_ne!(result, Expr::zero(), "variation should be nonzero");
}
