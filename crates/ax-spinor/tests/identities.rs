use ax_spinor::{
    apply_momentum_conservation, collect_mandelstam, expand_chain, expand_mandelstam,
    spinor_simplify, Label, SpinorExpr, SpinorFactor, SpinorTerm,
};
use num_rational::BigRational;
use num_traits::{One, Zero};

fn result_is_zero(expr: &SpinorExpr) -> bool {
    match expr {
        SpinorExpr::Numeric(n) => n.is_zero(),
        SpinorExpr::Sum(terms) => terms.is_empty() || terms.iter().all(result_is_zero),
        other => other.is_zero(),
    }
}

fn equivalent(a: &SpinorExpr, b: &SpinorExpr) -> bool {
    let diff = SpinorExpr::Sum(vec![a.clone(), SpinorExpr::Neg(Box::new(b.clone()))]);
    result_is_zero(&spinor_simplify(&diff, 4))
}

#[test]
fn angle_antisymmetry() {
    let a = SpinorExpr::angle(Label::new(1), Label::new(2));
    let b = SpinorExpr::angle(Label::new(2), Label::new(1));
    let sum = SpinorExpr::Sum(vec![a, b]);
    let result = spinor_simplify(&sum, 4);
    assert!(
        result_is_zero(&result),
        "<12> + <21> should be zero, got {:?}",
        result
    );
}

#[test]
fn square_antisymmetry() {
    let a = SpinorExpr::square(Label::new(1), Label::new(2));
    let b = SpinorExpr::square(Label::new(2), Label::new(1));
    let sum = SpinorExpr::Sum(vec![a, b]);
    let result = spinor_simplify(&sum, 4);
    assert!(
        result_is_zero(&result),
        "[12] + [21] should be zero, got {:?}",
        result
    );
}

#[test]
fn diagonal_zero() {
    let a = SpinorExpr::angle(Label::new(1), Label::new(1));
    assert!(a.is_zero());
    let b = SpinorExpr::square(Label::new(1), Label::new(1));
    assert!(b.is_zero());
}

#[test]
fn schouten_identity() {
    let l = |n: u16| Label::new(n);
    let t1 = SpinorExpr::Product(vec![SpinorTerm {
        coefficient: BigRational::one(),
        factors: vec![
            SpinorFactor::Angle(l(1), l(2)),
            SpinorFactor::Angle(l(3), l(4)),
        ],
    }]);
    let t2 = SpinorExpr::Product(vec![SpinorTerm {
        coefficient: BigRational::one(),
        factors: vec![
            SpinorFactor::Angle(l(1), l(3)),
            SpinorFactor::Angle(l(4), l(2)),
        ],
    }]);
    let t3 = SpinorExpr::Product(vec![SpinorTerm {
        coefficient: BigRational::one(),
        factors: vec![
            SpinorFactor::Angle(l(1), l(4)),
            SpinorFactor::Angle(l(2), l(3)),
        ],
    }]);
    let sum = SpinorExpr::Sum(vec![t1, t2, t3]);
    let result = spinor_simplify(&sum, 4);
    assert!(
        result_is_zero(&result),
        "Schouten identity failed, got {:?}",
        result
    );
}

#[test]
fn mandelstam_expansion_roundtrip() {
    let s12 = SpinorExpr::s(Label::new(1), Label::new(2));
    let expanded = expand_mandelstam(&s12);
    let collected = collect_mandelstam(&expanded);
    assert!(
        equivalent(&collected, &s12),
        "roundtrip failed: {:?}",
        collected
    );
}

#[test]
fn chain_single_momentum() {
    let chain = SpinorExpr::AngleSquareChain(Label::new(1), vec![Label::new(2)], Label::new(3));
    let expanded = expand_chain(&chain);
    let expected = SpinorExpr::Product(vec![SpinorTerm {
        coefficient: BigRational::one(),
        factors: vec![
            SpinorFactor::Angle(Label::new(1), Label::new(2)),
            SpinorFactor::Square(Label::new(2), Label::new(3)),
        ],
    }]);
    let diff = SpinorExpr::Sum(vec![expanded, SpinorExpr::Neg(Box::new(expected))]);
    let result = spinor_simplify(&diff, 4);
    assert!(
        result_is_zero(&result),
        "chain expansion wrong: {:?}",
        result
    );
}

#[test]
fn four_particle_mandelstam_relation() {
    let s = expand_mandelstam(&SpinorExpr::s(Label::new(1), Label::new(2)));
    let t = expand_mandelstam(&SpinorExpr::s(Label::new(1), Label::new(3)));
    let u = expand_mandelstam(&SpinorExpr::s(Label::new(1), Label::new(4)));
    let sum = SpinorExpr::Sum(vec![s, t, u]);
    let result = apply_momentum_conservation(&sum, 4, Label::new(4));
    let simplified = spinor_simplify(&result, 4);
    assert!(result_is_zero(&simplified), "s+t+u != 0: {:?}", simplified);
}
