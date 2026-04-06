use ax_graded::{graded_commutator, graded_simplify, GradedSymbolTable, Grading};
use ax_ir::{Expr, Interner};

#[test]
fn fermionic_nilpotency() {
    let interner = Interner::new();
    let theta = interner.get_or_intern("theta");
    let mut table = GradedSymbolTable::new();
    table.declare(theta, Grading::fermionic());
    let expr = Expr::mul(vec![Expr::Sym(theta), Expr::Sym(theta)]);
    let result = graded_simplify(&expr, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "theta^2 should be zero, got {:?}",
        result
    );
}

#[test]
fn fermionic_anticommutation() {
    let interner = Interner::new();
    let t1 = interner.get_or_intern("theta1");
    let t2 = interner.get_or_intern("theta2");
    let mut table = GradedSymbolTable::new();
    table.declare(t1, Grading::fermionic());
    table.declare(t2, Grading::fermionic());
    let a = Expr::mul(vec![Expr::Sym(t1), Expr::Sym(t2)]);
    let b = Expr::mul(vec![Expr::Sym(t2), Expr::Sym(t1)]);
    let sum = Expr::add(vec![a, b]);
    let result = graded_simplify(&sum, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "theta1*theta2 + theta2*theta1 should be zero, got {:?}",
        result
    );
}

#[test]
fn bosonic_commutation() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let mut table = GradedSymbolTable::new();
    table.declare(x, Grading::bosonic());
    table.declare(y, Grading::bosonic());
    let comm = graded_commutator(&Expr::Sym(x), &Expr::Sym(y), &table, &interner);
    let result = graded_simplify(&comm, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "[x,y] should be zero for bosonic symbols, got {:?}",
        result
    );
}

#[test]
fn mixed_grading_commutator() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    let t = interner.get_or_intern("theta");
    let mut table = GradedSymbolTable::new();
    table.declare(x, Grading::bosonic());
    table.declare(t, Grading::fermionic());
    let comm = graded_commutator(&Expr::Sym(x), &Expr::Sym(t), &table, &interner);
    let result = graded_simplify(&comm, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "boson-fermion commutator should be zero for free symbols"
    );
}

#[test]
fn fermionic_anticommutator() {
    let interner = Interner::new();
    let t1 = interner.get_or_intern("theta1");
    let t2 = interner.get_or_intern("theta2");
    let mut table = GradedSymbolTable::new();
    table.declare(t1, Grading::fermionic());
    table.declare(t2, Grading::fermionic());
    let comm = graded_commutator(&Expr::Sym(t1), &Expr::Sym(t2), &table, &interner);
    let result = graded_simplify(&comm, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "fermionic graded commutator (=anticommutator) should be zero for free symbols"
    );
}

#[test]
fn grading_inference_product() {
    let interner = Interner::new();
    let t1 = interner.get_or_intern("theta1");
    let t2 = interner.get_or_intern("theta2");
    let x = interner.get_or_intern("x");
    let mut table = GradedSymbolTable::new();
    table.declare(t1, Grading::fermionic());
    table.declare(t2, Grading::fermionic());
    table.declare(x, Grading::bosonic());
    let prod = Expr::mul(vec![Expr::Sym(t1), Expr::Sym(t2)]);
    let grading = table.infer_grading(&prod);
    assert!(
        grading.is_bosonic(),
        "product of two fermions should be bosonic, got {:?}",
        grading
    );
    let prod2 = Expr::mul(vec![Expr::Sym(x), Expr::Sym(t1)]);
    let grading2 = table.infer_grading(&prod2);
    assert!(
        grading2.is_fermionic(),
        "boson * fermion should be fermionic, got {:?}",
        grading2
    );
}

#[test]
fn triple_fermionic_product_zero() {
    let interner = Interner::new();
    let t1 = interner.get_or_intern("theta1");
    let t2 = interner.get_or_intern("theta2");
    let mut table = GradedSymbolTable::new();
    table.declare(t1, Grading::fermionic());
    table.declare(t2, Grading::fermionic());
    let expr = Expr::mul(vec![Expr::Sym(t1), Expr::Sym(t2), Expr::Sym(t1)]);
    let result = graded_simplify(&expr, &table, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "theta1*theta2*theta1 should be zero, got {:?}",
        result
    );
}
