use ax_forms::*;
use ax_ir::*;
use std::collections::BTreeMap;

fn int() -> Interner {
    Interner::new()
}

fn form(degree: usize, dim: usize, components: Vec<(Vec<usize>, Expr)>) -> DiffForm {
    DiffForm {
        degree,
        dim,
        components: components.into_iter().collect(),
    }
}

#[test]
fn wedge_antisymmetry() {
    // A ∧ B = -B ∧ A for 1-forms
    let interner = int();
    let a_sym = interner.get_or_intern("A");
    let b_sym = interner.get_or_intern("B");
    let a = form(1, 3, vec![(vec![0], Expr::Sym(a_sym))]);
    let b = form(1, 3, vec![(vec![1], Expr::Sym(b_sym))]);
    let ab = wedge(&a, &b, &interner);
    let ba = wedge(&b, &a, &interner);

    let ab_coeff = ab
        .components
        .get(&vec![0, 1])
        .cloned()
        .unwrap_or_else(Expr::zero);
    let ba_coeff = ba
        .components
        .get(&vec![0, 1])
        .cloned()
        .unwrap_or_else(Expr::zero);
    assert_eq!(
        Expr::add(vec![ab_coeff, ba_coeff]),
        Expr::zero(),
        "A∧B + B∧A should be 0"
    );
}

#[test]
fn exterior_derivative_squared_zero() {
    // d(dω) = 0 for any form ω
    let interner = int();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let coords = vec![x, y];
    let f = interner.get_or_intern("f");
    let omega = form(0, 2, vec![(vec![], Expr::Sym(f))]);
    let d_omega = exterior_derivative(&omega, &coords, &interner);
    let dd_omega = exterior_derivative(&d_omega, &coords, &interner);
    for (_, coeff) in &dd_omega.components {
        assert_eq!(
            coeff,
            &Expr::zero(),
            "d²ω component should be 0, got {:?}",
            coeff
        );
    }
}

#[test]
fn hodge_dual_of_dual() {
    // *(*ω) = ω for a 1-form in 3D Euclidean signature.
    let interner = int();
    let metric =
        ax_tensor::SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one(), Expr::one()]);
    let a = interner.get_or_intern("a");
    let omega = form(1, 3, vec![(vec![0], Expr::Sym(a))]);
    let star_omega = hodge_dual(&omega, &metric, &interner);
    let star_star_omega = hodge_dual(&star_omega, &metric, &interner);
    assert_eq!(
        form_to_expr(&star_star_omega),
        form_to_expr(&omega),
        "**ω should equal ω in 3D Euclidean"
    );
}

#[test]
fn wedge_degree() {
    // p-form ∧ q-form = (p+q)-form
    let interner = int();
    let a = DiffForm {
        degree: 1,
        dim: 4,
        components: BTreeMap::from([(vec![0], Expr::one())]),
    };
    let b = DiffForm {
        degree: 2,
        dim: 4,
        components: BTreeMap::from([(vec![1, 2], Expr::one())]),
    };
    let ab = wedge(&a, &b, &interner);
    assert_eq!(ab.degree, 3, "1-form ∧ 2-form should be 3-form");
}
