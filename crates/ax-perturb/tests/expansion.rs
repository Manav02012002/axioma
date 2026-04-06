use ax_ir::{Expr, Interner};
use ax_perturb::*;

fn setup_2d(interner: &Interner) -> PerturbationSetup {
    PerturbationSetup {
        full_field: interner.get_or_intern("g"),
        background: interner.get_or_intern("g0"),
        perturbations: vec![
            PerturbationOrder {
                order: 1,
                field: interner.get_or_intern("h"),
            },
            PerturbationOrder {
                order: 2,
                field: interner.get_or_intern("k"),
            },
        ],
        epsilon: interner.get_or_intern("eps"),
        inverse_background: Some(interner.get_or_intern("g0inv")),
        max_order: 2,
    }
}

#[test]
fn zeroth_order_is_background() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let g = Expr::Sym(interner.get_or_intern("g"));
    let expanded = perturb_expand(&g, &setup, &interner);
    let order0 = &expanded.orders.iter().find(|o| o.order == 0).unwrap().expr;
    assert_eq!(*order0, Expr::Sym(interner.get_or_intern("g0")));
}

#[test]
fn first_order_is_epsilon_times_perturbation() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let g = Expr::Sym(interner.get_or_intern("g"));
    let expanded = perturb_expand(&g, &setup, &interner);
    let order1 = &expanded.orders.iter().find(|o| o.order == 1).unwrap().expr;
    assert_eq!(*order1, Expr::Sym(interner.get_or_intern("h")));
}

#[test]
fn product_expansion_order_tracking() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let g = Expr::Sym(interner.get_or_intern("g"));
    let gg = Expr::mul(vec![g.clone(), g.clone()]);
    let expanded = perturb_expand(&gg, &setup, &interner);
    let order0 = &expanded.orders.iter().find(|o| o.order == 0).unwrap().expr;
    let g0 = Expr::Sym(interner.get_or_intern("g0"));
    let expected0 = Expr::mul(vec![g0.clone(), g0.clone()]);
    assert_eq!(*order0, expected0, "order 0 of g*g should be g0*g0");
}

#[test]
fn power_expansion() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let g = Expr::Sym(interner.get_or_intern("g"));
    let g_sq = Expr::pow(g.clone(), Expr::Int(2.into()));
    let expanded = perturb_expand(&g_sq, &setup, &interner);
    assert!(
        expanded.orders.len() >= 2,
        "should have at least orders 0 and 1"
    );
}

#[test]
fn inverse_metric_order1_structure() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let expanded = perturb_inverse_metric(&setup, &interner);
    let order1 = &expanded.orders.iter().find(|o| o.order == 1).unwrap().expr;
    let order1_str = format!("{:?}", order1);
    assert!(
        order1_str.contains("g0inv") || order1_str.contains("Neg"),
        "order 1 inverse metric should be -g0inv*h*g0inv, got: {}",
        order1_str
    );
}

#[test]
fn inverse_metric_order2_sign() {
    let interner = Interner::new();
    let setup = setup_2d(&interner);
    let expanded = perturb_inverse_metric(&setup, &interner);
    let order2 = expanded.orders.iter().find(|o| o.order == 2);
    assert!(order2.is_some(), "should have order 2 term");
    let order2_expr = &order2.unwrap().expr;
    match order2_expr {
        Expr::Neg(_) => panic!("order 2 should be positive, got Neg"),
        _ => {}
    }
}

#[test]
fn truncation_respected() {
    let interner = Interner::new();
    let mut setup = setup_2d(&interner);
    setup.max_order = 1;
    let g = Expr::Sym(interner.get_or_intern("g"));
    let expanded = perturb_expand(&g, &setup, &interner);
    for order_term in &expanded.orders {
        assert!(
            order_term.order <= 1,
            "got order {} but max_order is 1",
            order_term.order
        );
    }
}
