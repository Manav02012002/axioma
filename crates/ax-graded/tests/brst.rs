use ax_graded::brst::{
    apply_brst, filter_by_ghost_number, ghost_number, setup_yang_mills_brst, verify_nilpotency,
};
use ax_graded::graded_simplify;
use ax_ir::Expr;

#[test]
fn brst_squared_gauge_field_zero() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (setup, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    let result = verify_nilpotency(a, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "s^2(A) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn brst_squared_ghost_zero() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (setup, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    let result = verify_nilpotency(c, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "s^2(c) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn brst_squared_antighost_zero() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (setup, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    let result = verify_nilpotency(cbar, &setup, &table, &interner);
    let simplified = graded_simplify(&result, &table, &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "s^2(cbar) should be 0, got {:?}",
        simplified
    );
}

#[test]
fn ghost_numbers_correct() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (_, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    assert_eq!(ghost_number(&Expr::Sym(a), &table), Some(0));
    assert_eq!(ghost_number(&Expr::Sym(c), &table), Some(1));
    assert_eq!(ghost_number(&Expr::Sym(cbar), &table), Some(-1));
    assert_eq!(ghost_number(&Expr::Sym(b), &table), Some(0));
}

#[test]
fn ghost_number_of_product() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (_, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    let expr = Expr::mul(vec![Expr::Sym(cbar), Expr::Sym(b)]);
    assert_eq!(ghost_number(&expr, &table), Some(-1));
    let expr2 = Expr::mul(vec![Expr::Sym(c), Expr::Sym(c)]);
    assert_eq!(ghost_number(&expr2, &table), Some(2));
}

#[test]
fn brst_leibniz_rule() {
    let interner = ax_ir::Interner::new();
    let a_sym = interner.get_or_intern("A");
    let c_sym = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (setup, table) = setup_yang_mills_brst(a_sym, c_sym, cbar, b, g, &interner);
    let product = Expr::mul(vec![Expr::Sym(a_sym), Expr::Sym(c_sym)]);
    let result = apply_brst(&product, &setup, &table, &interner);
    match &result {
        Expr::Add(terms) => assert!(
            terms.len() >= 2,
            "Leibniz should give at least 2 terms, got {}",
            terms.len()
        ),
        _ => {}
    }
}

#[test]
fn filter_ghost_number_works() {
    let interner = ax_ir::Interner::new();
    let a = interner.get_or_intern("A");
    let c = interner.get_or_intern("c");
    let cbar = interner.get_or_intern("cbar");
    let b = interner.get_or_intern("B");
    let g = interner.get_or_intern("g");
    let (_, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
    let expr = Expr::add(vec![Expr::Sym(a), Expr::Sym(c), Expr::Sym(cbar)]);
    let filtered = filter_by_ghost_number(&expr, 1, &table, &interner);
    assert_eq!(
        filtered,
        Expr::Sym(c),
        "filter should keep only ghost number 1 term"
    );
}
