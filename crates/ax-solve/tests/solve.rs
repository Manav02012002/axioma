use ax_ir::*;
use ax_solve::*;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn solve_linear() {
    // 2x - 6 = 0 → x = 3
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = Expr::add(vec![
        Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(x)]),
        Expr::Int((-6).into()),
    ]);
    let result = solve(&eq, x, &interner);
    match &result {
        Expr::List(solutions) => {
            assert!(!solutions.is_empty(), "should have at least one solution");
            assert!(
                solutions.contains(&Expr::Int(3.into())),
                "solution should contain 3, got {:?}",
                result
            );
        }
        Expr::Int(n) if *n == 3.into() => {}
        _ => {
            let result_str = pretty_print(&result, &interner);
            assert!(
                result_str.contains('3'),
                "solution of 2x-6=0 should be 3, got {}",
                result_str
            );
        }
    }
}

#[test]
fn solve_quadratic() {
    // x^2 - 5x + 6 = 0 → x = 2 or x = 3
    let interner = int();
    let x = interner.get_or_intern("x");
    let eq = Expr::add(vec![
        Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
        Expr::mul(vec![Expr::Int((-5).into()), Expr::Sym(x)]),
        Expr::Int(6.into()),
    ]);
    let result = solve(&eq, x, &interner);
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains('2') && result_str.contains('3'),
        "solutions of x²-5x+6 should include 2 and 3, got {}",
        result_str
    );
}

#[test]
fn solve_linear_system_2x2() {
    // x + y = 5, x - y = 1 → x = 3, y = 2
    let interner = int();
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let eq1 = Expr::add(vec![Expr::Sym(x), Expr::Sym(y), Expr::Int((-5).into())]);
    let eq2 = Expr::add(vec![
        Expr::Sym(x),
        Expr::neg(Expr::Sym(y)),
        Expr::Int((-1).into()),
    ]);
    let result = solve_linear_system(&[eq1, eq2], &[x, y], &interner);
    assert!(result.is_some(), "2x2 system should have a solution");
    let solution = result.unwrap();
    let x_val = solution.iter().find(|(s, _)| *s == x).map(|(_, v)| v);
    let y_val = solution.iter().find(|(s, _)| *s == y).map(|(_, v)| v);
    assert_eq!(x_val, Some(&Expr::Int(3.into())), "x should be 3");
    assert_eq!(y_val, Some(&Expr::Int(2.into())), "y should be 2");
}
