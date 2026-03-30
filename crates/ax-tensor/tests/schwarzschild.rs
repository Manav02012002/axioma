use ax_ir::{Expr, Interner};
use ax_tensor::{
    christoffel_from_metric, ricci_from_riemann, riemann_from_christoffel, SymbolicMatrix,
};

fn simplify(expr: &Expr, interner: &Interner) -> Expr {
    ax_eval::eval(expr, &ax_eval::Env::new(), interner)
}

fn node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => 1,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::Indexed(base, _) => 1 + node_count(base),
        Expr::Let(_, val, body) => 1 + node_count(val) + node_count(body),
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
    }
}

fn aggressive_simplify(expr: &Expr, interner: &Interner) -> Expr {
    let mut current = expr.clone();
    for _ in 0..5 {
        let mut current_step = current.clone();
        if node_count(&current_step) <= 64 {
            current_step = ax_eval::simplify::expand(&current_step, interner);
        }
        let collected = ax_eval::simplify::collect_terms(&current_step, interner);
        let evaled = ax_eval::eval(&collected, &ax_eval::Env::new(), interner);
        if evaled == current {
            break;
        }
        current = evaled;
    }
    current
}

fn build_schwarzschild(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
    let t = interner.get_or_intern("t");
    let r_sym = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let phi = interner.get_or_intern("phi");
    let coords = vec![t, r_sym, theta, phi];

    let sin_sym = interner.get_or_intern("sin");

    let two = Expr::Int(2.into());
    let r = Expr::Sym(r_sym);
    let two_over_r = Expr::mul(vec![two.clone(), Expr::pow(r.clone(), Expr::Int((-1).into()))]);
    let f = Expr::add(vec![Expr::one(), Expr::neg(two_over_r.clone())]);
    let neg_f = Expr::neg(f.clone());
    let inv_f = Expr::pow(f.clone(), Expr::Int((-1).into()));
    let r_sq = Expr::pow(r.clone(), Expr::Int(2.into()));
    let sin_theta = Expr::Call(sin_sym, vec![Expr::Sym(theta)]);
    let sin_sq_theta = Expr::pow(sin_theta, Expr::Int(2.into()));
    let r_sq_sin_sq = Expr::mul(vec![r_sq.clone(), sin_sq_theta]);

    let mut g = SymbolicMatrix::new(4);
    g.set(0, 0, neg_f);
    g.set(1, 1, inv_f);
    g.set(2, 2, r_sq);
    g.set(3, 3, r_sq_sin_sq);

    (g, coords)
}

#[test]
fn schwarzschild_ricci_is_zero() {
    let interner = Interner::new();
    let (g, coords) = build_schwarzschild(&interner);

    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let riemann = riemann_from_christoffel(&gamma, &coords, &interner);
    let ricci = ricci_from_riemann(&riemann, 4);

    let mut nonzero = vec![];
    for j in 0..4 {
        for l in 0..4 {
            let component = aggressive_simplify(&simplify(&ricci[j][l], &interner), &interner);
            if component != Expr::zero() {
                nonzero.push(format!(
                    "Ricci[{}][{}] = {}",
                    j,
                    l,
                    ax_ir::pretty_print(&component, &interner)
                ));
            }
        }
    }

    assert!(
        nonzero.is_empty(),
        "Schwarzschild Ricci tensor has non-zero components:\n{}",
        nonzero.join("\n")
    );
}
