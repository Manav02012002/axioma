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
        Expr::Complex(re, im) => 1 + node_count(re) + node_count(im),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::FnDef(_, params, body) => params.len() + 1 + node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + node_count(lhs) + node_count(rhs),
        Expr::Import(path) => 1 + path.len(),
        Expr::Assume(_, assumptions) => 1 + assumptions.len(),
        Expr::SetConvention(field, value) => 1 + field.len() + value.len(),
        Expr::Piecewise(cases) => 1 + cases.iter().map(|(value, _)| node_count(value)).sum::<usize>(),
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

fn numeric_eval(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(n) => num_traits::ToPrimitive::to_f64(n),
        Expr::Rational(r) => Some(
            num_traits::ToPrimitive::to_f64(r.numer())?
                / num_traits::ToPrimitive::to_f64(r.denom())?,
        ),
        Expr::Float(f) => Some(*f),
        Expr::Add(terms) => {
            let mut acc = 0.0;
            for term in terms {
                acc += numeric_eval(term)?;
            }
            Some(acc)
        }
        Expr::Mul(factors) => {
            let mut acc = 1.0;
            for factor in factors {
                acc *= numeric_eval(factor)?;
            }
            Some(acc)
        }
        Expr::Pow(base, exp) => Some(numeric_eval(base)?.powf(numeric_eval(exp)?)),
        Expr::Neg(inner) => Some(-numeric_eval(inner)?),
        _ => None,
    }
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
    let two_over_r = Expr::mul(vec![
        two.clone(),
        Expr::pow(r.clone(), Expr::Int((-1).into())),
    ]);
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
    let riemann = riemann_from_christoffel(
        &gamma,
        &coords,
        &interner,
        &ax_ir::Convention::default(),
    );
    let ricci = ricci_from_riemann(&riemann, 4, &interner, &ax_ir::Convention::default());

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

#[test]
fn schwarzschild_ricci_zero_both_conventions() {
    let interner = Interner::new();
    let t = interner.get_or_intern("t");
    let r_sym = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let phi = interner.get_or_intern("phi");
    for riemann_sign in [ax_ir::RiemannSign::MTW, ax_ir::RiemannSign::Weinberg] {
        let (g, coords) = build_schwarzschild(&interner);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let mut convention = ax_ir::Convention::default();
        convention.riemann_sign = riemann_sign;

        let riemann = riemann_from_christoffel(&gamma, &coords, &interner, &convention);
        let ricci = ricci_from_riemann(&riemann, 4, &interner, &convention);
        let mut env = ax_eval::Env::new();
        env.bindings.insert(t, Expr::Float(0.0));
        env.bindings.insert(r_sym, Expr::Float(10.0));
        env.bindings.insert(theta, Expr::Float(1.0));
        env.bindings.insert(phi, Expr::Float(0.0));

        for row in ricci.iter().take(4) {
            for component in row.iter().take(4) {
                let numeric = ax_eval::eval(component, &env, &interner);
                let value = numeric_eval(&numeric)
                    .unwrap_or_else(|| panic!("expected numeric Ricci component, got {:?}", numeric));
                assert!(value.abs() < 1e-9, "component = {value}");
            }
        }
    }
}

#[test]
fn schwarzschild_christoffel_nonzero_components() {
    let interner = ax_ir::Interner::new();
    let t = interner.get_or_intern("t");
    let r_sym = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let phi = interner.get_or_intern("phi");
    let coords = vec![t, r_sym, theta, phi];
    let sin_sym = interner.get_or_intern("sin");

    let r = Expr::Sym(r_sym);
    let two_over_r = Expr::mul(vec![
        Expr::Int(2.into()),
        Expr::pow(r.clone(), Expr::Int((-1).into())),
    ]);
    let f = Expr::add(vec![Expr::one(), Expr::neg(two_over_r)]);

    let mut g = SymbolicMatrix::new(4);
    g.set(0, 0, Expr::neg(f.clone()));
    g.set(1, 1, Expr::pow(f.clone(), Expr::Int((-1).into())));
    g.set(2, 2, Expr::pow(r.clone(), Expr::Int(2.into())));
    let sin_theta = Expr::Call(sin_sym, vec![Expr::Sym(theta)]);
    g.set(
        3,
        3,
        Expr::mul(vec![
            Expr::pow(r.clone(), Expr::Int(2.into())),
            Expr::pow(sin_theta, Expr::Int(2.into())),
        ]),
    );

    let gamma = christoffel_from_metric(&g, &coords, &interner);

    let mut nonzero_count = 0;
    for plane in gamma.iter().take(4) {
        for row in plane.iter().take(4) {
            for component in row.iter().take(4) {
                if *component != Expr::zero() {
                    nonzero_count += 1;
                }
            }
        }
    }

    assert!(nonzero_count > 0, "expected non-zero Christoffel symbols");
    assert!(
        nonzero_count <= 13,
        "too many non-zero components: {}",
        nonzero_count
    );
}
