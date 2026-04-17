#![cfg(feature = "legacy-eval-tests")]

use ax_ir::*;
use ax_tensor::*;

fn int() -> Interner {
    Interner::new()
}

fn numeric_eval(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(n) => num_traits::ToPrimitive::to_f64(n),
        Expr::Rational(r) => Some(
            num_traits::ToPrimitive::to_f64(r.numer())?
                / num_traits::ToPrimitive::to_f64(r.denom())?,
        ),
        Expr::Float(f) => Some(*f),
        Expr::Add(terms) => terms
            .iter()
            .map(numeric_eval)
            .try_fold(0.0, |acc, val| Some(acc + val?)),
        Expr::Mul(factors) => factors
            .iter()
            .map(numeric_eval)
            .try_fold(1.0, |acc, val| Some(acc * val?)),
        Expr::Pow(base, exp) => Some(numeric_eval(base)?.powf(numeric_eval(exp)?)),
        Expr::Neg(inner) => Some(-numeric_eval(inner)?),
        _ => None,
    }
}

#[test]
fn minkowski_christoffel_zero() {
    let interner = int();
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let t = interner.get_or_intern("t");
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let coords = vec![t, x, y, z];
    let gamma = christoffel_from_metric(&g, &coords, &interner);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                assert_eq!(
                    gamma[i][j][k],
                    Expr::zero(),
                    "Minkowski Γ^{}_{}{} should be 0",
                    i,
                    j,
                    k
                );
            }
        }
    }
}

#[test]
fn minkowski_riemann_zero() {
    let interner = int();
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let coords: Vec<_> = ["t", "x", "y", "z"]
        .iter()
        .map(|s| interner.get_or_intern(s))
        .collect();
    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let riem = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    assert_eq!(
                        riem[a][b][c][d],
                        Expr::zero(),
                        "Minkowski R^{}_{}{}_{} should be 0",
                        a,
                        b,
                        c,
                        d
                    );
                }
            }
        }
    }
}

#[test]
fn einstein_tensor_trace() {
    // For Schwarzschild (vacuum), G_{ab} should be zero.
    let interner = int();
    let sin_sym = interner.get_or_intern("sin");
    let r_sym = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let coords: Vec<_> = ["t", "r", "theta", "phi"]
        .iter()
        .map(|s| interner.get_or_intern(s))
        .collect();
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
    let riem = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
    let ricci = ricci_from_riemann(&riem, 4, &interner, &Convention::default());
    let ginv = g.symbolic_inverse(&interner);
    let scalar = ricci_scalar(&ricci, &ginv, &interner);
    let ein = einstein_tensor(&ricci, &scalar, &g, &interner);

    let mut env = ax_eval::Env::new();
    env.bindings
        .insert(interner.get_or_intern("t"), Expr::Float(0.0));
    env.bindings.insert(r_sym, Expr::Float(10.0));
    env.bindings.insert(theta, Expr::Float(1.0));
    env.bindings
        .insert(interner.get_or_intern("phi"), Expr::Float(0.0));
    for (i, row) in ein.iter().enumerate().take(4) {
        let val = ax_eval::eval(&row[i], &env, &interner);
        if let Some(f) = numeric_eval(&val) {
            assert!(f.abs() < 1e-8, "G[{}][{}] should be ≈0, got {}", i, i, f);
        }
    }
}

#[test]
fn ricci_scalar_flat_is_zero() {
    let interner = int();
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let ginv = g.symbolic_inverse(&interner);
    let coords: Vec<_> = ["t", "x", "y", "z"]
        .iter()
        .map(|s| interner.get_or_intern(s))
        .collect();
    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let riem = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
    let ricci = ricci_from_riemann(&riem, 4, &interner, &Convention::default());
    let scalar = ricci_scalar(&ricci, &ginv, &interner);
    assert_eq!(
        scalar,
        Expr::zero(),
        "Ricci scalar for flat space should be 0, got {:?}",
        scalar
    );
}

#[test]
fn geodesic_flat_space() {
    // Flat space geodesics: ẍ^i = 0 (all Christoffel symbols zero)
    let interner = int();
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let coords: Vec<_> = ["t", "x", "y", "z"]
        .iter()
        .map(|s| interner.get_or_intern(s))
        .collect();
    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let geo = geodesic_equations(&gamma, &coords, &interner);
    for (i, eq) in geo.iter().enumerate() {
        assert_eq!(
            *eq,
            Expr::zero(),
            "flat space geodesic equation {} should be 0, got {:?}",
            i,
            eq
        );
    }
}

#[test]
fn frw_christoffel_only_depends_on_time_derivative_of_scale_factor() {
    let interner = int();
    let a = interner.get_or_intern("a");
    let t = interner.get_or_intern("t");
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let scale = Expr::Call(a, vec![Expr::Sym(t)]);
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::pow(scale.clone(), Expr::Int(2.into())),
        Expr::pow(scale.clone(), Expr::Int(2.into())),
        Expr::pow(scale.clone(), Expr::Int(2.into())),
    ]);
    let coords = vec![t, x, y, z];
    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let rendered = ax_ir::pretty_print(
        &Expr::List(
            gamma
                .iter()
                .map(|plane| {
                    Expr::List(
                        plane
                            .iter()
                            .map(|row| Expr::List(row.iter().cloned().collect()))
                            .collect(),
                    )
                })
                .collect(),
        ),
        &interner,
    );
    assert!(
        rendered.contains("diff(a(t), t)"),
        "expected FRW Christoffel to depend on time derivative, got {rendered}"
    );
    assert!(
        !rendered.contains("diff(a(t), x)"),
        "FRW Christoffel should not depend on diff(a(t), x), got {rendered}"
    );
    assert!(
        !rendered.contains("diff(a(t), y)"),
        "FRW Christoffel should not depend on diff(a(t), y), got {rendered}"
    );
    assert!(
        !rendered.contains("diff(a(t), z)"),
        "FRW Christoffel should not depend on diff(a(t), z), got {rendered}"
    );
}

#[test]
fn polar_plane_2d_curvature_pipeline_is_dimension_generic() {
    let interner = int();
    let r = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let coords = vec![r, theta];
    let radial = Expr::Sym(r);
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::one(),
        Expr::pow(radial.clone(), Expr::Int(2.into())),
    ]);

    let gamma = christoffel_from_metric(&g, &coords, &interner);
    assert_eq!(gamma.len(), 2);
    assert_eq!(gamma[0].len(), 2);
    assert_eq!(gamma[1].len(), 2);
    assert_eq!(
        gamma[0][1][1].clone(),
        Expr::mul(vec![Expr::Int((-1).into()), radial.clone()])
    );
    assert_eq!(
        gamma[1][0][1].clone(),
        Expr::pow(radial.clone(), Expr::Int((-1).into()))
    );
    assert_eq!(
        gamma[1][1][0].clone(),
        Expr::pow(radial.clone(), Expr::Int((-1).into()))
    );

    let riem = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
    assert_eq!(riem.len(), 2);
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    assert_eq!(
                        riem[a][b][c][d].clone(),
                        Expr::zero(),
                        "2D polar-plane Riemann component [{a}][{b}][{c}][{d}] should vanish"
                    );
                }
            }
        }
    }

    let ricci = ricci_from_riemann(&riem, g.dim, &interner, &Convention::default());
    assert_eq!(ricci.len(), 2);
    for a in 0..2 {
        for b in 0..2 {
            assert_eq!(
                ricci[a][b].clone(),
                Expr::zero(),
                "2D polar-plane Ricci component [{a}][{b}] should vanish"
            );
        }
    }

    let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
    assert_eq!(
        scalar,
        Expr::zero(),
        "2D polar-plane scalar curvature should vanish"
    );
}

#[test]
fn five_dimensional_einstein_path_is_dimension_generic() {
    let interner = int();
    let t = interner.get_or_intern("t");
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let w = interner.get_or_intern("w");
    let coords = vec![t, x, y, z, w];
    let g = SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);

    let gamma = christoffel_from_metric(&g, &coords, &interner);
    let riem = riemann_from_christoffel(&gamma, &coords, &interner, &Convention::default());
    let ricci = ricci_from_riemann(&riem, g.dim, &interner, &Convention::default());
    let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(&interner), &interner);
    let einstein = einstein_tensor(&ricci, &scalar, &g, &interner);

    assert_eq!(gamma.len(), 5);
    assert_eq!(riem.len(), 5);
    assert_eq!(ricci.len(), 5);
    assert_eq!(einstein.len(), 5);
    for row in &ricci {
        assert_eq!(row.len(), 5);
        assert!(row.iter().all(|entry| *entry == Expr::zero()));
    }
    for row in &einstein {
        assert_eq!(row.len(), 5);
        assert!(row.iter().all(|entry| *entry == Expr::zero()));
    }
    assert_eq!(scalar, Expr::zero());
}
