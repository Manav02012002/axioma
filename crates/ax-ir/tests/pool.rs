use ax_ir::*;

fn roundtrip(expr: &Expr) {
    let mut pool = ExprPool::new();
    let id = pool.from_expr(expr);
    let back = pool.to_expr(id);
    assert_eq!(*expr, back, "roundtrip failed for {:?}", expr);
}

fn roundtrip_float_bits(value: f64) {
    let expr = Expr::Float(value);
    let mut pool = ExprPool::new();
    let id = pool.from_expr(&expr);
    let back = pool.to_expr(id);
    match back {
        Expr::Float(back) => assert_eq!(value.to_bits(), back.to_bits()),
        other => panic!("float roundtrip returned {:?}", other),
    }
}

#[test]
fn rt_int() {
    roundtrip(&Expr::Int(42.into()));
}

#[test]
fn rt_neg_int() {
    roundtrip(&Expr::Int((-7).into()));
}

#[test]
fn rt_rational() {
    roundtrip(&Expr::Rational(num_rational::BigRational::new(
        1.into(),
        3.into(),
    )));
}

#[test]
fn rt_float() {
    roundtrip(&Expr::Float(3.14));
}

#[test]
fn rt_float_nan() {
    roundtrip_float_bits(f64::NAN);
}

#[test]
fn rt_float_neg_zero() {
    roundtrip_float_bits(-0.0);
}

#[test]
fn rt_sym() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Sym(x));
}

#[test]
fn rt_add() {
    roundtrip(&Expr::Add(vec![
        Expr::Int(1.into()),
        Expr::Int(2.into()),
        Expr::Int(3.into()),
    ]));
}

#[test]
fn rt_mul() {
    roundtrip(&Expr::Mul(vec![Expr::Int(2.into()), Expr::Int(3.into())]));
}

#[test]
fn rt_pow() {
    roundtrip(&Expr::Pow(
        Box::new(Expr::Int(2.into())),
        Box::new(Expr::Int(10.into())),
    ));
}

#[test]
fn rt_neg() {
    roundtrip(&Expr::Neg(Box::new(Expr::Int(5.into()))));
}

#[test]
fn rt_complex() {
    roundtrip(&Expr::Complex(
        Box::new(Expr::Int(1.into())),
        Box::new(Expr::Int(2.into())),
    ));
}

#[test]
fn rt_call() {
    let interner = Interner::new();
    let sin = interner.get_or_intern("sin");
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Call(sin, vec![Expr::Sym(x)]));
}

#[test]
fn rt_indexed() {
    let interner = Interner::new();
    let t = interner.get_or_intern("T");
    let mu = interner.get_or_intern("mu");
    roundtrip(&Expr::Indexed(
        Box::new(Expr::Sym(t)),
        vec![Index {
            name: mu,
            variance: Variance::Down,
            index_type: None,
        }],
    ));
}

#[test]
fn rt_list() {
    roundtrip(&Expr::List(vec![Expr::Int(1.into()), Expr::Int(2.into())]));
}

#[test]
fn rt_matrix() {
    roundtrip(&Expr::Matrix(vec![
        vec![Expr::Int(1.into()), Expr::Int(0.into())],
        vec![Expr::Int(0.into()), Expr::Int(1.into())],
    ]));
}

#[test]
fn rt_let() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Let(
        x,
        Box::new(Expr::Int(5.into())),
        Box::new(Expr::Sym(x)),
    ));
}

#[test]
fn rt_fndef() {
    let interner = Interner::new();
    let f = interner.get_or_intern("f");
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::FnDef(f, vec![x], Box::new(Expr::Sym(x))));
}

#[test]
fn rt_piecewise() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Piecewise(vec![
        (Expr::Sym(x), Condition::Gt(Expr::Sym(x), Expr::zero())),
        (Expr::neg(Expr::Sym(x)), Condition::True),
    ]));
}

#[test]
fn rt_rule() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Rule(
        Box::new(Expr::Sym(x)),
        Box::new(Expr::Int(1.into())),
        TrustLevel::Exact,
    ));
}

#[test]
fn rt_import() {
    let interner = Interner::new();
    let path = interner.get_or_intern("std.gr");
    roundtrip(&Expr::Import(vec![path]));
}

#[test]
fn rt_assume() {
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    roundtrip(&Expr::Assume(
        x,
        vec![Assumption::Positive, Assumption::Real],
    ));
}

#[test]
fn rt_set_convention() {
    roundtrip(&Expr::SetConvention(
        "metric_signature".into(),
        "mostly_plus".into(),
    ));
}

#[test]
fn deduplication() {
    let mut pool = ExprPool::new();
    let a = pool.from_expr(&Expr::Int(42.into()));
    let b = pool.from_expr(&Expr::Int(42.into()));
    assert_eq!(a, b, "identical expressions should get same ExprId");
}

#[test]
fn structural_eq_via_id() {
    let mut pool = ExprPool::new();
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    let sin = interner.get_or_intern("sin");
    let e1 = Expr::Call(sin, vec![Expr::Sym(x)]);
    let e2 = Expr::Call(sin, vec![Expr::Sym(x)]);
    let id1 = pool.from_expr(&e1);
    let id2 = pool.from_expr(&e2);
    assert!(pool.structural_eq(id1, id2));
}

#[test]
fn nested_dedup() {
    let mut pool = ExprPool::new();
    let interner = Interner::new();
    let x = interner.get_or_intern("x");
    let sin = interner.get_or_intern("sin");
    let sin_x = Expr::Call(sin, vec![Expr::Sym(x)]);
    let sum = Expr::Add(vec![sin_x.clone(), sin_x]);
    let _id = pool.from_expr(&sum);
    let call_count = (0..pool.len())
        .filter(|i| matches!(pool.get(ExprId(*i as u32)), PooledExpr::Call(_, _)))
        .count();
    assert_eq!(
        call_count, 1,
        "sin(x) should be stored once, found {} Call nodes",
        call_count
    );
}
