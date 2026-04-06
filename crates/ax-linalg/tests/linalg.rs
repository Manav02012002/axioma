use ax_ir::*;
use ax_linalg::*;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn determinant_2x2() {
    let interner = int();
    let m = vec![
        vec![Expr::Int(1.into()), Expr::Int(2.into())],
        vec![Expr::Int(3.into()), Expr::Int(4.into())],
    ];
    let det = determinant(&m, &interner);
    assert_eq!(
        det,
        Expr::Int((-2).into()),
        "det should be -2, got {:?}",
        det
    );
}

#[test]
fn determinant_3x3() {
    let interner = int();
    let m = vec![
        vec![
            Expr::Int(1.into()),
            Expr::Int(2.into()),
            Expr::Int(3.into()),
        ],
        vec![
            Expr::Int(4.into()),
            Expr::Int(5.into()),
            Expr::Int(6.into()),
        ],
        vec![
            Expr::Int(7.into()),
            Expr::Int(8.into()),
            Expr::Int(9.into()),
        ],
    ];
    let det = determinant(&m, &interner);
    let simplified = ax_eval::eval(&det, &ax_eval::Env::new(), &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "det of singular matrix should be 0, got {:?}",
        simplified
    );
}

#[test]
fn inverse_2x2() {
    let interner = int();
    let m = vec![
        vec![Expr::Int(1.into()), Expr::Int(0.into())],
        vec![Expr::Int(0.into()), Expr::Int(2.into())],
    ];
    let inv = inverse(&m, &interner);
    assert!(inv.is_some(), "diagonal matrix should be invertible");
    let inv = inv.unwrap();
    assert_eq!(inv[0][0], Expr::one());
    assert_eq!(
        inv[1][1],
        Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
    );
}

#[test]
fn inverse_singular_returns_none() {
    let interner = int();
    let m = vec![
        vec![Expr::Int(1.into()), Expr::Int(2.into())],
        vec![Expr::Int(2.into()), Expr::Int(4.into())],
    ];
    let inv = inverse(&m, &interner);
    assert!(inv.is_none(), "singular matrix inverse should return None");
}

#[test]
fn trace_identity() {
    let m = vec![
        vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(1.into()), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::Int(1.into())],
    ];
    let tr = trace(&m);
    assert_eq!(tr, Expr::Int(3.into()), "trace of 3x3 identity should be 3");
}

#[test]
fn tensor_product_2x2() {
    let a = vec![
        vec![Expr::Int(1.into()), Expr::zero()],
        vec![Expr::zero(), Expr::Int(1.into())],
    ];
    let b = vec![
        vec![Expr::zero(), Expr::Int(1.into())],
        vec![Expr::Int(1.into()), Expr::zero()],
    ];
    let result = tensor_product(&a, &b);
    assert_eq!(result.len(), 4, "I⊗σx should be 4x4");
    assert_eq!(result[0].len(), 4);
    assert_eq!(result[0][0], Expr::zero());
    assert_eq!(result[0][1], Expr::one());
}
