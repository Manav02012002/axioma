use ax_ir::*;
use ax_qm::*;
use num_rational::BigRational;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn pauli_commutation() {
    // [σ_x, σ_y] = 2i σ_z
    let interner = int();
    let sx = pauli_x(&interner);
    let sy = pauli_y(&interner);
    let comm = commutator(&sx, &sy, &interner);
    let c00 = &comm[0][0];
    match c00 {
        Expr::Complex(re, im) => {
            assert_eq!(**re, Expr::zero(), "real part of [σx,σy][0,0] should be 0");
            assert_eq!(
                **im,
                Expr::Int(2.into()),
                "imag part of [σx,σy][0,0] should be 2"
            );
        }
        Expr::Mul(_) => {
            let c00_str = pretty_print(c00, &interner);
            assert!(
                c00_str.contains('2') && (c00_str.contains('i') || c00_str.contains("Complex")),
                "[σx,σy][0,0] should be 2i, got {}",
                c00_str
            );
        }
        _ => panic!("[σx,σy][0,0] should be 2i, got {:?}", c00),
    }
}

#[test]
fn pauli_anticommutation() {
    // {σ_x, σ_x} = 2I
    let interner = int();
    let sx = pauli_x(&interner);
    let anti = anticommutator(&sx, &sx, &interner);
    assert_eq!(
        anti[0][0],
        Expr::Int(2.into()),
        "{{σx,σx}}[0,0] should be 2"
    );
    assert_eq!(anti[0][1], Expr::zero(), "{{σx,σx}}[0,1] should be 0");
    assert_eq!(anti[1][0], Expr::zero(), "{{σx,σx}}[1,0] should be 0");
    assert_eq!(
        anti[1][1],
        Expr::Int(2.into()),
        "{{σx,σx}}[1,1] should be 2"
    );
}

#[test]
fn gamma_trace_identity() {
    // Tr(γ^μ γ^ν) = 4 g^{μν}
    let interner = int();
    let metric = ax_tensor::SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let trace_00 = gamma_trace(
        &[GammaEntry::Index(0), GammaEntry::Index(0)],
        &metric,
        &interner,
    );
    assert_eq!(
        trace_00,
        Expr::Int((-4).into()),
        "Tr(γ⁰γ⁰) should be -4, got {:?}",
        trace_00
    );
    let trace_11 = gamma_trace(
        &[GammaEntry::Index(1), GammaEntry::Index(1)],
        &metric,
        &interner,
    );
    assert_eq!(
        trace_11,
        Expr::Int(4.into()),
        "Tr(γ¹γ¹) should be 4, got {:?}",
        trace_11
    );
}

#[test]
fn gamma_trace_odd_is_zero() {
    // Tr(γ^μ) = 0
    let interner = int();
    let metric = ax_tensor::SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let trace = gamma_trace(&[GammaEntry::Index(0)], &metric, &interner);
    assert_eq!(trace, Expr::zero(), "Tr(γ^μ) should be 0, got {:?}", trace);
}

#[test]
fn gamma_trace_four_indices() {
    // Tr(γ^1 γ^1 γ^1 γ^1) = 4(1*1 - 1*1 + 1*1) = 4
    let interner = int();
    let metric = ax_tensor::SymbolicMatrix::from_diagonal(vec![
        Expr::Int((-1).into()),
        Expr::one(),
        Expr::one(),
        Expr::one(),
    ]);
    let trace = gamma_trace(
        &[
            GammaEntry::Index(1),
            GammaEntry::Index(1),
            GammaEntry::Index(1),
            GammaEntry::Index(1),
        ],
        &metric,
        &interner,
    );
    assert_eq!(
        trace,
        Expr::Int(4.into()),
        "Tr(γ¹γ¹γ¹γ¹) should be 4, got {:?}",
        trace
    );
}

#[test]
fn braket_orthogonal() {
    let up = vec![Expr::one(), Expr::zero()];
    let down = vec![Expr::zero(), Expr::one()];
    let result = braket(&up, &down);
    assert_eq!(result, Expr::zero(), "⟨↑|↓⟩ should be 0");
}

#[test]
fn braket_normalized() {
    let up = vec![Expr::one(), Expr::zero()];
    let result = braket(&up, &up);
    assert_eq!(result, Expr::one(), "⟨↑|↑⟩ should be 1");
}

#[test]
fn density_matrix_pure_state() {
    let state = vec![Expr::one(), Expr::zero()];
    let rho = density_matrix(&state);
    assert_eq!(rho[0][0], Expr::one(), "ρ[0,0] should be 1");
    assert_eq!(rho[0][1], Expr::zero(), "ρ[0,1] should be 0");
    assert_eq!(rho[1][0], Expr::zero(), "ρ[1,0] should be 0");
    assert_eq!(rho[1][1], Expr::zero(), "ρ[1,1] should be 0");
}

#[test]
fn partial_trace_bell_state() {
    // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    let interner = int();
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv.clone()]),
    ];
    let rho = density_matrix(&bell);
    let reduced = partial_trace(&rho, 2, 2, 'B', &interner);
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let r00 = ax_eval::eval(&reduced[0][0], &ax_eval::Env::new(), &interner);
    assert_eq!(r00, half, "Tr_B(bell)[0,0] should be 1/2, got {:?}", r00);
}

#[test]
fn normal_order_basic() {
    // a * a† → a† a + 1 (for bosonic ladder operators)
    let interner = int();
    let a = interner.get_or_intern("a");
    let a_dag = interner.get_or_intern("a_dag");
    let mut ops = std::collections::HashMap::new();
    ops.insert(a, OperatorKind::Annihilation);
    ops.insert(a_dag, OperatorKind::Creation);
    let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(a_dag)]);
    let result = normal_order(&expr, &ops, &interner);
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains("a_dag"),
        "normal ordered should have a_dag, got {}",
        result_str
    );
}
