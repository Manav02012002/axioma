use ax_ir::*;
use ax_qm::*;
use num_rational::BigRational;
use std::collections::HashMap;

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
fn braket_complex_norm() {
    let interner = int();
    let state = vec![
        Expr::one(),
        Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one())),
    ];
    let result = ax_eval::eval(&braket(&state, &state), &ax_eval::Env::new(), &interner);
    assert_eq!(result, Expr::Int(2.into()));
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
fn density_matrix_complex_phase() {
    let interner = int();
    let state = vec![
        Expr::one(),
        Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one())),
    ];
    let rho = density_matrix(&state);
    assert_eq!(
        ax_eval::eval(&rho[0][0], &ax_eval::Env::new(), &interner),
        Expr::one()
    );
    assert_eq!(
        ax_eval::eval(&rho[0][1], &ax_eval::Env::new(), &interner),
        Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::Int((-1).into())))
    );
    assert_eq!(
        ax_eval::eval(&rho[1][0], &ax_eval::Env::new(), &interner),
        Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
    );
    assert_eq!(
        ax_eval::eval(&rho[1][1], &ax_eval::Env::new(), &interner),
        Expr::one()
    );
}

#[test]
fn try_braket_dimension_mismatch_errors() {
    let err = try_braket(&[Expr::one()], &[Expr::one(), Expr::zero()]);
    assert_eq!(
        err,
        Err(QmLinearAlgebraError::DimensionMismatch { left: 1, right: 2 })
    );
}

#[test]
fn try_ket_out_of_range_errors() {
    let err = try_ket(2, 2);
    assert_eq!(
        err,
        Err(QmLinearAlgebraError::BasisIndexOutOfRange { index: 2, dim: 2 })
    );
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
fn partial_trace_checked_bell_state_a() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let reduced = try_partial_trace(
        &rho,
        BipartiteDims { dim_a: 2, dim_b: 2 },
        PartialTraceTarget::A,
    )
    .unwrap();
    let expected = vec![
        vec![half.clone(), Expr::zero()],
        vec![Expr::zero(), half.clone()],
    ];
    let evaluated = reduced
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| ax_eval::eval(cell, &ax_eval::Env::new(), &interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluated, expected);
}

#[test]
fn partial_trace_checked_bell_state_b() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let reduced = try_partial_trace(
        &rho,
        BipartiteDims { dim_a: 2, dim_b: 2 },
        PartialTraceTarget::B,
    )
    .unwrap();
    let expected = vec![
        vec![half.clone(), Expr::zero()],
        vec![Expr::zero(), half.clone()],
    ];
    let evaluated = reduced
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| ax_eval::eval(cell, &ax_eval::Env::new(), &interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluated, expected);
}

#[test]
fn partial_trace_factor_bell_state_first_qubit() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let reduced = try_partial_trace_factor(&rho, &[2, 2], 0).unwrap();
    let evaluated = reduced
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| ax_eval::eval(cell, &ax_eval::Env::new(), &interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        evaluated,
        vec![
            vec![half.clone(), Expr::zero()],
            vec![Expr::zero(), half.clone()],
        ]
    );
}

#[test]
fn bell_state_entanglement_spectrum_is_half_half() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let bell = vec![
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
        Expr::zero(),
        Expr::zero(),
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
    ];

    let spectrum = entanglement_spectrum_from_state(&bell, 2, 2, &interner).unwrap();
    assert_eq!(spectrum, vec![half.clone(), half]);
}

#[test]
fn bell_state_schmidt_coefficients_are_inv_sqrt2() {
    let interner = int();
    let inv_sqrt2 = Expr::Call(
        interner.get_or_intern("sqrt"),
        vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
    );
    let bell = vec![
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
        Expr::zero(),
        Expr::zero(),
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
    ];

    let coefficients = schmidt_coefficients_from_state(&bell, 2, 2, &interner).unwrap();
    assert_eq!(coefficients, vec![inv_sqrt2.clone(), inv_sqrt2]);
}

#[test]
fn product_state_entanglement_spectrum_is_one_zero() {
    let interner = int();
    let product = vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()];

    let spectrum = entanglement_spectrum_from_state(&product, 2, 2, &interner).unwrap();
    assert_eq!(spectrum, vec![Expr::one(), Expr::zero()]);
}

#[test]
fn schmidt_coefficients_reject_bad_vector_length() {
    let interner = int();
    let err = schmidt_coefficients_from_state(&[Expr::one(), Expr::zero(), Expr::zero()], 2, 2, &interner);
    assert_eq!(
        err,
        Err(EntanglementError::StateDimensionMismatch {
            expected: 4,
            actual: 3,
        })
    );
}

#[test]
fn negativity_bell_state_is_one_half() {
    let interner = int();
    let bell = vec![
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
        Expr::zero(),
        Expr::zero(),
        Expr::pow(
            Expr::Int(2.into()),
            Expr::Rational(BigRational::new((-1).into(), 2.into())),
        ),
    ];
    let rho = density_matrix(&bell);

    let negativity = negativity_bipartite(&rho, 2, 2, 1, &interner).unwrap();
    assert_eq!(negativity, Expr::Rational(BigRational::new(1.into(), 2.into())));
}

#[test]
fn negativity_product_state_is_zero() {
    let interner = int();
    let product = vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()];
    let rho = density_matrix(&product);

    let negativity = negativity_bipartite(&rho, 2, 2, 1, &interner).unwrap();
    assert_eq!(negativity, Expr::zero());
}

#[test]
fn negativity_rejects_dimension_mismatch() {
    let interner = int();
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::one()],
    ];

    let err = negativity_bipartite(&rho, 2, 2, 1, &interner);
    assert_eq!(
        err,
        Err(NegativityError::DimensionMismatch {
            expected: 4,
            actual: 3,
        })
    );
}

#[test]
fn partial_trace_factor_bell_state_second_qubit() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let reduced = try_partial_trace_factor(&rho, &[2, 2], 1).unwrap();
    let evaluated = reduced
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| ax_eval::eval(cell, &ax_eval::Env::new(), &interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        evaluated,
        vec![
            vec![half.clone(), Expr::zero()],
            vec![Expr::zero(), half.clone()],
        ]
    );
}

#[test]
fn partial_trace_factor_one_factor_returns_total_trace() {
    let interner = int();
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let reduced = try_partial_trace_factor(&rho, &[2], 0).unwrap();
    let evaluated = reduced
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| ax_eval::eval(cell, &ax_eval::Env::new(), &interner))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluated, vec![vec![Expr::one()]]);
}

#[test]
fn partial_trace_factor_rejects_bad_index() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    let err = try_partial_trace_factor(&rho, &[2, 2], 2);
    assert_eq!(
        err,
        Err(CompositeSpaceError::InvalidFactorIndex {
            index: 2,
            factor_count: 2,
        })
    );
}

#[test]
fn partial_transpose_moves_offdiagonal_block() {
    let rho = vec![
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    let transposed = try_partial_transpose_factor(&rho, &[2, 2], 1).unwrap();
    let expected = vec![
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    assert_eq!(transposed, expected);
}

#[test]
fn partial_transpose_of_diagonal_product_state_is_unchanged() {
    let rho = vec![
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    assert_eq!(try_partial_transpose_factor(&rho, &[2, 2], 0).unwrap(), rho);
    assert_eq!(try_partial_transpose_factor(&rho, &[2, 2], 1).unwrap(), rho);
}

#[test]
fn permute_subsystems_swaps_two_qubits() {
    let rho = vec![
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    let permuted = try_permute_subsystems(&rho, &[2, 2], &[1, 0]).unwrap();
    let expected = vec![
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    assert_eq!(permuted, expected);
}

#[test]
fn permute_subsystems_rejects_bad_permutation() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    assert_eq!(
        try_permute_subsystems(&rho, &[2, 2], &[0]),
        Err(CompositeSpaceError::InvalidPermutationLength {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        try_permute_subsystems(&rho, &[2, 2], &[0, 2]),
        Err(CompositeSpaceError::InvalidPermutationEntry {
            value: 2,
            factor_count: 2,
        })
    );
    assert_eq!(
        try_permute_subsystems(&rho, &[2, 2], &[0, 0]),
        Err(CompositeSpaceError::DuplicatePermutationEntry { value: 0 })
    );
}

#[test]
fn partial_trace_rejects_non_square() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero()],
    ];
    let err = try_partial_trace(
        &rho,
        BipartiteDims { dim_a: 1, dim_b: 2 },
        PartialTraceTarget::A,
    );
    assert_eq!(
        err,
        Err(QmLinearAlgebraError::NonSquareMatrix { rows: 2, cols: 3 })
    );
}

#[test]
fn identity_channel_preserves_density_matrix() {
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let result = apply_kraus_channel(&identity_channel(2), &rho).unwrap();
    assert_eq!(result, rho);
}

#[test]
fn kraus_completeness_identity_channel_is_identity() {
    let completeness = kraus_completeness_matrix(&identity_channel(2)).unwrap();
    assert_eq!(
        completeness,
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ]
    );
}

#[test]
fn apply_kraus_channel_rejects_dimension_mismatch() {
    let kraus = identity_channel(2);
    let rho = vec![vec![Expr::one(), Expr::zero(), Expr::zero()]; 3];
    let err = apply_kraus_channel(&kraus, &rho);
    assert_eq!(
        err,
        Err(ChannelError::StateDimensionMismatch {
            expected: 2,
            actual: 3
        })
    );
}

#[test]
fn basis_projector_on_zero_state_gives_probability_one() {
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let projector = basis_projector(0, 2).unwrap();
    let probabilities = measurement_probabilities(&[projector], &rho).unwrap();
    assert_eq!(probabilities, vec![Expr::one()]);
}

#[test]
fn basis_projector_on_orthogonal_state_gives_probability_zero() {
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let projector = basis_projector(1, 2).unwrap();
    let probabilities = measurement_probabilities(&[projector], &rho).unwrap();
    assert_eq!(probabilities, vec![Expr::zero()]);
}

#[test]
fn expectation_value_pauli_z_on_zero_state_is_one() {
    let z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(expectation_value(&z, &rho0).unwrap(), Expr::one());
}

#[test]
fn variance_pauli_z_on_zero_state_is_zero() {
    let z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(variance(&z, &rho0).unwrap(), Expr::zero());
}

#[test]
fn expectation_value_pauli_z_on_maximally_mixed_state_is_zero() {
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    assert_eq!(expectation_value(&z, &rho_mixed).unwrap(), Expr::zero());
}

#[test]
fn variance_rejects_dimension_mismatch() {
    let z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let rho = vec![vec![Expr::one(), Expr::zero(), Expr::zero()]; 3];
    assert_eq!(
        variance(&z, &rho),
        Err(ObservableError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn purity_pure_state_is_one() {
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(purity(&rho0).unwrap(), Expr::one());
}

#[test]
fn purity_maximally_mixed_qubit_is_one_half() {
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    assert_eq!(
        purity(&rho_mixed).unwrap(),
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    );
}

#[test]
fn linear_entropy_pure_state_is_zero() {
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(linear_entropy(&rho0).unwrap(), Expr::zero());
}

#[test]
fn linear_entropy_maximally_mixed_qubit_is_one_half() {
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    assert_eq!(
        linear_entropy(&rho_mixed).unwrap(),
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    );
}

#[test]
fn purity_rejects_nonsquare_matrix() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(
        purity(&rho),
        Err(StateFunctionalError::StateNotSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn participation_ratio_pure_state_is_one() {
    let interner = int();
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(participation_ratio(&rho0, &interner).unwrap(), Expr::one());
}

#[test]
fn participation_ratio_maximally_mixed_qubit_is_two() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    assert_eq!(
        participation_ratio(&rho_mixed, &interner).unwrap(),
        Expr::Int(2.into())
    );
}

#[test]
fn participation_ratio_rejects_nonsquare_matrix() {
    let interner = int();
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(
        participation_ratio(&rho, &interner),
        Err(StateFunctionalError::StateNotSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn renyi2_entropy_pure_state_is_neg_log_one() {
    let interner = int();
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let result = renyi2_entropy(&rho0, &interner).unwrap();
    assert!(
        result == Expr::zero()
            || result == Expr::neg(Expr::Call(interner.get_or_intern("log"), vec![Expr::one()]))
    );
}

#[test]
fn renyi2_entropy_maximally_mixed_qubit_is_log_two() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    let result = renyi2_entropy(&rho_mixed, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(rendered == "log(2)" || rendered == "-log(1/2)" || rendered == "-1*log(1/2)");
}

#[test]
fn renyi2_entropy_factor_bell_pair_kept_qubit_is_log_two() {
    let interner = int();
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let result = renyi2_entropy_factor(&rho, &[2, 2], 0, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(rendered == "log(2)" || rendered == "-log(1/2)" || rendered == "-1*log(1/2)");
}

#[test]
fn renyi2_tripartite_information_product_state_is_zero() {
    let interner = int();
    let product = vec![
        Expr::one(),
        Expr::zero(),
        Expr::zero(),
        Expr::zero(),
        Expr::zero(),
        Expr::zero(),
        Expr::zero(),
        Expr::zero(),
    ];
    let rho = density_matrix(&product);
    assert_eq!(
        renyi2_tripartite_information(&rho, [2, 2, 2], &interner).unwrap(),
        Expr::zero()
    );
}

#[test]
fn renyi2_entropy_factor_rejects_bad_factor_index() {
    let interner = int();
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    let err = renyi2_entropy_factor(&rho, &[2, 2], 2, &interner);
    assert_eq!(
        err,
        Err(CompositeSpaceError::InvalidFactorIndex {
            index: 2,
            factor_count: 2
        })
    );
}

#[test]
fn von_neumann_entropy_pure_qubit_is_zero() {
    let interner = int();
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(von_neumann_entropy(&rho0, &interner).unwrap(), Expr::zero());
}

#[test]
fn von_neumann_entropy_maximally_mixed_qubit_is_log_two() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];
    let result = von_neumann_entropy(&rho_mixed, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(
        rendered == "log(2)"
            || rendered == "-log(1/2)"
            || rendered == "-1*log(1/2)"
            || rendered == "-2*1/2*log(1/2)"
    );
}

#[test]
fn von_neumann_entropy_diagonal_qutrit_probabilities() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let quarter = Expr::Rational(BigRational::new(1.into(), 4.into()));
    let rho = vec![
        vec![half, Expr::zero(), Expr::zero()],
        vec![Expr::zero(), quarter.clone(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), quarter],
    ];
    let result = von_neumann_entropy(&rho, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(
        rendered == "3/2*log(2)"
            || rendered == "-3/2*log(1/2)"
            || rendered == "-1/2*log(1/2) + -1/2*log(1/4)"
            || rendered == "-1/2*log(1/4) + -1/2*log(1/2)"
            || rendered == "-1/2*log(1/2) + -2*1/4*log(1/4)"
            || rendered == "-2*1/4*log(1/4) + -1/2*log(1/2)",
        "got {rendered}"
    );
}

#[test]
fn von_neumann_entropy_rejects_nonhermitian_matrix() {
    let interner = int();
    let matrix = vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::Int(2.into()), Expr::zero()],
    ];
    let err = von_neumann_entropy(&matrix, &interner);
    assert_eq!(err, Err(EntropyError::StateNotHermitian));
}

#[test]
fn mutual_information_bell_state_is_log_four() {
    let interner = int();
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let result = von_neumann_mutual_information_bipartite(&rho, 2, 2, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(
        rendered == "log(4)"
            || rendered == "2*log(2)"
            || rendered == "log(2) + log(2)"
            || rendered == "-2*log(1/2)"
            || rendered == "-log(1/2) + -log(1/2)"
            || rendered == "-1*log(1/2) + -1*log(1/2)"
            || rendered == "log(1) + -2*log(1/2)",
        "got {rendered}"
    );
}

#[test]
fn conditional_entropy_bell_state_is_minus_log_two() {
    let interner = int();
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let result = conditional_entropy_b_given_a(&rho, 2, 2, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(
        rendered == "-log(2)"
            || rendered == "log(1/2)"
            || rendered == "-1*log(2)"
            || rendered == "0 + -1*log(1/2)"
            || rendered == "-1*log(1/2)",
        "got {rendered}"
    );
}

#[test]
fn mutual_information_product_state_is_zero() {
    let interner = int();
    let product = vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()];
    let rho = density_matrix(&product);
    assert_eq!(
        von_neumann_mutual_information_bipartite(&rho, 2, 2, &interner).unwrap(),
        Expr::zero()
    );
}

#[test]
fn renyi2_mutual_information_bell_state_is_log_four() {
    let interner = int();
    let sqrt2_inv = Expr::pow(
        Expr::Int(2.into()),
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
    );
    let bell = vec![
        Expr::mul(vec![sqrt2_inv.clone()]),
        Expr::zero(),
        Expr::zero(),
        Expr::mul(vec![sqrt2_inv]),
    ];
    let rho = density_matrix(&bell);
    let result = renyi2_mutual_information_bipartite(&rho, 2, 2, &interner).unwrap();
    let rendered = pretty_print(&result, &interner);
    assert!(
        rendered == "log(4)"
            || rendered == "2*log(2)"
            || rendered == "log(2) + log(2)"
            || rendered == "-2*log(1/2)"
            || rendered == "-log(1/2) + -log(1/2)"
            || rendered == "-1*log(1/2) + -1*log(1/2)"
            || rendered == "log(1) + -2*log(1/2)",
        "got {rendered}"
    );
}

#[test]
fn bloch_vector_zero_state_is_001() {
    let rho0 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    assert_eq!(
        bloch_vector(&rho0).unwrap(),
        [Expr::zero(), Expr::zero(), Expr::one()]
    );
}

#[test]
fn bloch_vector_plus_state_is_100() {
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let rho_plus = vec![
        vec![half.clone(), half.clone()],
        vec![half.clone(), half.clone()],
    ];
    assert_eq!(
        bloch_vector(&rho_plus).unwrap(),
        [Expr::one(), Expr::zero(), Expr::zero()]
    );
}

#[test]
fn qubit_density_from_bloch_z_axis_is_zero_state() {
    let rho = qubit_density_from_bloch([Expr::zero(), Expr::zero(), Expr::one()]);
    assert_eq!(
        rho,
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn bloch_vector_rejects_non_qubit_matrix() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero()],
    ];
    assert_eq!(
        bloch_vector(&rho),
        Err(QubitStateError::NotTwoByTwo { rows: 3, cols: 3 })
    );
}

#[test]
fn post_measurement_state_of_certain_outcome_is_unchanged() {
    let rho = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let projector = basis_projector(0, 2).unwrap();
    let post = post_measurement_state(&projector, &rho, 0).unwrap();
    assert_eq!(post, rho);
}

#[test]
fn lindblad_rhs_vanishes_for_commuting_hamiltonian_and_state() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho_diag = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let rhs = lindblad_rhs(&h, &rho_diag, &[], &interner).unwrap();
    assert_eq!(
        rhs,
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn lindblad_rhs_vanishes_for_maximally_mixed_state_with_no_jumps() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho_mixed = vec![
        vec![half.clone(), Expr::zero()],
        vec![Expr::zero(), half.clone()],
    ];
    let rhs = lindblad_rhs(&h, &rho_mixed, &[], &interner).unwrap();
    assert_eq!(
        rhs,
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn lindblad_rhs_rejects_dimension_mismatch() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho_diag = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let jump_ops = vec![vec![vec![Expr::one(), Expr::zero(), Expr::zero()]]];
    let err = lindblad_rhs(&h, &rho_diag, &jump_ops, &interner);
    assert_eq!(
        err,
        Err(LindbladError::DimensionMismatch {
            expected: 2,
            actual: 1,
            which: "jump operator",
        })
    );
}

#[test]
fn hermitian_eigenvalues_pauli_z_are_pm_one() {
    let interner = int();
    let sigma_z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let eigenvalues = hermitian_eigenvalues_small(&sigma_z, &interner).unwrap();
    assert_eq!(eigenvalues, vec![Expr::one(), Expr::neg(Expr::one())]);
}

#[test]
fn hermitian_eigenprojectors_diagonal_qubit_are_basis_projectors() {
    let interner = int();
    let sigma_z = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ];
    let projectors = hermitian_eigenprojectors_small(&sigma_z, &interner).unwrap();
    assert_eq!(
        projectors,
        vec![
            vec![
                vec![Expr::one(), Expr::zero()],
                vec![Expr::zero(), Expr::zero()],
            ],
            vec![
                vec![Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::one()],
            ],
        ]
    );
}

#[test]
fn hermitian_eigenvalues_diagonal_qutrit_are_diagonal_entries() {
    let interner = int();
    let diag = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into()), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::Int(3.into())],
    ];
    let eigenvalues = hermitian_eigenvalues_small(&diag, &interner).unwrap();
    assert_eq!(
        eigenvalues,
        vec![Expr::one(), Expr::Int(2.into()), Expr::Int(3.into())]
    );
}

#[test]
fn hermitian_eigenvalues_reject_nonhermitian_matrix() {
    let interner = int();
    let matrix = vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::Int(2.into()), Expr::zero()],
    ];
    let err = hermitian_eigenvalues_small(&matrix, &interner);
    assert_eq!(err, Err(SpectralError::MatrixNotHermitian));
}

#[test]
fn hermitian_eigenprojectors_reject_degenerate_diagonal_case() {
    let interner = int();
    let matrix = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::one()],
    ];
    let err = hermitian_eigenprojectors_small(&matrix, &interner);
    assert_eq!(err, Err(SpectralError::DegenerateSpectrumUnsupported));
}

#[test]
fn partial_trace_rejects_wrong_total_dimension() {
    let rho = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::one()],
    ];
    let err = try_partial_trace(
        &rho,
        BipartiteDims { dim_a: 2, dim_b: 2 },
        PartialTraceTarget::B,
    );
    assert_eq!(
        err,
        Err(QmLinearAlgebraError::SubsystemDimensionMismatch {
            expected: 4,
            actual: 3
        })
    );
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
    let result = normal_order(&expr, &ops, &HashMap::new(), &interner);
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains("a_dag"),
        "normal ordered should have a_dag, got {}",
        result_str
    );
}

#[test]
fn permutation_sector_dimension_for_single_box_is_n() {
    let dimension = permutation_sector_dimension(&[1], 3).unwrap();
    assert_eq!(dimension, 3);
}
