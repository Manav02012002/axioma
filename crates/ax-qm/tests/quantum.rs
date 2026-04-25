use ax_ir::*;
use ax_qm::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::HashMap;

fn int() -> Interner {
    Interner::new()
}

fn simplify_test_expr(expr: Expr) -> Expr {
    fn normalize_rational(value: BigRational) -> Expr {
        if value.denom() == &1.into() {
            Expr::Int(value.numer().clone())
        } else {
            Expr::Rational(value)
        }
    }

    match expr {
        Expr::Add(terms) => {
            let simplified = terms
                .into_iter()
                .map(simplify_test_expr)
                .filter(|term| *term != Expr::zero())
                .collect::<Vec<_>>();
            if simplified.is_empty() {
                Expr::zero()
            } else if simplified
                .iter()
                .all(|term| matches!(term, Expr::Rational(_) | Expr::Int(_)))
            {
                let sum = simplified.into_iter().fold(
                    BigRational::from_integer(0.into()),
                    |acc, term| {
                        acc + match term {
                            Expr::Int(value) => BigRational::from_integer(value),
                            Expr::Rational(value) => value,
                            _ => unreachable!(),
                        }
                    },
                );
                normalize_rational(sum)
            } else {
                Expr::add(simplified)
            }
        }
        Expr::Pow(base, exp) => {
            let base = simplify_test_expr(*base);
            let exp = simplify_test_expr(*exp);
            match (&base, &exp) {
                (Expr::Call(_, args), Expr::Int(power))
                    if args.len() == 1 && power == &2.into() =>
                {
                    args[0].clone()
                }
                _ => Expr::pow(base, exp),
            }
        }
        Expr::Neg(inner) => match simplify_test_expr(*inner) {
            Expr::Int(value) => Expr::Int(-value),
            Expr::Rational(value) => normalize_rational(-value),
            other => Expr::neg(other),
        },
        other => other,
    }
}

fn simplify_test_matrix(matrix: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    matrix
        .into_iter()
        .map(|row| row.into_iter().map(simplify_test_expr).collect())
        .collect()
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
fn spin_half_jz_matches_pauli_z_over_two() {
    let interner = int();
    let expected = ax_linalg::mat_scale(
        &Expr::Rational(BigRational::new(1.into(), 2.into())),
        &pauli_z(&interner),
    );
    assert_eq!(jz_matrix(1, &interner).unwrap(), expected);
}

#[test]
fn spin_one_jz_has_entries_1_0_minus1() {
    let interner = int();
    assert_eq!(
        jz_matrix(2, &interner).unwrap(),
        vec![
            vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::Int((-1).into())],
        ]
    );
}

#[test]
fn spin_half_commutator_jx_jy_is_i_jz() {
    let interner = int();
    let jx = jx_matrix(1, &interner).unwrap();
    let jy = jy_matrix(1, &interner).unwrap();
    let jz = jz_matrix(1, &interner).unwrap();
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));
    let expected = ax_linalg::mat_scale(&i, &jz);
    assert_eq!(commutator(&jx, &jy, &interner), expected);
}

#[test]
fn spin_operator_rejects_invalid_two_j() {
    let interner = int();
    assert_eq!(
        jz_matrix(usize::MAX, &interner),
        Err(SpinError::InvalidSpinQuantumNumber)
    );
}

#[test]
fn hermitian_matrix_exponential_small_zero_matrix_is_identity() {
    let interner = int();
    let zero = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    assert_eq!(
        hermitian_matrix_exponential_small(&zero, &interner).unwrap(),
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ]
    );
}

#[test]
fn hermitian_matrix_exponential_small_diagonal_qubit_exponentiates_entries() {
    let interner = int();
    let matrix = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];

    assert_eq!(
        hermitian_matrix_exponential_small(&matrix, &interner).unwrap(),
        vec![
            vec![
                Expr::Call(interner.get_or_intern("exp"), vec![Expr::one()]),
                Expr::zero()
            ],
            vec![
                Expr::zero(),
                Expr::Call(interner.get_or_intern("exp"), vec![Expr::Int(2.into())]),
            ],
        ]
    );
}

#[test]
fn hermitian_matrix_exponential_small_diagonal_qutrit_exponentiates_entries() {
    let interner = int();
    let matrix = vec![
        vec![Expr::one(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into()), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::Int(3.into())],
    ];

    assert_eq!(
        hermitian_matrix_exponential_small(&matrix, &interner).unwrap(),
        vec![
            vec![
                Expr::Call(interner.get_or_intern("exp"), vec![Expr::one()]),
                Expr::zero(),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::Call(interner.get_or_intern("exp"), vec![Expr::Int(2.into())]),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::Call(interner.get_or_intern("exp"), vec![Expr::Int(3.into())]),
            ],
        ]
    );
}

#[test]
fn hermitian_matrix_exponential_small_rejects_nonhermitian_input() {
    let interner = int();
    let matrix = vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::Int(2.into()), Expr::zero()],
    ];

    assert_eq!(
        hermitian_matrix_exponential_small(&matrix, &interner),
        Err(MatrixExponentialError::MatrixNotHermitian)
    );
}

#[test]
fn time_evolution_operator_zero_matrix_returns_identity() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let zero = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    assert_eq!(
        time_evolution_operator(&zero, t, &interner).unwrap(),
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ]
    );
}

#[test]
fn time_evolution_operator_diagonal_qubit_returns_expected_phases() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];

    assert_eq!(
        time_evolution_operator(&h, t.clone(), &interner).unwrap(),
        vec![
            vec![
                Expr::Call(
                    interner.get_or_intern("exp"),
                    vec![Expr::mul(vec![Expr::neg(i.clone()), t.clone()])]
                ),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::Call(
                    interner.get_or_intern("exp"),
                    vec![Expr::mul(vec![Expr::Int((-2).into()), i, t])]
                ),
            ],
        ]
    );
}

#[test]
fn time_evolution_operator_rejects_nonhermitian_input() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let h = vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::Int(2.into()), Expr::zero()],
    ];

    assert_eq!(
        time_evolution_operator(&h, t, &interner),
        Err(MatrixExponentialError::MatrixNotHermitian)
    );
}

#[test]
fn schrodinger_evolve_zero_hamiltonian_leaves_state_unchanged() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let h = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let psi0 = vec![Expr::one(), Expr::zero()];

    assert_eq!(
        schrodinger_evolve_state(&h, &psi0, t, &interner).unwrap(),
        psi0
    );
}

#[test]
fn schrodinger_evolve_pauli_z_on_zero_state_picks_up_phase() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));

    assert_eq!(
        schrodinger_evolve_state(
            &pauli_z(&interner),
            &[Expr::one(), Expr::zero()],
            t.clone(),
            &interner
        )
        .unwrap(),
        vec![
            Expr::Call(
                interner.get_or_intern("exp"),
                vec![Expr::mul(vec![Expr::neg(i), t])]
            ),
            Expr::zero(),
        ]
    );
}

#[test]
fn schrodinger_evolve_rejects_dimension_mismatch() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));

    assert_eq!(
        schrodinger_evolve_state(&pauli_z(&interner), &[Expr::one()], t, &interner),
        Err(StateEvolutionError::StateDimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn heisenberg_evolve_zero_hamiltonian_leaves_operator_unchanged() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let h = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let op0 = pauli_x(&interner);

    assert_eq!(
        heisenberg_evolve_operator(&h, &op0, t, &interner).unwrap(),
        op0
    );
}

#[test]
fn heisenberg_evolve_pauli_z_preserves_pauli_z() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));
    let op0 = pauli_z(&interner);

    assert_eq!(
        heisenberg_evolve_operator(&pauli_z(&interner), &op0, t, &interner).unwrap(),
        op0
    );
}

#[test]
fn heisenberg_evolve_rejects_dimension_mismatch() {
    let interner = int();
    let t = Expr::Sym(interner.get_or_intern("t"));

    assert_eq!(
        heisenberg_evolve_operator(&pauli_z(&interner), &[vec![Expr::one()]], t, &interner),
        Err(OperatorEvolutionError::OperatorDimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn liouville_rhs_commuting_diagonal_case_is_zero() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho_diag = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    assert_eq!(
        liouville_von_neumann_rhs(&h, &rho_diag, &interner).unwrap(),
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn liouville_rhs_maximally_mixed_state_is_zero_for_any_diagonal_h() {
    let interner = int();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];
    let rho_mixed = vec![vec![half.clone(), Expr::zero()], vec![Expr::zero(), half]];

    assert_eq!(
        liouville_von_neumann_rhs(&h, &rho_mixed, &interner).unwrap(),
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn liouville_rhs_rejects_dimension_mismatch() {
    let interner = int();
    let h = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ];

    assert_eq!(
        liouville_von_neumann_rhs(&h, &[vec![Expr::one()]], &interner),
        Err(LiouvilleError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn dyson_series_order_zero_is_one() {
    let interner = int();

    assert_eq!(
        dyson_series(
            Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
            0,
            &interner,
        ),
        Expr::one()
    );
}

#[test]
fn dyson_series_order_one_contains_minus_i_and_one_integral_of_h_t1() {
    let interner = int();
    let result = dyson_series(
        Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
        1,
        &interner,
    );
    let Expr::Add(terms) = result else {
        panic!("expected additive Dyson expansion");
    };

    assert!(terms.contains(&Expr::one()));
    assert!(terms.contains(&Expr::mul(vec![
        Expr::neg(Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))),
        Expr::Rational(BigRational::new(1.into(), 1.into())),
        Expr::Call(
            interner.get_or_intern("integral"),
            vec![
                Expr::Sym(interner.get_or_intern("t1")),
                Expr::zero(),
                time_symbol(&interner),
                Expr::Call(
                    interner.get_or_intern("H"),
                    vec![Expr::Sym(interner.get_or_intern("t1"))],
                ),
            ],
        ),
    ])));
}

#[test]
fn dyson_series_order_two_contains_time_order_wrapper_and_half_factorial() {
    let interner = int();
    let result = dyson_series(
        Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
        2,
        &interner,
    );
    let Expr::Add(terms) = result else {
        panic!("expected additive Dyson expansion");
    };

    assert!(terms.contains(&Expr::mul(vec![
        Expr::pow(
            Expr::neg(Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))),
            Expr::Int(2.into()),
        ),
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::Call(
            interner.get_or_intern("integral"),
            vec![
                Expr::Sym(interner.get_or_intern("t1")),
                Expr::zero(),
                time_symbol(&interner),
                Expr::Call(
                    interner.get_or_intern("integral"),
                    vec![
                        Expr::Sym(interner.get_or_intern("t2")),
                        Expr::zero(),
                        time_symbol(&interner),
                        Expr::Call(
                            interner.get_or_intern("time_order"),
                            vec![Expr::mul(vec![
                                Expr::Call(
                                    interner.get_or_intern("H"),
                                    vec![Expr::Sym(interner.get_or_intern("t1"))],
                                ),
                                Expr::Call(
                                    interner.get_or_intern("H"),
                                    vec![Expr::Sym(interner.get_or_intern("t2"))],
                                ),
                            ])],
                        ),
                    ],
                ),
            ],
        ),
    ])));
}

#[test]
fn magnus_expansion_order_zero_is_zero() {
    let interner = int();

    assert_eq!(
        magnus_expansion(
            Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
            0,
            &interner,
        ),
        Expr::zero()
    );
}

#[test]
fn magnus_expansion_order_one_contains_one_integral_of_h() {
    let interner = int();

    assert_eq!(
        magnus_expansion(
            Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
            1,
            &interner,
        ),
        Expr::mul(vec![
            Expr::neg(Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))),
            Expr::Call(
                interner.get_or_intern("integral"),
                vec![
                    Expr::Sym(interner.get_or_intern("t1")),
                    Expr::zero(),
                    time_symbol(&interner),
                    Expr::Call(
                        interner.get_or_intern("H"),
                        vec![Expr::Sym(interner.get_or_intern("t1"))],
                    ),
                ],
            ),
        ])
    );
}

#[test]
fn magnus_expansion_order_two_contains_a_commutator_term() {
    let interner = int();
    let result = magnus_expansion(
        Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
        2,
        &interner,
    );
    let Expr::Add(terms) = result else {
        panic!("expected additive Magnus expansion");
    };

    assert!(terms.contains(&Expr::mul(vec![
        Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2usize))),
        Expr::Call(
            interner.get_or_intern("integral"),
            vec![
                Expr::Sym(interner.get_or_intern("t1")),
                Expr::zero(),
                time_symbol(&interner),
                Expr::Call(
                    interner.get_or_intern("integral"),
                    vec![
                        Expr::Sym(interner.get_or_intern("t2")),
                        Expr::zero(),
                        time_symbol(&interner),
                        Expr::Call(
                            interner.get_or_intern("commutator"),
                            vec![
                                Expr::Call(
                                    interner.get_or_intern("H"),
                                    vec![Expr::Sym(interner.get_or_intern("t1"))],
                                ),
                                Expr::Call(
                                    interner.get_or_intern("H"),
                                    vec![Expr::Sym(interner.get_or_intern("t2"))],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ])));
}

#[test]
fn magnus_expansion_order_three_contains_required_nested_commutator_structure() {
    fn contains_nested_commutator_sum(expr: &Expr, left: &Expr, right: &Expr) -> bool {
        match expr {
            Expr::Add(terms) => {
                (terms.contains(left) && terms.contains(right))
                    || terms
                        .iter()
                        .any(|term| contains_nested_commutator_sum(term, left, right))
            }
            Expr::Complex(re, im) => {
                contains_nested_commutator_sum(re, left, right)
                    || contains_nested_commutator_sum(im, left, right)
            }
            Expr::Mul(factors) => factors
                .iter()
                .any(|factor| contains_nested_commutator_sum(factor, left, right)),
            Expr::Pow(base, exp) => {
                contains_nested_commutator_sum(base, left, right)
                    || contains_nested_commutator_sum(exp, left, right)
            }
            Expr::Neg(inner) | Expr::Group(inner, _) => {
                contains_nested_commutator_sum(inner, left, right)
            }
            Expr::Call(_, args) | Expr::List(args) => args
                .iter()
                .any(|arg| contains_nested_commutator_sum(arg, left, right)),
            Expr::Matrix(rows) => rows
                .iter()
                .flatten()
                .any(|entry| contains_nested_commutator_sum(entry, left, right)),
            Expr::FnDef(_, _, body) => contains_nested_commutator_sum(body, left, right),
            Expr::Rule(lhs, rhs, _) | Expr::Let(_, lhs, rhs) => {
                contains_nested_commutator_sum(lhs, left, right)
                    || contains_nested_commutator_sum(rhs, left, right)
            }
            Expr::Piecewise(cases) => cases
                .iter()
                .any(|(value, _)| contains_nested_commutator_sum(value, left, right)),
            Expr::Indexed(base, _) => contains_nested_commutator_sum(base, left, right),
            _ => false,
        }
    }

    let interner = int();
    let result = magnus_expansion(
        Expr::Call(interner.get_or_intern("H"), vec![time_symbol(&interner)]),
        3,
        &interner,
    );
    let expected_left = Expr::Call(
        interner.get_or_intern("commutator"),
        vec![
            Expr::Call(
                interner.get_or_intern("H"),
                vec![Expr::Sym(interner.get_or_intern("t1"))],
            ),
            Expr::Call(
                interner.get_or_intern("commutator"),
                vec![
                    Expr::Call(
                        interner.get_or_intern("H"),
                        vec![Expr::Sym(interner.get_or_intern("t2"))],
                    ),
                    Expr::Call(
                        interner.get_or_intern("H"),
                        vec![Expr::Sym(interner.get_or_intern("t3"))],
                    ),
                ],
            ),
        ],
    );
    let expected_right = Expr::Call(
        interner.get_or_intern("commutator"),
        vec![
            Expr::Call(
                interner.get_or_intern("H"),
                vec![Expr::Sym(interner.get_or_intern("t3"))],
            ),
            Expr::Call(
                interner.get_or_intern("commutator"),
                vec![
                    Expr::Call(
                        interner.get_or_intern("H"),
                        vec![Expr::Sym(interner.get_or_intern("t2"))],
                    ),
                    Expr::Call(
                        interner.get_or_intern("H"),
                        vec![Expr::Sym(interner.get_or_intern("t1"))],
                    ),
                ],
            ),
        ],
    );

    assert!(contains_nested_commutator_sum(
        &result,
        &expected_left,
        &expected_right,
    ));
}

#[test]
fn kubo_response_contains_minus_i_theta_trace_and_commutator() {
    let interner = int();
    let a = Expr::Sym(interner.get_or_intern("A"));
    let b = Expr::Sym(interner.get_or_intern("B"));
    let rho0 = Expr::Sym(interner.get_or_intern("rho0"));
    let t = Expr::Sym(interner.get_or_intern("t"));

    assert_eq!(
        kubo_response_function(a.clone(), b.clone(), rho0.clone(), t.clone(), &interner),
        Expr::mul(vec![
            Expr::neg(Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))),
            Expr::Call(interner.get_or_intern("theta"), vec![t.clone()]),
            Expr::Call(
                interner.get_or_intern("trace"),
                vec![Expr::mul(vec![
                    rho0,
                    Expr::Call(
                        interner.get_or_intern("commutator"),
                        vec![
                            Expr::Call(interner.get_or_intern("A"), vec![t]),
                            Expr::Call(interner.get_or_intern("B"), vec![Expr::zero()]),
                        ],
                    ),
                ])],
            ),
        ])
    );
}

#[test]
fn susceptibility_fourier_contains_integral_with_exp_i_omega_t() {
    let interner = int();
    let omega = Expr::Sym(interner.get_or_intern("omega"));
    let response = Expr::Sym(interner.get_or_intern("chi_t"));

    assert_eq!(
        susceptibility_fourier(response.clone(), omega.clone(), &interner),
        Expr::Call(
            interner.get_or_intern("integral"),
            vec![
                Expr::Sym(interner.get_or_intern("t")),
                Expr::Sym(interner.get_or_intern("neg_inf")),
                Expr::Sym(interner.get_or_intern("inf")),
                Expr::mul(vec![
                    Expr::Call(
                        interner.get_or_intern("exp"),
                        vec![Expr::mul(vec![
                            Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one())),
                            omega,
                            Expr::Sym(interner.get_or_intern("t")),
                        ])],
                    ),
                    response,
                ]),
            ],
        )
    );
}

#[test]
fn singlet_state_has_expected_components() {
    let interner = int();
    let inv_sqrt_two = Expr::Call(
        interner.get_or_intern("sqrt"),
        vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
    );

    assert_eq!(
        two_spin_half_singlet_state(&interner),
        vec![
            Expr::zero(),
            inv_sqrt_two.clone(),
            Expr::neg(inv_sqrt_two),
            Expr::zero(),
        ]
    );
}

#[test]
fn triplet_states_have_expected_components() {
    let interner = int();
    let inv_sqrt_two = Expr::Call(
        interner.get_or_intern("sqrt"),
        vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
    );
    let [m_plus_one, m_zero, m_minus_one] = two_spin_half_triplet_states(&interner);

    assert_eq!(
        m_plus_one,
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()]
    );
    assert_eq!(
        m_zero,
        vec![
            Expr::zero(),
            inv_sqrt_two.clone(),
            inv_sqrt_two,
            Expr::zero(),
        ]
    );
    assert_eq!(
        m_minus_one,
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()]
    );
}

#[test]
fn singlet_and_triplet_projectors_sum_to_identity_on_four_dim_space() {
    let interner = int();
    let total = ax_linalg::mat_add(
        &two_spin_half_singlet_projector(&interner),
        &two_spin_half_triplet_projector(&interner),
    );

    assert_eq!(
        simplify_test_matrix(total),
        simplify_test_matrix(identity_matrix(4, &interner))
    );
}

#[test]
fn singlet_projector_annihilates_triplet_m_plus_one() {
    let interner = int();
    let projector = two_spin_half_singlet_projector(&interner);
    let [triplet_m_plus_one, _, _] = two_spin_half_triplet_states(&interner);
    let triplet_column = triplet_m_plus_one
        .into_iter()
        .map(|entry| vec![entry])
        .collect::<Vec<_>>();

    assert_eq!(
        ax_linalg::mat_mul(&projector, &triplet_column, &interner),
        vec![
            vec![Expr::zero()],
            vec![Expr::zero()],
            vec![Expr::zero()],
            vec![Expr::zero()],
        ]
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
    let err = schmidt_coefficients_from_state(
        &[Expr::one(), Expr::zero(), Expr::zero()],
        2,
        2,
        &interner,
    );
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
    assert_eq!(
        negativity,
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    );
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
fn compose_identity_channel_with_itself_is_identity_channel() {
    let composed = compose_kraus_channels(&identity_channel(2), &identity_channel(2)).unwrap();
    assert_eq!(
        composed,
        vec![vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ]]
    );
}

#[test]
fn compose_channels_preserves_application_order() {
    let left = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    let right = vec![vec![
        vec![Expr::Int(3.into()), Expr::zero()],
        vec![Expr::zero(), Expr::Int(4.into())],
    ]];

    let composed = compose_kraus_channels(&left, &right).unwrap();
    assert_eq!(
        composed,
        vec![vec![
            vec![Expr::Int(3.into()), Expr::zero()],
            vec![Expr::zero(), Expr::Int(8.into())],
        ]]
    );
}

#[test]
fn compose_channels_rejects_dimension_mismatch() {
    let left = identity_channel(2);
    let right = identity_channel(3);

    let err = compose_kraus_channels(&left, &right);
    assert_eq!(
        err,
        Err(ChannelError::CompositionDimensionMismatch {
            left_dim: 2,
            right_dim: 3,
        })
    );
}

#[test]
fn tensor_product_identity_channels_gives_identity_channel_on_product_space() {
    let product =
        tensor_product_kraus_channels(&identity_channel(2), &identity_channel(2)).unwrap();
    assert_eq!(
        product,
        vec![vec![
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
        ]]
    );
}

#[test]
fn tensor_product_single_kraus_diagonal_channels_has_expected_operator() {
    let left = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    let right = vec![vec![
        vec![Expr::Int(3.into()), Expr::zero()],
        vec![Expr::zero(), Expr::Int(4.into())],
    ]];

    let product = tensor_product_kraus_channels(&left, &right).unwrap();
    assert_eq!(
        product,
        vec![vec![
            vec![
                Expr::Int(3.into()),
                Expr::zero(),
                Expr::zero(),
                Expr::zero()
            ],
            vec![
                Expr::zero(),
                Expr::Int(4.into()),
                Expr::zero(),
                Expr::zero()
            ],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::Int(6.into()),
                Expr::zero()
            ],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
                Expr::Int(8.into())
            ],
        ]]
    );
}

#[test]
fn tensor_product_channels_rejects_invalid_empty_input() {
    let err = tensor_product_kraus_channels(&[], &identity_channel(2));
    assert_eq!(err, Err(ChannelError::EmptyKrausSet));
}

#[test]
fn choi_matrix_identity_qubit_channel_has_rank_one_bell_like_form() {
    let choi = choi_matrix_from_kraus(&identity_channel(2)).unwrap();
    assert_eq!(
        choi,
        vec![
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::one()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::one()],
        ]
    );
}

#[test]
fn choi_matrix_single_diagonal_kraus_channel_matches_vec_outer() {
    let kraus = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    let expected = outer_column_vector(&vec_column_major(&kraus[0]));
    let choi = choi_matrix_from_kraus(&kraus).unwrap();
    assert_eq!(choi, expected);
}

#[test]
fn choi_matrix_rejects_invalid_kraus_set() {
    let err = choi_matrix_from_kraus(&[]);
    assert!(matches!(err, Err(ChannelError::EmptyKrausSet)));
}

#[test]
fn identity_channel_choi_is_completely_positive() {
    let interner = int();
    let choi = choi_matrix_from_kraus(&identity_channel(2)).unwrap();
    assert!(is_completely_positive_choi_small(&choi, &interner).unwrap());
}

#[test]
fn negative_diagonal_choi_is_not_completely_positive() {
    let interner = int();
    let choi = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int((-1).into())],
    ];
    assert!(!is_completely_positive_choi_small(&choi, &interner).unwrap());
}

#[test]
fn cp_check_rejects_symbolic_non_numeric_input() {
    let interner = int();
    let x = Expr::Sym(interner.get_or_intern("x"));
    let choi = vec![vec![x]];
    let err = is_completely_positive_choi_small(&choi, &interner);
    assert_eq!(err, Err(ChannelError::NonNumericChoiMatrix));
}

#[test]
fn kraus_from_choi_recovers_identity_single_kraus() {
    let choi = choi_matrix_from_kraus(&identity_channel(2)).unwrap();
    let kraus = kraus_from_choi_small(&choi).unwrap();
    assert_eq!(
        kraus,
        vec![vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ]]
    );
}

#[test]
fn kraus_from_choi_recovers_diagonal_single_kraus() {
    let input = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    let choi = choi_matrix_from_kraus(&input).unwrap();
    let kraus = kraus_from_choi_small(&choi).unwrap();
    assert_eq!(kraus, input);
}

#[test]
fn kraus_from_choi_rejects_generic_unsupported_case() {
    let kraus = vec![
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ],
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one()],
        ],
    ];
    let choi = choi_matrix_from_kraus(&kraus).unwrap();
    let err = kraus_from_choi_small(&choi);
    assert_eq!(err, Err(ChannelError::UnsupportedChoiRecovery));
}

#[test]
fn identity_channel_is_trace_preserving() {
    let interner = int();
    assert!(is_trace_preserving_exact(&identity_channel(2), &interner).unwrap());
}

#[test]
fn trace_preserving_residual_identity_channel_is_zero_matrix() {
    let interner = int();
    assert_eq!(
        trace_preserving_residual(&identity_channel(2), &interner).unwrap(),
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn non_tp_single_kraus_channel_is_detected() {
    let interner = int();
    let kraus = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    assert!(!is_trace_preserving_exact(&kraus, &interner).unwrap());
}

#[test]
fn identity_channel_is_unital() {
    let interner = int();
    assert!(is_unital_exact(&identity_channel(2), &interner).unwrap());
}

#[test]
fn unital_residual_identity_channel_is_zero_matrix() {
    let interner = int();
    assert_eq!(
        unital_residual(&identity_channel(2), &interner).unwrap(),
        vec![
            vec![Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn amplitude_damping_like_channel_is_not_unital() {
    let interner = int();
    let g = Expr::Sym(interner.get_or_intern("g"));
    let sqrt = |expr| Expr::Call(interner.get_or_intern("sqrt"), vec![expr]);
    let kraus = vec![
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![
                Expr::zero(),
                sqrt(Expr::add(vec![Expr::one(), Expr::neg(g.clone())])),
            ],
        ],
        vec![
            vec![Expr::zero(), sqrt(g)],
            vec![Expr::zero(), Expr::zero()],
        ],
    ];
    assert!(!is_unital_exact(&kraus, &interner).unwrap());
}

#[test]
fn choi_distance_identical_channels_is_zero() {
    let interner = int();
    assert_eq!(
        choi_frobenius_distance(&identity_channel(2), &identity_channel(2), &interner).unwrap(),
        Expr::zero()
    );
}

#[test]
fn choi_distance_distinct_channels_is_nonzero() {
    let interner = int();
    let right = vec![vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::Int(2.into())],
    ]];
    let distance = choi_frobenius_distance(&identity_channel(2), &right, &interner).unwrap();
    assert_ne!(distance, Expr::zero());
}

#[test]
fn choi_distance_rejects_dimension_mismatch() {
    let interner = int();
    let err = choi_frobenius_distance(&identity_channel(2), &identity_channel(3), &interner);
    assert_eq!(
        err,
        Err(ChannelError::CompositionDimensionMismatch {
            left_dim: 2,
            right_dim: 3,
        })
    );
}

#[test]
fn depolarizing_channel_has_four_kraus_ops() {
    let interner = int();
    let p = Expr::Sym(interner.get_or_intern("p"));
    assert_eq!(depolarizing_channel_qubit(p, &interner).len(), 4);
}

#[test]
fn dephasing_channel_has_two_kraus_ops() {
    let interner = int();
    let p = Expr::Sym(interner.get_or_intern("p"));
    assert_eq!(dephasing_channel_qubit(p, &interner).len(), 2);
}

#[test]
fn amplitude_damping_channel_kraus_shapes_are_2x2() {
    let interner = int();
    let gamma = Expr::Sym(interner.get_or_intern("gamma"));
    let channel = amplitude_damping_channel_qubit(gamma, &interner);
    assert_eq!(channel.len(), 2);
    assert!(channel
        .iter()
        .all(|operator| { operator.len() == 2 && operator.iter().all(|row| row.len() == 2) }));
}

#[test]
fn bit_flip_identity_limit_at_p_zero() {
    let interner = int();
    assert_eq!(
        bit_flip_channel_qubit(Expr::zero(), &interner),
        identity_channel(2)
    );
}

#[test]
fn canonical_channel_family_hint_names_known_constructor_outputs() {
    let interner = int();
    let p = Expr::Sym(interner.get_or_intern("p"));
    let gamma = Expr::Sym(interner.get_or_intern("gamma"));

    assert_eq!(
        canonical_channel_family_hint(&identity_channel(2), &interner).as_deref(),
        Some("identity")
    );
    assert_eq!(
        canonical_channel_family_hint(&depolarizing_channel_qubit(p.clone(), &interner), &interner)
            .as_deref(),
        Some("depolarizing")
    );
    assert_eq!(
        canonical_channel_family_hint(&dephasing_channel_qubit(p.clone(), &interner), &interner)
            .as_deref(),
        Some("dephasing")
    );
    assert_eq!(
        canonical_channel_family_hint(
            &amplitude_damping_channel_qubit(gamma, &interner),
            &interner
        )
        .as_deref(),
        Some("amplitude_damping")
    );
    assert_eq!(
        canonical_channel_family_hint(&bit_flip_channel_qubit(p.clone(), &interner), &interner)
            .as_deref(),
        Some("bit_flip")
    );
    assert_eq!(
        canonical_channel_family_hint(&bit_phase_flip_channel_qubit(p, &interner), &interner)
            .as_deref(),
        Some("bit_phase_flip")
    );
}

#[test]
fn all_canonical_channels_are_trace_preserving_exactly() {
    let interner = int();
    let p = Expr::Sym(interner.get_or_intern("p"));
    let gamma = Expr::Sym(interner.get_or_intern("gamma"));

    println!(
        "depolarizing residual = {:?}",
        trace_preserving_residual(&depolarizing_channel_qubit(p.clone(), &interner), &interner)
    );
    println!(
        "amplitude residual = {:?}",
        trace_preserving_residual(
            &amplitude_damping_channel_qubit(gamma.clone(), &interner),
            &interner
        )
    );
    assert!(is_trace_preserving_exact(
        &depolarizing_channel_qubit(p.clone(), &interner),
        &interner
    )
    .unwrap());
    assert!(
        is_trace_preserving_exact(&dephasing_channel_qubit(p.clone(), &interner), &interner)
            .unwrap()
    );
    assert!(is_trace_preserving_exact(
        &amplitude_damping_channel_qubit(gamma, &interner),
        &interner
    )
    .unwrap());
    assert!(
        is_trace_preserving_exact(&bit_flip_channel_qubit(p.clone(), &interner), &interner)
            .unwrap()
    );
    assert!(
        is_trace_preserving_exact(&phase_flip_channel_qubit(p.clone(), &interner), &interner)
            .unwrap()
    );
    assert!(
        is_trace_preserving_exact(&bit_phase_flip_channel_qubit(p, &interner), &interner).unwrap()
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
fn lindbladian_superoperator_zero_hamiltonian_no_jumps_is_zero_matrix() {
    let interner = int();
    let zero_2x2 = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    assert_eq!(
        lindbladian_superoperator(&zero_2x2, &[], &interner).unwrap(),
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()],
        ]
    );
}

#[test]
fn sparse_lindbladian_zero_generator_has_no_entries() {
    let interner = int();
    let zero_2x2 = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    let sparse = sparse_lindbladian_superoperator_numeric(&zero_2x2, &[], &interner).unwrap();
    assert_eq!(sparse.rows, 4);
    assert_eq!(sparse.cols, 4);
    assert!(sparse.entries.is_empty());
}

#[test]
fn sparse_lindbladian_identity_hamiltonian_numeric_path_builds() {
    let interner = int();
    let identity_2x2 = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::one()],
    ];

    let sparse = sparse_lindbladian_superoperator_numeric(&identity_2x2, &[], &interner).unwrap();
    assert_eq!(sparse.rows, 4);
    assert_eq!(sparse.cols, 4);
}

#[test]
fn sparse_lindbladian_rejects_symbolic_input() {
    let interner = int();
    let zero_2x2 = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let g = interner.get_or_intern("g");
    let jump_ops = vec![vec![
        vec![Expr::Sym(g), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ]];

    assert_eq!(
        sparse_lindbladian_superoperator_numeric(&zero_2x2, &jump_ops, &interner),
        Err(LindbladianSpectrumError::NonNumericOperator)
    );
}

#[test]
fn lindbladian_eigenvalues_zero_generator_are_all_zero() {
    let interner = int();
    let zero_2x2 = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];

    assert_eq!(
        lindbladian_eigenvalues_small(&zero_2x2, &[], &interner).unwrap(),
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::zero()]
    );
}

#[test]
fn unsupported_non_diagonal_2_level_dissipator_spectrum_returns_unsupported_dimension() {
    let interner = int();
    let zero_2x2 = vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ];
    let jump_ops = vec![vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::zero(), Expr::zero()],
    ]];

    assert_eq!(
        lindbladian_eigenvalues_small(&zero_2x2, &jump_ops, &interner),
        Err(LindbladianSpectrumError::UnsupportedDimension { dim: 2 })
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
    let result = normal_order(&expr, &ops, &HashMap::new(), &HashMap::new(), &interner);
    let result_str = pretty_print(&result, &interner);
    assert!(
        result_str.contains("a_dag"),
        "normal ordered should have a_dag, got {}",
        result_str
    );
}

#[test]
fn bosonic_creation_raises_selected_mode() {
    let interner = int();
    let result = bosonic_creation_on_basis(1, &[2, 0, 1], &interner).unwrap();
    let expected = Expr::mul(vec![
        Expr::Call(interner.get_or_intern("sqrt"), vec![Expr::Int(1.into())]),
        Expr::Call(
            interner.get_or_intern("fock_state"),
            vec![Expr::List(vec![
                Expr::Int(2.into()),
                Expr::Int(1.into()),
                Expr::Int(1.into()),
            ])],
        ),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn bosonic_annihilation_lowers_selected_mode() {
    let interner = int();
    let result = bosonic_annihilation_on_basis(0, &[2, 0, 1], &interner).unwrap();
    let expected = Expr::mul(vec![
        Expr::Call(interner.get_or_intern("sqrt"), vec![Expr::Int(2.into())]),
        Expr::Call(
            interner.get_or_intern("fock_state"),
            vec![Expr::List(vec![
                Expr::Int(1.into()),
                Expr::Int(0.into()),
                Expr::Int(1.into()),
            ])],
        ),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn bosonic_annihilation_of_zero_occupation_is_zero() {
    let interner = int();
    let result = bosonic_annihilation_on_basis(1, &[2, 0, 1], &interner).unwrap();
    assert_eq!(result, Expr::zero());
}

#[test]
fn bosonic_basis_state_rejects_empty_list() {
    let interner = int();
    let err = bosonic_basis_state(&[], &interner);
    assert_eq!(err, Err(BosonicBasisError::EmptyOccupationList));
}

#[test]
fn fermionic_creation_applies_jw_sign() {
    let interner = int();
    let result = fermionic_creation_on_basis(1, &[1, 0, 0], &interner).unwrap();
    let expected = Expr::neg(Expr::Call(
        interner.get_or_intern("fermion_state"),
        vec![Expr::List(vec![
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(0.into()),
        ])],
    ));
    assert_eq!(result, expected);
}

#[test]
fn fermionic_annihilation_applies_jw_sign() {
    let interner = int();
    let result = fermionic_annihilation_on_basis(2, &[1, 1, 1], &interner).unwrap();
    let expected = Expr::Call(
        interner.get_or_intern("fermion_state"),
        vec![Expr::List(vec![
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(0.into()),
        ])],
    );
    assert_eq!(result, expected);
}

#[test]
fn fermionic_creation_on_occupied_mode_is_zero() {
    let interner = int();
    let result = fermionic_creation_on_basis(0, &[1, 0, 0], &interner).unwrap();
    assert_eq!(result, Expr::zero());
}

#[test]
fn fermion_state_rejects_invalid_occupation() {
    let interner = int();
    let err = fermionic_basis_state(&[1, 2, 0], &interner);
    assert_eq!(
        err,
        Err(FermionicBasisError::InvalidOccupation { index: 1, value: 2 })
    );
}

#[test]
fn permutation_sector_dimension_for_single_box_is_n() {
    let dimension = permutation_sector_dimension(&[1], 3).unwrap();
    assert_eq!(dimension, 3);
}
