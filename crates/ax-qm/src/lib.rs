#![forbid(unsafe_code)]
#![allow(
    clippy::manual_contains,
    clippy::manual_range_patterns,
    clippy::needless_range_loop,
    clippy::only_used_in_recursion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use ax_ir::{
    CliffordConvention, DiracBarMetadata, Expr, GammaConventionMetadata, GammaMatrixMetadata,
    Index, ModeMetadata, ModeStatistics, SpinorClass, SpinorMetadata, TensorProperty, Variance,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};

use ax_tensor::PropertyLookup;

pub fn permutation_sector_dimension(shape: &[usize], n: usize) -> anyhow::Result<u64> {
    let diagram = ax_young::YoungDiagram::try_new(shape.to_vec())?;
    Ok(ax_young::dimension_of_representation(&diagram, n))
}

/// Wrap an expression in the canonical symbolic time-ordering form `time_order(expr)`.
pub fn time_ordered(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("time_order"), vec![expr])
}

/// Wrap an expression in the canonical symbolic anti-time-ordering form
/// `anti_time_order(expr)`.
pub fn anti_time_ordered(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("anti_time_order"), vec![expr])
}

/// Construct the canonical symbolic time variable `t`.
pub fn time_symbol(interner: &ax_ir::Interner) -> Expr {
    Expr::Sym(interner.get_or_intern("t"))
}

fn substitute_symbol(expr: &Expr, target: lasso::Spur, replacement: &Expr) -> Expr {
    match expr {
        Expr::Sym(sym) if *sym == target => replacement.clone(),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_symbol(re, target, replacement)),
            Box::new(substitute_symbol(im, target, replacement)),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_symbol(term, target, replacement))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_symbol(factor, target, replacement))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_symbol(base, target, replacement),
            substitute_symbol(exp, target, replacement),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_symbol(inner, target, replacement)),
        Expr::Call(name, args) => Expr::Call(
            *name,
            args.iter()
                .map(|arg| substitute_symbol(arg, target, replacement))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_symbol(body, target, replacement)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_symbol(lhs, target, replacement)),
            Box::new(substitute_symbol(rhs, target, replacement)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_symbol(value, target, replacement),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_symbol(base, target, replacement)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(substitute_symbol(inner, target, replacement)),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_symbol(value, target, replacement)),
            Box::new(substitute_symbol(body, target, replacement)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_symbol(item, target, replacement))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| substitute_symbol(entry, target, replacement))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

/// Construct a symbolic nested time-integral placeholder of the requested
/// order using canonical `integral(variable, lower, upper, integrand)` calls.
///
/// Order `0` returns `integrand` unchanged. Higher orders wrap the integrand
/// in nested integrals over `t1, ..., tn`, each spanning the symbolic interval
/// from `0` to `t`.
pub fn ordered_time_integral_placeholder(
    order: usize,
    integrand: Expr,
    interner: &ax_ir::Interner,
) -> Expr {
    let integral = interner.get_or_intern("integral");
    let mut current = integrand;

    for idx in (1..=order).rev() {
        current = Expr::Call(
            integral,
            vec![
                Expr::Sym(interner.get_or_intern(&format!("t{idx}"))),
                Expr::zero(),
                time_symbol(interner),
                current,
            ],
        );
    }

    current
}

/// Construct the finite-order symbolic Dyson expansion for a time-dependent
/// Hamiltonian `H(t)`.
///
/// Order `0` returns `1`. Higher orders add the canonical symbolic terms
/// `(-i)^n / n!` multiplying nested placeholder time integrals of the
/// time-ordered product `T(H(t1) ... H(tn))`.
pub fn dyson_series(h_of_t: Expr, order: usize, interner: &ax_ir::Interner) -> Expr {
    let mut terms = vec![Expr::one()];
    let Expr::Sym(t_sym) = time_symbol(interner) else {
        unreachable!("time_symbol always returns a symbol");
    };

    for n in 1..=order {
        let product = Expr::mul(
            (1..=n)
                .map(|idx| {
                    substitute_symbol(
                        &h_of_t,
                        t_sym,
                        &Expr::Sym(interner.get_or_intern(&format!("t{idx}"))),
                    )
                })
                .collect(),
        );
        let integrand = if n == 1 {
            product
        } else {
            time_ordered(product, interner)
        };
        let integral = ordered_time_integral_placeholder(n, integrand, interner);
        let Expr::Int(factorial) = factorial_expr(n) else {
            unreachable!("factorial_expr always returns an integer expression");
        };
        let phase = if n == 1 {
            Expr::neg(imag_unit())
        } else {
            Expr::pow(Expr::neg(imag_unit()), Expr::Int(BigInt::from(n)))
        };
        terms.push(Expr::mul(vec![
            phase,
            Expr::Rational(BigRational::new(BigInt::one(), factorial)),
            integral,
        ]));
    }

    Expr::add(terms)
}

/// Construct the symbolic Magnus term `Ω_n` for a time-dependent Hamiltonian
/// `H(t)` through the explicitly supported orders `1`, `2`, and `3`.
///
/// The returned expression uses canonical symbolic `integral(...)` and
/// `commutator(...)` calls. Order `0` and orders above `3` return `0`.
pub fn magnus_term(h_of_t: Expr, order: usize, interner: &ax_ir::Interner) -> Expr {
    let Expr::Sym(t_sym) = time_symbol(interner) else {
        unreachable!("time_symbol always returns a symbol");
    };
    let h_at = |idx: usize| {
        substitute_symbol(
            &h_of_t,
            t_sym,
            &Expr::Sym(interner.get_or_intern(&format!("t{idx}"))),
        )
    };

    match order {
        1 => Expr::mul(vec![
            Expr::neg(imag_unit()),
            ordered_time_integral_placeholder(1, h_at(1), interner),
        ]),
        2 => Expr::mul(vec![
            Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2usize))),
            ordered_time_integral_placeholder(
                2,
                commutator_expr(h_at(1), h_at(2), interner),
                interner,
            ),
        ]),
        3 => {
            let nested_commutator_sum = Expr::add(vec![
                commutator_expr(
                    h_at(1),
                    commutator_expr(h_at(2), h_at(3), interner),
                    interner,
                ),
                commutator_expr(
                    h_at(3),
                    commutator_expr(h_at(2), h_at(1), interner),
                    interner,
                ),
            ]);
            Expr::mul(vec![
                imag_unit(),
                Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(6usize))),
                ordered_time_integral_placeholder(3, nested_commutator_sum, interner),
            ])
        }
        _ => Expr::zero(),
    }
}

/// Construct the finite-order symbolic Magnus expansion for a time-dependent
/// Hamiltonian `H(t)`.
///
/// Order `0` returns `0`. Orders `1`, `2`, and `3` return `Ω1`, `Ω1 + Ω2`,
/// and `Ω1 + Ω2 + Ω3` respectively using explicit symbolic nested integrals
/// and commutators. Orders above `3` return the sum of the implemented terms
/// through `Ω3`.
pub fn magnus_expansion(h_of_t: Expr, order: usize, interner: &ax_ir::Interner) -> Expr {
    if order == 0 {
        return Expr::zero();
    }

    Expr::add(
        (1..=order.min(3))
            .map(|n| magnus_term(h_of_t.clone(), n, interner))
            .collect(),
    )
}

/// Construct the canonical symbolic commutator wrapper `commutator(a, b)`.
pub fn commutator_expr(a: Expr, b: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("commutator"), vec![a, b])
}

fn theta_expr(t: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("theta"), vec![t])
}

fn trace_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("trace"), vec![expr])
}

fn operator_at_time(op: Expr, time: Expr, interner: &ax_ir::Interner) -> Expr {
    match op {
        Expr::Sym(sym) => Expr::Call(sym, vec![time]),
        other => Expr::Call(interner.get_or_intern("at_time"), vec![other, time]),
    }
}

/// Construct the canonical symbolic Kubo linear-response function
/// `-i theta(t) trace(rho0 * commutator(A(t), B(0)))`.
///
/// The result is purely symbolic: it introduces the Heaviside step function
/// as `theta(t)`, the trace as `trace(...)`, and the operator commutator as
/// `commutator(...)` without attempting any expansion or evaluation.
pub fn kubo_response_function(
    a_op: Expr,
    b_op: Expr,
    rho0: Expr,
    t: Expr,
    interner: &ax_ir::Interner,
) -> Expr {
    let a_t = operator_at_time(a_op, t.clone(), interner);
    let b_0 = operator_at_time(b_op, Expr::zero(), interner);
    Expr::mul(vec![
        Expr::neg(imag_unit()),
        theta_expr(t, interner),
        trace_expr(
            Expr::mul(vec![rho0, commutator_expr(a_t, b_0, interner)]),
            interner,
        ),
    ])
}

/// Construct the canonical symbolic Fourier susceptibility
/// `integral(t, neg_inf, inf, exp(i * omega * t) * response)`.
///
/// This helper is symbolic only in the current implementation. It uses the
/// canonical time variable `t` together with symbolic `neg_inf` and `inf`
/// bounds.
pub fn susceptibility_fourier(response: Expr, omega: Expr, interner: &ax_ir::Interner) -> Expr {
    let t = time_symbol(interner);
    Expr::Call(
        interner.get_or_intern("integral"),
        vec![
            t.clone(),
            Expr::Sym(interner.get_or_intern("neg_inf")),
            Expr::Sym(interner.get_or_intern("inf")),
            Expr::mul(vec![
                Expr::Call(
                    interner.get_or_intern("exp"),
                    vec![Expr::mul(vec![imag_unit(), omega, t])],
                ),
                response,
            ]),
        ],
    )
}

fn creation_operator_expr(mode: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("creation"), vec![mode])
}

fn annihilation_operator_expr(mode: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("annihilation"), vec![mode])
}

fn conj_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("conj"), vec![expr])
}

/// Construct the canonical symbolic Dirac-bar expression `bar(expr)`.
pub fn diracbar_of(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("bar"), vec![expr])
}

/// Construct the canonical symbolic adjoint expression `adjoint(expr)`.
pub fn adjoint_of(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("adjoint"), vec![expr])
}

/// Return a gamma-chain product in reversed order.
pub fn reverse_gamma_chain(exprs: &[Expr]) -> Vec<Expr> {
    exprs.iter().cloned().rev().collect()
}

/// Construct the canonical left chiral projector `P_L = (1 - gamma5) / 2`.
pub fn projector_left(interner: &ax_ir::Interner) -> Expr {
    Expr::mul(vec![
        Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(2usize))),
        Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::Sym(interner.get_or_intern("gamma5"))),
        ]),
    ])
}

/// Construct the canonical right chiral projector `P_R = (1 + gamma5) / 2`.
pub fn projector_right(interner: &ax_ir::Interner) -> Expr {
    Expr::mul(vec![
        Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(2usize))),
        Expr::add(vec![
            Expr::one(),
            Expr::Sym(interner.get_or_intern("gamma5")),
        ]),
    ])
}

/// Construct the canonical Lorentz-generator spin matrix `sigma(mu, nu)`.
///
/// The symbolic convention is `sigma^{mu nu} = (i/2) [gamma^mu, gamma^nu]`.
pub fn sigma_matrix(mu: Expr, nu: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("sigma"), vec![mu, nu])
}

fn gamma_factor(index: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("gamma"), vec![index])
}

fn i_over_two() -> Expr {
    Expr::Complex(
        Box::new(Expr::zero()),
        Box::new(Expr::Rational(BigRational::new(
            BigInt::one(),
            BigInt::from(2usize),
        ))),
    )
}

fn minus_two_i() -> Expr {
    Expr::Complex(
        Box::new(Expr::zero()),
        Box::new(Expr::Int(BigInt::from(-2i32))),
    )
}

/// Expand canonical `sigma(mu, nu)` factors into gamma-matrix commutators.
///
/// The top-level conversion is `sigma(mu,nu) -> (i/2)(gamma(mu) gamma(nu)
/// - gamma(nu) gamma(mu))`; other expression structure is traversed so sigma
/// factors inside sums, products, powers, and calls are converted as well.
pub fn sigma_to_gamma_commutator(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Call(sym, args) if interner.resolve(*sym) == "sigma" && args.len() == 2 => {
            let mu = args[0].clone();
            let nu = args[1].clone();
            Expr::mul(vec![
                i_over_two(),
                Expr::add(vec![
                    Expr::mul(vec![
                        gamma_factor(mu.clone(), interner),
                        gamma_factor(nu.clone(), interner),
                    ]),
                    Expr::neg(Expr::mul(vec![
                        gamma_factor(nu, interner),
                        gamma_factor(mu, interner),
                    ])),
                ]),
            ])
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| sigma_to_gamma_commutator(term, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| sigma_to_gamma_commutator(factor, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            sigma_to_gamma_commutator(base, interner),
            sigma_to_gamma_commutator(exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(sigma_to_gamma_commutator(inner, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(sigma_to_gamma_commutator(re, interner)),
            Box::new(sigma_to_gamma_commutator(im, interner)),
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| sigma_to_gamma_commutator(arg, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(sigma_to_gamma_commutator(base, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(sigma_to_gamma_commutator(inner, interner)), *rel)
        }
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| sigma_to_gamma_commutator(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| sigma_to_gamma_commutator(item, interner))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn gamma_call_index(expr: &Expr, interner: &ax_ir::Interner) -> Option<Expr> {
    match expr {
        Expr::Call(sym, args) if interner.resolve(*sym) == "gamma" && args.len() == 1 => {
            Some(args[0].clone())
        }
        _ => None,
    }
}

fn gamma_product_indices(expr: &Expr, interner: &ax_ir::Interner) -> Option<(Expr, Expr)> {
    match expr {
        Expr::Mul(factors) if factors.len() == 2 => Some((
            gamma_call_index(&factors[0], interner)?,
            gamma_call_index(&factors[1], interner)?,
        )),
        _ => None,
    }
}

fn signed_unit_term(expr: &Expr) -> Option<(i8, Expr)> {
    match expr {
        Expr::Neg(inner) => Some((-1, inner.as_ref().clone())),
        Expr::Mul(factors) if !factors.is_empty() => match &factors[0] {
            Expr::Int(n) if *n == BigInt::from(-1i32) => {
                Some((-1, Expr::mul(factors[1..].to_vec())))
            }
            Expr::Rational(r) if *r == BigRational::from_integer(BigInt::from(-1i32)) => {
                Some((-1, Expr::mul(factors[1..].to_vec())))
            }
            _ => Some((1, expr.clone())),
        },
        _ => Some((1, expr.clone())),
    }
}

fn gamma_commutator_indices_from_difference(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<(Expr, Expr)> {
    let Expr::Add(terms) = expr else {
        return None;
    };
    if terms.len() != 2 {
        return None;
    }
    let (sign_a, base_a) = signed_unit_term(&terms[0])?;
    let (sign_b, base_b) = signed_unit_term(&terms[1])?;
    let (mu_a, nu_a) = gamma_product_indices(&base_a, interner)?;
    let (mu_b, nu_b) = gamma_product_indices(&base_b, interner)?;

    match (sign_a, sign_b) {
        (1, -1) if mu_a == nu_b && nu_a == mu_b => Some((mu_a, nu_a)),
        (-1, 1) if mu_b == nu_a && nu_b == mu_a => Some((mu_b, nu_b)),
        _ => None,
    }
}

/// Convert an exact gamma-matrix commutator pattern back to the sigma basis.
///
/// Recognized forms are `commutator(gamma(mu), gamma(nu))` and the exact
/// two-term product difference `gamma(mu) gamma(nu) - gamma(nu) gamma(mu)`.
/// Nonmatching inputs are returned unchanged.
pub fn gamma_commutator_to_sigma(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let indices = match expr {
        Expr::Call(sym, args) if interner.resolve(*sym) == "commutator" && args.len() == 2 => {
            Some((
                gamma_call_index(&args[0], interner),
                gamma_call_index(&args[1], interner),
            ))
            .and_then(|(mu, nu)| Some((mu?, nu?)))
        }
        _ => gamma_commutator_indices_from_difference(expr, interner),
    };

    if let Some((mu, nu)) = indices {
        Expr::mul(vec![minus_two_i(), sigma_matrix(mu, nu, interner)])
    } else {
        expr.clone()
    }
}

/// Construct the exact factorial `n!` as an integer expression.
pub fn factorial_expr(n: usize) -> Expr {
    Expr::Int(factorial(n))
}

fn truncated_exponential_series(generator: Expr, order: usize) -> Expr {
    let mut terms = Vec::with_capacity(order + 1);
    terms.push(Expr::one());
    for n in 1..=order {
        let Expr::Int(factorial) = factorial_expr(n) else {
            unreachable!("factorial_expr always returns an integer expression");
        };
        terms.push(Expr::mul(vec![
            Expr::Rational(BigRational::new(BigInt::one(), factorial)),
            Expr::pow(generator.clone(), Expr::Int(BigInt::from(n))),
        ]));
    }
    Expr::add(terms)
}

/// Construct the truncated symbolic series for the bosonic displacement operator
/// `D(alpha) = exp(alpha a† - conj(alpha) a)` through the requested order.
///
/// Order `0` returns `1`. Higher orders return the exponential power series
/// `Σ_{n=0}^order generator^n / n!` without attempting operator simplification.
pub fn displacement_operator_series(
    alpha: Expr,
    mode: Expr,
    order: usize,
    interner: &ax_ir::Interner,
) -> Expr {
    let generator = Expr::add(vec![
        Expr::mul(vec![
            alpha.clone(),
            creation_operator_expr(mode.clone(), interner),
        ]),
        Expr::neg(Expr::mul(vec![
            conj_expr(alpha, interner),
            annihilation_operator_expr(mode, interner),
        ])),
    ]);
    truncated_exponential_series(generator, order)
}

/// Construct the truncated symbolic series for the bosonic squeezing operator
/// `S(zeta) = exp(1/2 (zeta a† a† - conj(zeta) a a))` through the requested order.
///
/// Order `0` returns `1`. Higher orders return the exponential power series
/// `Σ_{n=0}^order generator^n / n!` without attempting operator simplification.
pub fn squeezing_operator_series(
    zeta: Expr,
    mode: Expr,
    order: usize,
    interner: &ax_ir::Interner,
) -> Expr {
    let creation = creation_operator_expr(mode.clone(), interner);
    let annihilation = annihilation_operator_expr(mode, interner);
    let generator = Expr::mul(vec![
        Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(2usize))),
        Expr::add(vec![
            Expr::mul(vec![zeta.clone(), creation.clone(), creation]),
            Expr::neg(Expr::mul(vec![
                conj_expr(zeta, interner),
                annihilation.clone(),
                annihilation,
            ])),
        ]),
    ]);
    truncated_exponential_series(generator, order)
}

/// Construct the finite-order Baker-Campbell-Hausdorff expansion for `log(exp(A) exp(B))`.
///
/// Order `0` returns `0`. Orders `1` through `4` add the standard BCH terms through the
/// requested truncation order.
pub fn bch_expand(a: Expr, b: Expr, order: usize, interner: &ax_ir::Interner) -> Expr {
    fn rational(numer: i64, denom: i64) -> Expr {
        Expr::Rational(BigRational::new(numer.into(), denom.into()))
    }

    if order == 0 {
        return Expr::zero();
    }

    let ab = commutator_expr(a.clone(), b.clone(), interner);
    let aab = commutator_expr(a.clone(), ab.clone(), interner);
    let bba = commutator_expr(
        b.clone(),
        commutator_expr(b.clone(), a.clone(), interner),
        interner,
    );
    let baab = commutator_expr(
        b.clone(),
        commutator_expr(a.clone(), aab.clone(), interner),
        interner,
    );

    let mut terms = vec![a, b];
    if order >= 2 {
        terms.push(Expr::mul(vec![rational(1, 2), ab]));
    }
    if order >= 3 {
        terms.push(Expr::mul(vec![rational(1, 12), aab]));
        terms.push(Expr::mul(vec![rational(1, 12), bba]));
    }
    if order >= 4 {
        terms.push(Expr::mul(vec![rational(-1, 24), baab]));
    }
    Expr::add(terms)
}

fn fock_state_expr(occupations: &[usize], interner: &ax_ir::Interner) -> Expr {
    Expr::Call(
        interner.get_or_intern("fock_state"),
        vec![Expr::List(
            occupations
                .iter()
                .map(|n| Expr::Int(BigInt::from(*n)))
                .collect(),
        )],
    )
}

fn fermion_state_expr(occupations: &[usize], interner: &ax_ir::Interner) -> Expr {
    Expr::Call(
        interner.get_or_intern("fermion_state"),
        vec![Expr::List(
            occupations
                .iter()
                .map(|n| Expr::Int(BigInt::from(*n)))
                .collect(),
        )],
    )
}

/// Construct the canonical multimode bosonic occupation-basis state `fock_state([n0, n1, ...])`.
pub fn bosonic_basis_state(
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, BosonicBasisError> {
    if occupations.is_empty() {
        return Err(BosonicBasisError::EmptyOccupationList);
    }
    Ok(fock_state_expr(occupations, interner))
}

/// Apply a bosonic creation operator on the selected mode of a multimode occupation-basis state.
pub fn bosonic_creation_on_basis(
    mode: usize,
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, BosonicBasisError> {
    if occupations.is_empty() {
        return Err(BosonicBasisError::EmptyOccupationList);
    }
    if mode >= occupations.len() {
        return Err(BosonicBasisError::ModeIndexOutOfRange {
            index: mode,
            mode_count: occupations.len(),
        });
    }
    let mut raised = occupations.to_vec();
    raised[mode] += 1;
    Ok(Expr::mul(vec![
        exact_sqrt_expr(Expr::Int(BigInt::from(occupations[mode] + 1)), interner),
        fock_state_expr(&raised, interner),
    ]))
}

/// Apply a bosonic annihilation operator on the selected mode of a multimode occupation-basis state.
pub fn bosonic_annihilation_on_basis(
    mode: usize,
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, BosonicBasisError> {
    if occupations.is_empty() {
        return Err(BosonicBasisError::EmptyOccupationList);
    }
    if mode >= occupations.len() {
        return Err(BosonicBasisError::ModeIndexOutOfRange {
            index: mode,
            mode_count: occupations.len(),
        });
    }
    if occupations[mode] == 0 {
        return Ok(Expr::zero());
    }
    let mut lowered = occupations.to_vec();
    lowered[mode] -= 1;
    Ok(Expr::mul(vec![
        exact_sqrt_expr(Expr::Int(BigInt::from(occupations[mode])), interner),
        fock_state_expr(&lowered, interner),
    ]))
}

fn validate_fermionic_occupations(occupations: &[usize]) -> Result<(), FermionicBasisError> {
    if occupations.is_empty() {
        return Err(FermionicBasisError::EmptyOccupationList);
    }
    for (index, value) in occupations.iter().copied().enumerate() {
        if value > 1 {
            return Err(FermionicBasisError::InvalidOccupation { index, value });
        }
    }
    Ok(())
}

fn fermionic_sign_expr(mode: usize, occupations: &[usize], _interner: &ax_ir::Interner) -> Expr {
    let parity = occupations[..mode].iter().copied().sum::<usize>() % 2;
    let state = Expr::one();
    if parity == 0 {
        state
    } else {
        Expr::neg(state)
    }
}

/// Construct the canonical multimode fermionic occupation-basis state `fermion_state([n0, n1, ...])`.
pub fn fermionic_basis_state(
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, FermionicBasisError> {
    validate_fermionic_occupations(occupations)?;
    Ok(fermion_state_expr(occupations, interner))
}

/// Apply a fermionic creation operator on the selected mode of a multimode occupation-basis state,
/// including the exact Jordan-Wigner sign from occupied earlier modes.
pub fn fermionic_creation_on_basis(
    mode: usize,
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, FermionicBasisError> {
    validate_fermionic_occupations(occupations)?;
    if mode >= occupations.len() {
        return Err(FermionicBasisError::ModeIndexOutOfRange {
            index: mode,
            mode_count: occupations.len(),
        });
    }
    if occupations[mode] == 1 {
        return Ok(Expr::zero());
    }
    let mut raised = occupations.to_vec();
    raised[mode] = 1;
    let state = fermion_state_expr(&raised, interner);
    let sign = fermionic_sign_expr(mode, occupations, interner);
    Ok(if sign == Expr::one() {
        state
    } else {
        Expr::neg(state)
    })
}

/// Apply a fermionic annihilation operator on the selected mode of a multimode occupation-basis
/// state, including the exact Jordan-Wigner sign from occupied earlier modes.
pub fn fermionic_annihilation_on_basis(
    mode: usize,
    occupations: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, FermionicBasisError> {
    validate_fermionic_occupations(occupations)?;
    if mode >= occupations.len() {
        return Err(FermionicBasisError::ModeIndexOutOfRange {
            index: mode,
            mode_count: occupations.len(),
        });
    }
    if occupations[mode] == 0 {
        return Ok(Expr::zero());
    }
    let mut lowered = occupations.to_vec();
    lowered[mode] = 0;
    let state = fermion_state_expr(&lowered, interner);
    let sign = fermionic_sign_expr(mode, occupations, interner);
    Ok(if sign == Expr::one() {
        state
    } else {
        Expr::neg(state)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorKind {
    Creation,
    Annihilation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorStatistics {
    Bosonic,
    Fermionic,
}

#[derive(Clone, Debug)]
pub enum GammaEntry {
    Gamma(lasso::Spur),
    Index(usize),
    Gamma5,
    Identity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearPair {
    pub psi1: lasso::Spur,
    pub gamma_a: Vec<lasso::Spur>,
    pub psi2: lasso::Spur,
    pub psi3: lasso::Spur,
    pub gamma_b: Vec<lasso::Spur>,
    pub psi4: lasso::Spur,
    pub remaining_factors: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FierzError {
    NoBilinearPair,
    AmbiguousBilinears(usize),
    MalformedBilinear,
    AmbiguousSpinorOrder,
    SpinorOrderMismatch,
    IncompatibleSpinorMetadata,
    IncompatibleSpinorDimension,
    IncompatibleSpinorChirality,
    InsufficientConventionData,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpinError {
    #[error("invalid spin quantum number")]
    InvalidSpinQuantumNumber,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QmLinearAlgebraError {
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch { left: usize, right: usize },
    #[error("basis index out of range: index={index}, dim={dim}")]
    BasisIndexOutOfRange { index: usize, dim: usize },
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("subsystem dimension mismatch: expected={expected}, actual={actual}")]
    SubsystemDimensionMismatch { expected: usize, actual: usize },
    #[error("invalid trace target: target={target}")]
    InvalidTraceTarget { target: char },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChannelError {
    #[error("empty Kraus set")]
    EmptyKrausSet,
    #[error("channel composition dimension mismatch: left={left_dim}, right={right_dim}")]
    CompositionDimensionMismatch { left_dim: usize, right_dim: usize },
    #[error("non-numeric Choi matrix")]
    NonNumericChoiMatrix,
    #[error("unsupported complete positivity check for Choi dimension {dim}")]
    UnsupportedCompletePositivityCheck { dim: usize },
    #[error("unsupported Choi recovery")]
    UnsupportedChoiRecovery,
    #[error("invalid Choi dimension: dim={dim}")]
    InvalidChoiDimension { dim: usize },
    #[error("invalid Kraus set")]
    InvalidKrausSet,
    #[error("non-square Kraus operator at index {index}: rows={rows}, cols={cols}")]
    NonSquareKraus {
        index: usize,
        rows: usize,
        cols: usize,
    },
    #[error(
        "Kraus operator dimension mismatch at index {index}: expected={expected}, actual={actual}"
    )]
    KrausDimensionMismatch {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MeasurementError {
    #[error("projector dimension mismatch at index {index}: expected={expected}, actual={actual}")]
    ProjectorDimensionMismatch {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
    #[error("zero-probability outcome at index {index}")]
    ZeroProbabilityOutcome { index: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LindbladError {
    #[error("Hamiltonian not square: rows={rows}, cols={cols}")]
    HamiltonianNotSquare { rows: usize, cols: usize },
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
    #[error("dimension mismatch for {which}: expected={expected}, actual={actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        which: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LindbladianSpectrumError {
    #[error(transparent)]
    Lindblad(#[from] LindbladError),
    #[error("unsupported dimension: dim={dim}")]
    UnsupportedDimension { dim: usize },
    #[error("operator contains non-numeric entries")]
    NonNumericOperator,
    #[error(transparent)]
    SparseConversion(#[from] ax_linalg::SparseMatrixError),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ObservableError {
    #[error("operator not square: rows={rows}, cols={cols}")]
    OperatorNotSquare { rows: usize, cols: usize },
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
    #[error("dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StateFunctionalError {
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QubitStateError {
    #[error("matrix is not 2x2: rows={rows}, cols={cols}")]
    NotTwoByTwo { rows: usize, cols: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BosonicBasisError {
    #[error("empty occupation list")]
    EmptyOccupationList,
    #[error("negative occupation unsupported")]
    NegativeOccupationUnsupported,
    #[error("mode index out of range: index={index}, mode_count={mode_count}")]
    ModeIndexOutOfRange { index: usize, mode_count: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FermionicBasisError {
    #[error("empty occupation list")]
    EmptyOccupationList,
    #[error("invalid occupation at index {index}: value={value}")]
    InvalidOccupation { index: usize, value: usize },
    #[error("mode index out of range: index={index}, mode_count={mode_count}")]
    ModeIndexOutOfRange { index: usize, mode_count: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpectralError {
    #[error("matrix is not square: rows={rows}, cols={cols}")]
    MatrixNotSquare { rows: usize, cols: usize },
    #[error("unsupported spectral dimension: dim={dim}")]
    UnsupportedDimension { dim: usize },
    #[error("matrix is not Hermitian")]
    MatrixNotHermitian,
    #[error("degenerate spectrum unsupported")]
    DegenerateSpectrumUnsupported,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MatrixExponentialError {
    #[error("matrix is not square: rows={rows}, cols={cols}")]
    MatrixNotSquare { rows: usize, cols: usize },
    #[error("matrix is not Hermitian")]
    MatrixNotHermitian,
    #[error("unsupported dimension: dim={dim}")]
    UnsupportedDimension { dim: usize },
    #[error("degenerate spectrum unsupported")]
    DegenerateSpectrumUnsupported,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StateEvolutionError {
    #[error(transparent)]
    Hamiltonian(#[from] MatrixExponentialError),
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OperatorEvolutionError {
    #[error(transparent)]
    Hamiltonian(#[from] MatrixExponentialError),
    #[error("operator dimension mismatch: expected={expected}, actual={actual}")]
    OperatorDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LiouvilleError {
    #[error("Hamiltonian not square: rows={rows}, cols={cols}")]
    HamiltonianNotSquare { rows: usize, cols: usize },
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
    #[error("dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EntropyError {
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
    #[error("state is not Hermitian")]
    StateNotHermitian,
    #[error(transparent)]
    Spectral(#[from] SpectralError),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EntanglementError {
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
    #[error("density dimension mismatch: expected={expected}, actual={actual}")]
    DensityDimensionMismatch { expected: usize, actual: usize },
    #[error(transparent)]
    UnsupportedSpectrum(#[from] SpectralError),
    #[error(transparent)]
    PartialTrace(#[from] QmLinearAlgebraError),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NegativityError {
    #[error(transparent)]
    PartialTranspose(#[from] CompositeSpaceError),
    #[error(transparent)]
    Spectral(#[from] SpectralError),
    #[error("dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PerturbationError {
    #[error("unperturbed Hamiltonian is not square: rows={rows}, cols={cols}")]
    UnperturbedHamiltonianNotSquare { rows: usize, cols: usize },
    #[error("perturbation is not square: rows={rows}, cols={cols}")]
    PerturbationNotSquare { rows: usize, cols: usize },
    #[error("dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error(transparent)]
    UnsupportedSpectrum(#[from] SpectralError),
    #[error("degenerate unperturbed spectrum")]
    DegenerateUnperturbedSpectrum,
    #[error("empty degenerate subspace")]
    EmptyDegenerateSubspace,
    #[error("selected subspace is not degenerate in H0")]
    SelectedSubspaceNotDegenerate,
    #[error("state index out of range: index={index}, dim={dim}")]
    StateIndexOutOfRange { index: usize, dim: usize },
    #[error("degenerate subspace index out of range: index={index}, dim={dim}")]
    DegenerateSubspaceIndexOutOfRange { index: usize, dim: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CompositeSpaceError {
    #[error("empty factor list")]
    EmptyFactorList,
    #[error("invalid factor index {index} for factor count {factor_count}")]
    InvalidFactorIndex { index: usize, factor_count: usize },
    #[error("invalid permutation length: expected={expected}, actual={actual}")]
    InvalidPermutationLength { expected: usize, actual: usize },
    #[error("invalid permutation entry {value} for factor count {factor_count}")]
    InvalidPermutationEntry { value: usize, factor_count: usize },
    #[error("duplicate permutation entry {value}")]
    DuplicatePermutationEntry { value: usize },
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("total dimension mismatch: expected={expected}, actual={actual}")]
    TotalDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialTraceTarget {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BipartiteDims {
    pub dim_a: usize,
    pub dim_b: usize,
}

fn qm_error_expr(name: &str, expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern(name), vec![expr.clone()])
}

fn property_sym(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Call(sym, _) => Some(*sym),
        Expr::Indexed(base, _) => property_sym(base),
        _ => None,
    }
}

fn expr_has_property(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    kind: &TensorProperty,
) -> bool {
    property_sym(expr)
        .map(|sym| properties.has_property_kind(sym, kind))
        .unwrap_or(false)
}

fn prop_sort_order(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<Vec<lasso::Spur>> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::SortOrder(order) => Some(order.clone()),
            _ => None,
        })
}

fn declared_spinor_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<SpinorMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::SpinorMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

fn declared_gamma_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaMatrixMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::GammaMatrixMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

fn declared_gamma_convention_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaConventionMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::GammaConventionMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

fn gamma_convention_conflict_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> bool {
    let Some(sym) = property_sym(expr) else {
        return false;
    };
    let conventions = properties
        .get_properties(sym)
        .into_iter()
        .filter_map(|prop| match prop {
            TensorProperty::GammaConventionMeta(metadata) => Some(metadata),
            _ => None,
        })
        .collect::<Vec<_>>();
    conventions
        .windows(2)
        .any(|pair| pair[0].signature != pair[1].signature || pair[0].clifford != pair[1].clifford)
}

fn declared_diracbar_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::DiracBarMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

/// Return the structured quantum-mode metadata attached to a symbol when available.
pub fn mode_metadata_of_symbol(
    symbol: lasso::Spur,
    props: &dyn PropertyLookup,
) -> Option<ax_ir::ModeMetadata> {
    props
        .get_properties(symbol)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::ModeMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

/// Return the operator statistics associated with an operator or mode expression.
///
/// Structured `ModeMeta` metadata takes precedence over any legacy fallback.
/// Bosonic and fermionic mode metadata map directly to operator statistics,
/// while spin-mode metadata returns `None` because the operator layer does not
/// model spin modes as bosonic or fermionic ladder operators.
pub fn operator_statistics_of_expr(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<OperatorStatistics> {
    let symbol = match expr {
        Expr::Call(_, args) if args.len() == 1 => property_sym(&args[0]),
        _ => property_sym(expr),
    }?;
    match mode_metadata_of_symbol(symbol, props).map(|metadata| metadata.statistics) {
        Some(ModeStatistics::Bosonic) => Some(OperatorStatistics::Bosonic),
        Some(ModeStatistics::Fermionic) => Some(OperatorStatistics::Fermionic),
        Some(ModeStatistics::Spin) => None,
        None => None,
    }
}

/// Return the canonical zero-based mode index associated with an operator or mode expression.
pub fn operator_mode_index_of_expr(expr: &Expr, props: &dyn PropertyLookup) -> Option<usize> {
    let symbol = match expr {
        Expr::Call(_, args) if args.len() == 1 => property_sym(&args[0]),
        _ => property_sym(expr),
    }?;
    mode_metadata_of_symbol(symbol, props).map(|metadata| metadata.mode_index)
}

fn operator_mode_identity_of_expr(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<(Option<lasso::Spur>, usize)> {
    let symbol = match expr {
        Expr::Call(_, args) if args.len() == 1 => property_sym(&args[0]),
        _ => property_sym(expr),
    }?;
    let metadata = mode_metadata_of_symbol(symbol, props)?;
    Some((metadata.subsystem, metadata.mode_index))
}

/// Return the structured spinor metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When no structured metadata is present,
/// this synthesizes a best-effort fallback from the legacy `Spinor`,
/// `MajoranaSpinor`, and `WeylSpinor` markers to preserve existing behavior.
pub fn spinor_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<SpinorMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_spinor_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }

    let has_spinor = properties.has_property_kind(sym, &TensorProperty::Spinor)
        || properties.has_property_kind(sym, &TensorProperty::MajoranaSpinor)
        || properties.has_property_kind(sym, &TensorProperty::WeylSpinor);
    if !has_spinor {
        return None;
    }

    let class = match (
        properties.has_property_kind(sym, &TensorProperty::MajoranaSpinor),
        properties.has_property_kind(sym, &TensorProperty::WeylSpinor),
    ) {
        (true, true) => SpinorClass::MajoranaWeyl,
        (true, false) => SpinorClass::Majorana,
        (false, true) => SpinorClass::Weyl,
        (false, false) => SpinorClass::Dirac,
    };

    Some(SpinorMetadata {
        class,
        dimension: None,
        chirality: None,
        index_family: None,
    })
}

/// Return the structured gamma-matrix metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When absent, this falls back to the
/// legacy `GammaMatrixProp` marker and synthesizes empty metadata so older
/// declarations continue to work.
pub fn gamma_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaMatrixMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_gamma_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }
    if let Some(metadata) = declared_gamma_convention_metadata_of_symbol(sym, properties) {
        return Some(GammaMatrixMetadata {
            dimension: metadata.dimension,
            metric_symbol: None,
            index_family: None,
            has_gamma5: metadata.gamma5.is_some(),
        });
    }
    properties
        .has_property_kind(sym, &TensorProperty::GammaMatrixProp)
        .then_some(GammaMatrixMetadata {
            dimension: None,
            metric_symbol: None,
            index_family: None,
            has_gamma5: false,
        })
}

/// Return the structured gamma/Clifford convention metadata attached to a gamma expression.
///
/// This is the primary convention source for gamma algebra. Callers should use
/// this before falling back to legacy dimension or sign heuristics.
pub fn gamma_convention_of_expr(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<ax_ir::GammaConventionMetadata> {
    let sym = property_sym(expr)?;
    declared_gamma_convention_metadata_of_symbol(sym, props)
}

/// Return the dimension associated with a gamma expression.
///
/// Structured `GammaConventionMeta.dimension` takes precedence over the older
/// `GammaMatrixMeta.dimension` fallback.
pub fn gamma_dimension_of_expr(expr: &Expr, props: &dyn PropertyLookup) -> Option<usize> {
    gamma_convention_of_expr(expr, props)
        .and_then(|metadata| metadata.dimension)
        .or_else(|| gamma_declared_dimension(expr, props))
}

/// Return the Clifford sign convention associated with a gamma expression.
///
/// `PlusTwoG` means `{gamma^mu, gamma^nu} = +2 g^{mu nu}`;
/// `MinusTwoG` means `{gamma^mu, gamma^nu} = -2 g^{mu nu}`.
pub fn gamma_clifford_sign(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<ax_ir::CliffordConvention> {
    gamma_convention_of_expr(expr, props).map(|metadata| metadata.clifford)
}

/// Return the structured Dirac-bar metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When absent, this falls back to the
/// legacy `DiracBar` marker and preserves the old default behavior of reversing
/// gamma chains under Dirac-bar expansion.
pub fn diracbar_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_diracbar_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }
    properties
        .has_property_kind(sym, &TensorProperty::DiracBar)
        .then_some(DiracBarMetadata {
            gamma_symbol: None,
            spinor_family: None,
            reverse_gamma_order: true,
        })
}

fn is_majorana_spinor_expr(expr: &Expr, properties: &dyn ax_tensor::PropertyLookup) -> bool {
    spinor_metadata_of_expr(expr, properties)
        .map(|metadata| {
            matches!(
                metadata.class,
                SpinorClass::Majorana | SpinorClass::MajoranaWeyl
            )
        })
        .unwrap_or(false)
}

fn is_weyl_spinor_expr(expr: &Expr, properties: &dyn ax_tensor::PropertyLookup) -> bool {
    spinor_metadata_of_expr(expr, properties)
        .map(|metadata| {
            matches!(
                metadata.class,
                SpinorClass::Weyl | SpinorClass::MajoranaWeyl
            )
        })
        .unwrap_or(false)
}

fn index_family_name(
    idx: &Index,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    idx.index_type.or_else(|| {
        properties
            .index_families()
            .and_then(|families| families.get(&idx.name).map(|family| family.name))
    })
}

fn index_family_dimension(
    idx: &Index,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    let family_name = index_family_name(idx, properties)?;
    properties
        .index_families()
        .and_then(|families| {
            families
                .get(&family_name)
                .and_then(|family| family.dimension)
        })
        .or_else(|| {
            properties
                .index_families()
                .and_then(|families| families.get(&idx.name).and_then(|family| family.dimension))
        })
}

fn collect_all_index_names(expr: &Expr, out: &mut HashSet<lasso::Spur>) {
    match expr {
        Expr::Indexed(base, indices) => {
            for idx in indices {
                out.insert(idx.name);
            }
            collect_all_index_names(base, out);
        }
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) | Expr::Call(_, items) => {
            for item in items {
                collect_all_index_names(item, out);
            }
        }
        Expr::Pow(base, exp) => {
            collect_all_index_names(base, out);
            collect_all_index_names(exp, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_all_index_names(inner, out),
        Expr::Complex(re, im) => {
            collect_all_index_names(re, out);
            collect_all_index_names(im, out);
        }
        Expr::FnDef(_, _, body) => collect_all_index_names(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_all_index_names(lhs, out);
            collect_all_index_names(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_all_index_names(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_all_index_names(value, out);
            collect_all_index_names(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_all_index_names(cell, out);
                }
            }
        }
        _ => {}
    }
}

#[derive(Clone, Debug)]
struct GammaExprData {
    head: Expr,
    #[allow(dead_code)]
    sym: Option<lasso::Spur>,
    indices: Vec<Index>,
}

fn gamma_expr_data(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaExprData> {
    match expr {
        Expr::Call(sym, args) if gamma_metadata_of_expr(&Expr::Sym(*sym), properties).is_some() => {
            let indices = args
                .iter()
                .filter_map(|arg| match arg {
                    Expr::Sym(name) => Some(Index {
                        name: *name,
                        variance: Variance::Up,
                        index_type: None,
                    }),
                    Expr::Indexed(_, idxs) if idxs.len() == 1 => Some(idxs[0].clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            Some(GammaExprData {
                head: Expr::Sym(*sym),
                sym: Some(*sym),
                indices,
            })
        }
        Expr::Indexed(base, indices) if gamma_metadata_of_expr(base, properties).is_some() => {
            Some(GammaExprData {
                head: (**base).clone(),
                sym: property_sym(base),
                indices: indices.clone(),
            })
        }
        _ => None,
    }
}

fn build_gamma_expr(head: &Expr, indices: &[Index]) -> Expr {
    match head {
        Expr::Sym(sym)
            if indices
                .iter()
                .all(|idx| idx.index_type.is_none() && idx.variance == Variance::Up) =>
        {
            Expr::Call(
                *sym,
                indices.iter().map(|idx| Expr::Sym(idx.name)).collect(),
            )
        }
        _ => Expr::Indexed(Box::new(head.clone()), indices.to_vec()),
    }
}

fn build_metric_contraction(metric: &Expr, left: &Index, right: &Index) -> Expr {
    match metric {
        Expr::Sym(_) | Expr::Call(_, _) | Expr::Indexed(_, _) => Expr::Indexed(
            Box::new(metric.clone()),
            vec![
                Index {
                    name: left.name,
                    variance: Variance::Up,
                    index_type: left.index_type,
                },
                Index {
                    name: right.name,
                    variance: Variance::Up,
                    index_type: right.index_type,
                },
            ],
        ),
        _ => Expr::mul(vec![
            metric.clone(),
            Expr::Sym(left.name),
            Expr::Sym(right.name),
        ]),
    }
}

fn build_generalised_delta(uppers: &[Index], lowers: &[Index], interner: &ax_ir::Interner) -> Expr {
    let sym = interner.get_or_intern("generalised_delta");
    let mut args = Vec::with_capacity(uppers.len() + lowers.len());
    args.extend(uppers.iter().map(|idx| Expr::Sym(idx.name)));
    args.extend(lowers.iter().map(|idx| Expr::Sym(idx.name)));
    Expr::Call(sym, args)
}

fn permutation_parity(selection: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..selection.len() {
        for j in i + 1..selection.len() {
            if selection[i] > selection[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

fn combinations_of(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn helper(
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            helper(i + 1, n, k, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    let mut current = Vec::new();
    helper(0, n, k, &mut current, &mut out);
    out
}

fn factorial(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, k| acc * BigInt::from(k))
}

fn fresh_dummy_from_family(
    family: &ax_ir::IndexFamily,
    used: &mut HashSet<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    for value in &family.values {
        if used.insert(*value) {
            return *value;
        }
    }
    let mut counter = 0usize;
    loop {
        let candidate =
            interner.get_or_intern(&format!("{}_{}", interner.resolve(family.name), counter));
        counter += 1;
        if used.insert(candidate) {
            return candidate;
        }
    }
}

impl FierzError {
    fn symbol_name(&self) -> &'static str {
        match self {
            FierzError::NoBilinearPair => "fierz_no_bilinear_pair",
            FierzError::AmbiguousBilinears(_) => "fierz_ambiguous_bilinears",
            FierzError::MalformedBilinear => "fierz_malformed_bilinear",
            FierzError::AmbiguousSpinorOrder => "fierz_ambiguous_spinor_order",
            FierzError::SpinorOrderMismatch => "fierz_spinor_order_mismatch",
            FierzError::IncompatibleSpinorMetadata => "fierz_incompatible_spinor_metadata",
            FierzError::IncompatibleSpinorDimension => "fierz_incompatible_spinor_dimension",
            FierzError::IncompatibleSpinorChirality => "fierz_incompatible_spinor_chirality",
            FierzError::InsufficientConventionData => "fierz_insufficient_convention_data",
        }
    }
}

fn operator_info(
    expr: &Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<(OperatorKind, Option<Expr>, OperatorStatistics)> {
    match expr {
        Expr::Sym(sym) => operators.get(sym).copied().map(|kind| {
            (
                kind,
                Some(Expr::Sym(*sym)),
                operator_statistics_of_expr(&Expr::Sym(*sym), properties)
                    .or_else(|| operator_statistics.get(sym).copied())
                    .unwrap_or(OperatorStatistics::Bosonic),
            )
        }),
        Expr::Call(f, args) if args.len() == 1 => match interner.resolve(*f) {
            "creation" => Some((
                OperatorKind::Creation,
                Some(args[0].clone()),
                operator_statistics_of_expr(&args[0], properties)
                    .or_else(|| match &args[0] {
                        Expr::Sym(sym) => operator_statistics.get(sym).copied(),
                        _ => None,
                    })
                    .unwrap_or(OperatorStatistics::Bosonic),
            )),
            "annihilation" => Some((
                OperatorKind::Annihilation,
                Some(args[0].clone()),
                operator_statistics_of_expr(&args[0], properties)
                    .or_else(|| match &args[0] {
                        Expr::Sym(sym) => operator_statistics.get(sym).copied(),
                        _ => None,
                    })
                    .unwrap_or(OperatorStatistics::Bosonic),
            )),
            _ => None,
        },
        _ => None,
    }
}

fn modes_match(lhs: &Option<Expr>, rhs: &Option<Expr>, properties: &dyn PropertyLookup) -> bool {
    match (lhs, rhs) {
        (Some(a), Some(b)) => {
            let left_identity = operator_mode_identity_of_expr(a, properties);
            let right_identity = operator_mode_identity_of_expr(b, properties);
            if left_identity.is_some() || right_identity.is_some() {
                left_identity.is_some() && left_identity == right_identity
            } else {
                a == b
            }
        }
        _ => false,
    }
}

fn mode_metadata_of_expr(expr: &Expr, props: &dyn PropertyLookup) -> Option<ModeMetadata> {
    let symbol = match expr {
        Expr::Call(_, args) if args.len() == 1 => property_sym(&args[0]),
        _ => property_sym(expr),
    }?;
    mode_metadata_of_symbol(symbol, props)
}

fn ccr_car_operator_info(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<(OperatorKind, Expr, OperatorStatistics)> {
    match expr {
        Expr::Call(f, args) if args.len() == 1 => {
            let kind = match interner.resolve(*f) {
                "creation" => OperatorKind::Creation,
                "annihilation" => OperatorKind::Annihilation,
                _ => return None,
            };
            Some((
                kind,
                args[0].clone(),
                operator_statistics_of_expr(expr, props)?,
            ))
        }
        _ => None,
    }
}

fn same_mode_for_ccr_car(lhs_mode: &Expr, rhs_mode: &Expr, props: &dyn PropertyLookup) -> bool {
    match (
        mode_metadata_of_expr(lhs_mode, props),
        mode_metadata_of_expr(rhs_mode, props),
    ) {
        (Some(lhs), Some(rhs)) => {
            let subsystem_compatible = match (lhs.subsystem, rhs.subsystem) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };
            subsystem_compatible && lhs.mode_index == rhs.mode_index
        }
        _ => lhs_mode == rhs_mode,
    }
}

fn distinct_mode_for_ccr_car(lhs_mode: &Expr, rhs_mode: &Expr, props: &dyn PropertyLookup) -> bool {
    let Some(lhs) = mode_metadata_of_expr(lhs_mode, props) else {
        return false;
    };
    let Some(rhs) = mode_metadata_of_expr(rhs_mode, props) else {
        return false;
    };
    if lhs.statistics != rhs.statistics {
        return false;
    }
    match (lhs.subsystem, rhs.subsystem) {
        (Some(a), Some(b)) if a != b => false,
        _ => lhs.mode_index != rhs.mode_index,
    }
}

fn should_swap_distinct_modes(
    left_kind: OperatorKind,
    left_mode: &Expr,
    right_kind: OperatorKind,
    right_mode: &Expr,
    props: &dyn PropertyLookup,
) -> bool {
    match (left_kind, right_kind) {
        (OperatorKind::Annihilation, OperatorKind::Creation) => true,
        (OperatorKind::Creation, OperatorKind::Annihilation) => false,
        _ => {
            let (Some(left_index), Some(right_index)) = (
                operator_mode_index_of_expr(left_mode, props),
                operator_mode_index_of_expr(right_mode, props),
            ) else {
                return false;
            };
            left_index > right_index
        }
    }
}

fn rewrite_ccr_car_pair(
    left: &Expr,
    right: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let (left_kind, left_mode, left_stats) = ccr_car_operator_info(left, props, interner)?;
    let (right_kind, right_mode, right_stats) = ccr_car_operator_info(right, props, interner)?;

    if left_stats != right_stats {
        return None;
    }

    let swapped = Expr::mul(vec![right.clone(), left.clone()]);

    if same_mode_for_ccr_car(&left_mode, &right_mode, props) {
        return match (left_stats, left_kind, right_kind) {
            (OperatorStatistics::Bosonic, OperatorKind::Annihilation, OperatorKind::Creation) => {
                Some(simplify_expr(Expr::add(vec![Expr::one(), swapped])))
            }
            (OperatorStatistics::Fermionic, OperatorKind::Annihilation, OperatorKind::Creation) => {
                Some(simplify_expr(Expr::add(vec![
                    Expr::one(),
                    Expr::neg(swapped),
                ])))
            }
            _ => None,
        };
    }

    if !distinct_mode_for_ccr_car(&left_mode, &right_mode, props)
        || !should_swap_distinct_modes(left_kind, &left_mode, right_kind, &right_mode, props)
    {
        return None;
    }

    match left_stats {
        OperatorStatistics::Bosonic => Some(swapped),
        OperatorStatistics::Fermionic => Some(simplify_expr(Expr::neg(swapped))),
    }
}

fn simplify_ccr_car_mul_once(
    factors: &[Expr],
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    for index in 0..factors.len().saturating_sub(1) {
        let Some(rewritten_pair) =
            rewrite_ccr_car_pair(&factors[index], &factors[index + 1], props, interner)
        else {
            continue;
        };
        let mut rebuilt = Vec::with_capacity(factors.len() - 1);
        rebuilt.extend_from_slice(&factors[..index]);
        rebuilt.push(rewritten_pair);
        rebuilt.extend_from_slice(&factors[(index + 2)..]);
        return Some(simplify_expr(Expr::mul(rebuilt)));
    }
    None
}

/// Apply one explicit CCR/CAR rewrite step to an expression when a graded ladder-operator
/// relation matches, otherwise return the expression unchanged.
pub fn simplify_ccr_car_once(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            if let Some(rewritten) = simplify_ccr_car_mul_once(factors, props, interner) {
                return rewritten;
            }
            for (index, factor) in factors.iter().enumerate() {
                let updated = simplify_ccr_car_once(factor, props, interner);
                if updated != *factor {
                    let mut rebuilt = factors.clone();
                    rebuilt[index] = updated;
                    return Expr::mul(rebuilt);
                }
            }
            let canonical = ax_tensor::canonicalize_noncommutative_product(expr, props, interner);
            if canonical != *expr {
                canonical
            } else {
                expr.clone()
            }
        }
        Expr::Add(terms) => {
            for (index, term) in terms.iter().enumerate() {
                let updated = simplify_ccr_car_once(term, props, interner);
                if updated != *term {
                    let mut rebuilt = terms.clone();
                    rebuilt[index] = updated;
                    return Expr::add(rebuilt);
                }
            }
            expr.clone()
        }
        Expr::Pow(base, exp) => {
            let new_base = simplify_ccr_car_once(base, props, interner);
            if new_base != **base {
                return Expr::pow(new_base, (**exp).clone());
            }
            let new_exp = simplify_ccr_car_once(exp, props, interner);
            if new_exp != **exp {
                return Expr::pow((**base).clone(), new_exp);
            }
            expr.clone()
        }
        Expr::Neg(inner) => {
            let updated = simplify_ccr_car_once(inner, props, interner);
            if updated != **inner {
                Expr::neg(updated)
            } else {
                expr.clone()
            }
        }
        Expr::Complex(re, im) => {
            let new_re = simplify_ccr_car_once(re, props, interner);
            if new_re != **re {
                return Expr::Complex(Box::new(new_re), im.clone());
            }
            let new_im = simplify_ccr_car_once(im, props, interner);
            if new_im != **im {
                return Expr::Complex(re.clone(), Box::new(new_im));
            }
            expr.clone()
        }
        Expr::Call(f, args) => {
            for (index, arg) in args.iter().enumerate() {
                let updated = simplify_ccr_car_once(arg, props, interner);
                if updated != *arg {
                    let mut rebuilt = args.clone();
                    rebuilt[index] = updated;
                    return Expr::Call(*f, rebuilt);
                }
            }
            expr.clone()
        }
        Expr::FnDef(name, params, body) => {
            let updated = simplify_ccr_car_once(body, props, interner);
            if updated != **body {
                Expr::FnDef(*name, params.clone(), Box::new(updated))
            } else {
                expr.clone()
            }
        }
        Expr::Rule(lhs, rhs, trust) => {
            let new_lhs = simplify_ccr_car_once(lhs, props, interner);
            if new_lhs != **lhs {
                return Expr::Rule(Box::new(new_lhs), rhs.clone(), *trust);
            }
            let new_rhs = simplify_ccr_car_once(rhs, props, interner);
            if new_rhs != **rhs {
                return Expr::Rule(lhs.clone(), Box::new(new_rhs), *trust);
            }
            expr.clone()
        }
        Expr::Piecewise(cases) => {
            for (index, (value, condition)) in cases.iter().enumerate() {
                let updated = simplify_ccr_car_once(value, props, interner);
                if updated != *value {
                    let mut rebuilt = cases.clone();
                    rebuilt[index] = (updated, condition.clone());
                    return Expr::Piecewise(rebuilt);
                }
            }
            expr.clone()
        }
        Expr::Indexed(base, indices) => {
            let updated = simplify_ccr_car_once(base, props, interner);
            if updated != **base {
                Expr::Indexed(Box::new(updated), indices.clone())
            } else {
                expr.clone()
            }
        }
        Expr::Let(name, value, body) => {
            let new_value = simplify_ccr_car_once(value, props, interner);
            if new_value != **value {
                return Expr::Let(*name, Box::new(new_value), body.clone());
            }
            let new_body = simplify_ccr_car_once(body, props, interner);
            if new_body != **body {
                return Expr::Let(*name, value.clone(), Box::new(new_body));
            }
            expr.clone()
        }
        Expr::List(items) => {
            for (index, item) in items.iter().enumerate() {
                let updated = simplify_ccr_car_once(item, props, interner);
                if updated != *item {
                    let mut rebuilt = items.clone();
                    rebuilt[index] = updated;
                    return Expr::List(rebuilt);
                }
            }
            expr.clone()
        }
        Expr::Matrix(rows) => {
            for (row_index, row) in rows.iter().enumerate() {
                for (col_index, cell) in row.iter().enumerate() {
                    let updated = simplify_ccr_car_once(cell, props, interner);
                    if updated != *cell {
                        let mut rebuilt = rows.clone();
                        rebuilt[row_index][col_index] = updated;
                        return Expr::Matrix(rebuilt);
                    }
                }
            }
            expr.clone()
        }
        _ => expr.clone(),
    }
}

/// Repeatedly apply explicit CCR/CAR rewrites until the expression reaches a fixed point.
pub fn simplify_ccr_car_full(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut current = expr.clone();
    loop {
        let next = simplify_expr(simplify_ccr_car_once(&current, props, interner));
        if next == current {
            return current;
        }
        current = next;
    }
}

fn graded_reorder_mul(
    factors: Vec<Expr>,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    for i in 0..factors.len().saturating_sub(1) {
        let left = operator_info(
            &factors[i],
            operators,
            operator_statistics,
            properties,
            interner,
        );
        let right = operator_info(
            &factors[i + 1],
            operators,
            operator_statistics,
            properties,
            interner,
        );
        if let (
            Some((OperatorKind::Annihilation, _, left_stats)),
            Some((OperatorKind::Creation, _, right_stats)),
        ) = (left, right)
        {
            let mut swapped = factors.clone();
            swapped.swap(i, i + 1);
            let reordered = graded_reorder_mul(
                swapped,
                operators,
                operator_statistics,
                properties,
                interner,
            );
            return if left_stats == OperatorStatistics::Fermionic
                && right_stats == OperatorStatistics::Fermionic
            {
                Expr::neg(reordered)
            } else {
                reordered
            };
        }
    }
    Expr::mul(factors)
}

fn normal_order_mul(
    factors: Vec<Expr>,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    for i in 0..factors.len().saturating_sub(1) {
        let left = operator_info(
            &factors[i],
            operators,
            operator_statistics,
            properties,
            interner,
        );
        let right = operator_info(
            &factors[i + 1],
            operators,
            operator_statistics,
            properties,
            interner,
        );
        if let (
            Some((OperatorKind::Annihilation, left_mode, left_stats)),
            Some((OperatorKind::Creation, right_mode, right_stats)),
        ) = (left, right)
        {
            let mut swapped = factors.clone();
            swapped.swap(i, i + 1);
            let reordered = normal_order_mul(
                swapped,
                operators,
                operator_statistics,
                properties,
                interner,
            );
            let reordered = if left_stats == OperatorStatistics::Fermionic
                && right_stats == OperatorStatistics::Fermionic
            {
                Expr::neg(reordered)
            } else {
                reordered
            };
            if modes_match(&left_mode, &right_mode, properties) && left_stats == right_stats {
                let mut remaining = factors.clone();
                remaining.remove(i + 1);
                remaining.remove(i);
                let contraction = if remaining.is_empty() {
                    Expr::one()
                } else {
                    normal_order_mul(
                        remaining,
                        operators,
                        operator_statistics,
                        properties,
                        interner,
                    )
                };
                return simplify_expr(Expr::add(vec![reordered, contraction]));
            }
            return simplify_expr(reordered);
        }
    }
    Expr::mul(factors)
}

pub fn normal_order_simple(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let simplified = factors
                .iter()
                .map(|factor| {
                    normal_order_simple(
                        factor,
                        operators,
                        operator_statistics,
                        properties,
                        interner,
                    )
                })
                .collect();
            normal_order_mul(
                simplified,
                operators,
                operator_statistics,
                properties,
                interner,
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    normal_order_simple(term, operators, operator_statistics, properties, interner)
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            normal_order_simple(base, operators, operator_statistics, properties, interner),
            normal_order_simple(exp, operators, operator_statistics, properties, interner),
        ),
        Expr::Neg(inner) => Expr::neg(normal_order_simple(
            inner,
            operators,
            operator_statistics,
            properties,
            interner,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normal_order_simple(
                re,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
            Box::new(normal_order_simple(
                im,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| {
                    normal_order_simple(arg, operators, operator_statistics, properties, interner)
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(normal_order_simple(
                body,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(normal_order_simple(
                lhs,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
            Box::new(normal_order_simple(
                rhs,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        normal_order_simple(
                            value,
                            operators,
                            operator_statistics,
                            properties,
                            interner,
                        ),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(normal_order_simple(
                base,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
            indices.clone(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(normal_order_simple(
                value,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
            Box::new(normal_order_simple(
                body,
                operators,
                operator_statistics,
                properties,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| {
                    normal_order_simple(item, operators, operator_statistics, properties, interner)
                })
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| {
                            normal_order_simple(
                                cell,
                                operators,
                                operator_statistics,
                                properties,
                                interner,
                            )
                        })
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn normal_order(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    normal_order_simple(expr, operators, operator_statistics, properties, interner)
}

fn contraction_mode_key(
    expr: &Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<(OperatorKind, lasso::Spur, OperatorStatistics)> {
    let (kind, mode, statistics) =
        operator_info(expr, operators, operator_statistics, properties, interner)?;
    match mode {
        Some(Expr::Sym(sym)) => Some((kind, sym, statistics)),
        _ => None,
    }
}

fn wick_partial_pairings(indices: &[usize]) -> Vec<Vec<(usize, usize)>> {
    fn recurse(indices: &[usize]) -> Vec<Vec<(usize, usize)>> {
        let Some((&first, rest)) = indices.split_first() else {
            return vec![Vec::new()];
        };

        let mut pairings = Vec::new();

        for mut tail in recurse(rest) {
            let _ = first;
            pairings.push(std::mem::take(&mut tail));
        }

        for partner_offset in 0..rest.len() {
            let partner = rest[partner_offset];
            let mut remaining = Vec::with_capacity(rest.len().saturating_sub(1));
            remaining.extend_from_slice(&rest[..partner_offset]);
            remaining.extend_from_slice(&rest[(partner_offset + 1)..]);
            for mut tail in recurse(&remaining) {
                let mut pairs = Vec::with_capacity(tail.len() + 1);
                pairs.push((first, partner));
                pairs.append(&mut tail);
                pairings.push(pairs);
            }
        }

        pairings
    }

    recurse(indices)
}

/// Generate every complete pairing of an even-length ordered operator list.
///
/// Each returned pairing preserves the input ordering inside each pair, so every tuple is
/// `(i, j)` with `i < j`.
pub fn wick_pairings(indices: &[usize]) -> Vec<Vec<(usize, usize)>> {
    if indices.len() % 2 == 1 {
        return Vec::new();
    }
    if indices.is_empty() {
        return vec![Vec::new()];
    }

    let first = indices[0];
    let mut pairings = Vec::new();
    for partner_offset in 1..indices.len() {
        let partner = indices[partner_offset];
        let mut remaining = Vec::with_capacity(indices.len() - 2);
        remaining.extend_from_slice(&indices[1..partner_offset]);
        remaining.extend_from_slice(&indices[(partner_offset + 1)..]);
        for mut tail in wick_pairings(&remaining) {
            let mut pairs = Vec::with_capacity(tail.len() + 1);
            pairs.push((first, partner));
            pairs.append(&mut tail);
            pairings.push(pairs);
        }
    }
    pairings
}

/// Return the fermionic sign for a chosen set of Wick contractions on an ordered operator list.
///
/// The sign is the parity of the permutation needed to bring each contracted pair adjacent while
/// preserving pair order and the relative order of any unpaired operators. Equivalently, for
/// complete pairings this matches the parity of line crossings in the standard pairing diagram.
pub fn fermionic_pairing_sign(pairs: &[(usize, usize)], operator_count: usize) -> Expr {
    if pairs.is_empty() || operator_count < 2 {
        return Expr::one();
    }

    let mut pair_by_left = pairs.to_vec();
    pair_by_left.sort_unstable_by_key(|&(left, _)| left);

    let mut right_by_left = HashMap::with_capacity(pair_by_left.len());
    let mut paired = HashSet::with_capacity(pair_by_left.len() * 2);
    for &(left, right) in &pair_by_left {
        if left >= right || right >= operator_count {
            return Expr::one();
        }
        right_by_left.insert(left, right);
        paired.insert(left);
        paired.insert(right);
    }

    let mut target = Vec::with_capacity(operator_count);
    for index in 0..operator_count {
        if let Some(&right) = right_by_left.get(&index) {
            target.push(index);
            target.push(right);
        } else if !paired.contains(&index) {
            target.push(index);
        }
    }

    let mut positions = vec![0usize; operator_count];
    for (target_index, original_index) in target.iter().copied().enumerate() {
        positions[original_index] = target_index;
    }

    let mut inversions = 0usize;
    for i in 0..operator_count {
        for j in (i + 1)..operator_count {
            if positions[i] > positions[j] {
                inversions += 1;
            }
        }
    }

    if inversions % 2 == 0 {
        Expr::one()
    } else {
        Expr::neg(Expr::one())
    }
}

fn is_pure_fermionic_operator_product(
    factors: &[Expr],
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> bool {
    !factors.is_empty()
        && factors.iter().all(|factor| {
            matches!(
                contraction_mode_key(factor, operators, operator_statistics, properties, interner,),
                Some((_, _, OperatorStatistics::Fermionic))
            )
        })
}

fn wick_expand_fermionic_mul(
    factors: &[ax_ir::Expr],
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut operator_modes = Vec::with_capacity(factors.len());
    for factor in factors {
        let Some((_, mode, _)) =
            contraction_mode_key(factor, operators, operator_statistics, properties, interner)
        else {
            return wick_expand_single(
                factors,
                operators,
                operator_statistics,
                properties,
                contractions,
                interner,
            );
        };
        operator_modes.push(mode);
    }

    let graded = graded_reorder_mul(
        factors.to_vec(),
        operators,
        operator_statistics,
        properties,
        interner,
    );
    let mut terms = vec![graded.clone()];
    let mut found_contraction = false;

    for pairs in wick_partial_pairings(&(0..factors.len()).collect::<Vec<_>>()) {
        if pairs.is_empty() {
            continue;
        }

        let mut contraction_factors = Vec::with_capacity(pairs.len() + 1);
        contraction_factors.push(fermionic_pairing_sign(&pairs, factors.len()));

        let mut valid = true;
        let mut paired_positions = HashSet::with_capacity(pairs.len() * 2);
        for &(left, right) in &pairs {
            paired_positions.insert(left);
            paired_positions.insert(right);
            let lhs = operator_modes[left];
            let rhs = operator_modes[right];
            let Some(contraction) = contractions.get(&(lhs, rhs)) else {
                valid = false;
                break;
            };
            contraction_factors.push(contraction.clone());
        }
        if !valid {
            continue;
        }
        found_contraction = true;

        let remaining = factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| {
                if paired_positions.contains(&index) {
                    None
                } else {
                    Some(factor.clone())
                }
            })
            .collect::<Vec<_>>();
        let remainder = if remaining.is_empty() {
            Expr::one()
        } else {
            graded_reorder_mul(
                remaining,
                operators,
                operator_statistics,
                properties,
                interner,
            )
        };

        contraction_factors.push(remainder);
        terms.push(Expr::mul(contraction_factors));
    }

    if !found_contraction {
        return normal_order_simple(
            &Expr::mul(factors.to_vec()),
            operators,
            operator_statistics,
            properties,
            interner,
        );
    }

    simplify_expr(Expr::add(terms))
}

pub fn wick_expand_single(
    factors: &[ax_ir::Expr],
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    if is_pure_fermionic_operator_product(
        factors,
        operators,
        operator_statistics,
        properties,
        interner,
    ) {
        return wick_expand_fermionic_mul(
            factors,
            operators,
            operator_statistics,
            properties,
            contractions,
            interner,
        );
    }

    let mut terms = Vec::new();
    let mut found_contraction = false;

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let Some((_, lhs, _)) = contraction_mode_key(
                &factors[i],
                operators,
                operator_statistics,
                properties,
                interner,
            ) else {
                continue;
            };
            let Some((_, rhs, _)) = contraction_mode_key(
                &factors[j],
                operators,
                operator_statistics,
                properties,
                interner,
            ) else {
                continue;
            };
            if let Some(contraction) = contractions.get(&(lhs, rhs)) {
                found_contraction = true;
                let remaining = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, factor)| {
                        if idx == i || idx == j {
                            None
                        } else {
                            Some(factor.clone())
                        }
                    })
                    .collect::<Vec<_>>();
                let ordered_remaining = if remaining.is_empty() {
                    Expr::one()
                } else {
                    wick_expand(
                        &Expr::mul(remaining),
                        operators,
                        operator_statistics,
                        properties,
                        contractions,
                        interner,
                    )
                };
                let signed_contraction =
                    simplify_expr(Expr::mul(vec![Expr::one(), contraction.clone()]));
                terms.push(Expr::mul(vec![signed_contraction, ordered_remaining]));
            }
        }
    }

    if !found_contraction {
        return normal_order_simple(
            &Expr::mul(factors.to_vec()),
            operators,
            operator_statistics,
            properties,
            interner,
        );
    }

    terms.insert(
        0,
        graded_reorder_mul(
            factors.to_vec(),
            operators,
            operator_statistics,
            properties,
            interner,
        ),
    );

    simplify_expr(Expr::add(terms))
}

pub fn wick_expand(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    properties: &dyn PropertyLookup,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => wick_expand_single(
            factors,
            operators,
            operator_statistics,
            properties,
            contractions,
            interner,
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    wick_expand(
                        term,
                        operators,
                        operator_statistics,
                        properties,
                        contractions,
                        interner,
                    )
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            wick_expand(
                base,
                operators,
                operator_statistics,
                properties,
                contractions,
                interner,
            ),
            wick_expand(
                exp,
                operators,
                operator_statistics,
                properties,
                contractions,
                interner,
            ),
        ),
        Expr::Neg(inner) => Expr::neg(wick_expand(
            inner,
            operators,
            operator_statistics,
            properties,
            contractions,
            interner,
        )),
        _ => normal_order_simple(expr, operators, operator_statistics, properties, interner),
    }
}

fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let mut grouped: Vec<(Expr, usize)> = Vec::new();
            for term in terms.into_iter().map(simplify_expr) {
                if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == term)
                {
                    *count += 1;
                } else {
                    grouped.push((term, 1));
                }
            }
            Expr::add(
                grouped
                    .into_iter()
                    .map(|(term, count)| {
                        if count == 1 {
                            term
                        } else {
                            simplify_expr(Expr::mul(vec![Expr::Int((count as i64).into()), term]))
                        }
                    })
                    .collect(),
            )
        }
        Expr::Mul(factors) => {
            let mut scalar_re = BigRational::one();
            let mut scalar_im = BigRational::zero();
            let mut simplified_factors = Vec::new();

            for factor in factors.into_iter().map(simplify_expr) {
                if let Some((re, im)) = exact_numeric_expr(&factor) {
                    let next_re = scalar_re.clone() * re.clone() - scalar_im.clone() * im.clone();
                    let next_im = scalar_re * im + scalar_im * re;
                    scalar_re = next_re;
                    scalar_im = next_im;
                } else {
                    simplified_factors.push(factor);
                }
            }

            let mut collapsed_factors = Vec::new();
            let mut used = vec![false; simplified_factors.len()];
            for idx in 0..simplified_factors.len() {
                if used[idx] {
                    continue;
                }

                let factor = &simplified_factors[idx];
                if let Expr::Call(_, args) = factor {
                    if args.len() == 1 {
                        if let Some(next_idx) =
                            ((idx + 1)..simplified_factors.len()).find(|candidate| {
                                !used[*candidate] && simplified_factors[*candidate] == *factor
                            })
                        {
                            used[idx] = true;
                            used[next_idx] = true;
                            collapsed_factors.push(args[0].clone());
                            continue;
                        }
                    }
                }

                used[idx] = true;
                collapsed_factors.push(factor.clone());
            }

            if scalar_re.is_zero() && scalar_im.is_zero() {
                Expr::zero()
            } else {
                let scalar_expr = if scalar_re.is_one() && scalar_im.is_zero() {
                    None
                } else {
                    Some(expr_from_exact_complex(scalar_re, scalar_im))
                };

                let mut factors = Vec::new();
                if let Some(scalar_expr) = scalar_expr {
                    factors.push(scalar_expr);
                }
                factors.extend(collapsed_factors);

                match factors.len() {
                    0 => Expr::one(),
                    1 => factors.into_iter().next().unwrap_or_else(Expr::one),
                    _ => Expr::mul(factors),
                }
            }
        }
        Expr::Pow(base, exp) => {
            let base = simplify_expr(*base);
            let exp = simplify_expr(*exp);
            match (&base, &exp) {
                // Keep `f(x)^2` consistent with the unary-call collapse already used in `Mul`.
                (Expr::Call(_, args), Expr::Int(power))
                    if args.len() == 1 && power == &2.into() =>
                {
                    args[0].clone()
                }
                _ => Expr::pow(base, exp),
            }
        }
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner)),
        other => other,
    }
}

fn simplify_matrix(matrix: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    matrix
        .into_iter()
        .map(|row| row.into_iter().map(simplify_expr).collect())
        .collect()
}

fn zero_matrix(dim: usize) -> Vec<Vec<Expr>> {
    vec![vec![Expr::zero(); dim]; dim]
}

fn matrix_shape(matrix: &[Vec<Expr>]) -> Option<(usize, usize)> {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    matrix
        .iter()
        .all(|row| row.len() == cols)
        .then_some((rows, cols))
}

fn adjoint_matrix(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    ax_linalg::transpose(matrix)
        .into_iter()
        .map(|row| row.into_iter().map(|cell| conjugate_expr(&cell)).collect())
        .collect()
}

fn diagonal_entries(matrix: &[Vec<Expr>]) -> Option<Vec<Expr>> {
    let dim = matrix_shape(matrix)?;
    if dim.0 != dim.1 {
        return None;
    }
    if matrix.iter().enumerate().any(|(row_idx, row)| {
        row.iter()
            .enumerate()
            .any(|(col_idx, cell)| row_idx != col_idx && *cell != Expr::zero())
    }) {
        return None;
    }
    Some(
        matrix
            .iter()
            .enumerate()
            .map(|(idx, row)| row[idx].clone())
            .collect(),
    )
}

fn basis_projector_matrix(dim: usize, index: usize) -> Vec<Vec<Expr>> {
    let mut projector = vec![vec![Expr::zero(); dim]; dim];
    if index < dim {
        projector[index][index] = Expr::one();
    }
    projector
}

fn trace_of_matrix_product(
    left: &[Vec<Expr>],
    right: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Expr {
    simplify_expr(ax_linalg::trace(&ax_linalg::mat_mul(left, right, interner)))
}

fn matrix_subtract(left: &[Vec<Expr>], right: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    simplify_matrix(ax_linalg::mat_add(
        left,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), right),
    ))
}

fn scalar_inverse(expr: Expr) -> Expr {
    Expr::pow(expr, Expr::Int((-1).into()))
}

fn exact_sqrt_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("sqrt"), vec![expr])
}

fn simplified_sqrt_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let expr = simplify_expr(expr);
    if expr == Expr::zero() || expr == Expr::one() {
        expr
    } else {
        exact_sqrt_expr(expr, interner)
    }
}

fn numeric_scalar_expr_exact(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(value) => Some(BigRational::from_integer(value.clone())),
        Expr::Rational(value) => Some(value.clone()),
        _ => None,
    }
}

fn exact_numeric_expr(expr: &Expr) -> Option<(BigRational, BigRational)> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) => {
            Some((numeric_scalar_expr_exact(expr)?, BigRational::zero()))
        }
        Expr::Complex(re, im) => Some((
            numeric_scalar_expr_exact(re)?,
            numeric_scalar_expr_exact(im)?,
        )),
        _ => None,
    }
}

fn expr_from_exact_rational(value: BigRational) -> Expr {
    if value.denom() == &BigInt::one() {
        Expr::Int(value.numer().clone())
    } else {
        Expr::Rational(value)
    }
}

fn expr_from_exact_complex(re: BigRational, im: BigRational) -> Expr {
    if im.is_zero() {
        expr_from_exact_rational(re)
    } else {
        Expr::Complex(
            Box::new(expr_from_exact_rational(re)),
            Box::new(expr_from_exact_rational(im)),
        )
    }
}

fn linear_term_coeff(expr: Expr) -> (BigRational, Option<Expr>) {
    match expr {
        Expr::Int(value) => (BigRational::from_integer(value), None),
        Expr::Rational(value) => (value, None),
        Expr::Neg(inner) => {
            let (coeff, basis) = linear_term_coeff(*inner);
            (-coeff, basis)
        }
        Expr::Mul(factors) => {
            let mut coeff = BigRational::one();
            let mut basis = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Int(value) => coeff *= BigRational::from_integer(value),
                    Expr::Rational(value) => coeff *= value,
                    other => basis.push(other),
                }
            }
            let basis_expr = match basis.len() {
                0 => None,
                1 => basis.into_iter().next(),
                _ => Some(Expr::mul(basis)),
            };
            (coeff, basis_expr)
        }
        other => (BigRational::one(), Some(other)),
    }
}

fn expr_is_structurally_zero(expr: &Expr) -> bool {
    let expr = simplify_expr(expr.clone());
    if expr == Expr::zero() {
        return true;
    }

    let Expr::Add(terms) = expr else {
        return false;
    };

    let mut constant = BigRational::zero();
    let mut grouped: Vec<(Expr, BigRational)> = Vec::new();

    for term in terms {
        let (coeff, basis) = linear_term_coeff(term);
        if coeff.is_zero() {
            continue;
        }
        if let Some(basis) = basis {
            if let Some((_, existing_coeff)) = grouped
                .iter_mut()
                .find(|(existing_basis, _)| *existing_basis == basis)
            {
                *existing_coeff += coeff;
            } else {
                grouped.push((basis, coeff));
            }
        } else {
            constant += coeff;
        }
    }

    constant.is_zero() && grouped.into_iter().all(|(_, coeff)| coeff.is_zero())
}

fn prune_zero_kraus_ops(kraus: Vec<Vec<Vec<Expr>>>) -> Vec<Vec<Vec<Expr>>> {
    kraus
        .into_iter()
        .filter(|operator| {
            operator
                .iter()
                .flat_map(|row| row.iter())
                .any(|entry| entry != &Expr::zero())
        })
        .collect()
}

fn simplify_visible_scalar_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Call(sym, args) if args.len() == 1 => {
            let arg = simplify_visible_scalar_expr(args[0].clone());
            if arg == Expr::one() {
                Expr::one()
            } else {
                Expr::Call(sym, vec![arg])
            }
        }
        Expr::Mul(factors) => simplify_expr(Expr::mul(
            factors
                .into_iter()
                .map(simplify_visible_scalar_expr)
                .collect::<Vec<_>>(),
        )),
        Expr::Neg(inner) => Expr::neg(simplify_visible_scalar_expr(*inner)),
        other => other,
    }
}

fn explicit_negative_magnitude(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Int(value) if value < &BigInt::zero() => Some(Expr::Int(-value.clone())),
        Expr::Rational(value) if value < &BigRational::zero() => {
            Some(Expr::Rational(-value.clone()))
        }
        Expr::Neg(inner) => Some(simplify_visible_scalar_expr((**inner).clone())),
        Expr::Mul(factors) => {
            let Some((first, rest)) = factors.split_first() else {
                return None;
            };
            match first {
                Expr::Int(value) if value < &BigInt::zero() => {
                    Some(simplify_visible_scalar_expr(Expr::mul(
                        std::iter::once(Expr::Int(-value.clone()))
                            .chain(rest.iter().cloned())
                            .collect::<Vec<_>>(),
                    )))
                }
                Expr::Rational(value) if value < &BigRational::zero() => {
                    Some(simplify_visible_scalar_expr(Expr::mul(
                        std::iter::once(Expr::Rational(-value.clone()))
                            .chain(rest.iter().cloned())
                            .collect::<Vec<_>>(),
                    )))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn induced_block_indices(matrix: &[Vec<Expr>]) -> Vec<Vec<usize>> {
    let dim = matrix.len();
    let mut visited = vec![false; dim];
    let mut blocks = Vec::new();

    for start in 0..dim {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut block = Vec::new();
        visited[start] = true;

        while let Some(node) = stack.pop() {
            block.push(node);
            for next in 0..dim {
                if visited[next] {
                    continue;
                }
                let connected = if node == next {
                    !is_zero_expr(&matrix[node][node])
                } else {
                    !is_zero_expr(&matrix[node][next]) || !is_zero_expr(&matrix[next][node])
                };
                if connected {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        block.sort_unstable();
        blocks.push(block);
    }

    blocks
}

fn submatrix_from_indices(matrix: &[Vec<Expr>], indices: &[usize]) -> Vec<Vec<Expr>> {
    indices
        .iter()
        .map(|&row| {
            indices
                .iter()
                .map(|&col| matrix[row][col].clone())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn spectral_eigenvalues_supported_blocks(
    matrix: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, SpectralError> {
    let dim = is_square_matrix(matrix)?;
    if !matrix_is_exactly_hermitian(matrix) {
        return Err(SpectralError::MatrixNotHermitian);
    }
    if dim <= 3 {
        return hermitian_eigenvalues_small(matrix, interner);
    }

    let mut eigenvalues = Vec::new();
    for block in induced_block_indices(matrix) {
        match block.len() {
            0 => {}
            1 => eigenvalues.push(matrix[block[0]][block[0]].clone()),
            2 | 3 => {
                let submatrix = submatrix_from_indices(matrix, &block);
                eigenvalues.extend(hermitian_eigenvalues_small(&submatrix, interner)?);
            }
            size => return Err(SpectralError::UnsupportedDimension { dim: size }),
        }
    }
    Ok(eigenvalues)
}

/// Construct an exact symbolic natural logarithm expression.
pub fn expr_log(arg: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("log"), vec![arg])
}

/// Construct an exact symbolic exponential expression `exp(arg)`.
pub fn scalar_exp_expr(arg: Expr, interner: &ax_ir::Interner) -> Expr {
    if arg == Expr::zero() {
        Expr::one()
    } else {
        Expr::Call(interner.get_or_intern("exp"), vec![arg])
    }
}

/// Construct the exact entropy contribution `-λ log(λ)` with `0` handled exactly.
pub fn entropy_term(lambda: &Expr, interner: &ax_ir::Interner) -> Expr {
    if *lambda == Expr::zero() || *lambda == Expr::one() {
        Expr::zero()
    } else {
        simplify_expr(Expr::neg(Expr::mul(vec![
            lambda.clone(),
            expr_log(lambda.clone(), interner),
        ])))
    }
}

/// Validate that a matrix slice is square and return its common dimension.
pub fn is_square_matrix(mat: &[Vec<Expr>]) -> Result<usize, SpectralError> {
    let rows = mat.len();
    let cols = mat.first().map(|row| row.len()).unwrap_or(0);
    let Some((actual_rows, actual_cols)) = matrix_shape(mat) else {
        return Err(SpectralError::MatrixNotSquare { rows, cols });
    };
    if actual_rows != actual_cols {
        return Err(SpectralError::MatrixNotSquare {
            rows: actual_rows,
            cols: actual_cols,
        });
    }
    Ok(actual_rows)
}

/// Return the exact conjugate transpose of a symbolic matrix.
pub fn matrix_conjugate_transpose(mat: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    ax_linalg::transpose(mat)
        .into_iter()
        .map(|row| row.into_iter().map(|cell| conjugate_expr(&cell)).collect())
        .collect()
}

/// Check Hermiticity by exact structural equality against the conjugate transpose.
pub fn matrix_is_exactly_hermitian(mat: &[Vec<Expr>]) -> bool {
    mat == matrix_conjugate_transpose(mat)
}

/// Construct the exact identity matrix of the requested finite dimension.
pub fn identity_matrix(dim: usize, _interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    ax_linalg::identity(dim)
}

/// Return exact small-dimensional Hermitian eigenvalues for supported matrix classes.
pub fn hermitian_eigenvalues_small(
    mat: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, SpectralError> {
    let dim = is_square_matrix(mat)?;
    if !matrix_is_exactly_hermitian(mat) {
        return Err(SpectralError::MatrixNotHermitian);
    }

    match dim {
        2 => {
            if let Some(diagonal) = diagonal_entries(mat) {
                return Ok(diagonal);
            }

            let a = mat[0][0].clone();
            let b = mat[0][1].clone();
            let d = mat[1][1].clone();
            let trace = Expr::add(vec![a.clone(), d.clone()]);
            let diff = Expr::add(vec![a.clone(), Expr::neg(d.clone())]);
            let b_norm_sq = Expr::mul(vec![b.clone(), conjugate_expr(&b)]);
            let discriminant = Expr::add(vec![
                Expr::pow(diff, Expr::Int(2.into())),
                Expr::mul(vec![Expr::Int(4.into()), b_norm_sq]),
            ]);
            let sqrt_discriminant = Expr::Call(interner.get_or_intern("sqrt"), vec![discriminant]);
            let half_trace = Expr::mul(vec![half(), trace]);
            let half_gap = Expr::mul(vec![half(), sqrt_discriminant]);
            Ok(vec![
                simplify_expr(Expr::add(vec![half_trace.clone(), half_gap.clone()])),
                simplify_expr(Expr::add(vec![half_trace, Expr::neg(half_gap)])),
            ])
        }
        3 => diagonal_entries(mat).ok_or(SpectralError::UnsupportedDimension { dim: 3 }),
        _ => Err(SpectralError::UnsupportedDimension { dim }),
    }
}

/// Return exact spectral projectors for supported small Hermitian matrices.
pub fn hermitian_eigenprojectors_small(
    mat: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<Expr>>>, SpectralError> {
    let dim = is_square_matrix(mat)?;
    if !matrix_is_exactly_hermitian(mat) {
        return Err(SpectralError::MatrixNotHermitian);
    }

    if let Some(diagonal) = diagonal_entries(mat) {
        if diagonal
            .iter()
            .enumerate()
            .any(|(idx, entry)| diagonal.iter().skip(idx + 1).any(|other| other == entry))
        {
            return Err(SpectralError::DegenerateSpectrumUnsupported);
        }
        return Ok((0..dim)
            .map(|idx| basis_projector_matrix(dim, idx))
            .collect());
    }

    if dim != 2 {
        return Err(SpectralError::UnsupportedDimension { dim });
    }

    let eigenvalues = hermitian_eigenvalues_small(mat, interner)?;
    if eigenvalues.len() != 2 || eigenvalues[0] == eigenvalues[1] {
        return Err(SpectralError::DegenerateSpectrumUnsupported);
    }

    let identity = identity_matrix(dim, interner);
    let projectors = (0..2)
        .map(|i| {
            let j = 1 - i;
            let numerator = matrix_subtract(mat, &ax_linalg::mat_scale(&eigenvalues[j], &identity));
            let denominator = Expr::add(vec![
                eigenvalues[i].clone(),
                Expr::neg(eigenvalues[j].clone()),
            ]);
            simplify_matrix(ax_linalg::mat_scale(
                &scalar_inverse(denominator),
                &numerator,
            ))
        })
        .collect();

    Ok(projectors)
}

fn map_square_error(matrix: &[Vec<Expr>], is_h0: bool) -> Result<usize, PerturbationError> {
    is_square_matrix(matrix).map_err(|error| match error {
        SpectralError::MatrixNotSquare { rows, cols } => {
            if is_h0 {
                PerturbationError::UnperturbedHamiltonianNotSquare { rows, cols }
            } else {
                PerturbationError::PerturbationNotSquare { rows, cols }
            }
        }
        other => PerturbationError::UnsupportedSpectrum(other),
    })
}

fn distinct_unperturbed_eigenvalues(eigenvalues: &[Expr]) -> Result<(), PerturbationError> {
    if eigenvalues
        .iter()
        .enumerate()
        .any(|(idx, value)| eigenvalues.iter().skip(idx + 1).any(|other| other == value))
    {
        return Err(PerturbationError::DegenerateUnperturbedSpectrum);
    }
    Ok(())
}

fn supported_unperturbed_eigenbasis(
    h0: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<(Vec<Expr>, Option<Vec<Vec<Vec<Expr>>>>), PerturbationError> {
    if let Some(diagonal) = diagonal_entries(h0) {
        distinct_unperturbed_eigenvalues(&diagonal)?;
        return Ok((diagonal, None));
    }

    let dim = h0.len();
    if dim != 2 {
        return Err(PerturbationError::UnsupportedSpectrum(
            SpectralError::UnsupportedDimension { dim },
        ));
    }

    let eigenvalues = hermitian_eigenvalues_small(h0, interner)?;
    distinct_unperturbed_eigenvalues(&eigenvalues)?;
    let projectors = hermitian_eigenprojectors_small(h0, interner)?;
    Ok((eigenvalues, Some(projectors)))
}

fn diagonal_h0_energies(h0: &[Vec<Expr>]) -> Result<Vec<Expr>, PerturbationError> {
    let dim = map_square_error(h0, true)?;
    diagonal_entries(h0).ok_or(PerturbationError::UnsupportedSpectrum(
        SpectralError::UnsupportedDimension { dim },
    ))
}

fn validate_degenerate_subspace(
    energies: &[Expr],
    subspace: &[usize],
) -> Result<(), PerturbationError> {
    if subspace.is_empty() {
        return Err(PerturbationError::EmptyDegenerateSubspace);
    }

    for &index in subspace {
        if index >= energies.len() {
            return Err(PerturbationError::DegenerateSubspaceIndexOutOfRange {
                index,
                dim: energies.len(),
            });
        }
    }

    let reference = &energies[subspace[0]];
    if subspace
        .iter()
        .skip(1)
        .any(|&index| energies[index] != *reference)
    {
        return Err(PerturbationError::SelectedSubspaceNotDegenerate);
    }

    Ok(())
}

/// Return the exact perturbation matrix restricted to a chosen degenerate
/// subspace of a diagonal unperturbed Hamiltonian `H0`.
///
/// The `subspace` slice lists computational-basis indices spanning a
/// degenerate eigenspace of `H0`. Since this helper is restricted to diagonal
/// `H0`, the eigenbasis of `H0` is the computational basis, so the effective
/// perturbation is obtained by extracting the corresponding principal submatrix
/// of `V`.
pub fn degenerate_subspace_effective_perturbation(
    h0: &[Vec<Expr>],
    v: &[Vec<Expr>],
    subspace: &[usize],
    _interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, PerturbationError> {
    let energies = diagonal_h0_energies(h0)?;
    let dim = energies.len();
    let v_dim = map_square_error(v, false)?;
    if v_dim != dim {
        return Err(PerturbationError::DimensionMismatch {
            expected: dim,
            actual: v_dim,
        });
    }
    validate_degenerate_subspace(&energies, subspace)?;

    Ok(subspace
        .iter()
        .map(|&row| {
            subspace
                .iter()
                .map(|&col| simplify_expr(v[row][col].clone()))
                .collect::<Vec<_>>()
        })
        .collect())
}

/// Return the exact first-order level splittings inside a chosen degenerate
/// subspace by diagonalizing the effective perturbation matrix `V_eff`.
///
/// The `subspace` slice lists computational-basis indices spanning a
/// degenerate eigenspace of a diagonal `H0`. The returned eigenvalues are the
/// first-order perturbative splittings of that degenerate manifold.
pub fn degenerate_first_order_splittings(
    h0: &[Vec<Expr>],
    v: &[Vec<Expr>],
    subspace: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, PerturbationError> {
    let effective = degenerate_subspace_effective_perturbation(h0, v, subspace, interner)?;
    if effective.len() == 1 {
        return Ok(vec![simplify_expr(effective[0][0].clone())]);
    }
    if let Some(diagonal) = diagonal_entries(&effective) {
        return Ok(diagonal.into_iter().map(simplify_expr).collect());
    }
    if effective.len() == 2
        && effective[0][0] == effective[1][1]
        && effective[0][1] == effective[1][0]
        && conjugate_expr(&effective[0][1]) == effective[0][1]
    {
        let center = simplify_expr(effective[0][0].clone());
        let gap = simplify_expr(effective[0][1].clone());
        return Ok(vec![
            simplify_expr(Expr::add(vec![center.clone(), gap.clone()])),
            simplify_expr(Expr::add(vec![center, Expr::neg(gap)])),
        ]);
    }
    let splittings = hermitian_eigenvalues_small(&effective, interner)?;
    Ok(splittings.into_iter().map(simplify_expr).collect())
}

/// Return the exact first-order stationary perturbation-theory energy shift
/// `E_n^(1) = <n|V|n>` for a supported finite-dimensional unperturbed
/// Hamiltonian `H0`.
///
/// Supported `H0` classes are:
/// - diagonal, nondegenerate matrices of any finite dimension
/// - nondegenerate Hermitian `2x2` matrices supported by the exact spectral
///   helpers
///
/// The matrix element is evaluated in the eigenbasis of `H0`. Degenerate
/// unperturbed spectra are rejected explicitly.
pub fn first_order_energy_shift(
    h0: &[Vec<Expr>],
    v: &[Vec<Expr>],
    state_index: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, PerturbationError> {
    let dim = map_square_error(h0, true)?;
    let v_dim = map_square_error(v, false)?;
    if v_dim != dim {
        return Err(PerturbationError::DimensionMismatch {
            expected: dim,
            actual: v_dim,
        });
    }
    if state_index >= dim {
        return Err(PerturbationError::StateIndexOutOfRange {
            index: state_index,
            dim,
        });
    }

    let (_, projectors) = supported_unperturbed_eigenbasis(h0, interner)?;
    let shift = if let Some(projectors) = projectors {
        trace_of_matrix_product(&projectors[state_index], v, interner)
    } else {
        simplify_expr(v[state_index][state_index].clone())
    };
    Ok(shift)
}

/// Return the exact second-order stationary perturbation-theory energy shift
/// `E_n^(2) = Σ_{m != n} |V_mn|^2 / (E_n^(0) - E_m^(0))` for a supported
/// finite-dimensional unperturbed Hamiltonian `H0`.
///
/// Supported `H0` classes are:
/// - diagonal, nondegenerate matrices of any finite dimension
/// - nondegenerate Hermitian `2x2` matrices supported by the exact spectral
///   helpers
///
/// The sum is evaluated in the eigenbasis of `H0`. The off-diagonal matrix
/// element magnitude is kept exact as `V_mn * conj(V_mn)`.
pub fn second_order_energy_shift(
    h0: &[Vec<Expr>],
    v: &[Vec<Expr>],
    state_index: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, PerturbationError> {
    let dim = map_square_error(h0, true)?;
    let v_dim = map_square_error(v, false)?;
    if v_dim != dim {
        return Err(PerturbationError::DimensionMismatch {
            expected: dim,
            actual: v_dim,
        });
    }
    if state_index >= dim {
        return Err(PerturbationError::StateIndexOutOfRange {
            index: state_index,
            dim,
        });
    }

    let (eigenvalues, projectors) = supported_unperturbed_eigenbasis(h0, interner)?;
    let shift = if let Some(projectors) = projectors {
        let other = 1 - state_index;
        let v_dag = adjoint_matrix(v);
        let numerator = ax_linalg::mat_mul(
            &ax_linalg::mat_mul(&projectors[other], v, interner),
            &ax_linalg::mat_mul(&projectors[state_index], &v_dag, interner),
            interner,
        );
        let denominator = simplify_expr(Expr::add(vec![
            eigenvalues[state_index].clone(),
            Expr::neg(eigenvalues[other].clone()),
        ]));
        simplify_expr(Expr::mul(vec![
            ax_linalg::trace(&numerator),
            scalar_inverse(denominator),
        ]))
    } else {
        let mut terms = Vec::new();
        for m in 0..dim {
            if m == state_index {
                continue;
            }
            let v_mn = simplify_expr(v[m][state_index].clone());
            let numerator = simplify_expr(Expr::mul(vec![v_mn.clone(), conjugate_expr(&v_mn)]));
            let denominator = simplify_expr(Expr::add(vec![
                eigenvalues[state_index].clone(),
                Expr::neg(eigenvalues[m].clone()),
            ]));
            terms.push(simplify_expr(Expr::mul(vec![
                numerator,
                scalar_inverse(denominator),
            ])));
        }
        simplify_expr(Expr::add(terms))
    };
    Ok(shift)
}

/// Construct the symbolic Berry-connection one-form component
/// `A(parameter) = i <psi | d/dparameter | psi>`.
///
/// The derivative uses the canonical symbolic `diff(psi, parameter)`
/// representation. The resulting coefficient is wrapped in the shared
/// symbolic one-form placeholder from `ax-forms`.
pub fn berry_connection(psi: Expr, parameter: Expr, interner: &ax_ir::Interner) -> Expr {
    let bra = Expr::Call(interner.get_or_intern("bra"), vec![psi.clone()]);
    let derivative = Expr::Call(interner.get_or_intern("diff"), vec![psi, parameter.clone()]);
    let coefficient = simplify_expr(Expr::mul(vec![imag_unit(), bra, derivative]));
    ax_forms::one_form_component(parameter, coefficient, interner)
}

/// Construct the symbolic geometric phase
/// `γ = ∮ A(parameter) dparameter`.
///
/// The contour is represented symbolically by `closed_path(parameter)` while
/// reusing the canonical `integral(variable, lower, upper, integrand)` call
/// shape used elsewhere in the codebase.
pub fn geometric_phase(line_integrand: Expr, parameter: Expr, interner: &ax_ir::Interner) -> Expr {
    let closed_path = Expr::Call(
        interner.get_or_intern("closed_path"),
        vec![parameter.clone()],
    );
    Expr::Call(
        interner.get_or_intern("integral"),
        vec![parameter, closed_path.clone(), closed_path, line_integrand],
    )
}

fn map_spectral_error_to_matrix_exponential_error(error: SpectralError) -> MatrixExponentialError {
    match error {
        SpectralError::MatrixNotSquare { rows, cols } => {
            MatrixExponentialError::MatrixNotSquare { rows, cols }
        }
        SpectralError::MatrixNotHermitian => MatrixExponentialError::MatrixNotHermitian,
        SpectralError::UnsupportedDimension { dim } => {
            MatrixExponentialError::UnsupportedDimension { dim }
        }
        SpectralError::DegenerateSpectrumUnsupported => {
            MatrixExponentialError::DegenerateSpectrumUnsupported
        }
    }
}

/// Construct `exp(A)` from the exact spectral formula `Σ_i exp(λ_i) P_i`
/// for supported small Hermitian matrices.
///
/// This uses the existing exact small-dimensional Hermitian eigenspectrum and
/// eigenprojector machinery. Degenerate spectra that prevent projector
/// construction return `DegenerateSpectrumUnsupported`.
pub fn matrix_exponential_from_spectral_decomposition(
    mat: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, MatrixExponentialError> {
    let dim = is_square_matrix(mat).map_err(map_spectral_error_to_matrix_exponential_error)?;
    if !matrix_is_exactly_hermitian(mat) {
        return Err(MatrixExponentialError::MatrixNotHermitian);
    }

    if dim == 1 {
        return Ok(vec![vec![scalar_exp_expr(mat[0][0].clone(), interner)]]);
    }

    let eigenvalues = hermitian_eigenvalues_small(mat, interner)
        .map_err(map_spectral_error_to_matrix_exponential_error)?;
    let projectors = hermitian_eigenprojectors_small(mat, interner)
        .map_err(map_spectral_error_to_matrix_exponential_error)?;

    let mut result = zero_matrix(dim);
    for (eigenvalue, projector) in eigenvalues.into_iter().zip(projectors.into_iter()) {
        let weighted = ax_linalg::mat_scale(&scalar_exp_expr(eigenvalue, interner), &projector);
        result = simplify_matrix(ax_linalg::mat_add(&result, &weighted));
    }
    Ok(result)
}

/// Compute exact small Hermitian matrix exponentials for the supported cases:
/// `1x1`, diagonal `2x2`, nondegenerate Hermitian `2x2`, and diagonal `3x3`.
///
/// The implementation returns explicit symbolic matrices rather than an
/// unevaluated `exp(A)` wrapper. Non-diagonal `3x3` inputs are rejected as
/// unsupported, and degenerate non-diagonal `2x2` inputs return
/// `DegenerateSpectrumUnsupported`.
pub fn hermitian_matrix_exponential_small(
    mat: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, MatrixExponentialError> {
    let dim = is_square_matrix(mat).map_err(map_spectral_error_to_matrix_exponential_error)?;
    if !matrix_is_exactly_hermitian(mat) {
        return Err(MatrixExponentialError::MatrixNotHermitian);
    }

    if let Some(diagonal) = diagonal_entries(mat) {
        let mut result = zero_matrix(dim);
        for (idx, entry) in diagonal.into_iter().enumerate() {
            result[idx][idx] = scalar_exp_expr(entry, interner);
        }
        return Ok(result);
    }

    match dim {
        2 => matrix_exponential_from_spectral_decomposition(mat, interner),
        3 => Err(MatrixExponentialError::UnsupportedDimension { dim: 3 }),
        _ => Err(MatrixExponentialError::UnsupportedDimension { dim }),
    }
}

/// Construct the exact constant-Hamiltonian propagator `U(t) = exp(-i t H)`
/// for the same supported small-matrix classes as
/// [`hermitian_matrix_exponential_small`].
///
/// The Hamiltonian `H` must be square and exactly Hermitian. Diagonal `1x1`,
/// diagonal `2x2`, nondegenerate Hermitian `2x2`, and diagonal `3x3`
/// Hamiltonians are supported. The result is returned as an explicit symbolic
/// matrix with entries built from exact `exp(...)` expressions.
pub fn time_evolution_operator(
    h: &[Vec<Expr>],
    t: Expr,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, MatrixExponentialError> {
    let dim = is_square_matrix(h).map_err(map_spectral_error_to_matrix_exponential_error)?;
    if !matrix_is_exactly_hermitian(h) {
        return Err(MatrixExponentialError::MatrixNotHermitian);
    }

    let minus_i_t = Expr::mul(vec![Expr::neg(imag_unit()), t]);
    let _scaled = h
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| simplify_expr(Expr::mul(vec![minus_i_t.clone(), entry.clone()])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    if let Some(diagonal) = diagonal_entries(h) {
        let mut result = zero_matrix(dim);
        for (idx, entry) in diagonal.into_iter().enumerate() {
            result[idx][idx] = scalar_exp_expr(
                simplify_expr(Expr::mul(vec![minus_i_t.clone(), entry])),
                interner,
            );
        }
        return Ok(result);
    }

    match dim {
        2 => {
            let eigenvalues = hermitian_eigenvalues_small(h, interner)
                .map_err(map_spectral_error_to_matrix_exponential_error)?;
            let projectors = hermitian_eigenprojectors_small(h, interner)
                .map_err(map_spectral_error_to_matrix_exponential_error)?;

            let mut result = zero_matrix(dim);
            for (eigenvalue, projector) in eigenvalues.into_iter().zip(projectors.into_iter()) {
                let phase = scalar_exp_expr(
                    simplify_expr(Expr::mul(vec![minus_i_t.clone(), eigenvalue])),
                    interner,
                );
                let weighted = ax_linalg::mat_scale(&phase, &projector);
                result = simplify_matrix(ax_linalg::mat_add(&result, &weighted));
            }
            Ok(result)
        }
        1 => Ok(vec![vec![scalar_exp_expr(
            simplify_expr(Expr::mul(vec![minus_i_t, h[0][0].clone()])),
            interner,
        )]]),
        3 => Err(MatrixExponentialError::UnsupportedDimension { dim: 3 }),
        _ => Err(MatrixExponentialError::UnsupportedDimension { dim }),
    }
}

/// Multiply an explicit symbolic matrix by an explicit symbolic state vector.
///
/// The result is returned as a concrete component vector with each entry
/// simplified after summing the row-wise products.
pub fn matrix_vector_mul(mat: &[Vec<Expr>], vec: &[Expr]) -> Vec<Expr> {
    mat.iter()
        .map(|row| {
            simplify_expr(Expr::add(
                row.iter()
                    .zip(vec.iter())
                    .map(|(entry, value)| {
                        simplify_expr(Expr::mul(vec![entry.clone(), value.clone()]))
                    })
                    .collect(),
            ))
        })
        .collect()
}

/// Evolve a finite-dimensional pure state in the Schrödinger picture under a
/// constant Hermitian Hamiltonian using `psi(t) = U(t) psi0`.
///
/// This returns the explicit evolved state vector rather than an unevaluated
/// symbolic `U * psi0` wrapper. The Hamiltonian must be a supported square
/// Hermitian matrix, and the input state vector length must match its
/// dimension.
pub fn schrodinger_evolve_state(
    h: &[Vec<Expr>],
    psi0: &[Expr],
    t: Expr,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, StateEvolutionError> {
    let dim = is_square_matrix(h).map_err(map_spectral_error_to_matrix_exponential_error)?;
    if psi0.len() != dim {
        return Err(StateEvolutionError::StateDimensionMismatch {
            expected: dim,
            actual: psi0.len(),
        });
    }

    let evolution = time_evolution_operator(h, t, interner)?;
    Ok(matrix_vector_mul(&evolution, psi0))
}

fn combine_exponential_factors(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let exp_sym = interner.get_or_intern("exp");
    match expr {
        Expr::Add(terms) => simplify_expr(Expr::add(
            terms
                .into_iter()
                .map(|term| combine_exponential_factors(term, interner))
                .collect(),
        )),
        Expr::Mul(factors) => {
            let mut combined_exp_args = Vec::new();
            let mut other_factors = Vec::new();

            for factor in factors
                .into_iter()
                .map(|factor| combine_exponential_factors(factor, interner))
            {
                match factor {
                    Expr::Call(sym, args) if sym == exp_sym && args.len() == 1 => {
                        combined_exp_args.push(args[0].clone());
                    }
                    other => other_factors.push(other),
                }
            }

            if !combined_exp_args.is_empty() {
                let combined_arg = simplify_expr(Expr::add(combined_exp_args));
                if combined_arg != Expr::zero() {
                    other_factors.push(Expr::Call(exp_sym, vec![combined_arg]));
                }
            }

            simplify_expr(Expr::mul(other_factors))
        }
        Expr::Neg(inner) => simplify_expr(Expr::neg(combine_exponential_factors(*inner, interner))),
        other => other,
    }
}

/// Evolve a finite-dimensional operator in the Heisenberg picture under a
/// constant Hermitian Hamiltonian using `O(t) = U†(t) O0 U(t)`.
///
/// This returns the explicit evolved operator matrix rather than an
/// unevaluated symbolic `U† O0 U` wrapper. The Hamiltonian must be a
/// supported square Hermitian matrix, and the input operator must be a square
/// matrix of matching dimension.
pub fn heisenberg_evolve_operator(
    h: &[Vec<Expr>],
    op0: &[Vec<Expr>],
    t: Expr,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, OperatorEvolutionError> {
    let dim = is_square_matrix(h).map_err(map_spectral_error_to_matrix_exponential_error)?;
    let (rows, cols) =
        matrix_shape(op0).ok_or(OperatorEvolutionError::OperatorDimensionMismatch {
            expected: dim,
            actual: op0.len(),
        })?;
    if rows != dim || cols != dim {
        return Err(OperatorEvolutionError::OperatorDimensionMismatch {
            expected: dim,
            actual: rows,
        });
    }

    let evolution = time_evolution_operator(h, t, interner)?;
    let evolution_dag = matrix_conjugate_transpose(&evolution);
    let left = ax_linalg::mat_mul(&evolution_dag, op0, interner);
    Ok(
        simplify_matrix(ax_linalg::mat_mul(&left, &evolution, interner))
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| combine_exponential_factors(entry, interner))
                    .collect()
            })
            .collect(),
    )
}

/// Construct the closed-system density-matrix right-hand side
/// `ρ̇ = -i (Hρ - ρH)` for a finite-dimensional Hamiltonian and density matrix.
///
/// Both `H` and `ρ` must be square matrices of the same dimension. The result
/// is returned as an explicit symbolic matrix with elementwise simplification
/// applied after the commutator and overall `-i` scaling.
pub fn liouville_von_neumann_rhs(
    h: &[Vec<Expr>],
    rho: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, LiouvilleError> {
    let (h_rows, h_cols) = matrix_shape(h).unwrap_or((h.len(), h.first().map_or(0, Vec::len)));
    if h_rows != h_cols {
        return Err(LiouvilleError::HamiltonianNotSquare {
            rows: h_rows,
            cols: h_cols,
        });
    }

    let (rho_rows, rho_cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map_or(0, Vec::len)));
    if rho_rows != rho_cols {
        return Err(LiouvilleError::StateNotSquare {
            rows: rho_rows,
            cols: rho_cols,
        });
    }

    if rho_rows != h_rows {
        return Err(LiouvilleError::DimensionMismatch {
            expected: h_rows,
            actual: rho_rows,
        });
    }

    Ok(simplify_matrix(ax_linalg::mat_scale(
        &Expr::neg(imag_unit()),
        &commutator(h, rho, interner),
    )))
}

fn is_zero_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero()) || matches!(expr, Expr::Rational(r) if r.is_zero())
}

const NUMERIC_CP_TOLERANCE: f64 = 1.0e-12;

fn numeric_scalar_expr(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(value) => num_traits::ToPrimitive::to_f64(value),
        Expr::Rational(value) => {
            let numer = num_traits::ToPrimitive::to_f64(value.numer())?;
            let denom = num_traits::ToPrimitive::to_f64(value.denom())?;
            Some(numer / denom)
        }
        Expr::Float(value) => Some(*value),
        _ => None,
    }
}

/// Convert a numeric scalar expression into an explicit complex `(re, im)` pair.
///
/// Accepted forms are integers, rationals, floats, and explicit complex numbers
/// whose real and imaginary parts are themselves numeric scalar expressions.
pub fn numeric_complex_from_expr(expr: &Expr) -> Result<(f64, f64), LindbladianSpectrumError> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => numeric_scalar_expr(expr)
            .map(|value| (value, 0.0))
            .ok_or(LindbladianSpectrumError::NonNumericOperator),
        Expr::Complex(re, im) => Ok((
            numeric_scalar_expr(re).ok_or(LindbladianSpectrumError::NonNumericOperator)?,
            numeric_scalar_expr(im).ok_or(LindbladianSpectrumError::NonNumericOperator)?,
        )),
        _ => Err(LindbladianSpectrumError::NonNumericOperator),
    }
}

/// Convert a dense exact expression matrix into numeric complex entries.
///
/// Every element must be representable by [`numeric_complex_from_expr`].
pub fn dense_numeric_matrix_from_expr(
    mat: &[Vec<Expr>],
) -> Result<Vec<Vec<(f64, f64)>>, LindbladianSpectrumError> {
    mat.iter()
        .map(|row| {
            row.iter()
                .map(numeric_complex_from_expr)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn numeric_complex_expr(expr: &Expr) -> Result<(f64, f64), ChannelError> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => numeric_scalar_expr(expr)
            .map(|value| (value, 0.0))
            .ok_or(ChannelError::NonNumericChoiMatrix),
        Expr::Complex(re, im) => Ok((
            numeric_scalar_expr(re).ok_or(ChannelError::NonNumericChoiMatrix)?,
            numeric_scalar_expr(im).ok_or(ChannelError::NonNumericChoiMatrix)?,
        )),
        _ => Err(ChannelError::NonNumericChoiMatrix),
    }
}

fn numeric_choi_matrix(choi: &[Vec<Expr>]) -> Result<Vec<Vec<(f64, f64)>>, ChannelError> {
    choi.iter()
        .map(|row| {
            row.iter()
                .map(numeric_complex_expr)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn numeric_complex_is_zero((re, im): (f64, f64)) -> bool {
    re.abs() <= NUMERIC_CP_TOLERANCE && im.abs() <= NUMERIC_CP_TOLERANCE
}

fn numeric_complex_conjugate((re, im): (f64, f64)) -> (f64, f64) {
    (re, -im)
}

fn numeric_complex_close(left: (f64, f64), right: (f64, f64)) -> bool {
    (left.0 - right.0).abs() <= NUMERIC_CP_TOLERANCE
        && (left.1 - right.1).abs() <= NUMERIC_CP_TOLERANCE
}

fn numeric_matrix_is_hermitian(matrix: &[Vec<(f64, f64)>]) -> bool {
    for row in 0..matrix.len() {
        if matrix[row][row].1.abs() > NUMERIC_CP_TOLERANCE {
            return false;
        }
        for col in (row + 1)..matrix.len() {
            if !numeric_complex_close(
                matrix[row][col],
                numeric_complex_conjugate(matrix[col][row]),
            ) {
                return false;
            }
        }
    }
    true
}

fn numeric_induced_block_indices(matrix: &[Vec<(f64, f64)>]) -> Vec<Vec<usize>> {
    let dim = matrix.len();
    let mut visited = vec![false; dim];
    let mut blocks = Vec::new();

    for start in 0..dim {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut block = Vec::new();
        visited[start] = true;

        while let Some(node) = stack.pop() {
            block.push(node);
            for next in 0..dim {
                if visited[next] {
                    continue;
                }
                let connected = if node == next {
                    !numeric_complex_is_zero(matrix[node][node])
                } else {
                    !numeric_complex_is_zero(matrix[node][next])
                        || !numeric_complex_is_zero(matrix[next][node])
                };
                if connected {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        block.sort_unstable();
        blocks.push(block);
    }

    blocks
}

fn numeric_hermitian_2x2_eigenvalues(block: &[Vec<(f64, f64)>]) -> [f64; 2] {
    let a = block[0][0].0;
    let d = block[1][1].0;
    let b = block[0][1];
    let b_norm_sq = b.0 * b.0 + b.1 * b.1;
    let discriminant = ((a - d) * (a - d) + 4.0 * b_norm_sq).sqrt();
    [0.5 * (a + d + discriminant), 0.5 * (a + d - discriminant)]
}

/// Check complete positivity from a low-dimensional numeric Choi matrix.
///
/// Supported cases are:
/// - `1x1`
/// - Hermitian `2x2`
/// - Hermitian `4x4` matrices that reduce to supported `1x1` and `2x2` blocks by exact zero pattern
///
/// Symbolic Choi entries are rejected with `NonNumericChoiMatrix`, and larger or
/// otherwise unsupported matrices return `UnsupportedCompletePositivityCheck`.
pub fn is_completely_positive_choi_small(
    choi: &[Vec<Expr>],
    _interner: &ax_ir::Interner,
) -> Result<bool, ChannelError> {
    let Some((rows, cols)) = matrix_shape(choi) else {
        return Err(ChannelError::UnsupportedCompletePositivityCheck { dim: choi.len() });
    };
    if rows != cols {
        return Err(ChannelError::UnsupportedCompletePositivityCheck { dim: rows });
    }

    let numeric = numeric_choi_matrix(choi)?;
    if !numeric_matrix_is_hermitian(&numeric) {
        return Ok(false);
    }

    match rows {
        1 => Ok(numeric[0][0].0 >= -NUMERIC_CP_TOLERANCE),
        2 => Ok(numeric_hermitian_2x2_eigenvalues(&numeric)
            .into_iter()
            .all(|eigenvalue| eigenvalue >= -NUMERIC_CP_TOLERANCE)),
        4 => {
            let blocks = numeric_induced_block_indices(&numeric);
            if blocks.iter().any(|block| block.len() > 2) {
                return Err(ChannelError::UnsupportedCompletePositivityCheck { dim: 4 });
            }

            for block in blocks {
                match block.len() {
                    0 => {}
                    1 => {
                        if numeric[block[0]][block[0]].0 < -NUMERIC_CP_TOLERANCE {
                            return Ok(false);
                        }
                    }
                    2 => {
                        let submatrix = block
                            .iter()
                            .map(|&row| {
                                block
                                    .iter()
                                    .map(|&col| numeric[row][col])
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        if !numeric_hermitian_2x2_eigenvalues(&submatrix)
                            .into_iter()
                            .all(|eigenvalue| eigenvalue >= -NUMERIC_CP_TOLERANCE)
                        {
                            return Ok(false);
                        }
                    }
                    _ => return Err(ChannelError::UnsupportedCompletePositivityCheck { dim: 4 }),
                }
            }

            Ok(true)
        }
        dim => Err(ChannelError::UnsupportedCompletePositivityCheck { dim }),
    }
}

fn validate_kraus_set(kraus: &[Vec<Vec<Expr>>]) -> Result<usize, ChannelError> {
    if kraus.is_empty() {
        return Err(ChannelError::EmptyKrausSet);
    }

    let mut expected_dim = None;
    for (index, operator) in kraus.iter().enumerate() {
        let rows = operator.len();
        let cols = operator.first().map(|row| row.len()).unwrap_or(0);
        let Some((actual_rows, actual_cols)) = matrix_shape(operator) else {
            return Err(ChannelError::NonSquareKraus { index, rows, cols });
        };
        if actual_rows != actual_cols {
            return Err(ChannelError::NonSquareKraus {
                index,
                rows: actual_rows,
                cols: actual_cols,
            });
        }

        if let Some(expected) = expected_dim {
            if actual_rows != expected {
                return Err(ChannelError::KrausDimensionMismatch {
                    expected,
                    actual: actual_rows,
                    index,
                });
            }
        } else {
            expected_dim = Some(actual_rows);
        }
    }

    expected_dim.ok_or(ChannelError::InvalidKrausSet)
}

fn validate_square_state_dimension(
    matrix: &[Vec<Expr>],
    expected: usize,
) -> Result<(), MeasurementError> {
    let (rows, cols) = matrix_shape(matrix).unwrap_or((
        matrix.len(),
        matrix.first().map(|row| row.len()).unwrap_or(0),
    ));
    if rows != cols || rows != expected {
        return Err(MeasurementError::StateDimensionMismatch {
            expected,
            actual: rows,
        });
    }
    Ok(())
}

fn validate_projector_set(
    projectors: &[Vec<Vec<Expr>>],
    expected: usize,
) -> Result<(), MeasurementError> {
    for (index, projector) in projectors.iter().enumerate() {
        let (rows, cols) = matrix_shape(projector).unwrap_or((
            projector.len(),
            projector.first().map(|row| row.len()).unwrap_or(0),
        ));
        if rows != cols || rows != expected {
            return Err(MeasurementError::ProjectorDimensionMismatch {
                expected,
                actual: rows,
                index,
            });
        }
    }
    Ok(())
}

fn validate_lindblad_square_matrix(
    matrix: &[Vec<Expr>],
    which: &'static str,
) -> Result<usize, LindbladError> {
    let (rows, cols) = matrix_shape(matrix).unwrap_or((
        matrix.len(),
        matrix.first().map(|row| row.len()).unwrap_or(0),
    ));
    if rows != cols {
        return Err(match which {
            "Hamiltonian" => LindbladError::HamiltonianNotSquare { rows, cols },
            "state" => LindbladError::StateNotSquare { rows, cols },
            _ => LindbladError::DimensionMismatch {
                expected: rows,
                actual: cols,
                which,
            },
        });
    }
    Ok(rows)
}

fn validate_lindblad_jump_ops(
    jump_ops: &[Vec<Vec<Expr>>],
    expected: usize,
) -> Result<(), LindbladError> {
    for operator in jump_ops {
        let (rows, cols) = matrix_shape(operator).unwrap_or((
            operator.len(),
            operator.first().map(|row| row.len()).unwrap_or(0),
        ));
        if rows != cols {
            return Err(LindbladError::DimensionMismatch {
                expected,
                actual: rows,
                which: "jump operator",
            });
        }
        if rows != expected {
            return Err(LindbladError::DimensionMismatch {
                expected,
                actual: rows,
                which: "jump operator",
            });
        }
    }
    Ok(())
}

fn validate_observable_inputs(
    operator: &[Vec<Expr>],
    rho: &[Vec<Expr>],
) -> Result<usize, ObservableError> {
    let (operator_rows, operator_cols) = matrix_shape(operator).unwrap_or((
        operator.len(),
        operator.first().map(|row| row.len()).unwrap_or(0),
    ));
    if operator_rows != operator_cols {
        return Err(ObservableError::OperatorNotSquare {
            rows: operator_rows,
            cols: operator_cols,
        });
    }

    let (state_rows, state_cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if state_rows != state_cols {
        return Err(ObservableError::StateNotSquare {
            rows: state_rows,
            cols: state_cols,
        });
    }

    if state_rows != operator_rows {
        return Err(ObservableError::DimensionMismatch {
            expected: operator_rows,
            actual: state_rows,
        });
    }

    Ok(operator_rows)
}

fn validate_state_functional_input(rho: &[Vec<Expr>]) -> Result<usize, StateFunctionalError> {
    let (rows, cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if rows != cols {
        return Err(StateFunctionalError::StateNotSquare { rows, cols });
    }
    Ok(rows)
}

pub fn pauli_x(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero()],
    ]
}

pub fn pauli_y(interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    let _ = interner;
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));
    let neg_i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())));
    vec![vec![Expr::zero(), neg_i], vec![i, Expr::zero()]]
}

pub fn pauli_z(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ]
}

fn validate_spin_dimension(two_j: usize) -> Result<usize, SpinError> {
    two_j
        .checked_add(1)
        .ok_or(SpinError::InvalidSpinQuantumNumber)
}

/// Return the Hilbert-space dimension `2j + 1` for the exact integer label `two_j = 2j`.
pub fn spin_j_dimension(two_j: usize) -> usize {
    two_j + 1
}

/// Construct the exact `J_z` matrix for an arbitrary spin `j`, represented by `two_j = 2j`,
/// in the standard ordered basis `|j,m⟩` with `m = j, j-1, ..., -j`.
pub fn jz_matrix(two_j: usize, _interner: &ax_ir::Interner) -> Result<Vec<Vec<Expr>>, SpinError> {
    let dim = validate_spin_dimension(two_j)?;
    let mut matrix = zero_matrix(dim);
    for idx in 0..dim {
        let two_m = BigInt::from(two_j) - BigInt::from(2usize) * BigInt::from(idx);
        matrix[idx][idx] = expr_from_exact_rational(BigRational::new(two_m, BigInt::from(2usize)));
    }
    Ok(matrix)
}

/// Construct the exact raising operator `J_+` matrix for an arbitrary spin `j`, represented by
/// `two_j = 2j`, in the standard ordered basis `|j,m⟩` with `m = j, j-1, ..., -j`.
pub fn jplus_matrix(two_j: usize, interner: &ax_ir::Interner) -> Result<Vec<Vec<Expr>>, SpinError> {
    let dim = validate_spin_dimension(two_j)?;
    let mut matrix = zero_matrix(dim);
    for idx in 1..dim {
        let coefficient = BigInt::from(idx) * BigInt::from(two_j - idx + 1);
        matrix[idx - 1][idx] = simplified_sqrt_expr(Expr::Int(coefficient), interner);
    }
    Ok(matrix)
}

/// Construct the exact lowering operator `J_-` matrix for an arbitrary spin `j`, represented by
/// `two_j = 2j`, in the standard ordered basis `|j,m⟩` with `m = j, j-1, ..., -j`.
pub fn jminus_matrix(
    two_j: usize,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, SpinError> {
    let dim = validate_spin_dimension(two_j)?;
    let mut matrix = zero_matrix(dim);
    for idx in 0..(dim.saturating_sub(1)) {
        let coefficient = BigInt::from(two_j - idx) * BigInt::from(idx + 1);
        matrix[idx + 1][idx] = simplified_sqrt_expr(Expr::Int(coefficient), interner);
    }
    Ok(matrix)
}

/// Construct the exact Cartesian angular-momentum operator `J_x = (J_+ + J_-)/2` for an
/// arbitrary spin `j`, represented by `two_j = 2j`.
pub fn jx_matrix(two_j: usize, interner: &ax_ir::Interner) -> Result<Vec<Vec<Expr>>, SpinError> {
    let jp = jplus_matrix(two_j, interner)?;
    let jm = jminus_matrix(two_j, interner)?;
    Ok(simplify_matrix(ax_linalg::mat_scale(
        &half(),
        &ax_linalg::mat_add(&jp, &jm),
    )))
}

/// Construct the exact Cartesian angular-momentum operator `J_y = (J_+ - J_-)/(2i)` for an
/// arbitrary spin `j`, represented by `two_j = 2j`.
pub fn jy_matrix(two_j: usize, interner: &ax_ir::Interner) -> Result<Vec<Vec<Expr>>, SpinError> {
    let jp = jplus_matrix(two_j, interner)?;
    let jm = jminus_matrix(two_j, interner)?;
    let minus_i_over_two = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(half())));
    Ok(simplify_matrix(ax_linalg::mat_scale(
        &minus_i_over_two,
        &ax_linalg::mat_add(&jp, &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &jm)),
    )))
}

fn inv_sqrt_two(interner: &ax_ir::Interner) -> Expr {
    exact_sqrt_expr(half(), interner)
}

/// Return the explicit two-spin-1/2 singlet state in the computational basis
/// `|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩`.
///
/// The state is
/// `(|↑↓⟩ - |↓↑⟩) / sqrt(2)`.
pub fn two_spin_half_singlet_state(interner: &ax_ir::Interner) -> Vec<Expr> {
    let coeff = inv_sqrt_two(interner);
    vec![Expr::zero(), coeff.clone(), Expr::neg(coeff), Expr::zero()]
}

/// Return the explicit two-spin-1/2 triplet states in the computational basis
/// `|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩`.
///
/// The states are ordered as
/// `|1,1⟩ = |↑↑⟩`,
/// `|1,0⟩ = (|↑↓⟩ + |↓↑⟩) / sqrt(2)`,
/// `|1,-1⟩ = |↓↓⟩`.
pub fn two_spin_half_triplet_states(interner: &ax_ir::Interner) -> [Vec<Expr>; 3] {
    let coeff = inv_sqrt_two(interner);
    [
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), coeff.clone(), coeff, Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
    ]
}

/// Return the exact singlet projector `|S⟩⟨S|` for two spin-1/2 systems in the
/// computational basis `|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩`.
pub fn two_spin_half_singlet_projector(interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    density_matrix(&two_spin_half_singlet_state(interner))
}

/// Return the exact triplet projector for two spin-1/2 systems in the
/// computational basis `|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩`.
///
/// This projector is the sum of the three rank-one triplet projectors
/// `|1,1⟩⟨1,1| + |1,0⟩⟨1,0| + |1,-1⟩⟨1,-1|`.
pub fn two_spin_half_triplet_projector(interner: &ax_ir::Interner) -> Vec<Vec<Expr>> {
    let mut projector = zero_matrix(4);
    for state in two_spin_half_triplet_states(interner) {
        projector = simplify_matrix(ax_linalg::mat_add(&projector, &density_matrix(&state)));
    }
    projector
}

fn imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
}

fn neg_imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())))
}

pub fn gamma_matrices_dirac(_interner: &ax_ir::Interner) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    let i = imag_unit();
    let neg_i = neg_imag_unit();
    vec![
        vec![
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
            ],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
            ],
            vec![
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
            ],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), neg_i.clone()],
            vec![Expr::zero(), Expr::zero(), i.clone(), Expr::zero()],
            vec![Expr::zero(), i, Expr::zero(), Expr::zero()],
            vec![neg_i, Expr::zero(), Expr::zero(), Expr::zero()],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
            ],
            vec![
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
            ],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
        ],
    ]
}

pub fn gamma5(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
    ]
}

pub fn gamma_trace_recursive(
    indices: &[lasso::Spur],
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;
    let n = indices.len();
    if n == 0 {
        return Expr::Int(4.into());
    }
    if n % 2 != 0 {
        return Expr::zero();
    }
    if n == 2 {
        return Expr::mul(vec![
            Expr::Int(4.into()),
            Expr::Indexed(
                Box::new(Expr::Sym(metric_sym)),
                vec![
                    ax_ir::Index {
                        name: indices[0],
                        variance: ax_ir::Variance::Up,
                        index_type: None,
                    },
                    ax_ir::Index {
                        name: indices[1],
                        variance: ax_ir::Variance::Up,
                        index_type: None,
                    },
                ],
            ),
        ]);
    }

    let a1 = indices[0];
    let mut terms = Vec::new();
    for k in 1..n {
        let sign = if (k - 1) % 2 == 0 {
            Expr::one()
        } else {
            Expr::neg(Expr::one())
        };
        let metric_factor = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                ax_ir::Index {
                    name: a1,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: indices[k],
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
            ],
        );
        let remaining = indices[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k - 1)
            .map(|(_, sym)| *sym)
            .collect::<Vec<_>>();
        let sub_trace = gamma_trace_recursive(&remaining, metric_sym, interner);
        terms.push(Expr::mul(vec![sign, metric_factor, sub_trace]));
    }
    Expr::add(terms)
}

pub fn gamma_trace(
    indices: &[GammaEntry],
    metric: &ax_tensor::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let metric_sym = interner.get_or_intern("g");
    let epsilon_sym = interner.get_or_intern("epsilon");

    let mut gamma_indices = Vec::new();
    let mut numeric_indices = Vec::new();
    let mut has_symbolic_indices = false;
    let mut has_numeric_indices = false;
    let mut gamma5_count = 0usize;
    for entry in indices {
        match entry {
            GammaEntry::Gamma(sym) => {
                gamma_indices.push(*sym);
                has_symbolic_indices = true;
            }
            GammaEntry::Index(index) => {
                numeric_indices.push(*index);
                has_numeric_indices = true;
            }
            GammaEntry::Gamma5 => gamma5_count += 1,
            GammaEntry::Identity => {}
        }
    }

    if has_numeric_indices && !has_symbolic_indices && gamma5_count == 0 {
        return gamma_trace_numeric(&numeric_indices, metric);
    }

    if has_numeric_indices {
        gamma_indices.extend(numeric_indices.into_iter().map(|index| {
            let name = format!("mu{index}");
            interner.get_or_intern(&name)
        }));
    }

    if gamma5_count > 1 {
        return Expr::zero();
    }

    if gamma5_count == 1 {
        return match gamma_indices.len() {
            0 | 1 | 2 | 3 => Expr::zero(),
            4 => Expr::mul(vec![
                Expr::Int((-4).into()),
                imag_unit(),
                Expr::Indexed(
                    Box::new(Expr::Sym(epsilon_sym)),
                    gamma_indices
                        .iter()
                        .map(|sym| ax_ir::Index {
                            name: *sym,
                            variance: ax_ir::Variance::Up,
                            index_type: None,
                        })
                        .collect(),
                ),
            ]),
            _ if gamma_indices.len() % 2 != 0 => Expr::zero(),
            _ => Expr::zero(),
        };
    }

    gamma_trace_recursive(&gamma_indices, metric_sym, interner)
}

fn gamma_trace_numeric(indices: &[usize], metric: &ax_tensor::SymbolicMatrix) -> Expr {
    let n = indices.len();
    if n == 0 {
        return Expr::Int(4.into());
    }
    if n % 2 != 0 {
        return Expr::zero();
    }
    if n == 2 {
        return Expr::mul(vec![
            Expr::Int(4.into()),
            metric.get(indices[0], indices[1]).clone(),
        ]);
    }

    let first = indices[0];
    let mut terms = Vec::new();
    for k in 1..n {
        let metric_factor = metric.get(first, indices[k]).clone();
        let remaining = indices[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k - 1)
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        let term = Expr::mul(vec![metric_factor, gamma_trace_numeric(&remaining, metric)]);
        if (k - 1) % 2 == 0 {
            terms.push(term);
        } else {
            terms.push(Expr::neg(term));
        }
    }
    simplify_expr(Expr::add(terms))
}

pub fn commutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(
        &ab,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &ba),
    ))
}

pub fn anticommutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(&ab, &ba))
}

fn half() -> Expr {
    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
}

pub fn angular_momentum_matrices(
    j: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<(
    Vec<Vec<ax_ir::Expr>>,
    Vec<Vec<ax_ir::Expr>>,
    Vec<Vec<ax_ir::Expr>>,
)> {
    let two_j = match j {
        Expr::Rational(r) if r.denom() == &BigInt::from(2usize) && !r.is_negative() => {
            r.numer().to_usize()?
        }
        Expr::Int(n) if !n.is_negative() => n.to_usize()?.checked_mul(2)?,
        _ => return None,
    };
    Some((
        jx_matrix(two_j, interner).ok()?,
        jy_matrix(two_j, interner).ok()?,
        jz_matrix(two_j, interner).ok()?,
    ))
}

pub fn density_matrix(state: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    match try_density_matrix(state) {
        Ok(matrix) => matrix,
        Err(_) => Vec::new(),
    }
}

pub fn partial_trace(
    rho: &[Vec<ax_ir::Expr>],
    dim_a: usize,
    dim_b: usize,
    trace_over: char,
    _interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let target = match trace_over {
        'A' => PartialTraceTarget::A,
        'B' => PartialTraceTarget::B,
        other => {
            let _ = QmLinearAlgebraError::InvalidTraceTarget { target: other };
            return Vec::new();
        }
    };
    try_partial_trace(rho, BipartiteDims { dim_a, dim_b }, target).unwrap_or_default()
}

pub fn ket(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    match try_ket(index, dim) {
        Ok(vec) => vec,
        Err(_) => vec![Expr::zero(); dim],
    }
}

pub fn bra(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    match try_bra(index, dim) {
        Ok(vec) => vec,
        Err(_) => vec![Expr::zero(); dim],
    }
}

pub fn braket(bra: &[ax_ir::Expr], ket: &[ax_ir::Expr]) -> ax_ir::Expr {
    match try_braket(bra, ket) {
        Ok(expr) => expr,
        Err(_) => Expr::add(
            bra.iter()
                .zip(ket.iter())
                .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
                .collect(),
        ),
    }
}

pub fn outer(a: &[ax_ir::Expr], b: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    match try_outer(a, b) {
        Ok(matrix) => matrix,
        Err(_) => a
            .iter()
            .map(|ai| {
                b.iter()
                    .map(|bj| Expr::mul(vec![ai.clone(), bj.clone()]))
                    .collect()
            })
            .collect(),
    }
}

/// Return the complex conjugate of an expression while preserving symbolic structure.
pub fn conjugate_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => expr.clone(),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(conjugate_expr(re)),
            Box::new(Expr::neg(conjugate_expr(im))),
        ),
        Expr::Add(items) => Expr::add(items.iter().map(conjugate_expr).collect()),
        Expr::Mul(items) => Expr::mul(items.iter().map(conjugate_expr).collect()),
        Expr::Pow(base, exp) => Expr::pow(conjugate_expr(base), conjugate_expr(exp)),
        Expr::Neg(inner) => Expr::neg(conjugate_expr(inner)),
        Expr::Call(sym, args) => Expr::Call(*sym, args.iter().map(conjugate_expr).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(conjugate_expr(body)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(conjugate_expr(lhs)),
            Box::new(conjugate_expr(rhs)),
            *trust,
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(conjugate_expr(base)), indices.clone())
        }
        Expr::Group(inner, rel) => Expr::Group(Box::new(conjugate_expr(inner)), *rel),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(conjugate_expr(value)),
            Box::new(conjugate_expr(body)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(conjugate_expr).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(conjugate_expr).collect())
                .collect(),
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (conjugate_expr(value), condition.clone()))
                .collect(),
        ),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => expr.clone(),
    }
}

/// Return the Hermitian adjoint of a state vector represented as a flat expression slice.
pub fn adjoint_vector(vec: &[Expr]) -> Vec<Expr> {
    vec.iter().map(conjugate_expr).collect()
}

/// Build a computational basis ket with bounds checking.
pub fn try_ket(index: usize, dim: usize) -> Result<Vec<Expr>, QmLinearAlgebraError> {
    if index >= dim {
        return Err(QmLinearAlgebraError::BasisIndexOutOfRange { index, dim });
    }
    let mut out = vec![Expr::zero(); dim];
    out[index] = Expr::one();
    Ok(out)
}

/// Build a computational basis bra as the adjoint of the corresponding basis ket.
pub fn try_bra(index: usize, dim: usize) -> Result<Vec<Expr>, QmLinearAlgebraError> {
    let ket = try_ket(index, dim)?;
    Ok(adjoint_vector(&ket))
}

/// Compute the inner product `⟨bra|ket⟩ = Σ_i conj(bra_i) * ket_i`.
pub fn try_braket(bra: &[Expr], ket: &[Expr]) -> Result<Expr, QmLinearAlgebraError> {
    if bra.len() != ket.len() {
        return Err(QmLinearAlgebraError::DimensionMismatch {
            left: bra.len(),
            right: ket.len(),
        });
    }
    Ok(Expr::add(
        bra.iter()
            .zip(ket.iter())
            .map(|(bra_i, ket_i)| Expr::mul(vec![conjugate_expr(bra_i), ket_i.clone()]))
            .collect(),
    ))
}

/// Compute the outer product `|ket⟩⟨bra|` with element `(i, j) = ket[i] * conj(bra[j])`.
pub fn try_outer(ket: &[Expr], bra: &[Expr]) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    Ok(ket
        .iter()
        .map(|ket_i| {
            bra.iter()
                .map(|bra_j| Expr::mul(vec![ket_i.clone(), conjugate_expr(bra_j)]))
                .collect()
        })
        .collect())
}

/// Flatten a matrix into a single column-stacked vector in column-major order.
pub fn vec_column_major(mat: &[Vec<Expr>]) -> Vec<Expr> {
    let Some((rows, cols)) = matrix_shape(mat) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in mat.iter().take(rows) {
            out.push(row[col].clone());
        }
    }
    out
}

/// Form the conjugation-aware outer product `vec vec†` of a column vector.
pub fn outer_column_vector(vec: &[Expr]) -> Vec<Vec<Expr>> {
    match try_outer(vec, vec) {
        Ok(matrix) => matrix,
        Err(_) => Vec::new(),
    }
}

/// Compute the pure-state density matrix `|ψ⟩⟨ψ|`.
pub fn try_density_matrix(state: &[Expr]) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    try_outer(state, state)
}

/// Construct the computational-basis projector `|index⟩⟨index|`.
pub fn basis_projector(index: usize, dim: usize) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    let ket = try_ket(index, dim)?;
    try_outer(&ket, &ket)
}

/// Compute the Kraus completeness matrix `Σ_k K_k† K_k` for a finite-dimensional channel.
pub fn kraus_completeness_matrix(kraus: &[Vec<Vec<Expr>>]) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = kraus_dimension(kraus)?;
    let interner = ax_ir::Interner::new();
    let mut completeness = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let term = ax_linalg::mat_mul(&adjoint, operator, &interner);
        completeness = simplify_matrix(ax_linalg::mat_add(&completeness, &term));
    }
    Ok(simplify_matrix(completeness))
}

/// Compute the exact trace-preserving residual `Σ_k K_k† K_k - I` for a Kraus channel.
///
/// The input Kraus set must be non-empty, and every operator must be square with
/// a common dimension.
pub fn trace_preserving_residual(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = kraus_dimension(kraus)?;
    let mut completeness = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let term = ax_linalg::mat_mul(&adjoint, operator, interner);
        completeness = simplify_matrix(ax_linalg::mat_add(&completeness, &term));
    }
    let identity = identity_matrix(dim, interner);
    Ok(simplify_matrix(ax_linalg::mat_add(
        &completeness,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &identity),
    )))
}

/// Check whether a Kraus channel is exactly trace preserving.
///
/// This returns `true` iff every entry of the residual `Σ_k K_k† K_k - I` is
/// structurally zero after simplification.
pub fn is_trace_preserving_exact(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<bool, ChannelError> {
    let residual = trace_preserving_residual(kraus, interner)?;
    Ok(residual
        .iter()
        .flat_map(|row| row.iter())
        .all(expr_is_structurally_zero))
}

/// Compute the exact unital residual `Σ_k K_k K_k† - I` for a Kraus channel.
///
/// The input Kraus set must be non-empty, and every operator must be square with
/// a common dimension.
pub fn unital_residual(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = kraus_dimension(kraus)?;
    let mut completeness = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let term = ax_linalg::mat_mul(operator, &adjoint, interner);
        completeness = simplify_matrix(ax_linalg::mat_add(&completeness, &term));
    }
    let identity = identity_matrix(dim, interner);
    Ok(simplify_matrix(ax_linalg::mat_add(
        &completeness,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &identity),
    )))
}

/// Check whether a Kraus channel is exactly unital.
///
/// This returns `true` iff every entry of the residual `Σ_k K_k K_k† - I` is
/// structurally zero after simplification.
pub fn is_unital_exact(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<bool, ChannelError> {
    let residual = unital_residual(kraus, interner)?;
    Ok(residual
        .iter()
        .flat_map(|row| row.iter())
        .all(expr_is_structurally_zero))
}

/// Compute the Frobenius distance between two channels via their Choi matrices.
///
/// This constructs both Choi matrices and returns `||J(left) - J(right)||_F`,
/// where the norm is represented exactly as `sqrt(Tr((J1 - J2)† (J1 - J2)))`
/// unless the result simplifies structurally to zero.
pub fn choi_frobenius_distance(
    left: &[Vec<Vec<Expr>>],
    right: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Expr, ChannelError> {
    let left_dim = kraus_dimension(left)?;
    let right_dim = kraus_dimension(right)?;
    if left_dim != right_dim {
        return Err(ChannelError::CompositionDimensionMismatch {
            left_dim,
            right_dim,
        });
    }

    let left_choi = choi_matrix_from_kraus(left)?;
    let right_choi = choi_matrix_from_kraus(right)?;
    choi_frobenius_distance_from_choi(&left_choi, &right_choi, interner)
}

/// Compute the Frobenius distance between two Choi matrices.
///
/// The input Choi matrices must be square and have matching shape. The returned
/// expression is `sqrt(Tr((J1 - J2)† (J1 - J2)))`, with an exact `0` returned
/// when the squared norm simplifies structurally to zero.
pub fn choi_frobenius_distance_from_choi(
    left: &[Vec<Expr>],
    right: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Expr, ChannelError> {
    let Some((left_rows, left_cols)) = matrix_shape(left) else {
        return Err(ChannelError::InvalidChoiDimension { dim: left.len() });
    };
    let Some((right_rows, right_cols)) = matrix_shape(right) else {
        return Err(ChannelError::InvalidChoiDimension { dim: right.len() });
    };
    if left_rows != left_cols {
        return Err(ChannelError::InvalidChoiDimension { dim: left_rows });
    }
    if right_rows != right_cols {
        return Err(ChannelError::InvalidChoiDimension { dim: right_rows });
    }
    if left_rows != right_rows {
        return Err(ChannelError::CompositionDimensionMismatch {
            left_dim: left_rows,
            right_dim: right_rows,
        });
    }

    let delta = matrix_subtract(left, right);
    let delta_adjoint = adjoint_matrix(&delta);
    let gram = simplify_matrix(ax_linalg::mat_mul(&delta_adjoint, &delta, interner));
    let norm_sq = simplify_expr(ax_linalg::trace(&gram));
    if norm_sq == Expr::zero() {
        Ok(Expr::zero())
    } else {
        Ok(exact_sqrt_expr(norm_sq, interner))
    }
}

/// Construct the Choi matrix `J(E) = Σ_k vec(K_k) vec(K_k)†` from Kraus operators.
///
/// This uses column-stacking vectorization, so a `d x d` channel produces a
/// `d^2 x d^2` Choi matrix.
pub fn choi_matrix_from_kraus(kraus: &[Vec<Vec<Expr>>]) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = kraus_dimension(kraus)?;
    let choi_dim = dim.checked_mul(dim).ok_or(ChannelError::InvalidKrausSet)?;
    let mut choi = zero_matrix(choi_dim);

    for operator in kraus {
        let vec_k = vec_column_major(operator);
        let term = outer_column_vector(&vec_k);
        choi = simplify_matrix(ax_linalg::mat_add(&choi, &term));
    }

    Ok(simplify_matrix(choi))
}

fn sqrt_argument<'a>(expr: &'a Expr, interner: &ax_ir::Interner) -> Option<&'a Expr> {
    let Expr::Call(sym, args) = expr else {
        return None;
    };
    if interner.resolve(*sym) == "sqrt" {
        let [arg] = args.as_slice() else {
            return None;
        };
        Some(arg)
    } else {
        None
    }
}

fn one_minus_parameter(expr: &Expr, interner: &ax_ir::Interner) -> Option<Expr> {
    let arg = sqrt_argument(expr, interner)?;
    let Expr::Add(terms) = arg else {
        return None;
    };
    let [one, neg_param] = terms.as_slice() else {
        return None;
    };
    if one != &Expr::one() {
        return None;
    }
    match neg_param {
        Expr::Neg(param) => Some((**param).clone()),
        _ => None,
    }
}

fn scaled_by_one_third_parameter(expr: &Expr, interner: &ax_ir::Interner) -> Option<Expr> {
    let arg = sqrt_argument(expr, interner)?;
    let Expr::Mul(factors) = arg else {
        return None;
    };
    let [left, right] = factors.as_slice() else {
        return None;
    };
    let one_third = Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(3usize)));
    if right == &one_third {
        Some(left.clone())
    } else if left == &one_third {
        Some(right.clone())
    } else {
        None
    }
}

fn exact_canonical_qubit_family_hint(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Option<&'static str> {
    if kraus.len() == 4 {
        let p = kraus
            .first()
            .and_then(|k| k.first())
            .and_then(|row| row.first())
            .and_then(|entry| one_minus_parameter(entry, interner))
            .or_else(|| {
                kraus
                    .get(1)
                    .and_then(|k| k.first())
                    .and_then(|row| row.get(1))
                    .and_then(|entry| scaled_by_one_third_parameter(entry, interner))
            })?;
        if depolarizing_channel_qubit(p, interner) == kraus {
            return Some("depolarizing");
        }
        return None;
    }

    if kraus.len() != 2 {
        return None;
    }

    let gamma = kraus
        .get(1)
        .and_then(|k| k.first())
        .and_then(|row| row.get(1))
        .and_then(|entry| sqrt_argument(entry, interner))
        .cloned();
    if let Some(gamma) = gamma {
        if amplitude_damping_channel_qubit(gamma, interner) == kraus {
            return Some("amplitude_damping");
        }
    }

    let p = kraus
        .first()
        .and_then(|k| k.first())
        .and_then(|row| row.first())
        .and_then(|entry| one_minus_parameter(entry, interner))?;
    if bit_flip_channel_qubit(p.clone(), interner) == kraus {
        return Some("bit_flip");
    }
    if dephasing_channel_qubit(p.clone(), interner) == kraus {
        return Some("dephasing");
    }
    if phase_flip_channel_qubit(p.clone(), interner) == kraus {
        return Some("phase_flip");
    }
    if bit_phase_flip_channel_qubit(p, interner) == kraus {
        return Some("bit_phase_flip");
    }
    None
}

/// Return a canonical family hint for Kraus lists produced exactly by this crate.
///
/// The recognizer verifies the input against the concrete Kraus operators emitted
/// by the channel constructors and does not classify structurally similar,
/// partially matching, or reordered arbitrary Kraus sets.
pub fn canonical_channel_family_hint(
    kraus: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Option<String> {
    if let Ok(dim) = kraus_dimension(kraus) {
        if identity_channel(dim) == kraus {
            return Some("identity".to_string());
        }
    }

    exact_canonical_qubit_family_hint(kraus, interner).map(str::to_string)
}

fn matrix_from_column_major_vec(vec: &[Expr], dim: usize) -> Option<Vec<Vec<Expr>>> {
    if vec.len() != dim.checked_mul(dim)? {
        return None;
    }

    let mut matrix = vec![vec![Expr::zero(); dim]; dim];
    let mut cursor = 0usize;
    for col in 0..dim {
        for row in matrix.iter_mut().take(dim) {
            row[col] = vec[cursor].clone();
            cursor += 1;
        }
    }
    Some(matrix)
}

/// Recover Kraus operators from exact small Choi matrices in narrowly supported cases.
///
/// Supported cases are:
/// - a `1x1` trivial channel with Choi matrix `[[1]]`
/// - a `4x4` rank-1 Choi matrix equal to `vec(K) vec(K)†` for a single `2x2` Kraus operator
///
/// Any other Choi matrix returns `UnsupportedChoiRecovery` rather than attempting a
/// generic spectral decomposition.
pub fn kraus_from_choi_small(choi: &[Vec<Expr>]) -> Result<Vec<Vec<Vec<Expr>>>, ChannelError> {
    let Some((rows, cols)) = matrix_shape(choi) else {
        return Err(ChannelError::InvalidChoiDimension { dim: choi.len() });
    };
    if rows != cols {
        return Err(ChannelError::InvalidChoiDimension { dim: rows });
    }

    match rows {
        1 => {
            if choi[0][0] == Expr::one() {
                Ok(vec![vec![vec![Expr::one()]]])
            } else {
                Err(ChannelError::UnsupportedChoiRecovery)
            }
        }
        4 => {
            let Some(pivot) = (0..4).find(|&idx| !is_zero_expr(&choi[idx][idx])) else {
                return Err(ChannelError::UnsupportedChoiRecovery);
            };

            if choi[pivot][pivot] != Expr::one() {
                return Err(ChannelError::UnsupportedChoiRecovery);
            }

            let vec_k = choi
                .iter()
                .map(|row| row[pivot].clone())
                .collect::<Vec<_>>();
            let rebuilt = simplify_matrix(outer_column_vector(&vec_k));
            if rebuilt != simplify_matrix(choi.to_vec()) {
                return Err(ChannelError::UnsupportedChoiRecovery);
            }

            let kraus = matrix_from_column_major_vec(&vec_k, 2)
                .ok_or(ChannelError::UnsupportedChoiRecovery)?;
            Ok(vec![kraus])
        }
        dim => Err(ChannelError::InvalidChoiDimension { dim }),
    }
}

/// Apply a Kraus channel to a density matrix via `Σ_k K_k ρ K_k†`.
pub fn apply_kraus_channel(
    kraus: &[Vec<Vec<Expr>>],
    rho: &[Vec<Expr>],
) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = kraus_dimension(kraus)?;
    let (rows, cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if rows != cols || rows != dim {
        return Err(ChannelError::StateDimensionMismatch {
            expected: dim,
            actual: rows,
        });
    }

    let interner = ax_ir::Interner::new();
    let mut output = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let left = ax_linalg::mat_mul(operator, rho, &interner);
        let term = ax_linalg::mat_mul(&left, &adjoint, &interner);
        output = simplify_matrix(ax_linalg::mat_add(&output, &term));
    }

    Ok(simplify_matrix(output))
}

/// Return the common square dimension of a validated Kraus set.
///
/// This requires a non-empty Kraus list where every operator is square and all
/// operators share the same dimension.
pub fn kraus_dimension(kraus: &[Vec<Vec<Expr>>]) -> Result<usize, ChannelError> {
    let dim = validate_kraus_set(kraus)?;
    if dim == 0 {
        return Err(ChannelError::InvalidKrausSet);
    }
    Ok(dim)
}

/// Compose two Kraus channels by applying `right` first and then `left`.
///
/// The resulting Kraus operators are ordered as `{L_i R_j}` with `i` varying
/// slowest, so `{L1, L2}` composed with `{R1, R2}` yields
/// `{L1 R1, L1 R2, L2 R1, L2 R2}`.
pub fn compose_kraus_channels(
    left: &[Vec<Vec<Expr>>],
    right: &[Vec<Vec<Expr>>],
) -> Result<Vec<Vec<Vec<Expr>>>, ChannelError> {
    let left_dim = kraus_dimension(left)?;
    let right_dim = kraus_dimension(right)?;
    if left_dim != right_dim {
        return Err(ChannelError::CompositionDimensionMismatch {
            left_dim,
            right_dim,
        });
    }

    let interner = ax_ir::Interner::new();
    let mut composed = Vec::with_capacity(left.len() * right.len());
    for left_operator in left {
        for right_operator in right {
            composed.push(simplify_matrix(ax_linalg::mat_mul(
                left_operator,
                right_operator,
                &interner,
            )));
        }
    }
    Ok(composed)
}

/// Form the tensor-product channel whose Kraus operators are `{L_i ⊗ R_j}`.
///
/// The output preserves lexicographic pair ordering with `left` varying
/// slowest, so `{L1, L2}` tensored with `{R1, R2}` yields
/// `{L1 ⊗ R1, L1 ⊗ R2, L2 ⊗ R1, L2 ⊗ R2}`.
pub fn tensor_product_kraus_channels(
    left: &[Vec<Vec<Expr>>],
    right: &[Vec<Vec<Expr>>],
) -> Result<Vec<Vec<Vec<Expr>>>, ChannelError> {
    let _left_dim = kraus_dimension(left)?;
    let _right_dim = kraus_dimension(right)?;

    let mut product = Vec::with_capacity(left.len() * right.len());
    for left_operator in left {
        for right_operator in right {
            product.push(simplify_matrix(ax_linalg::tensor_product(
                left_operator,
                right_operator,
            )));
        }
    }

    Ok(product)
}

/// Construct the finite-dimensional identity channel with a single identity Kraus operator.
pub fn identity_channel(dim: usize) -> Vec<Vec<Vec<Expr>>> {
    vec![ax_linalg::identity(dim)]
}

/// Construct the canonical qubit depolarizing channel.
///
/// The Kraus operators are `{sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z}`.
pub fn depolarizing_channel_qubit(p: Expr, interner: &ax_ir::Interner) -> Vec<Vec<Vec<Expr>>> {
    let identity = identity_matrix(2, interner);
    let pauli_x = pauli_x(interner);
    let pauli_y = pauli_y(interner);
    let pauli_z = pauli_z(interner);
    let one_minus_p = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(p.clone())]));
    let p_over_three = simplify_expr(Expr::mul(vec![
        p,
        Expr::Rational(BigRational::new(1.into(), 3.into())),
    ]));
    let weight_identity = simplified_sqrt_expr(one_minus_p, interner);
    let weight_pauli = simplified_sqrt_expr(p_over_three, interner);

    prune_zero_kraus_ops(vec![
        simplify_matrix(ax_linalg::mat_scale(&weight_identity, &identity)),
        simplify_matrix(ax_linalg::mat_scale(&weight_pauli, &pauli_x)),
        simplify_matrix(ax_linalg::mat_scale(&weight_pauli, &pauli_y)),
        simplify_matrix(ax_linalg::mat_scale(&weight_pauli, &pauli_z)),
    ])
}

/// Construct the canonical qubit dephasing channel.
///
/// The Kraus operators are `{sqrt(1-p) I, sqrt(p) Z}`.
pub fn dephasing_channel_qubit(p: Expr, interner: &ax_ir::Interner) -> Vec<Vec<Vec<Expr>>> {
    let identity = identity_matrix(2, interner);
    let pauli_z = pauli_z(interner);
    let one_minus_p = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(p.clone())]));
    let weight_identity = simplified_sqrt_expr(one_minus_p, interner);
    let weight_phase = simplified_sqrt_expr(p, interner);

    prune_zero_kraus_ops(vec![
        simplify_matrix(ax_linalg::mat_scale(&weight_identity, &identity)),
        simplify_matrix(ax_linalg::mat_scale(&weight_phase, &pauli_z)),
    ])
}

/// Construct the canonical qubit amplitude-damping channel.
///
/// The Kraus operators are `{[[1,0],[0,sqrt(1-gamma)]], [[0,sqrt(gamma)],[0,0]]}`.
pub fn amplitude_damping_channel_qubit(
    gamma: Expr,
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Expr>>> {
    let sqrt = |expr| simplified_sqrt_expr(expr, interner);
    let one_minus_gamma = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(gamma.clone())]));

    prune_zero_kraus_ops(vec![
        vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), sqrt(one_minus_gamma)],
        ],
        vec![
            vec![Expr::zero(), sqrt(gamma)],
            vec![Expr::zero(), Expr::zero()],
        ],
    ])
}

/// Construct the canonical qubit bit-flip channel.
///
/// The Kraus operators are `{sqrt(1-p) I, sqrt(p) X}`.
pub fn bit_flip_channel_qubit(p: Expr, interner: &ax_ir::Interner) -> Vec<Vec<Vec<Expr>>> {
    let identity = identity_matrix(2, interner);
    let pauli_x = pauli_x(interner);
    let one_minus_p = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(p.clone())]));
    let weight_identity = simplified_sqrt_expr(one_minus_p, interner);
    let weight_flip = simplified_sqrt_expr(p, interner);

    prune_zero_kraus_ops(vec![
        simplify_matrix(ax_linalg::mat_scale(&weight_identity, &identity)),
        simplify_matrix(ax_linalg::mat_scale(&weight_flip, &pauli_x)),
    ])
}

/// Construct the canonical qubit phase-flip channel.
///
/// The Kraus operators are `{sqrt(1-p) I, sqrt(p) Z}`.
pub fn phase_flip_channel_qubit(p: Expr, interner: &ax_ir::Interner) -> Vec<Vec<Vec<Expr>>> {
    let identity = identity_matrix(2, interner);
    let pauli_z = pauli_z(interner);
    let one_minus_p = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(p.clone())]));
    let weight_identity = simplified_sqrt_expr(one_minus_p, interner);
    let weight_flip = simplified_sqrt_expr(p, interner);

    prune_zero_kraus_ops(vec![
        simplify_matrix(ax_linalg::mat_scale(&weight_identity, &identity)),
        simplify_matrix(ax_linalg::mat_scale(&weight_flip, &pauli_z)),
    ])
}

/// Construct the canonical qubit bit-phase-flip channel.
///
/// The Kraus operators are `{sqrt(1-p) I, sqrt(p) Y}`.
pub fn bit_phase_flip_channel_qubit(p: Expr, interner: &ax_ir::Interner) -> Vec<Vec<Vec<Expr>>> {
    let identity = identity_matrix(2, interner);
    let pauli_y = pauli_y(interner);
    let one_minus_p = simplify_expr(Expr::add(vec![Expr::one(), Expr::neg(p.clone())]));
    let weight_identity = simplified_sqrt_expr(one_minus_p, interner);
    let weight_flip = simplified_sqrt_expr(p, interner);

    prune_zero_kraus_ops(vec![
        simplify_matrix(ax_linalg::mat_scale(&weight_identity, &identity)),
        simplify_matrix(ax_linalg::mat_scale(&weight_flip, &pauli_y)),
    ])
}

/// Compute projective-measurement probabilities `p_i = Tr(P_i ρ)`.
pub fn measurement_probabilities(
    projectors: &[Vec<Vec<Expr>>],
    rho: &[Vec<Expr>],
) -> Result<Vec<Expr>, MeasurementError> {
    let dim = rho.len();
    validate_square_state_dimension(rho, dim)?;
    validate_projector_set(projectors, dim)?;
    let interner = ax_ir::Interner::new();

    Ok(projectors
        .iter()
        .map(|projector| {
            let product = ax_linalg::mat_mul(projector, rho, &interner);
            simplify_expr(ax_linalg::trace(&product))
        })
        .collect())
}

/// Compute the expectation value `Tr(ρ O)` of an observable against a density operator.
pub fn expectation_value(
    operator: &[Vec<Expr>],
    rho: &[Vec<Expr>],
) -> Result<Expr, ObservableError> {
    validate_observable_inputs(operator, rho)?;
    let interner = ax_ir::Interner::new();
    let product = ax_linalg::mat_mul(rho, operator, &interner);
    Ok(simplify_expr(ax_linalg::trace(&product)))
}

/// Compute the variance `Tr(ρ O^2) - (Tr(ρ O))^2` of an observable against a density operator.
pub fn variance(operator: &[Vec<Expr>], rho: &[Vec<Expr>]) -> Result<Expr, ObservableError> {
    validate_observable_inputs(operator, rho)?;
    let interner = ax_ir::Interner::new();
    let operator_sq = ax_linalg::mat_mul(operator, operator, &interner);
    let rho_operator_sq = ax_linalg::mat_mul(rho, &operator_sq, &interner);
    let second_moment = simplify_expr(ax_linalg::trace(&rho_operator_sq));
    let mean = expectation_value(operator, rho)?;
    Ok(simplify_expr(Expr::add(vec![
        second_moment,
        Expr::neg(Expr::pow(mean, Expr::Int(2.into()))),
    ])))
}

/// Compute the purity `Tr(ρ^2)` of a finite-dimensional density operator.
pub fn purity(rho: &[Vec<Expr>]) -> Result<Expr, StateFunctionalError> {
    validate_state_functional_input(rho)?;
    let interner = ax_ir::Interner::new();
    let rho_sq = ax_linalg::mat_mul(rho, rho, &interner);
    Ok(simplify_expr(ax_linalg::trace(&rho_sq)))
}

/// Compute the linear entropy `1 - Tr(ρ^2)` of a finite-dimensional density operator.
pub fn linear_entropy(rho: &[Vec<Expr>]) -> Result<Expr, StateFunctionalError> {
    let purity_value = purity(rho)?;
    Ok(simplify_expr(Expr::add(vec![
        Expr::one(),
        Expr::neg(purity_value),
    ])))
}

/// Compute the participation ratio `1 / Tr(ρ^2)` of a finite-dimensional density operator.
pub fn participation_ratio(
    rho: &[Vec<Expr>],
    _interner: &ax_ir::Interner,
) -> Result<Expr, StateFunctionalError> {
    let purity_value = purity(rho)?;
    Ok(simplify_expr(Expr::pow(
        purity_value,
        Expr::Int((-1).into()),
    )))
}

/// Compute the Renyi-2 entropy `-log(Tr(ρ^2))` of a finite-dimensional density operator.
pub fn renyi2_entropy(
    rho: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Expr, StateFunctionalError> {
    let purity_value = purity(rho)?;
    if purity_value == Expr::one() {
        return Ok(Expr::zero());
    }
    Ok(simplify_expr(Expr::neg(Expr::Call(
        interner.get_or_intern("log"),
        vec![purity_value],
    ))))
}

/// Compute the von Neumann entropy `S(ρ) = -Σ_i λ_i log(λ_i)` for small supported density matrices.
pub fn von_neumann_entropy(
    rho: &[Vec<Expr>],
    interner: &ax_ir::Interner,
) -> Result<Expr, EntropyError> {
    let (rows, cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if rows != cols {
        return Err(EntropyError::StateNotSquare { rows, cols });
    }

    if rows == 1 {
        let lambda = rho[0][0].clone();
        if lambda == Expr::one() {
            return Ok(Expr::zero());
        }
        return Ok(simplify_expr(entropy_term(&lambda, interner)));
    }

    if !matrix_is_exactly_hermitian(rho) {
        return Err(EntropyError::StateNotHermitian);
    }

    if purity(rho).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            EntropyError::StateNotSquare { rows, cols }
        }
    })? == Expr::one()
    {
        return Ok(Expr::zero());
    }

    let eigenvalues = hermitian_eigenvalues_small(rho, interner)?;
    Ok(simplify_expr(Expr::add(
        eigenvalues
            .iter()
            .map(|lambda| entropy_term(lambda, interner))
            .collect(),
    )))
}

/// Compute the bipartite von Neumann mutual information `S(ρ_A) + S(ρ_B) - S(ρ_AB)`.
///
/// The input density matrix must be arranged in row-major lexicographic order with total
/// dimension `dim_a * dim_b`. The reduced states are constructed using the checked bipartite
/// partial-trace helper.
pub fn von_neumann_mutual_information_bipartite(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, EntropyError> {
    let expected = dim_a.saturating_mul(dim_b);
    let rows = rho_ab.len();
    let cols = rho_ab.first().map(|row| row.len()).unwrap_or(0);
    if rows != expected || cols != expected || rho_ab.iter().any(|row| row.len() != cols) {
        return Err(EntropyError::StateNotSquare { rows, cols });
    }

    let rho_a = try_partial_trace(
        rho_ab,
        BipartiteDims { dim_a, dim_b },
        PartialTraceTarget::B,
    )
    .map_err(|err| match err {
        QmLinearAlgebraError::NonSquareMatrix { rows, cols } => {
            EntropyError::StateNotSquare { rows, cols }
        }
        QmLinearAlgebraError::SubsystemDimensionMismatch { .. }
        | QmLinearAlgebraError::InvalidTraceTarget { .. }
        | QmLinearAlgebraError::DimensionMismatch { .. }
        | QmLinearAlgebraError::BasisIndexOutOfRange { .. } => {
            EntropyError::StateNotSquare { rows, cols }
        }
    })?;
    let rho_b = try_partial_trace(
        rho_ab,
        BipartiteDims { dim_a, dim_b },
        PartialTraceTarget::A,
    )
    .map_err(|err| match err {
        QmLinearAlgebraError::NonSquareMatrix { rows, cols } => {
            EntropyError::StateNotSquare { rows, cols }
        }
        QmLinearAlgebraError::SubsystemDimensionMismatch { .. }
        | QmLinearAlgebraError::InvalidTraceTarget { .. }
        | QmLinearAlgebraError::DimensionMismatch { .. }
        | QmLinearAlgebraError::BasisIndexOutOfRange { .. } => {
            EntropyError::StateNotSquare { rows, cols }
        }
    })?;
    let s_a = von_neumann_entropy(&rho_a, interner)?;
    let s_b = von_neumann_entropy(&rho_b, interner)?;
    let s_ab = von_neumann_entropy(rho_ab, interner)?;
    Ok(simplify_expr(Expr::add(vec![s_a, s_b, Expr::neg(s_ab)])))
}

/// Compute the bipartite conditional entropy `S(B|A) = S(ρ_AB) - S(ρ_A)`.
///
/// The input density matrix must be arranged in row-major lexicographic order with total
/// dimension `dim_a * dim_b`. The reduced state `ρ_A` is obtained via the checked bipartite
/// partial-trace helper by tracing out subsystem `B`.
pub fn conditional_entropy_b_given_a(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, EntropyError> {
    let expected = dim_a.saturating_mul(dim_b);
    let rows = rho_ab.len();
    let cols = rho_ab.first().map(|row| row.len()).unwrap_or(0);
    if rows != expected || cols != expected || rho_ab.iter().any(|row| row.len() != cols) {
        return Err(EntropyError::StateNotSquare { rows, cols });
    }

    let rho_a = try_partial_trace(
        rho_ab,
        BipartiteDims { dim_a, dim_b },
        PartialTraceTarget::B,
    )
    .map_err(|err| match err {
        QmLinearAlgebraError::NonSquareMatrix { rows, cols } => {
            EntropyError::StateNotSquare { rows, cols }
        }
        QmLinearAlgebraError::SubsystemDimensionMismatch { .. }
        | QmLinearAlgebraError::InvalidTraceTarget { .. }
        | QmLinearAlgebraError::DimensionMismatch { .. }
        | QmLinearAlgebraError::BasisIndexOutOfRange { .. } => {
            EntropyError::StateNotSquare { rows, cols }
        }
    })?;
    let s_ab = von_neumann_entropy(rho_ab, interner)?;
    let s_a = von_neumann_entropy(&rho_a, interner)?;
    Ok(simplify_expr(Expr::add(vec![s_ab, Expr::neg(s_a)])))
}

/// Compute the Schmidt coefficients of a bipartite pure state from the reduced spectrum on A.
pub fn schmidt_coefficients_from_state(
    state: &[Expr],
    dim_a: usize,
    dim_b: usize,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, EntanglementError> {
    let spectrum = entanglement_spectrum_from_state(state, dim_a, dim_b, interner)?;
    Ok(spectrum
        .into_iter()
        .map(|lambda| {
            if lambda == Expr::zero() || lambda == Expr::one() {
                lambda
            } else {
                exact_sqrt_expr(lambda, interner)
            }
        })
        .collect())
}

/// Compute the bipartite entanglement spectrum of a pure state as the eigenvalues of `ρ_A`.
pub fn entanglement_spectrum_from_state(
    state: &[Expr],
    dim_a: usize,
    dim_b: usize,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, EntanglementError> {
    let expected = dim_a.saturating_mul(dim_b);
    let actual = state.len();
    if actual != expected {
        return Err(EntanglementError::StateDimensionMismatch { expected, actual });
    }

    let rho_ab = try_density_matrix(state)?;
    entanglement_spectrum_from_density(&rho_ab, dim_a, dim_b, 'A', interner)
}

/// Compute the entanglement spectrum from a bipartite density matrix by reducing to the kept side.
pub fn entanglement_spectrum_from_density(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    kept: char,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, EntanglementError> {
    let expected = dim_a.saturating_mul(dim_b);
    let rows = rho_ab.len();
    let cols = rho_ab.first().map(|row| row.len()).unwrap_or(0);
    if rows != expected || cols != expected || rho_ab.iter().any(|row| row.len() != cols) {
        return Err(EntanglementError::DensityDimensionMismatch {
            expected,
            actual: rows,
        });
    }

    let trace_target = match kept {
        'A' => PartialTraceTarget::B,
        'B' => PartialTraceTarget::A,
        other => {
            return Err(EntanglementError::PartialTrace(
                QmLinearAlgebraError::InvalidTraceTarget { target: other },
            ))
        }
    };
    let reduced = try_partial_trace(rho_ab, BipartiteDims { dim_a, dim_b }, trace_target)?;
    hermitian_eigenvalues_small(&reduced, interner).map_err(EntanglementError::from)
}

/// Compute the negativity from a partial-transpose spectrum by summing exact visible negative parts.
pub fn negativity_from_partial_transpose_spectrum(
    eigs: &[Expr],
    _interner: &ax_ir::Interner,
) -> Expr {
    simplify_expr(Expr::add(
        eigs.iter()
            .filter_map(explicit_negative_magnitude)
            .collect::<Vec<_>>(),
    ))
}

/// Compute bipartite negativity from the exact spectrum of the chosen partial transpose.
pub fn negativity_bipartite(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    transposed_factor: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, NegativityError> {
    let expected = dim_a.saturating_mul(dim_b);
    let rows = rho_ab.len();
    let cols = rho_ab.first().map(|row| row.len()).unwrap_or(0);
    if rows != expected || cols != expected || rho_ab.iter().any(|row| row.len() != cols) {
        return Err(NegativityError::DimensionMismatch {
            expected,
            actual: rows,
        });
    }

    let partial_transpose =
        try_partial_transpose_factor(rho_ab, &[dim_a, dim_b], transposed_factor)?;
    let eigenvalues = spectral_eigenvalues_supported_blocks(&partial_transpose, interner)?;
    Ok(negativity_from_partial_transpose_spectrum(
        &eigenvalues,
        interner,
    ))
}

/// Compute the exact supported spectrum of the chosen bipartite partial transpose.
pub fn partial_transpose_spectrum_bipartite(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    transposed_factor: usize,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, NegativityError> {
    let expected = dim_a.saturating_mul(dim_b);
    let rows = rho_ab.len();
    let cols = rho_ab.first().map(|row| row.len()).unwrap_or(0);
    if rows != expected || cols != expected || rho_ab.iter().any(|row| row.len() != cols) {
        return Err(NegativityError::DimensionMismatch {
            expected,
            actual: rows,
        });
    }

    let partial_transpose =
        try_partial_transpose_factor(rho_ab, &[dim_a, dim_b], transposed_factor)?;
    spectral_eigenvalues_supported_blocks(&partial_transpose, interner)
        .map_err(NegativityError::from)
}

/// Compute logarithmic negativity `log(1 + 2 N(ρ))` from the exact bipartite negativity.
pub fn logarithmic_negativity_bipartite(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    transposed_factor: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, NegativityError> {
    let negativity = negativity_bipartite(rho_ab, dim_a, dim_b, transposed_factor, interner)?;
    Ok(simplify_expr(expr_log(
        Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![Expr::Int(2.into()), negativity]),
        ]),
        interner,
    )))
}

/// Compute the bipartite Renyi-2 mutual information `S2(ρ_A) + S2(ρ_B) - S2(ρ_AB)`.
pub fn renyi2_mutual_information_bipartite(
    rho_ab: &[Vec<Expr>],
    dim_a: usize,
    dim_b: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, QmLinearAlgebraError> {
    let rho_a = try_partial_trace(
        rho_ab,
        BipartiteDims { dim_a, dim_b },
        PartialTraceTarget::B,
    )?;
    let rho_b = try_partial_trace(
        rho_ab,
        BipartiteDims { dim_a, dim_b },
        PartialTraceTarget::A,
    )?;
    let s2_a = renyi2_entropy(&rho_a, interner).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            QmLinearAlgebraError::NonSquareMatrix { rows, cols }
        }
    })?;
    let s2_b = renyi2_entropy(&rho_b, interner).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            QmLinearAlgebraError::NonSquareMatrix { rows, cols }
        }
    })?;
    let s2_ab = renyi2_entropy(rho_ab, interner).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            QmLinearAlgebraError::NonSquareMatrix { rows, cols }
        }
    })?;
    Ok(simplify_expr(Expr::add(vec![s2_a, s2_b, Expr::neg(s2_ab)])))
}

/// Compute the Renyi-2 entropy of the reduced state obtained by keeping one tensor factor.
///
/// The input density matrix is interpreted in row-major lexicographic order induced by
/// `factor_dims`. The subsystem indexed by `kept_factor` is retained, and every other factor is
/// traced out exactly.
pub fn renyi2_entropy_factor(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    kept_factor: usize,
    interner: &ax_ir::Interner,
) -> Result<Expr, CompositeSpaceError> {
    validate_factor_index(factor_dims, kept_factor)?;
    renyi2_entropy_factors_kept(rho, factor_dims, &[kept_factor], interner)
}

/// Compute the Renyi-2 entropy of the reduced state obtained by keeping an arbitrary factor subset.
///
/// The factors listed in `kept_factors` are preserved in their original relative order from
/// `factor_dims`, while all complementary factors are traced out. Repeated entries are rejected.
/// An empty `kept_factors` list denotes the scalar reduced state on the trivial subsystem.
pub fn renyi2_entropy_factors_kept(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    kept_factors: &[usize],
    interner: &ax_ir::Interner,
) -> Result<Expr, CompositeSpaceError> {
    validate_composite_space_matrix(rho, factor_dims)?;

    let mut kept_mask = vec![false; factor_dims.len()];
    let mut kept_order = Vec::with_capacity(kept_factors.len());
    for &factor in kept_factors {
        validate_factor_index(factor_dims, factor)?;
        if std::mem::replace(&mut kept_mask[factor], true) {
            return Err(CompositeSpaceError::DuplicatePermutationEntry { value: factor });
        }
        kept_order.push(factor);
    }

    let traced_factors = (0..factor_dims.len())
        .filter(|&factor| !kept_mask[factor])
        .collect::<Vec<_>>();
    let kept_dims = kept_order
        .iter()
        .map(|&factor| factor_dims[factor])
        .collect::<Vec<_>>();
    let traced_dims = traced_factors
        .iter()
        .map(|&factor| factor_dims[factor])
        .collect::<Vec<_>>();
    let kept_total = kept_dims.iter().product::<usize>();
    let traced_total = traced_dims.iter().product::<usize>();
    let mut reduced = vec![vec![Expr::zero(); kept_total]; kept_total];

    for out_row in 0..kept_total {
        let kept_row_multi = multi_index_from_linear(out_row, &kept_dims);
        for out_col in 0..kept_total {
            let kept_col_multi = multi_index_from_linear(out_col, &kept_dims);
            let terms = (0..traced_total)
                .map(|traced_linear| {
                    let traced_multi = multi_index_from_linear(traced_linear, &traced_dims);
                    let mut full_row = vec![0usize; factor_dims.len()];
                    let mut full_col = vec![0usize; factor_dims.len()];

                    for (cursor, &factor) in kept_order.iter().enumerate() {
                        full_row[factor] = kept_row_multi[cursor];
                        full_col[factor] = kept_col_multi[cursor];
                    }
                    for (cursor, &factor) in traced_factors.iter().enumerate() {
                        full_row[factor] = traced_multi[cursor];
                        full_col[factor] = traced_multi[cursor];
                    }

                    let row_index = linear_index_from_multi(&full_row, factor_dims);
                    let col_index = linear_index_from_multi(&full_col, factor_dims);
                    rho[row_index][col_index].clone()
                })
                .collect::<Vec<_>>();
            reduced[out_row][out_col] = simplify_expr(Expr::add(terms));
        }
    }

    renyi2_entropy(&reduced, interner).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            CompositeSpaceError::NonSquareMatrix { rows, cols }
        }
    })
}

/// Compute the tripartite Renyi-2 information
/// `I_3(A:B:C) = S2(A) + S2(B) + S2(C) - S2(AB) - S2(AC) - S2(BC) + S2(ABC)`.
pub fn renyi2_tripartite_information(
    rho_abc: &[Vec<Expr>],
    dims: [usize; 3],
    interner: &ax_ir::Interner,
) -> Result<Expr, CompositeSpaceError> {
    let s2_a = renyi2_entropy_factor(rho_abc, &dims, 0, interner)?;
    let s2_b = renyi2_entropy_factor(rho_abc, &dims, 1, interner)?;
    let s2_c = renyi2_entropy_factor(rho_abc, &dims, 2, interner)?;
    let s2_ab = renyi2_entropy_factors_kept(rho_abc, &dims, &[0, 1], interner)?;
    let s2_ac = renyi2_entropy_factors_kept(rho_abc, &dims, &[0, 2], interner)?;
    let s2_bc = renyi2_entropy_factors_kept(rho_abc, &dims, &[1, 2], interner)?;
    let s2_abc = renyi2_entropy(rho_abc, interner).map_err(|err| match err {
        StateFunctionalError::StateNotSquare { rows, cols } => {
            CompositeSpaceError::NonSquareMatrix { rows, cols }
        }
    })?;

    Ok(simplify_expr(Expr::add(vec![
        s2_a,
        s2_b,
        s2_c,
        Expr::neg(s2_ab),
        Expr::neg(s2_ac),
        Expr::neg(s2_bc),
        s2_abc,
    ])))
}

/// Compute the Bloch-vector components `[x, y, z]` for a `2x2` density matrix.
pub fn bloch_vector(rho: &[Vec<Expr>]) -> Result<[Expr; 3], QubitStateError> {
    let (rows, cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if rows != 2 || cols != 2 {
        return Err(QubitStateError::NotTwoByTwo { rows, cols });
    }
    let interner = ax_ir::Interner::new();
    let sigma_x = pauli_x(&interner);
    let sigma_y = pauli_y(&interner);
    let sigma_z = pauli_z(&interner);
    let x = expectation_value(&sigma_x, rho)
        .map_err(|_| QubitStateError::NotTwoByTwo { rows, cols })?;
    let y = expectation_value(&sigma_y, rho)
        .map_err(|_| QubitStateError::NotTwoByTwo { rows, cols })?;
    let z = expectation_value(&sigma_z, rho)
        .map_err(|_| QubitStateError::NotTwoByTwo { rows, cols })?;
    Ok([x, y, z])
}

/// Construct the qubit density matrix `1/2 (I + x σ_x + y σ_y + z σ_z)` from a Bloch vector.
pub fn qubit_density_from_bloch(r: [Expr; 3]) -> Vec<Vec<Expr>> {
    let interner = ax_ir::Interner::new();
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    let identity = vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::one()],
    ];
    let sigma_x = pauli_x(&interner);
    let sigma_y = pauli_y(&interner);
    let sigma_z = pauli_z(&interner);
    let x_term = ax_linalg::mat_scale(&r[0], &sigma_x);
    let y_term = ax_linalg::mat_scale(&r[1], &sigma_y);
    let z_term = ax_linalg::mat_scale(&r[2], &sigma_z);
    let sum = ax_linalg::mat_add(
        &ax_linalg::mat_add(&identity, &x_term),
        &ax_linalg::mat_add(&y_term, &z_term),
    );
    simplify_matrix(ax_linalg::mat_scale(&half, &sum))
}

/// Compute the normalized post-measurement state `ρ_i = P_i ρ P_i / p_i`.
pub fn post_measurement_state(
    projector: &[Vec<Expr>],
    rho: &[Vec<Expr>],
    outcome_index: usize,
) -> Result<Vec<Vec<Expr>>, MeasurementError> {
    let dim = rho.len();
    validate_square_state_dimension(rho, dim)?;
    validate_projector_set(&[projector.to_vec()], dim)?;
    let probability = measurement_probabilities(&[projector.to_vec()], rho)?
        .into_iter()
        .next()
        .unwrap_or_else(Expr::zero);
    if is_zero_expr(&probability) {
        return Err(MeasurementError::ZeroProbabilityOutcome {
            index: outcome_index,
        });
    }

    let interner = ax_ir::Interner::new();
    let left = ax_linalg::mat_mul(projector, rho, &interner);
    let numerator = ax_linalg::mat_mul(&left, projector, &interner);
    let inv_probability = Expr::pow(probability, Expr::Int((-1).into()));
    Ok(simplify_matrix(ax_linalg::mat_scale(
        &inv_probability,
        &numerator,
    )))
}

/// Vectorize a density matrix in canonical column-major order.
///
/// For a matrix `rho`, this returns
/// `[rho_00, rho_10, ..., rho_(d-1)0, rho_01, rho_11, ..., rho_(d-1)(d-1)]`.
pub fn vec_column_major_density(rho: &[Vec<Expr>]) -> Vec<Expr> {
    let cols = rho.first().map(|row| row.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(rho.len().saturating_mul(cols));
    for col in 0..cols {
        for row in rho {
            if let Some(entry) = row.get(col) {
                out.push(entry.clone());
            }
        }
    }
    out
}

/// Construct the finite-dimensional Lindbladian as a superoperator acting on
/// the column-major vectorization `vec(rho)`.
///
/// For a Hilbert-space dimension `d`, the returned superoperator has shape
/// `d^2 x d^2`. It is built exactly by applying `lindblad_rhs` to each
/// matrix-unit basis density operator `E_ij`, vectorizing the result in
/// column-major order, and placing that vector in the corresponding
/// superoperator column.
pub fn lindbladian_superoperator(
    h: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, LindbladianSpectrumError> {
    let dim = validate_lindblad_square_matrix(h, "Hamiltonian")?;
    validate_lindblad_jump_ops(jump_ops, dim)?;

    let super_dim = dim.saturating_mul(dim);
    let mut superoperator = vec![vec![Expr::zero(); super_dim]; super_dim];

    for basis_col in 0..dim {
        for basis_row in 0..dim {
            let mut basis_density = zero_matrix(dim);
            basis_density[basis_row][basis_col] = Expr::one();
            let rhs = lindblad_rhs(h, &basis_density, jump_ops, interner)?;
            let column = vec_column_major_density(&rhs);
            let super_col = basis_col * dim + basis_row;
            for (super_row, entry) in column.into_iter().enumerate() {
                superoperator[super_row][super_col] = entry;
            }
        }
    }

    Ok(superoperator)
}

/// Construct a sparse numeric Lindbladian superoperator for finite-dimensional inputs.
///
/// This helper first builds the exact dense superoperator using
/// [`lindbladian_superoperator`], then converts its entries into numeric complex
/// pairs and finally drops explicit zeros when emitting COO entries.
pub fn sparse_lindbladian_superoperator_numeric(
    h: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<ax_linalg::SparseMatrixCoo<(f64, f64)>, LindbladianSpectrumError> {
    let superoperator = lindbladian_superoperator(h, jump_ops, interner)?;
    let numeric_superoperator = dense_numeric_matrix_from_expr(&superoperator)?;
    ax_linalg::dense_to_sparse_coo(&numeric_superoperator, &(0.0, 0.0)).map_err(Into::into)
}

/// Return exact Lindbladian eigenvalues for explicitly supported low-dimensional cases.
///
/// Supported cases:
/// - `d = 1`
/// - `d = 2` only when the constructed `4x4` Lindbladian superoperator is diagonal
///
/// All other cases return `UnsupportedDimension { dim }`, where `dim` is the
/// Hilbert-space dimension rather than the superoperator dimension.
pub fn lindbladian_eigenvalues_small(
    h: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, LindbladianSpectrumError> {
    let dim = validate_lindblad_square_matrix(h, "Hamiltonian")?;
    validate_lindblad_jump_ops(jump_ops, dim)?;

    let superoperator = lindbladian_superoperator(h, jump_ops, interner)?;
    match dim {
        1 => Ok(vec![superoperator[0][0].clone()]),
        2 => diagonal_entries(&superoperator)
            .ok_or(LindbladianSpectrumError::UnsupportedDimension { dim }),
        _ => Err(LindbladianSpectrumError::UnsupportedDimension { dim }),
    }
}

/// Construct the finite-dimensional Lindblad right-hand side
/// `ρ̇ = -i [H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})`.
pub fn lindblad_rhs(
    h: &[Vec<Expr>],
    rho: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, LindbladError> {
    let dim = validate_lindblad_square_matrix(h, "Hamiltonian")?;
    let rho_dim = validate_lindblad_square_matrix(rho, "state")?;
    if rho_dim != dim {
        return Err(LindbladError::DimensionMismatch {
            expected: dim,
            actual: rho_dim,
            which: "state",
        });
    }
    validate_lindblad_jump_ops(jump_ops, dim)?;

    let coherent = simplify_matrix(ax_linalg::mat_scale(
        &Expr::neg(imag_unit()),
        &commutator(h, rho, interner),
    ));

    let mut dissipator = zero_matrix(dim);
    for jump in jump_ops {
        let jump_dagger = adjoint_matrix(jump);
        let jump_rho = ax_linalg::mat_mul(jump, rho, interner);
        let gain = ax_linalg::mat_mul(&jump_rho, &jump_dagger, interner);
        let jump_norm = ax_linalg::mat_mul(&jump_dagger, jump, interner);
        let loss = ax_linalg::mat_scale(&half(), &anticommutator(&jump_norm, rho, interner));
        let term = ax_linalg::mat_add(&gain, &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &loss));
        dissipator = simplify_matrix(ax_linalg::mat_add(&dissipator, &term));
    }

    Ok(simplify_matrix(ax_linalg::mat_add(&coherent, &dissipator)))
}

/// Convert a multi-index in row-major tensor-product order into a flattened linear index.
pub fn linear_index_from_multi(indices: &[usize], dims: &[usize]) -> usize {
    indices
        .iter()
        .zip(dims.iter())
        .fold(0usize, |acc, (index, dim)| {
            acc.saturating_mul(*dim).saturating_add(*index)
        })
}

/// Convert a flattened linear index into its row-major tensor-product multi-index.
pub fn multi_index_from_linear(index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }

    let mut remaining = index;
    let mut out = vec![0; dims.len()];
    for pos in (0..dims.len()).rev() {
        let dim = dims[pos];
        if dim == 0 {
            out[pos] = 0;
        } else {
            out[pos] = remaining % dim;
            remaining /= dim;
        }
    }
    out
}

fn validate_composite_space_matrix(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
) -> Result<(), CompositeSpaceError> {
    if factor_dims.is_empty() {
        return Err(CompositeSpaceError::EmptyFactorList);
    }

    let rows = rho.len();
    let cols = rho.first().map(|row| row.len()).unwrap_or(0);
    if rho.iter().any(|row| row.len() != cols) || rows != cols {
        return Err(CompositeSpaceError::NonSquareMatrix { rows, cols });
    }

    let expected = factor_dims.iter().product::<usize>();
    if rows != expected {
        return Err(CompositeSpaceError::TotalDimensionMismatch {
            expected,
            actual: rows,
        });
    }

    Ok(())
}

fn validate_factor_index(
    factor_dims: &[usize],
    factor_index: usize,
) -> Result<(), CompositeSpaceError> {
    if factor_index >= factor_dims.len() {
        return Err(CompositeSpaceError::InvalidFactorIndex {
            index: factor_index,
            factor_count: factor_dims.len(),
        });
    }
    Ok(())
}

fn validate_permutation(
    factor_dims: &[usize],
    permutation: &[usize],
) -> Result<(), CompositeSpaceError> {
    if permutation.len() != factor_dims.len() {
        return Err(CompositeSpaceError::InvalidPermutationLength {
            expected: factor_dims.len(),
            actual: permutation.len(),
        });
    }

    let mut seen = vec![false; factor_dims.len()];
    for &value in permutation {
        if value >= factor_dims.len() {
            return Err(CompositeSpaceError::InvalidPermutationEntry {
                value,
                factor_count: factor_dims.len(),
            });
        }
        if std::mem::replace(&mut seen[value], true) {
            return Err(CompositeSpaceError::DuplicatePermutationEntry { value });
        }
    }

    Ok(())
}

/// Trace out one tensor factor from a density matrix over an arbitrary ordered factorization.
///
/// The output preserves the original relative order of all remaining factors. When the input has
/// a single factor, tracing that factor returns the `1x1` matrix whose only entry is `Tr(rho)`.
pub fn try_partial_trace_factor(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    traced_factor: usize,
) -> Result<Vec<Vec<Expr>>, CompositeSpaceError> {
    validate_composite_space_matrix(rho, factor_dims)?;
    validate_factor_index(factor_dims, traced_factor)?;

    if factor_dims.len() == 1 {
        return Ok(vec![vec![simplify_expr(ax_linalg::trace(rho))]]);
    }

    let traced_dim = factor_dims[traced_factor];
    let remaining_dims = factor_dims
        .iter()
        .enumerate()
        .filter_map(|(idx, dim)| (idx != traced_factor).then_some(*dim))
        .collect::<Vec<_>>();
    let remaining_total = remaining_dims.iter().product::<usize>();
    let mut out = vec![vec![Expr::zero(); remaining_total]; remaining_total];

    for out_row in 0..remaining_total {
        let row_multi = multi_index_from_linear(out_row, &remaining_dims);
        for out_col in 0..remaining_total {
            let col_multi = multi_index_from_linear(out_col, &remaining_dims);
            let terms = (0..traced_dim)
                .map(|traced_index| {
                    let mut full_row = Vec::with_capacity(factor_dims.len());
                    let mut full_col = Vec::with_capacity(factor_dims.len());
                    let mut row_cursor = 0usize;
                    let mut col_cursor = 0usize;
                    for factor_idx in 0..factor_dims.len() {
                        if factor_idx == traced_factor {
                            full_row.push(traced_index);
                            full_col.push(traced_index);
                        } else {
                            full_row.push(row_multi[row_cursor]);
                            full_col.push(col_multi[col_cursor]);
                            row_cursor += 1;
                            col_cursor += 1;
                        }
                    }
                    let row_index = linear_index_from_multi(&full_row, factor_dims);
                    let col_index = linear_index_from_multi(&full_col, factor_dims);
                    rho[row_index][col_index].clone()
                })
                .collect::<Vec<_>>();
            out[out_row][out_col] = simplify_expr(Expr::add(terms));
        }
    }

    Ok(out)
}

/// Partially transpose one tensor factor of a density matrix in lexicographic product order.
///
/// The basis ordering is the row-major lexicographic order induced by `factor_dims`. Only the
/// selected subsystem index is transposed, meaning that the chosen factor's bra and ket digits are
/// swapped while every other subsystem index is left unchanged.
pub fn try_partial_transpose_factor(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    transposed_factor: usize,
) -> Result<Vec<Vec<Expr>>, CompositeSpaceError> {
    validate_composite_space_matrix(rho, factor_dims)?;
    validate_factor_index(factor_dims, transposed_factor)?;

    let total_dim = factor_dims.iter().product::<usize>();
    let mut out = vec![vec![Expr::zero(); total_dim]; total_dim];

    for out_row in 0..total_dim {
        let row_multi = multi_index_from_linear(out_row, factor_dims);
        for out_col in 0..total_dim {
            let col_multi = multi_index_from_linear(out_col, factor_dims);
            let mut source_row = row_multi.clone();
            let mut source_col = col_multi.clone();
            source_row[transposed_factor] = col_multi[transposed_factor];
            source_col[transposed_factor] = row_multi[transposed_factor];
            let source_row_index = linear_index_from_multi(&source_row, factor_dims);
            let source_col_index = linear_index_from_multi(&source_col, factor_dims);
            out[out_row][out_col] = rho[source_row_index][source_col_index].clone();
        }
    }

    Ok(out)
}

/// Permute tensor-product subsystems by exact basis relabeling in lexicographic product order.
///
/// The `permutation` slice specifies the new factor order: output factor `i` is input factor
/// `permutation[i]`. Both row and column multi-indices are reordered by that same relabeling, and
/// matrix entries are copied exactly without algebraic modification.
pub fn try_permute_subsystems(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    permutation: &[usize],
) -> Result<Vec<Vec<Expr>>, CompositeSpaceError> {
    validate_composite_space_matrix(rho, factor_dims)?;
    validate_permutation(factor_dims, permutation)?;

    let permuted_dims = permutation
        .iter()
        .map(|&index| factor_dims[index])
        .collect::<Vec<_>>();
    let total_dim = factor_dims.iter().product::<usize>();
    let mut out = vec![vec![Expr::zero(); total_dim]; total_dim];

    for source_row in 0..total_dim {
        let row_multi = multi_index_from_linear(source_row, factor_dims);
        let permuted_row = permutation
            .iter()
            .map(|&index| row_multi[index])
            .collect::<Vec<_>>();
        let target_row = linear_index_from_multi(&permuted_row, &permuted_dims);
        for source_col in 0..total_dim {
            let col_multi = multi_index_from_linear(source_col, factor_dims);
            let permuted_col = permutation
                .iter()
                .map(|&index| col_multi[index])
                .collect::<Vec<_>>();
            let target_col = linear_index_from_multi(&permuted_col, &permuted_dims);
            out[target_row][target_col] = rho[source_row][source_col].clone();
        }
    }

    Ok(out)
}

pub fn try_partial_trace(
    rho: &[Vec<Expr>],
    dims: BipartiteDims,
    target: PartialTraceTarget,
) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    let traced_factor = match target {
        PartialTraceTarget::A => 0,
        PartialTraceTarget::B => 1,
    };
    try_partial_trace_factor(rho, &[dims.dim_a, dims.dim_b], traced_factor).map_err(|err| match err
    {
        CompositeSpaceError::EmptyFactorList | CompositeSpaceError::InvalidFactorIndex { .. } => {
            QmLinearAlgebraError::InvalidTraceTarget { target: '?' }
        }
        CompositeSpaceError::InvalidPermutationLength { .. }
        | CompositeSpaceError::InvalidPermutationEntry { .. }
        | CompositeSpaceError::DuplicatePermutationEntry { .. } => {
            QmLinearAlgebraError::InvalidTraceTarget { target: '?' }
        }
        CompositeSpaceError::NonSquareMatrix { rows, cols } => {
            QmLinearAlgebraError::NonSquareMatrix { rows, cols }
        }
        CompositeSpaceError::TotalDimensionMismatch { expected, actual } => {
            QmLinearAlgebraError::SubsystemDimensionMismatch { expected, actual }
        }
    })
}

/// Join (contract) a product of gamma matrices.
///
/// gamma(a) * gamma(b) → gamma(a, b) + g(a+, b+)
/// gamma(a) * gamma(b, c) → gamma(a, b, c) + g(a+, b+) * gamma(c) - g(a+, c+) * gamma(b)
///
/// Uses the recursive contraction identity.
pub fn join_gamma_pair(
    indices1: &[lasso::Spur],
    indices2: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    if indices1.is_empty() {
        return make_gamma(indices2, gamma_sym);
    }
    if indices2.is_empty() {
        return make_gamma(indices1, gamma_sym);
    }

    let a1 = indices1[0];
    let rest1 = &indices1[1..];

    if rest1.is_empty() {
        join_single_with_multi(a1, indices2, gamma_sym, metric_sym, interner)
    } else {
        let inner = join_gamma_pair(rest1, indices2, gamma_sym, metric_sym, interner);
        join_single_with_expr(a1, &inner, gamma_sym, metric_sym, interner)
    }
}

fn join_single_with_multi(
    a: lasso::Spur,
    bs: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    _interner: &ax_ir::Interner,
) -> Expr {
    let mut terms = Vec::new();

    // First term: gamma with all indices combined
    let mut all = vec![a];
    all.extend_from_slice(bs);
    terms.push(make_gamma(&all, gamma_sym));

    // Contraction terms: Σ_k (-1)^k g^{a b_k} γ^{bs \ b_k}
    for k in 0..bs.len() {
        let metric = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: bs[k],
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );

        let remaining: Vec<lasso::Spur> = bs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .map(|(_, &b)| b)
            .collect();

        let gamma_part = if remaining.is_empty() {
            Expr::one()
        } else {
            make_gamma(&remaining, gamma_sym)
        };

        let term = Expr::mul(vec![metric, gamma_part]);
        if k % 2 == 0 {
            terms.push(term);
        } else {
            terms.push(Expr::neg(term));
        }
    }

    Expr::add(terms)
}

fn join_single_with_expr(
    a: lasso::Spur,
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_single_with_expr(a, t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            // Find the first gamma factor and join with it
            for (i, factor) in factors.iter().enumerate() {
                if let Expr::Call(f, args) = factor {
                    if *f == gamma_sym {
                        let gamma_indices: Vec<lasso::Spur> = args
                            .iter()
                            .filter_map(|arg| {
                                if let Expr::Sym(s) = arg {
                                    Some(*s)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let joined = join_single_with_multi(
                            a,
                            &gamma_indices,
                            gamma_sym,
                            metric_sym,
                            interner,
                        );
                        let mut rest: Vec<Expr> = factors
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, f)| f.clone())
                            .collect();
                        rest.push(joined);
                        return Expr::mul(rest);
                    }
                }
            }
            // No gamma factor found — prepend a single-index gamma
            let mut new_factors = vec![make_gamma(&[a], gamma_sym)];
            new_factors.extend(factors.iter().cloned());
            Expr::mul(new_factors)
        }
        Expr::Neg(e) => Expr::neg(join_single_with_expr(a, e, gamma_sym, metric_sym, interner)),
        _ => {
            // Check if expr itself is a gamma call
            if let Expr::Call(f, args) = expr {
                if *f == gamma_sym {
                    let gamma_indices: Vec<lasso::Spur> = args
                        .iter()
                        .filter_map(|arg| {
                            if let Expr::Sym(s) = arg {
                                Some(*s)
                            } else {
                                None
                            }
                        })
                        .collect();
                    return join_single_with_multi(
                        a,
                        &gamma_indices,
                        gamma_sym,
                        metric_sym,
                        interner,
                    );
                }
            }
            Expr::mul(vec![make_gamma(&[a], gamma_sym), expr.clone()])
        }
    }
}

fn make_gamma(indices: &[lasso::Spur], gamma_sym: lasso::Spur) -> Expr {
    Expr::Call(gamma_sym, indices.iter().map(|&i| Expr::Sym(i)).collect())
}

/// Extract gamma indices from a `gamma(a, b, ...)` Call expression.
fn gamma_indices(args: &[Expr]) -> Vec<lasso::Spur> {
    args.iter()
        .filter_map(|arg| {
            if let Expr::Sym(s) = arg {
                Some(*s)
            } else {
                None
            }
        })
        .collect()
}

/// Walk an expression and join all adjacent gamma-matrix Call nodes in products.
pub fn join_gammas_in_expr(
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Recursively process each factor first
            let factors: Vec<Expr> = factors
                .iter()
                .map(|f| join_gammas_in_expr(f, gamma_sym, metric_sym, interner))
                .collect();

            // Now fold adjacent gamma pairs left-to-right
            let mut result: Vec<Expr> = Vec::new();
            for factor in factors {
                if let Some(last) = result.last() {
                    if let (Expr::Call(f1, a1), Expr::Call(f2, a2)) = (last, &factor) {
                        if *f1 == gamma_sym && *f2 == gamma_sym {
                            let i1 = gamma_indices(a1);
                            let i2 = gamma_indices(a2);
                            let joined = join_gamma_pair(&i1, &i2, gamma_sym, metric_sym, interner);
                            result.pop();
                            // The joined expression may be an Add — wrap in a group
                            // by pushing the whole joined expression, then distributing
                            // remaining factors over it at the end.
                            result.push(joined);
                            continue;
                        }
                    }
                }
                result.push(factor);
            }

            if result.len() == 1 {
                result.remove(0)
            } else {
                Expr::mul(result)
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_gammas_in_expr(t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(join_gammas_in_expr(e, gamma_sym, metric_sym, interner)),
        _ => expr.clone(),
    }
}

fn structured_spinor_family(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    property_sym(expr)
        .and_then(|sym| declared_spinor_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.index_family)
}

fn structured_gamma_family(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    property_sym(expr)
        .and_then(|sym| declared_gamma_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.index_family)
}

fn gamma_declared_dimension(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    property_sym(expr)
        .and_then(|sym| declared_gamma_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.dimension)
}

fn gamma_effective_dimension(
    gam1: &Expr,
    g1: &GammaExprData,
    gam2: &Expr,
    g2: &GammaExprData,
    dimension: Option<usize>,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    gamma_dimension_of_expr(gam1, properties)
        .or_else(|| gamma_dimension_of_expr(gam2, properties))
        .or_else(|| {
            g1.indices
                .iter()
                .chain(g2.indices.iter())
                .filter_map(|idx| index_family_dimension(idx, properties))
                .max()
        })
        .or(dimension)
}

fn gamma_conventions_compatible(
    gam1: &Expr,
    gam2: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> bool {
    if gamma_convention_conflict_of_expr(gam1, properties)
        || gamma_convention_conflict_of_expr(gam2, properties)
    {
        return false;
    }

    match (
        gamma_convention_of_expr(gam1, properties),
        gamma_convention_of_expr(gam2, properties),
    ) {
        (Some(lhs), Some(rhs)) => lhs.signature == rhs.signature && lhs.clifford == rhs.clifford,
        _ => true,
    }
}

fn gamma_contraction_sign(
    expr: &Expr,
    contraction_count: usize,
    properties: &dyn ax_tensor::PropertyLookup,
) -> i32 {
    match gamma_clifford_sign(expr, properties) {
        Some(CliffordConvention::MinusTwoG) if contraction_count % 2 == 1 => -1,
        _ => 1,
    }
}

fn gamma_effective_families(
    expr: &Expr,
    data: &GammaExprData,
    properties: &dyn ax_tensor::PropertyLookup,
) -> HashSet<lasso::Spur> {
    let mut families = HashSet::new();
    if let Some(family) = structured_gamma_family(expr, properties) {
        families.insert(family);
    }
    families.extend(
        data.indices
            .iter()
            .filter_map(|idx| index_family_name(idx, properties)),
    );
    families
}

fn gamma_indices_have_duplicate_in_same_family(
    indices: &[Index],
    properties: &dyn ax_tensor::PropertyLookup,
) -> bool {
    let mut seen = HashSet::new();
    for idx in indices {
        let key = (idx.name, index_family_name(idx, properties));
        if !seen.insert(key) {
            return true;
        }
    }
    false
}

fn structured_diracbar_metadata(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    property_sym(expr).and_then(|sym| declared_diracbar_metadata_of_symbol(sym, properties))
}

fn sort_spinor_metadata_error(
    expr: &Expr,
    bar_factor: &Expr,
    gamma_factor: Option<&Expr>,
    left_spinor: &Expr,
    right_spinor: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let left_family = structured_spinor_family(left_spinor, properties);
    let right_family = structured_spinor_family(right_spinor, properties);

    if let (Some(left), Some(right)) = (left_family, right_family) {
        if left != right {
            return Some(qm_error_expr(
                "sort_spinors_spinor_family_mismatch",
                expr,
                interner,
            ));
        }
    }

    if let Some(metadata) = structured_diracbar_metadata(bar_factor, properties) {
        if let Some(expected_gamma_symbol) = metadata.gamma_symbol {
            if gamma_factor
                .and_then(property_sym)
                .is_some_and(|actual| actual != expected_gamma_symbol)
            {
                return Some(qm_error_expr(
                    "sort_spinors_gamma_family_mismatch",
                    expr,
                    interner,
                ));
            }
        }
        if let Some(expected_spinor_family) = metadata.spinor_family {
            for actual in [left_family, right_family].into_iter().flatten() {
                if actual != expected_spinor_family {
                    return Some(qm_error_expr(
                        "sort_spinors_spinor_family_mismatch",
                        expr,
                        interner,
                    ));
                }
            }
        }
    }

    if let Some(gamma_factor) = gamma_factor {
        if let Some(gamma_family) = structured_gamma_family(gamma_factor, properties) {
            for actual in [left_family, right_family].into_iter().flatten() {
                if actual != gamma_family {
                    return Some(qm_error_expr(
                        "sort_spinors_gamma_family_mismatch",
                        expr,
                        interner,
                    ));
                }
            }
        }
    }

    None
}

pub fn sort_spinors(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let mapped = factors
                .iter()
                .map(|factor| sort_spinors(factor, properties, interner))
                .collect::<Vec<_>>();

            let mut out = mapped.clone();
            for i in 0..mapped.len() {
                let bar_factor = &mapped[i];
                let Expr::Call(bar_sym, args) = bar_factor else {
                    continue;
                };
                if diracbar_metadata_of_expr(&Expr::Sym(*bar_sym), properties).is_none() {
                    continue;
                }
                if args.len() != 1 || !is_majorana_spinor_expr(&args[0], properties) {
                    continue;
                }
                let left_spinor = args[0].clone();
                let left_sym = match property_sym(&left_spinor) {
                    Some(sym) => sym,
                    None => continue,
                };

                let mut gamma_pos = None;
                let mut spinor_pos = None;
                for j in i + 1..mapped.len() {
                    let candidate = &mapped[j];
                    if diracbar_metadata_of_expr(candidate, properties).is_some() {
                        break;
                    }
                    if gamma_metadata_of_expr(candidate, properties).is_some() {
                        if gamma_pos.is_some() {
                            return qm_error_expr("sort_spinors_join_gamma_first", expr, interner);
                        }
                        gamma_pos = Some(j);
                        continue;
                    }
                    if spinor_metadata_of_expr(candidate, properties).is_some() {
                        spinor_pos = Some(j);
                        break;
                    }
                    if !matches!(candidate, Expr::Int(_) | Expr::Rational(_) | Expr::Float(_)) {
                        break;
                    }
                }

                let Some(j) = spinor_pos else {
                    continue;
                };
                let right_spinor = mapped[j].clone();
                if let Some(error) = sort_spinor_metadata_error(
                    expr,
                    bar_factor,
                    gamma_pos.map(|pos| &mapped[pos]),
                    &left_spinor,
                    &right_spinor,
                    properties,
                    interner,
                ) {
                    return error;
                }
                if !is_majorana_spinor_expr(&right_spinor, properties) {
                    return qm_error_expr("sort_spinors_second_not_majorana", expr, interner);
                }
                let right_sym = match property_sym(&right_spinor) {
                    Some(sym) => sym,
                    None => continue,
                };
                let Some(order) = prop_sort_order(left_sym, properties) else {
                    continue;
                };
                let Some(pos_left) = order.iter().position(|sym| *sym == left_sym) else {
                    continue;
                };
                let Some(pos_right) = order.iter().position(|sym| *sym == right_sym) else {
                    continue;
                };
                if pos_left <= pos_right {
                    continue;
                }

                let gamma_rank = gamma_pos
                    .and_then(|pos| {
                        gamma_expr_data(&mapped[pos], properties).map(|data| data.indices.len())
                    })
                    .unwrap_or(0);
                let majorana_sign = if ((gamma_rank * (gamma_rank + 1)) / 2) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let comparison =
                    ax_tensor::subtree_compare(&left_spinor, &right_spinor, properties, interner);
                let swap_sign = ax_tensor::can_swap(
                    &left_spinor,
                    &right_spinor,
                    comparison,
                    properties,
                    interner,
                    false,
                );
                if swap_sign == 0 {
                    continue;
                }
                let total_sign = majorana_sign * swap_sign;
                out[i] = Expr::Call(*bar_sym, vec![right_spinor.clone()]);
                out[j] = left_spinor.clone();
                let reordered = Expr::mul(out);
                return if total_sign < 0 {
                    Expr::neg(reordered)
                } else {
                    reordered
                };
            }

            Expr::mul(mapped)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| sort_spinors(term, properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(sort_spinors(inner, properties, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            sort_spinors(base, properties, interner),
            sort_spinors(exp, properties, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(sort_spinors(re, properties, interner)),
            Box::new(sort_spinors(im, properties, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| sort_spinors(arg, properties, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(sort_spinors(base, properties, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(sort_spinors(inner, properties, interner)), *rel)
        }
        _ => expr.clone(),
    }
}

pub fn join_gamma_full(
    gam1: &Expr,
    gam2: &Expr,
    dimension: Option<usize>,
    expand: bool,
    use_generalised_delta: bool,
    metric: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let Some(g1) = gamma_expr_data(gam1, properties) else {
        return Expr::mul(vec![gam1.clone(), gam2.clone()]);
    };
    let Some(g2) = gamma_expr_data(gam2, properties) else {
        return Expr::mul(vec![gam1.clone(), gam2.clone()]);
    };

    if !gamma_conventions_compatible(gam1, gam2, properties) {
        return qm_error_expr(
            "join_gamma_convention_mismatch",
            &Expr::mul(vec![gam1.clone(), gam2.clone()]),
            interner,
        );
    }

    let rank1 = g1.indices.len();
    let rank2 = g2.indices.len();
    let dim = gamma_effective_dimension(gam1, &g1, gam2, &g2, dimension, properties);

    let families1 = gamma_effective_families(gam1, &g1, properties);
    let families2 = gamma_effective_families(gam2, &g2, properties);
    if !families1.is_empty() && !families2.is_empty() && families1 != families2 {
        return qm_error_expr(
            "join_gamma_family_mismatch",
            &Expr::mul(vec![gam1.clone(), gam2.clone()]),
            interner,
        );
    }

    let mut terms = Vec::new();
    let max_i = rank1.min(rank2);
    for i in 0..=max_i {
        let free_rank = rank1 + rank2 - 2 * i;
        if dim.is_some_and(|d| free_rank > d) {
            continue;
        }
        let coeff = BigRational::new(
            factorial(rank1) * factorial(rank2),
            factorial(rank1 - i) * factorial(rank2 - i) * factorial(i),
        );

        if i == 0 {
            let mut free = g1.indices.clone();
            free.extend(g2.indices.clone());
            let gamma = if free.is_empty() {
                Expr::one()
            } else if gamma_indices_have_duplicate_in_same_family(&free, properties) {
                Expr::zero()
            } else {
                build_gamma_expr(&g1.head, &free)
            };
            if gamma != Expr::zero() {
                terms.push(if coeff.is_one() {
                    gamma
                } else {
                    Expr::mul(vec![Expr::Rational(coeff), gamma])
                });
            }
            continue;
        }

        let left_choices = combinations_of(rank1, i);
        let right_choices = combinations_of(rank2, i);
        let mut contracted_terms = Vec::new();
        let clifford_contraction_sign = gamma_contraction_sign(gam1, i, properties);
        for left in &left_choices {
            for right in &right_choices {
                let left_sign = permutation_parity(left);
                let right_sign = permutation_parity(right);
                let mut free = g1
                    .indices
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, item)| (!left.contains(&idx)).then_some(item.clone()))
                    .collect::<Vec<_>>();
                free.extend(
                    g2.indices
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, item)| (!right.contains(&idx)).then_some(item.clone())),
                );
                let gamma_part = if free.is_empty() {
                    Expr::one()
                } else if gamma_indices_have_duplicate_in_same_family(&free, properties) {
                    Expr::zero()
                } else {
                    build_gamma_expr(&g1.head, &free)
                };
                let contraction = if use_generalised_delta {
                    let uppers = left
                        .iter()
                        .map(|idx| g1.indices[*idx].clone())
                        .collect::<Vec<_>>();
                    let lowers = right
                        .iter()
                        .map(|idx| g2.indices[*idx].clone())
                        .collect::<Vec<_>>();
                    build_generalised_delta(&uppers, &lowers, interner)
                } else {
                    let metrics = left
                        .iter()
                        .zip(right.iter())
                        .map(|(li, ri)| {
                            build_metric_contraction(metric, &g1.indices[*li], &g2.indices[*ri])
                        })
                        .collect::<Vec<_>>();
                    if metrics.is_empty() {
                        Expr::one()
                    } else {
                        Expr::mul(metrics)
                    }
                };
                if gamma_part == Expr::zero() {
                    continue;
                }
                let mut term = Expr::mul(vec![gamma_part, contraction]);
                if left_sign * right_sign * clifford_contraction_sign < 0 {
                    term = Expr::neg(term);
                }
                contracted_terms.push(term);
                if !expand {
                    break;
                }
            }
            if !expand {
                break;
            }
        }

        let contraction_sum = if contracted_terms.len() == 1 {
            contracted_terms.pop().unwrap()
        } else {
            Expr::add(contracted_terms)
        };
        if expand && !coeff.is_one() {
            match contraction_sum {
                Expr::Add(items) => {
                    terms.extend(
                        items
                            .into_iter()
                            .map(|item| Expr::mul(vec![Expr::Rational(coeff.clone()), item])),
                    );
                }
                other => terms.push(Expr::mul(vec![Expr::Rational(coeff), other])),
            }
        } else {
            terms.push(if coeff.is_one() {
                contraction_sum
            } else {
                Expr::mul(vec![Expr::Rational(coeff), contraction_sum])
            });
        }
    }

    if terms.is_empty() {
        Expr::zero()
    } else {
        Expr::add(terms)
    }
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute Fierz rearrangement coefficients for each antisymmetric gamma rank.
///
/// Returns a list of `(coefficient, rank)` pairs for ranks 0..=dim.
/// The coefficient for rank k is:
///   c_k = -(-1)^{k(k+1)/2} * C(d,k) / (k! * spinor_dim)
/// where spinor_dim = 2^(d/2).
pub fn fierz_coefficients(dim: usize) -> Vec<(num_rational::BigRational, usize)> {
    let spinor_dim = 1usize << (dim / 2); // 2^(d/2)
    let mut result = Vec::new();

    for k in 0..=dim {
        let sign = if (k * (k + 1) / 2) % 2 == 0 {
            1i64
        } else {
            -1i64
        };
        let binom = binomial(dim, k);
        let coeff = num_rational::BigRational::new(
            (sign * binom as i64).into(),
            (spinor_dim as i64).into(),
        );
        // Divide by k! for the normalisation of the antisymmetric gamma basis element
        let k_fact: i64 = (1..=k as i64).product();
        let final_coeff = num_rational::BigRational::new(
            coeff.numer().clone(),
            coeff.denom().clone() * num_bigint::BigInt::from(k_fact),
        );
        result.push((final_coeff, k));
    }

    // Overall minus sign from Fierz rearrangement
    for (c, _) in &mut result {
        *c = -c.clone();
    }

    result
}

fn gamma5_symbol(interner: &ax_ir::Interner) -> Expr {
    Expr::Sym(interner.get_or_intern("gamma5"))
}

fn convention_dimension_matches_spinor(
    spinor_meta: &ax_ir::SpinorMetadata,
    gamma_meta: &ax_ir::GammaConventionMetadata,
) -> bool {
    match (spinor_meta.dimension, gamma_meta.dimension) {
        (Some(spinor_dim), Some(gamma_dim)) => spinor_dim == gamma_dim,
        _ => false,
    }
}

/// Select the Fierz gamma-basis sectors implied by structured spinor and gamma metadata.
///
/// In four dimensions, unconstrained Dirac and Majorana spinors use the full
/// five-sector basis `1`, `gamma(mu)`, `sigma(mu,nu)`, `gamma(mu) gamma5`,
/// and `gamma5`. Explicitly chiral Weyl or Majorana-Weyl spinors use a reduced
/// chiral basis that omits scalar, pseudoscalar, and axial-vector sectors that
/// are not independent in a fixed chirality sector.
pub fn fierz_basis_for_metadata(
    spinor_meta: &ax_ir::SpinorMetadata,
    gamma_meta: Option<&ax_ir::GammaConventionMetadata>,
    interner: &ax_ir::Interner,
) -> Vec<Expr> {
    let Some(gamma_meta) = gamma_meta else {
        return Vec::new();
    };
    if !convention_dimension_matches_spinor(spinor_meta, gamma_meta)
        || spinor_meta.dimension != Some(4)
        || gamma_meta.gamma5.is_none()
    {
        return Vec::new();
    }

    let mu = Expr::Sym(interner.get_or_intern("mu"));
    let nu = Expr::Sym(interner.get_or_intern("nu"));
    let gamma_mu = gamma_factor(mu.clone(), interner);
    let sigma_munu = sigma_matrix(mu.clone(), nu, interner);
    let gamma5 = gamma5_symbol(interner);

    match &spinor_meta.class {
        SpinorClass::Dirac | SpinorClass::Majorana if spinor_meta.chirality.is_none() => vec![
            Expr::one(),
            gamma_mu.clone(),
            sigma_munu,
            Expr::mul(vec![gamma_mu, gamma5.clone()]),
            gamma5,
        ],
        SpinorClass::Weyl | SpinorClass::MajoranaWeyl if spinor_meta.chirality.is_some() => {
            vec![gamma_mu, sigma_munu]
        }
        SpinorClass::Dirac | SpinorClass::Majorana if spinor_meta.chirality.is_some() => {
            vec![gamma_mu, sigma_munu]
        }
        _ => Vec::new(),
    }
}

fn fierz_basis_rank(expr: &Expr, interner: &ax_ir::Interner) -> Option<usize> {
    if *expr == Expr::one() {
        return Some(0);
    }
    match expr {
        Expr::Call(sym, args) if interner.resolve(*sym) == "gamma" && args.len() == 1 => Some(1),
        Expr::Call(sym, args) if interner.resolve(*sym) == "sigma" && args.len() == 2 => Some(2),
        Expr::Sym(sym) if interner.resolve(*sym) == "gamma5" => Some(4),
        Expr::Mul(factors) if factors.len() == 2 => {
            let has_gamma = factors
                .iter()
                .any(|factor| matches!(factor, Expr::Call(sym, args) if interner.resolve(*sym) == "gamma" && args.len() == 1));
            let has_gamma5 = factors.iter().any(
                |factor| matches!(factor, Expr::Sym(sym) if interner.resolve(*sym) == "gamma5"),
            );
            (has_gamma && has_gamma5).then_some(3)
        }
        _ => None,
    }
}

/// Return Fierz coefficients for a metadata-selected basis.
///
/// Coefficients are returned in the same order as `basis`. The mapping uses the
/// standard four-dimensional rank table for the corresponding sector:
/// scalar rank 0, vector rank 1, sigma rank 2, axial-vector rank 3, and
/// pseudoscalar rank 4. Missing dimension, missing gamma5 convention, or an
/// unrecognized basis sector returns `InsufficientConventionData`.
pub fn fierz_coefficients_for_basis(
    basis: &[Expr],
    spinor_meta: &ax_ir::SpinorMetadata,
    gamma_meta: Option<&ax_ir::GammaConventionMetadata>,
    interner: &ax_ir::Interner,
) -> Result<Vec<Expr>, FierzError> {
    let Some(gamma_meta) = gamma_meta else {
        return Err(FierzError::InsufficientConventionData);
    };
    if basis.is_empty()
        || !convention_dimension_matches_spinor(spinor_meta, gamma_meta)
        || spinor_meta.dimension != Some(4)
        || gamma_meta.gamma5.is_none()
    {
        return Err(FierzError::InsufficientConventionData);
    }

    let table = fierz_coefficients(4)
        .into_iter()
        .map(|(coefficient, rank)| (rank, coefficient))
        .collect::<HashMap<_, _>>();

    basis
        .iter()
        .map(|sector| {
            let rank =
                fierz_basis_rank(sector, interner).ok_or(FierzError::InsufficientConventionData)?;
            table
                .get(&rank)
                .cloned()
                .map(Expr::Rational)
                .ok_or(FierzError::InsufficientConventionData)
        })
        .collect()
}

/// Perform a Fierz rearrangement.
///
/// Given an expression of the form (ψ̄₁ Γ ψ₂)(ψ̄₃ Γ ψ₄), rearrange to
/// a sum over the Fierz basis: Σ_n c_n (ψ̄₁ Γ_n ψ₄)(ψ̄₃ Γ_n ψ₂)
pub fn fierz_rearrange(
    expr: &Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> Expr {
    fierz(expr, dim, spinor_order, interner)
}

fn is_name(name: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| name == *candidate)
}

fn has_property(
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    sym: lasso::Spur,
    property: &TensorProperty,
) -> bool {
    properties
        .map(|props| props.has_property_kind(sym, property))
        .unwrap_or(false)
}

fn expr_head_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Call(sym, _) => Some(*sym),
        Expr::Indexed(base, _) => expr_head_symbol(base),
        _ => None,
    }
}

fn is_dirac_bar_call(
    sym: lasso::Spur,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    properties
        .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(sym), props))
        .is_some()
        || is_name(
            interner.resolve(sym),
            &["dirac_bar", "diracbar", "bar", "DiracBar"],
        )
}

fn barred_spinor_symbol(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => {
            if properties
                .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(*sym), props))
                .is_some()
            {
                return Some(*sym);
            }
            let name = interner.resolve(*sym);
            if name.contains("bar")
                || name.contains("Bar")
                || name.ends_with("bar")
                || name.ends_with("_bar")
                || name.ends_with("Bar")
            {
                Some(*sym)
            } else {
                None
            }
        }
        Expr::Call(f, args) => {
            if is_dirac_bar_call(*f, properties, interner) {
                args.first().and_then(spinor_symbol)
            } else {
                None
            }
        }
        Expr::Indexed(base, _) => barred_spinor_symbol(base, properties, interner),
        _ => None,
    }
}

fn spinor_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Indexed(base, _) => spinor_symbol(base),
        _ => None,
    }
}

fn spinor_symbol_with_properties(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => {
            if properties
                .map(|props| {
                    spinor_metadata_of_expr(&Expr::Sym(*sym), props).is_some()
                        || props.has_property_kind(*sym, &TensorProperty::AntiCommuting)
                })
                .unwrap_or(true)
            {
                Some(*sym)
            } else {
                None
            }
        }
        Expr::Indexed(base, _) => spinor_symbol_with_properties(base, properties),
        _ => None,
    }
}

fn gamma_factor_indices(
    expr: &Expr,
    gamma_sym: Option<lasso::Spur>,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<Vec<lasso::Spur>> {
    match expr {
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            if Some(*f) == gamma_sym
                || properties
                    .and_then(|props| gamma_metadata_of_expr(&Expr::Sym(*f), props))
                    .is_some()
                || is_name(name, &["gamma", "Gamma", "γ"])
            {
                Some(
                    args.iter()
                        .filter_map(|arg| match arg {
                            Expr::Sym(sym) => Some(*sym),
                            _ => None,
                        })
                        .collect(),
                )
            } else if is_name(name, &["gamma5", "Gamma5", "γ5"]) {
                Some(vec![interner.get_or_intern("5")])
            } else {
                None
            }
        }
        Expr::Indexed(base, indices) => match base.as_ref() {
            Expr::Sym(sym)
                if Some(*sym) == gamma_sym
                    || properties
                        .and_then(|props| gamma_metadata_of_expr(&Expr::Sym(*sym), props))
                        .is_some()
                    || is_name(interner.resolve(*sym), &["gamma", "Gamma", "γ"]) =>
            {
                Some(indices.iter().map(|idx| idx.name).collect())
            }
            _ => None,
        },
        _ => None,
    }
}

fn parse_bilinear_at(
    factors: &[Expr],
    start: usize,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<(lasso::Spur, Vec<lasso::Spur>, lasso::Spur, usize, bool)> {
    let barred = barred_spinor_symbol(&factors[start], properties, interner)?;
    let expected_gamma_sym = properties.and_then(|props| {
        diracbar_metadata_of_expr(&factors[start], props).and_then(|metadata| metadata.gamma_symbol)
    });
    let mut gamma_indices = Vec::new();
    let mut cursor = start + 1;
    let mut saw_non_gamma_before_spinor = false;
    while cursor < factors.len() {
        let Some(mut indices) =
            gamma_factor_indices(&factors[cursor], expected_gamma_sym, properties, interner)
        else {
            break;
        };
        gamma_indices.append(&mut indices);
        cursor += 1;
    }
    if cursor >= factors.len() {
        return None;
    }
    let spinor = spinor_symbol_with_properties(&factors[cursor], properties)?;

    let mut trailing_cursor = cursor + 1;
    while trailing_cursor < factors.len() {
        if let Some(mut indices) =
            gamma_factor_indices(&factors[trailing_cursor], None, properties, interner)
        {
            gamma_indices.append(&mut indices);
            saw_non_gamma_before_spinor = true;
            trailing_cursor += 1;
        } else {
            break;
        }
    }

    Some((
        barred,
        gamma_indices,
        spinor,
        trailing_cursor,
        saw_non_gamma_before_spinor,
    ))
}

pub fn find_bilinears(expr: &Expr, interner: &ax_ir::Interner) -> Option<BilinearPair> {
    find_bilinears_impl(expr, None, interner).ok()
}

pub fn find_bilinears_with_properties(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<BilinearPair> {
    find_bilinears_impl(expr, Some(properties), interner).ok()
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedBilinear {
    barred: lasso::Spur,
    gamma_indices: Vec<lasso::Spur>,
    spinor: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedFierzInput {
    pair: BilinearPair,
    sign: i64,
}

fn flatten_mul_factors(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Mul(factors) => {
            for factor in factors {
                flatten_mul_factors(factor, out);
            }
        }
        other => out.push(other.clone()),
    }
}

fn factor_contains_diracbar(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    match expr {
        Expr::Sym(sym) => {
            barred_spinor_symbol(expr, properties, interner).is_some()
                || properties
                    .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(*sym), props))
                    .is_some()
        }
        Expr::Call(sym, args) => {
            is_dirac_bar_call(*sym, properties, interner)
                || args
                    .iter()
                    .any(|arg| factor_contains_diracbar(arg, properties, interner))
        }
        Expr::Indexed(base, _) => factor_contains_diracbar(base, properties, interner),
        Expr::Mul(factors) | Expr::Add(factors) => factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner)),
        Expr::Neg(inner) => factor_contains_diracbar(inner, properties, interner),
        Expr::Pow(base, exp) => {
            factor_contains_diracbar(base, properties, interner)
                || factor_contains_diracbar(exp, properties, interner)
        }
        Expr::Complex(re, im) => {
            factor_contains_diracbar(re, properties, interner)
                || factor_contains_diracbar(im, properties, interner)
        }
        _ => false,
    }
}

fn is_anticommuting_spinor(
    sym: lasso::Spur,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    properties
        .map(|props| props.has_property_kind(sym, &TensorProperty::AntiCommuting))
        .unwrap_or_else(|| {
            let name = interner.resolve(sym);
            name.starts_with("psi")
                || name.starts_with("chi")
                || name.starts_with("theta")
                || name.contains("spinor")
        })
}

fn anticommuting_reorder_sign(
    input_order: &[lasso::Spur],
    output_order: &[lasso::Spur],
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<i64, FierzError> {
    if input_order.len() != output_order.len() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let input_set: HashSet<_> = input_order.iter().copied().collect();
    let output_set: HashSet<_> = output_order.iter().copied().collect();
    if input_set != output_set || input_set.len() != input_order.len() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let mut current = input_order.to_vec();
    let mut sign = 1i64;
    for target_pos in 0..output_order.len() {
        let Some(found_pos) = current[target_pos..]
            .iter()
            .position(|sym| *sym == output_order[target_pos])
            .map(|pos| pos + target_pos)
        else {
            return Err(FierzError::SpinorOrderMismatch);
        };

        for pos in (target_pos..found_pos).rev() {
            if is_anticommuting_spinor(current[pos], properties, interner)
                && is_anticommuting_spinor(current[pos + 1], properties, interner)
            {
                sign = -sign;
            }
            current.swap(pos, pos + 1);
        }
    }
    Ok(sign)
}

fn find_bilinears_impl(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<BilinearPair, FierzError> {
    parse_fierz_input(expr, properties, interner).map(|parsed| parsed.pair)
}

fn parse_fierz_input(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<ParsedFierzInput, FierzError> {
    let mut factors = Vec::new();
    flatten_mul_factors(expr, &mut factors);
    if factors.len() < 4 {
        if factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner))
        {
            return Err(FierzError::MalformedBilinear);
        }
        return Err(FierzError::NoBilinearPair);
    }

    let mut bilinears: Vec<ParsedBilinear> = Vec::new();
    let mut remaining_factors = Vec::new();
    let mut reordered_within_bilinear = false;
    let mut cursor = 0usize;
    while cursor < factors.len() {
        if bilinears.len() < 2 {
            if let Some((barred, gamma_indices, spinor, next, reordered)) =
                parse_bilinear_at(&factors, cursor, properties, interner)
            {
                bilinears.push(ParsedBilinear {
                    barred,
                    gamma_indices,
                    spinor,
                });
                reordered_within_bilinear |= reordered;
                cursor = next;
                continue;
            }
        }
        remaining_factors.push(factors[cursor].clone());
        cursor += 1;
    }

    if bilinears.len() < 2 {
        if factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner))
        {
            return Err(FierzError::MalformedBilinear);
        }
        return Err(FierzError::NoBilinearPair);
    }

    let mut probe = 0usize;
    let mut total_bilinears = 0usize;
    while probe < factors.len() {
        if let Some((_, _, _, next, _)) = parse_bilinear_at(&factors, probe, properties, interner) {
            total_bilinears += 1;
            probe = next;
        } else {
            probe += 1;
        }
    }
    if total_bilinears > 2 {
        return Err(FierzError::AmbiguousBilinears(total_bilinears));
    }

    let first = bilinears[0].clone();
    let second = bilinears[1].clone();
    let pair = BilinearPair {
        psi1: first.barred,
        gamma_a: first.gamma_indices,
        psi2: first.spinor,
        psi3: second.barred,
        gamma_b: second.gamma_indices,
        psi4: second.spinor,
        remaining_factors,
    };

    let sign = if reordered_within_bilinear { -1 } else { 1 };

    Ok(ParsedFierzInput { pair, sign })
}

fn gamma_index_count(expr: &Expr, gamma_sym: lasso::Spur) -> Option<usize> {
    match expr {
        Expr::Call(f, args) if *f == gamma_sym => Some(args.len()),
        Expr::Indexed(base, indices) if expr_head_symbol(base) == Some(gamma_sym) => {
            Some(indices.len())
        }
        _ => None,
    }
}

fn is_gamma_expr(expr: &Expr, gamma_sym: lasso::Spur) -> bool {
    gamma_index_count(expr, gamma_sym).is_some()
}

fn is_diracbar_call(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<(lasso::Spur, Expr)> {
    match expr {
        Expr::Call(sym, args)
            if args.len() == 1
                && diracbar_metadata_of_expr(&Expr::Sym(*sym), properties).is_some() =>
        {
            Some((*sym, args[0].clone()))
        }
        _ => None,
    }
}

fn is_diracbar_scalar_factor(expr: &Expr, properties: &dyn ax_tensor::PropertyLookup) -> bool {
    if matches!(
        expr,
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Complex(_, _)
    ) {
        return true;
    }
    gamma_metadata_of_expr(expr, properties).is_none()
        && spinor_metadata_of_expr(expr, properties).is_none()
        && is_diracbar_call(expr, properties).is_none()
}

fn conjugate_scalar_factor(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => expr,
        Expr::Complex(re, im) => Expr::Complex(re, Box::new(Expr::neg(*im))),
        other => conj_expr(other, interner),
    }
}

/// Return the canonical Dirac-bar action on a gamma factor.
///
/// Gamma factors are represented as `bar_action(gamma(...))` so the operation
/// is explicit and does not collapse to a sign inferred only from chain length.
pub fn diracbar_action_on_gamma(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if gamma_metadata_of_expr(expr, props).is_some() {
        Expr::Call(interner.get_or_intern("bar_action"), vec![expr.clone()])
    } else {
        adjoint_of(expr.clone(), interner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChiralProjectorKind {
    Left,
    Right,
}

fn chiral_projector_kind(expr: &Expr, interner: &ax_ir::Interner) -> Option<ChiralProjectorKind> {
    if *expr == projector_left(interner) {
        return Some(ChiralProjectorKind::Left);
    }
    if *expr == projector_right(interner) {
        return Some(ChiralProjectorKind::Right);
    }
    match expr {
        Expr::Sym(sym) => match interner.resolve(*sym) {
            "P_L" | "PL" | "projector_left" => Some(ChiralProjectorKind::Left),
            "P_R" | "PR" | "projector_right" => Some(ChiralProjectorKind::Right),
            _ => None,
        },
        Expr::Call(sym, args) if args.is_empty() => match interner.resolve(*sym) {
            "P_L" | "PL" | "projector_left" => Some(ChiralProjectorKind::Left),
            "P_R" | "PR" | "projector_right" => Some(ChiralProjectorKind::Right),
            _ => None,
        },
        _ => None,
    }
}

fn chiral_projector_expr(kind: ChiralProjectorKind, interner: &ax_ir::Interner) -> Expr {
    match kind {
        ChiralProjectorKind::Left => projector_left(interner),
        ChiralProjectorKind::Right => projector_right(interner),
    }
}

fn chiral_projector_body_kind(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ChiralProjectorKind> {
    let gamma5 = Expr::Sym(interner.get_or_intern("gamma5"));
    if *expr == Expr::add(vec![Expr::one(), Expr::neg(gamma5.clone())]) {
        Some(ChiralProjectorKind::Left)
    } else if *expr == Expr::add(vec![Expr::one(), gamma5]) {
        Some(ChiralProjectorKind::Right)
    } else {
        None
    }
}

fn chiral_projector_denominator(count: usize) -> BigInt {
    (0..count).fold(BigInt::one(), |acc, _| acc * BigInt::from(2usize))
}

fn coalesce_chiral_projector_factors(factors: Vec<Expr>, interner: &ax_ir::Interner) -> Vec<Expr> {
    let mut numeric_coeff = None;
    let mut projector_kinds = Vec::new();
    let mut other = Vec::new();

    for factor in factors {
        if let Some(kind) = chiral_projector_kind(&factor, interner)
            .or_else(|| chiral_projector_body_kind(&factor, interner))
        {
            projector_kinds.push(kind);
        } else if let Expr::Rational(coeff) = &factor {
            numeric_coeff = Some(coeff.clone());
        } else {
            other.push(factor);
        }
    }

    if projector_kinds.is_empty() {
        if let Some(coeff) = numeric_coeff {
            other.insert(0, Expr::Rational(coeff));
        }
        return other;
    }

    let expected = BigRational::new(
        BigInt::one(),
        chiral_projector_denominator(projector_kinds.len()),
    );
    if numeric_coeff.as_ref() != Some(&expected) {
        if let Some(coeff) = numeric_coeff {
            other.insert(0, Expr::Rational(coeff));
        }
        other.extend(projector_kinds.into_iter().map(|kind| match kind {
            ChiralProjectorKind::Left => Expr::add(vec![
                Expr::one(),
                Expr::neg(Expr::Sym(interner.get_or_intern("gamma5"))),
            ]),
            ChiralProjectorKind::Right => Expr::add(vec![
                Expr::one(),
                Expr::Sym(interner.get_or_intern("gamma5")),
            ]),
        }));
        return other;
    }

    let mut coalesced = projector_kinds
        .into_iter()
        .map(|kind| chiral_projector_expr(kind, interner))
        .collect::<Vec<_>>();
    coalesced.extend(other);
    coalesced
}

fn spinor_chirality_of_expr(expr: &Expr, props: &dyn PropertyLookup) -> Option<ax_ir::Chirality> {
    let metadata = spinor_metadata_of_expr(expr, props)?;
    let is_weyl = matches!(
        metadata.class,
        SpinorClass::Weyl | SpinorClass::MajoranaWeyl
    );
    is_weyl.then_some(metadata.chirality).flatten()
}

fn apply_chiral_projector_to_spinor(
    projector: ChiralProjectorKind,
    spinor: &Expr,
    props: &dyn PropertyLookup,
) -> Option<Expr> {
    match (projector, spinor_chirality_of_expr(spinor, props)?) {
        (ChiralProjectorKind::Left, ax_ir::Chirality::Left)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Right) => Some(spinor.clone()),
        (ChiralProjectorKind::Left, ax_ir::Chirality::Right)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Left) => Some(Expr::zero()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BilinearSector {
    Scalar,
    Pseudoscalar,
    Vector,
    AxialVector,
    Tensor,
}

fn explicit_structured_weyl_chirality(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<ax_ir::Chirality> {
    let sym = property_sym(expr)?;
    let metadata = declared_spinor_metadata_of_symbol(sym, props)?;
    if metadata.dimension != Some(4) {
        return None;
    }
    match metadata.class {
        SpinorClass::Weyl | SpinorClass::MajoranaWeyl => metadata.chirality,
        _ => None,
    }
}

fn explicit_structured_majorana_4d(expr: &Expr, props: &dyn PropertyLookup) -> Option<lasso::Spur> {
    let sym = property_sym(expr)?;
    let metadata = declared_spinor_metadata_of_symbol(sym, props)?;
    if metadata.dimension != Some(4) {
        return None;
    }
    match metadata.class {
        SpinorClass::Majorana | SpinorClass::MajoranaWeyl => Some(sym),
        _ => None,
    }
}

fn gamma_factor_has_supported_4d_convention(expr: &Expr, props: &dyn PropertyLookup) -> bool {
    gamma_convention_of_expr(expr, props).is_some_and(|metadata| metadata.dimension == Some(4))
}

fn gamma5_factor_has_supported_4d_convention(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: Option<&ax_ir::Interner>,
) -> bool {
    match expr {
        Expr::Sym(sym) | Expr::Call(sym, _) => {
            if let Some(interner) = interner {
                let name = interner.resolve(*sym);
                if name == "gamma5" || name == "Gamma5" || name == "γ5" {
                    return declared_gamma_convention_metadata_of_symbol(
                        interner.get_or_intern("gamma"),
                        props,
                    )
                    .is_some_and(|metadata| {
                        metadata.dimension == Some(4) && metadata.gamma5.is_some()
                    });
                }
            }
            false
        }
        _ => false,
    }
}

fn bilinear_sector_from_middle(
    middle: Option<&Expr>,
    props: &dyn PropertyLookup,
    interner: Option<&ax_ir::Interner>,
) -> Option<BilinearSector> {
    let Some(middle) = middle else {
        return Some(BilinearSector::Scalar);
    };
    if *middle == Expr::one() {
        return Some(BilinearSector::Scalar);
    }
    if gamma5_factor_has_supported_4d_convention(middle, props, interner) {
        return Some(BilinearSector::Pseudoscalar);
    }
    if let Some(data) = gamma_expr_data(middle, props) {
        if !gamma_factor_has_supported_4d_convention(&data.head, props)
            && !gamma_factor_has_supported_4d_convention(middle, props)
        {
            return None;
        }
        return match data.indices.len() {
            1 => Some(BilinearSector::Vector),
            2 => Some(BilinearSector::Tensor),
            _ => None,
        };
    }
    if let Expr::Mul(factors) = middle {
        if factors.len() == 2 {
            let gamma_pos = factors
                .iter()
                .position(|factor| gamma_expr_data(factor, props).is_some())?;
            let gamma = &factors[gamma_pos];
            let gamma5 = &factors[1 - gamma_pos];
            let data = gamma_expr_data(gamma, props)?;
            if data.indices.len() == 1
                && gamma_factor_has_supported_4d_convention(&data.head, props)
                && gamma5_factor_has_supported_4d_convention(gamma5, props, interner)
            {
                return Some(BilinearSector::AxialVector);
            }
        }
    }
    if let Some(interner) = interner {
        if let Expr::Call(sym, args) = middle {
            if interner.resolve(*sym) == "sigma" && args.len() == 2 {
                return declared_gamma_convention_metadata_of_symbol(
                    interner.get_or_intern("gamma"),
                    props,
                )
                .is_some_and(|metadata| metadata.dimension == Some(4))
                .then_some(BilinearSector::Tensor);
            }
        }
    }
    None
}

fn diracbar_spinor_arg<'a>(expr: &'a Expr, props: &dyn PropertyLookup) -> Option<&'a Expr> {
    match expr {
        Expr::Call(sym, args)
            if args.len() == 1 && diracbar_metadata_of_expr(&Expr::Sym(*sym), props).is_some() =>
        {
            Some(&args[0])
        }
        _ => None,
    }
}

fn opposite_chirality(chirality: ax_ir::Chirality) -> ax_ir::Chirality {
    match chirality {
        ax_ir::Chirality::Left => ax_ir::Chirality::Right,
        ax_ir::Chirality::Right => ax_ir::Chirality::Left,
    }
}

fn projector_forbids_barred_weyl(
    projector: ChiralProjectorKind,
    left_chirality: ax_ir::Chirality,
) -> bool {
    match (projector, opposite_chirality(left_chirality)) {
        (ChiralProjectorKind::Left, ax_ir::Chirality::Right)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Left) => true,
        (ChiralProjectorKind::Left, ax_ir::Chirality::Left)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Right) => false,
    }
}

fn projector_forbids_right_weyl(
    projector: ChiralProjectorKind,
    right_chirality: ax_ir::Chirality,
) -> bool {
    match (projector, right_chirality) {
        (ChiralProjectorKind::Left, ax_ir::Chirality::Right)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Left) => true,
        (ChiralProjectorKind::Left, ax_ir::Chirality::Left)
        | (ChiralProjectorKind::Right, ax_ir::Chirality::Right) => false,
    }
}

fn middle_factor_list(middle: Option<&Expr>) -> Vec<&Expr> {
    match middle {
        Some(Expr::Mul(factors)) => factors.iter().collect(),
        Some(expr) => vec![expr],
        None => Vec::new(),
    }
}

fn weyl_bilinear_forbidden(
    left_spinor: &Expr,
    middle: Option<&Expr>,
    right: &Expr,
    props: &dyn PropertyLookup,
    interner: Option<&ax_ir::Interner>,
) -> bool {
    let (Some(left_chirality), Some(right_chirality)) = (
        explicit_structured_weyl_chirality(left_spinor, props),
        explicit_structured_weyl_chirality(right, props),
    ) else {
        return false;
    };

    let middle_factors = middle_factor_list(middle);
    if let Some(first) = middle_factors.first() {
        if let Some(projector) = interner.and_then(|i| {
            chiral_projector_kind(first, i).or_else(|| chiral_projector_body_kind(first, i))
        }) {
            if projector_forbids_barred_weyl(projector, left_chirality.clone()) {
                return true;
            }
        }
    }
    if let Some(last) = middle_factors.last() {
        if let Some(projector) = interner.and_then(|i| {
            chiral_projector_kind(last, i).or_else(|| chiral_projector_body_kind(last, i))
        }) {
            if projector_forbids_right_weyl(projector, right_chirality.clone()) {
                return true;
            }
        }
    }

    let non_projector_middle = middle_factors
        .iter()
        .copied()
        .filter(|factor| {
            interner
                .and_then(|i| {
                    chiral_projector_kind(factor, i)
                        .or_else(|| chiral_projector_body_kind(factor, i))
                })
                .is_none()
        })
        .cloned()
        .collect::<Vec<_>>();
    let middle_expr = match non_projector_middle.len() {
        0 => None,
        1 => non_projector_middle.first().cloned(),
        _ => return false,
    };
    let Some(sector) = bilinear_sector_from_middle(middle_expr.as_ref(), props, interner) else {
        return false;
    };

    let same_chirality = left_chirality == right_chirality;
    match sector {
        BilinearSector::Scalar | BilinearSector::Pseudoscalar | BilinearSector::Tensor => {
            same_chirality
        }
        BilinearSector::Vector | BilinearSector::AxialVector => !same_chirality,
    }
}

fn majorana_self_bilinear_forbidden(
    left_spinor: &Expr,
    middle: Option<&Expr>,
    right: &Expr,
    props: &dyn PropertyLookup,
    interner: Option<&ax_ir::Interner>,
) -> bool {
    let (Some(left_sym), Some(right_sym)) = (
        explicit_structured_majorana_4d(left_spinor, props),
        explicit_structured_majorana_4d(right, props),
    ) else {
        return false;
    };
    if left_sym != right_sym {
        return false;
    }
    let Some(sector) = bilinear_sector_from_middle(middle, props, interner) else {
        return false;
    };
    matches!(sector, BilinearSector::Vector | BilinearSector::Tensor)
}

fn bilinear_is_forbidden_by_metadata_impl(
    left: &Expr,
    middle: Option<&Expr>,
    right: &Expr,
    props: &dyn PropertyLookup,
    interner: Option<&ax_ir::Interner>,
) -> bool {
    let Some(left_spinor) = diracbar_spinor_arg(left, props) else {
        return false;
    };
    weyl_bilinear_forbidden(left_spinor, middle, right, props, interner)
        || majorana_self_bilinear_forbidden(left_spinor, middle, right, props, interner)
}

/// Return whether a single barred-spinor bilinear is forbidden by declared metadata.
///
/// The check is intentionally conservative: it only returns `true` when the
/// left factor is a declared Dirac-bar call, the participating spinors carry
/// structured 4D spinor metadata, and the middle factor can be classified from
/// structured gamma/convention metadata. If any required metadata is missing,
/// the bilinear is treated as unknown and this returns `false`.
pub fn bilinear_is_forbidden_by_metadata(
    left: &Expr,
    middle: Option<&Expr>,
    right: &Expr,
    props: &dyn PropertyLookup,
) -> bool {
    bilinear_is_forbidden_by_metadata_impl(left, middle, right, props, None)
}

#[derive(Clone, Debug)]
struct ImplicitSpinorChain {
    bar_factor: Expr,
    gamma_factors: Vec<Expr>,
    right_spinor: Expr,
    spinor_family: lasso::Spur,
}

fn compatible_spinor_chain_family(
    bar_factor: &Expr,
    left_spinor: &Expr,
    gamma_factors: &[Expr],
    right_spinor: &Expr,
    props: &dyn PropertyLookup,
) -> Option<lasso::Spur> {
    let left_meta = spinor_metadata_of_expr(left_spinor, props)?;
    let right_meta = spinor_metadata_of_expr(right_spinor, props)?;
    let family = left_meta.index_family?;
    if right_meta.index_family != Some(family) {
        return None;
    }

    let bar_meta = diracbar_metadata_of_expr(bar_factor, props)?;
    if bar_meta
        .spinor_family
        .is_some_and(|expected| expected != family)
    {
        return None;
    }

    for gamma in gamma_factors {
        let gamma_meta = gamma_metadata_of_expr(gamma, props)?;
        if gamma_meta.index_family != Some(family) {
            return None;
        }
        if let Some(expected_gamma) = bar_meta.gamma_symbol {
            if property_sym(gamma).is_some_and(|actual| actual != expected_gamma) {
                return None;
            }
        }
    }

    Some(family)
}

fn parse_implicit_spinor_chain(
    expr: &Expr,
    props: &dyn PropertyLookup,
) -> Option<ImplicitSpinorChain> {
    let Expr::Mul(factors) = expr else {
        return None;
    };
    if factors.len() < 2 {
        return None;
    }
    if matches!(factors.first(), Some(Expr::Indexed(_, _)))
        || matches!(factors.last(), Some(Expr::Indexed(_, _)))
    {
        return None;
    }
    let bar_factor = factors.first()?.clone();
    let left_spinor = diracbar_spinor_arg(&bar_factor, props)?.clone();
    let right_spinor = factors.last()?.clone();
    if spinor_metadata_of_expr(&right_spinor, props).is_none() {
        return None;
    }

    let gamma_factors = factors[1..factors.len() - 1].to_vec();
    if gamma_factors.iter().any(|factor| {
        matches!(factor, Expr::Indexed(_, _)) || gamma_metadata_of_expr(factor, props).is_none()
    }) {
        return None;
    }

    let spinor_family = compatible_spinor_chain_family(
        &bar_factor,
        &left_spinor,
        &gamma_factors,
        &right_spinor,
        props,
    )?;
    Some(ImplicitSpinorChain {
        bar_factor,
        gamma_factors,
        right_spinor,
        spinor_family,
    })
}

fn fresh_spinor_chain_indices(
    count: usize,
    family_name: lasso::Spur,
    used: &mut HashSet<lasso::Spur>,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Vec<lasso::Spur> {
    if let Some(family) = props
        .index_families()
        .and_then(|families| families.get(&family_name))
    {
        return (0..count)
            .map(|_| fresh_dummy_from_family(family, used, interner))
            .collect();
    }

    let prefix = interner.resolve(family_name).to_string();
    let mut out = Vec::with_capacity(count);
    let mut counter = 0usize;
    while out.len() < count {
        let candidate = interner.get_or_intern(&format!("{prefix}_{counter}"));
        counter += 1;
        if used.insert(candidate) {
            out.push(candidate);
        }
    }
    out
}

fn spinor_chain_index(name: lasso::Spur, variance: Variance, family: lasso::Spur) -> Index {
    Index {
        name,
        variance,
        index_type: Some(family),
    }
}

fn insert_explicit_spinor_indices_for_chain(
    chain: ImplicitSpinorChain,
    used: &mut HashSet<lasso::Spur>,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let names = fresh_spinor_chain_indices(
        chain.gamma_factors.len() + 1,
        chain.spinor_family,
        used,
        props,
        interner,
    );
    let mut explicit = Vec::with_capacity(chain.gamma_factors.len() + 2);
    explicit.push(Expr::Indexed(
        Box::new(chain.bar_factor),
        vec![spinor_chain_index(
            names[0],
            Variance::Down,
            chain.spinor_family,
        )],
    ));
    for (idx, gamma) in chain.gamma_factors.into_iter().enumerate() {
        explicit.push(Expr::Indexed(
            Box::new(gamma),
            vec![
                spinor_chain_index(names[idx], Variance::Up, chain.spinor_family),
                spinor_chain_index(names[idx + 1], Variance::Down, chain.spinor_family),
            ],
        ));
    }
    explicit.push(Expr::Indexed(
        Box::new(chain.right_spinor),
        vec![spinor_chain_index(
            *names.last().expect("spinor chain has at least one index"),
            Variance::Up,
            chain.spinor_family,
        )],
    ));
    Expr::mul(explicit)
}

/// Insert explicit contracted spinor indices for supported implicit spinor chains.
///
/// The bridge recognizes canonical metadata-backed products of the form
/// `bar(psi) * gamma(...) * ... * chi`, including the no-gamma bilinear
/// `bar(psi) * chi`. It requires structured Dirac-bar, spinor, gamma, and
/// common spinor index-family metadata. Fresh dummy names are drawn from the
/// declared `IndexFamily` when available. Expressions outside this supported
/// canonical shape are returned unchanged.
pub fn insert_explicit_spinor_indices(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut used = HashSet::new();
    collect_all_index_names(expr, &mut used);
    insert_explicit_spinor_indices_inner(expr, props, interner, &mut used)
}

fn insert_explicit_spinor_indices_inner(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
    used: &mut HashSet<lasso::Spur>,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let mapped = Expr::mul(
                factors
                    .iter()
                    .map(|factor| {
                        insert_explicit_spinor_indices_inner(factor, props, interner, used)
                    })
                    .collect(),
            );
            if let Some(chain) = parse_implicit_spinor_chain(&mapped, props) {
                insert_explicit_spinor_indices_for_chain(chain, used, props, interner)
            } else {
                mapped
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| insert_explicit_spinor_indices_inner(term, props, interner, used))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            insert_explicit_spinor_indices_inner(base, props, interner, used),
            insert_explicit_spinor_indices_inner(exp, props, interner, used),
        ),
        Expr::Neg(inner) => Expr::neg(insert_explicit_spinor_indices_inner(
            inner, props, interner, used,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(insert_explicit_spinor_indices_inner(
                re, props, interner, used,
            )),
            Box::new(insert_explicit_spinor_indices_inner(
                im, props, interner, used,
            )),
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| insert_explicit_spinor_indices_inner(arg, props, interner, used))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(insert_explicit_spinor_indices_inner(
                base, props, interner, used,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(insert_explicit_spinor_indices_inner(
                inner, props, interner, used,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(insert_explicit_spinor_indices_inner(
                value, props, interner, used,
            )),
            Box::new(insert_explicit_spinor_indices_inner(
                body, props, interner, used,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| insert_explicit_spinor_indices_inner(item, props, interner, used))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| {
                            insert_explicit_spinor_indices_inner(entry, props, interner, used)
                        })
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(insert_explicit_spinor_indices_inner(
                body, props, interner, used,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(insert_explicit_spinor_indices_inner(
                lhs, props, interner, used,
            )),
            Box::new(insert_explicit_spinor_indices_inner(
                rhs, props, interner, used,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        insert_explicit_spinor_indices_inner(value, props, interner, used),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn indexed_single_spinor_factor<'a>(
    expr: &'a Expr,
    props: &dyn PropertyLookup,
) -> Option<(&'a Expr, &'a Index)> {
    let Expr::Indexed(base, indices) = expr else {
        return None;
    };
    if indices.len() != 1 || spinor_metadata_of_expr(base, props).is_none() {
        return None;
    }
    Some((base.as_ref(), &indices[0]))
}

fn indexed_barred_spinor_factor<'a>(
    expr: &'a Expr,
    props: &dyn PropertyLookup,
) -> Option<(&'a Expr, &'a Expr, &'a Index)> {
    let Expr::Indexed(base, indices) = expr else {
        return None;
    };
    if indices.len() != 1 {
        return None;
    }
    let spinor = diracbar_spinor_arg(base, props)?;
    Some((base.as_ref(), spinor, &indices[0]))
}

fn indexed_gamma_spinor_factor<'a>(
    expr: &'a Expr,
    props: &dyn PropertyLookup,
) -> Option<(&'a Expr, &'a Index, &'a Index)> {
    let Expr::Indexed(base, indices) = expr else {
        return None;
    };
    if indices.len() != 2 || gamma_metadata_of_expr(base, props).is_none() {
        return None;
    }
    Some((base.as_ref(), &indices[0], &indices[1]))
}

fn same_spinor_chain_index(
    left: &Index,
    left_variance: Variance,
    right: &Index,
    right_variance: Variance,
    family: lasso::Spur,
) -> bool {
    left.name == right.name
        && left.variance == left_variance
        && right.variance == right_variance
        && left.index_type == Some(family)
        && right.index_type == Some(family)
}

fn remove_explicit_spinor_chain(expr: &Expr, props: &dyn PropertyLookup) -> Option<Expr> {
    let Expr::Mul(factors) = expr else {
        return None;
    };
    if factors.len() < 2 {
        return None;
    }
    let (bar_factor, left_spinor, bar_index) =
        indexed_barred_spinor_factor(factors.first()?, props)?;
    if bar_index.variance != Variance::Down {
        return None;
    }
    let (right_spinor, right_index) = indexed_single_spinor_factor(factors.last()?, props)?;
    if right_index.variance != Variance::Up {
        return None;
    }

    let gamma_factors = factors[1..factors.len() - 1]
        .iter()
        .map(|factor| indexed_gamma_spinor_factor(factor, props))
        .collect::<Option<Vec<_>>>()?;
    let gamma_bases = gamma_factors
        .iter()
        .map(|(base, _, _)| (*base).clone())
        .collect::<Vec<_>>();
    let family =
        compatible_spinor_chain_family(bar_factor, left_spinor, &gamma_bases, right_spinor, props)?;

    if gamma_factors.is_empty() {
        if !same_spinor_chain_index(bar_index, Variance::Down, right_index, Variance::Up, family) {
            return None;
        }
    } else {
        let (_, first_in, _) = gamma_factors.first()?;
        if !same_spinor_chain_index(bar_index, Variance::Down, first_in, Variance::Up, family) {
            return None;
        }
        for adjacent in gamma_factors.windows(2) {
            let (_, _, prev_out) = adjacent[0];
            let (_, next_in, _) = adjacent[1];
            if !same_spinor_chain_index(prev_out, Variance::Down, next_in, Variance::Up, family) {
                return None;
            }
        }
        let (_, _, last_out) = gamma_factors.last()?;
        if !same_spinor_chain_index(last_out, Variance::Down, right_index, Variance::Up, family) {
            return None;
        }
    }

    let mut implicit = Vec::with_capacity(factors.len());
    implicit.push(bar_factor.clone());
    implicit.extend(gamma_bases);
    implicit.push(right_spinor.clone());
    Some(Expr::mul(implicit))
}

/// Collapse explicit canonical spinor-index contractions back to implicit chains.
///
/// This inverse bridge recognizes only the unambiguous contraction pattern
/// produced by `insert_explicit_spinor_indices`: `bar(psi)_a gamma^a_b ...
/// chi^b`, with all spinor slots in one declared spinor index family. Any
/// noncanonical or unsupported indexed expression is returned unchanged.
pub fn remove_trivial_spinor_indices(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let mapped = Expr::mul(
                factors
                    .iter()
                    .map(|factor| remove_trivial_spinor_indices(factor, props, interner))
                    .collect(),
            );
            remove_explicit_spinor_chain(&mapped, props).unwrap_or(mapped)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| remove_trivial_spinor_indices(term, props, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            remove_trivial_spinor_indices(base, props, interner),
            remove_trivial_spinor_indices(exp, props, interner),
        ),
        Expr::Neg(inner) => Expr::neg(remove_trivial_spinor_indices(inner, props, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(remove_trivial_spinor_indices(re, props, interner)),
            Box::new(remove_trivial_spinor_indices(im, props, interner)),
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| remove_trivial_spinor_indices(arg, props, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(remove_trivial_spinor_indices(base, props, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(remove_trivial_spinor_indices(inner, props, interner)),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(remove_trivial_spinor_indices(value, props, interner)),
            Box::new(remove_trivial_spinor_indices(body, props, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| remove_trivial_spinor_indices(item, props, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| remove_trivial_spinor_indices(entry, props, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(remove_trivial_spinor_indices(body, props, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(remove_trivial_spinor_indices(lhs, props, interner)),
            Box::new(remove_trivial_spinor_indices(rhs, props, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        remove_trivial_spinor_indices(value, props, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn bilinear_window_is_forbidden(
    factors: &[Expr],
    start: usize,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<usize> {
    let left = factors.get(start)?;
    diracbar_spinor_arg(left, props)?;
    for end in (start + 1)..factors.len() {
        if spinor_metadata_of_expr(&factors[end], props).is_none() {
            continue;
        }
        let middle = match end - start {
            1 => None,
            2 => Some(factors[start + 1].clone()),
            _ => Some(Expr::mul(factors[(start + 1)..end].to_vec())),
        };
        if bilinear_is_forbidden_by_metadata_impl(
            left,
            middle.as_ref(),
            &factors[end],
            props,
            Some(interner),
        ) {
            return Some(end + 1);
        }
        return None;
    }
    None
}

/// Simplify supported 4D Majorana and Weyl spinor bilinears using metadata selection rules.
///
/// The simplifier scans products for `bar(psi) [sector/projector] chi`
/// bilinear windows. It replaces a containing product by exact `0` only when
/// structured metadata proves that the bilinear is forbidden, including
/// explicit Weyl-chirality projector/action mismatches and the supported 4D
/// Majorana self-bilinear sector table (`S`, `P`, `V`, `A`, `T`, with `V` and
/// `T` forbidden for a self-bilinear in the declared convention). Expressions
/// with insufficient metadata are returned unchanged.
pub fn simplify_spinor_bilinear_selection_rules(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| simplify_spinor_bilinear_selection_rules(term, props, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let simplified = factors
                .iter()
                .map(|factor| simplify_spinor_bilinear_selection_rules(factor, props, interner))
                .collect::<Vec<_>>();
            let mut idx = 0usize;
            while idx < simplified.len() {
                if let Some(_next) = bilinear_window_is_forbidden(&simplified, idx, props, interner)
                {
                    return Expr::zero();
                }
                idx += 1;
            }
            Expr::mul(simplified)
        }
        Expr::Pow(base, exp) => Expr::pow(
            simplify_spinor_bilinear_selection_rules(base, props, interner),
            simplify_spinor_bilinear_selection_rules(exp, props, interner),
        ),
        Expr::Neg(inner) => Expr::neg(simplify_spinor_bilinear_selection_rules(
            inner, props, interner,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_spinor_bilinear_selection_rules(
                re, props, interner,
            )),
            Box::new(simplify_spinor_bilinear_selection_rules(
                im, props, interner,
            )),
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| simplify_spinor_bilinear_selection_rules(arg, props, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(simplify_spinor_bilinear_selection_rules(
                body, props, interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(simplify_spinor_bilinear_selection_rules(
                lhs, props, interner,
            )),
            Box::new(simplify_spinor_bilinear_selection_rules(
                rhs, props, interner,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        simplify_spinor_bilinear_selection_rules(value, props, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(simplify_spinor_bilinear_selection_rules(
                base, props, interner,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(simplify_spinor_bilinear_selection_rules(
                inner, props, interner,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(simplify_spinor_bilinear_selection_rules(
                value, props, interner,
            )),
            Box::new(simplify_spinor_bilinear_selection_rules(
                body, props, interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| simplify_spinor_bilinear_selection_rules(item, props, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| {
                            simplify_spinor_bilinear_selection_rules(entry, props, interner)
                        })
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

/// Simplify canonical chiral projectors and their action on chiral Weyl spinors.
///
/// The rules include `P_L^2 = P_L`, `P_R^2 = P_R`, orthogonality
/// `P_L P_R = P_R P_L = 0`, completeness `P_L + P_R = 1`, and projector
/// action on Weyl spinors with explicit chirality metadata.
pub fn simplify_chiral_projectors(
    expr: &Expr,
    props: &dyn PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let simplified_terms = terms
                .iter()
                .map(|term| simplify_chiral_projectors(term, props, interner))
                .collect::<Vec<_>>();
            let has_left = simplified_terms.iter().any(|term| {
                chiral_projector_kind(term, interner) == Some(ChiralProjectorKind::Left)
            });
            let has_right = simplified_terms.iter().any(|term| {
                chiral_projector_kind(term, interner) == Some(ChiralProjectorKind::Right)
            });
            if has_left && has_right && simplified_terms.len() == 2 {
                Expr::one()
            } else {
                Expr::add(simplified_terms)
            }
        }
        Expr::Mul(factors) => {
            let mut out = Vec::new();
            let simplified = coalesce_chiral_projector_factors(
                factors
                    .iter()
                    .map(|factor| simplify_chiral_projectors(factor, props, interner))
                    .collect::<Vec<_>>(),
                interner,
            );
            let mut idx = 0usize;
            while idx < simplified.len() {
                if idx + 1 < simplified.len() {
                    let left_kind = chiral_projector_kind(&simplified[idx], interner);
                    let right_kind = chiral_projector_kind(&simplified[idx + 1], interner);
                    match (left_kind, right_kind) {
                        (Some(lhs), Some(rhs)) if lhs == rhs => {
                            out.push(chiral_projector_expr(lhs, interner));
                            idx += 2;
                            continue;
                        }
                        (Some(_), Some(_)) => return Expr::zero(),
                        (Some(projector), None) => {
                            if let Some(applied) = apply_chiral_projector_to_spinor(
                                projector,
                                &simplified[idx + 1],
                                props,
                            ) {
                                if applied == Expr::zero() {
                                    return Expr::zero();
                                }
                                out.push(applied);
                                idx += 2;
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
                out.push(simplified[idx].clone());
                idx += 1;
            }
            Expr::mul(out)
        }
        Expr::Pow(base, exp) => {
            let simplified_base = simplify_chiral_projectors(base, props, interner);
            if matches!(exp.as_ref(), Expr::Int(n) if *n == BigInt::from(2usize)) {
                if let Some(kind) = chiral_projector_kind(&simplified_base, interner) {
                    return chiral_projector_expr(kind, interner);
                }
            }
            Expr::pow(
                simplified_base,
                simplify_chiral_projectors(exp, props, interner),
            )
        }
        Expr::Neg(inner) => Expr::neg(simplify_chiral_projectors(inner, props, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_chiral_projectors(re, props, interner)),
            Box::new(simplify_chiral_projectors(im, props, interner)),
        ),
        Expr::Call(sym, args) => {
            if args.is_empty() {
                if let Some(kind) = chiral_projector_kind(expr, interner) {
                    return chiral_projector_expr(kind, interner);
                }
            }
            Expr::Call(
                *sym,
                args.iter()
                    .map(|arg| simplify_chiral_projectors(arg, props, interner))
                    .collect(),
            )
        }
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(simplify_chiral_projectors(base, props, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(simplify_chiral_projectors(inner, props, interner)),
            *rel,
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| simplify_chiral_projectors(item, props, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| simplify_chiral_projectors(entry, props, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => {
            if let Some(kind) = chiral_projector_kind(expr, interner) {
                chiral_projector_expr(kind, interner)
            } else {
                expr.clone()
            }
        }
    }
}

fn expand_diracbar_inner(inner: &Expr, diracbar_sym: lasso::Spur, gamma_sym: lasso::Spur) -> Expr {
    if let Expr::Neg(nested) = inner {
        return Expr::neg(expand_diracbar_inner(nested, diracbar_sym, gamma_sym));
    }

    let Expr::Mul(factors) = inner else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    if factors.len() > 1 {
        if let Expr::Int(n) = &factors[0] {
            if *n == (-1).into() {
                return Expr::neg(expand_diracbar_inner(
                    &Expr::mul(factors[1..].to_vec()),
                    diracbar_sym,
                    gamma_sym,
                ));
            }
        }
    }

    let mut gamma_chain = Vec::new();
    let mut spinor = None;
    for factor in factors {
        if is_gamma_expr(factor, gamma_sym) && spinor.is_none() {
            gamma_chain.push(factor.clone());
        } else if spinor.is_none() {
            spinor = Some(factor.clone());
        } else {
            return Expr::Call(diracbar_sym, vec![inner.clone()]);
        }
    }

    if gamma_chain.is_empty() {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    }

    let Some(spinor) = spinor else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    let total_gamma_indices: usize = gamma_chain
        .iter()
        .filter_map(|gamma| gamma_index_count(gamma, gamma_sym))
        .sum();
    let mut factors = vec![Expr::Call(diracbar_sym, vec![spinor])];
    factors.extend(gamma_chain.into_iter().rev());
    let result = Expr::mul(factors);

    if (total_gamma_indices * total_gamma_indices.saturating_sub(1) / 2) % 2 == 1 {
        Expr::neg(result)
    } else {
        result
    }
}

pub fn expand_diracbar(
    expr: &Expr,
    diracbar_sym: lasso::Spur,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = (metric_sym, interner);
    match expr {
        Expr::Call(f, args) if *f == diracbar_sym && args.len() == 1 => {
            let inner = expand_diracbar(&args[0], diracbar_sym, gamma_sym, metric_sym, interner);
            expand_diracbar_inner(&inner, diracbar_sym, gamma_sym)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| expand_diracbar(term, diracbar_sym, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    expand_diracbar(factor, diracbar_sym, gamma_sym, metric_sym, interner)
                })
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(expand_diracbar(
            inner,
            diracbar_sym,
            gamma_sym,
            metric_sym,
            interner,
        )),
        Expr::Pow(base, exp) => Expr::pow(
            expand_diracbar(base, diracbar_sym, gamma_sym, metric_sym, interner),
            expand_diracbar(exp, diracbar_sym, gamma_sym, metric_sym, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_diracbar(
                re,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
            Box::new(expand_diracbar(
                im,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| expand_diracbar(arg, diracbar_sym, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(expand_diracbar(
                base,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
            indices.clone(),
        ),
        _ => expr.clone(),
    }
}

fn expand_diracbar_full_inner(
    inner: &Expr,
    diracbar_sym: lasso::Spur,
    metadata: &DiracBarMetadata,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if let Expr::Add(terms) = inner {
        return Expr::add(
            terms
                .iter()
                .map(|term| {
                    expand_diracbar_full_inner(term, diracbar_sym, metadata, properties, interner)
                })
                .collect(),
        );
    }

    if let Expr::Neg(nested) = inner {
        return Expr::neg(expand_diracbar_full_inner(
            nested,
            diracbar_sym,
            metadata,
            properties,
            interner,
        ));
    }

    if let Some((_nested_bar_sym, nested_inner)) = is_diracbar_call(inner, properties) {
        return adjoint_of(Expr::Call(diracbar_sym, vec![nested_inner]), interner);
    }

    let Expr::Mul(factors) = inner else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    if factors.len() > 1 {
        if let Expr::Int(n) = &factors[0] {
            if *n == (-1).into() {
                return Expr::neg(expand_diracbar_full_inner(
                    &Expr::mul(factors[1..].to_vec()),
                    diracbar_sym,
                    metadata,
                    properties,
                    interner,
                ));
            }
        }
    }

    let mut gamma_chain = Vec::new();
    let mut scalar_factors = Vec::new();
    let mut spinor = None;
    for factor in factors {
        if gamma_metadata_of_expr(factor, properties).is_some() && spinor.is_none() {
            if let Some(expected_gamma_symbol) = metadata.gamma_symbol {
                if property_sym(factor) != Some(expected_gamma_symbol) {
                    return Expr::Call(diracbar_sym, vec![inner.clone()]);
                }
            }
            gamma_chain.push(factor.clone());
        } else if is_diracbar_scalar_factor(factor, properties) && spinor.is_none() {
            scalar_factors.push(factor.clone());
        } else if spinor.is_none() {
            if let Some(expected_spinor_family) = metadata.spinor_family {
                if structured_spinor_family(factor, properties)
                    .is_some_and(|family| family != expected_spinor_family)
                {
                    return Expr::Call(diracbar_sym, vec![inner.clone()]);
                }
            }
            spinor = Some(factor.clone());
        } else {
            return Expr::Call(diracbar_sym, vec![inner.clone()]);
        }
    }

    if gamma_chain.is_empty() && scalar_factors.is_empty() {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    }
    let Some(spinor) = spinor else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    let ordered_gamma_chain = if metadata.reverse_gamma_order {
        reverse_gamma_chain(&gamma_chain)
    } else {
        gamma_chain
    };
    let mut factors = scalar_factors
        .into_iter()
        .map(|factor| conjugate_scalar_factor(factor, interner))
        .collect::<Vec<_>>();
    factors.push(Expr::Call(diracbar_sym, vec![spinor]));
    factors.extend(
        ordered_gamma_chain
            .iter()
            .map(|gamma| diracbar_action_on_gamma(gamma, properties, interner)),
    );
    Expr::mul(factors)
}

fn expand_diracbar_full_legacy_inner(
    inner: &Expr,
    diracbar_sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Expr {
    if let Expr::Neg(nested) = inner {
        return Expr::neg(expand_diracbar_full_legacy_inner(
            nested,
            diracbar_sym,
            properties,
        ));
    }

    let Expr::Mul(factors) = inner else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    if factors.len() > 1 {
        if let Expr::Int(n) = &factors[0] {
            if *n == (-1).into() {
                return Expr::neg(expand_diracbar_full_legacy_inner(
                    &Expr::mul(factors[1..].to_vec()),
                    diracbar_sym,
                    properties,
                ));
            }
        }
    }

    let mut gamma_chain = Vec::new();
    let mut spinor = None;
    for factor in factors {
        if gamma_metadata_of_expr(factor, properties).is_some() && spinor.is_none() {
            gamma_chain.push(factor.clone());
        } else if spinor.is_none() {
            spinor = Some(factor.clone());
        } else {
            return Expr::Call(diracbar_sym, vec![inner.clone()]);
        }
    }

    if gamma_chain.is_empty() {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    }
    let Some(spinor) = spinor else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    let total_gamma_indices: usize = gamma_chain
        .iter()
        .filter_map(|gamma| gamma_expr_data(gamma, properties).map(|data| data.indices.len()))
        .sum();
    let mut factors = vec![Expr::Call(diracbar_sym, vec![spinor])];
    factors.extend(gamma_chain.into_iter().rev());
    let result = Expr::mul(factors);

    if ((total_gamma_indices * (total_gamma_indices + 1)) / 2) % 2 == 1 {
        Expr::neg(result)
    } else {
        result
    }
}

pub fn expand_diracbar_full(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Call(f, args) if args.len() == 1 => {
            if let Some(metadata) = declared_diracbar_metadata_of_symbol(*f, properties) {
                let inner = expand_diracbar_full(&args[0], properties, interner);
                return expand_diracbar_full_inner(&inner, *f, &metadata, properties, interner);
            }
            if !properties.has_property_kind(*f, &TensorProperty::DiracBar) {
                return Expr::Call(
                    *f,
                    args.iter()
                        .map(|arg| expand_diracbar_full(arg, properties, interner))
                        .collect(),
                );
            }
            let inner = expand_diracbar_full(&args[0], properties, interner);
            expand_diracbar_full_legacy_inner(&inner, *f, properties)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| expand_diracbar_full(term, properties, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| expand_diracbar_full(factor, properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(expand_diracbar_full(inner, properties, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            expand_diracbar_full(base, properties, interner),
            expand_diracbar_full(exp, properties, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_diracbar_full(re, properties, interner)),
            Box::new(expand_diracbar_full(im, properties, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| expand_diracbar_full(arg, properties, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(expand_diracbar_full(base, properties, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(expand_diracbar_full(inner, properties, interner)),
            *rel,
        ),
        _ => expr.clone(),
    }
}

pub fn diracbar_sort(
    expr: &Expr,
    diracbar_sym: lasso::Spur,
    gamma_sym: lasso::Spur,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = (operators, interner);
    match expr {
        Expr::Mul(factors) => {
            let sorted = factors
                .iter()
                .map(|factor| diracbar_sort(factor, diracbar_sym, gamma_sym, operators, interner))
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            let mut cursor = 0usize;
            while cursor < sorted.len() {
                let factor = &sorted[cursor];
                if matches!(factor, Expr::Call(f, _) if *f == diracbar_sym) {
                    out.push(factor.clone());
                    cursor += 1;
                    let mut gammas = Vec::new();
                    let mut spinor = None;
                    let mut others = Vec::new();
                    while cursor < sorted.len() {
                        let next = &sorted[cursor];
                        if matches!(next, Expr::Call(f, _) if *f == diracbar_sym) {
                            break;
                        }
                        if is_gamma_expr(next, gamma_sym) {
                            gammas.push(next.clone());
                        } else if spinor.is_none() {
                            spinor = Some(next.clone());
                        } else {
                            others.push(next.clone());
                        }
                        cursor += 1;
                    }
                    out.extend(gammas);
                    if let Some(spinor) = spinor {
                        out.push(spinor);
                    }
                    out.extend(others);
                } else {
                    out.push(factor.clone());
                    cursor += 1;
                }
            }
            Expr::mul(out)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| diracbar_sort(term, diracbar_sym, gamma_sym, operators, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(diracbar_sort(
            inner,
            diracbar_sym,
            gamma_sym,
            operators,
            interner,
        )),
        Expr::Pow(base, exp) => Expr::pow(
            diracbar_sort(base, diracbar_sym, gamma_sym, operators, interner),
            diracbar_sort(exp, diracbar_sym, gamma_sym, operators, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(diracbar_sort(
                re,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
            Box::new(diracbar_sort(
                im,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| diracbar_sort(arg, diracbar_sym, gamma_sym, operators, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(diracbar_sort(
                base,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
            indices.clone(),
        ),
        _ => expr.clone(),
    }
}

pub fn fierz_full(
    expr: &Expr,
    spinor_order: &[Expr; 4],
    dimension: usize,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let parsed = parse_fierz_input(expr, Some(properties), interner).ok()?;
    let desired = spinor_order
        .iter()
        .map(property_sym)
        .collect::<Option<Vec<_>>>()?;
    let desired = [desired[0], desired[1], desired[2], desired[3]];
    let right_order = [
        parsed.pair.psi1,
        parsed.pair.psi2,
        parsed.pair.psi3,
        parsed.pair.psi4,
    ];
    let wrong_order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    if desired == right_order {
        return None;
    }
    if desired != wrong_order {
        return None;
    }

    let spinor_dim = if dimension % 2 == 0 {
        1usize << (dimension / 2)
    } else {
        1usize << ((dimension - 1) / 2)
    };
    let use_weyl = spinor_order
        .iter()
        .all(|spinor| is_weyl_spinor_expr(spinor, properties));
    let max_rank = if use_weyl { dimension / 2 } else { dimension };
    let gamma_sym = property_sym(&Expr::Call(interner.get_or_intern("gamma"), vec![]))
        .unwrap_or_else(|| interner.get_or_intern("gamma"));

    let mut used = HashSet::new();
    collect_all_index_names(expr, &mut used);
    let family = properties
        .index_families()
        .and_then(|families| families.values().next().cloned());

    let mut terms = Vec::new();
    for rank in 0..=max_rank {
        let coeff = -BigRational::new(BigInt::one(), BigInt::from(spinor_dim) * factorial(rank));
        let gamma_indices = if let Some(info) = &family {
            (0..rank)
                .map(|_| fresh_dummy_from_family(info, &mut used, interner))
                .collect::<Vec<_>>()
        } else {
            (0..rank)
                .map(|idx| interner.get_or_intern(&format!("_fierz{rank}_{idx}")))
                .collect::<Vec<_>>()
        };

        let first_gamma = if gamma_indices.is_empty() {
            None
        } else {
            Some(Expr::Call(
                gamma_sym,
                gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
            ))
        };
        let mut second_chain = Vec::new();
        if !parsed.pair.gamma_a.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                parsed
                    .pair
                    .gamma_a
                    .iter()
                    .map(|idx| Expr::Sym(*idx))
                    .collect(),
            ));
        }
        if !gamma_indices.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
            ));
        }
        if !parsed.pair.gamma_b.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                parsed
                    .pair
                    .gamma_b
                    .iter()
                    .rev()
                    .map(|idx| Expr::Sym(*idx))
                    .collect(),
            ));
        }

        let mut first_bilinear = vec![Expr::Sym(desired[0])];
        if let Some(gamma) = first_gamma {
            first_bilinear.push(gamma);
        }
        first_bilinear.push(Expr::Sym(desired[1]));

        let mut second_bilinear = vec![Expr::Sym(desired[2])];
        second_bilinear.extend(second_chain);
        second_bilinear.push(Expr::Sym(desired[3]));

        let mut term_factors = parsed.pair.remaining_factors.clone();
        term_factors.push(Expr::Rational(coeff));
        term_factors.push(Expr::mul(first_bilinear));
        term_factors.push(Expr::mul(second_bilinear));
        terms.push(Expr::mul(term_factors));
    }

    Some(Expr::add(terms))
}

pub fn split_gamma_full(
    gamma_expr: &Expr,
    on_back: bool,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let Some(data) = gamma_expr_data(gamma_expr, properties) else {
        return gamma_expr.clone();
    };
    if gamma_convention_conflict_of_expr(gamma_expr, properties) {
        return qm_error_expr("join_gamma_convention_mismatch", gamma_expr, interner);
    }
    if data.indices.len() <= 1 {
        return gamma_expr.clone();
    }

    let metric = gamma_metadata_of_expr(gamma_expr, properties)
        .and_then(|metadata| metadata.metric_symbol)
        .map(Expr::Sym)
        .unwrap_or_else(|| Expr::Sym(interner.get_or_intern("g")));
    let (left_indices, right_indices) = if on_back {
        (
            data.indices[..data.indices.len() - 1].to_vec(),
            vec![data.indices[data.indices.len() - 1].clone()],
        )
    } else {
        (vec![data.indices[0].clone()], data.indices[1..].to_vec())
    };
    let lhs = build_gamma_expr(&data.head, &left_indices);
    let rhs = build_gamma_expr(&data.head, &right_indices);
    let product = Expr::mul(vec![lhs.clone(), rhs.clone()]);
    let joined = join_gamma_full(&lhs, &rhs, None, true, true, &metric, properties, interner);
    if matches!(joined, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_family_mismatch" || interner.resolve(sym) == "join_gamma_convention_mismatch")
    {
        return joined;
    }

    let joined_terms = match joined {
        Expr::Add(terms) => terms,
        other => vec![other],
    };
    let original_data = gamma_expr_data(gamma_expr, properties);
    let mut rest = Vec::new();
    for term in joined_terms {
        let rank_match = match &term {
            Expr::Mul(factors) => factors.iter().any(|factor| {
                gamma_expr_data(factor, properties)
                    .map(|candidate| {
                        original_data
                            .as_ref()
                            .map(|d| candidate.indices == d.indices)
                            .unwrap_or(false)
                    })
                    .unwrap_or(false)
            }),
            _ => gamma_expr_data(&term, properties)
                .map(|candidate| {
                    original_data
                        .as_ref()
                        .map(|d| candidate.indices == d.indices)
                        .unwrap_or(false)
                })
                .unwrap_or(false),
        };
        if !rank_match {
            rest.push(Expr::neg(term));
        }
    }
    let mut out_terms = vec![product];
    out_terms.extend(rest);
    Expr::add(out_terms)
}

fn fresh_fierz_indices(
    rank: usize,
    counter: &mut usize,
    interner: &ax_ir::Interner,
) -> Vec<lasso::Spur> {
    (0..rank)
        .map(|_| {
            let name = format!("_f{}", *counter);
            *counter += 1;
            interner.get_or_intern(&name)
        })
        .collect()
}

fn bilinear_expr(
    left: lasso::Spur,
    gamma_indices: &[lasso::Spur],
    right: lasso::Spur,
    gamma_sym: lasso::Spur,
) -> Expr {
    let mut factors = vec![Expr::Sym(left)];
    if !gamma_indices.is_empty() {
        factors.push(Expr::Call(
            gamma_sym,
            gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
        ));
    }
    factors.push(Expr::Sym(right));
    Expr::mul(factors)
}

fn replace_fierz_basis_placeholders(
    expr: &Expr,
    replacements: &HashMap<lasso::Spur, Expr>,
) -> Expr {
    match expr {
        Expr::Sym(sym) => replacements.get(sym).cloned().unwrap_or(Expr::Sym(*sym)),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| replace_fierz_basis_placeholders(term, replacements))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| replace_fierz_basis_placeholders(factor, replacements))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            replace_fierz_basis_placeholders(base, replacements),
            replace_fierz_basis_placeholders(exp, replacements),
        ),
        Expr::Neg(inner) => Expr::neg(replace_fierz_basis_placeholders(inner, replacements)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(replace_fierz_basis_placeholders(re, replacements)),
            Box::new(replace_fierz_basis_placeholders(im, replacements)),
        ),
        Expr::Call(sym, args) => Expr::Call(
            *sym,
            args.iter()
                .map(|arg| replace_fierz_basis_placeholders(arg, replacements))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(replace_fierz_basis_placeholders(base, replacements)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(replace_fierz_basis_placeholders(inner, replacements)),
            *rel,
        ),
        other => other.clone(),
    }
}

fn instantiate_fierz_basis_expr(
    basis: &Expr,
    counter: &mut usize,
    interner: &ax_ir::Interner,
) -> Expr {
    let rank = fierz_basis_rank(basis, interner).unwrap_or(0);
    if rank == 0 || rank == 4 {
        return basis.clone();
    }
    let mut replacements = HashMap::new();
    let placeholders = ["mu", "nu"];
    for name in placeholders.iter().take(rank.min(placeholders.len())) {
        let fresh = interner.get_or_intern(&format!("_f{}", *counter));
        *counter += 1;
        replacements.insert(interner.get_or_intern(name), Expr::Sym(fresh));
    }
    replace_fierz_basis_placeholders(basis, &replacements)
}

fn bilinear_expr_with_basis(left: lasso::Spur, basis: Expr, right: lasso::Spur) -> Expr {
    let mut factors = vec![Expr::Sym(left)];
    if basis != Expr::one() {
        factors.push(basis);
    }
    factors.push(Expr::Sym(right));
    Expr::mul(factors)
}

fn fierz_error_expr(error: &FierzError, expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let sym = interner.get_or_intern(error.symbol_name());
    Expr::Call(sym, vec![expr.clone()])
}

fn validate_fierz_spinor_metadata(
    pair: &BilinearPair,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
) -> Result<(), FierzError> {
    let Some(properties) = properties else {
        return Ok(());
    };

    let spinors = [pair.psi1, pair.psi2, pair.psi3, pair.psi4]
        .into_iter()
        .map(|sym| spinor_metadata_of_expr(&Expr::Sym(sym), properties))
        .collect::<Vec<_>>();

    let dimensions = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().and_then(|metadata| metadata.dimension))
        .collect::<HashSet<_>>();
    if dimensions.len() > 1 {
        return Err(FierzError::IncompatibleSpinorDimension);
    }

    let chiralities = spinors
        .iter()
        .filter_map(|metadata| {
            metadata
                .as_ref()
                .and_then(|metadata| metadata.chirality.clone())
        })
        .collect::<HashSet<_>>();
    if chiralities.len() > 1 {
        return Err(FierzError::IncompatibleSpinorChirality);
    }

    let classes = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().map(|metadata| metadata.class.clone()))
        .collect::<HashSet<_>>();
    if classes.len() > 1 {
        return Err(FierzError::IncompatibleSpinorMetadata);
    }

    let families = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().and_then(|metadata| metadata.index_family))
        .collect::<HashSet<_>>();
    if families.len() > 1 {
        return Err(FierzError::IncompatibleSpinorMetadata);
    }

    Ok(())
}

fn structured_fierz_spinor_metadata(
    pair: &BilinearPair,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Result<Option<SpinorMetadata>, FierzError> {
    let spinors = [pair.psi1, pair.psi2, pair.psi3, pair.psi4]
        .into_iter()
        .map(|sym| declared_spinor_metadata_of_symbol(sym, properties))
        .collect::<Vec<_>>();
    if spinors.iter().all(Option::is_none) {
        return Ok(None);
    }
    let Some(first) = spinors[0].clone() else {
        return Err(FierzError::InsufficientConventionData);
    };
    if spinors.iter().any(Option::is_none) {
        return Err(FierzError::InsufficientConventionData);
    }
    Ok(Some(first))
}

fn structured_fierz_gamma_convention(
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<GammaConventionMetadata> {
    declared_gamma_convention_metadata_of_symbol(interner.get_or_intern("gamma"), properties)
}

fn build_fierz_sum(
    parsed: ParsedFierzInput,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let pair = parsed.pair;
    let expected = [pair.psi1, pair.psi4, pair.psi3, pair.psi2];
    let explicit_set: HashSet<_> = spinor_order.iter().copied().collect();
    if explicit_set.len() != 4 || explicit_set != expected.iter().copied().collect() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let input_order = [pair.psi1, pair.psi2, pair.psi3, pair.psi4];
    let mut sign = anticommuting_reorder_sign(&input_order, &spinor_order, properties, interner)?;
    if parsed.sign < 0 {
        sign = -sign;
    }

    let coeffs = fierz_coefficients(dim);
    let gamma_sym = interner.get_or_intern("gamma");
    let [psi1, psi4, psi3, psi2] = spinor_order;
    let mut counter = 0usize;

    let terms = coeffs
        .into_iter()
        .map(|(coefficient, rank)| {
            let gamma_indices = fresh_fierz_indices(rank, &mut counter, interner);
            let first = bilinear_expr(psi1, &gamma_indices, psi4, gamma_sym);
            let second = bilinear_expr(psi3, &gamma_indices, psi2, gamma_sym);

            let mut factors = pair.remaining_factors.clone();
            let signed_coefficient = if sign < 0 { -coefficient } else { coefficient };
            factors.push(Expr::Rational(signed_coefficient));
            factors.push(first);
            factors.push(second);
            Expr::mul(factors)
        })
        .collect();
    Ok(ax_ir::Expr::add(terms))
}

fn build_fierz_sum_with_basis(
    parsed: ParsedFierzInput,
    spinor_order: [lasso::Spur; 4],
    properties: &dyn ax_tensor::PropertyLookup,
    basis: &[Expr],
    coefficients: &[Expr],
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let pair = parsed.pair;
    if basis.len() != coefficients.len() || basis.is_empty() {
        return Err(FierzError::InsufficientConventionData);
    }

    let expected = [pair.psi1, pair.psi4, pair.psi3, pair.psi2];
    let explicit_set: HashSet<_> = spinor_order.iter().copied().collect();
    if explicit_set.len() != 4 || explicit_set != expected.iter().copied().collect() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let input_order = [pair.psi1, pair.psi2, pair.psi3, pair.psi4];
    let mut sign =
        anticommuting_reorder_sign(&input_order, &spinor_order, Some(properties), interner)?;
    if parsed.sign < 0 {
        sign = -sign;
    }

    let [psi1, psi4, psi3, psi2] = spinor_order;
    let mut counter = 0usize;
    let terms = basis
        .iter()
        .zip(coefficients.iter())
        .map(|(basis_sector, coefficient)| {
            let instantiated_basis =
                instantiate_fierz_basis_expr(basis_sector, &mut counter, interner);
            let first = bilinear_expr_with_basis(psi1, instantiated_basis.clone(), psi4);
            let second = bilinear_expr_with_basis(psi3, instantiated_basis, psi2);

            let mut factors = pair.remaining_factors.clone();
            let signed_coefficient = if sign < 0 {
                Expr::neg(coefficient.clone())
            } else {
                coefficient.clone()
            };
            factors.push(signed_coefficient);
            factors.push(first);
            factors.push(second);
            Expr::mul(factors)
        })
        .collect();
    Ok(ax_ir::Expr::add(terms))
}

pub fn try_fierz(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, None, interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, None)?;
    build_fierz_sum(parsed, dim, spinor_order, None, interner)
}

pub fn try_fierz_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, Some(properties), interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, Some(properties))?;
    if let Some(spinor_meta) = structured_fierz_spinor_metadata(&parsed.pair, properties)? {
        if spinor_meta.dimension != Some(dim) {
            return Err(FierzError::IncompatibleSpinorDimension);
        }
        let gamma_meta = structured_fierz_gamma_convention(properties, interner)
            .ok_or(FierzError::InsufficientConventionData)?;
        let basis = fierz_basis_for_metadata(&spinor_meta, Some(&gamma_meta), interner);
        let coefficients =
            fierz_coefficients_for_basis(&basis, &spinor_meta, Some(&gamma_meta), interner)?;
        return build_fierz_sum_with_basis(
            parsed,
            spinor_order,
            properties,
            &basis,
            &coefficients,
            interner,
        );
    }
    build_fierz_sum(parsed, dim, spinor_order, Some(properties), interner)
}

/// Apply Fierz identity to a concrete product of two spinor bilinears.
pub fn fierz(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match try_fierz(expr, dim, spinor_order, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

pub fn fierz_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match try_fierz_with_properties(expr, dim, spinor_order, properties, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

pub fn try_fierz_auto(
    expr: &ax_ir::Expr,
    dim: usize,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, None, interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, None)?;
    let order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    build_fierz_sum(parsed, dim, order, None, interner)
}

pub fn try_fierz_auto_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, Some(properties), interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, Some(properties))?;
    let order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    if let Some(spinor_meta) = structured_fierz_spinor_metadata(&parsed.pair, properties)? {
        if spinor_meta.dimension != Some(dim) {
            return Err(FierzError::IncompatibleSpinorDimension);
        }
        let gamma_meta = structured_fierz_gamma_convention(properties, interner)
            .ok_or(FierzError::InsufficientConventionData)?;
        let basis = fierz_basis_for_metadata(&spinor_meta, Some(&gamma_meta), interner);
        let coefficients =
            fierz_coefficients_for_basis(&basis, &spinor_meta, Some(&gamma_meta), interner)?;
        return build_fierz_sum_with_basis(
            parsed,
            order,
            properties,
            &basis,
            &coefficients,
            interner,
        );
    }
    build_fierz_sum(parsed, dim, order, Some(properties), interner)
}

pub fn fierz_auto(expr: &ax_ir::Expr, dim: usize, interner: &ax_ir::Interner) -> ax_ir::Expr {
    match try_fierz_auto(expr, dim, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

/// Return the abstract Fierz coefficient expansion used by the old API.
pub fn fierz_simple(dim: usize, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let coeffs = fierz_coefficients(dim);
    let terms: Vec<ax_ir::Expr> = coeffs
        .iter()
        .map(|(c, k)| {
            ax_ir::Expr::mul(vec![
                ax_ir::Expr::Rational(c.clone()),
                ax_ir::Expr::Call(
                    interner.get_or_intern("gamma_basis"),
                    vec![ax_ir::Expr::Int(BigInt::from(*k))],
                ),
            ])
        })
        .collect();
    ax_ir::Expr::add(terms)
}

// ─── split_gamma ──────────────────────────────────────────────────────────────

/// Split one index off a multi-index antisymmetric gamma matrix.
///
/// Uses the join identity in reverse:
/// ```text
/// γ^{a} γ^{b…z} = γ^{a b…z} + contraction terms
/// γ^{a b…z} γ^{z} = γ^{a b…} + contraction terms
/// ```
/// So:
/// ```text
/// γ^{a b…z} = γ^{a b…} γ^{z} − (contraction terms)   [on_back = true]
/// γ^{a b…z} = γ^{a} γ^{b…z} − (contraction terms)   [on_back = false]
/// ```
///
/// Parameters:
/// - `gamma_sym`: symbol for the gamma matrix
/// - `metric_sym`: symbol for the metric used in contractions
/// - `on_back`: if `true`, split the last index; if `false`, split the first
pub fn split_gamma(
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    on_back: bool,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = interner;
    match expr {
        Expr::Call(f, args) if *f == gamma_sym && args.len() > 1 => {
            let indices: Vec<lasso::Spur> = args
                .iter()
                .filter_map(|a| if let Expr::Sym(s) = a { Some(*s) } else { None })
                .collect();

            if indices.len() <= 1 {
                return expr.clone();
            }

            // Choose which index to split off and what remains
            let (split_idx, remaining_indices) = if on_back {
                let last = *indices.last().unwrap();
                (last, indices[..indices.len() - 1].to_vec())
            } else {
                let first = indices[0];
                (first, indices[1..].to_vec())
            };

            // Main term: γ(remaining) * γ(split)  [on_back]
            //         or γ(split) * γ(remaining)  [on_front]
            let main = if on_back {
                Expr::mul(vec![
                    make_gamma(&remaining_indices, gamma_sym),
                    make_gamma(&[split_idx], gamma_sym),
                ])
            } else {
                Expr::mul(vec![
                    make_gamma(&[split_idx], gamma_sym),
                    make_gamma(&remaining_indices, gamma_sym),
                ])
            };

            // Contraction terms come from the join identity:
            //   γ(remaining) γ(split) = γ(full) + Σ_k (±1) g^{split rem_k} γ(remaining \ rem_k)
            // Rearranging: γ(full) = main − Σ_k (±1) g^{split rem_k} γ(remaining \ rem_k)
            //
            // Signs: k-th contraction gets (-1)^k when splitting from back,
            //        and (-1)^k when splitting from front (same rule, position counts from 0).
            let mut all_terms = vec![main];

            for (k, &rem_idx) in remaining_indices.iter().enumerate() {
                // Sign: (-1)^k  (k is 0-based position in remaining)
                let negate = k % 2 != 0;

                let metric = Expr::Indexed(
                    Box::new(Expr::Sym(metric_sym)),
                    vec![
                        Index {
                            name: split_idx,
                            variance: Variance::Up,
                            index_type: None,
                        },
                        Index {
                            name: rem_idx,
                            variance: Variance::Up,
                            index_type: None,
                        },
                    ],
                );

                // Sub-gamma: remaining indices with rem_idx removed
                let sub_remaining: Vec<lasso::Spur> = remaining_indices
                    .iter()
                    .filter(|&&i| i != rem_idx)
                    .copied()
                    .collect();

                let sub_gamma = if sub_remaining.is_empty() {
                    Expr::one()
                } else {
                    make_gamma(&sub_remaining, gamma_sym)
                };

                let contraction = Expr::mul(vec![metric, sub_gamma]);
                // Subtract the contraction: − (±contraction)
                let signed = if negate {
                    contraction
                } else {
                    Expr::neg(contraction)
                };
                all_terms.push(signed);
            }

            Expr::add(all_terms)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| split_gamma(f, gamma_sym, metric_sym, on_back, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| split_gamma(t, gamma_sym, metric_sym, on_back, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(split_gamma(e, gamma_sym, metric_sym, on_back, interner)),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prop_map() -> HashMap<lasso::Spur, Vec<TensorProperty>> {
        HashMap::new()
    }

    fn spinor_selection_rule_props(
        interner: &ax_ir::Interner,
    ) -> (
        HashMap<lasso::Spur, Vec<TensorProperty>>,
        lasso::Spur,
        lasso::Spur,
    ) {
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![
                TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                    dimension: Some(4),
                    metric_symbol: Some(interner.get_or_intern("eta")),
                    index_family: Some(spin),
                    has_gamma5: true,
                }),
                TensorProperty::GammaConventionMeta(GammaConventionMetadata {
                    signature: ax_ir::MetricSignature::MostlyPlus,
                    clifford: CliffordConvention::PlusTwoG,
                    gamma5: Some(ax_ir::GammaFiveConvention::AbstractChiral),
                    epsilon_symbol: Some(interner.get_or_intern("eps")),
                    dimension: Some(4),
                }),
            ],
        );
        (props, bar, gamma)
    }

    struct SpinorBridgeProps {
        props: HashMap<lasso::Spur, Vec<TensorProperty>>,
        families: HashMap<lasso::Spur, ax_ir::IndexFamily>,
    }

    impl PropertyLookup for SpinorBridgeProps {
        fn get_properties(&self, name: lasso::Spur) -> Vec<ax_ir::TensorProperty> {
            self.props.get(&name).cloned().unwrap_or_default()
        }

        fn get_properties_with_indices(
            &self,
            name: lasso::Spur,
            _indices: &[ax_ir::Index],
            _successor: Option<(lasso::Spur, &[ax_ir::Index])>,
        ) -> Vec<ax_ir::TensorProperty> {
            self.get_properties(name)
        }

        fn has_property_kind(&self, name: lasso::Spur, kind: &ax_ir::TensorProperty) -> bool {
            self.get_properties(name)
                .into_iter()
                .any(|prop| std::mem::discriminant(&prop) == std::mem::discriminant(kind))
        }

        fn index_families(&self) -> Option<&HashMap<lasso::Spur, ax_ir::IndexFamily>> {
            Some(&self.families)
        }
    }

    fn spinor_bridge_props(
        interner: &ax_ir::Interner,
    ) -> (SpinorBridgeProps, lasso::Spur, lasso::Spur, lasso::Spur) {
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("s_a");
        let b = interner.get_or_intern("s_b");
        let c = interner.get_or_intern("s_c");
        let d = interner.get_or_intern("s_d");
        let mut props = HashMap::new();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: Some(interner.get_or_intern("eta")),
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Dirac,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        let mut families = HashMap::new();
        families.insert(
            spin,
            ax_ir::IndexFamily {
                name: spin,
                values: vec![a, b, c, d],
                position: ax_ir::IndexPosition::Free,
                dimension: Some(4),
                parent: None,
            },
        );
        (SpinorBridgeProps { props, families }, bar, gamma, psi)
    }

    fn operator_stats() -> HashMap<lasso::Spur, OperatorStatistics> {
        HashMap::new()
    }

    #[test]
    fn pauli_commutation() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let sy = pauli_y(&interner);
        let comm = commutator(&sx, &sy, &interner);
        let simplified = ax_eval::eval(&comm[0][0], &ax_eval::Env::new(), &interner);
        let expected = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::Int(2.into())));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn anticommutator_pauli() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let anti = anticommutator(&sx, &sx, &interner);
        let simplified_00 = ax_eval::eval(&anti[0][0], &ax_eval::Env::new(), &interner);
        assert_eq!(simplified_00, Expr::Int(2.into()));
    }

    #[test]
    fn ket_basis_vectors() {
        let interner = ax_ir::Interner::new();
        let ket0 = vec![Expr::one(), Expr::zero()];
        let ket1 = vec![Expr::zero(), Expr::one()];
        let inner = Expr::add(
            ket0.iter()
                .zip(ket1.iter())
                .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
                .collect::<Vec<_>>(),
        );
        let result = ax_eval::eval(&inner, &ax_eval::Env::new(), &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn trace_identity() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let result = gamma_trace_recursive(&[], g, &interner);
        assert_eq!(result, Expr::Int(4.into()));
    }

    #[test]
    fn trace_single_gamma_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let result = gamma_trace_recursive(&[mu], g, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn trace_two_gammas() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let result = gamma_trace_recursive(&[mu, nu], g_sym, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("4") && pp.contains("g"), "got: {}", pp);
    }

    #[test]
    fn trace_odd_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");
        let result = gamma_trace_recursive(&[mu, nu, rho], g, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn normal_order_puts_creation_first() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);
        let statistics = operator_stats();
        let properties = prop_map();

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &statistics, &properties, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2);
            assert_eq!(factors[0], Expr::Sym(a_dag));
            assert_eq!(factors[1], Expr::Sym(a));
        } else {
            panic!("expected Mul");
        }
    }

    #[test]
    fn normal_order_preserves_scalars() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);
        let statistics = operator_stats();
        let properties = prop_map();

        let expr = Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &statistics, &properties, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("3"), "got: {}", pp);
    }

    #[test]
    fn normal_order_bosonic_same_mode_adds_plus_identity() {
        let interner = ax_ir::Interner::new();
        let operators = HashMap::new();
        let statistics = operator_stats();
        let properties = prop_map();
        let a = interner.get_or_intern("a");
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
        ]);
        let result = normal_order_simple(&expr, &operators, &statistics, &properties, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            ]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn normal_order_fermionic_same_mode_adds_minus_identity_term() {
        let interner = ax_ir::Interner::new();
        let operators = HashMap::new();
        let c = interner.get_or_intern("c");
        let mut statistics = operator_stats();
        statistics.insert(c, OperatorStatistics::Fermionic);
        let properties = prop_map();
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
        ]);
        let result = normal_order_simple(&expr, &operators, &statistics, &properties, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            ])),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn normal_order_uses_mode_metadata_before_legacy_operator_statistics() {
        let interner = ax_ir::Interner::new();
        let c = interner.get_or_intern("c");
        let operators = HashMap::new();
        let mut statistics = operator_stats();
        statistics.insert(c, OperatorStatistics::Bosonic);
        let mut properties = prop_map();
        properties.insert(
            c,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Fermionic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
        ]);
        let result = normal_order_simple(&expr, &operators, &statistics, &properties, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            ])),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn wick_pairings_generates_all_complete_pairings() {
        let pairings = wick_pairings(&[0, 1, 2, 3]);
        assert_eq!(pairings.len(), 3);
        assert!(pairings.contains(&vec![(0, 1), (2, 3)]));
        assert!(pairings.contains(&vec![(0, 2), (1, 3)]));
        assert!(pairings.contains(&vec![(0, 3), (1, 2)]));
    }

    #[test]
    fn fermionic_pairing_sign_flips_for_crossing_pairing() {
        let noncrossing = fermionic_pairing_sign(&[(0, 1), (2, 3)], 4);
        let crossing = fermionic_pairing_sign(&[(0, 2), (1, 3)], 4);

        assert_eq!(noncrossing, Expr::one());
        assert_eq!(crossing, Expr::neg(Expr::one()));
    }

    #[test]
    fn wick_expand_fermionic_pairings_include_correct_signs() {
        let interner = ax_ir::Interner::new();
        let c1 = interner.get_or_intern("c1");
        let c2 = interner.get_or_intern("c2");
        let annihilation = interner.get_or_intern("annihilation");
        let creation = interner.get_or_intern("creation");

        let operators = HashMap::new();
        let mut statistics = operator_stats();
        statistics.insert(c1, OperatorStatistics::Fermionic);
        statistics.insert(c2, OperatorStatistics::Fermionic);
        let properties = prop_map();

        let expr = Expr::mul(vec![
            Expr::Call(annihilation, vec![Expr::Sym(c1)]),
            Expr::Call(annihilation, vec![Expr::Sym(c2)]),
            Expr::Call(creation, vec![Expr::Sym(c2)]),
            Expr::Call(creation, vec![Expr::Sym(c1)]),
        ]);

        let mut contractions = HashMap::new();
        contractions.insert((c1, c1), Expr::one());
        contractions.insert((c2, c2), Expr::one());

        let result = wick_expand(
            &expr,
            &operators,
            &statistics,
            &properties,
            &contractions,
            &interner,
        );

        let expected = Expr::add(vec![
            Expr::mul(vec![
                Expr::Call(creation, vec![Expr::Sym(c2)]),
                Expr::Call(creation, vec![Expr::Sym(c1)]),
                Expr::Call(annihilation, vec![Expr::Sym(c1)]),
                Expr::Call(annihilation, vec![Expr::Sym(c2)]),
            ]),
            Expr::neg(Expr::mul(vec![
                Expr::Call(creation, vec![Expr::Sym(c2)]),
                Expr::Call(annihilation, vec![Expr::Sym(c2)]),
            ])),
            Expr::neg(Expr::mul(vec![
                Expr::Call(creation, vec![Expr::Sym(c1)]),
                Expr::Call(annihilation, vec![Expr::Sym(c1)]),
            ])),
            Expr::one(),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn wick_expand_bosonic_path_is_unchanged() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let annihilation = interner.get_or_intern("annihilation");
        let creation = interner.get_or_intern("creation");

        let operators = HashMap::new();
        let statistics = operator_stats();
        let properties = prop_map();

        let expr = Expr::mul(vec![
            Expr::Call(annihilation, vec![Expr::Sym(a)]),
            Expr::Call(creation, vec![Expr::Sym(a)]),
        ]);

        let mut contractions = HashMap::new();
        contractions.insert((a, a), Expr::one());

        let result = wick_expand(
            &expr,
            &operators,
            &statistics,
            &properties,
            &contractions,
            &interner,
        );
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Call(creation, vec![Expr::Sym(a)]),
                Expr::Call(annihilation, vec![Expr::Sym(a)]),
            ]),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn time_ordered_constructor_uses_canonical_call() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let expr = time_ordered(Expr::Sym(a), &interner);

        assert_eq!(
            expr,
            Expr::Call(interner.get_or_intern("time_order"), vec![Expr::Sym(a)])
        );
    }

    #[test]
    fn anti_time_ordered_constructor_uses_canonical_call() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let expr = anti_time_ordered(Expr::Sym(a), &interner);

        assert_eq!(
            expr,
            Expr::Call(
                interner.get_or_intern("anti_time_order"),
                vec![Expr::Sym(a)]
            )
        );
    }

    #[test]
    fn bch_expand_order_one_is_a_plus_b() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("A"));
        let b = Expr::Sym(interner.get_or_intern("B"));

        assert_eq!(
            bch_expand(a.clone(), b.clone(), 1, &interner),
            Expr::add(vec![a, b])
        );
    }

    #[test]
    fn bch_expand_order_two_contains_half_commutator() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("A"));
        let b = Expr::Sym(interner.get_or_intern("B"));
        let result = bch_expand(a.clone(), b.clone(), 2, &interner);
        let Expr::Add(terms) = result else {
            panic!("expected additive BCH expansion");
        };

        assert!(terms.contains(&Expr::mul(vec![
            Expr::Rational(BigRational::new(1.into(), 2.into())),
            commutator_expr(a, b, &interner),
        ])));
    }

    #[test]
    fn bch_expand_order_four_contains_required_quartic_term() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("A"));
        let b = Expr::Sym(interner.get_or_intern("B"));
        let result = bch_expand(a.clone(), b.clone(), 4, &interner);
        let Expr::Add(terms) = result else {
            panic!("expected additive BCH expansion");
        };

        let quartic = Expr::mul(vec![
            Expr::Rational(BigRational::new((-1).into(), 24.into())),
            commutator_expr(
                b.clone(),
                commutator_expr(
                    a.clone(),
                    commutator_expr(
                        a,
                        commutator_expr(
                            Expr::Sym(interner.get_or_intern("A")),
                            Expr::Sym(interner.get_or_intern("B")),
                            &interner,
                        ),
                        &interner,
                    ),
                    &interner,
                ),
                &interner,
            ),
        ]);

        assert!(terms.contains(&quartic));
    }

    #[test]
    fn bch_expand_order_zero_is_zero() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("A"));
        let b = Expr::Sym(interner.get_or_intern("B"));

        assert_eq!(bch_expand(a, b, 0, &interner), Expr::zero());
    }

    #[test]
    fn displacement_operator_series_order_zero_is_one() {
        let interner = ax_ir::Interner::new();
        let alpha = Expr::Sym(interner.get_or_intern("alpha"));
        let mode = Expr::Sym(interner.get_or_intern("a"));

        assert_eq!(
            displacement_operator_series(alpha, mode, 0, &interner),
            Expr::one()
        );
    }

    #[test]
    fn displacement_operator_series_order_one_is_one_plus_generator() {
        let interner = ax_ir::Interner::new();
        let alpha = Expr::Sym(interner.get_or_intern("alpha"));
        let mode = Expr::Sym(interner.get_or_intern("a"));
        let generator = Expr::add(vec![
            Expr::mul(vec![
                alpha.clone(),
                Expr::Call(interner.get_or_intern("creation"), vec![mode.clone()]),
            ]),
            Expr::neg(Expr::mul(vec![
                Expr::Call(interner.get_or_intern("conj"), vec![alpha]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![mode]),
            ])),
        ]);

        assert_eq!(
            displacement_operator_series(
                Expr::Sym(interner.get_or_intern("alpha")),
                Expr::Sym(interner.get_or_intern("a")),
                1,
                &interner,
            ),
            Expr::add(vec![Expr::one(), generator])
        );
    }

    #[test]
    fn squeezing_operator_series_order_two_contains_generator_squared_over_two_factorial() {
        let interner = ax_ir::Interner::new();
        let zeta = Expr::Sym(interner.get_or_intern("zeta"));
        let mode = Expr::Sym(interner.get_or_intern("a"));
        let result = squeezing_operator_series(zeta.clone(), mode.clone(), 2, &interner);
        let Expr::Add(terms) = result else {
            panic!("expected additive squeezing expansion");
        };

        let creation = Expr::Call(interner.get_or_intern("creation"), vec![mode.clone()]);
        let annihilation = Expr::Call(interner.get_or_intern("annihilation"), vec![mode]);
        let generator = Expr::mul(vec![
            Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(2usize))),
            Expr::add(vec![
                Expr::mul(vec![zeta.clone(), creation.clone(), creation]),
                Expr::neg(Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("conj"), vec![zeta]),
                    annihilation.clone(),
                    annihilation,
                ])),
            ]),
        ]);

        assert!(terms.contains(&Expr::mul(vec![
            Expr::Rational(BigRational::new(BigInt::one(), BigInt::from(2usize))),
            Expr::pow(generator, Expr::Int(BigInt::from(2usize))),
        ])));
    }

    #[test]
    fn simplify_ccr_car_bosonic_same_mode_annihilation_creation() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let mut properties = prop_map();
        properties.insert(
            a,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Bosonic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
        ]);
        let result = simplify_ccr_car_full(&expr, &properties, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            ]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn simplify_ccr_car_fermionic_same_mode_annihilation_creation() {
        let interner = ax_ir::Interner::new();
        let c = interner.get_or_intern("c");
        let mut properties = prop_map();
        properties.insert(
            c,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Fermionic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
        ]);
        let result = simplify_ccr_car_full(&expr, &properties, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            ])),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn simplify_ccr_car_distinct_bosons_reorder_without_sign() {
        let interner = ax_ir::Interner::new();
        let a0 = interner.get_or_intern("a0");
        let a1 = interner.get_or_intern("a1");
        let mut properties = prop_map();
        properties.insert(
            a0,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Bosonic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        properties.insert(
            a1,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Bosonic,
                subsystem: None,
                mode_index: 1,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a1)]),
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a0)]),
        ]);
        let result = simplify_ccr_car_full(&expr, &properties, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a0)]),
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a1)]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn simplify_ccr_car_distinct_fermions_reorder_with_minus_sign() {
        let interner = ax_ir::Interner::new();
        let c0 = interner.get_or_intern("c0");
        let c1 = interner.get_or_intern("c1");
        let mut properties = prop_map();
        properties.insert(
            c0,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Fermionic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        properties.insert(
            c1,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Fermionic,
                subsystem: None,
                mode_index: 1,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c1)]),
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c0)]),
        ]);
        let result = simplify_ccr_car_full(&expr, &properties, &interner);
        let expected = simplify_expr(Expr::neg(Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c0)]),
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c1)]),
        ])));
        assert_eq!(result, expected);
    }

    #[test]
    fn simplify_ccr_car_mixed_boson_fermion_product_is_unchanged() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let c = interner.get_or_intern("c");
        let mut properties = prop_map();
        properties.insert(
            a,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Bosonic,
                subsystem: None,
                mode_index: 0,
                label: None,
            })],
        );
        properties.insert(
            c,
            vec![TensorProperty::ModeMeta(ModeMetadata {
                statistics: ModeStatistics::Fermionic,
                subsystem: None,
                mode_index: 1,
                label: None,
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
        ]);
        let result = simplify_ccr_car_full(&expr, &properties, &interner);
        assert_eq!(result, expr);
    }

    #[test]
    fn spinor_bilinear_selection_rules_weyl_projector_mismatch_vanishes() {
        let interner = ax_ir::Interner::new();
        let (mut props, bar, _) = spinor_selection_rule_props(&interner);
        let spin = interner.get_or_intern("spin");
        let xi_l = interner.get_or_intern("xi_l");
        let chi_r = interner.get_or_intern("chi_r");
        props.insert(
            xi_l,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            chi_r,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Right),
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(xi_l)]),
            projector_left(&interner),
            Expr::Sym(chi_r),
        ]);

        assert_eq!(
            simplify_spinor_bilinear_selection_rules(&expr, &props, &interner),
            Expr::zero()
        );
    }

    #[test]
    fn spinor_bilinear_selection_rules_majorana_self_vector_vanishes() {
        let interner = ax_ir::Interner::new();
        let (mut props, bar, gamma) = spinor_selection_rule_props(&interner);
        let spin = interner.get_or_intern("spin");
        let psi = interner.get_or_intern("psi");
        let mu = interner.get_or_intern("mu");
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi),
        ]);

        assert_eq!(
            simplify_spinor_bilinear_selection_rules(&expr, &props, &interner),
            Expr::zero()
        );
    }

    #[test]
    fn spinor_bilinear_selection_rules_insufficient_metadata_leaves_unchanged() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let mu = interner.get_or_intern("mu");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi),
        ]);

        assert_eq!(
            simplify_spinor_bilinear_selection_rules(&expr, &props, &interner),
            expr
        );
    }

    #[test]
    fn spinor_index_bridge_simple_bilinear_round_trips() {
        let interner = ax_ir::Interner::new();
        let (props, bar, _, psi) = spinor_bridge_props(&interner);
        let expr = Expr::mul(vec![Expr::Call(bar, vec![Expr::Sym(psi)]), Expr::Sym(psi)]);

        let explicit = insert_explicit_spinor_indices(&expr, &props, &interner);
        assert_ne!(explicit, expr, "bridge should insert explicit spinor slots");
        assert_eq!(
            remove_trivial_spinor_indices(&explicit, &props, &interner),
            expr
        );
    }

    #[test]
    fn spinor_index_bridge_gamma_chain_round_trips() {
        let interner = ax_ir::Interner::new();
        let (props, bar, gamma, psi) = spinor_bridge_props(&interner);
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Call(gamma, vec![Expr::Sym(nu)]),
            Expr::Sym(psi),
        ]);

        let explicit = insert_explicit_spinor_indices(&expr, &props, &interner);
        assert_ne!(
            explicit, expr,
            "bridge should expose gamma-chain spinor slots"
        );
        assert_eq!(
            remove_trivial_spinor_indices(&explicit, &props, &interner),
            expr
        );
    }

    #[test]
    fn spinor_index_bridge_unsupported_noncanonical_is_unchanged() {
        let interner = ax_ir::Interner::new();
        let (props, bar, gamma, psi) = spinor_bridge_props(&interner);
        let mu = interner.get_or_intern("mu");
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Sym(psi),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
        ]);

        assert_eq!(
            insert_explicit_spinor_indices(&expr, &props, &interner),
            expr
        );
        assert_eq!(
            remove_trivial_spinor_indices(&expr, &props, &interner),
            expr
        );
    }

    #[test]
    fn join_two_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // γ^a γ^b = γ^{ab} + g^{ab}
        let result = join_gamma_pair(&[a], &[b], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                2,
                "expected γ^{{ab}} + g^{{ab}}, got {terms:?}"
            );
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_three_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // γ^a γ^{bc} = γ^{abc} + g^{ab} γ^c - g^{ac} γ^b
        let result = join_gamma_pair(&[a], &[b, c], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 3, "expected 3 terms, got {terms:?}");
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_empty_left_is_identity() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");

        // Identity * γ^a = γ^a
        let result = join_gamma_pair(&[], &[a], gamma, g, &interner);
        assert_eq!(result, Expr::Call(gamma, vec![Expr::Sym(a)]));
    }

    #[test]
    fn join_gammas_in_product() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // join_gammas_in_expr(gamma(a) * gamma(b)) → Add(...)
        let expr = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
        ]);
        let result = join_gammas_in_expr(&expr, gamma, g, &interner);
        assert!(
            matches!(result, Expr::Add(_)),
            "expected Add, got {result:?}"
        );
    }

    #[test]
    fn expand_bar_single_gamma() {
        // bar(gamma(a) psi) = bar(psi) gamma(a)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let inner = Expr::mul(vec![Expr::Call(gamma, vec![Expr::Sym(a)]), Expr::Sym(psi)]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let result_str = ax_ir::pretty_print(&result, &interner);
        assert!(
            result_str.contains("bar") && result_str.contains("gamma"),
            "should contain bar(psi) and gamma, got {}",
            result_str
        );
    }

    #[test]
    fn expand_bar_double_gamma_reverses() {
        // bar(gamma(a) gamma(b) psi) = -bar(psi) gamma(b) gamma(a)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let inner = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Sym(psi),
        ]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let result_str = format!("{:?}", result);
        assert!(
            result_str.contains("Neg") || result_str.contains("-1"),
            "double gamma reversal should introduce a sign, got {}",
            result_str
        );
    }

    #[test]
    fn expand_bar_multi_index_gamma_chain_reverses_with_total_rank_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let inner = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Sym(psi),
        ]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let expected = Expr::neg(Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]));
        assert_eq!(
            result, expected,
            "rank-3 gamma chain should reverse and pick a minus sign"
        );
    }

    #[test]
    fn expand_bar_nested_negative_chain_keeps_transpose_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let inner = Expr::neg(Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Sym(psi),
        ]));
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]);
        assert_eq!(
            result, expected,
            "explicit minus and two-gamma transpose minus should cancel"
        );
    }

    #[test]
    fn expand_bar_no_gamma() {
        // bar(psi) should stay as bar(psi)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let expr = Expr::Call(bar, vec![Expr::Sym(psi)]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        assert_eq!(result, expr, "bar(psi) with no gammas should be unchanged");
    }

    #[test]
    fn fierz_coefficients_4d() {
        let coeffs = fierz_coefficients(4);
        // ranks 0, 1, 2, 3, 4 in 4D → 5 entries
        assert_eq!(coeffs.len(), 5);
        // Verify ranks are 0..=4
        for (i, (_, rank)) in coeffs.iter().enumerate() {
            assert_eq!(*rank, i);
        }
        // spinor_dim = 4; overall minus; check signs
        // k=0: sign=(0%2==0)→+1, binom=1, coeff_raw=1/4, k!=1 → raw=1/4, after minus: -1/4
        // k=1: sign=(1%2==1)→-1, binom=4, coeff_raw=-1/1, k!=1 → raw=-1/1, after minus: 1/1
        // k=2: sign=(3%2==1)→-1, binom=6, coeff_raw=-3/2, k!=2 → raw=-3/4, after minus: 3/4
        // k=3: sign=(6%2==0)→+1, binom=4, coeff_raw=1/1, k!=6 → raw=1/6, after minus: -1/6
        // k=4: sign=(10%2==0)→+1, binom=1, coeff_raw=1/4, k!=24 → raw=1/96, after minus: -1/96
        let expected: Vec<num_rational::BigRational> = vec![
            num_rational::BigRational::new((-1i64).into(), 4i64.into()),
            num_rational::BigRational::new(1i64.into(), 1i64.into()),
            num_rational::BigRational::new(3i64.into(), 4i64.into()),
            num_rational::BigRational::new((-1i64).into(), 6i64.into()),
            num_rational::BigRational::new((-1i64).into(), 96i64.into()),
        ];
        for (i, (c, _)) in coeffs.iter().enumerate() {
            assert_eq!(c, &expected[i], "mismatch at rank {i}");
        }
    }

    #[test]
    fn fierz_coefficients_sum_check() {
        // In d=4, the 16 gamma matrix basis elements are counted by C(4,k):
        // C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1 = 16 = spinor_dim^2
        let dim = 4;
        let coeffs = fierz_coefficients(dim);
        assert!(!coeffs.is_empty());
        assert_eq!(coeffs.len(), dim + 1);
        // Completeness: sum of |c_k| * C(d,k) * k! * spinor_dim should equal total basis size
        // As a basic sanity check, verify no coefficient is zero
        for (c, _) in &coeffs {
            assert_ne!(*c, num_rational::BigRational::new(0i64.into(), 1i64.into()));
        }
    }

    #[test]
    fn fierz_4d_unit_unit() {
        // (psibar1 psi2)(psibar3 psi4) Fierz rearranged in 4D.
        let interner = ax_ir::Interner::new();
        let coeffs = fierz_coefficients(4);
        let total_basis: usize = coeffs.iter().map(|(_, k)| binomial(4, *k) as usize).sum();
        assert_eq!(total_basis, 16, "total gamma basis size in 4D should be 16");

        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");
        let expr = Expr::mul(vec![
            Expr::Sym(psibar1),
            Expr::Sym(psi2),
            Expr::Sym(psibar3),
            Expr::Sym(psi4),
        ]);
        let result = fierz(&expr, 4, [psibar1, psi4, psibar3, psi2], &interner);
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), coeffs.len()),
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    fn collect_rationals(expr: &Expr, out: &mut Vec<num_rational::BigRational>) {
        match expr {
            Expr::Rational(value) => out.push(value.clone()),
            Expr::Mul(factors) | Expr::Add(factors) => {
                for factor in factors {
                    collect_rationals(factor, out);
                }
            }
            Expr::Neg(inner) => {
                let mut nested = Vec::new();
                collect_rationals(inner, &mut nested);
                out.extend(nested.into_iter().map(|value| -value));
            }
            _ => {}
        }
    }

    #[test]
    fn fierz_detects_nontrivial_gamma_chains() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let expr = Expr::mul(vec![
            Expr::Sym(psibar1),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Sym(psi2),
            Expr::Sym(psibar3),
            Expr::Call(gamma, vec![Expr::Sym(d)]),
            Expr::Sym(psi4),
        ]);
        let pair = find_bilinears(&expr, &interner).expect("gamma-chain bilinears should parse");
        assert_eq!(pair.gamma_a, vec![a, b, c]);
        assert_eq!(pair.gamma_b, vec![d]);

        let result = fierz_auto(&expr, 4, &interner);
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), fierz_coefficients(4).len()),
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    #[test]
    fn fierz_auto_infers_standard_spinor_order_in_nested_product() {
        let interner = ax_ir::Interner::new();
        let scalar = interner.get_or_intern("m");
        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");

        let expr = Expr::mul(vec![
            Expr::Sym(scalar),
            Expr::mul(vec![Expr::Sym(psibar1), Expr::Sym(psi2)]),
            Expr::mul(vec![Expr::Sym(psibar3), Expr::Sym(psi4)]),
        ]);
        let result =
            try_fierz_auto(&expr, 4, &interner).expect("standard product should infer order");
        match result {
            Expr::Add(terms) => {
                assert_eq!(terms.len(), fierz_coefficients(4).len());
                assert!(
                    matches!(&terms[0], Expr::Mul(factors) if factors.contains(&Expr::Sym(scalar))),
                    "remaining scalar should be preserved"
                );
            }
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    #[test]
    fn fierz_ambiguous_three_bilinears_fails_clearly() {
        let interner = ax_ir::Interner::new();
        let s = ["psibar1", "psi2", "psibar3", "psi4", "psibar5", "psi6"]
            .iter()
            .map(|name| interner.get_or_intern(name))
            .collect::<Vec<_>>();
        let expr = Expr::mul(s.iter().map(|sym| Expr::Sym(*sym)).collect());
        let error = try_fierz_auto(&expr, 4, &interner).expect_err("three bilinears are ambiguous");
        assert_eq!(error, FierzError::AmbiguousBilinears(3));

        let wrapped = fierz_auto(&expr, 4, &interner);
        assert!(
            matches!(wrapped, Expr::Call(sym, _) if interner.resolve(sym) == "fierz_ambiguous_bilinears")
        );
    }

    #[test]
    fn fierz_malformed_bar_fails_clearly() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let psi = interner.get_or_intern("psi");
        let expr = Expr::mul(vec![Expr::Call(bar, vec![Expr::Sym(psi)])]);
        let error = try_fierz_auto(&expr, 4, &interner).expect_err("single bar is malformed");
        assert_eq!(error, FierzError::MalformedBilinear);
    }

    #[test]
    fn fierz_anticommuting_spinors_flip_rearrangement_sign() {
        let interner = ax_ir::Interner::new();
        let s1 = interner.get_or_intern("s1bar");
        let s2 = interner.get_or_intern("s2");
        let s3 = interner.get_or_intern("s3bar");
        let s4 = interner.get_or_intern("s4");
        let expr = Expr::mul(vec![
            Expr::Sym(s1),
            Expr::Sym(s2),
            Expr::Sym(s3),
            Expr::Sym(s4),
        ]);

        let plain = try_fierz_auto(&expr, 4, &interner).expect("plain spinors should rearrange");
        let mut props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
        for sym in [s1, s2, s3, s4] {
            props.insert(sym, vec![TensorProperty::AntiCommuting]);
        }
        let graded =
            try_fierz_auto_with_properties(&expr, 4, &props, &interner).expect("graded spinors");

        let mut plain_coeffs = Vec::new();
        collect_rationals(&plain, &mut plain_coeffs);
        let mut graded_coeffs = Vec::new();
        collect_rationals(&graded, &mut graded_coeffs);
        let mut negated_plain = plain_coeffs
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        graded_coeffs.sort_by_key(|value| format!("{value:?}"));
        negated_plain.sort_by_key(|value| format!("{value:?}"));
        assert_eq!(
            graded_coeffs, negated_plain,
            "moving the fourth anticommuting spinor through the third should flip every Fierz coefficient"
        );
    }

    #[test]
    fn fierz_basis_4d_dirac_metadata_yields_full_five_sector_basis() {
        let interner = ax_ir::Interner::new();
        let eps = interner.get_or_intern("eps");
        let spinor_meta = SpinorMetadata {
            class: SpinorClass::Dirac,
            dimension: Some(4),
            chirality: None,
            index_family: None,
        };
        let gamma_meta = GammaConventionMetadata {
            signature: ax_ir::MetricSignature::MostlyPlus,
            clifford: CliffordConvention::PlusTwoG,
            gamma5: Some(ax_ir::GammaFiveConvention::AbstractChiral),
            epsilon_symbol: Some(eps),
            dimension: Some(4),
        };

        let basis = fierz_basis_for_metadata(&spinor_meta, Some(&gamma_meta), &interner);

        assert_eq!(basis.len(), 5);
        assert!(basis.contains(&Expr::one()));
        assert!(
            basis
                .iter()
                .any(|sector| matches!(sector, Expr::Call(sym, args) if interner.resolve(*sym) == "gamma" && args.len() == 1))
        );
        assert!(
            basis
                .iter()
                .any(|sector| matches!(sector, Expr::Call(sym, args) if interner.resolve(*sym) == "sigma" && args.len() == 2))
        );
        assert!(basis.iter().any(|sector| {
            matches!(sector, Expr::Mul(factors) if factors.iter().any(|factor| matches!(factor, Expr::Call(sym, args) if interner.resolve(*sym) == "gamma" && args.len() == 1))
                && factors.iter().any(|factor| matches!(factor, Expr::Sym(sym) if interner.resolve(*sym) == "gamma5")))
        }));
        assert!(basis
            .iter()
            .any(|sector| matches!(sector, Expr::Sym(sym) if interner.resolve(*sym) == "gamma5")));
    }

    #[test]
    fn fierz_basis_chiral_metadata_yields_reduced_basis() {
        let interner = ax_ir::Interner::new();
        let eps = interner.get_or_intern("eps");
        let spinor_meta = SpinorMetadata {
            class: SpinorClass::Weyl,
            dimension: Some(4),
            chirality: Some(ax_ir::Chirality::Left),
            index_family: None,
        };
        let gamma_meta = GammaConventionMetadata {
            signature: ax_ir::MetricSignature::MostlyPlus,
            clifford: CliffordConvention::PlusTwoG,
            gamma5: Some(ax_ir::GammaFiveConvention::AbstractChiral),
            epsilon_symbol: Some(eps),
            dimension: Some(4),
        };

        let basis = fierz_basis_for_metadata(&spinor_meta, Some(&gamma_meta), &interner);

        assert_eq!(basis.len(), 2);
        assert!(!basis.contains(&Expr::one()));
        assert!(!basis
            .iter()
            .any(|sector| matches!(sector, Expr::Sym(sym) if interner.resolve(*sym) == "gamma5")));
    }

    #[test]
    fn fierz_coefficients_missing_convention_data_errors() {
        let interner = ax_ir::Interner::new();
        let spinor_meta = SpinorMetadata {
            class: SpinorClass::Dirac,
            dimension: Some(4),
            chirality: None,
            index_family: None,
        };

        assert_eq!(
            fierz_coefficients_for_basis(&[Expr::one()], &spinor_meta, None, &interner),
            Err(FierzError::InsufficientConventionData)
        );
    }

    #[test]
    fn fierz_structured_metadata_compatible_still_succeeds() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let eps = interner.get_or_intern("eps");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(
            gamma,
            vec![
                TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                    dimension: Some(4),
                    metric_symbol: None,
                    index_family: Some(spin),
                    has_gamma5: true,
                }),
                TensorProperty::GammaConventionMeta(GammaConventionMetadata {
                    signature: ax_ir::MetricSignature::MostlyPlus,
                    clifford: CliffordConvention::PlusTwoG,
                    gamma5: Some(ax_ir::GammaFiveConvention::AbstractChiral),
                    epsilon_symbol: Some(eps),
                    dimension: Some(4),
                }),
            ],
        );
        for sym in [psi1, psi2, psi3, psi4] {
            props.insert(
                sym,
                vec![
                    TensorProperty::SpinorMeta(SpinorMetadata {
                        class: SpinorClass::Majorana,
                        dimension: Some(4),
                        chirality: None,
                        index_family: Some(spin),
                    }),
                    TensorProperty::AntiCommuting,
                ],
            );
        }
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let result = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect("compatible structured metadata should allow Fierz");
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn fierz_structured_metadata_missing_convention_fails() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        for sym in [psi1, psi2, psi3, psi4] {
            props.insert(
                sym,
                vec![TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Dirac,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                })],
            );
        }
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);

        let error = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect_err("structured Fierz requires declared gamma convention");
        assert_eq!(error, FierzError::InsufficientConventionData);
    }

    #[test]
    fn fierz_structured_metadata_dimension_mismatch_fails() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi1,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi2,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(2),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi3,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi4,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let error = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect_err("dimension mismatch should be rejected");
        assert_eq!(error, FierzError::IncompatibleSpinorDimension);
    }

    #[test]
    fn fierz_structured_metadata_chirality_mismatch_fails() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi1,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi2,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Right),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi3,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi4,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let error = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect_err("chirality mismatch should be rejected");
        assert_eq!(error, FierzError::IncompatibleSpinorChirality);
    }

    // ── split_gamma tests ─────────────────────────────────────────────────────

    #[test]
    fn split_gamma_three_indices() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // gamma(a, b, c) split from back → gamma(a,b)*gamma(c) + contractions
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert!(
                terms.len() >= 2,
                "expected main term + contraction terms, got {:?}",
                result
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_back_vs_front_differ() {
        // Splitting from back vs front should produce different expressions
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let back = split_gamma(&expr, gamma, g, true, &interner);
        let front = split_gamma(&expr, gamma, g, false, &interner);

        assert_ne!(back, front, "splitting from back vs front should differ");
    }

    #[test]
    fn split_gamma_two_indices_back() {
        // gamma(a, b) split from back → gamma(a)*gamma(b) − g^{ba} * 1
        // (2-index: remaining = [a], split = b, k=0 → sign = +, negate=false → subtract metric)
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                2,
                "gamma(a,b) split should give 2 terms: main + one contraction"
            );
            // First term should be a Mul (gamma(a) * gamma(b))
            assert!(
                matches!(&terms[0], Expr::Mul(_)),
                "first term should be a product"
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_two_indices_front() {
        // gamma(a, b) split from front → gamma(a)*gamma(b) − g^{ab} * 1
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, false, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "gamma(a,b) split-front should give 2 terms");
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_single_index_unchanged() {
        // gamma(a) has only one index — cannot be split, returned as-is
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);
        assert_eq!(result, expr, "single-index gamma should be unchanged");
    }

    #[test]
    fn split_gamma_four_indices_term_count() {
        // gamma(a,b,c,d) split from back has remaining=[a,b,c] → 3 contractions + 1 main = 4 terms
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let expr = Expr::Call(
            gamma,
            vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c), Expr::Sym(d)],
        );
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                4,
                "4-index gamma split should give 4 terms (1 main + 3 contractions)"
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_non_gamma_call_unchanged() {
        // A call to a different function should not be touched
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let f_sym = interner.get_or_intern("f");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(f_sym, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);
        assert_eq!(result, expr, "non-gamma call should be unchanged");
    }

    #[test]
    fn split_gamma_distributes_in_sum() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // gamma(a,b,c) + gamma(a,b) → both are processed
        let g3 = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let g2 = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let expr = Expr::add(vec![g3, g2]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        // Result should still be an Add (may have more terms after expansion)
        assert!(
            matches!(result, Expr::Add(_)),
            "result of split on a sum should be an Add, got {:?}",
            result
        );
    }

    #[test]
    fn sort_spinors_majorana_flip_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi,
            vec![
                TensorProperty::Spinor,
                TensorProperty::MajoranaSpinor,
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::Spinor,
                TensorProperty::MajoranaSpinor,
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(chi)]),
            Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::Sym(psi),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn sort_spinors_structured_metadata_family_match_succeeds() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );

        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(chi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(psi),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn sort_spinors_structured_metadata_family_mismatch_returns_diagnostic() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let spin = interner.get_or_intern("spin");
        let other_spin = interner.get_or_intern("other_spin");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(other_spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );

        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "sort_spinors_spinor_family_mismatch")
        );
    }

    #[test]
    fn join_gamma_rank1_rank1_4d() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma, vec![Expr::Sym(b)]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), 2),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn join_gamma_rank2_rank1_4d() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c)]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert!(terms.len() >= 3),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn join_gamma_duplicate_index_zero() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(a)]),
            &Expr::Call(gamma, vec![]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn join_gamma_generalised_delta() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c), Expr::Sym(d)]),
            Some(4),
            true,
            true,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        let printed = ax_ir::pretty_print(&result, &interner);
        assert!(printed.contains("generalised_delta"));
    }

    #[test]
    fn join_gamma_family_mismatch_returns_diagnostic() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let spin = interner.get_or_intern("spin");
        let flavor = interner.get_or_intern("flavor");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        let gamma_flavor = interner.get_or_intern("gamma_flavor");
        props.insert(
            gamma_flavor,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(flavor),
                has_gamma5: false,
            })],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma_flavor, vec![Expr::Sym(b)]),
            None,
            true,
            false,
            &Expr::Sym(interner.get_or_intern("eta")),
            &props,
            &mut interner,
        );
        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_family_mismatch")
        );
    }

    #[test]
    fn join_gamma_dimension_comes_from_metadata_before_fallback() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c)]),
            Some(2),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert!(
                terms.len() >= 3,
                "metadata dimension=4 should win over fallback dimension=2"
            ),
            other => panic!("expected add, got {other:?}"),
        }
    }

    fn gamma_convention_meta(
        signature: ax_ir::MetricSignature,
        clifford: CliffordConvention,
        dimension: Option<usize>,
    ) -> TensorProperty {
        TensorProperty::GammaConventionMeta(GammaConventionMetadata {
            signature,
            clifford,
            gamma5: None,
            epsilon_symbol: None,
            dimension,
        })
    }

    fn upper_index(name: lasso::Spur) -> Index {
        Index {
            name,
            variance: Variance::Up,
            index_type: None,
        }
    }

    #[test]
    fn join_gamma_plus_two_g_respects_positive_metric_term() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![gamma_convention_meta(
                ax_ir::MetricSignature::MostlyPlus,
                CliffordConvention::PlusTwoG,
                Some(4),
            )],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma, vec![Expr::Sym(b)]),
            None,
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        let expected_metric =
            build_metric_contraction(&Expr::Sym(eta), &upper_index(a), &upper_index(b));

        let Expr::Add(terms) = result else {
            panic!("expected additive gamma product");
        };
        assert!(
            terms.contains(&expected_metric),
            "PlusTwoG should keep a positive metric contraction, got {terms:?}"
        );
    }

    #[test]
    fn join_gamma_minus_two_g_flips_metric_term_sign() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![gamma_convention_meta(
                ax_ir::MetricSignature::MostlyPlus,
                CliffordConvention::MinusTwoG,
                Some(4),
            )],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma, vec![Expr::Sym(b)]),
            None,
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        let expected_metric = Expr::neg(build_metric_contraction(
            &Expr::Sym(eta),
            &upper_index(a),
            &upper_index(b),
        ));

        let Expr::Add(terms) = result else {
            panic!("expected additive gamma product");
        };
        assert!(
            terms.contains(&expected_metric),
            "MinusTwoG should flip the metric contraction sign, got {terms:?}"
        );
    }

    #[test]
    fn join_gamma_convention_mismatch_returns_diagnostic() {
        let mut interner = ax_ir::Interner::new();
        let gamma_a = interner.get_or_intern("gamma_a");
        let gamma_b = interner.get_or_intern("gamma_b");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            gamma_a,
            vec![gamma_convention_meta(
                ax_ir::MetricSignature::MostlyPlus,
                CliffordConvention::PlusTwoG,
                Some(4),
            )],
        );
        props.insert(
            gamma_b,
            vec![gamma_convention_meta(
                ax_ir::MetricSignature::MostlyMinus,
                CliffordConvention::PlusTwoG,
                Some(4),
            )],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma_a, vec![Expr::Sym(a)]),
            &Expr::Call(gamma_b, vec![Expr::Sym(b)]),
            None,
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );

        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_convention_mismatch")
        );
    }

    #[test]
    fn split_gamma_convention_mismatch_returns_diagnostic() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![
                gamma_convention_meta(
                    ax_ir::MetricSignature::MostlyPlus,
                    CliffordConvention::PlusTwoG,
                    Some(4),
                ),
                gamma_convention_meta(
                    ax_ir::MetricSignature::MostlyPlus,
                    CliffordConvention::MinusTwoG,
                    Some(4),
                ),
            ],
        );

        let result = split_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            true,
            &props,
            &mut interner,
        );

        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_convention_mismatch")
        );
    }

    #[test]
    fn expand_diracbar_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let eps = interner.get_or_intern("eps");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let odd = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a)]),
                Expr::Sym(eps),
            ])],
        );
        let even = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]),
                Expr::Sym(psi),
            ])],
        );
        let odd_result = expand_diracbar_full(&odd, &props, &interner);
        let even_result = expand_diracbar_full(&even, &props, &interner);
        let odd_str = format!("{odd_result:?}");
        assert!(matches!(odd_result, Expr::Neg(_)) || odd_str.contains("-1"));
        assert!(!matches!(even_result, Expr::Neg(_)));
    }

    #[test]
    fn expand_diracbar_full_structured_metadata_respects_reverse_gamma_order() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );

        let expr = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a)]),
                Expr::Call(gamma, vec![Expr::Sym(b)]),
                Expr::Sym(psi),
            ])],
        );
        let result = expand_diracbar_full(&expr, &props, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(
                interner.get_or_intern("bar_action"),
                vec![Expr::Call(gamma, vec![Expr::Sym(b)])],
            ),
            Expr::Call(
                interner.get_or_intern("bar_action"),
                vec![Expr::Call(gamma, vec![Expr::Sym(a)])],
            ),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn expand_diracbar_full_distributes_over_sums() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: None,
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        for spinor in [psi, chi] {
            props.insert(
                spinor,
                vec![TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Dirac,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                })],
            );
        }

        let expr = Expr::Call(bar, vec![Expr::add(vec![Expr::Sym(psi), Expr::Sym(chi)])]);
        let expected = Expr::add(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(bar, vec![Expr::Sym(chi)]),
        ]);

        assert_eq!(expand_diracbar_full(&expr, &props, &interner), expected);
    }

    #[test]
    fn expand_diracbar_full_conjugates_scalar_coefficients() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let psi = interner.get_or_intern("psi");
        let c = interner.get_or_intern("c");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: None,
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Dirac,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );

        let expr = Expr::Call(bar, vec![Expr::mul(vec![Expr::Sym(c), Expr::Sym(psi)])]);
        let expected = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("conj"), vec![Expr::Sym(c)]),
            Expr::Call(bar, vec![Expr::Sym(psi)]),
        ]);

        assert_eq!(expand_diracbar_full(&expr, &props, &interner), expected);
    }

    #[test]
    fn expand_diracbar_full_gamma_products_reverse_with_metadata() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Dirac,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );

        let expr = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a)]),
                Expr::Call(gamma, vec![Expr::Sym(b)]),
                Expr::Sym(psi),
            ])],
        );
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(
                interner.get_or_intern("bar_action"),
                vec![Expr::Call(gamma, vec![Expr::Sym(b)])],
            ),
            Expr::Call(
                interner.get_or_intern("bar_action"),
                vec![Expr::Call(gamma, vec![Expr::Sym(a)])],
            ),
        ]);

        assert_eq!(expand_diracbar_full(&expr, &props, &interner), expected);
    }

    #[test]
    fn expand_diracbar_full_nested_bars_become_canonical_adjoint_form() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let psi = interner.get_or_intern("psi");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: None,
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Dirac,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );

        let expr = Expr::Call(bar, vec![Expr::Call(bar, vec![Expr::Sym(psi)])]);
        let expected = Expr::Call(
            interner.get_or_intern("adjoint"),
            vec![Expr::Call(bar, vec![Expr::Sym(psi)])],
        );

        assert_eq!(expand_diracbar_full(&expr, &props, &interner), expected);
    }

    #[test]
    fn chiral_projector_left_is_idempotent() {
        let interner = ax_ir::Interner::new();
        let props = prop_map();
        let pl = projector_left(&interner);

        assert_eq!(
            simplify_chiral_projectors(
                &Expr::pow(pl.clone(), Expr::Int(2.into())),
                &props,
                &interner
            ),
            pl
        );
    }

    #[test]
    fn chiral_projector_right_is_idempotent() {
        let interner = ax_ir::Interner::new();
        let props = prop_map();
        let pr = projector_right(&interner);

        assert_eq!(
            simplify_chiral_projectors(
                &Expr::pow(pr.clone(), Expr::Int(2.into())),
                &props,
                &interner
            ),
            pr
        );
    }

    #[test]
    fn chiral_projector_left_right_product_is_zero() {
        let interner = ax_ir::Interner::new();
        let props = prop_map();
        let expr = Expr::mul(vec![projector_left(&interner), projector_right(&interner)]);

        assert_eq!(
            simplify_chiral_projectors(&expr, &props, &interner),
            Expr::zero()
        );
    }

    #[test]
    fn chiral_projector_completeness_sum_is_one() {
        let interner = ax_ir::Interner::new();
        let props = prop_map();
        let expr = Expr::add(vec![projector_left(&interner), projector_right(&interner)]);

        assert_eq!(
            simplify_chiral_projectors(&expr, &props, &interner),
            Expr::one()
        );
    }

    #[test]
    fn chiral_projectors_act_on_weyl_spinors_by_metadata() {
        let interner = ax_ir::Interner::new();
        let psi_l = interner.get_or_intern("psi_l");
        let psi_r = interner.get_or_intern("psi_r");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            psi_l,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi_r,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Right),
                index_family: Some(spin),
            })],
        );

        assert_eq!(
            simplify_chiral_projectors(
                &Expr::mul(vec![projector_left(&interner), Expr::Sym(psi_l)]),
                &props,
                &interner,
            ),
            Expr::Sym(psi_l)
        );
        assert_eq!(
            simplify_chiral_projectors(
                &Expr::mul(vec![projector_right(&interner), Expr::Sym(psi_r)]),
                &props,
                &interner,
            ),
            Expr::Sym(psi_r)
        );
        assert_eq!(
            simplify_chiral_projectors(
                &Expr::mul(vec![projector_left(&interner), Expr::Sym(psi_r)]),
                &props,
                &interner,
            ),
            Expr::zero()
        );
        assert_eq!(
            simplify_chiral_projectors(
                &Expr::mul(vec![projector_right(&interner), Expr::Sym(psi_l)]),
                &props,
                &interner,
            ),
            Expr::zero()
        );
    }

    #[test]
    fn sigma_matrix_expands_to_gamma_commutator() {
        let interner = ax_ir::Interner::new();
        let mu = Expr::Sym(interner.get_or_intern("mu"));
        let nu = Expr::Sym(interner.get_or_intern("nu"));

        let result =
            sigma_to_gamma_commutator(&sigma_matrix(mu.clone(), nu.clone(), &interner), &interner);
        let expected = Expr::mul(vec![
            i_over_two(),
            Expr::add(vec![
                Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("gamma"), vec![mu.clone()]),
                    Expr::Call(interner.get_or_intern("gamma"), vec![nu.clone()]),
                ]),
                Expr::neg(Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("gamma"), vec![nu]),
                    Expr::Call(interner.get_or_intern("gamma"), vec![mu]),
                ])),
            ]),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn gamma_commutator_pattern_converts_to_sigma_with_correct_factor() {
        let interner = ax_ir::Interner::new();
        let mu = Expr::Sym(interner.get_or_intern("mu"));
        let nu = Expr::Sym(interner.get_or_intern("nu"));
        let gamma_mu = Expr::Call(interner.get_or_intern("gamma"), vec![mu.clone()]);
        let gamma_nu = Expr::Call(interner.get_or_intern("gamma"), vec![nu.clone()]);
        let commutator_pattern = Expr::add(vec![
            Expr::mul(vec![gamma_mu.clone(), gamma_nu.clone()]),
            Expr::neg(Expr::mul(vec![gamma_nu, gamma_mu])),
        ]);

        let result = gamma_commutator_to_sigma(&commutator_pattern, &interner);
        let expected = Expr::mul(vec![minus_two_i(), sigma_matrix(mu, nu, &interner)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn gamma_commutator_call_converts_to_sigma_with_correct_factor() {
        let interner = ax_ir::Interner::new();
        let mu = Expr::Sym(interner.get_or_intern("mu"));
        let nu = Expr::Sym(interner.get_or_intern("nu"));
        let commutator = Expr::Call(
            interner.get_or_intern("commutator"),
            vec![
                Expr::Call(interner.get_or_intern("gamma"), vec![mu.clone()]),
                Expr::Call(interner.get_or_intern("gamma"), vec![nu.clone()]),
            ],
        );

        let result = gamma_commutator_to_sigma(&commutator, &interner);
        let expected = Expr::mul(vec![minus_two_i(), sigma_matrix(mu, nu, &interner)]);

        assert_eq!(result, expected);
    }

    #[test]
    fn gamma_to_sigma_leaves_nonmatching_input_unchanged() {
        let interner = ax_ir::Interner::new();
        let mu = Expr::Sym(interner.get_or_intern("mu"));
        let nu = Expr::Sym(interner.get_or_intern("nu"));
        let nonmatching = Expr::add(vec![
            Expr::mul(vec![
                Expr::Call(interner.get_or_intern("gamma"), vec![mu]),
                Expr::Call(interner.get_or_intern("gamma"), vec![nu]),
            ]),
            Expr::one(),
        ]);

        assert_eq!(
            gamma_commutator_to_sigma(&nonmatching, &interner),
            nonmatching
        );
    }

    #[test]
    fn split_gamma_back_full() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma_full(&expr, true, &props, &mut interner);
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn split_gamma_front_full() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma_full(&expr, false, &props, &mut interner);
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn fierz_full_reorders_wrong_spinor_order() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let bar = interner.get_or_intern("bar");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(bar, vec![TensorProperty::DiracBar]);
        for sym in [psi1, psi2, psi3, psi4] {
            props.insert(
                sym,
                vec![TensorProperty::Spinor, TensorProperty::AntiCommuting],
            );
        }
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi4),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi2),
        ]);
        let order = [
            Expr::Sym(psi1),
            Expr::Sym(psi2),
            Expr::Sym(psi3),
            Expr::Sym(psi4),
        ];
        let result = fierz_full(&expr, &order, 4, &props, &mut interner);
        assert!(matches!(result, Some(Expr::Add(_))));
    }

    #[test]
    fn first_order_energy_shift_diagonal_two_level_returns_diagonal_entries() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("a"));
        let b = Expr::Sym(interner.get_or_intern("b"));
        let h0 = vec![
            vec![Expr::Int(1.into()), Expr::zero()],
            vec![Expr::zero(), Expr::Int(2.into())],
        ];
        let v = vec![vec![a.clone(), Expr::zero()], vec![Expr::zero(), b.clone()]];

        assert_eq!(first_order_energy_shift(&h0, &v, 0, &interner).unwrap(), a);
        assert_eq!(first_order_energy_shift(&h0, &v, 1, &interner).unwrap(), b);
    }

    #[test]
    fn second_order_energy_shift_diagonal_two_level_off_diagonal_coupling_is_exact() {
        let interner = ax_ir::Interner::new();
        let g = Expr::Sym(interner.get_or_intern("g"));
        let h0 = vec![
            vec![Expr::Int(1.into()), Expr::zero()],
            vec![Expr::zero(), Expr::Int(2.into())],
        ];
        let v = vec![vec![Expr::zero(), g.clone()], vec![g.clone(), Expr::zero()]];

        let shift = second_order_energy_shift(&h0, &v, 0, &interner).unwrap();
        let expected = Expr::neg(Expr::pow(g, Expr::Int(2.into())));
        assert_eq!(shift, expected);
    }

    #[test]
    fn perturbation_theory_rejects_degenerate_unperturbed_diagonal_spectrum() {
        let interner = ax_ir::Interner::new();
        let h0 = identity_matrix(2, &interner);
        let v = identity_matrix(2, &interner);

        assert_eq!(
            first_order_energy_shift(&h0, &v, 0, &interner),
            Err(PerturbationError::DegenerateUnperturbedSpectrum)
        );
    }

    #[test]
    fn degenerate_subspace_effective_perturbation_extracts_requested_block() {
        let interner = ax_ir::Interner::new();
        let a = Expr::Sym(interner.get_or_intern("a"));
        let b = Expr::Sym(interner.get_or_intern("b"));
        let c = Expr::Sym(interner.get_or_intern("c"));
        let d = Expr::Sym(interner.get_or_intern("d"));
        let h0 = vec![
            vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::Int(1.into()), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::Int(2.into())],
        ];
        let v = vec![
            vec![a.clone(), b.clone(), Expr::zero()],
            vec![c.clone(), d.clone(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::one()],
        ];

        assert_eq!(
            degenerate_subspace_effective_perturbation(&h0, &v, &[0, 1], &interner).unwrap(),
            vec![vec![a, b], vec![c, d]]
        );
    }

    #[test]
    fn degenerate_first_order_splittings_two_state_off_diagonal_are_pm_g() {
        let interner = ax_ir::Interner::new();
        let g = Expr::Sym(interner.get_or_intern("g"));
        let h0 = vec![
            vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::Int(1.into()), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::Int(2.into())],
        ];
        let v = vec![
            vec![Expr::zero(), g.clone(), Expr::zero()],
            vec![g.clone(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::one()],
        ];

        assert_eq!(
            degenerate_first_order_splittings(&h0, &v, &[0, 1], &interner).unwrap(),
            vec![g.clone(), Expr::neg(g)]
        );
    }

    #[test]
    fn degenerate_subspace_rejects_nondegenerate_selection() {
        let interner = ax_ir::Interner::new();
        let h0 = vec![
            vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::Int(1.into()), Expr::zero()],
            vec![Expr::zero(), Expr::zero(), Expr::Int(2.into())],
        ];
        let v = identity_matrix(3, &interner);

        assert_eq!(
            degenerate_subspace_effective_perturbation(&h0, &v, &[0, 2], &interner),
            Err(PerturbationError::SelectedSubspaceNotDegenerate)
        );
    }

    #[test]
    fn berry_connection_contains_i_bra_and_diff() {
        let interner = ax_ir::Interner::new();
        let psi = interner.get_or_intern("psi");
        let theta = interner.get_or_intern("theta");
        let psi_theta = Expr::Call(psi, vec![Expr::Sym(theta)]);
        let result = berry_connection(psi_theta, Expr::Sym(theta), &interner);

        let Expr::Call(form_sym, args) = result else {
            panic!("expected one_form_component call");
        };
        assert_eq!(interner.resolve(form_sym), "one_form_component");
        assert_eq!(args[0], Expr::Sym(theta));

        let rendered = ax_ir::pretty_print(&args[1], &interner);
        assert!(rendered.contains("i"), "got {rendered}");
        assert!(rendered.contains("bra(psi(theta))"), "got {rendered}");
        assert!(
            rendered.contains("diff(psi(theta), theta)"),
            "got {rendered}"
        );
    }

    #[test]
    fn geometric_phase_contains_canonical_integral_representation() {
        let interner = ax_ir::Interner::new();
        let theta = interner.get_or_intern("theta");
        let a = Expr::Sym(interner.get_or_intern("A"));
        let result = geometric_phase(a, Expr::Sym(theta), &interner);

        let Expr::Call(sym, args) = result else {
            panic!("expected integral call");
        };
        assert_eq!(interner.resolve(sym), "integral");
        assert_eq!(args.len(), 4);
        assert_eq!(args[0], Expr::Sym(theta));
        assert!(
            matches!(&args[1], Expr::Call(path_sym, inner) if interner.resolve(*path_sym) == "closed_path" && inner == &vec![Expr::Sym(theta)])
        );
        assert!(
            matches!(&args[2], Expr::Call(path_sym, inner) if interner.resolve(*path_sym) == "closed_path" && inner == &vec![Expr::Sym(theta)])
        );
    }
}
