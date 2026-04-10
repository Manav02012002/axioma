use crate::{
    collect_terms_expr, covariant_derivative_covector, eval_expr, simplify_expr, SymbolicMatrix,
};
use ax_ir::Expr;
use num_rational::BigRational;

#[derive(Debug, Clone, PartialEq)]
pub struct NullTetrad {
    /// Contravariant null vector l^a.
    pub l: Vec<ax_ir::Expr>,
    /// Contravariant null vector n^a.
    pub n: Vec<ax_ir::Expr>,
    /// Contravariant complex null vector m^a.
    pub m: Vec<ax_ir::Expr>,
    /// Contravariant conjugate null vector \bar m^a.
    pub m_bar: Vec<ax_ir::Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpinCoefficients {
    pub kappa: ax_ir::Expr,
    pub sigma: ax_ir::Expr,
    pub lambda: ax_ir::Expr,
    pub nu: ax_ir::Expr,
    pub rho: ax_ir::Expr,
    pub mu: ax_ir::Expr,
    pub tau: ax_ir::Expr,
    pub pi: ax_ir::Expr,
    pub epsilon: ax_ir::Expr,
    pub gamma: ax_ir::Expr,
    pub alpha: ax_ir::Expr,
    pub beta: ax_ir::Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeylScalars {
    pub psi0: ax_ir::Expr,
    pub psi1: ax_ir::Expr,
    pub psi2: ax_ir::Expr,
    pub psi3: ax_ir::Expr,
    pub psi4: ax_ir::Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PetrovType {
    I,
    II,
    D,
    III,
    N,
    O,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NewmanPenroseError {
    #[error("null_tetrad_from_metric requires a 4-dimensional metric")]
    RequiresFourDimensions,
    #[error("null_tetrad_from_metric requires a diagonal Lorentzian metric for auto-construction")]
    UnsupportedMetricForAutoTetrad,
    #[error("verify_null_tetrad failed: l·l is not zero")]
    LNotNull,
    #[error("verify_null_tetrad failed: n·n is not zero")]
    NNotNull,
    #[error("verify_null_tetrad failed: m·m is not zero")]
    MNotNull,
    #[error("verify_null_tetrad failed: m_bar·m_bar is not zero")]
    MBarNotNull,
    #[error("verify_null_tetrad failed: l·n is not normalized to -1")]
    LNNormalization,
    #[error("verify_null_tetrad failed: m·m_bar is not normalized to 1")]
    MMBarNormalization,
    #[error("verify_null_tetrad failed: tetrad contains non-orthogonal cross terms")]
    CrossTermNotZero,
    #[error("spin_coefficients dimension mismatch")]
    SpinCoefficientDimensionMismatch,
    #[error("weyl_scalars dimension mismatch")]
    WeylScalarDimensionMismatch,
    #[error("petrov_classify requires exact algebraic vanishing/non-vanishing information after simplification")]
    IndeterminatePetrovType,
}

fn sqrt_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("sqrt"), vec![expr])
}

fn reciprocal_sqrt(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::pow(sqrt_expr(expr, interner), Expr::Int((-1).into()))
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn i_sym(interner: &ax_ir::Interner) -> Expr {
    let _ = interner;
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
}

fn simplified_sign_and_abs(expr: &Expr) -> Option<(i8, Expr)> {
    match expr {
        Expr::Group(inner, _) => simplified_sign_and_abs(inner),
        Expr::Neg(inner) => Some((-1, inner.as_ref().clone())),
        Expr::Int(n) if *n < 0.into() => Some((-1, Expr::Int((-n).clone()))),
        Expr::Rational(r) if *r < BigRational::from_integer(0.into()) => {
            Some((-1, Expr::Rational(-r.clone())))
        }
        Expr::Float(v) if *v < 0.0 => Some((-1, Expr::Float(-v))),
        Expr::Mul(factors) => {
            let first = factors.first()?;
            let (sign, abs_first) = simplified_sign_and_abs(first)?;
            let mut abs_factors = Vec::new();
            if abs_first != Expr::one() || factors.len() == 1 {
                abs_factors.push(abs_first);
            }
            abs_factors.extend(factors.iter().skip(1).cloned());
            let abs_expr = match abs_factors.as_slice() {
                [] => Expr::one(),
                [single] => single.clone(),
                _ => Expr::mul(abs_factors),
            };
            Some((sign, abs_expr))
        }
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Some((1, expr.clone())),
        _ => Some((1, expr.clone())),
    }
}

fn metric_contract(
    g: &SymbolicMatrix,
    lhs: &[Expr],
    rhs: &[Expr],
    interner: &ax_ir::Interner,
) -> Expr {
    let mut terms = Vec::new();
    for a in 0..g.dim {
        for b in 0..g.dim {
            let gab = g.get(a, b).clone();
            if gab == Expr::zero() || lhs[a] == Expr::zero() || rhs[b] == Expr::zero() {
                continue;
            }
            terms.push(Expr::mul(vec![gab, lhs[a].clone(), rhs[b].clone()]));
        }
    }
    finalize_scalar(Expr::add(terms), interner)
}

fn lower_vector(v: &[Expr], g: &SymbolicMatrix, interner: &ax_ir::Interner) -> Vec<Expr> {
    (0..g.dim)
        .map(|a| {
            let terms = (0..g.dim)
                .filter_map(|b| {
                    let gab = g.get(a, b).clone();
                    if gab == Expr::zero() || v[b] == Expr::zero() {
                        None
                    } else {
                        Some(Expr::mul(vec![gab, v[b].clone()]))
                    }
                })
                .collect::<Vec<_>>();
            np_simplify(Expr::add(terms), interner)
        })
        .collect()
}

fn lower_first_index_rank4(
    t: &[Vec<Vec<Vec<Expr>>>],
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<Vec<Expr>>>> {
    let n = g.dim;
    let mut out = vec![vec![vec![vec![Expr::zero(); n]; n]; n]; n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let terms = (0..n)
                        .filter_map(|m| {
                            let gam = g.get(a, m).clone();
                            if gam == Expr::zero() || t[m][b][c][d] == Expr::zero() {
                                None
                            } else {
                                Some(Expr::mul(vec![gam, t[m][b][c][d].clone()]))
                            }
                        })
                        .collect::<Vec<_>>();
                    out[a][b][c][d] = np_simplify(Expr::add(terms), interner);
                }
            }
        }
    }
    out
}

fn directional_covariant_derivative_covector(
    covector: &[Expr],
    direction: &[Expr],
    gamma: &[Vec<Vec<Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Expr> {
    let n = covector.len();
    let derivatives = (0..n)
        .map(|coord_index| {
            covariant_derivative_covector(covector, gamma, coord_index, coords, interner)
        })
        .collect::<Vec<_>>();
    (0..n)
        .map(|a| {
            let terms = (0..n)
                .filter_map(|b| {
                    if direction[b] == Expr::zero() || derivatives[b][a] == Expr::zero() {
                        None
                    } else {
                        Some(Expr::mul(vec![
                            direction[b].clone(),
                            derivatives[b][a].clone(),
                        ]))
                    }
                })
                .collect::<Vec<_>>();
            np_simplify(Expr::add(terms), interner)
        })
        .collect()
}

fn tetrad_projection(
    projector: &[Expr],
    direction: &[Expr],
    target_covector: &[Expr],
    gamma: &[Vec<Vec<Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Expr {
    let directional = directional_covariant_derivative_covector(
        target_covector,
        direction,
        gamma,
        coords,
        interner,
    );
    let terms = projector
        .iter()
        .zip(directional.iter())
        .filter_map(|(lhs, rhs)| {
            if *lhs == Expr::zero() || *rhs == Expr::zero() {
                None
            } else {
                Some(Expr::mul(vec![lhs.clone(), rhs.clone()]))
            }
        })
        .collect::<Vec<_>>();
    np_simplify(Expr::add(terms), interner)
}

fn determinant_3x3(m: [[Expr; 3]; 3], interner: &ax_ir::Interner) -> Expr {
    let term1 = Expr::mul(vec![m[0][0].clone(), m[1][1].clone(), m[2][2].clone()]);
    let term2 = Expr::mul(vec![m[0][1].clone(), m[1][2].clone(), m[2][0].clone()]);
    let term3 = Expr::mul(vec![m[0][2].clone(), m[1][0].clone(), m[2][1].clone()]);
    let term4 = Expr::mul(vec![m[0][2].clone(), m[1][1].clone(), m[2][0].clone()]);
    let term5 = Expr::mul(vec![m[0][0].clone(), m[1][2].clone(), m[2][1].clone()]);
    let term6 = Expr::mul(vec![m[0][1].clone(), m[1][0].clone(), m[2][2].clone()]);
    np_simplify(
        Expr::add(vec![
            term1,
            term2,
            term3,
            Expr::neg(term4),
            Expr::neg(term5),
            Expr::neg(term6),
        ]),
        interner,
    )
}

fn normalize_np_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| normalize_np_expr(term, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .into_iter()
                .map(|factor| normalize_np_expr(factor, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(normalize_np_expr(*inner, interner)),
        Expr::Pow(base, exp) => {
            let base = normalize_np_expr(*base, interner);
            let exp = normalize_np_expr(*exp, interner);
            if matches!(exp, Expr::Int(ref n) if *n == (-2).into()) {
                if let Expr::Call(sym, args) = &base {
                    if args.len() == 1 && interner.resolve(*sym) == "sqrt" {
                        return Expr::pow(args[0].clone(), Expr::Int((-1).into()));
                    }
                }
            }
            Expr::pow(base, exp)
        }
        Expr::Call(sym, args) => Expr::Call(
            sym,
            args.into_iter()
                .map(|arg| normalize_np_expr(arg, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normalize_np_expr(*re, interner)),
            Box::new(normalize_np_expr(*im, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| normalize_np_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|item| normalize_np_expr(item, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Group(inner, relation) => {
            Expr::Group(Box::new(normalize_np_expr(*inner, interner)), relation)
        }
        other => other,
    }
}

fn np_simplify(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    simplify_expr(normalize_np_expr(expr, interner), interner)
}

fn finalize_scalar(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let simplified = np_simplify(expr, interner);
    let mut current = simplified;
    for _ in 0..4 {
        let collected = collect_terms_expr(&current, interner);
        let next = eval_expr(&collected);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn is_exact_zero(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    finalize_scalar(expr.clone(), interner) == Expr::zero()
}

fn is_exact_nonzero(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    !is_exact_zero(expr, interner)
}

/// Verify the Newman-Penrose null tetrad normalization and orthogonality conditions.
pub fn verify_null_tetrad(
    tetrad: &NullTetrad,
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<(), NewmanPenroseError> {
    if metric_contract(g, &tetrad.l, &tetrad.l, interner) != Expr::zero() {
        return Err(NewmanPenroseError::LNotNull);
    }
    if metric_contract(g, &tetrad.n, &tetrad.n, interner) != Expr::zero() {
        return Err(NewmanPenroseError::NNotNull);
    }
    if metric_contract(g, &tetrad.m, &tetrad.m, interner) != Expr::zero() {
        return Err(NewmanPenroseError::MNotNull);
    }
    if metric_contract(g, &tetrad.m_bar, &tetrad.m_bar, interner) != Expr::zero() {
        return Err(NewmanPenroseError::MBarNotNull);
    }
    if metric_contract(g, &tetrad.l, &tetrad.n, interner) != Expr::Int((-1).into()) {
        return Err(NewmanPenroseError::LNNormalization);
    }
    if metric_contract(g, &tetrad.m, &tetrad.m_bar, interner) != Expr::one() {
        return Err(NewmanPenroseError::MMBarNormalization);
    }
    let cross_terms = [
        metric_contract(g, &tetrad.l, &tetrad.m, interner),
        metric_contract(g, &tetrad.l, &tetrad.m_bar, interner),
        metric_contract(g, &tetrad.n, &tetrad.m, interner),
        metric_contract(g, &tetrad.n, &tetrad.m_bar, interner),
    ];
    if cross_terms.iter().any(|term| *term != Expr::zero()) {
        return Err(NewmanPenroseError::CrossTermNotZero);
    }
    Ok(())
}

/// Auto-construct a Newman-Penrose null tetrad for a diagonal Lorentzian 4-metric.
pub fn null_tetrad_from_metric(
    g: &crate::SymbolicMatrix,
    _coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<NullTetrad, NewmanPenroseError> {
    if g.dim != 4 {
        return Err(NewmanPenroseError::RequiresFourDimensions);
    }
    for a in 0..4 {
        for b in 0..4 {
            if a == b {
                continue;
            }
            if np_simplify(g.get(a, b).clone(), interner) != Expr::zero() {
                return Err(NewmanPenroseError::UnsupportedMetricForAutoTetrad);
            }
        }
    }

    let diagonals = (0..4)
        .map(|i| np_simplify(g.get(i, i).clone(), interner))
        .collect::<Vec<_>>();
    let mut signs_and_abs = diagonals
        .iter()
        .map(simplified_sign_and_abs)
        .collect::<Option<Vec<_>>>()
        .ok_or(NewmanPenroseError::UnsupportedMetricForAutoTetrad)?;
    let mut negative_count = signs_and_abs.iter().filter(|(sign, _)| *sign < 0).count();
    if negative_count != 1
        && signs_and_abs
            .iter()
            .enumerate()
            .skip(1)
            .all(|(_, (sign, _))| *sign > 0)
    {
        signs_and_abs[0] = (-1, np_simplify(Expr::neg(diagonals[0].clone()), interner));
        negative_count = signs_and_abs.iter().filter(|(sign, _)| *sign < 0).count();
    }
    if negative_count != 1 {
        return Err(NewmanPenroseError::UnsupportedMetricForAutoTetrad);
    }
    let time_slot = signs_and_abs
        .iter()
        .position(|(sign, _)| *sign < 0)
        .ok_or(NewmanPenroseError::UnsupportedMetricForAutoTetrad)?;
    let space_slots = (0..4).filter(|slot| *slot != time_slot).collect::<Vec<_>>();
    if space_slots.len() != 3 {
        return Err(NewmanPenroseError::UnsupportedMetricForAutoTetrad);
    }
    for coord_slot in &space_slots {
        if signs_and_abs[*coord_slot].0 < 0 {
            return Err(NewmanPenroseError::UnsupportedMetricForAutoTetrad);
        }
    }
    let coeff = |slot: usize| {
        reciprocal_sqrt(
            Expr::mul(vec![Expr::Int(2.into()), signs_and_abs[slot].1.clone()]),
            interner,
        )
    };
    let i = i_sym(interner);
    let mut l = vec![Expr::zero(); 4];
    let mut n = vec![Expr::zero(); 4];
    let mut m = vec![Expr::zero(); 4];
    let mut m_bar = vec![Expr::zero(); 4];
    l[time_slot] = coeff(time_slot);
    l[space_slots[0]] = coeff(space_slots[0]);
    n[time_slot] = coeff(time_slot);
    n[space_slots[0]] = np_simplify(Expr::neg(coeff(space_slots[0])), interner);
    m[space_slots[1]] = coeff(space_slots[1]);
    m[space_slots[2]] = np_simplify(Expr::mul(vec![i.clone(), coeff(space_slots[2])]), interner);
    m_bar[space_slots[1]] = coeff(space_slots[1]);
    m_bar[space_slots[2]] = np_simplify(
        Expr::neg(Expr::mul(vec![i.clone(), coeff(space_slots[2])])),
        interner,
    );
    let tetrad = NullTetrad { l, n, m, m_bar };
    verify_null_tetrad(&tetrad, g, interner)?;
    Ok(tetrad)
}

/// Compute the 12 Newman-Penrose spin coefficients from the Levi-Civita covariant derivatives
/// of the tetrad one-forms. The conventions used here are:
/// κ = -m^a D l_a, σ = -m^a δ l_a, ρ = -m^a \barδ l_a, τ = -m^a Δ l_a,
/// λ = \bar m^a \barδ n_a, ν = \bar m^a Δ n_a, μ = \bar m^a δ n_a, π = \bar m^a D n_a,
/// ε = 1/2 (n^a D l_a - m^a D \bar m_a), γ = 1/2 (n^a Δ l_a - m^a Δ \bar m_a),
/// α = 1/2 (n^a \barδ l_a - m^a \barδ \bar m_a), β = 1/2 (n^a δ l_a - m^a δ \bar m_a),
/// where D=l^b∇_b, Δ=n^b∇_b, δ=m^b∇_b, and \barδ=\bar m^b∇_b.
pub fn spin_coefficients(
    tetrad: &NullTetrad,
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    g: &crate::SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<SpinCoefficients, NewmanPenroseError> {
    let n = g.dim;
    if n != 4
        || gamma.len() != n
        || coords.len() != n
        || tetrad.l.len() != n
        || tetrad.n.len() != n
        || tetrad.m.len() != n
        || tetrad.m_bar.len() != n
        || gamma
            .iter()
            .any(|plane| plane.len() != n || plane.iter().any(|row| row.len() != n))
    {
        return Err(NewmanPenroseError::SpinCoefficientDimensionMismatch);
    }

    let l_cov = lower_vector(&tetrad.l, g, interner);
    let n_cov = lower_vector(&tetrad.n, g, interner);
    let m_bar_cov = lower_vector(&tetrad.m_bar, g, interner);

    let d_l = tetrad_projection(&tetrad.m, &tetrad.l, &l_cov, gamma, coords, interner);
    let delta_l = tetrad_projection(&tetrad.m, &tetrad.m, &l_cov, gamma, coords, interner);
    let delta_bar_l = tetrad_projection(&tetrad.m, &tetrad.m_bar, &l_cov, gamma, coords, interner);
    let delta_n = tetrad_projection(&tetrad.m_bar, &tetrad.m, &n_cov, gamma, coords, interner);
    let delta_bar_n = tetrad_projection(
        &tetrad.m_bar,
        &tetrad.m_bar,
        &n_cov,
        gamma,
        coords,
        interner,
    );
    let delta_big_n = tetrad_projection(&tetrad.m_bar, &tetrad.n, &n_cov, gamma, coords, interner);
    let d_n = tetrad_projection(&tetrad.m_bar, &tetrad.l, &n_cov, gamma, coords, interner);
    let eps_l = tetrad_projection(&tetrad.n, &tetrad.l, &l_cov, gamma, coords, interner);
    let eps_mbar = tetrad_projection(&tetrad.m, &tetrad.l, &m_bar_cov, gamma, coords, interner);
    let gamma_l = tetrad_projection(&tetrad.n, &tetrad.n, &l_cov, gamma, coords, interner);
    let gamma_mbar = tetrad_projection(&tetrad.m, &tetrad.n, &m_bar_cov, gamma, coords, interner);
    let alpha_l = tetrad_projection(&tetrad.n, &tetrad.m_bar, &l_cov, gamma, coords, interner);
    let alpha_mbar = tetrad_projection(
        &tetrad.m,
        &tetrad.m_bar,
        &m_bar_cov,
        gamma,
        coords,
        interner,
    );
    let beta_l = tetrad_projection(&tetrad.n, &tetrad.m, &l_cov, gamma, coords, interner);
    let beta_mbar = tetrad_projection(&tetrad.m, &tetrad.m, &m_bar_cov, gamma, coords, interner);

    Ok(SpinCoefficients {
        kappa: np_simplify(Expr::neg(d_l), interner),
        sigma: np_simplify(Expr::neg(delta_l), interner),
        lambda: np_simplify(delta_bar_n, interner),
        nu: np_simplify(delta_big_n, interner),
        rho: np_simplify(Expr::neg(delta_bar_l), interner),
        mu: np_simplify(delta_n, interner),
        tau: np_simplify(
            Expr::neg(tetrad_projection(
                &tetrad.m, &tetrad.n, &l_cov, gamma, coords, interner,
            )),
            interner,
        ),
        pi: np_simplify(d_n, interner),
        epsilon: np_simplify(
            Expr::mul(vec![half(), Expr::add(vec![eps_l, Expr::neg(eps_mbar)])]),
            interner,
        ),
        gamma: np_simplify(
            Expr::mul(vec![
                half(),
                Expr::add(vec![gamma_l, Expr::neg(gamma_mbar)]),
            ]),
            interner,
        ),
        alpha: np_simplify(
            Expr::mul(vec![
                half(),
                Expr::add(vec![alpha_l, Expr::neg(alpha_mbar)]),
            ]),
            interner,
        ),
        beta: np_simplify(
            Expr::mul(vec![half(), Expr::add(vec![beta_l, Expr::neg(beta_mbar)])]),
            interner,
        ),
    })
}

/// Compute the five Newman-Penrose Weyl scalars by contracting the fully covariant Weyl tensor
/// with the null tetrad in the standard NP order.
pub fn weyl_scalars(
    weyl: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    tetrad: &NullTetrad,
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<WeylScalars, NewmanPenroseError> {
    let n = g.dim;
    if n != 4
        || weyl.len() != n
        || tetrad.l.len() != n
        || tetrad.n.len() != n
        || tetrad.m.len() != n
        || tetrad.m_bar.len() != n
        || weyl.iter().any(|cube| {
            cube.len() != n
                || cube
                    .iter()
                    .any(|plane| plane.len() != n || plane.iter().any(|row| row.len() != n))
        })
    {
        return Err(NewmanPenroseError::WeylScalarDimensionMismatch);
    }
    let cov_weyl = lower_first_index_rank4(weyl, g, interner);
    let contract = |u: &[Expr], v: &[Expr], w: &[Expr], x: &[Expr]| -> Expr {
        let mut terms = Vec::new();
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        let cabcd = cov_weyl[a][b][c][d].clone();
                        if cabcd == Expr::zero()
                            || u[a] == Expr::zero()
                            || v[b] == Expr::zero()
                            || w[c] == Expr::zero()
                            || x[d] == Expr::zero()
                        {
                            continue;
                        }
                        terms.push(Expr::mul(vec![
                            cabcd,
                            u[a].clone(),
                            v[b].clone(),
                            w[c].clone(),
                            x[d].clone(),
                        ]));
                    }
                }
            }
        }
        finalize_scalar(Expr::neg(Expr::add(terms)), interner)
    };
    Ok(WeylScalars {
        psi0: contract(&tetrad.l, &tetrad.m, &tetrad.l, &tetrad.m),
        psi1: contract(&tetrad.l, &tetrad.n, &tetrad.l, &tetrad.m),
        psi2: contract(&tetrad.l, &tetrad.m, &tetrad.m_bar, &tetrad.n),
        psi3: contract(&tetrad.l, &tetrad.n, &tetrad.m_bar, &tetrad.n),
        psi4: contract(&tetrad.n, &tetrad.m_bar, &tetrad.n, &tetrad.m_bar),
    })
}

/// Classify the Weyl tensor algebraically using the Newman-Penrose Weyl scalars.
pub fn petrov_classify(
    scalars: &WeylScalars,
    interner: &ax_ir::Interner,
) -> Result<PetrovType, NewmanPenroseError> {
    let psi0 = np_simplify(scalars.psi0.clone(), interner);
    let psi1 = np_simplify(scalars.psi1.clone(), interner);
    let psi2 = np_simplify(scalars.psi2.clone(), interner);
    let psi3 = np_simplify(scalars.psi3.clone(), interner);
    let psi4 = np_simplify(scalars.psi4.clone(), interner);

    if [
        psi0.clone(),
        psi1.clone(),
        psi2.clone(),
        psi3.clone(),
        psi4.clone(),
    ]
    .iter()
    .all(|expr| is_exact_zero(expr, interner))
    {
        return Ok(PetrovType::O);
    }

    let z0 = is_exact_zero(&psi0, interner);
    let z1 = is_exact_zero(&psi1, interner);
    let z2 = is_exact_zero(&psi2, interner);
    let z3 = is_exact_zero(&psi3, interner);
    let z4 = is_exact_zero(&psi4, interner);
    if z1 && z2 && z3 && z0 != z4 {
        return Ok(PetrovType::N);
    }
    if z0 && z1 && z2 && is_exact_nonzero(&psi3, interner) && z4 {
        return Ok(PetrovType::III);
    }
    if z1 && z2 && z3 && is_exact_nonzero(&psi0, interner) && z4 {
        return Ok(PetrovType::N);
    }
    if z0 && z1 && z3 && z4 && is_exact_nonzero(&psi2, interner) {
        return Ok(PetrovType::D);
    }

    let i_inv = np_simplify(
        Expr::add(vec![
            Expr::mul(vec![psi0.clone(), psi4.clone()]),
            Expr::neg(Expr::mul(vec![
                Expr::Int(4.into()),
                psi1.clone(),
                psi3.clone(),
            ])),
            Expr::mul(vec![
                Expr::Int(3.into()),
                Expr::pow(psi2.clone(), Expr::Int(2.into())),
            ]),
        ]),
        interner,
    );
    let j_inv = determinant_3x3(
        [
            [psi4.clone(), psi3.clone(), psi2.clone()],
            [psi3.clone(), psi2.clone(), psi1.clone()],
            [psi2.clone(), psi1.clone(), psi0.clone()],
        ],
        interner,
    );
    let invariant_relation = np_simplify(
        Expr::add(vec![
            Expr::pow(i_inv.clone(), Expr::Int(3.into())),
            Expr::neg(Expr::mul(vec![
                Expr::Int(27.into()),
                Expr::pow(j_inv.clone(), Expr::Int(2.into())),
            ])),
        ]),
        interner,
    );
    if is_exact_zero(&invariant_relation, interner) && !(z0 && z1 && z2 && z3 && z4) {
        return Ok(PetrovType::II);
    }

    Ok(PetrovType::I)
}

#[cfg(test)]
mod tests {
    use super::{
        null_tetrad_from_metric, petrov_classify, verify_null_tetrad, weyl_scalars,
        NewmanPenroseError, PetrovType,
    };
    use crate::{
        christoffel_from_metric, ricci_from_riemann, ricci_scalar, riemann_from_christoffel,
        weyl_from_curvature, SymbolicMatrix,
    };
    use ax_ir::{Convention, Expr, Interner};

    fn schwarzschild_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let t = interner.get_or_intern("t");
        let r = interner.get_or_intern("r");
        let theta = interner.get_or_intern("theta");
        let phi = interner.get_or_intern("phi");
        let radial = Expr::Sym(r);
        let f = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Int((-2).into()),
                Expr::pow(radial.clone(), Expr::Int((-1).into())),
            ]),
        ]);
        let sin = interner.get_or_intern("sin");
        let mut g = SymbolicMatrix::new(4);
        g.set(0, 0, Expr::neg(f.clone()));
        g.set(1, 1, Expr::pow(f, Expr::Int((-1).into())));
        g.set(2, 2, Expr::pow(radial.clone(), Expr::Int(2.into())));
        g.set(
            3,
            3,
            Expr::mul(vec![
                Expr::pow(radial, Expr::Int(2.into())),
                Expr::pow(Expr::Call(sin, vec![Expr::Sym(theta)]), Expr::Int(2.into())),
            ]),
        );
        (g, vec![t, r, theta, phi])
    }

    fn flat_frw_metric(interner: &Interner) -> (SymbolicMatrix, Vec<lasso::Spur>) {
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let a = interner.get_or_intern("a");
        let scale = Expr::Call(a, vec![Expr::Sym(t)]);
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::pow(scale.clone(), Expr::Int(2.into())),
            Expr::pow(scale.clone(), Expr::Int(2.into())),
            Expr::pow(scale, Expr::Int(2.into())),
        ]);
        (g, vec![t, x, y, z])
    }

    fn curvature_data(
        g: &SymbolicMatrix,
        coords: &[lasso::Spur],
        interner: &Interner,
    ) -> (
        Vec<Vec<Vec<Expr>>>,
        Vec<Vec<Vec<Vec<Expr>>>>,
        Vec<Vec<Expr>>,
        Expr,
        Vec<Vec<Vec<Vec<Expr>>>>,
    ) {
        let gamma = christoffel_from_metric(g, coords, interner);
        let riemann = riemann_from_christoffel(&gamma, coords, interner, &Convention::default());
        let ricci = ricci_from_riemann(&riemann, g.dim, interner, &Convention::default());
        let scalar = ricci_scalar(&ricci, &g.symbolic_inverse(interner), interner);
        let weyl = weyl_from_curvature(&riemann, &ricci, &scalar, g, interner).expect("weyl");
        (gamma, riemann, ricci, scalar, weyl)
    }

    #[test]
    fn minkowski_auto_tetrad_verifies() {
        let interner = Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let tetrad = null_tetrad_from_metric(&g, &coords, &interner).expect("tetrad");
        verify_null_tetrad(&tetrad, &g, &interner).expect("verify");
    }

    #[test]
    fn minkowski_weyl_scalars_all_zero_and_petrov_o() {
        let interner = Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let tetrad = null_tetrad_from_metric(&g, &coords, &interner).expect("tetrad");
        let (_, _, _, _, weyl) = curvature_data(&g, &coords, &interner);
        let scalars = weyl_scalars(&weyl, &tetrad, &g, &interner).expect("weyl scalars");
        assert_eq!(scalars.psi0, Expr::zero());
        assert_eq!(scalars.psi1, Expr::zero());
        assert_eq!(scalars.psi2, Expr::zero());
        assert_eq!(scalars.psi3, Expr::zero());
        assert_eq!(scalars.psi4, Expr::zero());
        assert_eq!(
            petrov_classify(&scalars, &interner).expect("petrov"),
            PetrovType::O
        );
    }

    #[test]
    fn schwarzschild_auto_tetrad_gives_only_psi2_nonzero() {
        let interner = Interner::new();
        let (g, coords) = schwarzschild_metric(&interner);
        let tetrad = null_tetrad_from_metric(&g, &coords, &interner).expect("tetrad");
        let (_, _, _, _, weyl) = curvature_data(&g, &coords, &interner);
        let scalars = weyl_scalars(&weyl, &tetrad, &g, &interner).expect("weyl scalars");
        assert_eq!(scalars.psi0, Expr::zero());
        assert_eq!(scalars.psi1, Expr::zero());
        assert_ne!(scalars.psi2, Expr::zero());
        assert_eq!(scalars.psi3, Expr::zero());
        assert_eq!(scalars.psi4, Expr::zero());
    }

    #[test]
    fn schwarzschild_petrov_is_d() {
        let interner = Interner::new();
        let (g, coords) = schwarzschild_metric(&interner);
        let tetrad = null_tetrad_from_metric(&g, &coords, &interner).expect("tetrad");
        let (_, _, _, _, weyl) = curvature_data(&g, &coords, &interner);
        let scalars = weyl_scalars(&weyl, &tetrad, &g, &interner).expect("weyl scalars");
        assert_eq!(
            petrov_classify(&scalars, &interner).expect("petrov"),
            PetrovType::D
        );
    }

    #[test]
    fn flat_frw_petrov_is_o() {
        let interner = Interner::new();
        let (g, coords) = flat_frw_metric(&interner);
        let tetrad = null_tetrad_from_metric(&g, &coords, &interner).expect("tetrad");
        let (_, _, _, _, weyl) = curvature_data(&g, &coords, &interner);
        let scalars = weyl_scalars(&weyl, &tetrad, &g, &interner).expect("weyl scalars");
        assert_eq!(
            petrov_classify(&scalars, &interner).expect("petrov"),
            PetrovType::O
        );
    }

    #[test]
    fn auto_tetrad_rejects_nondiagonal_metric() {
        let interner = Interner::new();
        let coords = vec![
            interner.get_or_intern("t"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let mut g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        g.set(0, 1, Expr::one());
        assert_eq!(
            null_tetrad_from_metric(&g, &coords, &interner),
            Err(NewmanPenroseError::UnsupportedMetricForAutoTetrad)
        );
    }

    #[test]
    fn auto_tetrad_rejects_non4d_metric() {
        let interner = Interner::new();
        let coords = vec![interner.get_or_intern("t"), interner.get_or_intern("x")];
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int((-1).into()), Expr::one()]);
        assert_eq!(
            null_tetrad_from_metric(&g, &coords, &interner),
            Err(NewmanPenroseError::RequiresFourDimensions)
        );
    }
}
