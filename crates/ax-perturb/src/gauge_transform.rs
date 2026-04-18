use crate::domain::{FrwBackgroundSpec, TimeCoordinate};
use crate::error::CosmologyError;
use crate::gauge::{bardeen_expressions, svt_decompose_perturbation};
use ax_ir::Expr;

/// Names the first-order scalar gauge generator components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarGaugeGenerator {
    /// Time-shift generator.
    pub time_shift: lasso::Spur,
    /// Scalar spatial-shift generator.
    pub spatial_shift: lasso::Spur,
}

/// Stores the first-order scalar gauge variation of the FRW metric scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarGaugeVariation {
    /// Variation of `Phi`.
    pub delta_phi: ax_ir::Expr,
    /// Variation of `Psi`.
    pub delta_psi: ax_ir::Expr,
    /// Variation of `B`.
    pub delta_b: ax_ir::Expr,
    /// Variation of `E`.
    pub delta_e: ax_ir::Expr,
}

/// Records whether a named quantity is gauge invariant after normalization.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeInvariantCheck {
    /// Name of the checked quantity.
    pub name: lasso::Spur,
    /// Normalized first-order gauge variation.
    pub variation: ax_ir::Expr,
    /// Whether the normalized variation is exactly zero.
    pub is_invariant: bool,
}

/// Builds the default scalar gauge generator symbols `T` and `L`.
pub fn default_scalar_gauge_generator(interner: &ax_ir::Interner) -> ScalarGaugeGenerator {
    ScalarGaugeGenerator {
        time_shift: interner.get_or_intern("T"),
        spatial_shift: interner.get_or_intern("L"),
    }
}

/// Computes the first-order scalar gauge variation of the FRW scalar metric modes.
pub fn scalar_metric_gauge_variation(
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Result<ScalarGaugeVariation, CosmologyError> {
    if bg.time_coordinate != TimeCoordinate::Conformal {
        return Err(CosmologyError::IncompatibleTimeCoordinate {
            time_coordinate: bg.time_coordinate,
            operation: "scalar first-order gauge transformation".to_string(),
        });
    }

    let eta = Expr::Sym(bg.conformal_time);
    let hubble = Expr::Sym(bg.conformal_hubble);
    let time_shift = Expr::Sym(generator.time_shift);
    let spatial_shift = Expr::Sym(generator.spatial_shift);
    let time_shift_prime = diff_expr(time_shift.clone(), eta.clone(), interner);
    let spatial_shift_prime = diff_expr(spatial_shift.clone(), eta, interner);

    Ok(ScalarGaugeVariation {
        delta_phi: Expr::neg(Expr::add(vec![
            time_shift_prime,
            Expr::mul(vec![hubble.clone(), time_shift.clone()]),
        ])),
        delta_psi: Expr::mul(vec![hubble, time_shift.clone()]),
        delta_b: Expr::add(vec![time_shift, Expr::neg(spatial_shift_prime)]),
        delta_e: Expr::neg(spatial_shift),
    })
}

/// Computes and normalizes the first-order gauge variation of the Bardeen potentials.
pub fn bardeen_variations(
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Result<Vec<GaugeInvariantCheck>, CosmologyError> {
    let decomp = svt_decompose_perturbation(bg.spatial_dim, interner)?;
    let (phi_b, psi_b) = bardeen_expressions(&decomp, bg, interner)?;
    let variation = scalar_metric_gauge_variation(bg, generator, interner)?;
    let mode_names = crate::gauge::standard_svt_mode_names(interner);

    let phi_variation = normalize_scalar_gauge_expr(
        substitute_scalar_variation(phi_b, &mode_names, &variation, interner),
        bg,
        generator,
        interner,
    );
    let psi_variation = normalize_scalar_gauge_expr(
        substitute_scalar_variation(psi_b, &mode_names, &variation, interner),
        bg,
        generator,
        interner,
    );

    Ok(vec![
        GaugeInvariantCheck {
            name: interner.get_or_intern("Phi_B"),
            is_invariant: phi_variation == Expr::zero(),
            variation: phi_variation,
        },
        GaugeInvariantCheck {
            name: interner.get_or_intern("Psi_B"),
            is_invariant: psi_variation == Expr::zero(),
            variation: psi_variation,
        },
    ])
}

/// Applies the targeted first-order scalar gauge normalization used by the invariance checks.
pub fn normalize_scalar_gauge_expr(
    expr: ax_ir::Expr,
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut current = expr;
    loop {
        let next = normalize_scalar_gauge_expr_once(current.clone(), bg, generator, interner);
        if next == current {
            return next;
        }
        current = next;
    }
}

fn normalize_scalar_gauge_expr_once(
    expr: ax_ir::Expr,
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| normalize_scalar_gauge_expr(term, bg, generator, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let normalized_factors = factors
                .into_iter()
                .map(|factor| normalize_scalar_gauge_expr(factor, bg, generator, interner))
                .collect::<Vec<_>>();

            if let Some(rewritten) =
                rewrite_mul_patterns(normalized_factors.clone(), bg, generator, interner)
            {
                rewritten
            } else {
                Expr::mul(normalized_factors)
            }
        }
        Expr::Pow(base, exp) => Expr::pow(
            normalize_scalar_gauge_expr(*base, bg, generator, interner),
            normalize_scalar_gauge_expr(*exp, bg, generator, interner),
        ),
        Expr::Neg(inner) => Expr::neg(normalize_scalar_gauge_expr(*inner, bg, generator, interner)),
        Expr::Call(sym, args) => {
            let normalized_args = args
                .into_iter()
                .map(|arg| normalize_scalar_gauge_expr(arg, bg, generator, interner))
                .collect::<Vec<_>>();
            if interner.resolve(sym) == "diff" {
                rewrite_diff(normalized_args, bg, interner)
            } else {
                Expr::Call(sym, normalized_args)
            }
        }
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normalize_scalar_gauge_expr(*re, bg, generator, interner)),
            Box::new(normalize_scalar_gauge_expr(*im, bg, generator, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            name,
            params,
            Box::new(normalize_scalar_gauge_expr(*body, bg, generator, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(normalize_scalar_gauge_expr(*lhs, bg, generator, interner)),
            Box::new(normalize_scalar_gauge_expr(*rhs, bg, generator, interner)),
            trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| {
                    (
                        normalize_scalar_gauge_expr(value, bg, generator, interner),
                        condition,
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(normalize_scalar_gauge_expr(*base, bg, generator, interner)),
            indices,
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(normalize_scalar_gauge_expr(*inner, bg, generator, interner)),
            rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            name,
            Box::new(normalize_scalar_gauge_expr(*value, bg, generator, interner)),
            Box::new(normalize_scalar_gauge_expr(*body, bg, generator, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| normalize_scalar_gauge_expr(item, bg, generator, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| normalize_scalar_gauge_expr(cell, bg, generator, interner))
                        .collect()
                })
                .collect(),
        ),
        other => other,
    }
}

fn rewrite_diff(args: Vec<Expr>, bg: &FrwBackgroundSpec, interner: &ax_ir::Interner) -> Expr {
    if args.len() != 2 {
        return Expr::Call(interner.get_or_intern("diff"), args);
    }

    let mut iter = args.into_iter();
    let inner = match iter.next() {
        Some(expr) => expr,
        None => return Expr::Call(interner.get_or_intern("diff"), Vec::new()),
    };
    let variable = match iter.next() {
        Some(expr) => expr,
        None => return Expr::Call(interner.get_or_intern("diff"), vec![inner]),
    };

    let eta_expr = Expr::Sym(bg.conformal_time);
    if variable != eta_expr {
        return Expr::Call(interner.get_or_intern("diff"), vec![inner, variable]);
    }

    match inner {
        Expr::Neg(inner) => Expr::neg(diff_expr(*inner, eta_expr, interner)),
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| diff_expr(term, eta_expr.clone(), interner))
                .collect(),
        ),
        other => Expr::Call(interner.get_or_intern("diff"), vec![other, eta_expr]),
    }
}

fn rewrite_mul_patterns(
    factors: Vec<Expr>,
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if let Some(rewritten) = replace_pair(
        &factors,
        |expr| matches_a_inverse(expr, bg),
        |expr| matches_diff_a_t(expr, bg, generator, interner),
        || t_prime_plus_h_t(bg, generator, interner),
    ) {
        return Some(rewritten);
    }

    replace_pair(
        &factors,
        |expr| matches_a_inverse(expr, bg),
        |expr| matches_diff_a(expr, bg, interner),
        || Expr::Sym(bg.conformal_hubble),
    )
}

fn replace_pair(
    factors: &[Expr],
    left_match: impl Fn(&Expr) -> bool,
    right_match: impl Fn(&Expr) -> bool,
    replacement: impl Fn() -> Expr,
) -> Option<Expr> {
    let left_idx = factors.iter().position(left_match)?;
    let right_idx = factors
        .iter()
        .enumerate()
        .find_map(|(idx, expr)| (idx != left_idx && right_match(expr)).then_some(idx))?;
    let first_idx = left_idx.min(right_idx);

    let mut rebuilt = Vec::new();
    let mut inserted = false;
    for (idx, expr) in factors.iter().enumerate() {
        if idx == first_idx && !inserted {
            rebuilt.push(replacement());
            inserted = true;
        }
        if idx == left_idx || idx == right_idx {
            continue;
        }
        rebuilt.push(expr.clone());
    }

    if !inserted {
        rebuilt.push(replacement());
    }
    Some(Expr::mul(rebuilt))
}

fn matches_a_inverse(expr: &Expr, bg: &FrwBackgroundSpec) -> bool {
    matches!(
        expr,
        Expr::Pow(base, exp)
            if matches!(base.as_ref(), Expr::Sym(sym) if *sym == bg.scale_factor)
                && matches!(exp.as_ref(), Expr::Int(n) if *n == (-1).into())
    )
}

fn matches_diff_a(expr: &Expr, bg: &FrwBackgroundSpec, interner: &ax_ir::Interner) -> bool {
    matches!(
        expr,
        Expr::Call(sym, args)
            if interner.resolve(*sym) == "diff"
                && args.len() == 2
                && matches!(&args[0], Expr::Sym(scale) if *scale == bg.scale_factor)
                && matches!(&args[1], Expr::Sym(time) if *time == bg.conformal_time)
    )
}

fn matches_diff_a_t(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> bool {
    let Expr::Call(sym, args) = expr else {
        return false;
    };
    if interner.resolve(*sym) != "diff" || args.len() != 2 {
        return false;
    }
    if !matches!(&args[1], Expr::Sym(time) if *time == bg.conformal_time) {
        return false;
    }

    matches_a_times_t(&args[0], bg, generator)
}

fn matches_a_times_t(
    expr: &Expr,
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
) -> bool {
    let Expr::Mul(factors) = expr else {
        return false;
    };
    if factors.len() != 2 {
        return false;
    }
    factors
        .iter()
        .any(|factor| matches!(factor, Expr::Sym(sym) if *sym == bg.scale_factor))
        && factors
            .iter()
            .any(|factor| matches!(factor, Expr::Sym(sym) if *sym == generator.time_shift))
}

fn t_prime_plus_h_t(
    bg: &FrwBackgroundSpec,
    generator: &ScalarGaugeGenerator,
    interner: &ax_ir::Interner,
) -> Expr {
    let eta = Expr::Sym(bg.conformal_time);
    let time_shift = Expr::Sym(generator.time_shift);
    Expr::add(vec![
        diff_expr(time_shift.clone(), eta, interner),
        Expr::mul(vec![Expr::Sym(bg.conformal_hubble), time_shift]),
    ])
}

fn substitute_scalar_variation(
    expr: Expr,
    names: &crate::domain::SvtModeNames,
    variation: &ScalarGaugeVariation,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Sym(sym) if sym == names.phi => variation.delta_phi.clone(),
        Expr::Sym(sym) if sym == names.psi => variation.delta_psi.clone(),
        Expr::Sym(sym) if sym == names.b => variation.delta_b.clone(),
        Expr::Sym(sym) if sym == names.e => variation.delta_e.clone(),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_scalar_variation(*re, names, variation, interner)),
            Box::new(substitute_scalar_variation(*im, names, variation, interner)),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| substitute_scalar_variation(term, names, variation, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .into_iter()
                .map(|factor| substitute_scalar_variation(factor, names, variation, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_scalar_variation(*base, names, variation, interner),
            substitute_scalar_variation(*exp, names, variation, interner),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_scalar_variation(
            *inner, names, variation, interner,
        )),
        Expr::Call(sym, args) => Expr::Call(
            sym,
            args.into_iter()
                .map(|arg| substitute_scalar_variation(arg, names, variation, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            name,
            params,
            Box::new(substitute_scalar_variation(
                *body, names, variation, interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_scalar_variation(
                *lhs, names, variation, interner,
            )),
            Box::new(substitute_scalar_variation(
                *rhs, names, variation, interner,
            )),
            trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| {
                    (
                        substitute_scalar_variation(value, names, variation, interner),
                        condition,
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_scalar_variation(
                *base, names, variation, interner,
            )),
            indices,
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(substitute_scalar_variation(
                *inner, names, variation, interner,
            )),
            rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            name,
            Box::new(substitute_scalar_variation(
                *value, names, variation, interner,
            )),
            Box::new(substitute_scalar_variation(
                *body, names, variation, interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| substitute_scalar_variation(item, names, variation, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| substitute_scalar_variation(cell, names, variation, interner))
                        .collect()
                })
                .collect(),
        ),
        other => other,
    }
}

fn diff_expr(expr: Expr, variable: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, variable])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_metric_gauge_variation_matches_standard_formulas() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let generator = default_scalar_gauge_generator(&interner);
        let variation = scalar_metric_gauge_variation(&bg, &generator, &interner).unwrap();

        let t = Expr::Sym(generator.time_shift);
        let l = Expr::Sym(generator.spatial_shift);
        let eta = Expr::Sym(bg.conformal_time);
        let h = Expr::Sym(bg.conformal_hubble);

        assert_eq!(
            variation.delta_phi,
            Expr::neg(Expr::add(vec![
                diff_expr(t.clone(), eta.clone(), &interner),
                Expr::mul(vec![h.clone(), t.clone()]),
            ]))
        );
        assert_eq!(variation.delta_psi, Expr::mul(vec![h, t.clone()]));
        assert_eq!(
            variation.delta_b,
            Expr::add(vec![
                t,
                Expr::neg(diff_expr(l.clone(), eta.clone(), &interner))
            ])
        );
        assert_eq!(variation.delta_e, Expr::neg(l));
    }

    #[test]
    fn scalar_metric_gauge_variation_rejects_cosmic_time_background() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_cosmic(&interner);
        let generator = default_scalar_gauge_generator(&interner);
        let result = scalar_metric_gauge_variation(&bg, &generator, &interner);
        match result {
            Err(CosmologyError::IncompatibleTimeCoordinate {
                time_coordinate,
                operation,
            }) => {
                assert_eq!(time_coordinate, TimeCoordinate::Cosmic);
                assert_eq!(operation, "scalar first-order gauge transformation");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn normalize_scalar_gauge_expr_reduces_a_inverse_diff_a_to_hubble() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let generator = default_scalar_gauge_generator(&interner);
        let eta = Expr::Sym(bg.conformal_time);
        let expr = Expr::mul(vec![
            Expr::pow(Expr::Sym(bg.scale_factor), Expr::Int((-1).into())),
            diff_expr(Expr::Sym(bg.scale_factor), eta, &interner),
        ]);

        assert_eq!(
            normalize_scalar_gauge_expr(expr, &bg, &generator, &interner),
            Expr::Sym(bg.conformal_hubble)
        );
    }

    #[test]
    fn normalize_scalar_gauge_expr_reduces_a_inverse_diff_a_t_to_expected_sum() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let generator = default_scalar_gauge_generator(&interner);
        let eta = Expr::Sym(bg.conformal_time);
        let t = Expr::Sym(generator.time_shift);
        let expr = Expr::mul(vec![
            Expr::pow(Expr::Sym(bg.scale_factor), Expr::Int((-1).into())),
            diff_expr(
                Expr::mul(vec![Expr::Sym(bg.scale_factor), t.clone()]),
                eta.clone(),
                &interner,
            ),
        ]);

        assert_eq!(
            normalize_scalar_gauge_expr(expr, &bg, &generator, &interner),
            Expr::add(vec![
                diff_expr(t.clone(), eta, &interner),
                Expr::mul(vec![Expr::Sym(bg.conformal_hubble), t]),
            ])
        );
    }

    #[test]
    fn bardeen_variations_are_exactly_zero() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let generator = default_scalar_gauge_generator(&interner);
        let checks = bardeen_variations(&bg, &generator, &interner).unwrap();

        assert_eq!(checks.len(), 2);
        assert_eq!(interner.resolve(checks[0].name), "Phi_B");
        assert_eq!(interner.resolve(checks[1].name), "Psi_B");
        assert_eq!(checks[0].variation, Expr::zero());
        assert_eq!(checks[1].variation, Expr::zero());
        assert!(checks[0].is_invariant);
        assert!(checks[1].is_invariant);
    }
}
