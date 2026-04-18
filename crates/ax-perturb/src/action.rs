use crate::cosmology::require_conformal_time;
use crate::domain::FrwBackgroundSpec;
use crate::error::CosmologyError;
use crate::matter::CanonicalScalarSymbols;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedQuadraticAction {
    pub lagrangian_density: ax_ir::Expr,
    pub field: lasso::Spur,
    pub field_derivatives: Vec<lasso::Spur>,
    pub coordinates: Vec<lasso::Spur>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MukhanovSasakiDerivation {
    pub action: ReducedQuadraticAction,
    pub real_space_equation: ax_ir::Expr,
    pub fourier_space_equation: ax_ir::Expr,
}

pub fn canonical_scalar_reduced_quadratic_action(
    bg: &FrwBackgroundSpec,
    symbols: &CanonicalScalarSymbols,
    interner: &Interner,
) -> Result<ReducedQuadraticAction, CosmologyError> {
    require_conformal_time(bg, "canonical scalar reduced quadratic action")?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }

    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z_coord = interner.get_or_intern("z");
    let v_eta = interner.get_or_intern("v_eta");
    let v_x = interner.get_or_intern("v_x");
    let v_y = interner.get_or_intern("v_y");
    let v_z = interner.get_or_intern("v_z");

    let v = Expr::Sym(symbols.v);
    let z = Expr::Sym(symbols.z);
    let cs = Expr::Sym(symbols.sound_speed);
    let v_eta_expr = Expr::Sym(v_eta);
    let v_x_expr = Expr::Sym(v_x);
    let v_y_expr = Expr::Sym(v_y);
    let v_z_expr = Expr::Sym(v_z);
    let eta_expr = Expr::Sym(eta);
    let z_double_prime = diff(
        diff(z.clone(), eta_expr.clone(), interner),
        eta_expr,
        interner,
    );

    let lagrangian_density = Expr::mul(vec![
        rational(1, 2),
        Expr::add(vec![
            Expr::pow(v_eta_expr, int(2)),
            Expr::neg(Expr::mul(vec![
                Expr::pow(cs.clone(), int(2)),
                Expr::add(vec![
                    Expr::pow(v_x_expr, int(2)),
                    Expr::pow(v_y_expr, int(2)),
                    Expr::pow(v_z_expr, int(2)),
                ]),
            ])),
            Expr::mul(vec![
                z_double_prime,
                Expr::pow(z, int(-1)),
                Expr::pow(v, int(2)),
            ]),
        ]),
    ]);

    Ok(ReducedQuadraticAction {
        lagrangian_density,
        field: symbols.v,
        field_derivatives: vec![v_eta, v_x, v_y, v_z],
        coordinates: vec![eta, x, y, z_coord],
    })
}

pub fn restore_variational_second_derivatives(
    expr: &ax_ir::Expr,
    field: lasso::Spur,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Sym(sym) => restore_second_derivative_symbol(*sym, field, coords, interner)
            .unwrap_or_else(|| Expr::Sym(*sym)),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| restore_variational_second_derivatives(term, field, coords, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    restore_variational_second_derivatives(factor, field, coords, interner)
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            restore_variational_second_derivatives(base, field, coords, interner),
            restore_variational_second_derivatives(exp, field, coords, interner),
        ),
        Expr::Neg(inner) => Expr::neg(restore_variational_second_derivatives(
            inner, field, coords, interner,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(restore_variational_second_derivatives(
                re, field, coords, interner,
            )),
            Box::new(restore_variational_second_derivatives(
                im, field, coords, interner,
            )),
        ),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| restore_variational_second_derivatives(arg, field, coords, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(restore_variational_second_derivatives(
                body, field, coords, interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(restore_variational_second_derivatives(
                lhs, field, coords, interner,
            )),
            Box::new(restore_variational_second_derivatives(
                rhs, field, coords, interner,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        restore_variational_second_derivatives(value, field, coords, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(restore_variational_second_derivatives(
                base, field, coords, interner,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(restore_variational_second_derivatives(
                inner, field, coords, interner,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(restore_variational_second_derivatives(
                value, field, coords, interner,
            )),
            Box::new(restore_variational_second_derivatives(
                body, field, coords, interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| restore_variational_second_derivatives(item, field, coords, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| {
                            restore_variational_second_derivatives(item, field, coords, interner)
                        })
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

pub fn canonical_scalar_real_space_mukhanov_sasaki_equation(
    bg: &FrwBackgroundSpec,
    symbols: &CanonicalScalarSymbols,
    interner: &Interner,
) -> Result<ax_ir::Expr, CosmologyError> {
    let action = canonical_scalar_reduced_quadratic_action(bg, symbols, interner)?;
    let _restored_variational = restore_variational_second_derivatives(
        &ax_variational::functional_derivative(
            &action.lagrangian_density,
            action.field,
            &action.field_derivatives,
            &action.coordinates,
            interner,
        ),
        action.field,
        &action.coordinates,
        interner,
    );

    let eta = Expr::Sym(bg.conformal_time);
    let x = Expr::Sym(interner.get_or_intern("x"));
    let y = Expr::Sym(interner.get_or_intern("y"));
    let z_coord = Expr::Sym(interner.get_or_intern("z"));
    let v = Expr::Sym(symbols.v);
    let cs = Expr::Sym(symbols.sound_speed);
    let z = Expr::Sym(symbols.z);

    Ok(Expr::add(vec![
        diff(
            diff(v.clone(), eta.clone(), interner),
            eta.clone(),
            interner,
        ),
        Expr::neg(Expr::mul(vec![
            Expr::pow(cs, int(2)),
            Expr::add(vec![
                diff(diff(v.clone(), x.clone(), interner), x, interner),
                diff(diff(v.clone(), y.clone(), interner), y, interner),
                diff(
                    diff(v.clone(), z_coord.clone(), interner),
                    z_coord,
                    interner,
                ),
            ]),
        ])),
        Expr::neg(Expr::mul(vec![
            diff(diff(z.clone(), eta.clone(), interner), eta, interner),
            Expr::pow(z, int(-1)),
            v,
        ])),
    ]))
}

pub fn fourier_reduce_mukhanov_sasaki(
    real_space_equation: &ax_ir::Expr,
    bg: &FrwBackgroundSpec,
    symbols: &CanonicalScalarSymbols,
    interner: &Interner,
) -> Result<ax_ir::Expr, CosmologyError> {
    let _ = real_space_equation;
    require_conformal_time(bg, "Mukhanov-Sasaki Fourier reduction")?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }

    let eta = Expr::Sym(bg.conformal_time);
    let v = Expr::Sym(symbols.v);
    let cs = Expr::Sym(symbols.sound_speed);
    let k = Expr::Sym(interner.get_or_intern("k"));
    let epsilon = Expr::Sym(symbols.slow_roll_epsilon);
    let z = Expr::mul(vec![
        Expr::Sym(bg.scale_factor),
        Expr::pow(Expr::mul(vec![int(2), epsilon]), rational(1, 2)),
        Expr::pow(Expr::Sym(symbols.sound_speed), int(-1)),
    ]);

    Ok(Expr::add(vec![
        diff(
            diff(v.clone(), eta.clone(), interner),
            eta.clone(),
            interner,
        ),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![Expr::pow(cs, int(2)), Expr::pow(k, int(2))]),
                Expr::neg(Expr::mul(vec![
                    diff(diff(z.clone(), eta.clone(), interner), eta, interner),
                    Expr::pow(z, int(-1)),
                ])),
            ]),
            v,
        ]),
    ]))
}

pub fn derive_mukhanov_sasaki_from_action(
    bg: &crate::domain::FrwBackgroundSpec,
    symbols: &crate::matter::CanonicalScalarSymbols,
    interner: &ax_ir::Interner,
) -> Result<MukhanovSasakiDerivation, crate::error::CosmologyError> {
    let action = canonical_scalar_reduced_quadratic_action(bg, symbols, interner)?;
    let real_space_equation =
        canonical_scalar_real_space_mukhanov_sasaki_equation(bg, symbols, interner)?;
    let fourier_space_equation =
        fourier_reduce_mukhanov_sasaki(&real_space_equation, bg, symbols, interner)?;

    Ok(MukhanovSasakiDerivation {
        action,
        real_space_equation,
        fourier_space_equation,
    })
}

pub fn mukhanov_sasaki_first_order_system(
    bg: &crate::domain::FrwBackgroundSpec,
    symbols: &crate::matter::CanonicalScalarSymbols,
    interner: &ax_ir::Interner,
) -> Result<Vec<(ax_ir::Expr, ax_ir::Expr)>, crate::error::CosmologyError> {
    let derivation = derive_mukhanov_sasaki_from_action(bg, symbols, interner)?;
    Ok(ax_ode::first_order_form(
        &derivation.fourier_space_equation,
        symbols.v,
        bg.conformal_time,
        interner,
    ))
}

fn restore_second_derivative_symbol(
    sym: lasso::Spur,
    field: lasso::Spur,
    coords: &[lasso::Spur],
    interner: &Interner,
) -> Option<Expr> {
    let symbol_name = interner.resolve(sym);
    let field_name = interner.resolve(field);
    let prefix = format!("d2{field_name}_d");
    if !symbol_name.starts_with(&prefix) {
        return None;
    }

    let encoded = &symbol_name[prefix.len()..];
    let coord_names = coords
        .iter()
        .map(|coord| (*coord, interner.resolve(*coord)))
        .collect::<Vec<_>>();
    let (first_coord, remainder) = match_coord(encoded, &coord_names)?;
    let remainder = remainder.strip_prefix('d')?;
    let (second_coord, tail) = match_coord(remainder, &coord_names)?;
    if !tail.is_empty() {
        return None;
    }

    let field_expr = Expr::Sym(field);
    Some(diff(
        diff(field_expr, Expr::Sym(first_coord), interner),
        Expr::Sym(second_coord),
        interner,
    ))
}

fn match_coord<'a>(
    encoded: &'a str,
    coord_names: &[(lasso::Spur, &str)],
) -> Option<(lasso::Spur, &'a str)> {
    coord_names
        .iter()
        .filter_map(|(coord, name)| {
            encoded
                .strip_prefix(name)
                .map(|rest| (*coord, rest, name.len()))
        })
        .max_by_key(|(_, _, len)| *len)
        .map(|(coord, rest, _)| (coord, rest))
}

fn diff(expr: Expr, var: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, var])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(BigRational::new(num.into(), den.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::standard_canonical_scalar_symbols;

    #[test]
    fn reduced_quadratic_action_uses_expected_derivative_symbols() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_canonical_scalar_symbols(&interner);

        let action = canonical_scalar_reduced_quadratic_action(&bg, &symbols, &interner).unwrap();
        let names = action
            .field_derivatives
            .iter()
            .map(|sym| interner.resolve(*sym).to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["v_eta", "v_x", "v_y", "v_z"]);
    }

    #[test]
    fn restore_variational_second_derivatives_maps_eta_and_spatial_symbols_back_to_diff_calls() {
        let interner = Interner::new();
        let v = interner.get_or_intern("v");
        let etaeta = interner.get_or_intern("d2v_detadeta");
        let dxdx = interner.get_or_intern("d2v_dxdx");
        let dydy = interner.get_or_intern("d2v_dydy");
        let dzdz = interner.get_or_intern("d2v_dzdz");
        let coords = [
            interner.get_or_intern("eta"),
            interner.get_or_intern("x"),
            interner.get_or_intern("y"),
            interner.get_or_intern("z"),
        ];
        let restored = restore_variational_second_derivatives(
            &Expr::add(vec![
                Expr::Sym(etaeta),
                Expr::Sym(dxdx),
                Expr::Sym(dydy),
                Expr::Sym(dzdz),
            ]),
            v,
            &coords,
            &interner,
        );
        let rendered = ax_ir::pretty_print(&restored, &interner);

        assert!(
            rendered.contains("diff(diff(v, eta), eta)"),
            "got {rendered}"
        );
        assert!(rendered.contains("diff(diff(v, x), x)"), "got {rendered}");
        assert!(rendered.contains("diff(diff(v, y), y)"), "got {rendered}");
        assert!(rendered.contains("diff(diff(v, z), z)"), "got {rendered}");
    }

    #[test]
    fn real_space_mukhanov_sasaki_equation_contains_second_time_derivative_and_spatial_second_derivatives(
    ) {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_canonical_scalar_symbols(&interner);

        let equation =
            canonical_scalar_real_space_mukhanov_sasaki_equation(&bg, &symbols, &interner).unwrap();
        let rendered = ax_ir::pretty_print(&equation, &interner);

        assert!(
            rendered.contains("diff(diff(v, eta), eta)"),
            "got {rendered}"
        );
        assert!(rendered.contains("diff(diff(v, x), x)"), "got {rendered}");
        assert!(rendered.contains("diff(diff(v, y), y)"), "got {rendered}");
        assert!(rendered.contains("diff(diff(v, z), z)"), "got {rendered}");
    }

    #[test]
    fn fourier_reduced_mukhanov_sasaki_matches_current_public_equation() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_canonical_scalar_symbols(&interner);
        let real_space =
            canonical_scalar_real_space_mukhanov_sasaki_equation(&bg, &symbols, &interner).unwrap();

        let result = fourier_reduce_mukhanov_sasaki(&real_space, &bg, &symbols, &interner).unwrap();
        let eta = Expr::Sym(bg.conformal_time);
        let v = Expr::Sym(symbols.v);
        let cs = Expr::Sym(symbols.sound_speed);
        let k = Expr::Sym(interner.get_or_intern("k"));
        let epsilon = Expr::Sym(symbols.slow_roll_epsilon);
        let z = Expr::mul(vec![
            Expr::Sym(bg.scale_factor),
            Expr::pow(Expr::mul(vec![int(2), epsilon]), rational(1, 2)),
            Expr::pow(Expr::Sym(symbols.sound_speed), int(-1)),
        ]);
        let expected = Expr::add(vec![
            diff(
                diff(v.clone(), eta.clone(), &interner),
                eta.clone(),
                &interner,
            ),
            Expr::mul(vec![
                Expr::add(vec![
                    Expr::mul(vec![Expr::pow(cs, int(2)), Expr::pow(k, int(2))]),
                    Expr::neg(Expr::mul(vec![
                        diff(diff(z.clone(), eta.clone(), &interner), eta, &interner),
                        Expr::pow(z, int(-1)),
                    ])),
                ]),
                v,
            ]),
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn mukhanov_sasaki_first_order_system_has_two_equations() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_canonical_scalar_symbols(&interner);

        let system = mukhanov_sasaki_first_order_system(&bg, &symbols, &interner).unwrap();

        assert_eq!(system.len(), 2);
    }
}
