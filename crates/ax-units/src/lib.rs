use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub dimensions: BTreeMap<BaseDimension, BigRational>,
    pub scale: BigRational,
    pub offset: BigRational,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseDimension {
    Length,
    Mass,
    Time,
    Current,
    Temperature,
    Amount,
    Luminosity,
}

pub fn rational(numer: i64, denom: i64) -> BigRational {
    BigRational::new(numer.into(), denom.into())
}

pub fn rational_f64(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap_or_else(BigRational::zero)
}

fn clean_dimensions(dimensions: &mut BTreeMap<BaseDimension, BigRational>) {
    dimensions.retain(|_, exp| !exp.is_zero());
}

pub fn unit_single(dim: BaseDimension) -> Unit {
    let mut dimensions = BTreeMap::new();
    dimensions.insert(dim, BigRational::one());
    Unit {
        dimensions,
        scale: BigRational::one(),
        offset: BigRational::zero(),
    }
}

pub fn unit_derived(dims: &[(BaseDimension, i64)]) -> Unit {
    unit_with_scale(dims, BigRational::one())
}

pub fn unit_scaled(dim: BaseDimension, scale: impl Into<BigRational>) -> Unit {
    let mut unit = unit_single(dim);
    unit.scale = scale.into();
    unit
}

pub fn unit_with_scale(dims: &[(BaseDimension, i64)], scale: BigRational) -> Unit {
    let mut dimensions = BTreeMap::new();
    for (dim, exp) in dims {
        dimensions.insert(*dim, BigRational::from_integer((*exp).into()));
    }
    clean_dimensions(&mut dimensions);
    Unit {
        dimensions,
        scale,
        offset: BigRational::zero(),
    }
}

pub fn mul_units(a: &Unit, b: &Unit) -> Unit {
    let mut dimensions = a.dimensions.clone();
    for (dim, exp) in &b.dimensions {
        let next = dimensions
            .get(dim)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            + exp.clone();
        if next.is_zero() {
            dimensions.remove(dim);
        } else {
            dimensions.insert(*dim, next);
        }
    }
    Unit {
        dimensions,
        scale: a.scale.clone() * b.scale.clone(),
        offset: BigRational::zero(),
    }
}

pub fn pow_unit(unit: &Unit, power: i64) -> Unit {
    let mut dimensions = BTreeMap::new();
    for (dim, exp) in &unit.dimensions {
        let next = exp.clone() * BigRational::from_integer(power.into());
        if !next.is_zero() {
            dimensions.insert(*dim, next);
        }
    }

    let scale = if power == 0 {
        BigRational::one()
    } else if power > 0 {
        let p = u32::try_from(power).unwrap_or(0);
        BigRational::new(
            unit.scale.numer().clone().pow(p),
            unit.scale.denom().clone().pow(p),
        )
    } else {
        let p = u32::try_from(-power).unwrap_or(0);
        BigRational::new(
            unit.scale.denom().clone().pow(p),
            unit.scale.numer().clone().pow(p),
        )
    };

    Unit {
        dimensions,
        scale,
        offset: BigRational::zero(),
    }
}

fn dimensionless() -> Unit {
    Unit {
        dimensions: BTreeMap::new(),
        scale: BigRational::one(),
        offset: BigRational::zero(),
    }
}

pub fn si_units(interner: &ax_ir::Interner) -> HashMap<lasso::Spur, Unit> {
    let mut units = HashMap::new();

    units.insert(
        interner.get_or_intern("m"),
        unit_single(BaseDimension::Length),
    );
    units.insert(
        interner.get_or_intern("kg"),
        unit_single(BaseDimension::Mass),
    );
    units.insert(
        interner.get_or_intern("s"),
        unit_single(BaseDimension::Time),
    );
    units.insert(
        interner.get_or_intern("A"),
        unit_single(BaseDimension::Current),
    );
    units.insert(
        interner.get_or_intern("K"),
        unit_single(BaseDimension::Temperature),
    );
    units.insert(
        interner.get_or_intern("mol"),
        unit_single(BaseDimension::Amount),
    );
    units.insert(
        interner.get_or_intern("cd"),
        unit_single(BaseDimension::Luminosity),
    );

    units.insert(
        interner.get_or_intern("N"),
        unit_derived(&[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, 1),
            (BaseDimension::Time, -2),
        ]),
    );
    units.insert(
        interner.get_or_intern("J"),
        unit_derived(&[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, 2),
            (BaseDimension::Time, -2),
        ]),
    );
    units.insert(
        interner.get_or_intern("W"),
        unit_derived(&[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, 2),
            (BaseDimension::Time, -3),
        ]),
    );
    units.insert(
        interner.get_or_intern("Pa"),
        unit_derived(&[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, -1),
            (BaseDimension::Time, -2),
        ]),
    );
    units.insert(
        interner.get_or_intern("Hz"),
        unit_derived(&[(BaseDimension::Time, -1)]),
    );
    units.insert(
        interner.get_or_intern("C"),
        unit_derived(&[(BaseDimension::Current, 1), (BaseDimension::Time, 1)]),
    );
    units.insert(
        interner.get_or_intern("V"),
        unit_derived(&[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, 2),
            (BaseDimension::Time, -3),
            (BaseDimension::Current, -1),
        ]),
    );

    units.insert(
        interner.get_or_intern("km"),
        unit_scaled(
            BaseDimension::Length,
            BigRational::from_integer(1000.into()),
        ),
    );
    units.insert(
        interner.get_or_intern("cm"),
        unit_scaled(BaseDimension::Length, rational(1, 100)),
    );
    units.insert(
        interner.get_or_intern("mm"),
        unit_scaled(BaseDimension::Length, rational(1, 1000)),
    );
    units.insert(
        interner.get_or_intern("g"),
        unit_scaled(BaseDimension::Mass, rational(1, 1000)),
    );

    units.insert(
        interner.get_or_intern("eV"),
        unit_with_scale(
            &[
                (BaseDimension::Mass, 1),
                (BaseDimension::Length, 2),
                (BaseDimension::Time, -2),
            ],
            rational_f64(1.602176634e-19),
        ),
    );

    units
}

pub fn natural_units(interner: &ax_ir::Interner) -> HashMap<lasso::Spur, Unit> {
    let mut units = HashMap::new();
    let gev = unit_with_scale(
        &[
            (BaseDimension::Mass, 1),
            (BaseDimension::Length, 2),
            (BaseDimension::Time, -2),
        ],
        rational_f64(1.602176634e-10),
    );
    units.insert(interner.get_or_intern("GeV"), gev.clone());
    units.insert(interner.get_or_intern("c"), dimensionless());
    units.insert(interner.get_or_intern("hbar"), dimensionless());
    units.insert(interner.get_or_intern("kB"), dimensionless());
    units.insert(
        interner.get_or_intern("fm"),
        Unit {
            dimensions: [(BaseDimension::Length, BigRational::one())]
                .into_iter()
                .collect(),
            scale: rational_f64(1e-15),
            offset: BigRational::zero(),
        },
    );
    units
}

fn unitless_or_same(unit: &Unit, name: &str) -> Result<Unit, String> {
    if unit.dimensions.is_empty() {
        Ok(dimensionless())
    } else {
        Err(format!("{name} requires a dimensionless argument"))
    }
}

pub fn check_dimensions(
    expr: &ax_ir::Expr,
    units_env: &HashMap<lasso::Spur, Unit>,
    interner: &ax_ir::Interner,
) -> Result<Unit, String> {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Complex(_, _) => {
            Ok(dimensionless())
        }
        Expr::Sym(sym) => Ok(units_env.get(sym).cloned().unwrap_or_else(dimensionless)),
        Expr::Neg(inner) => check_dimensions(inner, units_env, interner),
        Expr::Add(terms) => {
            let mut iter = terms.iter();
            let first = iter
                .next()
                .map(|term| check_dimensions(term, units_env, interner))
                .transpose()?
                .unwrap_or_else(dimensionless);
            for term in iter {
                let next = check_dimensions(term, units_env, interner)?;
                if next.dimensions != first.dimensions {
                    return Err("cannot add quantities with different dimensions".into());
                }
            }
            Ok(first)
        }
        Expr::Mul(factors) => {
            let mut acc = dimensionless();
            for factor in factors {
                let next = check_dimensions(factor, units_env, interner)?;
                acc = mul_units(&acc, &next);
            }
            Ok(acc)
        }
        Expr::Pow(base, exp) => {
            let base_unit = check_dimensions(base, units_env, interner)?;
            match exp.as_ref() {
                Expr::Int(n) => {
                    let p = n
                        .to_i64()
                        .ok_or_else(|| "unit powers must fit in i64".to_string())?;
                    Ok(pow_unit(&base_unit, p))
                }
                _ => {
                    if base_unit.dimensions.is_empty() {
                        Ok(dimensionless())
                    } else {
                        Err(
                            "non-integer powers of dimensionful quantities are not supported"
                                .into(),
                        )
                    }
                }
            }
        }
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            match name {
                "sin" | "cos" | "exp" | "log" => {
                    let arg = args
                        .first()
                        .ok_or_else(|| format!("{name} expects an argument"))?;
                    unitless_or_same(&check_dimensions(arg, units_env, interner)?, name)
                }
                "abs" => {
                    let arg = args
                        .first()
                        .ok_or_else(|| "abs expects an argument".to_string())?;
                    check_dimensions(arg, units_env, interner)
                }
                "sqrt" => {
                    let arg = args
                        .first()
                        .ok_or_else(|| "sqrt expects an argument".to_string())?;
                    let unit = check_dimensions(arg, units_env, interner)?;
                    let mut dimensions = BTreeMap::new();
                    for (dim, exp) in unit.dimensions {
                        let half_exp = exp / rational(2, 1);
                        if half_exp.denom() != &BigInt::one() {
                            return Err("sqrt of this unit produces fractional dimensions".into());
                        }
                        dimensions.insert(dim, half_exp);
                    }
                    Ok(Unit {
                        dimensions,
                        scale: unit.scale,
                        offset: BigRational::zero(),
                    })
                }
                _ => Ok(dimensionless()),
            }
        }
        Expr::FnDef(_, _, body) => check_dimensions(body, units_env, interner),
        Expr::Rule(_, rhs, _) => check_dimensions(rhs, units_env, interner),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => Ok(dimensionless()),
        Expr::Piecewise(cases) => {
            let mut unit: Option<Unit> = None;
            for (value, _) in cases {
                let next = check_dimensions(value, units_env, interner)?;
                if let Some(existing) = &unit {
                    if existing.dimensions != next.dimensions {
                        return Err("piecewise branches must have matching dimensions".into());
                    }
                } else {
                    unit = Some(next);
                }
            }
            Ok(unit.unwrap_or_else(dimensionless))
        }
        Expr::Indexed(base, _) => check_dimensions(base, units_env, interner),
        Expr::Let(name, value, body) => {
            let mut extended = units_env.clone();
            let value_unit = check_dimensions(value, units_env, interner)?;
            extended.insert(*name, value_unit);
            check_dimensions(body, &extended, interner)
        }
        Expr::List(_) | Expr::Matrix(_) => Ok(dimensionless()),
    }
}

pub fn convert(value: &ax_ir::Expr, from: &Unit, to: &Unit) -> Result<ax_ir::Expr, String> {
    if from.dimensions != to.dimensions {
        return Err("cannot convert between incompatible dimensions".into());
    }

    let scale = from.scale.clone() / to.scale.clone();
    let offset_adjust = from.offset.clone() - to.offset.clone();

    let scaled = Expr::mul(vec![
        value.clone(),
        if scale.is_integer() {
            Expr::Int(scale.to_integer())
        } else {
            Expr::Rational(scale)
        },
    ]);

    if offset_adjust.is_zero() {
        Ok(scaled)
    } else {
        Ok(Expr::add(vec![
            scaled,
            if offset_adjust.is_integer() {
                Expr::Int(offset_adjust.to_integer())
            } else {
                Expr::Rational(offset_adjust)
            },
        ]))
    }
}

pub fn unit_to_expr(unit: &Unit, interner: &ax_ir::Interner) -> Expr {
    let mut items = Vec::new();
    for (dim, exp) in &unit.dimensions {
        let name = match dim {
            BaseDimension::Length => "Length",
            BaseDimension::Mass => "Mass",
            BaseDimension::Time => "Time",
            BaseDimension::Current => "Current",
            BaseDimension::Temperature => "Temperature",
            BaseDimension::Amount => "Amount",
            BaseDimension::Luminosity => "Luminosity",
        };
        let exp_expr = if exp.is_integer() {
            Expr::Int(exp.to_integer())
        } else {
            Expr::Rational(exp.clone())
        };
        items.push(Expr::List(vec![
            Expr::Sym(interner.get_or_intern(name)),
            exp_expr,
        ]));
    }
    Expr::List(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_is_mass_times_acceleration() {
        let interner = ax_ir::Interner::new();
        let units = si_units(&interner);
        let kg = units[&interner.get_or_intern("kg")].clone();
        let m = units[&interner.get_or_intern("m")].clone();
        let s = units[&interner.get_or_intern("s")].clone();

        let n = units[&interner.get_or_intern("N")].clone();
        let computed = mul_units(&mul_units(&kg, &m), &pow_unit(&s, -2));
        assert_eq!(computed.dimensions, n.dimensions);
    }

    #[test]
    fn cannot_add_length_and_time() {
        let interner = ax_ir::Interner::new();
        let units = si_units(&interner);
        let mut env = HashMap::new();
        let x = interner.get_or_intern("x");
        let t = interner.get_or_intern("t");
        env.insert(x, units[&interner.get_or_intern("m")].clone());
        env.insert(t, units[&interner.get_or_intern("s")].clone());

        let expr = ax_ir::Expr::add(vec![ax_ir::Expr::Sym(x), ax_ir::Expr::Sym(t)]);
        let result = check_dimensions(&expr, &env, &interner);
        assert!(result.is_err());
    }
}
