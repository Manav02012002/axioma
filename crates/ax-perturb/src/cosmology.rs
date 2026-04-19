use crate::domain::{FrwBackgroundSpec, NamedEquation, SectorKind, TimeCoordinate};
use crate::error::CosmologyError;
use crate::gauge::SVTDecomposition;
use crate::matter;
use ax_ir::{Expr, Interner};
use lasso::Spur;
use num_bigint::BigInt;

pub fn frw_background(interner: &Interner) -> FrwBackgroundSpec {
    FrwBackgroundSpec::default_flat_conformal(interner)
}

pub fn require_conformal_time(
    bg: &FrwBackgroundSpec,
    operation: &str,
) -> Result<(), CosmologyError> {
    if bg.time_coordinate == TimeCoordinate::Conformal {
        Ok(())
    } else {
        Err(CosmologyError::IncompatibleTimeCoordinate {
            time_coordinate: bg.time_coordinate,
            operation: operation.to_string(),
        })
    }
}

pub fn linearized_einstein_scalar(
    bg: &FrwBackgroundSpec,
    decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<NamedEquation>, CosmologyError> {
    let _ = decomp;
    crate::linearized::linearized_scalar_equations_as_named(bg, interner)
}

pub fn linearized_einstein_vector(
    bg: &FrwBackgroundSpec,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    Ok(
        crate::vector_tensor::derive_linear_vector_einstein_equations_poisson(bg, interner)?
            .equations,
    )
}

pub fn linearized_einstein_tensor(
    bg: &FrwBackgroundSpec,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    Ok(crate::vector_tensor::derive_linear_tensor_einstein_equations(bg, interner)?.equations)
}

pub fn tensor_mode_equation(
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedExpr>, crate::error::CosmologyError> {
    let derivation = crate::vector_tensor::derive_tensor_mode_equations(bg, interner)?;
    Ok(vec![
        crate::domain::NamedExpr {
            name: interner.get_or_intern("h_plus_eq"),
            expr: derivation.plus_equation_fourier_space,
        },
        crate::domain::NamedExpr {
            name: interner.get_or_intern("h_cross_eq"),
            expr: derivation.cross_equation_fourier_space,
        },
    ])
}

pub fn mukhanov_sasaki_equation(
    bg: &FrwBackgroundSpec,
    slow_roll_epsilon: Spur,
    interner: &Interner,
) -> Result<Expr, CosmologyError> {
    require_conformal_time(bg, "Mukhanov-Sasaki equation")?;
    let mut symbols = matter::standard_canonical_scalar_symbols(interner);
    symbols.slow_roll_epsilon = slow_roll_epsilon;
    Ok(
        crate::action::derive_mukhanov_sasaki_from_action(bg, &symbols, interner)?
            .fourier_space_equation,
    )
}

pub fn perfect_fluid_linear_conservation(
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    Ok(matter::perfect_fluid_linear_conservation_equations_newtonian(bg, interner)?.equations)
}

pub fn power_spectrum_leading(
    _bg: &FrwBackgroundSpec,
    slow_roll_epsilon: Spur,
    hubble_at_crossing: Spur,
    interner: &Interner,
) -> Expr {
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    Expr::mul(vec![
        Expr::pow(Expr::Sym(hubble_at_crossing), int(2)),
        Expr::pow(
            Expr::mul(vec![
                int(8),
                Expr::pow(pi, int(2)),
                Expr::Sym(slow_roll_epsilon),
            ]),
            int(-1),
        ),
    ])
}

pub fn spectral_index(slow_roll_epsilon: Spur, slow_roll_eta: Spur, _interner: &Interner) -> Expr {
    Expr::add(vec![
        Expr::one(),
        Expr::neg(Expr::mul(vec![int(6), Expr::Sym(slow_roll_epsilon)])),
        Expr::mul(vec![int(2), Expr::Sym(slow_roll_eta)]),
    ])
}

pub fn tensor_to_scalar_ratio(slow_roll_epsilon: Spur, _interner: &Interner) -> Expr {
    Expr::mul(vec![int(16), Expr::Sym(slow_roll_epsilon)])
}

pub fn linearized_einstein_second_order(
    bg: &FrwBackgroundSpec,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<NamedEquation>, CosmologyError> {
    let system = crate::second_order::derive_second_order_scalar_einstein_system(bg, interner)?;
    Ok(system
        .equations
        .into_iter()
        .map(|equation| NamedEquation {
            label: equation.label,
            expr: equation.full,
            order: 2,
            sector: SectorKind::Scalar,
        })
        .collect())
}

pub fn second_order_einstein_vector(
    bg: &FrwBackgroundSpec,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    Ok(
        crate::second_order_vector_tensor::derive_second_order_vector_system(bg, interner)?
            .equations
            .into_iter()
            .map(|equation| crate::domain::NamedEquation {
                label: equation.label,
                expr: equation.full,
                order: 2,
                sector: SectorKind::Vector,
            })
            .collect(),
    )
}

pub fn second_order_einstein_tensor(
    bg: &FrwBackgroundSpec,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Result<Vec<crate::domain::NamedEquation>, crate::error::CosmologyError> {
    Ok(
        crate::second_order_vector_tensor::derive_second_order_tensor_system(bg, interner)?
            .equations
            .into_iter()
            .map(|equation| crate::domain::NamedEquation {
                label: equation.label,
                expr: equation.full,
                order: 2,
                sector: SectorKind::Tensor,
            })
            .collect(),
    )
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::svt_decompose_perturbation;

    #[test]
    fn mukhanov_sasaki_requires_conformal_time() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_cosmic(&interner);
        let eps = interner.get_or_intern("epsilon");
        let result = mukhanov_sasaki_equation(&bg, eps, &interner);
        match result {
            Err(CosmologyError::IncompatibleTimeCoordinate {
                time_coordinate,
                operation,
            }) => {
                assert_eq!(time_coordinate, TimeCoordinate::Cosmic);
                assert_eq!(operation, "Mukhanov-Sasaki equation");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn linearized_einstein_scalar_returns_named_equations() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let decomp = svt_decompose_perturbation(3, &interner).unwrap();
        let equations = linearized_einstein_scalar(&bg, &decomp, &interner);
        assert!(equations.is_ok());
        let equations = equations.unwrap();
        assert_eq!(equations.len(), 4);
        assert_eq!(equations[0].label, "00_constraint");
        assert_eq!(equations[1].label, "0i_momentum");
        assert_eq!(equations[2].label, "ij_trace");
        assert_eq!(equations[3].label, "ij_traceless");
    }

    #[test]
    fn inflation_observables_are_symbolic() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let eps = interner.get_or_intern("epsilon");
        let eta = interner.get_or_intern("eta_sr");
        let h_star = interner.get_or_intern("H_star");
        assert!(matches!(
            power_spectrum_leading(&bg, eps, h_star, &interner),
            Expr::Mul(_)
        ));
        assert!(matches!(spectral_index(eps, eta, &interner), Expr::Add(_)));
        assert!(matches!(
            tensor_to_scalar_ratio(eps, &interner),
            Expr::Mul(_)
        ));
    }

    #[test]
    fn second_order_equations_have_sources() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let decomp = svt_decompose_perturbation(3, &interner).unwrap();
        let equations = linearized_einstein_second_order(&bg, &decomp, &interner).unwrap();
        assert_eq!(equations.len(), 4);
    }
}
