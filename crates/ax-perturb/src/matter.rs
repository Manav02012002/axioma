use crate::cosmology::require_conformal_time;
use crate::domain::{FrwBackgroundSpec, NamedEquation, SectorKind};
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerfectFluidSymbols {
    pub rho: lasso::Spur,
    pub pressure: lasso::Spur,
    pub delta_rho: lasso::Spur,
    pub delta_pressure: lasso::Spur,
    pub velocity_potential: lasso::Spur,
    pub anisotropic_stress: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalScalarSymbols {
    pub background_field: lasso::Spur,
    pub background_field_prime: lasso::Spur,
    pub perturbation: lasso::Spur,
    pub potential: lasso::Spur,
    pub potential_prime: lasso::Spur,
    pub z: lasso::Spur,
    pub v: lasso::Spur,
    pub sound_speed: lasso::Spur,
    pub slow_roll_epsilon: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatterEquationSet {
    pub equations: Vec<crate::domain::NamedEquation>,
}

pub fn standard_perfect_fluid_symbols(interner: &ax_ir::Interner) -> PerfectFluidSymbols {
    PerfectFluidSymbols {
        rho: interner.get_or_intern("rho"),
        pressure: interner.get_or_intern("P"),
        delta_rho: interner.get_or_intern("delta_rho"),
        delta_pressure: interner.get_or_intern("delta_P"),
        velocity_potential: interner.get_or_intern("v"),
        anisotropic_stress: interner.get_or_intern("Pi"),
    }
}

pub fn standard_canonical_scalar_symbols(interner: &ax_ir::Interner) -> CanonicalScalarSymbols {
    CanonicalScalarSymbols {
        background_field: interner.get_or_intern("phi0"),
        background_field_prime: interner.get_or_intern("phi0_prime"),
        perturbation: interner.get_or_intern("delta_phi"),
        potential: interner.get_or_intern("V"),
        potential_prime: interner.get_or_intern("V_phi"),
        z: interner.get_or_intern("z"),
        v: interner.get_or_intern("v"),
        sound_speed: interner.get_or_intern("c_s"),
        slow_roll_epsilon: interner.get_or_intern("epsilon"),
    }
}

pub fn perfect_fluid_linear_conservation_equations_newtonian(
    bg: &FrwBackgroundSpec,
    interner: &Interner,
) -> Result<MatterEquationSet, CosmologyError> {
    require_conformal_time(bg, "perfect fluid linear conservation equations")?;

    let symbols = standard_perfect_fluid_symbols(interner);
    let phi = Expr::Sym(interner.get_or_intern("Phi"));
    let psi = Expr::Sym(interner.get_or_intern("Psi"));
    let eta = Expr::Sym(bg.conformal_time);
    let h = Expr::Sym(bg.conformal_hubble);
    let rho = Expr::Sym(symbols.rho);
    let pressure = Expr::Sym(symbols.pressure);
    let delta_rho = Expr::Sym(symbols.delta_rho);
    let delta_pressure = Expr::Sym(symbols.delta_pressure);
    let velocity = Expr::Sym(symbols.velocity_potential);
    let anisotropic_stress = Expr::Sym(symbols.anisotropic_stress);
    let rho_plus_pressure = Expr::add(vec![rho, pressure]);

    let continuity = Expr::add(vec![
        diff(delta_rho.clone(), eta.clone(), interner),
        Expr::mul(vec![
            int(3),
            h.clone(),
            Expr::add(vec![delta_rho, delta_pressure.clone()]),
        ]),
        Expr::neg(Expr::mul(vec![
            int(3),
            rho_plus_pressure.clone(),
            diff(psi, eta.clone(), interner),
        ])),
        Expr::mul(vec![
            rho_plus_pressure.clone(),
            laplacian(velocity.clone(), interner),
        ]),
    ]);
    let euler = Expr::add(vec![
        diff(velocity, eta, interner),
        Expr::mul(vec![h, Expr::Sym(symbols.velocity_potential)]),
        phi,
        Expr::mul(vec![
            Expr::Sym(symbols.delta_pressure),
            Expr::pow(rho_plus_pressure, int(-1)),
        ]),
        anisotropic_stress,
    ]);

    Ok(MatterEquationSet {
        equations: vec![
            NamedEquation {
                label: "fluid_continuity".to_string(),
                expr: continuity,
                order: 1,
                sector: SectorKind::Scalar,
            },
            NamedEquation {
                label: "fluid_euler".to_string(),
                expr: euler,
                order: 1,
                sector: SectorKind::Scalar,
            },
        ],
    })
}

fn diff(expr: Expr, var: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, var])
}

fn laplacian(expr: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("laplacian"), vec![expr])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_perfect_fluid_symbols_use_expected_names() {
        let interner = Interner::new();
        let symbols = standard_perfect_fluid_symbols(&interner);

        assert_eq!(interner.resolve(symbols.rho), "rho");
        assert_eq!(interner.resolve(symbols.pressure), "P");
        assert_eq!(interner.resolve(symbols.delta_rho), "delta_rho");
        assert_eq!(interner.resolve(symbols.delta_pressure), "delta_P");
        assert_eq!(interner.resolve(symbols.velocity_potential), "v");
        assert_eq!(interner.resolve(symbols.anisotropic_stress), "Pi");
    }

    #[test]
    fn perfect_fluid_linear_conservation_returns_two_named_equations() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);

        let equations =
            perfect_fluid_linear_conservation_equations_newtonian(&bg, &interner).unwrap();

        assert_eq!(equations.equations.len(), 2);
        assert_eq!(equations.equations[0].label, "fluid_continuity");
        assert_eq!(equations.equations[1].label, "fluid_euler");
    }
}
