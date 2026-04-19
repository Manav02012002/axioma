use crate::cosmology::require_conformal_time;
use crate::domain::{NamedEquation, SectorKind};
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiFieldSymbols {
    pub field_count: usize,
    pub background_fields: Vec<lasso::Spur>,
    pub background_field_primes: Vec<lasso::Spur>,
    pub perturbations: Vec<lasso::Spur>,
    pub potentials_first: Vec<lasso::Spur>,
    pub potentials_second: Vec<Vec<lasso::Spur>>,
    pub curvature_mode: lasso::Spur,
    pub entropy_modes: Vec<lasso::Spur>,
    pub turn_rate: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdiabaticEntropyBasis {
    pub adiabatic_unit: Vec<ax_ir::Expr>,
    pub entropy_basis: Vec<Vec<ax_ir::Expr>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiFieldMassData {
    pub mass_matrix: ax_linalg::SymbolicMatrix,
    pub effective_entropy_mass_matrix: ax_linalg::SymbolicMatrix,
    pub turn_rate: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiFieldEquationSet {
    pub equations: Vec<crate::domain::NamedEquation>,
}

pub fn standard_multifield_symbols(
    field_count: usize,
    interner: &ax_ir::Interner,
) -> Result<MultiFieldSymbols, crate::error::CosmologyError> {
    if field_count < 2 {
        return Err(CosmologyError::InvalidFieldCount { got: field_count });
    }
    let background_fields = (1..=field_count)
        .map(|idx| interner.get_or_intern(&format!("phi0_{idx}")))
        .collect::<Vec<_>>();
    let background_field_primes = (1..=field_count)
        .map(|idx| interner.get_or_intern(&format!("phi0_{idx}_prime")))
        .collect::<Vec<_>>();
    let perturbations = (1..=field_count)
        .map(|idx| interner.get_or_intern(&format!("delta_phi_{idx}")))
        .collect::<Vec<_>>();
    let potentials_first = (1..=field_count)
        .map(|idx| interner.get_or_intern(&format!("V_{idx}")))
        .collect::<Vec<_>>();
    let potentials_second = (1..=field_count)
        .map(|row| {
            (1..=field_count)
                .map(|col| interner.get_or_intern(&format!("V_{row}{col}")))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let entropy_modes = (1..field_count)
        .map(|idx| interner.get_or_intern(&format!("S_{idx}")))
        .collect::<Vec<_>>();
    Ok(MultiFieldSymbols {
        field_count,
        background_fields,
        background_field_primes,
        perturbations,
        potentials_first,
        potentials_second,
        curvature_mode: interner.get_or_intern("R"),
        entropy_modes,
        turn_rate: interner.get_or_intern("Omega"),
    })
}

pub fn adiabatic_entropy_basis(
    symbols: &MultiFieldSymbols,
    interner: &ax_ir::Interner,
) -> Result<AdiabaticEntropyBasis, crate::error::CosmologyError> {
    let primes = symbols
        .background_field_primes
        .iter()
        .map(|sym| Expr::Sym(*sym))
        .collect::<Vec<_>>();
    let norm = sqrt(sum_of_squares(&primes, interner), interner);
    let adiabatic_unit = primes
        .iter()
        .map(|prime| Expr::mul(vec![prime.clone(), Expr::pow(norm.clone(), int(-1))]))
        .collect::<Vec<_>>();

    let entropy_basis = match symbols.field_count {
        2 => {
            let p1 = Expr::Sym(symbols.background_field_primes[0]);
            let p2 = Expr::Sym(symbols.background_field_primes[1]);
            vec![vec![
                Expr::neg(Expr::mul(vec![p2, Expr::pow(norm.clone(), int(-1))])),
                Expr::mul(vec![p1, Expr::pow(norm, int(-1))]),
            ]]
        }
        3 => {
            let p1 = Expr::Sym(symbols.background_field_primes[0]);
            let p2 = Expr::Sym(symbols.background_field_primes[1]);
            let p3 = Expr::Sym(symbols.background_field_primes[2]);
            let norm12 = sqrt(Expr::add(vec![sq(p1.clone()), sq(p2.clone())]), interner);
            vec![
                vec![
                    Expr::mul(vec![p2.clone(), Expr::pow(norm12.clone(), int(-1))]),
                    Expr::neg(Expr::mul(vec![
                        p1.clone(),
                        Expr::pow(norm12.clone(), int(-1)),
                    ])),
                    Expr::zero(),
                ],
                vec![
                    Expr::mul(vec![
                        p1.clone(),
                        p3.clone(),
                        Expr::pow(norm.clone(), int(-1)),
                        Expr::pow(norm12.clone(), int(-1)),
                    ]),
                    Expr::mul(vec![
                        p2.clone(),
                        p3,
                        Expr::pow(norm.clone(), int(-1)),
                        Expr::pow(norm12.clone(), int(-1)),
                    ]),
                    Expr::neg(Expr::mul(vec![
                        Expr::add(vec![sq(p1), sq(p2)]),
                        Expr::pow(norm, int(-1)),
                        Expr::pow(norm12, int(-1)),
                    ])),
                ],
            ]
        }
        _ => {
            return Err(CosmologyError::AdiabaticEntropyRotationFailure {
                operation: "adiabatic_entropy_basis".to_string(),
            });
        }
    };

    Ok(AdiabaticEntropyBasis {
        adiabatic_unit,
        entropy_basis,
    })
}

pub fn multifield_mass_data(
    bg: &crate::domain::FrwBackgroundSpec,
    symbols: &MultiFieldSymbols,
    interner: &ax_ir::Interner,
) -> Result<MultiFieldMassData, crate::error::CosmologyError> {
    require_conformal_time(bg, "multifield_mass_data")?;
    let basis = adiabatic_entropy_basis(symbols, interner)?;
    let mass_matrix_data = symbols
        .potentials_second
        .iter()
        .map(|row| row.iter().map(|sym| Expr::Sym(*sym)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let entropy_matrix = basis.entropy_basis;
    let projected = ax_linalg::mat_mul(
        &ax_linalg::mat_mul(&entropy_matrix, &mass_matrix_data, interner),
        &ax_linalg::transpose(&entropy_matrix),
        interner,
    );
    Ok(MultiFieldMassData {
        mass_matrix: ax_linalg::SymbolicMatrix::from_data(mass_matrix_data),
        effective_entropy_mass_matrix: ax_linalg::SymbolicMatrix::from_data(projected),
        turn_rate: Expr::Sym(symbols.turn_rate),
    })
}

pub fn derive_multifield_curvature_entropy_equations(
    bg: &crate::domain::FrwBackgroundSpec,
    symbols: &MultiFieldSymbols,
    interner: &ax_ir::Interner,
) -> Result<MultiFieldEquationSet, crate::error::CosmologyError> {
    require_conformal_time(bg, "derive_multifield_curvature_entropy_equations")?;
    let _basis = adiabatic_entropy_basis(symbols, interner)?;
    let mass_data = multifield_mass_data(bg, symbols, interner)?;
    let eta = bg.conformal_time;
    let hubble = Expr::Sym(bg.conformal_hubble);
    let k_sq = sq(Expr::Sym(interner.get_or_intern("k")));
    let curvature = Expr::Sym(symbols.curvature_mode);
    let curvature_equation = Expr::add(vec![
        diff(diff(curvature.clone(), eta, interner), eta, interner),
        Expr::mul(vec![
            int(2),
            hubble.clone(),
            diff(curvature.clone(), eta, interner),
        ]),
        Expr::mul(vec![k_sq.clone(), curvature.clone()]),
        Expr::mul(vec![
            mass_data.turn_rate.clone(),
            Expr::Sym(symbols.entropy_modes[0]),
        ]),
    ]);

    let mut equations = vec![NamedEquation {
        label: "multifield_curvature".to_string(),
        expr: curvature_equation,
        order: 1,
        sector: SectorKind::Scalar,
    }];

    for (idx, entropy_mode) in symbols.entropy_modes.iter().enumerate() {
        let entropy = Expr::Sym(*entropy_mode);
        let mass_terms = (0..symbols.entropy_modes.len())
            .map(|col| {
                Expr::mul(vec![
                    mass_data
                        .effective_entropy_mass_matrix
                        .get(idx, col)
                        .clone(),
                    Expr::Sym(symbols.entropy_modes[col]),
                ])
            })
            .collect::<Vec<_>>();
        let expr = Expr::add(vec![
            diff(diff(entropy.clone(), eta, interner), eta, interner),
            Expr::mul(vec![
                int(2),
                hubble.clone(),
                diff(entropy.clone(), eta, interner),
            ]),
            Expr::mul(vec![k_sq.clone(), entropy]),
            Expr::add(mass_terms),
            Expr::mul(vec![mass_data.turn_rate.clone(), curvature.clone()]),
        ]);
        equations.push(NamedEquation {
            label: format!("multifield_entropy_{}", idx + 1),
            expr,
            order: 1,
            sector: SectorKind::Scalar,
        });
    }

    Ok(MultiFieldEquationSet { equations })
}

fn diff(expr: Expr, var: lasso::Spur, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, Expr::Sym(var)])
}

fn sqrt(expr: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("sqrt"), vec![expr])
}

fn sq(expr: Expr) -> Expr {
    Expr::pow(expr, int(2))
}

fn sum_of_squares(values: &[Expr], _interner: &Interner) -> Expr {
    Expr::add(values.iter().cloned().map(sq).collect())
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrwBackgroundSpec;

    #[test]
    fn standard_multifield_symbols_rejects_field_count_below_two() {
        let interner = Interner::new();
        let result = standard_multifield_symbols(1, &interner);
        match result {
            Err(CosmologyError::InvalidFieldCount { got }) => assert_eq!(got, 1),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn standard_multifield_symbols_two_fields_use_expected_names() {
        let interner = Interner::new();
        let symbols = standard_multifield_symbols(2, &interner).unwrap();
        assert_eq!(interner.resolve(symbols.background_fields[0]), "phi0_1");
        assert_eq!(interner.resolve(symbols.background_fields[1]), "phi0_2");
        assert_eq!(
            interner.resolve(symbols.background_field_primes[0]),
            "phi0_1_prime"
        );
        assert_eq!(interner.resolve(symbols.perturbations[1]), "delta_phi_2");
        assert_eq!(interner.resolve(symbols.potentials_first[1]), "V_2");
        assert_eq!(interner.resolve(symbols.potentials_second[0][1]), "V_12");
        assert_eq!(interner.resolve(symbols.curvature_mode), "R");
        assert_eq!(interner.resolve(symbols.entropy_modes[0]), "S_1");
        assert_eq!(interner.resolve(symbols.turn_rate), "Omega");
    }

    #[test]
    fn adiabatic_entropy_basis_two_fields_returns_one_entropy_direction() {
        let interner = Interner::new();
        let symbols = standard_multifield_symbols(2, &interner).unwrap();
        let basis = adiabatic_entropy_basis(&symbols, &interner).unwrap();
        assert_eq!(basis.adiabatic_unit.len(), 2);
        assert_eq!(basis.entropy_basis.len(), 1);
        assert_eq!(basis.entropy_basis[0].len(), 2);
    }

    #[test]
    fn adiabatic_entropy_basis_three_fields_returns_two_entropy_directions() {
        let interner = Interner::new();
        let symbols = standard_multifield_symbols(3, &interner).unwrap();
        let basis = adiabatic_entropy_basis(&symbols, &interner).unwrap();
        assert_eq!(basis.adiabatic_unit.len(), 3);
        assert_eq!(basis.entropy_basis.len(), 2);
        assert_eq!(basis.entropy_basis[0].len(), 3);
        assert_eq!(basis.entropy_basis[1].len(), 3);
    }

    #[test]
    fn adiabatic_entropy_basis_rejects_more_than_three_fields() {
        let interner = Interner::new();
        let symbols = standard_multifield_symbols(4, &interner).unwrap();
        let result = adiabatic_entropy_basis(&symbols, &interner);
        match result {
            Err(CosmologyError::AdiabaticEntropyRotationFailure { operation }) => {
                assert_eq!(operation, "adiabatic_entropy_basis");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn multifield_mass_data_builds_square_mass_matrix() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_multifield_symbols(3, &interner).unwrap();
        let data = multifield_mass_data(&bg, &symbols, &interner).unwrap();
        assert_eq!(data.mass_matrix.dim, 3);
        assert_eq!(data.mass_matrix.data.len(), 3);
        assert_eq!(data.effective_entropy_mass_matrix.dim, 2);
    }

    #[test]
    fn derive_multifield_curvature_entropy_equations_returns_expected_labels_for_two_fields() {
        let interner = Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
        let symbols = standard_multifield_symbols(2, &interner).unwrap();
        let equations =
            derive_multifield_curvature_entropy_equations(&bg, &symbols, &interner).unwrap();
        let labels = equations
            .equations
            .iter()
            .map(|eq| eq.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["multifield_curvature", "multifield_entropy_1"]);
    }
}
