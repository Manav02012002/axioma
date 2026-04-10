use crate::{covariant_derivative_covector, simplify_expr};
use ax_ir::Expr;

#[derive(Debug, Clone, PartialEq)]
pub struct KillingSystem {
    /// Covariant components ξ_a(x) treated as unknown functions.
    pub covector_components: Vec<ax_ir::Expr>,
    /// Independent symmetric equations ∇_a ξ_b + ∇_b ξ_a = 0 in lexicographic (a,b) order with a <= b.
    pub equations: Vec<ax_ir::Expr>,
    /// Matching index pairs for `equations`.
    pub slot_pairs: Vec<(usize, usize)>,
}

impl Eq for KillingSystem {}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KillingError {
    #[error(
        "killing_equations dimension mismatch: gamma dim {gamma_dim}, coords len {coords_len}"
    )]
    DimensionMismatch { gamma_dim: usize, coords_len: usize },
    #[error("killing_equations requires a consistently shaped rank-3 connection")]
    InvalidChristoffelShape,
}

fn validate_gamma_shape(gamma: &[Vec<Vec<Expr>>], n: usize) -> bool {
    gamma.len() == n
        && gamma
            .iter()
            .all(|plane| plane.len() == n && plane.iter().all(|row| row.len() == n))
}

/// Build the Killing equations ∇_a ξ_b + ∇_b ξ_a = 0 for an unknown covector field ξ_b(x).
///
/// The unknown components are represented as calls
///   xi_0(t, x, ...)
///   xi_1(t, x, ...)
///   ...
/// using the provided `field_prefix`.
pub fn killing_equations(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    field_prefix: &str,
    interner: &ax_ir::Interner,
) -> Result<KillingSystem, KillingError> {
    let gamma_dim = gamma.len();
    let coords_len = coords.len();
    if gamma_dim != coords_len {
        return Err(KillingError::DimensionMismatch {
            gamma_dim,
            coords_len,
        });
    }
    if !validate_gamma_shape(gamma, gamma_dim) {
        return Err(KillingError::InvalidChristoffelShape);
    }

    let coord_exprs = coords.iter().copied().map(Expr::Sym).collect::<Vec<_>>();
    let covector_components = (0..gamma_dim)
        .map(|i| {
            let sym = interner.get_or_intern(&format!("{field_prefix}_{i}"));
            Expr::Call(sym, coord_exprs.clone())
        })
        .collect::<Vec<_>>();

    let mut equations = Vec::with_capacity(gamma_dim * (gamma_dim + 1) / 2);
    let mut slot_pairs = Vec::with_capacity(gamma_dim * (gamma_dim + 1) / 2);
    for a in 0..gamma_dim {
        let nabla_a =
            covariant_derivative_covector(&covector_components, gamma, a, coords, interner);
        for b in a..gamma_dim {
            let nabla_b =
                covariant_derivative_covector(&covector_components, gamma, b, coords, interner);
            equations.push(simplify_expr(
                Expr::add(vec![nabla_a[b].clone(), nabla_b[a].clone()]),
                interner,
            ));
            slot_pairs.push((a, b));
        }
    }

    Ok(KillingSystem {
        covector_components,
        equations,
        slot_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::{killing_equations, KillingError};
    use crate::{christoffel_from_metric, diff_component, SymbolicMatrix};
    use ax_ir::Expr;

    fn pair_position(pairs: &[(usize, usize)], target: (usize, usize)) -> usize {
        pairs
            .iter()
            .position(|pair| *pair == target)
            .expect("slot pair present")
    }

    #[test]
    fn minkowski_killing_system_shape_is_correct() {
        let interner = ax_ir::Interner::new();
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
        let gamma = christoffel_from_metric(&g, &coords, &interner);

        let system = killing_equations(&gamma, &coords, "xi", &interner).expect("killing system");
        assert_eq!(system.covector_components.len(), 4);
        assert_eq!(system.equations.len(), 10);
        assert_eq!(system.slot_pairs.len(), 10);
    }

    #[test]
    fn minkowski_diagonal_equations_are_plain_coordinate_derivatives() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let coords = vec![t, x];
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int((-1).into()), Expr::one()]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);

        let system = killing_equations(&gamma, &coords, "xi", &interner).expect("killing system");
        let xi0 = system.covector_components[0].clone();
        let xi1 = system.covector_components[1].clone();

        let expected_00 = Expr::mul(vec![
            Expr::Int(2.into()),
            diff_component(&xi0, t, &interner),
        ]);
        let expected_01 = Expr::add(vec![
            diff_component(&xi1, t, &interner),
            diff_component(&xi0, x, &interner),
        ]);
        let expected_11 = Expr::mul(vec![
            Expr::Int(2.into()),
            diff_component(&xi1, x, &interner),
        ]);

        assert_eq!(system.slot_pairs, vec![(0, 0), (0, 1), (1, 1)]);
        assert_eq!(system.equations[0], expected_00);
        assert_eq!(system.equations[1], expected_01);
        assert_eq!(system.equations[2], expected_11);
    }

    #[test]
    fn polar_plane_killing_system_contains_connection_terms() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("r");
        let theta = interner.get_or_intern("theta");
        let coords = vec![r, theta];
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::one(),
            Expr::pow(Expr::Sym(r), Expr::Int(2.into())),
        ]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);

        let system = killing_equations(&gamma, &coords, "xi", &interner).expect("killing system");
        let idx = pair_position(&system.slot_pairs, (0, 1));
        let xi1 = system.covector_components[1].clone();
        let expected = Expr::add(vec![
            diff_component(&xi1, r, &interner),
            diff_component(&system.covector_components[0], theta, &interner),
            Expr::neg(Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new(2.into(), 1.into())),
                Expr::pow(Expr::Sym(r), Expr::Int((-1).into())),
                xi1,
            ])),
        ]);
        assert_eq!(system.equations[idx], expected);
    }

    #[test]
    fn invalid_shape_errors() {
        let interner = ax_ir::Interner::new();
        let coords = vec![interner.get_or_intern("t"), interner.get_or_intern("x")];
        let gamma = vec![vec![vec![Expr::zero(); 2]; 2], vec![vec![Expr::zero(); 2]]];
        assert_eq!(
            killing_equations(&gamma, &coords, "xi", &interner),
            Err(KillingError::InvalidChristoffelShape)
        );
    }

    #[test]
    fn dimension_mismatch_errors() {
        let interner = ax_ir::Interner::new();
        let coords = vec![interner.get_or_intern("t")];
        let gamma = vec![vec![vec![Expr::zero(); 2]; 2]; 2];
        assert_eq!(
            killing_equations(&gamma, &coords, "xi", &interner),
            Err(KillingError::DimensionMismatch {
                gamma_dim: 2,
                coords_len: 1
            })
        );
    }
}
