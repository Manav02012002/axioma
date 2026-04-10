use std::collections::BTreeMap;

use ax_forms::DiffForm;
use ax_ir::Expr;
use num_rational::BigRational;

use crate::{christoffel_from_metric, diff_component, simplify_expr, SymbolicMatrix};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CartanError {
    #[error("torsion/connection metric dimension mismatch")]
    DimensionMismatch,
    #[error("spin_connection requires a square vielbein matrix")]
    InvalidVielbein,
    #[error("spin_connection requires coordinates length {coords_len} to match vielbein dimension {dim}")]
    CoordinateMismatch { coords_len: usize, dim: usize },
    #[error("spin_connection failed because the vielbein inverse does not exist")]
    VielbeinInverseFailed,
}

fn valid_square_matrix(matrix: &SymbolicMatrix) -> bool {
    matrix.data.len() == matrix.dim && matrix.data.iter().all(|row| row.len() == matrix.dim)
}

fn valid_rank3_shape(tensor: &[Vec<Vec<Expr>>], n: usize) -> bool {
    tensor.len() == n
        && tensor
            .iter()
            .all(|plane| plane.len() == n && plane.iter().all(|row| row.len() == n))
}

fn safe_inverse(matrix: &SymbolicMatrix, interner: &ax_ir::Interner) -> Option<SymbolicMatrix> {
    let is_diagonal = (0..matrix.dim)
        .all(|row| (0..matrix.dim).all(|col| row == col || matrix.data[row][col] == Expr::zero()));
    if is_diagonal {
        let mut inverse = SymbolicMatrix::new(matrix.dim);
        for i in 0..matrix.dim {
            inverse.data[i][i] = simplify_expr(
                Expr::pow(matrix.data[i][i].clone(), Expr::Int((-1).into())),
                interner,
            );
        }
        return Some(inverse);
    }
    ax_linalg::inverse(&matrix.data, interner).map(|data| SymbolicMatrix {
        dim: matrix.dim,
        data: data
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| simplify_expr(cell, interner))
                    .collect()
            })
            .collect(),
    })
}

fn one_half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn add_forms(lhs: &DiffForm, rhs: &DiffForm, interner: &ax_ir::Interner) -> DiffForm {
    debug_assert_eq!(lhs.degree, rhs.degree);
    debug_assert_eq!(lhs.dim, rhs.dim);
    let mut components = lhs.components.clone();
    for (basis, value) in &rhs.components {
        let updated = simplify_expr(
            Expr::add(vec![
                components.get(basis).cloned().unwrap_or_else(Expr::zero),
                value.clone(),
            ]),
            interner,
        );
        if updated == Expr::zero() {
            components.remove(basis);
        } else {
            components.insert(basis.clone(), updated);
        }
    }
    DiffForm {
        degree: lhs.degree,
        dim: lhs.dim,
        components,
    }
}

fn zero_form_of_degree(dim: usize, degree: usize) -> DiffForm {
    DiffForm {
        degree,
        dim,
        components: BTreeMap::new(),
    }
}

fn one_form_from_components(components: &[Expr]) -> DiffForm {
    DiffForm {
        degree: 1,
        dim: components.len(),
        components: components
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != Expr::zero())
            .map(|(index, value)| (vec![index], value.clone()))
            .collect(),
    }
}

pub fn contorsion_tensor(
    torsion: &[Vec<Vec<ax_ir::Expr>>],
    g: &crate::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<ax_ir::Expr>>>, CartanError> {
    if !valid_square_matrix(g) || !valid_rank3_shape(torsion, g.dim) {
        return Err(CartanError::DimensionMismatch);
    }
    let Some(g_inv) = safe_inverse(g, interner) else {
        return Err(CartanError::DimensionMismatch);
    };
    let n = g.dim;

    let mut lowered_first = vec![vec![vec![Expr::zero(); n]; n]; n];
    for b in 0..n {
        for e in 0..n {
            for c in 0..n {
                let terms = (0..n)
                    .filter_map(|d| {
                        if g.get(b, d) == &Expr::zero() || torsion[d][e][c] == Expr::zero() {
                            None
                        } else {
                            Some(Expr::mul(vec![
                                g.get(b, d).clone(),
                                torsion[d][e][c].clone(),
                            ]))
                        }
                    })
                    .collect::<Vec<_>>();
                lowered_first[b][e][c] = simplify_expr(Expr::add(terms), interner);
            }
        }
    }

    let mut contorsion = vec![vec![vec![Expr::zero(); n]; n]; n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let tb_ac = (0..n)
                    .filter_map(|e| {
                        if g_inv.get(a, e) == &Expr::zero()
                            || lowered_first[b][e][c] == Expr::zero()
                        {
                            None
                        } else {
                            Some(Expr::mul(vec![
                                g_inv.get(a, e).clone(),
                                lowered_first[b][e][c].clone(),
                            ]))
                        }
                    })
                    .collect::<Vec<_>>();
                let tc_ab = (0..n)
                    .filter_map(|e| {
                        if g_inv.get(a, e) == &Expr::zero()
                            || lowered_first[c][e][b] == Expr::zero()
                        {
                            None
                        } else {
                            Some(Expr::mul(vec![
                                g_inv.get(a, e).clone(),
                                lowered_first[c][e][b].clone(),
                            ]))
                        }
                    })
                    .collect::<Vec<_>>();
                contorsion[a][b][c] = simplify_expr(
                    Expr::mul(vec![
                        one_half(),
                        Expr::add(vec![
                            torsion[a][b][c].clone(),
                            Expr::neg(Expr::add(tb_ac)),
                            Expr::neg(Expr::add(tc_ab)),
                        ]),
                    ]),
                    interner,
                );
            }
        }
    }
    Ok(contorsion)
}

pub fn connection_with_torsion(
    christoffel: &[Vec<Vec<ax_ir::Expr>>],
    contorsion: &[Vec<Vec<ax_ir::Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<ax_ir::Expr>>>, CartanError> {
    let n = christoffel.len();
    if !valid_rank3_shape(christoffel, n) || !valid_rank3_shape(contorsion, n) {
        return Err(CartanError::DimensionMismatch);
    }
    let mut connection = vec![vec![vec![Expr::zero(); n]; n]; n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                connection[a][b][c] = simplify_expr(
                    Expr::add(vec![
                        christoffel[a][b][c].clone(),
                        contorsion[a][b][c].clone(),
                    ]),
                    interner,
                );
            }
        }
    }
    Ok(connection)
}

pub fn spin_connection(
    vielbein: &crate::SymbolicMatrix,
    g: &crate::SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Vec<ax_ir::Expr>>>, CartanError> {
    if !valid_square_matrix(vielbein) {
        return Err(CartanError::InvalidVielbein);
    }
    if !valid_square_matrix(g) || g.dim != vielbein.dim {
        return Err(CartanError::DimensionMismatch);
    }
    if coords.len() != vielbein.dim {
        return Err(CartanError::CoordinateMismatch {
            coords_len: coords.len(),
            dim: vielbein.dim,
        });
    }

    let Some(vielbein_inv) = safe_inverse(vielbein, interner) else {
        return Err(CartanError::VielbeinInverseFailed);
    };
    let gamma = christoffel_from_metric(g, coords, interner);
    let n = vielbein.dim;
    let mut omega = vec![vec![vec![Expr::zero(); n]; n]; n];

    for mu in 0..n {
        for a in 0..n {
            for b in 0..n {
                let mut nu_terms = Vec::new();
                for nu in 0..n {
                    if vielbein.get(a, nu) == &Expr::zero() {
                        continue;
                    }
                    let derivative = diff_component(vielbein_inv.get(nu, b), coords[mu], interner);
                    let gamma_terms = (0..n)
                        .filter_map(|rho| {
                            if gamma[nu][mu][rho] == Expr::zero()
                                || vielbein_inv.get(rho, b) == &Expr::zero()
                            {
                                None
                            } else {
                                Some(Expr::mul(vec![
                                    gamma[nu][mu][rho].clone(),
                                    vielbein_inv.get(rho, b).clone(),
                                ]))
                            }
                        })
                        .collect::<Vec<_>>();
                    let inner = simplify_expr(
                        Expr::add(std::iter::once(derivative).chain(gamma_terms).collect()),
                        interner,
                    );
                    if inner == Expr::zero() {
                        continue;
                    }
                    nu_terms.push(Expr::mul(vec![vielbein.get(a, nu).clone(), inner]));
                }
                omega[mu][a][b] = simplify_expr(Expr::add(nu_terms), interner);
            }
        }
    }

    Ok(omega)
}

pub fn first_cartan_structure(
    vielbein: &crate::SymbolicMatrix,
    spin_connection: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<ax_forms::DiffForm>, CartanError> {
    if !valid_square_matrix(vielbein) {
        return Err(CartanError::InvalidVielbein);
    }
    if coords.len() != vielbein.dim {
        return Err(CartanError::CoordinateMismatch {
            coords_len: coords.len(),
            dim: vielbein.dim,
        });
    }
    if !valid_rank3_shape(spin_connection, vielbein.dim) {
        return Err(CartanError::DimensionMismatch);
    }

    let n = vielbein.dim;
    let mut torsion_forms = Vec::with_capacity(n);
    for a in 0..n {
        // Convention: row `a` of the vielbein matrix is the coframe one-form e^a = e^a_mu dx^mu.
        let e_a = one_form_from_components(&vielbein.data[a]);
        let mut torsion = ax_forms::exterior_derivative(&e_a, coords, interner);
        for b in 0..n {
            let omega_ab_components = (0..n)
                .map(|mu| spin_connection[mu][a][b].clone())
                .collect::<Vec<_>>();
            let omega_ab = one_form_from_components(&omega_ab_components);
            torsion = add_forms(
                &torsion,
                &ax_forms::wedge(&omega_ab, &e_a_from_index(vielbein, b), interner),
                interner,
            );
        }
        torsion_forms.push(torsion);
    }
    Ok(torsion_forms)
}

fn e_a_from_index(vielbein: &SymbolicMatrix, a: usize) -> DiffForm {
    one_form_from_components(&vielbein.data[a])
}

pub fn second_cartan_structure(
    spin_connection: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<ax_forms::DiffForm>>, CartanError> {
    let n = coords.len();
    if !valid_rank3_shape(spin_connection, n) {
        return Err(CartanError::DimensionMismatch);
    }

    let mut curvature = vec![vec![zero_form_of_degree(n, 2); n]; n];
    for a in 0..n {
        for b in 0..n {
            let omega_ab = one_form_from_components(
                &(0..n)
                    .map(|mu| spin_connection[mu][a][b].clone())
                    .collect::<Vec<_>>(),
            );
            let mut curvature_ab = ax_forms::exterior_derivative(&omega_ab, coords, interner);
            for c in 0..n {
                let omega_ac = one_form_from_components(
                    &(0..n)
                        .map(|mu| spin_connection[mu][a][c].clone())
                        .collect::<Vec<_>>(),
                );
                let omega_cb = one_form_from_components(
                    &(0..n)
                        .map(|mu| spin_connection[mu][c][b].clone())
                        .collect::<Vec<_>>(),
                );
                curvature_ab = add_forms(
                    &curvature_ab,
                    &ax_forms::wedge(&omega_ac, &omega_cb, interner),
                    interner,
                );
            }
            curvature[a][b] = curvature_ab;
        }
    }
    Ok(curvature)
}

#[cfg(test)]
mod tests {
    use super::{
        connection_with_torsion, contorsion_tensor, first_cartan_structure,
        second_cartan_structure, spin_connection, CartanError,
    };
    use crate::{simplify_expr, SymbolicMatrix};
    use ax_ir::{Expr, Interner};

    fn zero_rank3(n: usize) -> Vec<Vec<Vec<Expr>>> {
        vec![vec![vec![Expr::zero(); n]; n]; n]
    }

    #[test]
    fn zero_torsion_gives_zero_contorsion() {
        let interner = Interner::new();
        let torsion = zero_rank3(2);
        let g = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        assert_eq!(
            contorsion_tensor(&torsion, &g, &interner).expect("contorsion"),
            zero_rank3(2)
        );
    }

    #[test]
    fn torsionful_connection_is_sum_of_lc_and_contorsion() {
        let interner = Interner::new();
        let gamma = zero_rank3(2);
        let mut contorsion = zero_rank3(2);
        contorsion[0][1][1] = Expr::one();
        let connection = connection_with_torsion(&gamma, &contorsion, &interner).expect("sum");
        assert_eq!(connection, contorsion);
    }

    #[test]
    fn flat_cartesian_vielbein_has_zero_spin_connection() {
        let interner = Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let e = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        let g = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        let omega = spin_connection(&e, &g, &[x, y], &interner).expect("spin connection");
        assert_eq!(omega, zero_rank3(2));
    }

    #[test]
    fn polar_plane_vielbein_has_expected_nonzero_spin_connection_component() {
        let interner = Interner::new();
        let r = interner.get_or_intern("r");
        let theta = interner.get_or_intern("theta");
        let e = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::Sym(r)]);
        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::one(),
            Expr::pow(Expr::Sym(r), Expr::Int(2.into())),
        ]);
        let omega = spin_connection(&e, &g, &[r, theta], &interner).expect("spin connection");
        assert_eq!(
            simplify_expr(omega[1][0][1].clone(), &interner),
            Expr::Int((-1).into())
        );
    }

    #[test]
    fn first_cartan_structure_vanishes_for_flat_cartesian_vielbein() {
        let interner = Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let e = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        let omega = zero_rank3(2);
        let torsion = first_cartan_structure(&e, &omega, &[x, y], &interner).expect("torsion");
        assert!(torsion.iter().all(|form| form.components.is_empty()));
    }

    #[test]
    fn second_cartan_structure_vanishes_for_flat_cartesian_vielbein() {
        let interner = Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let omega = zero_rank3(2);
        let curvature = second_cartan_structure(&omega, &[x, y], &interner).expect("curvature");
        assert!(curvature
            .iter()
            .flatten()
            .all(|form| form.components.is_empty()));
    }

    #[test]
    fn shape_and_coordinate_errors() {
        let interner = Interner::new();
        let x = interner.get_or_intern("x");
        let e = SymbolicMatrix {
            dim: 2,
            data: vec![vec![Expr::one()]],
        };
        let g = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        assert_eq!(
            spin_connection(&e, &g, &[x], &interner),
            Err(CartanError::InvalidVielbein)
        );
        let e_good = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        assert_eq!(
            spin_connection(&e_good, &g, &[x], &interner),
            Err(CartanError::CoordinateMismatch {
                coords_len: 1,
                dim: 2
            })
        );
        assert_eq!(
            contorsion_tensor(&zero_rank3(3), &g, &interner),
            Err(CartanError::DimensionMismatch)
        );
        assert_eq!(
            connection_with_torsion(&zero_rank3(2), &zero_rank3(3), &interner),
            Err(CartanError::DimensionMismatch)
        );
    }
}
