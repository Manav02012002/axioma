use crate::cosmology::require_conformal_time;
use crate::domain::{HarmonicBasisKind, NamedEquation, SectorKind, SpatialCurvature};
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use lasso::Spur;
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarHarmonicSpec {
    pub mode_symbol: lasso::Spur,
    pub wave_symbol: lasso::Spur,
    pub curvature: crate::domain::SpatialCurvature,
    pub basis: crate::domain::HarmonicBasisKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorHarmonicSpec {
    pub mode_x: lasso::Spur,
    pub mode_y: lasso::Spur,
    pub mode_z: lasso::Spur,
    pub wave_symbol: lasso::Spur,
    pub curvature: crate::domain::SpatialCurvature,
    pub basis: crate::domain::HarmonicBasisKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorHarmonicSpec {
    pub plus_mode: lasso::Spur,
    pub cross_mode: lasso::Spur,
    pub wave_symbol: lasso::Spur,
    pub curvature: crate::domain::SpatialCurvature,
    pub basis: crate::domain::HarmonicBasisKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicProjectionRule {
    pub label: String,
    pub before: ax_ir::Expr,
    pub after: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedEquationSet {
    pub equations: Vec<crate::domain::NamedEquation>,
    pub rules_applied: Vec<HarmonicProjectionRule>,
}

pub fn standard_scalar_harmonic_spec(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> ScalarHarmonicSpec {
    ScalarHarmonicSpec {
        mode_symbol: interner.get_or_intern("Q"),
        wave_symbol: interner.get_or_intern("k"),
        curvature,
        basis: match curvature {
            SpatialCurvature::Flat => HarmonicBasisKind::FourierFlat,
            SpatialCurvature::Closed | SpatialCurvature::Open => HarmonicBasisKind::ScalarHarmonics,
        },
    }
}

pub fn standard_vector_harmonic_spec(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> VectorHarmonicSpec {
    VectorHarmonicSpec {
        mode_x: interner.get_or_intern("QV_x"),
        mode_y: interner.get_or_intern("QV_y"),
        mode_z: interner.get_or_intern("QV_z"),
        wave_symbol: interner.get_or_intern("k"),
        curvature,
        basis: match curvature {
            SpatialCurvature::Flat => HarmonicBasisKind::FourierFlat,
            SpatialCurvature::Closed | SpatialCurvature::Open => HarmonicBasisKind::VectorHarmonics,
        },
    }
}

pub fn standard_tensor_harmonic_spec(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> TensorHarmonicSpec {
    TensorHarmonicSpec {
        plus_mode: interner.get_or_intern("GT_plus"),
        cross_mode: interner.get_or_intern("GT_cross"),
        wave_symbol: interner.get_or_intern("k"),
        curvature,
        basis: match curvature {
            SpatialCurvature::Flat => HarmonicBasisKind::FourierFlat,
            SpatialCurvature::Closed | SpatialCurvature::Open => HarmonicBasisKind::TensorHarmonics,
        },
    }
}

pub fn scalar_laplacian_eigenvalue(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, crate::error::CosmologyError> {
    let k = Expr::Sym(interner.get_or_intern("k"));
    let k_sq = Expr::pow(k, int(2));
    let kappa = Expr::Sym(interner.get_or_intern("K"));
    Ok(match curvature {
        SpatialCurvature::Flat => Expr::neg(k_sq),
        SpatialCurvature::Closed => Expr::neg(Expr::add(vec![
            k_sq,
            Expr::neg(Expr::mul(vec![int(3), kappa])),
        ])),
        SpatialCurvature::Open => Expr::neg(Expr::add(vec![k_sq, Expr::mul(vec![int(3), kappa])])),
    })
}

pub fn vector_laplacian_eigenvalue(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, crate::error::CosmologyError> {
    let k = Expr::Sym(interner.get_or_intern("k"));
    let k_sq = Expr::pow(k, int(2));
    let kappa = Expr::Sym(interner.get_or_intern("K"));
    Ok(match curvature {
        SpatialCurvature::Flat => Expr::neg(k_sq),
        SpatialCurvature::Closed => Expr::neg(Expr::add(vec![
            k_sq,
            Expr::neg(Expr::mul(vec![int(2), kappa])),
        ])),
        SpatialCurvature::Open => Expr::neg(Expr::add(vec![k_sq, Expr::mul(vec![int(2), kappa])])),
    })
}

pub fn tensor_laplacian_eigenvalue(
    curvature: crate::domain::SpatialCurvature,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, crate::error::CosmologyError> {
    let k = Expr::Sym(interner.get_or_intern("k"));
    let k_sq = Expr::pow(k, int(2));
    let kappa = Expr::Sym(interner.get_or_intern("K"));
    Ok(match curvature {
        SpatialCurvature::Flat => Expr::neg(k_sq),
        SpatialCurvature::Closed => Expr::neg(Expr::add(vec![
            k_sq,
            Expr::neg(Expr::mul(vec![int(2), kappa])),
        ])),
        SpatialCurvature::Open => Expr::neg(Expr::add(vec![k_sq, Expr::mul(vec![int(2), kappa])])),
    })
}

pub fn project_scalar_equations_to_harmonic_space(
    equations: &[crate::domain::NamedEquation],
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ProjectedEquationSet, crate::error::CosmologyError> {
    require_conformal_time(bg, "project_scalar_equations_to_harmonic_space")?;
    let eigenvalue = scalar_laplacian_eigenvalue(bg.spatial_curvature, interner)?;
    let mode = standard_scalar_harmonic_spec(bg.spatial_curvature, interner).mode_symbol;
    project_equations_to_harmonic_space(
        equations,
        &eigenvalue,
        Expr::Sym(mode),
        spatial_coords(interner),
        "project_scalar_equations_to_harmonic_space",
        "scalar",
        interner,
    )
}

pub fn project_vector_equations_to_harmonic_space(
    equations: &[crate::domain::NamedEquation],
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ProjectedEquationSet, crate::error::CosmologyError> {
    require_conformal_time(bg, "project_vector_equations_to_harmonic_space")?;
    let eigenvalue = vector_laplacian_eigenvalue(bg.spatial_curvature, interner)?;
    let spec = standard_vector_harmonic_spec(bg.spatial_curvature, interner);
    project_equations_to_harmonic_space(
        equations,
        &eigenvalue,
        Expr::List(vec![
            Expr::Sym(spec.mode_x),
            Expr::Sym(spec.mode_y),
            Expr::Sym(spec.mode_z),
        ]),
        spatial_coords(interner),
        "project_vector_equations_to_harmonic_space",
        "vector",
        interner,
    )
}

pub fn project_tensor_equations_to_harmonic_space(
    equations: &[crate::domain::NamedEquation],
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ProjectedEquationSet, crate::error::CosmologyError> {
    require_conformal_time(bg, "project_tensor_equations_to_harmonic_space")?;
    let eigenvalue = tensor_laplacian_eigenvalue(bg.spatial_curvature, interner)?;
    let spec = standard_tensor_harmonic_spec(bg.spatial_curvature, interner);
    project_equations_to_harmonic_space(
        equations,
        &eigenvalue,
        Expr::List(vec![Expr::Sym(spec.plus_mode), Expr::Sym(spec.cross_mode)]),
        spatial_coords(interner),
        "project_tensor_equations_to_harmonic_space",
        "tensor",
        interner,
    )
}

pub fn tensor_helicity_basis_flat(interner: &ax_ir::Interner) -> Vec<(String, lasso::Spur)> {
    vec![
        ("plus".to_string(), interner.get_or_intern("h_plus")),
        ("cross".to_string(), interner.get_or_intern("h_cross")),
    ]
}

pub fn render_harmonic_spec_unicode(
    curvature: crate::domain::SpatialCurvature,
    sector: crate::domain::SectorKind,
    interner: &ax_ir::Interner,
) -> String {
    let _ = interner;
    let curvature_name = match curvature {
        SpatialCurvature::Flat => "flat",
        SpatialCurvature::Closed => "closed",
        SpatialCurvature::Open => "open",
    };
    let sector_name = match sector {
        SectorKind::Scalar => "ScalarHarmonics",
        SectorKind::Vector => "VectorHarmonics",
        SectorKind::Tensor => "TensorHarmonics",
    };
    format!("{sector_name}({curvature_name}, k)")
}

fn project_equations_to_harmonic_space(
    equations: &[NamedEquation],
    eigenvalue: &Expr,
    mode_factor: Expr,
    spatial_coords: [Spur; 3],
    operation: &str,
    sector_label: &str,
    interner: &Interner,
) -> Result<ProjectedEquationSet, CosmologyError> {
    let k = Expr::Sym(interner.get_or_intern("k"));
    let mut rules_applied = vec![HarmonicProjectionRule {
        label: format!("{sector_label}_harmonic_factor"),
        before: mode_factor,
        after: Expr::one(),
    }];
    let projected = equations
        .iter()
        .map(|equation| {
            let (mut expr, mut rules) = rewrite_harmonic_expr(
                &equation.expr,
                eigenvalue,
                &spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            if sector_label == "scalar" && !contains_symbol(&expr, interner.get_or_intern("k")) {
                let before = expr.clone();
                expr = Expr::mul(vec![k.clone(), expr]);
                rules.push(HarmonicProjectionRule {
                    label: "scalar_wave_factor".to_string(),
                    before,
                    after: expr.clone(),
                });
            }
            rules_applied.append(&mut rules);
            if contains_explicit_spatial_derivative(&expr, &spatial_coords, interner) {
                return Err(CosmologyError::HarmonicProjectionFailure {
                    operation: operation.to_string(),
                });
            }
            Ok(NamedEquation {
                label: equation.label.clone(),
                expr,
                order: equation.order,
                sector: equation.sector,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ProjectedEquationSet {
        equations: projected,
        rules_applied,
    })
}

fn rewrite_harmonic_expr(
    expr: &Expr,
    eigenvalue: &Expr,
    spatial_coords: &[Spur; 3],
    operation: &str,
    sector_label: &str,
    interner: &Interner,
) -> Result<(Expr, Vec<HarmonicProjectionRule>), CosmologyError> {
    match expr {
        Expr::Call(name, args) if interner.resolve(*name) == "laplacian" && args.len() == 1 => {
            let (inner, rules) = rewrite_harmonic_expr(
                &args[0],
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            let after = Expr::mul(vec![eigenvalue.clone(), inner]);
            let mut all_rules = vec![HarmonicProjectionRule {
                label: format!("{sector_label}_laplacian_eigenvalue"),
                before: expr.clone(),
                after: after.clone(),
            }];
            all_rules.extend(rules);
            Ok((after, all_rules))
        }
        Expr::Call(name, args) if interner.resolve(*name) == "diff" && args.len() == 2 => {
            if let Expr::Sym(var) = &args[1] {
                if spatial_coords.contains(var) {
                    let replacement =
                        rewrite_spatial_diff(&args[0], *var, eigenvalue, spatial_coords, interner);
                    return Ok((
                        replacement.clone(),
                        vec![HarmonicProjectionRule {
                            label: format!("{sector_label}_spatial_derivative_projection"),
                            before: expr.clone(),
                            after: replacement,
                        }],
                    ));
                }
            }
            let (arg0, mut rules0) = rewrite_harmonic_expr(
                &args[0],
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            let (arg1, mut rules1) = rewrite_harmonic_expr(
                &args[1],
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            rules0.append(&mut rules1);
            Ok((Expr::Call(*name, vec![arg0, arg1]), rules0))
        }
        Expr::Add(terms) => rewrite_nary(
            terms,
            |items| Expr::add(items),
            eigenvalue,
            spatial_coords,
            operation,
            sector_label,
            interner,
        ),
        Expr::Mul(terms) => rewrite_nary(
            terms,
            |items| Expr::mul(items),
            eigenvalue,
            spatial_coords,
            operation,
            sector_label,
            interner,
        ),
        Expr::Pow(base, exp) => {
            let (base, mut rules0) = rewrite_harmonic_expr(
                base,
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            let (exp, mut rules1) = rewrite_harmonic_expr(
                exp,
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            rules0.append(&mut rules1);
            Ok((Expr::pow(base, exp), rules0))
        }
        Expr::Neg(inner) => {
            let (inner, rules) = rewrite_harmonic_expr(
                inner,
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            Ok((Expr::neg(inner), rules))
        }
        Expr::Group(inner, rel) => {
            let (inner, rules) = rewrite_harmonic_expr(
                inner,
                eigenvalue,
                spatial_coords,
                operation,
                sector_label,
                interner,
            )?;
            Ok((Expr::Group(Box::new(inner), *rel), rules))
        }
        Expr::Call(name, args) => {
            let mut rewritten = Vec::with_capacity(args.len());
            let mut rules = Vec::new();
            for arg in args {
                let (item, mut item_rules) = rewrite_harmonic_expr(
                    arg,
                    eigenvalue,
                    spatial_coords,
                    operation,
                    sector_label,
                    interner,
                )?;
                rewritten.push(item);
                rules.append(&mut item_rules);
            }
            Ok((Expr::Call(*name, rewritten), rules))
        }
        Expr::List(items) => rewrite_nary(
            items,
            Expr::List,
            eigenvalue,
            spatial_coords,
            operation,
            sector_label,
            interner,
        ),
        Expr::Matrix(rows) => {
            let mut rewritten_rows = Vec::with_capacity(rows.len());
            let mut rules = Vec::new();
            for row in rows {
                let (rewritten_row, mut row_rules) = rewrite_nary(
                    row,
                    |items| Expr::List(items),
                    eigenvalue,
                    spatial_coords,
                    operation,
                    sector_label,
                    interner,
                )?;
                match rewritten_row {
                    Expr::List(items) => rewritten_rows.push(items),
                    _ => {
                        return Err(CosmologyError::HarmonicProjectionFailure {
                            operation: operation.to_string(),
                        });
                    }
                }
                rules.append(&mut row_rules);
            }
            Ok((Expr::Matrix(rewritten_rows), rules))
        }
        other => Ok((other.clone(), Vec::new())),
    }
}

fn rewrite_nary(
    terms: &[Expr],
    rebuild: impl Fn(Vec<Expr>) -> Expr,
    eigenvalue: &Expr,
    spatial_coords: &[Spur; 3],
    operation: &str,
    sector_label: &str,
    interner: &Interner,
) -> Result<(Expr, Vec<HarmonicProjectionRule>), CosmologyError> {
    let mut rewritten = Vec::with_capacity(terms.len());
    let mut rules = Vec::new();
    for term in terms {
        let (item, mut item_rules) = rewrite_harmonic_expr(
            term,
            eigenvalue,
            spatial_coords,
            operation,
            sector_label,
            interner,
        )?;
        rewritten.push(item);
        rules.append(&mut item_rules);
    }
    Ok((rebuild(rewritten), rules))
}

fn contains_symbol(expr: &Expr, target: Spur) -> bool {
    match expr {
        Expr::Sym(sym) => *sym == target,
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
            items.iter().any(|item| contains_symbol(item, target))
        }
        Expr::Pow(base, exp) => contains_symbol(base, target) || contains_symbol(exp, target),
        Expr::Neg(inner) | Expr::Group(inner, _) | Expr::Indexed(inner, _) => {
            contains_symbol(inner, target)
        }
        Expr::Call(_, args) => args.iter().any(|arg| contains_symbol(arg, target)),
        Expr::FnDef(_, _, body) => contains_symbol(body, target),
        Expr::Rule(lhs, rhs, _) | Expr::Let(_, lhs, rhs) => {
            contains_symbol(lhs, target) || contains_symbol(rhs, target)
        }
        Expr::Piecewise(items) => items
            .iter()
            .any(|(branch, _)| contains_symbol(branch, target)),
        Expr::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|item| contains_symbol(item, target)),
        _ => false,
    }
}

fn rewrite_spatial_diff(
    expr: &Expr,
    coord: Spur,
    eigenvalue: &Expr,
    spatial_coords: &[Spur; 3],
    interner: &Interner,
) -> Expr {
    let Expr::Call(name, args) = expr else {
        return Expr::zero();
    };
    if interner.resolve(*name) != "diff" || args.len() != 2 {
        return Expr::zero();
    }
    let Expr::Sym(inner_coord) = &args[1] else {
        return Expr::zero();
    };
    if !spatial_coords.contains(inner_coord) {
        return Expr::zero();
    }
    if *inner_coord == coord {
        Expr::mul(vec![rational(1, 3), eigenvalue.clone(), args[0].clone()])
    } else {
        Expr::zero()
    }
}

fn contains_explicit_spatial_derivative(
    expr: &Expr,
    spatial_coords: &[Spur; 3],
    interner: &Interner,
) -> bool {
    match expr {
        Expr::Call(name, args) if interner.resolve(*name) == "laplacian" => true,
        Expr::Call(name, args) if interner.resolve(*name) == "diff" && args.len() == 2 => {
            match &args[1] {
                Expr::Sym(var) if spatial_coords.contains(var) => true,
                _ => args
                    .iter()
                    .any(|arg| contains_explicit_spatial_derivative(arg, spatial_coords, interner)),
            }
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => terms
            .iter()
            .any(|term| contains_explicit_spatial_derivative(term, spatial_coords, interner)),
        Expr::Pow(base, exp) => {
            contains_explicit_spatial_derivative(base, spatial_coords, interner)
                || contains_explicit_spatial_derivative(exp, spatial_coords, interner)
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => {
            contains_explicit_spatial_derivative(inner, spatial_coords, interner)
        }
        Expr::Call(_, args) => args
            .iter()
            .any(|arg| contains_explicit_spatial_derivative(arg, spatial_coords, interner)),
        Expr::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|item| contains_explicit_spatial_derivative(item, spatial_coords, interner)),
        _ => false,
    }
}

fn spatial_coords(interner: &Interner) -> [Spur; 3] {
    [
        interner.get_or_intern("x"),
        interner.get_or_intern("y"),
        interner.get_or_intern("z"),
    ]
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
    use crate::FrwBackgroundSpec;

    fn default_bg(interner: &Interner) -> FrwBackgroundSpec {
        FrwBackgroundSpec::default_flat_conformal(interner)
    }

    fn closed_bg(interner: &Interner) -> FrwBackgroundSpec {
        let mut bg = default_bg(interner);
        bg.spatial_curvature = SpatialCurvature::Closed;
        bg
    }

    fn open_bg(interner: &Interner) -> FrwBackgroundSpec {
        let mut bg = default_bg(interner);
        bg.spatial_curvature = SpatialCurvature::Open;
        bg
    }

    fn assert_no_spatial_diff(rendered: &str) {
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
    }

    #[test]
    fn standard_scalar_harmonic_spec_uses_expected_symbols() {
        let interner = Interner::new();
        let spec = standard_scalar_harmonic_spec(SpatialCurvature::Flat, &interner);
        assert_eq!(interner.resolve(spec.mode_symbol), "Q");
        assert_eq!(interner.resolve(spec.wave_symbol), "k");
        assert_eq!(spec.basis, HarmonicBasisKind::FourierFlat);
    }

    #[test]
    fn standard_vector_harmonic_spec_uses_expected_symbols() {
        let interner = Interner::new();
        let spec = standard_vector_harmonic_spec(SpatialCurvature::Closed, &interner);
        assert_eq!(interner.resolve(spec.mode_x), "QV_x");
        assert_eq!(interner.resolve(spec.mode_y), "QV_y");
        assert_eq!(interner.resolve(spec.mode_z), "QV_z");
        assert_eq!(interner.resolve(spec.wave_symbol), "k");
        assert_eq!(spec.basis, HarmonicBasisKind::VectorHarmonics);
    }

    #[test]
    fn standard_tensor_harmonic_spec_uses_expected_symbols() {
        let interner = Interner::new();
        let spec = standard_tensor_harmonic_spec(SpatialCurvature::Open, &interner);
        assert_eq!(interner.resolve(spec.plus_mode), "GT_plus");
        assert_eq!(interner.resolve(spec.cross_mode), "GT_cross");
        assert_eq!(interner.resolve(spec.wave_symbol), "k");
        assert_eq!(spec.basis, HarmonicBasisKind::TensorHarmonics);
    }

    #[test]
    fn scalar_laplacian_eigenvalues_match_expected_forms() {
        let interner = Interner::new();
        let k = Expr::Sym(interner.get_or_intern("k"));
        let k_sq = Expr::pow(k, int(2));
        let kappa = Expr::Sym(interner.get_or_intern("K"));
        assert_eq!(
            scalar_laplacian_eigenvalue(SpatialCurvature::Flat, &interner),
            Ok(Expr::neg(k_sq.clone()))
        );
        assert_eq!(
            scalar_laplacian_eigenvalue(SpatialCurvature::Closed, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq.clone(),
                Expr::neg(Expr::mul(vec![int(3), kappa.clone()])),
            ])))
        );
        assert_eq!(
            scalar_laplacian_eigenvalue(SpatialCurvature::Open, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq,
                Expr::mul(vec![int(3), kappa])
            ])))
        );
    }

    #[test]
    fn vector_laplacian_eigenvalues_match_expected_forms() {
        let interner = Interner::new();
        let k = Expr::Sym(interner.get_or_intern("k"));
        let k_sq = Expr::pow(k, int(2));
        let kappa = Expr::Sym(interner.get_or_intern("K"));
        assert_eq!(
            vector_laplacian_eigenvalue(SpatialCurvature::Flat, &interner),
            Ok(Expr::neg(k_sq.clone()))
        );
        assert_eq!(
            vector_laplacian_eigenvalue(SpatialCurvature::Closed, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq.clone(),
                Expr::neg(Expr::mul(vec![int(2), kappa.clone()])),
            ])))
        );
        assert_eq!(
            vector_laplacian_eigenvalue(SpatialCurvature::Open, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq,
                Expr::mul(vec![int(2), kappa])
            ])))
        );
    }

    #[test]
    fn tensor_laplacian_eigenvalues_match_expected_forms() {
        let interner = Interner::new();
        let k = Expr::Sym(interner.get_or_intern("k"));
        let k_sq = Expr::pow(k, int(2));
        let kappa = Expr::Sym(interner.get_or_intern("K"));
        assert_eq!(
            tensor_laplacian_eigenvalue(SpatialCurvature::Flat, &interner),
            Ok(Expr::neg(k_sq.clone()))
        );
        assert_eq!(
            tensor_laplacian_eigenvalue(SpatialCurvature::Closed, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq.clone(),
                Expr::neg(Expr::mul(vec![int(2), kappa.clone()])),
            ])))
        );
        assert_eq!(
            tensor_laplacian_eigenvalue(SpatialCurvature::Open, &interner),
            Ok(Expr::neg(Expr::add(vec![
                k_sq,
                Expr::mul(vec![int(2), kappa])
            ])))
        );
    }

    #[test]
    fn project_scalar_equations_to_harmonic_space_removes_explicit_spatial_derivatives() {
        let interner = Interner::new();
        let bg = default_bg(&interner);
        let decomp = crate::gauge::svt_decompose_perturbation(3, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));
        let equations = crate::cosmology::linearized_einstein_scalar(&bg, &decomp, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));
        let projected = project_scalar_equations_to_harmonic_space(&equations, &bg, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));
        let rendered = ax_ir::pretty_print(
            &Expr::List(
                projected
                    .equations
                    .iter()
                    .map(|eq| eq.expr.clone())
                    .collect(),
            ),
            &interner,
        );
        assert_no_spatial_diff(&rendered);
    }

    #[test]
    fn project_vector_equations_to_harmonic_space_removes_explicit_spatial_derivatives() {
        let interner = Interner::new();
        let bg = closed_bg(&interner);
        let equations =
            crate::vector_tensor::derive_linear_vector_einstein_equations_poisson(&bg, &interner)
                .unwrap_or_else(|err| panic!("{err:?}"))
                .equations;
        let projected = project_vector_equations_to_harmonic_space(&equations, &bg, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));
        let rendered = ax_ir::pretty_print(
            &Expr::List(
                projected
                    .equations
                    .iter()
                    .map(|eq| eq.expr.clone())
                    .collect(),
            ),
            &interner,
        );
        assert_no_spatial_diff(&rendered);
    }

    #[test]
    fn project_tensor_equations_to_harmonic_space_removes_explicit_spatial_derivatives() {
        let interner = Interner::new();
        let bg = open_bg(&interner);
        let equations =
            crate::vector_tensor::derive_linear_tensor_einstein_equations(&bg, &interner)
                .unwrap_or_else(|err| panic!("{err:?}"))
                .equations;
        let projected = project_tensor_equations_to_harmonic_space(&equations, &bg, &interner)
            .unwrap_or_else(|err| panic!("{err:?}"));
        let rendered = ax_ir::pretty_print(
            &Expr::List(
                projected
                    .equations
                    .iter()
                    .map(|eq| eq.expr.clone())
                    .collect(),
            ),
            &interner,
        );
        assert_no_spatial_diff(&rendered);
    }

    #[test]
    fn tensor_helicity_basis_flat_returns_plus_and_cross() {
        let interner = Interner::new();
        let basis = tensor_helicity_basis_flat(&interner);
        assert_eq!(
            basis,
            vec![
                ("plus".to_string(), interner.get_or_intern("h_plus")),
                ("cross".to_string(), interner.get_or_intern("h_cross")),
            ]
        );
    }

    #[test]
    fn render_harmonic_spec_unicode_formats_expected_string() {
        let interner = Interner::new();
        assert_eq!(
            render_harmonic_spec_unicode(SpatialCurvature::Flat, SectorKind::Scalar, &interner),
            "ScalarHarmonics(flat, k)"
        );
        assert_eq!(
            render_harmonic_spec_unicode(SpatialCurvature::Closed, SectorKind::Vector, &interner),
            "VectorHarmonics(closed, k)"
        );
        assert_eq!(
            render_harmonic_spec_unicode(SpatialCurvature::Open, SectorKind::Tensor, &interner),
            "TensorHarmonics(open, k)"
        );
    }
}
