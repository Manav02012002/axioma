use crate::cosmology::require_conformal_time;
use crate::domain::FrwBackgroundSpec;
use crate::error::CosmologyError;
use ax_ir::{Expr, Interner};
use num_bigint::BigInt;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubicInteractionChannel {
    ScalarScalarScalar,
    TensorTensorTensor,
    ScalarScalarTensor,
    ScalarTensorTensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedCubicAction {
    pub channel: CubicInteractionChannel,
    pub lagrangian_density: ax_ir::Expr,
    pub fields: Vec<lasso::Spur>,
    pub coordinates: Vec<lasso::Spur>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FourierKernel {
    pub channel: CubicInteractionChannel,
    pub kernel: ax_ir::Expr,
    pub momenta: [lasso::Spur; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct BispectrumShapeValue {
    pub shape_name: String,
    pub kernel: ax_ir::Expr,
    pub evaluated_form: ax_ir::Expr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractionVertexExport {
    pub channel: CubicInteractionChannel,
    pub code: String,
}

pub fn reduced_cubic_scalar_action(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ReducedCubicAction, crate::error::CosmologyError> {
    validate_cubic_background(bg, "reduced_cubic_scalar_action")?;
    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let r = interner.get_or_intern("R");
    let r_eta = interner.get_or_intern("R_eta");
    let r_x = interner.get_or_intern("R_x");
    let r_y = interner.get_or_intern("R_y");
    let r_z = interner.get_or_intern("R_z");
    let a_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let epsilon = Expr::Sym(interner.get_or_intern("epsilon"));
    let eta_sr = Expr::Sym(interner.get_or_intern("eta_sr"));
    let r_expr = Expr::Sym(r);

    let lagrangian_density = Expr::add(vec![
        Expr::mul(vec![
            a_sq.clone(),
            Expr::pow(epsilon.clone(), int(2)),
            r_expr.clone(),
            Expr::pow(Expr::Sym(r_eta), int(2)),
        ]),
        Expr::mul(vec![
            a_sq.clone(),
            Expr::pow(epsilon.clone(), int(2)),
            r_expr.clone(),
            Expr::add(vec![
                Expr::pow(Expr::Sym(r_x), int(2)),
                Expr::pow(Expr::Sym(r_y), int(2)),
                Expr::pow(Expr::Sym(r_z), int(2)),
            ]),
        ]),
        Expr::mul(vec![
            a_sq,
            epsilon,
            eta_sr,
            Expr::pow(r_expr, int(2)),
            Expr::Sym(r_eta),
        ]),
    ]);

    Ok(ReducedCubicAction {
        channel: CubicInteractionChannel::ScalarScalarScalar,
        lagrangian_density,
        fields: vec![r],
        coordinates: vec![eta, x, y, z],
    })
}

pub fn reduced_cubic_tensor_action(
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ReducedCubicAction, crate::error::CosmologyError> {
    validate_cubic_background(bg, "reduced_cubic_tensor_action")?;
    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let h_plus = interner.get_or_intern("h_plus");
    let h_cross = interner.get_or_intern("h_cross");
    let h_plus_eta = interner.get_or_intern("h_plus_eta");
    let h_plus_x = interner.get_or_intern("h_plus_x");
    let h_plus_y = interner.get_or_intern("h_plus_y");
    let h_plus_z = interner.get_or_intern("h_plus_z");
    let h_cross_eta = interner.get_or_intern("h_cross_eta");
    let h_cross_x = interner.get_or_intern("h_cross_x");
    let h_cross_y = interner.get_or_intern("h_cross_y");
    let h_cross_z = interner.get_or_intern("h_cross_z");
    let a_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(2));

    let lagrangian_density = Expr::add(vec![
        Expr::mul(vec![
            a_sq.clone(),
            Expr::Sym(h_plus),
            Expr::pow(Expr::Sym(h_plus_eta), int(2)),
        ]),
        Expr::mul(vec![
            a_sq.clone(),
            Expr::Sym(h_cross),
            Expr::pow(Expr::Sym(h_cross_eta), int(2)),
        ]),
        Expr::mul(vec![
            a_sq.clone(),
            Expr::Sym(h_plus),
            Expr::Sym(h_cross),
            Expr::Sym(h_cross_eta),
        ]),
        Expr::mul(vec![
            a_sq,
            Expr::Sym(h_cross),
            Expr::add(vec![
                Expr::pow(Expr::Sym(h_plus_x), int(2)),
                Expr::pow(Expr::Sym(h_plus_y), int(2)),
                Expr::pow(Expr::Sym(h_plus_z), int(2)),
                Expr::pow(Expr::Sym(h_cross_x), int(2)),
                Expr::pow(Expr::Sym(h_cross_y), int(2)),
                Expr::pow(Expr::Sym(h_cross_z), int(2)),
            ]),
        ]),
    ]);

    Ok(ReducedCubicAction {
        channel: CubicInteractionChannel::TensorTensorTensor,
        lagrangian_density,
        fields: vec![h_plus, h_cross],
        coordinates: vec![eta, x, y, z],
    })
}

pub fn reduced_cubic_mixed_action(
    channel: CubicInteractionChannel,
    bg: &crate::domain::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Result<ReducedCubicAction, crate::error::CosmologyError> {
    validate_cubic_background(bg, "reduced_cubic_mixed_action")?;
    let eta = bg.conformal_time;
    let x = interner.get_or_intern("x");
    let y = interner.get_or_intern("y");
    let z = interner.get_or_intern("z");
    let a_sq = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let epsilon = Expr::Sym(interner.get_or_intern("epsilon"));
    let r = interner.get_or_intern("R");
    let r_eta = interner.get_or_intern("R_eta");
    let r_x = interner.get_or_intern("R_x");
    let r_y = interner.get_or_intern("R_y");
    let r_z = interner.get_or_intern("R_z");
    let h_plus = interner.get_or_intern("h_plus");
    let h_cross = interner.get_or_intern("h_cross");
    let h_plus_eta = interner.get_or_intern("h_plus_eta");
    let h_plus_x = interner.get_or_intern("h_plus_x");
    let h_plus_y = interner.get_or_intern("h_plus_y");
    let h_plus_z = interner.get_or_intern("h_plus_z");
    let h_cross_eta = interner.get_or_intern("h_cross_eta");

    let lagrangian_density = match channel {
        CubicInteractionChannel::ScalarScalarTensor => Expr::add(vec![
            Expr::mul(vec![
                a_sq.clone(),
                epsilon.clone(),
                Expr::Sym(h_plus),
                Expr::pow(Expr::Sym(r_eta), int(2)),
            ]),
            Expr::mul(vec![
                a_sq,
                epsilon,
                Expr::Sym(h_cross),
                Expr::add(vec![
                    Expr::mul(vec![Expr::Sym(r_x), Expr::Sym(h_plus_x)]),
                    Expr::mul(vec![Expr::Sym(r_y), Expr::Sym(h_plus_y)]),
                    Expr::mul(vec![Expr::Sym(r_z), Expr::Sym(h_plus_z)]),
                ]),
            ]),
        ]),
        CubicInteractionChannel::ScalarTensorTensor => Expr::add(vec![
            Expr::mul(vec![
                a_sq.clone(),
                epsilon.clone(),
                Expr::Sym(r),
                Expr::pow(Expr::Sym(h_plus_eta), int(2)),
            ]),
            Expr::mul(vec![
                a_sq,
                epsilon,
                Expr::Sym(r),
                Expr::Sym(h_plus),
                Expr::Sym(h_cross_eta),
            ]),
        ]),
        _ => {
            return Err(CosmologyError::UnsupportedCubicChannel {
                channel: format!("{channel:?}"),
            });
        }
    };

    Ok(ReducedCubicAction {
        channel,
        lagrangian_density,
        fields: vec![r, h_plus, h_cross],
        coordinates: vec![eta, x, y, z],
    })
}

pub fn cubic_fourier_kernel(
    action: &ReducedCubicAction,
    interner: &ax_ir::Interner,
) -> Result<FourierKernel, crate::error::CosmologyError> {
    let k1 = interner.get_or_intern("k1");
    let k2 = interner.get_or_intern("k2");
    let k3 = interner.get_or_intern("k3");
    let field_names = action
        .fields
        .iter()
        .map(|field| interner.resolve(*field).to_string())
        .collect::<Vec<_>>();
    let kernel = substitute_kernel_symbols(
        &action.lagrangian_density,
        &field_names,
        interner,
        Expr::Sym(k1),
        Expr::Sym(k2),
        Expr::Sym(k3),
    );
    Ok(FourierKernel {
        channel: action.channel,
        kernel,
        momenta: [k1, k2, k3],
    })
}

pub fn bispectrum_shape(
    kernel: &FourierKernel,
    shape: &str,
    interner: &ax_ir::Interner,
) -> Result<BispectrumShapeValue, crate::error::CosmologyError> {
    let p = Expr::Sym(interner.get_or_intern("p"));
    let q = Expr::Sym(interner.get_or_intern("q"));
    let (k1_sub, k2_sub, k3_sub) = match shape {
        "local" => (q.clone(), q.clone(), p.clone()),
        "equilateral" => (q.clone(), q.clone(), q.clone()),
        "squeezed" => (p.clone(), q.clone(), q.clone()),
        _ => {
            return Err(CosmologyError::UnsupportedBispectrumShape {
                shape: shape.to_string(),
            });
        }
    };
    let evaluated_form = substitute_many(
        &kernel.kernel,
        &[
            (kernel.momenta[0], k1_sub),
            (kernel.momenta[1], k2_sub),
            (kernel.momenta[2], k3_sub),
        ],
    );
    Ok(BispectrumShapeValue {
        shape_name: shape.to_string(),
        kernel: kernel.kernel.clone(),
        evaluated_form,
    })
}

pub fn export_cubic_vertex(
    target: &str,
    kernel: &FourierKernel,
    interner: &ax_ir::Interner,
) -> Result<InteractionVertexExport, crate::error::CosmologyError> {
    let k1 = interner.get_or_intern("k1");
    let k2 = interner.get_or_intern("k2");
    let k3 = interner.get_or_intern("k3");
    let a = interner.get_or_intern("a");
    let epsilon = interner.get_or_intern("epsilon");
    let mut args = vec![k1, k2, k3, a, epsilon];
    let mut extra = collect_kernel_symbols(&kernel.kernel, interner)
        .into_iter()
        .filter(|sym| !args.contains(sym))
        .collect::<Vec<_>>();
    extra.sort_by_key(|sym| interner.resolve(*sym).to_string());
    args.extend(extra);
    let code = match target {
        "python" => {
            ax_codegen::emit_python_function("cubic_vertex", &args, &kernel.kernel, interner)
        }
        "rust" => ax_codegen::emit_rust_function("cubic_vertex", &args, &kernel.kernel, interner),
        "cpp" => ax_codegen::emit_cpp_function("cubic_vertex", &args, &kernel.kernel, interner),
        _ => {
            return Err(CosmologyError::UnsupportedCubicChannel {
                channel: target.to_string(),
            });
        }
    };
    Ok(InteractionVertexExport {
        channel: kernel.channel,
        code,
    })
}

fn validate_cubic_background(
    bg: &FrwBackgroundSpec,
    operation: &str,
) -> Result<(), CosmologyError> {
    require_conformal_time(bg, operation)?;
    if bg.spatial_dim != 3 {
        return Err(CosmologyError::MetricAnsatzRequiresThreeSpatialDimensions {
            got: bg.spatial_dim,
        });
    }
    Ok(())
}

fn substitute_kernel_symbols(
    expr: &Expr,
    field_names: &[String],
    interner: &Interner,
    k1: Expr,
    k2: Expr,
    k3: Expr,
) -> Expr {
    match expr {
        Expr::Sym(sym) => {
            let name = interner.resolve(*sym);
            if field_names.iter().any(|field| field == name) {
                Expr::one()
            } else if name.ends_with("_x") {
                k1
            } else if name.ends_with("_y") {
                k2
            } else if name.ends_with("_z") {
                k3
            } else if name.ends_with("_eta") {
                Expr::add(vec![k1, k2, k3])
            } else {
                Expr::Sym(*sym)
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    substitute_kernel_symbols(
                        term,
                        field_names,
                        interner,
                        k1.clone(),
                        k2.clone(),
                        k3.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    substitute_kernel_symbols(
                        factor,
                        field_names,
                        interner,
                        k1.clone(),
                        k2.clone(),
                        k3.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_kernel_symbols(
                base,
                field_names,
                interner,
                k1.clone(),
                k2.clone(),
                k3.clone(),
            ),
            substitute_kernel_symbols(exp, field_names, interner, k1, k2, k3),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_kernel_symbols(
            inner,
            field_names,
            interner,
            k1,
            k2,
            k3,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_kernel_symbols(
                re,
                field_names,
                interner,
                k1.clone(),
                k2.clone(),
                k3.clone(),
            )),
            Box::new(substitute_kernel_symbols(
                im,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
        ),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| {
                    substitute_kernel_symbols(
                        arg,
                        field_names,
                        interner,
                        k1.clone(),
                        k2.clone(),
                        k3.clone(),
                    )
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_kernel_symbols(
                body,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_kernel_symbols(
                lhs,
                field_names,
                interner,
                k1.clone(),
                k2.clone(),
                k3.clone(),
            )),
            Box::new(substitute_kernel_symbols(
                rhs,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_kernel_symbols(
                            value,
                            field_names,
                            interner,
                            k1.clone(),
                            k2.clone(),
                            k3.clone(),
                        ),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_kernel_symbols(
                base,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(substitute_kernel_symbols(
                inner,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
            *rel,
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_kernel_symbols(
                value,
                field_names,
                interner,
                k1.clone(),
                k2.clone(),
                k3.clone(),
            )),
            Box::new(substitute_kernel_symbols(
                body,
                field_names,
                interner,
                k1,
                k2,
                k3,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| {
                    substitute_kernel_symbols(
                        item,
                        field_names,
                        interner,
                        k1.clone(),
                        k2.clone(),
                        k3.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| {
                            substitute_kernel_symbols(
                                item,
                                field_names,
                                interner,
                                k1.clone(),
                                k2.clone(),
                                k3.clone(),
                            )
                        })
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn collect_kernel_symbols(expr: &Expr, interner: &Interner) -> BTreeSet<lasso::Spur> {
    let mut out = BTreeSet::new();
    collect_kernel_symbols_inner(expr, interner, &mut out);
    out
}

fn collect_kernel_symbols_inner(expr: &Expr, interner: &Interner, out: &mut BTreeSet<lasso::Spur>) {
    match expr {
        Expr::Sym(sym) => {
            let name = interner.resolve(*sym);
            if name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                out.insert(*sym);
            }
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_kernel_symbols_inner(term, interner, out);
            }
        }
        Expr::Pow(base, exp) | Expr::Rule(base, exp, _) => {
            collect_kernel_symbols_inner(base, interner, out);
            collect_kernel_symbols_inner(exp, interner, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) | Expr::Indexed(inner, _) => {
            collect_kernel_symbols_inner(inner, interner, out)
        }
        Expr::Complex(re, im) => {
            collect_kernel_symbols_inner(re, interner, out);
            collect_kernel_symbols_inner(im, interner, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_kernel_symbols_inner(arg, interner, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_kernel_symbols_inner(body, interner, out),
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_kernel_symbols_inner(value, interner, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_kernel_symbols_inner(value, interner, out);
            collect_kernel_symbols_inner(body, interner, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for item in row {
                    collect_kernel_symbols_inner(item, interner, out);
                }
            }
        }
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => {}
    }
}

fn substitute_many(expr: &Expr, replacements: &[(lasso::Spur, Expr)]) -> Expr {
    match expr {
        Expr::Sym(sym) => replacements
            .iter()
            .find_map(|(target, replacement)| (*target == *sym).then(|| replacement.clone()))
            .unwrap_or_else(|| Expr::Sym(*sym)),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_many(term, replacements))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_many(factor, replacements))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_many(base, replacements),
            substitute_many(exp, replacements),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_many(inner, replacements)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_many(re, replacements)),
            Box::new(substitute_many(im, replacements)),
        ),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| substitute_many(arg, replacements))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_many(body, replacements)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_many(lhs, replacements)),
            Box::new(substitute_many(rhs, replacements)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (substitute_many(value, replacements), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_many(base, replacements)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_many(inner, replacements)), *rel)
        }
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_many(value, replacements)),
            Box::new(substitute_many(body, replacements)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_many(item, replacements))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| substitute_many(item, replacements))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bg(interner: &Interner) -> FrwBackgroundSpec {
        FrwBackgroundSpec::default_flat_conformal(interner)
    }

    #[test]
    fn reduced_cubic_scalar_action_contains_expected_r_terms() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let rendered = ax_ir::pretty_print(&action.lagrangian_density, &interner);
        assert!(rendered.contains("R_eta"), "got {rendered}");
        assert!(rendered.contains("R_x"), "got {rendered}");
        assert!(rendered.contains("eta_sr"), "got {rendered}");
    }

    #[test]
    fn reduced_cubic_tensor_action_contains_expected_h_terms() {
        let interner = Interner::new();
        let action = reduced_cubic_tensor_action(&bg(&interner), &interner).unwrap();
        let rendered = ax_ir::pretty_print(&action.lagrangian_density, &interner);
        assert!(rendered.contains("h_plus_eta"), "got {rendered}");
        assert!(rendered.contains("h_cross"), "got {rendered}");
    }

    #[test]
    fn reduced_cubic_mixed_action_rejects_invalid_channel() {
        let interner = Interner::new();
        let result = reduced_cubic_mixed_action(
            CubicInteractionChannel::ScalarScalarScalar,
            &bg(&interner),
            &interner,
        );
        match result {
            Err(CosmologyError::UnsupportedCubicChannel { channel }) => {
                assert!(channel.contains("ScalarScalarScalar"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn cubic_fourier_kernel_removes_explicit_spatial_derivatives() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let rendered = ax_ir::pretty_print(&kernel.kernel, &interner);
        assert!(!rendered.contains("_x"), "got {rendered}");
        assert!(!rendered.contains("_y"), "got {rendered}");
        assert!(!rendered.contains("_z"), "got {rendered}");
        assert!(rendered.contains("k1"), "got {rendered}");
    }

    #[test]
    fn bispectrum_shape_local_substitutes_p_and_q() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let shape = bispectrum_shape(&kernel, "local", &interner).unwrap();
        let rendered = ax_ir::pretty_print(&shape.evaluated_form, &interner);
        assert!(rendered.contains("p"), "got {rendered}");
        assert!(rendered.contains("q"), "got {rendered}");
    }

    #[test]
    fn bispectrum_shape_equilateral_sets_all_momenta_equal() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let shape = bispectrum_shape(&kernel, "equilateral", &interner).unwrap();
        let rendered = ax_ir::pretty_print(&shape.evaluated_form, &interner);
        assert!(rendered.contains("q"), "got {rendered}");
        assert!(!rendered.contains("k1"), "got {rendered}");
        assert!(!rendered.contains("k2"), "got {rendered}");
        assert!(!rendered.contains("k3"), "got {rendered}");
    }

    #[test]
    fn bispectrum_shape_squeezed_uses_expected_substitution() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let shape = bispectrum_shape(&kernel, "squeezed", &interner).unwrap();
        let rendered = ax_ir::pretty_print(&shape.evaluated_form, &interner);
        assert!(rendered.contains("p"), "got {rendered}");
        assert!(rendered.contains("q"), "got {rendered}");
    }

    #[test]
    fn bispectrum_shape_rejects_unknown_shape() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let result = bispectrum_shape(&kernel, "folded", &interner);
        match result {
            Err(CosmologyError::UnsupportedBispectrumShape { shape }) => {
                assert_eq!(shape, "folded")
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn export_cubic_vertex_python_contains_def_cubic_vertex() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let export = export_cubic_vertex("python", &kernel, &interner).unwrap();
        assert!(export.code.contains("def cubic_vertex("));
    }

    #[test]
    fn export_cubic_vertex_rust_contains_pub_fn_cubic_vertex() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let export = export_cubic_vertex("rust", &kernel, &interner).unwrap();
        assert!(export.code.contains("pub fn cubic_vertex("));
    }

    #[test]
    fn export_cubic_vertex_cpp_contains_double_cubic_vertex() {
        let interner = Interner::new();
        let action = reduced_cubic_scalar_action(&bg(&interner), &interner).unwrap();
        let kernel = cubic_fourier_kernel(&action, &interner).unwrap();
        let export = export_cubic_vertex("cpp", &kernel, &interner).unwrap();
        assert!(export.code.contains("double cubic_vertex("));
    }
}
