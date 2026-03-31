#![forbid(unsafe_code)]

use ax_ir::Expr;

fn second_derivative_symbol(
    field: lasso::Spur,
    coord_a: lasso::Spur,
    coord_b: lasso::Spur,
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    let field_name = interner.resolve(field);
    let coord_a_name = interner.resolve(coord_a);
    let coord_b_name = interner.resolve(coord_b);
    interner.get_or_intern(&format!("d2{field_name}_d{coord_a_name}d{coord_b_name}"))
}

fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(terms.into_iter().map(simplify_expr).collect()),
        Expr::Mul(factors) => Expr::mul(factors.into_iter().map(simplify_expr).collect()),
        Expr::Pow(base, exp) => Expr::pow(simplify_expr(*base), simplify_expr(*exp)),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(simplify_expr(*re)),
            Box::new(simplify_expr(*im)),
        ),
        Expr::Call(f, args) => Expr::Call(f, args.into_iter().map(simplify_expr).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(name, params, Box::new(simplify_expr(*body)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(simplify_expr(*lhs)),
            Box::new(simplify_expr(*rhs)),
            trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| (simplify_expr(value), condition))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(Box::new(simplify_expr(*base)), indices),
        Expr::Let(name, value, body) => Expr::Let(
            name,
            Box::new(simplify_expr(*value)),
            Box::new(simplify_expr(*body)),
        ),
        Expr::List(items) => Expr::List(items.into_iter().map(simplify_expr).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| row.into_iter().map(simplify_expr).collect())
                .collect(),
        ),
        other => other,
    }
}

fn total_derivative(
    expr: &ax_ir::Expr,
    coord: lasso::Spur,
    field: lasso::Spur,
    field_derivs: &[lasso::Spur],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut terms = Vec::new();
    terms.push(ax_tensor::diff_component(expr, coord, interner));

    if let Some(coord_idx) = coords.iter().position(|c| *c == coord) {
        if coord_idx < field_derivs.len() {
            let df_dphi = ax_tensor::diff_component(expr, field, interner);
            let dphi_dxi = Expr::Sym(field_derivs[coord_idx]);
            terms.push(Expr::mul(vec![df_dphi, dphi_dxi]));
        }
    }

    for (j, field_deriv) in field_derivs.iter().enumerate() {
        let df_dfdj = ax_tensor::diff_component(expr, *field_deriv, interner);
        let coord_j = coords.get(j).copied().unwrap_or(coord);
        let second = second_derivative_symbol(field, coord_j, coord, interner);
        terms.push(Expr::mul(vec![df_dfdj, Expr::Sym(second)]));
    }

    simplify_expr(Expr::add(terms))
}

pub fn functional_derivative(
    lagrangian: &ax_ir::Expr,
    field: lasso::Spur,
    field_derivs: &[lasso::Spur],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let d_l_d_field = ax_tensor::diff_component(lagrangian, field, interner);
    let derivative_sum = coords
        .iter()
        .zip(field_derivs.iter())
        .map(|(coord, field_deriv)| {
            let d_l_d_field_deriv = ax_tensor::diff_component(lagrangian, *field_deriv, interner);
            total_derivative(
                &d_l_d_field_deriv,
                *coord,
                field,
                field_derivs,
                coords,
                interner,
            )
        })
        .collect::<Vec<_>>();

    simplify_expr(Expr::add(vec![
        d_l_d_field,
        Expr::neg(Expr::add(derivative_sum)),
    ]))
}

pub fn euler_lagrange_system(
    lagrangian: &ax_ir::Expr,
    fields: &[(lasso::Spur, Vec<lasso::Spur>)],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    fields
        .iter()
        .map(|(field, derivs)| functional_derivative(lagrangian, *field, derivs, coords, interner))
        .collect()
}

pub fn vary_action(
    lagrangian: &ax_ir::Expr,
    field: lasso::Spur,
    variation: lasso::Spur,
    field_derivs: &[lasso::Spur],
    variation_derivs: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let direct = Expr::mul(vec![
        ax_tensor::diff_component(lagrangian, field, interner),
        Expr::Sym(variation),
    ]);

    let deriv_terms = field_derivs
        .iter()
        .zip(variation_derivs.iter())
        .map(|(field_deriv, variation_deriv)| {
            Expr::mul(vec![
                ax_tensor::diff_component(lagrangian, *field_deriv, interner),
                Expr::Sym(*variation_deriv),
            ])
        })
        .collect::<Vec<_>>();

    simplify_expr(Expr::add(vec![direct, Expr::add(deriv_terms)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_lagrange_free_particle() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let v = interner.get_or_intern("v");
        let t = interner.get_or_intern("t");
        let m = interner.get_or_intern("m");

        let lagrangian = Expr::mul(vec![
            Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
            Expr::Sym(m),
            Expr::pow(Expr::Sym(v), Expr::Int(2.into())),
        ]);

        let result = functional_derivative(&lagrangian, x, &[v], &[t], &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("m"), "got: {}", pp);
    }

    #[test]
    fn euler_lagrange_harmonic_oscillator() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let v = interner.get_or_intern("v");
        let t = interner.get_or_intern("t");
        let m = interner.get_or_intern("m");
        let k = interner.get_or_intern("k");

        let lagrangian = Expr::add(vec![
            Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
                Expr::Sym(m),
                Expr::pow(Expr::Sym(v), Expr::Int(2.into())),
            ]),
            Expr::neg(Expr::mul(vec![
                Expr::Rational(num_rational::BigRational::new(1.into(), 2.into())),
                Expr::Sym(k),
                Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            ])),
        ]);

        let result = functional_derivative(&lagrangian, x, &[v], &[t], &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("k") || pp.contains("m"), "got: {}", pp);
    }
}
