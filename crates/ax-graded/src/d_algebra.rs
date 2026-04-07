use crate::superspace::{SuperspaceSetup, ThetaMonomial};
use crate::GradedSymbolTable;
use ax_ir::{Expr, Interner};
use lasso::Spur;
use num_bigint::BigInt;

pub enum SuperspaceMeasure {
    FullSuperspace,
    Chiral,
    AntiChiral,
}

pub fn apply_d_alpha(
    expr: &Expr,
    alpha: usize,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    assert!(alpha < setup.theta.len(), "theta spinor index out of range");
    let theta_derivative = grassmann_derivative(expr, setup.theta[alpha], table, interner);
    crate::graded_simplify(&theta_derivative, table, interner)
}

pub fn apply_d_bar_alpha_dot(
    expr: &Expr,
    alpha_dot: usize,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    assert!(
        alpha_dot < setup.theta_bar.len(),
        "theta-bar spinor index out of range"
    );
    let theta_bar_derivative = Expr::neg(grassmann_derivative(
        expr,
        setup.theta_bar[alpha_dot],
        table,
        interner,
    ));
    crate::graded_simplify(&theta_bar_derivative, table, interner)
}

pub fn d_squared(
    expr: &Expr,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let d1 = apply_d_alpha(expr, 1, setup, table, interner);
    let d0d1 = apply_d_alpha(&d1, 0, setup, table, interner);
    crate::graded_simplify(&Expr::mul(vec![int(2), d0d1]), table, interner)
}

pub fn d_bar_squared(
    expr: &Expr,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let dbar1 = apply_d_bar_alpha_dot(expr, 1, setup, table, interner);
    let dbar0dbar1 = apply_d_bar_alpha_dot(&dbar1, 0, setup, table, interner);
    crate::graded_simplify(&Expr::mul(vec![int(-2), dbar0dbar1]), table, interner)
}

pub fn superspace_integrate(
    expr: &Expr,
    measure: SuperspaceMeasure,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let target = match measure {
        SuperspaceMeasure::FullSuperspace => ThetaMonomial {
            theta_powers: vec![1; setup.theta.len()],
            theta_bar_powers: vec![1; setup.theta_bar.len()],
        },
        SuperspaceMeasure::Chiral => ThetaMonomial {
            theta_powers: vec![1; setup.theta.len()],
            theta_bar_powers: vec![0; setup.theta_bar.len()],
        },
        SuperspaceMeasure::AntiChiral => ThetaMonomial {
            theta_powers: vec![0; setup.theta.len()],
            theta_bar_powers: vec![1; setup.theta_bar.len()],
        },
    };
    crate::superspace::extract_component(expr, &target, setup, table, interner)
}

fn grassmann_derivative(
    expr: &Expr,
    variable: Spur,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    match expr {
        Expr::Sym(s) => {
            if *s == variable {
                Expr::one()
            } else {
                Expr::zero()
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| grassmann_derivative(term, variable, table, interner))
                .collect(),
        ),
        Expr::Mul(factors) => grassmann_derivative_product(factors, variable, table, interner),
        Expr::Neg(inner) => Expr::neg(grassmann_derivative(inner, variable, table, interner)),
        Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == BigInt::from(1)) => {
            grassmann_derivative(base, variable, table, interner)
        }
        Expr::Call(f, args) if contains_symbol_in_exprs(args, variable) => Expr::Call(
            interner.get_or_intern("dtheta"),
            vec![Expr::Sym(variable), Expr::Call(*f, args.clone())],
        ),
        Expr::Indexed(base, _) if contains_symbol(base, variable) => Expr::Call(
            interner.get_or_intern("dtheta"),
            vec![Expr::Sym(variable), expr.clone()],
        ),
        _ => Expr::zero(),
    }
}

fn contains_symbol_in_exprs(exprs: &[Expr], variable: Spur) -> bool {
    exprs.iter().any(|expr| contains_symbol(expr, variable))
}

fn contains_symbol(expr: &Expr, variable: Spur) -> bool {
    match expr {
        Expr::Sym(sym) => *sym == variable,
        Expr::Complex(re, im) => contains_symbol(re, variable) || contains_symbol(im, variable),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_symbol(term, variable))
        }
        Expr::Pow(base, exp) => contains_symbol(base, variable) || contains_symbol(exp, variable),
        Expr::Neg(inner) | Expr::Indexed(inner, _) => contains_symbol(inner, variable),
        Expr::Group(inner, _) => contains_symbol(inner, variable),
        Expr::Call(_, args) => contains_symbol_in_exprs(args, variable),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_symbol(cell, variable))),
        Expr::Let(_, value, body) => {
            contains_symbol(value, variable) || contains_symbol(body, variable)
        }
        Expr::FnDef(_, _, body) => contains_symbol(body, variable),
        Expr::Rule(lhs, rhs, _) => contains_symbol(lhs, variable) || contains_symbol(rhs, variable),
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, _)| contains_symbol(value, variable)),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => false,
    }
}

fn grassmann_derivative_product(
    factors: &[Expr],
    variable: Spur,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let mut terms = Vec::new();
    for (idx, factor) in factors.iter().enumerate() {
        let deriv = grassmann_derivative(factor, variable, table, interner);
        if is_zero(&deriv) {
            continue;
        }
        let sign = factors[..idx].iter().fold(1, |acc, prior| {
            acc * table
                .infer_grading(prior)
                .commutation_sign(&crate::Grading::fermionic())
        });
        let mut product = factors[..idx].to_vec();
        product.push(deriv);
        product.extend_from_slice(&factors[idx + 1..]);
        let term = crate::graded_multiply(&product, table, interner);
        terms.push(if sign < 0 { Expr::neg(term) } else { term });
    }
    Expr::add(terms)
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n == &BigInt::from(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superspace::setup_n1_superspace;

    #[test]
    fn theta_derivative_removes_theta() {
        let interner = Interner::new();
        let (setup, table) = setup_n1_superspace(&interner);
        let f = interner.get_or_intern("f");
        let expr = Expr::mul(vec![Expr::Sym(setup.theta[0]), Expr::Sym(f)]);
        let out = apply_d_alpha(&expr, 0, &setup, &table, &interner);
        assert!(format!("{out:?}").contains("Sym"));
    }

    #[test]
    fn full_superspace_integral_extracts_top_component() {
        let interner = Interner::new();
        let (setup, table) = setup_n1_superspace(&interner);
        let d = interner.get_or_intern("D");
        let expr = Expr::mul(vec![
            Expr::Sym(setup.theta[0]),
            Expr::Sym(setup.theta[1]),
            Expr::Sym(setup.theta_bar[0]),
            Expr::Sym(setup.theta_bar[1]),
            Expr::Sym(d),
        ]);
        assert_eq!(
            superspace_integrate(
                &expr,
                SuperspaceMeasure::FullSuperspace,
                &setup,
                &table,
                &interner,
            ),
            Expr::Sym(d)
        );
    }
}
