#![forbid(unsafe_code)]

use ax_ir::Expr;
use ax_tensor::SymbolicMatrix;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct DiffForm {
    pub degree: usize,
    pub dim: usize,
    pub components: BTreeMap<Vec<usize>, ax_ir::Expr>,
}

fn simplify_expr(expr: Expr, interner: &ax_ir::Interner) -> Expr {
    let _ = interner;
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .into_iter()
                .map(|term| simplify_expr(term, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .into_iter()
                .map(|factor| simplify_expr(factor, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(simplify_expr(*base, interner), simplify_expr(*exp, interner)),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner, interner)),
        Expr::Call(f, args) => Expr::Call(
            f,
            args.into_iter()
                .map(|arg| simplify_expr(arg, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(name, params, Box::new(simplify_expr(*body, interner)))
        }
        Expr::Rule(lhs, rhs) => Expr::Rule(
            Box::new(simplify_expr(*lhs, interner)),
            Box::new(simplify_expr(*rhs, interner)),
        ),
        Expr::Import(path) => Expr::Import(path),
        Expr::Assume(name, assumptions) => Expr::Assume(name, assumptions),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| (simplify_expr(value, interner), condition))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(Box::new(simplify_expr(*base, interner)), indices),
        Expr::Let(name, val, body) => Expr::Let(
            name,
            Box::new(simplify_expr(*val, interner)),
            Box::new(simplify_expr(*body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .into_iter()
                .map(|item| simplify_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.into_iter()
                .map(|row| row.into_iter().map(|cell| simplify_expr(cell, interner)).collect())
                .collect(),
        ),
        other => other,
    }
}

fn add_component(map: &mut BTreeMap<Vec<usize>, Expr>, key: Vec<usize>, value: Expr, interner: &ax_ir::Interner) {
    let value = simplify_expr(value, interner);
    if value == Expr::zero() {
        return;
    }
    map.entry(key)
        .and_modify(|existing| *existing = simplify_expr(Expr::add(vec![existing.clone(), value.clone()]), interner))
        .or_insert(value);
}

pub fn permutation_sign(perm: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

pub fn wedge(a: &DiffForm, b: &DiffForm, interner: &ax_ir::Interner) -> DiffForm {
    assert_eq!(a.dim, b.dim);
    let mut components = BTreeMap::new();

    for (ia, va) in &a.components {
        for (ib, vb) in &b.components {
            if ia.iter().any(|idx| ib.contains(idx)) {
                continue;
            }

            let mut merged = ia.clone();
            merged.extend(ib.iter().copied());
            let sign = permutation_sign(&merged);
            merged.sort_unstable();

            let mut term = Expr::mul(vec![va.clone(), vb.clone()]);
            if sign < 0 {
                term = Expr::neg(term);
            }
            add_component(&mut components, merged, term, interner);
        }
    }

    DiffForm {
        degree: a.degree + b.degree,
        dim: a.dim,
        components,
    }
}

pub fn exterior_derivative(
    form: &DiffForm,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> DiffForm {
    assert_eq!(form.dim, coords.len());
    let mut components = BTreeMap::new();

    for (basis, value) in &form.components {
        for (i, coord) in coords.iter().enumerate() {
            if basis.contains(&i) {
                continue;
            }

            let derivative = ax_tensor::diff_component(value, *coord, interner);
            if derivative == Expr::zero() {
                continue;
            }

            let position = basis.partition_point(|idx| *idx < i);
            let mut new_basis = basis.clone();
            new_basis.insert(position, i);

            let term = if position % 2 == 0 {
                derivative
            } else {
                Expr::neg(derivative)
            };
            add_component(&mut components, new_basis, term, interner);
        }
    }

    DiffForm {
        degree: form.degree + 1,
        dim: form.dim,
        components,
    }
}

pub fn hodge_dual(
    form: &DiffForm,
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> DiffForm {
    assert_eq!(form.dim, g.dim);

    let ginv = g.symbolic_inverse(interner);
    let det_g = Expr::mul((0..g.dim).map(|i| g.get(i, i).clone()).collect());
    let sqrt_abs_det_g = Expr::Call(
        interner.get_or_intern("sqrt"),
        vec![Expr::Call(interner.get_or_intern("abs"), vec![det_g])],
    );

    let mut components = BTreeMap::new();
    for (basis, value) in &form.components {
        let complement = (0..form.dim)
            .filter(|idx| !basis.contains(idx))
            .collect::<Vec<_>>();
        let mut perm = basis.clone();
        perm.extend(complement.iter().copied());
        let sign = permutation_sign(&perm);

        let mut factors = vec![sqrt_abs_det_g.clone(), value.clone()];
        for idx in basis {
            factors.push(ginv.get(*idx, *idx).clone());
        }

        let k_fact = (1..=form.degree).fold(BigInt::one(), |acc, n| acc * BigInt::from(n as i64));
        if k_fact != BigInt::one() {
            factors.push(Expr::Rational(BigRational::new(BigInt::one(), k_fact)));
        }

        let mut term = Expr::mul(factors);
        if sign < 0 {
            term = Expr::neg(term);
        }
        add_component(&mut components, complement, term, interner);
    }

    DiffForm {
        degree: form.dim - form.degree,
        dim: form.dim,
        components,
    }
}

pub fn one_form_from_expr(expr: &Expr) -> Option<DiffForm> {
    let Expr::List(items) = expr else {
        return None;
    };
    let mut components = BTreeMap::new();
    for (i, item) in items.iter().enumerate() {
        if *item != Expr::zero() {
            components.insert(vec![i], item.clone());
        }
    }
    Some(DiffForm {
        degree: 1,
        dim: items.len(),
        components,
    })
}

pub fn two_form_from_expr(expr: &Expr) -> Option<DiffForm> {
    let Expr::Matrix(rows) = expr else {
        return None;
    };
    let n = rows.len();
    if rows.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut components = BTreeMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if rows[i][j] != Expr::zero() {
                components.insert(vec![i, j], rows[i][j].clone());
            }
        }
    }

    Some(DiffForm {
        degree: 2,
        dim: n,
        components,
    })
}

pub fn scalar_form(expr: &Expr, dim: usize) -> DiffForm {
    let mut components = BTreeMap::new();
    if *expr != Expr::zero() {
        components.insert(vec![], expr.clone());
    }
    DiffForm {
        degree: 0,
        dim,
        components,
    }
}

pub fn form_to_expr(form: &DiffForm) -> Expr {
    match form.degree {
        0 => form.components.get(&Vec::new()).cloned().unwrap_or_else(Expr::zero),
        1 => {
            let mut items = vec![Expr::zero(); form.dim];
            for (basis, value) in &form.components {
                if let Some(idx) = basis.first() {
                    items[*idx] = value.clone();
                }
            }
            Expr::List(items)
        }
        2 => {
            let mut rows = vec![vec![Expr::zero(); form.dim]; form.dim];
            for (basis, value) in &form.components {
                if basis.len() == 2 {
                    let i = basis[0];
                    let j = basis[1];
                    rows[i][j] = value.clone();
                    rows[j][i] = Expr::neg(value.clone());
                }
            }
            Expr::Matrix(rows)
        }
        _ => Expr::List(
            form.components
                .iter()
                .map(|(basis, value)| {
                    Expr::List(vec![
                        Expr::List(
                            basis.iter()
                                .map(|idx| Expr::Int((*idx as i64).into()))
                                .collect(),
                        ),
                        value.clone(),
                    ])
                })
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exterior_d_of_scalar() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let coords = vec![x, y];
        let f = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::pow(Expr::Sym(y), Expr::Int(2.into())),
        ]);
        let form = DiffForm {
            degree: 0,
            dim: 2,
            components: {
                let mut m = BTreeMap::new();
                m.insert(vec![], f);
                m
            },
        };
        let df = exterior_derivative(&form, &coords, &interner);
        assert_eq!(df.degree, 1);
        assert_eq!(df.components.len(), 2);
    }

    #[test]
    fn dd_is_zero() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let coords = vec![x, y];
        let f = Expr::mul(vec![Expr::Sym(x), Expr::Sym(y)]);
        let form0 = DiffForm {
            degree: 0,
            dim: 2,
            components: {
                let mut m = BTreeMap::new();
                m.insert(vec![], f);
                m
            },
        };
        let df = exterior_derivative(&form0, &coords, &interner);
        let ddf = exterior_derivative(&df, &coords, &interner);
        for val in ddf.components.values() {
            let simplified = simplify_expr(val.clone(), &interner);
            assert_eq!(simplified, Expr::zero(), "d(d(f)) component = {:?}", val);
        }
    }

    #[test]
    fn wedge_anticommutative() {
        let interner = ax_ir::Interner::new();
        let mut dx = DiffForm {
            degree: 1,
            dim: 2,
            components: BTreeMap::new(),
        };
        dx.components.insert(vec![0], Expr::one());
        let mut dy = DiffForm {
            degree: 1,
            dim: 2,
            components: BTreeMap::new(),
        };
        dy.components.insert(vec![1], Expr::one());

        let dxdy = wedge(&dx, &dy, &interner);
        let dydx = wedge(&dy, &dx, &interner);

        assert_eq!(
            *dxdy.components.get(&vec![0, 1]).unwrap_or(&Expr::zero()),
            Expr::one()
        );
        assert_eq!(
            *dydx.components.get(&vec![0, 1]).unwrap_or(&Expr::zero()),
            Expr::neg(Expr::one())
        );
    }
}
