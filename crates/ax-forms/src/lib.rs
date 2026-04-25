#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use ax_ir::{
    DualityKind, Expr, RestrictedSymmetryMode, SymmetrySource, TableauAttachment, TensorSymmetry,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicMatrix {
    pub dim: usize,
    pub data: Vec<Vec<ax_ir::Expr>>,
}

impl SymbolicMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![vec![Expr::zero(); dim]; dim],
        }
    }

    pub fn from_diagonal(diag: Vec<ax_ir::Expr>) -> Self {
        let dim = diag.len();
        let mut matrix = Self::new(dim);
        for (i, value) in diag.into_iter().enumerate() {
            matrix.data[i][i] = value;
        }
        matrix
    }

    pub fn get(&self, row: usize, col: usize) -> &ax_ir::Expr {
        &self.data[row][col]
    }

    pub fn symbolic_inverse(&self, interner: &ax_ir::Interner) -> Self {
        let is_diagonal = (0..self.dim)
            .all(|row| (0..self.dim).all(|col| row == col || self.data[row][col] == Expr::zero()));

        if is_diagonal {
            let mut inverse = Self::new(self.dim);
            for i in 0..self.dim {
                inverse.data[i][i] = simplify_expr(
                    Expr::pow(self.data[i][i].clone(), Expr::Int((-1).into())),
                    interner,
                );
            }
            return inverse;
        }

        match ax_linalg::inverse(&self.data, interner) {
            Some(inv_data) => Self {
                dim: self.dim,
                data: inv_data
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|cell| simplify_expr(cell, interner))
                            .collect()
                    })
                    .collect(),
            },
            None => panic!("metric tensor is singular (determinant is zero)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffForm {
    pub degree: usize,
    pub dim: usize,
    pub components: BTreeMap<Vec<usize>, ax_ir::Expr>,
}

pub fn k_form_tableau(rank: usize) -> TensorSymmetry {
    TensorSymmetry {
        tableaux: vec![TableauAttachment {
            shape: vec![1; rank],
            slot_map: (0..rank).collect(),
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: DualityKind::None,
            restricted_mode: RestrictedSymmetryMode::FullYoung,
            trace_free: false,
            dimension_guard: None,
            source: SymmetrySource::Declared,
            label: None,
        }],
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    }
}

pub fn hodge_dual_rank(rank: usize, dim: usize) -> usize {
    ax_young::duality::hodge_dual_form_degree(rank, dim)
}

pub fn middle_degree_selfdual_symmetry(dim: usize) -> anyhow::Result<TensorSymmetry> {
    if dim % 2 != 0 {
        anyhow::bail!("middle-degree selfdual symmetry requires even dimension");
    }
    Ok(ax_young::induced_form_tableau_duality(
        dim / 2,
        dim,
        DualityKind::SelfDual,
    )?)
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
        Expr::Mul(factors) => Expr::mul({
            let mut simplified = factors
                .into_iter()
                .map(|factor| simplify_expr(factor, interner))
                .collect::<Vec<_>>();
            simplified.sort_by_key(|expr| format!("{expr:?}"));
            simplified
        }),
        Expr::Pow(base, exp) => Expr::pow(
            simplify_expr(*base, interner),
            simplify_expr(*exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner, interner)),
        Expr::Call(f, args) => {
            let args = args
                .into_iter()
                .map(|arg| simplify_expr(arg, interner))
                .collect::<Vec<_>>();
            let abs = interner.get_or_intern("abs");
            let sqrt = interner.get_or_intern("sqrt");
            if f == abs && args.len() == 1 {
                match &args[0] {
                    Expr::Int(n) if *n >= BigInt::from(0) => return args[0].clone(),
                    Expr::Int(n) => return Expr::Int(-n.clone()),
                    _ => {}
                }
            }
            if f == sqrt && args.len() == 1 && args[0] == Expr::one() {
                return Expr::one();
            }
            Expr::Call(f, args)
        }
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(name, params, Box::new(simplify_expr(*body, interner)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(simplify_expr(*lhs, interner)),
            Box::new(simplify_expr(*rhs, interner)),
            trust,
        ),
        Expr::Import(path) => Expr::Import(path),
        Expr::Assume(name, assumptions) => Expr::Assume(name, assumptions),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .into_iter()
                .map(|(value, condition)| (simplify_expr(value, interner), condition))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(simplify_expr(*base, interner)), indices)
        }
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
                .map(|row| {
                    row.into_iter()
                        .map(|cell| simplify_expr(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        other => other,
    }
}

pub fn diff_component(
    expr: &ax_ir::Expr,
    coord: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
            Expr::Complex(re, im) => contains_var(re, var) || contains_var(im, var),
            Expr::Sym(s) => *s == var,
            Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
                items.iter().any(|item| contains_var(item, var))
            }
            Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
            Expr::Neg(e) => contains_var(e, var),
            Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
            Expr::FnDef(_, _, body) => contains_var(body, var),
            Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
            Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => false,
            Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_var(value, var)),
            Expr::Indexed(base, _) => contains_var(base, var),
            Expr::Group(inner, _) => contains_var(inner, var),
            Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
            Expr::Matrix(rows) => rows
                .iter()
                .any(|row| row.iter().any(|cell| contains_var(cell, var))),
        }
    }

    fn one_half() -> Expr {
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    }

    fn diff(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
            Expr::Complex(re, im) => Expr::Complex(
                Box::new(diff(re, var, interner)),
                Box::new(diff(im, var, interner)),
            ),
            Expr::Sym(s) => {
                if *s == var {
                    Expr::one()
                } else {
                    Expr::zero()
                }
            }
            Expr::Add(terms) => Expr::add(terms.iter().map(|t| diff(t, var, interner)).collect()),
            Expr::Neg(e) => Expr::neg(diff(e, var, interner)),
            Expr::Mul(factors) => Expr::add(
                factors
                    .iter()
                    .enumerate()
                    .map(|(i, factor)| {
                        let mut product = Vec::with_capacity(factors.len());
                        product.extend(factors[..i].iter().cloned());
                        product.push(diff(factor, var, interner));
                        product.extend(factors[i + 1..].iter().cloned());
                        Expr::mul(product)
                    })
                    .collect(),
            ),
            Expr::Pow(base, exp) if !contains_var(exp, var) => Expr::mul(vec![
                exp.as_ref().clone(),
                Expr::pow(
                    base.as_ref().clone(),
                    Expr::add(vec![exp.as_ref().clone(), Expr::neg(Expr::one())]),
                ),
                diff(base, var, interner),
            ]),
            Expr::Pow(_, _) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
            Expr::Call(f, args) if args.len() == 1 => match interner.resolve(*f) {
                "sin" => Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("cos"), args.clone()),
                    diff(&args[0], var, interner),
                ]),
                "cos" => Expr::mul(vec![
                    Expr::neg(Expr::Call(interner.get_or_intern("sin"), args.clone())),
                    diff(&args[0], var, interner),
                ]),
                "exp" => Expr::mul(vec![
                    Expr::Call(interner.get_or_intern("exp"), args.clone()),
                    diff(&args[0], var, interner),
                ]),
                "log" => Expr::mul(vec![
                    Expr::pow(args[0].clone(), Expr::neg(Expr::one())),
                    diff(&args[0], var, interner),
                ]),
                "sqrt" => diff(&Expr::pow(args[0].clone(), one_half()), var, interner),
                _ if !contains_var(&args[0], var) => Expr::zero(),
                _ => Expr::Call(
                    interner.get_or_intern("diff"),
                    vec![expr.clone(), Expr::Sym(var)],
                ),
            },
            Expr::Call(_, args) if args.iter().all(|arg| !contains_var(arg, var)) => Expr::zero(),
            Expr::Call(_, _) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
            Expr::FnDef(_, _, _)
            | Expr::Rule(_, _, _)
            | Expr::Import(_)
            | Expr::Assume(_, _)
            | Expr::SetConvention(_, _)
            | Expr::Piecewise(_)
            | Expr::Indexed(_, _)
            | Expr::Group(_, _)
            | Expr::Let(_, _, _)
            | Expr::List(_)
            | Expr::Matrix(_) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
        }
    }

    simplify_expr(diff(expr, coord, interner), interner)
}

fn add_component(
    map: &mut BTreeMap<Vec<usize>, Expr>,
    key: Vec<usize>,
    value: Expr,
    interner: &ax_ir::Interner,
) {
    let value = simplify_expr(value, interner);
    if value == Expr::zero() {
        return;
    }
    map.entry(key)
        .and_modify(|existing| {
            *existing = simplify_expr(Expr::add(vec![existing.clone(), value.clone()]), interner)
        })
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
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
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

            let derivative = diff_component(value, *coord, interner);
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

pub fn hodge_dual(form: &DiffForm, g: &SymbolicMatrix, interner: &ax_ir::Interner) -> DiffForm {
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
    match expr {
        Expr::List(items)
            if items
                .iter()
                .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2)) =>
        {
            let generic = form_from_expr(expr)?;
            (generic.degree == 1).then_some(generic)
        }
        Expr::List(items) => {
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
        _ => None,
    }
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

pub fn form_from_expr(expr: &Expr) -> Option<DiffForm> {
    match expr {
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Add(_)
        | Expr::Mul(_)
        | Expr::Pow(_, _)
        | Expr::Neg(_)
        | Expr::Call(_, _)
        | Expr::Group(_, _) => Some(scalar_form(expr, 0)),
        Expr::List(items)
            if items
                .iter()
                .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2)) =>
        {
            let mut components = BTreeMap::new();
            let mut degree = None;
            let mut max_idx = None;
            for item in items {
                let Expr::List(pair) = item else {
                    return None;
                };
                let basis = match &pair[0] {
                    Expr::List(indices) => indices
                        .iter()
                        .map(|idx| match idx {
                            Expr::Int(n) => num_traits::ToPrimitive::to_usize(n),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?,
                    _ => return None,
                };
                if let Some(existing_degree) = degree {
                    if existing_degree != basis.len() {
                        return None;
                    }
                } else {
                    degree = Some(basis.len());
                }
                if basis.windows(2).any(|window| window[0] >= window[1]) {
                    return None;
                }
                if let Some(last) = basis.last() {
                    max_idx = Some(max_idx.map_or(*last, |current: usize| current.max(*last)));
                }
                let value = pair[1].clone();
                if value != Expr::zero() {
                    components.insert(basis, value);
                }
            }
            Some(DiffForm {
                degree: degree.unwrap_or(0),
                dim: max_idx.map_or(0, |idx| idx + 1),
                components,
            })
        }
        Expr::List(items) => one_form_from_expr(expr).or_else(|| {
            Some(DiffForm {
                degree: 1,
                dim: items.len(),
                components: items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| **item != Expr::zero())
                    .map(|(i, item)| (vec![i], item.clone()))
                    .collect(),
            })
        }),
        Expr::Matrix(_) => two_form_from_expr(expr),
        _ => None,
    }
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

pub fn resize_form(form: &DiffForm, dim: usize) -> DiffForm {
    assert!(dim >= form.dim);
    DiffForm {
        degree: form.degree,
        dim,
        components: form.components.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaPairingTerm {
    pub pairings: Vec<(usize, usize)>,
    pub coefficient: i64,
}

fn permutation_parity(perm: &[usize]) -> i64 {
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

fn next_permutation(values: &mut [usize]) -> bool {
    if values.len() < 2 {
        return false;
    }
    let mut pivot = values.len() - 2;
    while values[pivot] >= values[pivot + 1] {
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut swap_with = values.len() - 1;
    while values[swap_with] <= values[pivot] {
        swap_with -= 1;
    }
    values.swap(pivot, swap_with);
    values[pivot + 1..].reverse();
    true
}

pub fn epsilon_pairing_rank(rank: usize) -> anyhow::Result<Vec<DeltaPairingTerm>> {
    if rank == 0 {
        return Ok(vec![DeltaPairingTerm {
            pairings: Vec::new(),
            coefficient: 1,
        }]);
    }

    let mut permutation: Vec<usize> = (0..rank).collect();
    let mut terms = Vec::new();
    loop {
        terms.push(DeltaPairingTerm {
            pairings: (0..rank).map(|slot| (slot, permutation[slot])).collect(),
            coefficient: permutation_parity(&permutation),
        });
        if !next_permutation(&mut permutation) {
            break;
        }
    }
    terms.sort_by(|left, right| left.pairings.cmp(&right.pairings));
    Ok(terms)
}

pub fn interior_product(vector: &[Expr], form: &DiffForm, interner: &ax_ir::Interner) -> DiffForm {
    assert_eq!(vector.len(), form.dim);
    if form.degree == 0 {
        return scalar_form(&Expr::zero(), form.dim);
    }
    let mut components = BTreeMap::new();
    for (basis, value) in &form.components {
        for (pos, idx) in basis.iter().enumerate() {
            let vector_component = vector[*idx].clone();
            if vector_component == Expr::zero() {
                continue;
            }
            let mut reduced_basis = basis.clone();
            reduced_basis.remove(pos);
            let mut term = Expr::mul(vec![vector_component, value.clone()]);
            if pos % 2 == 1 {
                term = Expr::neg(term);
            }
            add_component(&mut components, reduced_basis, term, interner);
        }
    }
    DiffForm {
        degree: form.degree.saturating_sub(1),
        dim: form.dim,
        components,
    }
}

pub fn codifferential(
    form: &DiffForm,
    g: &SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> DiffForm {
    let star = hodge_dual(form, g, interner);
    let d_star = exterior_derivative(&star, coords, interner);
    let mut result = hodge_dual(&d_star, g, interner);
    let exponent = (form.dim * (form.degree + 1) + 1) % 2;
    if exponent == 1 {
        result.components = result
            .components
            .into_iter()
            .map(|(basis, value)| (basis, Expr::neg(value)))
            .collect();
    }
    result
}

pub fn lie_derivative_form(
    vector: &[Expr],
    form: &DiffForm,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> DiffForm {
    let interior_then_d = interior_product(
        vector,
        &exterior_derivative(form, coords, interner),
        interner,
    );
    let d_then_interior =
        exterior_derivative(&interior_product(vector, form, interner), coords, interner);
    let mut components = interior_then_d.components.clone();
    for (basis, value) in d_then_interior.components {
        add_component(&mut components, basis, value, interner);
    }
    DiffForm {
        degree: form.degree,
        dim: form.dim,
        components,
    }
}

pub fn form_to_expr(form: &DiffForm) -> Expr {
    match form.degree {
        0 => form
            .components
            .get(&Vec::new())
            .cloned()
            .unwrap_or_else(Expr::zero),
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
                            basis
                                .iter()
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

/// Build a symbolic one-form component placeholder for `coefficient * d(symbol)`.
pub fn one_form_component(
    symbol: ax_ir::Expr,
    coefficient: ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    Expr::Call(
        interner.get_or_intern("one_form_component"),
        vec![symbol, coefficient],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_degree_selfdual_symmetry_is_exact_in_four_dimensions() {
        let symmetry = middle_degree_selfdual_symmetry(4).unwrap();
        assert_eq!(symmetry.tableaux.len(), 1);
        assert_eq!(symmetry.tableaux[0].shape, vec![1, 1]);
        assert_eq!(symmetry.tableaux[0].slot_map, vec![0, 1]);
        assert_eq!(symmetry.tableaux[0].duality, DualityKind::SelfDual);
    }

    #[test]
    fn middle_degree_selfdual_symmetry_rejects_odd_dimension() {
        assert!(middle_degree_selfdual_symmetry(3).is_err());
    }

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

    #[test]
    fn parse_generic_three_form_round_trip() {
        let expr = Expr::List(vec![Expr::List(vec![
            Expr::List(vec![
                Expr::Int(0.into()),
                Expr::Int(1.into()),
                Expr::Int(2.into()),
            ]),
            Expr::one(),
        ])]);
        let form = form_from_expr(&expr).expect("generic form");
        assert_eq!(form.degree, 3);
        assert_eq!(form.dim, 3);
        assert_eq!(form_to_expr(&form), expr);
    }

    #[test]
    fn interior_product_reduces_degree() {
        let interner = ax_ir::Interner::new();
        let vector = vec![Expr::one(), Expr::zero(), Expr::zero()];
        let form = DiffForm {
            degree: 2,
            dim: 3,
            components: BTreeMap::from([(vec![0, 2], Expr::one())]),
        };
        let result = interior_product(&vector, &form, &interner);
        assert_eq!(result.degree, 1);
        assert_eq!(result.components.get(&vec![2]), Some(&Expr::one()));
    }

    #[test]
    fn codifferential_of_constant_one_form_is_zero() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let metric = SymbolicMatrix::from_diagonal(vec![Expr::one(), Expr::one()]);
        let form = DiffForm {
            degree: 1,
            dim: 2,
            components: BTreeMap::from([(vec![0], Expr::one())]),
        };
        let delta = codifferential(&form, &metric, &[x, y], &interner);
        assert!(delta
            .components
            .values()
            .all(|value| *value == Expr::zero()));
    }

    #[test]
    fn lie_derivative_form_matches_cartan_on_constant_vector() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let omega = DiffForm {
            degree: 1,
            dim: 2,
            components: BTreeMap::from([(vec![0], Expr::Sym(x)), (vec![1], Expr::zero())]),
        };
        let result = lie_derivative_form(&[Expr::one(), Expr::zero()], &omega, &[x, y], &interner);
        assert_eq!(
            form_to_expr(&result),
            Expr::List(vec![Expr::one(), Expr::zero()])
        );
    }

    #[test]
    fn epsilon_pairing_rank_two_matches_tensor_engine() {
        assert_eq!(
            epsilon_pairing_rank(2).unwrap(),
            vec![
                DeltaPairingTerm {
                    pairings: vec![(0, 0), (1, 1)],
                    coefficient: 1,
                },
                DeltaPairingTerm {
                    pairings: vec![(0, 1), (1, 0)],
                    coefficient: -1,
                },
            ]
        );
    }

    #[test]
    fn one_form_component_builds_symbolic_one_form_placeholder() {
        let interner = ax_ir::Interner::new();
        let theta = Expr::Sym(interner.get_or_intern("theta"));
        let coeff = Expr::Sym(interner.get_or_intern("A"));
        assert_eq!(
            one_form_component(theta.clone(), coeff.clone(), &interner),
            Expr::Call(
                interner.get_or_intern("one_form_component"),
                vec![theta, coeff]
            )
        );
    }
}
