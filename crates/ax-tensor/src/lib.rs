#![forbid(unsafe_code)]

use ax_ir::{Expr, Interner};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
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

    pub fn set(&mut self, row: usize, col: usize, val: ax_ir::Expr) {
        self.data[row][col] = val;
    }

    pub fn symbolic_inverse(&self, interner: &ax_ir::Interner) -> Self {
        let is_diagonal = (0..self.dim).all(|row| {
            (0..self.dim).all(|col| row == col || self.data[row][col] == Expr::zero())
        });

        if is_diagonal {
            let mut inverse = Self::new(self.dim);
            for i in 0..self.dim {
                inverse.data[i][i] =
                    simplify_expr(Expr::pow(self.data[i][i].clone(), Expr::Int((-1).into())), interner);
            }
            return inverse;
        }

        match ax_linalg::inverse(&self.data, interner) {
            Some(inv_data) => {
                let mut result = Self::new(self.dim);
                for (i, row) in inv_data.iter().enumerate().take(self.dim) {
                    for (j, cell) in row.iter().enumerate().take(self.dim) {
                        result.data[i][j] = simplify_expr(cell.clone(), interner);
                    }
                }
                result
            }
            None => panic!("metric tensor is singular (determinant is zero)"),
        }
    }
}

pub fn detect_contractions(indices: &[ax_ir::Index]) -> Vec<(usize, usize)> {
    let mut used = HashSet::new();
    let mut pairs = Vec::new();

    for i in 0..indices.len() {
        if used.contains(&i) {
            continue;
        }
        for j in (i + 1)..indices.len() {
            if used.contains(&j) {
                continue;
            }
            if indices[i].name == indices[j].name
                && indices[i].variance != indices[j].variance
                && (indices[i].index_type == indices[j].index_type
                    || indices[i].index_type.is_none()
                    || indices[j].index_type.is_none())
            {
                used.insert(i);
                used.insert(j);
                pairs.push((i, j));
                break;
            }
        }
    }

    pairs
}

fn permutation_parity(original: &[ax_ir::Index], sorted: &[ax_ir::Index]) -> bool {
    let mut used = vec![false; sorted.len()];
    let mut permutation = Vec::with_capacity(original.len());
    for item in original {
        let pos = sorted
            .iter()
            .enumerate()
            .find(|(idx, candidate)| !used[*idx] && *candidate == item)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        used[pos] = true;
        permutation.push(pos);
    }

    let mut inversions = 0usize;
    for i in 0..permutation.len() {
        for j in (i + 1)..permutation.len() {
            if permutation[i] > permutation[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 1
}

fn sort_key(index: &ax_ir::Index, interner: &Interner) -> (String, u8) {
    (
        interner.resolve(index.name).to_string(),
        match index.variance {
            ax_ir::Variance::Down => 0,
            ax_ir::Variance::Up => 1,
        },
    )
}

pub fn canonicalize_indices(
    expr: &ax_ir::Expr,
    properties: &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let base_expr = canonicalize_indices(base, properties, interner);
            let mut indices = indices.clone();
            let mut negate = false;

            if let Expr::Sym(sym) = &base_expr {
                if let Some(props) = properties.get(sym) {
                    for prop in props {
                        match prop {
                            ax_ir::TensorProperty::Symmetric(positions) => {
                                let mut original = positions
                                    .iter()
                                    .filter_map(|&pos| indices.get(pos).cloned())
                                    .collect::<Vec<_>>();
                                let mut sorted = original.clone();
                                sorted.sort_by_key(|idx| sort_key(idx, interner));
                                for (slot, value) in positions.iter().zip(sorted.iter()) {
                                    if let Some(target) = indices.get_mut(*slot) {
                                        *target = value.clone();
                                    }
                                }
                                original.clear();
                            }
                            ax_ir::TensorProperty::AntiSymmetric(positions) => {
                                let original = positions
                                    .iter()
                                    .filter_map(|&pos| indices.get(pos).cloned())
                                    .collect::<Vec<_>>();
                                if original.len() >= 2 {
                                    for i in 0..original.len() {
                                        for j in (i + 1)..original.len() {
                                            if original[i] == original[j] {
                                                return Expr::zero();
                                            }
                                        }
                                    }
                                }
                                let mut sorted = original.clone();
                                sorted.sort_by_key(|idx| sort_key(idx, interner));
                                if permutation_parity(&original, &sorted) {
                                    negate = !negate;
                                }
                                for (slot, value) in positions.iter().zip(sorted.iter()) {
                                    if let Some(target) = indices.get_mut(*slot) {
                                        *target = value.clone();
                                    }
                                }
                            }
                            ax_ir::TensorProperty::RiemannSymmetry => {
                                if indices.len() == 4 {
                                    let pairs = [vec![0usize, 1usize], vec![2usize, 3usize]];
                                    for positions in pairs {
                                        let original = positions
                                            .iter()
                                            .filter_map(|&pos| indices.get(pos).cloned())
                                            .collect::<Vec<_>>();
                                        if original[0] == original[1] {
                                            return Expr::zero();
                                        }
                                        let mut sorted = original.clone();
                                        sorted.sort_by_key(|idx| sort_key(idx, interner));
                                        if permutation_parity(&original, &sorted) {
                                            negate = !negate;
                                        }
                                        for (slot, value) in positions.iter().zip(sorted.iter()) {
                                            indices[*slot] = value.clone();
                                        }
                                    }

                                    let left = vec![indices[0].clone(), indices[1].clone()];
                                    let right = vec![indices[2].clone(), indices[3].clone()];
                                    let left_key = left.iter().map(|i| sort_key(i, interner)).collect::<Vec<_>>();
                                    let right_key = right.iter().map(|i| sort_key(i, interner)).collect::<Vec<_>>();
                                    if right_key < left_key {
                                        indices.swap(0, 2);
                                        indices.swap(1, 3);
                                    }
                                }
                            }
                            ax_ir::TensorProperty::Traceless
                            | ax_ir::TensorProperty::Metric
                            | ax_ir::TensorProperty::InverseMetric
                            | ax_ir::TensorProperty::KroneckerDelta
                            | ax_ir::TensorProperty::EpsilonTensor
                            | ax_ir::TensorProperty::Derivative
                            | ax_ir::TensorProperty::PartialDerivative
                            | ax_ir::TensorProperty::CovariantDerivative
                            | ax_ir::TensorProperty::Depends(_) => {}
                        }
                    }
                }
            }

            let out = Expr::Indexed(Box::new(base_expr), indices);
            if negate { Expr::neg(out) } else { out }
        }
        Expr::Add(terms) => Expr::add(
            terms.iter()
                .map(|term| canonicalize_indices(term, properties, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| canonicalize_indices(factor, properties, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            canonicalize_indices(base, properties, interner),
            canonicalize_indices(exp, properties, interner),
        ),
        Expr::Neg(inner) => Expr::neg(canonicalize_indices(inner, properties, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| canonicalize_indices(arg, properties, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(canonicalize_indices(re, properties, interner)),
            Box::new(canonicalize_indices(im, properties, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(canonicalize_indices(body, properties, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(canonicalize_indices(lhs, properties, interner)),
            Box::new(canonicalize_indices(rhs, properties, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, cond)| (canonicalize_indices(value, properties, interner), cond.clone()))
                .collect(),
        ),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(canonicalize_indices(val, properties, interner)),
            Box::new(canonicalize_indices(body, properties, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| canonicalize_indices(item, properties, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| canonicalize_indices(cell, properties, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn rename_dummy_indices(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> ax_ir::Expr {
    fn collect(
        expr: &Expr,
        seen: &mut Vec<(lasso::Spur, ax_ir::Variance)>,
        counts: &mut HashMap<lasso::Spur, Vec<ax_ir::Variance>>,
    ) {
        match expr {
            Expr::Indexed(_, indices) => {
                for index in indices {
                    seen.push((index.name, index.variance.clone()));
                    counts.entry(index.name).or_default().push(index.variance.clone());
                }
            }
            Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
                for item in items {
                    collect(item, seen, counts);
                }
            }
            Expr::Pow(base, exp) => {
                collect(base, seen, counts);
                collect(exp, seen, counts);
            }
            Expr::Neg(inner) => collect(inner, seen, counts),
            Expr::Call(_, args) => {
                for arg in args {
                    collect(arg, seen, counts);
                }
            }
            Expr::Complex(re, im) => {
                collect(re, seen, counts);
                collect(im, seen, counts);
            }
            Expr::FnDef(_, _, body) => collect(body, seen, counts),
            Expr::Rule(lhs, rhs, _) => {
                collect(lhs, seen, counts);
                collect(rhs, seen, counts);
            }
            Expr::Piecewise(cases) => {
                for (value, _) in cases {
                    collect(value, seen, counts);
                }
            }
            Expr::SetConvention(_, _) => {}
            Expr::Let(_, val, body) => {
                collect(val, seen, counts);
                collect(body, seen, counts);
            }
            Expr::Matrix(rows) => {
                for cell in rows.iter().flatten() {
                    collect(cell, seen, counts);
                }
            }
            _ => {}
        }
    }

    fn replace(expr: &Expr, mapping: &HashMap<lasso::Spur, lasso::Spur>) -> Expr {
        match expr {
            Expr::Indexed(base, indices) => Expr::Indexed(
                Box::new(replace(base, mapping)),
                indices
                    .iter()
                    .map(|index| ax_ir::Index {
                        name: mapping.get(&index.name).copied().unwrap_or(index.name),
                        variance: index.variance.clone(),
                        index_type: index.index_type,
                    })
                    .collect(),
            ),
            Expr::Add(items) => Expr::add(items.iter().map(|item| replace(item, mapping)).collect()),
            Expr::Mul(items) => Expr::mul(items.iter().map(|item| replace(item, mapping)).collect()),
            Expr::Pow(base, exp) => Expr::pow(replace(base, mapping), replace(exp, mapping)),
            Expr::Neg(inner) => Expr::neg(replace(inner, mapping)),
            Expr::Call(f, args) => Expr::Call(*f, args.iter().map(|arg| replace(arg, mapping)).collect()),
            Expr::Complex(re, im) => Expr::Complex(Box::new(replace(re, mapping)), Box::new(replace(im, mapping))),
            Expr::FnDef(name, params, body) => Expr::FnDef(*name, params.clone(), Box::new(replace(body, mapping))),
            Expr::Rule(lhs, rhs, trust) => Expr::Rule(
                Box::new(replace(lhs, mapping)),
                Box::new(replace(rhs, mapping)),
                *trust,
            ),
            Expr::Piecewise(cases) => Expr::Piecewise(
                cases.iter().map(|(value, cond)| (replace(value, mapping), cond.clone())).collect(),
            ),
            Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
            Expr::Let(name, val, body) => Expr::Let(*name, Box::new(replace(val, mapping)), Box::new(replace(body, mapping))),
            Expr::List(items) => Expr::List(items.iter().map(|item| replace(item, mapping)).collect()),
            Expr::Matrix(rows) => Expr::Matrix(rows.iter().map(|row| row.iter().map(|cell| replace(cell, mapping)).collect()).collect()),
            _ => expr.clone(),
        }
    }

    let mut seen = Vec::new();
    let mut counts = HashMap::new();
    collect(expr, &mut seen, &mut counts);

    let mut mapping = HashMap::new();
    let mut next = 0usize;
    for (sym, _) in seen {
        if mapping.contains_key(&sym) {
            continue;
        }
        if let Some(vars) = counts.get(&sym) {
            if vars.len() == 2 && vars[0] != vars[1] {
                let canonical = interner.get_or_intern(&format!("_d{next}"));
                mapping.insert(sym, canonical);
                next += 1;
            }
        }
    }

    replace(expr, &mapping)
}

fn contains_index(expr: &Expr, idx: lasso::Spur) -> bool {
    match expr {
        Expr::Indexed(_, indices) => indices.iter().any(|i| i.name == idx),
        Expr::Mul(factors) => factors.iter().any(|f| contains_index(f, idx)),
        Expr::Add(terms) => terms.iter().any(|t| contains_index(t, idx)),
        Expr::Neg(e) => contains_index(e, idx),
        _ => false,
    }
}

fn replace_index(expr: &Expr, from: lasso::Spur, to: lasso::Spur) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    if idx.name == from {
                        ax_ir::Index {
                            name: to,
                            variance: idx.variance.clone(),
                            index_type: idx.index_type,
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();
            Expr::Indexed(base.clone(), new_indices)
        }
        Expr::Mul(factors) => {
            Expr::mul(factors.iter().map(|f| replace_index(f, from, to)).collect())
        }
        Expr::Add(terms) => Expr::add(terms.iter().map(|t| replace_index(t, from, to)).collect()),
        Expr::Neg(e) => Expr::neg(replace_index(e, from, to)),
        _ => expr.clone(),
    }
}

pub fn eliminate_kronecker(
    expr: &ax_ir::Expr,
    delta_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let mut remaining = factors.clone();
            let mut changed = true;
            while changed {
                changed = false;
                for i in 0..remaining.len() {
                    let delta_info = match &remaining[i] {
                        Expr::Indexed(base, indices) => match base.as_ref() {
                            Expr::Sym(s) if *s == delta_sym && indices.len() == 2 => Some((
                                indices[0].name,
                                indices[0].variance.clone(),
                                indices[1].name,
                                indices[1].variance.clone(),
                            )),
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some((left_name, left_var, right_name, right_var)) = delta_info else {
                        continue;
                    };

                    let (from, to) = if left_var == ax_ir::Variance::Up
                        && right_var == ax_ir::Variance::Down
                    {
                        (right_name, left_name)
                    } else if left_var == ax_ir::Variance::Down
                        && right_var == ax_ir::Variance::Up
                    {
                        (left_name, right_name)
                    } else {
                        continue;
                    };

                    let mut found = false;
                    for j in 0..remaining.len() {
                        if i == j {
                            continue;
                        }
                        if contains_index(&remaining[j], from) {
                            remaining[j] = replace_index(&remaining[j], from, to);
                            found = true;
                        }
                    }
                    if found {
                        remaining.remove(i);
                        changed = true;
                        break;
                    }

                    if left_name == right_name {
                        remaining[i] = Expr::Sym(interner.get_or_intern("dim"));
                        changed = true;
                        break;
                    }
                }
            }
            Expr::mul(
                remaining
                    .into_iter()
                    .map(|f| eliminate_kronecker(&f, delta_sym, interner))
                    .collect(),
            )
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| eliminate_kronecker(t, delta_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(eliminate_kronecker(e, delta_sym, interner)),
        Expr::Indexed(base, indices) => {
            if let Expr::Sym(s) = base.as_ref() {
                if *s == delta_sym && indices.len() == 2 && indices[0].name == indices[1].name {
                    return Expr::Sym(interner.get_or_intern("dim"));
                }
            }
            expr.clone()
        }
        _ => expr.clone(),
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
            Expr::Import(_) => false,
            Expr::Assume(_, _) => false,
            Expr::SetConvention(_, _) => false,
            Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_var(value, var)),
            Expr::Indexed(base, _) => contains_var(base, var),
            Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
            Expr::Matrix(rows) => rows
                .iter()
                .any(|row| row.iter().any(|cell| contains_var(cell, var))),
        }
    }

    fn one_half() -> Expr {
        Expr::Rational(BigRational::new(1.into(), 2.into()))
    }

    fn diff(expr: &Expr, var: lasso::Spur, interner: &Interner) -> Expr {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::zero(),
            Expr::Complex(re, im) => {
                Expr::Complex(Box::new(diff(re, var, interner)), Box::new(diff(im, var, interner)))
            }
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
                _ => Expr::Call(
                    interner.get_or_intern("diff"),
                    vec![expr.clone(), Expr::Sym(var)],
                ),
            },
            Expr::Call(_, _) | Expr::Indexed(_, _) => Expr::Call(
                interner.get_or_intern("diff"),
                vec![expr.clone(), Expr::Sym(var)],
            ),
            Expr::FnDef(name, params, body) => {
                Expr::FnDef(*name, params.clone(), Box::new(diff(body, var, interner)))
            }
            Expr::Rule(lhs, rhs, trust) => Expr::Rule(
                Box::new(diff(lhs, var, interner)),
                Box::new(diff(rhs, var, interner)),
                *trust,
            ),
            Expr::Import(path) => Expr::Import(path.clone()),
            Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
            Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
            Expr::Piecewise(cases) => Expr::Piecewise(
                cases.iter()
                    .map(|(value, condition)| (diff(value, var, interner), condition.clone()))
                    .collect(),
            ),
            Expr::Let(name, val, body) => {
                Expr::Let(*name, val.clone(), Box::new(diff(body, var, interner)))
            }
            Expr::List(items) => Expr::List(items.iter().map(|i| diff(i, var, interner)).collect()),
            Expr::Matrix(rows) => Expr::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|cell| diff(cell, var, interner)).collect())
                    .collect(),
            ),
        }
    }

    diff(expr, coord, interner)
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn numeric_coeff(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

#[allow(dead_code)]
fn decompose_term(term: &Expr) -> (BigRational, Expr) {
    match term {
        Expr::Mul(factors) if !factors.is_empty() => {
            if let Some(coeff) = numeric_coeff(&factors[0]) {
                let rest = factors[1..].to_vec();
                let base = if rest.is_empty() {
                    Expr::one()
                } else {
                    Expr::mul(rest)
                };
                (coeff, base)
            } else {
                (BigRational::one(), term.clone())
            }
        }
        Expr::Neg(inner) => (
            BigRational::from_integer((-1).into()),
            inner.as_ref().clone(),
        ),
        Expr::Int(n) => (BigRational::from_integer(n.clone()), Expr::one()),
        Expr::Rational(r) => (r.clone(), Expr::one()),
        _ => (BigRational::one(), term.clone()),
    }
}

#[allow(dead_code)]
fn factor_base_and_exp(expr: &Expr) -> (Expr, BigRational) {
    match expr {
        Expr::Pow(base, exp) => {
            if let Some(n) = numeric_coeff(exp) {
                ((*base.clone()), n)
            } else {
                (expr.clone(), BigRational::one())
            }
        }
        _ => (expr.clone(), BigRational::one()),
    }
}

#[allow(dead_code)]
fn factor_list(expr: &Expr) -> Vec<(Expr, BigRational)> {
    match expr {
        Expr::Mul(factors) => factors.iter().map(factor_base_and_exp).collect(),
        Expr::Int(_) | Expr::Rational(_) => Vec::new(),
        _ => vec![factor_base_and_exp(expr)],
    }
}

#[allow(dead_code)]
fn remove_common_factor(factors: &[(Expr, BigRational)], common: &[(Expr, BigRational)]) -> Expr {
    let mut remaining = factors.to_vec();

    for (common_base, common_exp) in common {
        if let Some((_, exp)) = remaining.iter_mut().find(|(base, _)| *base == *common_base) {
            *exp -= common_exp.clone();
        }
    }

    let rebuilt = remaining
        .into_iter()
        .filter_map(|(base, exp)| {
            if exp.is_zero() {
                None
            } else if exp.is_one() {
                Some(base)
            } else {
                Some(Expr::pow(base, Expr::Rational(exp)))
            }
        })
        .collect::<Vec<_>>();

    Expr::mul(rebuilt)
}

#[allow(dead_code)]
fn extract_common_factor(terms: &[Expr]) -> Option<(Expr, Vec<Expr>)> {
    if terms.len() < 2 {
        return None;
    }

    let mut common = factor_list(&terms[0]);
    if common.is_empty() {
        return None;
    }

    for term in &terms[1..] {
        let factors = factor_list(term);
        common.retain_mut(|(common_base, common_exp)| {
            if let Some((_, exp)) = factors.iter().find(|(base, _)| *base == *common_base) {
                if *exp < *common_exp {
                    *common_exp = exp.clone();
                }
                !common_exp.is_zero()
            } else {
                false
            }
        });

        if common.is_empty() {
            return None;
        }
    }

    if !common.iter().any(|(_, exp)| exp.is_negative()) {
        return None;
    }

    let common_expr = Expr::mul(
        common
            .iter()
            .map(|(base, exp)| {
                if exp.is_one() {
                    base.clone()
                } else {
                    Expr::pow(base.clone(), Expr::Rational(exp.clone()))
                }
            })
            .collect(),
    );

    if common_expr == Expr::one() {
        return None;
    }

    let remainders = terms
        .iter()
        .map(|term| remove_common_factor(&factor_list(term), &common))
        .collect::<Vec<_>>();

    Some((common_expr, remainders))
}

#[allow(dead_code)]
fn collect_flat_add(expr: &Expr) -> Expr {
    let Expr::Add(terms) = expr else {
        return expr.clone();
    };

    let mut groups: Vec<(Expr, BigRational)> = Vec::new();
    for term in terms {
        let (coeff, base) = decompose_term(term);
        if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
            *acc += coeff;
        } else {
            groups.push((base, coeff));
        }
    }

    Expr::add(
        groups
            .into_iter()
            .filter_map(|(base, coeff)| {
                if coeff.is_zero() {
                    return None;
                }

                let coeff_expr = if coeff.is_integer() {
                    Expr::Int(coeff.to_integer())
                } else {
                    Expr::Rational(coeff)
                };

                Some(if base == Expr::one() {
                    coeff_expr
                } else {
                    Expr::mul(vec![coeff_expr, base])
                })
            })
            .collect(),
    )
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = numeric_coeff(base)?;
    match exp {
        Expr::Int(n) => {
            if let Some(pow) = n.to_u32() {
                let numer = base_r.numer().clone().pow(pow);
                let denom = base_r.denom().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else if n.is_negative() {
                let pow = (-n).to_u32()?;
                let numer = base_r.denom().clone().pow(pow);
                let denom = base_r.numer().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else {
                None
            }
        }
        Expr::Rational(_) => None,
        _ => None,
    }
}

fn eval_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(Box::new(eval_expr(re)), Box::new(eval_expr(im))),
        Expr::Sym(s) => Expr::Sym(*s),
        Expr::Add(terms) => Expr::add(terms.iter().map(eval_expr).collect()),
        Expr::Mul(factors) => Expr::mul(factors.iter().map(eval_expr).collect()),
        Expr::Pow(base, exp) => {
            let evaled_base = eval_expr(base);
            let evaled_exp = eval_expr(exp);
            if let Some(out) = numeric_pow(&evaled_base, &evaled_exp) {
                out
            } else {
                Expr::pow(evaled_base, evaled_exp)
            }
        }
        Expr::Neg(e) => Expr::neg(eval_expr(e)),
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(eval_expr).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(eval_expr(body)))
        }
        Expr::Rule(lhs, rhs, trust) => {
            Expr::Rule(Box::new(eval_expr(lhs)), Box::new(eval_expr(rhs)), *trust)
        }
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (eval_expr(value), condition.clone()))
                .collect(),
        ),
        Expr::Let(name, val, body) => {
            Expr::Let(*name, Box::new(eval_expr(val)), Box::new(eval_expr(body)))
        }
        Expr::Indexed(base, indices) => Expr::Indexed(Box::new(eval_expr(base)), indices.clone()),
        Expr::List(items) => Expr::List(items.iter().map(eval_expr).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(eval_expr).collect())
                .collect(),
        ),
    }
}

#[allow(dead_code)]
fn node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => 1,
        Expr::Complex(re, im) => 1 + node_count(re) + node_count(im),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::FnDef(_, params, body) => params.len() + 1 + node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + node_count(lhs) + node_count(rhs),
        Expr::Import(path) => 1 + path.len(),
        Expr::Assume(_, assumptions) => 1 + assumptions.len(),
        Expr::SetConvention(field, value) => 1 + field.len() + value.len(),
        Expr::Piecewise(cases) => {
            1 + cases.iter().map(|(value, _)| node_count(value)).sum::<usize>()
        }
        Expr::Indexed(base, _) => 1 + node_count(base),
        Expr::Let(_, val, body) => 1 + node_count(val) + node_count(body),
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
    }
}

#[allow(dead_code)]
fn expand_expr(expr: &Expr, interner: &Interner) -> Expr {
    let _ = interner;
    match expr {
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_expr(re, interner)),
            Box::new(expand_expr(im, interner)),
        ),
        Expr::Add(terms) => {
            let expanded = Expr::add(terms.iter().map(|t| expand_expr(t, interner)).collect());
            collect_flat_add(&expanded)
        }
        Expr::Mul(factors) => {
            let expanded_factors = factors
                .iter()
                .map(|f| expand_expr(f, interner))
                .collect::<Vec<_>>();

            if expanded_factors.len() > 6 {
                return Expr::mul(expanded_factors);
            }

            if let Some((idx, terms)) =
                expanded_factors.iter().enumerate().find_map(|(i, factor)| {
                    if let Expr::Add(terms) = factor {
                        Some((i, terms.clone()))
                    } else {
                        None
                    }
                })
            {
                let rest = expanded_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(i, factor)| if i != idx { Some(factor.clone()) } else { None })
                    .collect::<Vec<_>>();

                let distributed = terms
                    .into_iter()
                    .map(|term| {
                        let mut factors = Vec::with_capacity(rest.len() + 1);
                        factors.push(term);
                        factors.extend(rest.clone());
                        Expr::mul(factors)
                    })
                    .collect::<Vec<_>>();

                return expand_expr(&Expr::add(distributed), interner);
            }

            Expr::mul(expanded_factors)
        }
        Expr::Pow(base, exp) => {
            let expanded_base = expand_expr(base, interner);
            let expanded_exp = expand_expr(exp, interner);
            if let (Expr::Add(terms), Expr::Int(n)) = (&expanded_base, &expanded_exp) {
                if *n > 1.into() {
                    if let Some(power) = n.to_u32() {
                        if (2..=8).contains(&power) && terms.len() * (power as usize) <= 12 {
                            let sum = Expr::Add(terms.clone());
                            let repeated = (0..power).map(|_| sum.clone()).collect::<Vec<_>>();
                            return expand_expr(&Expr::Mul(repeated), interner);
                        }
                    }
                }
            }
            Expr::pow(expanded_base, expanded_exp)
        }
        Expr::Neg(e) => {
            let inner = expand_expr(e, interner);
            if let Expr::Add(terms) = inner {
                let expanded = Expr::add(terms.into_iter().map(Expr::neg).collect());
                collect_flat_add(&expanded)
            } else {
                Expr::neg(inner)
            }
        }
        Expr::Call(f, args) => {
            Expr::Call(*f, args.iter().map(|a| expand_expr(a, interner)).collect())
        }
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(expand_expr(body, interner)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(expand_expr(lhs, interner)),
            Box::new(expand_expr(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (expand_expr(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(expand_expr(base, interner)), indices.clone())
        }
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(expand_expr(val, interner)),
            Box::new(expand_expr(body, interner)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(|i| expand_expr(i, interner)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|cell| expand_expr(cell, interner)).collect())
                .collect(),
        ),
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => Expr::Sym(*s),
    }
}

#[allow(dead_code)]
fn collect_terms_expr(expr: &Expr, interner: &Interner) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let normalized_terms = terms
                .iter()
                .map(|term| collect_terms_expr(term, interner))
                .collect::<Vec<_>>();

            let mut groups: Vec<(Expr, BigRational)> = Vec::new();
            for term in &normalized_terms {
                let (coeff, base) = decompose_term(term);
                if let Some((_, acc)) = groups.iter_mut().find(|(existing, _)| *existing == base) {
                    *acc += coeff;
                } else {
                    groups.push((base, coeff));
                }
            }

            let rebuilt = groups
                .into_iter()
                .filter_map(|(base, coeff)| {
                    if coeff.is_zero() {
                        return None;
                    }

                    let coeff_expr = if coeff.is_integer() {
                        Expr::Int(coeff.to_integer())
                    } else {
                        Expr::Rational(coeff)
                    };

                    Some(if base == Expr::one() {
                        coeff_expr
                    } else {
                        Expr::mul(vec![coeff_expr, base])
                    })
                })
                .collect::<Vec<_>>();

            let combined = Expr::add(rebuilt);
            if let Expr::Add(combined_terms) = &combined {
                if let Some((common, remainders)) = extract_common_factor(combined_terms) {
                    let inner = collect_terms_expr(&Expr::add(remainders), interner);
                    return Expr::mul(vec![common, inner]);
                }
            }

            combined
        }
        Expr::Mul(factors) => {
            let normalized = factors
                .iter()
                .map(|factor| collect_terms_expr(factor, interner))
                .collect::<Vec<_>>();

            let add_factor = normalized.iter().enumerate().find_map(|(idx, factor)| {
                if let Expr::Add(terms) = factor {
                    Some((idx, terms.clone()))
                } else {
                    None
                }
            });

            if let Some((idx, terms)) = add_factor {
                if normalized.len() <= 4 {
                    let rest = normalized
                        .iter()
                        .enumerate()
                        .filter_map(
                            |(i, factor)| if i != idx { Some(factor.clone()) } else { None },
                        )
                        .collect::<Vec<_>>();

                    let is_pure_sign_flip = rest.len() == 1
                        && matches!(rest.first(), Some(Expr::Int(n)) if *n == (-1).into());

                    if is_pure_sign_flip {
                        return Expr::mul(normalized);
                    }

                    let distributed = terms
                        .into_iter()
                        .map(|term| {
                            let mut factors = Vec::with_capacity(rest.len() + 1);
                            factors.push(term);
                            factors.extend(rest.clone());
                            Expr::mul(factors)
                        })
                        .collect::<Vec<_>>();

                    return Expr::add(
                        distributed
                            .iter()
                            .map(|term| collect_terms_expr(term, interner))
                            .collect(),
                    );
                }
            }

            Expr::mul(normalized)
        }
        Expr::Pow(base, exp) => Expr::pow(
            collect_terms_expr(base, interner),
            collect_terms_expr(exp, interner),
        ),
        Expr::Neg(inner) => Expr::neg(collect_terms_expr(inner, interner)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| collect_terms_expr(arg, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(collect_terms_expr(body, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(collect_terms_expr(lhs, interner)),
            Box::new(collect_terms_expr(rhs, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (collect_terms_expr(value, interner), condition.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(collect_terms_expr(base, interner)),
            indices.clone(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(collect_terms_expr(val, interner)),
            Box::new(collect_terms_expr(body, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| collect_terms_expr(item, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| collect_terms_expr(cell, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

fn simplify_expr(expr: Expr, interner: &Interner) -> Expr {
    let _ = interner;
    eval_expr(&expr)
}

pub fn christoffel_from_metric(
    g: &SymbolicMatrix,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    let ginv = g.symbolic_inverse(interner);
    let n = coords.len();
    assert_eq!(n, g.dim);

    let mut gamma = vec![vec![vec![Expr::zero(); n]; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let mut sum_terms = Vec::with_capacity(n);
                for l in 0..n {
                    let dj_glk = diff_component(g.get(j, l), coords[k], interner);
                    let dk_glj = diff_component(g.get(k, l), coords[j], interner);
                    let dl_gjk = diff_component(g.get(j, k), coords[l], interner);
                    let inner = Expr::add(vec![dj_glk, dk_glj, Expr::neg(dl_gjk)]);
                    sum_terms.push(Expr::mul(vec![ginv.get(i, l).clone(), inner]));
                }
                let expr = Expr::mul(vec![half(), Expr::add(sum_terms)]);
                gamma[i][j][k] = simplify_expr(expr, interner);
            }
        }
    }
    gamma
}

pub fn riemann_from_christoffel(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
    convention: &ax_ir::Convention,
) -> Vec<Vec<Vec<Vec<ax_ir::Expr>>>> {
    let n = coords.len();
    let mut riemann = vec![vec![vec![vec![Expr::zero(); n]; n]; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let term1 = diff_component(&gamma[i][l][j], coords[k], interner);
                    let term2 = diff_component(&gamma[i][k][j], coords[l], interner);

                    let mut pos_terms = Vec::with_capacity(n);
                    let mut neg_terms = Vec::with_capacity(n);
                    for m in 0..n {
                        pos_terms.push(Expr::mul(vec![
                            gamma[i][k][m].clone(),
                            gamma[m][l][j].clone(),
                        ]));
                        neg_terms.push(Expr::mul(vec![
                            gamma[i][l][m].clone(),
                            gamma[m][k][j].clone(),
                        ]));
                    }

                    let expr = match convention.riemann_sign {
                        ax_ir::RiemannSign::MTW => Expr::add(vec![
                            term1,
                            Expr::neg(term2),
                            Expr::add(pos_terms),
                            Expr::neg(Expr::add(neg_terms)),
                        ]),
                        ax_ir::RiemannSign::Weinberg => Expr::add(vec![
                            term2,
                            Expr::neg(term1),
                            Expr::add(neg_terms),
                            Expr::neg(Expr::add(pos_terms)),
                        ]),
                    };
                    riemann[i][j][k][l] = simplify_expr(expr, interner);
                }
            }
        }
    }

    riemann
}

pub fn ricci_from_riemann(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    n: usize,
    interner: &ax_ir::Interner,
    convention: &ax_ir::Convention,
) -> Vec<Vec<ax_ir::Expr>> {
    let mut ricci = vec![vec![Expr::zero(); n]; n];
    for j in 0..n {
        for l in 0..n {
            let terms = (0..n)
                .map(|i| match convention.ricci_contraction {
                    ax_ir::RicciContraction::FirstThird => riemann[i][j][i][l].clone(),
                    ax_ir::RicciContraction::FirstFourth => riemann[i][j][l][i].clone(),
                })
                .collect::<Vec<_>>();
            ricci[j][l] = simplify_expr(Expr::add(terms), interner);
        }
    }
    ricci
}

pub fn ricci_scalar(
    ricci: &[Vec<ax_ir::Expr>],
    ginv: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let n = ginv.dim;
    let mut terms = Vec::with_capacity(n * n);
    for j in 0..n {
        for l in 0..n {
            terms.push(Expr::mul(vec![ginv.get(j, l).clone(), ricci[j][l].clone()]));
        }
    }
    let _ = interner;
    Expr::add(terms)
}

pub fn einstein_tensor(
    ricci: &[Vec<ax_ir::Expr>],
    scalar: &ax_ir::Expr,
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let n = g.dim;
    let mut einstein = vec![vec![Expr::zero(); n]; n];

    for j in 0..n {
        for l in 0..n {
            let expr = Expr::add(vec![
                ricci[j][l].clone(),
                Expr::neg(Expr::mul(vec![half(), g.get(j, l).clone(), scalar.clone()])),
            ]);
            einstein[j][l] = simplify_expr(expr, interner);
        }
    }

    einstein
}

pub fn kretschner_scalar(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    g: &SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let n = g.dim;
    let ginv = g.symbolic_inverse(interner);
    let mut terms = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let r = &riemann[i][j][k][l];
                    if *r == Expr::zero() {
                        continue;
                    }
                    let term = Expr::mul(vec![
                        g.get(i, i).clone(),
                        ginv.get(j, j).clone(),
                        ginv.get(k, k).clone(),
                        ginv.get(l, l).clone(),
                        Expr::pow(r.clone(), Expr::Int(2.into())),
                    ]);
                    terms.push(term);
                }
            }
        }
    }

    simplify_expr(Expr::add(terms), interner)
}

pub fn covariant_derivative_vector(
    v: &[ax_ir::Expr],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = v.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = vec![diff_component(&v[i], coords[coord_index], interner)];
        for (j, vj) in v.iter().enumerate() {
            terms.push(Expr::mul(vec![
                gamma[i][coord_index][j].clone(),
                vj.clone(),
            ]));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

pub fn covariant_derivative_covector(
    w: &[ax_ir::Expr],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = w.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = vec![diff_component(&w[i], coords[coord_index], interner)];
        for (j, wj) in w.iter().enumerate() {
            terms.push(Expr::neg(Expr::mul(vec![
                gamma[j][coord_index][i].clone(),
                wj.clone(),
            ])));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

pub fn covariant_derivative_tensor2(
    t: &[Vec<ax_ir::Expr>],
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coord_index: usize,
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let n = t.len();
    let mut out = vec![vec![Expr::zero(); n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut terms = vec![diff_component(&t[i][j], coords[coord_index], interner)];
            for m in 0..n {
                terms.push(Expr::mul(vec![
                    gamma[i][coord_index][m].clone(),
                    t[m][j].clone(),
                ]));
                terms.push(Expr::mul(vec![
                    gamma[j][coord_index][m].clone(),
                    t[i][m].clone(),
                ]));
            }
            out[i][j] = simplify_expr(Expr::add(terms), interner);
        }
    }

    out
}

pub fn geodesic_equations(
    gamma: &[Vec<Vec<ax_ir::Expr>>],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = coords.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = Vec::new();
        for j in 0..n {
            for k in 0..n {
                let dot_j = Expr::Sym(
                    interner.get_or_intern(&format!("dot_{}", interner.resolve(coords[j]))),
                );
                let dot_k = Expr::Sym(
                    interner.get_or_intern(&format!("dot_{}", interner.resolve(coords[k]))),
                );
                terms.push(Expr::mul(vec![
                    gamma[i][j][k].clone(),
                    dot_j,
                    dot_k,
                ]));
            }
        }
        out[i] = simplify_expr(Expr::neg(Expr::add(terms)), interner);
    }

    out
}

pub fn lie_derivative_scalar(
    f: &ax_ir::Expr,
    v: &[ax_ir::Expr],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let terms = v
        .iter()
        .enumerate()
        .map(|(i, vi)| Expr::mul(vec![vi.clone(), diff_component(f, coords[i], interner)]))
        .collect::<Vec<_>>();
    simplify_expr(Expr::add(terms), interner)
}

pub fn lie_derivative_vector(
    w: &[ax_ir::Expr],
    v: &[ax_ir::Expr],
    coords: &[lasso::Spur],
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::Expr> {
    let n = w.len();
    let mut out = vec![Expr::zero(); n];

    for i in 0..n {
        let mut terms = Vec::new();
        for j in 0..n {
            terms.push(Expr::mul(vec![
                v[j].clone(),
                diff_component(&w[i], coords[j], interner),
            ]));
            terms.push(Expr::neg(Expr::mul(vec![
                w[j].clone(),
                diff_component(&v[i], coords[j], interner),
            ])));
        }
        out[i] = simplify_expr(Expr::add(terms), interner);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn diagonal_inverse() {
        let interner = ax_ir::Interner::new();
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int(2.into()), Expr::Int(3.into())]);
        let ginv = g.symbolic_inverse(&interner);
        let expected = Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
        assert_eq!(*ginv.get(0, 0), expected);
    }

    #[test]
    fn non_diagonal_metric_inverse() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        let mut g = SymbolicMatrix::new(2);
        g.set(0, 0, Expr::Sym(a));
        g.set(0, 1, Expr::Sym(b));
        g.set(1, 0, Expr::Sym(b));
        g.set(1, 1, Expr::Sym(c));

        let ginv = g.symbolic_inverse(&interner);

        assert!(
            ginv.get(0, 0) != &Expr::zero(),
            "ginv[0][0] should not be zero for general metric"
        );
        assert!(
            ginv.get(0, 1) != &Expr::zero(),
            "ginv[0][1] should not be zero for off-diagonal metric"
        );
    }

    #[test]
    fn non_diagonal_inverse_times_original_is_identity() {
        let interner = ax_ir::Interner::new();
        let mut g = SymbolicMatrix::new(2);
        g.set(0, 0, Expr::Int(2.into()));
        g.set(0, 1, Expr::Int(1.into()));
        g.set(1, 0, Expr::Int(1.into()));
        g.set(1, 1, Expr::Int(3.into()));

        let ginv = g.symbolic_inverse(&interner);

        let product = ax_linalg::mat_mul(&g.data, &ginv.data, &interner);
        let product_simplified: Vec<Vec<Expr>> = product
            .iter()
            .map(|row| {
                row.iter()
                    .map(|e| simplify_expr(e.clone(), &interner))
                    .collect()
            })
            .collect();

        assert_eq!(product_simplified[0][0], Expr::one());
        assert_eq!(product_simplified[0][1], Expr::zero());
        assert_eq!(product_simplified[1][0], Expr::zero());
        assert_eq!(product_simplified[1][1], Expr::one());
    }

    #[test]
    fn detect_contraction_pair() {
        let interner = ax_ir::Interner::new();
        let mu = interner.get_or_intern("mu");
        let indices = vec![
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Up,
                index_type: None,
            },
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Down,
                index_type: None,
            },
        ];
        let pairs = detect_contractions(&indices);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn minkowski_christoffel_is_zero() {
        let interner = ax_ir::Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
        ]);

        let gamma = christoffel_from_metric(&g, &coords, &interner);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert_eq!(
                        gamma[i][j][k],
                        Expr::zero(),
                        "Gamma[{}][{}][{}] = {:?}",
                        i,
                        j,
                        k,
                        gamma[i][j][k]
                    );
                }
            }
        }
    }

    #[test]
    fn minkowski_kretschner_is_zero() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
            Expr::Int(1.into()),
        ]);

        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let riemann = riemann_from_christoffel(
            &gamma,
            &coords,
            &interner,
            &ax_ir::Convention::default(),
        );
        let k = kretschner_scalar(&riemann, &g, &interner);
        assert_eq!(k, Expr::zero());
    }

    #[test]
    fn minkowski_geodesic_is_zero() {
        let interner = Interner::new();
        let t = interner.get_or_intern("t");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let coords = vec![t, x, y, z];

        let g = SymbolicMatrix::from_diagonal(vec![
            Expr::Int((-1).into()),
            Expr::one(),
            Expr::one(),
            Expr::one(),
        ]);
        let gamma = christoffel_from_metric(&g, &coords, &interner);
        let geod = geodesic_equations(&gamma, &coords, &interner);
        for (i, eq) in geod.iter().enumerate() {
            assert_eq!(*eq, Expr::zero(), "geodesic[{}] = {:?}", i, eq);
        }
    }

    #[test]
    fn symmetric_tensor_canonicalize() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        let mut properties = HashMap::new();
        properties.insert(g, vec![ax_ir::TensorProperty::Symmetric(vec![0, 1])]);

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalize_indices(&expr, &properties, &interner);
        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices[0].name, mu);
            assert_eq!(indices[1].name, nu);
        } else {
            panic!("expected Indexed");
        }
    }

    #[test]
    fn antisymmetric_trace_is_zero() {
        let interner = ax_ir::Interner::new();
        let f_sym = interner.get_or_intern("F");
        let mu = interner.get_or_intern("mu");

        let mut properties = HashMap::new();
        properties.insert(
            f_sym,
            vec![ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])],
        );

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = canonicalize_indices(&expr, &properties, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn kronecker_eliminates_in_product() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let a_sym = interner.get_or_intern("A");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");

        let delta_expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let a_expr = Expr::Indexed(
            Box::new(Expr::Sym(a_sym)),
            vec![
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: rho,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let product = Expr::mul(vec![delta_expr, a_expr]);
        let result = eliminate_kronecker(&product, delta, &interner);

        if let Expr::Indexed(_, indices) = &result {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, mu);
            assert_eq!(indices[1].name, rho);
        } else {
            panic!("expected Indexed, got {:?}", result);
        }
    }

    #[test]
    fn kronecker_trace_gives_dim() {
        let interner = ax_ir::Interner::new();
        let delta = interner.get_or_intern("delta");
        let mu = interner.get_or_intern("mu");
        let dim = interner.get_or_intern("dim");

        let expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let result = eliminate_kronecker(&expr, delta, &interner);
        assert_eq!(result, Expr::Sym(dim));
    }
}
