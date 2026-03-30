#![forbid(unsafe_code)]

use ax_ir::{Expr, Interner};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::HashSet;

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
        let _ = interner;
        for row in 0..self.dim {
            for col in 0..self.dim {
                if row != col && self.data[row][col] != Expr::zero() {
                    panic!("symbolic inverse of non-diagonal matrices not yet implemented");
                }
            }
        }

        let mut inverse = Self::new(self.dim);
        for i in 0..self.dim {
            inverse.data[i][i] = match &self.data[i][i] {
                Expr::Int(n) => Expr::Rational(BigRational::new(1.into(), n.clone())),
                Expr::Rational(r) => {
                    Expr::Rational(BigRational::new(r.denom().clone(), r.numer().clone()))
                }
                other => Expr::pow(other.clone(), Expr::Int((-1).into())),
            };
        }
        inverse
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
            if indices[i].name == indices[j].name && indices[i].variance != indices[j].variance {
                used.insert(i);
                used.insert(j);
                pairs.push((i, j));
                break;
            }
        }
    }

    pairs
}

pub fn diff_component(
    expr: &ax_ir::Expr,
    coord: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
        match expr {
            Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
            Expr::Sym(s) => *s == var,
            Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => {
                items.iter().any(|item| contains_var(item, var))
            }
            Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
            Expr::Neg(e) => contains_var(e, var),
            Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
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
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::Indexed(base, _) => 1 + node_count(base),
        Expr::Let(_, val, body) => 1 + node_count(val) + node_count(body),
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
    }
}

#[allow(dead_code)]
fn expand_expr(expr: &Expr, interner: &Interner) -> Expr {
    let _ = interner;
    match expr {
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

                    let expr = Expr::add(vec![
                        term1,
                        Expr::neg(term2),
                        Expr::add(pos_terms),
                        Expr::neg(Expr::add(neg_terms)),
                    ]);
                    riemann[i][j][k][l] = expr;
                }
            }
        }
    }

    riemann
}

pub fn ricci_from_riemann(
    riemann: &[Vec<Vec<Vec<ax_ir::Expr>>>],
    n: usize,
    _interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let mut ricci = vec![vec![Expr::zero(); n]; n];
    for j in 0..n {
        for l in 0..n {
            let terms = (0..n)
                .map(|i| riemann[i][j][i][l].clone())
                .collect::<Vec<_>>();
            ricci[j][l] = Expr::add(terms);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_inverse() {
        let interner = ax_ir::Interner::new();
        let g = SymbolicMatrix::from_diagonal(vec![Expr::Int(2.into()), Expr::Int(3.into())]);
        let ginv = g.symbolic_inverse(&interner);
        let expected = Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
        assert_eq!(*ginv.get(0, 0), expected);
    }

    #[test]
    fn detect_contraction_pair() {
        let interner = ax_ir::Interner::new();
        let mu = interner.get_or_intern("mu");
        let indices = vec![
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Up,
            },
            ax_ir::Index {
                name: mu,
                variance: ax_ir::Variance::Down,
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
        let riemann = riemann_from_christoffel(&gamma, &coords, &interner);
        let k = kretschner_scalar(&riemann, &g, &interner);
        assert_eq!(k, Expr::zero());
    }
}
