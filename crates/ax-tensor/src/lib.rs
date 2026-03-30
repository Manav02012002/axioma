#![forbid(unsafe_code)]

use ax_ir::{Expr, Interner};
use num_rational::BigRational;
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
                Expr::Rational(r) => Expr::Rational(BigRational::new(
                    r.denom().clone(),
                    r.numer().clone(),
                )),
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

pub fn diff_component(expr: &ax_ir::Expr, coord: lasso::Spur, interner: &ax_ir::Interner) -> ax_ir::Expr {
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
                gamma[i][j][k] = expr;
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
}
