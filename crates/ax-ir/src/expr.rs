use lasso::Key;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

pub type Sym = lasso::Spur;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Variance {
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index {
    pub name: Sym,
    pub variance: Variance,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Int(BigInt),
    Rational(BigRational),
    Float(f64),
    Sym(Sym),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Call(Sym, Vec<Expr>),
    Indexed(Box<Expr>, Vec<Index>),
    Let(Sym, Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
    Matrix(Vec<Vec<Expr>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExprSortKey {
    Numeric(BigRational),
    Sym(usize),
    Other(String),
}

impl Ord for ExprSortKey {
    fn cmp(&self, other: &Self) -> Ordering {
        use ExprSortKey::{Numeric, Other, Sym};

        match (self, other) {
            (Numeric(lhs), Numeric(rhs)) => lhs.cmp(rhs),
            (Numeric(_), _) => Ordering::Less,
            (_, Numeric(_)) => Ordering::Greater,
            (Sym(lhs), Sym(rhs)) => lhs.cmp(rhs),
            (Sym(_), Other(_)) => Ordering::Less,
            (Other(_), Sym(_)) => Ordering::Greater,
            (Other(lhs), Other(rhs)) => lhs.cmp(rhs),
        }
    }
}

impl PartialOrd for ExprSortKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn normalize_numeric(n: BigRational) -> Expr {
    if n.is_integer() {
        Expr::Int(n.to_integer())
    } else {
        Expr::Rational(n)
    }
}

fn as_numeric(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn is_zero_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero()) || matches!(expr, Expr::Rational(r) if r.is_zero())
}

fn is_one_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_one()) || matches!(expr, Expr::Rational(r) if r.is_one())
}

fn expr_sort_key(e: &Expr) -> ExprSortKey {
    match e {
        Expr::Int(n) => ExprSortKey::Numeric(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => ExprSortKey::Numeric(r.clone()),
        Expr::Sym(s) => ExprSortKey::Sym(s.into_usize()),
        _ => ExprSortKey::Other(format!("{e:?}")),
    }
}

impl Expr {
    pub fn zero() -> Expr {
        Expr::Int(0.into())
    }

    pub fn one() -> Expr {
        Expr::Int(1.into())
    }

    pub fn add(mut terms: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for term in terms.drain(..) {
            match term {
                Expr::Add(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }

        let mut numeric_sum = BigRational::zero();
        let mut has_numeric = false;
        let mut result = Vec::new();

        for term in flat {
            if let Some(n) = as_numeric(&term) {
                numeric_sum += n;
                has_numeric = true;
            } else if !is_zero_expr(&term) {
                result.push(term);
            }
        }

        if has_numeric && (!numeric_sum.is_zero() || result.is_empty()) {
            result.push(normalize_numeric(numeric_sum));
        }

        result.retain(|term| !is_zero_expr(term));
        result.sort_by_key(expr_sort_key);

        match result.len() {
            0 => Expr::zero(),
            1 => result.pop().unwrap(),
            _ => Expr::Add(result),
        }
    }

    pub fn mul(mut factors: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for factor in factors.drain(..) {
            match factor {
                Expr::Mul(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }

        if flat.iter().any(|factor| matches!(factor, Expr::Int(n) if n.is_zero())) {
            return Expr::zero();
        }

        let mut numeric_product = BigRational::one();
        let mut has_numeric = false;
        let mut result = Vec::new();

        for factor in flat {
            if let Some(n) = as_numeric(&factor) {
                numeric_product *= n;
                has_numeric = true;
            } else if !is_one_expr(&factor) {
                result.push(factor);
            }
        }

        if numeric_product.is_zero() {
            return Expr::zero();
        }

        if has_numeric && (!numeric_product.is_one() || result.is_empty()) {
            result.push(normalize_numeric(numeric_product));
        }

        result.retain(|factor| !is_one_expr(factor));
        result.sort_by_key(expr_sort_key);

        match result.len() {
            0 => Expr::one(),
            1 => result.pop().unwrap(),
            _ => Expr::Mul(result),
        }
    }

    pub fn pow(base: Expr, exp: Expr) -> Expr {
        if matches!(exp, Expr::Int(ref n) if n.is_zero()) {
            return Expr::one();
        }
        if matches!(exp, Expr::Int(ref n) if n.is_one()) {
            return base;
        }
        if matches!(base, Expr::Int(ref n) if n.is_zero()) {
            return Expr::zero();
        }
        if matches!(base, Expr::Int(ref n) if n.is_one()) {
            return Expr::one();
        }

        if let (Expr::Int(base_int), Expr::Int(exp_int)) = (&base, &exp) {
            if let Some(pow) = exp_int.to_u32() {
                return Expr::Int(base_int.pow(pow));
            }

            if exp_int.is_negative() {
                let abs = (-exp_int).to_u32();
                if let Some(pow) = abs {
                    let denom = base_int.pow(pow);
                    return normalize_numeric(BigRational::new(BigInt::one(), denom));
                }
            }
        }

        Expr::Pow(Box::new(base), Box::new(exp))
    }

    pub fn neg(e: Expr) -> Expr {
        match e {
            Expr::Int(n) => Expr::Int(-n),
            Expr::Rational(r) => Expr::Rational(-r),
            Expr::Neg(inner) => *inner,
            other => Expr::Neg(Box::new(other)),
        }
    }
}
