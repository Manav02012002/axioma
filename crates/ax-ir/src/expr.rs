use lasso::Key;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

pub type Sym = lasso::Spur;
pub type IndexType = Option<Sym>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variance {
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Index {
    pub name: Sym,
    pub variance: Variance,
    pub index_type: IndexType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexFamily {
    pub name: Sym,
    pub values: Vec<Sym>,
    pub position: IndexPosition,
    pub dimension: Option<usize>,
    pub parent: Option<Sym>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IndexPosition {
    #[default]
    Free,
    Fixed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Assumption {
    Real,
    Positive,
    Negative,
    NonZero,
    Integer,
    Even,
    Odd,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorProperty {
    Symmetric(Vec<usize>),
    AntiSymmetric(Vec<usize>),
    RiemannSymmetry,
    Traceless,
    Diagonal,
    Trace,
    Metric,
    InverseMetric,
    KroneckerDelta,
    EpsilonTensor,
    Derivative,
    PartialDerivative,
    CovariantDerivative,
    TableauInherit,
    Depends(Vec<Sym>),
    Spinor,
    DiracBar,
    GammaMatrixProp,
    Commuting,
    AntiCommuting,
    NonCommuting,
    CommutingWith(Vec<Sym>),
    AntiCommutingWith(Vec<Sym>),
    NonCommutingWith(Vec<Sym>),
    SelfAntiCommuting,
    SelfNonCommuting,
    SelfCommuting,
    CommutingAsProduct,
    CommutingAsSum,
    MajoranaSpinor,
    WeylSpinor,
    ImplicitIndex,
    SortOrder(Vec<Sym>),
    TableauSymmetry {
        shape: Vec<usize>,
        indices: Vec<usize>,
    },
    SatisfiesBianchi {
        slots: Vec<usize>,
    },
    DimensionDependentIdentity,
    WeylTensor,
    DifferentialFormDegree(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Grading {
    Even,
    Odd,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Convention {
    pub metric_signature: MetricSignature,
    pub riemann_sign: RiemannSign,
    pub ricci_contraction: RicciContraction,
    pub levi_civita_norm: LeviCivitaNorm,
    pub fourier_sign: FourierSign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricSignature {
    MostlyPlus,
    MostlyMinus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RiemannSign {
    MTW,
    Weinberg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RicciContraction {
    FirstThird,
    FirstFourth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LeviCivitaNorm {
    PlusOne,
    MinusOne,
    SqrtG,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FourierSign {
    MinusI,
    PlusI,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TrustLevel {
    Exact,
    UnderAssumptions,
    Heuristic,
    NumericallyChecked,
    #[default]
    Unverified,
}

impl Default for Convention {
    fn default() -> Self {
        Self {
            metric_signature: MetricSignature::MostlyPlus,
            riemann_sign: RiemannSign::MTW,
            ricci_contraction: RicciContraction::FirstThird,
            levi_civita_norm: LeviCivitaNorm::PlusOne,
            fourier_sign: FourierSign::MinusI,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    Gt(Expr, Expr),
    Lt(Expr, Expr),
    Ge(Expr, Expr),
    Le(Expr, Expr),
    Eq(Expr, Expr),
    Ne(Expr, Expr),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    True,
    False,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParentRel {
    ExplicitGroup,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Int(BigInt),
    Rational(BigRational),
    Float(f64),
    Complex(Box<Expr>, Box<Expr>),
    Sym(Sym),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Call(Sym, Vec<Expr>),
    FnDef(Sym, Vec<Sym>, Box<Expr>),
    Rule(Box<Expr>, Box<Expr>, TrustLevel),
    Import(Vec<Sym>),
    Assume(Sym, Vec<Assumption>),
    SetConvention(String, String),
    Piecewise(Vec<(Expr, Condition)>),
    Indexed(Box<Expr>, Vec<Index>),
    Group(Box<Expr>, ParentRel),
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
        Expr::Group(inner, _) => as_numeric(inner),
        _ => None,
    }
}

fn as_complex(expr: &Expr) -> Option<(Expr, Expr)> {
    match expr {
        Expr::Complex(re, im) => Some((re.as_ref().clone(), im.as_ref().clone())),
        Expr::Group(inner, _) => as_complex(inner),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Some((expr.clone(), Expr::zero())),
        _ => None,
    }
}

fn normalize_complex(re: Expr, im: Expr) -> Expr {
    if is_zero_expr(&im) {
        re
    } else {
        Expr::Complex(Box::new(re), Box::new(im))
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
        Expr::Complex(_, _) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Sym(s) => ExprSortKey::Sym(s.into_usize()),
        Expr::FnDef(_, _, _) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Rule(_, _, _) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Import(_) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Assume(_, _) => ExprSortKey::Other(format!("{e:?}")),
        Expr::SetConvention(_, _) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Piecewise(_) => ExprSortKey::Other(format!("{e:?}")),
        Expr::Group(_, _) => ExprSortKey::Other(format!("{e:?}")),
        _ => ExprSortKey::Other(format!("{e:?}")),
    }
}

fn numeric_expr(n: BigRational) -> Expr {
    normalize_numeric(n)
}

fn factor_base_and_exp(expr: Expr) -> (Expr, BigRational) {
    match expr {
        Expr::Pow(base, exp) => {
            if let Some(n) = as_numeric(&exp) {
                ((*base).clone(), n)
            } else {
                (Expr::Pow(base, exp), BigRational::one())
            }
        }
        other => (other, BigRational::one()),
    }
}

fn split_add_term(expr: Expr) -> (BigRational, Expr) {
    match expr {
        Expr::Int(n) => (BigRational::from_integer(n), Expr::one()),
        Expr::Rational(r) => (r, Expr::one()),
        Expr::Mul(factors) if !factors.is_empty() => {
            if let Some(coeff) = as_numeric(&factors[0]) {
                let rest = factors[1..].to_vec();
                let base = if rest.is_empty() {
                    Expr::one()
                } else if rest.len() == 1 {
                    rest[0].clone()
                } else {
                    Expr::Mul(rest)
                };
                (coeff, base)
            } else {
                (BigRational::one(), Expr::Mul(factors))
            }
        }
        Expr::Neg(inner) => {
            let (coeff, base) = split_add_term(*inner);
            (-coeff, base)
        }
        other => (BigRational::one(), other),
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
        let mut complex_sum: Option<(Expr, Expr)> = None;
        let mut result = Vec::new();

        for term in flat {
            if let Some((re, im)) = as_complex(&term) {
                if is_zero_expr(&im) {
                    if let Some(n) = as_numeric(&re) {
                        numeric_sum += n;
                        has_numeric = true;
                    } else if !is_zero_expr(&re) {
                        result.push(re);
                    }
                } else {
                    complex_sum = Some(match complex_sum {
                        Some((acc_re, acc_im)) => {
                            (Expr::add(vec![acc_re, re]), Expr::add(vec![acc_im, im]))
                        }
                        None => (re, im),
                    });
                }
            } else if let Some(n) = as_numeric(&term) {
                numeric_sum += n;
                has_numeric = true;
            } else if !is_zero_expr(&term) {
                result.push(term);
            }
        }

        if has_numeric && (!numeric_sum.is_zero() || result.is_empty()) {
            result.push(normalize_numeric(numeric_sum));
        }

        if let Some((re, im)) = complex_sum {
            result.push(normalize_complex(re, im));
        }

        let mut grouped: Vec<(Expr, BigRational)> = Vec::new();
        for term in result {
            let (coeff, base) = split_add_term(term);
            if let Some((_, existing)) = grouped.iter_mut().find(|(b, _)| *b == base) {
                *existing += coeff;
            } else {
                grouped.push((base, coeff));
            }
        }

        let mut result = grouped
            .into_iter()
            .filter_map(|(base, coeff)| {
                if coeff.is_zero() {
                    None
                } else if base == Expr::one() {
                    Some(normalize_numeric(coeff))
                } else if coeff == BigRational::from_integer((-1).into()) {
                    Some(Expr::neg(base))
                } else if coeff.is_one() {
                    Some(base)
                } else {
                    Some(Expr::mul(vec![normalize_numeric(coeff), base]))
                }
            })
            .collect::<Vec<_>>();

        result.retain(|term| !is_zero_expr(term));
        result.sort_by_key(expr_sort_key);

        if result.len() == 2 {
            if let Some(n) = as_numeric(&result[0]) {
                if n == BigRational::from_integer((-1).into()) {
                    let negated = result.into_iter().map(Expr::neg).collect::<Vec<_>>();
                    return Expr::mul(vec![Expr::Int((-1).into()), Expr::Add(negated)]);
                }
            }
        }

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

        if flat.iter().any(is_zero_expr) {
            return Expr::zero();
        }

        let mut complex_acc: Option<(Expr, Expr)> = None;
        let mut non_complex = Vec::new();
        for factor in flat {
            if let Some((re, im)) = as_complex(&factor) {
                if is_zero_expr(&im) {
                    non_complex.push(re);
                } else {
                    complex_acc = Some(match complex_acc {
                        Some((a, b)) => {
                            let ac = Expr::mul(vec![a.clone(), re.clone()]);
                            let bd = Expr::mul(vec![b.clone(), im.clone()]);
                            let ad = Expr::mul(vec![a, im.clone()]);
                            let bc = Expr::mul(vec![b, re.clone()]);
                            (Expr::add(vec![ac, Expr::neg(bd)]), Expr::add(vec![ad, bc]))
                        }
                        None => (re, im),
                    });
                }
            } else {
                non_complex.push(factor);
            }
        }

        let mut numeric_product = BigRational::one();
        let mut has_numeric = false;
        let mut symbolic = Vec::<(Expr, BigRational)>::new();

        for factor in non_complex {
            let factor = match factor {
                Expr::Neg(inner) => {
                    numeric_product = -numeric_product;
                    has_numeric = true;
                    *inner
                }
                other => other,
            };

            if let Some(n) = as_numeric(&factor) {
                numeric_product *= n;
                has_numeric = true;
            } else if !is_one_expr(&factor) {
                let (base, exp) = factor_base_and_exp(factor);
                if let Some((_, existing_exp)) =
                    symbolic.iter_mut().find(|(existing, _)| *existing == base)
                {
                    *existing_exp += exp;
                } else {
                    symbolic.push((base, exp));
                }
            }
        }

        if numeric_product.is_zero() {
            return Expr::zero();
        }

        let mut result = symbolic
            .into_iter()
            .filter_map(|(base, exp)| {
                if exp.is_zero() {
                    None
                } else if exp.is_one() {
                    Some(base)
                } else {
                    Some(Expr::pow(base, numeric_expr(exp)))
                }
            })
            .collect::<Vec<_>>();

        if has_numeric && (!numeric_product.is_one() || result.is_empty()) {
            result.insert(0, normalize_numeric(numeric_product));
        }

        result.retain(|factor| !is_one_expr(factor));
        let plain = match result.len() {
            0 => Expr::one(),
            1 => result.pop().unwrap(),
            _ => Expr::Mul(result),
        };

        if let Some((re, im)) = complex_acc {
            let scaled_re = Expr::mul(vec![plain.clone(), re]);
            let scaled_im = Expr::mul(vec![plain, im]);
            normalize_complex(scaled_re, scaled_im)
        } else {
            plain
        }
    }

    pub fn pow(base: Expr, exp: Expr) -> Expr {
        if let Expr::Group(inner, _) = base {
            return Expr::pow(*inner, exp);
        }

        if let Expr::Neg(inner) = base.clone() {
            if let Some(exp_num) = as_numeric(&exp) {
                if exp_num.is_integer() {
                    let exp_int = exp_num.to_integer();
                    let reduced = Expr::pow(*inner, exp);
                    if exp_int.is_odd() {
                        return Expr::neg(reduced);
                    }
                    return reduced;
                }
            }
        }

        if let (Expr::Complex(re, im), Expr::Int(n)) = (&base, &exp) {
            if let Some(pow) = n.to_u32() {
                let mut acc = Expr::one();
                let complex = Expr::Complex(re.clone(), im.clone());
                for _ in 0..pow {
                    acc = Expr::mul(vec![acc, complex.clone()]);
                }
                return acc;
            }
        }
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

        if matches!(exp, Expr::Int(ref n) if *n == (-1).into()) {
            if let Expr::Mul(factors) = &base {
                return Expr::mul(
                    factors
                        .iter()
                        .cloned()
                        .map(|factor| Expr::pow(factor, Expr::Int((-1).into())))
                        .collect(),
                );
            }
        }

        if let Expr::Pow(inner_base, inner_exp) = &base {
            if let (Some(lhs), Some(rhs)) = (as_numeric(inner_exp), as_numeric(&exp)) {
                return Expr::pow((**inner_base).clone(), numeric_expr(lhs * rhs));
            }
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

        if let (Some(base_num), Some(exp_num)) = (as_numeric(&base), as_numeric(&exp)) {
            if exp_num.is_integer() {
                let exp_int = exp_num.to_integer();
                if let Some(pow) = exp_int.to_u32() {
                    let numer = base_num.numer().clone().pow(pow);
                    let denom = base_num.denom().clone().pow(pow);
                    return normalize_numeric(BigRational::new(numer, denom));
                }
                if exp_int.is_negative() {
                    let pow = (-exp_int).to_u32();
                    if let Some(pow) = pow {
                        let numer = base_num.denom().clone().pow(pow);
                        let denom = base_num.numer().clone().pow(pow);
                        return normalize_numeric(BigRational::new(numer, denom));
                    }
                }
            }
        }

        Expr::Pow(Box::new(base), Box::new(exp))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(e: Expr) -> Expr {
        match e {
            Expr::Group(inner, rel) => Expr::Group(Box::new(Expr::neg(*inner)), rel),
            Expr::Int(n) => Expr::Int(-n),
            Expr::Rational(r) => Expr::Rational(-r),
            Expr::Complex(re, im) => normalize_complex(Expr::neg(*re), Expr::neg(*im)),
            Expr::Neg(inner) => *inner,
            Expr::Add(terms) => Expr::add(terms.into_iter().map(Expr::neg).collect()),
            Expr::Mul(mut factors) => {
                if let Some(first) = factors.first_mut() {
                    if let Some(n) = as_numeric(first) {
                        *first = normalize_numeric(-n);
                        return Expr::mul(factors);
                    }
                }

                let mut negated = Vec::with_capacity(factors.len() + 1);
                negated.push(Expr::Int((-1).into()));
                negated.extend(factors);
                Expr::mul(negated)
            }
            other => Expr::Neg(Box::new(other)),
        }
    }

    pub fn group(e: Expr) -> Expr {
        Expr::Group(Box::new(e), ParentRel::ExplicitGroup)
    }

    pub fn group_with_rel(e: Expr, rel: ParentRel) -> Expr {
        Expr::Group(Box::new(e), rel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_addition() {
        let a = Expr::Complex(Box::new(Expr::Int(1.into())), Box::new(Expr::Int(2.into())));
        let b = Expr::Complex(Box::new(Expr::Int(3.into())), Box::new(Expr::Int(4.into())));
        let sum = Expr::add(vec![a, b]);
        assert_eq!(
            sum,
            Expr::Complex(Box::new(Expr::Int(4.into())), Box::new(Expr::Int(6.into())))
        );
    }

    #[test]
    fn complex_multiplication() {
        let a = Expr::Complex(Box::new(Expr::one()), Box::new(Expr::one()));
        let b = Expr::Complex(Box::new(Expr::one()), Box::new(Expr::neg(Expr::one())));
        let prod = Expr::mul(vec![a, b]);
        assert_eq!(prod, Expr::Int(2.into()));
    }

    #[test]
    fn grouped_factor_is_not_collapsed_into_power() {
        let sym = lasso::Spur::default();
        let x = Expr::Sym(sym);
        let grouped = Expr::group(x.clone());
        let prod = Expr::mul(vec![grouped.clone(), x.clone()]);
        assert_eq!(prod, Expr::Mul(vec![grouped, x]));
    }

    #[test]
    fn grouped_term_is_not_combined_with_ungrouped_term() {
        let sym = lasso::Spur::default();
        let x = Expr::Sym(sym);
        let grouped = Expr::group(x.clone());
        let sum = Expr::add(vec![grouped.clone(), x.clone()]);
        assert_eq!(sum, Expr::Add(vec![x, grouped]));
    }

    #[test]
    fn inverse_of_negated_product_distributes_sign() {
        let sym = lasso::Spur::default();
        let x = Expr::Sym(sym);
        let expr = Expr::pow(Expr::neg(x.clone()), Expr::Int((-1).into()));
        assert_eq!(expr, Expr::neg(Expr::pow(x, Expr::Int((-1).into()))));
    }
}
