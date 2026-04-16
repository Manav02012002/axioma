#![allow(clippy::manual_is_multiple_of)]

pub mod brst;
pub mod d_algebra;
pub mod superspace;

use ax_ir::Expr;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Grading {
    Z2(u8),
    Z(i32),
    Product(Vec<(String, GradingValue)>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GradingValue {
    Mod2(u8),
    Integer(i32),
}

impl Grading {
    pub fn bosonic() -> Self {
        Self::Z2(0)
    }

    pub fn fermionic() -> Self {
        Self::Z2(1)
    }

    pub fn ghost(n: i32) -> Self {
        Self::Z(n)
    }

    pub fn is_bosonic(&self) -> bool {
        self.parity() == 0
    }

    pub fn is_fermionic(&self) -> bool {
        self.parity() == 1
    }

    pub fn add(&self, other: &Grading) -> Grading {
        match (self, other) {
            (Grading::Z2(a), Grading::Z2(b)) => Grading::Z2((a + b) % 2),
            (Grading::Z(a), Grading::Z(b)) => Grading::Z(a + b),
            (Grading::Product(a), Grading::Product(b)) => {
                let mut out = BTreeMap::<String, GradingValue>::new();
                for (name, value) in a.iter().chain(b.iter()) {
                    out.entry(name.clone())
                        .and_modify(|existing| *existing = add_values(existing, value))
                        .or_insert_with(|| normalize_value(value.clone()));
                }
                Grading::Product(out.into_iter().collect())
            }
            (Grading::Product(a), b) => add_product_and_scalar(a, b),
            (a, Grading::Product(b)) => add_product_and_scalar(b, a),
            (a, b) => Grading::Product(vec![
                (
                    "z2".to_string(),
                    GradingValue::Mod2((a.parity() + b.parity()) % 2),
                ),
                (
                    "z".to_string(),
                    GradingValue::Integer(a.integer_degree() + b.integer_degree()),
                ),
            ]),
        }
    }

    pub fn commutation_sign(&self, other: &Grading) -> i32 {
        if (self.parity() * other.parity()) % 2 == 0 {
            1
        } else {
            -1
        }
    }

    fn parity(&self) -> u8 {
        match self {
            Grading::Z2(n) => n % 2,
            Grading::Z(n) => n.rem_euclid(2) as u8,
            Grading::Product(values) => values
                .iter()
                .fold(0u8, |acc, (_, value)| (acc + value.parity()) % 2),
        }
    }

    fn integer_degree(&self) -> i32 {
        match self {
            Grading::Z2(n) => i32::from(n % 2),
            Grading::Z(n) => *n,
            Grading::Product(values) => {
                values.iter().map(|(_, value)| value.integer_degree()).sum()
            }
        }
    }
}

impl GradingValue {
    fn parity(&self) -> u8 {
        match self {
            GradingValue::Mod2(n) => n % 2,
            GradingValue::Integer(n) => n.rem_euclid(2) as u8,
        }
    }

    fn integer_degree(&self) -> i32 {
        match self {
            GradingValue::Mod2(n) => i32::from(n % 2),
            GradingValue::Integer(n) => *n,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GradedSymbolTable {
    gradings: HashMap<lasso::Spur, Grading>,
}

impl GradedSymbolTable {
    pub fn new() -> Self {
        Self {
            gradings: HashMap::new(),
        }
    }

    pub fn declare(&mut self, sym: lasso::Spur, grading: Grading) {
        self.gradings.insert(sym, grading);
    }

    pub fn get(&self, sym: lasso::Spur) -> Option<&Grading> {
        self.gradings.get(&sym)
    }

    pub fn infer_grading(&self, expr: &Expr) -> Grading {
        match expr {
            Expr::Sym(s) => self.get(*s).cloned().unwrap_or_else(Grading::bosonic),
            Expr::Add(terms) => terms
                .first()
                .map(|term| self.infer_grading(term))
                .unwrap_or_else(Grading::bosonic),
            Expr::Mul(factors) => factors.iter().fold(Grading::bosonic(), |acc, factor| {
                acc.add(&self.infer_grading(factor))
            }),
            Expr::Pow(base, _) => self.infer_grading(base),
            Expr::Neg(inner) => self.infer_grading(inner),
            Expr::Indexed(base, _) => self.infer_grading(base),
            Expr::Group(inner, _) => self.infer_grading(inner),
            Expr::Call(f, args) => self.get(*f).cloned().unwrap_or_else(|| {
                args.iter().fold(Grading::bosonic(), |acc, arg| {
                    acc.add(&self.infer_grading(arg))
                })
            }),
            Expr::List(items) => items
                .first()
                .map(|item| self.infer_grading(item))
                .unwrap_or_else(Grading::bosonic),
            Expr::Matrix(rows) => rows
                .iter()
                .flatten()
                .next()
                .map(|item| self.infer_grading(item))
                .unwrap_or_else(Grading::bosonic),
            Expr::Let(_, _, body) => self.infer_grading(body),
            Expr::Complex(_, _)
            | Expr::Int(_)
            | Expr::Rational(_)
            | Expr::Float(_)
            | Expr::FnDef(_, _, _)
            | Expr::Rule(_, _, _)
            | Expr::Import(_)
            | Expr::Assume(_, _)
            | Expr::SetConvention(_, _)
            | Expr::Piecewise(_) => Grading::bosonic(),
        }
    }
}

impl Default for GradedSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

pub fn graded_sign_for_slot_swap(left_parity_odd: bool, right_parity_odd: bool) -> i32 {
    if left_parity_odd && right_parity_odd {
        -1
    } else {
        1
    }
}

pub fn graded_multiply(
    factors: &[Expr],
    table: &GradedSymbolTable,
    interner: &ax_ir::Interner,
) -> Expr {
    let mut flat = Vec::new();
    let mut coeff = BigRational::one();
    flatten_factors(factors, &mut flat, &mut coeff);

    if coeff.is_zero() {
        return Expr::zero();
    }

    for factor in &flat {
        if nilpotent_power_is_zero(factor, table) {
            return Expr::zero();
        }
    }

    let mut seen_fermions = HashSet::<String>::new();
    for factor in &flat {
        if table.infer_grading(factor).is_fermionic() {
            let key = canonical_key(factor, interner);
            if !seen_fermions.insert(key) {
                return Expr::zero();
            }
        }
    }

    let mut sign = 1;
    let mut sorted = flat;
    let len = sorted.len();
    for i in 0..len {
        for j in 0..len.saturating_sub(1 + i) {
            if expr_cmp(&sorted[j], &sorted[j + 1], interner) == Ordering::Greater {
                sign *= table
                    .infer_grading(&sorted[j])
                    .commutation_sign(&table.infer_grading(&sorted[j + 1]));
                sorted.swap(j, j + 1);
            }
        }
    }

    if sign < 0 {
        coeff = -coeff;
    }

    let mut out = Vec::new();
    if !coeff.is_one() || sorted.is_empty() {
        out.push(rational_expr(coeff));
    }
    out.extend(sorted);
    Expr::mul(out)
}

pub fn graded_commutator(
    a: &Expr,
    b: &Expr,
    table: &GradedSymbolTable,
    interner: &ax_ir::Interner,
) -> Expr {
    let ab = graded_multiply(&[a.clone(), b.clone()], table, interner);
    let ba = graded_multiply(&[b.clone(), a.clone()], table, interner);
    let sign = table
        .infer_grading(a)
        .commutation_sign(&table.infer_grading(b));

    if sign == 1 {
        Expr::add(vec![ab, Expr::neg(ba)])
    } else {
        Expr::add(vec![ab, ba])
    }
}

pub fn graded_simplify(expr: &Expr, table: &GradedSymbolTable, interner: &ax_ir::Interner) -> Expr {
    let mut current = expr.clone();
    for _ in 0..8 {
        let next = graded_simplify_once(&current, table, interner);
        if next == current {
            return next;
        }
        current = next;
    }
    current
}

fn graded_simplify_once(
    expr: &Expr,
    table: &GradedSymbolTable,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| graded_simplify(term, table, interner))
                .filter(|term| !is_zero(term))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let simplified = factors
                .iter()
                .map(|factor| graded_simplify(factor, table, interner))
                .collect::<Vec<_>>();
            graded_multiply(&simplified, table, interner)
        }
        Expr::Pow(base, exp) => {
            let base = graded_simplify(base, table, interner);
            if nilpotent_exponent(&base, exp, table) {
                Expr::zero()
            } else {
                Expr::pow(base, graded_simplify(exp, table, interner))
            }
        }
        Expr::Neg(inner) => Expr::neg(graded_simplify(inner, table, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(graded_simplify(re, table, interner)),
            Box::new(graded_simplify(im, table, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| graded_simplify(arg, table, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(graded_simplify(base, table, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(graded_simplify(inner, table, interner)), *rel)
        }
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| graded_simplify(item, table, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| graded_simplify(item, table, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(graded_simplify(value, table, interner)),
            Box::new(graded_simplify(body, table, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(graded_simplify(body, table, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(graded_simplify(lhs, table, interner)),
            Box::new(graded_simplify(rhs, table, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (graded_simplify(value, table, interner), condition.clone())
                })
                .collect(),
        ),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => expr.clone(),
    }
}

fn add_values(a: &GradingValue, b: &GradingValue) -> GradingValue {
    match (a, b) {
        (GradingValue::Mod2(x), GradingValue::Mod2(y)) => GradingValue::Mod2((x + y) % 2),
        _ => GradingValue::Integer(a.integer_degree() + b.integer_degree()),
    }
}

fn normalize_value(value: GradingValue) -> GradingValue {
    match value {
        GradingValue::Mod2(n) => GradingValue::Mod2(n % 2),
        GradingValue::Integer(n) => GradingValue::Integer(n),
    }
}

fn add_product_and_scalar(product: &[(String, GradingValue)], scalar: &Grading) -> Grading {
    let mut out = BTreeMap::<String, GradingValue>::new();
    for (name, value) in product {
        out.insert(name.clone(), normalize_value(value.clone()));
    }
    let value = match scalar {
        Grading::Z2(n) => GradingValue::Mod2(*n),
        Grading::Z(n) => GradingValue::Integer(*n),
        Grading::Product(_) => unreachable!(),
    };
    out.entry("scalar".to_string())
        .and_modify(|existing| *existing = add_values(existing, &value))
        .or_insert(value);
    Grading::Product(out.into_iter().collect())
}

fn flatten_factors(factors: &[Expr], out: &mut Vec<Expr>, coeff: &mut BigRational) {
    for factor in factors {
        match factor {
            Expr::Mul(inner) => flatten_factors(inner, out, coeff),
            Expr::Neg(inner) => {
                *coeff = -coeff.clone();
                flatten_factors(std::slice::from_ref(inner.as_ref()), out, coeff);
            }
            Expr::Int(n) => *coeff *= BigRational::from_integer(n.clone()),
            Expr::Rational(r) => *coeff *= r.clone(),
            other if is_one(other) => {}
            other => out.push(other.clone()),
        }
    }
}

fn nilpotent_power_is_zero(expr: &Expr, table: &GradedSymbolTable) -> bool {
    match expr {
        Expr::Pow(base, exp) => nilpotent_exponent(base, exp, table),
        _ => false,
    }
}

fn nilpotent_exponent(base: &Expr, exp: &Expr, table: &GradedSymbolTable) -> bool {
    table.infer_grading(base).is_fermionic() && matches!(exp, Expr::Int(n) if *n > BigInt::one())
}

fn expr_cmp(a: &Expr, b: &Expr, interner: &ax_ir::Interner) -> Ordering {
    canonical_key(a, interner).cmp(&canonical_key(b, interner))
}

fn canonical_key(expr: &Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => format!("00:{n}"),
        Expr::Rational(r) => format!("01:{}/{}", r.numer(), r.denom()),
        Expr::Float(f) => format!("02:{:016x}", f.to_bits()),
        Expr::Sym(s) => format!("03:{}", interner.resolve(*s)),
        Expr::Indexed(base, indices) => format!(
            "04:{}:{:?}",
            canonical_key(base, interner),
            indices
                .iter()
                .map(|idx| (interner.resolve(idx.name).to_string(), &idx.variance))
                .collect::<Vec<_>>()
        ),
        Expr::Call(f, args) => format!(
            "05:{}({})",
            interner.resolve(*f),
            args.iter()
                .map(|arg| canonical_key(arg, interner))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Expr::Pow(base, exp) => format!(
            "06:{}^{}",
            canonical_key(base, interner),
            canonical_key(exp, interner)
        ),
        Expr::Neg(inner) => format!("07:-{}", canonical_key(inner, interner)),
        Expr::Mul(factors) => format!(
            "08:{}",
            factors
                .iter()
                .map(|factor| canonical_key(factor, interner))
                .collect::<Vec<_>>()
                .join("*")
        ),
        Expr::Add(terms) => format!(
            "09:{}",
            terms
                .iter()
                .map(|term| canonical_key(term, interner))
                .collect::<Vec<_>>()
                .join("+")
        ),
        other => format!("99:{other:?}"),
    }
}

fn rational_expr(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Int(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero())
        || matches!(expr, Expr::Rational(r) if r.is_zero())
        || matches!(expr, Expr::Float(f) if *f == 0.0)
}

fn is_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_one()) || matches!(expr, Expr::Rational(r) if r.is_one())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fermions_anticommute() {
        let interner = ax_ir::Interner::new();
        let theta = interner.get_or_intern("theta");
        let eta = interner.get_or_intern("eta");
        let mut table = GradedSymbolTable::new();
        table.declare(theta, Grading::fermionic());
        table.declare(eta, Grading::fermionic());

        let theta_eta = graded_multiply(&[Expr::Sym(theta), Expr::Sym(eta)], &table, &interner);
        let eta_theta = graded_multiply(&[Expr::Sym(eta), Expr::Sym(theta)], &table, &interner);
        assert_eq!(Expr::add(vec![theta_eta, eta_theta]), Expr::zero());
    }

    #[test]
    fn fermion_square_is_zero() {
        let interner = ax_ir::Interner::new();
        let theta = interner.get_or_intern("theta");
        let mut table = GradedSymbolTable::new();
        table.declare(theta, Grading::fermionic());
        assert_eq!(
            graded_simplify(
                &Expr::mul(vec![Expr::Sym(theta), Expr::Sym(theta)]),
                &table,
                &interner
            ),
            Expr::zero()
        );
    }

    #[test]
    fn graded_commutator_of_fermions_is_anticommutator() {
        let interner = ax_ir::Interner::new();
        let theta = interner.get_or_intern("theta");
        let eta = interner.get_or_intern("eta");
        let mut table = GradedSymbolTable::new();
        table.declare(theta, Grading::fermionic());
        table.declare(eta, Grading::fermionic());
        let out = graded_commutator(&Expr::Sym(theta), &Expr::Sym(eta), &table, &interner);
        assert_eq!(out, Expr::zero());
    }

    #[test]
    fn graded_slot_swap_sign_matches_parity_rule() {
        assert_eq!(graded_sign_for_slot_swap(true, true), -1);
        assert_eq!(graded_sign_for_slot_swap(true, false), 1);
    }
}
