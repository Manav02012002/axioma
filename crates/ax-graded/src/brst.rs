use crate::{graded_multiply, graded_simplify, GradedSymbolTable, Grading, GradingValue};
use ax_ir::{Expr, Interner};
use lasso::Spur;
use num_traits::ToPrimitive;

#[derive(Clone, Debug)]
pub struct BRSTSetup {
    pub fields: Vec<BRSTField>,
    pub brst_rules: Vec<(Spur, Expr)>,
}

#[derive(Clone, Debug)]
pub struct BRSTField {
    pub name: Spur,
    pub ghost_number: i32,
    pub statistics: Grading,
}

pub fn setup_yang_mills_brst(
    gauge_field: Spur,
    ghost: Spur,
    antighost: Spur,
    nakanishi_lautrup: Spur,
    _coupling: Spur,
    interner: &Interner,
) -> (BRSTSetup, GradedSymbolTable) {
    let mut table = GradedSymbolTable::new();
    let fields = vec![
        BRSTField {
            name: gauge_field,
            ghost_number: 0,
            statistics: Grading::bosonic(),
        },
        BRSTField {
            name: ghost,
            ghost_number: 1,
            statistics: Grading::fermionic(),
        },
        BRSTField {
            name: antighost,
            ghost_number: -1,
            statistics: Grading::fermionic(),
        },
        BRSTField {
            name: nakanishi_lautrup,
            ghost_number: 0,
            statistics: Grading::bosonic(),
        },
    ];

    for field in &fields {
        table.declare(
            field.name,
            Grading::Product(vec![
                (
                    "statistics".to_string(),
                    statistics_value(&field.statistics),
                ),
                (
                    "ghost".to_string(),
                    GradingValue::Integer(field.ghost_number),
                ),
            ]),
        );
    }

    let partial = interner.get_or_intern("partial");
    let brst_rules = vec![
        (gauge_field, Expr::Call(partial, vec![Expr::Sym(ghost)])),
        (ghost, Expr::zero()),
        (antighost, Expr::Sym(nakanishi_lautrup)),
        (nakanishi_lautrup, Expr::zero()),
    ];

    (BRSTSetup { fields, brst_rules }, table)
}

pub fn apply_brst(
    expr: &Expr,
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let out = match expr {
        Expr::Sym(s) => setup
            .brst_rules
            .iter()
            .find(|(field, _)| field == s)
            .map(|(_, rhs)| rhs.clone())
            .unwrap_or_else(Expr::zero),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| apply_brst(term, setup, table, interner))
                .collect(),
        ),
        Expr::Mul(factors) => apply_brst_product(factors, setup, table, interner),
        Expr::Pow(base, exp) => apply_brst_power(base, exp, setup, table, interner),
        Expr::Neg(inner) => Expr::neg(apply_brst(inner, setup, table, interner)),
        Expr::Call(f, args) => {
            let derived_args = args
                .iter()
                .enumerate()
                .filter_map(|(idx, _)| {
                    let derived = apply_brst(&args[idx], setup, table, interner);
                    if derived == Expr::zero() {
                        return None;
                    }
                    let mut call_args = args.clone();
                    call_args[idx] = derived;
                    Some(Expr::Call(*f, call_args))
                })
                .collect();
            Expr::add(derived_args)
        }
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(apply_brst(base, setup, table, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(apply_brst(inner, setup, table, interner)),
            *rel,
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| apply_brst(item, setup, table, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| apply_brst(item, setup, table, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(apply_brst(value, setup, table, interner)),
            Box::new(apply_brst(body, setup, table, interner)),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(apply_brst(re, setup, table, interner)),
            Box::new(apply_brst(im, setup, table, interner)),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(apply_brst(body, setup, table, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(apply_brst(lhs, setup, table, interner)),
            Box::new(apply_brst(rhs, setup, table, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (apply_brst(value, setup, table, interner), condition.clone())
                })
                .collect(),
        ),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => Expr::zero(),
    };
    graded_simplify(&out, table, interner)
}

pub fn verify_nilpotency(
    field: Spur,
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let first = apply_brst(&Expr::Sym(field), setup, table, interner);
    apply_brst(&first, setup, table, interner)
}

pub fn ghost_number(expr: &Expr, table: &GradedSymbolTable) -> Option<i32> {
    match expr {
        Expr::Sym(s) => Some(symbol_ghost_number(*s, table)),
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Complex(_, _) => Some(0),
        Expr::Add(terms) => {
            let mut iter = terms.iter().filter_map(|term| ghost_number(term, table));
            let first = iter.next().unwrap_or(0);
            if iter.all(|n| n == first) {
                Some(first)
            } else {
                None
            }
        }
        Expr::Mul(factors) => factors.iter().try_fold(0, |acc, factor| {
            ghost_number(factor, table).map(|n| acc + n)
        }),
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Int(n) => ghost_number(base, table).and_then(|g| n.to_i32().map(|p| g * p)),
            _ => None,
        },
        Expr::Neg(inner) | Expr::Indexed(inner, _) => ghost_number(inner, table),
        Expr::Group(inner, _) => ghost_number(inner, table),
        Expr::Call(f, args) => {
            let f_ghost = symbol_ghost_number(*f, table);
            args.iter().try_fold(f_ghost, |acc, arg| {
                ghost_number(arg, table).map(|n| acc + n)
            })
        }
        Expr::List(items) => ghost_number_of_collection(items, table),
        Expr::Matrix(rows) => {
            let items = rows.iter().flatten().cloned().collect::<Vec<_>>();
            ghost_number_of_collection(&items, table)
        }
        Expr::Let(_, _, body) => ghost_number(body, table),
        Expr::FnDef(_, _, _)
        | Expr::Rule(_, _, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Piecewise(_) => Some(0),
    }
}

pub fn filter_by_ghost_number(
    expr: &Expr,
    target: i32,
    table: &GradedSymbolTable,
    _interner: &Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .filter(|term| ghost_number(term, table) == Some(target))
                .cloned()
                .collect(),
        ),
        other if ghost_number(other, table) == Some(target) => other.clone(),
        _ => Expr::zero(),
    }
}

pub fn brst_exact_check(
    expr: &Expr,
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> bool {
    let target_ghost = ghost_number(expr, table).map(|n| n - 1);
    setup.fields.iter().any(|field| {
        Some(field.ghost_number) == target_ghost
            && graded_simplify(
                &apply_brst(&Expr::Sym(field.name), setup, table, interner),
                table,
                interner,
            ) == graded_simplify(expr, table, interner)
    })
}

pub fn brst_closed_check(
    expr: &Expr,
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> bool {
    graded_simplify(&apply_brst(expr, setup, table, interner), table, interner) == Expr::zero()
}

fn apply_brst_product(
    factors: &[Expr],
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let mut terms = Vec::new();
    for (idx, factor) in factors.iter().enumerate() {
        let derived = apply_brst(factor, setup, table, interner);
        if derived == Expr::zero() {
            continue;
        }
        let sign = factors[..idx]
            .iter()
            .filter_map(|prior| ghost_number(prior, table))
            .fold(1, |acc, n| if n.rem_euclid(2) == 0 { acc } else { -acc });
        let mut product = factors[..idx].to_vec();
        product.push(derived);
        product.extend_from_slice(&factors[idx + 1..]);
        let term = graded_multiply(&product, table, interner);
        terms.push(if sign < 0 { Expr::neg(term) } else { term });
    }
    Expr::add(terms)
}

fn apply_brst_power(
    base: &Expr,
    exp: &Expr,
    setup: &BRSTSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    match exp {
        Expr::Int(n) => {
            let Some(power) = n.to_usize() else {
                return Expr::zero();
            };
            if power == 0 {
                return Expr::zero();
            }
            let terms = (0..power)
                .map(|idx| {
                    let mut factors = Vec::new();
                    factors.extend(std::iter::repeat(base.clone()).take(idx));
                    factors.push(apply_brst(base, setup, table, interner));
                    factors.extend(std::iter::repeat(base.clone()).take(power - idx - 1));
                    graded_multiply(&factors, table, interner)
                })
                .collect();
            Expr::add(terms)
        }
        _ => Expr::zero(),
    }
}

fn statistics_value(statistics: &Grading) -> GradingValue {
    if statistics.is_fermionic() {
        GradingValue::Mod2(1)
    } else {
        GradingValue::Mod2(0)
    }
}

fn symbol_ghost_number(sym: Spur, table: &GradedSymbolTable) -> i32 {
    match table.get(sym) {
        Some(Grading::Z(n)) => *n,
        Some(Grading::Product(values)) => values
            .iter()
            .find_map(|(name, value)| match (name.as_str(), value) {
                ("ghost", GradingValue::Integer(n)) => Some(*n),
                _ => None,
            })
            .unwrap_or(0),
        _ => 0,
    }
}

fn ghost_number_of_collection(items: &[Expr], table: &GradedSymbolTable) -> Option<i32> {
    let mut iter = items.iter().filter_map(|item| ghost_number(item, table));
    let first = iter.next().unwrap_or(0);
    if iter.all(|n| n == first) {
        Some(first)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yang_mills_assigns_ghost_numbers() {
        let interner = Interner::new();
        let a = interner.get_or_intern("A");
        let c = interner.get_or_intern("c");
        let cbar = interner.get_or_intern("c_bar");
        let b = interner.get_or_intern("B");
        let g = interner.get_or_intern("g");
        let (_, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
        assert_eq!(ghost_number(&Expr::Sym(c), &table), Some(1));
        assert_eq!(ghost_number(&Expr::Sym(cbar), &table), Some(-1));
    }

    #[test]
    fn brst_antighost_maps_to_b() {
        let interner = Interner::new();
        let a = interner.get_or_intern("A");
        let c = interner.get_or_intern("c");
        let cbar = interner.get_or_intern("c_bar");
        let b = interner.get_or_intern("B");
        let g = interner.get_or_intern("g");
        let (setup, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
        assert_eq!(
            apply_brst(&Expr::Sym(cbar), &setup, &table, &interner),
            Expr::Sym(b)
        );
        assert_eq!(
            verify_nilpotency(cbar, &setup, &table, &interner),
            Expr::zero()
        );
    }

    #[test]
    fn filter_keeps_matching_ghost_number_terms() {
        let interner = Interner::new();
        let a = interner.get_or_intern("A");
        let c = interner.get_or_intern("c");
        let cbar = interner.get_or_intern("c_bar");
        let b = interner.get_or_intern("B");
        let g = interner.get_or_intern("g");
        let (_, table) = setup_yang_mills_brst(a, c, cbar, b, g, &interner);
        let expr = Expr::add(vec![Expr::Sym(a), Expr::Sym(c)]);
        assert_eq!(
            filter_by_ghost_number(&expr, 1, &table, &interner),
            Expr::Sym(c)
        );
    }
}
