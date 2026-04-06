use ax_ir::{Condition, Expr, Index, Interner, Variance};
use lasso::Spur;
use std::sync::atomic::{AtomicUsize, Ordering};

fn equation_parts<'a>(expr: &'a Expr, interner: &Interner) -> Option<(&'a Expr, &'a Expr)> {
    match expr {
        Expr::Call(f, args) if interner.resolve(*f) == "__eq" && args.len() == 2 => {
            Some((&args[0], &args[1]))
        }
        _ => None,
    }
}

pub fn is_equation(expr: &Expr, interner: &Interner) -> bool {
    equation_parts(expr, interner).is_some()
}

pub fn make_equation(lhs: Expr, rhs: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("__eq"), vec![lhs, rhs])
}

pub fn get_lhs(expr: &Expr, interner: &Interner) -> Option<Expr> {
    equation_parts(expr, interner).map(|(lhs, _)| lhs.clone())
}

pub fn get_rhs(expr: &Expr, interner: &Interner) -> Option<Expr> {
    equation_parts(expr, interner).map(|(_, rhs)| rhs.clone())
}

pub fn swap_sides(expr: &Expr, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => make_equation(rhs.clone(), lhs.clone(), interner),
        None => expr.clone(),
    }
}

pub fn equation_to_rule(expr: &Expr, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => Expr::Rule(
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
            ax_ir::TrustLevel::Exact,
        ),
        None => expr.clone(),
    }
}

pub fn equation_to_subrule(expr: &Expr, interner: &Interner) -> Expr {
    equation_to_rule(expr, interner)
}

pub fn multiply_through(expr: &Expr, factor: &Expr, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => make_equation(
            Expr::mul(vec![lhs.clone(), factor.clone()]),
            Expr::mul(vec![rhs.clone(), factor.clone()]),
            interner,
        ),
        None => Expr::mul(vec![expr.clone(), factor.clone()]),
    }
}

pub fn add_through(expr: &Expr, term: &Expr, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => make_equation(
            Expr::add(vec![lhs.clone(), term.clone()]),
            Expr::add(vec![rhs.clone(), term.clone()]),
            interner,
        ),
        None => Expr::add(vec![expr.clone(), term.clone()]),
    }
}

pub fn apply_through(
    expr: &Expr,
    operation: &dyn Fn(&Expr, &Interner) -> Expr,
    interner: &Interner,
) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => {
            make_equation(operation(lhs, interner), operation(rhs, interner), interner)
        }
        None => operation(expr, interner),
    }
}

pub fn to_rhs(expr: &Expr, target: &Expr, interner: &Interner) -> Expr {
    let Some((lhs, rhs)) = equation_parts(expr, interner) else {
        return expr.clone();
    };

    if let Expr::Add(terms) = lhs {
        let (moved, remaining): (Vec<_>, Vec<_>) = terms
            .iter()
            .cloned()
            .partition(|term| expr_contains(term, target));

        if moved.is_empty() {
            return expr.clone();
        }

        let moved_to_rhs = moved.into_iter().map(Expr::neg).collect::<Vec<_>>();
        make_equation(
            Expr::add(remaining),
            Expr::add(std::iter::once(rhs.clone()).chain(moved_to_rhs).collect()),
            interner,
        )
    } else if expr_contains(lhs, target) {
        make_equation(
            Expr::zero(),
            Expr::add(vec![rhs.clone(), Expr::neg(lhs.clone())]),
            interner,
        )
    } else {
        expr.clone()
    }
}

fn condition_contains(condition: &Condition, target: &Expr) -> bool {
    match condition {
        Condition::Gt(lhs, rhs)
        | Condition::Lt(lhs, rhs)
        | Condition::Ge(lhs, rhs)
        | Condition::Le(lhs, rhs)
        | Condition::Eq(lhs, rhs)
        | Condition::Ne(lhs, rhs) => expr_contains(lhs, target) || expr_contains(rhs, target),
        Condition::And(lhs, rhs) | Condition::Or(lhs, rhs) => {
            condition_contains(lhs, target) || condition_contains(rhs, target)
        }
        Condition::Not(inner) => condition_contains(inner, target),
        Condition::True | Condition::False => false,
    }
}

pub fn expr_contains(expr: &Expr, target: &Expr) -> bool {
    if expr == target {
        return true;
    }

    match expr {
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| expr_contains(term, target))
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) | Expr::Rule(base, exp, _) => {
            expr_contains(base, target) || expr_contains(exp, target)
        }
        Expr::Neg(inner) | Expr::Indexed(inner, _) => expr_contains(inner, target),
        Expr::Call(_, args) => args.iter().any(|arg| expr_contains(arg, target)),
        Expr::FnDef(_, _, body) => expr_contains(body, target),
        Expr::Piecewise(branches) => branches.iter().any(|(value, condition)| {
            expr_contains(value, target) || condition_contains(condition, target)
        }),
        Expr::Let(_, value, body) => expr_contains(value, target) || expr_contains(body, target),
        Expr::Matrix(rows) => rows
            .iter()
            .flat_map(|row| row.iter())
            .any(|entry| expr_contains(entry, target)),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => false,
    }
}

pub fn to_lhs(expr: &Expr, target: &Expr, interner: &Interner) -> Expr {
    swap_sides(
        &to_rhs(&swap_sides(expr, interner), target, interner),
        interner,
    )
}

fn separate_lhs_terms(expr: &Expr, target: &Expr, interner: &Interner) -> Expr {
    let Some((lhs, rhs)) = equation_parts(expr, interner) else {
        return expr.clone();
    };

    let Expr::Add(terms) = lhs else {
        return expr.clone();
    };

    let (containing, not_containing): (Vec<_>, Vec<_>) = terms
        .iter()
        .cloned()
        .partition(|term| expr_contains(term, target));

    if containing.is_empty() || not_containing.is_empty() {
        return expr.clone();
    }

    let moved = not_containing
        .into_iter()
        .map(Expr::neg)
        .collect::<Vec<_>>();
    make_equation(
        Expr::add(containing),
        Expr::add(std::iter::once(rhs.clone()).chain(moved).collect()),
        interner,
    )
}

fn isolate_product(lhs: &Expr, rhs: &Expr, target: &Expr, interner: &Interner) -> Option<Expr> {
    if lhs == target {
        return Some(make_equation(target.clone(), rhs.clone(), interner));
    }

    let Expr::Mul(factors) = lhs else {
        return None;
    };

    let target_positions = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (factor == target).then_some(idx))
        .collect::<Vec<_>>();

    if target_positions.len() != 1 {
        return None;
    }

    let target_idx = target_positions[0];
    let other_factors = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != target_idx).then_some(factor.clone()))
        .collect::<Vec<_>>();
    let coefficient = Expr::mul(other_factors);
    let isolated_rhs = if coefficient == Expr::one() {
        rhs.clone()
    } else {
        Expr::mul(vec![
            rhs.clone(),
            Expr::pow(coefficient, Expr::Int((-1).into())),
        ])
    };
    Some(make_equation(target.clone(), isolated_rhs, interner))
}

pub fn isolate(expr: &Expr, target: &Expr, interner: &Interner) -> Expr {
    let mut separated = to_lhs(expr, target, interner);
    separated = separate_lhs_terms(&separated, target, interner);

    let Some((lhs, rhs)) = equation_parts(&separated, interner) else {
        return separated;
    };

    if lhs == target {
        return make_equation(target.clone(), rhs.clone(), interner);
    }

    if let Some(result) = isolate_product(lhs, rhs, target, interner) {
        return result;
    }

    if let Expr::Neg(inner) = lhs {
        if expr_contains(inner, target) {
            let negated = make_equation((**inner).clone(), Expr::neg(rhs.clone()), interner);
            return isolate(&negated, target, interner);
        }
    }

    if let Expr::Add(terms) = lhs {
        let target_terms = terms
            .iter()
            .filter(|term| {
                *term == target || matches!(term, Expr::Mul(_) if expr_contains(term, target))
            })
            .cloned()
            .collect::<Vec<_>>();
        if target_terms.len() == 1 {
            let focus = target_terms[0].clone();
            let others = terms
                .iter()
                .filter(|term| **term != focus)
                .cloned()
                .map(Expr::neg)
                .collect::<Vec<_>>();
            let reduced = make_equation(
                focus,
                Expr::add(std::iter::once(rhs.clone()).chain(others).collect()),
                interner,
            );
            return isolate(&reduced, target, interner);
        }
    }

    separated
}

pub fn get_factor(expr: &Expr, target: &Expr) -> (Expr, Expr) {
    if expr == target {
        return (Expr::one(), target.clone());
    }

    match expr {
        Expr::Mul(factors) => {
            let (containing, not_containing): (Vec<_>, Vec<_>) = factors
                .iter()
                .cloned()
                .partition(|factor| expr_contains(factor, target));
            (Expr::mul(not_containing), Expr::mul(containing))
        }
        Expr::Neg(inner) => {
            let (coefficient, remainder) = get_factor(inner, target);
            (Expr::neg(coefficient), remainder)
        }
        _ => (Expr::one(), expr.clone()),
    }
}

pub fn multiply_through_indexed(
    expr: &Expr,
    factor: &Expr,
    side: &str,
    interner: &Interner,
) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => match side {
            "left" => make_equation(
                Expr::mul(vec![factor.clone(), lhs.clone()]),
                Expr::mul(vec![factor.clone(), rhs.clone()]),
                interner,
            ),
            "right" => make_equation(
                Expr::mul(vec![lhs.clone(), factor.clone()]),
                Expr::mul(vec![rhs.clone(), factor.clone()]),
                interner,
            ),
            _ => expr.clone(),
        },
        None => match side {
            "left" => Expr::mul(vec![factor.clone(), expr.clone()]),
            "right" => Expr::mul(vec![expr.clone(), factor.clone()]),
            _ => expr.clone(),
        },
    }
}

pub fn contract_through(expr: &Expr, tensor: &Expr, interner: &Interner) -> Expr {
    multiply_through_indexed(expr, tensor, "left", interner)
}

fn fresh_equation_dummy(interner: &Interner) -> Spur {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    interner.get_or_intern(&format!("_eq{n}"))
}

fn indexed_metric_factor(sym: Spur, first: Spur, second: Spur, variance: Variance) -> Expr {
    Expr::Indexed(
        Box::new(Expr::Sym(sym)),
        vec![
            Index {
                name: first,
                variance: variance.clone(),
                index_type: None,
            },
            Index {
                name: second,
                variance,
                index_type: None,
            },
        ],
    )
}

pub fn raise_equation(
    expr: &Expr,
    metric_sym: Spur,
    inv_metric_sym: Spur,
    index_to_raise: Spur,
    interner: &Interner,
) -> Expr {
    let fresh = fresh_equation_dummy(interner);
    let factor = indexed_metric_factor(inv_metric_sym, index_to_raise, fresh, Variance::Up);
    apply_through(
        &multiply_through_indexed(expr, &factor, "left", interner),
        &|side, interner| ax_tensor::eliminate_metric(side, metric_sym, inv_metric_sym, interner),
        interner,
    )
}

pub fn lower_equation(
    expr: &Expr,
    metric_sym: Spur,
    inv_metric_sym: Spur,
    index_to_lower: Spur,
    interner: &Interner,
) -> Expr {
    let fresh = fresh_equation_dummy(interner);
    let factor = indexed_metric_factor(metric_sym, index_to_lower, fresh, Variance::Down);
    apply_through(
        &multiply_through_indexed(expr, &factor, "left", interner),
        &|side, interner| ax_tensor::eliminate_metric(side, metric_sym, inv_metric_sym, interner),
        interner,
    )
}

fn has_indexed(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => terms.iter().any(has_indexed),
        Expr::Pow(base, exp) | Expr::Complex(base, exp) | Expr::Rule(base, exp, _) => {
            has_indexed(base) || has_indexed(exp)
        }
        Expr::Neg(inner) => has_indexed(inner),
        Expr::Call(_, args) => args.iter().any(has_indexed),
        Expr::FnDef(_, _, body) => has_indexed(body),
        Expr::Piecewise(branches) => branches
            .iter()
            .any(|(value, condition)| has_indexed(value) || condition_has_indexed(condition)),
        Expr::Let(_, value, body) => has_indexed(value) || has_indexed(body),
        Expr::Matrix(rows) => rows.iter().flat_map(|row| row.iter()).any(has_indexed),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => false,
    }
}

fn condition_has_indexed(condition: &Condition) -> bool {
    match condition {
        Condition::Gt(lhs, rhs)
        | Condition::Lt(lhs, rhs)
        | Condition::Ge(lhs, rhs)
        | Condition::Le(lhs, rhs)
        | Condition::Eq(lhs, rhs)
        | Condition::Ne(lhs, rhs) => has_indexed(lhs) || has_indexed(rhs),
        Condition::And(lhs, rhs) | Condition::Or(lhs, rhs) => {
            condition_has_indexed(lhs) || condition_has_indexed(rhs)
        }
        Condition::Not(inner) => condition_has_indexed(inner),
        Condition::True | Condition::False => false,
    }
}

fn substitute_one_side(
    expr: &Expr,
    target: &Expr,
    replacement: &Expr,
    interner: &Interner,
) -> Expr {
    if has_indexed(target) || has_indexed(replacement) {
        crate::substitute_with_indices(expr, target, replacement, &crate::Env::new(), interner)
    } else {
        crate::symbolic_substitute(expr, target, replacement, interner)
    }
}

pub fn substitute_equation(
    expr: &Expr,
    target: &Expr,
    replacement: &Expr,
    interner: &Interner,
) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => make_equation(
            substitute_one_side(lhs, target, replacement, interner),
            substitute_one_side(rhs, target, replacement, interner),
            interner,
        ),
        None => substitute_one_side(expr, target, replacement, interner),
    }
}

pub fn differentiate_equation(expr: &Expr, var: Spur, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => make_equation(
            crate::differentiate(lhs, var, interner),
            crate::differentiate(rhs, var, interner),
            interner,
        ),
        None => crate::differentiate(expr, var, interner),
    }
}

pub fn integrate_equation(expr: &Expr, var: Spur, interner: &Interner) -> Expr {
    match equation_parts(expr, interner) {
        Some((lhs, rhs)) => {
            let constant = Expr::Sym(interner.get_or_intern("C"));
            make_equation(
                crate::integrate::integrate(lhs, var, interner),
                Expr::add(vec![
                    crate::integrate::integrate(rhs, var, interner),
                    constant,
                ]),
                interner,
            )
        }
        None => crate::integrate::integrate(expr, var, interner),
    }
}
