#![forbid(unsafe_code)]

pub mod integrate;
pub mod series;
pub mod simplify;

use ax_ir::{Assumption, Condition, Expr};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct Env {
    pub bindings: HashMap<lasso::Spur, Expr>,
    pub parent: Option<Box<Env>>,
    pub rules: Vec<ax_rewrite::RewriteRule>,
    pub assumptions: HashMap<lasso::Spur, Vec<Assumption>>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
            rules: Vec::new(),
            assumptions: HashMap::new(),
        }
    }

    pub fn lookup(&self, sym: lasso::Spur) -> Option<&Expr> {
        self.bindings
            .get(&sym)
            .or_else(|| self.parent.as_deref().and_then(|parent| parent.lookup(sym)))
    }

    pub fn extend(&self, sym: lasso::Spur, val: Expr) -> Env {
        let mut bindings = HashMap::new();
        bindings.insert(sym, val);
        Env {
            bindings,
            parent: Some(Box::new(self.clone())),
            rules: self.rules.clone(),
            assumptions: self.assumptions.clone(),
        }
    }
}

fn has_assumption(env: &Env, sym: lasso::Spur, assumption: &Assumption) -> bool {
    env.assumptions
        .get(&sym)
        .is_some_and(|assumptions| assumptions.contains(assumption))
        || env
            .parent
            .as_deref()
            .is_some_and(|parent| has_assumption(parent, sym, assumption))
}

fn to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = to_rational(base)?;
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

fn perfect_square_root(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    let root = n.sqrt();
    if &root * &root == *n {
        Some(root)
    } else {
        None
    }
}

fn one_half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn diff_call(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    let diff_sym = interner.get_or_intern("diff");
    Expr::Call(diff_sym, vec![expr.clone(), Expr::Sym(var)])
}

fn builtin_unary(name: &str, arg: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern(name), vec![arg])
}

fn collapse_duplicate_sum_terms(terms: Vec<Expr>) -> Expr {
    let mut grouped: Vec<(Expr, usize)> = Vec::new();

    for term in terms {
        if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == term) {
            *count += 1;
        } else {
            grouped.push((term, 1));
        }
    }

    Expr::add(
        grouped
            .into_iter()
            .map(|(term, count)| {
                if count == 1 {
                    term
                } else {
                    Expr::mul(vec![Expr::Int((count as i64).into()), term])
                }
            })
            .collect(),
    )
}

fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        Expr::Sym(s) => *s == var,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_var(term, var))
        }
        Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
        Expr::Neg(e) => contains_var(e, var),
        Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
        Expr::FnDef(_, _, body) => contains_var(body, var),
        Expr::Rule(lhs, rhs) => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::Import(_) => false,
        Expr::Assume(_, _) => false,
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, condition)| contains_var(value, var) || condition_contains_var(condition, var)),
        Expr::Indexed(base, _) => contains_var(base, var),
        Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_var(cell, var))),
    }
}

fn condition_contains_var(condition: &Condition, var: lasso::Spur) -> bool {
    match condition {
        Condition::Gt(a, b)
        | Condition::Lt(a, b)
        | Condition::Ge(a, b)
        | Condition::Le(a, b)
        | Condition::Eq(a, b)
        | Condition::Ne(a, b) => contains_var(a, var) || contains_var(b, var),
        Condition::And(a, b) | Condition::Or(a, b) => {
            condition_contains_var(a, var) || condition_contains_var(b, var)
        }
        Condition::Not(c) => condition_contains_var(c, var),
        Condition::True | Condition::False => false,
    }
}

fn compare_numeric(a: &Expr, b: &Expr) -> Option<Ordering> {
    let fa = to_f64(a)?;
    let fb = to_f64(b)?;
    fa.partial_cmp(&fb)
}

fn eval_condition(cond: &Condition, env: &Env, interner: &ax_ir::Interner) -> Option<bool> {
    match cond {
        Condition::True => Some(true),
        Condition::False => Some(false),
        Condition::Gt(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord == Ordering::Greater)
        }
        Condition::Lt(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord == Ordering::Less)
        }
        Condition::Ge(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord != Ordering::Less)
        }
        Condition::Le(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord != Ordering::Greater)
        }
        Condition::Eq(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            Some(ea == eb)
        }
        Condition::Ne(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            Some(ea != eb)
        }
        Condition::And(a, b) => Some(eval_condition(a, env, interner)? && eval_condition(b, env, interner)?),
        Condition::Or(a, b) => Some(eval_condition(a, env, interner)? || eval_condition(b, env, interner)?),
        Condition::Not(c) => eval_condition(c, env, interner).map(|v| !v),
    }
}

fn expr_to_pattern(expr: &Expr, interner: &ax_ir::Interner) -> ax_rewrite::Pattern {
    match expr {
        Expr::Sym(s) => {
            let name = interner.resolve(*s);
            if name.ends_with('_') {
                ax_rewrite::Pattern::Slot(*s)
            } else {
                ax_rewrite::Pattern::Exact(expr.clone())
            }
        }
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => {
            ax_rewrite::Pattern::Exact(expr.clone())
        }
        Expr::Add(terms) => {
            ax_rewrite::Pattern::Add(terms.iter().map(|t| expr_to_pattern(t, interner)).collect())
        }
        Expr::Mul(factors) => ax_rewrite::Pattern::Mul(
            factors
                .iter()
                .map(|f| expr_to_pattern(f, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => ax_rewrite::Pattern::Pow(
            Box::new(expr_to_pattern(base, interner)),
            Box::new(expr_to_pattern(exp, interner)),
        ),
        Expr::Neg(e) => ax_rewrite::Pattern::Neg(Box::new(expr_to_pattern(e, interner))),
        Expr::Call(f, args) => ax_rewrite::Pattern::Call(
            *f,
            args.iter().map(|a| expr_to_pattern(a, interner)).collect(),
        ),
        _ => ax_rewrite::Pattern::Exact(expr.clone()),
    }
}

pub fn register_rule(
    rule_expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    if let Expr::Rule(lhs, rhs) = rule_expr {
        let rule = ax_rewrite::RewriteRule {
            name: format!("user_rule_{}", env.rules.len()),
            pattern: expr_to_pattern(lhs, interner),
            replacement: *rhs.clone(),
            condition: None,
        };
        let name = rule.name.clone();
        env.rules.push(rule);
        Some(name)
    } else {
        None
    }
}

pub fn resolve_import(
    path: &[lasso::Spur],
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Option<PathBuf> {
    if path.is_empty() {
        return None;
    }

    let parts = path
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect::<Vec<_>>();

    let mut rel = PathBuf::new();
    for part in &parts {
        rel.push(part);
    }
    rel.set_extension("ax");

    let rel_without_std = if parts.first().is_some_and(|part| part == "std") && parts.len() > 1 {
        let mut p = PathBuf::new();
        for part in &parts[1..] {
            p.push(part);
        }
        p.set_extension("ax");
        Some(p)
    } else {
        None
    };

    for root in search_paths {
        let candidate = root.join(&rel);
        if candidate.is_file() {
            return Some(candidate);
        }

        if let Some(rel_without_std) = &rel_without_std {
            let candidate = root.join(rel_without_std);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

pub fn differentiate(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::Int(0.into()),
        Expr::Sym(s) => {
            if *s == var {
                Expr::Int(1.into())
            } else {
                Expr::Int(0.into())
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| differentiate(term, var, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(differentiate(e, var, interner)),
        Expr::Mul(factors) => {
            let terms = factors
                .iter()
                .enumerate()
                .map(|(i, factor)| {
                    let mut product = Vec::with_capacity(factors.len());
                    product.extend(factors[..i].iter().cloned());
                    product.push(differentiate(factor, var, interner));
                    product.extend(factors[i + 1..].iter().cloned());
                    Expr::mul(product)
                })
                .collect();
            collapse_duplicate_sum_terms(terms)
        }
        Expr::Pow(base, exp) => {
            if !contains_var(exp, var) {
                Expr::mul(vec![
                    exp.as_ref().clone(),
                    Expr::pow(
                        base.as_ref().clone(),
                        Expr::add(vec![exp.as_ref().clone(), Expr::neg(Expr::one())]),
                    ),
                    differentiate(base, var, interner),
                ])
            } else if !contains_var(base, var) {
                match base.as_ref() {
                    Expr::Sym(sym) if interner.resolve(*sym) == "e" => {
                        Expr::mul(vec![expr.clone(), differentiate(exp, var, interner)])
                    }
                    Expr::Call(f, args) if interner.resolve(*f) == "exp" && args.len() == 1 => {
                        Expr::mul(vec![expr.clone(), differentiate(exp, var, interner)])
                    }
                    _ => diff_call(expr, var, interner),
                }
            } else {
                diff_call(expr, var, interner)
            }
        }
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            if args.len() != 1 {
                return diff_call(expr, var, interner);
            }

            let arg = args[0].clone();
            let darg = differentiate(&args[0], var, interner);
            match name {
                "sin" => Expr::mul(vec![builtin_unary("cos", arg, interner), darg]),
                "cos" => Expr::mul(vec![Expr::neg(builtin_unary("sin", arg, interner)), darg]),
                "exp" => Expr::mul(vec![builtin_unary("exp", arg, interner), darg]),
                "log" => Expr::mul(vec![Expr::pow(arg, Expr::neg(Expr::one())), darg]),
                "sqrt" => differentiate(&Expr::pow(arg, one_half()), var, interner),
                _ => diff_call(expr, var, interner),
            }
        }
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(differentiate(body, var, interner)),
        ),
        Expr::Rule(lhs, rhs) => Expr::Rule(
            Box::new(differentiate(lhs, var, interner)),
            Box::new(differentiate(rhs, var, interner)),
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases.iter()
                .map(|(value, condition)| (differentiate(value, var, interner), condition.clone()))
                .collect(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            val.clone(),
            Box::new(differentiate(body, var, interner)),
        ),
        Expr::Indexed(_, _) => diff_call(expr, var, interner),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| differentiate(item, var, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| differentiate(cell, var, interner))
                        .collect()
                })
                .collect(),
        ),
    }
}

fn builtin_call(
    name: &str,
    f: lasso::Spur,
    args: Vec<Expr>,
    interner: &ax_ir::Interner,
    env: &Env,
) -> Expr {
    match name {
        "N" => {
            if args.len() == 1 {
                if let Some(v) = to_f64(&args[0]) {
                    Expr::Float(v)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sin" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.sin()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cos" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.cos()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "exp" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.exp()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "log" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v > 0.0 => Expr::Float(v.ln()),
                    Expr::Int(n) if n.is_one() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sqrt" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v >= 0.0 => Expr::Float(v.sqrt()),
                    Expr::Int(n) => {
                        if let Some(root) = perfect_square_root(n) {
                            Expr::Int(root)
                        } else {
                            Expr::pow(args[0].clone(), one_half())
                        }
                    }
                    Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into()) => {
                        match base.as_ref() {
                            Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Positive) => {
                                Expr::Sym(*sym)
                            }
                            Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Real) => {
                                Expr::Call(interner.get_or_intern("abs"), vec![Expr::Sym(*sym)])
                            }
                            _ => Expr::pow(args[0].clone(), one_half()),
                        }
                    }
                    _ => Expr::pow(args[0].clone(), one_half()),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "abs" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Int(n) => Expr::Int(n.abs()),
                    Expr::Float(v) => Expr::Float(v.abs()),
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Positive) => {
                        Expr::Sym(*sym)
                    }
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Negative) => {
                        Expr::neg(Expr::Sym(*sym))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "expand" => {
            if args.len() == 1 {
                simplify::expand(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "simplify" => {
            if args.len() == 1 {
                simplify::simplify(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "rewrite" => {
            if args.len() == 1 {
                ax_rewrite::rewrite_fixed_point(&env.rules, &args[0], interner, 100)
            } else {
                Expr::Call(f, args)
            }
        }
        "solve" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (expr, Expr::Sym(var)) => ax_solve::solve(expr, *var, interner),
                    (Expr::List(equations), Expr::List(vars_expr)) => {
                        let vars = vars_expr
                            .iter()
                            .map(|expr| {
                                if let Expr::Sym(sym) = expr {
                                    Some(*sym)
                                } else {
                                    None
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(vars) = vars {
                            if let Some(solution) =
                                ax_solve::solve_linear_system(equations, &vars, interner)
                            {
                                Expr::List(solution.into_iter().map(|(_, value)| value).collect())
                            } else {
                                Expr::Call(f, args)
                            }
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "det" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => ax_linalg::determinant(rows, interner),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "inv" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => {
                        if let Some(inv) = ax_linalg::inverse(rows, interner) {
                            Expr::Matrix(inv)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "transpose" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => Expr::Matrix(ax_linalg::transpose(rows)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "trace_mat" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => ax_linalg::trace(rows),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eigenvalues" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => {
                        if rows.len() == 2 && rows.iter().all(|row| row.len() == 2) {
                            Expr::List(ax_linalg::eigenvalues_2x2(rows, interner))
                        } else {
                            ax_linalg::eigenvalues_symbolic(rows, interner)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "matmul" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_linalg::mat_mul(a, b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "identity" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Int(n) => {
                        if let Some(dim) = num_traits::ToPrimitive::to_usize(n) {
                            Expr::Matrix(ax_linalg::identity(dim))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_product" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_linalg::tensor_product(a, b))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "diff" => {
            if args.len() == 2 {
                if let Expr::Sym(var_sym) = args[1] {
                    let diffed = differentiate(&args[0], var_sym, interner);
                    let diff_sym = interner.get_or_intern("diff");
                    if matches!(&diffed, Expr::Call(sym, inner_args) if *sym == diff_sym && inner_args == &args)
                    {
                        diffed
                    } else {
                        eval(&diffed, &Env::new(), interner)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "christoffel" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (metric_expr, Expr::List(coords_exprs)) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            let coords = coords_exprs
                                .iter()
                                .map(|expr| {
                                    if let Expr::Sym(sym) = expr {
                                        Some(*sym)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Option<Vec<_>>>();
                            if let Some(coords) = coords {
                                expr_3d_to_list(ax_tensor::christoffel_from_metric(
                                    &metric, &coords, interner,
                                ))
                            } else {
                                Expr::Call(f, args)
                            }
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "riemann" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), &args[1]) {
                    (Some(gamma), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| {
                                if let Expr::Sym(sym) = expr {
                                    Some(*sym)
                                } else {
                                    None
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            expr_4d_to_list(ax_tensor::riemann_from_christoffel(
                                &gamma, &coords, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ricci" => {
            if args.len() == 1 {
                if let Some(riemann) = expr_to_4d(&args[0]) {
                    let n = riemann.len();
                    Expr::Matrix(ax_tensor::ricci_from_riemann(&riemann, n, interner))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ricci_scalar" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(ricci), ginv_expr) => {
                        if let Some(ginv) = matrix_to_symbolic(ginv_expr) {
                            ax_tensor::ricci_scalar(ricci, &ginv, interner)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "einstein" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (Expr::Matrix(ricci), scalar, metric_expr) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            Expr::Matrix(ax_tensor::einstein_tensor(
                                ricci, scalar, &metric, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "kretschner" => {
            if args.len() == 2 {
                match (expr_to_4d(&args[0]), &args[1]) {
                    (Some(riemann), metric_expr) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            ax_tensor::kretschner_scalar(&riemann, &metric, interner)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "metric" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Call(diag_f, diag_args) if interner.resolve(*diag_f) == "diag" => {
                        symbolic_to_matrix(&ax_tensor::SymbolicMatrix::from_diagonal(
                            diag_args.clone(),
                        ))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "diag" => symbolic_to_matrix(&ax_tensor::SymbolicMatrix::from_diagonal(args)),
        "integrate" => {
            if args.len() == 2 {
                if let Expr::Sym(var_sym) = args[1] {
                    let integrated = integrate::integrate(&args[0], var_sym, interner);
                    eval(&integrated, &Env::new(), interner)
                } else {
                    Expr::Call(f, args)
                }
            } else if args.len() == 4 {
                if let Expr::Sym(var_sym) = args[1] {
                    let integrated = integrate::integrate(&args[0], var_sym, interner);
                    let integrate_sym = interner.get_or_intern("integrate");
                    if matches!(&integrated, Expr::Call(sym, _) if *sym == integrate_sym) {
                        Expr::Call(f, args)
                    } else {
                        let mut hi_env = Env::new();
                        hi_env.bindings.insert(var_sym, args[3].clone());
                        let hi_val = eval(&integrated, &hi_env, interner);

                        let mut lo_env = Env::new();
                        lo_env.bindings.insert(var_sym, args[2].clone());
                        let lo_val = eval(&integrated, &lo_env, interner);

                        eval(
                            &Expr::add(vec![hi_val, Expr::neg(lo_val)]),
                            &Env::new(),
                            interner,
                        )
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "series" => {
            if args.len() == 4 {
                if let (Expr::Sym(var_sym), Expr::Int(order)) = (&args[1], &args[3]) {
                    if let Some(order) = order.to_usize() {
                        series::taylor_series(&args[0], *var_sym, &args[2], order, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        _ => Expr::Call(f, args),
    }
}

pub fn eval(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Sym(s) => {
            if let Some(val) = env.lookup(*s) {
                if matches!(val, Expr::Sym(bound) if *bound == *s) {
                    Expr::Sym(*s)
                } else {
                    eval(val, env, interner)
                }
            } else {
                Expr::Sym(*s)
            }
        }
        Expr::Add(terms) => {
            let evaluated = terms.iter().map(|term| eval(term, env, interner)).collect();
            Expr::add(evaluated)
        }
        Expr::Mul(factors) => {
            let evaluated = factors
                .iter()
                .map(|factor| eval(factor, env, interner))
                .collect();
            Expr::mul(evaluated)
        }
        Expr::Pow(base, exp) => {
            let evaled_base = eval(base, env, interner);
            let evaled_exp = eval(exp, env, interner);
            if let Some(out) = numeric_pow(&evaled_base, &evaled_exp) {
                out
            } else if matches!(&evaled_base, Expr::Int(n) if *n == (-1).into()) {
                match &evaled_exp {
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Even) => Expr::one(),
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Odd) => {
                        Expr::Int((-1).into())
                    }
                    _ => Expr::pow(evaled_base, evaled_exp),
                }
            } else {
                Expr::pow(evaled_base, evaled_exp)
            }
        }
        Expr::Neg(e) => Expr::neg(eval(e, env, interner)),
        Expr::Call(f, args) => {
            let evaled_args: Vec<Expr> = args.iter().map(|arg| eval(arg, env, interner)).collect();
            let name = interner.resolve(*f);
            let result = builtin_call(name, *f, evaled_args.clone(), interner, env);
            if let Expr::Call(returned_f, _) = &result {
                if *returned_f == *f {
                    if let Some(Expr::FnDef(_, params, body)) = env.lookup(*f) {
                        if params.len() == evaled_args.len() {
                            let mut child_env = env.clone();
                            for (param, arg) in params.iter().zip(evaled_args.iter()) {
                                child_env.bindings.insert(*param, arg.clone());
                            }
                            return eval(body, &child_env, interner);
                        }
                    }
                }
            }
            result
        }
        Expr::FnDef(name, params, body) => Expr::FnDef(*name, params.clone(), body.clone()),
        Expr::Rule(lhs, rhs) => Expr::Rule(lhs.clone(), rhs.clone()),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::Piecewise(cases) => {
            for (value, condition) in cases {
                match eval_condition(condition, env, interner) {
                    Some(true) => return eval(value, env, interner),
                    Some(false) => continue,
                    None => {
                        return Expr::Piecewise(
                            cases.iter()
                                .map(|(value, condition)| (eval(value, env, interner), condition.clone()))
                                .collect(),
                        );
                    }
                }
            }
            Expr::zero()
        }
        Expr::Let(name, val, body) => {
            let evaled_val = eval(val, env, interner);
            let child = env.extend(*name, evaled_val);
            eval(body, &child, interner)
        }
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(eval(base, env, interner)), indices.clone())
        }
        Expr::List(items) => {
            let evaled: Vec<Expr> = items.iter().map(|item| eval(item, env, interner)).collect();
            if let Some(ncols) = evaled.first().and_then(|e| {
                if let Expr::List(inner) = e {
                    Some(inner.len())
                } else {
                    None
                }
            }) {
                if evaled
                    .iter()
                    .all(|e| matches!(e, Expr::List(inner) if inner.len() == ncols))
                {
                    let rows = evaled
                        .into_iter()
                        .map(|e| {
                            if let Expr::List(inner) = e {
                                inner
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    return Expr::Matrix(rows);
                }
            }
            Expr::List(evaled)
        }
        Expr::Matrix(rows) => {
            let evaled_rows = rows
                .iter()
                .map(|row| row.iter().map(|cell| eval(cell, env, interner)).collect())
                .collect();
            Expr::Matrix(evaled_rows)
        }
    }
}

fn to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(n) => n.to_f64(),
        Expr::Rational(r) => Some(r.numer().to_f64()? / r.denom().to_f64()?),
        Expr::Float(f) => Some(*f),
        _ => None,
    }
}

fn expr_to_3d(expr: &Expr) -> Option<Vec<Vec<Vec<Expr>>>> {
    let Expr::List(level1) = expr else {
        return None;
    };
    level1
        .iter()
        .map(|item| {
            let Expr::List(level2) = item else {
                return None;
            };
            level2
                .iter()
                .map(|row| {
                    let Expr::List(level3) = row else {
                        return None;
                    };
                    Some(level3.clone())
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

fn expr_3d_to_list(data: Vec<Vec<Vec<Expr>>>) -> Expr {
    Expr::List(
        data.into_iter()
            .map(|level2| Expr::List(level2.into_iter().map(Expr::List).collect()))
            .collect(),
    )
}

fn expr_to_4d(expr: &Expr) -> Option<Vec<Vec<Vec<Vec<Expr>>>>> {
    let Expr::List(level1) = expr else {
        return None;
    };
    level1
        .iter()
        .map(|item| {
            let Expr::List(level2) = item else {
                return None;
            };
            level2
                .iter()
                .map(|item2| {
                    let Expr::List(level3) = item2 else {
                        return None;
                    };
                    level3
                        .iter()
                        .map(|item3| {
                            let Expr::List(level4) = item3 else {
                                return None;
                            };
                            Some(level4.clone())
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

fn expr_4d_to_list(data: Vec<Vec<Vec<Vec<Expr>>>>) -> Expr {
    Expr::List(
        data.into_iter()
            .map(|level2| {
                Expr::List(
                    level2
                        .into_iter()
                        .map(|level3| Expr::List(level3.into_iter().map(Expr::List).collect()))
                        .collect(),
                )
            })
            .collect(),
    )
}

fn matrix_to_symbolic(expr: &Expr) -> Option<ax_tensor::SymbolicMatrix> {
    let Expr::Matrix(rows) = expr else {
        return None;
    };
    let dim = rows.len();
    if rows.iter().any(|row| row.len() != dim) {
        return None;
    }
    Some(ax_tensor::SymbolicMatrix {
        dim,
        data: rows.clone(),
    })
}

fn symbolic_to_matrix(m: &ax_tensor::SymbolicMatrix) -> Expr {
    Expr::Matrix(m.data.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(
            result.errors.is_empty(),
            "lower errors: {:?}",
            result.errors
        );
        let expr = result.expr.expect("expected expression");
        let env = Env::new();
        (eval(&expr, &env, &interner), interner)
    }

    #[test]
    fn eval_arithmetic() {
        let (e, _) = eval_src("2 + 3 * 4;");
        assert_eq!(e, ax_ir::Expr::Int(14.into()));
    }

    #[test]
    fn eval_symbolic_stays() {
        let (e, int) = eval_src("x + 1;");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "expected x in output: {}", pp);
    }

    #[test]
    fn eval_let_binding() {
        let (e, _) = eval_src("let x = 5 in x + 3;");
        assert_eq!(e, ax_ir::Expr::Int(8.into()));
    }

    #[test]
    fn eval_nested_let() {
        let (e, _) = eval_src("let x = 2 in let y = 3 in x + y;");
        assert_eq!(e, ax_ir::Expr::Int(5.into()));
    }

    #[test]
    fn eval_sqrt_perfect_square() {
        let (e, _) = eval_src("sqrt(9);");
        assert_eq!(e, ax_ir::Expr::Int(3.into()));
    }

    #[test]
    fn eval_zero_times_anything() {
        let (e, _) = eval_src("0 * x;");
        assert_eq!(e, ax_ir::Expr::Int(0.into()));
    }

    #[test]
    fn eval_diag() {
        let (e, _) = eval_src("diag(1, 2, 3);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0][0], Expr::Int(1.into()));
                assert_eq!(rows[1][1], Expr::Int(2.into()));
                assert_eq!(rows[2][2], Expr::Int(3.into()));
                assert_eq!(rows[0][1], Expr::zero());
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn det_2x2() {
        let (e, _) = eval_src("det([[1, 2], [3, 4]]);");
        assert_eq!(e, Expr::Int((-2).into()));
    }

    #[test]
    fn det_3x3() {
        let (e, _) = eval_src("det([[1,0,0],[0,1,0],[0,0,1]]);");
        assert_eq!(e, Expr::Int(1.into()));
    }

    #[test]
    fn transpose_test() {
        let (e, _) = eval_src("transpose([[1,2],[3,4]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows[0][1], Expr::Int(3.into()));
                assert_eq!(rows[1][0], Expr::Int(2.into()));
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn trace_test() {
        let (e, _) = eval_src("trace_mat([[1,2],[3,4]]);");
        assert_eq!(e, Expr::Int(5.into()));
    }

    #[test]
    fn inverse_2x2() {
        let (e, _) = eval_src("inv([[1,0],[0,2]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(
                    rows[1][1],
                    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
                );
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn diff_polynomial() {
        let (e, int) = eval_src("diff(x^3, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("3") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn diff_sum() {
        let (e, _) = eval_src("diff(x + 1, x);");
        assert_eq!(e, ax_ir::Expr::Int(1.into()));
    }

    #[test]
    fn diff_constant() {
        let (e, _) = eval_src("diff(5, x);");
        assert_eq!(e, ax_ir::Expr::Int(0.into()));
    }

    #[test]
    fn diff_product() {
        let (e, int) = eval_src("diff(x * x, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("2") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn diff_sin() {
        let (e, int) = eval_src("diff(sin(x), x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("cos"), "got: {}", pp);
    }

    #[test]
    fn eval_user_function() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("f(x) = x^2", &interner);
        let f_call = ax_core_ir::lower("f(3)", &interner);

        let mut env = Env::new();
        let def_expr = f_def.expr.unwrap();
        let def_result = eval(&def_expr, &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let call_expr = f_call.expr.unwrap();
        let result = eval(&call_expr, &env, &interner);
        assert_eq!(result, Expr::Int(9.into()));
    }

    #[test]
    fn eval_user_function_two_args() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("g(x, y) = x + y", &interner);
        let f_call = ax_core_ir::lower("g(3, 4)", &interner);

        let mut env = Env::new();
        let def_result = eval(&f_def.expr.unwrap(), &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let result = eval(&f_call.expr.unwrap(), &env, &interner);
        assert_eq!(result, Expr::Int(7.into()));
    }

    #[test]
    fn diff_user_function() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("f(x) = x^3", &interner);
        let mut env = Env::new();
        let def_result = eval(&f_def.expr.unwrap(), &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let diff_call = ax_core_ir::lower("diff(f(x), x)", &interner);
        let result = eval(&diff_call.expr.unwrap(), &env, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("3") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn apply_user_rule() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();

        let rule_src = "rule pythag: sin(x_)^2 + cos(x_)^2 => 1";
        let rule_expr = eval(
            &ax_core_ir::lower(rule_src, &interner).expr.unwrap(),
            &env,
            &interner,
        );
        register_rule(&rule_expr, &mut env, &interner);

        let test_src = "rewrite(sin(a)^2 + cos(a)^2)";
        let test_expr = ax_core_ir::lower(test_src, &interner).expr.unwrap();
        let result = eval(&test_expr, &env, &interner);
        assert_eq!(result, Expr::Int(1.into()));
    }

    #[test]
    fn assume_positive_sqrt() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let mut env = Env::new();
        env.assumptions.insert(x, vec![ax_ir::Assumption::Positive]);

        let expr = ax_core_ir::lower("sqrt(x^2)", &interner).expr.unwrap();
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::Sym(x));
    }

    #[test]
    fn eval_if_then_else_numeric() {
        let (e, _) = eval_src("if 3 > 2 then 10 else 20");
        assert_eq!(e, Expr::Int(10.into()));
    }

    #[test]
    fn eval_if_symbolic() {
        let (e, _) = eval_src("if x > 0 then x else -x");
        assert!(matches!(e, Expr::Piecewise(_)));
    }
}
