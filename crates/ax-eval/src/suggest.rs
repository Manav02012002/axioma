use crate::Env;
use ax_ir::Expr;
use std::collections::{HashMap, HashSet};

pub struct Suggestion {
    pub algorithm: String,
    pub reason: String,
}

pub struct MissingProperty {
    pub symbol: String,
    pub suggestion: String,
}

pub struct SuggestResult {
    pub suggestions: Vec<Suggestion>,
    pub missing: Vec<MissingProperty>,
}

fn contains_indexed(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => terms.iter().any(contains_indexed),
        Expr::Matrix(rows) => rows.iter().flatten().any(contains_indexed),
        Expr::Pow(base, exp) => contains_indexed(base) || contains_indexed(exp),
        Expr::Neg(inner) => contains_indexed(inner),
        Expr::Complex(re, im) => contains_indexed(re) || contains_indexed(im),
        Expr::Call(_, args) => args.iter().any(contains_indexed),
        Expr::FnDef(_, _, body) => contains_indexed(body),
        Expr::Rule(lhs, rhs, _) => contains_indexed(lhs) || contains_indexed(rhs),
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_indexed(value)),
        Expr::Let(_, value, body) => contains_indexed(value) || contains_indexed(body),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Sym(_)
        | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
    }
}

fn contains_named_call(expr: &Expr, names: &[&str], interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Call(f, args) => {
            if names.contains(&interner.resolve(*f)) {
                true
            } else {
                args.iter().any(|arg| contains_named_call(arg, names, interner))
            }
        }
        Expr::Indexed(base, _) => contains_named_call(base, names, interner),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_named_call(term, names, interner))
        }
        Expr::Matrix(rows) => rows.iter().flatten().any(|cell| contains_named_call(cell, names, interner)),
        Expr::Pow(base, exp) => {
            contains_named_call(base, names, interner) || contains_named_call(exp, names, interner)
        }
        Expr::Neg(inner) => contains_named_call(inner, names, interner),
        Expr::Complex(re, im) => {
            contains_named_call(re, names, interner) || contains_named_call(im, names, interner)
        }
        Expr::FnDef(_, _, body) => contains_named_call(body, names, interner),
        Expr::Rule(lhs, rhs, _) => contains_named_call(lhs, names, interner) || contains_named_call(rhs, names, interner),
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_named_call(value, names, interner)),
        Expr::Let(_, value, body) => contains_named_call(value, names, interner) || contains_named_call(body, names, interner),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Sym(_)
        | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
    }
}

fn contains_derivative_call(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    contains_named_call(expr, &["partial", "nabla", "D", "d", "diff", "partial_derivative"], interner)
}

fn collect_indices(expr: &Expr, out: &mut Vec<ax_ir::Index>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_indices(base, out);
            out.extend(indices.iter().cloned());
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_indices(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_indices(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_indices(base, out);
            collect_indices(exp, out);
        }
        Expr::Neg(inner) => collect_indices(inner, out),
        Expr::Complex(re, im) => {
            collect_indices(re, out);
            collect_indices(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_indices(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_indices(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_indices(lhs, out);
            collect_indices(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_indices(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_indices(value, out);
            collect_indices(body, out);
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Sym(_)
        | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => {}
    }
}

fn collect_indexed_base_symbols(expr: &Expr, out: &mut Vec<ax_ir::expr::Sym>) {
    match expr {
        Expr::Indexed(base, _) => {
            if let Expr::Sym(sym) = base.as_ref() {
                out.push(*sym);
            } else {
                collect_indexed_base_symbols(base, out);
            }
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_indexed_base_symbols(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_indexed_base_symbols(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_indexed_base_symbols(base, out);
            collect_indexed_base_symbols(exp, out);
        }
        Expr::Neg(inner) => collect_indexed_base_symbols(inner, out),
        Expr::Complex(re, im) => {
            collect_indexed_base_symbols(re, out);
            collect_indexed_base_symbols(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_indexed_base_symbols(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_indexed_base_symbols(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_indexed_base_symbols(lhs, out);
            collect_indexed_base_symbols(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_indexed_base_symbols(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_indexed_base_symbols(value, out);
            collect_indexed_base_symbols(body, out);
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Sym(_)
        | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => {}
    }
}

fn add_suggestion(
    suggestions: &mut Vec<Suggestion>,
    seen: &mut HashSet<String>,
    algorithm: &str,
    reason: &str,
) {
    if seen.insert(algorithm.to_string()) {
        suggestions.push(Suggestion {
            algorithm: algorithm.to_string(),
            reason: reason.to_string(),
        });
    }
}

pub fn suggest_for_expr(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> SuggestResult {
    let mut suggestions = Vec::new();
    let mut seen = HashSet::new();

    let mut all_indices = Vec::new();
    collect_indices(expr, &mut all_indices);
    let mut by_name: HashMap<ax_ir::expr::Sym, Vec<ax_ir::Index>> = HashMap::new();
    for idx in all_indices {
        by_name.entry(idx.name).or_default().push(idx);
    }
    let has_dummy_indices = by_name
        .values()
        .any(|occs| occs.len() == 2 && occs[0].variance != occs[1].variance);

    let mut indexed_syms = Vec::new();
    collect_indexed_base_symbols(expr, &mut indexed_syms);
    indexed_syms.sort_by_key(|s| interner.resolve(*s).to_string());
    indexed_syms.dedup();

    let mut missing = Vec::new();

    if contains_indexed(expr) {
        let mut has_symmetry = false;
        let mut symmetry_reason = "expression has tensors with symmetry properties".to_string();
        let mut has_metric = false;
        let mut has_delta = false;
        let mut has_epsilon = false;

        for sym in &indexed_syms {
            match env.tensor_properties.get(sym) {
                Some(props) => {
                    for prop in props {
                        match prop {
                            ax_ir::TensorProperty::RiemannSymmetry => {
                                has_symmetry = true;
                                symmetry_reason = "expression has tensors with RiemannSymmetry".to_string();
                            }
                            ax_ir::TensorProperty::Symmetric(_) => {
                                has_symmetry = true;
                                if symmetry_reason == "expression has tensors with symmetry properties" {
                                    symmetry_reason = "expression has tensors with Symmetric properties".to_string();
                                }
                            }
                            ax_ir::TensorProperty::AntiSymmetric(_) => {
                                has_symmetry = true;
                                if symmetry_reason == "expression has tensors with symmetry properties" {
                                    symmetry_reason = "expression has tensors with AntiSymmetric properties".to_string();
                                }
                            }
                            ax_ir::TensorProperty::Metric => has_metric = true,
                            ax_ir::TensorProperty::KroneckerDelta => has_delta = true,
                            ax_ir::TensorProperty::EpsilonTensor => has_epsilon = true,
                            _ => {}
                        }
                    }
                }
                None => {
                    let name = interner.resolve(*sym).to_string();
                    missing.push(MissingProperty {
                        symbol: name.clone(),
                        suggestion: format!("declare symmetry properties for {name} to enable canonicalise"),
                    });
                }
            }
        }

        if has_symmetry {
            add_suggestion(&mut suggestions, &mut seen, "canonicalise", &symmetry_reason);
        }
        if matches!(expr, Expr::Add(terms) if terms.len() > 1 && terms.iter().all(contains_indexed)) {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "meld",
                "sum contains multiple indexed terms",
            );
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "collect_terms",
                "expression is an additive tensor combination",
            );
        }
        if has_metric {
            add_suggestion(&mut suggestions, &mut seen, "eliminate_metric", "expression contains metric tensors");
        }
        if has_delta {
            add_suggestion(&mut suggestions, &mut seen, "eliminate_kronecker", "expression contains Kronecker deltas");
        }
        if has_epsilon {
            add_suggestion(&mut suggestions, &mut seen, "epsilon_to_delta", "expression contains epsilon tensors");
        }
        if matches!(expr, Expr::Mul(factors) if factors.iter().any(contains_indexed)) {
            add_suggestion(&mut suggestions, &mut seen, "sort_product", "expression is a product of indexed terms");
        }
        if has_dummy_indices {
            add_suggestion(&mut suggestions, &mut seen, "rename_dummies", "expression contains dummy index pairs");
        }
    }

    if let Expr::Add(terms) = expr {
        add_suggestion(&mut suggestions, &mut seen, "simplify", "general simplification for sums");
        add_suggestion(&mut suggestions, &mut seen, "collect_terms", "expression is a sum");
        if terms.len() > 4 {
            add_suggestion(&mut suggestions, &mut seen, "factor_out", "expression has many additive terms");
        }
    }

    if let Expr::Mul(factors) = expr {
        if factors.iter().any(|factor| contains_derivative_call(factor, interner)) {
            add_suggestion(&mut suggestions, &mut seen, "product_rule", "product contains a derivative call");
            add_suggestion(&mut suggestions, &mut seen, "unwrap", "product contains a derivative call");
        }
    }

    if contains_named_call(
        expr,
        &[
            "sin", "cos", "tan", "sec", "csc", "cot", "asin", "arcsin", "acos", "arccos",
            "atan", "arctan", "sinh", "cosh", "tanh", "asinh", "arcsinh", "acosh",
            "arccosh", "atanh", "arctanh",
        ],
        interner,
    ) {
        add_suggestion(&mut suggestions, &mut seen, "trig_simplify", "expression contains trigonometric functions");
    }

    if matches!(expr, Expr::Call(f, _) if interner.resolve(*f) == "christoffel")
        || contains_named_call(expr, &["christoffel"], interner)
    {
        add_suggestion(&mut suggestions, &mut seen, "riemann", "expression contains Christoffel symbols");
    }

    if contains_named_call(expr, &["gamma"], interner) {
        add_suggestion(&mut suggestions, &mut seen, "join_gamma", "expression contains gamma matrix calls");
        add_suggestion(&mut suggestions, &mut seen, "gamma_trace", "expression contains gamma matrix calls");
    }

    add_suggestion(&mut suggestions, &mut seen, "simplify", "general simplification");

    SuggestResult { suggestions, missing }
}
