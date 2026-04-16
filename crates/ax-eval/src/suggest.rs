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
    pub note: Option<String>,
}

fn contains_indexed(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(contains_indexed)
        }
        Expr::Matrix(rows) => rows.iter().flatten().any(contains_indexed),
        Expr::Pow(base, exp) => contains_indexed(base) || contains_indexed(exp),
        Expr::Neg(inner) | Expr::Group(inner, _) => contains_indexed(inner),
        Expr::Complex(re, im) => contains_indexed(re) || contains_indexed(im),
        Expr::Call(_, args) => args.iter().any(contains_indexed),
        Expr::FnDef(_, _, body) => contains_indexed(body),
        Expr::Rule(lhs, rhs, _) => contains_indexed(lhs) || contains_indexed(rhs),
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_indexed(value)),
        Expr::Let(_, value, body) => contains_indexed(value) || contains_indexed(body),
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Sym(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => false,
    }
}

fn contains_named_call(expr: &Expr, names: &[&str], interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Call(f, args) => {
            if names.contains(&interner.resolve(*f)) {
                true
            } else {
                args.iter()
                    .any(|arg| contains_named_call(arg, names, interner))
            }
        }
        Expr::Indexed(base, _) => contains_named_call(base, names, interner),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => terms
            .iter()
            .any(|term| contains_named_call(term, names, interner)),
        Expr::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|cell| contains_named_call(cell, names, interner)),
        Expr::Pow(base, exp) => {
            contains_named_call(base, names, interner) || contains_named_call(exp, names, interner)
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => contains_named_call(inner, names, interner),
        Expr::Complex(re, im) => {
            contains_named_call(re, names, interner) || contains_named_call(im, names, interner)
        }
        Expr::FnDef(_, _, body) => contains_named_call(body, names, interner),
        Expr::Rule(lhs, rhs, _) => {
            contains_named_call(lhs, names, interner) || contains_named_call(rhs, names, interner)
        }
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, _)| contains_named_call(value, names, interner)),
        Expr::Let(_, value, body) => {
            contains_named_call(value, names, interner)
                || contains_named_call(body, names, interner)
        }
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Sym(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => false,
    }
}

fn contains_derivative_call(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    contains_named_call(
        expr,
        &["partial", "nabla", "D", "d", "diff", "partial_derivative"],
        interner,
    )
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
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_indices(inner, out),
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
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Sym(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => {}
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
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_indexed_base_symbols(inner, out),
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
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Sym(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => {}
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

fn goal_priorities(goal: &str) -> Vec<&'static str> {
    let g = goal.to_lowercase();
    if g.contains("simplif") {
        vec![
            "canonicalise",
            "rename_dummies",
            "collect_terms",
            "meld",
            "simplify",
            "factor_out",
        ]
    } else if g.contains("canonical") || g.contains("canon") {
        vec!["canonicalise", "rename_dummies", "collect_terms", "meld"]
    } else if g.contains("evaluat") || g.contains("component") {
        vec!["evaluate_components", "simplify"]
    } else if g.contains("zero")
        || g.contains("vanish")
        || g.contains("prove")
        || g.contains("identity")
    {
        vec![
            "canonicalise",
            "rename_dummies",
            "meld",
            "collect_terms",
            "simplify",
        ]
    } else if g.contains("contract") || g.contains("eliminat") {
        vec![
            "eliminate_metric",
            "eliminate_kronecker",
            "rename_dummies",
            "collect_terms",
        ]
    } else if g.contains("expand") || g.contains("distribut") {
        vec!["distribute", "expand", "collect_terms", "simplify"]
    } else if g.contains("gamma") || g.contains("spinor") || g.contains("dirac") {
        vec![
            "join_gamma",
            "sort_product",
            "gamma_trace",
            "fierz",
            "sort_spinors",
            "collect_terms",
            "simplify",
        ]
    } else if g.contains("factor") {
        vec!["factor_out", "factor_in", "collect_factors", "simplify"]
    } else if g.contains("integrat") {
        vec!["integrate_by_parts", "integrate", "simplify"]
    } else if g.contains("decompos") || g.contains("irrep") {
        vec![
            "decompose_product",
            "decompose",
            "young_project",
            "collect_terms",
        ]
    } else if g.contains("curv")
        || g.contains("riemann")
        || g.contains("ricci")
        || g.contains("einstein")
    {
        vec![
            "define_metric",
            "christoffel",
            "riemann",
            "ricci",
            "einstein",
            "scalar_curvature",
            "kretschner",
        ]
    } else {
        vec![]
    }
}

fn prioritise_suggestions(
    mut suggestions: Vec<Suggestion>,
    goal: &str,
) -> (Vec<Suggestion>, Option<String>) {
    let priorities = goal_priorities(goal);
    if priorities.is_empty() {
        return (
            suggestions,
            Some(format!(
                "No goal-specific priority profile matched '{}'; returning the general suggestions.",
                goal
            )),
        );
    }

    let priority_positions = priorities
        .iter()
        .enumerate()
        .map(|(idx, alg)| (*alg, idx))
        .collect::<HashMap<_, _>>();

    let mut priority = Vec::new();
    let mut other = Vec::new();
    let mut seen = suggestions
        .iter()
        .map(|s| s.algorithm.clone())
        .collect::<HashSet<_>>();

    for suggestion in suggestions.drain(..) {
        if priority_positions.contains_key(suggestion.algorithm.as_str()) {
            priority.push(suggestion);
        } else {
            other.push(suggestion);
        }
    }

    priority.sort_by_key(|suggestion| {
        priority_positions
            .get(suggestion.algorithm.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });

    for algorithm in priorities {
        if seen.insert(algorithm.to_string()) {
            priority.push(Suggestion {
                algorithm: algorithm.to_string(),
                reason: format!(
                    "recommended for goal: {} (may require additional property declarations)",
                    goal
                ),
            });
        }
    }

    priority.extend(other);
    (priority, None)
}

pub fn suggest_for_expr(
    expr: &Expr,
    env: &Env,
    interner: &ax_ir::Interner,
    goal: Option<&str>,
) -> SuggestResult {
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
            let props = env.property_store.get_all(*sym);
            if props.is_empty() {
                let name = interner.resolve(*sym).to_string();
                missing.push(MissingProperty {
                    symbol: name.clone(),
                    suggestion: format!(
                        "declare symmetry properties for {name} to enable canonicalise"
                    ),
                });
            } else {
                for prop in props {
                    match prop {
                        ax_ir::TensorProperty::RiemannSymmetry => {
                            has_symmetry = true;
                            symmetry_reason =
                                "expression has tensors with RiemannSymmetry".to_string();
                        }
                        ax_ir::TensorProperty::Symmetric(_) => {
                            has_symmetry = true;
                            if symmetry_reason == "expression has tensors with symmetry properties"
                            {
                                symmetry_reason =
                                    "expression has tensors with Symmetric properties".to_string();
                            }
                        }
                        ax_ir::TensorProperty::AntiSymmetric(_) => {
                            has_symmetry = true;
                            if symmetry_reason == "expression has tensors with symmetry properties"
                            {
                                symmetry_reason =
                                    "expression has tensors with AntiSymmetric properties"
                                        .to_string();
                            }
                        }
                        ax_ir::TensorProperty::Metric => has_metric = true,
                        ax_ir::TensorProperty::KroneckerDelta => has_delta = true,
                        ax_ir::TensorProperty::EpsilonTensor => has_epsilon = true,
                        _ => {}
                    }
                }
            }
        }

        if has_symmetry {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "canonicalise",
                &symmetry_reason,
            );
        }
        if matches!(expr, Expr::Add(terms) if terms.len() > 1 && terms.iter().all(contains_indexed))
        {
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
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "eliminate_metric",
                "expression contains metric tensors",
            );
        }
        if has_delta {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "eliminate_kronecker",
                "expression contains Kronecker deltas",
            );
        }
        if has_epsilon {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "epsilon_to_delta",
                "expression contains epsilon tensors",
            );
        }
        if matches!(expr, Expr::Mul(factors) if factors.iter().any(contains_indexed)) {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "sort_product",
                "expression is a product of indexed terms",
            );
        }
        if has_dummy_indices {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "rename_dummies",
                "expression contains dummy index pairs",
            );
        }
    }

    if let Expr::Add(terms) = expr {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "simplify",
            "general simplification for sums",
        );
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "collect_terms",
            "expression is a sum",
        );
        if terms.len() > 4 {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "factor_out",
                "expression has many additive terms",
            );
        }
    }

    if let Expr::Mul(factors) = expr {
        if factors
            .iter()
            .any(|factor| contains_derivative_call(factor, interner))
        {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "product_rule",
                "product contains a derivative call",
            );
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "unwrap",
                "product contains a derivative call",
            );
        }
    }

    if contains_named_call(
        expr,
        &[
            "sin", "cos", "tan", "sec", "csc", "cot", "asin", "arcsin", "acos", "arccos", "atan",
            "arctan", "sinh", "cosh", "tanh", "asinh", "arcsinh", "acosh", "arccosh", "atanh",
            "arctanh",
        ],
        interner,
    ) {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "trig_simplify",
            "expression contains trigonometric functions",
        );
    }

    if matches!(expr, Expr::Call(f, _) if interner.resolve(*f) == "christoffel")
        || contains_named_call(expr, &["christoffel"], interner)
    {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "riemann",
            "expression contains Christoffel symbols",
        );
    }

    if contains_named_call(expr, &["gamma"], interner) {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "join_gamma",
            "expression contains gamma matrix calls",
        );
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "gamma_trace",
            "expression contains gamma matrix calls",
        );
    }

    add_suggestion(
        &mut suggestions,
        &mut seen,
        "simplify",
        "general simplification",
    );

    let (suggestions, note) = if let Some(goal) = goal {
        prioritise_suggestions(suggestions, goal)
    } else {
        (suggestions, None)
    };

    SuggestResult {
        suggestions,
        missing,
        note,
    }
}
