use crate::{registry::format_tensor_property, Env};
use ax_ir::{Expr, Index, Interner, TensorProperty, Variance};
use num_rational::BigRational;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

fn numeric_value(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Group(inner, _) => numeric_value(inner),
        Expr::Neg(inner) => numeric_value(inner).map(|n| -n),
        _ => None,
    }
}

fn split_numeric_coeff(expr: &Expr) -> (BigRational, Expr) {
    if let Some(value) = numeric_value(expr) {
        return (value, Expr::Int(1.into()));
    }
    match expr {
        Expr::Mul(factors) => {
            let mut coeff = BigRational::from_integer(1.into());
            let mut rest = Vec::new();
            for factor in factors {
                if let Some(value) = numeric_value(factor) {
                    coeff *= value;
                } else {
                    rest.push(factor.clone());
                }
            }
            let structure = match rest.len() {
                0 => Expr::Int(1.into()),
                1 => rest[0].clone(),
                _ => Expr::Mul(rest),
            };
            (coeff, structure)
        }
        Expr::Neg(inner) => {
            let (coeff, rest) = split_numeric_coeff(inner);
            (-coeff, rest)
        }
        Expr::Group(inner, rel) => {
            let (coeff, rest) = split_numeric_coeff(inner);
            (coeff, Expr::Group(Box::new(rest), *rel))
        }
        _ => (BigRational::from_integer(1.into()), expr.clone()),
    }
}

fn render_expr(expr: &Expr, interner: &Interner) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Float(f) => f.to_string(),
        Expr::Sym(sym) => interner.resolve(*sym).to_string(),
        Expr::Indexed(base, indices) => format!(
            "{}[{}]",
            render_expr(base, interner),
            indices
                .iter()
                .map(|idx| format!(
                    "{}{}",
                    interner.resolve(idx.name),
                    match idx.variance {
                        Variance::Up => "+",
                        Variance::Down => "-",
                    }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Add(terms) => terms
            .iter()
            .map(|term| render_expr(term, interner))
            .collect::<Vec<_>>()
            .join(" + "),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| render_expr(factor, interner))
            .collect::<Vec<_>>()
            .join(" * "),
        Expr::Pow(base, exp) => format!(
            "({})^({})",
            render_expr(base, interner),
            render_expr(exp, interner)
        ),
        Expr::Neg(inner) => format!("-{}", render_expr(inner, interner)),
        Expr::Call(name, args) => format!(
            "{}({})",
            interner.resolve(*name),
            args.iter()
                .map(|arg| render_expr(arg, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Group(inner, _) => format!("({})", render_expr(inner, interner)),
        Expr::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| render_expr(item, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Matrix(rows) => format!(
            "[{}]",
            rows.iter()
                .map(|row| format!(
                    "[{}]",
                    row.iter()
                        .map(|cell| render_expr(cell, interner))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Complex(re, im) => format!(
            "{} + {} i",
            render_expr(re, interner),
            render_expr(im, interner)
        ),
        Expr::FnDef(name, _, body) => format!(
            "fn {} => {}",
            interner.resolve(*name),
            render_expr(body, interner)
        ),
        Expr::Rule(lhs, rhs, _) => format!(
            "{} -> {}",
            render_expr(lhs, interner),
            render_expr(rhs, interner)
        ),
        Expr::Import(path) => format!(
            "import {}",
            path.iter()
                .map(|sym| interner.resolve(*sym).to_string())
                .collect::<Vec<_>>()
                .join(".")
        ),
        Expr::Assume(sym, _) => format!("assume {}", interner.resolve(*sym)),
        Expr::SetConvention(field, value) => format!("set_convention({}, {})", field, value),
        Expr::Piecewise(cases) => format!("piecewise({} cases)", cases.len()),
        Expr::Let(sym, value, body) => format!(
            "let {} = {} in {}",
            interner.resolve(*sym),
            render_expr(value, interner),
            render_expr(body, interner)
        ),
    }
}

fn base_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Indexed(base, _) => base_symbol(base),
        Expr::Group(inner, _) => base_symbol(inner),
        _ => None,
    }
}

fn collect_indices(expr: &Expr, out: &mut Vec<Index>) {
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

fn collect_indexed_symbols(expr: &Expr, out: &mut Vec<lasso::Spur>) {
    match expr {
        Expr::Indexed(base, _) => {
            if let Some(sym) = base_symbol(base) {
                out.push(sym);
            }
            collect_indexed_symbols(base, out);
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_indexed_symbols(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_indexed_symbols(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_indexed_symbols(base, out);
            collect_indexed_symbols(exp, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_indexed_symbols(inner, out),
        Expr::Complex(re, im) => {
            collect_indexed_symbols(re, out);
            collect_indexed_symbols(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_indexed_symbols(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_indexed_symbols(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_indexed_symbols(lhs, out);
            collect_indexed_symbols(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_indexed_symbols(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_indexed_symbols(value, out);
            collect_indexed_symbols(body, out);
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

fn same_structure_ignoring_coefficients(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Add(a_terms), Expr::Add(b_terms)) => {
            if a_terms.len() != b_terms.len() {
                return false;
            }
            let mut left = a_terms
                .iter()
                .map(|term| {
                    let (_, rest) = split_numeric_coeff(term);
                    format!("{rest:?}")
                })
                .collect::<Vec<_>>();
            let mut right = b_terms
                .iter()
                .map(|term| {
                    let (_, rest) = split_numeric_coeff(term);
                    format!("{rest:?}")
                })
                .collect::<Vec<_>>();
            left.sort();
            right.sort();
            left == right
        }
        _ => {
            let (_, a_rest) = split_numeric_coeff(a);
            let (_, b_rest) = split_numeric_coeff(b);
            a_rest == b_rest
        }
    }
}

fn additive_coefficient_difference(a: &Expr, b: &Expr) -> bool {
    let (Expr::Add(a_terms), Expr::Add(b_terms)) = (a, b) else {
        return false;
    };
    if a_terms.len() != b_terms.len() {
        return false;
    }
    let mut left = a_terms
        .iter()
        .map(|term| {
            let (coeff, rest) = split_numeric_coeff(term);
            (format!("{rest:?}"), coeff)
        })
        .collect::<Vec<_>>();
    let mut right = b_terms
        .iter()
        .map(|term| {
            let (coeff, rest) = split_numeric_coeff(term);
            (format!("{rest:?}"), coeff)
        })
        .collect::<Vec<_>>();
    left.sort_by(|a, b| a.0.cmp(&b.0));
    right.sort_by(|a, b| a.0.cmp(&b.0));
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(lhs, rhs)| lhs.0 == rhs.0)
        && left
            .iter()
            .zip(right.iter())
            .any(|(lhs, rhs)| lhs.1 != rhs.1)
}

fn same_structure_ignoring_index_names(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Indexed(a_base, a_idx), Expr::Indexed(b_base, b_idx)) => {
            a_idx.len() == b_idx.len()
                && same_structure_ignoring_index_names(a_base, b_base)
                && a_idx
                    .iter()
                    .zip(b_idx.iter())
                    .all(|(lhs, rhs)| lhs.variance == rhs.variance)
        }
        (Expr::Sym(a_sym), Expr::Sym(b_sym)) => a_sym == b_sym,
        (Expr::Add(a_terms), Expr::Add(b_terms))
        | (Expr::Mul(a_terms), Expr::Mul(b_terms))
        | (Expr::List(a_terms), Expr::List(b_terms)) => {
            a_terms.len() == b_terms.len()
                && a_terms
                    .iter()
                    .zip(b_terms.iter())
                    .all(|(lhs, rhs)| same_structure_ignoring_index_names(lhs, rhs))
        }
        (Expr::Pow(a_base, a_exp), Expr::Pow(b_base, b_exp)) => {
            same_structure_ignoring_index_names(a_base, b_base)
                && same_structure_ignoring_index_names(a_exp, b_exp)
        }
        (Expr::Neg(a_inner), Expr::Neg(b_inner))
        | (Expr::Group(a_inner, _), Expr::Group(b_inner, _)) => {
            same_structure_ignoring_index_names(a_inner, b_inner)
        }
        (Expr::Call(a_name, a_args), Expr::Call(b_name, b_args)) => {
            a_name == b_name
                && a_args.len() == b_args.len()
                && a_args
                    .iter()
                    .zip(b_args.iter())
                    .all(|(lhs, rhs)| same_structure_ignoring_index_names(lhs, rhs))
        }
        (Expr::Int(a_n), Expr::Int(b_n)) => a_n == b_n,
        (Expr::Rational(a_r), Expr::Rational(b_r)) => a_r == b_r,
        (Expr::Float(a_f), Expr::Float(b_f)) => a_f == b_f,
        (Expr::Complex(a_re, a_im), Expr::Complex(b_re, b_im)) => {
            same_structure_ignoring_index_names(a_re, b_re)
                && same_structure_ignoring_index_names(a_im, b_im)
        }
        (Expr::Matrix(a_rows), Expr::Matrix(b_rows)) => {
            a_rows.len() == b_rows.len()
                && a_rows.iter().zip(b_rows.iter()).all(|(a_row, b_row)| {
                    a_row.len() == b_row.len()
                        && a_row
                            .iter()
                            .zip(b_row.iter())
                            .all(|(lhs, rhs)| same_structure_ignoring_index_names(lhs, rhs))
                })
        }
        _ => a == b,
    }
}

fn same_structure_ignoring_index_positions(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Indexed(a_base, a_idx), Expr::Indexed(b_base, b_idx)) => {
            a_idx.len() == b_idx.len()
                && same_structure_ignoring_index_positions(a_base, b_base)
                && a_idx
                    .iter()
                    .zip(b_idx.iter())
                    .all(|(lhs, rhs)| lhs.name == rhs.name)
        }
        (Expr::Add(a_terms), Expr::Add(b_terms))
        | (Expr::Mul(a_terms), Expr::Mul(b_terms))
        | (Expr::List(a_terms), Expr::List(b_terms)) => {
            a_terms.len() == b_terms.len()
                && a_terms
                    .iter()
                    .zip(b_terms.iter())
                    .all(|(lhs, rhs)| same_structure_ignoring_index_positions(lhs, rhs))
        }
        (Expr::Pow(a_base, a_exp), Expr::Pow(b_base, b_exp)) => {
            same_structure_ignoring_index_positions(a_base, b_base)
                && same_structure_ignoring_index_positions(a_exp, b_exp)
        }
        (Expr::Neg(a_inner), Expr::Neg(b_inner))
        | (Expr::Group(a_inner, _), Expr::Group(b_inner, _)) => {
            same_structure_ignoring_index_positions(a_inner, b_inner)
        }
        (Expr::Call(a_name, a_args), Expr::Call(b_name, b_args)) => {
            a_name == b_name
                && a_args.len() == b_args.len()
                && a_args
                    .iter()
                    .zip(b_args.iter())
                    .all(|(lhs, rhs)| same_structure_ignoring_index_positions(lhs, rhs))
        }
        _ => a == b,
    }
}

fn expr_signature(expr: &Expr, interner: &Interner) -> String {
    match expr {
        Expr::Sym(sym) => interner.resolve(*sym).to_string(),
        Expr::Indexed(base, indices) => format!(
            "{}[{}]",
            expr_signature(base, interner),
            indices
                .iter()
                .map(|idx| format!(
                    "{}{}",
                    interner.resolve(idx.name),
                    match idx.variance {
                        Variance::Up => "+",
                        Variance::Down => "-",
                    }
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Expr::Add(terms) => {
            let mut parts = terms
                .iter()
                .map(|term| expr_signature(term, interner))
                .collect::<Vec<_>>();
            parts.sort();
            format!("Add({})", parts.join("|"))
        }
        Expr::Mul(factors) => {
            let mut parts = factors
                .iter()
                .map(|factor| expr_signature(factor, interner))
                .collect::<Vec<_>>();
            parts.sort();
            format!("Mul({})", parts.join("|"))
        }
        _ => render_expr(expr, interner),
    }
}

fn has_same_term_or_factor_multiset(a: &Expr, b: &Expr, interner: &Interner) -> bool {
    match (a, b) {
        (Expr::Add(a_terms), Expr::Add(b_terms)) | (Expr::Mul(a_terms), Expr::Mul(b_terms)) => {
            if a_terms.len() != b_terms.len() {
                return false;
            }
            let mut left = a_terms
                .iter()
                .map(|term| expr_signature(term, interner))
                .collect::<Vec<_>>();
            let mut right = b_terms
                .iter()
                .map(|term| expr_signature(term, interner))
                .collect::<Vec<_>>();
            left.sort();
            right.sort();
            left == right
        }
        _ => false,
    }
}

fn symmetry_properties(props: &[&TensorProperty]) -> bool {
    props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::Symmetric(_)
                | TensorProperty::AntiSymmetric(_)
                | TensorProperty::TableauSymmetry(_)
                | TensorProperty::RiemannSymmetry
        )
    })
}

fn prop_names(props: &[&TensorProperty], interner: &Interner) -> Vec<String> {
    let mut names = props
        .iter()
        .map(|prop| format_tensor_property(prop, interner))
        .collect::<Vec<_>>();
    names.sort();
    names
}

pub fn diff_expressions(a: &Expr, b: &Expr, interner: &Interner) -> Value {
    if a == b {
        return json!({ "identical": true });
    }

    let mut details = Vec::new();

    if let (Expr::Add(a_terms), Expr::Add(b_terms)) = (a, b) {
        if a_terms.len() != b_terms.len() {
            details.push("term_count_differs".to_string());
        }
    }
    if let (Expr::Mul(a_factors), Expr::Mul(b_factors)) = (a, b) {
        if a_factors.len() != b_factors.len() {
            details.push("factor_count_differs".to_string());
        }
    }

    if same_structure_ignoring_coefficients(a, b) || additive_coefficient_difference(a, b) {
        let (a_coeff, a_rest) = split_numeric_coeff(a);
        let (b_coeff, b_rest) = split_numeric_coeff(b);
        if (a_rest == b_rest && a_coeff != b_coeff) || additive_coefficient_difference(a, b) {
            details.push("coefficient_differs".to_string());
        }
    }

    if same_structure_ignoring_index_names(a, b) {
        let mut a_indices = Vec::new();
        let mut b_indices = Vec::new();
        collect_indices(a, &mut a_indices);
        collect_indices(b, &mut b_indices);
        if a_indices.len() == b_indices.len()
            && a_indices
                .iter()
                .zip(b_indices.iter())
                .any(|(lhs, rhs)| lhs.name != rhs.name)
        {
            details.push("index_names_differ".to_string());
        }
    }

    if same_structure_ignoring_index_positions(a, b) {
        let mut a_indices = Vec::new();
        let mut b_indices = Vec::new();
        collect_indices(a, &mut a_indices);
        collect_indices(b, &mut b_indices);
        if a_indices.len() == b_indices.len()
            && a_indices
                .iter()
                .zip(b_indices.iter())
                .any(|(lhs, rhs)| lhs.variance != rhs.variance)
        {
            details.push("index_positions_differ".to_string());
        }
    }

    let description = if has_same_term_or_factor_multiset(a, b, interner) {
        "The expressions contain the same terms or factors, but in a different order.".to_string()
    } else if details.contains(&"coefficient_differs".to_string()) {
        "The expressions have the same algebraic structure but different numerical coefficients."
            .to_string()
    } else if details.contains(&"index_names_differ".to_string()) {
        "The tensor structure matches, but the index names differ; rename_dummies may unify them."
            .to_string()
    } else if details.contains(&"index_positions_differ".to_string()) {
        "The tensor structure matches, but one or more indices appear with different variance."
            .to_string()
    } else {
        details.push("completely_different".to_string());
        format!(
            "The expressions are structurally different: '{}' versus '{}'.",
            render_expr(a, interner),
            render_expr(b, interner)
        )
    };

    json!({
        "identical": false,
        "description": description,
        "details": details,
    })
}

pub fn check_properties(expr: &Expr, algorithm: &str, env: &Env, interner: &Interner) -> Value {
    let mut indexed_symbols = Vec::new();
    collect_indexed_symbols(expr, &mut indexed_symbols);
    indexed_symbols.sort_by_key(|sym| interner.resolve(*sym).to_string());
    indexed_symbols.dedup();

    let mut symbol_rows = Vec::new();
    let mut issues = Vec::new();

    for sym in indexed_symbols {
        let props = env.property_store.get_all(sym);
        let property_names = prop_names(&props, interner);
        let mut row = json!({
            "name": interner.resolve(sym),
            "properties": property_names,
            "ok": true,
        });
        let needs_symmetry = matches!(algorithm, "canonicalise" | "meld");
        if needs_symmetry && !symmetry_properties(&props) {
            row["ok"] = json!(false);
            row["issue"] = json!("no symmetry properties declared");
            issues.push(format!(
                "Symbol {} has no symmetry properties — {} requires at least Symmetric, AntiSymmetric, TableauSymmetry, or RiemannSymmetry",
                interner.resolve(sym),
                algorithm
            ));
        }
        symbol_rows.push(row);
    }

    let mut indices = Vec::new();
    collect_indices(expr, &mut indices);
    let mut index_rows = Vec::new();
    let mut seen_index_rows = BTreeSet::new();
    let mut by_name: HashMap<lasso::Spur, Vec<Index>> = HashMap::new();
    for index in &indices {
        by_name.entry(index.name).or_default().push(index.clone());
        let family = env
            .index_to_family
            .get(&index.name)
            .copied()
            .or_else(|| env.property_store.index_to_family.get(&index.name).copied());
        let key = (
            interner.resolve(index.name).to_string(),
            family.map(|fam| interner.resolve(fam).to_string()),
        );
        if seen_index_rows.insert(key.clone()) {
            let mut row = json!({
                "name": key.0,
                "family": key.1,
                "ok": true,
            });
            if key.1.is_none() {
                row["ok"] = json!(false);
                row["issue"] = json!("index has no declared family");
                issues.push(format!(
                    "Index {} has no declared index family",
                    interner.resolve(index.name)
                ));
            }
            index_rows.push(row);
        }
    }

    for (name, occs) in &by_name {
        if occs.len() == 2 && occs[0].variance == occs[1].variance {
            issues.push(format!(
                "Dummy index {} appears twice with the same variance",
                interner.resolve(*name)
            ));
        } else if occs.len() > 2 {
            issues.push(format!(
                "Index {} appears {} times; dummy pairs should be well-formed",
                interner.resolve(*name),
                occs.len()
            ));
        }
    }

    match algorithm {
        "meld" => {
            if !matches!(expr, Expr::Add(_)) {
                issues.push("meld requires a sum expression".to_string());
            }
        }
        "eliminate_metric" => {
            let has_metric = env.property_store.symbols().into_iter().any(|sym| {
                env.property_store
                    .get_all(sym)
                    .iter()
                    .any(|prop| matches!(prop, TensorProperty::Metric))
            });
            if !has_metric {
                issues.push(
                    "No symbol with Metric property is declared — eliminate_metric needs a metric tensor"
                        .to_string(),
                );
            }
            let has_dummy_pair = by_name
                .values()
                .any(|occs| occs.len() == 2 && occs[0].variance != occs[1].variance);
            if !has_dummy_pair {
                issues.push(
                    "No matching up/down dummy index pairs were found for a metric contraction"
                        .to_string(),
                );
            }
        }
        "eliminate_kronecker" => {
            let has_delta = env.property_store.symbols().into_iter().any(|sym| {
                env.property_store
                    .get_all(sym)
                    .iter()
                    .any(|prop| matches!(prop, TensorProperty::KroneckerDelta))
            });
            if !has_delta {
                issues.push(
                    "No symbol with KroneckerDelta property is declared — eliminate_kronecker needs one"
                        .to_string(),
                );
            }
        }
        "sort_product" => {
            if !matches!(expr, Expr::Mul(_)) {
                issues.push("sort_product requires a product expression".to_string());
            }
            let mut has_sort_order = false;
            let mut has_commutativity = false;
            let mut symbols = Vec::new();
            collect_indexed_symbols(expr, &mut symbols);
            for sym in symbols {
                for prop in env.property_store.get_all(sym) {
                    if matches!(prop, TensorProperty::SortOrder(_)) {
                        has_sort_order = true;
                    }
                    if matches!(
                        prop,
                        TensorProperty::Commuting
                            | TensorProperty::AntiCommuting
                            | TensorProperty::NonCommuting
                            | TensorProperty::SelfCommuting
                            | TensorProperty::SelfAntiCommuting
                            | TensorProperty::SelfNonCommuting
                            | TensorProperty::CommutingAsProduct
                            | TensorProperty::CommutingAsSum
                    ) {
                        has_commutativity = true;
                    }
                }
            }
            if !has_sort_order {
                issues
                    .push("No SortOrder property is declared for the product factors".to_string());
            }
            if !has_commutativity {
                issues.push(
                    "No commutativity properties are declared for the product factors".to_string(),
                );
            }
        }
        "evaluate_components" => {
            if env.coordinates.is_empty() {
                issues.push(
                    "No coordinates are declared — evaluate_components needs active coordinates"
                        .to_string(),
                );
            }
            if !indexed_symbols_present(expr) {
                issues.push(
                    "The expression has no indexed tensors to evaluate into components".to_string(),
                );
            }
            let mut needed = Vec::new();
            collect_indexed_symbols(expr, &mut needed);
            needed.sort_by_key(|sym| interner.resolve(*sym).to_string());
            needed.dedup();
            for sym in needed {
                if !env.component_rule_symbols.contains(&sym) {
                    issues.push(format!(
                        "No component rules are known for symbol {}",
                        interner.resolve(sym)
                    ));
                }
            }
        }
        "rename_dummies" => {
            for (name, occs) in &by_name {
                if occs.len() == 2 {
                    let family = env
                        .index_to_family
                        .get(name)
                        .copied()
                        .or_else(|| env.property_store.index_to_family.get(name).copied());
                    if family.is_none() {
                        issues.push(format!(
                            "Dummy index {} has no declared index family",
                            interner.resolve(*name)
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    let ready = issues.is_empty();
    json!({
        "algorithm": algorithm,
        "ready": ready,
        "symbols": symbol_rows,
        "indices": index_rows,
        "issues": issues,
    })
}

fn indexed_symbols_present(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(indexed_symbols_present)
        }
        Expr::Matrix(rows) => rows.iter().flatten().any(indexed_symbols_present),
        Expr::Pow(base, exp) => indexed_symbols_present(base) || indexed_symbols_present(exp),
        Expr::Neg(inner) | Expr::Group(inner, _) => indexed_symbols_present(inner),
        Expr::Complex(re, im) => indexed_symbols_present(re) || indexed_symbols_present(im),
        Expr::Call(_, args) => args.iter().any(indexed_symbols_present),
        Expr::FnDef(_, _, body) => indexed_symbols_present(body),
        Expr::Rule(lhs, rhs, _) => indexed_symbols_present(lhs) || indexed_symbols_present(rhs),
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, _)| indexed_symbols_present(value)),
        Expr::Let(_, value, body) => {
            indexed_symbols_present(value) || indexed_symbols_present(body)
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

pub fn explain_algorithm(algorithm: &str, expr: &Expr, env: &Env, interner: &Interner) -> String {
    let summary = match algorithm {
        "canonicalise" => "canonicalise reorders tensor indices and dummy names into a canonical form using declared symmetry properties such as Symmetric, AntiSymmetric, TableauSymmetry, and RiemannSymmetry.",
        "meld" => "meld combines equivalent tensor terms in a sum after using declared symmetries and dummy-index renaming to identify matches.",
        "collect_terms" => "collect_terms groups algebraically identical terms and adds their coefficients.",
        "simplify" => "simplify runs the general algebraic simplification pipeline, including arithmetic cleanup, expansion/collection steps, and lightweight exact rewrites.",
        "sort_product" => "sort_product reorders factors in a product using declared SortOrder and commutativity properties, including graded sign changes for anticommuting factors.",
        "eliminate_metric" => "eliminate_metric uses metric and inverse-metric factors to raise or lower contracted indices and removes the metric factors that were consumed.",
        "eliminate_kronecker" => "eliminate_kronecker contracts Kronecker deltas against matching indices and substitutes the surviving index labels through the expression.",
        "rename_dummies" => "rename_dummies rewrites repeated up/down dummy pairs to a canonical naming scheme within each declared index family.",
        "evaluate_components" => "evaluate_components replaces indexed tensors with explicit component expressions using declared coordinates, tensor properties, and the component rules supplied to the call.",
        "epsilon_to_delta" => "epsilon_to_delta rewrites products of epsilon tensors into antisymmetrised Kronecker deltas or metric contractions.",
        "expand_delta" => "expand_delta expands generalised delta objects into sums of simpler contractions.",
        "reduce_delta" => "reduce_delta simplifies products and chains of deltas by composing their substitutions.",
        "distribute" => "distribute pushes products across sums to expand multiplicative structure.",
        "unwrap" => "unwrap removes wrapper constructs and nested containers when they can be flattened without changing meaning.",
        "product_rule" => "product_rule applies the Leibniz rule to derivatives of products.",
        "integrate_by_parts" => "integrate_by_parts performs one integration-by-parts step, moving a derivative away from the chosen variable.",
        "factor_out" => "factor_out extracts common prefactors from terms in a sum.",
        "factor_in" => "factor_in regroups terms so common prefactors are absorbed back into grouped factors.",
        _ => "This algorithm transforms expressions according to its own structural rules and declared tensor metadata.",
    };

    let checks = check_properties(expr, algorithm, env, interner);
    let issues = checks["issues"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|issue| issue.as_str().map(str::to_string))
        .collect::<Vec<_>>();

    if issues.is_empty() {
        format!(
            "{summary} It may not have changed this expression because the expression is already in the target form."
        )
    } else {
        format!(
            "{summary} It may not have changed this expression because:\n- {}",
            issues.join("\n- ")
        )
    }
}
