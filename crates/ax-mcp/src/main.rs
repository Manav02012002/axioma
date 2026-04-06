use ax_ir::Expr;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

struct McpState {
    interner: ax_ir::Interner,
    env: ax_eval::Env,
    expressions: HashMap<String, Expr>,
    next_id: u64,
}

impl McpState {
    fn new() -> Self {
        Self {
            interner: ax_ir::Interner::new(),
            env: ax_eval::Env::new(),
            expressions: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_expr_id(&mut self) -> String {
        let id = format!("expr_{}", self.next_id);
        self.next_id += 1;
        id
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolDef {
    name: &'static str,
    description: &'static str,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

fn tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "axioma_eval",
            description: "Parse and evaluate .ax code, returning the result.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": { "type": "string" }
                },
                "required": ["code"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "axioma_parse",
            description: "Parse .ax code and return diagnostics without evaluation.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": { "type": "string" }
                },
                "required": ["code"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "axioma_inspect",
            description: "Inspect a stored expression by ID.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expr_id": { "type": "string" }
                },
                "required": ["expr_id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "axioma_suggest",
            description: "Suggest applicable algorithms for a stored expression.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expr_id": { "type": "string" }
                },
                "required": ["expr_id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "axioma_render",
            description: "Render a stored expression to LaTeX or Unicode.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expr_id": { "type": "string" },
                    "format": {
                        "type": "string",
                        "enum": ["latex", "unicode"]
                    }
                },
                "required": ["expr_id", "format"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "axioma_env",
            description: "Return the current Axioma environment state.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        },
    ]
}

fn make_result_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn make_error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "axioma-mcp",
            "version": "0.1.0"
        }
    })
}

fn tools_list_result() -> Value {
    json!({
        "tools": tool_definitions()
    })
}

fn not_implemented(tool_name: &str) -> Value {
    json!({
        "error": "not yet implemented",
        "tool": tool_name
    })
}

fn require_expr_id(arguments: Option<&Value>) -> Result<&str, Value> {
    arguments
        .and_then(|args| args.get("expr_id"))
        .and_then(Value::as_str)
        .ok_or_else(|| json!({ "error": "missing or invalid 'expr_id'" }))
}

fn severity_name(severity: ax_syntax::diag::Severity) -> &'static str {
    match severity {
        ax_syntax::diag::Severity::Error => "error",
        ax_syntax::diag::Severity::Warning => "warning",
    }
}

fn syntax_diag_json(diag: &ax_syntax::diag::Diagnostic) -> Value {
    json!({
        "severity": severity_name(diag.severity),
        "message": diag.message,
        "span": {
            "start": diag.span.start,
            "end": diag.span.end
        },
        "fixits": diag.fixits.iter().map(|fix| {
            json!({
                "span": {
                    "start": fix.span.start,
                    "end": fix.span.end
                },
                "replacement": fix.replacement,
                "message": fix.message
            })
        }).collect::<Vec<_>>()
    })
}

fn lower_error_json(err: &ax_core_ir::LowerError) -> Value {
    json!({
        "severity": "error",
        "message": err.message,
        "span": {
            "start": err.span.start,
            "end": err.span.end
        },
        "fixits": []
    })
}

fn default_search_paths() -> Vec<PathBuf> {
    let mut search_paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.clone());
        search_paths.push(cwd.join("std"));
    }
    if let Ok(std_path) = std::env::var("AXIOMA_STD_PATH") {
        search_paths.push(PathBuf::from(std_path));
    }
    search_paths
}

fn handle_import(
    path: &[ax_ir::expr::Sym],
    state: &mut McpState,
    search_paths: &[PathBuf],
) -> Result<(), String> {
    let import_name = path
        .iter()
        .map(|sym| state.interner.resolve(*sym))
        .collect::<Vec<_>>()
        .join(".");
    let file_path = ax_eval::resolve_import(path, &state.interner, search_paths)
        .ok_or_else(|| format!("import not found: {import_name}"))?;
    let source =
        std::fs::read_to_string(&file_path).map_err(|e| format!("failed to read import: {e}"))?;
    let lowered = ax_core_ir::lower(&source, &state.interner);
    if !lowered.errors.is_empty() {
        let joined = lowered
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("failed to lower import {import_name}: {joined}"));
    }

    let mut nested_search_paths = vec![file_path.parent().unwrap_or(Path::new(".")).to_path_buf()];
    nested_search_paths.extend_from_slice(search_paths);

    for expr in lowered.exprs {
        apply_expr_side_effects(state, &expr, &nested_search_paths)?;
    }

    Ok(())
}

fn apply_expr_side_effects(
    state: &mut McpState,
    expr: &Expr,
    search_paths: &[PathBuf],
) -> Result<Expr, String> {
    match expr {
        Expr::Import(path) => {
            handle_import(path, state, search_paths)?;
            Ok(Expr::Import(path.clone()))
        }
        Expr::Let(name, val, body) => {
            let evaled_val = ax_eval::eval(val, &state.env, &state.interner);
            state.env.bindings.insert(*name, evaled_val.clone());
            let rendered = if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                evaled_val
            } else {
                ax_eval::eval(body, &state.env, &state.interner)
            };
            Ok(rendered)
        }
        Expr::FnDef(name, _, _) => {
            let result = ax_eval::eval(expr, &state.env, &state.interner);
            state.env.bindings.insert(*name, result.clone());
            Ok(result)
        }
        Expr::Rule(_, _, _) => {
            let result = ax_eval::eval(expr, &state.env, &state.interner);
            let _ = ax_eval::register_rule(&result, &mut state.env, &state.interner);
            Ok(result)
        }
        Expr::Assume(var, assumptions) => {
            state
                .env
                .assumptions
                .entry(*var)
                .or_default()
                .extend(assumptions.clone());
            Ok(expr.clone())
        }
        Expr::SetConvention(_, _) => {
            let _ = ax_eval::apply_set_convention(expr, &mut state.env);
            Ok(expr.clone())
        }
        _ => {
            if ax_eval::apply_property_declaration(expr, &mut state.env, &state.interner).is_some()
            {
                return Ok(expr.clone());
            }
            if ax_eval::apply_coordinate_declaration(expr, &mut state.env, &state.interner)
                .is_some()
            {
                return Ok(expr.clone());
            }
            if ax_eval::apply_index_declaration(expr, &mut state.env, &state.interner).is_some() {
                return Ok(expr.clone());
            }

            if let Expr::Call(f, args) = expr {
                if state.interner.resolve(*f) == "__declare_depends" && args.len() == 2 {
                    if let (Expr::Sym(tensor), Expr::List(dep_list)) = (&args[0], &args[1]) {
                        let deps: Vec<_> = dep_list
                            .iter()
                            .filter_map(|e| if let Expr::Sym(s) = e { Some(*s) } else { None })
                            .collect();
                        state
                            .env
                            .tensor_properties
                            .entry(*tensor)
                            .or_default()
                            .push(ax_ir::TensorProperty::Depends(deps));
                        return Ok(expr.clone());
                    }
                }
            }

            if let Expr::Call(f, args) = expr {
                if state.interner.resolve(*f) == "__declare_weight" && args.len() >= 2 {
                    if let (Expr::Sym(sym), Expr::Int(w)) = (&args[0], &args[1]) {
                        let label = args
                            .get(2)
                            .and_then(|e| {
                                if let Expr::Sym(s) = e {
                                    Some(state.interner.resolve(*s).to_string())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        let weight_val: i64 = w.to_string().parse().unwrap_or(0);
                        state.env.weights.insert((*sym, label), weight_val);
                        return Ok(expr.clone());
                    }
                }
            }

            let result = ax_eval::eval(expr, &state.env, &state.interner);
            if let Some(rule_name) = ax_eval::register_rule(&result, &mut state.env, &state.interner)
            {
                let _ = rule_name;
                return Ok(result);
            }
            if let Expr::Assume(var, assumptions) = &result {
                state
                    .env
                    .assumptions
                    .entry(*var)
                    .or_default()
                    .extend(assumptions.clone());
                return Ok(result);
            }
            if let Expr::FnDef(name, _, _) = &result {
                state.env.bindings.insert(*name, result.clone());
                return Ok(result);
            }
            let _ = ax_eval::apply_grassmann_declaration(&result, &mut state.env, &state.interner);
            let _ = ax_eval::apply_operator_declaration(&result, &mut state.env, &state.interner);
            Ok(result)
        }
    }
}

fn require_code(arguments: Option<&Value>) -> Result<&str, Value> {
    arguments
        .and_then(|args| args.get("code"))
        .and_then(Value::as_str)
        .ok_or_else(|| json!({ "error": "missing or invalid 'code'" }))
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Indexed(_, _) => "indexed",
        Expr::Matrix(_) => "matrix",
        Expr::List(_) => "list",
        Expr::Add(_) => "sum",
        Expr::Mul(_) => "product",
        Expr::Call(_, _) => "function_call",
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => "scalar",
        _ => "other",
    }
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

fn variance_name(variance: &ax_ir::Variance) -> &'static str {
    match variance {
        ax_ir::Variance::Up => "up",
        ax_ir::Variance::Down => "down",
    }
}

fn property_name(prop: &ax_ir::TensorProperty, interner: &ax_ir::Interner) -> String {
    match prop {
        ax_ir::TensorProperty::Symmetric(pos) => format!("Symmetric({pos:?})"),
        ax_ir::TensorProperty::AntiSymmetric(pos) => format!("AntiSymmetric({pos:?})"),
        ax_ir::TensorProperty::RiemannSymmetry => "RiemannSymmetry".to_string(),
        ax_ir::TensorProperty::Traceless => "Traceless".to_string(),
        ax_ir::TensorProperty::Metric => "Metric".to_string(),
        ax_ir::TensorProperty::InverseMetric => "InverseMetric".to_string(),
        ax_ir::TensorProperty::KroneckerDelta => "KroneckerDelta".to_string(),
        ax_ir::TensorProperty::EpsilonTensor => "EpsilonTensor".to_string(),
        ax_ir::TensorProperty::Derivative => "Derivative".to_string(),
        ax_ir::TensorProperty::PartialDerivative => "PartialDerivative".to_string(),
        ax_ir::TensorProperty::CovariantDerivative => "CovariantDerivative".to_string(),
        ax_ir::TensorProperty::Depends(syms) => format!(
            "Depends({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::Spinor => "Spinor".to_string(),
        ax_ir::TensorProperty::DiracBar => "DiracBar".to_string(),
        ax_ir::TensorProperty::GammaMatrixProp => "GammaMatrixProp".to_string(),
        ax_ir::TensorProperty::Commuting => "Commuting".to_string(),
        ax_ir::TensorProperty::AntiCommuting => "AntiCommuting".to_string(),
        ax_ir::TensorProperty::NonCommuting => "NonCommuting".to_string(),
        ax_ir::TensorProperty::SortOrder(syms) => format!(
            "SortOrder({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::TableauSymmetry { shape, indices } => {
            format!("TableauSymmetry(shape={shape:?}, indices={indices:?})")
        }
        ax_ir::TensorProperty::SatisfiesBianchi => "SatisfiesBianchi".to_string(),
        ax_ir::TensorProperty::WeylTensor => "WeylTensor".to_string(),
        ax_ir::TensorProperty::DifferentialFormDegree(n) => {
            format!("DifferentialFormDegree({n})")
        }
    }
}

fn collect_symbols(expr: &Expr, out: &mut Vec<ax_ir::expr::Sym>) {
    match expr {
        Expr::Sym(sym) => out.push(*sym),
        Expr::Indexed(base, _) => collect_symbols(base, out),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_symbols(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_symbols(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_symbols(base, out);
            collect_symbols(exp, out);
        }
        Expr::Neg(inner) => collect_symbols(inner, out),
        Expr::Complex(re, im) => {
            collect_symbols(re, out);
            collect_symbols(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_symbols(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_symbols(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_symbols(lhs, out);
            collect_symbols(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_symbols(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_symbols(value, out);
            collect_symbols(body, out);
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Int(_)
        | Expr::Rational(_) | Expr::Float(_) => {}
    }
}

fn node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => 1,
        Expr::Indexed(base, indices) => 1 + node_count(base) + indices.len(),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Complex(re, im) => 1 + node_count(re) + node_count(im),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::FnDef(_, _, body) => 1 + node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + node_count(lhs) + node_count(rhs),
        Expr::Piecewise(cases) => 1 + cases.iter().map(|(value, _)| node_count(value)).sum::<usize>(),
        Expr::Let(_, value, body) => 1 + node_count(value) + node_count(body),
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

fn contains_indexed(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(contains_indexed)
        }
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
        Expr::Rule(lhs, rhs, _) => {
            contains_named_call(lhs, names, interner) || contains_named_call(rhs, names, interner)
        }
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, _)| contains_named_call(value, names, interner)),
        Expr::Let(_, value, body) => {
            contains_named_call(value, names, interner) || contains_named_call(body, names, interner)
        }
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) | Expr::Sym(_)
        | Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
    }
}

fn contains_derivative_call(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    contains_named_call(
        expr,
        &["partial", "nabla", "D", "d", "diff", "partial_derivative"],
        interner,
    )
}

fn add_suggestion(
    suggestions: &mut Vec<(String, String)>,
    seen: &mut std::collections::HashSet<String>,
    algorithm: &str,
    reason: &str,
) {
    if seen.insert(algorithm.to_string()) {
        suggestions.push((algorithm.to_string(), reason.to_string()));
    }
}

fn convention_json(convention: &ax_ir::Convention) -> Value {
    json!({
        "metric_signature": format!("{:?}", convention.metric_signature),
        "riemann_sign": format!("{:?}", convention.riemann_sign),
        "ricci_contraction": format!("{:?}", convention.ricci_contraction),
        "levi_civita_norm": format!("{:?}", convention.levi_civita_norm),
        "fourier_sign": format!("{:?}", convention.fourier_sign)
    })
}

fn sorted_expression_ids(expressions: &HashMap<String, Expr>) -> Vec<String> {
    let mut ids = expressions.keys().cloned().collect::<Vec<_>>();
    ids.sort();
    ids
}

fn handle_axioma_eval(state: &mut McpState, arguments: Option<&Value>) -> Value {
    let code = match require_code(arguments) {
        Ok(code) => code,
        Err(err) => return err,
    };

    let (_node, parse_diags) = ax_syntax::parser::parse_file(code);
    let lowered = ax_core_ir::lower(code, &state.interner);

    if !lowered.errors.is_empty() {
        let mut diagnostics = parse_diags
            .iter()
            .map(syntax_diag_json)
            .collect::<Vec<_>>();
        diagnostics.extend(lowered.errors.iter().map(lower_error_json));
        return json!({
            "diagnostics": diagnostics,
            "valid": false
        });
    }

    let search_paths = default_search_paths();
    let mut last_result = Expr::zero();
    for expr in &lowered.exprs {
        match apply_expr_side_effects(state, expr, &search_paths) {
            Ok(result) => last_result = result,
            Err(message) => {
                return json!({
                    "diagnostics": [{
                        "severity": "error",
                        "message": message,
                        "span": { "start": 0, "end": 0 },
                        "fixits": []
                    }],
                    "valid": false
                });
            }
        }
    }

    let expr_id = state.alloc_expr_id();
    state.expressions.insert(expr_id.clone(), last_result.clone());

    json!({
        "expr_id": expr_id,
        "result_latex": ax_render::to_latex(&last_result, &state.interner),
        "result_unicode": ax_render::to_unicode(&last_result, &state.interner),
        "result_raw": format!("{last_result:?}")
    })
}

fn handle_axioma_parse(_state: &mut McpState, arguments: Option<&Value>) -> Value {
    let code = match require_code(arguments) {
        Ok(code) => code,
        Err(err) => return err,
    };

    let (_node, parse_diags) = ax_syntax::parser::parse_file(code);
    let lowered = ax_core_ir::lower(code, &ax_ir::Interner::new());
    let mut diagnostics = parse_diags
        .iter()
        .map(syntax_diag_json)
        .collect::<Vec<_>>();
    diagnostics.extend(lowered.errors.iter().map(lower_error_json));

    json!({
        "diagnostics": diagnostics,
        "valid": diagnostics.is_empty()
    })
}

fn handle_axioma_inspect(state: &mut McpState, arguments: Option<&Value>) -> Value {
    let expr_id = match require_expr_id(arguments) {
        Ok(expr_id) => expr_id,
        Err(err) => return err,
    };

    let Some(expr) = state.expressions.get(expr_id).cloned() else {
        return json!({
            "error": "expression not found",
            "expr_id": expr_id
        });
    };

    let mut all_indices = Vec::new();
    collect_indices(&expr, &mut all_indices);

    let mut by_name: HashMap<ax_ir::expr::Sym, Vec<ax_ir::Index>> = HashMap::new();
    for idx in all_indices {
        by_name.entry(idx.name).or_default().push(idx);
    }

    let free_indices = by_name
        .iter()
        .filter_map(|(name, occs)| {
            if occs.len() == 1 {
                let idx = &occs[0];
                Some(json!({
                    "name": state.interner.resolve(*name),
                    "variance": variance_name(&idx.variance),
                    "family": state.env.index_to_family.get(name).map(|f| state.interner.resolve(*f).to_string())
                }))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let dummy_pairs = by_name
        .iter()
        .filter_map(|(name, occs)| {
            if occs.len() == 2 && occs[0].variance != occs[1].variance {
                let rendered = state.interner.resolve(*name).to_string();
                Some(json!([rendered, rendered]))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut symbols = Vec::new();
    collect_symbols(&expr, &mut symbols);
    symbols.sort_by_key(|s| state.interner.resolve(*s).to_string());
    symbols.dedup();

    let properties = symbols
        .iter()
        .filter_map(|sym| {
            state.env.tensor_properties.get(sym).map(|props| {
                json!({
                    "symbol": state.interner.resolve(*sym),
                    "properties": props.iter().map(|p| property_name(p, &state.interner)).collect::<Vec<_>>()
                })
            })
        })
        .collect::<Vec<_>>();

    let depends_on = symbols
        .iter()
        .map(|sym| state.interner.resolve(*sym).to_string())
        .collect::<Vec<_>>();

    let assumptions = symbols
        .iter()
        .filter_map(|sym| {
            state.env.assumptions.get(sym).map(|asms| {
                (
                    state.interner.resolve(*sym).to_string(),
                    Value::Array(
                        asms.iter()
                            .map(|a| Value::String(format!("{a:?}")))
                            .collect::<Vec<_>>(),
                    ),
                )
            })
        })
        .collect::<serde_json::Map<String, Value>>();

    json!({
        "expr_id": expr_id,
        "kind": expr_kind(&expr),
        "free_indices": free_indices,
        "dummy_pairs": dummy_pairs,
        "properties": properties,
        "depends_on": depends_on,
        "assumptions": assumptions,
        "node_count": node_count(&expr),
        "latex": ax_render::to_latex(&expr, &state.interner),
        "unicode": ax_render::to_unicode(&expr, &state.interner)
    })
}

fn handle_axioma_suggest(state: &mut McpState, arguments: Option<&Value>) -> Value {
    let expr_id = match require_expr_id(arguments) {
        Ok(expr_id) => expr_id,
        Err(err) => return err,
    };

    let Some(expr) = state.expressions.get(expr_id).cloned() else {
        return json!({
            "error": "expression not found",
            "expr_id": expr_id
        });
    };

    let mut suggestions: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut all_indices = Vec::new();
    collect_indices(&expr, &mut all_indices);
    let mut by_name: HashMap<ax_ir::expr::Sym, Vec<ax_ir::Index>> = HashMap::new();
    for idx in all_indices {
        by_name.entry(idx.name).or_default().push(idx);
    }
    let has_dummy_indices = by_name
        .values()
        .any(|occs| occs.len() == 2 && occs[0].variance != occs[1].variance);

    let mut indexed_syms = Vec::new();
    collect_indexed_base_symbols(&expr, &mut indexed_syms);
    indexed_syms.sort_by_key(|s| state.interner.resolve(*s).to_string());
    indexed_syms.dedup();

    let has_indexed = contains_indexed(&expr);
    let mut missing_properties = Vec::new();

    if has_indexed {
        let mut has_symmetry = false;
        let mut has_metric = false;
        let mut has_delta = false;
        let mut has_epsilon = false;

        for sym in &indexed_syms {
            match state.env.tensor_properties.get(sym) {
                Some(props) => {
                    for prop in props {
                        match prop {
                            ax_ir::TensorProperty::Symmetric(_)
                            | ax_ir::TensorProperty::AntiSymmetric(_)
                            | ax_ir::TensorProperty::RiemannSymmetry => has_symmetry = true,
                            ax_ir::TensorProperty::Metric => has_metric = true,
                            ax_ir::TensorProperty::KroneckerDelta => has_delta = true,
                            ax_ir::TensorProperty::EpsilonTensor => has_epsilon = true,
                            _ => {}
                        }
                    }
                }
                None => {
                    missing_properties.push(json!({
                        "symbol": state.interner.resolve(*sym),
                        "suggestion": format!(
                            "declare symmetry properties for {} to enable canonicalise",
                            state.interner.resolve(*sym)
                        )
                    }));
                }
            }
        }

        if has_symmetry {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "canonicalise",
                "expression has tensors with symmetry properties",
            );
        }
        if matches!(&expr, Expr::Add(terms) if terms.len() > 1 && terms.iter().all(contains_indexed))
        {
            add_suggestion(
                &mut suggestions,
                &mut seen,
                "meld",
                "expression is a tensor sum with multiple indexed terms",
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
        if matches!(&expr, Expr::Mul(factors) if factors.iter().any(contains_indexed)) {
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

    if let Expr::Add(terms) = &expr {
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

    if let Expr::Mul(factors) = &expr {
        if factors
            .iter()
            .any(|factor| contains_derivative_call(factor, &state.interner))
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
        &expr,
        &[
            "sin", "cos", "tan", "sec", "csc", "cot", "asin", "arcsin", "acos", "arccos",
            "atan", "arctan", "sinh", "cosh", "tanh", "asinh", "arcsinh", "acosh",
            "arccosh", "atanh", "arctanh",
        ],
        &state.interner,
    ) {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "trig_simplify",
            "expression contains trigonometric functions",
        );
    }

    if matches!(&expr, Expr::Call(f, _) if state.interner.resolve(*f) == "christoffel")
        || contains_named_call(&expr, &["christoffel"], &state.interner)
    {
        add_suggestion(
            &mut suggestions,
            &mut seen,
            "riemann",
            "expression contains Christoffel symbols",
        );
    }

    if contains_named_call(&expr, &["gamma"], &state.interner) {
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

    json!({
        "expr_id": expr_id,
        "suggestions": suggestions
            .into_iter()
            .map(|(algorithm, reason)| json!({ "algorithm": algorithm, "reason": reason }))
            .collect::<Vec<_>>(),
        "missing_properties": missing_properties
    })
}

fn handle_axioma_render(state: &mut McpState, arguments: Option<&Value>) -> Value {
    let Some(args) = arguments else {
        return json!({ "error": "missing arguments" });
    };
    let expr_id = match args.get("expr_id").and_then(Value::as_str) {
        Some(expr_id) => expr_id,
        None => return json!({ "error": "missing or invalid 'expr_id'" }),
    };
    let format = match args.get("format").and_then(Value::as_str) {
        Some(format @ ("latex" | "unicode")) => format,
        _ => return json!({ "error": "missing or invalid 'format'" }),
    };

    let Some(expr) = state.expressions.get(expr_id) else {
        return json!({
            "error": "expression not found",
            "expr_id": expr_id
        });
    };

    let rendered = match format {
        "latex" => ax_render::to_latex(expr, &state.interner),
        "unicode" => ax_render::to_unicode(expr, &state.interner),
        _ => unreachable!(),
    };

    json!({
        "expr_id": expr_id,
        "format": format,
        "rendered": rendered
    })
}

fn handle_axioma_env(state: &mut McpState, _arguments: Option<&Value>) -> Value {
    let mut binding_keys = state.env.bindings.keys().copied().collect::<Vec<_>>();
    binding_keys.sort_by_key(|sym| state.interner.resolve(*sym).to_string());
    let bindings = binding_keys
        .into_iter()
        .map(|sym| {
            (
                state.interner.resolve(sym).to_string(),
                Value::String(ax_render::to_unicode(
                    state.env.bindings.get(&sym).expect("binding key exists"),
                    &state.interner,
                )),
            )
        })
        .collect::<serde_json::Map<String, Value>>();

    let mut rules = state
        .env
        .rules
        .iter()
        .map(|rule| {
            json!({
                "name": rule.name,
                "trust": format!("{:?}", rule.trust_level)
            })
        })
        .collect::<Vec<_>>();
    rules.sort_by_key(|entry| {
        entry
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    });

    let mut assumption_keys = state.env.assumptions.keys().copied().collect::<Vec<_>>();
    assumption_keys.sort_by_key(|sym| state.interner.resolve(*sym).to_string());
    let assumptions = assumption_keys
        .into_iter()
        .map(|sym| {
            let mut values = state
                .env
                .assumptions
                .get(&sym)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|assumption| format!("{assumption:?}"))
                .collect::<Vec<_>>();
            values.sort();
            values.dedup();
            (
                state.interner.resolve(sym).to_string(),
                Value::Array(values.into_iter().map(Value::String).collect()),
            )
        })
        .collect::<serde_json::Map<String, Value>>();

    let mut tensor_keys = state
        .env
        .tensor_properties
        .keys()
        .copied()
        .collect::<Vec<_>>();
    tensor_keys.sort_by_key(|sym| state.interner.resolve(*sym).to_string());
    let tensor_properties = tensor_keys
        .into_iter()
        .map(|sym| {
            let mut values = state
                .env
                .tensor_properties
                .get(&sym)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|prop| property_name(&prop, &state.interner))
                .collect::<Vec<_>>();
            values.sort();
            values.dedup();
            (
                state.interner.resolve(sym).to_string(),
                Value::Array(values.into_iter().map(Value::String).collect()),
            )
        })
        .collect::<serde_json::Map<String, Value>>();

    let mut family_keys = state.env.index_families.keys().copied().collect::<Vec<_>>();
    family_keys.sort_by_key(|sym| state.interner.resolve(*sym).to_string());
    let index_families = family_keys
        .into_iter()
        .map(|sym| {
            let family = state
                .env
                .index_families
                .get(&sym)
                .expect("index family key exists");
            let mut indices = family
                .values
                .iter()
                .map(|value| state.interner.resolve(*value).to_string())
                .collect::<Vec<_>>();
            indices.sort();
            (
                state.interner.resolve(sym).to_string(),
                json!({
                    "indices": indices,
                    "dimension": family.dimension
                }),
            )
        })
        .collect::<serde_json::Map<String, Value>>();

    let mut coordinates = state
        .env
        .coordinates
        .iter()
        .map(|sym| state.interner.resolve(*sym).to_string())
        .collect::<Vec<_>>();
    coordinates.sort();

    json!({
        "bindings": bindings,
        "rules": rules,
        "assumptions": assumptions,
        "tensor_properties": tensor_properties,
        "index_families": index_families,
        "coordinates": coordinates,
        "convention": convention_json(&state.env.convention),
        "stored_expressions": sorted_expression_ids(&state.expressions)
    })
}

fn dispatch_tool_call(state: &mut McpState, params: Option<&Value>) -> Result<Value, &'static str> {
    let params = params.ok_or("missing params")?;
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or("missing tool name")?;
    let arguments = params.get("arguments");

    let result = match tool_name {
        "axioma_eval" => handle_axioma_eval(state, arguments),
        "axioma_parse" => handle_axioma_parse(state, arguments),
        "axioma_inspect" => handle_axioma_inspect(state, arguments),
        "axioma_suggest" => handle_axioma_suggest(state, arguments),
        "axioma_render" => handle_axioma_render(state, arguments),
        "axioma_env" => handle_axioma_env(state, arguments),
        _ => return Err("unknown tool"),
    };

    Ok(result)
}

fn handle_request(state: &mut McpState, request: &Value) -> Option<Value> {
    let id = request.get("id").cloned();
    let method = request.get("method").and_then(Value::as_str);

    let Some(method) = method else {
        return id.map(|id| make_error_response(id, -32600, "invalid request: missing method"));
    };

    match method {
        "initialize" => id.map(|id| make_result_response(id, initialize_result())),
        "tools/list" => id.map(|id| make_result_response(id, tools_list_result())),
        "tools/call" => {
            let Some(id) = id else {
                return None;
            };
            match dispatch_tool_call(state, request.get("params")) {
                Ok(result) => Some(make_result_response(id, result)),
                Err("missing params") => Some(make_error_response(id, -32602, "missing params")),
                Err("missing tool name") => {
                    Some(make_error_response(id, -32602, "missing tool name"))
                }
                Err("unknown tool") => Some(make_error_response(id, -32601, "unknown tool")),
                Err(message) => Some(make_error_response(id, -32000, message)),
            }
        }
        _ => id.map(|id| make_error_response(id, -32601, "method not found")),
    }
}

fn write_response(stdout: &mut impl Write, response: &Value) -> io::Result<()> {
    serde_json::to_writer(&mut *stdout, response)?;
    stdout.write_all(b"\n")?;
    stdout.flush()
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut lines = stdin.lock().lines();
    let mut out = stdout.lock();
    let mut state = McpState::new();

    while let Some(line) = lines.next() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => handle_request(&mut state, &request),
            Err(_) => Some(make_error_response(Value::Null, -32700, "parse error")),
        };

        if let Some(response) = response {
            write_response(&mut out, &response)?;
        }
    }

    Ok(())
}
