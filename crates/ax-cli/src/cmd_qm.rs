//! First-class QM CLI workflows backed by the existing evaluator and renderer.

use crate::cmd_run;
use ax_ir::Expr;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum QmCliErrorKind {
    Evaluation,
    Qm,
}

#[derive(Debug)]
pub struct QmCliError {
    kind: QmCliErrorKind,
    message: String,
}

impl QmCliError {
    fn evaluation(message: impl Into<String>) -> Self {
        Self {
            kind: QmCliErrorKind::Evaluation,
            message: message.into(),
        }
    }

    fn qm(message: impl Into<String>) -> Self {
        Self {
            kind: QmCliErrorKind::Qm,
            message: message.into(),
        }
    }

    pub fn heading(&self) -> &'static str {
        match self.kind {
            QmCliErrorKind::Evaluation => "Evaluation error:",
            QmCliErrorKind::Qm => "QM error:",
        }
    }
}

impl fmt::Display for QmCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for QmCliError {}

struct EvaluatedExpr {
    expr: Expr,
    interner: ax_ir::Interner,
}

fn lower_source(source: &str, interner: &ax_ir::Interner) -> Result<Vec<Expr>, QmCliError> {
    let lowered = ax_core_ir::lower(source, interner);
    if !lowered.errors.is_empty() {
        let joined = lowered
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(QmCliError::evaluation(joined));
    }
    if lowered.exprs.is_empty() {
        if let Some(expr) = lowered.expr {
            return Ok(vec![expr]);
        }
    }
    Ok(lowered.exprs)
}

fn split_top_level_csv(text: &str) -> Result<Vec<&str>, QmCliError> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;

    for (idx, ch) in text.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => {
                if paren_depth == 0 {
                    return Err(QmCliError::evaluation(
                        "unbalanced `)` while parsing matrix literal",
                    ));
                }
                paren_depth -= 1;
            }
            '[' => bracket_depth += 1,
            ']' => {
                if bracket_depth == 0 {
                    return Err(QmCliError::evaluation(
                        "unbalanced `]` while parsing matrix literal",
                    ));
                }
                bracket_depth -= 1;
            }
            '{' => brace_depth += 1,
            '}' => {
                if brace_depth == 0 {
                    return Err(QmCliError::evaluation(
                        "unbalanced `}` while parsing matrix literal",
                    ));
                }
                brace_depth -= 1;
            }
            ',' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                let part = text[start..idx].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if paren_depth != 0 || bracket_depth != 0 || brace_depth != 0 {
        return Err(QmCliError::evaluation(
            "unbalanced delimiters while parsing matrix literal",
        ));
    }

    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    Ok(parts)
}

fn apply_side_effect_declaration(
    expr: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) -> bool {
    ax_eval::apply_set_convention(expr, env).is_some()
        || ax_eval::apply_parallel_declaration(expr, env, interner).is_some()
        || ax_eval::apply_graded_declaration(expr, env, interner).is_some()
        || ax_eval::apply_superspace_setup(expr, env, interner).is_some()
        || ax_eval::apply_brst_setup(expr, env, interner).is_some()
        || ax_eval::apply_property_declaration(expr, env, interner).is_some()
        || ax_eval::apply_coordinate_declaration(expr, env, interner).is_some()
        || ax_eval::apply_index_declaration(expr, env, interner).is_some()
}

fn apply_result_side_effect(
    result: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) -> bool {
    ax_eval::register_rule(result, env, interner).is_some()
        || ax_eval::apply_grassmann_declaration(result, env, interner).is_some()
        || ax_eval::apply_operator_declaration(result, env, interner).is_some()
        || ax_eval::apply_named_operator_declaration(result, env, interner).is_some()
        || ax_eval::apply_named_contraction_declaration(result, env, interner).is_some()
}

fn evaluate_expr_silent(
    expr: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<Option<Expr>, QmCliError> {
    if let Expr::Import(path) = expr {
        cmd_run::handle_import(path, env, interner, search_paths)
            .map_err(|err: anyhow::Error| QmCliError::evaluation(err.to_string()))?;
        return Ok(None);
    }

    if apply_side_effect_declaration(expr, env, interner) {
        return Ok(None);
    }

    if let Expr::Let(name, value, body) = expr {
        let evaluated_value = ax_eval::eval(value, env, interner);
        env.bindings.insert(*name, evaluated_value.clone());
        let rendered = if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
            evaluated_value
        } else {
            ax_eval::eval(body, env, interner)
        };
        return Ok(Some(rendered));
    }

    let result = ax_eval::eval(expr, env, interner);

    if let Expr::Assume(sym, assumptions) = &result {
        env.assumptions
            .entry(*sym)
            .or_default()
            .extend(assumptions.clone());
        return Ok(None);
    }

    if let Expr::FnDef(name, _, _) = &result {
        env.bindings.insert(*name, result.clone());
        return Ok(None);
    }

    if apply_result_side_effect(&result, env, interner) {
        return Ok(None);
    }

    Ok(Some(result))
}

fn evaluate_with_state(
    source: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<Expr, QmCliError> {
    let exprs = lower_source(source, interner)?;
    let mut last_result = None;

    for expr in exprs {
        if let Some(value) = evaluate_expr_silent(&expr, env, interner, search_paths)? {
            last_result = Some(value);
        }
    }

    last_result.ok_or_else(|| {
        QmCliError::evaluation("no evaluable expression produced by the provided source")
    })
}

fn try_parse_matrix_literal(
    source: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<Option<Expr>, QmCliError> {
    let trimmed = source.trim();
    if !(trimmed.starts_with("[[") && trimmed.ends_with("]]")) {
        return Ok(None);
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let row_texts = split_top_level_csv(inner)?;
    if row_texts.is_empty()
        || !row_texts
            .iter()
            .all(|row| row.starts_with('[') && row.ends_with(']'))
    {
        return Ok(None);
    }

    let mut rows = Vec::new();
    for row_text in row_texts {
        let row_inner = &row_text[1..row_text.len() - 1];
        let entries = split_top_level_csv(row_inner)?;
        let mut row = Vec::new();
        for entry in entries {
            row.push(evaluate_with_state(entry, env, interner, search_paths)?);
        }
        rows.push(row);
    }
    Ok(Some(Expr::Matrix(rows)))
}

fn evaluate_source(source: &str) -> Result<EvaluatedExpr, QmCliError> {
    let interner = ax_ir::Interner::new();
    let expr = evaluate_source_with_interner(source, &interner)?;

    Ok(EvaluatedExpr { expr, interner })
}

fn evaluate_source_with_interner(
    source: &str,
    interner: &ax_ir::Interner,
) -> Result<Expr, QmCliError> {
    let mut env = ax_eval::Env::new();
    let search_paths = cmd_run::default_search_paths(None);
    if let Some(matrix) = try_parse_matrix_literal(source, &mut env, interner, &search_paths)? {
        return Ok(matrix);
    }
    evaluate_with_state(source, &mut env, interner, &search_paths)
}

fn rendered_error_symbol(expr: &Expr, interner: &ax_ir::Interner) -> Option<String> {
    let Expr::Sym(sym) = expr else {
        return None;
    };
    let message = interner.resolve(*sym);
    if message.chars().any(char::is_whitespace) {
        Some(message.to_string())
    } else {
        None
    }
}

fn evaluate_qm_expr(source: &str) -> Result<EvaluatedExpr, QmCliError> {
    let evaluated = evaluate_source(source)?;
    if let Some(message) = rendered_error_symbol(&evaluated.expr, &evaluated.interner) {
        return Err(QmCliError::qm(message));
    }
    Ok(evaluated)
}

fn require_matrix(expr: Expr) -> Result<Vec<Vec<Expr>>, QmCliError> {
    match expr {
        Expr::Matrix(rows) => Ok(rows),
        _ => Err(QmCliError::qm(
            "expected the provided expression to evaluate to a matrix",
        )),
    }
}

fn parse_dims_csv(dims_csv: &str) -> Result<Vec<usize>, QmCliError> {
    let dims = dims_csv
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| QmCliError::evaluation(format!("invalid dimension `{part}`")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.is_empty() {
        return Err(QmCliError::evaluation(
            "expected at least one subsystem dimension in --dims",
        ));
    }
    Ok(dims)
}

fn parse_jump_expressions(jumps: &str) -> Result<Vec<String>, QmCliError> {
    let parsed = jumps
        .split(';')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        return Err(QmCliError::evaluation(
            "expected at least one jump operator in --jumps",
        ));
    }
    Ok(parsed)
}

fn expr_to_display(expr: &Expr, interner: &ax_ir::Interner) -> String {
    ax_ir::pretty_print(expr, interner)
}

fn eval_qm_builtin(
    name: &str,
    args: Vec<Expr>,
    interner: &ax_ir::Interner,
) -> Result<Expr, QmCliError> {
    let func = interner.get_or_intern(name);
    let result = ax_eval::eval(&Expr::Call(func, args), &ax_eval::Env::new(), interner);
    if let Some(message) = rendered_error_symbol(&result, interner) {
        return Err(QmCliError::qm(message));
    }
    Ok(result)
}

/// Evaluate a density-matrix expression and print a compact QM summary.
pub fn summarize(expr: &str) -> Result<(), QmCliError> {
    let evaluated = evaluate_qm_expr(expr)?;
    let matrix_expr = evaluated.expr;
    let rows = require_matrix(matrix_expr.clone())?;
    let dims = format!("{}x{}", rows.len(), rows.first().map(Vec::len).unwrap_or(0));
    let trace = eval_qm_builtin("trace", vec![matrix_expr.clone()], &evaluated.interner)?;
    let purity = eval_qm_builtin("purity", vec![matrix_expr.clone()], &evaluated.interner)?;
    println!("Dimension: {dims}");
    println!("Trace: {}", expr_to_display(&trace, &evaluated.interner));
    println!("Purity: {}", expr_to_display(&purity, &evaluated.interner));
    if let Ok(entropy) = eval_qm_builtin(
        "von_neumann_entropy",
        vec![matrix_expr],
        &evaluated.interner,
    ) {
        println!(
            "Entropy: {}",
            expr_to_display(&entropy, &evaluated.interner)
        );
    }
    Ok(())
}

/// Evaluate a matrix expression and print its von Neumann entropy.
pub fn entropy(matrix_expr: &str) -> Result<(), QmCliError> {
    let evaluated = evaluate_qm_expr(matrix_expr)?;
    let result = eval_qm_builtin(
        "von_neumann_entropy",
        vec![evaluated.expr],
        &evaluated.interner,
    )?;
    println!("{}", expr_to_display(&result, &evaluated.interner));
    Ok(())
}

/// Evaluate a matrix expression and print a factor partial trace.
pub fn partial_trace(matrix_expr: &str, dims_csv: &str, factor: usize) -> Result<(), QmCliError> {
    let dims = parse_dims_csv(dims_csv)?;
    let evaluated = evaluate_qm_expr(matrix_expr)?;
    let dims_expr = Expr::List(
        dims.into_iter()
            .map(|dim| Expr::Int(dim.into()))
            .collect::<Vec<_>>(),
    );
    let result = eval_qm_builtin(
        "partial_trace_factor",
        vec![evaluated.expr, dims_expr, Expr::Int(factor.into())],
        &evaluated.interner,
    )?;
    println!("{}", expr_to_display(&result, &evaluated.interner));
    Ok(())
}

/// Evaluate a qubit density matrix and print its Bloch vector.
pub fn bloch(matrix_expr: &str) -> Result<(), QmCliError> {
    let evaluated = evaluate_qm_expr(matrix_expr)?;
    let result = eval_qm_builtin("bloch_vector", vec![evaluated.expr], &evaluated.interner)?;
    println!("{}", expr_to_display(&result, &evaluated.interner));
    Ok(())
}

/// Evaluate a Lindblad steady state from a Hamiltonian and semicolon-separated jump operators.
pub fn steady_state(hamiltonian_expr: &str, jumps: &str) -> Result<(), QmCliError> {
    let jumps = parse_jump_expressions(jumps)?;
    let interner = ax_ir::Interner::new();
    let h_expr = evaluate_source_with_interner(hamiltonian_expr, &interner)?;
    if let Some(message) = rendered_error_symbol(&h_expr, &interner) {
        return Err(QmCliError::qm(message));
    }
    let mut jump_exprs = Vec::new();
    for jump in jumps {
        let jump_expr = evaluate_source_with_interner(&jump, &interner)?;
        if let Some(message) = rendered_error_symbol(&jump_expr, &interner) {
            return Err(QmCliError::qm(message));
        }
        jump_exprs.push(jump_expr);
    }
    let result = eval_qm_builtin(
        "lindblad_steady_state",
        vec![h_expr, Expr::List(jump_exprs)],
        &interner,
    )?;
    println!("{}", expr_to_display(&result, &interner));
    Ok(())
}
