#![forbid(unsafe_code)]

use anyhow::{anyhow, Result};
use ax_ir::Expr;
use std::path::{Path, PathBuf};
use tiny_http::{Header, Method, Response, Server};

#[derive(serde::Serialize, Debug)]
pub struct EvalResponse {
    pub unicode: Option<String>,
    pub latex: Option<String>,
    pub error: Option<String>,
    pub svg: Option<String>,
}

fn default_search_paths() -> Vec<PathBuf> {
    let mut search_paths = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd);
    }

    if let Ok(std_path) = std::env::var("AXIOMA_STD_PATH") {
        search_paths.push(PathBuf::from(std_path));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            search_paths.push(dir.to_path_buf());
            search_paths.push(dir.join("std"));
            if let Some(parent) = dir.parent() {
                search_paths.push(parent.to_path_buf());
                search_paths.push(parent.join("std"));
            }
        }
    }

    search_paths
}

fn assume_message(
    var: ax_ir::expr::Sym,
    assumptions: &[ax_ir::Assumption],
    interner: &ax_ir::Interner,
) -> String {
    format!(
        "assumed {} is {}",
        interner.resolve(var),
        assumptions
            .iter()
            .map(|a| format!("{a:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn handle_import(
    path: &[ax_ir::expr::Sym],
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<()> {
    let import_name = path
        .iter()
        .map(|sym| interner.resolve(*sym))
        .collect::<Vec<_>>()
        .join(".");
    let file_path = ax_eval::resolve_import(path, interner, search_paths)
        .ok_or_else(|| anyhow!("import not found: {import_name}"))?;
    let source = std::fs::read_to_string(&file_path)?;
    let lowered = ax_core_ir::lower(&source, interner);

    if !lowered.errors.is_empty() {
        let joined = lowered
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(anyhow!("failed to lower import {import_name}: {joined}"));
    }

    let mut nested_search_paths = vec![file_path.parent().unwrap_or(Path::new(".")).to_path_buf()];
    nested_search_paths.extend_from_slice(search_paths);

    for expr in lowered.exprs {
        match &expr {
            Expr::Import(import_path) => handle_import(import_path, env, interner, &nested_search_paths)?,
            Expr::Let(name, val, _) => {
                let evaled_val = ax_eval::eval(val, env, interner);
                env.bindings.insert(*name, evaled_val);
            }
            Expr::FnDef(name, _, _) => {
                let result = ax_eval::eval(&expr, env, interner);
                env.bindings.insert(*name, result);
            }
            Expr::Rule(_, _, _) => {
                let result = ax_eval::eval(&expr, env, interner);
                let _ = ax_eval::register_rule(&result, env, interner);
            }
            Expr::Assume(var, assumptions) => {
                env.assumptions.entry(*var).or_default().extend(assumptions.clone());
            }
            Expr::SetConvention(_, _) => {
                let _ = ax_eval::apply_set_convention(&expr, env);
            }
            _ => {
                let _ = ax_eval::eval(&expr, env, interner);
            }
        }
    }

    Ok(())
}

fn is_plot_call(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    matches!(expr, Expr::Call(f, _) if interner.resolve(*f) == "plot")
}

pub fn handle_eval(
    body: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> EvalResponse {
    let request: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return EvalResponse {
                unicode: None,
                latex: None,
                error: Some(e.to_string()),
                svg: None,
            }
        }
    };

    let source = match request.get("source").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return EvalResponse {
                unicode: None,
                latex: None,
                error: Some("missing 'source' field".into()),
                svg: None,
            }
        }
    };

    let lowered = ax_core_ir::lower(source, interner);
    if !lowered.errors.is_empty() {
        let msg = lowered
            .errors
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return EvalResponse {
            unicode: None,
            latex: None,
            error: Some(msg),
            svg: None,
        };
    }

    let mut last_unicode = None;
    let mut last_latex = None;
    let mut last_svg = None;

    for expr in &lowered.exprs {
        let rewrite_target = match expr {
            Expr::Call(f, args) if interner.resolve(*f) == "rewrite" && args.len() == 1 => {
                Some(args[0].clone())
            }
            _ => None,
        };

        if let Expr::Import(path) = expr {
            if let Err(err) = handle_import(path, env, interner, search_paths) {
                return EvalResponse {
                    unicode: None,
                    latex: None,
                    error: Some(err.to_string()),
                    svg: None,
                };
            }
            last_unicode = Some(format!(
                "imported {}",
                path.iter()
                    .map(|sym| interner.resolve(*sym))
                    .collect::<Vec<_>>()
                    .join(".")
            ));
            last_latex = None;
            continue;
        }

        if let Some(description) = ax_eval::apply_set_convention(expr, env) {
            last_unicode = Some(format!("active convention: {description}"));
            last_latex = None;
            continue;
        }

        if let Expr::Let(name, val, body) = expr {
            let evaled = ax_eval::eval(val, env, interner);
            env.bindings.insert(*name, evaled.clone());
            let display = if matches!(body.as_ref(), Expr::Sym(s) if *s == *name) {
                evaled
            } else {
                ax_eval::eval(body, env, interner)
            };
            if is_plot_call(body, interner) {
                last_svg = std::fs::read_to_string("axioma_plot.svg").ok();
                last_unicode = Some("plot saved to axioma_plot.svg".to_string());
                last_latex = None;
            } else {
                last_unicode = Some(ax_render::to_unicode(&display, interner));
                last_latex = Some(ax_render::to_latex(&display, interner));
            }
            continue;
        }

        let result = ax_eval::eval(expr, env, interner);

        if let Some(rule_name) = ax_eval::register_rule(&result, env, interner) {
            last_unicode = Some(format!("registered rule: {rule_name}"));
            last_latex = None;
            continue;
        }
        if let Expr::FnDef(name, _, _) = &result {
            env.bindings.insert(*name, result.clone());
            last_unicode = Some(format!("defined {}", interner.resolve(*name)));
            last_latex = None;
            continue;
        }
        if let Expr::Assume(var, assumptions) = &result {
            env.assumptions.entry(*var).or_default().extend(assumptions.clone());
            last_unicode = Some(assume_message(*var, assumptions, interner));
            last_latex = None;
            continue;
        }
        if let Some(message) = ax_eval::apply_grassmann_declaration(&result, env, interner) {
            last_unicode = Some(message);
            last_latex = None;
            continue;
        }
        if let Some(message) = ax_eval::apply_operator_declaration(&result, env, interner) {
            last_unicode = Some(message);
            last_latex = None;
            continue;
        }

        if is_plot_call(expr, interner) {
            last_svg = std::fs::read_to_string("axioma_plot.svg").ok();
            last_unicode = Some("plot saved to axioma_plot.svg".to_string());
            last_latex = None;
        } else {
        last_unicode = Some(ax_render::to_unicode(&result, interner));
        last_latex = Some(ax_render::to_latex(&result, interner));
        if let Some(target) = rewrite_target {
            let (_, trace) = ax_eval::rewrite_with_trace(&target, env, interner);
            let trust = ax_eval::describe_rewrite_trace(&trace);
            last_unicode = Some(match last_unicode.take() {
                Some(text) => format!("{text}\n{trust}"),
                None => trust,
            });
        }
    }
    }

    EvalResponse {
        unicode: last_unicode,
        latex: last_latex,
        error: None,
        svg: last_svg,
    }
}

pub fn start_server(port: u16) -> Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let server = Server::http(&addr).map_err(|e| anyhow!("{e}"))?;
    println!("Axioma notebook running at http://localhost:{port}");

    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let search_paths = default_search_paths();

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        match (method, url.as_str()) {
            (Method::Get, "/") => {
                let html = include_str!("notebook.html");
                let response = Response::from_string(html).with_header(
                    Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                );
                let _ = request.respond(response);
            }
            (Method::Post, "/eval") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = handle_eval(&body, &mut env, &interner, &search_paths);
                let json = serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());
                let response = Response::from_string(json)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            (Method::Post, "/reset") => {
                env = ax_eval::Env::new();
                let response = Response::from_string(r#"{"ok": true}"#)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            _ => {
                let response = Response::from_string("Not Found").with_status_code(404);
                let _ = request.respond(response);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_eval_simple() {
        let interner = ax_ir::Interner::new();
        let mut env = ax_eval::Env::new();
        let search_paths = vec![];
        let result = handle_eval(r#"{"source": "1 + 2"}"#, &mut env, &interner, &search_paths);
        assert!(result.error.is_none());
        assert_eq!(result.unicode.as_deref(), Some("3"));
    }

    #[test]
    fn handle_eval_with_error() {
        let interner = ax_ir::Interner::new();
        let mut env = ax_eval::Env::new();
        let search_paths = vec![];
        let result = handle_eval(r#"{"source": "$$$"}"#, &mut env, &interner, &search_paths);
        assert!(result.error.is_some());
    }
}
