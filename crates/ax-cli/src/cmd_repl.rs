use anyhow::Result;
use ax_ir::Expr;
use rustyline::error::ReadlineError;

fn convention_lines(env: &ax_eval::Env) -> Vec<String> {
    vec![
        format!("  metric_signature: {:?}", env.convention.metric_signature),
        format!("  riemann_sign: {:?}", env.convention.riemann_sign),
        format!(
            "  ricci_contraction: {:?}",
            env.convention.ricci_contraction
        ),
        format!("  levi_civita_norm: {:?}", env.convention.levi_civita_norm),
        format!("  fourier_sign: {:?}", env.convention.fourier_sign),
    ]
}

fn candidate_last_expr(
    expr: &Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    match expr {
        Expr::Import(_) | Expr::SetConvention(_, _) => None,
        Expr::Let(name, val, body) => {
            let evaled = ax_eval::eval(val, env, interner);
            Some(if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                evaled
            } else {
                ax_eval::eval(body, env, interner)
            })
        }
        _ => {
            let result = ax_eval::eval(expr, env, interner);
            match &result {
                Expr::FnDef(_, _, _)
                | Expr::Assume(_, _)
                | Expr::Rule(_, _, _)
                | Expr::SetConvention(_, _) => None,
                Expr::Call(f, _) if interner.resolve(*f) == "__set_parallel" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "grassmann" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "creation" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "annihilation" => None,
                _ => Some(result),
            }
        }
    }
}

fn trust_for_expr(
    expr: &Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    match expr {
        Expr::Call(f, args) if interner.resolve(*f) == "rewrite" && args.len() == 1 => {
            let (_, trace) = ax_eval::rewrite_with_trace(&args[0], env, interner);
            Some(ax_eval::describe_rewrite_trace(&trace))
        }
        _ => None,
    }
}

fn print_help() {
    println!("Axioma REPL commands:");
    println!("  :quit, :q          Exit");
    println!("  :help, :h          Show this help");
    println!("  :env               Show all bindings");
    println!("  :rules             Show all user-defined rules");
    println!("  :assumptions       Show all assumptions");
    println!("  :convention        Show active convention");
    println!("  :inspect [expr]    Inspect an expression or the last result");
    println!("  :suggest [expr]    Suggest algorithms for an expression or the last result");
    println!("  :pool on           Enable pooled expression storage");
    println!("  :pool off          Disable pooled expression storage");
    println!("  :pool stats        Show pooled unique-node count");
    println!("  :parallel on       Enable parallel tensor canonicalisation");
    println!("  :parallel off      Disable parallel tensor canonicalisation");
    println!("  :codegen python    Generate Python for last result");
    println!("  :codegen rust      Generate Rust for last result");
    println!("  :codegen cpp       Generate C++ for last result");
    println!("  :reset             Clear all bindings and rules");
    println!("  :trust             Show trust level of last result");
}

fn print_env(env: &ax_eval::Env, interner: &ax_ir::Interner) {
    for (sym, val) in &env.bindings {
        println!(
            "  {} = {}",
            interner.resolve(*sym),
            ax_render::to_unicode(val, interner)
        );
    }
}

fn print_rules(env: &ax_eval::Env) {
    for (i, rule) in env.rules.iter().enumerate() {
        println!("  [{}] {} ({:?})", i, rule.name, rule.trust_level);
    }
}

fn print_assumptions(env: &ax_eval::Env, interner: &ax_ir::Interner) {
    for (sym, assumptions) in &env.assumptions {
        let names = assumptions
            .iter()
            .map(|a| format!("{a:?}").to_lowercase())
            .collect::<Vec<_>>();
        println!("  {} is {}", interner.resolve(*sym), names.join(", "));
    }
}

fn inspect_target_expr(
    text: &str,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Result<Expr> {
    let effective_input = if text.contains('\\') || text.contains("_{") || text.contains("^{") {
        ax_core_ir::latex_to_axioma(text)
    } else {
        text.to_string()
    };
    let lowered = ax_core_ir::lower(&effective_input, interner);
    if !lowered.errors.is_empty() {
        return Err(anyhow::anyhow!(
            lowered
                .errors
                .into_iter()
                .map(|error| error.message)
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    let expr = lowered
        .expr
        .ok_or_else(|| anyhow::anyhow!("expected exactly one expression"))?;
    Ok(ax_eval::eval(&expr, env, interner))
}

fn variance_arrow(variance: &str) -> &'static str {
    match variance {
        "up" => "↑",
        "down" => "↓",
        _ => "?",
    }
}

fn print_inspect_result(result: &ax_eval::inspect::InspectResult) {
    println!("Kind: {}", result.kind);
    if result.free_indices.is_empty() {
        println!("Free indices: (none)");
    } else {
        println!(
            "Free indices: {}",
            result
                .free_indices
                .iter()
                .map(|(name, variance)| format!("{name}{}", variance_arrow(variance)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if result.dummy_pairs.is_empty() {
        println!("Dummy pairs: (none)");
    } else {
        println!(
            "Dummy pairs: {}",
            result
                .dummy_pairs
                .iter()
                .map(|(a, b)| format!("({a}, {b})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if result.properties.is_empty() {
        println!("Properties: (none)");
    } else {
        println!(
            "Properties: {}",
            result
                .properties
                .iter()
                .map(|(symbol, props)| format!("{symbol} → [{}]", props.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    println!("Symbols: [{}]", result.symbols.join(", "));
    println!("Node count: {}", result.node_count);
}

fn print_suggest_result(result: &ax_eval::suggest::SuggestResult) {
    println!("Suggested algorithms:");
    for suggestion in &result.suggestions {
        println!("  → {}: {}", suggestion.algorithm, suggestion.reason);
    }
    if result.missing.is_empty() {
        println!();
        println!("Missing properties: (none)");
    } else {
        println!();
        println!("Missing properties:");
        for missing in &result.missing {
            println!("  → {} has no declared properties; consider: {}", missing.symbol, missing.suggestion);
        }
    }
}

pub fn is_complete(input: &str) -> bool {
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut braces = 0i32;
    for ch in input.chars() {
        match ch {
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' => braces += 1,
            '}' => braces -= 1,
            _ => {}
        }
    }
    parens <= 0 && brackets <= 0 && braces <= 0 && !input.trim_end().ends_with('\\')
}

pub fn run() -> Result<()> {
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let mut editor = rustyline::DefaultEditor::new()?;
    let search_paths = crate::cmd_run::default_search_paths(None);
    let history_path = std::env::var("HOME")
        .ok()
        .map(|home| std::path::PathBuf::from(home).join(".axioma_history"));
    let mut last_result: Option<Expr> = None;
    let mut last_trust: Option<String> = None;

    if let Some(path) = &history_path {
        let _ = editor.load_history(path);
    }

    println!("axioma v0.1.0 — type an expression, or :quit to exit");

    loop {
        let mut input = String::new();
        loop {
            let prompt = if input.is_empty() {
                "axioma> "
            } else {
                "   ...> "
            };
            match editor.readline(prompt) {
                Ok(line) => {
                    if input.is_empty() && matches!(line.trim(), ":quit" | ":q") {
                        if let Some(path) = &history_path {
                            let _ = editor.save_history(path);
                        }
                        return Ok(());
                    }
                    if !input.is_empty() {
                        input.push('\n');
                    }
                    input.push_str(&line);
                    if is_complete(&input) {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    input.clear();
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    if let Some(path) = &history_path {
                        let _ = editor.save_history(path);
                    }
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            }
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            ":quit" | ":q" => break,
            ":help" | ":h" => {
                print_help();
                continue;
            }
            ":env" => {
                print_env(&env, &interner);
                continue;
            }
            ":rules" => {
                print_rules(&env);
                continue;
            }
            ":assumptions" => {
                print_assumptions(&env, &interner);
                continue;
            }
            ":convention" => {
                for line in convention_lines(&env) {
                    println!("{line}");
                }
                continue;
            }
            ":pool on" => {
                env.enable_pool();
                println!("Expression pool enabled.");
                continue;
            }
            ":pool off" => {
                env.expr_pool = None;
                println!("Expression pool disabled.");
                continue;
            }
            ":pool stats" => {
                if let Some(pool) = &env.expr_pool {
                    println!("Unique pooled nodes: {}", pool.len());
                } else {
                    println!("Expression pool is disabled.");
                }
                continue;
            }
            ":parallel on" => {
                env.parallel = true;
                println!("Parallel mode enabled.");
                continue;
            }
            ":parallel off" => {
                env.parallel = false;
                println!("Parallel mode disabled.");
                continue;
            }
            s if s.starts_with(":inspect") => {
                let arg = s.strip_prefix(":inspect").unwrap_or_default().trim();
                let expr = if arg.is_empty() {
                    match &last_result {
                        Some(expr) => expr.clone(),
                        None => {
                            eprintln!("No previous result to inspect.");
                            continue;
                        }
                    }
                } else {
                    match inspect_target_expr(arg, &env, &interner) {
                        Ok(expr) => expr,
                        Err(err) => {
                            eprintln!("{err}");
                            continue;
                        }
                    }
                };
                let result = ax_eval::inspect::inspect_expr(&expr, &env, &interner);
                print_inspect_result(&result);
                continue;
            }
            s if s.starts_with(":suggest") => {
                let arg = s.strip_prefix(":suggest").unwrap_or_default().trim();
                let expr = if arg.is_empty() {
                    match &last_result {
                        Some(expr) => expr.clone(),
                        None => {
                            eprintln!("No previous result to analyze.");
                            continue;
                        }
                    }
                } else {
                    match inspect_target_expr(arg, &env, &interner) {
                        Ok(expr) => expr,
                        Err(err) => {
                            eprintln!("{err}");
                            continue;
                        }
                    }
                };
                let result = ax_eval::suggest::suggest_for_expr(&expr, &env, &interner);
                print_suggest_result(&result);
                continue;
            }
            ":reset" => {
                env = ax_eval::Env::new();
                last_result = None;
                last_trust = None;
                println!("Environment reset.");
                continue;
            }
            ":trust" => {
                if let Some(trust) = &last_trust {
                    println!("{trust}");
                } else {
                    eprintln!("No trust information for the last result.");
                }
                continue;
            }
            s if s.starts_with(":codegen ") => {
                if let Some(last) = &last_result {
                    let target = s.strip_prefix(":codegen ").unwrap_or_default().trim();
                    let target = match target {
                        "python" | "py" => ax_codegen::Target::Python,
                        "rust" | "rs" => ax_codegen::Target::Rust,
                        "cpp" | "c++" => ax_codegen::Target::Cpp,
                        _ => {
                            eprintln!("Unknown target: {}", target);
                            continue;
                        }
                    };
                    let params = crate::cmd_codegen::infer_params(last, &env, &interner);
                    let code = ax_codegen::generate(last, target, &interner, None, &params);
                    println!("{code}");
                } else {
                    eprintln!("No previous result to generate code for.");
                }
                continue;
            }
            _ => {}
        }

        let _ = editor.add_history_entry(trimmed);

        let effective_input =
            if trimmed.contains('\\') || trimmed.contains("_{") || trimmed.contains("^{") {
                ax_core_ir::latex_to_axioma(trimmed)
            } else {
                trimmed.to_string()
            };
        let lowered = ax_core_ir::lower(&effective_input, &interner);
        if !lowered.errors.is_empty() {
            for error in lowered.errors {
                eprintln!("{}", error.message);
            }
            continue;
        }

        if let Some(expr) = lowered.expr {
            let candidate_last = candidate_last_expr(&expr, &env, &interner);
            let trust = trust_for_expr(&expr, &env, &interner);
            if let Some(message) =
                crate::cmd_run::execute_expr(&expr, &mut env, &interner, &search_paths)?
            {
                println!("{message}");
            }
            if let Some(expr) = candidate_last {
                last_result = Some(expr);
            }
            last_trust = trust;
        }
    }

    if let Some(path) = &history_path {
        let _ = editor.save_history(path);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_complete;

    #[test]
    fn incomplete_parens() {
        assert!(!is_complete("f(x,"));
        assert!(is_complete("f(x, y)"));
        assert!(!is_complete("[1, 2,"));
        assert!(is_complete("[1, 2, 3]"));
    }

    #[test]
    fn backslash_continuation() {
        assert!(!is_complete("1 + \\"));
        assert!(is_complete("1 + 2"));
    }
}
