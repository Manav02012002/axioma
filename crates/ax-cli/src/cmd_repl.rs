use anyhow::Result;
use rustyline::error::ReadlineError;

pub fn run() -> Result<()> {
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let mut editor = rustyline::DefaultEditor::new()?;
    let search_paths = crate::cmd_run::default_search_paths(None);
    let mut last_expr: Option<ax_ir::Expr> = None;

    println!("axioma v0.1.0 — type an expression, or :quit to exit");

    loop {
        match editor.readline("axioma> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed == ":quit" || trimmed == ":q" {
                    break;
                }
                if let Some(target) = trimmed.strip_prefix(":codegen ").map(str::trim) {
                    let Some(expr) = &last_expr else {
                        eprintln!("no previous expression to generate code for");
                        continue;
                    };
                    let target = match target {
                        "python" => ax_codegen::Target::Python,
                        "rust" => ax_codegen::Target::Rust,
                        "cpp" => ax_codegen::Target::Cpp,
                        other => {
                            eprintln!("unknown codegen target: {other}");
                            continue;
                        }
                    };
                    let params = crate::cmd_codegen::infer_params(expr, &env, &interner);
                    println!(
                        "{}",
                        ax_codegen::generate(expr, target, &interner, None, &params)
                    );
                    continue;
                }

                let _ = editor.add_history_entry(trimmed);

                let lowered = ax_core_ir::lower(trimmed, &interner);
                if !lowered.errors.is_empty() {
                    for error in lowered.errors {
                        eprintln!("{}", error.message);
                    }
                    continue;
                }

                if let Some(expr) = lowered.expr {
                    let candidate_last = match &expr {
                        ax_ir::Expr::Import(_) | ax_ir::Expr::SetConvention(_, _) => None,
                        ax_ir::Expr::Let(name, val, body) => {
                            let evaled = ax_eval::eval(val, &env, &interner);
                            Some(if matches!(body.as_ref(), ax_ir::Expr::Sym(sym) if *sym == *name) {
                                evaled
                            } else {
                                ax_eval::eval(body, &env, &interner)
                            })
                        }
                        _ => {
                            let result = ax_eval::eval(&expr, &env, &interner);
                            match &result {
                                ax_ir::Expr::FnDef(_, _, _)
                                | ax_ir::Expr::Assume(_, _)
                                | ax_ir::Expr::Rule(_, _, _)
                                | ax_ir::Expr::SetConvention(_, _) => None,
                                ax_ir::Expr::Call(f, _) if interner.resolve(*f) == "grassmann" => None,
                                _ => Some(result),
                            }
                        }
                    };
                    if let Some(message) =
                        crate::cmd_run::execute_expr(&expr, &mut env, &interner, &search_paths)?
                    {
                        println!("{message}");
                    }
                    if let Some(expr) = candidate_last {
                        last_expr = Some(expr);
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}
