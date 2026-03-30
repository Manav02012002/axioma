use anyhow::Result;
use ax_ir::Expr;
use rustyline::error::ReadlineError;

pub fn run() -> Result<()> {
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let mut editor = rustyline::DefaultEditor::new()?;

    println!("axioma v0.1.0 — type an expression, or :quit to exit");

    loop {
        match editor.readline("axioma> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed == ":quit" || trimmed == ":q" {
                    break;
                }

                let _ = editor.add_history_entry(trimmed);

                let mut input = trimmed.to_string();
                if !input.ends_with(';') {
                    input.push(';');
                }

                let lowered = ax_core_ir::lower(&input, &interner);
                if !lowered.errors.is_empty() {
                    for error in lowered.errors {
                        eprintln!("{}", error.message);
                    }
                    continue;
                }

                if let Some(expr) = lowered.expr {
                    if trimmed.starts_with("let") {
                        if let Expr::Let(name, val, body) = expr {
                            let evaled_val = ax_eval::eval(&val, &env, &interner);
                            env.bindings.insert(name, evaled_val);
                            let result = ax_eval::eval(&body, &env, &interner);
                            println!("{}", ax_render::to_unicode(&result, &interner));
                            println!("  LaTeX: {}", ax_render::to_latex(&result, &interner));
                            continue;
                        }
                    }

                    let result = ax_eval::eval(&expr, &env, &interner);
                    println!("{}", ax_render::to_unicode(&result, &interner));
                    println!("  LaTeX: {}", ax_render::to_latex(&result, &interner));
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}
