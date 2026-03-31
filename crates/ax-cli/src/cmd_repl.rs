use anyhow::Result;
use rustyline::error::ReadlineError;

pub fn run() -> Result<()> {
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let mut editor = rustyline::DefaultEditor::new()?;
    let search_paths = crate::cmd_run::default_search_paths(None);

    println!("axioma v0.1.0 — type an expression, or :quit to exit");

    loop {
        match editor.readline("axioma> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed == ":quit" || trimmed == ":q" {
                    break;
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
                    if let Some(message) =
                        crate::cmd_run::execute_expr(&expr, &mut env, &interner, &search_paths)?
                    {
                        println!("{message}");
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}
