use anyhow::{bail, Result};
use std::path::Path;

pub fn run(file: &Path, format: &str) -> Result<i32> {
    let source = std::fs::read_to_string(file)?;
    let interner = ax_ir::Interner::new();
    let lowered = ax_core_ir::lower(&source, &interner);

    if !lowered.errors.is_empty() {
        for error in lowered.errors {
            eprintln!("{}", error.message);
        }
        return Ok(2);
    }

    let env = ax_eval::Env::new();
    for expr in lowered.exprs {
        let result = ax_eval::eval(&expr, &env, &interner);
        match format {
            "latex" => println!("{}", ax_render::to_latex(&result, &interner)),
            "ascii" => println!("{}", ax_ir::pretty_print(&result, &interner)),
            _ => bail!("unknown format"),
        }
    }

    Ok(0)
}
