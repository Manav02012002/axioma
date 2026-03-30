use anyhow::Result;
use ax_ir::Expr;
use std::path::Path;

pub fn run(file: &Path) -> Result<i32> {
    let source = std::fs::read_to_string(file)?;
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let lowered = ax_core_ir::lower(&source, &interner);

    if !lowered.errors.is_empty() {
        for error in lowered.errors {
            eprintln!("{}", error.message);
        }
        return Ok(2);
    }

    for expr in lowered.exprs {
        if let Expr::Let(name, val, body) = &expr {
            let evaled_val = ax_eval::eval(val, &env, &interner);
            env.bindings.insert(*name, evaled_val.clone());

            if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                println!("{}", ax_render::to_unicode(&evaled_val, &interner));
                println!("  LaTeX: {}", ax_render::to_latex(&evaled_val, &interner));
            } else {
                let result = ax_eval::eval(body, &env, &interner);
                println!("{}", ax_render::to_unicode(&result, &interner));
                println!("  LaTeX: {}", ax_render::to_latex(&result, &interner));
            }
            continue;
        }

        let result = ax_eval::eval(&expr, &env, &interner);
        println!("{}", ax_render::to_unicode(&result, &interner));
        println!("  LaTeX: {}", ax_render::to_latex(&result, &interner));
    }

    Ok(0)
}
