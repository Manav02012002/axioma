#![forbid(unsafe_code)]

use anyhow::{anyhow, Result};

fn eval_cpt_source(source: &str) -> Result<(ax_ir::Expr, ax_ir::Interner)> {
    let interner = ax_ir::Interner::new();
    let env = ax_eval::Env::new();
    let lowered = ax_core_ir::lower(source, &interner);
    let expr = lowered
        .exprs
        .into_iter()
        .next()
        .or(lowered.expr)
        .ok_or_else(|| anyhow!("no expression parsed from CPT source"))?;
    Ok((ax_eval::eval(&expr, &env, &interner), interner))
}

pub fn run_demo() -> Result<()> {
    let source = "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic));";
    let (expr, interner) = eval_cpt_source(source)?;
    println!("{}", ax_render::to_unicode(&expr, &interner));
    Ok(())
}

pub fn run_export(target: &str) -> Result<()> {
    let source = format!(
        "cpt_export_mode_rhs({target}, frw_background_spec(conformal, flat, 3), cpt_matter(canonical_scalar));"
    );
    let (expr, interner) = eval_cpt_source(&source)?;
    println!("{}", ax_render::to_unicode(&expr, &interner));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_expr_contains_00_constraint() {
        let (expr, interner) = eval_cpt_source(
            "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic));",
        )
        .unwrap();
        let rendered = ax_render::to_unicode(&expr, &interner);
        assert!(rendered.contains("00_constraint"));
    }

    #[test]
    fn export_python_contains_def() {
        let (expr, interner) = eval_cpt_source(
            "cpt_export_mode_rhs(python, frw_background_spec(conformal, flat, 3), cpt_matter(canonical_scalar));",
        )
        .unwrap();
        assert!(ax_render::to_unicode(&expr, &interner).contains("def ms_rhs("));
    }
}
