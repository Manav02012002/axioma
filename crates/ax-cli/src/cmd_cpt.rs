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

pub fn run_parity() -> Result<()> {
    let source = "cpt_parity_report();";
    let (expr, interner) = eval_cpt_source(source)?;
    println!("{}", ax_render::to_unicode(&expr, &interner));
    Ok(())
}

pub fn run_hierarchy(species: &str, lmax: usize, gauge: &str, closure: &str) -> Result<()> {
    let source = match species {
        "neutrino" => format!("neutrino_hierarchy({lmax}, {gauge}, {closure});"),
        "photon" => format!("photon_hierarchy({lmax}, {gauge}, {closure});"),
        other => return Err(anyhow!("unsupported hierarchy species `{other}`")),
    };
    let (expr, interner) = eval_cpt_source(&source)?;
    println!("{}", ax_render::to_unicode(&expr, &interner));
    Ok(())
}

pub fn run_hierarchy_export(
    target: &str,
    species: &str,
    lmax: usize,
    gauge: &str,
    closure: &str,
) -> Result<()> {
    let source = format!("export_hierarchy({target}, {species}, {lmax}, {gauge}, {closure});");
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

    #[test]
    fn parity_output_contains_ma_bertschinger_suite() {
        let (expr, interner) = eval_cpt_source("cpt_parity_report();").unwrap();
        assert!(ax_render::to_unicode(&expr, &interner).contains("ma_bertschinger_scalar_labels"));
    }

    #[test]
    fn hierarchy_output_contains_f_nu_0() {
        let (expr, interner) =
            eval_cpt_source("neutrino_hierarchy(3, newtonian, power_law);").unwrap();
        assert!(ax_render::to_unicode(&expr, &interner).contains("F_nu_0"));
    }

    #[test]
    fn hierarchy_export_contains_class_target() {
        let (expr, interner) =
            eval_cpt_source("export_hierarchy(class_hook, neutrino, 3, newtonian, power_law);")
                .unwrap();
        assert!(ax_render::to_unicode(&expr, &interner).contains("\"target\":\"class\""));
    }
}
