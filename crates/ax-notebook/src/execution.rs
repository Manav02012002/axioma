use anyhow::{anyhow, Result};
use ax_ir::Expr;
use std::path::{Path, PathBuf};

pub fn import_name(path: &[ax_ir::expr::Sym], interner: &ax_ir::Interner) -> String {
    path.iter()
        .map(|sym| interner.resolve(*sym))
        .collect::<Vec<_>>()
        .join(".")
}

pub fn assume_message(
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

pub fn is_plot_call(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    matches!(expr, Expr::Call(f, _) if interner.resolve(*f) == "plot")
}

pub fn apply_import(
    path: &[ax_ir::expr::Sym],
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<String> {
    let import_name = import_name(path, interner);
    let file_path = ax_eval::resolve_import(path, interner, search_paths)
        .ok_or_else(|| anyhow!(ax_context::format_import_resolution_error(&import_name, search_paths)))?;
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
    for path in search_paths {
        if !nested_search_paths.iter().any(|existing| existing == path) {
            nested_search_paths.push(path.clone());
        }
    }

    for expr in lowered.exprs {
        match &expr {
            Expr::Import(import_path) => {
                let _ = apply_import(import_path, env, interner, &nested_search_paths)?;
            }
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
                env.assumptions
                    .entry(*var)
                    .or_default()
                    .extend(assumptions.clone());
            }
            Expr::SetConvention(_, _) => {
                let _ = ax_eval::apply_set_convention(&expr, env);
            }
            _ => {
                let _ = ax_eval::eval(&expr, env, interner);
            }
        }
    }

    Ok(import_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("axioma-{label}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn apply_import_loads_nested_modules_with_shared_search_path_logic() {
        let root = unique_dir("notebook-import");
        let std_dir = root.join("std");
        std::fs::create_dir_all(&std_dir).expect("create std dir");
        std::fs::write(std_dir.join("inner.ax"), "let y = 7").expect("write inner");
        std::fs::write(std_dir.join("outer.ax"), "import std.inner\nlet x = y").expect("write outer");

        let mut env = ax_eval::Env::new();
        let interner = ax_ir::Interner::new();
        let search_paths = vec![root.clone()];
        let outer = vec![interner.get_or_intern("std"), interner.get_or_intern("outer")];

        let imported = apply_import(&outer, &mut env, &interner, &search_paths).expect("apply import");

        assert_eq!(imported, "std.outer");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        assert_eq!(
            env.lookup(x).map(|expr| ax_render::to_unicode(expr, &interner)),
            Some("7".to_string())
        );
        assert_eq!(
            env.lookup(y).map(|expr| ax_render::to_unicode(expr, &interner)),
            Some("7".to_string())
        );
    }
}
