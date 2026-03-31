use anyhow::{anyhow, Result};
use ax_ir::Expr;
use std::path::Path;

fn collect_syms(expr: &Expr, out: &mut Vec<ax_ir::expr::Sym>) {
    match expr {
        Expr::Sym(sym) => out.push(*sym),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_syms(term, out);
            }
        }
        Expr::Pow(base, exp) => {
            collect_syms(base, out);
            collect_syms(exp, out);
        }
        Expr::Neg(inner) => collect_syms(inner, out),
        Expr::Call(_, args) => {
            for arg in args {
                collect_syms(arg, out);
            }
        }
        Expr::Complex(re, im) => {
            collect_syms(re, out);
            collect_syms(im, out);
        }
        Expr::FnDef(_, _, body) => collect_syms(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_syms(lhs, out);
            collect_syms(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_syms(value, out);
            }
        }
        Expr::Indexed(base, _) => collect_syms(base, out),
        Expr::Let(_, value, body) => {
            collect_syms(value, out);
            collect_syms(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_syms(cell, out);
                }
            }
        }
        _ => {}
    }
}

pub fn infer_params(
    expr: &Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Vec<ax_ir::expr::Sym> {
    let mut syms = Vec::new();
    collect_syms(expr, &mut syms);
    let reserved = ["pi", "e", "i", "inf", "infty", "neg_inf"];
    syms.retain(|sym| env.lookup(*sym).is_none() && !reserved.contains(&interner.resolve(*sym)));
    syms.sort_by_key(|sym| interner.resolve(*sym).to_string());
    syms.dedup();
    syms
}

fn parse_target(target: &str) -> Result<ax_codegen::Target> {
    match target {
        "python" => Ok(ax_codegen::Target::Python),
        "rust" => Ok(ax_codegen::Target::Rust),
        "cpp" => Ok(ax_codegen::Target::Cpp),
        other => Err(anyhow!("unknown codegen target: {other}")),
    }
}

pub fn run(file: &Path, target: &str, fn_name: Option<&str>) -> Result<i32> {
    let source = std::fs::read_to_string(file)?;
    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let lowered = ax_core_ir::lower(&source, &interner);
    let search_paths = crate::cmd_run::default_search_paths(Some(file));

    if !lowered.errors.is_empty() {
        for error in lowered.errors {
            eprintln!("{}", error.message);
        }
        return Ok(2);
    }

    let mut last_expr = None;
    for expr in lowered.exprs {
        if let Expr::Import(path) = &expr {
            crate::cmd_run::handle_import(path, &mut env, &interner, &search_paths)?;
            continue;
        }
        if ax_eval::apply_set_convention(&expr, &mut env).is_some() {
            continue;
        }
        match &expr {
            Expr::Let(name, val, body) => {
                let evaled_val = ax_eval::eval(val, &env, &interner);
                env.bindings.insert(*name, evaled_val.clone());
                last_expr = Some(if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                    evaled_val
                } else {
                    ax_eval::eval(body, &env, &interner)
                });
            }
            _ => {
                let result = ax_eval::eval(&expr, &env, &interner);
                if ax_eval::register_rule(&result, &mut env, &interner).is_some() {
                    continue;
                }
                if let Expr::Assume(var, assumptions) = &result {
                    env.assumptions.entry(*var).or_default().extend(assumptions.clone());
                    continue;
                }
                if let Expr::FnDef(name, _, _) = &result {
                    env.bindings.insert(*name, result.clone());
                    continue;
                }
                if ax_eval::apply_grassmann_declaration(&result, &mut env, &interner).is_some() {
                    continue;
                }
                last_expr = Some(result);
            }
        }
    }

    let expr = last_expr.ok_or_else(|| anyhow!("no expression available for code generation"))?;
    let params = infer_params(&expr, &env, &interner);
    let code = ax_codegen::generate(&expr, parse_target(target)?, &interner, fn_name, &params);
    println!("{code}");
    Ok(0)
}
