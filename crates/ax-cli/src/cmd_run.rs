use anyhow::{anyhow, Result};
use ax_ir::Expr;
use std::path::{Path, PathBuf};

fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
    let mut current = start_dir.to_path_buf();
    loop {
        if current.join("axioma.toml").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn qm_settings_from_config(qm: Option<&ax_context::QmSection>) -> ax_eval::QmSettings {
    let Some(qm) = qm else {
        return ax_eval::QmSettings::default();
    };

    ax_eval::QmSettings {
        log_base: qm
            .log_base
            .as_deref()
            .and_then(ax_eval::QmLogBase::from_config_value),
        tensor_product_basis_order: qm
            .tensor_product_basis_order
            .as_deref()
            .and_then(ax_eval::TensorProductBasisOrder::from_config_value),
        gamma_signature: qm
            .gamma_signature
            .as_deref()
            .and_then(ax_eval::GammaSignature::from_config_value),
        clifford_convention: qm
            .clifford_convention
            .as_deref()
            .and_then(ax_eval::CliffordConvention::from_config_value),
        pretty_bra_ket_unicode: qm.pretty_bra_ket_unicode,
        solver_backend: qm.solver_backend.clone(),
        sparse_threshold_dim: qm.sparse_threshold_dim,
        abs_tolerance: qm.abs_tolerance,
        rel_tolerance: qm.rel_tolerance,
    }
}

pub(crate) fn load_project_qm_settings(start_dir: Option<&Path>) -> Result<ax_eval::QmSettings> {
    let start_dir = match start_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?,
    };
    let Some(root) = find_project_root(&start_dir) else {
        return Ok(ax_eval::QmSettings::default());
    };

    let project_paths = ax_context::load_project_paths(Some(root.to_string_lossy().as_ref()))?;
    let config = ax_context::load_config(&project_paths)?;
    Ok(qm_settings_from_config(config.qm.as_ref()))
}

pub fn default_search_paths(current_file: Option<&Path>) -> Vec<PathBuf> {
    let mut search_paths =
        ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
            env_std_path: std::env::var_os("AXIOMA_STD_PATH"),
            working_dir: std::env::current_dir().ok(),
            executable: std::env::current_exe().ok(),
        });

    if let Some(dir) = current_file.and_then(|path| path.parent()) {
        if !search_paths.iter().any(|existing| existing == dir) {
            search_paths.insert(0, dir.to_path_buf());
        }
    }

    search_paths
}

fn assume_message(
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

pub fn handle_import(
    path: &[ax_ir::expr::Sym],
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<()> {
    let import_name = path
        .iter()
        .map(|sym| interner.resolve(*sym))
        .collect::<Vec<_>>()
        .join(".");
    let file_path = ax_eval::resolve_import(path, interner, search_paths)
        .ok_or_else(|| anyhow!("import not found: {import_name}"))?;
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
    nested_search_paths.extend_from_slice(search_paths);

    for expr in lowered.exprs {
        match &expr {
            Expr::Import(import_path) => {
                handle_import(import_path, env, interner, &nested_search_paths)?;
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
            _ if ax_eval::apply_parallel_declaration(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_graded_declaration(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_superspace_setup(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_brst_setup(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_named_operator_declaration(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_named_contraction_declaration(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_property_declaration(&expr, env, interner).is_some() => {}
            _ if ax_eval::apply_index_declaration(&expr, env, interner).is_some() => {}
            _ => {
                let result = ax_eval::eval(&expr, env, interner);
                let _ = ax_eval::apply_coordinate_declaration(&result, env, interner);
                let _ = ax_eval::apply_grassmann_declaration(&result, env, interner);
                let _ = ax_eval::apply_operator_declaration(&result, env, interner);
                let _ = ax_eval::apply_named_operator_declaration(&result, env, interner);
                let _ = ax_eval::apply_named_contraction_declaration(&result, env, interner);
            }
        }
    }

    Ok(())
}

pub fn execute_expr(
    expr: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<Option<String>> {
    let rewrite_target = match expr {
        Expr::Call(f, args) if interner.resolve(*f) == "rewrite" && args.len() == 1 => {
            Some(args[0].clone())
        }
        _ => None,
    };

    if let Expr::Import(path) = expr {
        handle_import(path, env, interner, search_paths)?;
        return Ok(Some(format!(
            "imported {}",
            path.iter()
                .map(|sym| interner.resolve(*sym))
                .collect::<Vec<_>>()
                .join(".")
        )));
    }

    if let Some(description) = ax_eval::apply_set_convention(expr, env) {
        return Ok(Some(format!("active convention: {description}")));
    }
    if let Some(message) = ax_eval::apply_parallel_declaration(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_graded_declaration(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_superspace_setup(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_brst_setup(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_property_declaration(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_coordinate_declaration(expr, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_index_declaration(expr, env, interner) {
        return Ok(Some(message));
    }

    if let Expr::Call(f, args) = expr {
        if interner.resolve(*f) == "__declare_depends" && args.len() == 2 {
            if let (Expr::Sym(tensor), Expr::List(dep_list)) = (&args[0], &args[1]) {
                let deps: Vec<_> = dep_list
                    .iter()
                    .filter_map(|e| if let Expr::Sym(s) = e { Some(*s) } else { None })
                    .collect();
                env.tensor_properties
                    .entry(*tensor)
                    .or_default()
                    .push(ax_ir::TensorProperty::Depends(deps));
                return Ok(Some(format!(
                    "attached Depends to {}",
                    interner.resolve(*tensor)
                )));
            }
        }
    }

    if let Expr::Call(f, args) = expr {
        if interner.resolve(*f) == "__declare_weight" && args.len() >= 2 {
            if let (Expr::Sym(sym), Expr::Int(w)) = (&args[0], &args[1]) {
                let label = args
                    .get(2)
                    .and_then(|e| {
                        if let Expr::Sym(s) = e {
                            Some(interner.resolve(*s).to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                // Convert BigInt to i64 via string parse
                let weight_val: i64 = w.to_string().parse().unwrap_or(0);
                env.weights.insert((*sym, label), weight_val);
                return Ok(Some(format!(
                    "attached weight {} to {}",
                    weight_val,
                    interner.resolve(*sym)
                )));
            }
        }
    }

    if let Expr::Let(name, val, body) = expr {
        let evaled_val = ax_eval::eval(val, env, interner);
        env.bindings.insert(*name, evaled_val.clone());

        let rendered = if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
            evaled_val
        } else {
            ax_eval::eval(body, env, interner)
        };
        println!("{}", ax_render::to_unicode(&rendered, interner));
        println!("  LaTeX: {}", ax_render::to_latex(&rendered, interner));
        return Ok(None);
    }

    let result = ax_eval::eval(expr, env, interner);
    if let Some(rule_name) = ax_eval::register_rule(&result, env, interner) {
        return Ok(Some(format!("registered rule: {rule_name}")));
    }
    if let Expr::Assume(var, assumptions) = &result {
        env.assumptions
            .entry(*var)
            .or_default()
            .extend(assumptions.clone());
        return Ok(Some(assume_message(*var, assumptions, interner)));
    }
    if let Expr::FnDef(name, _, _) = &result {
        env.bindings.insert(*name, result.clone());
        return Ok(Some(format!("defined {}", interner.resolve(*name))));
    }
    if let Some(message) = ax_eval::apply_grassmann_declaration(&result, env, interner) {
        return Ok(Some(message));
    }
    if let Some(message) = ax_eval::apply_operator_declaration(&result, env, interner) {
        return Ok(Some(message));
    }

    println!("{}", ax_render::to_unicode(&result, interner));
    println!("  LaTeX: {}", ax_render::to_latex(&result, interner));
    if let Some(target) = rewrite_target {
        let (_, trace) = ax_eval::rewrite_with_trace(&target, env, interner);
        println!("  {}", ax_eval::describe_rewrite_trace(&trace));
    }
    Ok(None)
}

pub fn run(file: &Path) -> Result<i32> {
    let source = std::fs::read_to_string(file)?;
    let interner = ax_ir::Interner::new();
    let qm_settings = load_project_qm_settings(file.parent())?;
    let mut env = ax_eval::Env::with_qm_settings(qm_settings);
    let lowered = ax_core_ir::lower(&source, &interner);
    let search_paths = default_search_paths(Some(file));

    if !lowered.errors.is_empty() {
        for error in lowered.errors {
            eprintln!("{}", error.message);
        }
        return Ok(2);
    }

    for expr in lowered.exprs {
        if let Some(message) = execute_expr(&expr, &mut env, &interner, &search_paths)? {
            println!("{message}");
        }
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_eval::registry::{
        algorithm_entries, builtin_entries, property_entries, std_modules, syntax_rules,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the unix epoch")
            .as_nanos();
        path.push(format!("axioma-{label}-{nanos}"));
        fs::create_dir_all(&path).expect("temporary directory should be created");
        path
    }

    fn sweep_search_paths() -> Vec<PathBuf> {
        let mut search_paths = default_search_paths(None);
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .map(Path::to_path_buf)
            .expect("ax-cli manifest should live under crates/ax-cli");
        if !search_paths.iter().any(|existing| existing == &repo_root) {
            search_paths.insert(0, repo_root);
        }
        search_paths
    }

    fn execute_code_block(
        code: &str,
        env: &mut ax_eval::Env,
        interner: &ax_ir::Interner,
        search_paths: &[PathBuf],
    ) -> Result<()> {
        let effective_code = if code.contains('\\') || code.contains("_{") || code.contains("^{") {
            ax_core_ir::latex_to_axioma(code)
        } else {
            code.to_string()
        };
        let lowered = ax_core_ir::lower(&effective_code, interner);
        if !lowered.errors.is_empty() {
            let joined = lowered
                .errors
                .iter()
                .map(|error| error.message.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(anyhow!("lowering failed: {joined}"));
        }

        for expr in lowered.exprs {
            execute_expr(&expr, env, interner, search_paths)?;
        }

        Ok(())
    }

    fn assert_examples_execute(
        label: &str,
        examples: impl IntoIterator<Item = (&'static str, &'static str)>,
    ) {
        let interner = ax_ir::Interner::new();
        let search_paths = sweep_search_paths();
        let mut failures = Vec::new();

        for (name, code) in examples {
            let mut env = ax_eval::Env::new();
            if let Err(error) = execute_code_block(code, &mut env, &interner, &search_paths) {
                failures.push(format!("{name}: {error} | code: {code}"));
            }
        }

        assert!(
            failures.is_empty(),
            "{label} failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn builtin_examples_execute_end_to_end() {
        assert_examples_execute(
            "builtin examples",
            builtin_entries()
                .into_iter()
                .map(|entry| (entry.name, entry.example)),
        );
    }

    #[test]
    fn algorithm_examples_execute_end_to_end() {
        assert_examples_execute(
            "algorithm examples",
            algorithm_entries()
                .into_iter()
                .map(|entry| (entry.name, entry.example)),
        );
    }

    #[test]
    fn property_examples_execute_end_to_end() {
        assert_examples_execute(
            "property examples",
            property_entries()
                .into_iter()
                .map(|entry| (entry.name, entry.example)),
        );
    }

    #[test]
    fn syntax_rule_examples_execute_end_to_end() {
        assert_examples_execute(
            "syntax rule examples",
            syntax_rules()
                .into_iter()
                .filter(|entry| !matches!(entry.pattern, "// comment" | "/* comment */"))
                .map(|entry| (entry.pattern, entry.example)),
        );
    }

    #[test]
    fn std_module_imports_execute_end_to_end() {
        let interner = ax_ir::Interner::new();
        let search_paths = sweep_search_paths();
        let mut failures = Vec::new();

        for module in std_modules() {
            let mut env = ax_eval::Env::new();
            let code = format!("import std.{}", module.path.replace('/', "."));
            let lowered = ax_core_ir::lower(&code, &interner);
            if !lowered.errors.is_empty() {
                let joined = lowered
                    .errors
                    .iter()
                    .map(|error| error.message.as_str())
                    .collect::<Vec<_>>()
                    .join("; ");
                failures.push(format!(
                    "{}: lowering failed: {joined} | code: {code}",
                    module.path
                ));
                continue;
            }
            let mut import_ok = false;
            for expr in lowered.exprs {
                match execute_expr(&expr, &mut env, &interner, &search_paths) {
                    Ok(Some(message)) if message.starts_with("imported ") => {
                        import_ok = true;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        failures.push(format!("{}: {error} | code: {code}", module.path));
                        import_ok = false;
                        break;
                    }
                }
            }
            if !import_ok {
                failures.push(format!(
                    "{}: import did not execute as an import command | code: {code}",
                    module.path
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "std module import failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn load_project_qm_settings_populates_evaluator_env() {
        let root = unique_temp_dir("qm-config");
        let nested = root.join("src").join("examples");
        fs::create_dir_all(&nested).expect("nested temp directory should be created");
        fs::write(
            root.join("axioma.toml"),
            r#"[axioma]
version = "0.1.0"

[paths]
spec_dir = "spec"
build_dir = "build"

[qm]
log_base = "2"
tensor_product_basis_order = "left_to_right_lexicographic"
gamma_signature = "mostly_minus"
clifford_convention = "minus_two_g"
pretty_bra_ket_unicode = false
solver_backend = "plugin_sparse"
sparse_threshold_dim = 12
abs_tolerance = 1e-11
rel_tolerance = 1e-8
"#,
        )
        .expect("test axioma.toml should be written");

        let qm = load_project_qm_settings(Some(&nested)).expect("qm settings should load");
        assert_eq!(qm.log_base, Some(ax_eval::QmLogBase::Two));
        assert_eq!(
            qm.tensor_product_basis_order,
            Some(ax_eval::TensorProductBasisOrder::LeftToRightLexicographic)
        );
        assert_eq!(
            qm.gamma_signature,
            Some(ax_eval::GammaSignature::MostlyMinus)
        );
        assert_eq!(
            qm.clifford_convention,
            Some(ax_eval::CliffordConvention::MinusTwoG)
        );
        assert_eq!(qm.pretty_bra_ket_unicode, Some(false));
        assert_eq!(qm.solver_backend.as_deref(), Some("plugin_sparse"));
        assert_eq!(qm.sparse_threshold_dim, Some(12));
        assert_eq!(qm.abs_tolerance, Some(1e-11));
        assert_eq!(qm.rel_tolerance, Some(1e-8));

        let env = ax_eval::Env::with_qm_settings(qm.clone());
        assert_eq!(env.qm, qm);

        fs::remove_dir_all(&root).expect("temporary directory should be removed");
    }
}
