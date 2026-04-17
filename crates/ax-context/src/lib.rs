//! Project context discovery + config loading.

#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct ProjectPaths {
    pub root: PathBuf,
    pub spec_dir: PathBuf,
    pub build_dir: PathBuf,
    pub config_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AxiomaConfig {
    #[serde(default)]
    pub axioma: Option<AxiomaSection>,

    #[serde(default)]
    pub paths: PathsSection,

    #[serde(default)]
    pub dependencies: BTreeMap<String, DependencyConfig>,

    /// `[plugins.<id>]` tables
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginConfig>,

    #[serde(default)]
    pub symmetry: SymmetryConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AxiomaSection {
    pub version: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PathsSection {
    #[serde(default = "default_spec_dir")]
    pub spec_dir: String,
    #[serde(default = "default_build_dir")]
    pub build_dir: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DependencyConfig {
    pub version: Option<String>,
    pub path: Option<String>,
    pub git: Option<String>,
}

fn default_spec_dir() -> String {
    "spec".to_string()
}
fn default_build_dir() -> String {
    "build".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct PluginConfig {
    pub wasm: String,
    #[serde(default)]
    pub allow: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SymmetryConfig {
    #[serde(default)]
    pub default_dimension: Option<usize>,
    #[serde(default = "default_projector_max_terms")]
    pub projector_max_terms: usize,
    #[serde(default = "default_sparse_projector_cache_capacity")]
    pub sparse_projector_cache_capacity: usize,
    #[serde(default)]
    pub render_unicode: bool,
}

impl Default for SymmetryConfig {
    fn default() -> Self {
        Self {
            default_dimension: None,
            projector_max_terms: default_projector_max_terms(),
            sparse_projector_cache_capacity: default_sparse_projector_cache_capacity(),
            render_unicode: false,
        }
    }
}

fn default_projector_max_terms() -> usize {
    256
}

fn default_sparse_projector_cache_capacity() -> usize {
    128
}

pub fn load_project_paths(root_override: Option<&str>) -> Result<ProjectPaths> {
    let root = match root_override {
        Some(r) => PathBuf::from(r),
        None => find_root(std::env::current_dir().context("cwd")?)?,
    };

    let config_path = root.join("axioma.toml");
    if !config_path.exists() {
        bail!("axioma.toml not found at {}", config_path.display());
    }

    // Read config to resolve spec/build dirs.
    let cfg = load_config_from_path(&config_path)?;
    let spec_dir = root.join(cfg.paths.spec_dir);
    let build_dir = root.join(cfg.paths.build_dir);

    Ok(ProjectPaths {
        root,
        spec_dir,
        build_dir,
        config_path,
    })
}

pub fn load_config(paths: &ProjectPaths) -> Result<AxiomaConfig> {
    load_config_from_path(&paths.config_path)
}

#[derive(Debug, Clone, Default)]
pub struct ImportSearchPathConfig {
    pub env_std_path: Option<OsString>,
    pub working_dir: Option<PathBuf>,
    pub executable: Option<PathBuf>,
}

/// Builds the import search path in deterministic precedence order:
/// 1. Explicit `AXIOMA_STD_PATH` override entries, in the order they appear.
/// 2. The explicitly provided working directory for the notebook/kernel process.
/// 3. Executable-relative fallback locations: `<exe dir>`, `<exe dir>/std`,
///    `<exe dir>/..`, `<exe dir>/../std`.
///
/// Every entry is normalized to an absolute canonical directory when possible,
/// non-directories are dropped, and duplicates are removed while preserving
/// the first occurrence.
pub fn build_import_search_paths(config: &ImportSearchPathConfig) -> Vec<PathBuf> {
    let mut search_paths = Vec::new();

    if let Some(env_std_path) = &config.env_std_path {
        for path in std::env::split_paths(env_std_path) {
            push_search_path(&mut search_paths, path);
        }
    }

    if let Some(working_dir) = &config.working_dir {
        push_search_path(&mut search_paths, working_dir.clone());
    }

    if let Some(executable) = &config.executable {
        for candidate in executable_relative_search_paths(executable) {
            push_search_path(&mut search_paths, candidate);
        }
    }

    search_paths
}

pub fn format_import_resolution_error(module_name: &str, search_paths: &[PathBuf]) -> String {
    let searched = if search_paths.is_empty() {
        "  (none)".to_string()
    } else {
        search_paths
            .iter()
            .map(|path| format!("  - {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "import not found: {module_name}\nsearched paths (highest precedence first):\n{searched}"
    )
}

fn executable_relative_search_paths(executable: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = executable.parent() {
        candidates.push(dir.to_path_buf());
        candidates.push(dir.join("std"));
        if let Some(parent) = dir.parent() {
            candidates.push(parent.to_path_buf());
            candidates.push(parent.join("std"));
        }
    }
    candidates
}

fn push_search_path(search_paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    let normalized = normalize_search_path(&candidate);
    if let Some(path) = normalized {
        if !search_paths.iter().any(|existing| existing == &path) {
            search_paths.push(path);
        }
    }
}

fn normalize_search_path(candidate: &Path) -> Option<PathBuf> {
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(candidate)
    };

    if !absolute.is_dir() {
        return None;
    }

    std::fs::canonicalize(&absolute).ok()
}

fn load_config_from_path(path: &Path) -> Result<AxiomaConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let cfg: AxiomaConfig = toml::from_str(&text)
        .with_context(|| format!("failed to parse TOML: {}", path.display()))?;
    Ok(cfg)
}

fn find_root(start: PathBuf) -> Result<PathBuf> {
    let mut cur = start;
    loop {
        let candidate = cur.join("axioma.toml");
        if candidate.exists() {
            return Ok(cur);
        }
        if !cur.pop() {
            bail!("could not find axioma.toml by walking up directories");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn parse_config_with_dependencies() {
        let toml_str = r#"
[axioma]
version = "0.1.0"

[paths]
spec_dir = "spec"
build_dir = "build"

[dependencies]
my-rules = { path = "../my-rules" }
gr-utils = { version = "0.2" }
"#;
        let cfg: AxiomaConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.dependencies.contains_key("my-rules"));
        assert_eq!(
            cfg.dependencies["my-rules"].path.as_deref(),
            Some("../my-rules")
        );
    }

    #[test]
    fn symmetry_config_preserves_projector_max_terms() {
        let toml_str = r#"
[symmetry]
projector_max_terms = 4096
sparse_projector_cache_capacity = 32
render_unicode = true
"#;
        let cfg: AxiomaConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.symmetry.projector_max_terms, 4096);
        assert_eq!(cfg.symmetry.sparse_projector_cache_capacity, 32);
        assert!(cfg.symmetry.render_unicode);
    }

    #[test]
    fn default_import_search_paths_includes_current_dir() {
        let paths = build_import_search_paths(&ImportSearchPathConfig {
            working_dir: Some(std::env::current_dir().expect("cwd")),
            ..ImportSearchPathConfig::default()
        });
        let cwd = std::env::current_dir().expect("cwd");
        assert!(
            paths.iter().any(|path| path == &cwd),
            "expected current dir in search paths: {paths:?}"
        );
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axioma-ax-context-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn import_search_paths_prioritize_env_then_cwd_then_executable() {
        let env_dir = temp_dir("env");
        let cwd_dir = temp_dir("cwd");
        let exe_root = temp_dir("exe-root");
        let bin_dir = exe_root.join("bin");
        let bin_std = bin_dir.join("std");
        let root_std = exe_root.join("std");
        std::fs::create_dir_all(&bin_std).expect("bin/std");
        std::fs::create_dir_all(&root_std).expect("root/std");

        let paths = build_import_search_paths(&ImportSearchPathConfig {
            env_std_path: Some(env_dir.clone().into_os_string()),
            working_dir: Some(cwd_dir.clone()),
            executable: Some(bin_dir.join("axioma-jupyter")),
        });

        assert_eq!(
            paths[0],
            std::fs::canonicalize(&env_dir).expect("env canonical")
        );
        assert_eq!(
            paths[1],
            std::fs::canonicalize(&cwd_dir).expect("cwd canonical")
        );
        assert_eq!(
            paths[2],
            std::fs::canonicalize(&bin_dir).expect("bin canonical")
        );
        assert_eq!(
            paths[3],
            std::fs::canonicalize(&bin_std).expect("bin/std canonical")
        );
        assert_eq!(
            paths[4],
            std::fs::canonicalize(&exe_root).expect("root canonical")
        );
        assert_eq!(
            paths[5],
            std::fs::canonicalize(&root_std).expect("root/std canonical")
        );
    }

    #[test]
    fn import_search_paths_split_env_and_dedup() {
        let shared = temp_dir("shared");
        let env = std::env::join_paths([shared.clone(), shared.clone(), shared.clone().join(".")])
            .expect("join paths");

        let paths = build_import_search_paths(&ImportSearchPathConfig {
            env_std_path: Some(env),
            ..ImportSearchPathConfig::default()
        });

        assert_eq!(paths.len(), 1, "expected duplicate elimination: {paths:?}");
    }

    #[test]
    fn format_import_resolution_error_lists_paths_in_order() {
        let first = temp_dir("first");
        let second = temp_dir("second");
        let rendered = format_import_resolution_error("std.demo", &[first.clone(), second.clone()]);
        assert!(rendered.contains("import not found: std.demo"));
        let first_rendered = format!("- {}", first.display());
        let second_rendered = format!("- {}", second.display());
        let first_index = rendered.find(&first_rendered).expect("first");
        let second_index = rendered.find(&second_rendered).expect("second");
        assert!(
            first_index < second_index,
            "expected precedence order in {rendered}"
        );
    }

    fn write_module(root: &Path, module: &str, source: &str) -> PathBuf {
        let mut module_path = root.to_path_buf();
        for part in module.split('.') {
            module_path.push(part);
        }
        module_path.set_extension("ax");
        if let Some(parent) = module_path.parent() {
            std::fs::create_dir_all(parent).expect("module parent");
        }
        std::fs::write(&module_path, source).expect("write module");
        module_path
    }

    fn resolve_module(module: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
        let interner = ax_ir::Interner::new();
        let parts = module
            .split('.')
            .map(|part| interner.get_or_intern(part))
            .collect::<Vec<_>>();
        ax_eval::resolve_import(&parts, &interner, search_paths)
    }

    #[test]
    fn env_std_path_resolves_module() {
        let env_dir = temp_dir("env-module");
        let module_path = write_module(&env_dir, "std.demo", "let value = 7");
        let search_paths = build_import_search_paths(&ImportSearchPathConfig {
            env_std_path: Some(OsString::from(env_dir.clone())),
            ..ImportSearchPathConfig::default()
        });
        assert_eq!(
            resolve_module("std.demo", &search_paths),
            Some(std::fs::canonicalize(module_path).expect("canonical module"))
        );
    }

    #[test]
    fn cwd_resolves_module() {
        let cwd_dir = temp_dir("cwd-module");
        let module_path = write_module(&cwd_dir, "local.demo", "let value = 9");
        let search_paths = build_import_search_paths(&ImportSearchPathConfig {
            working_dir: Some(cwd_dir),
            ..ImportSearchPathConfig::default()
        });
        assert_eq!(
            resolve_module("local.demo", &search_paths),
            Some(std::fs::canonicalize(module_path).expect("canonical module"))
        );
    }

    #[test]
    fn executable_relative_std_path_resolves_module() {
        let root = temp_dir("exe-module");
        let bin_dir = root.join("bin");
        let std_dir = root.join("std");
        std::fs::create_dir_all(&bin_dir).expect("bin dir");
        std::fs::create_dir_all(&std_dir).expect("std dir");
        let module_path = write_module(&std_dir, "tensor.demo", "let value = 11");
        let search_paths = build_import_search_paths(&ImportSearchPathConfig {
            executable: Some(bin_dir.join("axioma-jupyter")),
            ..ImportSearchPathConfig::default()
        });
        assert_eq!(
            resolve_module("std.tensor.demo", &search_paths),
            Some(std::fs::canonicalize(module_path).expect("canonical module"))
        );
    }
}
