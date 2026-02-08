//! Project context discovery + config loading.

#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
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

    /// `[plugins.<id>]` tables
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AxiomaSection {
    pub version: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default, Default)]
pub struct PathsSection {
    #[serde(default = "default_spec_dir")]
    pub spec_dir: String,
    #[serde(default = "default_build_dir")]
    pub build_dir: String,
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
