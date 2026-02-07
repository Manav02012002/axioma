use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Deserialize)]
pub struct AxiomaToml {
    pub axioma: AxiomaSection,
    pub paths: Option<PathsSection>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AxiomaSection {
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PathsSection {
    pub spec_dir: Option<String>,
    pub build_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectPaths {
    pub root: PathBuf,
    pub spec_dir: PathBuf,
    pub build_dir: PathBuf,
}

fn find_root_from(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        if cur.join("axioma.toml").is_file() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

pub fn resolve_root(cli_root: Option<&str>) -> Result<PathBuf> {
    if let Some(r) = cli_root {
        return Ok(PathBuf::from(r));
    }
    if let Ok(r) = env::var("AXIOMA_ROOT") {
        return Ok(PathBuf::from(r));
    }
    let cwd = env::current_dir().context("failed to get current dir")?;
    find_root_from(&cwd).ok_or_else(|| {
        anyhow::anyhow!("AXROOT0001: project root not found (no axioma.toml found in parents)")
    })
}

pub fn load_project_paths(cli_root: Option<&str>) -> Result<ProjectPaths> {
    let root = resolve_root(cli_root)?;
    let cfg_path = root.join("axioma.toml");
    let txt = fs::read_to_string(&cfg_path)
        .with_context(|| format!("failed to read {}", cfg_path.display()))?;
    let cfg: AxiomaToml =
        toml::from_str(&txt).with_context(|| format!("invalid TOML in {}", cfg_path.display()))?;

    let spec_rel = cfg
        .paths
        .as_ref()
        .and_then(|p| p.spec_dir.clone())
        .unwrap_or_else(|| "spec".to_string());
    let build_rel = cfg
        .paths
        .as_ref()
        .and_then(|p| p.build_dir.clone())
        .unwrap_or_else(|| "build".to_string());

    // sanity: version presence
    if cfg.axioma.version.trim().is_empty() {
        bail!("AXROOT0002: axioma.version is empty in axioma.toml");
    }

    Ok(ProjectPaths {
        root: root.clone(),
        spec_dir: root.join(spec_rel),
        build_dir: root.join(build_rel),
    })
}
