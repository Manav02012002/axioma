use anyhow::{anyhow, bail, Result};
use std::path::Path;

pub fn run(package: &str, path: Option<&str>, git: Option<&str>) -> Result<()> {
    if git.is_some() {
        println!("git dependencies not yet supported. Use --path instead.");
        return Ok(());
    }

    let Some(dep_path) = path else {
        println!("registry dependencies not yet supported.");
        return Ok(());
    };

    let dep_root = Path::new(dep_path);
    if !dep_root.exists() {
        bail!("dependency path does not exist: {}", dep_root.display());
    }

    let has_ax_files = contains_ax_files(dep_root)?;
    if !has_ax_files {
        bail!(
            "dependency path does not contain any .ax files: {}",
            dep_root.display()
        );
    }

    let config_path = std::env::current_dir()?.join("axioma.toml");
    if !config_path.is_file() {
        bail!("axioma.toml not found in current directory");
    }

    let text = std::fs::read_to_string(&config_path)?;
    let mut doc: toml::Value = toml::from_str(&text)?;
    let table = doc
        .as_table_mut()
        .ok_or_else(|| anyhow!("axioma.toml root must be a table"))?;
    let dependencies = table
        .entry("dependencies")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let deps_table = dependencies
        .as_table_mut()
        .ok_or_else(|| anyhow!("[dependencies] must be a table"))?;

    let mut dep_entry = toml::map::Map::new();
    dep_entry.insert(
        "path".to_string(),
        toml::Value::String(dep_path.to_string()),
    );
    deps_table.insert(package.to_string(), toml::Value::Table(dep_entry));

    std::fs::write(&config_path, toml::to_string_pretty(&doc)?)?;
    println!(
        "installed dependency `{package}` from {}",
        dep_root.display()
    );
    Ok(())
}

fn contains_ax_files(path: &Path) -> Result<bool> {
    if path.is_file() {
        return Ok(path.extension().is_some_and(|ext| ext == "ax"));
    }

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            if contains_ax_files(&entry_path)? {
                return Ok(true);
            }
        } else if entry_path.extension().is_some_and(|ext| ext == "ax") {
            return Ok(true);
        }
    }

    Ok(false)
}
