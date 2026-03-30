use std::path::{Path, PathBuf};

pub fn run(file: &Path, diags_json: Option<PathBuf>) -> anyhow::Result<i32> {
    let src = std::fs::read_to_string(file)?;
    let (_node, diags) = ax_syntax::parser::parse_file(&src);

    if let Some(p) = diags_json {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&diags)?;
        std::fs::write(&p, json)?;
    } else {
        let json = serde_json::to_string_pretty(&diags)?;
        println!("{json}");
    }

    Ok(if diags.is_empty() { 0 } else { 2 })
}
