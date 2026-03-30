use std::path::{Path, PathBuf};

fn hash_file(path: &Path) -> anyhow::Result<blake3::Hash> {
    let bytes = std::fs::read(path)?;
    Ok(blake3::hash(&bytes))
}

pub fn fix(file: &Path, max_iter: usize, diags_json: Option<PathBuf>) -> anyhow::Result<i32> {
    for _ in 0..max_iter {
        let before = hash_file(file)?;
        let code = crate::cmd_fix::run(file, diags_json.clone(), true)?;
        if code == 0 {
            return Ok(0);
        }
        let after = hash_file(file)?;
        if before == after {
            return Ok(2);
        }
    }
    Ok(2)
}
