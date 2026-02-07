use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub run_id: String,
    pub axioma_version: String,
    pub schema_hash: String,
    pub script_hash: String,
    pub exit_code: i32,
    pub elapsed_ms: u128,
    pub diagnostics_json: serde_json::Value,
}

fn atomic_write_json(path: &Path, value: &serde_json::Value) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let mut tmp = PathBuf::from(path);
    tmp.set_extension("tmp");

    let mut f = fs::File::create(&tmp)?;
    let bytes = serde_json::to_vec_pretty(value).expect("serialize trace json");
    f.write_all(&bytes)?;
    f.write_all(b"\n")?;
    f.sync_all()?;

    fs::rename(tmp, path)?;
    Ok(())
}

impl TraceReport {
    pub fn write_to_build_dir(&self) -> std::io::Result<PathBuf> {
        let out_path = PathBuf::from("build")
            .join("trace")
            .join(format!("{}.json", self.run_id));

        let v = serde_json::to_value(self).expect("trace to json value");
        atomic_write_json(&out_path, &v)?;
        Ok(out_path)
    }
}
