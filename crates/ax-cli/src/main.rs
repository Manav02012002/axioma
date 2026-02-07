use anyhow::{bail, Context, Result};
use ax_diagnostics::Diagnostic;
use jsonschema::JSONSchema;
use serde_json::Value;
use std::{env, fs, path::PathBuf};

fn repo_root_from_env() -> Option<PathBuf> {
    env::var("AXIOMA_ROOT").ok().map(PathBuf::from)
}

fn load_schema() -> Result<Value> {
    let schema_path = match repo_root_from_env() {
        Some(root) => root.join("spec/aas.schema.json"),
        None => PathBuf::from("spec/aas.schema.json"),
    };

    let text = fs::read_to_string(&schema_path)
        .with_context(|| format!("failed to read schema at {}", schema_path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("schema is not valid JSON: {}", schema_path.display()))?;
    Ok(v)
}

fn validate_aas(schema: &Value, doc: &Value) -> Vec<Diagnostic> {
    let compiled = match JSONSchema::options().compile(schema) {
        Ok(c) => c,
        Err(e) => {
            return vec![Diagnostic::error(
                "AXAAS0002",
                format!("failed to compile AAS schema: {e}"),
            )];
        }
    };

    let mut diags = Vec::new();
    if let Err(errors) = compiled.validate(doc) {
        for e in errors {
            diags.push(Diagnostic::error(
                "AXAAS0001",
                format!("AAS validation error at {}: {}", e.instance_path, e),
            ));
        }
    }
    diags
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    if cmd != "validate" {
        bail!("usage: axioma validate <script.json>");
    }

    let path = args.next().context("missing <script.json>")?;
    let text = fs::read_to_string(&path).with_context(|| format!("failed to read {path}"))?;
    let doc: Value =
        serde_json::from_str(&text).with_context(|| format!("invalid JSON in {path}"))?;

    let schema = load_schema()?;
    let diags = validate_aas(&schema, &doc);

    if diags.is_empty() {
        println!("ok: AAS is valid");
        return Ok(());
    }

    let out = serde_json::json!({ "diagnostics": diags });
    println!("{}", serde_json::to_string_pretty(&out)?);
    std::process::exit(1);
}
