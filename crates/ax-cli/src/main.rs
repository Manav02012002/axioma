use anyhow::{bail, Context, Result};
use ax_context::load_project_paths;
use ax_diagnostics::Diagnostic;
use ax_trace::TraceReport;
use blake3::Hasher;
use jsonschema::JSONSchema;
use serde_json::Value;
use std::{env, fs, time::Instant};

const AXIOMA_VERSION: &str = "0.1.0";

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn run_id(schema_hash_hex: &str, script_hash_hex: &str) -> String {
    let mut h = Hasher::new();
    h.update(b"axioma/");
    h.update(AXIOMA_VERSION.as_bytes());
    h.update(b"\n");
    h.update(schema_hash_hex.as_bytes());
    h.update(b"\n");
    h.update(script_hash_hex.as_bytes());
    h.finalize().to_hex().to_string()
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
    let start = Instant::now();

    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    if cmd != "validate" {
        bail!("usage: axioma validate <script.json>");
    }

    let script_path = args.next().context("missing <script.json>")?;

    // Resolve root/spec/build paths
    let paths = load_project_paths(None)?;

    // Read inputs as bytes for hashing (deterministic)
    let script_bytes =
        fs::read(&script_path).with_context(|| format!("failed to read {script_path}"))?;

    let schema_path = paths.spec_dir.join("aas.schema.json");
    let schema_bytes = fs::read(&schema_path)
        .with_context(|| format!("failed to read schema at {}", schema_path.display()))?;

    let schema_hash = blake3_hex(&schema_bytes);
    let script_hash = blake3_hex(&script_bytes);
    let rid = run_id(&schema_hash, &script_hash);

    // Parse JSON after hashing
    let schema_json: Value =
        serde_json::from_slice(&schema_bytes).with_context(|| "schema is not valid JSON")?;
    let doc: Value =
        serde_json::from_slice(&script_bytes).with_context(|| "invalid JSON in script")?;

    let diags = validate_aas(&schema_json, &doc);
    let ok = diags.is_empty();

    let diagnostics_json = if ok {
        serde_json::json!({ "diagnostics": [] })
    } else {
        serde_json::json!({ "diagnostics": diags })
    };

    let exit_code = if ok { 0 } else { 1 };

    let trace = TraceReport {
        run_id: rid.clone(),
        axioma_version: AXIOMA_VERSION.to_string(),
        schema_hash,
        script_hash,
        exit_code,
        elapsed_ms: start.elapsed().as_millis(),
        diagnostics_json: diagnostics_json.clone(),
    };

    let _ = trace.write_to_build_dir(&paths.build_dir);

    if ok {
        println!("ok: AAS is valid");
        return Ok(());
    }

    println!("{}", serde_json::to_string_pretty(&diagnostics_json)?);
    std::process::exit(1);
}
