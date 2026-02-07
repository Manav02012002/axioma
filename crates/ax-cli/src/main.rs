use anyhow::{bail, Context, Result};
use ax_context::load_project_paths;
use ax_diagnostics::Diagnostic;
use ax_trace::TraceReport;
use blake3::Hasher;
use clap::{Parser, Subcommand};
use jsonschema::JSONSchema;
use serde_json::Value;
use std::{fs, time::Instant};

const AXIOMA_VERSION: &str = "0.1.0";

#[derive(Debug, Parser)]
#[command(name = "axioma", version = AXIOMA_VERSION)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate an Axioma ActionScript (AAS) JSON file against the schema.
    Validate {
        /// Path to the AAS JSON script
        script: String,

        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,

        /// Do not write a trace receipt to build/trace/
        #[arg(long)]
        no_trace: bool,
    },

    /// Print resolved project paths as JSON (root/spec/build).
    Paths {
        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,
    },
}

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
    let cli = Cli::parse();

    match cli.cmd {
        Command::Paths { root } => {
            let paths = load_project_paths(root.as_deref())?;
            let out = serde_json::json!({
                "root": paths.root,
                "spec_dir": paths.spec_dir,
                "build_dir": paths.build_dir
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }

        Command::Validate {
            script,
            root,
            no_trace,
        } => {
            let start = Instant::now();
            let paths = load_project_paths(root.as_deref())?;

            let script_bytes =
                fs::read(&script).with_context(|| format!("failed to read {script}"))?;

            let schema_path = paths.spec_dir.join("aas.schema.json");
            let schema_bytes = fs::read(&schema_path)
                .with_context(|| format!("failed to read schema at {}", schema_path.display()))?;

            let schema_hash = blake3_hex(&schema_bytes);
            let script_hash = blake3_hex(&script_bytes);
            let rid = run_id(&schema_hash, &script_hash);

            let schema_json: Value = serde_json::from_slice(&schema_bytes)
                .with_context(|| "schema is not valid JSON")?;
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

            if !no_trace {
                let trace = TraceReport {
                    run_id: rid,
                    axioma_version: AXIOMA_VERSION.to_string(),
                    schema_hash,
                    script_hash,
                    exit_code,
                    elapsed_ms: start.elapsed().as_millis(),
                    diagnostics_json: diagnostics_json.clone(),
                };
                let _ = trace.write_to_build_dir(&paths.build_dir);
            }

            if ok {
                println!("ok: AAS is valid");
                return Ok(());
            }

            println!("{}", serde_json::to_string_pretty(&diagnostics_json)?);
            std::process::exit(1);
        }
    }
}
