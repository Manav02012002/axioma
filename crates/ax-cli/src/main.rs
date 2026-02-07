#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use ax_context::load_project_paths;
use ax_plugin_api::{PluginRequest, PluginResponse};
use ax_plugin_host::WasmPlugin;
use ax_trace::TraceReport;
use blake3::Hasher;
use clap::{Parser, Subcommand};
use jsonschema::JSONSchema;
use serde_json::Value;
use std::{fs, time::Instant};

const AXIOMA_VERSION: &str = "0.1.0";

#[derive(Debug, Parser)]
#[command(name = "axioma")]
#[command(about = "Axioma CLI", version = AXIOMA_VERSION)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate an Axioma ActionScript (AAS) JSON file against the schema.
    Validate {
        /// Path to AAS json
        path: String,

        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,

        /// Do not write a trace receipt
        #[arg(long)]
        no_trace: bool,
    },

    /// Print resolved project paths (root/spec/build).
    Paths {
        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,
    },

    /// WASM plugin operations.
    Plugin {
        #[command(subcommand)]
        cmd: PluginCmd,
    },
}

#[derive(Debug, Subcommand)]
enum PluginCmd {
    /// Run a WASM plugin (sandboxed), JSON in/out.
    Run {
        /// Path to plugin .wasm
        #[arg(long)]
        wasm: String,

        /// Plugin id (defaults to wasm file stem)
        #[arg(long)]
        plugin: Option<String>,

        /// Operation name (e.g. "plan", "rewrite", "transform")
        #[arg(long)]
        op: String,

        /// JSON args (inline JSON string)
        #[arg(long)]
        args: String,

        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,

        /// Do not write a trace receipt
        #[arg(long)]
        no_trace: bool,
    },
}

fn blake3_hex(bytes: &[u8]) -> String {
    let mut h = Hasher::new();
    h.update(bytes);
    h.finalize().to_hex().to_string()
}

fn run_id(schema_hash: &str, script_hash: &str) -> String {
    let mut h = Hasher::new();
    h.update(schema_hash.as_bytes());
    h.update(b":");
    h.update(script_hash.as_bytes());
    h.finalize().to_hex().to_string()
}

fn load_schema(schema_path: &std::path::Path) -> Result<Value> {
    let text = fs::read_to_string(schema_path)
        .with_context(|| format!("failed to read schema at {}", schema_path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("schema is not valid JSON: {}", schema_path.display()))?;
    Ok(v)
}

fn main() -> std::process::ExitCode {
    if let Err(e) = real_main() {
        eprintln!("Error: {e:#}");
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::from(0)
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Command::Paths { root } => {
            let paths = load_project_paths(root.as_deref())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": paths.root,
                    "spec": paths.spec_dir,
                    "build": paths.build_dir,
                }))?
            );
            Ok(())
        }

        Command::Validate {
            path,
            root,
            no_trace,
        } => {
            let start = Instant::now();
            let paths = load_project_paths(root.as_deref())?;

            let schema_path = paths.spec_dir.join("aas.schema.json");
            let schema_value = load_schema(&schema_path)?;
            let schema_static: &'static Value = Box::leak(Box::new(schema_value));
            let schema_bytes = serde_json::to_vec(schema_static).context("serialize schema")?;
            let schema_hash = blake3_hex(&schema_bytes);

            let script_path = std::path::PathBuf::from(path);
            let script_text = fs::read_to_string(&script_path)
                .with_context(|| format!("failed to read {}", script_path.display()))?;
            let script_json: Value = serde_json::from_str(&script_text)
                .with_context(|| format!("script is not valid JSON: {}", script_path.display()))?;
            let script_bytes = serde_json::to_vec(&script_json).context("serialize script")?;
            let script_hash = blake3_hex(&script_bytes);

            let rid = run_id(&schema_hash, &script_hash);

            let compiled = JSONSchema::compile(schema_static)?;
            let mut diags = vec![];

            if let Err(errors) = compiled.validate(&script_json) {
                for e in errors {
                    diags.push(serde_json::json!({
                        "code": "AXAAS0001",
                        "level": "error",
                        "message": format!("AAS validation error at {}: {}", e.instance_path, e),
                    }));
                }
            }

            let ok = diags.is_empty();
            let exit_code = if ok { 0 } else { 1 };

            let diagnostics_json = if ok {
                serde_json::json!({ "diagnostics": [] })
            } else {
                serde_json::json!({ "diagnostics": diags })
            };

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
                Ok(())
            } else {
                println!("{}", serde_json::to_string_pretty(&diagnostics_json)?);
                bail!("AAS validation failed")
            }
        }

        Command::Plugin { cmd } => match cmd {
            PluginCmd::Run {
                wasm,
                plugin,
                op,
                args,
                root,
                no_trace,
            } => {
                let start = Instant::now();
                let paths = load_project_paths(root.as_deref())?;

                let wasm_path = std::path::PathBuf::from(&wasm);
                let wasm_bytes = fs::read(&wasm_path)
                    .with_context(|| format!("failed to read wasm at {}", wasm_path.display()))?;

                let plugin_id = plugin.unwrap_or_else(|| {
                    wasm_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("plugin")
                        .to_string()
                });

                let args_json: Value =
                    serde_json::from_str(&args).with_context(|| "invalid --args JSON")?;

                let req = PluginRequest {
                    plugin: plugin_id,
                    op,
                    args: args_json,
                };

                let req_bytes = serde_json::to_vec(&req).context("serialize PluginRequest")?;

                // For plugin runs, treat the wasm as "schema" and the request as "script".
                let schema_hash = blake3_hex(&wasm_bytes);
                let script_hash = blake3_hex(&req_bytes);
                let rid = run_id(&schema_hash, &script_hash);

                let host = WasmPlugin::from_file(&wasm_path)?;
                let resp: PluginResponse = host.call(&req)?;
                let resp_json = serde_json::to_value(&resp).context("response to json")?;
                let exit_code = if resp.ok { 0 } else { 1 };

                if !no_trace {
                    let trace = TraceReport {
                        run_id: rid,
                        axioma_version: AXIOMA_VERSION.to_string(),
                        schema_hash,
                        script_hash,
                        exit_code,
                        elapsed_ms: start.elapsed().as_millis(),
                        diagnostics_json: resp_json.clone(),
                    };
                    let _ = trace.write_to_build_dir(&paths.build_dir);
                }

                println!("{}", serde_json::to_string_pretty(&resp_json)?);

                if resp.ok {
                    Ok(())
                } else {
                    bail!("plugin returned ok=false")
                }
            }
        },
    }
}
