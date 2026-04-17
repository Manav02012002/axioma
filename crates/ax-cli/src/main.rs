#![forbid(unsafe_code)]
mod cmd_ai;
mod cmd_ai_apply;
mod cmd_ai_pack;

mod cmd_codegen;
pub mod cmd_docgen;
pub mod cmd_export;
mod cmd_fix;
mod cmd_install;
mod cmd_parse;
mod cmd_render;
mod cmd_repl;
mod cmd_run;
mod cmd_tableau;
use anyhow::{bail, Context, Result};
#[cfg(feature = "plugins")]
use ax_context::load_config;
use ax_context::load_project_paths;
#[cfg(feature = "plugins")]
use ax_plugin_api::{PluginRequest, PluginResponse};
#[cfg(feature = "plugins")]
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
enum AiCmd {
    Fix {
        file: std::path::PathBuf,
        #[arg(long, default_value_t = 8)]
        max_iter: usize,
        #[arg(long)]
        diags_json: Option<std::path::PathBuf>,
    },
    Pack {
        file: std::path::PathBuf,
        #[arg(long, default_value = "build/ai_packet.json")]
        out: std::path::PathBuf,
    },
    Apply {
        file: std::path::PathBuf,
        edits_json: std::path::PathBuf,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        #[arg(long)]
        print: bool,
    },
}

#[derive(Debug, Subcommand)]
enum TableauCmd {
    Render {
        #[arg(long)]
        shape: String,
        #[arg(long)]
        slots: Option<String>,
    },
    Trace {
        #[arg(long)]
        shape: String,
    },
    Canonicalize {
        #[arg(long)]
        shape: String,
        #[arg(long)]
        slots: String,
    },
    Summary {
        #[arg(long)]
        expr: String,
    },
    Character {
        #[arg(long)]
        shape: String,
        #[arg(long)]
        cycle: String,
    },
    Frobenius {
        #[arg(long)]
        shape: String,
    },
}

#[derive(Debug, Subcommand)]

enum Command {
    Ai {
        #[command(subcommand)]
        cmd: AiCmd,
    },

    Fix {
        file: std::path::PathBuf,
        #[arg(long)]
        diags_json: Option<std::path::PathBuf>,
        #[arg(long)]
        apply: bool,
    },

    Parse {
        file: std::path::PathBuf,
        #[arg(long)]
        diags_json: Option<std::path::PathBuf>,
    },
    Render {
        file: std::path::PathBuf,
        #[arg(long, default_value = "latex")]
        format: String,
    },
    Tableau {
        #[command(subcommand)]
        cmd: TableauCmd,
    },
    Export {
        file: std::path::PathBuf,
        #[arg(long, default_value = "latex")]
        format: String,
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        #[arg(long = "no-input")]
        no_input: bool,
        #[arg(long = "no-output")]
        no_output: bool,
        #[arg(long)]
        fragment: bool,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        author: Option<String>,
        #[arg(long = "class", default_value = "article")]
        class: String,
    },
    Codegen {
        file: std::path::PathBuf,
        #[arg(long, default_value = "python")]
        target: String,
        #[arg(long)]
        fn_name: Option<String>,
    },
    Docgen {
        #[arg(long, default_value = "share/axioma-llm-context.md")]
        output: std::path::PathBuf,
    },
    Install {
        /// Package name or path
        package: String,
        #[arg(long)]
        path: Option<String>,
        #[arg(long)]
        git: Option<String>,
    },
    Init,
    Run {
        file: std::path::PathBuf,
    },
    #[cfg(feature = "notebook")]
    Notebook {
        #[arg(long, default_value_t = 8888)]
        port: u16,
    },
    Repl,
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
    #[cfg(feature = "plugins")]
    Plugin {
        #[command(subcommand)]
        cmd: PluginCmd,
    },
}

#[cfg(feature = "plugins")]
#[derive(Debug, Subcommand)]
enum PluginCmd {
    /// List plugins registered in axioma.toml
    List {
        /// Project root override (directory containing axioma.toml)
        #[arg(long)]
        root: Option<String>,
    },

    /// Run a WASM plugin (sandboxed), JSON in/out.
    Run {
        /// Path to plugin .wasm (override registry)
        #[arg(long)]
        wasm: Option<String>,

        /// Plugin id from registry (preferred)
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

        Command::Parse { file, diags_json } => {
            let code = cmd_parse::run(&file, diags_json)?;

            std::process::exit(code);
        }
        Command::Render { file, format } => {
            let code = cmd_render::run(&file, &format)?;
            std::process::exit(code);
        }
        Command::Tableau { cmd } => {
            match cmd {
                TableauCmd::Render { shape, slots } => {
                    print!("{}", cmd_tableau::render(&shape, slots.as_deref())?);
                }
                TableauCmd::Trace { shape } => {
                    print!("{}", cmd_tableau::trace(&shape)?);
                }
                TableauCmd::Canonicalize { shape, slots } => {
                    print!("{}", cmd_tableau::canonicalize(&shape, &slots)?);
                }
                TableauCmd::Summary { expr } => {
                    print!("{}", cmd_tableau::summary(&expr)?);
                }
                TableauCmd::Character { shape, cycle } => {
                    print!("{}", cmd_tableau::character(&shape, &cycle)?);
                }
                TableauCmd::Frobenius { shape } => {
                    print!("{}", cmd_tableau::frobenius(&shape)?);
                }
            }
            Ok(())
        }
        Command::Export {
            file,
            format,
            output,
            no_input,
            no_output,
            fragment,
            title,
            author,
            class,
        } => {
            let format = match format.as_str() {
                "latex" | "tex" => cmd_export::ExportFormat::Latex,
                "html" => cmd_export::ExportFormat::Html,
                other => bail!("unsupported export format: {other}; expected latex or html"),
            };
            let default_extension = match format {
                cmd_export::ExportFormat::Latex => "tex",
                cmd_export::ExportFormat::Html => "html",
            };
            let options = cmd_export::ExportOptions {
                format,
                include_input: !no_input,
                include_output: !no_output,
                standalone: !fragment,
                title,
                author,
                document_class: class,
            };
            let source = fs::read_to_string(&file)
                .with_context(|| format!("failed to read {}", file.display()))?;
            let interner = ax_ir::Interner::new();
            let rendered = cmd_export::export(&source, &options, &interner);
            let output_path = output.unwrap_or_else(|| file.with_extension(default_extension));
            fs::write(&output_path, rendered)
                .with_context(|| format!("failed to write {}", output_path.display()))?;
            println!("wrote {}", output_path.display());
            Ok(())
        }
        Command::Codegen {
            file,
            target,
            fn_name,
        } => {
            let code = cmd_codegen::run(&file, &target, fn_name.as_deref())?;
            std::process::exit(code);
        }
        Command::Docgen { output } => {
            cmd_docgen::run(&output)?;
            Ok(())
        }
        Command::Install { package, path, git } => {
            cmd_install::run(&package, path.as_deref(), git.as_deref())?;
            Ok(())
        }
        Command::Init => {
            let config_path = std::env::current_dir()?.join("axioma.toml");
            if config_path.exists() {
                bail!("axioma.toml already exists at {}", config_path.display());
            }
            std::fs::write(
                &config_path,
                "[axioma]\nversion = \"0.1.0\"\n\n[paths]\nspec_dir = \"spec\"\nbuild_dir = \"build\"\n",
            )?;
            println!("created {}", config_path.display());
            Ok(())
        }
        Command::Run { file } => {
            let code = cmd_run::run(&file)?;
            std::process::exit(code);
        }
        #[cfg(feature = "notebook")]
        Command::Notebook { port } => {
            ax_notebook::start_server(port)?;
            Ok(())
        }
        Command::Repl => {
            cmd_repl::run()?;
            Ok(())
        }
        Command::Fix {
            file,
            diags_json,
            apply,
        } => {
            let code = cmd_fix::run(&file, diags_json, apply)?;
            std::process::exit(code);
        }
        Command::Ai { cmd } => match cmd {
            AiCmd::Fix {
                file,
                max_iter,
                diags_json,
            } => {
                let code = cmd_ai::fix(&file, max_iter, diags_json)?;
                std::process::exit(code);
            }
            AiCmd::Pack { file, out } => {
                cmd_ai_pack::pack(&file, &out, AXIOMA_VERSION)?;
                std::process::exit(0);
            }
            AiCmd::Apply {
                file,
                edits_json,
                out,
                print,
            } => {
                cmd_ai_apply::run(&file, &edits_json, out.as_deref(), print)?;
                std::process::exit(0);
            }
        },

        #[cfg(feature = "plugins")]
        Command::Plugin { cmd } => match cmd {
            PluginCmd::List { root } => {
                let paths = load_project_paths(root.as_deref())?;
                let cfg = load_config(&paths)?;
                let mut items = vec![];
                for (id, pcfg) in cfg.plugins.iter() {
                    items.push(serde_json::json!({
                        "id": id,
                        "wasm": pcfg.wasm,
                        "allow": pcfg.allow,
                    }));
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({ "plugins": items }))?
                );
                Ok(())
            }

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
                let cfg = load_config(&paths)?;

                // Resolve wasm path:
                // - if --wasm provided: use it
                // - else require --plugin and resolve from registry
                let (plugin_id, wasm_path) = match (plugin, wasm) {
                    (Some(pid), Some(wp)) => (pid, std::path::PathBuf::from(wp)),
                    (Some(pid), None) => {
                        let pcfg = cfg
                            .plugins
                            .get(&pid)
                            .with_context(|| format!("plugin not found in axioma.toml: {}", pid))?;
                        (pid, paths.root.join(&pcfg.wasm))
                    }
                    (None, Some(wp)) => {
                        let p = std::path::PathBuf::from(&wp);
                        let pid = p
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("plugin")
                            .to_string();
                        (pid, p)
                    }
                    (None, None) => {
                        bail!("must provide --plugin <id> (registry) or --wasm <path>");
                    }
                };

                let wasm_bytes = fs::read(&wasm_path)
                    .with_context(|| format!("failed to read wasm at {}", wasm_path.display()))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_all_top_level_commands() {
        let cases = [
            vec!["axioma", "init"],
            vec!["axioma", "repl"],
            vec!["axioma", "run", "examples/gr_tutorial.ax"],
            vec!["axioma", "parse", "examples/gr_tutorial.ax"],
            vec!["axioma", "render", "examples/gr_tutorial.ax"],
            vec!["axioma", "tableau", "render", "--shape", "2,1"],
            vec![
                "axioma",
                "tableau",
                "summary",
                "--expr",
                "tableau_symmetry([[2,1]], slots=[[0,1,2]])",
            ],
            vec!["axioma", "export", "examples/gr_tutorial.ax"],
            vec!["axioma", "codegen", "examples/gr_tutorial.ax"],
            vec!["axioma", "docgen"],
            vec!["axioma", "install", "demo"],
            vec!["axioma", "validate", "examples/aas/hello_world.json"],
            vec!["axioma", "paths"],
            vec!["axioma", "fix", "examples/gr_tutorial.ax"],
            vec!["axioma", "ai", "fix", "examples/gr_tutorial.ax"],
            vec!["axioma", "ai", "pack", "examples/gr_tutorial.ax"],
            vec![
                "axioma",
                "ai",
                "apply",
                "examples/gr_tutorial.ax",
                "build/ai_packet.json",
            ],
        ];

        for args in cases {
            let cli = Cli::try_parse_from(args.clone())
                .unwrap_or_else(|error| panic!("failed to parse {:?}: {error}", args));
            match cli.cmd {
                Command::Ai { .. }
                | Command::Fix { .. }
                | Command::Parse { .. }
                | Command::Render { .. }
                | Command::Tableau { .. }
                | Command::Export { .. }
                | Command::Codegen { .. }
                | Command::Docgen { .. }
                | Command::Install { .. }
                | Command::Init
                | Command::Run { .. }
                | Command::Repl
                | Command::Validate { .. }
                | Command::Paths { .. } => {}
                #[cfg(feature = "notebook")]
                Command::Notebook { .. } => {}
                #[cfg(feature = "plugins")]
                Command::Plugin { .. } => {}
            }
        }
    }

    #[test]
    fn cli_parses_optional_feature_commands() {
        #[cfg(feature = "notebook")]
        {
            let cli = Cli::try_parse_from(["axioma", "notebook"]).expect("notebook should parse");
            assert!(matches!(cli.cmd, Command::Notebook { .. }));
        }

        #[cfg(feature = "plugins")]
        {
            let list = Cli::try_parse_from(["axioma", "plugin", "list"])
                .expect("plugin list should parse");
            assert!(matches!(
                list.cmd,
                Command::Plugin {
                    cmd: PluginCmd::List { .. }
                }
            ));

            let run = Cli::try_parse_from([
                "axioma", "plugin", "run", "--plugin", "axp-echo", "--op", "plan", "--args", "{}",
            ])
            .expect("plugin run should parse");
            assert!(matches!(
                run.cmd,
                Command::Plugin {
                    cmd: PluginCmd::Run { .. }
                }
            ));
        }
    }
}
