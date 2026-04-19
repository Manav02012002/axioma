use anyhow::{anyhow, Context, Result};
use hmac::{Hmac, Mac};
use ax_notebook::{
    execution::{apply_import, import_name, is_plot_call},
    MimeBundle,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

pub fn symmetry_summary_mime_bundle(
    summary: &ax_ai_proto::SymmetryExplainResponse,
) -> BTreeMap<String, String> {
    let mut bundle = BTreeMap::new();
    bundle.insert("text/plain".to_string(), summary.rendered_ascii.clone());
    let json = serde_json::to_string(&summary.summary).unwrap_or_else(|_| "{}".to_string());
    bundle.insert("application/json".to_string(), json);
    bundle
}

#[derive(Deserialize)]
pub struct ConnectionInfo {
    pub shell_port: u16,
    pub iopub_port: u16,
    pub stdin_port: u16,
    pub control_port: u16,
    pub hb_port: u16,
    pub ip: String,
    pub transport: String,
    pub key: String,
    pub signature_scheme: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StartupConfig {
    connection_file: PathBuf,
    working_dir: PathBuf,
    env_std_path: Option<OsString>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct KernelSpecOptions {
    kernel_name: String,
    display_name: String,
    binary_path: Option<PathBuf>,
    working_dir: Option<PathBuf>,
    std_path: Option<PathBuf>,
    prefix: Option<PathBuf>,
    user: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CliCommand {
    Run { connection_file: PathBuf },
    Install(KernelSpecOptions),
    PrintKernelspec(KernelSpecOptions),
    Help,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelSpec {
    pub argv: Vec<String>,
    pub display_name: String,
    pub language: String,
    #[serde(default)]
    pub env: serde_json::Map<String, Value>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interrupt_mode: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Channel {
    Shell,
    Control,
}

impl Channel {
    fn label(self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Control => "control",
        }
    }
}

#[derive(Clone, Debug)]
struct JupyterMessage {
    identities: Vec<Vec<u8>>,
    header: Value,
    parent_header: Value,
    metadata: Value,
    content: Value,
}

#[derive(Clone, Debug)]
struct DecodedMessage {
    identities: Vec<Vec<u8>>,
    header: Value,
    content_bytes: Vec<u8>,
}

impl DecodedMessage {
    fn msg_type(&self) -> Result<&str, ProtocolError> {
        self.header
            .get("msg_type")
            .and_then(Value::as_str)
            .ok_or(ProtocolError::MissingMsgType)
    }

    fn content_json(&self) -> Result<Value, ProtocolError> {
        serde_json::from_slice(&self.content_bytes)
            .map_err(|err| ProtocolError::InvalidContentJson(err.to_string()))
    }
}

#[derive(Debug)]
enum OutboundTarget {
    Reply(Channel),
    Iopub,
}

#[derive(Debug)]
struct OutboundMessage {
    target: OutboundTarget,
    message: JupyterMessage,
}

#[derive(Debug, Default)]
struct ProcessResult {
    outbound: Vec<OutboundMessage>,
    logs: Vec<String>,
    traces: Vec<String>,
    shutdown: bool,
}

impl ProcessResult {
    fn push_reply(&mut self, channel: Channel, message: JupyterMessage) {
        self.outbound.push(OutboundMessage {
            target: OutboundTarget::Reply(channel),
            message,
        });
    }

    fn push_iopub(&mut self, message: JupyterMessage) {
        self.outbound.push(OutboundMessage {
            target: OutboundTarget::Iopub,
            message,
        });
    }

    fn push_trace(&mut self, event: &str, fields: &[(&str, String)]) {
        self.traces.push(trace_line(event, fields));
    }

    fn extend(&mut self, mut other: ProcessResult) {
        self.outbound.append(&mut other.outbound);
        self.logs.append(&mut other.logs);
        self.traces.append(&mut other.traces);
        self.shutdown |= other.shutdown;
    }
}

#[derive(Debug)]
enum ReceiveError {
    Transport(String),
    Protocol(ProtocolError),
}

#[derive(Debug)]
enum ProtocolError {
    MissingDelimiter,
    IncompleteMessage { frame_count: usize },
    InvalidSignatureFrame(String),
    InvalidSignature,
    InvalidHeaderJson(String),
    InvalidParentHeaderJson(String),
    InvalidMetadataJson(String),
    InvalidContentJson(String),
    MissingMsgType,
    MissingExecuteCode,
    UnsupportedMessageType(String),
}

#[derive(Debug)]
enum HandlerError {
    Execute(String),
    Interrupted,
}

#[derive(Clone, Debug)]
struct StreamOutput {
    name: &'static str,
    text: String,
}

#[derive(Clone, Debug)]
enum KernelOutput {
    Stream(StreamOutput),
    ExecuteResult(MimeBundle),
    DisplayData(MimeBundle),
}

#[derive(Debug, Default)]
struct EvalOutcome {
    outputs: Vec<KernelOutput>,
}

#[derive(Debug)]
struct KernelCatalog {
    builtin_names: Vec<String>,
    algorithm_names: Vec<String>,
    std_modules: Vec<String>,
    module_docs: HashMap<String, (String, String, String)>,
    builtin_docs: HashMap<String, (String, String, String)>,
    algorithm_docs: HashMap<String, (String, String, String)>,
    assumption_names: Vec<String>,
    convention_fields: Vec<String>,
    convention_values: HashMap<String, Vec<String>>,
}

impl KernelCatalog {
    fn new() -> Self {
        let builtin_docs = ax_eval::builtin_entries()
            .into_iter()
            .map(|entry| {
                (
                    entry.name.to_string(),
                    (
                        entry.signature.to_string(),
                        entry.description.to_string(),
                        entry.example.to_string(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let algorithm_docs = ax_eval::algorithm_entries()
            .into_iter()
            .map(|entry| {
                (
                    entry.name.to_string(),
                    (
                        entry.signature.to_string(),
                        entry.description.to_string(),
                        entry.example.to_string(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let module_docs = ax_eval::std_modules()
            .into_iter()
            .map(|entry| {
                (
                    format!("std.{}", entry.path.replace('/', ".")),
                    (
                        entry.path.to_string(),
                        entry.description.to_string(),
                        entry.provides.to_string(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut builtin_names = builtin_docs.keys().cloned().collect::<Vec<_>>();
        builtin_names.sort();
        let mut algorithm_names = algorithm_docs.keys().cloned().collect::<Vec<_>>();
        algorithm_names.sort();
        let mut std_modules = module_docs.keys().cloned().collect::<Vec<_>>();
        std_modules.sort();
        let mut assumption_names = ax_eval::assumption_entries()
            .into_iter()
            .map(|entry| entry.name.to_string())
            .collect::<Vec<_>>();
        assumption_names.sort();
        assumption_names.dedup();
        let mut convention_values = HashMap::new();
        let mut convention_fields = Vec::new();
        for entry in ax_eval::convention_entries() {
            convention_fields.push(entry.field.to_string());
            convention_values.insert(
                entry.field.to_string(),
                entry
                    .options
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(|part| part.to_ascii_lowercase())
                    .collect::<Vec<_>>(),
            );
        }
        convention_fields.sort();
        convention_fields.dedup();

        Self {
            builtin_names,
            algorithm_names,
            std_modules,
            module_docs,
            builtin_docs,
            algorithm_docs,
            assumption_names,
            convention_fields,
            convention_values,
        }
    }

    fn convention_values_for(&self, field: &str) -> &[String] {
        self.convention_values
            .get(field)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn scope_names(&self, env: &ax_eval::Env, interner: &ax_ir::Interner) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut names = Vec::new();

        for sym in env.bindings.keys().copied() {
            let name = interner.resolve(sym).to_string();
            if seen.insert(name.clone()) {
                names.push(name);
            }
        }
        for sym in env.assumptions.keys().copied() {
            let name = interner.resolve(sym).to_string();
            if seen.insert(name.clone()) {
                names.push(name);
            }
        }
        for sym in env.coordinates.iter().copied() {
            let name = interner.resolve(sym).to_string();
            if seen.insert(name.clone()) {
                names.push(name);
            }
        }
        for sym in env.index_families.keys().copied() {
            let name = interner.resolve(sym).to_string();
            if seen.insert(name.clone()) {
                names.push(name);
            }
        }
        for rule in &env.rules {
            if seen.insert(rule.name.clone()) {
                names.push(rule.name.clone());
            }
        }
        for name in &self.builtin_names {
            if seen.insert(name.clone()) {
                names.push(name.clone());
            }
        }
        for name in &self.algorithm_names {
            if seen.insert(name.clone()) {
                names.push(name.clone());
            }
        }
        for keyword in [
            "let",
            "import",
            "module",
            "assume",
            "convention",
            "rule",
            "in",
            "indexset",
        ] {
            let keyword = keyword.to_string();
            if seen.insert(keyword.clone()) {
                names.push(keyword);
            }
        }

        names.sort();
        names
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HistoryEntry {
    session: String,
    line_number: u64,
    code: String,
    output: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Completeness {
    Complete,
    Incomplete,
    Invalid,
}

#[derive(Debug)]
struct PendingExecute {
    channel: Channel,
    message: DecodedMessage,
    code: String,
    execution_count: u64,
    store_history: bool,
    silent: bool,
    cancellation: ax_ir::CancellationToken,
    started_at: Instant,
}

#[derive(Debug)]
enum ExecutionCompletion {
    Finished(Result<(EvalOutcome, ax_eval::Env), HandlerError>),
    Fatal(String),
}

#[derive(Debug)]
enum RuntimeError {
    Send {
        channel: &'static str,
        message: String,
    },
}

fn trace_line(event: &str, fields: &[(&str, String)]) -> String {
    let mut line = format!("event={event}");
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(value);
    }
    line
}

fn message_id(message: &DecodedMessage) -> &str {
    message
        .header
        .get("msg_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDelimiter => write!(f, "missing Jupyter delimiter"),
            Self::IncompleteMessage { frame_count } => {
                write!(f, "incomplete Jupyter message with {frame_count} frames")
            }
            Self::InvalidSignatureFrame(err) => write!(f, "invalid signature frame: {err}"),
            Self::InvalidSignature => write!(f, "invalid Jupyter signature"),
            Self::InvalidHeaderJson(err) => write!(f, "invalid header JSON: {err}"),
            Self::InvalidParentHeaderJson(err) => write!(f, "invalid parent header JSON: {err}"),
            Self::InvalidMetadataJson(err) => write!(f, "invalid metadata JSON: {err}"),
            Self::InvalidContentJson(err) => write!(f, "invalid content JSON: {err}"),
            Self::MissingMsgType => write!(f, "missing msg_type in Jupyter header"),
            Self::MissingExecuteCode => write!(f, "execute_request missing string 'code' field"),
            Self::UnsupportedMessageType(msg_type) => {
                write!(f, "unsupported Jupyter message type: {msg_type}")
            }
        }
    }
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execute(err) => write!(f, "{err}"),
            Self::Interrupted => write!(f, "execution interrupted"),
        }
    }
}

impl std::fmt::Display for ReceiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(err) => write!(f, "transport error: {err}"),
            Self::Protocol(err) => write!(f, "{err}"),
        }
    }
}

fn endpoint(conn: &ConnectionInfo, port: u16) -> String {
    format!("{}://{}:{}", conn.transport, conn.ip, port)
}

fn default_kernelspec_options() -> KernelSpecOptions {
    KernelSpecOptions {
        kernel_name: "axioma".to_string(),
        display_name: "Axioma".to_string(),
        binary_path: None,
        working_dir: None,
        std_path: None,
        prefix: None,
        user: false,
    }
}

fn parse_cli_command(args: &[String]) -> Result<CliCommand> {
    if args.len() <= 1 {
        return Ok(CliCommand::Help);
    }

    match args[1].as_str() {
        "install" => {
            let mut options = default_kernelspec_options();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--user" => {
                        options.user = true;
                        i += 1;
                    }
                    "--prefix" => {
                        let value = args.get(i + 1).ok_or_else(|| anyhow!("missing value for --prefix"))?;
                        options.prefix = Some(PathBuf::from(value));
                        i += 2;
                    }
                    "--name" => {
                        let value = args.get(i + 1).ok_or_else(|| anyhow!("missing value for --name"))?;
                        options.kernel_name = value.clone();
                        i += 2;
                    }
                    "--display-name" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --display-name"))?;
                        options.display_name = value.clone();
                        i += 2;
                    }
                    "--binary" => {
                        let value = args.get(i + 1).ok_or_else(|| anyhow!("missing value for --binary"))?;
                        options.binary_path = Some(PathBuf::from(value));
                        i += 2;
                    }
                    "--working-dir" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --working-dir"))?;
                        options.working_dir = Some(PathBuf::from(value));
                        i += 2;
                    }
                    "--std-path" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --std-path"))?;
                        options.std_path = Some(PathBuf::from(value));
                        i += 2;
                    }
                    other => {
                        return Err(anyhow!("unknown install option: {other}"));
                    }
                }
            }
            Ok(CliCommand::Install(options))
        }
        "print-kernelspec" => {
            let mut options = default_kernelspec_options();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--name" => {
                        let value = args.get(i + 1).ok_or_else(|| anyhow!("missing value for --name"))?;
                        options.kernel_name = value.clone();
                        i += 2;
                    }
                    "--display-name" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --display-name"))?;
                        options.display_name = value.clone();
                        i += 2;
                    }
                    "--binary" => {
                        let value = args.get(i + 1).ok_or_else(|| anyhow!("missing value for --binary"))?;
                        options.binary_path = Some(PathBuf::from(value));
                        i += 2;
                    }
                    "--working-dir" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --working-dir"))?;
                        options.working_dir = Some(PathBuf::from(value));
                        i += 2;
                    }
                    "--std-path" => {
                        let value =
                            args.get(i + 1).ok_or_else(|| anyhow!("missing value for --std-path"))?;
                        options.std_path = Some(PathBuf::from(value));
                        i += 2;
                    }
                    other => return Err(anyhow!("unknown print-kernelspec option: {other}")),
                }
            }
            Ok(CliCommand::PrintKernelspec(options))
        }
        "--help" | "-h" | "help" => Ok(CliCommand::Help),
        connection_file => Ok(CliCommand::Run {
            connection_file: PathBuf::from(connection_file),
        }),
    }
}

fn usage() -> &'static str {
    "Usage:\n  axioma-jupyter <connection-file>\n  axioma-jupyter install [--user] [--prefix DIR] [--name NAME] [--display-name NAME] [--binary PATH] [--working-dir DIR] [--std-path DIR]\n  axioma-jupyter print-kernelspec [--name NAME] [--display-name NAME] [--binary PATH] [--working-dir DIR] [--std-path DIR]"
}

fn resolve_kernel_binary_path(binary_override: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = binary_override {
        return Ok(path);
    }
    std::env::current_exe().context("failed to determine axioma-jupyter binary path")
}

fn build_kernelspec(options: &KernelSpecOptions) -> Result<KernelSpec> {
    let binary = resolve_kernel_binary_path(options.binary_path.clone())?;
    let mut env = serde_json::Map::new();
    if let Some(working_dir) = &options.working_dir {
        env.insert(
            "AXIOMA_JUPYTER_WORKDIR".to_string(),
            Value::String(working_dir.display().to_string()),
        );
    }
    if let Some(std_path) = &options.std_path {
        env.insert(
            "AXIOMA_STD_PATH".to_string(),
            Value::String(std_path.display().to_string()),
        );
    }

    let mut metadata = serde_json::Map::new();
    metadata.insert("debugger".to_string(), Value::Bool(false));
    metadata.insert(
        "axioma".to_string(),
        json!({
            "working_directory": options.working_dir.as_ref().map(|path| path.display().to_string()),
            "std_path": options.std_path.as_ref().map(|path| path.display().to_string()),
            "session_model": "frontend_session",
            "trust_model": "notebook_untrusted_export_sanitized",
        }),
    );

    Ok(KernelSpec {
        argv: vec![
            binary.display().to_string(),
            "{connection_file}".to_string(),
        ],
        display_name: options.display_name.clone(),
        language: "axioma".to_string(),
        env,
        metadata,
        interrupt_mode: Some("message".to_string()),
    })
}

fn user_kernels_dir_from_home(home: &std::path::Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library").join("Jupyter").join("kernels")
    } else if cfg!(target_os = "windows") {
        home.join("AppData").join("Roaming").join("jupyter").join("kernels")
    } else {
        home.join(".local").join("share").join("jupyter").join("kernels")
    }
}

fn kernelspec_install_dir(options: &KernelSpecOptions, home_override: Option<PathBuf>) -> Result<PathBuf> {
    if options.user && options.prefix.is_some() {
        return Err(anyhow!("--user and --prefix are mutually exclusive"));
    }
    let base = if let Some(prefix) = &options.prefix {
        prefix.join("share").join("jupyter").join("kernels")
    } else if options.user {
        let home = if let Some(home) = home_override {
            home
        } else {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow!("HOME is not set; pass --prefix or --binary"))?
        };
        user_kernels_dir_from_home(&home)
    } else {
        std::env::current_dir()
            .context("failed to determine current directory for local kernelspec install")?
            .join(".jupyter")
            .join("kernels")
    };
    Ok(base.join(&options.kernel_name))
}

fn install_kernelspec(options: &KernelSpecOptions) -> Result<PathBuf> {
    let kernelspec = build_kernelspec(options)?;
    let install_dir = kernelspec_install_dir(options, None)?;
    fs::create_dir_all(&install_dir)
        .with_context(|| format!("failed to create kernelspec directory {}", install_dir.display()))?;
    let kernel_json = serde_json::to_string_pretty(&kernelspec)?;
    fs::write(install_dir.join("kernel.json"), kernel_json)
        .with_context(|| format!("failed to write {}", install_dir.join("kernel.json").display()))?;
    Ok(install_dir)
}

fn resolve_startup_config_from_values(
    connection_file: &std::path::Path,
    working_dir_override: Option<OsString>,
    env_std_path: Option<OsString>,
    fallback_working_dir: PathBuf,
) -> StartupConfig {
    let working_dir = working_dir_override
        .map(PathBuf::from)
        .unwrap_or(fallback_working_dir);
    StartupConfig {
        connection_file: connection_file.to_path_buf(),
        working_dir,
        env_std_path,
    }
}

fn resolve_startup_config(connection_file: &std::path::Path) -> Result<StartupConfig> {
    let fallback_working_dir =
        std::env::current_dir().context("failed to determine Jupyter kernel working directory")?;
    Ok(resolve_startup_config_from_values(
        connection_file,
        std::env::var_os("AXIOMA_JUPYTER_WORKDIR"),
        std::env::var_os("AXIOMA_STD_PATH"),
        fallback_working_dir,
    ))
}

fn log_startup_config(config: &StartupConfig, conn: &ConnectionInfo, search_paths: &[PathBuf]) {
    eprintln!(
        "[axioma-jupyter] {}",
        trace_line(
            "startup.connection",
            &[
                (
                    "connection_file",
                    config.connection_file.display().to_string(),
                ),
                ("transport", conn.transport.clone()),
                ("ip", conn.ip.clone()),
                ("shell_port", conn.shell_port.to_string()),
                ("control_port", conn.control_port.to_string()),
                ("iopub_port", conn.iopub_port.to_string()),
                ("stdin_port", conn.stdin_port.to_string()),
                ("hb_port", conn.hb_port.to_string()),
            ],
        )
    );
    eprintln!(
        "[axioma-jupyter] {}",
        trace_line(
            "startup.paths",
            &[
                ("working_dir", config.working_dir.display().to_string()),
                (
                    "std_path",
                    config
                        .env_std_path
                        .as_ref()
                        .map(|path| PathBuf::from(path).display().to_string())
                        .unwrap_or_else(|| "(unset)".to_string()),
                ),
                (
                    "import_search_paths",
                    search_paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
        )
    );
}

fn timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    format!("{secs:.3}")
}

fn compute_hmac(key: &str, frames: &[&[u8]]) -> Result<String> {
    if key.is_empty() {
        return Ok(String::new());
    }
    let mut mac = HmacSha256::new_from_slice(key.as_bytes())
        .map_err(|e| anyhow!("invalid HMAC key: {e}"))?;
    for frame in frames {
        mac.update(frame);
    }
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn validate_auxiliary_frames(parent_header_bytes: &[u8], metadata_bytes: &[u8]) -> Result<(), ReceiveError> {
    let _: Value = serde_json::from_slice(parent_header_bytes).map_err(|err| {
        ReceiveError::Protocol(ProtocolError::InvalidParentHeaderJson(err.to_string()))
    })?;
    let _: Value = serde_json::from_slice(metadata_bytes).map_err(|err| {
        ReceiveError::Protocol(ProtocolError::InvalidMetadataJson(err.to_string()))
    })?;
    Ok(())
}

fn decode_jupyter_message(frames: Vec<Vec<u8>>, key: &str) -> Result<DecodedMessage, ReceiveError> {
    let delimiter = frames
        .iter()
        .position(|frame| frame == b"<IDS|MSG>")
        .ok_or(ReceiveError::Protocol(ProtocolError::MissingDelimiter))?;

    if frames.len() < delimiter + 6 {
        return Err(ReceiveError::Protocol(ProtocolError::IncompleteMessage {
            frame_count: frames.len(),
        }));
    }

    let identities = frames[..delimiter].to_vec();
    let signature = String::from_utf8(frames[delimiter + 1].clone()).map_err(|err| {
        ReceiveError::Protocol(ProtocolError::InvalidSignatureFrame(err.to_string()))
    })?;
    let header_bytes = &frames[delimiter + 2];
    let parent_header_bytes = &frames[delimiter + 3];
    let metadata_bytes = &frames[delimiter + 4];
    let content_bytes = frames[delimiter + 5].clone();

    if !key.is_empty() {
        let expected = compute_hmac(
            key,
            &[
                header_bytes,
                parent_header_bytes,
                metadata_bytes,
                &content_bytes,
            ],
        )
        .map_err(|err| ReceiveError::Transport(err.to_string()))?;
        if expected != signature {
            return Err(ReceiveError::Protocol(ProtocolError::InvalidSignature));
        }
    }

    validate_auxiliary_frames(parent_header_bytes, metadata_bytes)?;

    Ok(DecodedMessage {
        identities,
        header: serde_json::from_slice(header_bytes).map_err(|err| {
            ReceiveError::Protocol(ProtocolError::InvalidHeaderJson(err.to_string()))
        })?,
        content_bytes,
    })
}

fn receive_jupyter_message(socket: &zmq::Socket, key: &str) -> Result<DecodedMessage, ReceiveError> {
    let frames = socket
        .recv_multipart(0)
        .map_err(|err| ReceiveError::Transport(err.to_string()))?;
    decode_jupyter_message(frames, key)
}

fn send_jupyter_message(socket: &zmq::Socket, msg: &JupyterMessage, key: &str) -> Result<()> {
    let header = serde_json::to_vec(&msg.header)?;
    let parent_header = serde_json::to_vec(&msg.parent_header)?;
    let metadata = serde_json::to_vec(&msg.metadata)?;
    let content = serde_json::to_vec(&msg.content)?;
    let signature = compute_hmac(key, &[&header, &parent_header, &metadata, &content])?;

    let mut frames = msg.identities.clone();
    frames.push(b"<IDS|MSG>".to_vec());
    frames.push(signature.into_bytes());
    frames.push(header);
    frames.push(parent_header);
    frames.push(metadata);
    frames.push(content);

    socket.send_multipart(frames, 0)?;
    Ok(())
}

fn make_message(parent: &DecodedMessage, msg_type: &str, content: Value) -> JupyterMessage {
    let session = parent
        .header
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let username = parent
        .header
        .get("username")
        .and_then(Value::as_str)
        .unwrap_or("axioma");

    JupyterMessage {
        identities: parent.identities.clone(),
        header: json!({
            "msg_id": Uuid::new_v4().to_string(),
            "username": username,
            "session": session,
            "date": timestamp(),
            "msg_type": msg_type,
            "version": "5.4"
        }),
        parent_header: parent.header.clone(),
        metadata: json!({}),
        content,
    }
}

fn make_reply(parent: &DecodedMessage, msg_type: &str, content: Value) -> JupyterMessage {
    make_message(parent, msg_type, content)
}

fn make_iopub_message(parent: &DecodedMessage, msg_type: &str, content: Value) -> JupyterMessage {
    let mut msg = make_message(parent, msg_type, content);
    msg.identities = Vec::new();
    msg
}

fn make_output_message(
    parent: &DecodedMessage,
    output: &KernelOutput,
    execution_count: u64,
) -> JupyterMessage {
    match output {
        KernelOutput::Stream(stream) => make_iopub_message(
            parent,
            "stream",
            json!({
                "name": stream.name,
                "text": stream.text,
            }),
        ),
        KernelOutput::ExecuteResult(bundle) => make_iopub_message(
            parent,
            "execute_result",
            json!({
                "execution_count": execution_count,
                "data": bundle.to_jupyter_data(),
                "metadata": {},
            }),
        ),
        KernelOutput::DisplayData(bundle) => make_iopub_message(
            parent,
            "display_data",
            json!({
                "data": bundle.to_jupyter_data(),
                "metadata": {},
            }),
        ),
    }
}

fn make_status(parent: &DecodedMessage, state: &str) -> JupyterMessage {
    make_iopub_message(parent, "status", json!({ "execution_state": state }))
}

fn make_error_message(parent: &DecodedMessage, ename: &str, err: &str) -> JupyterMessage {
    make_iopub_message(
        parent,
        "error",
        json!({
            "ename": ename,
            "evalue": err,
            "traceback": [err],
        }),
    )
}

fn apply_result_side_effects(
    expr: &ax_ir::Expr,
    result: &ax_ir::Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) {
    if let ax_ir::Expr::Let(name, val, body) = expr {
        let evaled = ax_eval::eval(val, env, interner);
        env.bindings.insert(*name, evaled);
        if matches!(body.as_ref(), ax_ir::Expr::Sym(s) if *s == *name) {
            return;
        }
    }

    if let Some(rule_name) = ax_eval::register_rule(result, env, interner) {
        let sym = interner.get_or_intern(&rule_name);
        env.bindings.insert(sym, ax_ir::Expr::Sym(sym));
    }

    if let ax_ir::Expr::FnDef(name, _, _) = result {
        env.bindings.insert(*name, result.clone());
    }

    if let ax_ir::Expr::Assume(var, assumptions) = result {
        env.assumptions
            .entry(*var)
            .or_default()
            .extend(assumptions.clone());
    }

    let _ = ax_eval::apply_grassmann_declaration(result, env, interner);
    let _ = ax_eval::apply_operator_declaration(result, env, interner);
    let _ = ax_eval::apply_set_convention(result, env);
}

fn evaluate_code(
    code: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Result<EvalOutcome, String> {
    let lowered = ax_core_ir::lower(code, interner);
    if !lowered.errors.is_empty() {
        return Err(
            lowered
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    let mut outcome = EvalOutcome::default();
    for expr in &lowered.exprs {
        if let ax_ir::Expr::Import(path) = expr {
            let import_name = import_name(path, interner);
            let _ = apply_import(path, env, interner, search_paths).map_err(|err| err.to_string())?;
            outcome.outputs.push(KernelOutput::Stream(StreamOutput {
                name: "stdout",
                text: format!("imported {import_name}\n"),
            }));
            continue;
        }

        let result = ax_eval::eval(expr, env, interner);
        apply_result_side_effects(expr, &result, env, interner);
        let display = match expr {
            ax_ir::Expr::Let(name, val, body)
                if matches!(body.as_ref(), ax_ir::Expr::Sym(s) if *s == *name) =>
            {
                ax_eval::eval(val, env, interner)
            }
            _ => result,
        };

        let output = match expr {
            ax_ir::Expr::Let(_, _, body) if is_plot_call(body, interner) => {
                std::fs::read_to_string("axioma_plot.svg")
                    .ok()
                    .map(|svg| {
                        KernelOutput::DisplayData(
                            MimeBundle::svg(svg).with_plain("plot saved to axioma_plot.svg"),
                        )
                    })
                    .unwrap_or_else(|| {
                        KernelOutput::ExecuteResult(MimeBundle::plain(
                            "plot saved to axioma_plot.svg",
                        ))
                    })
            }
            _ if is_plot_call(expr, interner) => std::fs::read_to_string("axioma_plot.svg")
                .ok()
                .map(|svg| {
                    KernelOutput::DisplayData(
                        MimeBundle::svg(svg).with_plain("plot saved to axioma_plot.svg"),
                    )
                })
                .unwrap_or_else(|| {
                    KernelOutput::ExecuteResult(MimeBundle::plain(
                        "plot saved to axioma_plot.svg",
                    ))
                }),
            _ => KernelOutput::ExecuteResult(MimeBundle::from_expr(&display, interner)),
        };
        outcome.outputs.push(output);
    }

    Ok(outcome)
}

fn evaluate_code_transactional(
    code: &str,
    env: ax_eval::Env,
    interner: Arc<ax_ir::Interner>,
    search_paths: &[PathBuf],
) -> Result<(EvalOutcome, ax_eval::Env), HandlerError> {
    let mut trial_env = env;
    let cancellation = ax_ir::current_cancellation();
    let result = ax_ir::with_cancellation(cancellation, || {
        panic::catch_unwind(AssertUnwindSafe(|| {
            evaluate_code(code, &mut trial_env, &interner, search_paths)
        }))
    });

    match result {
        Ok(Ok(outcome)) => Ok((outcome, trial_env)),
        Ok(Err(err)) => Err(HandlerError::Execute(err)),
        Err(payload) => match payload.downcast::<ax_ir::ExecutionAbort>() {
            Ok(abort) => match *abort {
                ax_ir::ExecutionAbort::Interrupted => Err(HandlerError::Interrupted),
                ax_ir::ExecutionAbort::TimedOut => {
                    Err(HandlerError::Execute("computation timed out".to_string()))
                }
            },
            Err(payload) => panic::resume_unwind(payload),
        },
    }
}

fn structural_incomplete(input: &str) -> bool {
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut braces = 0i32;
    for ch in input.chars() {
        match ch {
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' => braces += 1,
            '}' => braces -= 1,
            _ => {}
        }
    }
    parens > 0 || brackets > 0 || braces > 0 || input.trim_end().ends_with('\\')
}

fn completeness_for_code(code: &str, interner: &ax_ir::Interner) -> Completeness {
    if structural_incomplete(code) {
        return Completeness::Incomplete;
    }
    let lowered = ax_core_ir::lower(code, interner);
    if lowered.errors.is_empty() {
        Completeness::Complete
    } else {
        Completeness::Invalid
    }
}

fn current_line_prefix(code: &str, cursor_pos: usize) -> &str {
    let safe_pos = cursor_pos.min(code.len());
    let line_start = code[..safe_pos].rfind('\n').map(|idx| idx + 1).unwrap_or(0);
    &code[line_start..safe_pos]
}

fn import_completion_span(code: &str, cursor_pos: usize) -> Option<(usize, usize, String)> {
    let safe_pos = cursor_pos.min(code.len());
    let line_prefix = current_line_prefix(code, safe_pos);
    let trimmed = line_prefix.trim_start();
    let leading = line_prefix.len().saturating_sub(trimmed.len());
    let import_kw = "import ";
    let after_import = trimmed.strip_prefix(import_kw)?;
    let line_start = safe_pos - line_prefix.len();
    let replace_start = line_start + leading + import_kw.len();
    Some((replace_start, safe_pos, after_import.to_string()))
}

fn identifier_span(code: &str, cursor_pos: usize) -> Option<(usize, usize)> {
    let safe_pos = cursor_pos.min(code.len());
    let (tokens, _) = ax_syntax::lexer::lex(code);
    for (kind, span) in tokens {
        if kind == ax_syntax::kind::SyntaxKind::Eof {
            break;
        }
        let is_name = matches!(
            kind,
            ax_syntax::kind::SyntaxKind::Ident
                | ax_syntax::kind::SyntaxKind::KwModule
                | ax_syntax::kind::SyntaxKind::KwImport
                | ax_syntax::kind::SyntaxKind::KwLet
                | ax_syntax::kind::SyntaxKind::KwIn
                | ax_syntax::kind::SyntaxKind::KwIndexset
        );
        if !is_name {
            continue;
        }
        if span.start <= safe_pos && safe_pos <= span.end {
            return Some((span.start, span.end));
        }
    }

    let bytes = code.as_bytes();
    let mut start = safe_pos;
    while start > 0 {
        let ch = bytes[start - 1] as char;
        if ch.is_ascii_alphanumeric() || ch == '_' {
            start -= 1;
        } else {
            break;
        }
    }
    let mut end = safe_pos;
    while end < bytes.len() {
        let ch = bytes[end] as char;
        if ch.is_ascii_alphanumeric() || ch == '_' {
            end += 1;
        } else {
            break;
        }
    }
    (start < end).then_some((start, end))
}

fn identifier_at_cursor(code: &str, cursor_pos: usize) -> Option<(usize, usize, String)> {
    let (start, end) = identifier_span(code, cursor_pos)?;
    Some((start, end, code[start..end].to_string()))
}

fn imported_module_name_at_cursor(code: &str, cursor_pos: usize) -> Option<String> {
    let (start, _, prefix) = import_completion_span(code, cursor_pos)?;
    if start == cursor_pos || prefix.trim().is_empty() {
        return None;
    }
    Some(prefix.trim().to_string())
}

fn completion_matches(
    code: &str,
    cursor_pos: usize,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
    catalog: &KernelCatalog,
) -> (usize, usize, Vec<String>) {
    let safe_pos = cursor_pos.min(code.len());
    let line_prefix = current_line_prefix(code, safe_pos);
    let trimmed = line_prefix.trim_start();
    let mut matches = Vec::new();

    if let Some((start, end, prefix)) = import_completion_span(code, safe_pos) {
        for module in &catalog.std_modules {
            if module.starts_with(&prefix) && module.as_str() != prefix {
                matches.push(module.clone());
            }
        }
        return (start, end, matches);
    }

    if let Some(after_convention) = trimmed.strip_prefix("convention ") {
        let parts = after_convention.split_whitespace().collect::<Vec<_>>();
        let line_start = safe_pos - line_prefix.len();
        let replace_start = line_start + line_prefix.len() - after_convention.len();
        if parts.len() <= 1 && !after_convention.contains(char::is_whitespace) {
            let prefix = after_convention;
            for field in &catalog.convention_fields {
                if field.starts_with(prefix) && field != prefix {
                    matches.push(field.clone());
                }
            }
            return (replace_start, safe_pos, matches);
        }
        if parts.len() >= 2 {
            let field = parts[0];
            let value_prefix = after_convention
                .strip_prefix(field)
                .unwrap_or_default()
                .trim_start();
            let value_start = safe_pos - value_prefix.len();
            for value in catalog.convention_values_for(field) {
                if value.starts_with(value_prefix) && value != value_prefix {
                    matches.push(value.clone());
                }
            }
            return (value_start, safe_pos, matches);
        }
    }

    if let Some(after_assume) = trimmed.strip_prefix("assume ") {
        let parts = after_assume.split_whitespace().collect::<Vec<_>>();
        if parts.len() >= 2 {
            let prefix = parts.last().copied().unwrap_or_default();
            let start = safe_pos - prefix.len();
            for name in &catalog.assumption_names {
                if name.starts_with(prefix) && name != prefix {
                    matches.push(name.clone());
                }
            }
            return (start, safe_pos, matches);
        }
    }

    if let Some((start, end, token)) = identifier_at_cursor(code, safe_pos) {
        for name in catalog.scope_names(env, interner) {
            if name.starts_with(&token) && name != token {
                matches.push(name);
            }
        }
        matches.sort();
        matches.dedup();
        return (start, end, matches);
    }

    (safe_pos, safe_pos, Vec::new())
}

fn assumption_labels(assumptions: &[ax_ir::Assumption]) -> Vec<String> {
    assumptions
        .iter()
        .map(|assumption| format!("{assumption:?}").to_ascii_lowercase())
        .collect()
}

fn inspect_markdown(
    name: &str,
    value: Option<&ax_ir::Expr>,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
    catalog: &KernelCatalog,
) -> Option<MimeBundle> {
    let mut plain = Vec::new();
    let mut markdown = Vec::new();

    plain.push(format!("name: {name}"));
    markdown.push(format!("## `{name}`"));

    if let Some(expr) = value {
        let inspected = ax_eval::inspect::inspect_expr(expr, env, interner);
        let unicode = ax_render::to_unicode(expr, interner);
        let latex = ax_render::to_latex(expr, interner);
        plain.push(format!("value: {unicode}"));
        plain.push(format!("kind: {}", inspected.kind));
        plain.push(format!("symbols: {}", inspected.symbols.join(", ")));
        plain.push(format!("node_count: {}", inspected.node_count));
        markdown.push(format!("**Value:** `{unicode}`"));
        markdown.push(format!("**Kind:** {}", inspected.kind));
        if !latex.is_empty() {
            markdown.push(format!("**LaTeX:** `${latex}$"));
        }
        if !inspected.symbols.is_empty() {
            markdown.push(format!("**Symbols:** {}", inspected.symbols.join(", ")));
        }
        if !inspected.properties.is_empty() {
            let rendered = inspected
                .properties
                .iter()
                .map(|(symbol, props)| format!("{symbol}: {}", props.join(", ")))
                .collect::<Vec<_>>()
                .join("; ");
            plain.push(format!("properties: {rendered}"));
            markdown.push(format!("**Properties:** {rendered}"));
        }
    }

    if let Some(assumptions) = env.assumptions.get(&interner.get_or_intern(name)) {
        let labels = assumption_labels(assumptions);
        if !labels.is_empty() {
            let joined = labels.join(", ");
            plain.push(format!("assumptions: {joined}"));
            markdown.push(format!("**Assumptions:** {joined}"));
        }
    }

    if let Some((signature, description, example)) = catalog
        .builtin_docs
        .get(name)
        .or_else(|| catalog.algorithm_docs.get(name))
    {
        plain.push(format!("signature: {signature}"));
        plain.push(description.clone());
        plain.push(format!("example: {example}"));
        markdown.push(format!("**Signature:** `{signature}`"));
        markdown.push(description.clone());
        markdown.push(format!("**Example:** `{example}`"));
    }

    if let Some((path, description, provides)) = catalog.module_docs.get(name) {
        plain.push(format!("module: {path}"));
        plain.push(description.clone());
        plain.push(format!("provides: {provides}"));
        markdown.push(format!("**Module:** `std.{}`", path.replace('/', ".")));
        markdown.push(description.clone());
        markdown.push(format!("**Provides:** `{provides}`"));
    }

    if plain.len() == 1 {
        return None;
    }

    Some(
        MimeBundle::plain(plain.join("\n"))
            .with_markdown(markdown.join("\n\n"))
            .with_json(json!({
                "name": name,
                "has_value": value.is_some(),
            })),
    )
}

struct KernelRuntime {
    // The main thread owns all mutable kernel state. Worker threads evaluate
    // against cloned state and return a completion for the main thread to
    // reconcile, which keeps protocol state and environment commits coherent.
    env: ax_eval::Env,
    interner: Arc<ax_ir::Interner>,
    search_paths: Arc<[PathBuf]>,
    catalog: KernelCatalog,
    execution_count: u64,
    history: Vec<HistoryEntry>,
    pending_execute: Option<PendingExecute>,
    completion_tx: mpsc::Sender<ExecutionCompletion>,
    completion_rx: mpsc::Receiver<ExecutionCompletion>,
    shutdown_requested: bool,
}

impl KernelRuntime {
    fn new(search_paths: Vec<PathBuf>) -> Self {
        let (completion_tx, completion_rx) = mpsc::channel();
        Self {
            env: ax_eval::Env::new(),
            interner: Arc::new(ax_ir::Interner::new()),
            search_paths: Arc::from(search_paths),
            catalog: KernelCatalog::new(),
            execution_count: 0,
            history: Vec::new(),
            pending_execute: None,
            completion_tx,
            completion_rx,
            shutdown_requested: false,
        }
    }

    fn is_busy(&self) -> bool {
        self.pending_execute.is_some()
    }

    fn maybe_shutdown_after_idle(&self, result: &mut ProcessResult) {
        if self.shutdown_requested && self.pending_execute.is_none() {
            result.shutdown = true;
        }
    }

    fn spawn_execute_worker(
        &self,
        code: String,
        env: ax_eval::Env,
        cancellation: ax_ir::CancellationToken,
    ) {
        let sender = self.completion_tx.clone();
        let interner = Arc::clone(&self.interner);
        let search_paths = Arc::clone(&self.search_paths);
        thread::spawn(move || {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                ax_ir::with_cancellation(Some(cancellation), || {
                    evaluate_code_transactional(&code, env, interner, search_paths.as_ref())
                })
            }));
            let completion = match outcome {
                Ok(result) => ExecutionCompletion::Finished(result),
                Err(payload) => {
                    let message = if let Some(text) = payload.downcast_ref::<&str>() {
                        (*text).to_string()
                    } else if let Some(text) = payload.downcast_ref::<String>() {
                        text.clone()
                    } else {
                        "kernel worker panicked".to_string()
                    };
                    ExecutionCompletion::Fatal(message)
                }
            };
            let _ = sender.send(completion);
        });
    }

    fn launch_execute(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
        code: &str,
        execution_count: u64,
        store_history: bool,
        silent: bool,
    ) {
        let cancellation = ax_ir::CancellationToken::new();
        self.spawn_execute_worker(code.to_string(), self.env.clone(), cancellation.clone());
        self.pending_execute = Some(PendingExecute {
            channel,
            message: message.clone(),
            code: code.to_string(),
            execution_count,
            store_history,
            silent,
            cancellation,
            started_at: Instant::now(),
        });
    }

    fn cancel_active_execution(&mut self) -> bool {
        if let Some(pending) = &self.pending_execute {
            pending.cancellation.cancel();
            true
        } else {
            false
        }
    }

    fn collect_finished_execution(&mut self, completion: ExecutionCompletion) -> ProcessResult {
        let mut result = ProcessResult::default();
        let Some(pending) = self.pending_execute.take() else {
            result.logs.push("received execute completion with no active execution".to_string());
            self.maybe_shutdown_after_idle(&mut result);
            return result;
        };

        match completion {
            ExecutionCompletion::Finished(Ok((outcome, committed_env))) => {
                let elapsed_ms = pending.started_at.elapsed().as_millis();
                self.env = committed_env;
                if pending.store_history {
                    self.execution_count = pending.execution_count;
                    let history_output = outcome.outputs.iter().rev().find_map(|output| match output {
                        KernelOutput::Stream(_) => None,
                        KernelOutput::ExecuteResult(bundle) | KernelOutput::DisplayData(bundle) => {
                            bundle.text_plain().map(str::to_string)
                        }
                    });
                    self.push_history(pending.execution_count, &pending.code, history_output);
                }
                if !pending.silent {
                    for output in &outcome.outputs {
                        result.push_iopub(make_output_message(
                            &pending.message,
                            output,
                            pending.execution_count,
                        ));
                    }
                }
                result.push_reply(
                    pending.channel,
                    make_reply(
                        &pending.message,
                        "execute_reply",
                        json!({
                            "status": "ok",
                            "execution_count": pending.execution_count,
                            "payload": [],
                            "user_expressions": {},
                        }),
                    ),
                );
                result.push_iopub(make_status(&pending.message, "idle"));
                result.push_trace(
                    "execute.completed",
                    &[
                        ("msg_id", message_id(&pending.message).to_string()),
                        ("status", "ok".to_string()),
                        ("execution_count", pending.execution_count.to_string()),
                        ("elapsed_ms", elapsed_ms.to_string()),
                        ("outputs", outcome.outputs.len().to_string()),
                    ],
                );
            }
            ExecutionCompletion::Finished(Err(err)) => {
                let elapsed_ms = pending.started_at.elapsed().as_millis();
                if pending.store_history {
                    self.execution_count = pending.execution_count;
                    self.push_history(pending.execution_count, &pending.code, Some(err.to_string()));
                }
                self.push_execute_handler_error(
                    &mut result,
                    pending.channel,
                    &pending.message,
                    err,
                    pending.execution_count,
                    pending.silent,
                );
                result.push_trace(
                    "execute.completed",
                    &[
                        ("msg_id", message_id(&pending.message).to_string()),
                        ("status", "error".to_string()),
                        ("execution_count", pending.execution_count.to_string()),
                        ("elapsed_ms", elapsed_ms.to_string()),
                    ],
                );
            }
            ExecutionCompletion::Fatal(message) => {
                let elapsed_ms = pending.started_at.elapsed().as_millis();
                result.logs.push(format!("kernel worker fatal failure: {message}"));
                result.push_iopub(make_error_message(
                    &pending.message,
                    "KernelFatalError",
                    &message,
                ));
                result.push_reply(
                    pending.channel,
                    make_reply(
                        &pending.message,
                        "execute_reply",
                        json!({
                            "status": "error",
                            "execution_count": pending.execution_count,
                            "ename": "KernelFatalError",
                            "evalue": message,
                            "traceback": [message],
                        }),
                    ),
                );
                result.push_iopub(make_status(&pending.message, "idle"));
                result.shutdown = true;
                result.push_trace(
                    "execute.completed",
                    &[
                        ("msg_id", message_id(&pending.message).to_string()),
                        ("status", "fatal".to_string()),
                        ("execution_count", pending.execution_count.to_string()),
                        ("elapsed_ms", elapsed_ms.to_string()),
                    ],
                );
            }
        }

        self.maybe_shutdown_after_idle(&mut result);
        result
    }

    fn try_collect_execution_result(&mut self) -> Option<ProcessResult> {
        match self.completion_rx.try_recv() {
            Ok(completion) => Some(self.collect_finished_execution(completion)),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                let mut result = ProcessResult::default();
                result.logs.push("kernel worker channel disconnected".to_string());
                result.shutdown = true;
                Some(result)
            }
        }
    }

    #[cfg(test)]
    fn wait_for_execution_result(&mut self) -> ProcessResult {
        match self
            .completion_rx
            .recv_timeout(std::time::Duration::from_secs(30))
        {
            Ok(completion) => self.collect_finished_execution(completion),
            Err(err) => {
                let mut result = ProcessResult::default();
                result.logs.push(format!("timed out waiting for execute completion: {err}"));
                result
            }
        }
    }

    #[cfg(test)]
    fn process_frames(&mut self, channel: Channel, frames: Vec<Vec<u8>>, key: &str) -> ProcessResult {
        match decode_jupyter_message(frames, key) {
            Ok(message) => {
                let mut result = self.process_message(channel, &message);
                if self.is_busy() && matches!(message.msg_type(), Ok("execute_request")) {
                    result.extend(self.wait_for_execution_result());
                }
                result
            }
            Err(err) => {
                let mut result = ProcessResult::default();
                result
                    .logs
                    .push(format!("{} channel receive failure: {err}", channel.label()));
                result
            }
        }
    }

    #[cfg(test)]
    fn execute_with_cancellation(
        &mut self,
        code: &str,
        cancellation: ax_ir::CancellationToken,
    ) -> Result<(EvalOutcome, ax_eval::Env), HandlerError> {
        ax_ir::with_cancellation(Some(cancellation), || {
            evaluate_code_transactional(
                code,
                self.env.clone(),
                Arc::clone(&self.interner),
                self.search_paths.as_ref(),
            )
        })
    }

    #[cfg(test)]
    fn process_frames_async(
        &mut self,
        channel: Channel,
        frames: Vec<Vec<u8>>,
        key: &str,
    ) -> ProcessResult {
        match decode_jupyter_message(frames, key) {
            Ok(message) => self.process_message(channel, &message),
            Err(err) => {
                let mut result = ProcessResult::default();
                result
                    .logs
                    .push(format!("{} channel receive failure: {err}", channel.label()));
                result
            }
        }
    }

    fn process_message(&mut self, channel: Channel, message: &DecodedMessage) -> ProcessResult {
        // The kernel intentionally implements the core interactive notebook
        // request set. Advanced protocol areas such as comms/debugger/stdin
        // remain out of scope until they are wired end to end.
        let mut result = ProcessResult::default();
        let msg_type = match message.msg_type() {
            Ok(msg_type) => {
                result.push_trace(
                    "request.received",
                    &[
                        ("channel", channel.label().to_string()),
                        ("msg_type", msg_type.to_string()),
                        ("msg_id", message_id(message).to_string()),
                        ("busy", self.is_busy().to_string()),
                    ],
                );
                msg_type
            }
            Err(err) => {
                result.logs.push(format!(
                    "{} channel protocol error: {err}",
                    channel.label()
                ));
                return result;
            }
        };

        let handled = match msg_type {
            "kernel_info_request" => self.process_kernel_info(channel, message),
            "execute_request" => self.process_execute_request(channel, message),
            "interrupt_request" => self.process_interrupt_request(channel, message),
            "complete_request" => self.process_complete_request(channel, message),
            "inspect_request" => self.process_inspect_request(channel, message),
            "history_request" => self.process_history_request(channel, message),
            "is_complete_request" => self.process_is_complete_request(channel, message),
            "shutdown_request" => self.process_shutdown(channel, message),
            other => {
                let mut handled = ProcessResult::default();
                handled.logs.push(format!(
                    "{} channel unsupported message type: {}",
                    channel.label(),
                    ProtocolError::UnsupportedMessageType(other.to_string())
                ));
                handled
            }
        };
        result.extend(handled);
        if msg_type != "execute_request" || !self.is_busy() {
            result.push_trace(
                "request.completed",
                &[
                    ("channel", channel.label().to_string()),
                    ("msg_type", msg_type.to_string()),
                    ("msg_id", message_id(message).to_string()),
                    ("busy", self.is_busy().to_string()),
                ],
            );
        }
        result
    }

    fn process_kernel_info(&mut self, channel: Channel, message: &DecodedMessage) -> ProcessResult {
        let mut result = ProcessResult::default();
        result.push_reply(
            channel,
            make_reply(
                message,
                "kernel_info_reply",
                json!({
                    "protocol_version": "5.4",
                    "implementation": "axioma",
                    "implementation_version": "0.1.0",
                    "language_info": {
                        "name": "axioma",
                        "version": "0.1.0",
                        "mimetype": "text/x-axioma",
                        "file_extension": ".ax",
                        "codemirror_mode": "text",
                        "pygments_lexer": "text"
                    },
                    "banner": "Axioma — Scientific Computing Language for Physicists",
                    "help_links": [
                        {
                            "text": "Axioma Repository",
                            "url": "https://github.com/Manav02012002/axioma"
                        }
                    ],
                    "status": "ok"
                }),
            ),
        );
        result
    }

    fn push_history(&mut self, line_number: u64, code: &str, output: Option<String>) {
        self.history.push(HistoryEntry {
            session: "axioma".to_string(),
            line_number,
            code: code.to_string(),
            output,
        });
        if self.history.len() > 1000 {
            let excess = self.history.len() - 1000;
            self.history.drain(0..excess);
        }
    }

    fn process_shutdown(&mut self, channel: Channel, message: &DecodedMessage) -> ProcessResult {
        let restart = message
            .content_json()
            .ok()
            .and_then(|content| content.get("restart").and_then(Value::as_bool))
            .unwrap_or(false);
        let mut result = ProcessResult::default();
        result.push_reply(
            channel,
            make_reply(
                message,
                "shutdown_reply",
                json!({ "restart": restart, "status": "ok" }),
            ),
        );
        self.shutdown_requested = true;
        let cancelled_active = self.cancel_active_execution();
        result.push_trace(
            "request.shutdown",
            &[
                ("channel", channel.label().to_string()),
                ("msg_id", message_id(message).to_string()),
                ("restart", restart.to_string()),
                ("cancelled_active", cancelled_active.to_string()),
            ],
        );
        self.maybe_shutdown_after_idle(&mut result);
        result
    }

    fn process_interrupt_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        let interrupted = self.cancel_active_execution();
        result.push_trace(
            "request.interrupt",
            &[
                ("channel", channel.label().to_string()),
                ("msg_id", message_id(message).to_string()),
                ("interrupted", interrupted.to_string()),
            ],
        );
        result.push_reply(
            channel,
            make_reply(
                message,
                "interrupt_reply",
                json!({ "status": "ok" }),
            ),
        );
        result
    }

    fn process_complete_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        match message.content_json() {
            Ok(content) => {
                let code = content
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let cursor_pos = content
                    .get("cursor_pos")
                    .and_then(Value::as_u64)
                    .map(|pos| pos as usize)
                    .unwrap_or(code.len())
                    .min(code.len());
                let (cursor_start, cursor_end, matches) =
                    completion_matches(code, cursor_pos, &self.env, &self.interner, &self.catalog);
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "complete_reply",
                        json!({
                            "status": "ok",
                            "matches": matches,
                            "cursor_start": cursor_start,
                            "cursor_end": cursor_end,
                            "metadata": {},
                        }),
                    ),
                );
            }
            Err(err) => {
                let rendered = err.to_string();
                result.logs.push(format!(
                    "{} channel complete protocol failure: {rendered}",
                    channel.label()
                ));
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "complete_reply",
                        json!({
                            "status": "error",
                            "matches": [],
                            "cursor_start": 0,
                            "cursor_end": 0,
                            "metadata": {},
                            "ename": "ProtocolError",
                            "evalue": rendered,
                            "traceback": [rendered],
                        }),
                    ),
                );
            }
        }
        result
    }

    fn process_inspect_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        match message.content_json() {
            Ok(content) => {
                let code = content
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let cursor_pos = content
                    .get("cursor_pos")
                    .and_then(Value::as_u64)
                    .map(|pos| pos as usize)
                    .unwrap_or(code.len())
                    .min(code.len());
                let mut bundle = imported_module_name_at_cursor(code, cursor_pos)
                    .and_then(|module_name| {
                        inspect_markdown(&module_name, None, &self.env, &self.interner, &self.catalog)
                    });

                if bundle.is_none() {
                    if let Some((_, _, name)) = identifier_at_cursor(code, cursor_pos) {
                        let sym = self.interner.get_or_intern(&name);
                        let value = self.env.lookup(sym);
                        bundle = inspect_markdown(&name, value, &self.env, &self.interner, &self.catalog);
                    }
                }

                let found = bundle.is_some();
                let data = bundle
                    .take()
                    .map(|bundle| bundle.to_jupyter_data())
                    .unwrap_or_default();
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "inspect_reply",
                        json!({
                            "status": "ok",
                            "found": found,
                            "data": data,
                            "metadata": {},
                        }),
                    ),
                );
            }
            Err(err) => {
                let rendered = err.to_string();
                result.logs.push(format!(
                    "{} channel inspect protocol failure: {rendered}",
                    channel.label()
                ));
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "inspect_reply",
                        json!({
                            "status": "error",
                            "found": false,
                            "data": {},
                            "metadata": {},
                            "ename": "ProtocolError",
                            "evalue": rendered,
                            "traceback": [rendered],
                        }),
                    ),
                );
            }
        }
        result
    }

    fn process_history_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        match message.content_json() {
            Ok(content) => {
                let access_type = content
                    .get("hist_access_type")
                    .and_then(Value::as_str)
                    .unwrap_or("tail");
                let include_output = content.get("output").and_then(Value::as_bool).unwrap_or(false);
                let history_entries = match access_type {
                    "tail" => {
                        let n = content.get("n").and_then(Value::as_u64).unwrap_or(10) as usize;
                        let start = self.history.len().saturating_sub(n);
                        self.history[start..].to_vec()
                    }
                    "range" => {
                        let start = content.get("start").and_then(Value::as_u64).unwrap_or(1);
                        let stop = content.get("stop").and_then(Value::as_u64).unwrap_or(u64::MAX);
                        self.history
                            .iter()
                            .filter(|entry| entry.line_number >= start && entry.line_number < stop)
                            .cloned()
                            .collect()
                    }
                    "search" => {
                        let pattern = content
                            .get("pattern")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let unique = content.get("unique").and_then(Value::as_bool).unwrap_or(false);
                        let mut entries = self
                            .history
                            .iter()
                            .filter(|entry| entry.code.contains(pattern))
                            .cloned()
                            .collect::<Vec<_>>();
                        if unique {
                            let mut seen = HashSet::new();
                            entries.retain(|entry| seen.insert(entry.code.clone()));
                        }
                        let n = content.get("n").and_then(Value::as_u64).unwrap_or(entries.len() as u64) as usize;
                        let start = entries.len().saturating_sub(n);
                        entries[start..].to_vec()
                    }
                    other => {
                        result.logs.push(format!(
                            "{} channel history unsupported access type: {other}",
                            channel.label()
                        ));
                        Vec::new()
                    }
                };

                let history = history_entries
                    .into_iter()
                    .map(|entry| {
                        if include_output {
                            json!([
                                entry.session,
                                entry.line_number,
                                [entry.code, entry.output.unwrap_or_default()]
                            ])
                        } else {
                            json!([entry.session, entry.line_number, entry.code])
                        }
                    })
                    .collect::<Vec<_>>();

                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "history_reply",
                        json!({
                            "status": "ok",
                            "history": history,
                        }),
                    ),
                );
            }
            Err(err) => {
                let rendered = err.to_string();
                result.logs.push(format!(
                    "{} channel history protocol failure: {rendered}",
                    channel.label()
                ));
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "history_reply",
                        json!({
                            "status": "error",
                            "history": [],
                            "ename": "ProtocolError",
                            "evalue": rendered,
                            "traceback": [rendered],
                        }),
                    ),
                );
            }
        }
        result
    }

    fn process_is_complete_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        match message.content_json() {
            Ok(content) => {
                let code = content
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let status = match completeness_for_code(code, &self.interner) {
                    Completeness::Complete => "complete",
                    Completeness::Incomplete => "incomplete",
                    Completeness::Invalid => "invalid",
                };
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "is_complete_reply",
                        json!({
                            "status": status,
                            "indent": "",
                        }),
                    ),
                );
            }
            Err(err) => {
                let rendered = err.to_string();
                result.logs.push(format!(
                    "{} channel is_complete protocol failure: {rendered}",
                    channel.label()
                ));
                result.push_reply(
                    channel,
                    make_reply(
                        message,
                        "is_complete_reply",
                        json!({
                            "status": "invalid",
                            "indent": "",
                            "ename": "ProtocolError",
                            "evalue": rendered,
                            "traceback": [rendered],
                        }),
                    ),
                );
            }
        }
        result
    }

    fn process_execute_request(
        &mut self,
        channel: Channel,
        message: &DecodedMessage,
    ) -> ProcessResult {
        let mut result = ProcessResult::default();
        if self.is_busy() {
            let rendered = "kernel is already executing a request";
            result.logs.push(format!(
                "{} channel execute rejected while busy: {rendered}",
                channel.label()
            ));
            result.push_reply(
                channel,
                make_reply(
                    message,
                    "execute_reply",
                    json!({
                        "status": "error",
                        "execution_count": self.execution_count,
                        "ename": "KernelBusy",
                        "evalue": rendered,
                        "traceback": [rendered],
                    }),
                ),
            );
            return result;
        }
        result.push_iopub(make_status(message, "busy"));

        let content = match message.content_json() {
            Ok(content) => content,
            Err(err) => {
                self.push_execute_protocol_error(
                    &mut result,
                    channel,
                    message,
                    err,
                    self.execution_count,
                    false,
                );
                return result;
            }
        };

        let code = match content.get("code").and_then(Value::as_str) {
            Some(code) => code,
            None => {
                self.push_execute_protocol_error(
                    &mut result,
                    channel,
                    message,
                    ProtocolError::MissingExecuteCode,
                    self.execution_count,
                    false,
                );
                return result;
            }
        };

        let silent = content
            .get("silent")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let store_history = if silent {
            false
        } else {
            content
                .get("store_history")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        };
        let execution_count = if store_history {
            self.execution_count + 1
        } else {
            self.execution_count
        };

        result.push_trace(
            "execute.launched",
            &[
                ("channel", channel.label().to_string()),
                ("msg_id", message_id(message).to_string()),
                ("silent", silent.to_string()),
                ("store_history", store_history.to_string()),
                ("execution_count", execution_count.to_string()),
            ],
        );

        if !silent {
            result.push_iopub(make_iopub_message(
                message,
                "execute_input",
                json!({
                    "code": code,
                    "execution_count": execution_count,
                }),
            ));
        }

        self.launch_execute(channel, message, code, execution_count, store_history, silent);
        result
    }

    fn push_execute_protocol_error(
        &self,
        result: &mut ProcessResult,
        channel: Channel,
        message: &DecodedMessage,
        err: ProtocolError,
        execution_count: u64,
        silent: bool,
    ) {
        let rendered = err.to_string();
        result.logs.push(format!(
            "{} channel execute protocol failure: {rendered}",
            channel.label()
        ));
        if !silent {
            result.push_iopub(make_error_message(message, "ProtocolError", &rendered));
        }
        result.push_reply(
            channel,
            make_reply(
                message,
                "execute_reply",
                json!({
                    "status": "error",
                    "execution_count": execution_count,
                    "ename": "ProtocolError",
                    "evalue": rendered,
                    "traceback": [rendered],
                }),
            ),
        );
        result.push_iopub(make_status(message, "idle"));
    }

    fn push_execute_handler_error(
        &self,
        result: &mut ProcessResult,
        channel: Channel,
        message: &DecodedMessage,
        err: HandlerError,
        execution_count: u64,
        silent: bool,
    ) {
        let rendered = err.to_string();
        let ename = match err {
            HandlerError::Interrupted => "Interrupted",
            HandlerError::Execute(_) => "EvalError",
        };
        result.logs.push(format!(
            "{} channel execute handler failure: {rendered}",
            channel.label()
        ));
        if !silent {
            result.push_iopub(make_error_message(message, ename, &rendered));
        }
        result.push_reply(
            channel,
            make_reply(
                message,
                "execute_reply",
                json!({
                    "status": "error",
                    "execution_count": execution_count,
                    "ename": ename,
                    "evalue": rendered,
                    "traceback": [rendered],
                }),
            ),
        );
        result.push_iopub(make_status(message, "idle"));
    }
}

fn flush_process_result(
    result: ProcessResult,
    shell: &zmq::Socket,
    control: &zmq::Socket,
    iopub: &zmq::Socket,
    key: &str,
) -> Result<bool, RuntimeError> {
    for line in result.logs {
        eprintln!("[axioma-jupyter] {line}");
    }
    for line in result.traces {
        eprintln!("[axioma-jupyter] {line}");
    }

    for outbound in result.outbound {
        match outbound.target {
            OutboundTarget::Reply(Channel::Shell) => {
                send_jupyter_message(shell, &outbound.message, key).map_err(|err| RuntimeError::Send {
                    channel: "shell",
                    message: err.to_string(),
                })?;
            }
            OutboundTarget::Reply(Channel::Control) => {
                send_jupyter_message(control, &outbound.message, key).map_err(|err| RuntimeError::Send {
                    channel: "control",
                    message: err.to_string(),
                })?;
            }
            OutboundTarget::Iopub => {
                send_jupyter_message(iopub, &outbound.message, key).map_err(|err| RuntimeError::Send {
                    channel: "iopub",
                    message: err.to_string(),
                })?;
            }
        }
    }

    Ok(!result.shutdown)
}

fn handle_socket_event(
    runtime: &mut KernelRuntime,
    channel: Channel,
    socket: &zmq::Socket,
    shell: &zmq::Socket,
    control: &zmq::Socket,
    iopub: &zmq::Socket,
    key: &str,
) -> Result<bool> {
    match receive_jupyter_message(socket, key) {
        Ok(message) => flush_process_result(
            runtime.process_message(channel, &message),
            shell,
            control,
            iopub,
            key,
        )
        .map_err(|err| match err {
            RuntimeError::Send { channel, message } => {
                anyhow!("fatal Jupyter {channel} send failure: {message}")
            }
        }),
        Err(err) => {
            eprintln!(
                "[axioma-jupyter] {}",
                trace_line(
                    "request.receive_failed",
                    &[
                        ("channel", channel.label().to_string()),
                        ("error", err.to_string()),
                    ],
                )
            );
            Ok(true)
        }
    }
}

fn flush_runtime_completions(
    runtime: &mut KernelRuntime,
    shell: &zmq::Socket,
    control: &zmq::Socket,
    iopub: &zmq::Socket,
    key: &str,
) -> Result<bool> {
    let mut keep_running = true;
    while let Some(result) = runtime.try_collect_execution_result() {
        keep_running = flush_process_result(result, shell, control, iopub, key).map_err(|err| {
            match err {
                RuntimeError::Send { channel, message } => {
                    anyhow!("fatal Jupyter {channel} send failure: {message}")
                }
            }
        })?;
        if !keep_running {
            break;
        }
    }
    Ok(keep_running)
}

pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match parse_cli_command(&args)? {
        CliCommand::Run { connection_file } => run_with_connection_file(&connection_file),
        CliCommand::Install(options) => {
            let install_dir = install_kernelspec(&options)?;
            eprintln!(
                "[axioma-jupyter] installed kernelspec '{}' at {}",
                options.kernel_name,
                install_dir.display()
            );
            Ok(())
        }
        CliCommand::PrintKernelspec(options) => {
            let spec = build_kernelspec(&options)?;
            println!("{}", serde_json::to_string_pretty(&spec)?);
            Ok(())
        }
        CliCommand::Help => {
            println!("{}", usage());
            Ok(())
        }
    }
}

pub fn run_with_connection_file(path: &std::path::Path) -> Result<()> {
    let startup = resolve_startup_config(path)?;
    let conn_file = std::fs::read_to_string(path)?;
    let conn: ConnectionInfo = serde_json::from_str(&conn_file)?;

    if conn.signature_scheme != "hmac-sha256" && !conn.signature_scheme.is_empty() {
        return Err(anyhow!(
            "unsupported signature scheme: {}",
            conn.signature_scheme
        ));
    }

    let ctx = zmq::Context::new();

    let shell = ctx.socket(zmq::ROUTER)?;
    shell.bind(&endpoint(&conn, conn.shell_port))?;

    let iopub = ctx.socket(zmq::PUB)?;
    iopub.bind(&endpoint(&conn, conn.iopub_port))?;

    let stdin_socket = ctx.socket(zmq::ROUTER)?;
    stdin_socket.bind(&endpoint(&conn, conn.stdin_port))?;

    let hb = ctx.socket(zmq::REP)?;
    hb.bind(&endpoint(&conn, conn.hb_port))?;

    let control = ctx.socket(zmq::ROUTER)?;
    control.bind(&endpoint(&conn, conn.control_port))?;

    std::thread::spawn(move || loop {
        match hb.recv_msg(0) {
            Ok(msg) => {
                let _ = hb.send(msg, 0);
            }
            Err(_) => break,
        }
    });

    let _ = stdin_socket;
    let search_paths = ax_context::build_import_search_paths(
        &ax_context::ImportSearchPathConfig {
            env_std_path: startup.env_std_path.clone(),
            working_dir: Some(startup.working_dir.clone()),
            executable: std::env::current_exe().ok(),
        },
    );
    log_startup_config(&startup, &conn, &search_paths);
    let mut runtime = KernelRuntime::new(search_paths);

    loop {
        if !flush_runtime_completions(&mut runtime, &shell, &control, &iopub, &conn.key)? {
            break;
        }

        let mut items = [
            shell.as_poll_item(zmq::POLLIN),
            control.as_poll_item(zmq::POLLIN),
        ];
        zmq::poll(&mut items, 50)?;

        if items[0].is_readable()
            && !handle_socket_event(
                &mut runtime,
                Channel::Shell,
                &shell,
                &shell,
                &control,
                &iopub,
                &conn.key,
            )?
        {
            break;
        }

        if !flush_runtime_completions(&mut runtime, &shell, &control, &iopub, &conn.key)? {
            break;
        }

        if items[1].is_readable()
            && !handle_socket_event(
                &mut runtime,
                Channel::Control,
                &control,
                &shell,
                &control,
                &iopub,
                &conn.key,
            )?
        {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod test_harness;
#[cfg(test)]
mod protocol_tests;
#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod startup_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn test_header(msg_type: &str) -> Value {
        json!({
            "msg_id": "test-msg",
            "username": "tester",
            "session": "session-1",
            "date": "0.0",
            "msg_type": msg_type,
            "version": "5.4"
        })
    }

    fn make_frames(msg_type: &str, content: &[u8]) -> Vec<Vec<u8>> {
        vec![
            b"client-1".to_vec(),
            b"<IDS|MSG>".to_vec(),
            Vec::new(),
            serde_json::to_vec(&test_header(msg_type)).expect("header"),
            serde_json::to_vec(&json!({})).expect("parent"),
            serde_json::to_vec(&json!({})).expect("metadata"),
            content.to_vec(),
        ]
    }

    fn execute_frames(code: &str) -> Vec<Vec<u8>> {
        make_frames(
            "execute_request",
            serde_json::to_string(&json!({ "code": code }))
                .expect("content")
                .as_bytes(),
        )
    }

    fn execute_frames_with_options(code: &str, extra: Value) -> Vec<Vec<u8>> {
        let mut content = json!({ "code": code });
        if let Some(obj) = content.as_object_mut() {
            if let Some(extra_obj) = extra.as_object() {
                for (key, value) in extra_obj {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }
        make_frames(
            "execute_request",
            serde_json::to_string(&content).expect("content").as_bytes(),
        )
    }

    fn complete_frames(code: &str, cursor_pos: usize) -> Vec<Vec<u8>> {
        make_frames(
            "complete_request",
            serde_json::to_string(&json!({
                "code": code,
                "cursor_pos": cursor_pos,
            }))
            .expect("content")
            .as_bytes(),
        )
    }

    fn inspect_frames(code: &str, cursor_pos: usize) -> Vec<Vec<u8>> {
        make_frames(
            "inspect_request",
            serde_json::to_string(&json!({
                "code": code,
                "cursor_pos": cursor_pos,
                "detail_level": 1,
            }))
            .expect("content")
            .as_bytes(),
        )
    }

    fn history_frames(content: Value) -> Vec<Vec<u8>> {
        make_frames(
            "history_request",
            serde_json::to_string(&content).expect("content").as_bytes(),
        )
    }

    fn is_complete_frames(code: &str) -> Vec<Vec<u8>> {
        make_frames(
            "is_complete_request",
            serde_json::to_string(&json!({ "code": code }))
                .expect("content")
                .as_bytes(),
        )
    }

    fn interrupt_frames() -> Vec<Vec<u8>> {
        make_frames(
            "interrupt_request",
            serde_json::to_string(&json!({})).expect("content").as_bytes(),
        )
    }

    fn shutdown_frames(restart: bool) -> Vec<Vec<u8>> {
        make_frames(
            "shutdown_request",
            serde_json::to_string(&json!({ "restart": restart }))
                .expect("content")
                .as_bytes(),
        )
    }

    fn message_types(result: &ProcessResult) -> Vec<String> {
        result
            .outbound
            .iter()
            .map(|outbound| {
                outbound.message.header["msg_type"]
                    .as_str()
                    .unwrap_or("missing")
                    .to_string()
            })
            .collect()
    }

    fn reply_content<'a>(result: &'a ProcessResult, msg_type: &str) -> &'a Value {
        &result
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == msg_type)
            .unwrap_or_else(|| panic!("missing {msg_type} in {:?}", message_types(result)))
            .message
            .content
    }

    fn assert_execute_ok(result: &ProcessResult, execution_count: u64, expected_text: &str) {
        assert!(
            result.logs.is_empty(),
            "expected no logs, got {:?}",
            result.logs
        );
        let reply = result
            .outbound
            .iter()
            .find(|outbound| matches!(outbound.target, OutboundTarget::Reply(_)))
            .expect("execute reply");
        assert_eq!(reply.message.content["status"], "ok");
        assert_eq!(reply.message.content["execution_count"], execution_count);
        assert!(
            !result
                .outbound
                .iter()
                .any(|outbound| outbound.message.header["msg_type"] == "display_data"),
            "plain execution must not emit display_data: {:?}",
            message_types(result)
        );

        let execute_result = result
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
            .expect("execute_result");
        assert_eq!(
            execute_result.message.content["data"]["text/plain"],
            expected_text
        );
    }

    fn output_data(msg: &JupyterMessage) -> &serde_json::Map<String, Value> {
        msg.content["data"].as_object().expect("mime bundle")
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axioma-ax-jupyter-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn write_module(root: &Path, module: &str, source: &str) -> PathBuf {
        let mut module_path = root.to_path_buf();
        for part in module.split('.') {
            module_path.push(part);
        }
        module_path.set_extension("ax");
        if let Some(parent) = module_path.parent() {
            std::fs::create_dir_all(parent).expect("module parent");
        }
        std::fs::write(&module_path, source).expect("write module");
        module_path
    }

    fn long_addition_code(terms: usize) -> String {
        std::iter::repeat("1").take(terms).collect::<Vec<_>>().join(" + ")
    }

    #[test]
    fn malformed_message_frames_do_not_break_following_execute() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let malformed = runtime.process_frames(Channel::Shell, vec![b"broken".to_vec()], "");
        assert!(malformed.outbound.is_empty());
        assert_eq!(runtime.execution_count, 0);
        assert!(
            malformed
                .logs
                .iter()
                .any(|line| line.contains("missing Jupyter delimiter")),
            "unexpected logs: {:?}",
            malformed.logs
        );

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 1, "3");
        assert_eq!(runtime.execution_count, 1);
    }

    #[test]
    fn invalid_json_content_returns_error_and_next_execute_works() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let invalid = runtime.process_frames(Channel::Shell, make_frames("execute_request", b"{"), "");
        assert_eq!(runtime.execution_count, 0);
        assert!(
            invalid
                .logs
                .iter()
                .any(|line| line.contains("invalid content JSON")),
            "unexpected logs: {:?}",
            invalid.logs
        );
        let reply = invalid
            .outbound
            .iter()
            .find(|outbound| matches!(outbound.target, OutboundTarget::Reply(Channel::Shell)))
            .expect("error reply");
        assert_eq!(reply.message.content["status"], "error");
        assert_eq!(reply.message.content["execution_count"], 0);
        let statuses: Vec<&str> = invalid
            .outbound
            .iter()
            .filter(|outbound| outbound.message.header["msg_type"] == "status")
            .map(|outbound| outbound.message.content["execution_state"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(statuses, vec!["busy", "idle"]);

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 1, "3");
        assert_eq!(runtime.execution_count, 1);
    }

    #[test]
    fn unsupported_message_type_on_control_does_not_break_following_execute() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let unsupported =
            runtime.process_frames(Channel::Control, make_frames("comm_open", b"{}"), "");
        assert!(unsupported.outbound.is_empty());
        assert_eq!(runtime.execution_count, 0);
        assert!(
            unsupported
                .logs
                .iter()
                .any(|line| line.contains("unsupported Jupyter message type: comm_open")),
            "unexpected logs: {:?}",
            unsupported.logs
        );

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 1, "3");
        assert_eq!(runtime.execution_count, 1);
    }

    #[test]
    fn handler_error_does_not_commit_env_and_next_execute_works() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let failed = runtime.process_frames(
            Channel::Shell,
            execute_frames("let x = 1\nimport does.not.exist"),
            "",
        );
        assert_eq!(runtime.execution_count, 1);
        assert!(
            !runtime
                .env
                .bindings
                .contains_key(&runtime.interner.get_or_intern("x")),
            "failed execution should not commit bindings"
        );
        assert!(
            failed
                .logs
                .iter()
                .any(|line| line.contains("import not found")),
            "unexpected logs: {:?}",
            failed.logs
        );
        let reply = failed
            .outbound
            .iter()
            .find(|outbound| matches!(outbound.target, OutboundTarget::Reply(Channel::Shell)))
            .expect("error reply");
        assert_eq!(reply.message.content["status"], "error");
        assert_eq!(reply.message.content["execution_count"], 1);

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 2, "3");
        assert_eq!(runtime.execution_count, 2);
    }

    #[test]
    fn invalid_signature_does_not_break_following_execute() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let mut frames = execute_frames("1 + 2");
        frames[2] = b"wrong-signature".to_vec();

        let invalid = runtime.process_frames(Channel::Shell, frames, "secret");
        assert!(invalid.outbound.is_empty());
        assert_eq!(runtime.execution_count, 0);
        assert!(
            invalid
                .logs
                .iter()
                .any(|line| line.contains("invalid Jupyter signature")),
            "unexpected logs: {:?}",
            invalid.logs
        );

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 1, "3");
        assert_eq!(runtime.execution_count, 1);
    }

    #[test]
    fn notebook_and_jupyter_resolve_same_import_identically() {
        let cwd = temp_dir("parity");
        write_module(&cwd, "shared.demo", "let imported_value = 64");
        let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
            working_dir: Some(cwd),
            ..ax_context::ImportSearchPathConfig::default()
        });

        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            r#"{"source": "import shared.demo\nimported_value"}"#,
            &mut notebook_env,
            &notebook_interner,
            &search_paths,
        );
        assert_eq!(notebook_result.error, None, "notebook error: {:?}", notebook_result.error);
        assert_eq!(notebook_result.unicode.as_deref(), Some("64"));

        let kernel_env = ax_eval::Env::new();
        let kernel_interner = Arc::new(ax_ir::Interner::new());
        let kernel_result = evaluate_code_transactional(
            "import shared.demo\nimported_value",
            kernel_env,
            kernel_interner,
            &search_paths,
        );
        let kernel_output = kernel_result.expect("kernel import").0;
        let final_result = kernel_output
            .outputs
            .iter()
            .find_map(|output| match output {
                KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
                _ => None,
            });
        assert_eq!(final_result, Some("64".to_string()));
    }

    #[test]
    fn plain_results_emit_execute_result_not_display_data() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let result = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&result, 1, "3");
        assert_eq!(
            message_types(&result),
            vec![
                "status",
                "execute_input",
                "execute_result",
                "execute_reply",
                "status",
            ]
        );
    }

    #[test]
    fn silent_execute_suppresses_visible_out_history() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let result = runtime.process_frames(
            Channel::Shell,
            execute_frames_with_options("1 + 2", json!({ "silent": true })),
            "",
        );

        assert_eq!(runtime.execution_count, 0);
        let message_types = message_types(&result);
        assert_eq!(message_types, vec!["status", "execute_reply", "status"]);
        let reply = result
            .outbound
            .iter()
            .find(|outbound| matches!(outbound.target, OutboundTarget::Reply(Channel::Shell)))
            .expect("execute reply");
        assert_eq!(reply.message.content["status"], "ok");
        assert_eq!(reply.message.content["execution_count"], 0);
        assert!(
            result
                .outbound
                .iter()
                .all(|outbound| outbound.message.header["msg_type"] != "execute_result"),
            "silent execute must not produce Out[n]: {:?}",
            message_types
        );
    }

    #[test]
    fn stream_output_precedes_final_result() {
        let cwd = temp_dir("stream-sequence");
        write_module(&cwd, "shared.demo", "let imported_value = 5");
        let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
            working_dir: Some(cwd),
            ..ax_context::ImportSearchPathConfig::default()
        });
        let mut runtime = KernelRuntime::new(search_paths);

        let result = runtime.process_frames(
            Channel::Shell,
            execute_frames("import shared.demo\nimported_value"),
            "",
        );
        assert_eq!(
            message_types(&result),
            vec![
                "status",
                "execute_input",
                "stream",
                "execute_result",
                "execute_reply",
                "status",
            ]
        );
        let stream = result
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "stream")
            .expect("stream output");
        assert_eq!(stream.message.content["name"], "stdout");
        assert_eq!(stream.message.content["text"], "imported shared.demo\n");
        let execute_result = result
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
            .expect("execute result");
        assert_eq!(execute_result.message.content["execution_count"], 1);
        assert_eq!(execute_result.message.content["data"]["text/plain"], "5");
    }

    #[test]
    fn sequential_executes_increment_out_numbering() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let first = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        let second = runtime.process_frames(Channel::Shell, execute_frames("2 + 3"), "");

        assert_execute_ok(&first, 1, "3");
        assert_execute_ok(&second, 2, "5");
        assert_eq!(runtime.execution_count, 2);
        let first_input = first
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "execute_input")
            .expect("first input");
        let second_input = second
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "execute_input")
            .expect("second input");
        assert_eq!(first_input.message.content["execution_count"], 1);
        assert_eq!(second_input.message.content["execution_count"], 2);
    }

    #[test]
    fn plain_bundle_emits_execute_result_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::ExecuteResult(MimeBundle::plain("plain text")),
            1,
        );
        assert_eq!(message.header["msg_type"], "execute_result");
        assert_eq!(message.content["execution_count"], 1);
        assert_eq!(output_data(&message)["text/plain"], "plain text");
    }

    #[test]
    fn notebook_and_kernel_render_cpt_linearized_einstein_with_identical_plain_labels() {
        let source = "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic))";
        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            &format!(r#"{{"source": "{source}"}}"#),
            &mut notebook_env,
            &notebook_interner,
            &[],
        );

        let kernel_result = evaluate_code_transactional(
            source,
            ax_eval::Env::new(),
            Arc::new(ax_ir::Interner::new()),
            &[],
        )
        .expect("kernel cpt result")
        .0;
        let kernel_plain = kernel_result.outputs.iter().find_map(|output| match output {
            KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
            _ => None,
        });

        assert_eq!(notebook_result.unicode, kernel_plain);
        assert!(
            notebook_result
                .unicode
                .as_deref()
                .unwrap_or("")
                .contains("00_constraint")
        );
    }

    #[test]
    fn notebook_and_kernel_render_cpt_spec_identically() {
        let source = "frw_background_spec(conformal, flat, 3)";
        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            &format!(r#"{{"source": "{source}"}}"#),
            &mut notebook_env,
            &notebook_interner,
            &[],
        );

        let kernel_result = evaluate_code_transactional(
            source,
            ax_eval::Env::new(),
            Arc::new(ax_ir::Interner::new()),
            &[],
        )
        .expect("kernel cpt spec")
        .0;
        let kernel_plain = kernel_result.outputs.iter().find_map(|output| match output {
            KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
            _ => None,
        });

        assert_eq!(notebook_result.unicode, kernel_plain);
        assert_eq!(
            notebook_result.unicode.as_deref(),
            Some("FRWBackground(time=conformal, curvature=flat, spatial_dim=3)")
        );
    }

    #[test]
    fn notebook_and_jupyter_render_scalar_equation_labels_identically() {
        let source = "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic))";
        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            &format!(r#"{{"source": "{source}"}}"#),
            &mut notebook_env,
            &notebook_interner,
            &[],
        );

        let kernel_result = evaluate_code_transactional(
            source,
            ax_eval::Env::new(),
            Arc::new(ax_ir::Interner::new()),
            &[],
        )
        .expect("kernel scalar cpt result")
        .0;
        let kernel_plain = kernel_result.outputs.iter().find_map(|output| match output {
            KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
            _ => None,
        });

        assert_eq!(notebook_result.unicode, kernel_plain);
        assert!(kernel_plain.unwrap_or_default().contains("00_constraint"));
    }

    #[test]
    fn notebook_and_jupyter_render_tensor_equation_labels_identically() {
        let source = "linearized_einstein_tensor()";
        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            &format!(r#"{{"source": "{source}"}}"#),
            &mut notebook_env,
            &notebook_interner,
            &[],
        );

        let kernel_result = evaluate_code_transactional(
            source,
            ax_eval::Env::new(),
            Arc::new(ax_ir::Interner::new()),
            &[],
        )
        .expect("kernel tensor cpt result")
        .0;
        let kernel_plain = kernel_result.outputs.iter().find_map(|output| match output {
            KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
            _ => None,
        });

        assert_eq!(notebook_result.unicode, kernel_plain);
        assert!(kernel_plain.unwrap_or_default().contains("tensor_xx"));
    }

    #[test]
    fn notebook_and_jupyter_render_harmonic_spec_identically() {
        let source = "tensor_harmonic_spec(flat)";
        let mut notebook_env = ax_eval::Env::new();
        let notebook_interner = ax_ir::Interner::new();
        let notebook_result = ax_notebook::handle_eval(
            &format!(r#"{{"source": "{source}"}}"#),
            &mut notebook_env,
            &notebook_interner,
            &[],
        );

        let kernel_result = evaluate_code_transactional(
            source,
            ax_eval::Env::new(),
            Arc::new(ax_ir::Interner::new()),
            &[],
        )
        .expect("kernel harmonic spec")
        .0;
        let kernel_plain = kernel_result.outputs.iter().find_map(|output| match output {
            KernelOutput::ExecuteResult(bundle) => bundle.text_plain().map(str::to_string),
            _ => None,
        });

        assert_eq!(notebook_result.unicode, kernel_plain);
        assert_eq!(kernel_plain.as_deref(), Some("TensorHarmonics(flat, k)"));
    }

    #[test]
    fn latex_bundle_emits_execute_result_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::ExecuteResult(MimeBundle::plain("x^2").with_latex("x^{2}")),
            2,
        );
        let data = output_data(&message);
        assert_eq!(message.header["msg_type"], "execute_result");
        assert_eq!(data["text/plain"], "x^2");
        assert_eq!(data["text/latex"], "x^{2}");
    }

    #[test]
    fn markdown_bundle_emits_display_data_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::DisplayData(
                MimeBundle::markdown("**bold**").with_plain("bold"),
            ),
            3,
        );
        let data = output_data(&message);
        assert_eq!(message.header["msg_type"], "display_data");
        assert_eq!(data["text/plain"], "bold");
        assert_eq!(data["text/markdown"], "**bold**");
    }

    #[test]
    fn html_bundle_emits_display_data_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::DisplayData(
                MimeBundle::html("<b>bold</b>").with_plain("bold"),
            ),
            4,
        );
        let data = output_data(&message);
        assert_eq!(message.header["msg_type"], "display_data");
        assert_eq!(data["text/plain"], "bold");
        assert_eq!(data["text/html"], "<b>bold</b>");
    }

    #[test]
    fn svg_bundle_emits_display_data_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::DisplayData(
                MimeBundle::svg("<svg></svg>").with_plain("plot"),
            ),
            5,
        );
        let data = output_data(&message);
        assert_eq!(message.header["msg_type"], "display_data");
        assert_eq!(data["text/plain"], "plot");
        assert_eq!(data["image/svg+xml"], "<svg></svg>");
    }

    #[test]
    fn json_bundle_emits_display_data_with_plain_fallback() {
        let parent = DecodedMessage {
            identities: Vec::new(),
            header: test_header("execute_request"),
            content_bytes: Vec::new(),
        };
        let message = make_output_message(
            &parent,
            &KernelOutput::DisplayData(
                MimeBundle::json(json!({"answer": 42})).with_plain("{\"answer\":42}"),
            ),
            6,
        );
        let data = output_data(&message);
        assert_eq!(message.header["msg_type"], "display_data");
        assert_eq!(data["text/plain"], "{\"answer\":42}");
        assert_eq!(data["application/json"], json!({"answer": 42}));
    }

    #[test]
    fn qm_execute_result_emits_application_json_bundle() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let result = runtime.process_frames(Channel::Shell, execute_frames("ket(psi)"), "");
        let execute_result = result
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "execute_result")
            .expect("execute_result");
        let data = output_data(&execute_result.message);

        assert!(data.contains_key("application/json"));
        assert_eq!(data["text/plain"], "|psi⟩");
        assert!(
            data["text/latex"]
                .as_str()
                .is_some_and(|latex| latex.contains("\\left|"))
        );
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");
        assert!(json.contains("\"object_kind\":\"ket\""), "got {json}");
    }

    #[test]
    fn cancellation_requested_before_evaluation_interrupts_without_committing_state() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let token = ax_ir::CancellationToken::new();
        token.cancel();

        let result = runtime.execute_with_cancellation("let x = 1", token);
        assert!(matches!(result, Err(HandlerError::Interrupted)));
        assert!(!runtime.env.bindings.contains_key(&runtime.interner.get_or_intern("x")));

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 1, "3");
    }

    #[test]
    fn cancellation_requested_during_long_evaluation_interrupts_and_subsequent_execute_works() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let token = ax_ir::CancellationToken::new();
        let code = long_addition_code(200_000);

        let interrupted = std::thread::scope(|scope| {
            let token_for_thread = token.clone();
            let handle = scope.spawn(|| runtime.execute_with_cancellation(&code, token_for_thread));
            std::thread::sleep(Duration::from_millis(1));
            token.cancel();
            handle.join().expect("evaluation thread")
        });

        assert!(matches!(interrupted, Err(HandlerError::Interrupted)));
        assert!(runtime.env.bindings.is_empty(), "interrupted evaluation must not mutate env");

        let success = runtime.process_frames(Channel::Shell, execute_frames("2 + 3"), "");
        assert_execute_ok(&success, 1, "5");
    }

    #[test]
    fn completion_uses_in_scope_symbol_names() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let _ = runtime.process_frames(Channel::Shell, execute_frames("let schwarzschild_metric = 1"), "");

        let result = runtime.process_frames(
            Channel::Shell,
            complete_frames("schw", 4),
            "",
        );
        let reply = reply_content(&result, "complete_reply");
        assert_eq!(reply["status"], "ok");
        assert_eq!(reply["cursor_start"], 0);
        assert_eq!(reply["cursor_end"], 4);
        let matches = reply["matches"].as_array().expect("matches");
        assert!(
            matches.iter().any(|entry| entry == "schwarzschild_metric"),
            "unexpected matches: {matches:?}"
        );
    }

    #[test]
    fn completion_uses_imported_names_after_execute() {
        let cwd = temp_dir("completion-import");
        write_module(&cwd, "shared.demo", "let imported_value = 64");
        let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
            working_dir: Some(cwd),
            ..ax_context::ImportSearchPathConfig::default()
        });
        let mut runtime = KernelRuntime::new(search_paths);
        let _ = runtime.process_frames(
            Channel::Shell,
            execute_frames("import shared.demo"),
            "",
        );

        let result = runtime.process_frames(
            Channel::Shell,
            complete_frames("imported_v", 10),
            "",
        );
        let matches = reply_content(&result, "complete_reply")["matches"]
            .as_array()
            .expect("matches");
        assert!(
            matches.iter().any(|entry| entry == "imported_value"),
            "unexpected matches: {matches:?}"
        );
    }

    #[test]
    fn inspect_reports_known_symbol_value_and_structure() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let _ = runtime.process_frames(Channel::Shell, execute_frames("let x = 42"), "");

        let result = runtime.process_frames(Channel::Shell, inspect_frames("x", 1), "");
        let reply = reply_content(&result, "inspect_reply");
        assert_eq!(reply["status"], "ok");
        assert_eq!(reply["found"], true);
        let data = reply["data"].as_object().expect("inspect data");
        let plain = data["text/plain"].as_str().expect("plain");
        assert!(plain.contains("name: x"), "plain: {plain}");
        assert!(plain.contains("value: 42"), "plain: {plain}");
        assert!(plain.contains("kind: scalar"), "plain: {plain}");
    }

    #[test]
    fn history_returns_prior_executes_in_order() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let _ = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        let _ = runtime.process_frames(Channel::Shell, execute_frames("2 + 3"), "");

        let result = runtime.process_frames(
            Channel::Shell,
            history_frames(json!({
                "hist_access_type": "tail",
                "n": 2,
                "output": true,
            })),
            "",
        );
        let history = reply_content(&result, "history_reply")["history"]
            .as_array()
            .expect("history");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0][1], 1);
        assert_eq!(history[0][2][0], "1 + 2");
        assert_eq!(history[0][2][1], "3");
        assert_eq!(history[1][1], 2);
        assert_eq!(history[1][2][0], "2 + 3");
        assert_eq!(history[1][2][1], "5");
    }

    #[test]
    fn is_complete_distinguishes_incomplete_complete_and_invalid() {
        let mut runtime = KernelRuntime::new(Vec::new());

        let incomplete = runtime.process_frames(Channel::Shell, is_complete_frames("f(x,"), "");
        assert_eq!(reply_content(&incomplete, "is_complete_reply")["status"], "incomplete");

        let complete = runtime.process_frames(Channel::Shell, is_complete_frames("f(x, y)"), "");
        assert_eq!(reply_content(&complete, "is_complete_reply")["status"], "complete");

        let invalid = runtime.process_frames(Channel::Shell, is_complete_frames("let x = )"), "");
        assert_eq!(reply_content(&invalid, "is_complete_reply")["status"], "invalid");
    }

    #[test]
    fn interrupt_request_stops_running_execute_and_next_execute_still_works() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let start = runtime.process_frames_async(
            Channel::Shell,
            execute_frames(&long_addition_code(200_000)),
            "",
        );
        assert_eq!(message_types(&start), vec!["status", "execute_input"]);

        let interrupt = runtime.process_frames(Channel::Control, interrupt_frames(), "");
        let interrupt_reply = reply_content(&interrupt, "interrupt_reply");
        assert_eq!(interrupt_reply["status"], "ok");

        let finish = runtime.wait_for_execution_result();
        assert_eq!(
            message_types(&finish),
            vec!["error", "execute_reply", "status"]
        );
        let execute_reply = reply_content(&finish, "execute_reply");
        assert_eq!(execute_reply["status"], "error");
        assert_eq!(execute_reply["ename"], "Interrupted");
        let error = finish
            .outbound
            .iter()
            .find(|outbound| outbound.message.header["msg_type"] == "error")
            .expect("error");
        assert_eq!(error.message.content["ename"], "Interrupted");
        let statuses: Vec<&str> = finish
            .outbound
            .iter()
            .filter(|outbound| outbound.message.header["msg_type"] == "status")
            .map(|outbound| outbound.message.content["execution_state"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(statuses, vec!["idle"]);

        let success = runtime.process_frames(Channel::Shell, execute_frames("1 + 2"), "");
        assert_execute_ok(&success, 2, "3");
    }

    #[test]
    fn shutdown_request_replies_cleanly_and_waits_for_inflight_execute_to_finish() {
        let mut runtime = KernelRuntime::new(Vec::new());
        let start = runtime.process_frames_async(
            Channel::Shell,
            execute_frames(&long_addition_code(200_000)),
            "",
        );
        assert_eq!(message_types(&start), vec!["status", "execute_input"]);

        let shutdown = runtime.process_frames(Channel::Control, shutdown_frames(false), "");
        let shutdown_reply = reply_content(&shutdown, "shutdown_reply");
        assert_eq!(shutdown_reply["status"], "ok");
        assert_eq!(shutdown_reply["restart"], false);
        assert!(!shutdown.shutdown, "kernel should exit only after inflight execute settles");

        let finish = runtime.wait_for_execution_result();
        let execute_reply = reply_content(&finish, "execute_reply");
        assert_eq!(execute_reply["status"], "error");
        assert_eq!(execute_reply["ename"], "Interrupted");
        assert!(finish.shutdown, "shutdown should trigger after idle is restored");
    }

    #[test]
    fn symmetry_mime_bundle_has_required_keys() {
        let summary = ax_ai_proto::SymmetryExplainResponse {
            summary: ax_ai_proto::TensorSymmetrySummary {
                tableaux: vec![ax_ai_proto::TensorSymmetryEntry {
                    shape: vec![2],
                    slots: vec![0, 1],
                    label: None,
                    trace_free: false,
                    duality: "none".to_string(),
                }],
            },
            rendered_ascii: "[][]".to_string(),
        };

        let bundle = symmetry_summary_mime_bundle(&summary);
        assert!(bundle.contains_key("text/plain"));
        assert!(bundle.contains_key("application/json"));
    }

    #[test]
    fn notebook_and_jupyter_helpers_consume_the_same_summary_object_exactly() {
        let summary = ax_ai_proto::SymmetryExplainResponse {
            summary: ax_ai_proto::TensorSymmetrySummary {
                tableaux: vec![
                    ax_ai_proto::TensorSymmetryEntry {
                        shape: vec![2, 1],
                        slots: vec![0, 1, 2],
                        label: Some("main".to_string()),
                        trace_free: false,
                        duality: "none".to_string(),
                    },
                    ax_ai_proto::TensorSymmetryEntry {
                        shape: vec![1, 1],
                        slots: vec![1, 2],
                        label: Some("alt".to_string()),
                        trace_free: true,
                        duality: "none".to_string(),
                    },
                ],
            },
            rendered_ascii: concat!(
                "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None, label=\"main\"\n",
                "tableau[1]: shape=[1, 1], slots=[1, 2], trace_free=true, duality=None, label=\"alt\""
            )
            .to_string(),
        };

        let bundle = symmetry_summary_mime_bundle(&summary);
        assert_eq!(bundle.get("text/plain").map(String::as_str), Some(summary.rendered_ascii.as_str()));
        assert_eq!(
            bundle.get("application/json").map(String::as_str),
            Some(
                "{\"tableaux\":[{\"shape\":[2,1],\"slots\":[0,1,2],\"label\":\"main\",\"trace_free\":false,\"duality\":\"none\"},{\"shape\":[1,1],\"slots\":[1,2],\"label\":\"alt\",\"trace_free\":true,\"duality\":\"none\"}]}"
            )
        );
        assert_eq!(
            ax_notebook::render_symmetry_cell(&summary.summary),
            concat!(
                "### tableau[0]\nshape=[2, 1]\nslots=[0, 1, 2]\ntrace_free=false\nduality=none\n\n",
                "### tableau[1]\nshape=[1, 1]\nslots=[1, 2]\ntrace_free=true\nduality=none"
            )
        );
    }
}
