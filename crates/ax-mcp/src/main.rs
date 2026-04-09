use ax_eval::registry::{EvalState, ParamType};
use ax_ir::Expr;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use thiserror::Error;
#[cfg(feature = "http")]
use tokio_stream::Stream;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool '{tool}' panicked: {message}")]
    Panic { tool: String, message: String },
    #[error("tool '{tool}' timed out after {seconds}s")]
    Timeout { tool: String, seconds: u64 },
    #[error("tool '{tool}' failed: {message}")]
    Failed { tool: String, message: String },
    #[error("unknown tool: {tool}")]
    UnknownTool { tool: String },
}

struct McpState {
    interner: ax_ir::Interner,
    env: ax_eval::Env,
    expressions: HashMap<String, Expr>,
    metrics: HashMap<String, (ax_tensor::SymbolicMatrix, Vec<ax_ir::expr::Sym>)>,
    christoffels: HashMap<String, Vec<Vec<Vec<Expr>>>>,
    riemanns: HashMap<String, Vec<Vec<Vec<Vec<Expr>>>>>,
    riccis: HashMap<String, Vec<Vec<Expr>>>,
    matrices: HashMap<String, Vec<Vec<Expr>>>,
    next_id: u64,
    deadline: Option<Instant>,
}

impl McpState {
    fn new() -> Self {
        let mut env = ax_eval::Env::new();
        env.enable_pool();
        env.parallel = true;
        Self {
            interner: ax_ir::Interner::new(),
            env,
            expressions: HashMap::new(),
            metrics: HashMap::new(),
            christoffels: HashMap::new(),
            riemanns: HashMap::new(),
            riccis: HashMap::new(),
            matrices: HashMap::new(),
            next_id: 1,
            deadline: None,
        }
    }

    fn alloc_expr_id(&mut self) -> String {
        let id = format!("expr_{}", self.next_id);
        self.next_id += 1;
        id
    }
}

impl EvalState for McpState {
    fn interner(&self) -> &ax_ir::Interner {
        &self.interner
    }
    fn interner_mut(&mut self) -> &mut ax_ir::Interner {
        &mut self.interner
    }
    fn env(&self) -> &ax_eval::Env {
        &self.env
    }
    fn env_mut(&mut self) -> &mut ax_eval::Env {
        &mut self.env
    }

    fn store_expr(&mut self, expr: Expr) -> String {
        for rule in ax_eval::parse_component_rules_expr(&expr) {
            self.env.component_rule_symbols.insert(rule.tensor);
        }
        let id = self.alloc_expr_id();
        if let Expr::Matrix(rows) = &expr {
            self.matrices.insert(id.clone(), rows.clone());
        }
        self.expressions.insert(id.clone(), expr);
        id
    }

    fn get_expr(&self, id: &str) -> Option<&Expr> {
        self.expressions.get(id)
    }

    fn parse_code(&mut self, code: &str) -> Result<Expr, String> {
        let mut sources = vec![code.to_string()];
        if !code.trim_end().ends_with(';') {
            sources.push(format!("{code};"));
        }

        let mut last_errors = Vec::new();
        let mut parsed = None;
        for source in sources {
            let lowered = ax_core_ir::lower(&source, &self.interner);
            if lowered.errors.is_empty() {
                if let Some(expr) = lowered.exprs.into_iter().next() {
                    parsed = Some(expr);
                    break;
                }
            } else {
                last_errors = lowered.errors.iter().map(|e| e.message.clone()).collect();
            }
        }

        let expr = parsed.ok_or_else(|| {
            if last_errors.is_empty() {
                "no expression parsed".to_string()
            } else {
                last_errors.join("; ")
            }
        })?;
        ax_eval::apply_index_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_coordinate_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_property_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_grassmann_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_operator_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_parallel_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_graded_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_superspace_setup(&expr, &mut self.env, &self.interner);
        ax_eval::apply_brst_setup(&expr, &mut self.env, &self.interner);
        Ok(ax_eval::eval(&expr, &self.env, &self.interner))
    }

    fn render_latex(&self, expr: &Expr) -> String {
        ax_render::to_latex(expr, &self.interner)
    }
    fn render_unicode(&self, expr: &Expr) -> String {
        ax_render::to_unicode(expr, &self.interner)
    }

    fn get_metric(&self, id: &str) -> Option<&(ax_tensor::SymbolicMatrix, Vec<ax_ir::expr::Sym>)> {
        self.metrics.get(id)
    }
    fn store_metric(
        &mut self,
        id: String,
        metric: ax_tensor::SymbolicMatrix,
        coords: Vec<ax_ir::expr::Sym>,
    ) {
        self.metrics.insert(id, (metric, coords));
    }
    fn get_christoffel(&self, id: &str) -> Option<&Vec<Vec<Vec<Expr>>>> {
        self.christoffels.get(id)
    }
    fn store_christoffel(&mut self, id: String, chris: Vec<Vec<Vec<Expr>>>) {
        self.christoffels.insert(id, chris);
    }
    fn get_riemann(&self, id: &str) -> Option<&Vec<Vec<Vec<Vec<Expr>>>>> {
        self.riemanns.get(id)
    }
    fn store_riemann(&mut self, id: String, riem: Vec<Vec<Vec<Vec<Expr>>>>) {
        self.riemanns.insert(id, riem);
    }
    fn get_ricci(&self, id: &str) -> Option<&Vec<Vec<Expr>>> {
        self.riccis.get(id)
    }
    fn store_ricci(&mut self, id: String, ric: Vec<Vec<Expr>>) {
        self.matrices.insert(id.clone(), ric.clone());
        self.riccis.insert(id, ric);
    }
    fn get_matrix_data(&self, id: &str) -> Option<Vec<Vec<Expr>>> {
        self.matrices
            .get(id)
            .cloned()
            .or_else(|| match self.expressions.get(id) {
                Some(Expr::Matrix(rows)) => Some(rows.clone()),
                _ => None,
            })
    }
    fn list_expression_ids(&self) -> Vec<String> {
        let mut ids = self.expressions.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }
    fn list_metric_ids(&self) -> Vec<String> {
        let mut ids = self.metrics.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }
    fn list_christoffel_ids(&self) -> Vec<String> {
        let mut ids = self.christoffels.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }
    fn list_riemann_ids(&self) -> Vec<String> {
        let mut ids = self.riemanns.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }
    fn list_ricci_ids(&self) -> Vec<String> {
        let mut ids = self.riccis.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }
    fn list_properties(&self) -> Vec<(String, Vec<String>)> {
        let mut entries = self
            .env
            .property_store
            .symbols()
            .into_iter()
            .map(|sym| {
                let mut props = self
                    .env
                    .property_store
                    .get_all(sym)
                    .into_iter()
                    .map(|prop| ax_eval::registry::format_tensor_property(prop, &self.interner))
                    .collect::<Vec<_>>();
                props.sort();
                (self.interner.resolve(sym).to_string(), props)
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }
    fn list_index_families(&self) -> Vec<(String, Vec<String>, Option<usize>)> {
        let mut families = self
            .env
            .index_families
            .values()
            .map(|family| {
                let mut values = family
                    .values
                    .iter()
                    .map(|sym| self.interner.resolve(*sym).to_string())
                    .collect::<Vec<_>>();
                values.sort();
                (
                    self.interner.resolve(family.name).to_string(),
                    values,
                    family.dimension,
                )
            })
            .collect::<Vec<_>>();
        families.sort_by(|a, b| a.0.cmp(&b.0));
        families
    }
    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

fn param_type_to_schema(param_type: &ParamType, description: &str) -> Value {
    match param_type {
        ParamType::ExprId | ParamType::Code | ParamType::Symbol => {
            json!({"type": "string", "description": description})
        }
        ParamType::SymbolList => {
            json!({"type": "array", "items": {"type": "string"}, "description": description})
        }
        ParamType::Bool => json!({"type": "boolean", "description": description}),
        ParamType::Integer => json!({"type": "integer", "description": description}),
        ParamType::Float => json!({"type": "number", "description": description}),
        ParamType::StringEnum(opts) => {
            json!({"type": "string", "enum": opts, "description": description})
        }
        ParamType::Matrix => {
            json!({"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": description})
        }
        ParamType::Optional(inner) => param_type_to_schema(inner, description),
    }
}

fn tool_definitions() -> Vec<Value> {
    ax_eval::callable_entries()
        .iter()
        .map(|entry| {
            let properties: Map<String, Value> = entry
                .parameters
                .iter()
                .map(|p| {
                    (
                        p.name.to_string(),
                        param_type_to_schema(&p.param_type, p.description),
                    )
                })
                .collect();
            let required: Vec<&str> = entry
                .parameters
                .iter()
                .filter(|p| p.required)
                .map(|p| p.name)
                .collect();
            json!({
                "name": format!("axioma_{}", entry.name),
                "category": entry.category,
                "description": entry.description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            })
        })
        .collect()
}

fn make_result_response(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn make_error_response(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn initialize_result(transport: &str, port: u16) -> Value {
    let mut result = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "axioma-mcp", "version": "0.1.0" }
    });
    if transport == "http" {
        result["capabilities"]["transport"] = json!({
            "type": "http+sse",
            "endpoint": format!("http://127.0.0.1:{port}/mcp")
        });
    }
    result
}

fn tools_list_result() -> Value {
    json!({ "tools": tool_definitions() })
}

fn error_value(error: ToolError) -> Value {
    json!({
        "status": "error",
        "message": error.to_string(),
    })
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "unknown panic".to_string()
    }
}

fn handle_tools_call_safe(
    state: &mut McpState,
    tool_name: &str,
    arguments: &Value,
    timeout_secs: u64,
) -> Value {
    #[cfg(test)]
    if tool_name == "axioma_test_panic" {
        let result = std::panic::catch_unwind(|| panic!("test panic"));
        return match result {
            Ok(_) => unreachable!(),
            Err(payload) => error_value(ToolError::Panic {
                tool: tool_name.to_string(),
                message: panic_message(payload),
            }),
        };
    }
    #[cfg(test)]
    if tool_name == "axioma_test_timeout" {
        state.deadline = Some(Instant::now() + Duration::from_secs(timeout_secs));
        let result = ax_ir::with_deadline(state.deadline, || {
            std::thread::sleep(Duration::from_millis(5));
            ax_ir::check_deadline()
        });
        state.deadline = None;
        return match result {
            Ok(_) => json!({"status": "ok"}),
            Err(_) => error_value(ToolError::Timeout {
                tool: tool_name.to_string(),
                seconds: timeout_secs,
            }),
        };
    }

    let name = tool_name.strip_prefix("axioma_").unwrap_or(tool_name);
    let entries = ax_eval::callable_entries();
    let Some(entry) = entries.iter().find(|e| e.name == name) else {
        return error_value(ToolError::UnknownTool {
            tool: tool_name.to_string(),
        });
    };

    let args: Vec<Value> = entry
        .parameters
        .iter()
        .map(|p| {
            arguments
                .get(p.name)
                .cloned()
                .or_else(|| {
                    (p.name == "expr")
                        .then(|| arguments.get("expr_id").cloned())
                        .flatten()
                })
                .unwrap_or(Value::Null)
        })
        .collect();

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    state.deadline = Some(deadline);

    let (tx, rx) = mpsc::channel();
    let handler = entry.handler;
    let args_owned = args;
    let tool_owned = tool_name.to_string();
    let state_addr = state as *mut McpState as usize;

    let worker = match std::thread::Builder::new()
        .name(format!("tool:{tool_owned}"))
        .stack_size(2 * 1024 * 1024 * 1024)
        .spawn(move || {
            let result = ax_ir::with_deadline(Some(deadline), || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let state = unsafe { &mut *(state_addr as *mut McpState) };
                    handler(&args_owned, state)
                }))
            });
            let _ = tx.send(result);
        }) {
        Ok(worker) => worker,
        Err(err) => {
            state.deadline = None;
            return error_value(ToolError::Failed {
                tool: tool_name.to_string(),
                message: format!("failed to spawn tool worker: {err}"),
            });
        }
    };
    let mut worker = Some(worker);

    let response = match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(result) => match result {
            Ok(Ok(value)) => value,
            Ok(Err(message)) => {
                if message.contains("computation timed out") {
                    error_value(ToolError::Timeout {
                        tool: tool_name.to_string(),
                        seconds: timeout_secs,
                    })
                } else {
                    error_value(ToolError::Failed {
                        tool: tool_name.to_string(),
                        message,
                    })
                }
            }
            Err(payload) => error_value(ToolError::Panic {
                tool: tool_name.to_string(),
                message: panic_message(payload),
            }),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
            state.deadline = None;
            return error_value(ToolError::Timeout {
                tool: tool_name.to_string(),
                seconds: timeout_secs,
            });
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => match worker.take().unwrap().join() {
            Ok(()) => error_value(ToolError::Failed {
                tool: tool_name.to_string(),
                message: "tool worker exited without returning a result".to_string(),
            }),
            Err(payload) => error_value(ToolError::Panic {
                tool: tool_name.to_string(),
                message: panic_message(payload),
            }),
        },
    };

    if let Some(worker) = worker.take() {
        let _ = worker.join();
    }
    state.deadline = None;
    response
}

fn handle_request(
    state: &mut McpState,
    request: &Value,
    timeout_secs: u64,
    transport: &str,
    port: u16,
) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return make_error_response(id, -32600, "missing method");
    };
    match method {
        "initialize" => make_result_response(id, initialize_result(transport, port)),
        "notifications/initialized" => Value::Null,
        "ping" => make_result_response(id, json!({})),
        "tools/list" => make_result_response(id, tools_list_result()),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return make_error_response(id, -32602, "missing tool name");
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            make_result_response(
                id,
                handle_tools_call_safe(state, name, &arguments, timeout_secs),
            )
        }
        _ => make_error_response(id, -32601, "method not found"),
    }
}

fn run_stdio(timeout_secs: u64) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = McpState::new();

    for line in stdin.lock().lines() {
        let line = line.map_err(|err| err.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => handle_request(&mut state, &request, timeout_secs, "stdio", 3000),
            Err(err) => make_error_response(Value::Null, -32700, &format!("parse error: {err}")),
        };
        if response != Value::Null {
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }
    }

    Ok(())
}

#[cfg(feature = "http")]
type HttpSharedState = std::sync::Arc<tokio::sync::Mutex<McpState>>;

#[cfg(feature = "http")]
fn build_http_app(state: HttpSharedState, timeout_secs: u64, port: u16) -> axum::Router {
    use axum::{routing::post, Router};
    use tower_http::cors::{Any, CorsLayer};

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/mcp", post(handle_http_request))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(cors)
        .with_state((state, timeout_secs, port))
}

#[cfg(feature = "http")]
async fn run_http(port: u16, timeout_secs: u64) -> Result<(), Box<dyn std::error::Error>> {
    let state = std::sync::Arc::new(tokio::sync::Mutex::new(McpState::new()));
    let app = build_http_app(state, timeout_secs, port);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!("Axioma MCP server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(feature = "http")]
async fn handle_http_request(
    axum::extract::State((state, timeout_secs, port)): axum::extract::State<(
        HttpSharedState,
        u64,
        u16,
    )>,
    axum::Json(request): axum::Json<Value>,
) -> axum::response::sse::Sse<
    std::pin::Pin<
        Box<dyn Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> + Send>,
    >,
> {
    use axum::response::sse::{Event, Sse};

    let mut state = state.lock().await;
    let response = handle_request(&mut state, &request, timeout_secs, "http", port);
    if response == Value::Null {
        return Sse::new(Box::pin(tokio_stream::empty()));
    }
    let json_str = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
    let stream = tokio_stream::once(Ok(Event::default().data(json_str)));
    Sse::new(Box::pin(stream))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let transport = args
        .iter()
        .position(|a| a == "--transport")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("stdio");
    let _port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let timeout_secs: u64 = args
        .iter()
        .position(|a| a == "--timeout")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    match transport {
        "stdio" => run_stdio(timeout_secs),
        #[cfg(feature = "http")]
        "http" => {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(run_http(port, timeout_secs))
        }
        #[cfg(not(feature = "http"))]
        "http" => {
            eprintln!("Unknown transport: http. Rebuild with --features http.");
            std::process::exit(1);
        }
        other => {
            eprintln!("Unknown transport: {}. Use 'stdio' or 'http'.", other);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_eval::ParamType;
    use serde_json::Map;

    #[derive(Clone)]
    struct TestFixtures {
        expr_id: Value,
        product_expr_id: Value,
        multivar_expr_id: Value,
        list_id: Value,
        list_rhs_id: Value,
        matrix_id: Value,
        eq_id: Value,
        focus_id: Value,
        remainder_id: Value,
        metric_id: Value,
        christoffel_id: Value,
        riemann_id: Value,
        ricci_id: Value,
    }

    fn expect_ok(result: Value, tool: &str) -> Value {
        assert!(
            matches!(
                result.get("status").and_then(Value::as_str),
                Some("ok" | "unchanged")
            ),
            "{tool} failed: {result:?}"
        );
        result
    }

    fn build_fixtures(state: &mut McpState) -> TestFixtures {
        let expr = expect_ok(
            handle_tools_call_safe(state, "axioma_eval", &json!({"code": "x"}), 5),
            "axioma_eval expr",
        );
        let product = expect_ok(
            handle_tools_call_safe(state, "axioma_eval", &json!({"code": "(a+b)*c"}), 5),
            "axioma_eval product",
        );
        let multivar = expect_ok(
            handle_tools_call_safe(state, "axioma_eval", &json!({"code": "x*y*z"}), 5),
            "axioma_eval multivar",
        );
        let list = expect_ok(
            handle_tools_call_safe(state, "axioma_eval", &json!({"code": "[x, y]"}), 5),
            "axioma_eval list",
        );
        let list_rhs = expect_ok(
            handle_tools_call_safe(state, "axioma_eval", &json!({"code": "[u, v]"}), 5),
            "axioma_eval list_rhs",
        );
        let matrix = expect_ok(
            handle_tools_call_safe(
                state,
                "axioma_outer",
                &json!({"left": list["expr_id"].clone(), "right": list_rhs["expr_id"].clone()}),
                5,
            ),
            "axioma_outer",
        );
        let eq = expect_ok(
            handle_tools_call_safe(
                state,
                "axioma_eq",
                &json!({"lhs": expr["expr_id"].clone(), "rhs": product["expr_id"].clone()}),
                5,
            ),
            "axioma_eq",
        );
        let zoom = expect_ok(
            handle_tools_call_safe(
                state,
                "axioma_zoom",
                &json!({"expr": expr["expr_id"].clone(), "pattern": "a"}),
                5,
            ),
            "axioma_zoom",
        );
        let metric = expect_ok(
            handle_tools_call_safe(
                state,
                "axioma_define_metric",
                &json!({
                    "name": "g",
                    "components": [
                        ["-(1 - 2/r)", "0", "0", "0"],
                        ["0", "1/(1 - 2/r)", "0", "0"],
                        ["0", "0", "r^2", "0"],
                        ["0", "0", "0", "r^2*sin(theta)^2"]
                    ],
                    "coordinates": ["t", "r", "theta", "phi"]
                }),
                5,
            ),
            "axioma_define_metric",
        );
        let christoffel = expect_ok(
            handle_tools_call_safe(state, "axioma_christoffel", &json!({"metric_id": "g"}), 5),
            "axioma_christoffel",
        );
        let riemann = expect_ok(
            handle_tools_call_safe(state, "axioma_riemann", &json!({"christoffel_id": "g"}), 5),
            "axioma_riemann",
        );
        let ricci = expect_ok(
            handle_tools_call_safe(state, "axioma_ricci", &json!({"riemann_id": "g"}), 5),
            "axioma_ricci",
        );
        expect_ok(
            handle_tools_call_safe(state, "axioma_setup_superspace", &json!({"N": 1}), 5),
            "axioma_setup_superspace",
        );
        expect_ok(
            handle_tools_call_safe(
                state,
                "axioma_setup_brst_ym",
                &json!({"A": "A", "c": "c", "cbar": "cbar", "B": "B", "g": "g"}),
                5,
            ),
            "axioma_setup_brst_ym",
        );

        TestFixtures {
            expr_id: expr["expr_id"].clone(),
            product_expr_id: product["expr_id"].clone(),
            multivar_expr_id: multivar["expr_id"].clone(),
            list_id: list["expr_id"].clone(),
            list_rhs_id: list_rhs["expr_id"].clone(),
            matrix_id: matrix["expr_id"].clone(),
            eq_id: eq["expr_id"].clone(),
            focus_id: zoom["focus_id"].clone(),
            remainder_id: zoom["remainder_id"].clone(),
            metric_id: metric["metric_id"].clone(),
            christoffel_id: christoffel["christoffel_id"].clone(),
            riemann_id: riemann["riemann_id"].clone(),
            ricci_id: ricci["ricci_id"].clone(),
        }
    }

    fn is_syntax_only(entry: &ax_eval::CallableEntry) -> bool {
        entry.name != "eval"
            && entry.parameters.len() == 1
            && matches!(entry.parameters[0].param_type, ParamType::Code)
    }

    fn expr_id_for(entry_name: &str, param_name: &str, fx: &TestFixtures) -> Value {
        if matches!(
            entry_name,
            "integrate"
                | "double_integral"
                | "dblint"
                | "triple_integral"
                | "tplint"
                | "definite_integral"
                | "defint"
        ) && param_name == "expr"
        {
            return fx.multivar_expr_id.clone();
        }
        match param_name {
            "basis" | "bra" | "ket" | "left" | "right" | "functions" | "state" => {
                fx.list_id.clone()
            }
            "tensor" | "metric" | "metric_inverse" => fx.matrix_id.clone(),
            "ricci" => fx.matrix_id.clone(),
            "vector" | "covector" => fx.list_id.clone(),
            "scalar" => fx.expr_id.clone(),
            "eq" | "equation" | "ode" => fx.eq_id.clone(),
            "focus" => fx.focus_id.clone(),
            "remainder" => fx.remainder_id.clone(),
            "lhs" => {
                if entry_name == "eq" || entry_name == "atan2" {
                    fx.expr_id.clone()
                } else {
                    fx.eq_id.clone()
                }
            }
            "rhs" => {
                if entry_name == "eq" || entry_name == "atan2" {
                    fx.product_expr_id.clone()
                } else {
                    fx.expr_id.clone()
                }
            }
            _ => fx.expr_id.clone(),
        }
    }

    fn value_for_param_type(
        entry_name: &str,
        param_name: &str,
        param_type: &ParamType,
        fx: &TestFixtures,
    ) -> Value {
        match param_type {
            ParamType::ExprId => expr_id_for(entry_name, param_name, fx),
            ParamType::Code => match param_name {
                "metric_id" => fx.metric_id.clone(),
                "christoffel_id" => fx.christoffel_id.clone(),
                "riemann_id" => fx.riemann_id.clone(),
                "code" => json!("1 + 1"),
                "target" => json!("a"),
                "replacement" => json!("b"),
                "pattern" => json!("a"),
                "point" => json!("0"),
                "lower_bound" => json!("0"),
                "upper_bound" => json!("1"),
                "equation" => json!("x + y - 2"),
                "equations" => json!(["x + y - 2", "x - y"]),
                "fields" => json!([["phi", ["phi_t", "phi_x"]]]),
                "name" => json!("g2"),
                _ => json!("x"),
            },
            ParamType::Symbol => match param_name {
                "A" => json!("A"),
                "c" => json!("c"),
                "cbar" => json!("cbar"),
                "B" => json!("B"),
                "g" => json!("g"),
                "field" => json!("phi"),
                "background" => json!("g0"),
                "background_inv" => json!("g0inv"),
                "perturbation" => json!("h"),
                "epsilon" => json!("eps"),
                "variation" => json!("delta_phi"),
                "dependent" => json!("y"),
                "independent" | "x" | "variable" => json!("x"),
                "y" => json!("y"),
                "z" => json!("z"),
                "spatial_var" => json!("x"),
                "temporal_var" => json!("t"),
                "index" => json!("a"),
                _ => json!("x"),
            },
            ParamType::SymbolList => match param_name {
                "coordinates" | "coords" => json!(["t", "r"]),
                "variables" | "targets" | "dependents" => json!(["x", "y"]),
                "field_derivatives" | "variation_derivatives" => json!(["phi_t", "phi_x"]),
                _ => json!(["x", "y"]),
            },
            ParamType::Bool => json!(true),
            ParamType::Integer => match param_name {
                "N" => json!(1),
                "coord_index" => json!(0),
                "order" => json!(1),
                "steps" => json!(8),
                "n" => json!(4),
                "l" => json!(2),
                "i" | "a" => json!(1),
                "j" | "b" | "eliminate" => json!(2),
                "k" | "c" => json!(3),
                "d" => json!(4),
                "e" => json!(5),
                "f" => json!(6),
                _ => json!(1),
            },
            ParamType::Float => match param_name {
                "x_end" => json!(1.0),
                "y0" => json!(1.0),
                _ => json!(0.0),
            },
            ParamType::StringEnum(options) => json!(options.first().copied().unwrap_or("")),
            ParamType::Matrix => json!([["1", "0"], ["0", "1"]]),
            ParamType::Optional(inner) => value_for_param_type(entry_name, param_name, inner, fx),
        }
    }

    fn arguments_for_entry(entry: &ax_eval::CallableEntry, fx: &TestFixtures) -> Value {
        let mut args = Map::new();
        if entry.name == "rk4_system" {
            args.insert("functions".to_string(), fx.list_id.clone());
            args.insert("independent".to_string(), json!("t"));
            args.insert("dependents".to_string(), json!(["x", "y"]));
            args.insert("x0".to_string(), json!(0.0));
            args.insert("y0s".to_string(), json!([1.0, 0.0]));
            args.insert("x_end".to_string(), json!(1.0));
            args.insert("steps".to_string(), json!(8));
            return Value::Object(args);
        }
        if entry.name == "bcfw_decomposition" {
            args.insert("n".to_string(), json!(4));
            args.insert("i".to_string(), json!(1));
            args.insert("j".to_string(), json!(2));
            args.insert("helicities".to_string(), json!([1, -1, 1, -1]));
            return Value::Object(args);
        }
        for param in entry.parameters {
            args.insert(
                param.name.to_string(),
                value_for_param_type(entry.name, param.name, &param.param_type, fx),
            );
        }
        Value::Object(args)
    }

    #[test]
    fn panic_in_tool_does_not_crash_server() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(&mut state, "axioma_test_panic", &json!({}), 5);
        assert_eq!(result["status"], "error");

        let result2 =
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "1 + 1"}), 5);
        assert!(result2.get("expr_id").is_some());
    }

    #[test]
    fn timeout_returns_error_not_hang() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(&mut state, "axioma_test_timeout", &json!({}), 0);
        assert_eq!(result["status"], "error");
        assert!(result["message"]
            .as_str()
            .unwrap_or("")
            .contains("timed out"));
    }

    #[test]
    fn real_gr_tool_timeout_returns_structured_error() {
        let mut state = McpState::new();
        let metric = handle_tools_call_safe(
            &mut state,
            "axioma_define_metric",
            &json!({
                "name": "g",
                "components": [
                    ["-(1 - 2/r)", "0", "0", "0"],
                    ["0", "1/(1 - 2/r)", "0", "0"],
                    ["0", "0", "r^2", "0"],
                    ["0", "0", "0", "r^2*sin(theta)^2"]
                ],
                "coordinates": ["t", "r", "theta", "phi"]
            }),
            5,
        );
        assert_eq!(metric["status"], "ok", "metric creation failed: {metric:?}");
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_christoffel",
            &json!({"metric_id": "g"}),
            0,
        );
        assert_eq!(result["status"], "error");
        assert!(result["message"]
            .as_str()
            .unwrap_or("")
            .contains("timed out"));
    }

    #[test]
    fn expr_response_includes_status() {
        let mut state = McpState::new();
        let result =
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "1 + 1"}), 5);
        assert_eq!(result["status"], "ok");
    }

    #[test]
    fn algorithm_response_shows_changed_true() {
        let mut state = McpState::new();
        let expr =
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "(a+b)*c"}), 5);
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_expand",
            &json!({"expr_id": expr["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(result["changed"], true);
        assert!(result["message"].as_str().unwrap_or("").contains("expand"));
    }

    #[test]
    fn algorithm_response_shows_unchanged() {
        let mut state = McpState::new();
        let expr = handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "a + b"}), 5);
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_simplify",
            &json!({"expr_id": expr["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "unchanged");
        assert_eq!(result["changed"], false);
    }

    #[test]
    fn error_response_has_status_error() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_simplify",
            &json!({"expr_id": "does_not_exist"}),
            5,
        );
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn all_handlers_return_status_field() {
        let mut state = McpState::new();
        let fixtures = build_fixtures(&mut state);
        for entry in ax_eval::callable_entries() {
            if is_syntax_only(&entry) {
                continue;
            }
            let tool_name = format!("axioma_{}", entry.name);
            let result = handle_tools_call_safe(
                &mut state,
                &tool_name,
                &arguments_for_entry(&entry, &fixtures),
                5,
            );
            assert!(
                result.get("status").is_some(),
                "{} returned no status field: {:?}",
                entry.name,
                result
            );
        }
    }

    #[test]
    fn list_expressions_returns_all_stored() {
        let mut state = McpState::new();
        handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "1 + 1"}), 5);
        handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "x^2"}), 5);
        let result = handle_tools_call_safe(&mut state, "axioma_list_expressions", &json!({}), 5);
        assert_eq!(result["status"], "ok");
        let exprs = result["expressions"].as_array().unwrap();
        assert!(exprs.len() >= 2, "{result:?}");
    }

    #[test]
    fn list_properties_after_declaration() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property R riemann_symmetry"}),
            5,
        );
        let result = handle_tools_call_safe(&mut state, "axioma_list_properties", &json!({}), 5);
        assert_eq!(result["status"], "ok");
        let props = result["properties"].as_array().unwrap();
        let r_entry = props.iter().find(|p| p["symbol"] == "R").unwrap();
        assert!(r_entry["properties"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p.as_str().unwrap().contains("Riemann")));
    }

    #[test]
    fn get_state_summary_covers_everything() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property R riemann_symmetry"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [mu, nu, rho, sigma] dim=4"}),
            5,
        );
        let result = handle_tools_call_safe(&mut state, "axioma_get_state_summary", &json!({}), 5);
        assert_eq!(result["status"], "ok");
        assert!(result.get("expression_count").is_some(), "{result:?}");
        assert!(result.get("properties").is_some(), "{result:?}");
        assert!(result.get("index_families").is_some(), "{result:?}");
    }

    #[test]
    fn state_tools_are_categorized_as_state() {
        let tools = tool_definitions();
        for name in [
            "axioma_list_expressions",
            "axioma_list_metrics",
            "axioma_list_properties",
            "axioma_list_index_families",
            "axioma_get_state_summary",
        ] {
            let tool = tools
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing tool definition for {name}"));
            assert_eq!(tool["category"], "state", "{tool:?}");
        }
    }

    #[test]
    fn diff_identical_expressions() {
        let mut state = McpState::new();
        let a = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "x + y"}), 5),
            "axioma_eval",
        );
        let b = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "x + y"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_diff",
            &json!({"expr_a": a["expr_id"].clone(), "expr_b": b["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(result["identical"], true, "{result:?}");
    }

    #[test]
    fn diff_coefficient_difference() {
        let mut state = McpState::new();
        let a = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "2*x + y"}), 5),
            "axioma_eval",
        );
        let b = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "3*x + y"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_diff",
            &json!({"expr_a": a["expr_id"].clone(), "expr_b": b["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        let details = result["details"].as_array().unwrap();
        assert!(
            details.iter().any(|d| d == "coefficient_differs"),
            "{result:?}"
        );
    }

    #[test]
    fn diff_index_name_difference() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b, c, d] dim=4"}),
            5,
        );
        let a = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "T[a-, b+]"}), 5),
            "axioma_eval",
        );
        let b = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "T[c-, d+]"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_diff",
            &json!({"expr_a": a["expr_id"].clone(), "expr_b": b["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        let details = result["details"].as_array().unwrap();
        assert!(
            details.iter().any(|d| d == "index_names_differ"),
            "{result:?}"
        );
    }

    #[test]
    fn check_properties_missing_symmetry() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b] dim=4"}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "T[a-, b+]"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_check_properties",
            &json!({"expr": expr["expr_id"].clone(), "algorithm": "canonicalise"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(result["ready"], false, "{result:?}");
        assert!(
            result["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|issue| issue
                    .as_str()
                    .unwrap_or("")
                    .contains("no symmetry properties")),
            "{result:?}"
        );
    }

    #[test]
    fn check_properties_all_ok() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property R riemann_symmetry"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b, c, d] dim=4"}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "R[a-, b-, c+, d+]"}),
                5,
            ),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_check_properties",
            &json!({"expr": expr["expr_id"].clone(), "algorithm": "canonicalise"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(result["ready"], true, "{result:?}");
    }

    #[test]
    fn explain_canonicalise_with_issues() {
        let mut state = McpState::new();
        let expr = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "T[a-, b+]"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_explain",
            &json!({"algorithm": "canonicalise", "expr": expr["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        let explanation = result["explanation"].as_str().unwrap_or("");
        assert!(explanation.contains("canonicalise"));
        assert!(explanation.contains("symmetry"), "{result:?}");
    }

    #[test]
    fn check_properties_evaluate_components_tracks_component_rules() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [mu, nu] dim=2"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_declare_coordinates",
            &json!({"coordinates": ["t", "x"]}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "T[mu-, nu-]"}),
                5,
            ),
            "axioma_eval",
        );
        let before = handle_tools_call_safe(
            &mut state,
            "axioma_check_properties",
            &json!({"expr": expr["expr_id"].clone(), "algorithm": "evaluate_components"}),
            5,
        );
        assert_eq!(before["status"], "ok");
        assert_eq!(before["ready"], false, "{before:?}");
        assert!(
            before["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|issue| issue
                    .as_str()
                    .unwrap_or("")
                    .contains("No component rules are known for symbol T")),
            "{before:?}"
        );

        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "[[T, [t, t], 1], [T, [x, x], 2]]"}),
            5,
        );
        let after = handle_tools_call_safe(
            &mut state,
            "axioma_check_properties",
            &json!({"expr": expr["expr_id"].clone(), "algorithm": "evaluate_components"}),
            5,
        );
        assert_eq!(after["status"], "ok");
        assert_eq!(after["ready"], true, "{after:?}");
    }

    #[test]
    fn diagnostic_tools_are_categorized_as_diagnostics() {
        let tools = tool_definitions();
        for name in ["axioma_diff", "axioma_check_properties", "axioma_explain"] {
            let tool = tools
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing tool definition for {name}"));
            assert_eq!(tool["category"], "diagnostics", "{tool:?}");
        }
    }

    #[test]
    fn suggest_without_goal_returns_all() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property T symmetric"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b, c, d] dim=4"}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "T[a-, b+] + T[c-, d+]"}),
                5,
            ),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_suggest",
            &json!({"expr": expr["expr_id"].clone()}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert!(
            result["suggestions"].as_array().unwrap().len() >= 2,
            "{result:?}"
        );
    }

    #[test]
    fn suggest_with_simplify_goal_prioritises_canonicalise() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property T symmetric"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b] dim=4"}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(&mut state, "axioma_eval", &json!({"code": "T[a-, b+]"}), 5),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_suggest",
            &json!({"expr": expr["expr_id"].clone(), "goal": "simplify to canonical form"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(
            result["suggestions"][0]["algorithm"], "canonicalise",
            "{result:?}"
        );
    }

    #[test]
    fn suggest_with_evaluate_goal_prioritises_evaluate_components() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [mu, nu] dim=2"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_declare_coordinates",
            &json!({"coordinates": ["t", "x"]}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "T[mu-, nu-]"}),
                5,
            ),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_suggest",
            &json!({"expr": expr["expr_id"].clone(), "goal": "evaluate in components"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        let algorithms = result["suggestions"]
            .as_array()
            .unwrap()
            .iter()
            .take(3)
            .filter_map(|item| item["algorithm"].as_str())
            .collect::<Vec<_>>();
        assert!(algorithms.contains(&"evaluate_components"), "{result:?}");
    }

    #[test]
    fn suggest_with_prove_zero_goal() {
        let mut state = McpState::new();
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "property T symmetric"}),
            5,
        );
        handle_tools_call_safe(
            &mut state,
            "axioma_eval",
            &json!({"code": "indices spacetime [a, b, c, d] dim=4"}),
            5,
        );
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "T[a-, b+] + T[c-, d+]"}),
                5,
            ),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_suggest",
            &json!({"expr": expr["expr_id"].clone(), "goal": "prove this vanishes"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        let algorithms = result["suggestions"]
            .as_array()
            .unwrap()
            .iter()
            .take(5)
            .filter_map(|item| item["algorithm"].as_str())
            .collect::<Vec<_>>();
        assert!(algorithms.contains(&"canonicalise"), "{result:?}");
        assert!(algorithms.contains(&"meld"), "{result:?}");
        assert!(algorithms.contains(&"collect_terms"), "{result:?}");
    }

    #[test]
    fn suggest_with_unknown_goal_returns_all_with_note() {
        let mut state = McpState::new();
        let expr = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "sin(x) + cos(x)"}),
                5,
            ),
            "axioma_eval",
        );
        let result = handle_tools_call_safe(
            &mut state,
            "axioma_suggest",
            &json!({"expr": expr["expr_id"].clone(), "goal": "quantum gravity loop corrections"}),
            5,
        );
        assert_eq!(result["status"], "ok");
        assert_eq!(result["goal"], "quantum gravity loop corrections");
        assert!(
            result["note"]
                .as_str()
                .unwrap_or("")
                .contains("No goal-specific priority profile matched"),
            "{result:?}"
        );
        assert!(
            !result["suggestions"].as_array().unwrap().is_empty(),
            "{result:?}"
        );
    }

    #[test]
    fn stdio_still_works_without_http_feature() {
        let mut state = McpState::new();
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        });
        let response = handle_request(&mut state, &request, 5, "stdio", 3000);
        assert_eq!(response["result"]["serverInfo"]["name"], "axioma-mcp");
    }

    #[cfg(feature = "http")]
    async fn spawn_http_test_server() -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = build_http_app(
            std::sync::Arc::new(tokio::sync::Mutex::new(McpState::new())),
            5,
            port,
        );
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (port, handle)
    }

    #[cfg(feature = "http")]
    async fn http_request(
        method: &str,
        port: u16,
        path: &str,
        body: Option<&str>,
        extra_headers: &[(&str, &str)],
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut last_err = None;
        let mut stream = {
            let mut connected = None;
            for _ in 0..40 {
                match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
                    Ok(stream) => {
                        connected = Some(stream);
                        break;
                    }
                    Err(err) => {
                        last_err = Some(err);
                        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    }
                }
            }
            connected.unwrap_or_else(|| {
                panic!(
                    "failed to connect to test HTTP server: {}",
                    last_err
                        .map(|err| err.to_string())
                        .unwrap_or_else(|| "unknown connection error".to_string())
                )
            })
        };
        let body = body.unwrap_or("");
        let mut request =
            format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
        for (name, value) in extra_headers {
            request.push_str(&format!("{name}: {value}\r\n"));
        }
        if !body.is_empty() {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        } else {
            request.push_str("Content-Length: 0\r\n");
        }
        request.push_str("\r\n");
        request.push_str(body);
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn http_initialize_returns_server_info() {
        let (port, handle) = spawn_http_test_server().await;
        let response = http_request(
            "POST",
            port,
            "/mcp",
            Some(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            &[],
        )
        .await;
        handle.abort();
        assert!(response.contains("text/event-stream"), "{response}");
        assert!(response.contains("\"serverInfo\""), "{response}");
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn http_tool_call_works() {
        let (port, handle) = spawn_http_test_server().await;
        let response = http_request(
            "POST",
            port,
            "/mcp",
            Some(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"axioma_eval","arguments":{"code":"1+1"}}}"#),
            &[],
        )
        .await;
        handle.abort();
        assert!(response.contains("data: "), "{response}");
        assert!(response.contains("\"expr_id\""), "{response}");
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn http_cors_headers_present() {
        let (port, handle) = spawn_http_test_server().await;
        let response = http_request(
            "OPTIONS",
            port,
            "/mcp",
            None,
            &[
                ("Origin", "http://example.com"),
                ("Access-Control-Request-Method", "POST"),
            ],
        )
        .await;
        handle.abort();
        assert!(
            response
                .to_lowercase()
                .contains("access-control-allow-origin"),
            "{response}"
        );
    }
}
