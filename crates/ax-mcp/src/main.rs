use ax_eval::registry::{EvalState, ParamType};
use ax_ir::Expr;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

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
        }
    }

    fn alloc_expr_id(&mut self) -> String {
        let id = format!("expr_{}", self.next_id);
        self.next_id += 1;
        id
    }
}

impl EvalState for McpState {
    fn interner(&self) -> &ax_ir::Interner { &self.interner }
    fn interner_mut(&mut self) -> &mut ax_ir::Interner { &mut self.interner }
    fn env(&self) -> &ax_eval::Env { &self.env }
    fn env_mut(&mut self) -> &mut ax_eval::Env { &mut self.env }

    fn store_expr(&mut self, expr: Expr) -> String {
        let id = self.alloc_expr_id();
        if let Expr::Matrix(rows) = &expr {
            self.matrices.insert(id.clone(), rows.clone());
        }
        self.expressions.insert(id.clone(), expr);
        id
    }

    fn get_expr(&self, id: &str) -> Option<&Expr> { self.expressions.get(id) }

    fn parse_code(&mut self, code: &str) -> Result<Expr, String> {
        let (_node, diags) = ax_syntax::parser::parse_file(code);
        let errors = diags
            .iter()
            .filter(|d| matches!(d.severity, ax_syntax::diag::Severity::Error))
            .map(|d| d.message.clone())
            .collect::<Vec<_>>();
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        let lowered = ax_core_ir::lower(code, &self.interner);
        if !lowered.errors.is_empty() {
            return Err(lowered.errors.iter().map(|e| e.message.clone()).collect::<Vec<_>>().join("; "));
        }
        let expr = lowered.exprs.into_iter().next().ok_or_else(|| "no expression parsed".to_string())?;
        Ok(ax_eval::eval(&expr, &self.env, &self.interner))
    }

    fn render_latex(&self, expr: &Expr) -> String { ax_render::to_latex(expr, &self.interner) }
    fn render_unicode(&self, expr: &Expr) -> String { ax_render::to_unicode(expr, &self.interner) }

    fn get_metric(&self, id: &str) -> Option<&(ax_tensor::SymbolicMatrix, Vec<ax_ir::expr::Sym>)> {
        self.metrics.get(id)
    }
    fn store_metric(&mut self, id: String, metric: ax_tensor::SymbolicMatrix, coords: Vec<ax_ir::expr::Sym>) {
        self.metrics.insert(id, (metric, coords));
    }
    fn get_christoffel(&self, id: &str) -> Option<&Vec<Vec<Vec<Expr>>>> { self.christoffels.get(id) }
    fn store_christoffel(&mut self, id: String, chris: Vec<Vec<Vec<Expr>>>) { self.christoffels.insert(id, chris); }
    fn get_riemann(&self, id: &str) -> Option<&Vec<Vec<Vec<Vec<Expr>>>>> { self.riemanns.get(id) }
    fn store_riemann(&mut self, id: String, riem: Vec<Vec<Vec<Vec<Expr>>>>) { self.riemanns.insert(id, riem); }
    fn get_ricci(&self, id: &str) -> Option<&Vec<Vec<Expr>>> { self.riccis.get(id) }
    fn store_ricci(&mut self, id: String, ric: Vec<Vec<Expr>>) {
        self.matrices.insert(id.clone(), ric.clone());
        self.riccis.insert(id, ric);
    }
    fn get_matrix_data(&self, id: &str) -> Option<Vec<Vec<Expr>>> {
        self.matrices.get(id).cloned().or_else(|| match self.expressions.get(id) {
            Some(Expr::Matrix(rows)) => Some(rows.clone()),
            _ => None,
        })
    }
}

fn param_type_to_schema(param_type: &ParamType, description: &str) -> Value {
    match param_type {
        ParamType::ExprId | ParamType::Code | ParamType::Symbol => json!({"type": "string", "description": description}),
        ParamType::SymbolList => json!({"type": "array", "items": {"type": "string"}, "description": description}),
        ParamType::Integer => json!({"type": "integer", "description": description}),
        ParamType::Float => json!({"type": "number", "description": description}),
        ParamType::StringEnum(opts) => json!({"type": "string", "enum": opts, "description": description}),
        ParamType::Matrix => json!({"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": description}),
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
                .map(|p| (p.name.to_string(), param_type_to_schema(&p.param_type, p.description)))
                .collect();
            let required: Vec<&str> = entry.parameters.iter().filter(|p| p.required).map(|p| p.name).collect();
            json!({
                "name": format!("axioma_{}", entry.name),
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

fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "axioma-mcp", "version": "0.1.0" }
    })
}

fn tools_list_result() -> Value {
    json!({ "tools": tool_definitions() })
}

fn handle_tools_call(state: &mut McpState, tool_name: &str, arguments: &Value) -> Value {
    let name = tool_name.strip_prefix("axioma_").unwrap_or(tool_name);
    let entries = ax_eval::callable_entries();
    if let Some(entry) = entries.iter().find(|e| e.name == name) {
        let args: Vec<Value> = entry
            .parameters
            .iter()
            .map(|p| arguments.get(p.name).cloned().unwrap_or(Value::Null))
            .collect();
        match (entry.handler)(&args, state) {
            Ok(result) => result,
            Err(e) => json!({"error": e}),
        }
    } else {
        json!({"error": format!("unknown tool: {}", tool_name)})
    }
}

fn handle_request(state: &mut McpState, request: &Value) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return make_error_response(id, -32600, "missing method");
    };
    match method {
        "initialize" => make_result_response(id, initialize_result()),
        "tools/list" => make_result_response(id, tools_list_result()),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return make_error_response(id, -32602, "missing tool name");
            };
            let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            make_result_response(id, handle_tools_call(state, name, &arguments))
        }
        _ => make_error_response(id, -32601, "method not found"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = McpState::new();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => handle_request(&mut state, &request),
            Err(err) => make_error_response(Value::Null, -32700, &format!("parse error: {err}")),
        };
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }

    Ok(())
}
