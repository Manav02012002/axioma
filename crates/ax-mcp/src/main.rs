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
        ax_eval::apply_named_operator_declaration(&expr, &mut self.env, &self.interner);
        ax_eval::apply_named_contraction_declaration(&expr, &mut self.env, &self.interner);
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
    let mut tools = ax_eval::callable_entries()
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
        .collect::<Vec<_>>();
    tools.extend(qm_tool_definitions());
    tools.push(symmetry_summary_tool_definition());
    tools.extend(cpt_tool_definitions());
    tools
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

fn qm_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "axioma_qm_density_summary",
            "category": "quantum",
            "description": "Summarize a stored density-matrix expression with trace, purity, and qubit diagnostics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Stored density-matrix expression id."
                    }
                },
                "required": ["expr"]
            }
        }),
        json!({
            "name": "axioma_qm_partial_trace",
            "category": "quantum",
            "description": "Reduce a stored density matrix by tracing out one tensor factor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Stored density-matrix expression id."
                    },
                    "dims": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Tensor-factor dimensions in lexicographic product order."
                    },
                    "factor_index": {
                        "type": "integer",
                        "description": "Zero-based factor index to trace out."
                    }
                },
                "required": ["expr", "dims", "factor_index"]
            }
        }),
        json!({
            "name": "axioma_qm_expectation_value",
            "category": "quantum",
            "description": "Compute an expectation value from stored operator and state expressions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operator_expr": {
                        "type": "string",
                        "description": "Stored observable matrix expression id."
                    },
                    "state_expr": {
                        "type": "string",
                        "description": "Stored density-matrix expression id."
                    }
                },
                "required": ["operator_expr", "state_expr"]
            }
        }),
        json!({
            "name": "axioma_qm_lindblad_steady_state",
            "category": "quantum",
            "description": "Solve for a finite-dimensional Lindblad steady state from stored Hamiltonian and jump operators.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hamiltonian_expr": {
                        "type": "string",
                        "description": "Stored Hamiltonian matrix expression id."
                    },
                    "jump_exprs": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Stored jump-operator matrix expression ids."
                    }
                },
                "required": ["hamiltonian_expr", "jump_exprs"]
            }
        }),
    ]
}

fn symmetry_summary_tool_definition() -> Value {
    json!({
        "name": "tensor.symmetry_summary",
        "category": "diagnostics",
        "description": "Parse a structured tableau symmetry expression and return a machine-readable summary with an ASCII render.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "tableau_symmetry(...) expression to summarize"
                }
            },
            "required": ["expression"]
        }
    })
}

fn cpt_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "cpt.background",
            "category": "cosmology",
            "description": "Build and render a compact CPT FRW background spec.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "curvature": {"type": "string"},
                    "spatial_dim": {"type": "integer"}
                },
                "required": ["time", "curvature", "spatial_dim"]
            }
        }),
        json!({
            "name": "cpt.linearized_einstein",
            "category": "cosmology",
            "description": "Return labelled CPT Einstein equations as structured JSON.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "time": {"type": "string"},
                    "curvature": {"type": "string"},
                    "spatial_dim": {"type": "integer"},
                    "gauge": {"type": "string"},
                    "matter": {"type": "string"}
                },
                "required": ["order", "time", "curvature", "spatial_dim", "gauge", "matter"]
            }
        }),
        json!({
            "name": "cpt.fluid_equations",
            "category": "cosmology",
            "description": "Return labelled CPT perfect-fluid conservation equations as structured JSON.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "curvature": {"type": "string"},
                    "spatial_dim": {"type": "integer"}
                },
                "required": ["time", "curvature", "spatial_dim"]
            }
        }),
        json!({
            "name": "cpt.mukhanov_sasaki",
            "category": "cosmology",
            "description": "Return the CPT Mukhanov-Sasaki equation renderings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "curvature": {"type": "string"},
                    "spatial_dim": {"type": "integer"},
                    "matter": {"type": "string"}
                },
                "required": ["time", "curvature", "spatial_dim", "matter"]
            }
        }),
        json!({
            "name": "cpt.export_mode_rhs",
            "category": "cosmology",
            "description": "Return exported CPT mode RHS code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "time": {"type": "string"},
                    "curvature": {"type": "string"},
                    "spatial_dim": {"type": "integer"},
                    "matter": {"type": "string"}
                },
                "required": ["target", "time", "curvature", "spatial_dim", "matter"]
            }
        }),
        json!({
            "name": "cpt.multifield",
            "category": "cosmology",
            "description": "Return multifield curvature and entropy equations as structured JSON.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "nfields": {"type": "integer"}
                },
                "required": ["nfields"]
            }
        }),
        json!({
            "name": "cpt.boltzmann_bridge",
            "category": "cosmology",
            "description": "Return a symbolic Einstein-Boltzmann bridge system.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.boltzmann_bridge_export",
            "category": "cosmology",
            "description": "Return exported Einstein-Boltzmann bridge code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"}
                },
                "required": ["target"]
            }
        }),
        json!({
            "name": "cpt.second_order_vector",
            "category": "cosmology",
            "description": "Return derived second-order vector Einstein equations.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.second_order_tensor",
            "category": "cosmology",
            "description": "Return derived second-order tensor Einstein equations.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.project_second_order_vector",
            "category": "cosmology",
            "description": "Return second-order vector equations projected to harmonic space.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.project_second_order_tensor",
            "category": "cosmology",
            "description": "Return second-order tensor equations projected to harmonic space.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.cubic_action",
            "category": "cosmology",
            "description": "Return a reduced cubic CPT action density.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"}
                },
                "required": ["channel"]
            }
        }),
        json!({
            "name": "cpt.cubic_kernel",
            "category": "cosmology",
            "description": "Return a cubic Fourier-space CPT kernel.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"}
                },
                "required": ["channel"]
            }
        }),
        json!({
            "name": "cpt.bispectrum_shape",
            "category": "cosmology",
            "description": "Return a cubic bispectrum shape evaluation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "shape": {"type": "string"}
                },
                "required": ["channel", "shape"]
            }
        }),
        json!({
            "name": "cpt.export_cubic_vertex",
            "category": "cosmology",
            "description": "Return exported cubic vertex code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["channel", "target"]
            }
        }),
        json!({
            "name": "cpt.eft_model",
            "category": "cosmology",
            "description": "Return a typed reduced EFT model description.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"}
                },
                "required": ["kind"]
            }
        }),
        json!({
            "name": "cpt.neutrino_hierarchy",
            "category": "cosmology",
            "description": "Return a symbolic neutrino hierarchy with explicit truncation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lmax": {"type": "integer"},
                    "gauge": {"type": "string"},
                    "closure": {"type": "string"}
                },
                "required": ["lmax", "gauge", "closure"]
            }
        }),
        json!({
            "name": "cpt.photon_hierarchy",
            "category": "cosmology",
            "description": "Return a symbolic photon hierarchy with explicit truncation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lmax": {"type": "integer"},
                    "gauge": {"type": "string"},
                    "closure": {"type": "string"}
                },
                "required": ["lmax", "gauge", "closure"]
            }
        }),
        json!({
            "name": "cpt.export_hierarchy",
            "category": "cosmology",
            "description": "Return exported hierarchy code or hook payload.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "species": {"type": "string"},
                    "lmax": {"type": "integer"},
                    "gauge": {"type": "string"},
                    "closure": {"type": "string"}
                },
                "required": ["target", "species", "lmax", "gauge", "closure"]
            }
        }),
        json!({
            "name": "cpt.parity_report",
            "category": "cosmology",
            "description": "Return built-in CPT parity benchmark reports.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cpt.eft_stability",
            "category": "cosmology",
            "description": "Return reduced EFT stability conditions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"}
                },
                "required": ["kind"]
            }
        }),
        json!({
            "name": "cpt.eft_mode_equations",
            "category": "cosmology",
            "description": "Return reduced EFT mode equations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"}
                },
                "required": ["kind"]
            }
        }),
        json!({
            "name": "cpt.eft_export_rhs",
            "category": "cosmology",
            "description": "Return exported reduced EFT mode RHS code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["kind", "target"]
            }
        }),
    ]
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

fn stored_matrix_from_expression_id(
    state: &McpState,
    expr_id: &str,
) -> Result<Vec<Vec<Expr>>, &'static str> {
    match state.expressions.get(expr_id) {
        Some(Expr::Matrix(rows)) => Ok(rows.clone()),
        Some(_) => Err("stored expression is not a matrix"),
        None => Err("expression id not found"),
    }
}

fn stored_named_matrix_from_expression_id(
    state: &McpState,
    expr_id: &str,
    not_found_message: &'static str,
) -> Result<Vec<Vec<Expr>>, &'static str> {
    match state.expressions.get(expr_id) {
        Some(Expr::Matrix(rows)) => Ok(rows.clone()),
        Some(_) => Err("stored expression is not a matrix"),
        None => Err(not_found_message),
    }
}

fn render_matrix_cell(state: &McpState, expr: &Expr) -> String {
    match expr {
        Expr::Int(value) => value.to_string(),
        Expr::Rational(value) => {
            if value.is_integer() {
                value.to_integer().to_string()
            } else {
                format!("{}/{}", value.numer(), value.denom())
            }
        }
        Expr::Neg(inner) => match inner.as_ref() {
            Expr::Int(value) => format!("-{value}"),
            Expr::Rational(value) if value.is_integer() => format!("-{}", value.to_integer()),
            Expr::Rational(value) => format!("-{}/{}", value.numer(), value.denom()),
            _ => state.render_unicode(expr),
        },
        _ => state.render_unicode(expr),
    }
}

fn render_matrix_cells(state: &McpState, matrix: &[Vec<Expr>]) -> Vec<Vec<String>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|cell| render_matrix_cell(state, cell)).collect())
        .collect()
}

fn parse_dims_argument(arguments: &Value) -> Result<Vec<usize>, &'static str> {
    let Some(items) = arguments.get("dims").and_then(Value::as_array) else {
        return Err("partial trace failed");
    };
    items.iter()
        .map(|item| {
            item.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("partial trace failed")
        })
        .collect()
}

fn handle_qm_density_summary(arguments: &Value, state: &McpState) -> Result<Value, &'static str> {
    let expr_id = arguments
        .get("expr")
        .and_then(Value::as_str)
        .ok_or("expression id not found")?;
    let rho = stored_matrix_from_expression_id(state, expr_id)?;
    let dimension = rho.len();
    let trace = ax_eval::eval(
        &Expr::add(
            rho.iter()
                .enumerate()
                .filter_map(|(i, row)| row.get(i).cloned())
                .collect(),
        ),
        &state.env,
        &state.interner,
    );
    let purity = ax_qm::purity(&rho).map_err(|_| "stored expression is not a matrix")?;
    let linear_entropy =
        ax_qm::linear_entropy(&rho).map_err(|_| "stored expression is not a matrix")?;
    let is_qubit = matches!(rho.as_slice(), [row_a, row_b] if row_a.len() == 2 && row_b.len() == 2);

    let mut payload = json!({
        "status": "ok",
        "dimension": dimension,
        "trace": state.render_unicode(&trace),
        "purity": state.render_unicode(&purity),
        "linear_entropy": state.render_unicode(&linear_entropy),
        "is_qubit": is_qubit,
    });

    if is_qubit {
        let bloch = ax_qm::bloch_vector(&rho).map_err(|_| "stored expression is not a matrix")?;
        payload["bloch_vector"] = Value::Array(
            bloch
                .into_iter()
                .map(|component| Value::String(state.render_unicode(&component)))
                .collect(),
        );
    }

    Ok(payload)
}

fn handle_qm_partial_trace(arguments: &Value, state: &McpState) -> Result<Value, &'static str> {
    let expr_id = arguments
        .get("expr")
        .and_then(Value::as_str)
        .ok_or("expression id not found")?;
    let rho = stored_matrix_from_expression_id(state, expr_id)?;
    let dims = parse_dims_argument(arguments)?;
    let factor_index = arguments
        .get("factor_index")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("partial trace failed")?;
    let reduced =
        ax_qm::try_partial_trace_factor(&rho, &dims, factor_index).map_err(|_| "partial trace failed")?;
    let reduced_expr = Expr::Matrix(reduced.clone());
    Ok(json!({
        "status": "ok",
        "matrix": render_matrix_cells(state, &reduced),
        "unicode": state.render_unicode(&reduced_expr),
        "latex": state.render_latex(&reduced_expr),
    }))
}

fn handle_qm_expectation_value(arguments: &Value, state: &McpState) -> Result<Value, &'static str> {
    let operator_expr = arguments
        .get("operator_expr")
        .and_then(Value::as_str)
        .ok_or("expression id not found")?;
    let state_expr = arguments
        .get("state_expr")
        .and_then(Value::as_str)
        .ok_or("expression id not found")?;
    let operator = stored_matrix_from_expression_id(state, operator_expr)?;
    let rho = stored_matrix_from_expression_id(state, state_expr)?;
    let value = ax_qm::expectation_value(&operator, &rho).map_err(|_| "expectation value failed")?;
    let unicode = state.render_unicode(&value);
    Ok(json!({
        "status": "ok",
        "value": unicode,
        "unicode": state.render_unicode(&value),
    }))
}

fn handle_qm_lindblad_steady_state(
    arguments: &Value,
    state: &McpState,
) -> Result<Value, &'static str> {
    let hamiltonian_expr = arguments
        .get("hamiltonian_expr")
        .and_then(Value::as_str)
        .ok_or("hamiltonian expression id not found")?;
    let h = stored_named_matrix_from_expression_id(
        state,
        hamiltonian_expr,
        "hamiltonian expression id not found",
    )?;

    let jump_exprs = arguments
        .get("jump_exprs")
        .and_then(Value::as_array)
        .ok_or("steady state solver failed")?;
    let jump_ops = jump_exprs
        .iter()
        .map(|value| {
            let expr_id = value.as_str().ok_or("jump expression id not found")?;
            stored_named_matrix_from_expression_id(state, expr_id, "jump expression id not found")
        })
        .collect::<Result<Vec<_>, _>>()?;

    let matrix = ax_solve::lindblad_steady_state_linear(&h, &jump_ops, &state.interner)
        .map_err(|_| "steady state solver failed")?;
    let matrix_expr = Expr::Matrix(matrix.clone());

    Ok(json!({
        "status": "ok",
        "matrix": render_matrix_cells(state, &matrix),
        "unicode": state.render_unicode(&matrix_expr),
        "latex": state.render_latex(&matrix_expr),
    }))
}

fn handle_tools_call_safe(
    state: &mut McpState,
    tool_name: &str,
    arguments: &Value,
    timeout_secs: u64,
) -> Value {
    if let Some(result) = handle_qm_tool_call(state, tool_name, arguments) {
        return result;
    }
    if let Some(result) = handle_cpt_tool_call(state, tool_name, arguments) {
        return result;
    }
    if tool_name == "tensor.symmetry_summary" {
        return match handle_tensor_symmetry_summary(arguments) {
            Ok(value) => value,
            Err(message) => error_value(ToolError::Failed {
                tool: tool_name.to_string(),
                message,
            }),
        };
    }

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

fn handle_qm_tool_call(state: &McpState, tool_name: &str, arguments: &Value) -> Option<Value> {
    let result = match tool_name {
        "axioma_qm_density_summary" => handle_qm_density_summary(arguments, state),
        "axioma_qm_partial_trace" => handle_qm_partial_trace(arguments, state),
        "axioma_qm_expectation_value" => handle_qm_expectation_value(arguments, state),
        "axioma_qm_lindblad_steady_state" => handle_qm_lindblad_steady_state(arguments, state),
        _ => return None,
    };
    Some(match result {
        Ok(value) => value,
        Err(message) => json!({
            "status": "error",
            "message": message,
        }),
    })
}

fn eval_cpt_source(state: &mut McpState, source: &str) -> Result<Expr, String> {
    state.parse_code(source)
}

fn cpt_labelled_equations_json(expr: &Expr, interner: &ax_ir::Interner) -> Option<Value> {
    let Expr::List(items) = expr else {
        return None;
    };
    let mut equations = Vec::with_capacity(items.len());
    for item in items {
        let Expr::List(pair) = item else {
            return None;
        };
        let [Expr::Sym(label), value] = pair.as_slice() else {
            return None;
        };
        equations.push(json!({
            "label": interner.resolve(*label),
            "unicode": ax_render::to_unicode(value, interner),
            "latex": ax_render::to_latex(value, interner),
        }));
    }
    Some(json!({ "status": "ok", "equations": equations }))
}

fn cpt_named_exprs_json(expr: &Expr, interner: &ax_ir::Interner) -> Option<Value> {
    let Expr::List(items) = expr else {
        return None;
    };
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Expr::List(pair) = item else {
            return None;
        };
        let [Expr::Sym(label), value] = pair.as_slice() else {
            return None;
        };
        values.push(json!({
            "label": interner.resolve(*label),
            "unicode": ax_render::to_unicode(value, interner),
            "latex": ax_render::to_latex(value, interner),
        }));
    }
    Some(json!({ "status": "ok", "entries": values }))
}

fn cpt_pair_equations_json(expr: &Expr, interner: &ax_ir::Interner) -> Option<Value> {
    let Expr::List(items) = expr else {
        return None;
    };
    let mut equations = Vec::with_capacity(items.len());
    let mut variables = Vec::with_capacity(items.len());
    for item in items {
        let Expr::List(pair) = item else {
            return None;
        };
        let [lhs, rhs] = pair.as_slice() else {
            return None;
        };
        variables.push(ax_render::to_unicode(lhs, interner));
        equations.push(json!({
            "lhs": ax_render::to_unicode(lhs, interner),
            "rhs": ax_render::to_unicode(rhs, interner),
        }));
    }
    Some(json!({ "status": "ok", "variables": variables, "equations": equations }))
}

fn background_source(arguments: &Value) -> Result<String, String> {
    let time = arguments
        .get("time")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing time".to_string())?;
    let curvature = arguments
        .get("curvature")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing curvature".to_string())?;
    let spatial_dim = arguments
        .get("spatial_dim")
        .and_then(Value::as_i64)
        .ok_or_else(|| "missing spatial_dim".to_string())?;
    Ok(format!(
        "frw_background_spec({time}, {curvature}, {spatial_dim})"
    ))
}

fn handle_cpt_tool_call(state: &mut McpState, tool_name: &str, arguments: &Value) -> Option<Value> {
    let result = match tool_name {
        "cpt.background" => {
            let source = background_source(arguments).ok()?;
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.linearized_einstein" => {
            let order = arguments.get("order").and_then(Value::as_i64)?;
            let gauge = arguments.get("gauge").and_then(Value::as_str)?;
            let matter = arguments.get("matter").and_then(Value::as_str)?;
            let bg = background_source(arguments).ok()?;
            let source = format!(
                "cpt_linearized_einstein({order}, {bg}, cpt_gauge({gauge}), cpt_matter({matter}))"
            );
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.fluid_equations" => {
            let bg = background_source(arguments).ok()?;
            let source = format!("cpt_fluid_equations({bg})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.mukhanov_sasaki" => {
            let matter = arguments.get("matter").and_then(Value::as_str)?;
            let bg = background_source(arguments).ok()?;
            let source = format!("cpt_mukhanov_sasaki({bg}, cpt_matter({matter}))");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.export_mode_rhs" => {
            let target = arguments.get("target").and_then(Value::as_str)?;
            let matter = arguments.get("matter").and_then(Value::as_str)?;
            let bg = background_source(arguments).ok()?;
            let source = format!("cpt_export_mode_rhs({target}, {bg}, cpt_matter({matter}))");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "code": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        "cpt.multifield" => {
            let nfields = arguments.get("nfields").and_then(Value::as_i64)?;
            let source = format!("multifield_equations({nfields})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.boltzmann_bridge" => {
            let expr = eval_cpt_source(state, "boltzmann_bridge()").ok()?;
            cpt_pair_equations_json(&expr, &state.interner)
        }
        "cpt.boltzmann_bridge_export" => {
            let target = arguments.get("target").and_then(Value::as_str)?;
            let source = format!("boltzmann_bridge_export({target})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "code": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        "cpt.second_order_vector" => {
            let expr = eval_cpt_source(state, "second_order_einstein_vector()").ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.second_order_tensor" => {
            let expr = eval_cpt_source(state, "second_order_einstein_tensor()").ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.project_second_order_vector" => {
            let expr = eval_cpt_source(state, "project_second_order_vector_harmonics()").ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.project_second_order_tensor" => {
            let expr = eval_cpt_source(state, "project_second_order_tensor_harmonics()").ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.cubic_action" => {
            let channel = arguments.get("channel").and_then(Value::as_str)?;
            let source = format!("cubic_action({channel})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.cubic_kernel" => {
            let channel = arguments.get("channel").and_then(Value::as_str)?;
            let source = format!("cubic_kernel({channel})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.bispectrum_shape" => {
            let channel = arguments.get("channel").and_then(Value::as_str)?;
            let shape = arguments.get("shape").and_then(Value::as_str)?;
            let source = format!("bispectrum_shape({channel}, {shape})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.export_cubic_vertex" => {
            let channel = arguments.get("channel").and_then(Value::as_str)?;
            let target = arguments.get("target").and_then(Value::as_str)?;
            let source = format!("export_cubic_vertex({channel}, {target})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "code": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        "cpt.eft_model" => {
            let kind = arguments.get("kind").and_then(Value::as_str)?;
            let source = format!("eft_model({kind})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "unicode": ax_render::to_unicode(&expr, &state.interner),
                "latex": ax_render::to_latex(&expr, &state.interner),
            }))
        }
        "cpt.neutrino_hierarchy" => {
            let lmax = arguments.get("lmax").and_then(Value::as_i64)?;
            let gauge = arguments.get("gauge").and_then(Value::as_str)?;
            let closure = arguments.get("closure").and_then(Value::as_str)?;
            let source = format!("neutrino_hierarchy({lmax}, {gauge}, {closure})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_pair_equations_json(&expr, &state.interner)
        }
        "cpt.photon_hierarchy" => {
            let lmax = arguments.get("lmax").and_then(Value::as_i64)?;
            let gauge = arguments.get("gauge").and_then(Value::as_str)?;
            let closure = arguments.get("closure").and_then(Value::as_str)?;
            let source = format!("photon_hierarchy({lmax}, {gauge}, {closure})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_pair_equations_json(&expr, &state.interner)
        }
        "cpt.export_hierarchy" => {
            let target = arguments.get("target").and_then(Value::as_str)?;
            let species = arguments.get("species").and_then(Value::as_str)?;
            let lmax = arguments.get("lmax").and_then(Value::as_i64)?;
            let gauge = arguments.get("gauge").and_then(Value::as_str)?;
            let closure = arguments.get("closure").and_then(Value::as_str)?;
            let source =
                format!("export_hierarchy({target}, {species}, {lmax}, {gauge}, {closure})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "code": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        "cpt.parity_report" => {
            let expr = eval_cpt_source(state, "cpt_parity_report()").ok()?;
            Some(json!({
                "status": "ok",
                "report": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        "cpt.eft_stability" => {
            let kind = arguments.get("kind").and_then(Value::as_str)?;
            let source = format!("eft_stability({kind})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_named_exprs_json(&expr, &state.interner)
        }
        "cpt.eft_mode_equations" => {
            let kind = arguments.get("kind").and_then(Value::as_str)?;
            let source = format!("eft_mode_equations({kind})");
            let expr = eval_cpt_source(state, &source).ok()?;
            cpt_labelled_equations_json(&expr, &state.interner)
        }
        "cpt.eft_export_rhs" => {
            let kind = arguments.get("kind").and_then(Value::as_str)?;
            let target = arguments.get("target").and_then(Value::as_str)?;
            let source = format!("eft_export_rhs({kind}, {target})");
            let expr = eval_cpt_source(state, &source).ok()?;
            Some(json!({
                "status": "ok",
                "code": ax_render::to_unicode(&expr, &state.interner),
            }))
        }
        _ => None,
    };
    result
}

fn handle_tensor_symmetry_summary(arguments: &Value) -> Result<Value, String> {
    let expression = arguments
        .get("expression")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing expression".to_string())?;
    let symmetry = ax_syntax::parse_tableau_symmetry(expression)
        .map_err(|diagnostics| format_syntax_diagnostics(&diagnostics))?;
    let summary = ax_ai_proto::TensorSymmetrySummary::from(&symmetry);
    let summary_json = serde_json::to_string(&summary).map_err(|err| err.to_string())?;
    Ok(json!({
        "status": "ok",
        "summary_json": summary_json,
        "rendered_ascii": ax_render::render_tensor_symmetry_summary(&symmetry),
    }))
}

fn format_syntax_diagnostics(diagnostics: &[ax_syntax::Diagnostic]) -> String {
    diagnostics
        .iter()
        .map(|diag| diag.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
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
    fn qm_density_summary_reports_purity_and_trace() {
        let mut state = McpState::new();
        let rho0 = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[1,0],[0,0]]"}),
                5,
            ),
            "axioma_eval rho0",
        );

        let result = handle_tools_call_safe(
            &mut state,
            "axioma_qm_density_summary",
            &json!({"expr": rho0["expr_id"].clone()}),
            5,
        );

        assert_eq!(result["status"], "ok", "{result:?}");
        assert!(result["trace"].as_str().unwrap_or("").contains("1"), "{result:?}");
        assert!(
            result["purity"].as_str().unwrap_or("").contains("1"),
            "{result:?}"
        );
    }

    #[test]
    fn qm_partial_trace_returns_reduced_qubit() {
        let mut state = McpState::new();
        let rho_bell = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[1/2,0,0,1/2],[0,0,0,0],[0,0,0,0],[1/2,0,0,1/2]]"}),
                5,
            ),
            "axioma_eval rho_bell",
        );

        let result = handle_tools_call_safe(
            &mut state,
            "axioma_qm_partial_trace",
            &json!({
                "expr": rho_bell["expr_id"].clone(),
                "dims": [2, 2],
                "factor_index": 1
            }),
            5,
        );

        assert_eq!(result["status"], "ok", "{result:?}");
        let matrix = result["matrix"].to_string();
        assert!(matrix.contains("1/2"), "{result:?}");
    }

    #[test]
    fn qm_expectation_value_reports_one_for_pauli_z_on_zero_state() {
        let mut state = McpState::new();
        let z = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[1,0],[0,-1]]"}),
                5,
            ),
            "axioma_eval Z",
        );
        let rho0 = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[1,0],[0,0]]"}),
                5,
            ),
            "axioma_eval rho0",
        );

        let result = handle_tools_call_safe(
            &mut state,
            "axioma_qm_expectation_value",
            &json!({
                "operator_expr": z["expr_id"].clone(),
                "state_expr": rho0["expr_id"].clone()
            }),
            5,
        );

        assert_eq!(result["status"], "ok", "{result:?}");
        assert!(result["value"].as_str().unwrap_or("").contains("1"), "{result:?}");
    }

    #[test]
    fn qm_lindblad_steady_state_tool_returns_ground_state_for_amplitude_damping() {
        let mut state = McpState::new();
        let h = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[0,0],[0,0]]"}),
                5,
            ),
            "axioma_eval H",
        );
        let l = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[0,1],[0,0]]"}),
                5,
            ),
            "axioma_eval L",
        );

        let result = handle_tools_call_safe(
            &mut state,
            "axioma_qm_lindblad_steady_state",
            &json!({
                "hamiltonian_expr": h["expr_id"].clone(),
                "jump_exprs": [l["expr_id"].clone()]
            }),
            5,
        );

        assert_eq!(result["status"], "ok", "{result:?}");
        assert_eq!(result["matrix"], json!([["1", "0"], ["0", "0"]]), "{result:?}");
    }

    #[test]
    fn qm_lindblad_steady_state_tool_reports_solver_failure_for_zero_generator() {
        let mut state = McpState::new();
        let h = expect_ok(
            handle_tools_call_safe(
                &mut state,
                "axioma_eval",
                &json!({"code": "[[0,0],[0,0]]"}),
                5,
            ),
            "axioma_eval H",
        );

        let result = handle_tools_call_safe(
            &mut state,
            "axioma_qm_lindblad_steady_state",
            &json!({
                "hamiltonian_expr": h["expr_id"].clone(),
                "jump_exprs": []
            }),
            5,
        );

        assert_eq!(result["status"], "error", "{result:?}");
        assert_eq!(result["message"], "steady state solver failed", "{result:?}");
    }

    #[test]
    fn qm_tools_are_categorized_as_quantum() {
        let tools = tool_definitions();
        for name in [
            "axioma_qm_density_summary",
            "axioma_qm_partial_trace",
            "axioma_qm_expectation_value",
            "axioma_qm_lindblad_steady_state",
        ] {
            let tool = tools
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("missing tool definition for {name}"));
            assert_eq!(tool["category"], "quantum", "{tool:?}");
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

    #[test]
    fn tensor_symmetry_summary_tool_returns_summary() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "tensor.symmetry_summary",
            &json!({"expression": "tableau_symmetry([[2,1]], slots=[[0,1,2]])"}),
            5,
        );
        assert_eq!(result["status"], "ok", "{result:?}");
        let rendered = result["rendered_ascii"].as_str().unwrap_or("");
        assert!(
            rendered.contains("tableau[0]")
                || result["summary_json"]
                    .as_str()
                    .unwrap_or("")
                    .contains("\"tableaux\""),
            "{result:?}"
        );
    }

    #[test]
    fn tensor_symmetry_summary_tool_rejects_invalid_input() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "tensor.symmetry_summary",
            &json!({"expression": "tableau_symmetry([], slots=[])"}),
            5,
        );
        assert_eq!(result["status"], "error", "{result:?}");
    }

    #[test]
    fn tensor_symmetry_summary_tool_returns_structured_payload_exactly() {
        let payload = handle_tensor_symmetry_summary(
            &json!({"expression": "tableau_symmetry([[2,1]], slots=[[0,1,2]])"}),
        )
        .expect("summary payload");

        assert_eq!(payload["status"], "ok");
        assert_eq!(
            payload["rendered_ascii"],
            "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None"
        );
        assert_eq!(
            payload["summary_json"],
            "{\"tableaux\":[{\"shape\":[2,1],\"slots\":[0,1,2],\"label\":null,\"trace_free\":false,\"duality\":\"none\"}]}"
        );
    }

    #[test]
    fn cpt_linearized_einstein_tool_returns_four_labels() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.linearized_einstein",
            &json!({
                "order": 1,
                "time": "conformal",
                "curvature": "flat",
                "spatial_dim": 3,
                "gauge": "newtonian",
                "matter": "symbolic"
            }),
            5,
        );
        let labels = result["equations"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(labels.len(), 4, "{result:?}");
        assert!(labels.contains(&"00_constraint"), "{result:?}");
        assert!(labels.contains(&"ij_traceless"), "{result:?}");
    }

    #[test]
    fn cpt_mukhanov_sasaki_returns_nonempty_unicode_string_containing_k() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.mukhanov_sasaki",
            &json!({
                "time": "conformal",
                "curvature": "flat",
                "spatial_dim": 3,
                "matter": "canonical_scalar"
            }),
            5,
        );
        let unicode = result["unicode"].as_str().unwrap_or("");
        assert!(!unicode.is_empty(), "{result:?}");
        assert!(unicode.contains("k"), "{result:?}");
    }

    #[test]
    fn cpt_export_mode_rhs_tool_returns_code() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.export_mode_rhs",
            &json!({
                "target": "python",
                "time": "conformal",
                "curvature": "flat",
                "spatial_dim": 3,
                "matter": "canonical_scalar"
            }),
            5,
        );
        let code = result["code"].as_str().unwrap_or("");
        assert!(code.contains("def ms_rhs("), "{result:?}");
    }

    #[test]
    fn cpt_multifield_tool_returns_expected_label_for_two_fields() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.multifield",
            &json!({
                "nfields": 2
            }),
            5,
        );
        let labels = result["equations"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"multifield_curvature"), "{result:?}");
    }

    #[test]
    fn cpt_boltzmann_bridge_export_tool_returns_code() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.boltzmann_bridge_export",
            &json!({
                "target": "python"
            }),
            5,
        );
        let code = result["code"].as_str().unwrap_or("");
        assert!(code.contains("def rhs_0("), "{result:?}");
    }

    #[test]
    fn cpt_cubic_kernel_tool_returns_nonempty_unicode() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.cubic_kernel",
            &json!({
                "channel": "scalar_scalar_scalar"
            }),
            5,
        );
        let unicode = result["unicode"].as_str().unwrap_or("");
        assert!(!unicode.is_empty(), "{result:?}");
        assert!(
            unicode.contains("k₁") || unicode.contains("k1"),
            "{result:?}"
        );
    }

    #[test]
    fn cpt_export_cubic_vertex_tool_returns_code() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.export_cubic_vertex",
            &json!({
                "channel": "scalar_scalar_scalar",
                "target": "python"
            }),
            5,
        );
        let code = result["code"].as_str().unwrap_or("");
        assert!(code.contains("def cubic_vertex("), "{result:?}");
    }

    #[test]
    fn cpt_eft_mode_equations_tool_returns_two_equations() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.eft_mode_equations",
            &json!({
                "kind": "canonical"
            }),
            5,
        );
        let equations = result["equations"].as_array().cloned().unwrap_or_default();
        assert_eq!(equations.len(), 2, "{result:?}");
    }

    #[test]
    fn cpt_eft_export_rhs_tool_returns_code() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.eft_export_rhs",
            &json!({
                "kind": "horndeski_like",
                "target": "python"
            }),
            5,
        );
        let code = result["code"].as_str().unwrap_or("");
        assert!(code.contains("def eft_scalar_rhs("), "{result:?}");
    }

    #[test]
    fn cpt_neutrino_hierarchy_tool_returns_equations() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(
            &mut state,
            "cpt.neutrino_hierarchy",
            &json!({
                "lmax": 3,
                "gauge": "newtonian",
                "closure": "power_law"
            }),
            5,
        );
        let equations = result["equations"].as_array().cloned().unwrap_or_default();
        assert_eq!(equations.len(), 4, "{result:?}");
    }

    #[test]
    fn cpt_parity_report_tool_returns_report_payload() {
        let mut state = McpState::new();
        let result = handle_tools_call_safe(&mut state, "cpt.parity_report", &json!({}), 5);
        let report = result["report"].as_str().unwrap_or("");
        assert!(
            report.contains("ma_bertschinger_scalar_labels"),
            "{result:?}"
        );
    }

    #[test]
    fn mcp_cpt_linearized_einstein_matches_eval_labels() {
        let mut state = McpState::new();
        let mcp_result = handle_tools_call_safe(
            &mut state,
            "cpt.linearized_einstein",
            &json!({
                "order": 1,
                "time": "conformal",
                "curvature": "flat",
                "spatial_dim": 3,
                "gauge": "newtonian",
                "matter": "symbolic"
            }),
            5,
        );
        let eval_expr = eval_cpt_source(
            &mut state,
            "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic))",
        )
        .expect("eval cpt linearized einstein");

        let mcp_labels = mcp_result["equations"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("label").and_then(Value::as_str))
            .map(str::to_string)
            .collect::<Vec<_>>();
        let eval_labels = cpt_labelled_equations_json(&eval_expr, &state.interner)
            .and_then(|payload| payload["equations"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entry| {
                entry
                    .get("label")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect::<Vec<_>>();

        assert_eq!(mcp_labels, eval_labels);
    }

    #[test]
    fn mcp_cpt_multifield_matches_eval_labels() {
        let mut state = McpState::new();
        let mcp_result = handle_tools_call_safe(
            &mut state,
            "cpt.multifield",
            &json!({
                "nfields": 2
            }),
            5,
        );
        let eval_expr = eval_cpt_source(&mut state, "multifield_equations(2)")
            .expect("eval multifield equations");

        let mcp_labels = mcp_result["equations"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("label").and_then(Value::as_str))
            .map(str::to_string)
            .collect::<Vec<_>>();
        let eval_labels = cpt_labelled_equations_json(&eval_expr, &state.interner)
            .and_then(|payload| payload["equations"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entry| {
                entry
                    .get("label")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect::<Vec<_>>();

        assert_eq!(mcp_labels, eval_labels);
    }

    #[test]
    fn mcp_cpt_export_code_is_nonempty_and_deterministic() {
        let mut state = McpState::new();
        let first = handle_tools_call_safe(
            &mut state,
            "cpt.boltzmann_bridge_export",
            &json!({
                "target": "python"
            }),
            5,
        );
        let second = handle_tools_call_safe(
            &mut state,
            "cpt.boltzmann_bridge_export",
            &json!({
                "target": "python"
            }),
            5,
        );

        let first_code = first["code"].as_str().unwrap_or("");
        let second_code = second["code"].as_str().unwrap_or("");

        assert!(!first_code.is_empty(), "{first:?}");
        assert_eq!(first_code, second_code);
    }
}
