#![forbid(unsafe_code)]

use anyhow::{anyhow, Context, Result};
use ax_ai_proto::TensorSymmetrySummary;
use ax_plugin_api::{PluginRequest, PluginResponse, SymmetrySummaryResponse};
use std::path::Path;
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

pub struct WasmPlugin {
    engine: Engine,
    module: Module,
}

impl WasmPlugin {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, path.as_ref())
            .with_context(|| format!("failed to load wasm module {}", path.as_ref().display()))?;
        Ok(Self { engine, module })
    }

    pub fn call(&self, req: &PluginRequest) -> Result<PluginResponse> {
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &self.module, &[])
            .context("failed to instantiate wasm module")?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("plugin missing exported memory"))?;

        let alloc: TypedFunc<i32, i32> = instance
            .get_typed_func(&mut store, "axioma_alloc")
            .context("missing export axioma_alloc(i32)->i32")?;

        let entry: TypedFunc<(i32, i32), i32> = instance
            .get_typed_func(&mut store, "axioma_entry")
            .context("missing export axioma_entry(i32,i32)->i32")?;

        let free: Option<TypedFunc<(i32, i32), ()>> =
            instance.get_typed_func(&mut store, "axioma_free").ok();

        let req_bytes = serde_json::to_vec(req).context("serialize PluginRequest")?;
        let req_len = i32::try_from(req_bytes.len()).context("request too large")?;
        let req_ptr = alloc
            .call(&mut store, req_len)
            .context("axioma_alloc failed")?;

        // write request bytes into guest memory
        let mem = memory.data_mut(&mut store);
        let start = req_ptr as usize;
        let end = start + req_bytes.len();
        if end > mem.len() {
            return Err(anyhow!("guest memory too small for request"));
        }
        mem[start..end].copy_from_slice(&req_bytes);

        // call entrypoint; returns pointer to response blob [u32 len][bytes...]
        let resp_ptr = entry
            .call(&mut store, (req_ptr, req_len))
            .context("axioma_entry failed")? as usize;

        let mem = memory.data(&store);
        if resp_ptr + 4 > mem.len() {
            return Err(anyhow!("invalid response pointer"));
        }
        let len = u32::from_le_bytes(mem[resp_ptr..resp_ptr + 4].try_into().unwrap()) as usize;
        let payload_start = resp_ptr + 4;
        let payload_end = payload_start + len;
        if payload_end > mem.len() {
            return Err(anyhow!("invalid response length"));
        }

        let resp_bytes = &mem[payload_start..payload_end];
        let resp: PluginResponse =
            serde_json::from_slice(resp_bytes).context("deserialize PluginResponse")?;

        if let Some(free) = free {
            let _ = free.call(&mut store, (req_ptr, req_len));
            let _ = free.call(&mut store, (resp_ptr as i32, (len + 4) as i32));
        }

        Ok(resp)
    }
}

pub fn summarize_symmetry_for_expr(expr: &str) -> anyhow::Result<SymmetrySummaryResponse> {
    summarize_symmetry_for_expr_impl(expr).context("failed to summarize symmetry for expression")
}

fn summarize_symmetry_for_expr_impl(expr: &str) -> Result<SymmetrySummaryResponse> {
    let symmetry = ax_syntax::parse_tableau_symmetry(expr)
        .map_err(|diagnostics| anyhow!(diagnostics_to_message(&diagnostics)))?;
    let summary = TensorSymmetrySummary::from(&symmetry);
    let summary_json = serde_json::to_string(&summary)?;
    Ok(SymmetrySummaryResponse {
        summary_json,
        rendered_ascii: ax_render::render_tensor_symmetry_summary(&symmetry),
    })
}

fn diagnostics_to_message(diagnostics: &[ax_syntax::Diagnostic]) -> String {
    diagnostics
        .iter()
        .map(|diag| diag.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}
