#![forbid(unsafe_code)]

use anyhow::{anyhow, Context, Result};
use ax_ai_proto::TensorSymmetrySummary;
use ax_plugin_api::{
    PluginRequest, PluginResponse, SparseEigenRequest, SparseEigenResponse, SymmetrySummaryResponse,
};
use num_traits::ToPrimitive;
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

fn expr_to_f64(expr: &ax_ir::Expr) -> Option<f64> {
    match expr {
        ax_ir::Expr::Int(n) => n.to_f64(),
        ax_ir::Expr::Rational(r) => Some(r.numer().to_f64()? / r.denom().to_f64()?),
        ax_ir::Expr::Float(f) => Some(*f),
        _ => None,
    }
}

fn numeric_matrix_error() -> anyhow::Error {
    anyhow!("sparse_eigenpairs_via_plugin requires a purely numeric matrix")
}

fn plugin_diagnostics_error(diagnostics: &[ax_plugin_api::PluginDiag]) -> anyhow::Error {
    anyhow!(
        "{}",
        diagnostics
            .iter()
            .map(|diag| diag.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    )
}

fn expr_to_complex_pair(expr: &ax_ir::Expr) -> anyhow::Result<(f64, f64)> {
    match expr {
        ax_ir::Expr::Int(n) => n
            .to_f64()
            .map(|value| (value, 0.0))
            .ok_or_else(numeric_matrix_error),
        ax_ir::Expr::Rational(r) => {
            let re = r
                .numer()
                .to_f64()
                .zip(r.denom().to_f64())
                .map(|(numer, denom)| numer / denom);
            re.map(|value| (value, 0.0))
                .ok_or_else(numeric_matrix_error)
        }
        ax_ir::Expr::Float(f) => Ok((*f, 0.0)),
        ax_ir::Expr::Complex(re, im) => {
            let re = expr_to_f64(re).ok_or_else(numeric_matrix_error)?;
            let im = expr_to_f64(im).ok_or_else(numeric_matrix_error)?;
            Ok((re, im))
        }
        _ => Err(numeric_matrix_error()),
    }
}

pub fn sparse_eigenpairs_via_plugin(
    plugin: &WasmPlugin,
    plugin_name: &str,
    matrix: &[Vec<ax_ir::Expr>],
    k: usize,
    which: &str,
) -> anyhow::Result<SparseEigenResponse> {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    if !matrix.iter().all(|row| row.len() == cols) || rows != cols {
        return Err(anyhow!(
            "sparse_eigenpairs_via_plugin expects a square matrix"
        ));
    }

    let matrix = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(expr_to_complex_pair)
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let req = PluginRequest {
        plugin: plugin_name.to_string(),
        op: "sparse_eigenpairs".to_string(),
        args: serde_json::to_value(SparseEigenRequest {
            matrix,
            k,
            which: which.to_string(),
        })
        .context("serialize SparseEigenRequest")?,
    };

    let response = plugin.call(&req)?;
    if !response.ok {
        return Err(plugin_diagnostics_error(&response.diagnostics));
    }

    serde_json::from_value(response.result).context("deserialize SparseEigenResponse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_to_complex_pair_accepts_real_and_complex_numeric_exprs() {
        assert_eq!(
            expr_to_complex_pair(&ax_ir::Expr::Int(2.into())).unwrap(),
            (2.0, 0.0)
        );
        assert_eq!(
            expr_to_complex_pair(&ax_ir::Expr::Complex(
                Box::new(ax_ir::Expr::Int(1.into())),
                Box::new(ax_ir::Expr::Int((-1).into())),
            ))
            .unwrap(),
            (1.0, -1.0)
        );
    }

    #[test]
    fn expr_to_complex_pair_rejects_symbolic_expr() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let err = expr_to_complex_pair(&ax_ir::Expr::Sym(x)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "sparse_eigenpairs_via_plugin requires a purely numeric matrix"
        );
    }
}
