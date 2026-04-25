use ax_plugin_api::{
    PluginDiag, PluginRequest, PluginResponse, SparseEigenRequest, SparseEigenResponse,
    SparseEigenpair,
};

#[no_mangle]
pub extern "C" fn axioma_alloc(len: i32) -> i32 {
    if len < 0 {
        return 0;
    }
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr as i32
}

#[no_mangle]
pub extern "C" fn axioma_free(ptr: i32, len: i32) {
    if ptr == 0 || len < 0 {
        return;
    }
    unsafe {
        let _ = Vec::<u8>::from_raw_parts(ptr as *mut u8, len as usize, len as usize);
    }
}

#[no_mangle]
pub extern "C" fn axioma_entry(req_ptr: i32, req_len: i32) -> i32 {
    let response = if req_ptr == 0 || req_len < 0 {
        error_response("invalid request pointer or length")
    } else {
        let req_bytes =
            unsafe { std::slice::from_raw_parts(req_ptr as *const u8, req_len as usize) };
        match serde_json::from_slice::<PluginRequest>(req_bytes) {
            Ok(req) => handle_plugin_request(req),
            Err(err) => error_response(format!("bad request json: {err}")),
        }
    };

    let out = match pack_response_bytes(&response) {
        Ok(out) => out,
        Err(err) => {
            let fallback = error_response(format!("failed to encode plugin response: {err}"));
            match pack_response_bytes(&fallback) {
                Ok(out) => out,
                Err(_) => return 0,
            }
        }
    };
    let ptr = out.as_ptr() as i32;
    std::mem::forget(out);
    ptr
}

pub fn handle_plugin_request(req: PluginRequest) -> PluginResponse {
    match req.op.as_str() {
        "sparse_eigenpairs" => handle_sparse_eigenpairs(req.args),
        other => error_response(format!("unsupported op: {other}")),
    }
}

fn handle_sparse_eigenpairs(args: serde_json::Value) -> PluginResponse {
    let request = match serde_json::from_value::<SparseEigenRequest>(args) {
        Ok(request) => request,
        Err(err) => return error_response(format!("invalid sparse_eigenpairs args: {err}")),
    };

    match solve_diagonal_eigenpairs(&request) {
        Ok(response) => value_response(response),
        Err(err) => error_response(err),
    }
}

fn solve_diagonal_eigenpairs(request: &SparseEigenRequest) -> Result<SparseEigenResponse, String> {
    validate_square_matrix(&request.matrix)?;
    if request.k == 0 {
        return Ok(SparseEigenResponse {
            eigenpairs: Vec::new(),
            converged: true,
        });
    }

    let n = request.matrix.len();
    let zero_tol = 1.0e-12;
    for (row_index, row) in request.matrix.iter().enumerate() {
        for (col_index, entry) in row.iter().enumerate() {
            if row_index != col_index && complex_norm(*entry) > zero_tol {
                return Err(
                    "axp-sparse-demo sparse_eigenpairs currently supports diagonal matrices"
                        .to_string(),
                );
            }
        }
    }

    let mut eigenpairs = request
        .matrix
        .iter()
        .enumerate()
        .map(|(index, row)| SparseEigenpair {
            eigenvalue: row[index],
            eigenvector: basis_vector(n, index),
        })
        .collect::<Vec<_>>();

    sort_eigenpairs(&mut eigenpairs, &request.which);
    eigenpairs.truncate(request.k.min(n));

    Ok(SparseEigenResponse {
        eigenpairs,
        converged: true,
    })
}

fn validate_square_matrix(matrix: &[Vec<(f64, f64)>]) -> Result<(), String> {
    let n = matrix.len();
    if n == 0 {
        return Err("sparse_eigenpairs expects a non-empty square matrix".to_string());
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err("sparse_eigenpairs expects a square matrix".to_string());
    }
    if matrix
        .iter()
        .flatten()
        .any(|(re, im)| !re.is_finite() || !im.is_finite())
    {
        return Err("sparse_eigenpairs expects finite numeric entries".to_string());
    }
    Ok(())
}

fn basis_vector(n: usize, index: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| if i == index { (1.0, 0.0) } else { (0.0, 0.0) })
        .collect()
}

fn sort_eigenpairs(eigenpairs: &mut [SparseEigenpair], which: &str) {
    match which {
        "SM" => eigenpairs.sort_by(|a, b| {
            complex_norm(a.eigenvalue)
                .total_cmp(&complex_norm(b.eigenvalue))
                .then_with(|| a.eigenvalue.0.total_cmp(&b.eigenvalue.0))
        }),
        "LR" => eigenpairs.sort_by(|a, b| b.eigenvalue.0.total_cmp(&a.eigenvalue.0)),
        "SR" => eigenpairs.sort_by(|a, b| a.eigenvalue.0.total_cmp(&b.eigenvalue.0)),
        "LI" => eigenpairs.sort_by(|a, b| b.eigenvalue.1.total_cmp(&a.eigenvalue.1)),
        "SI" => eigenpairs.sort_by(|a, b| a.eigenvalue.1.total_cmp(&b.eigenvalue.1)),
        _ => eigenpairs.sort_by(|a, b| {
            complex_norm(b.eigenvalue)
                .total_cmp(&complex_norm(a.eigenvalue))
                .then_with(|| a.eigenvalue.0.total_cmp(&b.eigenvalue.0))
        }),
    }
}

fn complex_norm((re, im): (f64, f64)) -> f64 {
    re.hypot(im)
}

fn value_response<T: serde::Serialize>(value: T) -> PluginResponse {
    match serde_json::to_value(value) {
        Ok(result) => PluginResponse {
            ok: true,
            result,
            diagnostics: Vec::new(),
        },
        Err(err) => error_response(format!(
            "failed to encode sparse_eigenpairs response: {err}"
        )),
    }
}

fn error_response(message: impl Into<String>) -> PluginResponse {
    PluginResponse {
        ok: false,
        result: serde_json::Value::Null,
        diagnostics: vec![PluginDiag {
            level: "error".to_string(),
            message: message.into(),
        }],
    }
}

fn pack_response_bytes(resp: &PluginResponse) -> Result<Vec<u8>, serde_json::Error> {
    let payload = serde_json::to_vec(resp)?;
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_eigenpairs_returns_diagonal_eigenvalues() {
        let req = PluginRequest {
            plugin: "axp-sparse-demo".to_string(),
            op: "sparse_eigenpairs".to_string(),
            args: serde_json::json!({
                "matrix": [[ [1.0, 0.0], [0.0, 0.0] ], [ [0.0, 0.0], [2.0, 0.0] ]],
                "k": 2,
                "which": "SM"
            }),
        };

        let resp = handle_plugin_request(req);
        assert!(resp.ok, "{:?}", resp.diagnostics);
        let decoded: SparseEigenResponse =
            serde_json::from_value(resp.result).expect("decode sparse eigen response");
        let mut values = decoded
            .eigenpairs
            .into_iter()
            .map(|pair| pair.eigenvalue)
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(values, vec![(1.0, 0.0), (2.0, 0.0)]);
    }
}
