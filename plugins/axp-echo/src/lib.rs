use ax_plugin_api::{PluginDiag, PluginRequest, PluginResponse};

#[no_mangle]
pub extern "C" fn axioma_alloc(len: i32) -> i32 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr as i32
}

#[no_mangle]
pub extern "C" fn axioma_free(ptr: i32, len: i32) {
    unsafe {
        let _ = Vec::<u8>::from_raw_parts(ptr as *mut u8, len as usize, len as usize);
    }
}

// Response format: [u32 len LE][bytes...]
fn pack_response_bytes(resp: &PluginResponse) -> Vec<u8> {
    let payload = serde_json::to_vec(resp).unwrap();
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

#[no_mangle]
pub extern "C" fn axioma_entry(req_ptr: i32, req_len: i32) -> i32 {
    let req_bytes = unsafe { std::slice::from_raw_parts(req_ptr as *const u8, req_len as usize) };

    let resp = match serde_json::from_slice::<PluginRequest>(req_bytes) {
        Ok(req) => PluginResponse {
            ok: true,
            result: serde_json::json!({
                "echo": req.args,
                "op": req.op,
                "plugin": req.plugin
            }),
            diagnostics: vec![],
        },
        Err(e) => PluginResponse {
            ok: false,
            result: serde_json::json!(null),
            diagnostics: vec![PluginDiag {
                level: "error".to_string(),
                message: format!("bad request json: {e}"),
            }],
        },
    };

    let out = pack_response_bytes(&resp);
    let ptr = out.as_ptr() as i32;
    std::mem::forget(out);
    ptr
}
