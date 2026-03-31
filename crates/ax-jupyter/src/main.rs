use anyhow::{anyhow, Result};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

#[derive(Deserialize)]
struct ConnectionInfo {
    shell_port: u16,
    iopub_port: u16,
    stdin_port: u16,
    control_port: u16,
    hb_port: u16,
    ip: String,
    transport: String,
    key: String,
    signature_scheme: String,
}

#[derive(Clone, Debug)]
struct JupyterMessage {
    identities: Vec<Vec<u8>>,
    header: Value,
    parent_header: Value,
    metadata: Value,
    content: Value,
}

fn endpoint(conn: &ConnectionInfo, port: u16) -> String {
    format!("{}://{}:{}", conn.transport, conn.ip, port)
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

fn recv_jupyter_message(socket: &zmq::Socket, key: &str) -> Result<JupyterMessage> {
    let frames = socket.recv_multipart(0)?;
    let delimiter = frames
        .iter()
        .position(|frame| frame == b"<IDS|MSG>")
        .ok_or_else(|| anyhow!("missing Jupyter delimiter"))?;

    if frames.len() < delimiter + 6 {
        return Err(anyhow!("incomplete Jupyter message"));
    }

    let identities = frames[..delimiter].to_vec();
    let signature = String::from_utf8(frames[delimiter + 1].clone())
        .map_err(|e| anyhow!("invalid signature frame: {e}"))?;
    let header_bytes = &frames[delimiter + 2];
    let parent_header_bytes = &frames[delimiter + 3];
    let metadata_bytes = &frames[delimiter + 4];
    let content_bytes = &frames[delimiter + 5];

    if !key.is_empty() {
        let expected = compute_hmac(
            key,
            &[header_bytes, parent_header_bytes, metadata_bytes, content_bytes],
        )?;
        if expected != signature {
            return Err(anyhow!("invalid Jupyter signature"));
        }
    }

    Ok(JupyterMessage {
        identities,
        header: serde_json::from_slice(header_bytes)?,
        parent_header: serde_json::from_slice(parent_header_bytes)?,
        metadata: serde_json::from_slice(metadata_bytes)?,
        content: serde_json::from_slice(content_bytes)?,
    })
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

fn make_message(parent: &JupyterMessage, msg_type: &str, content: Value) -> JupyterMessage {
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

fn make_reply(parent: &JupyterMessage, msg_type: &str, content: Value) -> JupyterMessage {
    make_message(parent, msg_type, content)
}

fn send_status(iopub: &zmq::Socket, parent: &JupyterMessage, state: &str, key: &str) -> Result<()> {
    let mut msg = make_message(parent, "status", json!({ "execution_state": state }));
    msg.identities = Vec::new();
    send_jupyter_message(iopub, &msg, key)
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

    let _ = ax_eval::apply_set_convention(result, env);
}

fn evaluate_code(
    code: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Result<Option<ax_ir::Expr>, String> {
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

    let mut last_result = None;
    for expr in &lowered.exprs {
        if let ax_ir::Expr::Import(path) = expr {
            let search_paths = vec![std::env::current_dir().unwrap_or_default()];
            let file_path = ax_eval::resolve_import(path, interner, &search_paths).ok_or_else(|| {
                format!(
                    "import not found: {}",
                    path.iter()
                        .map(|s| interner.resolve(*s))
                        .collect::<Vec<_>>()
                        .join(".")
                )
            })?;
            let imported = std::fs::read_to_string(&file_path)
                .map_err(|e| format!("failed to read import {}: {e}", file_path.display()))?;
            let _ = evaluate_code(&imported, env, interner)?;
            last_result = Some(ax_ir::Expr::zero());
            continue;
        }

        let result = ax_eval::eval(expr, env, interner);
        apply_result_side_effects(expr, &result, env, interner);
        last_result = Some(match expr {
            ax_ir::Expr::Let(name, val, body)
                if matches!(body.as_ref(), ax_ir::Expr::Sym(s) if *s == *name) =>
            {
                ax_eval::eval(val, env, interner)
            }
            _ => result,
        });
    }

    Ok(last_result)
}

fn handle_kernel_request(
    socket: &zmq::Socket,
    iopub: &zmq::Socket,
    raw: &JupyterMessage,
    conn: &ConnectionInfo,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    execution_count: &mut u64,
) -> Result<bool> {
    let msg_type = raw
        .header
        .get("msg_type")
        .and_then(Value::as_str)
        .unwrap_or("");

    match msg_type {
        "kernel_info_request" => {
            let reply = make_reply(
                raw,
                "kernel_info_reply",
                json!({
                    "protocol_version": "5.4",
                    "implementation": "axioma",
                    "implementation_version": "0.1.0",
                    "language_info": {
                        "name": "axioma",
                        "version": "0.1.0",
                        "mimetype": "text/x-axioma",
                        "file_extension": ".ax"
                    },
                    "banner": "Axioma — Scientific Computing Language for Physicists",
                    "status": "ok"
                }),
            );
            send_jupyter_message(socket, &reply, &conn.key)?;
        }
        "execute_request" => {
            *execution_count += 1;
            let code = raw
                .content
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("");

            send_status(iopub, raw, "busy", &conn.key)?;

            let execute_input = JupyterMessage {
                identities: Vec::new(),
                ..make_message(
                    raw,
                    "execute_input",
                    json!({
                        "code": code,
                        "execution_count": *execution_count
                    }),
                )
            };
            send_jupyter_message(iopub, &execute_input, &conn.key)?;

            let eval_result = evaluate_code(code, env, interner);
            let (output, error) = match eval_result {
                Ok(Some(result)) => {
                    let latex = ax_render::to_latex(&result, interner);
                    let unicode = ax_render::to_unicode(&result, interner);
                    (Some((latex, unicode)), None)
                }
                Ok(None) => (None, None),
                Err(err) => (None, Some(err)),
            };

            if let Some((latex, unicode)) = &output {
                let mut display = make_message(
                    raw,
                    "display_data",
                    json!({
                        "data": {
                            "text/latex": format!("$${}$$", latex),
                            "text/plain": unicode
                        },
                        "metadata": {}
                    }),
                );
                display.identities = Vec::new();
                send_jupyter_message(iopub, &display, &conn.key)?;
            }

            if let Some(err_msg) = &error {
                let mut err = make_message(
                    raw,
                    "error",
                    json!({
                        "ename": "EvalError",
                        "evalue": err_msg,
                        "traceback": [err_msg]
                    }),
                );
                err.identities = Vec::new();
                send_jupyter_message(iopub, &err, &conn.key)?;
            }

            let reply_content = if let Some(err_msg) = error {
                json!({
                    "status": "error",
                    "execution_count": *execution_count,
                    "ename": "EvalError",
                    "evalue": err_msg,
                    "traceback": [err_msg]
                })
            } else {
                json!({
                    "status": "ok",
                    "execution_count": *execution_count,
                    "payload": [],
                    "user_expressions": {}
                })
            };
            let reply = make_reply(raw, "execute_reply", reply_content);
            send_jupyter_message(socket, &reply, &conn.key)?;
            send_status(iopub, raw, "idle", &conn.key)?;
        }
        "shutdown_request" => {
            let reply = make_reply(raw, "shutdown_reply", json!({ "restart": false }));
            send_jupyter_message(socket, &reply, &conn.key)?;
            return Ok(false);
        }
        _ => {}
    }

    Ok(true)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: axioma-jupyter <connection-file>");
        std::process::exit(1);
    }

    let conn_file = std::fs::read_to_string(&args[1])?;
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

    let _stdin = ctx.socket(zmq::ROUTER)?;
    _stdin.bind(&endpoint(&conn, conn.stdin_port))?;

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

    let interner = ax_ir::Interner::new();
    let mut env = ax_eval::Env::new();
    let mut execution_count = 0u64;

    loop {
        let mut items = [
            shell.as_poll_item(zmq::POLLIN),
            control.as_poll_item(zmq::POLLIN),
        ];
        zmq::poll(&mut items, -1)?;

        if items[0].is_readable() {
            let raw = recv_jupyter_message(&shell, &conn.key)?;
            if !handle_kernel_request(
                &shell,
                &iopub,
                &raw,
                &conn,
                &mut env,
                &interner,
                &mut execution_count,
            )? {
                break;
            }
        }

        if items[1].is_readable() {
            let raw = recv_jupyter_message(&control, &conn.key)?;
            if !handle_kernel_request(
                &control,
                &iopub,
                &raw,
                &conn,
                &mut env,
                &interner,
                &mut execution_count,
            )? {
                break;
            }
        }
    }

    Ok(())
}
