#![forbid(unsafe_code)]

use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    let interner = ax_ir::Interner::new();

    loop {
        match read_message(&mut reader) {
            Ok(msg) => {
                if let Some(response) = handle_message(&msg, &interner) {
                    write_message(&mut writer, &response);
                }
            }
            Err(_) => break,
        }
    }
}

fn read_message(reader: &mut impl BufRead) -> io::Result<serde_json::Value> {
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading LSP headers",
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(len_str) = trimmed.strip_prefix("Content-Length: ") {
            content_length = len_str.parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_message(writer: &mut impl Write, msg: &serde_json::Value) {
    if let Ok(body) = serde_json::to_string(msg) {
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let _ = writer.write_all(header.as_bytes());
        let _ = writer.write_all(body.as_bytes());
        let _ = writer.flush();
    }
}

fn offset_to_position(text: &str, offset: usize) -> (usize, usize) {
    let clamped = offset.min(text.len());
    let prefix = &text[..clamped];
    let line = prefix.matches('\n').count();
    let character = clamped - prefix.rfind('\n').map_or(0, |p| p + 1);
    (line, character)
}

fn handle_message(msg: &serde_json::Value, interner: &ax_ir::Interner) -> Option<serde_json::Value> {
    let method = msg.get("method")?.as_str()?;

    match method {
        "initialize" => {
            let id = msg.get("id")?;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "completionProvider": {
                            "triggerCharacters": [".", "(", "["]
                        },
                        "diagnosticProvider": {
                            "interFileDependencies": false,
                            "workspaceDiagnostics": false
                        }
                    },
                    "serverInfo": {
                        "name": "axioma-lsp",
                        "version": "0.1.0"
                    }
                }
            }))
        }
        "initialized" => None,
        "textDocument/didOpen" | "textDocument/didChange" => {
            let params = msg.get("params")?;
            let text_doc = params.get("textDocument")?;
            let uri = text_doc.get("uri")?.as_str()?;

            let text = if method == "textDocument/didOpen" {
                text_doc.get("text")?.as_str()?
            } else {
                let changes = params.get("contentChanges")?.as_array()?;
                changes.first()?.get("text")?.as_str()?
            };

            let lowered = ax_core_ir::lower(text, interner);
            let diagnostics: Vec<serde_json::Value> = lowered
                .errors
                .iter()
                .map(|err| {
                    let (start_line, start_col) = offset_to_position(text, err.span.start);
                    let (end_line, end_col) = offset_to_position(text, err.span.end);
                    serde_json::json!({
                        "range": {
                            "start": { "line": start_line, "character": start_col },
                            "end": { "line": end_line, "character": end_col }
                        },
                        "severity": 1,
                        "message": err.message
                    })
                })
                .collect();

            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "diagnostics": diagnostics
                }
            }))
        }
        "textDocument/hover" => {
            let id = msg.get("id")?;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": null
            }))
        }
        "textDocument/completion" => {
            let id = msg.get("id")?;
            let items: Vec<serde_json::Value> = vec![
                "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
                "Gamma", "Delta", "Theta", "Lambda", "Sigma", "Phi", "Psi", "Omega",
                "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "diff", "integrate",
                "series", "solve", "expand", "simplify", "det", "inv", "transpose",
                "christoffel", "riemann", "ricci", "einstein", "kretschner",
                "metric", "diag", "ket", "bra", "commutator", "pauli_x", "pauli_y", "pauli_z",
                "plot", "dsolve", "rk4", "rewrite",
            ]
            .iter()
            .map(|name| {
                serde_json::json!({
                    "label": name,
                    "kind": if name.chars().next().unwrap_or('a').is_lowercase() && name.len() > 2 { 3 } else { 6 },
                })
            })
            .collect();

            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": items
            }))
        }
        "shutdown" => {
            let id = msg.get("id")?;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": null
            }))
        }
        "exit" => {
            std::process::exit(0);
        }
        _ => None,
    }
}
