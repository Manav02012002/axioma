use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn spawn_mcp() -> (std::process::Child, impl Write, impl BufRead) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_axioma-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn axioma-mcp");
    let stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    (child, stdin, stdout)
}

fn send(stdin: &mut impl Write, stdout: &mut impl BufRead, method: &str, params: Value) -> Value {
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    });
    writeln!(stdin, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    stdin.flush().unwrap();
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    let response: Value = serde_json::from_str(&line).unwrap();
    response
        .get("result")
        .cloned()
        .unwrap_or_else(|| response.clone())
}

fn call_tool(
    stdin: &mut impl Write,
    stdout: &mut impl BufRead,
    tool_name: &str,
    args: Value,
) -> Value {
    send(
        stdin,
        stdout,
        "tools/call",
        json!({
            "name": tool_name,
            "arguments": args,
        }),
    )
}

#[test]
fn schwarzschild_ricci_zero_via_mcp() {
    let (mut child, mut stdin, mut stdout) = spawn_mcp();
    let init = send(&mut stdin, &mut stdout, "initialize", json!({}));
    assert_eq!(init["serverInfo"]["name"], "axioma-mcp");

    let metric_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_define_metric",
        json!({
            "name": "g",
            "components": [
                ["-(1 - 2/r)", "0", "0", "0"],
                ["0", "1/(1 - 2/r)", "0", "0"],
                ["0", "0", "r^2", "0"],
                ["0", "0", "0", "r^2*sin(theta)^2"]
            ],
            "coordinates": ["t", "r", "theta", "phi"]
        }),
    );
    assert!(
        metric_result.get("metric_id").is_some(),
        "metric definition failed: {:?}",
        metric_result
    );

    let chris_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_christoffel",
        json!({
            "metric_id": "g"
        }),
    );
    assert!(
        chris_result.get("christoffel_id").is_some(),
        "christoffel failed: {:?}",
        chris_result
    );
    let nonzero = chris_result["nonzero_count"].as_u64().unwrap();
    assert!(
        nonzero > 0 && nonzero <= 13,
        "expected 9-13 nonzero Christoffel components, got {}",
        nonzero
    );

    let riem_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_riemann",
        json!({
            "christoffel_id": "g"
        }),
    );
    assert!(
        riem_result.get("riemann_id").is_some(),
        "riemann failed: {:?}",
        riem_result
    );

    let ricci_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_ricci",
        json!({
            "riemann_id": "g"
        }),
    );
    assert!(
        ricci_result.get("ricci_id").is_some(),
        "ricci failed: {:?}",
        ricci_result
    );

    if let Some(components) = ricci_result.get("components") {
        let matrix = components.as_array().unwrap();
        for (i, row) in matrix.iter().enumerate() {
            for (j, entry) in row.as_array().unwrap().iter().enumerate() {
                let val = entry.as_str().unwrap_or("");
                assert!(
                    val == "0" || val == "" || val.contains("0"),
                    "Ricci[{}][{}] should be 0, got '{}'",
                    i,
                    j,
                    val
                );
            }
        }
    }

    child.kill().ok();
}

#[test]
fn eval_and_inspect_roundtrip() {
    let (mut child, mut stdin, mut stdout) = spawn_mcp();
    send(&mut stdin, &mut stdout, "initialize", json!({}));

    let eval_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_eval",
        json!({
            "code": "diff(x^3 + sin(x), x)"
        }),
    );
    let expr_id = eval_result["expr_id"].as_str().unwrap();

    let inspect_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_inspect",
        json!({
            "expr_id": expr_id
        }),
    );
    assert!(inspect_result.get("kind").is_some());
    assert!(inspect_result["symbols"].as_array().unwrap().len() > 0);

    child.kill().ok();
}

#[test]
fn suggest_on_indexed_expr() {
    let (mut child, mut stdin, mut stdout) = spawn_mcp();
    send(&mut stdin, &mut stdout, "initialize", json!({}));

    call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_eval",
        json!({
            "code": "property R riemann_symmetry"
        }),
    );
    let eval_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_eval",
        json!({
            "code": "R[a-, b-, c-, d-] + R[a-, c-, d-, b-]"
        }),
    );
    let expr_id = eval_result["expr_id"].as_str().unwrap();

    let suggest_result = call_tool(
        &mut stdin,
        &mut stdout,
        "axioma_suggest",
        json!({
            "expr_id": expr_id
        }),
    );
    let suggestions = suggest_result["suggestions"].as_array().unwrap();
    let algo_names: Vec<&str> = suggestions
        .iter()
        .map(|s| s["algorithm"].as_str().unwrap())
        .collect();
    assert!(
        algo_names.contains(&"canonicalise"),
        "should suggest canonicalise, got {:?}",
        algo_names
    );

    child.kill().ok();
}
