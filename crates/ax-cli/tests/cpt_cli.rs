use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_axioma"))
}

#[test]
fn axioma_cpt_demo_output_contains_00_constraint() {
    let output = bin().arg("cpt-demo").output().expect("run cpt-demo");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("00_constraint"),
        "stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn axioma_cpt_export_python_output_contains_def() {
    let output = bin()
        .args(["cpt-export", "python"])
        .output()
        .expect("run cpt-export python");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("def ms_rhs("),
        "stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn axioma_cpt_export_rust_output_contains_pub_fn() {
    let output = bin()
        .args(["cpt-export", "rust"])
        .output()
        .expect("run cpt-export rust");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("pub fn ms_rhs("),
        "stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn axioma_cpt_export_cpp_output_contains_double_fn() {
    let output = bin()
        .args(["cpt-export", "cpp"])
        .output()
        .expect("run cpt-export cpp");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("double ms_rhs("),
        "stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
