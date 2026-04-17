use std::{path::PathBuf, process::Command};

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_axioma"));
    c.env("AXIOMA_ROOT", repo_root());
    c
}

#[test]
fn tableau_render_command_matches_exact_output() {
    let out = bin()
        .args(["tableau", "render", "--shape", "2,1", "--slots", "0,1,2"])
        .output()
        .expect("run tableau render");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[0][1]\n[2]\n");
}

#[test]
fn tableau_trace_command_reports_required_prefixes() {
    let out = bin()
        .args(["tableau", "trace", "--shape", "2,1"])
        .output()
        .expect("run tableau trace");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.lines().any(|line| line.starts_with("shape=")),
        "{stdout}"
    );
    assert!(
        stdout.lines().any(|line| line.starts_with("degree=")),
        "{stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("row_generator_count=")),
        "{stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("column_generator_count=")),
        "{stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("expanded_term_count=")),
        "{stdout}"
    );
}

#[test]
fn tableau_canonicalize_command_matches_exact_output() {
    let out = bin()
        .args(["tableau", "canonicalize", "--shape", "2", "--slots", "9,3"])
        .output()
        .expect("run tableau canonicalize");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "canonical_slots=3,9\n"
    );
}

#[test]
fn tableau_summary_command_matches_exact_output_for_multiple_tableaux() {
    let out = bin()
        .args([
            "tableau",
            "summary",
            "--expr",
            "tableau_symmetry([[2,1],[1,1]], slots=[[0,1,2],[1,2]], labels=[\"main\",\"alt\"], trace_free=[false,true])",
        ])
        .output()
        .expect("run tableau summary");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        concat!(
            "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None, label=\"main\"\n",
            "tableau[1]: shape=[1, 1], slots=[1, 2], trace_free=true, duality=None, label=\"alt\"\n"
        )
    );
}

#[test]
fn tableau_summary_command_renders_trace_free_curvature_attachment_exactly() {
    let out = bin()
        .args([
            "tableau",
            "summary",
            "--expr",
            "tableau_symmetry([[2,2]], slots=[[0,1,2,3]], labels=[\"weyl\"], trace_free=[true])",
        ])
        .output()
        .expect("run tableau summary");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "tableau[0]: shape=[2, 2], slots=[0, 1, 2, 3], trace_free=true, duality=None, label=\"weyl\"\n"
    );
}

#[test]
fn tableau_character_command_matches_exact_output() {
    let out = bin()
        .args(["tableau", "character", "--shape", "2,1", "--cycle", "2,1"])
        .output()
        .expect("run tableau character");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "character=0\n");
}

#[test]
fn tableau_frobenius_command_contains_power_sum_term() {
    let out = bin()
        .args(["tableau", "frobenius", "--shape", "2"])
        .output()
        .expect("run tableau frobenius");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("p_[2]"));
}
