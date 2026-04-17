use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn oracle_run_corpus_prints_passing_case_trace() {
    let output = Command::new(env!("CARGO_BIN_EXE_axioma"))
        .current_dir(workspace_root())
        .args([
            "oracle",
            "run",
            "--corpus",
            "tests/corpora/young_tableaux_oracle.json",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:#?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("case=sym_rank2_canonicalize"), "{stdout}");
    assert!(stdout.contains("passed=true"), "{stdout}");
}

#[test]
fn oracle_bench_manifest_prints_case_repetition_lines() {
    let output = Command::new(env!("CARGO_BIN_EXE_axioma"))
        .current_dir(workspace_root())
        .args([
            "oracle",
            "bench",
            "--manifest",
            "tests/corpora/benchmark_manifest.json",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:#?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("case=sym_rank2_canonicalization; repetitions=5"),
        "{stdout}"
    );
    assert!(
        stdout.contains("case=shape_21_projector; repetitions=5"),
        "{stdout}"
    );
    assert!(
        stdout.contains("case=triple_vector_lr; repetitions=5"),
        "{stdout}"
    );
}
