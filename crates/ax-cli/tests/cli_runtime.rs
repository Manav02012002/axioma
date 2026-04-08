use ax_ai_proto::{AiEditRequest, AiPacket, Edit, Span};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

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

fn repo_file(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("axioma-{label}-{nanos}"));
    fs::create_dir_all(&dir).expect("mkdir temp dir");
    dir
}

fn write(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir parent");
    }
    fs::write(path, text).expect("write file");
}

fn run_repl_script(current_dir: &Path, script: &str) -> std::process::Output {
    let mut child = bin()
        .current_dir(current_dir)
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write repl script");
    child.wait_with_output().expect("wait repl")
}

#[test]
fn parse_command_writes_diags_json() {
    let dir = unique_temp_dir("parse");
    let src = dir.join("ok.ax");
    let diags = dir.join("diags.json");
    write(&src, "1 + 2;");

    let out = bin()
        .args([
            "parse",
            src.to_string_lossy().as_ref(),
            "--diags-json",
            diags.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run parse");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let parsed: serde_json::Value =
        serde_json::from_slice(&fs::read(diags).expect("read diags")).expect("parse diags json");
    assert!(parsed.as_array().is_some(), "expected array diagnostics");
}

#[test]
fn render_command_supports_latex_and_ascii() {
    let dir = unique_temp_dir("render");
    let src = dir.join("expr.ax");
    write(&src, "x^2 + 1");

    let latex = bin()
        .args(["render", src.to_string_lossy().as_ref(), "--format", "latex"])
        .output()
        .expect("run render latex");
    assert!(latex.status.success());
    assert!(
        String::from_utf8_lossy(&latex.stdout).contains("{x}^{2}"),
        "stdout:\n{}",
        String::from_utf8_lossy(&latex.stdout)
    );

    let ascii = bin()
        .args(["render", src.to_string_lossy().as_ref(), "--format", "ascii"])
        .output()
        .expect("run render ascii");
    assert!(ascii.status.success());
    assert!(
        String::from_utf8_lossy(&ascii.stdout).contains("x"),
        "stdout:\n{}",
        String::from_utf8_lossy(&ascii.stdout)
    );
}

#[test]
fn export_command_writes_latex_and_html() {
    let dir = unique_temp_dir("export");
    let src = dir.join("cells.ax");
    write(&src, "let x = 1 + 2\nx");

    let tex = dir.join("out.tex");
    let html = dir.join("out.html");

    let latex = bin()
        .args([
            "export",
            src.to_string_lossy().as_ref(),
            "--format",
            "latex",
            "--output",
            tex.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run export latex");
    assert!(latex.status.success(), "stderr:\n{}", String::from_utf8_lossy(&latex.stderr));
    assert!(fs::read_to_string(&tex).expect("read tex").contains("\\documentclass"));

    let html_out = bin()
        .args([
            "export",
            src.to_string_lossy().as_ref(),
            "--format",
            "html",
            "--output",
            html.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run export html");
    assert!(html_out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&html_out.stderr));
    assert!(fs::read_to_string(&html).expect("read html").contains("<html"));
}

#[test]
fn codegen_command_emits_python() {
    let dir = unique_temp_dir("codegen");
    let src = dir.join("expr.ax");
    write(&src, "x^2 + 1");

    let out = bin()
        .args([
            "codegen",
            src.to_string_lossy().as_ref(),
            "--target",
            "python",
        ])
        .output()
        .expect("run codegen");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("x**2"),
        "stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn docgen_command_writes_reference_file() {
    let dir = unique_temp_dir("docgen");
    let out_file = dir.join("llm.md");

    let out = bin()
        .current_dir(repo_root())
        .args(["docgen", "--output", out_file.to_string_lossy().as_ref()])
        .output()
        .expect("run docgen");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let text = fs::read_to_string(out_file).expect("read docgen output");
    assert!(text.contains("# Axioma Language Reference"));
}

#[test]
fn run_command_executes_file() {
    let out = bin()
        .args(["run", repo_file("examples/equation_manipulation.ax").to_string_lossy().as_ref()])
        .output()
        .expect("run command");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("LaTeX:"),
        "stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn run_qm_tutorial_executes_end_to_end() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("examples/qm_tutorial.ax").to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run qm tutorial");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[[2i, 0], [0, -2i]]"), "stdout:\n{stdout}");
    assert!(stdout.contains("[[1, 0], [0, 0]]"), "stdout:\n{stdout}");
    assert!(stdout.contains("\n1\n"), "stdout:\n{stdout}");
}

#[test]
fn run_gr_tutorial_executes_end_to_end() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("examples/gr_tutorial.ax").to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run gr tutorial");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("imported std.conventions.mtw"), "stdout:\n{stdout}");
    assert!(stdout.contains("48M²r⁻⁶"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("dot_theta"), "stdout:\n{stdout}");
}

#[test]
fn run_cosmology_perturbation_example_executes_end_to_end() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("examples/cosmological_perturbations.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run cosmological perturbations example");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("00_constraint"), "stdout:\n{stdout}");
    assert!(stdout.contains("ij_traceless"), "stdout:\n{stdout}");
    assert!(stdout.contains("H_star"), "stdout:\n{stdout}");
    assert!(stdout.contains("eta_sr"), "stdout:\n{stdout}");
}

#[test]
fn run_qft_examples_execute_end_to_end() {
    for rel in ["examples/brst_qed.ax", "examples/wess_zumino_model.ax"] {
        let out = bin()
            .current_dir(repo_root())
            .args(["run", repo_file(rel).to_string_lossy().as_ref()])
            .output()
            .expect("run qft example");

        assert!(out.status.success(), "{rel} stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        if rel.ends_with("brst_qed.ax") {
            assert!(stdout.contains("initialized Yang-Mills BRST setup"), "stdout:\n{stdout}");
            assert!(stdout.contains("\n0\n"), "stdout:\n{stdout}");
            assert!(stdout.contains("\n-1\n"), "stdout:\n{stdout}");
        } else {
            assert!(stdout.contains("initialized N=1 superspace"), "stdout:\n{stdout}");
            assert!(stdout.contains("F_Phi"), "stdout:\n{stdout}");
            assert!(stdout.contains("F_bar_Phi_bar"), "stdout:\n{stdout}");
        }
    }
}

#[test]
fn run_qft_std_modules_execute_end_to_end() {
    for rel in [
        "std/qft/brst.ax",
        "std/qft/dirac.ax",
        "std/qft/gamma.ax",
        "std/qft/normal_ordering.ax",
        "std/qft/scalar_field.ax",
        "std/qft/spinor_helicity.ax",
        "std/qft/superspace.ax",
    ] {
        let out = bin()
            .current_dir(repo_root())
            .args(["run", repo_file(rel).to_string_lossy().as_ref()])
            .output()
            .expect("run qft std module");

        assert!(out.status.success(), "{rel} stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        match rel {
            "std/qft/brst.ax" => {
                assert!(stdout.contains("initialized Yang-Mills BRST setup"), "stdout:\n{stdout}");
                assert!(stdout.contains("partial(c)"), "stdout:\n{stdout}");
                assert!(stdout.contains("\ntrue\n"), "stdout:\n{stdout}");
            }
            "std/qft/dirac.ax" => {
                assert!(stdout.contains("bar(ψ)"), "stdout:\n{stdout}");
                assert!(stdout.contains("gamma(μ)"), "stdout:\n{stdout}");
                assert!(!stdout.contains("fierz("), "stdout:\n{stdout}");
            }
            "std/qft/gamma.ax" => {
                assert!(stdout.contains("gamma(μ, ν)"), "stdout:\n{stdout}");
                assert!(stdout.contains("4g^μ^ν"), "stdout:\n{stdout}");
            }
            "std/qft/normal_ordering.ax" => {
                assert!(stdout.contains("creation(a)annihilation(a)"), "stdout:\n{stdout}");
                assert!(stdout.contains("number_state(a, 1)"), "stdout:\n{stdout}");
            }
            "std/qft/scalar_field.ax" => {
                assert!(stdout.contains("dphi_dt"), "stdout:\n{stdout}");
                assert!(stdout.contains("m²φ"), "stdout:\n{stdout}");
            }
            "std/qft/spinor_helicity.ax" => {
                assert!(stdout.contains("⟨12⟩"), "stdout:\n{stdout}");
                assert!(stdout.contains("s_{12}"), "stdout:\n{stdout}");
                assert!(stdout.contains("z"), "stdout:\n{stdout}");
            }
            "std/qft/superspace.ax" => {
                assert!(stdout.contains("initialized N=1 superspace"), "stdout:\n{stdout}");
                assert!(stdout.contains("F_Phi"), "stdout:\n{stdout}");
                assert!(stdout.contains("theta_bar"), "stdout:\n{stdout}");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn run_frw_module_has_only_time_derivatives_of_scale_factor() {
    let out = bin()
        .current_dir(repo_root())
        .args(["run", repo_file("std/gr/frw.ax").to_string_lossy().as_ref()])
        .output()
        .expect("run frw module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("∂a(t)/∂t"), "stdout:\n{stdout}");
    assert!(!stdout.contains("∂a(t)/∂x"), "stdout:\n{stdout}");
    assert!(!stdout.contains("∂a(t)/∂y"), "stdout:\n{stdout}");
    assert!(!stdout.contains("∂a(t)/∂z"), "stdout:\n{stdout}");
}

#[test]
fn run_minkowski_module_produces_zero_christoffel() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/gr/minkowski.ax").to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run minkowski module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]"
        ),
        "stdout:\n{stdout}"
    );
}

#[test]
fn run_kerr_newman_module_exposes_full_metric_matrix() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/gr/kerr_newman.ax").to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run kerr-newman module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[t, r, θ, φ]"), "stdout:\n{stdout}");
    assert!(stdout.contains("sin(θ)²"), "stdout:\n{stdout}");
    assert!(stdout.contains("[[-1("), "stdout:\n{stdout}");
    assert!(stdout.contains("Q²"), "stdout:\n{stdout}");
}

#[test]
fn run_cosmology_perturbation_module_exposes_demo_bindings() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/cosmology/perturbation.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run cosmology perturbation module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Phi_B"), "stdout:\n{stdout}");
    assert!(stdout.contains("second_order_00_constraint"), "stdout:\n{stdout}");
    assert!(stdout.contains("H_star"), "stdout:\n{stdout}");
    assert!(stdout.contains("16ε"), "stdout:\n{stdout}");
}

#[test]
fn run_classical_mechanics_module_exposes_euler_lagrange_equations() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/physics/classical_mechanics.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run classical mechanics module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("d2x_dtdt"), "stdout:\n{stdout}");
    assert!(stdout.contains("kx"), "stdout:\n{stdout}");
    assert!(stdout.contains("sin(θ)"), "stdout:\n{stdout}");
}

#[test]
fn run_maxwell_module_exposes_field_strength_and_eom() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/physics/maxwell.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run maxwell module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("A1_t - A0_x"), "stdout:\n{stdout}");
    assert!(stdout.contains("d2A0_dxdx"), "stdout:\n{stdout}");
    assert!(stdout.contains("d2A1_dtdt"), "stdout:\n{stdout}");
}

#[test]
fn run_qm_spin_module_exposes_pauli_commutator() {
    let out = bin()
        .current_dir(repo_root())
        .args(["run", repo_file("std/qm/spin.ax").to_string_lossy().as_ref()])
        .output()
        .expect("run qm spin module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[[0, 1], [1, 0]]"), "stdout:\n{stdout}");
    assert!(stdout.contains("[[2i, 0], [0, -2i]]"), "stdout:\n{stdout}");
}

#[test]
fn run_qm_bell_module_exposes_reduced_density_matrix() {
    let out = bin()
        .current_dir(repo_root())
        .args(["run", repo_file("std/qm/bell.ax").to_string_lossy().as_ref()])
        .output()
        .expect("run qm bell module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[2^(-½), 0, 0, 2^(-½)]"), "stdout:\n{stdout}");
    assert!(stdout.contains("[[½, 0], [0, ½]]"), "stdout:\n{stdout}");
}

#[test]
fn run_qm_harmonic_oscillator_module_exposes_abstract_oscillator_algebra() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/qm/harmonic_oscillator.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run qm harmonic oscillator module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("annihilation(ho)"), "stdout:\n{stdout}");
    assert!(stdout.contains("creation(ho)"), "stdout:\n{stdout}");
    assert!(stdout.contains("number_state(ho, 1)"), "stdout:\n{stdout}");
    assert!(stdout.contains("√2number_state(ho, 2)"), "stdout:\n{stdout}");
    assert!(stdout.contains("1 + creation(ho)annihilation(ho)"), "stdout:\n{stdout}");
    assert!(stdout.contains("3/2hbarωnumber_state(ho, 1)"), "stdout:\n{stdout}");
}

#[test]
fn run_black_hole_perturbation_module_exposes_master_equations() {
    let out = bin()
        .current_dir(repo_root())
        .args([
            "run",
            repo_file("std/gr/black_hole_perturbation.ax")
                .to_string_lossy()
                .as_ref(),
        ])
        .output()
        .expect("run black-hole perturbation module");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("even_parity"), "stdout:\n{stdout}");
    assert!(stdout.contains("Psi_Z"), "stdout:\n{stdout}");
    assert!(stdout.contains("Psi_RW"), "stdout:\n{stdout}");
}

#[test]
fn init_and_install_commands_work_together() {
    let dir = unique_temp_dir("init-install");
    let dep = dir.join("dep");
    fs::create_dir_all(&dep).expect("mkdir dep");
    write(&dep.join("demo.ax"), "let demo = 1");

    let init = bin().current_dir(&dir).args(["init"]).output().expect("run init");
    assert!(init.status.success(), "stderr:\n{}", String::from_utf8_lossy(&init.stderr));
    assert!(dir.join("axioma.toml").exists());

    let install = bin()
        .current_dir(&dir)
        .args([
            "install",
            "demo_pkg",
            "--path",
            dep.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run install");
    assert!(install.status.success(), "stderr:\n{}", String::from_utf8_lossy(&install.stderr));
    let config = fs::read_to_string(dir.join("axioma.toml")).expect("read config");
    assert!(config.contains("demo_pkg"));
    assert!(config.contains(dep.to_string_lossy().as_ref()));
}

#[test]
fn fix_command_repairs_missing_semicolon() {
    let dir = unique_temp_dir("fix");
    let src = dir.join("bad.ax");
    let diags = dir.join("diags.json");
    write(&src, "module demo");

    let out = bin()
        .args([
            "fix",
            src.to_string_lossy().as_ref(),
            "--apply",
            "--diags-json",
            diags.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run fix");

    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(fs::read_to_string(&src).expect("read fixed"), "module demo;");
    let parsed: serde_json::Value =
        serde_json::from_slice(&fs::read(diags).expect("read diags")).expect("parse diags");
    assert_eq!(parsed, serde_json::json!([]));
}

#[test]
fn ai_fix_pack_and_apply_commands_work() {
    let dir = unique_temp_dir("ai");
    let src = dir.join("bad.ax");
    let packet = dir.join("packet.json");
    let edits = dir.join("edits.json");
    write(&src, "module demo");

    let fix = bin()
        .args([
            "ai",
            "fix",
            src.to_string_lossy().as_ref(),
            "--max-iter",
            "2",
        ])
        .output()
        .expect("run ai fix");
    assert!(fix.status.success(), "stderr:\n{}", String::from_utf8_lossy(&fix.stderr));
    assert_eq!(fs::read_to_string(&src).expect("read fixed"), "module demo;");

    let pack = bin()
        .args([
            "ai",
            "pack",
            src.to_string_lossy().as_ref(),
            "--out",
            packet.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run ai pack");
    assert!(pack.status.success(), "stderr:\n{}", String::from_utf8_lossy(&pack.stderr));
    let packet_json: AiPacket =
        serde_json::from_slice(&fs::read(&packet).expect("read packet")).expect("parse packet");
    assert_eq!(packet_json.tool, "axioma");

    let request = AiEditRequest {
        version: "1".to_string(),
        file_hash_blake3_hex: blake3::hash(fs::read(&src).expect("read src").as_slice())
            .to_hex()
            .to_string(),
        edits: vec![Edit::Replace {
            span: Span { start: 0, end: 6 },
            replacement: "module".to_string(),
        }],
        rationale: None,
    };
    write(
        &edits,
        &serde_json::to_string_pretty(&request).expect("serialize request"),
    );

    let apply = bin()
        .args([
            "ai",
            "apply",
            src.to_string_lossy().as_ref(),
            edits.to_string_lossy().as_ref(),
            "--print",
        ])
        .output()
        .expect("run ai apply");
    assert!(apply.status.success(), "stderr:\n{}", String::from_utf8_lossy(&apply.stderr));
    assert!(
        String::from_utf8_lossy(&apply.stdout).contains("\"applied\""),
        "stdout:\n{}",
        String::from_utf8_lossy(&apply.stdout)
    );
}

#[test]
fn repl_help_and_quit_work() {
    let dir = unique_temp_dir("repl-help");
    let out = run_repl_script(&dir, ":help\n:quit\n");
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Axioma REPL commands:"), "stdout:\n{stdout}");
    assert!(stdout.contains(":latex             Toggle LaTeX input mode"), "stdout:\n{stdout}");
}

#[test]
fn repl_meta_commands_execute_end_to_end() {
    let dir = unique_temp_dir("repl-meta");
    let out = run_repl_script(
        &dir,
        "1+2\n:trust\n:pool on\n:pool stats\n:pool off\n:parallel on\n:parallel off\n:latex\n\\alpha+1\n:inspect\n:suggest\n:codegen python\n:export latex session.tex\n:export html session.html\n:env\n:rules\n:assumptions\n:convention\n:reset\n:env\n:quit\n",
    );
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stdout.contains("[1] = 3"), "stdout:\n{stdout}");
    assert!(
        stderr.contains("error: No trust information for the last result."),
        "stderr:\n{stderr}"
    );
    assert!(stdout.contains("Expression pool enabled."), "stdout:\n{stdout}");
    assert!(stdout.contains("Unique pooled nodes:"), "stdout:\n{stdout}");
    assert!(stdout.contains("Parallel mode enabled."), "stdout:\n{stdout}");
    assert!(stdout.contains("Parallel mode disabled."), "stdout:\n{stdout}");
    assert!(stdout.contains("LaTeX input mode on."), "stdout:\n{stdout}");
    assert!(stdout.contains("[2] = 1 + α"), "stdout:\n{stdout}");
    assert!(stdout.contains("Kind: sum"), "stdout:\n{stdout}");
    assert!(stdout.contains("Suggested algorithms:"), "stdout:\n{stdout}");
    assert!(stdout.contains("1 + alpha"), "stdout:\n{stdout}");
    assert!(stdout.contains("wrote session.tex"), "stdout:\n{stdout}");
    assert!(stdout.contains("wrote session.html"), "stdout:\n{stdout}");
    assert!(stdout.contains("(no bindings)"), "stdout:\n{stdout}");
    assert!(stdout.contains("(no rules)"), "stdout:\n{stdout}");
    assert!(stdout.contains("(no assumptions)"), "stdout:\n{stdout}");
    assert!(stdout.contains("metric_signature: MostlyPlus"), "stdout:\n{stdout}");
    assert!(stdout.contains("Environment reset."), "stdout:\n{stdout}");
    assert!(dir.join("session.tex").exists());
    assert!(dir.join("session.html").exists());
}
