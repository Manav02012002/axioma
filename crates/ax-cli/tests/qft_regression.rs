use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
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

fn expression_lines(stdout: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with("declared index family:")
                && !line.starts_with("attached property ")
                && !line.starts_with("  LaTeX:")
        })
        .map(ToOwned::to_owned)
        .collect()
}

fn run_inline(label: &str, source: &str) -> Vec<String> {
    let dir = unique_temp_dir(label);
    let src = dir.join("main.ax");
    write(&src, source);

    let out = bin()
        .current_dir(repo_root())
        .args(["run", src.to_string_lossy().as_ref()])
        .output()
        .expect("run axioma inline source");

    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    expression_lines(&out.stdout)
}

#[test]
fn gamma_trace_regressions_match_canonical_forms() {
    let lines = run_inline(
        "qft-gamma-trace",
        r#"
indices spin [s0, s1, s2, s3] dim=4 values=[s0, s1, s2, s3]
declare_gamma_matrix_meta(gamma, 4, eta, spin, true)
declare_gamma5_convention(gamma, mostly_plus, plus_two_g, levi_civita, eps, 4)

gamma_trace([mu, nu], eta)
gamma_trace([mu, nu, rho], eta)
gamma5_trace([mu, nu, rho, sigma])
"#,
    );

    assert_eq!(
        lines,
        vec![
            "4η^μ^ν".to_string(),
            "0".to_string(),
            "-4ε^μ^ν^ρ^σi".to_string(),
        ]
    );
}

#[test]
fn chiral_projector_and_sigma_regressions_match_canonical_forms() {
    let lines = run_inline(
        "qft-chiral-projectors",
        r#"
indices spin [s0, s1, s2, s3] dim=4 values=[s0, s1, s2, s3]
declare_spinor_meta(xi_l, 4, Weyl, left, spin)
declare_spinor_meta(eta_l, 4, Weyl, left, spin)
declare_spinor_meta(eta_r, 4, Weyl, right, spin)
declare_gamma_matrix_meta(gamma, 4, eta, spin, true)
declare_gamma5_convention(gamma, mostly_plus, plus_two_g, abstract_chiral, eps, 4)
declare_dirac_bar_meta(bar, gamma, spin, true)

simplify_chiral(projector_left() * projector_right() * eta_l)
simplify_chiral(projector_left() + projector_right())
sigma_to_gamma(sigma(mu, nu))
simplify_spinor_bilinears(bar(xi_l) * gamma(mu) * eta_l)
"#,
    );

    assert_eq!(
        lines,
        vec![
            "0".to_string(),
            "1".to_string(),
            "½(gamma(μ)gamma(ν) + -1gamma(ν)gamma(μ))i".to_string(),
            "bar(xi_l)gamma(μ)eta_l".to_string(),
        ]
    );
}

#[test]
fn fierz_rearrangement_regression_matches_canonical_form() {
    let lines = run_inline(
        "qft-fierz",
        r#"
indices spin [s0, s1, s2, s3] dim=4 values=[s0, s1, s2, s3]
declare_spinor_meta(psi1, 4, Majorana, none, spin)
declare_spinor_meta(psi2, 4, Majorana, none, spin)
declare_spinor_meta(psi3, 4, Majorana, none, spin)
declare_spinor_meta(psi4, 4, Majorana, none, spin)
declare_gamma_matrix_meta(gamma, 4, eta, spin, true)
declare_gamma5_convention(gamma, mostly_plus, plus_two_g, abstract_chiral, eps, 4)
declare_dirac_bar_meta(bar, gamma, spin, true)

fierz(bar(psi1) * gamma(mu) * psi2 * bar(psi3) * psi4)
"#,
    );

    assert_eq!(
        lines,
        vec![String::from(
            "-¼psi1psi4psi3psi2 + -1/6psi1gamma(_f3)²gamma5²psi4psi3psi2 + -1/96psi1gamma5²psi4psi3psi2 + ¾psi1sigma(_f1, _f2)²psi4psi3psi2 + psi1gamma(_f0)²psi4psi3psi2"
        )]
    );
}
