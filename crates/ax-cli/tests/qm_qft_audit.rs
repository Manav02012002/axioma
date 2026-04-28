use ax_eval::registry::{builtin_entries, BuiltinEntry};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct AuditEntry {
    name: String,
    category: String,
    registry_doc_required: bool,
    std_example_paths: Vec<String>,
    test_paths: Vec<String>,
    notebook_or_jupyter_covered: bool,
    mcp_covered: bool,
}

fn repo_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path
}

fn manifest_path() -> PathBuf {
    repo_root().join("tests/qm_qft_audit_manifest.json")
}

fn load_manifest() -> Vec<AuditEntry> {
    let path = manifest_path();
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn qm_qft_property_builtin_names() -> &'static [&'static str] {
    &[
        "spinor",
        "dirac_bar",
        "diracbar",
        "gamma_matrix",
        "declare_spinor_meta",
        "declare_gamma_matrix_meta",
        "declare_gamma_convention",
        "declare_gamma5_convention",
        "declare_dirac_bar_meta",
        "declare_trace_space",
        "declare_hilbert_space",
        "declare_composite_space",
        "declare_quantum_object",
        "declare_operator_space",
        "declare_mode",
        "declare_mode_in_subsystem",
        "declare_mode_with_label",
        "declare_bosonic_truncated_mode",
        "declare_fermionic_mode",
        "declare_fock_space",
    ]
}

fn is_qm_qft_builtin(entry: &BuiltinEntry) -> bool {
    matches!(
        entry.category,
        "spinor" | "twistor" | "graded-algebra" | "superspace" | "brst" | "qm" | "quantum"
    ) || (entry.category == "properties" && qm_qft_property_builtin_names().contains(&entry.name))
}

fn qm_qft_registry_map() -> BTreeMap<&'static str, BuiltinEntry> {
    builtin_entries()
        .into_iter()
        .filter(is_qm_qft_builtin)
        .map(|entry| (entry.name, entry))
        .collect()
}

fn assert_registry_docs_present(entry: &BuiltinEntry) {
    assert!(
        !entry.signature.trim().is_empty(),
        "builtin `{}` is missing a registry signature",
        entry.name
    );
    assert!(
        !entry.description.trim().is_empty(),
        "builtin `{}` is missing a registry description",
        entry.name
    );
    assert!(
        !entry.example.trim().is_empty(),
        "builtin `{}` is missing a registry example",
        entry.name
    );
}

fn assert_repo_relative_file_exists(root: &Path, rel: &str, kind: &str, builtin: &str) {
    let path = root.join(rel);
    assert!(
        path.is_file(),
        "builtin `{}` references missing {} path `{}`",
        builtin,
        kind,
        rel
    );
}

#[test]
fn qm_qft_audit_manifest_entries_resolve() {
    let manifest = load_manifest();
    assert!(
        !manifest.is_empty(),
        "qm/qft audit manifest should not be empty"
    );

    let mut seen = BTreeSet::new();
    for entry in &manifest {
        assert!(
            seen.insert(entry.name.as_str()),
            "duplicate qm/qft audit manifest entry for `{}`",
            entry.name
        );
        assert!(
            !entry.category.trim().is_empty(),
            "builtin `{}` is missing a category",
            entry.name
        );
        assert!(
            !entry.std_example_paths.is_empty(),
            "builtin `{}` must list at least one std/example path",
            entry.name
        );
        assert!(
            !entry.test_paths.is_empty(),
            "builtin `{}` must list at least one test path",
            entry.name
        );

        let _surface_flags = (entry.notebook_or_jupyter_covered, entry.mcp_covered);
    }
}

#[test]
fn qm_qft_audit_registry_entries_exist() {
    let manifest = load_manifest();
    let registry = qm_qft_registry_map();

    let manifest_names = manifest
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<BTreeSet<_>>();
    let registry_names = registry.keys().copied().collect::<BTreeSet<_>>();

    assert_eq!(
        manifest_names, registry_names,
        "qm/qft audit manifest must exactly match the registry-derived builtin set"
    );

    for entry in &manifest {
        let registry_entry = registry
            .get(entry.name.as_str())
            .unwrap_or_else(|| panic!("missing registry builtin `{}`", entry.name));
        assert_eq!(
            entry.category, registry_entry.category,
            "manifest category mismatch for `{}`",
            entry.name
        );
        if entry.registry_doc_required {
            assert_registry_docs_present(registry_entry);
        }
    }
}

#[test]
fn qm_qft_audit_required_files_exist() {
    let manifest = load_manifest();
    let root = repo_root();

    for entry in &manifest {
        for rel in &entry.std_example_paths {
            assert_repo_relative_file_exists(&root, rel, "std/example", &entry.name);
        }
        for rel in &entry.test_paths {
            assert_repo_relative_file_exists(&root, rel, "test", &entry.name);
        }
    }
}
