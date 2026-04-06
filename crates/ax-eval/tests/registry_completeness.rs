use ax_eval::registry::*;

#[test]
fn all_builtins_have_callable_entries() {
    let builtins = builtin_entries();
    let callables = callable_entries();
    let callable_names: std::collections::HashSet<&str> =
        callables.iter().map(|c| c.name).collect();
    let mut missing = Vec::new();
    for b in &builtins {
        if !callable_names.contains(b.name) {
            missing.push(b.name);
        }
    }
    assert!(
        missing.is_empty(),
        "These builtins have no callable entry: {:?}\nThis means they won't appear in the MCP server.",
        missing
    );
}

#[test]
fn all_algorithms_have_callable_entries() {
    let algorithms = algorithm_entries();
    let callables = callable_entries();
    let callable_names: std::collections::HashSet<&str> =
        callables.iter().map(|c| c.name).collect();
    let mut missing = Vec::new();
    for a in &algorithms {
        if !callable_names.contains(a.name) {
            missing.push(a.name);
        }
    }
    let allowed_missing = [
        "eval",
        "resolve_import",
        "describe_rewrite_trace",
        "rewrite_with_trace",
        "match_tensor_pattern",
        "compute_weight",
        "diff_component",
        "rename_dummy_indices",
        "evaluate_components_v2",
    ];
    missing.retain(|name| !allowed_missing.contains(name));
    assert!(
        missing.is_empty(),
        "These algorithms have no callable entry: {:?}",
        missing
    );
}

#[test]
fn no_duplicate_callable_names() {
    let callables = callable_entries();
    let mut seen = std::collections::HashSet::new();
    let mut dupes = Vec::new();
    for c in &callables {
        if !seen.insert(c.name) {
            dupes.push(c.name);
        }
    }
    assert!(dupes.is_empty(), "Duplicate callable entries: {:?}", dupes);
}

#[test]
fn all_properties_have_entries() {
    let props = property_entries();
    let prop_names: Vec<&str> = props.iter().map(|p| p.name).collect();
    let expected = [
        "Symmetric",
        "AntiSymmetric",
        "RiemannSymmetry",
        "Traceless",
        "Metric",
        "InverseMetric",
        "KroneckerDelta",
        "EpsilonTensor",
        "Derivative",
        "PartialDerivative",
        "CovariantDerivative",
        "Depends",
        "Spinor",
        "DiracBar",
        "GammaMatrixProp",
        "Commuting",
        "AntiCommuting",
        "NonCommuting",
        "SortOrder",
        "TableauSymmetry",
        "SatisfiesBianchi",
        "WeylTensor",
        "DifferentialFormDegree",
    ];
    for e in &expected {
        assert!(prop_names.contains(e), "missing property entry for {}", e);
    }
}

#[test]
fn all_conventions_have_entries() {
    let convs = convention_entries();
    assert!(
        convs.len() >= 5,
        "should have at least 5 convention entries, got {}",
        convs.len()
    );
    let field_names: Vec<&str> = convs.iter().map(|c| c.field).collect();
    assert!(field_names.contains(&"metric_signature"));
    assert!(field_names.contains(&"riemann_sign"));
    assert!(field_names.contains(&"ricci_contraction"));
    assert!(field_names.contains(&"levi_civita_norm"));
    assert!(field_names.contains(&"fourier_sign"));
}

#[test]
fn all_assumptions_have_entries() {
    let asms = assumption_entries();
    assert!(
        asms.len() >= 7,
        "should have at least 7 assumption entries, got {}",
        asms.len()
    );
}

#[test]
fn std_modules_all_exist() {
    let modules = std_modules();
    for m in &modules {
        assert!(!m.path.is_empty(), "module path should not be empty");
        assert!(
            !m.description.is_empty(),
            "module {} should have description",
            m.path
        );
    }
}
