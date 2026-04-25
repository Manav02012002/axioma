use crate::{
    anticommutator_exprs, bra_exprs, braket_exprs, commutator_exprs, dagger_exprs, ket_exprs,
    normal_order_exprs, parser::parse_file, subsystem_label_exprs, tensor_product_exprs,
};

#[test]
fn parses_ket_expr() {
    let (root, diagnostics) = parse_file("|psi>;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(ket_exprs(&root).len(), 1);
}

#[test]
fn parses_bra_expr() {
    let (root, diagnostics) = parse_file("<phi|;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(bra_exprs(&root).len(), 1);
}

#[test]
fn parses_braket_expr() {
    let (root, diagnostics) = parse_file("<phi|psi>;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(braket_exprs(&root).len(), 1);
}

#[test]
fn parses_dagger_expr() {
    let (root, diagnostics) = parse_file("A†;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(dagger_exprs(&root).len(), 1);
}

#[test]
fn parses_tensor_product_expr() {
    let (root, diagnostics) = parse_file("A ⊗ B;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(tensor_product_exprs(&root).len(), 1);
}

#[test]
fn parses_tensor_product_with_dagger() {
    let (root, diagnostics) = parse_file("A† ⊗ B;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(dagger_exprs(&root).len(), 1);
    assert_eq!(tensor_product_exprs(&root).len(), 1);
}

#[test]
fn parses_dirac_and_tensor_product_together() {
    let (root, diagnostics) = parse_file("<phi| ⊗ |psi>;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(bra_exprs(&root).len(), 1);
    assert_eq!(ket_exprs(&root).len(), 1);
    assert_eq!(tensor_product_exprs(&root).len(), 1);
}

#[test]
fn parses_commutator_expr() {
    let (root, diagnostics) = parse_file("[A, B];");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(commutator_exprs(&root).len(), 1);
}

#[test]
fn parses_anticommutator_expr() {
    let (root, diagnostics) = parse_file("{A, B};");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(anticommutator_exprs(&root).len(), 1);
}

#[test]
fn parses_normal_order_expr() {
    let (root, diagnostics) = parse_file(":a*b:;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(normal_order_exprs(&root).len(), 1);
}

#[test]
fn parses_subsystem_label_expr() {
    let (root, diagnostics) = parse_file("A@QA;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(subsystem_label_exprs(&root).len(), 1);
}

#[test]
fn parses_dagger_then_subsystem_label() {
    let (root, diagnostics) = parse_file("A†@Q;");
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(dagger_exprs(&root).len(), 1);
    assert_eq!(subsystem_label_exprs(&root).len(), 1);
}

#[test]
fn reports_missing_ket_closer() {
    let (_root, diagnostics) = parse_file("|psi;");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected '>' to close ket"));
}

#[test]
fn reports_missing_braket_pipe() {
    let (_root, diagnostics) = parse_file("<phi;");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected '|' in bra or braket"));
}

#[test]
fn reports_missing_braket_closer() {
    let (_root, diagnostics) = parse_file("<phi|psi;");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected '>' to close braket"));
}

#[test]
fn reports_missing_commutator_comma() {
    let (_root, diagnostics) = parse_file("[A B];");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected ',' in commutator"));
}

#[test]
fn reports_missing_anticommutator_closer() {
    let (_root, diagnostics) = parse_file("{A, B;");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected '}' to close anticommutator"));
}

#[test]
fn reports_missing_normal_order_closer() {
    let (_root, diagnostics) = parse_file(":a*b;");
    assert!(diagnostics
        .iter()
        .any(|diag| diag.message == "expected ':' to close normal-order expression"));
}
