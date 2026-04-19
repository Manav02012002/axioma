use crate::{
    bra_exprs, braket_exprs, dagger_exprs, ket_exprs, parser::parse_file, tensor_product_exprs,
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
