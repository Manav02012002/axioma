use crate::parser::parse_file;

#[test]
fn parses_module_import_and_exprs() {
    let src = "module m; import a.b.c; f(1,2+3*4); T[a-, b+, c-];";
    let (_node, diags) = parse_file(src);
    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn root_exists_even_on_garbage() {
    let (node, diags) = parse_file("$$$");
    assert!(!diags.is_empty());
    assert_eq!(format!("{:?}", node.kind()), "Root");
}
