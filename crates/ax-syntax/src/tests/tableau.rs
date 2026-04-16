use crate::{parser::parse_file, tableau_symmetry_exprs};

#[test]
fn parses_single_tableau_symmetry_expression() {
    let src = "tableau_symmetry([[2,1]], slots=[[0,1,2]]);";
    let (node, diags) = parse_file(src);
    assert!(diags.is_empty(), "{diags:?}");
    let tableaux = tableau_symmetry_exprs(&node);
    assert_eq!(tableaux.len(), 1);
    assert_eq!(tableaux[0].tableau_shapes(), vec![vec![2, 1]]);
    assert_eq!(tableaux[0].tableau_slot_maps(), vec![vec![0, 1, 2]]);
}

#[test]
fn parses_multiple_tableau_attachments() {
    let src = "tableau_symmetry([[2,1],[1]], slots=[[0,1,2],[0]]);";
    let (node, diags) = parse_file(src);
    assert!(diags.is_empty(), "{diags:?}");
    let mut tableaux = tableau_symmetry_exprs(&node);
    let tableau = tableaux.remove(0);
    assert_eq!(tableau.tableau_shapes().len(), 2);
    assert_eq!(tableau.tableau_slot_maps().len(), 2);
}

#[test]
fn errors_when_shapes_missing() {
    let (_node, diags) = parse_file("tableau_symmetry([], slots=[]);");
    assert!(diags
        .iter()
        .any(|diag| diag.message == "tableau_symmetry requires at least one tableau shape"));
}

#[test]
fn errors_when_shapes_and_slots_lengths_do_not_match() {
    let (_node, diags) = parse_file("tableau_symmetry([[2,1]], slots=[[0,1],[2]]);");
    assert!(diags.iter().any(|diag| {
        diag.message == "tableau_symmetry shapes and slots lists must have the same length"
    }));
}

#[test]
fn parse_and_render_tableau_symmetry_round_trip_is_exact() {
    let symmetry =
        crate::parse_tableau_symmetry("tableau_symmetry([[2,1]], slots=[[0,1,2]])").unwrap();
    assert_eq!(
        ax_render::render_tensor_symmetry_summary(&symmetry),
        "tableau[0]: shape=[2, 1], slots=[0, 1, 2], trace_free=false, duality=None"
    );
}
