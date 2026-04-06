use ax_cli::cmd_export::*;
use ax_ir::Interner;

#[test]
fn latex_standalone_has_documentclass() {
    let interner = Interner::new();
    let source = "// Test\nlet x = 5\nx + 1";
    let options = ExportOptions::default();
    let result = export(source, &options, &interner);
    assert!(
        result.contains("\\documentclass"),
        "standalone LaTeX should have documentclass"
    );
    assert!(
        result.contains("\\begin{document}"),
        "should have begin document"
    );
    assert!(
        result.contains("\\end{document}"),
        "should have end document"
    );
}

#[test]
fn latex_fragment_no_preamble() {
    let interner = Interner::new();
    let source = "let x = 5";
    let mut options = ExportOptions::default();
    options.standalone = false;
    let result = export(source, &options, &interner);
    assert!(
        !result.contains("\\documentclass"),
        "fragment should not have documentclass"
    );
}

#[test]
fn latex_includes_input_code() {
    let interner = Interner::new();
    let source = "diff(x^2, x)";
    let options = ExportOptions::default();
    let result = export(source, &options, &interner);
    assert!(
        result.contains("lstlisting") || result.contains("verbatim"),
        "should include code listing"
    );
    assert!(result.contains("diff"), "should contain the input code");
}

#[test]
fn latex_includes_output() {
    let interner = Interner::new();
    let source = "1 + 2";
    let options = ExportOptions::default();
    let result = export(source, &options, &interner);
    assert!(result.contains("3"), "should contain the output 3");
}

#[test]
fn latex_no_input_flag() {
    let interner = Interner::new();
    let source = "1 + 2";
    let mut options = ExportOptions::default();
    options.include_input = false;
    let result = export(source, &options, &interner);
    assert!(
        !result.contains("lstlisting"),
        "should not contain code listing"
    );
    assert!(result.contains("3"), "should still contain output");
}

#[test]
fn latex_no_output_flag() {
    let interner = Interner::new();
    let source = "1 + 2";
    let mut options = ExportOptions::default();
    options.include_output = false;
    let result = export(source, &options, &interner);
    assert!(
        result.contains("1 + 2") || result.contains("lstlisting"),
        "should contain input"
    );
}

#[test]
fn latex_comment_as_text() {
    let interner = Interner::new();
    let source = "// This is a paragraph of explanation.\nlet x = 5";
    let options = ExportOptions::default();
    let result = export(source, &options, &interner);
    assert!(
        result.contains("This is a paragraph"),
        "comment should appear as text"
    );
    assert!(!result.contains("//"), "comment markers should be stripped");
}

#[test]
fn latex_heading_detection() {
    let interner = Interner::new();
    let source = "// ===== My Section =====\n// Some text\nlet x = 5";
    let options = ExportOptions::default();
    let result = export(source, &options, &interner);
    assert!(
        result.contains("\\section") || result.contains("\\subsection"),
        "=== lines should become section headings"
    );
}

#[test]
fn latex_title_and_author() {
    let interner = Interner::new();
    let source = "1 + 1";
    let mut options = ExportOptions::default();
    options.title = Some("My Computation".to_string());
    options.author = Some("Manav Rawal".to_string());
    let result = export(source, &options, &interner);
    assert!(result.contains("My Computation"), "should contain title");
    assert!(result.contains("Manav Rawal"), "should contain author");
    assert!(result.contains("\\maketitle"), "should have maketitle");
}

#[test]
fn latex_revtex_class() {
    let interner = Interner::new();
    let source = "1 + 1";
    let mut options = ExportOptions::default();
    options.document_class = "revtex4-2".to_string();
    let result = export(source, &options, &interner);
    assert!(
        result.contains("revtex4-2"),
        "should use revtex document class"
    );
}

#[test]
fn html_standalone_structure() {
    let interner = Interner::new();
    let source = "// Hello\n1 + 2";
    let mut options = ExportOptions::default();
    options.format = ExportFormat::Html;
    let result = export(source, &options, &interner);
    assert!(result.contains("<!DOCTYPE html>"), "should have doctype");
    assert!(result.contains("<html"), "should have html tag");
    assert!(result.contains("katex"), "should include KaTeX");
    assert!(result.contains("</html>"), "should close html");
}

#[test]
fn html_contains_math() {
    let interner = Interner::new();
    let source = "diff(x^3, x)";
    let mut options = ExportOptions::default();
    options.format = ExportFormat::Html;
    let result = export(source, &options, &interner);
    assert!(
        result.contains("$$"),
        "output should be wrapped in $$ for KaTeX"
    );
}

#[test]
fn html_syntax_highlight() {
    let interner = Interner::new();
    let source = "let x = 5";
    let mut options = ExportOptions::default();
    options.format = ExportFormat::Html;
    let result = export(source, &options, &interner);
    assert!(
        result.contains("keyword"),
        "let should be highlighted as keyword"
    );
}

#[test]
fn html_escapes_entities() {
    let interner = Interner::new();
    let source = "// x < y & z > w";
    let mut options = ExportOptions::default();
    options.format = ExportFormat::Html;
    let result = export(source, &options, &interner);
    assert!(result.contains("&lt;"), "< should be escaped");
    assert!(result.contains("&gt;"), "> should be escaped");
    assert!(result.contains("&amp;"), "& should be escaped");
}

#[test]
fn html_fragment_no_wrapper() {
    let interner = Interner::new();
    let source = "1 + 1";
    let mut options = ExportOptions::default();
    options.format = ExportFormat::Html;
    options.standalone = false;
    let result = export(source, &options, &interner);
    assert!(
        !result.contains("<!DOCTYPE"),
        "fragment should not have doctype"
    );
    assert!(
        !result.contains("<html"),
        "fragment should not have html tag"
    );
}

#[test]
fn collect_cells_separates_blocks() {
    let interner = Interner::new();
    let source = "// A comment\n\nlet x = 5\nlet y = 10\n\n// Another comment";
    let cells = collect_cells(source, &interner);
    let types: Vec<_> = cells
        .iter()
        .map(|c| match c.cell_type {
            CellType::Comment => "comment",
            CellType::Code => "code",
            CellType::Blank => "blank",
        })
        .collect();
    assert!(types.contains(&"comment"), "should have comment cells");
    assert!(types.contains(&"code"), "should have code cells");
}

#[test]
fn schwarzschild_export() {
    let interner = Interner::new();
    let source = std::fs::read_to_string("../../examples/schwarzschild.ax")
        .or_else(|_| std::fs::read_to_string("examples/schwarzschild.ax"))
        .unwrap_or_else(|_| {
            "// Schwarzschild\nlet g = metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))\nlet coords = [t, r, theta, phi]\nlet Gamma = christoffel(g, coords)".to_string()
        });
    let options = ExportOptions::default();
    let result = export(&source, &options, &interner);
    assert!(
        result.contains("Schwarzschild")
            || result.contains("christoffel")
            || result.contains("Gamma"),
        "Schwarzschild export should contain relevant content"
    );
    assert!(
        result.len() > 200,
        "export should be non-trivial, got {} bytes",
        result.len()
    );
}
