use crate::{FilledTableau, YoungDiagram};

#[allow(dead_code)]
pub(crate) fn render_diagram_ascii(diagram: &YoungDiagram) -> String {
    diagram
        .rows
        .iter()
        .map(|row_len| "[]".repeat(*row_len))
        .collect::<Vec<_>>()
        .join("\n")
}

#[allow(dead_code)]
pub(crate) fn render_tableau_rows<T: std::fmt::Display + Clone + Ord + Eq>(
    tableau: &FilledTableau<T>,
) -> String {
    tableau
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}
