use ax_ir::TensorSymmetry;

use crate::unicode::YOUNG_BOX;

pub fn render_young_diagram_ascii(rows: &[usize]) -> String {
    rows.iter()
        .map(|width| "[]".repeat(*width))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_young_diagram_unicode(rows: &[usize]) -> String {
    rows.iter()
        .map(|width| {
            std::iter::repeat(YOUNG_BOX)
                .take(*width)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_tableau_slot_map_ascii(rows: &[usize], slots: &[usize]) -> String {
    let mut next_slot = 0usize;
    rows.iter()
        .map(|width| {
            let mut row = String::new();
            for _ in 0..*width {
                if let Some(slot) = slots.get(next_slot) {
                    row.push_str(&format!("[{slot}]"));
                }
                next_slot += 1;
            }
            row
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_tensor_symmetry_summary(sym: &TensorSymmetry) -> String {
    sym.tableaux
        .iter()
        .enumerate()
        .map(|(idx, tableau)| {
            let mut line = format!(
                "tableau[{idx}]: shape={:?}, slots={:?}, trace_free={}, duality={:?}",
                tableau.shape, tableau.slot_map, tableau.trace_free, tableau.duality
            );
            if let Some(label) = &tableau.label {
                line.push_str(&format!(", label={label:?}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}
