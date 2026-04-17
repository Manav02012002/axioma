use anyhow::{bail, Result};

pub fn render(shape: &str, slots: Option<&str>) -> Result<String> {
    let rows = parse_csv_usize(shape)?;
    let rendered = if let Some(slots) = slots {
        let slot_map = parse_csv_usize(slots)?;
        ax_render::render_tableau_slot_map_ascii(&rows, &slot_map)
    } else {
        ax_render::render_young_diagram_ascii(&rows)
    };
    Ok(format!("{rendered}\n"))
}

pub fn summary(expr: &str) -> Result<String> {
    let symmetry = ax_syntax::parse_tableau_symmetry(expr)
        .map_err(|diags| anyhow::anyhow!(format_diags(&diags)))?;
    Ok(format!(
        "{}\n",
        ax_render::render_tensor_symmetry_summary(&symmetry)
    ))
}

pub fn character(shape: &str, cycle: &str) -> Result<String> {
    let rows = parse_csv_usize(shape)?;
    let cycle_type = parse_csv_usize(cycle)?;
    let diagram = ax_young::YoungDiagram::try_new(rows)?;
    let value = ax_young::symmetric_group_character(&diagram, &cycle_type)?;
    Ok(format!("character={value}\n"))
}

pub fn frobenius(shape: &str) -> Result<String> {
    let rows = parse_csv_usize(shape)?;
    let diagram = ax_young::YoungDiagram::try_new(rows)?;
    let rendered =
        ax_render::render_power_sum_expansion(&ax_young::frobenius_characteristic(&diagram)?);
    Ok(format!("{rendered}\n"))
}

pub fn trace(shape: &str) -> Result<String> {
    let rows = parse_csv_usize(shape)?;
    let diagram = ax_young::YoungDiagram::try_new(rows)?;
    let tableau = ax_young::YoungTableau::standard(&diagram)?;
    let (_, trace) = ax_young::build_projector_with_trace(
        &tableau,
        ax_young::ProjectorNormalization::Unnormalized,
    )?;

    Ok(format!(
        "shape={:?}\ndegree={}\nrow_generator_count={}\ncolumn_generator_count={}\nexpanded_term_count={}\n",
        trace.shape,
        trace.degree,
        trace.row_generator_count,
        trace.column_generator_count,
        trace.expanded_term_count
    ))
}

pub fn canonicalize(shape: &str, slots: &str) -> Result<String> {
    let rows = parse_csv_usize(shape)?;
    let slots = parse_csv_usize(slots)?;
    let diagram = ax_young::YoungDiagram::try_new(rows)?;
    let tableau = ax_young::YoungTableau::standard(&diagram)?;
    let (projector, _) = ax_young::build_projector_with_trace(
        &tableau,
        ax_young::ProjectorNormalization::Unnormalized,
    )?;
    let (canonical_slots, _) = ax_young::canonicalize_slots_with_trace(&projector, &slots)?;

    Ok(format!(
        "canonical_slots={}\n",
        canonical_slots
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn parse_csv_usize(input: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for part in input.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        values.push(trimmed.parse::<usize>()?);
    }
    if values.is_empty() {
        bail!("expected at least one integer");
    }
    Ok(values)
}

fn format_diags(diags: &[ax_syntax::Diagnostic]) -> String {
    diags
        .iter()
        .map(|diag| diag.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tableau_character_command_renders_exact_stdout() {
        assert_eq!(character("2,1", "2,1").unwrap(), "character=0\n");
    }

    #[test]
    fn tableau_frobenius_command_contains_power_sum_term() {
        let stdout = frobenius("2").unwrap();
        assert!(stdout.contains("p_[2]"));
    }
}
