use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MimeBundle {
    text_plain: Option<String>,
    text_latex: Option<String>,
    text_markdown: Option<String>,
    text_html: Option<String>,
    image_svg_xml: Option<String>,
    application_json: Option<Value>,
    custom: Map<String, Value>,
}

impl MimeBundle {
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text_plain: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn latex(text: impl Into<String>) -> Self {
        Self {
            text_latex: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn markdown(text: impl Into<String>) -> Self {
        Self {
            text_markdown: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn html(text: impl Into<String>) -> Self {
        Self {
            text_html: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn svg(text: impl Into<String>) -> Self {
        Self {
            image_svg_xml: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn json(value: Value) -> Self {
        Self {
            application_json: Some(value),
            ..Self::default()
        }
    }

    pub fn from_expr(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Self {
        if let Some(bundle) = cpt_mime_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_measurement_probability_plot_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_entropy_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_entanglement_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_mime_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_bloch_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_entanglement_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_spectral_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_density_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_channel_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_dynamics_summary_bundle(expr, interner) {
            return bundle;
        }
        Self::plain(ax_render::to_unicode(expr, interner))
            .with_latex(ax_render::to_latex(expr, interner))
    }

    pub fn with_plain(mut self, text: impl Into<String>) -> Self {
        self.text_plain = Some(text.into());
        self
    }

    pub fn with_latex(mut self, text: impl Into<String>) -> Self {
        self.text_latex = Some(text.into());
        self
    }

    pub fn with_markdown(mut self, text: impl Into<String>) -> Self {
        self.text_markdown = Some(text.into());
        self
    }

    pub fn with_html(mut self, text: impl Into<String>) -> Self {
        self.text_html = Some(text.into());
        self
    }

    pub fn with_svg(mut self, text: impl Into<String>) -> Self {
        self.image_svg_xml = Some(text.into());
        self
    }

    pub fn with_json(mut self, value: Value) -> Self {
        self.application_json = Some(value);
        self
    }

    pub fn with_custom(mut self, mime: impl Into<String>, value: Value) -> Self {
        self.custom.insert(mime.into(), value);
        self
    }

    pub fn text_plain(&self) -> Option<&str> {
        self.text_plain.as_deref()
    }

    pub fn text_latex(&self) -> Option<&str> {
        self.text_latex.as_deref()
    }

    pub fn image_svg_xml(&self) -> Option<&str> {
        self.image_svg_xml.as_deref()
    }

    pub fn application_json(&self) -> Option<&Value> {
        self.application_json.as_ref()
    }

    pub fn to_jupyter_data(&self) -> Map<String, Value> {
        // `serde_json::Map` in this build is deterministic but key-ordered, so
        // we populate it once here and let the stable map ordering define the
        // final MIME bundle layout.
        let mut data = Map::new();
        if let Some(text) = &self.text_plain {
            data.insert("text/plain".to_string(), Value::String(text.clone()));
        }
        if let Some(text) = &self.text_latex {
            data.insert("text/latex".to_string(), Value::String(text.clone()));
        }
        if let Some(text) = &self.text_markdown {
            data.insert("text/markdown".to_string(), Value::String(text.clone()));
        }
        if let Some(text) = &self.text_html {
            data.insert("text/html".to_string(), Value::String(text.clone()));
        }
        if let Some(text) = &self.image_svg_xml {
            data.insert("image/svg+xml".to_string(), Value::String(text.clone()));
        }
        if let Some(value) = &self.application_json {
            data.insert("application/json".to_string(), value.clone());
        }
        for (mime, value) in &self.custom {
            data.insert(mime.clone(), value.clone());
        }
        data
    }
}

const QUANTUM_WORKFLOW_MIME: &str = "application/vnd.axioma.quantum-workflow+json";
const QUANTUM_NARRATIVE_MIME: &str = "application/vnd.axioma.quantum-narrative+json";

fn canonical_call<'a>(
    expr: &'a ax_ir::Expr,
    name: &str,
    interner: &ax_ir::Interner,
) -> Option<&'a [ax_ir::Expr]> {
    let ax_ir::Expr::Call(sym, args) = expr else {
        return None;
    };
    (interner.resolve(*sym) == name).then_some(args.as_slice())
}

fn strip_groups(expr: &ax_ir::Expr) -> &ax_ir::Expr {
    let mut current = expr;
    while let ax_ir::Expr::Group(inner, _) = current {
        current = inner;
    }
    current
}

fn scalar_expr_to_f64(expr: &ax_ir::Expr) -> Option<f64> {
    match expr {
        ax_ir::Expr::Int(value) => value.to_string().parse().ok(),
        ax_ir::Expr::Rational(value) => {
            let numer = value.numer().to_string().parse::<f64>().ok()?;
            let denom = value.denom().to_string().parse::<f64>().ok()?;
            Some(numer / denom)
        }
        ax_ir::Expr::Float(value) => Some(*value),
        _ => None,
    }
}

fn scalar_exprs_to_f64(values: &[ax_ir::Expr]) -> Option<Vec<f64>> {
    values.iter().map(scalar_expr_to_f64).collect()
}

fn expr_matrix_dimension(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Option<usize> {
    match strip_groups(expr) {
        ax_ir::Expr::Matrix(rows) => {
            let dimension = rows.len();
            (dimension > 0 && rows.iter().all(|row| row.len() == dimension)).then_some(dimension)
        }
        ax_ir::Expr::List(rows) => {
            let dimension = rows.len();
            if dimension == 0 {
                return None;
            }
            rows.iter()
                .all(|row| matches!(strip_groups(row), ax_ir::Expr::List(cells) if cells.len() == dimension))
                .then_some(dimension)
        }
        _ => {
            let mut env = ax_eval::Env::new();
            let evaluated = ax_eval::eval(expr, &mut env, interner);
            (evaluated != *expr)
                .then(|| expr_matrix_dimension(&evaluated, interner))
                .flatten()
        }
    }
}

fn bool_option_label(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

/// Build a consistent two-column markdown table for QM summary bundles.
pub fn qm_multiline_summary_markdown(title: &str, rows: &[(&str, String)]) -> String {
    let mut markdown = String::new();
    if !title.is_empty() {
        markdown.push_str(title);
        markdown.push_str("\n\n");
    }
    markdown.push_str("| Quantity | Value |\n| --- | --- |");
    for (label, value) in rows {
        markdown.push_str(&format!("\n| {} | {} |", label, value));
    }
    markdown
}

fn qm_plot_packet(title: &str, kind: &str) -> Option<Value> {
    serde_json::to_value(ax_ai_proto::QuantumPlotPacket {
        title: title.to_string(),
        kind: kind.to_string(),
    })
    .ok()
}

fn quantum_workflow_value(
    workflow_kind: &str,
    title: &str,
    summary_lines: Vec<String>,
    json_payload_kind: &str,
) -> Option<Value> {
    serde_json::to_value(ax_ai_proto::QuantumWorkflowPacket {
        workflow_kind: workflow_kind.to_string(),
        title: title.to_string(),
        summary_lines,
        json_payload_kind: json_payload_kind.to_string(),
    })
    .ok()
}

fn with_quantum_workflow(
    bundle: MimeBundle,
    workflow_kind: &str,
    title: &str,
    summary_lines: Vec<String>,
    json_payload_kind: &str,
) -> MimeBundle {
    match quantum_workflow_value(workflow_kind, title, summary_lines, json_payload_kind) {
        Some(value) => bundle.with_custom(QUANTUM_WORKFLOW_MIME, value),
        None => bundle,
    }
}

fn quantum_narrative_value(trace: ax_trace::QuantumNarrativeTrace) -> Option<Value> {
    serde_json::to_value(ax_ai_proto::QuantumNarrativePacket {
        workflow_kind: trace.workflow_kind,
        explanation_steps: trace.explanation_steps,
    })
    .ok()
}

fn with_quantum_narrative(
    bundle: MimeBundle,
    trace: ax_trace::QuantumNarrativeTrace,
) -> MimeBundle {
    match quantum_narrative_value(trace) {
        Some(value) => bundle.with_custom(QUANTUM_NARRATIVE_MIME, value),
        None => bundle,
    }
}

/// Build a QM probability bar-chart bundle using the shared SVG plot helpers.
pub fn qm_probability_plot_bundle(labels: &[String], values: &[f64], title: &str) -> MimeBundle {
    let plain = format!(
        "{title}: [{}]",
        labels
            .iter()
            .zip(values.iter())
            .map(|(label, value)| format!("{label}={value:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let bundle = MimeBundle::svg(ax_plot::probability_bar_chart_svg(labels, values, title))
        .with_plain(plain);
    match qm_plot_packet(title, "probability_bar_chart") {
        Some(json) => bundle.with_json(json),
        None => bundle,
    }
}

/// Build a QM spectrum bar-chart bundle using the shared SVG plot helpers.
pub fn qm_spectrum_plot_bundle(labels: &[String], values: &[f64], title: &str) -> MimeBundle {
    let plain = format!(
        "{title}: [{}]",
        labels
            .iter()
            .zip(values.iter())
            .map(|(label, value)| format!("{label}={value:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let bundle = MimeBundle::svg(ax_plot::probability_bar_chart_svg(labels, values, title))
        .with_plain(plain);
    match qm_plot_packet(title, "spectrum_bar_chart") {
        Some(json) => bundle.with_json(json),
        None => bundle,
    }
}

/// Build a QM expectation-trajectory bundle using the shared SVG plot helpers.
pub fn qm_expectation_trajectory_plot_bundle(
    times: &[f64],
    values: &[f64],
    y_label: &str,
    title: &str,
) -> MimeBundle {
    let plain = format!(
        "{title}: [{}]",
        times
            .iter()
            .zip(values.iter())
            .map(|(time, value)| format!("({time:.6}, {value:.6})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let bundle = MimeBundle::svg(ax_plot::expectation_trajectory_svg(
        times, values, y_label, title,
    ))
    .with_plain(plain);
    let bundle = match qm_plot_packet(title, "expectation_trajectory_plot") {
        Some(json) => bundle.with_json(json),
        None => bundle,
    };
    with_quantum_workflow(
        bundle,
        "trajectory_summary",
        title,
        vec![
            format!("Trajectory contains {} sampled steps.", values.len()),
            format!("Observable label is {y_label}."),
        ],
        "QuantumPlotPacket",
    )
}

fn qm_measurement_probability_plot_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let [projectors_arg, rho_arg] =
        canonical_call(strip_groups(expr), "measurement_probabilities", interner)?
    else {
        return None;
    };

    let mut env = ax_eval::Env::new();
    let evaluated_projectors = ax_eval::eval(strip_groups(projectors_arg), &mut env, interner);
    let evaluated_rho = ax_eval::eval(strip_groups(rho_arg), &mut env, interner);
    let projectors = expr_to_kraus_channel(&evaluated_projectors)?;
    let ax_ir::Expr::Matrix(rho) = evaluated_rho else {
        return None;
    };

    let probabilities = ax_qm::measurement_probabilities(&projectors, &rho).ok()?;
    let values = scalar_exprs_to_f64(&probabilities)?;
    let labels = (0..values.len())
        .map(|index| index.to_string())
        .collect::<Vec<_>>();

    Some(qm_probability_plot_bundle(
        &labels,
        &values,
        "Measurement probabilities",
    ))
}

fn qm_object_kind(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Option<&'static str> {
    match expr {
        ax_ir::Expr::Call(sym, args) => match (interner.resolve(*sym), args.as_slice()) {
            ("ket", [_]) => Some("ket"),
            ("bra", [_]) => Some("bra"),
            ("dagger", [_]) => Some("dagger"),
            ("tensor_product", [_, _]) => Some("tensor_product"),
            ("braket", [lhs, rhs])
                if matches!(canonical_call(lhs, "bra", interner), Some([_]))
                    && matches!(canonical_call(rhs, "ket", interner), Some([_])) =>
            {
                Some("braket")
            }
            _ => None,
        },
        _ => None,
    }
}

pub fn qm_mime_bundle(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Option<MimeBundle> {
    let object_kind = qm_object_kind(expr, interner)?;
    let unicode = ax_render::to_unicode(expr, interner);
    let latex = ax_render::to_latex(expr, interner);
    let packet = ax_ai_proto::QuantumDisplayPacket {
        object_kind: object_kind.to_string(),
        unicode: unicode.clone(),
        latex: latex.clone(),
        dimension: None,
        subsystem_dims: Vec::new(),
    };
    let json = serde_json::to_value(packet).ok()?;

    Some(MimeBundle::plain(unicode).with_latex(latex).with_json(json))
}

pub fn qm_density_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let ax_ir::Expr::Matrix(rows) = expr else {
        return None;
    };
    let dimension = rows.len();
    if dimension == 0 || rows.iter().any(|row| row.len() != dimension) {
        return None;
    }

    let trace = ax_render::to_unicode(
        &ax_ir::Expr::add(
            rows.iter()
                .enumerate()
                .filter_map(|(i, row)| row.get(i).cloned())
                .collect(),
        ),
        interner,
    );
    let purity = ax_render::to_unicode(&ax_qm::purity(rows).ok()?, interner);
    let linear_entropy = ax_render::to_unicode(&ax_qm::linear_entropy(rows).ok()?, interner);
    let bloch_vector = ax_qm::bloch_vector(rows)
        .ok()
        .map(|vector| vector.map(|component| ax_render::to_unicode(&component, interner)));
    let is_qubit = bloch_vector.is_some();

    let packet = ax_ai_proto::QuantumDensitySummaryPacket {
        dimension,
        trace: trace.clone(),
        purity: purity.clone(),
        linear_entropy: linear_entropy.clone(),
        is_qubit,
        bloch_vector: bloch_vector.clone(),
    };
    let json = serde_json::to_value(&packet).ok()?;

    let mut rows = vec![
        ("Dimension", dimension.to_string()),
        ("Trace", trace.clone()),
        ("Purity", purity.clone()),
        ("Linear entropy", linear_entropy.clone()),
    ];
    if let Some([x, y, z]) = bloch_vector {
        rows.push(("Bloch vector", format!("[{x}, {y}, {z}]")));
    }
    let markdown = qm_multiline_summary_markdown("Quantum density summary", &rows);

    Some(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
            .with_markdown(markdown)
            .with_json(json),
    )
}

pub fn qm_bloch_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let ax_ir::Expr::Matrix(rows) = strip_groups(expr) else {
        return None;
    };
    if rows.len() != 2 || rows.iter().any(|row| row.len() != 2) {
        return None;
    }

    let bloch_vector = ax_qm::bloch_vector(rows).ok()?;
    let purity_expr = ax_qm::purity(rows).ok()?;
    let linear_entropy_expr = ax_qm::linear_entropy(rows).ok()?;
    let state_class = if purity_expr == ax_ir::Expr::one() {
        "pure"
    } else {
        "mixed"
    };

    let packet = ax_ai_proto::QuantumBlochSummaryPacket {
        dimension: 2,
        bloch_vector: bloch_vector
            .clone()
            .map(|component| ax_ir::pretty_print(&component, interner)),
        purity: ax_ir::pretty_print(&purity_expr, interner),
        linear_entropy: ax_ir::pretty_print(&linear_entropy_expr, interner),
        state_class: state_class.to_string(),
    };
    let json = serde_json::to_value(&packet).ok()?;
    let bloch_vector_unicode = ax_render::render_bloch_vector_unicode(&bloch_vector, interner);
    let purity = ax_render::to_unicode(&purity_expr, interner);
    let linear_entropy = ax_render::to_unicode(&linear_entropy_expr, interner);
    let markdown = qm_multiline_summary_markdown(
        "Quantum Bloch summary",
        &[
            ("Dimension", "2".to_string()),
            ("Bloch vector", bloch_vector_unicode.clone()),
            ("Purity", purity.clone()),
            ("Linear entropy", linear_entropy.clone()),
            ("State class", state_class.to_string()),
        ],
    );
    let plain = format!(
        "Quantum Bloch summary: dimension=2, bloch_vector={bloch_vector_unicode}, purity={purity}, linear_entropy={linear_entropy}, state_class={state_class}"
    );

    Some(
        MimeBundle::plain(plain)
            .with_markdown(markdown)
            .with_json(json),
    )
}

struct QuantumSpectralSummaryData {
    dimension: usize,
    eigenvalues: Vec<ax_ir::Expr>,
    entropy: Option<ax_ir::Expr>,
    renyi2_entropy: Option<ax_ir::Expr>,
    negativity: Option<ax_ir::Expr>,
    logarithmic_negativity: Option<ax_ir::Expr>,
}

struct QuantumEntanglementSummaryData {
    subsystem_dims: [usize; 2],
    reduced_spectrum_a: Vec<String>,
    reduced_spectrum_b: Vec<String>,
    von_neumann_entropy_a: Option<String>,
    von_neumann_entropy_b: Option<String>,
    renyi2_entropy_a: Option<String>,
    renyi2_entropy_b: Option<String>,
    negativity: Option<String>,
    logarithmic_negativity: Option<String>,
}

fn is_square_matrix(rows: &[Vec<ax_ir::Expr>]) -> bool {
    let dimension = rows.len();
    dimension > 0 && rows.iter().all(|row| row.len() == dimension)
}

fn expr_to_kraus_channel(expr: &ax_ir::Expr) -> Option<Vec<Vec<Vec<ax_ir::Expr>>>> {
    match strip_groups(expr) {
        ax_ir::Expr::List(items) if !items.is_empty() => items
            .iter()
            .map(|item| match strip_groups(item) {
                ax_ir::Expr::Matrix(rows) => Some(rows.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn matrix_trace_expr(rows: &[Vec<ax_ir::Expr>]) -> ax_ir::Expr {
    ax_ir::Expr::add(
        rows.iter()
            .enumerate()
            .filter_map(|(index, row)| row.get(index).cloned())
            .collect(),
    )
}

pub fn qm_channel_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let kraus = expr_to_kraus_channel(expr)?;
    let dimension = ax_qm::kraus_dimension(&kraus).ok()?;
    let kraus_count = kraus.len();
    let trace_preserving = ax_qm::is_trace_preserving_exact(&kraus, interner).ok()?;
    let unital = ax_qm::is_unital_exact(&kraus, interner).ok()?;
    let choi_dimension = dimension.checked_mul(dimension)?;
    let choi = ax_qm::choi_matrix_from_kraus(&kraus).ok()?;
    let choi_trace = ax_render::to_unicode(&matrix_trace_expr(&choi), interner);
    let completely_positive = ax_qm::is_completely_positive_choi_small(&choi, interner).ok();
    let family_hint = ax_qm::canonical_channel_family_hint(&kraus, interner);

    let packet = ax_ai_proto::QuantumChannelSummaryPacket {
        dimension,
        kraus_count,
        trace_preserving,
        unital,
        choi_dimension,
        family_hint: family_hint.clone(),
        choi_trace: choi_trace.clone(),
        completely_positive,
    };
    let json = serde_json::to_value(&packet).ok()?;
    let family_hint_label = family_hint
        .as_deref()
        .map(ax_render::render_channel_family_hint_unicode)
        .unwrap_or_else(|| "unknown".to_string());
    let plain = format!(
        "Quantum channel: dimension={dimension}, kraus_count={kraus_count}, trace_preserving={trace_preserving}, unital={unital}, choi_dimension={choi_dimension}, family_hint={family_hint_label}, choi_trace={choi_trace}, completely_positive={}",
        bool_option_label(completely_positive)
    );
    let markdown = qm_multiline_summary_markdown(
        "Quantum channel summary",
        &[
            ("Dimension", dimension.to_string()),
            ("Kraus count", kraus_count.to_string()),
            ("Trace preserving", trace_preserving.to_string()),
            ("Unital", unital.to_string()),
            ("Choi dimension", choi_dimension.to_string()),
            ("Family hint", family_hint_label),
            ("Choi trace", choi_trace),
            (
                "Completely positive",
                bool_option_label(completely_positive).to_string(),
            ),
        ],
    );

    Some(with_quantum_narrative(
        with_quantum_workflow(
            MimeBundle::plain(plain)
                .with_markdown(markdown)
                .with_json(json),
            "channel_summary",
            "Quantum channel summary",
            vec![
                format!("Dimension {dimension} channel with {kraus_count} Kraus operators."),
                format!("Trace preserving is {trace_preserving} and unital is {unital}."),
            ],
            "QuantumChannelSummaryPacket",
        ),
        ax_trace::narrative_for_channel_summary(dimension, kraus_count, trace_preserving, unital),
    ))
}

fn is_diagonal_matrix(rows: &[Vec<ax_ir::Expr>]) -> bool {
    rows.iter().enumerate().all(|(i, row)| {
        row.iter()
            .enumerate()
            .all(|(j, entry)| i == j || *entry == ax_ir::Expr::zero())
    })
}

fn bipartite_dim_candidates(total_dim: usize) -> Vec<[usize; 2]> {
    let mut candidates = Vec::new();
    for dim_a in 2..=total_dim {
        if total_dim % dim_a != 0 {
            continue;
        }
        let dim_b = total_dim / dim_a;
        if dim_b < 2 {
            continue;
        }
        candidates.push([dim_a, dim_b]);
    }
    candidates
}

fn pretty_expr_option(value: Option<ax_ir::Expr>, interner: &ax_ir::Interner) -> Option<String> {
    value
        .as_ref()
        .map(|expr| ax_ir::pretty_print(expr, interner))
}

fn entanglement_summary_data(
    rows: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Option<QuantumEntanglementSummaryData> {
    if !is_square_matrix(rows) || !ax_qm::matrix_is_exactly_hermitian(rows) {
        return None;
    }

    let total_dim = rows.len();
    for [dim_a, dim_b] in bipartite_dim_candidates(total_dim) {
        let Ok(reduced_spectrum_a) =
            ax_qm::entanglement_spectrum_from_density(rows, dim_a, dim_b, 'A', interner)
        else {
            continue;
        };
        let Ok(reduced_spectrum_b) =
            ax_qm::entanglement_spectrum_from_density(rows, dim_a, dim_b, 'B', interner)
        else {
            continue;
        };
        let Ok(rho_a) = ax_qm::try_partial_trace(
            rows,
            ax_qm::BipartiteDims { dim_a, dim_b },
            ax_qm::PartialTraceTarget::B,
        ) else {
            continue;
        };
        let Ok(rho_b) = ax_qm::try_partial_trace(
            rows,
            ax_qm::BipartiteDims { dim_a, dim_b },
            ax_qm::PartialTraceTarget::A,
        ) else {
            continue;
        };

        let von_neumann_entropy_a =
            pretty_expr_option(ax_qm::von_neumann_entropy(&rho_a, interner).ok(), interner);
        let von_neumann_entropy_b =
            pretty_expr_option(ax_qm::von_neumann_entropy(&rho_b, interner).ok(), interner);
        let renyi2_entropy_a =
            pretty_expr_option(ax_qm::renyi2_entropy(&rho_a, interner).ok(), interner);
        let renyi2_entropy_b =
            pretty_expr_option(ax_qm::renyi2_entropy(&rho_b, interner).ok(), interner);
        let negativity = pretty_expr_option(
            ax_qm::negativity_bipartite(rows, dim_a, dim_b, 1, interner).ok(),
            interner,
        );
        let logarithmic_negativity = pretty_expr_option(
            ax_qm::logarithmic_negativity_bipartite(rows, dim_a, dim_b, 1, interner).ok(),
            interner,
        );

        let has_entanglement_quantity = von_neumann_entropy_a.is_some()
            || von_neumann_entropy_b.is_some()
            || renyi2_entropy_a.is_some()
            || renyi2_entropy_b.is_some()
            || negativity.is_some()
            || logarithmic_negativity.is_some();
        if !has_entanglement_quantity {
            continue;
        }

        return Some(QuantumEntanglementSummaryData {
            subsystem_dims: [dim_a, dim_b],
            reduced_spectrum_a: reduced_spectrum_a
                .iter()
                .map(|value| ax_ir::pretty_print(value, interner))
                .collect(),
            reduced_spectrum_b: reduced_spectrum_b
                .iter()
                .map(|value| ax_ir::pretty_print(value, interner))
                .collect(),
            von_neumann_entropy_a,
            von_neumann_entropy_b,
            renyi2_entropy_a,
            renyi2_entropy_b,
            negativity,
            logarithmic_negativity,
        });
    }

    None
}

fn spectral_summary_data(
    rows: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Option<QuantumSpectralSummaryData> {
    if !is_square_matrix(rows) || !ax_qm::matrix_is_exactly_hermitian(rows) {
        return None;
    }

    let dimension = rows.len();
    let eigenvalues = match dimension {
        1 => vec![rows[0][0].clone()],
        2 => ax_qm::hermitian_eigenvalues_small(rows, interner).ok()?,
        3 if is_diagonal_matrix(rows) => rows
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].clone())
            .collect(),
        4 => {
            let negativity = ax_qm::negativity_bipartite(rows, 2, 2, 1, interner).ok()?;
            let logarithmic_negativity =
                ax_qm::logarithmic_negativity_bipartite(rows, 2, 2, 1, interner).ok()?;
            if ax_qm::purity(rows).ok()? != ax_ir::Expr::one() {
                return None;
            }
            return Some(QuantumSpectralSummaryData {
                dimension,
                eigenvalues: vec![
                    ax_ir::Expr::one(),
                    ax_ir::Expr::zero(),
                    ax_ir::Expr::zero(),
                    ax_ir::Expr::zero(),
                ],
                entropy: ax_qm::von_neumann_entropy(rows, interner).ok(),
                renyi2_entropy: ax_qm::renyi2_entropy(rows, interner).ok(),
                negativity: Some(negativity),
                logarithmic_negativity: Some(logarithmic_negativity),
            });
        }
        _ => return None,
    };

    Some(QuantumSpectralSummaryData {
        dimension,
        eigenvalues,
        entropy: ax_qm::von_neumann_entropy(rows, interner).ok(),
        renyi2_entropy: ax_qm::renyi2_entropy(rows, interner).ok(),
        negativity: None,
        logarithmic_negativity: None,
    })
}

pub fn qm_spectral_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let ax_ir::Expr::Matrix(rows) = expr else {
        return None;
    };
    let summary = spectral_summary_data(rows, interner)?;
    let unicode_eigenvalues =
        ax_render::render_eigenvalue_list_unicode(summary.eigenvalues.as_slice(), interner);
    let latex_eigenvalues =
        ax_render::render_eigenvalue_list_latex(summary.eigenvalues.as_slice(), interner);
    let packet = ax_ai_proto::QuantumSpectralSummaryPacket {
        dimension: summary.dimension,
        eigenvalues: summary
            .eigenvalues
            .iter()
            .map(|value| ax_ir::pretty_print(value, interner))
            .collect(),
        entropy: summary
            .entropy
            .as_ref()
            .map(|value| ax_ir::pretty_print(value, interner)),
        renyi2_entropy: summary
            .renyi2_entropy
            .as_ref()
            .map(|value| ax_ir::pretty_print(value, interner)),
        negativity: summary
            .negativity
            .as_ref()
            .map(|value| ax_ir::pretty_print(value, interner)),
        logarithmic_negativity: summary
            .logarithmic_negativity
            .as_ref()
            .map(|value| ax_ir::pretty_print(value, interner)),
    };
    let json = serde_json::to_value(&packet).ok()?;

    let mut rows = vec![
        ("Dimension", summary.dimension.to_string()),
        ("Eigenvalues", unicode_eigenvalues.clone()),
    ];
    if let Some(value) = &summary.entropy {
        rows.push((
            "Von Neumann entropy",
            ax_render::to_unicode(value, interner),
        ));
    }
    if let Some(value) = &summary.renyi2_entropy {
        rows.push(("Rényi-2 entropy", ax_render::to_unicode(value, interner)));
    }
    if let Some(value) = &summary.negativity {
        rows.push(("Negativity", ax_render::to_unicode(value, interner)));
    }
    if let Some(value) = &summary.logarithmic_negativity {
        rows.push((
            "Logarithmic negativity",
            ax_render::to_unicode(value, interner),
        ));
    }
    let markdown = qm_multiline_summary_markdown("Quantum spectral summary", &rows);
    let mut bundle = with_quantum_workflow(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
            .with_latex(latex_eigenvalues)
            .with_markdown(markdown)
            .with_json(json),
        "spectral_summary",
        "Quantum spectral summary",
        {
            let mut lines = vec![format!(
                "Dimension {} spectrum with eigenvalues {}.",
                summary.dimension, unicode_eigenvalues
            )];
            if let Some(value) = &summary.entropy {
                lines.push(format!(
                    "Von Neumann entropy is {}.",
                    ax_render::to_unicode(value, interner)
                ));
            }
            if let Some(value) = &summary.renyi2_entropy {
                lines.push(format!(
                    "Renyi-2 entropy is {}.",
                    ax_render::to_unicode(value, interner)
                ));
            }
            if let Some(value) = &summary.negativity {
                lines.push(format!(
                    "Negativity is {}.",
                    ax_render::to_unicode(value, interner)
                ));
            }
            if let Some(value) = &summary.logarithmic_negativity {
                lines.push(format!(
                    "Logarithmic negativity is {}.",
                    ax_render::to_unicode(value, interner)
                ));
            }
            lines
        },
        "QuantumSpectralSummaryPacket",
    );

    if let Some(values) = scalar_exprs_to_f64(summary.eigenvalues.as_slice()) {
        let labels = (0..values.len())
            .map(|index| format!("λ{}", index + 1))
            .collect::<Vec<_>>();
        bundle = bundle.with_svg(ax_plot::probability_bar_chart_svg(
            &labels, &values, "Spectrum",
        ));
    }

    Some(bundle)
}

pub fn qm_entanglement_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let ax_ir::Expr::Matrix(rows) = strip_groups(expr) else {
        return None;
    };
    let summary = entanglement_summary_data(rows, interner)?;
    let packet = ax_ai_proto::QuantumEntanglementSummaryPacket {
        subsystem_dims: summary.subsystem_dims.to_vec(),
        reduced_spectrum_a: summary.reduced_spectrum_a.clone(),
        reduced_spectrum_b: summary.reduced_spectrum_b.clone(),
        von_neumann_entropy_a: summary.von_neumann_entropy_a.clone(),
        von_neumann_entropy_b: summary.von_neumann_entropy_b.clone(),
        renyi2_entropy_a: summary.renyi2_entropy_a.clone(),
        renyi2_entropy_b: summary.renyi2_entropy_b.clone(),
        negativity: summary.negativity.clone(),
        logarithmic_negativity: summary.logarithmic_negativity.clone(),
    };
    let json = serde_json::to_value(&packet).ok()?;

    let mut markdown_rows = vec![
        (
            "Subsystem dimensions",
            format!(
                "[{}, {}]",
                summary.subsystem_dims[0], summary.subsystem_dims[1]
            ),
        ),
        (
            "Reduced spectrum A",
            ax_render::render_spectrum_unicode(&summary.reduced_spectrum_a),
        ),
        (
            "Reduced spectrum B",
            ax_render::render_spectrum_unicode(&summary.reduced_spectrum_b),
        ),
    ];
    if let Some(value) = &summary.von_neumann_entropy_a {
        markdown_rows.push(("Von Neumann entropy A", value.clone()));
    }
    if let Some(value) = &summary.von_neumann_entropy_b {
        markdown_rows.push(("Von Neumann entropy B", value.clone()));
    }
    if let Some(value) = &summary.renyi2_entropy_a {
        markdown_rows.push(("Rényi-2 entropy A", value.clone()));
    }
    if let Some(value) = &summary.renyi2_entropy_b {
        markdown_rows.push(("Rényi-2 entropy B", value.clone()));
    }
    if let Some(value) = &summary.negativity {
        markdown_rows.push(("Negativity", value.clone()));
    }
    if let Some(value) = &summary.logarithmic_negativity {
        markdown_rows.push(("Logarithmic negativity", value.clone()));
    }

    let markdown = qm_multiline_summary_markdown("Quantum entanglement summary", &markdown_rows);
    let mut plain_parts = vec![
        format!(
            "subsystem_dims=[{}, {}]",
            summary.subsystem_dims[0], summary.subsystem_dims[1]
        ),
        format!(
            "reduced_spectrum_a={}",
            ax_render::render_spectrum_unicode(&summary.reduced_spectrum_a)
        ),
        format!(
            "reduced_spectrum_b={}",
            ax_render::render_spectrum_unicode(&summary.reduced_spectrum_b)
        ),
    ];
    if let Some(value) = &summary.von_neumann_entropy_a {
        plain_parts.push(format!("von_neumann_entropy_a={value}"));
    }
    if let Some(value) = &summary.von_neumann_entropy_b {
        plain_parts.push(format!("von_neumann_entropy_b={value}"));
    }
    if let Some(value) = &summary.renyi2_entropy_a {
        plain_parts.push(format!("renyi2_entropy_a={value}"));
    }
    if let Some(value) = &summary.renyi2_entropy_b {
        plain_parts.push(format!("renyi2_entropy_b={value}"));
    }
    if let Some(value) = &summary.negativity {
        plain_parts.push(format!("negativity={value}"));
    }
    if let Some(value) = &summary.logarithmic_negativity {
        plain_parts.push(format!("logarithmic_negativity={value}"));
    }
    let plain = format!("Quantum entanglement summary: {}", plain_parts.join(", "));

    Some(with_quantum_narrative(
        with_quantum_workflow(
            MimeBundle::plain(plain)
                .with_markdown(markdown)
                .with_json(json),
            "entanglement_summary",
            "Quantum entanglement summary",
            {
                let mut lines = vec![format!(
                    "Subsystem dimensions are [{}, {}].",
                    summary.subsystem_dims[0], summary.subsystem_dims[1]
                )];
                lines.push(format!(
                    "Reduced spectra are A={} and B={}.",
                    ax_render::render_spectrum_unicode(&summary.reduced_spectrum_a),
                    ax_render::render_spectrum_unicode(&summary.reduced_spectrum_b)
                ));
                if let Some(value) = &summary.negativity {
                    lines.push(format!("Negativity is {value}."));
                }
                if let Some(value) = &summary.logarithmic_negativity {
                    lines.push(format!("Logarithmic negativity is {value}."));
                }
                lines
            },
            "QuantumEntanglementSummaryPacket",
        ),
        ax_trace::narrative_for_negativity(summary.subsystem_dims[0], summary.subsystem_dims[1]),
    ))
}

pub fn qm_entropy_bundle(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Option<MimeBundle> {
    let [arg] = canonical_call(strip_groups(expr), "von_neumann_entropy", interner)? else {
        return None;
    };
    let mut env = ax_eval::Env::new();
    let evaluated_arg = ax_eval::eval(strip_groups(arg), &mut env, interner);
    let ax_ir::Expr::Matrix(rows) = &evaluated_arg else {
        return None;
    };
    let value = ax_qm::von_neumann_entropy(rows, interner).ok()?;
    let value_unicode = ax_render::to_unicode(&value, interner);
    let value_latex = ax_render::to_latex(&value, interner);
    let json = serde_json::to_value(ax_ai_proto::QuantumEntropyPacket {
        kind: "von_neumann_entropy".to_string(),
        value_unicode,
        value_latex,
    })
    .ok()?;

    Some(with_quantum_narrative(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
            .with_latex(ax_render::to_latex(expr, interner))
            .with_json(json),
        ax_trace::narrative_for_entropy(rows.len(), "von Neumann entropy"),
    ))
}

pub fn qm_entanglement_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let args = canonical_call(strip_groups(expr), "negativity", interner)
        .or_else(|| canonical_call(strip_groups(expr), "logarithmic_negativity", interner))?;
    let [rho_arg, dim_a_arg, dim_b_arg] = args else {
        return None;
    };

    let mut env = ax_eval::Env::new();
    let evaluated_rho = ax_eval::eval(strip_groups(rho_arg), &mut env, interner);
    let evaluated_dim_a = ax_eval::eval(strip_groups(dim_a_arg), &mut env, interner);
    let evaluated_dim_b = ax_eval::eval(strip_groups(dim_b_arg), &mut env, interner);
    let ax_ir::Expr::Matrix(rows) = &evaluated_rho else {
        return None;
    };
    let ax_ir::Expr::Int(dim_a) = evaluated_dim_a else {
        return None;
    };
    let ax_ir::Expr::Int(dim_b) = evaluated_dim_b else {
        return None;
    };
    let dim_a = dim_a.to_string().parse::<usize>().ok()?;
    let dim_b = dim_b.to_string().parse::<usize>().ok()?;

    let spectrum =
        ax_qm::partial_transpose_spectrum_bipartite(rows, dim_a, dim_b, 1, interner).ok()?;
    let negativity = ax_qm::negativity_bipartite(rows, dim_a, dim_b, 1, interner).ok()?;
    let logarithmic_negativity =
        ax_qm::logarithmic_negativity_bipartite(rows, dim_a, dim_b, 1, interner).ok()?;
    let json = serde_json::to_value(ax_ai_proto::QuantumEntanglementPacket {
        spectrum: spectrum
            .iter()
            .map(|value| ax_ir::pretty_print(value, interner))
            .collect(),
        negativity: ax_ir::pretty_print(&negativity, interner),
        logarithmic_negativity: ax_ir::pretty_print(&logarithmic_negativity, interner),
    })
    .ok()?;

    Some(MimeBundle::plain(ax_render::to_unicode(expr, interner)).with_json(json))
}

fn qm_dynamics_summary_packet(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_ai_proto::QuantumDynamicsSummaryPacket> {
    let expr = strip_groups(expr);

    if let Some([h, _t]) = canonical_call(expr, "time_evolution_operator", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "operator".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "hamiltonian".to_string(),
            trace_preserving: None,
            purity_preserving: None,
        });
    }

    if let Some([h, _psi0, _t]) = canonical_call(expr, "schrodinger_evolve", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "state".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "hamiltonian".to_string(),
            trace_preserving: None,
            purity_preserving: Some(true),
        });
    }

    if let Some([h, _op0, _t]) = canonical_call(expr, "heisenberg_evolve", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "operator".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "hamiltonian".to_string(),
            trace_preserving: None,
            purity_preserving: None,
        });
    }

    if let Some([h, _rho]) = canonical_call(expr, "liouville_rhs", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "density_rhs".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "liouville".to_string(),
            trace_preserving: Some(true),
            purity_preserving: Some(true),
        });
    }

    if let Some([h, _rho, _jumps]) = canonical_call(expr, "lindblad_rhs", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "density_rhs".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "lindblad".to_string(),
            trace_preserving: Some(true),
            purity_preserving: Some(false),
        });
    }

    if let Some([h, _jumps]) = canonical_call(expr, "lindbladian_superoperator", interner) {
        return Some(ax_ai_proto::QuantumDynamicsSummaryPacket {
            object_kind: "superoperator".to_string(),
            dimension: expr_matrix_dimension(h, interner)?,
            generator_kind: "lindblad".to_string(),
            trace_preserving: Some(true),
            purity_preserving: Some(false),
        });
    }

    None
}

pub fn qm_dynamics_summary_bundle(
    expr: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<MimeBundle> {
    let packet = qm_dynamics_summary_packet(expr, interner)?;
    let json = serde_json::to_value(&packet).ok()?;
    let plain = format!(
        "Quantum dynamics summary: object_kind={}, dimension={}, generator_kind={}, trace_preserving={}, purity_preserving={}",
        packet.object_kind,
        packet.dimension,
        packet.generator_kind,
        bool_option_label(packet.trace_preserving),
        bool_option_label(packet.purity_preserving)
    );
    let markdown = qm_multiline_summary_markdown(
        "Quantum dynamics summary",
        &[
            ("Object kind", packet.object_kind.clone()),
            ("Dimension", packet.dimension.to_string()),
            ("Generator kind", packet.generator_kind.clone()),
            (
                "Trace preserving",
                bool_option_label(packet.trace_preserving).to_string(),
            ),
            (
                "Purity preserving",
                bool_option_label(packet.purity_preserving).to_string(),
            ),
        ],
    );

    Some(
        MimeBundle::plain(plain)
            .with_markdown(markdown)
            .with_json(json),
    )
}

pub fn cpt_mime_bundle(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> Option<MimeBundle> {
    if ax_render::is_labelled_equation_list(expr) {
        return Some(
            MimeBundle::plain(ax_render::render_labelled_equation_list_unicode(
                expr, interner,
            )?)
            .with_latex(ax_render::render_labelled_equation_list_latex(
                expr, interner,
            )?),
        );
    }
    if let Some(rendered) = ax_render::render_cpt_spec_unicode(expr, interner) {
        return Some(MimeBundle::plain(rendered.clone()).with_latex(rendered));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jupyter_data_is_emitted_in_stable_order() {
        let bundle = MimeBundle::plain("plain")
            .with_latex("latex")
            .with_markdown("markdown")
            .with_html("<b>html</b>")
            .with_svg("<svg></svg>")
            .with_json(serde_json::json!({"ok": true}));
        let keys = bundle
            .to_jupyter_data()
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            vec![
                "application/json",
                "image/svg+xml",
                "text/html",
                "text/latex",
                "text/markdown",
                "text/plain",
            ]
        );
    }

    #[test]
    fn qm_multiline_summary_markdown_formats_two_column_table() {
        let markdown = qm_multiline_summary_markdown(
            "Quantum summary",
            &[("Dimension", "2".to_string()), ("Trace", "1".to_string())],
        );

        assert!(markdown.contains("| Quantity | Value |"), "{markdown}");
        assert!(markdown.contains("| --- | --- |"), "{markdown}");
        assert!(markdown.contains("| Dimension | 2 |"), "{markdown}");
    }

    #[test]
    fn qm_probability_plot_bundle_contains_svg_mime() {
        let bundle = qm_probability_plot_bundle(
            &["0".to_string(), "1".to_string()],
            &[0.75, 0.25],
            "Measurement probabilities",
        );
        let data = bundle.to_jupyter_data();
        let svg = data["image/svg+xml"].as_str().expect("svg");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(data.contains_key("image/svg+xml"), "{data:?}");
        assert!(svg.contains("<svg"), "{svg}");
        assert!(
            json.contains("\"kind\":\"probability_bar_chart\""),
            "{json}"
        );
    }

    #[test]
    fn cpt_mime_bundle_formats_labelled_equations() {
        let interner = ax_ir::Interner::new();
        let label = interner.get_or_intern("eq0");
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::List(vec![ax_ir::Expr::List(vec![
            ax_ir::Expr::Sym(label),
            ax_ir::Expr::Sym(x),
        ])]);

        let bundle = cpt_mime_bundle(&expr, &interner);

        assert_eq!(
            bundle.and_then(|b| b.text_plain().map(str::to_string)),
            Some("eq0: x".to_string())
        );
    }

    #[test]
    fn qm_mime_bundle_for_ket_contains_json_and_latex() {
        let interner = ax_ir::Interner::new();
        let ket = interner.get_or_intern("ket");
        let psi = interner.get_or_intern("psi");
        let expr = ax_ir::Expr::Call(ket, vec![ax_ir::Expr::Sym(psi)]);

        let bundle = qm_mime_bundle(&expr, &interner).expect("qm bundle");
        assert_eq!(bundle.text_plain(), Some("|psi⟩"));
        assert!(bundle
            .text_latex()
            .is_some_and(|latex| latex.contains("\\left|")));
        let json = bundle.application_json().expect("json mime");
        let encoded = serde_json::to_string(json).expect("json encoding");
        assert!(encoded.contains("\"object_kind\":\"ket\""), "got {encoded}");
    }

    #[test]
    fn qm_density_summary_bundle_contains_markdown_and_json() {
        let expr = ax_ir::Expr::Matrix(vec![
            vec![ax_ir::Expr::one(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_density_summary_bundle(&expr, &interner).expect("density summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Purity"), "{markdown}");
        assert!(markdown.contains("1"), "{markdown}");
        assert!(json.contains("\"dimension\":2"), "{json}");
        assert!(json.contains("\"is_qubit\":true"), "{json}");
    }

    #[test]
    fn qm_bloch_summary_bundle_zero_state_contains_001() {
        let expr = ax_ir::Expr::Matrix(vec![
            vec![ax_ir::Expr::one(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_bloch_summary_bundle(&expr, &interner).expect("bloch summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Bloch vector"), "{markdown}");
        assert!(
            json.contains("\"bloch_vector\":[\"0\",\"0\",\"1\"]"),
            "{json}"
        );
        assert!(json.contains("\"state_class\":\"pure\""), "{json}");
    }

    #[test]
    fn qm_bloch_summary_bundle_plus_state_contains_100() {
        let expr = ax_ir::Expr::Matrix(ax_qm::qubit_density_from_bloch([
            ax_ir::Expr::one(),
            ax_ir::Expr::zero(),
            ax_ir::Expr::zero(),
        ]));
        let interner = ax_ir::Interner::new();

        let bundle = qm_bloch_summary_bundle(&expr, &interner).expect("bloch summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Bloch vector"), "{markdown}");
        assert!(
            json.contains("\"bloch_vector\":[\"1\",\"0\",\"0\"]"),
            "{json}"
        );
        assert!(json.contains("\"state_class\":\"pure\""), "{json}");
    }

    #[test]
    fn channel_summary_identity_has_family_hint_identity() {
        let expr = ax_ir::Expr::List(vec![ax_ir::Expr::Matrix(vec![
            vec![ax_ir::Expr::one(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), ax_ir::Expr::one()],
        ])]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_channel_summary_bundle(&expr, &interner).expect("channel summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = &data["application/json"];
        let json_text = serde_json::to_string(json).expect("json encoding");

        assert!(markdown.contains("| Dimension | 2 |"), "{markdown}");
        assert!(markdown.contains("| Kraus count | 1 |"), "{markdown}");
        assert!(
            markdown.contains("| Trace preserving | true |"),
            "{markdown}"
        );
        assert!(markdown.contains("| Unital | true |"), "{markdown}");
        assert!(markdown.contains("| Choi dimension | 4 |"), "{markdown}");
        assert!(markdown.contains("Family hint"), "{markdown}");
        assert!(
            markdown.contains("| Family hint | identity |"),
            "{markdown}"
        );
        assert!(markdown.contains("| Choi trace | 2 |"), "{markdown}");
        assert!(
            markdown.contains("| Completely positive | true |"),
            "{markdown}"
        );
        assert!(
            json_text.contains("\"family_hint\":\"identity\""),
            "{json_text}"
        );
        assert!(
            json_text.contains("\"trace_preserving\":true"),
            "{json_text}"
        );
        assert!(json_text.contains("\"unital\":true"), "{json_text}");
        assert_eq!(json["kraus_count"], 1);
        assert_eq!(json["trace_preserving"], true);
        assert_eq!(json["unital"], true);
        assert_eq!(json["choi_dimension"], 4);
        assert_eq!(json["family_hint"], "identity");
        assert_eq!(json["choi_trace"], "2");
        assert_eq!(json["completely_positive"], true);
    }

    #[test]
    fn channel_summary_amplitude_damping_has_family_hint() {
        let interner = ax_ir::Interner::new();
        let gamma = ax_ir::Expr::Sym(interner.get_or_intern("gamma"));
        let expr = ax_ir::Expr::List(
            ax_qm::amplitude_damping_channel_qubit(gamma, &interner)
                .into_iter()
                .map(ax_ir::Expr::Matrix)
                .collect(),
        );

        let bundle = qm_channel_summary_bundle(&expr, &interner).expect("channel summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = &data["application/json"];
        let json_text = serde_json::to_string(json).expect("json encoding");

        assert!(markdown.contains("Family hint"), "{markdown}");
        assert!(
            json_text.contains("\"family_hint\":\"amplitude_damping\""),
            "{json_text}"
        );
        assert_eq!(json["family_hint"], "amplitude_damping");
    }

    #[test]
    fn workflow_packet_is_included_for_channel_summary_bundle() {
        let expr = ax_ir::Expr::List(vec![ax_ir::Expr::Matrix(vec![
            vec![ax_ir::Expr::one(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), ax_ir::Expr::one()],
        ])]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_channel_summary_bundle(&expr, &interner).expect("channel summary bundle");
        let data = bundle.to_jupyter_data();
        let workflow = &data["application/vnd.axioma.quantum-workflow+json"];

        assert!(
            data.contains_key("application/vnd.axioma.quantum-workflow+json"),
            "{data:?}"
        );
        assert_eq!(workflow["workflow_kind"], "channel_summary", "{workflow:?}");
        assert!(
            workflow["summary_lines"]
                .as_array()
                .is_some_and(|lines| !lines.is_empty()),
            "{workflow:?}"
        );
    }

    #[test]
    fn qm_dynamics_summary_bundle_for_time_evolution_operator() {
        let interner = ax_ir::Interner::new();
        let time_evolution_operator = interner.get_or_intern("time_evolution_operator");
        let t = interner.get_or_intern("t");
        let expr = ax_ir::Expr::Call(
            time_evolution_operator,
            vec![
                ax_ir::Expr::Matrix(vec![
                    vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
                    vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
                ]),
                ax_ir::Expr::Sym(t),
            ],
        );

        let bundle = qm_dynamics_summary_bundle(&expr, &interner).expect("dynamics summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = &data["application/json"];

        assert!(
            markdown.contains("| Object kind | operator |"),
            "{markdown}"
        );
        assert!(
            markdown.contains("| Generator kind | hamiltonian |"),
            "{markdown}"
        );
        assert_eq!(json["object_kind"], "operator");
        assert_eq!(json["generator_kind"], "hamiltonian");
    }

    #[test]
    fn qm_dynamics_summary_bundle_for_lindbladian_superoperator() {
        let interner = ax_ir::Interner::new();
        let lindbladian_superoperator = interner.get_or_intern("lindbladian_superoperator");
        let expr = ax_ir::Expr::Call(
            lindbladian_superoperator,
            vec![
                ax_ir::Expr::Matrix(vec![
                    vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
                    vec![ax_ir::Expr::zero(), ax_ir::Expr::zero()],
                ]),
                ax_ir::Expr::List(vec![]),
            ],
        );

        let bundle = qm_dynamics_summary_bundle(&expr, &interner).expect("dynamics summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = &data["application/json"];

        assert!(
            markdown.contains("| Object kind | superoperator |"),
            "{markdown}"
        );
        assert_eq!(json["object_kind"], "superoperator");
        assert_eq!(json["generator_kind"], "lindblad");
        assert_eq!(json["trace_preserving"], true);
        assert_eq!(json["purity_preserving"], false);
    }

    #[test]
    fn qm_entropy_bundle_contains_json_kind() {
        let interner = ax_ir::Interner::new();
        let von_neumann_entropy = interner.get_or_intern("von_neumann_entropy");
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Call(
            von_neumann_entropy,
            vec![ax_ir::Expr::Matrix(vec![
                vec![half.clone(), ax_ir::Expr::zero()],
                vec![ax_ir::Expr::zero(), half],
            ])],
        );

        let bundle = qm_entropy_bundle(&expr, &interner).expect("entropy bundle");
        let json = serde_json::to_string(bundle.application_json().expect("json mime"))
            .expect("json encoding");

        assert!(json.contains("\"kind\":\"von_neumann_entropy\""), "{json}");
    }

    #[test]
    fn quantum_narrative_mime_is_present_in_entropy_summary_bundle() {
        let interner = ax_ir::Interner::new();
        let von_neumann_entropy = interner.get_or_intern("von_neumann_entropy");
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Call(
            von_neumann_entropy,
            vec![ax_ir::Expr::Matrix(vec![
                vec![half.clone(), ax_ir::Expr::zero()],
                vec![ax_ir::Expr::zero(), half],
            ])],
        );

        let bundle = qm_entropy_bundle(&expr, &interner).expect("entropy bundle");
        let data = bundle.to_jupyter_data();

        assert!(
            data.contains_key("application/vnd.axioma.quantum-narrative+json"),
            "{data:?}"
        );
    }

    #[test]
    fn qm_entanglement_bundle_contains_negativity_json() {
        let interner = ax_ir::Interner::new();
        let negativity = interner.get_or_intern("negativity");
        let density = interner.get_or_intern("density");
        let sqrt = interner.get_or_intern("sqrt");
        let inv_sqrt2 = ax_ir::Expr::pow(
            ax_ir::Expr::Call(sqrt, vec![ax_ir::Expr::Int(2.into())]),
            ax_ir::Expr::Int((-1).into()),
        );
        let expr = ax_ir::Expr::Call(
            negativity,
            vec![
                ax_ir::Expr::Call(
                    density,
                    vec![ax_ir::Expr::List(vec![
                        inv_sqrt2.clone(),
                        ax_ir::Expr::zero(),
                        ax_ir::Expr::zero(),
                        inv_sqrt2,
                    ])],
                ),
                ax_ir::Expr::Int(2.into()),
                ax_ir::Expr::Int(2.into()),
            ],
        );

        let bundle = qm_entanglement_bundle(&expr, &interner).expect("entanglement bundle");
        let json = serde_json::to_string(bundle.application_json().expect("json mime"))
            .expect("json encoding");

        assert!(json.contains("\"negativity\":\"1/2\""), "{json}");
    }

    #[test]
    fn entanglement_summary_bell_state_contains_half_half_spectra() {
        let interner = ax_ir::Interner::new();
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![
                half.clone(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                half.clone(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![half.clone(), ax_ir::Expr::zero(), ax_ir::Expr::zero(), half],
        ]);

        let bundle =
            qm_entanglement_summary_bundle(&expr, &interner).expect("entanglement summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Reduced spectrum A"), "{markdown}");
        assert!(
            json.contains("\"reduced_spectrum_a\":[\"1/2\",\"1/2\"]"),
            "{json}"
        );
        assert!(
            json.contains("\"reduced_spectrum_b\":[\"1/2\",\"1/2\"]"),
            "{json}"
        );
    }

    #[test]
    fn entanglement_summary_bell_state_contains_log_two_and_half() {
        let interner = ax_ir::Interner::new();
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![
                half.clone(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                half.clone(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![half.clone(), ax_ir::Expr::zero(), ax_ir::Expr::zero(), half],
        ]);

        let bundle =
            qm_entanglement_summary_bundle(&expr, &interner).expect("entanglement summary bundle");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(json.contains("\"negativity\":\"1/2\""), "{json}");
        assert!(
            json.contains("\"logarithmic_negativity\":\"log(2)\""),
            "{json}"
        );
        assert!(markdown.contains("Reduced spectrum A"), "{markdown}");
    }

    #[test]
    fn qm_spectral_summary_bundle_for_maximally_mixed_qubit_contains_entropy() {
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![half.clone(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), half],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_spectral_summary_bundle(&expr, &interner).expect("spectral summary");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Von Neumann entropy"), "{markdown}");
        assert!(json.contains("\"eigenvalues\":[\"1/2\",\"1/2\"]"), "{json}");
    }

    #[test]
    fn qm_spectrum_summary_numeric_case_contains_svg() {
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![half.clone(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), half],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_spectral_summary_bundle(&expr, &interner).expect("spectral summary");
        let data = bundle.to_jupyter_data();
        let svg = data["image/svg+xml"].as_str().expect("svg");

        assert!(data.contains_key("image/svg+xml"), "{data:?}");
        assert!(svg.contains("<svg"), "{svg}");
    }

    #[test]
    fn qm_spectral_summary_bundle_for_bell_state_contains_negativity() {
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![
                half.clone(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                half.clone(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
            ],
            vec![half.clone(), ax_ir::Expr::zero(), ax_ir::Expr::zero(), half],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_spectral_summary_bundle(&expr, &interner).expect("spectral summary");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Negativity"), "{markdown}");
        assert!(json.contains("\"negativity\":\"1/2\""), "{json}");
    }

    #[test]
    fn workflow_packet_is_included_for_spectral_summary_bundle() {
        let half = ax_ir::Expr::pow(ax_ir::Expr::Int(2.into()), ax_ir::Expr::Int((-1).into()));
        let expr = ax_ir::Expr::Matrix(vec![
            vec![half.clone(), ax_ir::Expr::zero()],
            vec![ax_ir::Expr::zero(), half],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_spectral_summary_bundle(&expr, &interner).expect("spectral summary");
        let data = bundle.to_jupyter_data();
        let workflow = &data["application/vnd.axioma.quantum-workflow+json"];

        assert!(
            data.contains_key("application/vnd.axioma.quantum-workflow+json"),
            "{data:?}"
        );
        assert_eq!(
            workflow["workflow_kind"], "spectral_summary",
            "{workflow:?}"
        );
        assert!(
            workflow["summary_lines"]
                .as_array()
                .is_some_and(|lines| !lines.is_empty()),
            "{workflow:?}"
        );
    }
}
