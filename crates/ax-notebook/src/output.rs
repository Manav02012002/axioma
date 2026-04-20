use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MimeBundle {
    text_plain: Option<String>,
    text_latex: Option<String>,
    text_markdown: Option<String>,
    text_html: Option<String>,
    image_svg_xml: Option<String>,
    application_json: Option<Value>,
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
        if let Some(bundle) = qm_entropy_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_entanglement_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_mime_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_spectral_summary_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = qm_density_summary_bundle(expr, interner) {
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
        data
    }
}

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
    let bloch_vector = ax_qm::bloch_vector(rows).ok().map(|vector| {
        vector.map(|component| ax_render::to_unicode(&component, interner))
    });
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

    let mut markdown = format!(
        "| Quantity | Value |\n| --- | --- |\n| Dimension | {} |\n| Trace | {} |\n| Purity | {} |\n| Linear entropy | {} |",
        dimension, trace, purity, linear_entropy
    );
    if let Some([x, y, z]) = bloch_vector {
        markdown.push_str(&format!("\n| Bloch vector | [{x}, {y}, {z}] |"));
    }

    Some(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
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

fn is_square_matrix(rows: &[Vec<ax_ir::Expr>]) -> bool {
    let dimension = rows.len();
    dimension > 0 && rows.iter().all(|row| row.len() == dimension)
}

fn is_diagonal_matrix(rows: &[Vec<ax_ir::Expr>]) -> bool {
    rows.iter().enumerate().all(|(i, row)| {
        row.iter()
            .enumerate()
            .all(|(j, entry)| i == j || *entry == ax_ir::Expr::zero())
    })
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

    let mut markdown = format!(
        "| Quantity | Value |\n| --- | --- |\n| Dimension | {} |\n| Eigenvalues | {} |",
        summary.dimension, unicode_eigenvalues
    );
    if let Some(value) = &summary.entropy {
        markdown.push_str(&format!(
            "\n| Von Neumann entropy | {} |",
            ax_render::to_unicode(value, interner)
        ));
    }
    if let Some(value) = &summary.renyi2_entropy {
        markdown.push_str(&format!(
            "\n| Rényi-2 entropy | {} |",
            ax_render::to_unicode(value, interner)
        ));
    }
    if let Some(value) = &summary.negativity {
        markdown.push_str(&format!(
            "\n| Negativity | {} |",
            ax_render::to_unicode(value, interner)
        ));
    }
    if let Some(value) = &summary.logarithmic_negativity {
        markdown.push_str(&format!(
            "\n| Logarithmic negativity | {} |",
            ax_render::to_unicode(value, interner)
        ));
    }

    Some(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
            .with_latex(latex_eigenvalues)
            .with_markdown(markdown)
            .with_json(json),
    )
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

    Some(
        MimeBundle::plain(ax_render::to_unicode(expr, interner))
            .with_latex(ax_render::to_latex(expr, interner))
            .with_json(json),
    )
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

    let spectrum = ax_qm::partial_transpose_spectrum_bipartite(rows, dim_a, dim_b, 1, interner)
        .ok()?;
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
            vec![
                half.clone(),
                ax_ir::Expr::zero(),
                ax_ir::Expr::zero(),
                half,
            ],
        ]);
        let interner = ax_ir::Interner::new();

        let bundle = qm_spectral_summary_bundle(&expr, &interner).expect("spectral summary");
        let data = bundle.to_jupyter_data();
        let markdown = data["text/markdown"].as_str().expect("markdown");
        let json = serde_json::to_string(&data["application/json"]).expect("json encoding");

        assert!(markdown.contains("Negativity"), "{markdown}");
        assert!(json.contains("\"negativity\":\"1/2\""), "{json}");
    }
}
