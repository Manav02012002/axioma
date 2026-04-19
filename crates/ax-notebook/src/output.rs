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
        if let Some(bundle) = qm_mime_bundle(expr, interner) {
            return bundle;
        }
        if let Some(bundle) = cpt_mime_bundle(expr, interner) {
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
}
