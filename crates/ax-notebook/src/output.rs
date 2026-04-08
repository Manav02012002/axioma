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
}
