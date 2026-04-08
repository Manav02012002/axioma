#![forbid(unsafe_code)]

pub mod execution;
pub mod output;

use anyhow::{anyhow, Context, Result};
use ax_ir::Expr;
use execution::{apply_import, assume_message, is_plot_call};
pub use output::MimeBundle;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Response, Server};

/// Trust policy for notebook content and outputs.
///
/// `TrustedLocal` is intended for notebooks authored and executed locally in
/// the current session. `Untrusted` is the safe default for imported/shared
/// notebooks and exported content from unknown sources.
#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, Debug, Default, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NotebookTrust {
    TrustedLocal,
    #[default]
    Untrusted,
}

#[derive(serde::Serialize, Debug)]
pub struct EvalResponse {
    pub unicode: Option<String>,
    pub latex: Option<String>,
    pub error: Option<String>,
    pub svg: Option<String>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct CellData {
    pub source: String,
    pub output_latex: Option<String>,
    pub output_unicode: Option<String>,
    pub cell_type: String,
}

#[derive(serde::Deserialize, Clone, Debug, Default)]
struct ExportRequest {
    cells: Vec<CellData>,
    title: Option<String>,
    author: Option<String>,
    #[serde(default)]
    trust: NotebookTrust,
}

struct NotebookSession {
    env: ax_eval::Env,
    interner: ax_ir::Interner,
    search_paths: Vec<PathBuf>,
    last_access: Instant,
}

impl NotebookSession {
    fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            env: ax_eval::Env::new(),
            interner: ax_ir::Interner::new(),
            search_paths,
            last_access: Instant::now(),
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    fn expired(&self, now: Instant, ttl: Duration) -> bool {
        now.saturating_duration_since(self.last_access) > ttl
    }

    fn reset(&mut self) {
        self.env = ax_eval::Env::new();
        self.interner = ax_ir::Interner::new();
        self.touch();
    }
}

struct SessionStore {
    sessions: Mutex<HashMap<String, Arc<Mutex<NotebookSession>>>>,
    base_search_paths: Vec<PathBuf>,
    ttl: Duration,
}

impl SessionStore {
    fn new(base_search_paths: Vec<PathBuf>, ttl: Duration) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            base_search_paths,
            ttl,
        }
    }

    fn get_or_create(&self, session_id: &str) -> Arc<Mutex<NotebookSession>> {
        self.cleanup_expired();
        let mut sessions = match self.sessions.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(NotebookSession::new(self.base_search_paths.clone()))))
            .clone()
    }

    fn eval(&self, session_id: &str, body: &str) -> EvalResponse {
        let session = self.get_or_create(session_id);
        let mut session = match session.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        session.touch();
        let NotebookSession {
            env,
            interner,
            search_paths,
            ..
        } = &mut *session;
        handle_eval(body, env, interner, search_paths)
    }

    fn reset(&self, session_id: &str) {
        let session = self.get_or_create(session_id);
        let mut session = match session.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        session.reset();
    }

    fn cleanup_expired(&self) {
        let now = Instant::now();
        let mut sessions = match self.sessions.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        sessions.retain(|_, session| {
            let session = match session.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            !session.expired(now, self.ttl)
        });
    }

    #[cfg(test)]
    fn session_count(&self) -> usize {
        match self.sessions.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

fn escape_latex_metadata(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '\\' => "\\textbackslash{}".to_string(),
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '$' => "\\$".to_string(),
            '&' => "\\&".to_string(),
            '%' => "\\%".to_string(),
            '#' => "\\#".to_string(),
            '_' => "\\_".to_string(),
            '^' => "\\^{}".to_string(),
            '~' => "\\~{}".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}

fn escape_html(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            '\'' => "&#39;".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}

fn sanitize_url(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("javascript:")
        || lower.starts_with("vbscript:")
        || lower.starts_with("data:")
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn sanitize_html_fragment(html: &str) -> String {
    let mut builder = ammonia::Builder::default();
    builder.tags(
        [
            "a", "abbr", "b", "blockquote", "br", "code", "div", "em", "h1", "h2", "h3", "h4",
            "h5", "h6", "hr", "i", "li", "ol", "p", "pre", "span", "strong", "ul",
        ]
        .into_iter()
        .collect(),
    );
    builder.tag_attributes(
        [
            ("a", ["href", "title"].into_iter().collect()),
            ("abbr", ["title"].into_iter().collect()),
        ]
        .into_iter()
        .collect(),
    );
    builder.url_schemes(["http", "https", "mailto"].into_iter().collect());
    builder.link_rel(Some("noopener noreferrer"));
    builder.clean(html).to_string()
}

fn render_markdown_to_html(markdown: &str) -> String {
    let parser = pulldown_cmark::Parser::new_ext(
        markdown,
        pulldown_cmark::Options::ENABLE_STRIKETHROUGH
            | pulldown_cmark::Options::ENABLE_TABLES
            | pulldown_cmark::Options::ENABLE_TASKLISTS,
    );
    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, parser);
    sanitize_html_fragment(&html)
}

fn escape_latex_text(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '\\' => "\\textbackslash{}".to_string(),
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '$' => "\\$".to_string(),
            '&' => "\\&".to_string(),
            '%' => "\\%".to_string(),
            '#' => "\\#".to_string(),
            '_' => "\\_".to_string(),
            '^' => "\\^{}".to_string(),
            '~' => "\\~{}".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}

fn split_math_segments(text: &str) -> Vec<(bool, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' {
            let delim = if i + 1 < chars.len() && chars[i + 1] == '$' {
                "$$"
            } else {
                "$"
            };
            let step = delim.len();
            let mut j = i + step;
            while j < chars.len() {
                if delim == "$$" {
                    if j + 1 < chars.len() && chars[j] == '$' && chars[j + 1] == '$' {
                        if !current.is_empty() {
                            segments.push((false, std::mem::take(&mut current)));
                        }
                        segments.push((true, chars[i..=j + 1].iter().collect()));
                        i = j + 2;
                        break;
                    }
                } else if chars[j] == '$' {
                    if !current.is_empty() {
                        segments.push((false, std::mem::take(&mut current)));
                    }
                    segments.push((true, chars[i..=j].iter().collect()));
                    i = j + 1;
                    break;
                }
                j += 1;
            }
            if j >= chars.len() {
                current.push(chars[i]);
                i += 1;
            }
        } else {
            current.push(chars[i]);
            i += 1;
        }
    }
    if !current.is_empty() {
        segments.push((false, current));
    }
    segments
}

fn escape_latex_preserving_math(text: &str) -> String {
    split_math_segments(text)
        .into_iter()
        .map(|(is_math, segment)| {
            if is_math {
                segment
            } else {
                escape_latex_text(&segment)
            }
        })
        .collect()
}

fn render_markdown_to_latex(markdown: &str) -> String {
    let parser = pulldown_cmark::Parser::new_ext(
        markdown,
        pulldown_cmark::Options::ENABLE_STRIKETHROUGH
            | pulldown_cmark::Options::ENABLE_TABLES
            | pulldown_cmark::Options::ENABLE_TASKLISTS,
    );
    let mut out = String::new();
    let mut list_stack: Vec<bool> = Vec::new();
    let mut in_code_block = false;

    for event in parser {
        match event {
            pulldown_cmark::Event::Start(tag) => match tag {
                pulldown_cmark::Tag::Paragraph => {}
                pulldown_cmark::Tag::Heading { level, .. } => {
                    let cmd = match level {
                        pulldown_cmark::HeadingLevel::H1 => "\\section*{",
                        pulldown_cmark::HeadingLevel::H2 => "\\subsection*{",
                        pulldown_cmark::HeadingLevel::H3 => "\\subsubsection*{",
                        _ => "\\paragraph{",
                    };
                    out.push_str(cmd);
                }
                pulldown_cmark::Tag::Strong => out.push_str("\\textbf{"),
                pulldown_cmark::Tag::Emphasis => out.push_str("\\emph{"),
                pulldown_cmark::Tag::BlockQuote(_) => out.push_str("\\begin{quote}\n"),
                pulldown_cmark::Tag::List(Some(_)) => {
                    list_stack.push(true);
                    out.push_str("\\begin{enumerate}\n");
                }
                pulldown_cmark::Tag::List(None) => {
                    list_stack.push(false);
                    out.push_str("\\begin{itemize}\n");
                }
                pulldown_cmark::Tag::Item => out.push_str("\\item "),
                pulldown_cmark::Tag::CodeBlock(_) => {
                    in_code_block = true;
                    out.push_str("\\begin{lstlisting}\n");
                }
                pulldown_cmark::Tag::Link { dest_url, .. } => {
                    if let Some(url) = sanitize_url(dest_url.as_ref()) {
                        out.push_str("\\href{");
                        out.push_str(&escape_latex_text(&url));
                        out.push_str("}{");
                    }
                }
                _ => {}
            },
            pulldown_cmark::Event::End(tag) => match tag {
                pulldown_cmark::TagEnd::Paragraph => out.push_str("\n\n"),
                pulldown_cmark::TagEnd::Heading(_) => out.push_str("}\n\n"),
                pulldown_cmark::TagEnd::Strong | pulldown_cmark::TagEnd::Emphasis => out.push('}'),
                pulldown_cmark::TagEnd::BlockQuote(_) => out.push_str("\\end{quote}\n\n"),
                pulldown_cmark::TagEnd::List(_) => {
                    let ordered = list_stack.pop().unwrap_or(false);
                    if ordered {
                        out.push_str("\\end{enumerate}\n");
                    } else {
                        out.push_str("\\end{itemize}\n");
                    }
                }
                pulldown_cmark::TagEnd::Item => out.push('\n'),
                pulldown_cmark::TagEnd::CodeBlock => {
                    in_code_block = false;
                    out.push_str("\\end{lstlisting}\n\n");
                }
                pulldown_cmark::TagEnd::Link => out.push('}'),
                _ => {}
            },
            pulldown_cmark::Event::Text(text) => {
                if in_code_block {
                    out.push_str(text.as_ref());
                } else {
                    out.push_str(&escape_latex_preserving_math(text.as_ref()));
                }
            }
            pulldown_cmark::Event::Code(text) => {
                out.push_str("\\texttt{");
                out.push_str(&escape_latex_text(text.as_ref()));
                out.push('}');
            }
            pulldown_cmark::Event::SoftBreak => out.push('\n'),
            pulldown_cmark::Event::HardBreak => out.push_str("\\\\\n"),
            pulldown_cmark::Event::Rule => out.push_str("\\hrule\n"),
            pulldown_cmark::Event::Html(html) | pulldown_cmark::Event::InlineHtml(html) => {
                out.push_str(&escape_latex_text(html.as_ref()));
            }
            pulldown_cmark::Event::DisplayMath(math) => {
                out.push_str("\\[\n");
                out.push_str(math.as_ref());
                out.push_str("\n\\]\n");
            }
            pulldown_cmark::Event::InlineMath(math) => {
                out.push('$');
                out.push_str(math.as_ref());
                out.push('$');
            }
            pulldown_cmark::Event::FootnoteReference(text) => {
                out.push_str(&escape_latex_text(text.as_ref()));
            }
            pulldown_cmark::Event::TaskListMarker(checked) => {
                out.push_str(if checked { "[x] " } else { "[ ] " });
            }
        }
    }
    out
}

fn sanitize_svg_fragment(svg: &str) -> Option<String> {
    let root = xmltree::Element::parse(std::io::Cursor::new(svg.as_bytes())).ok()?;
    let sanitized = sanitize_svg_element(&root)?;
    let mut out = Vec::new();
    sanitized.write(&mut out).ok()?;
    String::from_utf8(out).ok()
}

fn sanitize_svg_element(elem: &xmltree::Element) -> Option<xmltree::Element> {
    const TAGS: &[&str] = &[
        "svg", "g", "path", "circle", "ellipse", "line", "polyline", "polygon", "rect",
        "text", "tspan", "defs", "clipPath", "linearGradient", "radialGradient", "stop",
        "title", "desc",
    ];
    const ATTRS: &[&str] = &[
        "xmlns", "viewBox", "width", "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy",
        "r", "rx", "ry", "d", "points", "fill", "fill-opacity", "stroke", "stroke-width",
        "stroke-opacity", "stroke-linecap", "stroke-linejoin", "transform", "opacity",
        "font-size", "font-family", "text-anchor", "class", "id", "offset", "stop-color",
        "stop-opacity", "gradientUnits", "gradientTransform", "clip-path", "xmlns:xlink",
        "xlink:href", "href",
    ];
    if !TAGS.contains(&elem.name.as_str()) {
        return None;
    }
    let mut clean = xmltree::Element::new(&elem.name);
    for (key, value) in &elem.attributes {
        if key.starts_with("on") || !ATTRS.contains(&key.as_str()) {
            continue;
        }
        if (key == "href" || key == "xlink:href") && !value.trim_start().starts_with('#') {
            continue;
        }
        clean.attributes.insert(key.clone(), value.clone());
    }
    for child in &elem.children {
        match child {
            xmltree::XMLNode::Element(child) => {
                if let Some(safe_child) = sanitize_svg_element(child) {
                    clean.children.push(xmltree::XMLNode::Element(safe_child));
                }
            }
            xmltree::XMLNode::Text(text) | xmltree::XMLNode::CData(text) => {
                clean.children.push(xmltree::XMLNode::Text(text.clone()));
            }
            _ => {}
        }
    }
    Some(clean)
}

fn request_session_id(request: &tiny_http::Request) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv("X-Axioma-Session"))
        .map(|header| header.value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
}

fn export_latex_with_meta(
    cells: &[CellData],
    title: Option<&str>,
    author: Option<&str>,
    trust: NotebookTrust,
) -> String {
    let mut out = String::from(
        "\\documentclass[11pt]{article}\n\
\\usepackage{amsmath,amssymb,amsfonts}\n\
\\usepackage{listings}\n\
\\usepackage[margin=1in]{geometry}\n\n\
\\lstset{\n\
  basicstyle=\\ttfamily\\small,\n\
  frame=single,\n\
  breaklines=true,\n\
  columns=fullflexible\n\
}\n\n",
    );
    out.push_str(&format!(
        "\\title{{{}}}\n",
        escape_latex_metadata(title.unwrap_or("Axioma Notebook"))
    ));
    out.push_str(&format!(
        "\\author{{{}}}\n",
        escape_latex_metadata(author.unwrap_or(""))
    ));
    out.push_str("\\date{\\today}\n\n\\begin{document}\n\\maketitle\n\n");

    for cell in cells {
        if cell.cell_type == "markdown" {
            if !cell.source.trim().is_empty() {
                out.push_str(&render_markdown_to_latex(&cell.source));
                out.push_str("\n\n");
            }
            continue;
        }

        out.push_str("\\begin{lstlisting}\n");
        out.push_str(&cell.source);
        if !cell.source.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("\\end{lstlisting}\n");
        if let Some(latex) = cell.output_latex.as_deref() {
            if !latex.trim().is_empty() {
                if trust == NotebookTrust::TrustedLocal {
                    out.push_str("\\[\n");
                    out.push_str(latex);
                    out.push_str("\n\\]\n");
                } else {
                    out.push_str("\\begin{verbatim}\n");
                    out.push_str(latex);
                    out.push_str("\n\\end{verbatim}\n");
                }
            }
        } else if let Some(unicode) = cell.output_unicode.as_deref() {
            if !unicode.trim().is_empty() {
                out.push_str("\\begin{verbatim}\n");
                out.push_str(unicode);
                out.push_str("\n\\end{verbatim}\n");
            }
        }
        out.push('\n');
    }

    out.push_str("\\end{document}\n");
    out
}

pub fn export_latex(cells: &[CellData]) -> String {
    export_latex_with_meta(cells, Some("Axioma Notebook"), Some(""), NotebookTrust::Untrusted)
}

fn export_html_with_title(cells: &[CellData], title: Option<&str>) -> String {
    let title = title.unwrap_or("Axioma Export");
    let mut out = String::from(
        "<!DOCTYPE html>\n\
<html>\n\
<head>\n\
<meta charset=\"UTF-8\">\n",
    );
    out.push_str(&format!("<title>{}</title>\n", escape_html(title)));
    out.push_str(
        "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css\">\n\
<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js\"></script>\n\
<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js\"></script>\n\
<style>\n\
body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: serif; line-height: 1.6; }\n\
pre { background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }\n\
.output { margin: 16px 0; font-size: 1.2em; }\n\
</style>\n\
</head>\n\
<body>\n",
    );
    out.push_str(&format!("<h1>{}</h1>\n\n", escape_html(title)));

    for cell in cells {
        if cell.cell_type == "markdown" {
            out.push_str("<div class=\"prose\">");
            out.push_str(&render_markdown_to_html(&cell.source));
            out.push_str("</div>\n\n");
            continue;
        }

        out.push_str("<pre><code>");
        out.push_str(&escape_html(&cell.source));
        out.push_str("</code></pre>\n");
        if let Some(latex) = cell.output_latex.as_deref() {
            if !latex.trim().is_empty() {
                out.push_str("<div class=\"output\">$$");
                out.push_str(latex);
                out.push_str("$$</div>\n\n");
            }
        } else if let Some(unicode) = cell.output_unicode.as_deref() {
            if !unicode.trim().is_empty() {
                out.push_str("<div class=\"output\"><pre><code>");
                out.push_str(&escape_html(unicode));
                out.push_str("</code></pre></div>\n\n");
            }
        }
    }

    out.push_str("<script>renderMathInElement(document.body);</script>\n</body>\n</html>\n");
    out
}

pub fn export_html(cells: &[CellData]) -> String {
    export_html_with_title(cells, Some("Axioma Export"))
}

pub fn handle_eval(
    body: &str,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> EvalResponse {
    let request: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return EvalResponse {
                unicode: None,
                latex: None,
                error: Some(e.to_string()),
                svg: None,
            }
        }
    };

    let source = match request.get("source").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return EvalResponse {
                unicode: None,
                latex: None,
                error: Some("missing 'source' field".into()),
                svg: None,
            }
        }
    };

    let lowered = ax_core_ir::lower(source, interner);
    if !lowered.errors.is_empty() {
        let msg = lowered
            .errors
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return EvalResponse {
            unicode: None,
            latex: None,
            error: Some(msg),
            svg: None,
        };
    }

    let mut last_unicode = None;
    let mut last_latex = None;
    let mut last_svg = None;

    for expr in &lowered.exprs {
        let rewrite_target = match expr {
            Expr::Call(f, args) if interner.resolve(*f) == "rewrite" && args.len() == 1 => {
                Some(args[0].clone())
            }
            _ => None,
        };

        if let Expr::Import(path) = expr {
            if let Err(err) = apply_import(path, env, interner, search_paths) {
                return EvalResponse {
                    unicode: None,
                    latex: None,
                    error: Some(err.to_string()),
                    svg: None,
                };
            }
            last_unicode = Some(format!(
                "imported {}",
                path.iter()
                    .map(|sym| interner.resolve(*sym))
                    .collect::<Vec<_>>()
                    .join(".")
            ));
            last_latex = None;
            continue;
        }

        if let Some(description) = ax_eval::apply_set_convention(expr, env) {
            last_unicode = Some(format!("active convention: {description}"));
            last_latex = None;
            continue;
        }

        if let Expr::Let(name, val, body) = expr {
            let evaled = ax_eval::eval(val, env, interner);
            env.bindings.insert(*name, evaled.clone());
            let display = if matches!(body.as_ref(), Expr::Sym(s) if *s == *name) {
                evaled
            } else {
                ax_eval::eval(body, env, interner)
            };
            if is_plot_call(body, interner) {
                last_svg = std::fs::read_to_string("axioma_plot.svg")
                    .ok()
                    .and_then(|svg| sanitize_svg_fragment(&svg));
                last_unicode = Some("plot saved to axioma_plot.svg".to_string());
                last_latex = None;
            } else {
                let bundle = MimeBundle::from_expr(&display, interner);
                last_unicode = bundle.text_plain().map(str::to_string);
                last_latex = bundle.text_latex().map(str::to_string);
            }
            continue;
        }

        let result = ax_eval::eval(expr, env, interner);

        if let Some(rule_name) = ax_eval::register_rule(&result, env, interner) {
            last_unicode = Some(format!("registered rule: {rule_name}"));
            last_latex = None;
            continue;
        }
        if let Expr::FnDef(name, _, _) = &result {
            env.bindings.insert(*name, result.clone());
            last_unicode = Some(format!("defined {}", interner.resolve(*name)));
            last_latex = None;
            continue;
        }
        if let Expr::Assume(var, assumptions) = &result {
            env.assumptions
                .entry(*var)
                .or_default()
                .extend(assumptions.clone());
            last_unicode = Some(assume_message(*var, assumptions, interner));
            last_latex = None;
            continue;
        }
        if let Some(message) = ax_eval::apply_grassmann_declaration(&result, env, interner) {
            last_unicode = Some(message);
            last_latex = None;
            continue;
        }
        if let Some(message) = ax_eval::apply_operator_declaration(&result, env, interner) {
            last_unicode = Some(message);
            last_latex = None;
            continue;
        }

        if is_plot_call(expr, interner) {
            last_svg = std::fs::read_to_string("axioma_plot.svg")
                .ok()
                .and_then(|svg| sanitize_svg_fragment(&svg));
            last_unicode = Some("plot saved to axioma_plot.svg".to_string());
            last_latex = None;
        } else {
            let bundle = MimeBundle::from_expr(&result, interner);
            last_unicode = bundle.text_plain().map(str::to_string);
            last_latex = bundle.text_latex().map(str::to_string);
            if let Some(target) = rewrite_target {
                let (_, trace) = ax_eval::rewrite_with_trace(&target, env, interner);
                let trust = ax_eval::describe_rewrite_trace(&trace);
                last_unicode = Some(match last_unicode.take() {
                    Some(text) => format!("{text}\n{trust}"),
                    None => trust,
                });
            }
        }
    }

    EvalResponse {
        unicode: last_unicode,
        latex: last_latex,
        error: None,
        svg: last_svg,
    }
}

pub fn start_server(port: u16) -> Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let server = Server::http(&addr).map_err(|e| anyhow!("{e}"))?;
    println!("Axioma notebook running at http://localhost:{port}");

    // Capture the notebook working directory once at startup so module lookup
    // order is stable for the lifetime of the server.
    let working_dir = std::env::current_dir().context("failed to determine notebook working directory")?;
    let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
        env_std_path: std::env::var_os("AXIOMA_STD_PATH"),
        working_dir: Some(working_dir),
        executable: std::env::current_exe().ok(),
    });
    let sessions = SessionStore::new(search_paths, Duration::from_secs(60 * 60 * 2));

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        match (method, url.as_str()) {
            (Method::Get, "/") => {
                let html = include_str!("notebook.html");
                let response = Response::from_string(html).with_header(
                    Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                );
                let _ = request.respond(response);
            }
            (Method::Post, "/export/latex") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let req = serde_json::from_str::<ExportRequest>(&body).unwrap_or_default();
                let latex =
                    export_latex_with_meta(&req.cells, req.title.as_deref(), req.author.as_deref(), req.trust);
                let response = Response::from_string(latex)
                    .with_header(
                        Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap(),
                    )
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            (Method::Post, "/export/html") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let req = serde_json::from_str::<ExportRequest>(&body).unwrap_or_default();
                let html = export_html_with_title(&req.cells, req.title.as_deref());
                let response = Response::from_string(html)
                    .with_header(
                        Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                    )
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            (Method::Post, "/eval") => {
                let session_id = match request_session_id(&request) {
                    Some(session_id) => session_id,
                    None => {
                        let response = Response::from_string(r#"{"error":"missing X-Axioma-Session header"}"#)
                            .with_status_code(400)
                            .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                            .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                        let _ = request.respond(response);
                        continue;
                    }
                };
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = sessions.eval(&session_id, &body);
                let json = serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());
                let response = Response::from_string(json)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            (Method::Post, "/reset") => {
                let session_id = match request_session_id(&request) {
                    Some(session_id) => session_id,
                    None => {
                        let response = Response::from_string(r#"{"error":"missing X-Axioma-Session header"}"#)
                            .with_status_code(400)
                            .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                            .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                        let _ = request.respond(response);
                        continue;
                    }
                };
                sessions.reset(&session_id);
                let response = Response::from_string(r#"{"ok": true}"#)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                    .with_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
                let _ = request.respond(response);
            }
            _ => {
                let response = Response::from_string("Not Found").with_status_code(404);
                let _ = request.respond(response);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn handle_eval_simple() {
        let interner = ax_ir::Interner::new();
        let mut env = ax_eval::Env::new();
        let search_paths = vec![];
        let result = handle_eval(r#"{"source": "1 + 2"}"#, &mut env, &interner, &search_paths);
        assert!(result.error.is_none());
        assert_eq!(result.unicode.as_deref(), Some("3"));
    }

    #[test]
    fn handle_eval_with_error() {
        let interner = ax_ir::Interner::new();
        let mut env = ax_eval::Env::new();
        let search_paths = vec![];
        let result = handle_eval(r#"{"source": "$$$"}"#, &mut env, &interner, &search_paths);
        assert!(result.error.is_some());
    }

    #[test]
    fn latex_export_basic() {
        let cells = vec![CellData {
            source: "1 + 1".into(),
            output_latex: Some("2".into()),
            output_unicode: Some("2".into()),
            cell_type: "code".into(),
        }];
        let latex = export_latex(&cells);
        assert!(latex.contains("\\documentclass"));
        assert!(latex.contains("1 + 1"));
        assert!(latex.contains("2"));
    }

    #[test]
    fn html_export_basic() {
        let cells = vec![CellData {
            source: "x^2".into(),
            output_latex: Some("x^{2}".into()),
            output_unicode: Some("x²".into()),
            cell_type: "code".into(),
        }];
        let html = export_html(&cells);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("x^{2}"));
        assert!(html.contains("katex"));
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axioma-ax-notebook-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn write_module(root: &Path, module: &str, source: &str) -> PathBuf {
        let mut module_path = root.to_path_buf();
        for part in module.split('.') {
            module_path.push(part);
        }
        module_path.set_extension("ax");
        if let Some(parent) = module_path.parent() {
            std::fs::create_dir_all(parent).expect("module parent");
        }
        std::fs::write(&module_path, source).expect("write module");
        module_path
    }

    #[test]
    fn handle_eval_resolves_module_from_cwd_search_path() {
        let interner = ax_ir::Interner::new();
        let mut env = ax_eval::Env::new();
        let cwd = temp_dir("cwd-resolution");
        write_module(&cwd, "local.demo", "let imported_value = 42");
        let search_paths = ax_context::build_import_search_paths(&ax_context::ImportSearchPathConfig {
            working_dir: Some(cwd),
            ..ax_context::ImportSearchPathConfig::default()
        });

        let result = handle_eval(
            r#"{"source": "import local.demo\nimported_value"}"#,
            &mut env,
            &interner,
            &search_paths,
        );
        assert_eq!(result.error, None, "unexpected error: {:?}", result.error);
        assert_eq!(result.unicode.as_deref(), Some("42"));
    }

    #[test]
    fn html_export_sanitizes_scripts_handlers_and_urls() {
        let cells = vec![CellData {
            source: r#"<script>alert(1)</script><img src=x onerror=alert(1)><a href="javascript:alert(1)">bad</a><a href="https://example.com">ok</a>"#.into(),
            output_latex: None,
            output_unicode: None,
            cell_type: "markdown".into(),
        }];
        let html = export_html(&cells);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(!html.contains("alert(1)"));
        assert!(!html.contains("onerror"));
        assert!(!html.contains("javascript:"));
        assert!(html.contains("https://example.com"));
    }

    #[test]
    fn latex_export_escapes_untrusted_content_and_latex_output() {
        let cells = vec![
            CellData {
                source: r#"# Heading \input{evil}
Text with $x^2$ and \write18{boom}"#.into(),
                output_latex: None,
                output_unicode: None,
                cell_type: "markdown".into(),
            },
            CellData {
                source: "1 + 1".into(),
                output_latex: Some(r#"\input{evil}"#.into()),
                output_unicode: Some("2".into()),
                cell_type: "code".into(),
            },
        ];
        let latex = export_latex(&cells);
        assert!(latex.contains("\\section*{Heading \\textbackslash{}input\\{evil\\}}"));
        assert!(latex.contains("Text with $x^2$ and \\textbackslash{}write18\\{boom\\}"));
        assert!(latex.contains("\\begin{verbatim}\n\\input{evil}\n\\end{verbatim}"));
    }

    #[test]
    fn trusted_latex_export_keeps_kernel_latex() {
        let cells = vec![CellData {
            source: "x^2".into(),
            output_latex: Some("x^{2}".into()),
            output_unicode: Some("x²".into()),
            cell_type: "code".into(),
        }];
        let latex = export_latex_with_meta(&cells, Some("A"), Some("B"), NotebookTrust::TrustedLocal);
        assert!(latex.contains("\\[\nx^{2}\n\\]"));
    }

    #[test]
    fn sanitize_svg_removes_scripts_handlers_and_external_urls() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(1)</script><rect width="10" height="10"/><a href="javascript:alert(1)"><text>x</text></a><use href="https://evil.test/x"/></svg>"#;
        let sanitized = sanitize_svg_fragment(svg).expect("sanitized svg");
        assert!(!sanitized.contains("<script"));
        assert!(!sanitized.contains("onload"));
        assert!(!sanitized.contains("javascript:"));
        assert!(!sanitized.contains("https://evil.test"));
        assert!(sanitized.contains("<rect"));
    }

    fn eval_body(source: &str) -> String {
        serde_json::json!({ "source": source }).to_string()
    }

    #[test]
    fn sessions_do_not_share_variables() {
        let store = SessionStore::new(Vec::new(), Duration::from_secs(60));
        let a_set = store.eval("session-a", &eval_body("let x = 1"));
        let b_read = store.eval("session-b", &eval_body("x"));
        let a_read = store.eval("session-a", &eval_body("x"));

        assert_eq!(a_set.error, None);
        assert_eq!(b_read.unicode.as_deref(), Some("x"));
        assert_eq!(a_read.unicode.as_deref(), Some("1"));
    }

    #[test]
    fn reset_only_affects_target_session() {
        let store = SessionStore::new(Vec::new(), Duration::from_secs(60));
        let _ = store.eval("session-a", &eval_body("let x = 1"));
        let _ = store.eval("session-b", &eval_body("let x = 2"));

        store.reset("session-a");

        let a_read = store.eval("session-a", &eval_body("x"));
        let b_read = store.eval("session-b", &eval_body("x"));
        assert_eq!(a_read.unicode.as_deref(), Some("x"));
        assert_eq!(b_read.unicode.as_deref(), Some("2"));
    }

    #[test]
    fn concurrent_distinct_sessions_are_isolated() {
        let store = Arc::new(SessionStore::new(Vec::new(), Duration::from_secs(60)));
        let a = Arc::clone(&store);
        let b = Arc::clone(&store);

        let thread_a = thread::spawn(move || {
            let _ = a.eval("session-a", &eval_body("let x = 10"));
            a.eval("session-a", &eval_body("x"))
        });
        let thread_b = thread::spawn(move || {
            let _ = b.eval("session-b", &eval_body("let x = 20"));
            b.eval("session-b", &eval_body("x"))
        });

        let a_result = thread_a.join().expect("thread a");
        let b_result = thread_b.join().expect("thread b");
        assert_eq!(a_result.unicode.as_deref(), Some("10"));
        assert_eq!(b_result.unicode.as_deref(), Some("20"));
    }

    #[test]
    fn expired_sessions_are_cleaned_up() {
        let store = SessionStore::new(Vec::new(), Duration::from_millis(1));
        let _ = store.eval("session-a", &eval_body("1 + 1"));
        assert_eq!(store.session_count(), 1);
        {
            let session = store.get_or_create("session-a");
            let mut session = match session.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            session.last_access = Instant::now() - Duration::from_secs(10);
        }
        store.cleanup_expired();
        assert_eq!(store.session_count(), 0);
    }
}
