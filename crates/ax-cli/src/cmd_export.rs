use anyhow::{Context, Result};
use ax_ir::Expr;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Latex,
    Html,
}

#[derive(Clone, Debug)]
pub struct ExportOptions {
    pub format: ExportFormat,
    pub include_input: bool,
    pub include_output: bool,
    pub standalone: bool,
    pub title: Option<String>,
    pub author: Option<String>,
    pub document_class: String,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Latex,
            include_input: true,
            include_output: true,
            standalone: true,
            title: None,
            author: None,
            document_class: "article".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExportCell {
    pub input_source: String,
    pub input_line_start: usize,
    pub output_latex: Option<String>,
    pub output_unicode: Option<String>,
    pub cell_type: CellType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CellType {
    Code,
    Comment,
    Blank,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Code,
    Comment,
    Blank,
}

#[derive(Clone, Debug)]
struct SourceBlock {
    kind: BlockKind,
    line_start: usize,
    text: String,
}

pub fn collect_cells(source: &str, interner: &ax_ir::Interner) -> Vec<ExportCell> {
    let blocks = split_source_blocks(source);
    let mut env = ax_eval::Env::new();
    let mut cells = Vec::new();

    for block in blocks {
        match block.kind {
            BlockKind::Comment => cells.push(ExportCell {
                input_source: block.text,
                input_line_start: block.line_start,
                output_latex: None,
                output_unicode: None,
                cell_type: CellType::Comment,
            }),
            BlockKind::Code => {
                let (latex_outputs, unicode_outputs) =
                    evaluate_code_block(&block.text, interner, &mut env);
                cells.push(ExportCell {
                    input_source: block.text,
                    input_line_start: block.line_start,
                    output_latex: non_empty_join(latex_outputs),
                    output_unicode: non_empty_join(unicode_outputs),
                    cell_type: CellType::Code,
                });
            }
            BlockKind::Blank => cells.push(ExportCell {
                input_source: String::new(),
                input_line_start: block.line_start,
                output_latex: None,
                output_unicode: None,
                cell_type: CellType::Blank,
            }),
        }
    }

    cells
}

pub fn export_latex(cells: &[ExportCell], options: &ExportOptions) -> String {
    let mut out = String::new();

    if options.standalone {
        out.push_str(&format!(
            "\\documentclass{{{}}}\n",
            latex_escape_command_arg(&options.document_class)
        ));
        out.push_str("\\usepackage{amsmath,amssymb,amsfonts}\n");
        out.push_str("\\usepackage{listings}\n");
        out.push_str("\\usepackage{xcolor}\n");
        out.push_str("\\usepackage{geometry}\n");
        out.push_str("\\geometry{margin=2.5cm}\n");
        out.push_str("\\lstdefinelanguage{axioma}{\n");
        out.push_str("  keywords={let,import,assume,indices,coordinates,property,convention,rule,if,then,else,grassmann,depends,weight},\n");
        out.push_str("  comment=[l]{\\slash\\slash},\n");
        out.push_str("  morecomment=[s]{/*}{*/},\n");
        out.push_str("  string=[b]\",\n");
        out.push_str("  sensitive=true,\n");
        out.push_str("}\n");
        out.push_str("\\lstset{\n");
        out.push_str("  language=axioma,\n");
        out.push_str("  basicstyle=\\ttfamily\\small,\n");
        out.push_str("  keywordstyle=\\color{blue}\\bfseries,\n");
        out.push_str("  commentstyle=\\color{gray}\\itshape,\n");
        out.push_str("  backgroundcolor=\\color{gray!5},\n");
        out.push_str("  frame=single,\n");
        out.push_str("  framerule=0.5pt,\n");
        out.push_str("  breaklines=true,\n");
        out.push_str("  columns=fullflexible,\n");
        out.push_str("}\n");
        if let Some(title) = &options.title {
            out.push_str(&format!("\\title{{{}}}\n", latex_escape_text(title)));
        }
        if let Some(author) = &options.author {
            out.push_str(&format!("\\author{{{}}}\n", latex_escape_text(author)));
        }
        out.push_str("\\begin{document}\n");
        if options.title.is_some() || options.author.is_some() {
            out.push_str("\\maketitle\n\n");
        }
    }

    for cell in cells {
        match cell.cell_type {
            CellType::Blank => out.push_str("\\medskip\n\n"),
            CellType::Comment => {
                out.push_str(&format!("% Axioma source line {}\n", cell.input_line_start));
                let rendered = render_comment_latex(&cell.input_source);
                if !rendered.is_empty() {
                    out.push_str(&rendered);
                    out.push_str("\n\n");
                }
            }
            CellType::Code => {
                out.push_str(&format!("% Axioma source line {}\n", cell.input_line_start));
                if options.include_input {
                    out.push_str("\\begin{lstlisting}\n");
                    out.push_str(&sanitize_latex_listing(&cell.input_source));
                    if !cell.input_source.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str("\\end{lstlisting}\n\n");
                }
                if options.include_output {
                    if let Some(output) = &cell.output_latex {
                        out.push_str(&render_output_latex(output));
                        out.push_str("\n\n");
                    }
                }
            }
        }
    }

    if options.standalone {
        out.push_str("\\end{document}\n");
    }

    out
}

pub fn export_html(cells: &[ExportCell], options: &ExportOptions) -> String {
    let mut out = String::new();

    if options.standalone {
        out.push_str("<!DOCTYPE html>\n");
        out.push_str("<html lang=\"en\">\n");
        out.push_str("<head>\n");
        out.push_str("<meta charset=\"UTF-8\">\n");
        out.push_str(
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        out.push_str(&format!(
            "<title>{}</title>\n",
            html_escape(options.title.as_deref().unwrap_or("Axioma Export"))
        ));
        out.push_str("<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css\">\n");
        out.push_str("<script defer src=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js\"></script>\n");
        out.push_str("<script defer src=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js\"\n");
        out.push_str("  onload=\"renderMathJax()\"></script>\n");
        out.push_str("<style>\n");
        out.push_str("body { max-width: 800px; margin: 2em auto; font-family: 'Computer Modern Serif', Georgia, serif; line-height: 1.6; color: #222; padding: 0 1em; }\n");
        out.push_str("h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }\n");
        out.push_str("h2 { border-bottom: 1px solid #999; padding-bottom: 0.2em; }\n");
        out.push_str("pre.axioma-input { background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 0.8em 1em; overflow-x: auto; font-size: 0.9em; }\n");
        out.push_str(".axioma-output { margin: 0.5em 0 1.5em 0; padding: 0.5em 1em; border-left: 3px solid #4a90d9; background: #f8faff; }\n");
        out.push_str(".axioma-comment { color: #444; }\n");
        out.push_str("code { font-family: 'JetBrains Mono', 'Fira Code', monospace; }\n");
        out.push_str(".keyword { color: #0033b3; font-weight: bold; }\n");
        out.push_str(".comment { color: #8c8c8c; font-style: italic; }\n");
        out.push_str(".katex-display { overflow-x: auto; overflow-y: hidden; }\n");
        out.push_str("</style>\n");
        out.push_str("<script>\n");
        out.push_str("function renderMathJax() {\n");
        out.push_str("  renderMathInElement(document.body, {\n");
        out.push_str("    delimiters: [\n");
        out.push_str("      {left: '$$', right: '$$', display: true},\n");
        out.push_str("      {left: '$', right: '$', display: false},\n");
        out.push_str("    ]\n");
        out.push_str("  });\n");
        out.push_str("}\n");
        out.push_str("</script>\n");
        out.push_str("</head>\n");
        out.push_str("<body>\n");
        if let Some(title) = &options.title {
            out.push_str(&format!("<h1>{}</h1>\n", html_escape(title)));
        }
        if let Some(author) = &options.author {
            out.push_str(&format!(
                "<p class=\"author\">{}</p>\n",
                html_escape(author)
            ));
        }
    }

    for cell in cells {
        match cell.cell_type {
            CellType::Blank => {}
            CellType::Comment => out.push_str(&render_comment_html(
                &cell.input_source,
                cell.input_line_start,
            )),
            CellType::Code => {
                if options.include_input {
                    out.push_str(&format!(
                        "<pre class=\"axioma-input\" data-line=\"{}\"><code>",
                        cell.input_line_start
                    ));
                    out.push_str(&syntax_highlight_html(&cell.input_source));
                    out.push_str("</code></pre>\n");
                }
                if options.include_output {
                    if let Some(output) = &cell.output_latex {
                        out.push_str("<div class=\"axioma-output\">$$");
                        out.push_str(&html_math_output(output));
                        out.push_str("$$</div>\n");
                    } else if let Some(output) = &cell.output_unicode {
                        out.push_str("<div class=\"axioma-output\"><code>");
                        out.push_str(&html_escape(output).replace('\n', "<br>"));
                        out.push_str("</code></div>\n");
                    }
                }
            }
        }
    }

    if options.standalone {
        out.push_str("</body></html>\n");
    }

    out
}

pub fn export(source: &str, options: &ExportOptions, interner: &ax_ir::Interner) -> String {
    let cells = collect_cells(source, interner);
    match options.format {
        ExportFormat::Latex => export_latex(&cells, options),
        ExportFormat::Html => export_html(&cells, options),
    }
}

pub fn export_document(
    source: &str,
    interner: &ax_ir::Interner,
    options: &ExportOptions,
) -> String {
    export(source, options, interner)
}

pub fn export_file(path: &Path, options: &ExportOptions) -> Result<String> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let interner = ax_ir::Interner::new();
    Ok(export_document(&source, &interner, options))
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    input: &Path,
    output: Option<&Path>,
    format: &str,
    include_input: bool,
    include_output: bool,
    standalone: bool,
    title: Option<String>,
    author: Option<String>,
    document_class: String,
) -> Result<()> {
    let options = ExportOptions {
        format: parse_export_format(format)?,
        include_input,
        include_output,
        standalone,
        title,
        author,
        document_class,
    };
    let rendered = export_file(input, &options)?;

    if let Some(output_path) = output {
        std::fs::write(output_path, rendered)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    } else {
        print!("{rendered}");
    }

    Ok(())
}

fn parse_export_format(format: &str) -> Result<ExportFormat> {
    match format {
        "latex" | "tex" => Ok(ExportFormat::Latex),
        "html" => Ok(ExportFormat::Html),
        other => anyhow::bail!("unsupported export format: {other}; expected latex or html"),
    }
}

fn split_source_blocks(source: &str) -> Vec<SourceBlock> {
    let mut blocks = Vec::new();
    let mut current_kind: Option<BlockKind> = None;
    let mut current_text = String::new();
    let mut current_start = 1usize;

    for (idx, raw_line) in source.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim_start();

        if trimmed.is_empty() {
            flush_block(
                &mut blocks,
                &mut current_kind,
                &mut current_text,
                current_start,
            );
            blocks.push(SourceBlock {
                kind: BlockKind::Blank,
                line_start: line_no,
                text: String::new(),
            });
            continue;
        }

        let (kind, text) = if let Some(comment) = trimmed.strip_prefix("//") {
            (BlockKind::Comment, comment.trim_start().to_string())
        } else {
            (BlockKind::Code, raw_line.to_string())
        };

        if current_kind != Some(kind) {
            flush_block(
                &mut blocks,
                &mut current_kind,
                &mut current_text,
                current_start,
            );
            current_kind = Some(kind);
            current_start = line_no;
        }

        if !current_text.is_empty() {
            current_text.push('\n');
        }
        current_text.push_str(&text);
    }

    flush_block(
        &mut blocks,
        &mut current_kind,
        &mut current_text,
        current_start,
    );

    blocks
}

fn flush_block(
    blocks: &mut Vec<SourceBlock>,
    current_kind: &mut Option<BlockKind>,
    current_text: &mut String,
    current_start: usize,
) {
    if let Some(kind) = current_kind.take() {
        blocks.push(SourceBlock {
            kind,
            line_start: current_start,
            text: std::mem::take(current_text),
        });
    }
}

fn evaluate_code_block(
    code: &str,
    interner: &ax_ir::Interner,
    env: &mut ax_eval::Env,
) -> (Vec<String>, Vec<String>) {
    let _ = ax_syntax::parser::parse_file(code);
    let lowered = ax_core_ir::lower(code, interner);
    if !lowered.errors.is_empty() {
        let message = lowered
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return (
            vec![latex_text_output(&format!("Error: {message}"))],
            vec![format!("Error: {message}")],
        );
    }

    let mut latex_outputs = Vec::new();
    let mut unicode_outputs = Vec::new();

    for expr in lowered.exprs {
        if let Some(message) = apply_declaration_like_expr(&expr, env, interner) {
            latex_outputs.push(latex_text_output(&message));
            unicode_outputs.push(message);
            continue;
        }

        let result = eval_for_export(&expr, env, interner);
        if let Some(rule_name) = ax_eval::register_rule(&result, env, interner) {
            let message = format!("registered rule: {rule_name}");
            latex_outputs.push(latex_text_output(&message));
            unicode_outputs.push(message);
            continue;
        }
        if let Some(message) = apply_result_side_effects(&result, env, interner) {
            latex_outputs.push(latex_text_output(&message));
            unicode_outputs.push(message);
            continue;
        }

        latex_outputs.push(ax_render::to_latex(&result, interner));
        unicode_outputs.push(ax_render::to_unicode(&result, interner));
    }

    (latex_outputs, unicode_outputs)
}

fn eval_for_export(expr: &Expr, env: &mut ax_eval::Env, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Let(name, val, body) => {
            let evaled_val = ax_eval::eval(val, env, interner);
            env.bindings.insert(*name, evaled_val.clone());
            if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                evaled_val
            } else {
                ax_eval::eval(body, env, interner)
            }
        }
        Expr::FnDef(name, _, _) => {
            let result = ax_eval::eval(expr, env, interner);
            env.bindings.insert(*name, result.clone());
            result
        }
        _ => ax_eval::eval(expr, env, interner),
    }
}

fn apply_declaration_like_expr(
    expr: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    if let Some(description) = ax_eval::apply_set_convention(expr, env) {
        return Some(format!("active convention: {description}"));
    }
    ax_eval::apply_parallel_declaration(expr, env, interner)
        .or_else(|| ax_eval::apply_graded_declaration(expr, env, interner))
        .or_else(|| ax_eval::apply_superspace_setup(expr, env, interner))
        .or_else(|| ax_eval::apply_brst_setup(expr, env, interner))
        .or_else(|| ax_eval::apply_property_declaration(expr, env, interner))
        .or_else(|| ax_eval::apply_coordinate_declaration(expr, env, interner))
        .or_else(|| ax_eval::apply_index_declaration(expr, env, interner))
}

fn apply_result_side_effects(
    result: &Expr,
    env: &mut ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    match result {
        Expr::Assume(var, assumptions) => {
            env.assumptions
                .entry(*var)
                .or_default()
                .extend(assumptions.clone());
            Some(format!(
                "assumed {} is {}",
                interner.resolve(*var),
                assumptions
                    .iter()
                    .map(|assumption| format!("{assumption:?}").to_lowercase())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
        Expr::FnDef(name, _, _) => {
            env.bindings.insert(*name, result.clone());
            Some(format!("defined {}", interner.resolve(*name)))
        }
        _ => ax_eval::apply_grassmann_declaration(result, env, interner)
            .or_else(|| ax_eval::apply_operator_declaration(result, env, interner)),
    }
}

fn non_empty_join(outputs: Vec<String>) -> Option<String> {
    let filtered = outputs
        .into_iter()
        .filter(|output| !output.trim().is_empty())
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        None
    } else {
        Some(filtered.join("\n"))
    }
}

fn render_output_latex(output: &str) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    if lines.len() > 1 {
        let body = lines
            .into_iter()
            .map(|line| format!("  {}", line.trim()))
            .collect::<Vec<_>>()
            .join(" \\\\\n");
        format!("\\[\n\\begin{{aligned}}\n{body}\n\\end{{aligned}}\n\\]")
    } else {
        format!("\\[\n{}\n\\]", output.trim())
    }
}

fn render_comment_latex(comment: &str) -> String {
    let lines = comment
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    if lines.is_empty() {
        return String::new();
    }

    if let Some((level, title)) = heading_from_comment_lines(&lines) {
        return match level {
            1 => format!("\\section{{{}}}", latex_escape_text(&title)),
            _ => format!("\\subsection{{{}}}", latex_escape_text(&title)),
        };
    }

    lines
        .into_iter()
        .map(latex_escape_text)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_comment_html(comment: &str, line_start: usize) -> String {
    let lines = comment
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    if lines.is_empty() {
        return String::new();
    }

    if let Some((level, title)) = heading_from_comment_lines(&lines) {
        let tag = if level == 1 { "h2" } else { "h3" };
        return format!(
            "<{tag} class=\"axioma-comment\" data-line=\"{}\">{}</{tag}>\n",
            line_start,
            html_escape(&title)
        );
    }

    format!(
        "<p class=\"axioma-comment\" data-line=\"{}\">{}</p>\n",
        line_start,
        lines
            .into_iter()
            .map(html_escape)
            .collect::<Vec<_>>()
            .join("<br>")
    )
}

fn heading_from_comment_lines(lines: &[&str]) -> Option<(u8, String)> {
    if lines.len() >= 2 {
        if is_decoration_line(lines[0], '=') {
            return Some((1, lines[1].to_string()));
        }
        if is_decoration_line(lines[0], '-') {
            return Some((2, lines[1].to_string()));
        }
        if is_decoration_line(lines[1], '=') {
            return Some((1, lines[0].to_string()));
        }
        if is_decoration_line(lines[1], '-') {
            return Some((2, lines[0].to_string()));
        }
    }

    let first = lines[0];
    if first.starts_with("===") || first.ends_with("===") {
        return Some((1, trim_heading_marks(first, '=')));
    }
    if first.starts_with("---") || first.ends_with("---") {
        return Some((2, trim_heading_marks(first, '-')));
    }
    None
}

fn is_decoration_line(line: &str, marker: char) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && trimmed.chars().all(|ch| ch == marker)
}

fn trim_heading_marks(line: &str, marker: char) -> String {
    line.trim_matches(marker).trim().to_string()
}

fn syntax_highlight_html(code: &str) -> String {
    code.lines()
        .map(syntax_highlight_html_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn syntax_highlight_html_line(line: &str) -> String {
    let escaped = html_escape(line);
    if let Some(comment_start) = escaped.find("//") {
        let (code_part, comment_part) = escaped.split_at(comment_start);
        format!(
            "{}<span class=\"comment\">{}</span>",
            wrap_keywords_html(code_part),
            comment_part
        )
    } else {
        wrap_keywords_html(&escaped)
    }
}

fn wrap_keywords_html(input: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "let",
        "import",
        "assume",
        "indices",
        "coordinates",
        "property",
        "convention",
        "rule",
        "if",
        "then",
        "else",
        "grassmann",
        "depends",
        "weight",
        "diff",
        "integrate",
        "simplify",
        "solve",
        "canonicalise",
        "meld",
    ];

    let mut out = String::new();
    let chars = input.chars().collect::<Vec<_>>();
    let mut pos = 0;

    while pos < chars.len() {
        let ch = chars[pos];
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = pos;
            pos += 1;
            while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            let word = chars[start..pos].iter().collect::<String>();
            if KEYWORDS.contains(&word.as_str()) {
                out.push_str("<span class=\"keyword\">");
                out.push_str(&word);
                out.push_str("</span>");
            } else {
                out.push_str(&word);
            }
        } else {
            out.push(ch);
            pos += 1;
        }
    }

    out
}

fn html_math_output(output: &str) -> String {
    let lines = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    if lines.len() > 1 {
        format!("\\begin{{aligned}}{}\\end{{aligned}}", lines.join(" \\\\ "))
    } else {
        output.trim().to_string()
    }
}

fn sanitize_latex_listing(input: &str) -> String {
    input.replace("\\end{lstlisting}", "\\textbackslash{}end{lstlisting}")
}

fn latex_escape_command_arg(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
        .collect::<String>()
}

fn latex_escape_text(input: impl AsRef<str>) -> String {
    input.as_ref().chars().map(latex_escape_char).collect()
}

fn latex_text_output(input: &str) -> String {
    format!("\\text{{{}}}", latex_escape_text(input))
}

fn latex_escape_char(ch: char) -> String {
    match ch {
        '\\' => "\\textbackslash{}".to_string(),
        '{' => "\\{".to_string(),
        '}' => "\\}".to_string(),
        '$' => "\\$".to_string(),
        '&' => "\\&".to_string(),
        '%' => "\\%".to_string(),
        '#' => "\\#".to_string(),
        '_' => "\\_".to_string(),
        '^' => "\\textasciicircum{}".to_string(),
        '~' => "\\textasciitilde{}".to_string(),
        _ => ch.to_string(),
    }
}

fn html_escape(input: &str) -> String {
    input
        .chars()
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
