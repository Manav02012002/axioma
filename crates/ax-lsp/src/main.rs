#![forbid(unsafe_code)]

mod constants;

use ax_eval::{
    algorithm_entries, builtin_entries, callable_entries, registry::format_tensor_property,
    CallableEntry, Env, ParamType,
};
use ax_ir::{Expr, Interner, TensorProperty};
use constants::{
    convention_values, greek_to_unicode, property_documentation, qm_snippet_documentation,
    CPT_CALLABLE_DOCS, GREEK_LETTERS, KEYWORDS, PROPERTY_NAMES, QFT_SNIPPETS, QM_SNIPPETS,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::ops::Range;

#[derive(Debug, Clone)]
struct DocumentAnalysis {
    symbols: Vec<(String, Range<usize>, SymbolKind)>,
}

#[derive(Debug, Clone)]
enum SymbolKind {
    Variable,
    Function,
    Index,
    Property,
    TensorSymbol,
    Coordinate,
    Module,
}

struct LspState {
    interner: ax_ir::Interner,
    documents: HashMap<String, String>,
    analyses: HashMap<String, DocumentAnalysis>,
    env: ax_eval::Env,
}

#[derive(Clone, Copy)]
struct FunctionDoc {
    name: &'static str,
    signature: &'static str,
    description: &'static str,
    example: &'static str,
}

impl LspState {
    fn new() -> Self {
        Self {
            interner: Interner::new(),
            documents: HashMap::new(),
            analyses: HashMap::new(),
            env: Env::new(),
        }
    }

    fn upsert_document(&mut self, uri: String, text: String) {
        self.documents.insert(uri, text);
        self.rebuild_analyses();
    }

    fn rebuild_analyses(&mut self) {
        self.analyses.clear();
        self.env = Env::new();

        let mut uris: Vec<String> = self.documents.keys().cloned().collect();
        uris.sort();

        for uri in uris {
            if let Some(text) = self.documents.get(&uri).cloned() {
                let lowered = ax_core_ir::lower(&text, &self.interner);
                let exprs = if lowered.exprs.is_empty() {
                    lowered.expr.into_iter().collect()
                } else {
                    lowered.exprs
                };

                for expr in &exprs {
                    apply_expr_declarations(expr, &mut self.env, &self.interner);
                }

                let analysis = analyse_document(&text, &exprs, &self.env, &self.interner);
                self.analyses.insert(uri, analysis);
            }
        }
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut state = LspState::new();

    loop {
        match read_message(&mut reader) {
            Ok(msg) => {
                if let Some(response) = handle_message(&msg, &mut state) {
                    if response != Value::Null {
                        write_message(&mut writer, &response);
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn read_message(reader: &mut impl BufRead) -> io::Result<Value> {
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading LSP headers",
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(len_str) = trimmed.strip_prefix("Content-Length: ") {
            content_length = len_str.parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_message(writer: &mut impl Write, msg: &Value) {
    if let Ok(body) = serde_json::to_string(msg) {
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let _ = writer.write_all(header.as_bytes());
        let _ = writer.write_all(body.as_bytes());
        let _ = writer.flush();
    }
}

fn offset_to_position(text: &str, offset: usize) -> (usize, usize) {
    let clamped = offset.min(text.len());
    let mut line = 0usize;
    let mut character = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        if byte_idx >= clamped {
            break;
        }
        if ch == '\n' {
            line += 1;
            character = 0;
        } else {
            character += 1;
        }
    }

    (line, character)
}

fn position_to_offset(text: &str, line: usize, character: usize) -> usize {
    let mut current_line = 0usize;
    let mut current_col = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        if current_line == line && current_col == character {
            return byte_idx;
        }
        if ch == '\n' {
            if current_line == line {
                return byte_idx;
            }
            current_line += 1;
            current_col = 0;
        } else if current_line == line {
            current_col += 1;
        }
    }

    text.len()
}

fn is_ident_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn word_at_offset(text: &str, offset: usize) -> Option<&str> {
    let clamped = offset.min(text.len());
    let mut ranges = identifier_ranges(text);
    if let Some((start, end)) = ranges
        .drain(..)
        .find(|(start, end)| *start <= clamped && clamped <= *end)
    {
        return text.get(start..end);
    }
    None
}

fn identifier_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;

    for (idx, ch) in text.char_indices() {
        if is_ident_char(ch) {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(begin) = start.take() {
            out.push((begin, idx));
        }
    }
    if let Some(begin) = start {
        out.push((begin, text.len()));
    }
    out
}

fn handle_message(msg: &Value, state: &mut LspState) -> Option<Value> {
    let method = msg.get("method")?.as_str()?;

    match method {
        "initialize" => {
            let id = msg.get("id")?;
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "completionProvider": {
                            "triggerCharacters": [".", "(", "[", " "],
                            "resolveProvider": false
                        },
                        "signatureHelpProvider": {
                            "triggerCharacters": ["(", ","]
                        },
                        "codeActionProvider": true,
                        "diagnosticProvider": {
                            "interFileDependencies": false,
                            "workspaceDiagnostics": false
                        }
                    },
                    "serverInfo": {
                        "name": "axioma-lsp",
                        "version": "0.2.0"
                    }
                }
            }))
        }
        "initialized" => None,
        "textDocument/didOpen" => {
            let params = msg.get("params")?;
            let text_doc = params.get("textDocument")?;
            let uri = text_doc.get("uri")?.as_str()?.to_string();
            let text = text_doc.get("text")?.as_str()?.to_string();
            state.upsert_document(uri.clone(), text);
            Some(publish_diagnostics(state, &uri))
        }
        "textDocument/didChange" => {
            let params = msg.get("params")?;
            let text_doc = params.get("textDocument")?;
            let uri = text_doc.get("uri")?.as_str()?.to_string();
            let changes = params.get("contentChanges")?.as_array()?;
            let text = changes.first()?.get("text")?.as_str()?.to_string();
            state.upsert_document(uri.clone(), text);
            Some(publish_diagnostics(state, &uri))
        }
        "textDocument/hover" => {
            let id = msg.get("id")?;
            let params = msg.get("params")?;
            let result = handle_hover(state, params);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result.unwrap_or(Value::Null)
            }))
        }
        "textDocument/completion" => {
            let id = msg.get("id")?;
            let params = msg.get("params")?;
            let result = handle_completion(state, params);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result.unwrap_or_else(|| json!([]))
            }))
        }
        "textDocument/signatureHelp" => {
            let id = msg.get("id")?;
            let params = msg.get("params")?;
            let result = handle_signature_help(state, params);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result.unwrap_or(Value::Null)
            }))
        }
        "textDocument/codeAction" => {
            let id = msg.get("id")?;
            let params = msg.get("params")?;
            let result = handle_code_action(state, params);
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result.unwrap_or_else(|| json!([]))
            }))
        }
        "shutdown" => {
            let id = msg.get("id")?;
            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": Value::Null
            }))
        }
        "exit" => {
            std::process::exit(0);
        }
        _ => None,
    }
}

fn publish_diagnostics(state: &LspState, uri: &str) -> Value {
    let text = state.documents.get(uri).map(String::as_str).unwrap_or("");
    let lowered = ax_core_ir::lower(text, &state.interner);
    let parse_diags = syntax_diagnostics(text);

    let mut diagnostics = Vec::new();

    for err in &lowered.errors {
        let (start_line, start_col) = offset_to_position(text, err.span.start);
        let (end_line, end_col) = offset_to_position(text, err.span.end);
        diagnostics.push(json!({
            "range": {
                "start": { "line": start_line, "character": start_col },
                "end": { "line": end_line, "character": end_col }
            },
            "severity": 1,
            "source": "axioma-lsp",
            "message": err.message
        }));
    }

    for diag in parse_diags {
        let (start_line, start_col) = offset_to_position(text, diag.span.start);
        let (end_line, end_col) = offset_to_position(text, diag.span.end);
        diagnostics.push(json!({
            "range": {
                "start": { "line": start_line, "character": start_col },
                "end": { "line": end_line, "character": end_col }
            },
            "severity": match diag.severity {
                ax_syntax::Severity::Error => 1,
                ax_syntax::Severity::Warning => 2
            },
            "source": "axioma-syntax",
            "message": diag.message
        }));
    }

    for diag in lsp_heuristic_diagnostics(state, text) {
        let (start_line, start_col) = offset_to_position(text, diag.span.start);
        let (end_line, end_col) = offset_to_position(text, diag.span.end);
        diagnostics.push(json!({
            "range": {
                "start": { "line": start_line, "character": start_col },
                "end": { "line": end_line, "character": end_col }
            },
            "severity": 1,
            "source": "axioma-lsp",
            "message": diag.message
        }));
    }

    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics
        }
    })
}

fn handle_hover(state: &LspState, params: &Value) -> Option<Value> {
    let uri = params.get("textDocument")?.get("uri")?.as_str()?;
    let position = params.get("position")?;
    let line = position.get("line")?.as_u64()? as usize;
    let character = position.get("character")?.as_u64()? as usize;

    let text = state.documents.get(uri)?;
    let offset = position_to_offset(text, line, character);
    let word = word_at_offset(text, offset);

    let mut content_parts = Vec::new();
    if let Some(tableau_hover) = tableau_hover_content(text, offset) {
        content_parts.push(tableau_hover);
    }

    if let Some(word) = word {
        if let Some(doc) = qm_snippet_documentation(word) {
            content_parts.push(format!("**{}**\n\n{}", word, doc));
        }

        if let Some(doc) = lookup_function_doc(word) {
            content_parts.push(format!("**{}**\n\n{}", doc.name, doc.description));
            content_parts.push(format!("```\n{}\n```", doc.signature));
            content_parts.push(format!("Example: `{}`", doc.example));
        } else {
            let entries = callable_entries();
            if let Some(entry) = entries.iter().find(|entry| entry.name == word) {
                content_parts.push(format!("**{}**\n\n{}", entry.name, entry.description));
                if !entry.parameters.is_empty() {
                    let params_str = entry
                        .parameters
                        .iter()
                        .map(|param| {
                            let optional = if param.required { "" } else { "?" };
                            format!(
                                "{}{}: {}",
                                param.name,
                                optional,
                                format_param_type(&param.param_type)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    content_parts.push(format!("```\n{}({})\n```", entry.name, params_str));
                }
                content_parts.push(format!("Example: `{}`", brief_example(entry)));
            }
        }

        let sym = state.interner.get_or_intern(word);
        let props = state.env.property_store.get_all(sym);
        if !props.is_empty() {
            let props_str = props
                .iter()
                .map(|prop| format_tensor_property(prop, &state.interner))
                .collect::<Vec<_>>()
                .join(", ");
            content_parts.push(format!("**Properties:** {}", props_str));
        }

        if let Some(unicode) = greek_to_unicode(word) {
            content_parts.push(format!("Greek letter: `{}`", unicode));
        }

        if let Some(family) = state.env.index_families.get(&sym) {
            let values = family
                .values
                .iter()
                .map(|value| state.interner.resolve(*value).to_string())
                .collect::<Vec<_>>();
            content_parts.push(format!(
                "**Index family:** `{}`\n\nValues: {}\n\nDimension: {}",
                state.interner.resolve(family.name),
                if values.is_empty() {
                    "(none)".to_string()
                } else {
                    values.join(", ")
                },
                family
                    .dimension
                    .map(|dim| dim.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            ));
        } else if let Some(family_sym) = state.env.index_to_family.get(&sym) {
            if let Some(family) = state.env.index_families.get(family_sym) {
                let values = family
                    .values
                    .iter()
                    .map(|value| state.interner.resolve(*value).to_string())
                    .collect::<Vec<_>>();
                content_parts.push(format!(
                    "**Index:** `{}` belongs to family `{}`\n\nValues: {}",
                    word,
                    state.interner.resolve(*family_sym),
                    if values.is_empty() {
                        "(none)".to_string()
                    } else {
                        values.join(", ")
                    }
                ));
            }
        }
    }

    if content_parts.is_empty() {
        return None;
    }

    Some(json!({
        "contents": {
            "kind": "markdown",
            "value": content_parts.join("\n\n---\n\n")
        }
    }))
}

fn syntax_diagnostics(text: &str) -> Vec<ax_syntax::Diagnostic> {
    let (_node, diags) = ax_syntax::parser::parse_file(text);
    diags
}

#[derive(Debug, Clone)]
struct HeuristicDiagnostic {
    span: Range<usize>,
    message: &'static str,
}

fn lsp_heuristic_diagnostics(state: &LspState, text: &str) -> Vec<HeuristicDiagnostic> {
    let mut diagnostics = Vec::new();
    diagnostics.extend(invalid_on_subsystem_diagnostics(state, text));
    diagnostics.extend(incompatible_compose_operator_diagnostics(state, text));
    diagnostics
}

fn invalid_on_subsystem_diagnostics(state: &LspState, text: &str) -> Vec<HeuristicDiagnostic> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }

        let mut rhs_start = i + 1;
        while rhs_start < bytes.len() && bytes[rhs_start].is_ascii_whitespace() {
            rhs_start += 1;
        }
        if rhs_start >= bytes.len() || !text[rhs_start..].chars().next().is_some_and(is_ident_char)
        {
            i += 1;
            continue;
        }

        let mut rhs_end = rhs_start;
        while rhs_end < bytes.len() && text[rhs_end..].chars().next().is_some_and(is_ident_char) {
            rhs_end += text[rhs_end..].chars().next().unwrap().len_utf8();
        }

        let rhs = &text[rhs_start..rhs_end];
        let sym = state.interner.get_or_intern(rhs);
        let is_hilbert = state
            .env
            .property_store
            .get_all(sym)
            .into_iter()
            .any(|prop| matches!(prop, TensorProperty::HilbertSpaceMeta(_)));
        if !is_hilbert {
            out.push(HeuristicDiagnostic {
                span: rhs_start..rhs_end,
                message: "on_subsystem expects a previously declared Hilbert space symbol",
            });
        }
        i = rhs_end;
    }

    out
}

fn incompatible_compose_operator_diagnostics(
    state: &LspState,
    text: &str,
) -> Vec<HeuristicDiagnostic> {
    let mut out = Vec::new();
    let needle = "compose_operators(";
    let mut search_start = 0usize;

    while let Some(relative_start) = text[search_start..].find(needle) {
        let start = search_start + relative_start;
        let args_start = start + needle.len();
        if let Some((left, right, span, next_offset)) =
            parse_simple_binary_call_args(text, args_start)
        {
            let Some(left_meta) = operator_space_metadata_of_name(state, left) else {
                search_start = next_offset;
                continue;
            };
            let Some(right_meta) = operator_space_metadata_of_name(state, right) else {
                search_start = next_offset;
                continue;
            };

            if right_meta.codomain_space != left_meta.domain_space {
                out.push(HeuristicDiagnostic {
                    span,
                    message: "compose_operators requires codomain(right) = domain(left)",
                });
            }
            search_start = next_offset;
        } else {
            search_start = start + needle.len();
        }
    }

    out
}

fn parse_simple_binary_call_args<'a>(
    text: &'a str,
    mut offset: usize,
) -> Option<(&'a str, &'a str, Range<usize>, usize)> {
    while offset < text.len()
        && text[offset..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    let left_start = offset;
    while offset < text.len() && text[offset..].chars().next().is_some_and(is_ident_char) {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    let left = text.get(left_start..offset)?;
    if left.is_empty() {
        return None;
    }

    while offset < text.len()
        && text[offset..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    if text[offset..].chars().next()? != ',' {
        return None;
    }
    offset += 1;

    while offset < text.len()
        && text[offset..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    let right_start = offset;
    while offset < text.len() && text[offset..].chars().next().is_some_and(is_ident_char) {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    let right = text.get(right_start..offset)?;
    if right.is_empty() {
        return None;
    }

    while offset < text.len()
        && text[offset..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        offset += text[offset..].chars().next()?.len_utf8();
    }
    if text[offset..].chars().next()? != ')' {
        return None;
    }

    Some((left, right, left_start..offset + 1, offset + 1))
}

fn operator_space_metadata_of_name(
    state: &LspState,
    name: &str,
) -> Option<ax_ir::OperatorSpaceMetadata> {
    let sym = state.interner.get_or_intern(name);
    state
        .env
        .property_store
        .get_all(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::OperatorSpaceMeta(metadata) => Some(metadata.clone()),
            TensorProperty::QuantumObjectMeta(metadata)
                if matches!(
                    metadata.kind,
                    ax_ir::QuantumObjectKind::Operator
                        | ax_ir::QuantumObjectKind::DensityOperator
                        | ax_ir::QuantumObjectKind::Projector
                        | ax_ir::QuantumObjectKind::Observable
                        | ax_ir::QuantumObjectKind::Channel
                ) =>
            {
                Some(ax_ir::OperatorSpaceMetadata {
                    domain_space: metadata.space_symbol,
                    codomain_space: metadata.space_symbol,
                })
            }
            _ => None,
        })
}

fn tableau_hover_content(text: &str, offset: usize) -> Option<String> {
    let (root, _diags) = ax_syntax::parser::parse_file(text);
    let tableau = ax_syntax::tableau_symmetry_expr_at_offset(&root, offset)?;
    let shapes = tableau.tableau_shapes();
    let slots = tableau.tableau_slot_maps();

    let mut lines = vec![
        "**tableau_symmetry**".to_string(),
        format!("shape_count={}", shapes.len()),
    ];
    for shape in &shapes {
        lines.push(format!("shape={shape:?}"));
    }
    for slot_map in &slots {
        lines.push(format!("slots={slot_map:?}"));
    }
    Some(lines.join("\n"))
}

fn handle_completion(state: &LspState, params: &Value) -> Option<Value> {
    let uri = params.get("textDocument")?.get("uri")?.as_str()?;
    let position = params.get("position")?;
    let line_num = position.get("line")?.as_u64()? as usize;
    let character = position.get("character")?.as_u64()? as usize;

    let text = state.documents.get(uri)?;
    let line_text = text.lines().nth(line_num).unwrap_or("");
    let prefix = slice_to_char(line_text, character);

    let mut items = Vec::new();

    if is_after_property_context(prefix) {
        for name in PROPERTY_NAMES {
            items.push(json!({
                "label": name,
                "kind": 13,
                "detail": "tensor property",
                "documentation": property_documentation(name),
            }));
        }
    } else if is_after_assume_context(prefix) {
        for name in ["real", "positive", "negative", "integer"] {
            items.push(json!({
                "label": name,
                "kind": 14
            }));
        }
    } else if let Some(field) = detect_convention_context(prefix) {
        for value in convention_values(field) {
            items.push(json!({
                "label": value,
                "kind": 13
            }));
        }
    } else {
        for kw in KEYWORDS {
            items.push(json!({
                "label": kw,
                "kind": 14
            }));
        }

        for entry in callable_entries() {
            let snippet = make_snippet(&entry);
            items.push(json!({
                "label": entry.name,
                "kind": 3,
                "detail": brief_signature(&entry),
                "documentation": { "kind": "markdown", "value": format!("{}\n\nExample: `{}`", entry.description, brief_example(&entry)) },
                "insertText": snippet,
                "insertTextFormat": 2
            }));
        }

        for doc in function_docs() {
            items.push(json!({
                "label": doc.name,
                "kind": 3,
                "detail": doc.signature,
                "documentation": { "kind": "markdown", "value": format!("{}\n\nExample: `{}`", doc.description, doc.example) }
            }));
        }

        for &(name, snippet, documentation) in QM_SNIPPETS {
            items.push(json!({
                "label": name,
                "kind": 15,
                "detail": "qm snippet",
                "documentation": { "kind": "markdown", "value": documentation },
                "insertText": snippet,
                "insertTextFormat": 2
            }));
        }

        for &(name, snippet, documentation) in QFT_SNIPPETS {
            items.push(json!({
                "label": name,
                "kind": 15,
                "detail": "qft snippet",
                "documentation": { "kind": "markdown", "value": documentation },
                "insertText": snippet,
                "insertTextFormat": 2
            }));
        }

        for &(name, unicode) in GREEK_LETTERS {
            items.push(json!({
                "label": name,
                "kind": 6,
                "detail": unicode
            }));
        }

        for symbol in declared_symbols(state) {
            items.push(json!({
                "label": symbol,
                "kind": 6
            }));
        }
    }

    Some(json!(items))
}

fn handle_signature_help(state: &LspState, params: &Value) -> Option<Value> {
    let uri = params.get("textDocument")?.get("uri")?.as_str()?;
    let position = params.get("position")?;
    let line_num = position.get("line")?.as_u64()? as usize;
    let character = position.get("character")?.as_u64()? as usize;

    let text = state.documents.get(uri)?;
    let line_text = text.lines().nth(line_num)?;
    let prefix = slice_to_char(line_text, character);
    let (func_name, active_param) = find_function_call_context(prefix)?;

    if let Some(doc) = lookup_function_doc(func_name) {
        let params = parse_signature_parameters(doc.signature);
        let params_info: Vec<Value> = params
            .iter()
            .map(|param| json!({ "label": param, "documentation": "" }))
            .collect();
        return Some(json!({
            "signatures": [{
                "label": doc.signature,
                "documentation": { "kind": "markdown", "value": format!("{}\n\nExample: `{}`", doc.description, doc.example) },
                "parameters": params_info
            }],
            "activeSignature": 0,
            "activeParameter": active_param.min(params.len().saturating_sub(1))
        }));
    }

    let entries = callable_entries();
    let entry = entries.iter().find(|entry| entry.name == func_name)?;
    let params_info: Vec<Value> = entry
        .parameters
        .iter()
        .map(|param| {
            json!({
                "label": param.name,
                "documentation": param.description
            })
        })
        .collect();

    let sig_label = format!(
        "{}({})",
        entry.name,
        entry
            .parameters
            .iter()
            .map(|param| {
                if param.required {
                    param.name.to_string()
                } else {
                    format!("{}?", param.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    );

    Some(json!({
        "signatures": [{
            "label": sig_label,
            "documentation": { "kind": "markdown", "value": format!("{}\n\nExample: `{}`", entry.description, brief_example(entry)) },
            "parameters": params_info
        }],
        "activeSignature": 0,
        "activeParameter": active_param.min(entry.parameters.len().saturating_sub(1))
    }))
}

fn handle_code_action(state: &LspState, params: &Value) -> Option<Value> {
    let uri = params.get("textDocument")?.get("uri")?.as_str()?;
    let text = state.documents.get(uri)?;
    let (_node, diags) = ax_syntax::parser::parse_file(text);

    let actions: Vec<Value> = diags
        .iter()
        .flat_map(|diag| {
            diag.fixits.iter().map(|fixit| {
                let (start_line, start_col) = offset_to_position(text, fixit.span.start);
                let (end_line, end_col) = offset_to_position(text, fixit.span.end);
                json!({
                    "title": fixit.message,
                    "kind": "quickfix",
                    "edit": {
                        "changes": {
                            uri: [{
                                "range": {
                                    "start": { "line": start_line, "character": start_col },
                                    "end": { "line": end_line, "character": end_col }
                                },
                                "newText": fixit.replacement
                            }]
                        }
                    }
                })
            })
        })
        .collect();

    Some(json!(actions))
}

fn slice_to_char(text: &str, char_count: usize) -> &str {
    if char_count == 0 {
        return "";
    }
    let mut end = text.len();
    let mut seen = 0usize;
    for (idx, _) in text.char_indices() {
        if seen == char_count {
            end = idx;
            break;
        }
        seen += 1;
    }
    if seen < char_count {
        text
    } else {
        &text[..end]
    }
}

fn is_after_property_context(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    let mut parts = trimmed.split_whitespace();
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("property"), Some(ident), None, None) if ident.chars().all(is_ident_char)
    )
}

fn is_after_assume_context(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    let mut parts = trimmed.split_whitespace();
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("assume"), Some(ident), None, None) if ident.chars().all(is_ident_char)
    )
}

fn detect_convention_context(prefix: &str) -> Option<&str> {
    let trimmed = prefix.trim_end();
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts.as_slice() {
        ["convention", field] => Some(field),
        _ => None,
    }
}

fn brief_signature(entry: &CallableEntry) -> String {
    format!(
        "{}({})",
        entry.name,
        entry
            .parameters
            .iter()
            .map(|param| param.name)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn brief_example(entry: &CallableEntry) -> &'static str {
    ax_eval::builtin_entries()
        .into_iter()
        .find(|builtin| builtin.name == entry.name)
        .map(|builtin| builtin.example)
        .or_else(|| {
            ax_eval::algorithm_entries()
                .into_iter()
                .find(|algorithm| algorithm.name == entry.name)
                .map(|algorithm| algorithm.example)
        })
        .unwrap_or(entry.name)
}

fn make_snippet(entry: &CallableEntry) -> String {
    if entry.parameters.is_empty() {
        return entry.name.to_string();
    }

    let params = entry
        .parameters
        .iter()
        .enumerate()
        .map(|(idx, param)| format!("${{{}:{}}}", idx + 1, param.name))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", entry.name, params)
}

fn format_param_type(param_type: &ParamType) -> String {
    match param_type {
        ParamType::ExprId => "expr".to_string(),
        ParamType::Code => "code".to_string(),
        ParamType::Symbol => "symbol".to_string(),
        ParamType::SymbolList => "symbol[]".to_string(),
        ParamType::Bool => "bool".to_string(),
        ParamType::Integer => "integer".to_string(),
        ParamType::Float => "float".to_string(),
        ParamType::StringEnum(options) => format!("enum({})", options.join(" | ")),
        ParamType::Matrix => "matrix".to_string(),
        ParamType::Optional(inner) => format!("optional {}", format_param_type(inner)),
    }
}

fn find_function_call_context(prefix: &str) -> Option<(&str, usize)> {
    let mut depth = 0i32;
    let mut comma_count = 0usize;
    let bytes = prefix.as_bytes();
    let mut i = bytes.len();

    while i > 0 {
        i -= 1;
        match bytes[i] {
            b')' => depth += 1,
            b'(' => {
                if depth == 0 {
                    let before = &prefix[..i];
                    let func_name = before
                        .trim_end_matches(|c: char| c.is_whitespace())
                        .rsplit(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .next()?;
                    if func_name.is_empty() {
                        return None;
                    }
                    return Some((func_name, comma_count));
                }
                depth -= 1;
            }
            b',' if depth == 0 => comma_count += 1,
            _ => {}
        }
    }
    None
}

fn declared_symbols(state: &LspState) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for analysis in state.analyses.values() {
        for (name, _, _) in &analysis.symbols {
            if seen.insert(name.clone()) {
                out.push(name.clone());
            }
        }
    }

    for sym in state.env.property_store.symbols() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    for sym in state.env.coordinates.iter().copied() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    for sym in state.env.index_families.keys().copied() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    for sym in state.env.index_to_family.keys().copied() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    for sym in state.env.assumptions.keys().copied() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    for sym in state.env.bindings.keys().copied() {
        let name = state.interner.resolve(sym).to_string();
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }

    out.sort();
    out
}

fn function_docs() -> Vec<FunctionDoc> {
    let mut docs = Vec::new();
    let mut seen = HashSet::new();

    for builtin in builtin_entries() {
        if seen.insert(builtin.name) {
            docs.push(FunctionDoc {
                name: builtin.name,
                signature: builtin.signature,
                description: builtin.description,
                example: builtin.example,
            });
        }
    }

    for algorithm in algorithm_entries() {
        if seen.insert(algorithm.name) {
            docs.push(FunctionDoc {
                name: algorithm.name,
                signature: algorithm.signature,
                description: algorithm.description,
                example: algorithm.example,
            });
        }
    }

    for alias in [
        FunctionDoc {
            name: "diff",
            signature: "diff(expr, var)",
            description:
                "Take a symbolic derivative with chain, product, and builtin function rules.",
            example: "diff(sin(x^2), x)",
        },
        FunctionDoc {
            name: "dsolve",
            signature: "dsolve(equation, y, x)",
            description: "Solve simple separable or first-order linear ODEs symbolically.",
            example: "dsolve(y - x, y, x)",
        },
    ] {
        if seen.insert(alias.name) {
            docs.push(alias);
        }
    }

    for (name, description) in CPT_CALLABLE_DOCS {
        if seen.insert(name) {
            docs.push(FunctionDoc {
                name,
                signature: name,
                description,
                example: name,
            });
        }
    }

    docs
}

fn lookup_function_doc(name: &str) -> Option<FunctionDoc> {
    function_docs().into_iter().find(|doc| doc.name == name)
}

fn parse_signature_parameters(signature: &str) -> Vec<String> {
    let Some(open) = signature.find('(') else {
        return Vec::new();
    };
    let Some(close) = signature.rfind(')') else {
        return Vec::new();
    };
    if close <= open + 1 {
        return Vec::new();
    }
    signature[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.split(':')
                .next()
                .unwrap_or(part)
                .trim()
                .trim_start_matches('&')
                .to_string()
        })
        .collect()
}

fn apply_expr_declarations(expr: &Expr, env: &mut Env, interner: &Interner) {
    let _ = ax_eval::apply_coordinate_declaration(expr, env, interner);
    let _ = ax_eval::apply_property_declaration(expr, env, interner);
    let _ = ax_eval::apply_index_declaration(expr, env, interner);
    let _ = ax_eval::apply_parallel_declaration(expr, env, interner);
    let _ = ax_eval::apply_set_convention(expr, env);

    match expr {
        Expr::Assume(sym, assumptions) => {
            env.assumptions.insert(*sym, assumptions.clone());
        }
        Expr::FnDef(name, _, body) => {
            env.bindings.insert(*name, (**body).clone());
        }
        Expr::Let(name, value, _) => {
            env.bindings.insert(*name, (**value).clone());
        }
        Expr::Call(f, args) => match interner.resolve(*f) {
            "__declare_depends" => apply_depends_declaration(args, env),
            "__declare_weight" => apply_weight_declaration(args, env, interner),
            _ => {}
        },
        _ => {}
    }
}

fn apply_depends_declaration(args: &[Expr], env: &mut Env) {
    let Some(Expr::Sym(symbol)) = args.first() else {
        return;
    };
    let Some(Expr::List(deps)) = args.get(1) else {
        return;
    };
    let dependency_symbols = deps
        .iter()
        .filter_map(|expr| match expr {
            Expr::Sym(sym) => Some(*sym),
            _ => None,
        })
        .collect::<Vec<_>>();
    let property = TensorProperty::Depends(dependency_symbols);
    env.tensor_properties
        .entry(*symbol)
        .or_default()
        .push(property.clone());
    env.property_store.declare_simple(*symbol, property);
}

fn apply_weight_declaration(args: &[Expr], env: &mut Env, interner: &Interner) {
    let Some(Expr::Sym(symbol)) = args.first() else {
        return;
    };
    let Some(weight_expr) = args.get(1) else {
        return;
    };
    let label = match args.get(2) {
        Some(Expr::Sym(label_sym)) => interner.resolve(*label_sym).to_string(),
        _ => "field".to_string(),
    };

    let weight = match weight_expr {
        Expr::Int(value) => value.to_string().parse::<i64>().ok(),
        Expr::Neg(inner) => match inner.as_ref() {
            Expr::Int(value) => value.to_string().parse::<i64>().ok().map(|v| -v),
            _ => None,
        },
        _ => None,
    };

    if let Some(weight) = weight {
        env.weights.insert((*symbol, label), weight);
    }
}

fn analyse_document(
    text: &str,
    _exprs: &[Expr],
    env: &Env,
    interner: &Interner,
) -> DocumentAnalysis {
    let property_names: HashSet<&str> = PROPERTY_NAMES.iter().copied().collect();
    let callable_names: HashSet<&str> = callable_entries()
        .into_iter()
        .map(|entry| entry.name)
        .collect();
    let coordinates: HashSet<String> = env
        .coordinates
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect();
    let indices: HashSet<String> = env
        .index_to_family
        .keys()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect();
    let tensor_symbols: HashSet<String> = env
        .property_store
        .symbols()
        .into_iter()
        .map(|sym| interner.resolve(sym).to_string())
        .collect();

    let modules = detect_module_symbols(text);
    let properties = detect_property_symbols(text);

    let mut symbols = Vec::new();
    for (start, end) in identifier_ranges(text) {
        let name = text[start..end].to_string();
        let kind = if property_names.contains(name.as_str()) || properties.contains(&name) {
            SymbolKind::Property
        } else if callable_names.contains(name.as_str()) {
            SymbolKind::Function
        } else if indices.contains(&name) {
            SymbolKind::Index
        } else if coordinates.contains(&name) {
            SymbolKind::Coordinate
        } else if modules.contains(&name) {
            SymbolKind::Module
        } else if tensor_symbols.contains(&name) {
            SymbolKind::TensorSymbol
        } else {
            SymbolKind::Variable
        };
        symbols.push((name, start..end, kind));
    }

    DocumentAnalysis { symbols }
}

fn detect_property_symbols(text: &str) -> HashSet<String> {
    text.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.as_slice() {
                ["property", symbol, ..] => Some((*symbol).to_string()),
                _ => None,
            }
        })
        .collect()
}

fn detect_module_symbols(text: &str) -> HashSet<String> {
    let mut modules = HashSet::new();
    for line in text.lines() {
        let parts: Vec<&str> = line
            .split(|ch: char| ch.is_whitespace() || ch == '.')
            .filter(|part| !part.is_empty())
            .collect();
        if parts.first() == Some(&"module") || parts.first() == Some(&"import") {
            for part in parts.into_iter().skip(1) {
                if part.chars().all(is_ident_char) {
                    modules.insert(part.to_string());
                }
            }
        }
    }
    modules
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hover_params(uri: &str, line: usize, character: usize) -> Value {
        json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        })
    }

    fn completion_params(uri: &str, line: usize, character: usize) -> Value {
        hover_params(uri, line, character)
    }

    #[test]
    fn hover_on_known_function_returns_docs() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "diff(x^2, x)".into());
        let response = handle_hover(&state, &hover_params("test.ax", 0, 1)).unwrap();
        let value = response["contents"]["value"].as_str().unwrap();
        assert!(value.contains("diff"));
        assert!(value.contains("diff(expr, var)"));
    }

    #[test]
    fn completion_after_property_keyword_returns_property_names() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "property R ".into());
        let response = handle_completion(&state, &completion_params("test.ax", 0, 11)).unwrap();
        let labels = response
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"riemann_symmetry"));
        assert!(labels.contains(&"symmetric"));
        assert!(!labels.contains(&"diff"));
    }

    #[test]
    fn completion_general_returns_functions_and_keywords() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "".into());
        let response = handle_completion(&state, &completion_params("test.ax", 0, 0)).unwrap();
        let labels = response
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"let"));
        assert!(labels.contains(&"diff"));
    }

    #[test]
    fn completion_includes_cpt_linearized_einstein() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "".into());
        let response = handle_completion(&state, &completion_params("test.ax", 0, 0)).unwrap();
        let labels = response
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"cpt_linearized_einstein"));
    }

    #[test]
    fn completion_includes_qm_snippets() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "".into());
        let response = handle_completion(&state, &completion_params("test.ax", 0, 0)).unwrap();
        let items = response.as_array().unwrap();
        let labels = items
            .iter()
            .filter_map(|item| item.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"ket"));
        assert!(labels.contains(&"bra"));
        assert!(labels.contains(&"braket"));
        assert!(labels.contains(&"dagger"));
        assert!(labels.contains(&"tensor_product"));

        let docs = items
            .iter()
            .filter_map(|item| item.get("documentation"))
            .filter_map(|doc| doc.get("value").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(docs.contains(&"Dirac ket syntax."));
        assert!(docs.contains(&"Dirac inner-product syntax."));
        assert!(docs.contains(&"Adjoint / Hermitian-conjugate syntax."));
    }

    #[test]
    fn completion_includes_qft_surface_snippets() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "".into());
        let response = handle_completion(&state, &completion_params("test.ax", 0, 0)).unwrap();
        let items = response.as_array().unwrap();

        let labels = items
            .iter()
            .filter_map(|item| item.get("label").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(labels.contains(&"commutator"));
        assert!(labels.contains(&"anticommutator"));
        assert!(labels.contains(&"normal_order"));
        assert!(labels.contains(&"subsystem_label"));
        assert!(labels.contains(&"compose_operators"));

        assert!(items.iter().any(|item| {
            item.get("label").and_then(Value::as_str) == Some("commutator")
                && item.get("detail").and_then(Value::as_str) == Some("qft snippet")
        }));
        assert!(items.iter().any(|item| {
            item.get("label").and_then(Value::as_str) == Some("normal_order")
                && item.get("detail").and_then(Value::as_str) == Some("qm snippet")
        }));
    }

    #[test]
    fn hover_docs_include_frw_background_spec() {
        let mut state = LspState::new();
        state.upsert_document(
            "test.ax".into(),
            "frw_background_spec(conformal, flat, 3)".into(),
        );
        let response = handle_hover(&state, &hover_params("test.ax", 0, 3)).unwrap();
        let value = response["contents"]["value"].as_str().unwrap();
        assert!(value.contains("frw_background_spec"));
    }

    #[test]
    fn hover_docs_include_braket() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "braket".into());
        let response = handle_hover(&state, &hover_params("test.ax", 0, 3)).unwrap();
        let value = response["contents"]["value"].as_str().unwrap();
        assert!(value.contains("Dirac inner-product syntax."));
    }

    #[test]
    fn hover_docs_include_dagger() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "dagger".into());
        let response = handle_hover(&state, &hover_params("test.ax", 0, 3)).unwrap();
        let value = response["contents"]["value"].as_str().unwrap();
        assert!(value.contains("Adjoint / Hermitian-conjugate syntax."));
    }

    #[test]
    fn hover_docs_include_compose_operators() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "compose_operators".into());
        let response = handle_hover(&state, &hover_params("test.ax", 0, 3)).unwrap();
        let value = response["contents"]["value"].as_str().unwrap();
        assert!(value.contains("Type-aware operator composition."));
    }

    #[test]
    fn signature_help_inside_function_call() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "diff(x^2, ".into());
        let response = handle_signature_help(&state, &hover_params("test.ax", 0, 10)).unwrap();
        assert_eq!(response["activeParameter"].as_u64(), Some(1));
        let label = response["signatures"][0]["label"].as_str().unwrap();
        assert!(label.contains("diff(expr, var)"));
    }

    #[test]
    fn code_action_for_parse_error_with_fixit() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "let x = 1 + ".into());
        let response = handle_code_action(
            &state,
            &json!({
                "textDocument": { "uri": "test.ax" }
            }),
        )
        .unwrap();
        assert!(response.is_array());
    }

    #[test]
    fn tableau_hover_contains_shape_and_slots() {
        let hover = tableau_hover_content(
            "tableau_symmetry([[2,1]], slots=[[0,1,2]]);",
            "tableau_symmetry".len(),
        )
        .unwrap();
        assert!(hover.contains("shape=[2, 1]"));
        assert!(hover.contains("slots=[0, 1, 2]"));
    }

    #[test]
    fn tableau_parser_mismatch_diagnostic_is_forwarded() {
        let diags = syntax_diagnostics("tableau_symmetry([[2,1]], slots=[[0,1],[2]]);");
        assert!(diags.iter().any(|diag| {
            diag.message == "tableau_symmetry shapes and slots lists must have the same length"
        }));
    }

    #[test]
    fn tableau_hover_is_exact_for_multiple_tableaux() {
        let hover = tableau_hover_content(
            "tableau_symmetry([[2,1],[1,1]], slots=[[0,1,2],[1,2]]);",
            "tableau_symmetry".len(),
        )
        .unwrap();
        assert_eq!(
            hover,
            concat!(
                "**tableau_symmetry**\n",
                "shape_count=2\n",
                "shape=[2, 1]\n",
                "shape=[1, 1]\n",
                "slots=[0, 1, 2]\n",
                "slots=[1, 2]"
            )
        );
    }

    #[test]
    fn diagnostic_reports_invalid_on_subsystem_space() {
        let mut state = LspState::new();
        state.upsert_document("test.ax".into(), "A@Qbad;".into());
        let response = publish_diagnostics(&state, "test.ax");
        let messages = response["params"]["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|diag| diag.get("message").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(
            messages.contains(&"on_subsystem expects a previously declared Hilbert space symbol")
        );
    }

    #[test]
    fn diagnostic_reports_incompatible_compose_operators() {
        let mut state = LspState::new();
        state.upsert_document(
            "test.ax".into(),
            concat!(
                "declare_hilbert_space(HA, 2);\n",
                "declare_hilbert_space(HB, 2);\n",
                "declare_hilbert_space(HC, 2);\n",
                "declare_operator_space(A, HA, HB);\n",
                "declare_operator_space(B, HC, HC);\n",
                "compose_operators(A, B);\n"
            )
            .into(),
        );
        let response = publish_diagnostics(&state, "test.ax");
        let messages = response["params"]["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|diag| diag.get("message").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(messages.contains(&"compose_operators requires codomain(right) = domain(left)"));
    }
}
