#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use ax_ai_proto::{
    AiPacket, Diagnostic as ProtoDiag, FixIt as ProtoFixIt, LanguageCard, SourceFile,
    Span as ProtoSpan, AI_PACKET_VERSION,
};
use ax_syntax::parser::parse_file;
use std::fs;
use std::path::Path;

fn to_span(r: std::ops::Range<usize>) -> ProtoSpan {
    ProtoSpan {
        start: r.start,
        end: r.end,
    }
}

pub fn pack(file: &Path, out: &Path, tool_version: &str) -> Result<()> {
    let text = fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
    let hash = blake3::hash(text.as_bytes()).to_hex().to_string();

    let (_node, diags) = parse_file(&text);

    let diagnostics: Vec<ProtoDiag> = diags
        .into_iter()
        .map(|d| ProtoDiag {
            code: format!("{:?}", d.code),
            severity: format!("{:?}", d.severity),
            message: d.message,
            span: to_span(d.span),
            label: d.label,
            help: d.help,
            notes: d.notes,
            fixits: d
                .fixits
                .into_iter()
                .map(|f| ProtoFixIt {
                    span: to_span(f.span),
                    replacement: f.replacement,
                    message: f.message,
                })
                .collect(),
        })
        .collect();

    let language_card = LanguageCard {
        language: "axioma".to_string(),
        statement_terminator: ";".to_string(),
        notes: vec![
            "Statements end with ';' (Java-style).".to_string(),
            "Parser is error-tolerant; diagnostics include spans + optional fix-its.".to_string(),
            "Prefer small edits; never rewrite whole files if not necessary.".to_string(),
        ],
        examples: vec![
            "module m;".to_string(),
            "import a.b;".to_string(),
            "f(1, 2 + 3);".to_string(),
        ],
    };

    let packet = AiPacket {
        version: AI_PACKET_VERSION.to_string(),
        tool: "axioma".to_string(),
        tool_version: tool_version.to_string(),
        file: SourceFile {
            path: file.display().to_string(),
            text,
            hash_blake3_hex: hash,
        },
        diagnostics,
        language_card,
    };

    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;

    let json = serde_json::to_string_pretty(&packet).context("serialize packet")?;
    fs::write(out, json).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}
