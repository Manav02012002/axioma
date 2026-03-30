use std::ops::Range;
use std::path::{Path, PathBuf};

use ax_syntax::FixIt;

fn apply_fixits(src: &str, fixits: &[FixIt]) -> anyhow::Result<String> {
    let mut edits: Vec<(Range<usize>, &str)> = fixits
        .iter()
        .map(|f| (f.span.clone(), f.replacement.as_str()))
        .collect();
    edits.sort_by_key(|(r, _)| (r.start, r.end));
    for i in 1..edits.len() {
        let prev = &edits[i - 1].0;
        let cur = &edits[i].0;
        if cur.start < prev.end {
            anyhow::bail!("overlapping fix-its: {:?} overlaps {:?}", prev, cur);
        }
    }

    let mut out =
        String::with_capacity(src.len() + edits.iter().map(|(_, rep)| rep.len()).sum::<usize>());
    let mut cursor = 0usize;
    for (span, rep) in edits {
        if span.start > src.len() || span.end > src.len() || span.start > span.end {
            anyhow::bail!("invalid fix-it span: {:?}", span);
        }
        out.push_str(&src[cursor..span.start]);
        out.push_str(rep);
        cursor = span.end;
    }
    out.push_str(&src[cursor..]);
    Ok(out)
}

fn write_diags(path: &Path, diags: &[ax_syntax::Diagnostic]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(diags)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn run(file: &Path, diags_json: Option<PathBuf>, apply: bool) -> anyhow::Result<i32> {
    let src = std::fs::read_to_string(file)?;
    let (_node, diags) = ax_syntax::parser::parse_file(&src);

    if diags.is_empty() {
        if let Some(p) = diags_json.as_deref() {
            write_diags(p, &diags)?;
        }
        return Ok(0);
    }

    let mut fixits = Vec::new();
    for d in &diags {
        fixits.extend(d.fixits.clone());
    }

    if fixits.is_empty() {
        if let Some(p) = diags_json.as_deref() {
            write_diags(p, &diags)?;
        }
        return Ok(2);
    }

    let fixed = apply_fixits(&src, &fixits)?;

    if apply {
        std::fs::write(file, &fixed)?;
    } else {
        print!("{fixed}");
    }

    let (_node2, diags2) = ax_syntax::parser::parse_file(&fixed);

    if let Some(p) = diags_json.as_deref() {
        write_diags(p, &diags2)?;
    } else {
        let json = serde_json::to_string_pretty(&diags2)?;
        println!("{json}");
    }

    Ok(if diags2.is_empty() { 0 } else { 2 })
}
