#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use ax_ai_proto::{AiEditRequest, AiEditResult, Edit};
use std::{
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

fn validate_span(span: &Range<usize>, text_len: usize) -> Result<()> {
    if span.start > span.end {
        bail!("invalid span: start > end ({} > {})", span.start, span.end);
    }
    if span.end > text_len {
        bail!(
            "span out of bounds: {}..{} (len={})",
            span.start,
            span.end,
            text_len
        );
    }
    Ok(())
}

fn apply_edits(text: &str, edits: &[Edit]) -> Result<(String, usize, usize)> {
    let mut replacements = Vec::with_capacity(edits.len());
    for edit in edits {
        match edit {
            Edit::Replace { span, replacement } => {
                let range = span.start..span.end;
                validate_span(&range, text.len())?;
                replacements.push((range, replacement.as_str()));
            }
        }
    }

    replacements.sort_by_key(|(span, _)| (span.start, span.end));
    for window in replacements.windows(2) {
        let prev = &window[0].0;
        let cur = &window[1].0;
        if cur.start < prev.end {
            bail!("overlapping edits: {:?} overlaps {:?}", prev, cur);
        }
    }

    let mut output = text.to_string();
    let mut applied = 0usize;
    let mut rejected = 0usize;
    for (span, replacement) in replacements.into_iter().rev() {
        let before = &text[span.clone()];
        if before == replacement {
            rejected += 1;
            continue;
        }
        output.replace_range(span, replacement);
        applied += 1;
    }

    Ok((output, applied, rejected))
}

pub fn run(file: &Path, edits_json: &Path, out: Option<&Path>, print_result: bool) -> Result<()> {
    let text0 = fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;

    let req_s =
        fs::read_to_string(edits_json).with_context(|| format!("read {}", edits_json.display()))?;
    let req: AiEditRequest = serde_json::from_str(&req_s).context("parse AiEditRequest")?;

    let hash0 = blake3::hash(text0.as_bytes()).to_hex().to_string();
    if req.file_hash_blake3_hex != hash0 {
        bail!(
            "file hash mismatch: request={} actual={}",
            req.file_hash_blake3_hex,
            hash0
        );
    }

    let (text, applied, rejected) = apply_edits(&text0, &req.edits)?;

    let hash1 = blake3::hash(text.as_bytes()).to_hex().to_string();

    let res = AiEditResult {
        version: req.version,
        applied,
        rejected,
        output_hash_blake3_hex: hash1,
        output_text: text.clone(),
    };

    let out_text_path: PathBuf = match out {
        Some(p) => p.to_path_buf(),
        None => file.to_path_buf(),
    };
    fs::write(&out_text_path, &text)
        .with_context(|| format!("write {}", out_text_path.display()))?;

    if print_result {
        println!("{}", serde_json::to_string_pretty(&res)?);
    }

    Ok(())
}
