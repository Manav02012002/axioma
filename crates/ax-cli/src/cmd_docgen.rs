use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

fn esc(cell: &str) -> String {
    cell.replace('|', "\\|").replace('\n', "<br>")
}

fn compact_signature(sig: &str) -> String {
    let sig = sig.to_string();
    let mut out = String::new();
    let mut chars = sig.chars().peekable();
    let mut paren_depth = 0usize;
    let mut angle_depth = 0usize;
    let mut skipping_type = false;
    let mut seen_name = false;

    while let Some(ch) = chars.next() {
        if !seen_name {
            if ch == '<' {
                let mut depth = 1usize;
                for next in chars.by_ref() {
                    if next == '<' {
                        depth += 1;
                    } else if next == '>' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
                continue;
            }
            if ch == '(' {
                seen_name = true;
                paren_depth = 1;
                out.push(ch);
                continue;
            }
            out.push(ch);
            continue;
        }

        if skipping_type {
            match ch {
                '<' | '[' | '(' => {
                    angle_depth += 1;
                }
                '>' | ']' | ')' => {
                    if angle_depth > 0 {
                        angle_depth -= 1;
                    } else if ch == ')' && paren_depth == 1 {
                        skipping_type = false;
                        paren_depth -= 1;
                        out.push(')');
                    }
                }
                ',' if angle_depth == 0 && paren_depth == 1 => {
                    skipping_type = false;
                    out.push(',');
                    out.push(' ');
                    while let Some(' ') = chars.peek() {
                        chars.next();
                    }
                }
                _ => {}
            }
            continue;
        }

        match ch {
            ':' if paren_depth == 1 => {
                skipping_type = true;
            }
            '(' => {
                paren_depth += 1;
                out.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                out.push(ch);
                if paren_depth == 0 {
                    break;
                }
            }
            _ => out.push(ch),
        }
    }

    out.trim().to_string()
}

fn compact_algorithm_signature(sig: &str) -> String {
    let compact = compact_signature(sig);
    let Some(open) = compact.find('(') else {
        return compact;
    };
    let Some(close) = compact.rfind(')') else {
        return compact;
    };
    let name = &compact[..open];
    let args = &compact[open + 1..close];
    let hidden = [
        "interner",
        "env",
        "tensor_properties",
        "properties",
        "index_to_family",
        "index_families",
        "operators",
        "contractions",
        "weights",
        "label",
        "gamma_sym",
        "metric_sym",
        "epsilon_sym",
        "delta_sym",
        "dim_sym",
    ];
    let kept = args
        .split(',')
        .map(str::trim)
        .filter(|arg| !arg.is_empty())
        .filter(|arg| !hidden.contains(arg))
        .collect::<Vec<_>>();
    format!("{name}({})", kept.join(", "))
}

fn push_table_header(out: &mut String, cols: &[&str]) {
    out.push('|');
    for col in cols {
        let _ = write!(out, "{}|", col);
    }
    out.push('\n');
    out.push('|');
    for _ in cols {
        out.push_str("---|");
    }
    out.push('\n');
}

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

pub fn run(out: &Path) -> Result<()> {
    let builtins = ax_eval::builtin_entries();
    let properties = ax_eval::property_entries();
    let conventions = ax_eval::convention_entries();
    let assumptions = ax_eval::assumption_entries();
    let algorithms = ax_eval::algorithm_entries();
    let syntax = ax_eval::syntax_rules();
    let modules = ax_eval::std_modules();

    let schwarzschild = read_text(Path::new("examples/schwarzschild.ax"))?;
    let qm_spin = read_text(Path::new("std/qm/spin.ax"))?;
    let calculus_demo = read_text(Path::new("examples/calculus_demo.ax"))?;

    let mut doc = String::new();
    doc.push_str("# Axioma Language Reference (LLM Context)\n\n");
    doc.push_str("> This file is auto-generated. It is the complete reference for the Axioma scientific computing language. Inject this into your LLM system prompt or tool description when working with .ax files.\n\n");

    doc.push_str("## Syntax\n");
    push_table_header(&mut doc, &["pattern", "meaning", "example"]);
    for rule in syntax {
        let _ = writeln!(
            doc,
            "|{}|{}|{}|",
            esc(rule.pattern),
            esc(rule.meaning),
            esc(rule.example)
        );
    }
    doc.push('\n');

    doc.push_str("## Built-in Functions\n");
    let mut builtins_by_category: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for entry in builtins {
        builtins_by_category
            .entry(entry.category)
            .or_default()
            .push(entry);
    }
    for (category, mut entries) in builtins_by_category {
        entries.sort_by(|a, b| a.name.cmp(b.name).then(a.signature.cmp(b.signature)));
        let _ = writeln!(doc, "### {}", esc(category));
        push_table_header(&mut doc, &["name", "signature", "description"]);
        for entry in entries {
            let _ = writeln!(
                doc,
                "|{}|{}|{}|",
                esc(entry.name),
                esc(&compact_signature(entry.signature)),
                esc(entry.description)
            );
        }
        doc.push('\n');
    }

    doc.push_str("## Tensor Properties\n");
    push_table_header(&mut doc, &["property", "syntax", "description", "enables"]);
    for entry in properties {
        let _ = writeln!(
            doc,
            "|{}|{}|{}|{}|",
            esc(entry.name),
            esc(entry.syntax),
            esc(entry.description),
            esc(entry.enables)
        );
    }
    doc.push('\n');

    doc.push_str("## Conventions\n");
    push_table_header(&mut doc, &["field", "options", "default", "description"]);
    for entry in conventions {
        let _ = writeln!(
            doc,
            "|{}|{}|{}|{}|",
            esc(entry.field),
            esc(entry.options),
            esc(entry.default),
            esc(entry.description)
        );
    }
    doc.push('\n');

    doc.push_str("## Assumptions\n");
    push_table_header(&mut doc, &["name", "description"]);
    for entry in assumptions {
        let _ = writeln!(
            doc,
            "|{}|{}|",
            esc(entry.name),
            esc(entry.description)
        );
    }
    doc.push('\n');

    doc.push_str("## Algorithms\n");
    let mut algos_by_category: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for entry in algorithms {
        algos_by_category
            .entry(entry.category)
            .or_default()
            .push(entry);
    }
    for (category, mut entries) in algos_by_category {
        entries.sort_by(|a, b| a.name.cmp(b.name).then(a.signature.cmp(b.signature)));
        let _ = writeln!(doc, "### {}", esc(category));
        push_table_header(&mut doc, &["name", "signature", "preconditions", "description"]);
        for entry in entries {
            let _ = writeln!(
                doc,
                "|{}|{}|{}|{}|",
                esc(entry.name),
                esc(&compact_algorithm_signature(entry.signature)),
                esc(entry.preconditions),
                esc(entry.description)
            );
        }
        doc.push('\n');
    }

    doc.push_str("## Standard Library\n");
    push_table_header(&mut doc, &["module", "description", "provides"]);
    for module in modules {
        let _ = writeln!(
            doc,
            "|{}|{}|{}|",
            esc(module.path),
            esc(module.description),
            esc(module.provides)
        );
    }
    doc.push('\n');

    doc.push_str("## Common Workflows\n");

    doc.push_str("### Schwarzschild Ricci tensor\n");
    doc.push_str("```ax\n");
    doc.push_str(&schwarzschild);
    if !schwarzschild.ends_with('\n') {
        doc.push('\n');
    }
    doc.push_str("```\n\n");

    doc.push_str("### QM spin algebra\n");
    doc.push_str("```ax\n");
    doc.push_str(&qm_spin);
    if !qm_spin.ends_with('\n') {
        doc.push('\n');
    }
    doc.push_str("```\n\n");

    doc.push_str("### Calculus demo\n");
    doc.push_str("```ax\n");
    doc.push_str(&calculus_demo);
    if !calculus_demo.ends_with('\n') {
        doc.push('\n');
    }
    doc.push_str("```\n");

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(out, doc).with_context(|| format!("failed to write {}", out.display()))?;
    Ok(())
}
