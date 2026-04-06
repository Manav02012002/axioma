use ax_ir::{Expr, Index};
use std::collections::{HashMap, HashSet};

/// Classification of indices in a tensor expression.
#[derive(Clone, Debug)]
pub struct IndexClassification {
    /// Free indices: appear exactly once. Map from name to (position, Index).
    pub free: Vec<(lasso::Spur, usize, Index)>,
    /// Dummy pairs: appear exactly twice with opposite variance.
    /// Each entry is (name, pos1, pos2, Index1, Index2).
    pub dummy: Vec<(lasso::Spur, usize, usize, Index, Index)>,
    /// All indices in order of appearance.
    pub all: Vec<(usize, Index)>,
    /// Total number of index slots.
    pub total: usize,
}

/// Classify all indices in an expression into free and dummy.
pub fn classify_indices(expr: &Expr) -> IndexClassification {
    let mut all_indices: Vec<(usize, Index)> = Vec::new();
    let mut pos = 0usize;
    collect_indices_ordered(expr, &mut all_indices, &mut pos);

    let mut name_occurrences: HashMap<lasso::Spur, Vec<(usize, Index)>> = HashMap::new();
    for (p, idx) in &all_indices {
        name_occurrences
            .entry(idx.name)
            .or_default()
            .push((*p, idx.clone()));
    }

    let mut free = Vec::new();
    let mut dummy = Vec::new();

    for (name, occs) in &name_occurrences {
        if occs.len() == 1 {
            free.push((*name, occs[0].0, occs[0].1.clone()));
        } else if occs.len() == 2 && occs[0].1.variance != occs[1].1.variance {
            dummy.push((
                *name,
                occs[0].0,
                occs[1].0,
                occs[0].1.clone(),
                occs[1].1.clone(),
            ));
        } else {
            for (p, idx) in occs {
                free.push((*name, *p, idx.clone()));
            }
        }
    }

    free.sort_by_key(|(_, p, _)| *p);
    dummy.sort_by_key(|(_, p1, _, _, _)| *p1);

    IndexClassification {
        free,
        dummy,
        all: all_indices,
        total: pos,
    }
}

fn collect_indices_ordered(expr: &Expr, out: &mut Vec<(usize, Index)>, pos: &mut usize) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_indices_ordered(base, out, pos);
            for idx in indices {
                out.push((*pos, idx.clone()));
                *pos += 1;
            }
        }
        Expr::Mul(factors) => {
            for f in factors {
                collect_indices_ordered(f, out, pos);
            }
        }
        Expr::Neg(e) => collect_indices_ordered(e, out, pos),
        _ => {}
    }
}

/// Classify indices for a sum: each term independently, then verify free index consistency.
pub fn classify_indices_sum(terms: &[Expr]) -> Vec<IndexClassification> {
    terms.iter().map(classify_indices).collect()
}

/// Get a fresh dummy index name that doesn't conflict with any existing indices.
pub fn get_fresh_dummy(
    existing: &IndexClassification,
    prefix: &str,
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    let used: HashSet<lasso::Spur> = existing.all.iter().map(|(_, idx)| idx.name).collect();
    for i in 0..1000 {
        let candidate = interner.get_or_intern(&format!("{}_{}", prefix, i));
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    interner.get_or_intern(&format!("{}_fresh", prefix))
}
