pub mod symmetry_trace;

pub use symmetry_trace::{
    CanonicalizationTrace, CurvatureDecompositionTrace, DecompositionTrace,
    DummyCanonicalizationTrace, MultiplicityBasisTrace, MultitermReductionTrace, OracleCaseTrace,
    ProjectorBuildTrace, SparseProjectorTrace, TableauProjectionTrace,
};

use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub run_id: String,
    pub axioma_version: String,
    pub schema_hash: String,
    pub script_hash: String,
    pub exit_code: i32,
    pub elapsed_ms: u128,
    pub diagnostics_json: serde_json::Value,
}

/// Provenance record for deterministic numeric tolerance settings used by a workflow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericToleranceRecord {
    pub abs_tolerance: f64,
    pub rel_tolerance: f64,
    pub backend: String,
}

/// Concise AI-facing narrative trace for a quantum workflow result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantumNarrativeTrace {
    pub workflow_kind: String,
    pub explanation_steps: Vec<String>,
}

/// Builds a provenance record for selected numeric tolerances and backend policy.
pub fn record_numeric_tolerances(
    abs_tolerance: f64,
    rel_tolerance: f64,
    backend: impl Into<String>,
) -> NumericToleranceRecord {
    NumericToleranceRecord {
        abs_tolerance,
        rel_tolerance,
        backend: backend.into(),
    }
}

/// Build a narrative for tracing out one subsystem from a composite density matrix.
pub fn narrative_for_partial_trace(
    original_dims: &[usize],
    factor_index: usize,
) -> QuantumNarrativeTrace {
    let removed = original_dims
        .get(factor_index)
        .copied()
        .map(|dim| format!("subsystem {factor_index} with dimension {dim}"))
        .unwrap_or_else(|| format!("subsystem {factor_index}"));
    let remaining_dims = original_dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| (index != factor_index).then_some(dim.to_string()))
        .collect::<Vec<_>>();

    QuantumNarrativeTrace {
        workflow_kind: "partial_trace".to_string(),
        explanation_steps: vec![
            format!("The reduced state was formed after the calculation traced out {removed}."),
            format!(
                "The remaining subsystem dimensions are [{}].",
                remaining_dims.join(", ")
            ),
        ],
    }
}

/// Build a narrative for entropy-style summaries of a density matrix.
pub fn narrative_for_entropy(
    dimension: usize,
    entropy_functional: impl Into<String>,
) -> QuantumNarrativeTrace {
    let entropy_functional = entropy_functional.into();
    QuantumNarrativeTrace {
        workflow_kind: "spectral_summary".to_string(),
        explanation_steps: vec![
            format!(
                "The {dimension}x{dimension} density matrix was diagonalized or spectrally analyzed before evaluating state functionals."
            ),
            format!("The {entropy_functional} functional was then computed from that spectrum."),
        ],
    }
}

/// Build a narrative for bipartite negativity calculations.
pub fn narrative_for_negativity(dim_a: usize, dim_b: usize) -> QuantumNarrativeTrace {
    QuantumNarrativeTrace {
        workflow_kind: "entanglement_summary".to_string(),
        explanation_steps: vec![
            format!(
                "A partial transpose was taken on the {}x{} bipartite density matrix.",
                dim_a, dim_b
            ),
            "The partial-transpose spectrum was inspected to extract the negativity measures."
                .to_string(),
        ],
    }
}

/// Build a narrative for channel summary checks on a Kraus map.
pub fn narrative_for_channel_summary(
    dimension: usize,
    kraus_count: usize,
    trace_preserving: bool,
    unital: bool,
) -> QuantumNarrativeTrace {
    QuantumNarrativeTrace {
        workflow_kind: "channel_summary".to_string(),
        explanation_steps: vec![
            format!(
                "The {dimension}-dimensional channel was summarized from {kraus_count} Kraus operators."
            ),
            "The Choi matrix, trace-preserving condition, and unital condition were evaluated."
                .to_string(),
            format!(
                "The TP check returned {trace_preserving} and the unital check returned {unital}."
            ),
        ],
    }
}

fn atomic_write_json(path: &Path, value: &serde_json::Value) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let mut tmp = PathBuf::from(path);
    tmp.set_extension("tmp");

    let mut f = fs::File::create(&tmp)?;
    let bytes = serde_json::to_vec_pretty(value).expect("serialize trace json");
    f.write_all(&bytes)?;
    f.write_all(b"\n")?;
    f.sync_all()?;

    fs::rename(tmp, path)?;
    Ok(())
}

impl TraceReport {
    pub fn write_to_build_dir(&self, build_dir: &Path) -> std::io::Result<PathBuf> {
        let out_path = build_dir
            .join("trace")
            .join(format!("{}.json", self.run_id));

        let v = serde_json::to_value(self).expect("trace to json value");
        atomic_write_json(&out_path, &v)?;
        Ok(out_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tableau_projection_trace_round_trips_fields() {
        let trace = TableauProjectionTrace {
            input_expr: "T_{ba}".to_string(),
            projector_shapes: vec![vec![2]],
            slot_maps: vec![vec![0, 1]],
            canonical_slot_orders: vec![vec![0, 1]],
            output_expr: "T_{ab}".to_string(),
        };

        assert_eq!(trace.input_expr, "T_{ba}");
        assert_eq!(trace.projector_shapes, vec![vec![2]]);
        assert_eq!(trace.slot_maps, vec![vec![0, 1]]);
        assert_eq!(trace.canonical_slot_orders, vec![vec![0, 1]]);
        assert_eq!(trace.output_expr, "T_{ab}");
    }

    #[test]
    fn numeric_tolerance_record_serializes_and_deserializes() {
        let record = record_numeric_tolerances(1e-12, 1e-9, "plugin_sparse");
        let json = serde_json::to_string(&record).expect("serialize tolerance record");
        let decoded: NumericToleranceRecord =
            serde_json::from_str(&json).expect("deserialize tolerance record");
        assert_eq!(decoded, record);
    }

    #[test]
    fn projector_build_trace_round_trips_fields() {
        let trace = ProjectorBuildTrace {
            shape: vec![2, 1],
            degree: 3,
            row_generator_count: 1,
            column_generator_count: 1,
            expanded_term_count: 4,
        };

        assert_eq!(trace.shape, vec![2, 1]);
        assert_eq!(trace.degree, 3);
        assert_eq!(trace.row_generator_count, 1);
        assert_eq!(trace.column_generator_count, 1);
        assert_eq!(trace.expanded_term_count, 4);
    }

    #[test]
    fn canonicalization_trace_round_trips_fields() {
        let trace = CanonicalizationTrace {
            input_slots: vec![9, 3],
            candidate_count: 2,
            canonical_slots: vec![3, 9],
        };

        assert_eq!(trace.input_slots, vec![9, 3]);
        assert_eq!(trace.candidate_count, 2);
        assert_eq!(trace.canonical_slots, vec![3, 9]);
    }

    #[test]
    fn sparse_projector_trace_round_trips_fields() {
        let trace = SparseProjectorTrace {
            input_term_count: 1,
            explored_permutation_count: 4,
            emitted_term_count: 2,
            merged_term_count: 2,
            dropped_due_to_budget: false,
        };

        assert_eq!(trace.input_term_count, 1);
        assert_eq!(trace.explored_permutation_count, 4);
        assert_eq!(trace.emitted_term_count, 2);
        assert_eq!(trace.merged_term_count, 2);
        assert!(!trace.dropped_due_to_budget);
    }

    #[test]
    fn multiplicity_basis_trace_round_trips_fields() {
        let trace = MultiplicityBasisTrace {
            factors: vec![vec![1], vec![1], vec![1]],
            target: vec![2, 1],
            left_associated_basis: vec!["m0".to_string(), "m1".to_string()],
            right_associated_basis: vec!["m0".to_string(), "m1".to_string()],
            change_of_basis_matrix: vec![
                vec![
                    num_rational::BigRational::new(1.into(), 2.into()),
                    num_rational::BigRational::new((-1).into(), 2.into()),
                ],
                vec![
                    num_rational::BigRational::new(3.into(), 2.into()),
                    num_rational::BigRational::new(1.into(), 2.into()),
                ],
            ],
        };

        assert_eq!(trace.target, vec![2, 1]);
        assert_eq!(trace.left_associated_basis.len(), 2);
        assert_eq!(trace.right_associated_basis.len(), 2);
        assert_eq!(trace.change_of_basis_matrix.len(), 2);
    }

    #[test]
    fn dummy_canonicalization_trace_round_trips_fields() {
        let trace = DummyCanonicalizationTrace {
            original_slot_labels: vec!["j".into(), "i".into(), "j".into(), "i".into()],
            canonical_slot_labels: vec!["i".into(), "j".into(), "i".into(), "j".into()],
            original_slot_permutation: vec![0, 1, 2, 3],
            canonical_slot_permutation: vec![0, 1, 2, 3],
            dummy_orbit_count: 2,
            symmetry_orbit_count: 1,
            sign: 1,
        };

        assert_eq!(trace.original_slot_labels.len(), 4);
        assert_eq!(trace.canonical_slot_labels[0], "i");
        assert_eq!(trace.dummy_orbit_count, 2);
        assert_eq!(trace.sign, 1);
    }

    #[test]
    fn multiterm_reduction_trace_round_trips_fields() {
        let trace = MultitermReductionTrace {
            original_slots: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            pivot_slots: vec!["a".into(), "d".into(), "b".into(), "c".into()],
            reduced_term_count: 2,
            identity_kind: "FirstBianchi".into(),
        };

        assert_eq!(trace.original_slots.len(), 4);
        assert_eq!(trace.pivot_slots[1], "d");
        assert_eq!(trace.reduced_term_count, 2);
    }

    #[test]
    fn curvature_decomposition_trace_round_trips_fields() {
        let trace = CurvatureDecompositionTrace {
            dimension: 4,
            input_kind: "riemann_rank4".into(),
            output_kinds: vec![
                "weyl_rank4".into(),
                "metric_ricci_rank4".into(),
                "metric_scalar_rank4".into(),
            ],
            coefficient_numerators: vec![1, 1, -1],
            coefficient_denominators: vec![1, 2, 6],
        };

        assert_eq!(trace.dimension, 4);
        assert_eq!(trace.input_kind, "riemann_rank4");
        assert_eq!(trace.output_kinds.len(), 3);
        assert_eq!(trace.coefficient_denominators[2], 6);
    }

    #[test]
    fn oracle_case_trace_round_trips_fields() {
        let trace = OracleCaseTrace {
            case_name: "sym_rank2_canonicalize".into(),
            kind: "canonicalize".into(),
            expected: "T[a-, b-]".into(),
            actual: "T[a-, b-]".into(),
            passed: true,
        };

        assert_eq!(trace.case_name, "sym_rank2_canonicalize");
        assert_eq!(trace.kind, "canonicalize");
        assert_eq!(trace.expected, "T[a-, b-]");
        assert!(trace.passed);
    }

    #[test]
    fn narrative_for_partial_trace_contains_removed_subsystem_language() {
        let trace = narrative_for_partial_trace(&[2, 3, 5], 1);

        assert!(
            trace
                .explanation_steps
                .iter()
                .any(|step| step.contains("traced out")),
            "{trace:?}"
        );
    }

    #[test]
    fn narrative_for_negativity_mentions_partial_transpose() {
        let trace = narrative_for_negativity(2, 2);

        assert!(
            trace
                .explanation_steps
                .iter()
                .any(|step| step.contains("partial transpose")),
            "{trace:?}"
        );
    }
}
