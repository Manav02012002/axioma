pub mod symmetry_trace;

pub use symmetry_trace::{
    CanonicalizationTrace, DecompositionTrace, MultiplicityBasisTrace, ProjectorBuildTrace,
    SparseProjectorTrace, TableauProjectionTrace,
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
}
