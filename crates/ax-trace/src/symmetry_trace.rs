use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableauProjectionTrace {
    pub input_expr: String,
    pub projector_shapes: Vec<Vec<usize>>,
    pub slot_maps: Vec<Vec<usize>>,
    pub canonical_slot_orders: Vec<Vec<usize>>,
    pub output_expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecompositionTrace {
    pub factor_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub multiplicities: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectorBuildTrace {
    pub shape: Vec<usize>,
    pub degree: usize,
    pub row_generator_count: usize,
    pub column_generator_count: usize,
    pub expanded_term_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalizationTrace {
    pub input_slots: Vec<usize>,
    pub candidate_count: usize,
    pub canonical_slots: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseProjectorTrace {
    pub input_term_count: usize,
    pub explored_permutation_count: usize,
    pub emitted_term_count: usize,
    pub merged_term_count: usize,
    pub dropped_due_to_budget: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiplicityBasisTrace {
    pub factors: Vec<Vec<usize>>,
    pub target: Vec<usize>,
    pub left_associated_basis: Vec<String>,
    pub right_associated_basis: Vec<String>,
    pub change_of_basis_matrix: Vec<Vec<num_rational::BigRational>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DummyCanonicalizationTrace {
    pub original_slot_labels: Vec<String>,
    pub canonical_slot_labels: Vec<String>,
    pub original_slot_permutation: Vec<usize>,
    pub canonical_slot_permutation: Vec<usize>,
    pub dummy_orbit_count: usize,
    pub symmetry_orbit_count: usize,
    pub sign: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultitermReductionTrace {
    pub original_slots: Vec<String>,
    pub pivot_slots: Vec<String>,
    pub reduced_term_count: usize,
    pub identity_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurvatureDecompositionTrace {
    pub dimension: usize,
    pub input_kind: String,
    pub output_kinds: Vec<String>,
    pub coefficient_numerators: Vec<i64>,
    pub coefficient_denominators: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleCaseTrace {
    pub case_name: String,
    pub kind: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
}
