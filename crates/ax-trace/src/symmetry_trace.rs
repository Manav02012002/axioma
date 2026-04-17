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
