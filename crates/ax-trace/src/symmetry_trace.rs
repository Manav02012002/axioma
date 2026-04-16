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
