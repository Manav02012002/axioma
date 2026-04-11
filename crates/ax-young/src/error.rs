use thiserror::Error;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum YoungError {
    #[error("Young diagram cannot be empty")]
    EmptyDiagram,
    #[error("Young diagram cannot contain zero row lengths: {rows:?}")]
    ZeroRowLength { rows: Vec<usize> },
    #[error("Young diagram rows must be weakly decreasing: {rows:?}")]
    NonDecreasingRows { rows: Vec<usize> },
    #[error("Young diagram cell ({row}, {col}) is out of bounds")]
    InvalidCell { row: usize, col: usize },
    #[error("Filled tableau contains duplicate entries")]
    DuplicateEntry,
    #[error("Filled tableau entries are not a contiguous set")]
    MissingEntries,
    #[error("Filled tableau rows do not match declared Young diagram shape")]
    ShapeMismatch,
    #[error("Semistandard tableau row {row} is not weakly increasing")]
    InvalidSemistandardRow { row: usize },
    #[error("Semistandard tableau column {col} is not strictly increasing")]
    InvalidSemistandardColumn { col: usize },
    #[error("Tableau multiplicity must have nonzero positive denominator, got {numer}/{denom}")]
    InvalidMultiplicity { numer: i64, denom: i64 },
    #[error("Self-dual column constraint failed at column {column} with length {length}")]
    SelfDualInvalidColumn { column: usize, length: usize },
    #[error("Semistandard content weight does not match tableau size")]
    InvalidContentWeight,
    #[error("Littlewood-Richardson skew tableau placement is invalid")]
    InvalidLrSkewPlacement,
    #[error("Skew diagram inner shape {inner:?} is not contained in outer shape {outer:?}")]
    InnerDiagramNotContained { outer: Vec<usize>, inner: Vec<usize> },
    #[error("Skew diagram cell ({row}, {col}) is out of bounds")]
    SkewCellOutOfBounds { row: usize, col: usize },
    #[error("Reading word is not Littlewood-Richardson lattice/Yamanouchi")]
    InvalidReadingWord,
    #[error("Content multiplicities sum to {actual_total}, but tableau requires {expected_cells} cells")]
    ContentLengthMismatch {
        expected_cells: usize,
        actual_total: usize,
    },
    #[error("Multiplicity cannot be negative")]
    NegativeMultiplicity,
    #[error("Littlewood-Richardson target size mismatch: left {left_cells} + right {right_cells} != target {target_cells}")]
    LrShapeSizeMismatch {
        left_cells: usize,
        right_cells: usize,
        target_cells: usize,
    },
    #[error("Littlewood-Richardson target shape {target:?} does not contain left shape {left:?}")]
    TargetDoesNotContainLeftShape { left: Vec<usize>, target: Vec<usize> },
    #[error("Schur expansion cannot be empty")]
    EmptySchurExpansion,
    #[error("Plethysm outer expansion must be expressed in the Schur basis")]
    InvalidPlethysmOuter,
    #[error("Plethysm inner expansion must be expressed in the Schur basis")]
    InvalidPlethysmInner,
    #[error("Branching requested for shape {shape:?} in dimension {n}, which is too small for the irrep")]
    BranchingDimensionTooSmall { shape: Vec<usize>, n: usize },
    #[error("Multiplicity basis index {index} is out of range for multiplicity {multiplicity}")]
    MultiplicityBasisIndexOutOfRange { index: usize, multiplicity: usize },
    #[error("self-dual decomposition requires middle degree in even dimension, got rank {rank} in dimension {dim}")]
    InvalidSelfDualDimension { rank: usize, dim: usize },
    #[error("Permutation degree mismatch: expected {expected}, got {actual}")]
    PermutationDegreeMismatch { expected: usize, actual: usize },
}
