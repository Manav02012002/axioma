/// Errors produced by cosmological perturbation domain-model validation.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CosmologyError {
    #[error("FRW spatial dimension must be at least 1, got {got}")]
    InvalidSpatialDimension { got: usize },

    #[error("FRW spatial dimension greater than 16 is unsupported, got {got}")]
    UnsupportedSpatialDimension { got: usize },

    #[error("missing standard scalar mode `{name}` in SVT decomposition")]
    MissingScalarMode { name: String },

    #[error("gauge `{gauge:?}` is incompatible with sector `{sector:?}`")]
    IncompatibleGaugeSector {
        gauge: crate::domain::GaugeKind,
        sector: crate::domain::SectorKind,
    },

    #[error("harmonic basis `{basis:?}` is incompatible with sector `{sector:?}`")]
    IncompatibleHarmonicBasis {
        basis: crate::domain::HarmonicBasisKind,
        sector: crate::domain::SectorKind,
    },

    #[error("background time coordinate `{time_coordinate:?}` is incompatible with requested operation `{operation}`")]
    IncompatibleTimeCoordinate {
        time_coordinate: crate::domain::TimeCoordinate,
        operation: String,
    },

    #[error("scalar gauge generator is required for operation `{operation}`")]
    MissingScalarGaugeGenerator { operation: String },

    #[error("metric ansatz requires spatial dimension 3, got {got}")]
    MetricAnsatzRequiresThreeSpatialDimensions { got: usize },

    #[error("operation `{operation}` could not factor a common scalar spatial gradient along coordinate `{coordinate}`")]
    MissingCommonScalarGradient {
        operation: String,
        coordinate: String,
    },

    #[error("operation `{operation}` could not factor a common mixed scalar spatial gradient along coordinates `{first}`, `{second}`")]
    MissingCommonMixedScalarGradient {
        operation: String,
        first: String,
        second: String,
    },

    #[error("operation `{operation}` produced a matrix with unexpected dimension {got}, expected {expected}")]
    UnexpectedMatrixDimension {
        operation: String,
        got: usize,
        expected: usize,
    },

    #[error("second-order scalar equation `{label}` contains a term that could not be classified as linear-in-second-order or quadratic-in-first-order: {rendered}")]
    UnclassifiedSecondOrderTerm { label: String, rendered: String },

    #[error(
        "could not extract scalar metric modes from transformed metric for operation `{operation}`"
    )]
    ScalarModeExtractionFailure { operation: String },
}
