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

    #[error("harmonic basis `{basis:?}` is incompatible with spatial curvature `{curvature:?}`")]
    IncompatibleHarmonicCurvature {
        basis: crate::domain::HarmonicBasisKind,
        curvature: crate::domain::SpatialCurvature,
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

    #[error(
        "vector projection for operation `{operation}` failed along coordinate `{coordinate}`"
    )]
    VectorProjectionFailure {
        operation: String,
        coordinate: String,
    },

    #[error("tensor transverse-traceless projection for operation `{operation}` failed")]
    TensorProjectionFailure { operation: String },

    #[error("helicity basis `{basis}` is unsupported for operation `{operation}`")]
    UnsupportedHelicityBasis { basis: String, operation: String },

    #[error("harmonic mode projection for operation `{operation}` failed")]
    HarmonicProjectionFailure { operation: String },

    #[error("harmonic eigenvalue for sector `{sector:?}` and curvature `{curvature:?}` is undefined in this implementation")]
    UndefinedHarmonicEigenvalue {
        sector: crate::domain::SectorKind,
        curvature: crate::domain::SpatialCurvature,
    },

    #[error("multifield system requires at least 2 fields, got {got}")]
    InvalidFieldCount { got: usize },

    #[error("adiabatic-entropy rotation for `{operation}` failed")]
    AdiabaticEntropyRotationFailure { operation: String },

    #[error("Boltzmann bridge export target `{target}` is unsupported")]
    UnsupportedBoltzmannExportTarget { target: String },

    #[error("second-order vector equation `{label}` contains an unclassified term: {rendered}")]
    UnclassifiedSecondOrderVectorTerm { label: String, rendered: String },

    #[error("second-order tensor equation `{label}` contains an unclassified term: {rendered}")]
    UnclassifiedSecondOrderTensorTerm { label: String, rendered: String },

    #[error("second-order vector mode extraction failed for operation `{operation}`")]
    SecondOrderVectorExtractionFailure { operation: String },

    #[error("second-order tensor mode extraction failed for operation `{operation}`")]
    SecondOrderTensorExtractionFailure { operation: String },

    #[error("cubic bispectrum shape `{shape}` is unsupported")]
    UnsupportedBispectrumShape { shape: String },

    #[error("cubic interaction channel `{channel}` is unsupported")]
    UnsupportedCubicChannel { channel: String },

    #[error("EFT model `{model}` is unsupported")]
    UnsupportedEftModel { model: String },

    #[error("stability condition `{condition}` could not be extracted")]
    StabilityConditionExtractionFailure { condition: String },

    #[error("hierarchy truncation order must be at least 1, got {got}")]
    InvalidHierarchyOrder { got: usize },

    #[error("hierarchy closure `{closure}` is unsupported")]
    UnsupportedHierarchyClosure { closure: String },

    #[error("parity corpus fixture `{fixture}` could not be validated")]
    ParityFixtureValidationFailure { fixture: String },

    #[error("external solver hook target `{target}` is unsupported")]
    UnsupportedExternalSolverHook { target: String },
}
