use crate::error::CosmologyError;

/// Selects the time coordinate used to parameterize the FRW background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeCoordinate {
    /// Use conformal time.
    Conformal,
    /// Use cosmic proper time.
    Cosmic,
}

/// Encodes the sign of the FRW spatial curvature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpatialCurvature {
    /// Spatially flat slices.
    Flat,
    /// Positively curved closed spatial slices.
    Closed,
    /// Negatively curved open spatial slices.
    Open,
}

impl SpatialCurvature {
    /// Returns the conventional curvature sign `k` as `0`, `1`, or `-1`.
    pub fn k_sign(self) -> i8 {
        match self {
            Self::Flat => 0,
            Self::Closed => 1,
            Self::Open => -1,
        }
    }
}

/// Identifies the scalar, vector, or tensor sector of a perturbation object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SectorKind {
    /// Scalar sector.
    Scalar,
    /// Vector sector.
    Vector,
    /// Tensor sector.
    Tensor,
}

/// Enumerates named gauge choices used in cosmological perturbation theory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GaugeKind {
    /// Newtonian gauge.
    Newtonian,
    /// Synchronous gauge.
    Synchronous,
    /// Comoving gauge.
    Comoving,
    /// Spatially flat gauge.
    Flat,
    /// Uniform-density gauge.
    UniformDensity,
    /// Uniform-curvature gauge.
    UniformCurvature,
    /// Poisson gauge.
    Poisson,
}

/// Describes the matter content model used by a perturbative system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatterKind {
    /// A perfect fluid source.
    PerfectFluid,
    /// A fluid with imperfect stresses.
    ImperfectFluid,
    /// A single canonical scalar field.
    CanonicalScalar,
    /// Multiple canonical scalar fields.
    MultiCanonicalScalar {
        /// Number of scalar fields.
        fields: usize,
    },
    /// Symbolic matter content left unspecified.
    Symbolic,
}

/// Selects the harmonic basis or representation space for perturbation modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HarmonicBasisKind {
    /// Work directly in position space.
    PositionSpace,
    /// Use flat-space Fourier modes.
    FourierFlat,
    /// Use scalar harmonics.
    ScalarHarmonics,
    /// Use vector harmonics.
    VectorHarmonics,
    /// Use tensor harmonics.
    TensorHarmonics,
}

/// Canonical names for the standard scalar/vector/tensor metric mode fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvtModeNames {
    /// Scalar lapse perturbation.
    pub phi: lasso::Spur,
    /// Scalar curvature perturbation.
    pub psi: lasso::Spur,
    /// Scalar shift perturbation.
    pub b: lasso::Spur,
    /// Scalar spatial-shear perturbation.
    pub e: lasso::Spur,
    /// Transverse vector shift perturbation.
    pub s: lasso::Spur,
    /// Transverse vector spatial perturbation.
    pub f: lasso::Spur,
    /// Transverse-traceless tensor perturbation.
    pub h_tt: lasso::Spur,
}

/// A typed description of an FRW background and its preferred coordinate symbols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrwBackgroundSpec {
    /// Symbol for the scale factor.
    pub scale_factor: lasso::Spur,
    /// Symbol for the conformal Hubble parameter.
    pub conformal_hubble: lasso::Spur,
    /// Symbol for the cosmic-time Hubble parameter.
    pub cosmic_hubble: lasso::Spur,
    /// Symbol for conformal time.
    pub conformal_time: lasso::Spur,
    /// Symbol for cosmic time.
    pub cosmic_time: lasso::Spur,
    /// Number of spatial dimensions.
    pub spatial_dim: usize,
    /// Sign of the spatial curvature.
    pub spatial_curvature: SpatialCurvature,
    /// Active time coordinate choice.
    pub time_coordinate: TimeCoordinate,
}

impl FrwBackgroundSpec {
    /// Constructs a validated FRW background specification.
    pub fn new(
        scale_factor: lasso::Spur,
        conformal_hubble: lasso::Spur,
        cosmic_hubble: lasso::Spur,
        conformal_time: lasso::Spur,
        cosmic_time: lasso::Spur,
        spatial_dim: usize,
        spatial_curvature: SpatialCurvature,
        time_coordinate: TimeCoordinate,
    ) -> Result<Self, CosmologyError> {
        if spatial_dim == 0 {
            return Err(CosmologyError::InvalidSpatialDimension { got: spatial_dim });
        }
        if spatial_dim > 16 {
            return Err(CosmologyError::UnsupportedSpatialDimension { got: spatial_dim });
        }

        Ok(Self {
            scale_factor,
            conformal_hubble,
            cosmic_hubble,
            conformal_time,
            cosmic_time,
            spatial_dim,
            spatial_curvature,
            time_coordinate,
        })
    }

    /// Builds the conventional flat FRW background in conformal time.
    pub fn default_flat_conformal(interner: &ax_ir::Interner) -> Self {
        Self {
            scale_factor: interner.get_or_intern("a"),
            conformal_hubble: interner.get_or_intern("H"),
            cosmic_hubble: interner.get_or_intern("H_cosmic"),
            conformal_time: interner.get_or_intern("eta"),
            cosmic_time: interner.get_or_intern("t"),
            spatial_dim: 3,
            spatial_curvature: SpatialCurvature::Flat,
            time_coordinate: TimeCoordinate::Conformal,
        }
    }

    /// Builds the conventional flat FRW background in cosmic time.
    pub fn default_flat_cosmic(interner: &ax_ir::Interner) -> Self {
        Self {
            time_coordinate: TimeCoordinate::Cosmic,
            ..Self::default_flat_conformal(interner)
        }
    }

    /// Returns the active time symbol for the selected coordinate convention.
    pub fn active_time_symbol(&self) -> lasso::Spur {
        match self.time_coordinate {
            TimeCoordinate::Conformal => self.conformal_time,
            TimeCoordinate::Cosmic => self.cosmic_time,
        }
    }

    /// Returns the inactive time symbol for the selected coordinate convention.
    pub fn inactive_time_symbol(&self) -> lasso::Spur {
        match self.time_coordinate {
            TimeCoordinate::Conformal => self.cosmic_time,
            TimeCoordinate::Cosmic => self.conformal_time,
        }
    }
}

/// Symbol names for infinitesimal gauge generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaugeGeneratorNames {
    /// Time-shift generator.
    pub time_shift: lasso::Spur,
    /// Scalar spatial-shift generator.
    pub scalar_shift: lasso::Spur,
    /// Divergence-free vector-shift generator.
    pub vector_shift: lasso::Spur,
}

/// Couples a symbol name to a symbolic expression payload.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedExpr {
    /// Symbolic name of the expression.
    pub name: lasso::Spur,
    /// Stored expression.
    pub expr: ax_ir::Expr,
}

/// Couples a named equation to perturbative order and sector metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedEquation {
    /// Human-readable label for the equation.
    pub label: String,
    /// Symbolic equation expression.
    pub expr: ax_ir::Expr,
    /// Perturbative order associated with the equation.
    pub order: usize,
    /// Sector associated with the equation.
    pub sector: SectorKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_flat_conformal_uses_expected_symbols() {
        let interner = ax_ir::Interner::new();
        let bg = FrwBackgroundSpec::default_flat_conformal(&interner);

        assert_eq!(interner.resolve(bg.scale_factor), "a");
        assert_eq!(interner.resolve(bg.conformal_hubble), "H");
        assert_eq!(interner.resolve(bg.cosmic_hubble), "H_cosmic");
        assert_eq!(interner.resolve(bg.conformal_time), "eta");
        assert_eq!(interner.resolve(bg.cosmic_time), "t");
        assert_eq!(bg.spatial_dim, 3);
        assert_eq!(bg.spatial_curvature, SpatialCurvature::Flat);
        assert_eq!(bg.time_coordinate, TimeCoordinate::Conformal);
    }

    #[test]
    fn frw_constructor_rejects_zero_spatial_dimension() {
        let interner = ax_ir::Interner::new();
        let result = FrwBackgroundSpec::new(
            interner.get_or_intern("a"),
            interner.get_or_intern("H"),
            interner.get_or_intern("H_cosmic"),
            interner.get_or_intern("eta"),
            interner.get_or_intern("t"),
            0,
            SpatialCurvature::Flat,
            TimeCoordinate::Conformal,
        );

        assert_eq!(
            result,
            Err(CosmologyError::InvalidSpatialDimension { got: 0 })
        );
    }

    #[test]
    fn frw_constructor_rejects_spatial_dimension_above_guardrail() {
        let interner = ax_ir::Interner::new();
        let result = FrwBackgroundSpec::new(
            interner.get_or_intern("a"),
            interner.get_or_intern("H"),
            interner.get_or_intern("H_cosmic"),
            interner.get_or_intern("eta"),
            interner.get_or_intern("t"),
            17,
            SpatialCurvature::Flat,
            TimeCoordinate::Conformal,
        );

        assert_eq!(
            result,
            Err(CosmologyError::UnsupportedSpatialDimension { got: 17 })
        );
    }

    #[test]
    fn spatial_curvature_k_sign_matches_expected_values() {
        assert_eq!(SpatialCurvature::Flat.k_sign(), 0);
        assert_eq!(SpatialCurvature::Closed.k_sign(), 1);
        assert_eq!(SpatialCurvature::Open.k_sign(), -1);
    }
}
