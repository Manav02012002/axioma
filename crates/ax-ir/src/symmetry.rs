use thiserror::Error;

/// Duality information attached to a tensor tableau.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DualityKind {
    /// No duality constraint is attached.
    None,
    /// The tableau is constrained to be self-dual.
    SelfDual,
    /// The tableau is constrained to be anti-self-dual.
    AntiSelfDual,
}

/// Provenance for how a tensor symmetry attachment entered the system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymmetrySource {
    /// The symmetry was declared directly by source input.
    Declared,
    /// The symmetry was inherited from another tensor/composite.
    Inherited,
    /// The symmetry was derived from another property.
    Derived,
    /// The symmetry was produced by canonicalization.
    Canonicalized,
    /// The symmetry was produced by a projector/projection step.
    Projected,
}

/// Restriction mode for how much of the Young symmetry is active.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestrictedSymmetryMode {
    /// Full Young symmetrization/antisymmetrization is active.
    FullYoung,
    /// Only column-exchange relations are active.
    ColumnExchangeOnly,
    /// Only row-exchange relations are active.
    RowExchangeOnly,
}

/// Dimension constraints on when a tableau attachment is valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionGuard {
    /// Minimum allowed dimension, if any.
    pub min_dimension: Option<usize>,
    /// Maximum allowed dimension, if any.
    pub max_dimension: Option<usize>,
    /// Exact required dimension, if any.
    pub exact_dimension: Option<usize>,
}

impl DimensionGuard {
    /// Return whether this guard allows the provided dimension.
    pub fn allows(&self, dim: Option<usize>) -> bool {
        match dim {
            None => self.exact_dimension.is_none(),
            Some(dimension) => {
                if let Some(exact) = self.exact_dimension {
                    if dimension != exact {
                        return false;
                    }
                }
                if let Some(min_dimension) = self.min_dimension {
                    if dimension < min_dimension {
                        return false;
                    }
                }
                if let Some(max_dimension) = self.max_dimension {
                    if dimension > max_dimension {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// One tableau attachment contributing to a tensor's symmetry data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableauAttachment {
    /// Young tableau row shape.
    pub shape: Vec<usize>,
    /// Explicit slot mapping from tableau cells to tensor slots.
    pub slot_map: Vec<usize>,
    /// Rational multiplicity numerator.
    pub multiplicity_numer: i64,
    /// Rational multiplicity denominator.
    pub multiplicity_denom: i64,
    /// Duality constraint for this tableau.
    pub duality: DualityKind,
    /// Restriction mode for this tableau.
    pub restricted_mode: RestrictedSymmetryMode,
    /// Whether this tableau enforces trace-freeness.
    pub trace_free: bool,
    /// Optional dimension guard for the tableau.
    pub dimension_guard: Option<DimensionGuard>,
    /// Provenance for diagnostics and downstream handling.
    pub source: SymmetrySource,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// Structured tensor symmetry metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSymmetry {
    /// All tableaux attached to the tensor/property.
    pub tableaux: Vec<TableauAttachment>,
    /// Whether the symmetry should inherit under derivatives.
    pub inherits_under_derivative: bool,
    /// Whether the symmetry should inherit under tensor products.
    pub inherits_under_tensor_product: bool,
    /// Whether the symmetry should inherit under contractions.
    pub inherits_under_contraction: bool,
    /// Whether projection preserves trace-freeness.
    pub preserves_trace_free_under_projection: bool,
}

/// Validation failures for structured tensor symmetries.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum SymmetryValidationError {
    #[error("tensor symmetry tableau shape cannot be empty")]
    EmptyShape,
    #[error("tensor symmetry tableau shape must be weakly decreasing: {shape:?}")]
    NonDecreasingShape { shape: Vec<usize> },
    #[error("tensor symmetry tableau shape cannot contain zero row lengths: {shape:?}")]
    ZeroRowLength { shape: Vec<usize> },
    #[error("tensor symmetry slot map cannot be empty")]
    EmptySlotMap,
    #[error("tensor symmetry shape cell count {shape_cells} does not match slot map length {slot_count}")]
    ShapeSlotCountMismatch {
        shape_cells: usize,
        slot_count: usize,
    },
    #[error("tensor symmetry slot map contains duplicate slot {slot}")]
    DuplicateSlot { slot: usize },
    #[error(
        "tensor symmetry multiplicity must have nonzero positive denominator, got {numer}/{denom}"
    )]
    InvalidRationalMultiplicity { numer: i64, denom: i64 },
    #[error("tensor symmetry dimension guard is contradictory")]
    InvalidDimensionGuard,
    #[error("self-dual or anti-self-dual tableau requires even column length, column {column} has length {length}")]
    SelfDualWithoutEvenColumnLength { column: usize, length: usize },
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DualityValidationError {
    #[error("duality validation requires an exact ambient dimension")]
    DimensionRequired,
    #[error("self-dual and anti-self-dual tensors require even ambient dimension, got {dim}")]
    OddDimensionSelfDual { dim: usize },
    #[error("self-dual and anti-self-dual tensors require middle degree rank dim/2, got rank {rank} in dimension {dim}")]
    RankMismatchForMiddleDegree { rank: usize, dim: usize },
}

fn validate_dimension_guard(guard: &DimensionGuard) -> Result<(), SymmetryValidationError> {
    if let (Some(min_dimension), Some(max_dimension)) = (guard.min_dimension, guard.max_dimension) {
        if min_dimension > max_dimension {
            return Err(SymmetryValidationError::InvalidDimensionGuard);
        }
    }
    if let Some(exact_dimension) = guard.exact_dimension {
        if let Some(min_dimension) = guard.min_dimension {
            if exact_dimension < min_dimension {
                return Err(SymmetryValidationError::InvalidDimensionGuard);
            }
        }
        if let Some(max_dimension) = guard.max_dimension {
            if exact_dimension > max_dimension {
                return Err(SymmetryValidationError::InvalidDimensionGuard);
            }
        }
    }
    Ok(())
}

fn column_lengths(shape: &[usize]) -> Vec<usize> {
    let max_columns = shape.iter().copied().max().unwrap_or(0);
    (0..max_columns)
        .map(|column| shape.iter().filter(|row_len| **row_len > column).count())
        .collect()
}

/// Validate one tableau attachment.
pub fn validate_tableau_attachment(
    attachment: &TableauAttachment,
) -> Result<(), SymmetryValidationError> {
    if attachment.shape.is_empty() {
        return Err(SymmetryValidationError::EmptyShape);
    }
    if attachment.shape.iter().any(|row_len| *row_len == 0) {
        return Err(SymmetryValidationError::ZeroRowLength {
            shape: attachment.shape.clone(),
        });
    }
    if attachment.shape.windows(2).any(|rows| rows[0] < rows[1]) {
        return Err(SymmetryValidationError::NonDecreasingShape {
            shape: attachment.shape.clone(),
        });
    }
    if attachment.slot_map.is_empty() {
        return Err(SymmetryValidationError::EmptySlotMap);
    }
    let shape_cells = attachment.shape.iter().sum::<usize>();
    if shape_cells != attachment.slot_map.len() {
        return Err(SymmetryValidationError::ShapeSlotCountMismatch {
            shape_cells,
            slot_count: attachment.slot_map.len(),
        });
    }

    let mut seen_slots = std::collections::HashSet::new();
    for slot in &attachment.slot_map {
        if !seen_slots.insert(*slot) {
            return Err(SymmetryValidationError::DuplicateSlot { slot: *slot });
        }
    }

    if attachment.multiplicity_denom <= 0 {
        return Err(SymmetryValidationError::InvalidRationalMultiplicity {
            numer: attachment.multiplicity_numer,
            denom: attachment.multiplicity_denom,
        });
    }

    if let Some(guard) = &attachment.dimension_guard {
        validate_dimension_guard(guard)?;
    }

    if attachment.duality != DualityKind::None {
        for (column, length) in column_lengths(&attachment.shape).into_iter().enumerate() {
            if length % 2 != 0 {
                return Err(SymmetryValidationError::SelfDualWithoutEvenColumnLength {
                    column,
                    length,
                });
            }
        }
    }

    Ok(())
}

/// Validate a full tensor symmetry object.
pub fn validate_tensor_symmetry(sym: &TensorSymmetry) -> Result<(), SymmetryValidationError> {
    for attachment in &sym.tableaux {
        validate_tableau_attachment(attachment)?;
    }
    Ok(())
}

pub fn validate_duality_in_dimension(
    attachment: &TableauAttachment,
    dim: Option<usize>,
) -> Result<(), DualityValidationError> {
    if attachment.duality == DualityKind::None {
        return Ok(());
    }
    let dim = dim.ok_or(DualityValidationError::DimensionRequired)?;
    if dim % 2 != 0 {
        return Err(DualityValidationError::OddDimensionSelfDual { dim });
    }
    let rank = attachment.slot_map.len();
    if rank != dim / 2 {
        return Err(DualityValidationError::RankMismatchForMiddleDegree { rank, dim });
    }
    Ok(())
}

impl TensorSymmetry {
    /// Return the largest slot count across all attached tableaux.
    pub fn total_slots(&self) -> usize {
        self.tableaux
            .iter()
            .map(|attachment| attachment.slot_map.len())
            .max()
            .unwrap_or(0)
    }

    /// Return whether this symmetry has no attached tableaux.
    pub fn is_empty(&self) -> bool {
        self.tableaux.is_empty()
    }

    /// Validate this tensor symmetry.
    pub fn validate(&self) -> Result<(), SymmetryValidationError> {
        validate_tensor_symmetry(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_attachment() -> TableauAttachment {
        TableauAttachment {
            shape: vec![2, 1],
            slot_map: vec![0, 1, 2],
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: DualityKind::None,
            restricted_mode: RestrictedSymmetryMode::FullYoung,
            trace_free: false,
            dimension_guard: None,
            source: SymmetrySource::Declared,
            label: None,
        }
    }

    #[test]
    fn valid_single_tableau() {
        assert_eq!(validate_tableau_attachment(&base_attachment()), Ok(()));
    }

    #[test]
    fn shape_empty() {
        let mut attachment = base_attachment();
        attachment.shape.clear();
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::EmptyShape)
        );
    }

    #[test]
    fn zero_row() {
        let mut attachment = base_attachment();
        attachment.shape = vec![2, 0];
        attachment.slot_map = vec![0, 1];
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::ZeroRowLength { shape: vec![2, 0] })
        );
    }

    #[test]
    fn nondecreasing_violation() {
        let mut attachment = base_attachment();
        attachment.shape = vec![1, 2];
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::NonDecreasingShape { shape: vec![1, 2] })
        );
    }

    #[test]
    fn slot_count_mismatch() {
        let mut attachment = base_attachment();
        attachment.slot_map = vec![0, 1];
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::ShapeSlotCountMismatch {
                shape_cells: 3,
                slot_count: 2,
            })
        );
    }

    #[test]
    fn duplicate_slot() {
        let mut attachment = base_attachment();
        attachment.shape = vec![2];
        attachment.slot_map = vec![0, 0];
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::DuplicateSlot { slot: 0 })
        );
    }

    #[test]
    fn invalid_denominator_zero() {
        let mut attachment = base_attachment();
        attachment.multiplicity_denom = 0;
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::InvalidRationalMultiplicity { numer: 1, denom: 0 })
        );
    }

    #[test]
    fn contradictory_dimension_guard() {
        let mut attachment = base_attachment();
        attachment.dimension_guard = Some(DimensionGuard {
            min_dimension: Some(5),
            max_dimension: Some(3),
            exact_dimension: None,
        });
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::InvalidDimensionGuard)
        );
    }

    #[test]
    fn exact_dimension_outside_range() {
        let mut attachment = base_attachment();
        attachment.dimension_guard = Some(DimensionGuard {
            min_dimension: Some(5),
            max_dimension: Some(10),
            exact_dimension: Some(4),
        });
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::InvalidDimensionGuard)
        );

        attachment.dimension_guard = Some(DimensionGuard {
            min_dimension: Some(5),
            max_dimension: Some(10),
            exact_dimension: Some(11),
        });
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::InvalidDimensionGuard)
        );
    }

    #[test]
    fn selfdual_with_odd_column() {
        let mut attachment = base_attachment();
        attachment.shape = vec![1];
        attachment.slot_map = vec![0];
        attachment.duality = DualityKind::SelfDual;
        assert_eq!(
            validate_tableau_attachment(&attachment),
            Err(SymmetryValidationError::SelfDualWithoutEvenColumnLength {
                column: 0,
                length: 1,
            })
        );
    }

    #[test]
    fn multi_tableau_total_slots() {
        let sym = TensorSymmetry {
            tableaux: vec![
                base_attachment(),
                TableauAttachment {
                    shape: vec![1, 1],
                    slot_map: vec![0, 1],
                    multiplicity_numer: 1,
                    multiplicity_denom: 1,
                    duality: DualityKind::None,
                    restricted_mode: RestrictedSymmetryMode::FullYoung,
                    trace_free: false,
                    dimension_guard: None,
                    source: SymmetrySource::Declared,
                    label: None,
                },
            ],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };

        assert_eq!(sym.total_slots(), 3);
    }

    #[test]
    fn validate_duality_in_dimension_matches_exact_rules() {
        let mut attachment = base_attachment();
        attachment.shape = vec![1, 1];
        attachment.slot_map = vec![0, 1];
        attachment.duality = DualityKind::SelfDual;

        assert_eq!(validate_duality_in_dimension(&attachment, Some(4)), Ok(()));
        assert_eq!(
            validate_duality_in_dimension(&attachment, Some(3)),
            Err(DualityValidationError::OddDimensionSelfDual { dim: 3 })
        );

        attachment.shape = vec![1];
        attachment.slot_map = vec![0];
        assert_eq!(
            validate_duality_in_dimension(&attachment, Some(4)),
            Err(DualityValidationError::RankMismatchForMiddleDegree { rank: 1, dim: 4 })
        );
        assert_eq!(
            validate_duality_in_dimension(&attachment, None),
            Err(DualityValidationError::DimensionRequired)
        );
    }
}
