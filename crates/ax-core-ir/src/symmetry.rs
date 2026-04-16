use thiserror::Error;

/// Lowered structured tableau attachment with core-sized slot indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreTableauAttachment {
    /// Young tableau row shape.
    pub shape: Vec<usize>,
    /// Explicit slot mapping.
    pub slot_map: Vec<u32>,
    /// Rational multiplicity numerator.
    pub multiplicity_numer: i64,
    /// Rational multiplicity denominator.
    pub multiplicity_denom: i64,
    /// Duality metadata.
    pub duality: ax_ir::DualityKind,
    /// Restricted symmetry mode.
    pub restricted_mode: ax_ir::RestrictedSymmetryMode,
    /// Trace-freeness metadata.
    pub trace_free: bool,
    /// Optional dimension guard.
    pub dimension_guard: Option<ax_ir::DimensionGuard>,
    /// Source provenance.
    pub source: ax_ir::SymmetrySource,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// Lowered structured tensor symmetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreTensorSymmetry {
    /// All attached tableaux.
    pub tableaux: Vec<CoreTableauAttachment>,
    /// Whether the symmetry inherits under derivatives.
    pub inherits_under_derivative: bool,
    /// Whether the symmetry inherits under tensor products.
    pub inherits_under_tensor_product: bool,
    /// Whether the symmetry inherits under contractions.
    pub inherits_under_contraction: bool,
    /// Whether projection preserves trace-freeness.
    pub preserves_trace_free_under_projection: bool,
}

/// Errors while lowering structured tensor symmetries into core form.
#[derive(Error, Debug)]
pub enum CoreSymmetryLowerError {
    #[error("core symmetry lowering encountered slot index {slot} which does not fit into u32")]
    SlotOutOfRange { slot: usize },
    #[error("core symmetry lowering validation failed: {0}")]
    Validation(ax_ir::SymmetryValidationError),
}

/// Lower a validated IR tensor symmetry into core symmetry form.
pub fn lower_tensor_symmetry(
    sym: &ax_ir::TensorSymmetry,
) -> Result<CoreTensorSymmetry, CoreSymmetryLowerError> {
    sym.validate().map_err(CoreSymmetryLowerError::Validation)?;

    let mut tableaux = Vec::with_capacity(sym.tableaux.len());
    for attachment in &sym.tableaux {
        let mut slot_map = Vec::with_capacity(attachment.slot_map.len());
        for slot in &attachment.slot_map {
            let lowered = u32::try_from(*slot)
                .map_err(|_| CoreSymmetryLowerError::SlotOutOfRange { slot: *slot })?;
            slot_map.push(lowered);
        }
        tableaux.push(CoreTableauAttachment {
            shape: attachment.shape.clone(),
            slot_map,
            multiplicity_numer: attachment.multiplicity_numer,
            multiplicity_denom: attachment.multiplicity_denom,
            duality: attachment.duality.clone(),
            restricted_mode: attachment.restricted_mode.clone(),
            trace_free: attachment.trace_free,
            dimension_guard: attachment.dimension_guard.clone(),
            source: attachment.source.clone(),
            label: attachment.label.clone(),
        });
    }

    Ok(CoreTensorSymmetry {
        tableaux,
        inherits_under_derivative: sym.inherits_under_derivative,
        inherits_under_tensor_product: sym.inherits_under_tensor_product,
        inherits_under_contraction: sym.inherits_under_contraction,
        preserves_trace_free_under_projection: sym.preserves_trace_free_under_projection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_symmetry() -> ax_ir::TensorSymmetry {
        ax_ir::TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape: vec![2, 1],
                slot_map: vec![0, 1, 2],
                multiplicity_numer: 3,
                multiplicity_denom: 2,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::ColumnExchangeOnly,
                trace_free: true,
                dimension_guard: Some(ax_ir::DimensionGuard {
                    min_dimension: Some(3),
                    max_dimension: Some(8),
                    exact_dimension: None,
                }),
                source: ax_ir::SymmetrySource::Declared,
                label: Some("main".to_string()),
            }],
            inherits_under_derivative: true,
            inherits_under_tensor_product: false,
            inherits_under_contraction: true,
            preserves_trace_free_under_projection: true,
        }
    }

    #[test]
    fn successful_lowering_preserves_fields() {
        let lowered = lower_tensor_symmetry(&sample_symmetry()).unwrap();
        assert_eq!(lowered.tableaux.len(), 1);
        assert_eq!(lowered.tableaux[0].shape, vec![2, 1]);
        assert_eq!(lowered.tableaux[0].slot_map, vec![0, 1, 2]);
        assert_eq!(lowered.tableaux[0].multiplicity_numer, 3);
        assert_eq!(lowered.tableaux[0].multiplicity_denom, 2);
        assert_eq!(
            lowered.tableaux[0].restricted_mode,
            ax_ir::RestrictedSymmetryMode::ColumnExchangeOnly
        );
        assert!(lowered.inherits_under_derivative);
        assert!(lowered.inherits_under_contraction);
        assert!(lowered.preserves_trace_free_under_projection);
    }

    #[test]
    fn slot_larger_than_u32_max_fails() {
        let mut sym = sample_symmetry();
        sym.tableaux[0].slot_map = vec![usize::MAX];
        sym.tableaux[0].shape = vec![1];
        assert!(matches!(
            lower_tensor_symmetry(&sym),
            Err(CoreSymmetryLowerError::SlotOutOfRange { slot }) if slot == usize::MAX
        ));
    }

    #[test]
    fn invalid_source_symmetry_propagates_validation() {
        let mut sym = sample_symmetry();
        sym.tableaux[0].shape.clear();
        assert!(matches!(
            lower_tensor_symmetry(&sym),
            Err(CoreSymmetryLowerError::Validation(
                ax_ir::SymmetryValidationError::EmptyShape
            ))
        ));
    }
}
