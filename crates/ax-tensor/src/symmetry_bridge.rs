use ax_ir::{DualityKind, TensorProperty, TensorSymmetry};
use ax_young::{ProjectorNormalization, YoungDiagram, YoungTableau};
use num_bigint::BigInt;
use num_rational::BigRational;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct RealizedTableau {
    pub shape: Vec<usize>,
    pub slot_map: Vec<usize>,
    // Preserved-but-not-yet-consumed-on-every-path metadata is carried alongside the
    // projector so downstream tensor code can inspect `trace_free` and `duality`.
    pub projector: ax_young::group_action::GroupBackedProjector,
    pub trace_free: bool,
    pub duality: DualityKind,
}

#[derive(Error, Debug)]
pub enum SymmetryBridgeError {
    #[error("tensor symmetry bridge requires at least one tableau attachment")]
    MissingTableaux,
    #[error("tensor symmetry bridge Young conversion failed: {0}")]
    Young(#[from] ax_young::YoungError),
    #[error("tensor symmetry bridge projector construction failed: {0}")]
    Projector(#[from] ax_young::GroupProjectorError),
    #[error("tensor symmetry bridge validation failed: {0}")]
    Validation(#[from] ax_ir::SymmetryValidationError),
}

pub fn realized_tableaux_from_symmetry(
    symmetry: &TensorSymmetry,
) -> Result<Vec<RealizedTableau>, SymmetryBridgeError> {
    symmetry.validate()?;
    if symmetry.tableaux.is_empty() {
        return Err(SymmetryBridgeError::MissingTableaux);
    }

    symmetry
        .tableaux
        .iter()
        .map(|attachment| {
            let diagram = YoungDiagram::try_new(attachment.shape.clone())?;
            let standard = YoungTableau::standard(&diagram)?;
            let tableau = relabel_tableau_from_slot_map(
                &standard,
                &attachment.slot_map,
                attachment.multiplicity_numer,
                attachment.multiplicity_denom,
            )?;
            let projector = ax_young::build_group_backed_projector(
                &tableau,
                ProjectorNormalization::Unnormalized,
            )?;
            Ok(RealizedTableau {
                shape: attachment.shape.clone(),
                slot_map: attachment.slot_map.clone(),
                projector,
                trace_free: attachment.trace_free,
                duality: attachment.duality.clone(),
            })
        })
        .collect()
}

pub fn realized_tableaux_from_properties(
    properties: &[TensorProperty],
) -> Result<Vec<RealizedTableau>, SymmetryBridgeError> {
    let Some(symmetry) = crate::structured_curvature_properties_from_legacy(properties).0 else {
        return Err(SymmetryBridgeError::MissingTableaux);
    };
    realized_tableaux_from_symmetry(&symmetry)
}

fn relabel_tableau_from_slot_map(
    standard: &YoungTableau,
    slot_map: &[usize],
    multiplicity_numer: i64,
    multiplicity_denom: i64,
) -> Result<YoungTableau, SymmetryBridgeError> {
    let dense_labels = dense_projector_labels(slot_map);
    let rows = standard
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .filter_map(|cell| dense_labels.get(*cell).copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Preserve rational multiplicity in the realised projector tableau even though the
    // canonicalisation path currently only consumes the stabilizer groups.
    YoungTableau::with_metadata(
        rows,
        BigRational::new(
            BigInt::from(multiplicity_numer),
            BigInt::from(multiplicity_denom),
        ),
        0,
    )
    .map_err(SymmetryBridgeError::Young)
}

fn dense_projector_labels(slot_map: &[usize]) -> Vec<usize> {
    let mut sorted = slot_map.to_vec();
    sorted.sort_unstable();
    slot_map
        .iter()
        .map(|slot| {
            sorted
                .iter()
                .position(|candidate| candidate == slot)
                .unwrap_or(0)
        })
        .collect()
}
