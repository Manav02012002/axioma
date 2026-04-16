use ax_ir::{DualityKind, TensorSymmetry};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TensorSymmetrySummary {
    pub tableaux: Vec<TensorSymmetryEntry>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TensorSymmetryEntry {
    pub shape: Vec<usize>,
    pub slots: Vec<usize>,
    pub label: Option<String>,
    pub trace_free: bool,
    pub duality: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SymmetryExplainResponse {
    pub summary: TensorSymmetrySummary,
    pub rendered_ascii: String,
}

impl From<&TensorSymmetry> for TensorSymmetrySummary {
    fn from(symmetry: &TensorSymmetry) -> Self {
        Self {
            tableaux: symmetry
                .tableaux
                .iter()
                .map(|tableau| TensorSymmetryEntry {
                    shape: tableau.shape.clone(),
                    slots: tableau.slot_map.clone(),
                    label: tableau.label.clone(),
                    trace_free: tableau.trace_free,
                    duality: duality_name(&tableau.duality).to_string(),
                })
                .collect(),
        }
    }
}

fn duality_name(duality: &DualityKind) -> &'static str {
    match duality {
        DualityKind::None => "none",
        DualityKind::SelfDual => "self_dual",
        DualityKind::AntiSelfDual => "anti_self_dual",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetry_explain_response_round_trips() {
        let response = SymmetryExplainResponse {
            summary: TensorSymmetrySummary {
                tableaux: vec![TensorSymmetryEntry {
                    shape: vec![2, 1],
                    slots: vec![0, 1, 2],
                    label: Some("main".to_string()),
                    trace_free: false,
                    duality: "none".to_string(),
                }],
            },
            rendered_ascii: "[0][1]\n[2]".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let round_trip: SymmetryExplainResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, response);
    }

    #[test]
    fn tensor_symmetry_summary_json_round_trips_exactly() {
        let summary = TensorSymmetrySummary {
            tableaux: vec![
                TensorSymmetryEntry {
                    shape: vec![2, 1],
                    slots: vec![0, 1, 2],
                    label: Some("main".to_string()),
                    trace_free: false,
                    duality: "none".to_string(),
                },
                TensorSymmetryEntry {
                    shape: vec![1, 1],
                    slots: vec![1, 2],
                    label: Some("alt".to_string()),
                    trace_free: true,
                    duality: "none".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&summary).unwrap();
        let round_trip: TensorSymmetrySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, summary);
    }
}
