//! Axioma WASM plugin contract (JSON in/out).
#![forbid(unsafe_code)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginRequest {
    pub plugin: String,
    pub op: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginResponse {
    pub ok: bool,
    pub result: serde_json::Value,
    pub diagnostics: Vec<PluginDiag>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PluginDiag {
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SymmetrySummaryRequest {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SymmetrySummaryResponse {
    pub summary_json: String,
    pub rendered_ascii: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenRequest {
    pub matrix: Vec<Vec<(f64, f64)>>,
    pub k: usize,
    pub which: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenpair {
    pub eigenvalue: (f64, f64),
    pub eigenvector: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseEigenResponse {
    pub eigenpairs: Vec<SparseEigenpair>,
    pub converged: bool,
}

/// Numeric matrix exponential request.
///
/// When `vector` is absent, plugins may return the full `exp(time * matrix)`.
/// When `vector` is present, plugins may return the action `exp(time * matrix) v`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MatrixExponentialRequest {
    pub matrix: Vec<Vec<(f64, f64)>>,
    pub vector: Option<Vec<(f64, f64)>>,
    pub time: f64,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MatrixExponentialResponse {
    pub matrix_exponential: Option<Vec<Vec<(f64, f64)>>>,
    pub action_on_vector: Option<Vec<(f64, f64)>>,
    pub converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseLindbladianSpectrumRequest {
    pub superoperator: Vec<Vec<(f64, f64)>>,
    pub k: usize,
    pub which: String,
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseLindbladianSpectrumResponse {
    pub eigenvalues: Vec<(f64, f64)>,
    pub converged: bool,
    pub residual_norms: Vec<f64>,
}

/// Sparse Lindbladian/Liouvillian steady-state solve request.
///
/// `superoperator` is represented in the vectorized density basis and must be
/// sized `dim^2 x dim^2`. When `trace_constraint` is true, plugins must enforce
/// `Tr(rho) = 1` in the returned steady state.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseSteadyStateRequest {
    pub superoperator: Vec<Vec<(f64, f64)>>,
    pub dim: usize,
    pub trace_constraint: bool,
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SparseSteadyStateResponse {
    pub density_matrix: Vec<Vec<(f64, f64)>>,
    pub converged: bool,
    pub residual_norm: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CompletePositivityRequest {
    pub choi: Vec<Vec<(f64, f64)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CompletePositivityResponse {
    pub completely_positive: bool,
}

/// Local Hamiltonian term acting on the listed 1D lattice sites.
///
/// `operator` is a dense complex matrix encoded as `(re, im)` pairs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct LocalTerm1D {
    pub sites: Vec<usize>,
    pub operator: Vec<Vec<(f64, f64)>>,
}

/// Ground-state request for a 1D local Hamiltonian tensor-network workflow.
///
/// `terms` are local Hamiltonian terms on specified site subsets.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct TensorNetworkGroundStateRequest {
    pub chain_length: usize,
    pub local_dim: usize,
    pub terms: Vec<LocalTerm1D>,
    pub bond_dimension: usize,
    pub sweeps: usize,
}

/// Ground-state response for a 1D local Hamiltonian tensor-network workflow.
///
/// `expectation_values` correspond to requested observables only if the plugin
/// implementation chooses to supply them later; for now this required field may
/// be empty.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct TensorNetworkGroundStateResponse {
    pub energy: f64,
    pub converged: bool,
    pub expectation_values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_json_schema_round_trips<T: JsonSchema>() {
        let schema_json = serde_json::to_string(&schemars::schema_for!(T)).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(&schema_json).unwrap();
        assert!(decoded.is_object());
    }

    #[test]
    fn symmetry_request_round_trips() {
        let request = SymmetrySummaryRequest {
            expr: "tableau_symmetry([[2]], slots=[[0,1]])".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SymmetrySummaryRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn symmetry_response_round_trips() {
        let response = SymmetrySummaryResponse {
            summary_json: "{\"tableaux\":[]}".to_string(),
            rendered_ascii: "[][]".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SymmetrySummaryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn sparse_eigen_request_round_trips() {
        let request = SparseEigenRequest {
            matrix: vec![vec![(1.0, 0.0), (0.0, 0.0)], vec![(0.0, 0.0), (2.0, 0.0)]],
            k: 1,
            which: "LM".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SparseEigenRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn sparse_eigen_response_round_trips() {
        let response = SparseEigenResponse {
            eigenpairs: vec![SparseEigenpair {
                eigenvalue: (2.0, 0.0),
                eigenvector: vec![(1.0, 0.0), (0.0, 0.0)],
            }],
            converged: true,
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SparseEigenResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn matrix_exponential_request_round_trips() {
        let request = MatrixExponentialRequest {
            matrix: vec![vec![(0.0, 0.0), (-1.0, 0.0)], vec![(1.0, 0.0), (0.0, 0.0)]],
            vector: Some(vec![(1.0, 0.0), (0.0, 0.0)]),
            time: std::f64::consts::FRAC_PI_2,
            method: "krylov".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: MatrixExponentialRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        assert_json_schema_round_trips::<MatrixExponentialRequest>();
    }

    #[test]
    fn matrix_exponential_response_round_trips() {
        let response = MatrixExponentialResponse {
            matrix_exponential: Some(vec![
                vec![(0.0, 0.0), (-1.0, 0.0)],
                vec![(1.0, 0.0), (0.0, 0.0)],
            ]),
            action_on_vector: None,
            converged: true,
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: MatrixExponentialResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
        assert_json_schema_round_trips::<MatrixExponentialResponse>();
    }

    #[test]
    fn sparse_lindbladian_spectrum_request_round_trips() {
        let request = SparseLindbladianSpectrumRequest {
            superoperator: vec![
                vec![(-0.2, 0.0), (0.0, 0.0), (0.0, 0.0), (0.05, 0.0)],
                vec![(0.0, 0.0), (-0.1, -1.5), (0.02, 0.0), (0.0, 0.0)],
                vec![(0.0, 0.0), (0.02, 0.0), (-0.1, 1.5), (0.0, 0.0)],
                vec![(0.2, 0.0), (0.0, 0.0), (0.0, 0.0), (-0.05, 0.0)],
            ],
            k: 3,
            which: "LR".to_string(),
            tolerance: 1e-9,
            max_iterations: 4_096,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SparseLindbladianSpectrumRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        assert_json_schema_round_trips::<SparseLindbladianSpectrumRequest>();
    }

    #[test]
    fn sparse_lindbladian_spectrum_response_round_trips() {
        let response = SparseLindbladianSpectrumResponse {
            eigenvalues: vec![(0.0, 0.0), (-0.1, 1.5), (-0.1, -1.5)],
            converged: true,
            residual_norms: vec![2.0e-13, 7.5e-11, 8.0e-11],
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SparseLindbladianSpectrumResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
        assert_json_schema_round_trips::<SparseLindbladianSpectrumResponse>();
    }

    #[test]
    fn sparse_steady_state_request_round_trips() {
        let request = SparseSteadyStateRequest {
            superoperator: vec![
                vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.15, 0.0)],
                vec![(0.0, 0.0), (-0.075, -1.25), (0.0, 0.0), (0.0, 0.0)],
                vec![(0.0, 0.0), (0.0, 0.0), (-0.075, 1.25), (0.0, 0.0)],
                vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-0.15, 0.0)],
            ],
            dim: 2,
            trace_constraint: true,
            tolerance: 1e-10,
            max_iterations: 2_000,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: SparseSteadyStateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        assert_json_schema_round_trips::<SparseSteadyStateRequest>();
    }

    #[test]
    fn sparse_steady_state_response_round_trips() {
        let response = SparseSteadyStateResponse {
            density_matrix: vec![
                vec![(0.82, 0.0), (0.03, -0.04)],
                vec![(0.03, 0.04), (0.18, 0.0)],
            ],
            converged: true,
            residual_norm: 4.2e-13,
            iterations: 137,
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: SparseSteadyStateResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
        assert_json_schema_round_trips::<SparseSteadyStateResponse>();
    }

    #[test]
    fn complete_positivity_request_round_trips() {
        let request = CompletePositivityRequest {
            choi: vec![vec![(1.0, 0.0), (0.0, 0.0)], vec![(0.0, 0.0), (1.0, 0.0)]],
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: CompletePositivityRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn complete_positivity_response_round_trips() {
        let response = CompletePositivityResponse {
            completely_positive: true,
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: CompletePositivityResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn tensor_network_ground_state_request_round_trips() {
        let request = TensorNetworkGroundStateRequest {
            chain_length: 4,
            local_dim: 2,
            terms: vec![
                LocalTerm1D {
                    sites: vec![0],
                    operator: vec![vec![(1.0, 0.0), (0.0, 0.0)], vec![(0.0, 0.0), (-1.0, 0.0)]],
                },
                LocalTerm1D {
                    sites: vec![0, 1],
                    operator: vec![
                        vec![(0.25, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
                        vec![(0.0, 0.0), (-0.25, 0.0), (0.5, 0.0), (0.0, 0.0)],
                        vec![(0.0, 0.0), (0.5, 0.0), (-0.25, 0.0), (0.0, 0.0)],
                        vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.25, 0.0)],
                    ],
                },
            ],
            bond_dimension: 32,
            sweeps: 8,
        };
        let json = serde_json::to_string(&request).unwrap();
        let decoded: TensorNetworkGroundStateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        assert_json_schema_round_trips::<LocalTerm1D>();
        assert_json_schema_round_trips::<TensorNetworkGroundStateRequest>();
    }

    #[test]
    fn tensor_network_ground_state_response_round_trips() {
        let response = TensorNetworkGroundStateResponse {
            energy: -1.25,
            converged: true,
            expectation_values: Vec::new(),
        };
        let json = serde_json::to_string(&response).unwrap();
        let decoded: TensorNetworkGroundStateResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, response);
        assert_json_schema_round_trips::<TensorNetworkGroundStateResponse>();
    }
}
