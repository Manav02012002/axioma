#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NumericalGRError {
    #[error("parallel_transport requires nonempty curve data")]
    EmptyCurve,
    #[error("parallel_transport initial vector dimension {vector_dim} does not match curve dimension {curve_dim}")]
    ParallelTransportDimensionMismatch { vector_dim: usize, curve_dim: usize },
    #[error("integrate_geodesic requires position and velocity dimensions to match")]
    GeodesicInitialDataMismatch,
    #[error("integrate_geodesic requires n_steps > 0")]
    InvalidStepCount,
    #[error("integrate_geodesic tau range must satisfy tau_start <= tau_end")]
    InvalidTauRange,
}

fn zeros_rank3(dim: usize) -> Vec<Vec<Vec<f64>>> {
    vec![vec![vec![0.0; dim]; dim]; dim]
}

pub fn parallel_transport(
    initial_vector: &[f64],
    curve: &[Vec<f64>],
    gamma_numeric: &dyn Fn(&[f64]) -> Vec<Vec<Vec<f64>>>,
) -> Result<Vec<Vec<f64>>, NumericalGRError> {
    let Some(first_point) = curve.first() else {
        return Err(NumericalGRError::EmptyCurve);
    };
    if initial_vector.len() != first_point.len() {
        return Err(NumericalGRError::ParallelTransportDimensionMismatch {
            vector_dim: initial_vector.len(),
            curve_dim: first_point.len(),
        });
    }
    let dim = first_point.len();
    if curve.iter().any(|point| point.len() != dim) {
        return Err(NumericalGRError::ParallelTransportDimensionMismatch {
            vector_dim: initial_vector.len(),
            curve_dim: dim,
        });
    }

    let mut transported = Vec::with_capacity(curve.len());
    let mut current = initial_vector.to_vec();
    transported.push(current.clone());

    for segment in curve.windows(2) {
        let start = &segment[0];
        let end = &segment[1];
        let tangent = end
            .iter()
            .zip(start.iter())
            .map(|(x1, x0)| x1 - x0)
            .collect::<Vec<_>>();

        let system = |lambda: f64, vector: &[f64]| {
            let position = start
                .iter()
                .zip(tangent.iter())
                .map(|(x0, dx)| x0 + lambda * dx)
                .collect::<Vec<_>>();
            let gamma = gamma_numeric(&position);
            let gamma = if gamma.len() == dim {
                gamma
            } else {
                zeros_rank3(dim)
            };
            (0..dim)
                .map(|mu| {
                    let mut value = 0.0;
                    for nu in 0..dim {
                        for rho in 0..dim {
                            value -= gamma[mu][nu][rho] * vector[nu] * tangent[rho];
                        }
                    }
                    value
                })
                .collect::<Vec<_>>()
        };

        let result = ax_ode::rk4_system_numeric(&system, 0.0, &current, 1.0, 1);
        let next = result
            .last()
            .map(|row| row[1..].to_vec())
            .unwrap_or_else(|| current.clone());
        current = next;
        transported.push(current.clone());
    }

    Ok(transported)
}

pub fn integrate_geodesic(
    gamma_numeric: &dyn Fn(&[f64]) -> Vec<Vec<Vec<f64>>>,
    initial_position: &[f64],
    initial_velocity: &[f64],
    tau_range: (f64, f64),
    n_steps: usize,
) -> Result<Vec<(f64, Vec<f64>, Vec<f64>)>, NumericalGRError> {
    if initial_position.len() != initial_velocity.len() {
        return Err(NumericalGRError::GeodesicInitialDataMismatch);
    }
    if n_steps == 0 {
        return Err(NumericalGRError::InvalidStepCount);
    }
    if tau_range.0 > tau_range.1 {
        return Err(NumericalGRError::InvalidTauRange);
    }

    let dim = initial_position.len();
    let mut initial_state = Vec::with_capacity(2 * dim);
    initial_state.extend_from_slice(initial_position);
    initial_state.extend_from_slice(initial_velocity);

    let system = |_tau: f64, state: &[f64]| {
        let position = &state[..dim];
        let velocity = &state[dim..];
        let gamma = gamma_numeric(position);
        let gamma = if gamma.len() == dim {
            gamma
        } else {
            zeros_rank3(dim)
        };
        let mut derivative = vec![0.0; 2 * dim];
        derivative[..dim].copy_from_slice(velocity);
        for mu in 0..dim {
            let mut accel = 0.0;
            for nu in 0..dim {
                for rho in 0..dim {
                    accel -= gamma[mu][nu][rho] * velocity[nu] * velocity[rho];
                }
            }
            derivative[dim + mu] = accel;
        }
        derivative
    };

    let raw =
        ax_ode::rk4_system_numeric(&system, tau_range.0, &initial_state, tau_range.1, n_steps);
    Ok(raw
        .into_iter()
        .map(|row| {
            let tau = row[0];
            let state = &row[1..];
            let position = state[..dim].to_vec();
            let velocity = state[dim..].to_vec();
            (tau, position, velocity)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{integrate_geodesic, parallel_transport, NumericalGRError};

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "expected {lhs} ~= {rhs} within {tol}"
        );
    }

    #[test]
    fn minkowski_geodesic_is_uniform_motion() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 2]; 2]; 2];
        let result =
            integrate_geodesic(&gamma, &[0.0, 0.0], &[1.0, 2.0], (0.0, 1.0), 10).expect("geodesic");
        let (_, final_position, final_velocity) = result.last().expect("endpoint");
        assert_close(final_position[0], 1.0, 1e-12);
        assert_close(final_position[1], 2.0, 1e-12);
        for (_, _, velocity) in &result {
            assert_close(velocity[0], 1.0, 1e-12);
            assert_close(velocity[1], 2.0, 1e-12);
        }
        assert_close(final_velocity[0], 1.0, 1e-12);
        assert_close(final_velocity[1], 2.0, 1e-12);
    }

    #[test]
    fn minkowski_parallel_transport_preserves_vector() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 3]; 3]; 3];
        let curve = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![3.0, 5.0, 8.0],
        ];
        let transported = parallel_transport(&[2.0, -1.0, 4.0], &curve, &gamma).expect("transport");
        for vector in transported {
            assert_close(vector[0], 2.0, 1e-12);
            assert_close(vector[1], -1.0, 1e-12);
            assert_close(vector[2], 4.0, 1e-12);
        }
    }

    #[test]
    fn polar_plane_radial_geodesic_keeps_zero_angular_velocity() {
        let gamma = |x: &[f64]| {
            let r = x[0];
            let mut gamma = vec![vec![vec![0.0; 2]; 2]; 2];
            gamma[0][1][1] = -r;
            gamma[1][0][1] = 1.0 / r;
            gamma[1][1][0] = 1.0 / r;
            gamma
        };
        let result = integrate_geodesic(&gamma, &[1.0, 0.0], &[1.0, 0.0], (0.0, 1.0), 50)
            .expect("polar geodesic");
        for (_, _, velocity) in result {
            assert_close(velocity[1], 0.0, 1e-12);
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 2]; 2]; 2];
        assert_eq!(
            parallel_transport(&[1.0], &[vec![0.0, 0.0]], &gamma),
            Err(NumericalGRError::ParallelTransportDimensionMismatch {
                vector_dim: 1,
                curve_dim: 2
            })
        );
        assert_eq!(
            integrate_geodesic(&gamma, &[0.0], &[0.0, 1.0], (0.0, 1.0), 10),
            Err(NumericalGRError::GeodesicInitialDataMismatch)
        );
    }

    #[test]
    fn empty_curve_error() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 1]; 1]; 1];
        assert_eq!(
            parallel_transport(&[1.0], &[], &gamma),
            Err(NumericalGRError::EmptyCurve)
        );
    }

    #[test]
    fn invalid_tau_range_error() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 1]; 1]; 1];
        assert_eq!(
            integrate_geodesic(&gamma, &[0.0], &[1.0], (1.0, 0.0), 10),
            Err(NumericalGRError::InvalidTauRange)
        );
    }

    #[test]
    fn invalid_step_count_error() {
        let gamma = |_x: &[f64]| vec![vec![vec![0.0; 1]; 1]; 1];
        assert_eq!(
            integrate_geodesic(&gamma, &[0.0], &[1.0], (0.0, 1.0), 0),
            Err(NumericalGRError::InvalidStepCount)
        );
    }
}
