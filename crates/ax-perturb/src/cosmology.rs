use crate::gauge::{SVTComponent, SVTDecomposition};
use ax_ir::{Expr, Interner};
use lasso::Spur;
use num_bigint::BigInt;

pub struct FRWBackground {
    pub scale_factor: Spur,
    pub hubble: Spur,
    pub conformal_time: Spur,
    pub cosmic_time: Spur,
    pub spatial_dim: usize,
}

pub fn frw_background(interner: &Interner) -> FRWBackground {
    FRWBackground {
        scale_factor: interner.get_or_intern("a"),
        hubble: interner.get_or_intern("H"),
        conformal_time: interner.get_or_intern("eta"),
        cosmic_time: interner.get_or_intern("t"),
        spatial_dim: 3,
    }
}

pub fn linearized_einstein_scalar(
    bg: &FRWBackground,
    decomp: &SVTDecomposition,
    interner: &Interner,
) -> Vec<(String, Expr)> {
    let phi = Expr::Sym(mode_or_standard(decomp, SVTComponent::Phi, "Phi", interner));
    let psi = Expr::Sym(mode_or_standard(decomp, SVTComponent::Psi, "Psi", interner));
    let h = Expr::Sym(bg.hubble);
    let a = Expr::Sym(bg.scale_factor);
    let eta = Expr::Sym(bg.conformal_time);
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    let g_newton = Expr::Sym(interner.get_or_intern("G"));
    let delta_rho = Expr::Sym(interner.get_or_intern("delta_rho"));
    let velocity = Expr::Sym(interner.get_or_intern("v"));
    let rho = Expr::Sym(interner.get_or_intern("rho"));
    let pressure = Expr::Sym(interner.get_or_intern("P"));
    let delta_pressure = Expr::Sym(interner.get_or_intern("delta_P"));
    let anisotropic_stress = Expr::Sym(interner.get_or_intern("Pi"));
    let four_pi_g_a2 = Expr::mul(vec![
        int(4),
        pi.clone(),
        g_newton.clone(),
        Expr::pow(a.clone(), int(2)),
    ]);
    let eight_pi_g_a2 = Expr::mul(vec![int(8), pi, g_newton, Expr::pow(a.clone(), int(2))]);
    let psi_prime = diff(psi.clone(), eta.clone(), interner);
    let phi_prime = diff(phi.clone(), eta.clone(), interner);
    let h_phi = Expr::mul(vec![h.clone(), phi.clone()]);
    let shear_combo = Expr::add(vec![psi_prime.clone(), h_phi.clone()]);

    let eq_00 = Expr::add(vec![
        laplacian(psi.clone(), interner),
        Expr::neg(Expr::mul(vec![int(3), h.clone(), shear_combo.clone()])),
        Expr::neg(Expr::mul(vec![four_pi_g_a2.clone(), delta_rho])),
    ]);

    let eq_0i = Expr::add(vec![
        shear_combo.clone(),
        Expr::mul(vec![
            four_pi_g_a2.clone(),
            Expr::add(vec![rho, pressure]),
            velocity,
        ]),
    ]);

    let eq_trace = Expr::add(vec![
        diff(psi_prime.clone(), eta.clone(), interner),
        Expr::mul(vec![
            h.clone(),
            Expr::add(vec![Expr::mul(vec![int(2), psi_prime]), phi_prime]),
        ]),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![int(2), diff(h.clone(), eta, interner)]),
                Expr::pow(h.clone(), int(2)),
            ]),
            phi.clone(),
        ]),
        Expr::mul(vec![
            rational(1, 3),
            laplacian(
                Expr::add(vec![phi.clone(), Expr::neg(psi.clone())]),
                interner,
            ),
        ]),
        Expr::neg(Expr::mul(vec![four_pi_g_a2, delta_pressure])),
    ]);

    let eq_traceless = Expr::add(vec![
        phi,
        Expr::neg(psi),
        Expr::neg(Expr::mul(vec![eight_pi_g_a2, anisotropic_stress])),
    ]);

    vec![
        ("00_constraint".to_string(), eq_00),
        ("0i_momentum".to_string(), eq_0i),
        ("ij_trace".to_string(), eq_trace),
        ("ij_traceless".to_string(), eq_traceless),
    ]
}

pub fn mukhanov_sasaki_equation(
    bg: &FRWBackground,
    slow_roll_epsilon: Spur,
    interner: &Interner,
) -> Expr {
    let eta = Expr::Sym(bg.conformal_time);
    let v = Expr::Sym(interner.get_or_intern("v"));
    let k = Expr::Sym(interner.get_or_intern("k"));
    let epsilon = Expr::Sym(slow_roll_epsilon);
    let cs = Expr::Sym(interner.get_or_intern("c_s"));
    let z = Expr::mul(vec![
        Expr::Sym(bg.scale_factor),
        Expr::pow(Expr::mul(vec![int(2), epsilon]), rational(1, 2)),
        Expr::pow(cs, int(-1)),
    ]);
    Expr::add(vec![
        diff(
            diff(v.clone(), eta.clone(), interner),
            eta.clone(),
            interner,
        ),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::pow(k, int(2)),
                Expr::neg(Expr::mul(vec![
                    diff(diff(z.clone(), eta.clone(), interner), eta, interner),
                    Expr::pow(z, int(-1)),
                ])),
            ]),
            v,
        ]),
    ])
}

pub fn power_spectrum_leading(
    _bg: &FRWBackground,
    slow_roll_epsilon: Spur,
    hubble_at_crossing: Spur,
    interner: &Interner,
) -> Expr {
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    Expr::mul(vec![
        Expr::pow(Expr::Sym(hubble_at_crossing), int(2)),
        Expr::pow(
            Expr::mul(vec![
                int(8),
                Expr::pow(pi, int(2)),
                Expr::Sym(slow_roll_epsilon),
            ]),
            int(-1),
        ),
    ])
}

pub fn spectral_index(slow_roll_epsilon: Spur, slow_roll_eta: Spur, _interner: &Interner) -> Expr {
    Expr::add(vec![
        Expr::one(),
        Expr::neg(Expr::mul(vec![int(6), Expr::Sym(slow_roll_epsilon)])),
        Expr::mul(vec![int(2), Expr::Sym(slow_roll_eta)]),
    ])
}

pub fn tensor_to_scalar_ratio(slow_roll_epsilon: Spur, _interner: &Interner) -> Expr {
    Expr::mul(vec![int(16), Expr::Sym(slow_roll_epsilon)])
}

pub fn linearized_einstein_second_order(
    bg: &FRWBackground,
    _decomp: &SVTDecomposition,
    interner: &Interner,
) -> Vec<(String, Expr)> {
    let phi2 = Expr::Sym(interner.get_or_intern("Phi_2"));
    let psi2 = Expr::Sym(interner.get_or_intern("Psi_2"));
    let phi1 = Expr::Sym(interner.get_or_intern("Phi_1"));
    let psi1 = Expr::Sym(interner.get_or_intern("Psi_1"));
    let h = Expr::Sym(bg.hubble);
    let eta = Expr::Sym(bg.conformal_time);
    let pi = Expr::Sym(interner.get_or_intern("pi"));
    let g_newton = Expr::Sym(interner.get_or_intern("G"));
    let a2 = Expr::pow(Expr::Sym(bg.scale_factor), int(2));
    let source_00 = Expr::add(vec![
        Expr::pow(diff(psi1.clone(), eta.clone(), interner), int(2)),
        Expr::mul(vec![
            partial_i(psi1.clone(), interner),
            partial_i(psi1.clone(), interner),
        ]),
        Expr::mul(vec![phi1.clone(), laplacian(psi1.clone(), interner)]),
    ]);
    let source_0i = Expr::add(vec![
        Expr::mul(vec![phi1.clone(), partial_i(psi1.clone(), interner)]),
        Expr::mul(vec![
            diff(psi1.clone(), eta.clone(), interner),
            partial_i(psi1.clone(), interner),
        ]),
    ]);
    let source_trace = Expr::add(vec![
        Expr::mul(vec![
            diff(phi1.clone(), eta.clone(), interner),
            diff(psi1.clone(), eta.clone(), interner),
        ]),
        Expr::pow(diff(psi1.clone(), eta.clone(), interner), int(2)),
        Expr::mul(vec![
            partial_i(phi1.clone(), interner),
            partial_i(psi1.clone(), interner),
        ]),
    ]);
    let source_traceless = Expr::add(vec![
        Expr::mul(vec![
            partial_i(phi1.clone(), interner),
            partial_i(phi1, interner),
        ]),
        Expr::neg(Expr::mul(vec![
            partial_i(psi1.clone(), interner),
            partial_i(psi1, interner),
        ])),
    ]);
    let quadratic_prefactor = Expr::mul(vec![int(4), pi, g_newton, a2]);

    let psi2_prime = diff(psi2.clone(), eta.clone(), interner);
    let phi2_prime = diff(phi2.clone(), eta.clone(), interner);
    let linear_00 = Expr::add(vec![
        laplacian(psi2.clone(), interner),
        Expr::neg(Expr::mul(vec![
            int(3),
            h.clone(),
            Expr::add(vec![
                psi2_prime.clone(),
                Expr::mul(vec![h.clone(), phi2.clone()]),
            ]),
        ])),
    ]);
    let linear_0i = Expr::add(vec![
        psi2_prime.clone(),
        Expr::mul(vec![h.clone(), phi2.clone()]),
    ]);
    let linear_trace = Expr::add(vec![
        diff(psi2_prime, eta.clone(), interner),
        Expr::mul(vec![
            h.clone(),
            Expr::add(vec![
                Expr::mul(vec![int(2), diff(psi2.clone(), eta.clone(), interner)]),
                phi2_prime,
            ]),
        ]),
        Expr::mul(vec![
            Expr::add(vec![
                Expr::mul(vec![int(2), diff(h, eta, interner)]),
                Expr::pow(Expr::Sym(bg.hubble), int(2)),
            ]),
            phi2.clone(),
        ]),
        Expr::mul(vec![
            rational(1, 3),
            laplacian(
                Expr::add(vec![phi2.clone(), Expr::neg(psi2.clone())]),
                interner,
            ),
        ]),
    ]);
    let linear_traceless = Expr::add(vec![phi2, Expr::neg(psi2)]);

    vec![
        (
            "second_order_00_constraint".to_string(),
            Expr::add(vec![
                linear_00,
                Expr::neg(Expr::mul(vec![quadratic_prefactor.clone(), source_00])),
            ]),
        ),
        (
            "second_order_0i_momentum".to_string(),
            Expr::add(vec![
                linear_0i,
                Expr::neg(Expr::mul(vec![quadratic_prefactor.clone(), source_0i])),
            ]),
        ),
        (
            "second_order_ij_trace".to_string(),
            Expr::add(vec![
                linear_trace,
                Expr::neg(Expr::mul(vec![quadratic_prefactor.clone(), source_trace])),
            ]),
        ),
        (
            "second_order_ij_traceless".to_string(),
            Expr::add(vec![
                linear_traceless,
                Expr::neg(Expr::mul(vec![quadratic_prefactor, source_traceless])),
            ]),
        ),
    ]
}

fn mode_or_standard(
    decomp: &SVTDecomposition,
    component: SVTComponent,
    fallback: &str,
    interner: &Interner,
) -> Spur {
    decomp
        .scalar_modes
        .iter()
        .find(|mode| same_scalar_component(&mode.component, &component))
        .map(|mode| mode.name)
        .unwrap_or_else(|| interner.get_or_intern(fallback))
}

fn same_scalar_component(lhs: &SVTComponent, rhs: &SVTComponent) -> bool {
    matches!(
        (lhs, rhs),
        (SVTComponent::Phi, SVTComponent::Phi)
            | (SVTComponent::Psi, SVTComponent::Psi)
            | (SVTComponent::B, SVTComponent::B)
            | (SVTComponent::E, SVTComponent::E)
    )
}

fn diff(expr: Expr, var: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, var])
}

fn laplacian(expr: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("laplacian"), vec![expr])
}

fn partial_i(expr: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("partial_i"), vec![expr])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn rational(num: i64, den: i64) -> Expr {
    Expr::Rational(num_rational::BigRational::new(num.into(), den.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::svt_decompose_perturbation;

    #[test]
    fn scalar_equations_have_four_components() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let decomp = svt_decompose_perturbation(3, &interner);
        let equations = linearized_einstein_scalar(&bg, &decomp, &interner);
        assert_eq!(equations.len(), 4);
    }

    #[test]
    fn inflation_observables_are_symbolic() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let eps = interner.get_or_intern("epsilon");
        let eta = interner.get_or_intern("eta_sr");
        let h_star = interner.get_or_intern("H_star");
        assert!(matches!(
            power_spectrum_leading(&bg, eps, h_star, &interner),
            Expr::Mul(_)
        ));
        assert!(matches!(spectral_index(eps, eta, &interner), Expr::Add(_)));
        assert!(matches!(
            tensor_to_scalar_ratio(eps, &interner),
            Expr::Mul(_)
        ));
    }

    #[test]
    fn second_order_equations_have_sources() {
        let interner = Interner::new();
        let bg = frw_background(&interner);
        let decomp = svt_decompose_perturbation(3, &interner);
        let equations = linearized_einstein_second_order(&bg, &decomp, &interner);
        assert_eq!(equations.len(), 4);
    }
}
