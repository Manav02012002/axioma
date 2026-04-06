use ax_ir::{Expr, Index, Interner, Variance};
use lasso::Spur;
use num_bigint::BigInt;
use num_rational::BigRational;

#[derive(Clone, Debug)]
pub struct SVTDecomposition {
    pub scalar_modes: Vec<ScalarMode>,
    pub vector_modes: Vec<VectorMode>,
    pub tensor_modes: Vec<TensorMode>,
}

#[derive(Clone, Debug)]
pub struct ScalarMode {
    pub name: Spur,
    pub component: SVTComponent,
}

#[derive(Clone, Debug)]
pub enum SVTComponent {
    Phi,
    Psi,
    B,
    E,
}

#[derive(Clone, Debug)]
pub struct VectorMode {
    pub name: Spur,
    pub component: VectorSVT,
}

#[derive(Clone, Debug)]
pub enum VectorSVT {
    Si,
    Fi,
}

#[derive(Clone, Debug)]
pub struct TensorMode {
    pub name: Spur,
}

#[derive(Clone, Debug)]
pub struct ReggeWheelerDecomposition {
    pub even_parity: Vec<(Spur, Expr)>,
    pub odd_parity: Vec<(Spur, Expr)>,
}

pub fn svt_decompose_perturbation(spatial_dim: usize, interner: &Interner) -> SVTDecomposition {
    assert!(
        spatial_dim > 0,
        "SVT decomposition requires at least one spatial dimension"
    );

    let scalar_modes = vec![
        ScalarMode {
            name: interner.get_or_intern("Phi"),
            component: SVTComponent::Phi,
        },
        ScalarMode {
            name: interner.get_or_intern("Psi"),
            component: SVTComponent::Psi,
        },
        ScalarMode {
            name: interner.get_or_intern("B"),
            component: SVTComponent::B,
        },
        ScalarMode {
            name: interner.get_or_intern("E"),
            component: SVTComponent::E,
        },
    ];

    let vector_modes = if spatial_dim >= 2 {
        vec![
            VectorMode {
                name: interner.get_or_intern("S"),
                component: VectorSVT::Si,
            },
            VectorMode {
                name: interner.get_or_intern("F"),
                component: VectorSVT::Fi,
            },
        ]
    } else {
        Vec::new()
    };

    let tensor_modes = if spatial_dim >= 2 {
        vec![TensorMode {
            name: interner.get_or_intern("h_TT"),
        }]
    } else {
        Vec::new()
    };

    SVTDecomposition {
        scalar_modes,
        vector_modes,
        tensor_modes,
    }
}

pub fn bardeen_variables(
    decomp: &SVTDecomposition,
    scale_factor: Spur,
    conformal_time: Spur,
    interner: &Interner,
) -> Vec<(Spur, Expr)> {
    let phi = scalar_mode(decomp, SVTComponent::Phi);
    let psi = scalar_mode(decomp, SVTComponent::Psi);
    let b = scalar_mode(decomp, SVTComponent::B);
    let e = scalar_mode(decomp, SVTComponent::E);

    let a = Expr::Sym(scale_factor);
    let eta = Expr::Sym(conformal_time);
    let e_prime = diff(Expr::Sym(e), eta.clone(), interner);
    let shear = Expr::add(vec![Expr::Sym(b), Expr::neg(e_prime)]);
    let a_shear = Expr::mul(vec![a.clone(), shear.clone()]);
    let phi_b = Expr::add(vec![
        Expr::Sym(phi),
        Expr::mul(vec![
            Expr::pow(a.clone(), int(-1)),
            diff(a_shear, eta.clone(), interner),
        ]),
    ]);
    let psi_b = Expr::add(vec![
        Expr::Sym(psi),
        Expr::neg(Expr::mul(vec![
            diff(a.clone(), eta, interner),
            Expr::pow(a, int(-1)),
            shear,
        ])),
    ]);

    vec![
        (interner.get_or_intern("Phi_B"), phi_b),
        (interner.get_or_intern("Psi_B"), psi_b),
    ]
}

pub fn regge_wheeler_decompose(l: usize, interner: &Interner) -> ReggeWheelerDecomposition {
    let t = interner.get_or_intern("t");
    let r = interner.get_or_intern("r");
    let theta = interner.get_or_intern("theta");
    let phi = interner.get_or_intern("phi");
    let y_lm = spherical_harmonic(l, theta, phi, interner);

    let even_names = ["H0", "H1", "H2", "K"]
        .into_iter()
        .map(|name| {
            let sym = interner.get_or_intern(name);
            (sym, Expr::mul(vec![mode_function(sym, t, r), y_lm.clone()]))
        })
        .collect();

    let odd_theta = interner.get_or_intern("X_lm_theta");
    let odd_phi = interner.get_or_intern("X_lm_phi");
    let odd_names = [
        (
            "h0",
            Expr::Call(
                odd_theta,
                vec![int(l as i64), Expr::Sym(theta), Expr::Sym(phi)],
            ),
        ),
        (
            "h1",
            Expr::Call(
                odd_phi,
                vec![int(l as i64), Expr::Sym(theta), Expr::Sym(phi)],
            ),
        ),
    ]
    .into_iter()
    .map(|(name, angular)| {
        let sym = interner.get_or_intern(name);
        (sym, Expr::mul(vec![mode_function(sym, t, r), angular]))
    })
    .collect();

    ReggeWheelerDecomposition {
        even_parity: even_names,
        odd_parity: odd_names,
    }
}

pub fn zerilli_equation(l: usize, mass: Spur, interner: &Interner) -> Expr {
    let r = Expr::Sym(interner.get_or_intern("r"));
    let r_star = Expr::Sym(interner.get_or_intern("r_star"));
    let omega = Expr::Sym(interner.get_or_intern("omega"));
    let psi = Expr::Sym(interner.get_or_intern("Psi_Z"));
    let m = Expr::Sym(mass);
    let n = BigRational::new(BigInt::from(((l as i64) - 1) * ((l as i64) + 2)), 2.into());
    let n_expr = Expr::Rational(n.clone());
    let one_minus_2m_over_r = Expr::add(vec![
        Expr::one(),
        Expr::neg(Expr::mul(vec![
            int(2),
            m.clone(),
            Expr::pow(r.clone(), int(-1)),
        ])),
    ]);

    let numerator = Expr::add(vec![
        Expr::mul(vec![
            int(2),
            Expr::pow(n_expr.clone(), int(2)),
            Expr::add(vec![n_expr.clone(), Expr::one()]),
            Expr::pow(r.clone(), int(3)),
        ]),
        Expr::mul(vec![
            int(6),
            Expr::pow(n_expr.clone(), int(2)),
            m.clone(),
            Expr::pow(r.clone(), int(2)),
        ]),
        Expr::mul(vec![
            int(18),
            n_expr.clone(),
            Expr::pow(m.clone(), int(2)),
            r.clone(),
        ]),
        Expr::mul(vec![int(18), Expr::pow(m.clone(), int(3))]),
    ]);
    let denominator = Expr::mul(vec![
        Expr::pow(r.clone(), int(3)),
        Expr::pow(
            Expr::add(vec![
                Expr::mul(vec![n_expr, r.clone()]),
                Expr::mul(vec![int(3), m]),
            ]),
            int(2),
        ),
    ]);
    let potential = Expr::mul(vec![
        one_minus_2m_over_r,
        numerator,
        Expr::pow(denominator, int(-1)),
    ]);
    master_equation(psi, r_star, omega, potential, interner)
}

pub fn regge_wheeler_equation(l: usize, mass: Spur, interner: &Interner) -> Expr {
    let r = Expr::Sym(interner.get_or_intern("r"));
    let r_star = Expr::Sym(interner.get_or_intern("r_star"));
    let omega = Expr::Sym(interner.get_or_intern("omega"));
    let psi = Expr::Sym(interner.get_or_intern("Psi_RW"));
    let m = Expr::Sym(mass);
    let l_l_plus_one = int((l * (l + 1)) as i64);
    let one_minus_2m_over_r = Expr::add(vec![
        Expr::one(),
        Expr::neg(Expr::mul(vec![
            int(2),
            m.clone(),
            Expr::pow(r.clone(), int(-1)),
        ])),
    ]);
    let bracket = Expr::add(vec![
        Expr::mul(vec![l_l_plus_one, Expr::pow(r.clone(), int(-2))]),
        Expr::neg(Expr::mul(vec![int(6), m, Expr::pow(r.clone(), int(-3))])),
    ]);
    let potential = Expr::mul(vec![one_minus_2m_over_r, bracket]);
    master_equation(psi, r_star, omega, potential, interner)
}

fn scalar_mode(decomp: &SVTDecomposition, component: SVTComponent) -> Spur {
    decomp
        .scalar_modes
        .iter()
        .find(|mode| same_scalar_component(&mode.component, &component))
        .map(|mode| mode.name)
        .expect("SVT decomposition is missing a standard scalar mode")
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

fn spherical_harmonic(l: usize, theta: Spur, phi: Spur, interner: &Interner) -> Expr {
    let y = interner.get_or_intern("Y_lm");
    Expr::Call(y, vec![int(l as i64), Expr::Sym(theta), Expr::Sym(phi)])
}

fn mode_function(mode: Spur, t: Spur, r: Spur) -> Expr {
    Expr::Call(mode, vec![Expr::Sym(t), Expr::Sym(r)])
}

fn master_equation(
    psi: Expr,
    r_star: Expr,
    omega: Expr,
    potential: Expr,
    interner: &Interner,
) -> Expr {
    Expr::add(vec![
        diff(
            diff(psi.clone(), r_star.clone(), interner),
            r_star,
            interner,
        ),
        Expr::mul(vec![
            Expr::add(vec![Expr::pow(omega, int(2)), Expr::neg(potential)]),
            psi,
        ]),
    ])
}

fn diff(expr: Expr, var: Expr, interner: &Interner) -> Expr {
    Expr::Call(interner.get_or_intern("diff"), vec![expr, var])
}

fn int(n: i64) -> Expr {
    Expr::Int(BigInt::from(n))
}

#[allow(dead_code)]
fn indexed_mode(symbol: Spur, first: Spur, second: Spur) -> Expr {
    Expr::Indexed(
        Box::new(Expr::Sym(symbol)),
        vec![
            Index {
                name: first,
                variance: Variance::Down,
                index_type: None,
            },
            Index {
                name: second,
                variance: Variance::Down,
                index_type: None,
            },
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svt_has_standard_modes() {
        let interner = Interner::new();
        let decomp = svt_decompose_perturbation(3, &interner);
        assert_eq!(decomp.scalar_modes.len(), 4);
        assert_eq!(decomp.vector_modes.len(), 2);
        assert_eq!(decomp.tensor_modes.len(), 1);
    }

    #[test]
    fn bardeen_builds_two_variables() {
        let interner = Interner::new();
        let decomp = svt_decompose_perturbation(3, &interner);
        let a = interner.get_or_intern("a");
        let eta = interner.get_or_intern("eta");
        let vars = bardeen_variables(&decomp, a, eta, &interner);
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn rw_and_zerilli_equations_are_symbolic_equations() {
        let interner = Interner::new();
        let m = interner.get_or_intern("M");
        assert!(matches!(
            regge_wheeler_equation(2, m, &interner),
            Expr::Add(_)
        ));
        assert!(matches!(zerilli_equation(2, m, &interner), Expr::Add(_)));
    }
}
