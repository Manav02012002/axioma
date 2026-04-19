use ax_ir::{Expr, Interner};
use ax_perturb::{
    default_scalar_gauge_generator, default_second_order_scalar_modes,
    derive_second_order_scalar_einstein_system, scalar_metric_gauge_variation,
    second_order_scalar_gauge_variation, FrwBackgroundSpec,
};

fn substitute_many(expr: &Expr, replacements: &[(lasso::Spur, Expr)]) -> Expr {
    match expr {
        Expr::Sym(sym) => replacements
            .iter()
            .find_map(|(target, replacement)| (*target == *sym).then(|| replacement.clone()))
            .unwrap_or_else(|| Expr::Sym(*sym)),
        Expr::Add(items) => Expr::add(
            items
                .iter()
                .map(|item| substitute_many(item, replacements))
                .collect(),
        ),
        Expr::Mul(items) => Expr::mul(
            items
                .iter()
                .map(|item| substitute_many(item, replacements))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_many(base, replacements),
            substitute_many(exp, replacements),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_many(inner, replacements)),
        Expr::Call(fun, args) => {
            let rewritten = args
                .iter()
                .map(|arg| substitute_many(arg, replacements))
                .collect::<Vec<_>>();
            if rewritten.len() == 2
                && matches!(rewritten.first(), Some(Expr::Int(value)) if *value == 0.into())
            {
                Expr::zero()
            } else {
                Expr::Call(*fun, rewritten)
            }
        }
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_many(item, replacements))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| substitute_many(item, replacements))
                        .collect()
                })
                .collect(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(substitute_many(inner, replacements)), *rel)
        }
        other => other.clone(),
    }
}

#[test]
fn second_order_equations_have_stable_labels() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let system = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();
    let labels = system
        .equations
        .iter()
        .map(|eq| eq.label.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        labels,
        vec![
            "second_order_00_constraint",
            "second_order_0i_momentum",
            "second_order_ij_trace",
            "second_order_ij_traceless",
        ]
    );
}

#[test]
fn second_order_source_split_reconstructs_full_equation() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let system = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();

    for equation in system.equations {
        assert_eq!(
            Expr::add(vec![
                equation.linear_second_order.clone(),
                equation.quadratic_source.clone()
            ]),
            equation.full
        );
    }
}

#[test]
fn second_order_quadratic_source_vanishes_when_first_order_modes_are_zero() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let modes = default_second_order_scalar_modes(&interner);
    let system = derive_second_order_scalar_einstein_system(&bg, &interner).unwrap();
    let replacements = vec![
        (modes.phi_1, Expr::zero()),
        (modes.psi_1, Expr::zero()),
        (modes.b_1, Expr::zero()),
        (modes.e_1, Expr::zero()),
    ];

    for equation in system.equations {
        let reduced = substitute_many(&equation.quadratic_source, &replacements);
        assert_eq!(reduced, Expr::zero());
    }
}

#[test]
fn second_order_first_order_limit_matches_prompt3_gauge_transforms() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);
    let second_order = second_order_scalar_gauge_variation(&bg, &interner).unwrap();
    let second_order_generator = ax_perturb::default_second_order_gauge_generator(&interner);

    let first_order_generator = default_scalar_gauge_generator(&interner);
    let first_order =
        scalar_metric_gauge_variation(&bg, &first_order_generator, &interner).unwrap();

    let replacements = vec![
        (
            second_order_generator.time_shift_1,
            Expr::Sym(first_order_generator.time_shift),
        ),
        (
            second_order_generator.space_shift_1,
            Expr::Sym(first_order_generator.spatial_shift),
        ),
    ];

    assert_eq!(
        substitute_many(&second_order.delta_phi_1, &replacements),
        first_order.delta_phi
    );
    assert_eq!(
        substitute_many(&second_order.delta_psi_1, &replacements),
        first_order.delta_psi
    );
    assert_eq!(
        substitute_many(&second_order.delta_b_1, &replacements),
        first_order.delta_b
    );
    assert_eq!(
        substitute_many(&second_order.delta_e_1, &replacements),
        first_order.delta_e
    );
}
