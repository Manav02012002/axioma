use crate::{GradedSymbolTable, Grading};
use ax_ir::{Expr, Index, Interner};
use lasso::Spur;

#[derive(Clone, Debug)]
pub struct SuperspaceSetup {
    pub spacetime_coords: Vec<Spur>,
    pub theta: Vec<Spur>,
    pub theta_bar: Vec<Spur>,
    pub n_susy: usize,
}

#[derive(Clone, Debug)]
pub struct SuperfieldExpansion {
    pub components: Vec<SuperfieldComponent>,
}

#[derive(Clone, Debug)]
pub struct SuperfieldComponent {
    pub theta_structure: ThetaMonomial,
    pub field: Spur,
    pub field_indices: Vec<Index>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThetaMonomial {
    pub theta_powers: Vec<u8>,
    pub theta_bar_powers: Vec<u8>,
}

impl ThetaMonomial {
    pub fn is_zero(&self) -> bool {
        self.theta_powers.iter().any(|power| *power > 1)
            || self.theta_bar_powers.iter().any(|power| *power > 1)
    }

    pub fn total_theta(&self) -> usize {
        self.theta_powers
            .iter()
            .map(|power| usize::from(*power))
            .sum()
    }

    pub fn total_theta_bar(&self) -> usize {
        self.theta_bar_powers
            .iter()
            .map(|power| usize::from(*power))
            .sum()
    }

    pub fn mass_dimension_contribution(&self) -> i32 {
        -((self.total_theta() + self.total_theta_bar()) as i32)
    }
}

pub fn setup_n1_superspace(interner: &Interner) -> (SuperspaceSetup, GradedSymbolTable) {
    let spacetime_coords = (0..4)
        .map(|idx| interner.get_or_intern(&format!("x{idx}")))
        .collect::<Vec<_>>();
    let theta = (1..=2)
        .map(|idx| interner.get_or_intern(&format!("theta{idx}")))
        .collect::<Vec<_>>();
    let theta_bar = (1..=2)
        .map(|idx| interner.get_or_intern(&format!("theta_bar{idx}")))
        .collect::<Vec<_>>();

    let mut table = GradedSymbolTable::new();
    for coord in &spacetime_coords {
        table.declare(*coord, Grading::bosonic());
    }
    for theta_sym in theta.iter().chain(theta_bar.iter()) {
        table.declare(*theta_sym, Grading::fermionic());
    }

    (
        SuperspaceSetup {
            spacetime_coords,
            theta,
            theta_bar,
            n_susy: 1,
        },
        table,
    )
}

pub fn expand_superfield(
    name: Spur,
    setup: &SuperspaceSetup,
    interner: &Interner,
) -> SuperfieldExpansion {
    let base_name = interner.resolve(name);
    let mut components = Vec::new();
    for monomial in all_theta_monomials(setup) {
        if monomial.is_zero() {
            continue;
        }
        let field_name = component_name(base_name, &monomial);
        components.push(SuperfieldComponent {
            theta_structure: monomial,
            field: interner.get_or_intern(&field_name),
            field_indices: Vec::new(),
        });
    }
    SuperfieldExpansion { components }
}

pub fn superfield_to_expr(
    expansion: &SuperfieldExpansion,
    setup: &SuperspaceSetup,
    _interner: &Interner,
) -> Expr {
    Expr::add(
        expansion
            .components
            .iter()
            .map(|component| {
                let mut factors = theta_factors(&component.theta_structure, setup);
                factors.push(component_field_expr(component, setup));
                Expr::mul(factors)
            })
            .collect(),
    )
}

pub fn extract_component(
    expr: &Expr,
    theta_structure: &ThetaMonomial,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Expr {
    let simplified = crate::graded_simplify(&expand_products(expr), table, interner);
    let terms = match simplified {
        Expr::Add(terms) => terms,
        other => vec![other],
    };
    Expr::add(
        terms
            .into_iter()
            .filter_map(|term| {
                extract_theta_coefficient(&term, theta_structure, setup, table, interner)
            })
            .collect(),
    )
}

pub fn chiral_constraint(
    expansion: &SuperfieldExpansion,
    _setup: &SuperspaceSetup,
    _interner: &Interner,
) -> SuperfieldExpansion {
    SuperfieldExpansion {
        components: expansion
            .components
            .iter()
            .filter(|component| {
                component.theta_structure.total_theta_bar() == 0
                    && component.theta_structure.total_theta() <= 2
            })
            .cloned()
            .collect(),
    }
}

pub fn antichiral_constraint(
    expansion: &SuperfieldExpansion,
    _setup: &SuperspaceSetup,
    _interner: &Interner,
) -> SuperfieldExpansion {
    SuperfieldExpansion {
        components: expansion
            .components
            .iter()
            .filter(|component| {
                component.theta_structure.total_theta() == 0
                    && component.theta_structure.total_theta_bar() <= 2
            })
            .cloned()
            .collect(),
    }
}

pub fn vector_superfield_wz_gauge(
    name: Spur,
    setup: &SuperspaceSetup,
    interner: &Interner,
) -> SuperfieldExpansion {
    let base = interner.resolve(name);
    let structures = [
        (
            ThetaMonomial {
                theta_powers: vec![1; setup.theta.len()],
                theta_bar_powers: vec![0, 1],
            },
            format!("lambda_bar_{base}"),
        ),
        (
            ThetaMonomial {
                theta_powers: vec![0, 1],
                theta_bar_powers: vec![1; setup.theta_bar.len()],
            },
            format!("lambda_{base}"),
        ),
        (
            ThetaMonomial {
                theta_powers: vec![1, 0],
                theta_bar_powers: vec![0, 1],
            },
            format!("A_{base}"),
        ),
        (
            ThetaMonomial {
                theta_powers: vec![1; setup.theta.len()],
                theta_bar_powers: vec![1; setup.theta_bar.len()],
            },
            format!("D_{base}"),
        ),
    ];

    SuperfieldExpansion {
        components: structures
            .into_iter()
            .filter(|(monomial, _)| !monomial.is_zero())
            .map(|(theta_structure, field_name)| SuperfieldComponent {
                theta_structure,
                field: interner.get_or_intern(&field_name),
                field_indices: Vec::new(),
            })
            .collect(),
    }
}

fn all_theta_monomials(setup: &SuperspaceSetup) -> Vec<ThetaMonomial> {
    let theta_count = setup.theta.len();
    let theta_bar_count = setup.theta_bar.len();
    let theta_max = 1usize << theta_count;
    let theta_bar_max = 1usize << theta_bar_count;
    let mut out = Vec::new();
    for theta_mask in 0..theta_max {
        for theta_bar_mask in 0..theta_bar_max {
            out.push(ThetaMonomial {
                theta_powers: mask_to_powers(theta_mask, theta_count),
                theta_bar_powers: mask_to_powers(theta_bar_mask, theta_bar_count),
            });
        }
    }
    out
}

fn mask_to_powers(mask: usize, len: usize) -> Vec<u8> {
    (0..len)
        .map(|idx| if (mask & (1usize << idx)) == 0 { 0 } else { 1 })
        .collect()
}

fn component_name(base: &str, monomial: &ThetaMonomial) -> String {
    match (monomial.total_theta(), monomial.total_theta_bar()) {
        (0, 0) => format!("f_{base}"),
        (1, 0) => format!("psi_{base}"),
        (0, 1) => format!("chi_bar_{base}"),
        (2, 0) => format!("F_{base}"),
        (0, 2) => format!("F_bar_{base}"),
        (1, 1) => format!("v_{base}"),
        (2, 1) => format!("lambda_bar_{base}"),
        (1, 2) => format!("rho_{base}"),
        (2, 2) => format!("d_{base}"),
        (theta, theta_bar) => format!("component_{theta}_{theta_bar}_{base}"),
    }
}

fn theta_factors(monomial: &ThetaMonomial, setup: &SuperspaceSetup) -> Vec<Expr> {
    setup
        .theta
        .iter()
        .zip(&monomial.theta_powers)
        .flat_map(|(sym, power)| std::iter::repeat(Expr::Sym(*sym)).take(usize::from(*power)))
        .chain(
            setup
                .theta_bar
                .iter()
                .zip(&monomial.theta_bar_powers)
                .flat_map(|(sym, power)| {
                    std::iter::repeat(Expr::Sym(*sym)).take(usize::from(*power))
                }),
        )
        .collect()
}

fn component_field_expr(component: &SuperfieldComponent, setup: &SuperspaceSetup) -> Expr {
    let base = Expr::Call(
        component.field,
        setup
            .spacetime_coords
            .iter()
            .map(|coord| Expr::Sym(*coord))
            .collect(),
    );
    if component.field_indices.is_empty() {
        base
    } else {
        Expr::Indexed(Box::new(base), component.field_indices.clone())
    }
}

fn expand_products(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(terms.iter().map(expand_products).collect()),
        Expr::Mul(factors) => {
            let mut acc = vec![Expr::one()];
            for factor in factors {
                let factor = expand_products(factor);
                let factor_terms = match factor {
                    Expr::Add(terms) => terms,
                    other => vec![other],
                };
                let mut next = Vec::new();
                for lhs in &acc {
                    for rhs in &factor_terms {
                        next.push(Expr::mul(vec![lhs.clone(), rhs.clone()]));
                    }
                }
                acc = next;
            }
            Expr::add(acc)
        }
        Expr::Pow(base, exp) => Expr::pow(expand_products(base), expand_products(exp)),
        Expr::Neg(inner) => Expr::neg(expand_products(inner)),
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(expand_products).collect()),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(expand_products(base)), indices.clone())
        }
        _ => expr.clone(),
    }
}

fn extract_theta_coefficient(
    term: &Expr,
    target: &ThetaMonomial,
    setup: &SuperspaceSetup,
    table: &GradedSymbolTable,
    interner: &Interner,
) -> Option<Expr> {
    let factors = match term {
        Expr::Mul(factors) => factors.clone(),
        other => vec![other.clone()],
    };
    let mut theta_powers = vec![0u8; setup.theta.len()];
    let mut theta_bar_powers = vec![0u8; setup.theta_bar.len()];
    let mut coefficient = Vec::new();

    for factor in factors {
        match factor {
            Expr::Sym(s) => {
                if let Some(pos) = setup.theta.iter().position(|theta| *theta == s) {
                    theta_powers[pos] += 1;
                } else if let Some(pos) =
                    setup.theta_bar.iter().position(|theta_bar| *theta_bar == s)
                {
                    theta_bar_powers[pos] += 1;
                } else {
                    coefficient.push(Expr::Sym(s));
                }
            }
            other => coefficient.push(other),
        }
    }

    let actual = ThetaMonomial {
        theta_powers,
        theta_bar_powers,
    };
    if &actual == target {
        Some(crate::graded_simplify(
            &Expr::mul(coefficient),
            table,
            interner,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_declares_theta_as_fermionic() {
        let interner = Interner::new();
        let (setup, table) = setup_n1_superspace(&interner);
        assert_eq!(setup.spacetime_coords.len(), 4);
        assert!(table.get(setup.theta[0]).unwrap().is_fermionic());
        assert!(table.get(setup.spacetime_coords[0]).unwrap().is_bosonic());
    }

    #[test]
    fn chiral_constraint_keeps_only_theta_modes() {
        let interner = Interner::new();
        let (setup, _) = setup_n1_superspace(&interner);
        let phi = interner.get_or_intern("Phi");
        let expansion = expand_superfield(phi, &setup, &interner);
        let chiral = chiral_constraint(&expansion, &setup, &interner);
        assert!(chiral
            .components
            .iter()
            .all(|component| component.theta_structure.total_theta_bar() == 0));
    }

    #[test]
    fn extracts_theta_component() {
        let interner = Interner::new();
        let (setup, table) = setup_n1_superspace(&interner);
        let field = interner.get_or_intern("f");
        let expr = Expr::mul(vec![Expr::Sym(setup.theta[0]), Expr::Call(field, vec![])]);
        let target = ThetaMonomial {
            theta_powers: vec![1, 0],
            theta_bar_powers: vec![0, 0],
        };
        assert_eq!(
            extract_component(&expr, &target, &setup, &table, &interner),
            Expr::Call(field, vec![])
        );
    }
}
