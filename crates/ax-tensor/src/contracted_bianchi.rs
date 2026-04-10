use crate::PropertyLookup;
use ax_ir::{Expr, Index, Interner, TensorProperty};
use lasso::Spur;
use num_rational::BigRational;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContractedBianchiError {
    #[error("contracted_bianchi_reduce requires derivative symbol '{derivative}' to match the abstract covariant-derivative head in the expression")]
    DerivativeHeadMismatch { derivative: String },
    #[error("contracted_bianchi_reduce encountered repeated index '{index}' with identical variance; a contraction requires one up and one down occurrence")]
    InvalidRepeatedIndexVariance { index: String },
    #[error("contracted_bianchi_reduce encountered repeated index '{index}' across different index families")]
    MismatchedIndexFamilies { index: String },
}

fn indexed_symbol_and_indices(expr: &Expr) -> Option<(Spur, &[Index])> {
    let Expr::Indexed(base, indices) = expr else {
        return None;
    };
    let Expr::Sym(sym) = base.as_ref() else {
        return None;
    };
    Some((*sym, indices))
}

fn is_covariant_derivative(sym: Spur, properties: &dyn PropertyLookup) -> bool {
    properties
        .get_properties(sym)
        .into_iter()
        .any(|prop| matches!(prop, TensorProperty::CovariantDerivative))
}

fn parse_numeric_factor(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Neg(inner) => parse_numeric_factor(inner).map(std::ops::Neg::neg),
        _ => None,
    }
}

fn split_mul_coefficients(factors: &[Expr]) -> (BigRational, Vec<Expr>) {
    let mut coeff = BigRational::from_integer(1.into());
    let mut rest = Vec::new();
    for factor in factors {
        if let Some(value) = parse_numeric_factor(factor) {
            coeff *= value;
        } else {
            rest.push(factor.clone());
        }
    }
    (coeff, rest)
}

fn validate_repeated_pair(
    lhs: &Index,
    rhs: &Index,
    interner: &Interner,
) -> Result<bool, ContractedBianchiError> {
    if lhs.name != rhs.name {
        return Ok(false);
    }
    if lhs.index_type != rhs.index_type {
        return Err(ContractedBianchiError::MismatchedIndexFamilies {
            index: interner.resolve(lhs.name).to_string(),
        });
    }
    if lhs.variance == rhs.variance {
        return Err(ContractedBianchiError::InvalidRepeatedIndexVariance {
            index: interner.resolve(lhs.name).to_string(),
        });
    }
    Ok(true)
}

fn build_scalar_gradient(derivative_sym: Spur, scalar_sym: Spur, free_index: &Index) -> Expr {
    Expr::mul(vec![
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::Indexed(
            Box::new(Expr::Sym(derivative_sym)),
            vec![free_index.clone()],
        ),
        Expr::Sym(scalar_sym),
    ])
}

fn rewrite_product(
    factors: &[Expr],
    derivative_sym: Spur,
    ricci_sym: Spur,
    scalar_sym: Spur,
    einstein_sym: Option<Spur>,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Result<Option<Expr>, ContractedBianchiError> {
    let (coeff, nonnumeric) = split_mul_coefficients(factors);
    let derivative_name = interner.resolve(derivative_sym).to_string();

    for factor in &nonnumeric {
        let Some((sym, _)) = indexed_symbol_and_indices(factor) else {
            continue;
        };
        if sym != derivative_sym && is_covariant_derivative(sym, properties) {
            return Err(ContractedBianchiError::DerivativeHeadMismatch {
                derivative: derivative_name.clone(),
            });
        }
    }

    if nonnumeric.len() != 2 {
        return Ok(None);
    }

    let derivative_factor =
        nonnumeric
            .iter()
            .find_map(|factor| match indexed_symbol_and_indices(factor) {
                Some((sym, indices)) if sym == derivative_sym && indices.len() == 1 => {
                    Some((factor, &indices[0]))
                }
                _ => None,
            });
    let Some((_, derivative_index)) = derivative_factor else {
        return Ok(None);
    };

    let target_factor = nonnumeric
        .iter()
        .find(|factor| !matches!(indexed_symbol_and_indices(factor), Some((sym, _)) if sym == derivative_sym));
    let Some(target_factor) = target_factor else {
        return Ok(None);
    };
    let Some((target_sym, target_indices)) = indexed_symbol_and_indices(target_factor) else {
        return Ok(None);
    };
    if target_indices.len() != 2 {
        return Ok(None);
    }

    if !validate_repeated_pair(derivative_index, &target_indices[0], interner)? {
        return Ok(None);
    }

    let rewritten = if target_sym == ricci_sym {
        build_scalar_gradient(derivative_sym, scalar_sym, &target_indices[1])
    } else if Some(target_sym) == einstein_sym {
        Expr::zero()
    } else {
        return Ok(None);
    };

    if rewritten == Expr::zero() {
        return Ok(Some(Expr::zero()));
    }

    let scaled = if coeff == BigRational::from_integer(1.into()) {
        rewritten
    } else {
        Expr::mul(vec![Expr::Rational(coeff), rewritten])
    };
    Ok(Some(scaled))
}

fn rewrite_expr(
    expr: &Expr,
    derivative_sym: Spur,
    ricci_sym: Spur,
    scalar_sym: Spur,
    einstein_sym: Option<Spur>,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Result<Expr, ContractedBianchiError> {
    match expr {
        Expr::Add(terms) => {
            let mut rewritten = Vec::with_capacity(terms.len());
            for term in terms {
                rewritten.push(rewrite_expr(
                    term,
                    derivative_sym,
                    ricci_sym,
                    scalar_sym,
                    einstein_sym,
                    properties,
                    interner,
                )?);
            }
            Ok(Expr::add(rewritten))
        }
        Expr::Mul(factors) => {
            let mut rewritten = Vec::with_capacity(factors.len());
            for factor in factors {
                rewritten.push(rewrite_expr(
                    factor,
                    derivative_sym,
                    ricci_sym,
                    scalar_sym,
                    einstein_sym,
                    properties,
                    interner,
                )?);
            }
            match rewrite_product(
                &rewritten,
                derivative_sym,
                ricci_sym,
                scalar_sym,
                einstein_sym,
                properties,
                interner,
            )? {
                Some(reduced) => Ok(reduced),
                None => Ok(Expr::mul(rewritten)),
            }
        }
        Expr::Neg(inner) => Ok(Expr::neg(rewrite_expr(
            inner,
            derivative_sym,
            ricci_sym,
            scalar_sym,
            einstein_sym,
            properties,
            interner,
        )?)),
        Expr::Group(inner, rel) => Ok(Expr::Group(
            Box::new(rewrite_expr(
                inner,
                derivative_sym,
                ricci_sym,
                scalar_sym,
                einstein_sym,
                properties,
                interner,
            )?),
            *rel,
        )),
        Expr::Let(name, value, body) => Ok(Expr::Let(
            *name,
            Box::new(rewrite_expr(
                value,
                derivative_sym,
                ricci_sym,
                scalar_sym,
                einstein_sym,
                properties,
                interner,
            )?),
            Box::new(rewrite_expr(
                body,
                derivative_sym,
                ricci_sym,
                scalar_sym,
                einstein_sym,
                properties,
                interner,
            )?),
        )),
        Expr::List(items) => {
            let mut rewritten = Vec::with_capacity(items.len());
            for item in items {
                rewritten.push(rewrite_expr(
                    item,
                    derivative_sym,
                    ricci_sym,
                    scalar_sym,
                    einstein_sym,
                    properties,
                    interner,
                )?);
            }
            Ok(Expr::List(rewritten))
        }
        Expr::Matrix(rows) => {
            let mut rewritten_rows = Vec::with_capacity(rows.len());
            for row in rows {
                let mut rewritten_row = Vec::with_capacity(row.len());
                for item in row {
                    rewritten_row.push(rewrite_expr(
                        item,
                        derivative_sym,
                        ricci_sym,
                        scalar_sym,
                        einstein_sym,
                        properties,
                        interner,
                    )?);
                }
                rewritten_rows.push(rewritten_row);
            }
            Ok(Expr::Matrix(rewritten_rows))
        }
        _ => Ok(expr.clone()),
    }
}

/// Reduce abstract contracted-Bianchi patterns.
///
/// Supported identities:
///   ∇^a G_ab = 0
///   ∇_a G^a_b = 0
///   ∇^a R_ab = 1/2 ∇_b R
///   ∇_a R^a_b = 1/2 ∇_b R
///
/// The reducer is purely abstract/index-based. It does not insert metrics.
/// It recognizes contraction only when the derivative product already contains
/// one upper and one lower occurrence of the same abstract index name.
///
/// `ricci_sym`, `scalar_sym`, and `einstein_sym` are explicit target symbols.
/// `einstein_sym` may be None, in which case only the Ricci/scalar form is reduced.
pub fn contracted_bianchi_reduce(
    expr: &ax_ir::Expr,
    derivative_sym: lasso::Spur,
    ricci_sym: lasso::Spur,
    scalar_sym: lasso::Spur,
    einstein_sym: Option<lasso::Spur>,
    properties: &dyn crate::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, ContractedBianchiError> {
    let rewritten = rewrite_expr(
        expr,
        derivative_sym,
        ricci_sym,
        scalar_sym,
        einstein_sym,
        properties,
        interner,
    )?;
    Ok(crate::canonicalise(&rewritten, properties, interner))
}

#[cfg(test)]
mod tests {
    use super::{contracted_bianchi_reduce, ContractedBianchiError};
    use crate::PropertyLookup;
    use ax_ir::{Expr, Index, TensorProperty, Variance};
    use num_rational::BigRational;

    use std::collections::HashMap;

    #[derive(Default)]
    struct TestProps {
        props: HashMap<lasso::Spur, Vec<TensorProperty>>,
    }

    impl PropertyLookup for TestProps {
        fn get_properties(&self, name: lasso::Spur) -> Vec<TensorProperty> {
            self.props.get(&name).cloned().unwrap_or_default()
        }

        fn get_properties_with_indices(
            &self,
            name: lasso::Spur,
            _indices: &[Index],
            _successor: Option<(lasso::Spur, &[Index])>,
        ) -> Vec<TensorProperty> {
            self.get_properties(name)
        }

        fn has_property_kind(&self, name: lasso::Spur, kind: &TensorProperty) -> bool {
            self.get_properties(name)
                .into_iter()
                .any(|prop| std::mem::discriminant(&prop) == std::mem::discriminant(kind))
        }
    }

    fn idx(
        interner: &ax_ir::Interner,
        name: &str,
        variance: Variance,
        family: Option<&str>,
    ) -> Index {
        Index {
            name: interner.get_or_intern(name),
            variance,
            index_type: family.map(|label| interner.get_or_intern(label)),
        }
    }

    fn tensor(interner: &ax_ir::Interner, sym: &str, indices: Vec<Index>) -> Expr {
        Expr::Indexed(Box::new(Expr::Sym(interner.get_or_intern(sym))), indices)
    }

    fn props(interner: &ax_ir::Interner) -> TestProps {
        TestProps {
            props: HashMap::from([(
                interner.get_or_intern("nabla"),
                vec![TensorProperty::CovariantDerivative],
            )]),
        }
    }

    #[test]
    fn einstein_divergence_vanishes_lower_second_slot() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
            tensor(
                &interner,
                "G",
                vec![
                    idx(&interner, "a", Variance::Down, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            Some(interner.get_or_intern("G")),
            &properties,
            &interner,
        )
        .expect("einstein divergence should reduce");
        assert_eq!(reduced, Expr::zero());
    }

    #[test]
    fn einstein_divergence_vanishes_upper_contracted_slot() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Down, None)],
            ),
            tensor(
                &interner,
                "G",
                vec![
                    idx(&interner, "a", Variance::Up, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            Some(interner.get_or_intern("G")),
            &properties,
            &interner,
        )
        .expect("einstein divergence should reduce");
        assert_eq!(reduced, Expr::zero());
    }

    #[test]
    fn ricci_divergence_rewrites_to_half_scalar_gradient() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "a", Variance::Down, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let expected = crate::canonicalise(
            &Expr::mul(vec![
                Expr::Rational(BigRational::new(1.into(), 2.into())),
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "b", Variance::Down, None)],
                ),
                Expr::Sym(interner.get_or_intern("R")),
            ]),
            &properties,
            &interner,
        );
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect("ricci divergence should reduce");
        assert_eq!(reduced, expected);
    }

    #[test]
    fn ricci_divergence_rewrites_to_half_scalar_gradient_with_lower_derivative() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Down, None)],
            ),
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "a", Variance::Up, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let expected = crate::canonicalise(
            &Expr::mul(vec![
                Expr::Rational(BigRational::new(1.into(), 2.into())),
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "b", Variance::Down, None)],
                ),
                Expr::Sym(interner.get_or_intern("R")),
            ]),
            &properties,
            &interner,
        );
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect("ricci divergence should reduce");
        assert_eq!(reduced, expected);
    }

    #[test]
    fn explicit_contracted_bianchi_sum_reduces_to_zero_left_order() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::add(vec![
            Expr::mul(vec![
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "b", Variance::Down, None)],
                ),
                Expr::Sym(interner.get_or_intern("R")),
            ]),
            Expr::neg(Expr::mul(vec![
                Expr::Int(2.into()),
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "a", Variance::Up, None)],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                    ],
                ),
            ])),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect("contracted bianchi sum should reduce");
        assert_eq!(reduced, Expr::zero());
    }

    #[test]
    fn explicit_contracted_bianchi_sum_reduces_to_zero_right_order() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::add(vec![
            Expr::mul(vec![
                Expr::Int(2.into()),
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "a", Variance::Up, None)],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                    ],
                ),
            ]),
            Expr::neg(Expr::mul(vec![
                tensor(
                    &interner,
                    "nabla",
                    vec![idx(&interner, "b", Variance::Down, None)],
                ),
                Expr::Sym(interner.get_or_intern("R")),
            ])),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect("contracted bianchi sum should reduce");
        assert_eq!(reduced, Expr::zero());
    }

    #[test]
    fn family_mismatch_errors() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Up, Some("spacetime"))],
            ),
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "a", Variance::Down, Some("frame")),
                    idx(&interner, "b", Variance::Down, Some("spacetime")),
                ],
            ),
        ]);
        let err = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect_err("mismatched families should error");
        assert_eq!(
            err,
            ContractedBianchiError::MismatchedIndexFamilies {
                index: "a".to_string()
            }
        );
    }

    #[test]
    fn same_variance_repetition_errors() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "a", Variance::Up, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let err = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect_err("same variance contraction should error");
        assert_eq!(
            err,
            ContractedBianchiError::InvalidRepeatedIndexVariance {
                index: "a".to_string()
            }
        );
    }

    #[test]
    fn wrong_derivative_head_is_unchanged() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "D",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "a", Variance::Down, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            None,
            &properties,
            &interner,
        )
        .expect("unrelated derivative head should be unchanged");
        assert_eq!(reduced, crate::canonicalise(&expr, &properties, &interner));
    }

    #[test]
    fn nonmatching_tensor_symbol_is_unchanged() {
        let interner = ax_ir::Interner::new();
        let properties = props(&interner);
        let expr = Expr::mul(vec![
            tensor(
                &interner,
                "nabla",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
            tensor(
                &interner,
                "T",
                vec![
                    idx(&interner, "a", Variance::Down, None),
                    idx(&interner, "b", Variance::Down, None),
                ],
            ),
        ]);
        let reduced = contracted_bianchi_reduce(
            &expr,
            interner.get_or_intern("nabla"),
            interner.get_or_intern("Ric"),
            interner.get_or_intern("R"),
            Some(interner.get_or_intern("G")),
            &properties,
            &interner,
        )
        .expect("unrelated tensor should be unchanged");
        assert_eq!(reduced, crate::canonicalise(&expr, &properties, &interner));
    }
}
