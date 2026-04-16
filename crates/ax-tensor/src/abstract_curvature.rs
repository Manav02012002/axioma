use crate::PropertyLookup;
use ax_ir::{Expr, Index, Interner, Variance};
use lasso::Spur;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AbstractCurvatureReduceError {
    #[error("riemann_to_ricci requires a Ricci tensor symbol")]
    MissingRicciSymbol,
    #[error("riemann_to_ricci encountered a repeated index '{index}' with identical variance; a contraction requires one up and one down occurrence")]
    InvalidRepeatedIndexVariance { index: String },
    #[error(
        "riemann_to_ricci encountered repeated index '{index}' across different index families"
    )]
    MismatchedIndexFamilies { index: String },
}

#[derive(Clone, Debug)]
struct RepeatedOccurrence<'a> {
    slot: usize,
    index: &'a Index,
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

fn is_riemann_like(sym: Spur, indices: &[Index], properties: &dyn PropertyLookup) -> bool {
    let factor_props = properties.get_properties_with_indices(sym, indices, None);
    crate::structured_curvature_properties_from_legacy(&factor_props)
        .0
        .is_some()
}

fn validate_repeated_indices(
    indices: &[Index],
    interner: &Interner,
) -> Result<Vec<(usize, usize)>, AbstractCurvatureReduceError> {
    let mut by_name: HashMap<Spur, Vec<RepeatedOccurrence<'_>>> = HashMap::new();
    for (slot, index) in indices.iter().enumerate() {
        by_name
            .entry(index.name)
            .or_default()
            .push(RepeatedOccurrence { slot, index });
    }

    let mut pairs = Vec::new();
    for (name, occurrences) in by_name {
        if occurrences.len() < 2 {
            continue;
        }

        let first_family = occurrences[0].index.index_type;
        if occurrences
            .iter()
            .skip(1)
            .any(|occurrence| occurrence.index.index_type != first_family)
        {
            return Err(AbstractCurvatureReduceError::MismatchedIndexFamilies {
                index: interner.resolve(name).to_string(),
            });
        }

        let up_count = occurrences
            .iter()
            .filter(|occurrence| occurrence.index.variance == Variance::Up)
            .count();
        let down_count = occurrences.len().saturating_sub(up_count);
        if occurrences.len() != 2 || up_count != 1 || down_count != 1 {
            return Err(AbstractCurvatureReduceError::InvalidRepeatedIndexVariance {
                index: interner.resolve(name).to_string(),
            });
        }

        let mut pair = [occurrences[0].slot, occurrences[1].slot];
        pair.sort_unstable();
        pairs.push((pair[0], pair[1]));
    }

    pairs.sort_unstable();
    Ok(pairs)
}

fn collapse_traced_ricci_if_requested(
    expr: Expr,
    scalar_sym: Option<Spur>,
    interner: &Interner,
) -> Result<Expr, AbstractCurvatureReduceError> {
    let Some(scalar_sym) = scalar_sym else {
        return Ok(expr);
    };
    if let Expr::Neg(inner) = expr {
        let collapsed_inner =
            collapse_traced_ricci_if_requested(*inner, Some(scalar_sym), interner)?;
        return Ok(Expr::neg(collapsed_inner));
    }
    let Some((_, indices)) = indexed_symbol_and_indices(&expr) else {
        return Ok(expr);
    };
    if indices.len() != 2 {
        return Ok(expr);
    }
    let repeated = validate_repeated_indices(indices, interner)?;
    if repeated.len() == 1 && repeated[0] == (0, 1) {
        Ok(Expr::Sym(scalar_sym))
    } else {
        Ok(expr)
    }
}

fn rewrite_single_riemann_factor(
    sym: Spur,
    indices: &[Index],
    ricci_sym: Spur,
    scalar_sym: Option<Spur>,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Result<Option<Expr>, AbstractCurvatureReduceError> {
    if indices.len() != 4 || !is_riemann_like(sym, indices, properties) {
        return Ok(None);
    }

    let repeated_pairs = validate_repeated_indices(indices, interner)?;
    if repeated_pairs.is_empty() {
        return Ok(None);
    }

    let mk_ricci = |lhs: usize, rhs: usize| {
        Expr::Indexed(
            Box::new(Expr::Sym(ricci_sym)),
            vec![indices[lhs].clone(), indices[rhs].clone()],
        )
    };

    if repeated_pairs
        .iter()
        .any(|pair| matches!(pair, (0, 1) | (2, 3)))
    {
        return Ok(Some(Expr::zero()));
    }

    let first_pair = repeated_pairs[0];
    let reduced = match first_pair {
        (0, 2) => mk_ricci(1, 3),
        (0, 3) => Expr::neg(mk_ricci(1, 2)),
        (1, 2) => Expr::neg(mk_ricci(0, 3)),
        (1, 3) => mk_ricci(0, 2),
        _ => return Ok(None),
    };

    if repeated_pairs.len() == 1 {
        return Ok(Some(reduced));
    }

    let collapsed = collapse_traced_ricci_if_requested(reduced, scalar_sym, interner)?;
    Ok(Some(collapsed))
}

fn rewrite_expr(
    expr: &Expr,
    ricci_sym: Spur,
    scalar_sym: Option<Spur>,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Result<Expr, AbstractCurvatureReduceError> {
    match expr {
        Expr::Add(terms) => {
            let mut rewritten = Vec::with_capacity(terms.len());
            for term in terms {
                rewritten.push(rewrite_expr(
                    term, ricci_sym, scalar_sym, properties, interner,
                )?);
            }
            Ok(Expr::add(rewritten))
        }
        Expr::Mul(factors) => {
            let mut rewritten = Vec::with_capacity(factors.len());
            for factor in factors {
                rewritten.push(rewrite_expr(
                    factor, ricci_sym, scalar_sym, properties, interner,
                )?);
            }
            Ok(Expr::mul(rewritten))
        }
        Expr::Neg(inner) => Ok(Expr::neg(rewrite_expr(
            inner, ricci_sym, scalar_sym, properties, interner,
        )?)),
        Expr::Group(inner, rel) => Ok(Expr::Group(
            Box::new(rewrite_expr(
                inner, ricci_sym, scalar_sym, properties, interner,
            )?),
            *rel,
        )),
        Expr::Let(name, value, body) => Ok(Expr::Let(
            *name,
            Box::new(rewrite_expr(
                value, ricci_sym, scalar_sym, properties, interner,
            )?),
            Box::new(rewrite_expr(
                body, ricci_sym, scalar_sym, properties, interner,
            )?),
        )),
        Expr::List(items) => {
            let mut rewritten = Vec::with_capacity(items.len());
            for item in items {
                rewritten.push(rewrite_expr(
                    item, ricci_sym, scalar_sym, properties, interner,
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
                        item, ricci_sym, scalar_sym, properties, interner,
                    )?);
                }
                rewritten_rows.push(rewritten_row);
            }
            Ok(Expr::Matrix(rewritten_rows))
        }
        Expr::Indexed(base, indices) => {
            let base_rewritten = rewrite_expr(base, ricci_sym, scalar_sym, properties, interner)?;
            let candidate = Expr::Indexed(Box::new(base_rewritten), indices.clone());
            let Some((sym, factor_indices)) = indexed_symbol_and_indices(&candidate) else {
                return Ok(candidate);
            };
            match rewrite_single_riemann_factor(
                sym,
                factor_indices,
                ricci_sym,
                scalar_sym,
                properties,
                interner,
            )? {
                Some(rewritten) => Ok(rewritten),
                None => Ok(candidate),
            }
        }
        _ => Ok(expr.clone()),
    }
}

/// Rewrite contracted abstract Riemann-tensor factors into Ricci-tensor or scalar-curvature factors.
///
/// This function is purely abstract/index-based. It does not compute components.
/// It only rewrites contractions internal to a single indexed Riemann factor.
/// Cross-factor contractions remain the responsibility of canonicalise/tensor_reduce.
///
/// The target tensor is considered Riemann-like if it carries RiemannSymmetry or WeylTensor,
/// or if the property lookup returns both RiemannSymmetry and SatisfiesBianchi for that symbol.
///
/// `ricci_sym` is required. `scalar_sym` is optional. If `scalar_sym` is None, a doubly-contracted
/// Ricci tensor must remain as `Ric[a-,a+]` instead of collapsing to a scalar symbol.
pub fn riemann_to_ricci(
    expr: &ax_ir::Expr,
    ricci_sym: lasso::Spur,
    scalar_sym: Option<lasso::Spur>,
    properties: &dyn crate::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, AbstractCurvatureReduceError> {
    let rewritten = rewrite_expr(expr, ricci_sym, scalar_sym, properties, interner)?;
    Ok(crate::canonicalise(&rewritten, properties, interner))
}

#[cfg(test)]
mod tests {
    use super::{riemann_to_ricci, AbstractCurvatureReduceError};
    use crate::PropertyLookup;
    use ax_ir::{Expr, Index, ParentRel, TensorProperty, Variance};
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

    #[test]
    fn riemann_to_ricci_single_contractions_cover_all_ordered_patterns() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let ric = interner.get_or_intern("Ric");
        let scal = interner.get_or_intern("Scal");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };

        let cases = vec![
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "a", Variance::Up, None),
                    ],
                ),
                Expr::neg(tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                    ],
                )),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "a", Variance::Down, None),
                    ],
                ),
                Expr::neg(tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                    ],
                )),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                Expr::neg(tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                )),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                Expr::neg(tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                )),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                    ],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                    ],
                ),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                    ],
                ),
                tensor(
                    &interner,
                    "Ric",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                    ],
                ),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                Expr::zero(),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "d", Variance::Down, None),
                    ],
                ),
                Expr::zero(),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Down, None),
                        idx(&interner, "c", Variance::Up, None),
                    ],
                ),
                Expr::zero(),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "c", Variance::Up, None),
                        idx(&interner, "c", Variance::Down, None),
                    ],
                ),
                Expr::zero(),
            ),
        ];

        for (expr, expected) in cases {
            let reduced = riemann_to_ricci(&expr, ric, Some(scal), &props, &interner)
                .expect("single contraction should reduce");
            assert_eq!(reduced, expected, "unexpected reduction for {:?}", expr);
        }
    }

    #[test]
    fn riemann_to_ricci_double_contractions_cover_scalar_cases() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let ric = interner.get_or_intern("Ric");
        let scal = interner.get_or_intern("Scal");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };

        let cases = vec![
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "b", Variance::Up, None),
                    ],
                ),
                Expr::Sym(scal),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                        idx(&interner, "a", Variance::Up, None),
                    ],
                ),
                Expr::neg(Expr::Sym(scal)),
            ),
            (
                tensor(
                    &interner,
                    "R",
                    vec![
                        idx(&interner, "a", Variance::Down, None),
                        idx(&interner, "a", Variance::Up, None),
                        idx(&interner, "b", Variance::Down, None),
                        idx(&interner, "b", Variance::Up, None),
                    ],
                ),
                Expr::zero(),
            ),
        ];

        for (expr, expected) in cases {
            let reduced = riemann_to_ricci(&expr, ric, Some(scal), &props, &interner)
                .expect("double contraction should reduce");
            assert_eq!(reduced, expected, "unexpected reduction for {:?}", expr);
        }

        let no_scalar = tensor(
            &interner,
            "R",
            vec![
                idx(&interner, "a", Variance::Down, None),
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "a", Variance::Up, None),
                idx(&interner, "b", Variance::Up, None),
            ],
        );
        let explicit_trace = riemann_to_ricci(&no_scalar, ric, None, &props, &interner)
            .expect("double contraction without scalar symbol should stay traced Ricci");
        assert_eq!(
            explicit_trace,
            tensor(
                &interner,
                "Ric",
                vec![
                    idx(&interner, "b", Variance::Down, None),
                    idx(&interner, "b", Variance::Up, None),
                ],
            )
        );
    }

    #[test]
    fn riemann_to_ricci_leaves_non_riemann_and_cross_factor_contractions_unchanged() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let ric = interner.get_or_intern("Ric");
        let scal = interner.get_or_intern("Scal");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };

        let non_riemann = tensor(
            &interner,
            "T",
            vec![
                idx(&interner, "a", Variance::Down, None),
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "a", Variance::Up, None),
                idx(&interner, "d", Variance::Down, None),
            ],
        );
        assert_eq!(
            riemann_to_ricci(&non_riemann, ric, Some(scal), &props, &interner)
                .expect("non-riemann factor should pass through"),
            non_riemann
        );

        let cross_factor = Expr::mul(vec![
            tensor(
                &interner,
                "R",
                vec![
                    idx(&interner, "a", Variance::Down, None),
                    idx(&interner, "b", Variance::Down, None),
                    idx(&interner, "c", Variance::Down, None),
                    idx(&interner, "d", Variance::Down, None),
                ],
            ),
            tensor(
                &interner,
                "V",
                vec![idx(&interner, "a", Variance::Up, None)],
            ),
        ]);
        assert_eq!(
            riemann_to_ricci(&cross_factor, ric, Some(scal), &props, &interner)
                .expect("cross-factor contraction must stay untouched"),
            cross_factor
        );
    }

    #[test]
    fn riemann_to_ricci_reports_invalid_repeated_index_variance() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let ric = interner.get_or_intern("Ric");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };
        let expr = tensor(
            &interner,
            "R",
            vec![
                idx(&interner, "a", Variance::Down, None),
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "a", Variance::Down, None),
                idx(&interner, "d", Variance::Down, None),
            ],
        );

        let err = riemann_to_ricci(&expr, ric, None, &props, &interner)
            .expect_err("same-variance repeated index should be rejected");
        assert_eq!(
            err,
            AbstractCurvatureReduceError::InvalidRepeatedIndexVariance {
                index: "a".to_string(),
            }
        );
    }

    #[test]
    fn riemann_to_ricci_reports_mismatched_index_families() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let ric = interner.get_or_intern("Ric");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };
        let expr = tensor(
            &interner,
            "R",
            vec![
                idx(&interner, "a", Variance::Down, Some("spacetime")),
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "a", Variance::Up, Some("frame")),
                idx(&interner, "d", Variance::Down, None),
            ],
        );

        let err = riemann_to_ricci(&expr, ric, None, &props, &interner)
            .expect_err("same-name repeated indices from different families should error");
        assert_eq!(
            err,
            AbstractCurvatureReduceError::MismatchedIndexFamilies {
                index: "a".to_string(),
            }
        );
    }

    #[test]
    fn riemann_to_ricci_recurses_through_expression_containers() {
        let interner = ax_ir::Interner::new();
        let r = interner.get_or_intern("R");
        let x = interner.get_or_intern("x");
        let ric = interner.get_or_intern("Ric");
        let props = TestProps {
            props: HashMap::from([(r, vec![TensorProperty::RiemannSymmetry])]),
        };
        let contracted = tensor(
            &interner,
            "R",
            vec![
                idx(&interner, "a", Variance::Down, None),
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "a", Variance::Up, None),
                idx(&interner, "d", Variance::Down, None),
            ],
        );
        let expected = tensor(
            &interner,
            "Ric",
            vec![
                idx(&interner, "b", Variance::Down, None),
                idx(&interner, "d", Variance::Down, None),
            ],
        );
        let expr = Expr::Let(
            x,
            Box::new(Expr::Group(
                Box::new(contracted.clone()),
                ParentRel::ExplicitGroup,
            )),
            Box::new(Expr::List(vec![
                Expr::Neg(Box::new(contracted.clone())),
                Expr::Matrix(vec![vec![contracted.clone()]]),
                Expr::Add(vec![contracted.clone()]),
            ])),
        );

        let reduced = riemann_to_ricci(&expr, ric, None, &props, &interner)
            .expect("container recursion should succeed");
        let expected_expr = Expr::Let(
            x,
            Box::new(Expr::Group(
                Box::new(expected.clone()),
                ParentRel::ExplicitGroup,
            )),
            Box::new(Expr::List(vec![
                Expr::Neg(Box::new(expected.clone())),
                Expr::Matrix(vec![vec![expected.clone()]]),
                expected,
            ])),
        );
        assert_eq!(reduced, expected_expr);
    }
}
