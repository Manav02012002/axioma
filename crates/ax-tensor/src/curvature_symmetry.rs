use ax_ir::{
    riemann_identity_set, Expr, Index, SymmetrySource, TableauAttachment, TensorIdentitySet,
    TensorMultitermIdentity, TensorProperty, TensorSymmetry,
};

pub fn riemann_tensor_symmetry() -> TensorSymmetry {
    TensorSymmetry {
        tableaux: vec![TableauAttachment {
            shape: vec![2, 2],
            slot_map: vec![0, 1, 2, 3],
            multiplicity_numer: 1,
            multiplicity_denom: 1,
            duality: ax_ir::DualityKind::None,
            restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
            trace_free: false,
            dimension_guard: None,
            source: SymmetrySource::Derived,
            label: Some("riemann".into()),
        }],
        // Curvature symmetries should propagate through derivative and tensor-product
        // metadata flows; contraction inheritance is also enabled so structured
        // curvature semantics remain the primary source of truth downstream.
        inherits_under_derivative: true,
        inherits_under_tensor_product: true,
        inherits_under_contraction: true,
        preserves_trace_free_under_projection: false,
    }
}

pub fn weyl_tensor_symmetry() -> TensorSymmetry {
    let mut symmetry = riemann_tensor_symmetry();
    symmetry.tableaux[0].trace_free = true;
    symmetry.tableaux[0].label = Some("weyl".into());
    symmetry.preserves_trace_free_under_projection = true;
    symmetry
}

pub fn riemann_tensor_identities() -> TensorIdentitySet {
    riemann_identity_set()
}

pub fn weyl_tensor_identities() -> TensorIdentitySet {
    riemann_identity_set()
}

pub fn structured_curvature_properties_from_legacy(
    properties: &[TensorProperty],
) -> (Option<TensorSymmetry>, TensorIdentitySet) {
    let explicit_symmetry = merge_explicit_tableau_symmetries(properties);
    let mut identities = TensorIdentitySet::empty();
    let mut saw_explicit_identities = false;

    for property in properties {
        if let TensorProperty::TensorIdentities(explicit) = property {
            saw_explicit_identities = true;
            merge_identity_sets(&mut identities, explicit);
        }
    }

    let has_weyl = properties
        .iter()
        .any(|property| matches!(property, TensorProperty::WeylTensor));
    let has_riemann = properties
        .iter()
        .any(|property| matches!(property, TensorProperty::RiemannSymmetry));
    let has_traceless = properties
        .iter()
        .any(|property| matches!(property, TensorProperty::Traceless));

    let synthesized = if has_weyl {
        Some(weyl_tensor_symmetry())
    } else if has_riemann {
        Some(riemann_tensor_symmetry())
    } else {
        None
    }
    .map(|mut symmetry| {
        if has_traceless {
            for attachment in &mut symmetry.tableaux {
                attachment.trace_free = true;
            }
            symmetry.preserves_trace_free_under_projection = true;
        }
        symmetry
    });

    if !saw_explicit_identities {
        for property in properties {
            match property {
                TensorProperty::SatisfiesBianchi { slots } => {
                    if let Some(identity) = first_bianchi_from_legacy_slots(slots) {
                        push_identity(&mut identities, identity);
                    }
                }
                TensorProperty::WeylTensor => {
                    merge_identity_sets(&mut identities, &weyl_tensor_identities());
                }
                _ => {}
            }
        }
    } else {
        for property in properties {
            if let TensorProperty::SatisfiesBianchi { slots } = property {
                if let Some(identity) = first_bianchi_from_legacy_slots(slots) {
                    push_identity(&mut identities, identity);
                }
            }
        }
    }

    (explicit_symmetry.or(synthesized), identities)
}

pub fn first_bianchi_sum(tensor_symbol: lasso::Spur, slots: &[Index]) -> Expr {
    let terms = [
        [0usize, 1usize, 2usize, 3usize],
        [0usize, 2usize, 3usize, 1usize],
        [0usize, 3usize, 1usize, 2usize],
    ]
    .into_iter()
    .map(|ordering| {
        Expr::Indexed(
            Box::new(Expr::Sym(tensor_symbol)),
            ordering.iter().map(|slot| slots[*slot].clone()).collect(),
        )
    })
    .collect();
    Expr::add(terms)
}

fn merge_explicit_tableau_symmetries(properties: &[TensorProperty]) -> Option<TensorSymmetry> {
    let explicit = properties
        .iter()
        .filter_map(|property| match property {
            TensorProperty::TableauSymmetry(symmetry) => Some(symmetry.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if explicit.is_empty() {
        return None;
    }

    let mut merged = TensorSymmetry {
        tableaux: Vec::new(),
        inherits_under_derivative: false,
        inherits_under_tensor_product: false,
        inherits_under_contraction: false,
        preserves_trace_free_under_projection: false,
    };
    for symmetry in explicit {
        merged.tableaux.extend(symmetry.tableaux);
        merged.inherits_under_derivative |= symmetry.inherits_under_derivative;
        merged.inherits_under_tensor_product |= symmetry.inherits_under_tensor_product;
        merged.inherits_under_contraction |= symmetry.inherits_under_contraction;
        merged.preserves_trace_free_under_projection |=
            symmetry.preserves_trace_free_under_projection;
    }
    Some(merged)
}

fn first_bianchi_from_legacy_slots(slots: &[usize]) -> Option<TensorMultitermIdentity> {
    match slots {
        [a, b, c] => Some(TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [*a, *b, *c],
        }),
        [_, a, b, c] => Some(TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [*a, *b, *c],
        }),
        _ => None,
    }
}

fn merge_identity_sets(target: &mut TensorIdentitySet, source: &TensorIdentitySet) {
    for identity in &source.multiterm {
        push_identity(target, identity.clone());
    }
}

fn push_identity(target: &mut TensorIdentitySet, identity: TensorMultitermIdentity) {
    if !target.multiterm.contains(&identity) {
        target.multiterm.push(identity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{TensorMultitermIdentity, TensorProperty};
    use lasso::Key;

    fn explicit_symmetry() -> TensorSymmetry {
        TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![3, 1],
                slot_map: vec![0, 1, 2, 3],
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: Some("explicit".into()),
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        }
    }

    #[test]
    fn riemann_tensor_symmetry_is_exact() {
        let symmetry = riemann_tensor_symmetry();
        assert_eq!(symmetry.tableaux[0].shape, vec![2, 2]);
        assert_eq!(symmetry.tableaux[0].slot_map, vec![0, 1, 2, 3]);
        assert!(!symmetry.tableaux[0].trace_free);
    }

    #[test]
    fn weyl_tensor_symmetry_is_exact() {
        let symmetry = weyl_tensor_symmetry();
        assert_eq!(symmetry.tableaux[0].shape, vec![2, 2]);
        assert!(symmetry.tableaux[0].trace_free);
    }

    #[test]
    fn structured_curvature_properties_synthesize_riemann_symmetry() {
        let (symmetry, identities) =
            structured_curvature_properties_from_legacy(&[TensorProperty::RiemannSymmetry]);
        assert_eq!(symmetry, Some(riemann_tensor_symmetry()));
        assert!(identities.is_empty());
    }

    #[test]
    fn structured_curvature_properties_merge_weyl_and_bianchi() {
        let (symmetry, identities) = structured_curvature_properties_from_legacy(&[
            TensorProperty::WeylTensor,
            TensorProperty::Traceless,
            TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2, 3],
            },
        ]);
        assert!(symmetry.is_some_and(|sym| sym.tableaux[0].trace_free));
        assert!(identities
            .multiterm
            .contains(&TensorMultitermIdentity::FirstBianchi {
                cyclic_slots: [1, 2, 3]
            }));
    }

    #[test]
    fn explicit_symmetry_wins_but_legacy_bianchi_is_merged() {
        let (symmetry, identities) = structured_curvature_properties_from_legacy(&[
            TensorProperty::TableauSymmetry(explicit_symmetry()),
            TensorProperty::RiemannSymmetry,
            TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2, 3],
            },
        ]);
        assert_eq!(symmetry, Some(explicit_symmetry()));
        assert!(identities
            .multiterm
            .contains(&TensorMultitermIdentity::FirstBianchi {
                cyclic_slots: [1, 2, 3]
            }));
    }

    #[test]
    fn first_bianchi_sum_uses_exact_cyclic_order() {
        let r = lasso::Spur::try_from_usize(0).unwrap();
        let slots = (0..4)
            .map(|index| Index {
                name: lasso::Spur::try_from_usize(index + 1).unwrap(),
                variance: ax_ir::Variance::Down,
                index_type: None,
            })
            .collect::<Vec<_>>();
        let sum = first_bianchi_sum(r, &slots);
        let Expr::Add(terms) = sum else {
            panic!("expected Add");
        };
        assert_eq!(
            terms,
            vec![
                Expr::Indexed(Box::new(Expr::Sym(r)), slots.clone()),
                Expr::Indexed(
                    Box::new(Expr::Sym(r)),
                    vec![
                        slots[0].clone(),
                        slots[2].clone(),
                        slots[3].clone(),
                        slots[1].clone()
                    ]
                ),
                Expr::Indexed(
                    Box::new(Expr::Sym(r)),
                    vec![
                        slots[0].clone(),
                        slots[3].clone(),
                        slots[1].clone(),
                        slots[2].clone()
                    ]
                ),
            ]
        );
    }

    #[test]
    fn apply_first_bianchi_if_applicable_respects_identity_presence() {
        let r = lasso::Spur::try_from_usize(20).unwrap();
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(r)),
            (0..4)
                .map(|index| Index {
                    name: lasso::Spur::try_from_usize(30 + index).unwrap(),
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                })
                .collect(),
        );

        assert!(crate::apply_first_bianchi_if_applicable(&expr, &|symbol| {
            (symbol == r)
                .then_some(vec![TensorProperty::SatisfiesBianchi {
                    slots: vec![0, 1, 2, 3],
                }])
                .unwrap_or_default()
        })
        .is_some());

        assert!(crate::apply_first_bianchi_if_applicable(&expr, &|_| vec![]).is_none());
    }
}
