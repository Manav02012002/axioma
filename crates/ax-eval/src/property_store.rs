use ax_ir::{
    DiracBarMetadata, Expr, FockSpaceMetadata, GammaConventionMetadata, GammaMatrixMetadata,
    HilbertSpaceMetadata, Index, IndexFamily, ModeMetadata, ModeStatistics, OperatorSpaceMetadata,
    QuantumObjectKind, QuantumObjectMetadata, SpinorClass, SpinorMetadata, SymmetrySource,
    TableauAttachment, TensorProperty, TensorSymmetry, TraceSpaceMetadata, Variance,
};
use num_traits::ToPrimitive;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct PropertyPattern {
    pub base_name: lasso::Spur,
    pub index_slots: Vec<SlotSpec>,
}

#[derive(Clone, Debug)]
pub struct SlotSpec {
    pub variance: Variance,
    pub family: Option<lasso::Spur>,
}

#[derive(Clone, Debug)]
pub struct PropertyAttachment {
    pub pattern: PropertyPattern,
    pub property: TensorProperty,
}

#[derive(Clone, Debug, Default)]
pub struct PropertyStore {
    attachments: Vec<PropertyAttachment>,
    inheritance_rules: Vec<InheritanceRule>,
    pub index_to_family: HashMap<lasso::Spur, lasso::Spur>,
    pub index_families: HashMap<lasso::Spur, IndexFamily>,
}

#[derive(Clone, Debug)]
pub enum InheritanceRule {
    WeightInherit {
        label: String,
        combine: WeightCombine,
    },
    DependsInherit,
    IndexInherit,
}

#[derive(Clone, Debug)]
pub enum WeightCombine {
    Additive,
    Multiplicative,
}

impl PropertyStore {
    pub fn new() -> Self {
        Self {
            attachments: Vec::new(),
            inheritance_rules: Vec::new(),
            index_to_family: HashMap::new(),
            index_families: HashMap::new(),
        }
    }

    pub fn declare(&mut self, pattern: PropertyPattern, property: TensorProperty) {
        self.attachments
            .push(PropertyAttachment { pattern, property });
    }

    pub fn declare_simple(&mut self, name: lasso::Spur, property: TensorProperty) {
        self.declare(
            PropertyPattern {
                base_name: name,
                index_slots: Vec::new(),
            },
            property,
        );
    }

    pub fn declare_with_compatibility(
        &mut self,
        pattern: PropertyPattern,
        property: TensorProperty,
    ) {
        for property in expand_compatible_properties(property) {
            self.declare(pattern.clone(), property);
        }
    }

    pub fn declare_simple_with_compatibility(
        &mut self,
        name: lasso::Spur,
        property: TensorProperty,
    ) {
        let pattern = PropertyPattern {
            base_name: name,
            index_slots: Vec::new(),
        };
        self.declare_with_compatibility(pattern, property);
    }

    pub fn declare_spinor_meta(&mut self, name: lasso::Spur, metadata: SpinorMetadata) {
        self.declare_simple_with_compatibility(name, TensorProperty::SpinorMeta(metadata));
    }

    pub fn declare_gamma_matrix_meta(&mut self, name: lasso::Spur, metadata: GammaMatrixMetadata) {
        self.declare_simple_with_compatibility(name, TensorProperty::GammaMatrixMeta(metadata));
    }

    pub fn declare_gamma_convention_meta(
        &mut self,
        name: lasso::Spur,
        metadata: GammaConventionMetadata,
    ) {
        self.declare_simple(name, TensorProperty::GammaConventionMeta(metadata));
    }

    pub fn declare_dirac_bar_meta(&mut self, name: lasso::Spur, metadata: DiracBarMetadata) {
        self.declare_simple_with_compatibility(name, TensorProperty::DiracBarMeta(metadata));
    }

    pub fn declare_trace_space(&mut self, name: lasso::Spur, metadata: TraceSpaceMetadata) {
        self.declare_simple(name, TensorProperty::TraceSpaceMeta(metadata));
    }

    /// Attach structured Hilbert-space metadata to a symbol.
    pub fn declare_hilbert_space(&mut self, name: lasso::Spur, metadata: HilbertSpaceMetadata) {
        self.declare_simple(name, TensorProperty::HilbertSpaceMeta(metadata));
    }

    /// Attach structured Fock-space metadata to a symbol.
    pub fn declare_fock_space(&mut self, name: lasso::Spur, metadata: FockSpaceMetadata) {
        self.declare_simple(name, TensorProperty::FockSpaceMeta(metadata));
    }

    /// Attach structured quantum-object metadata to a symbol and add compatible legacy markers.
    pub fn declare_quantum_object(&mut self, name: lasso::Spur, metadata: QuantumObjectMetadata) {
        self.declare_simple(name, TensorProperty::QuantumObjectMeta(metadata.clone()));
        if matches!(
            metadata.kind,
            QuantumObjectKind::Operator
                | QuantumObjectKind::DensityOperator
                | QuantumObjectKind::Projector
                | QuantumObjectKind::Observable
                | QuantumObjectKind::Channel
        ) {
            self.declare_simple(name, TensorProperty::NonCommuting);
        }
    }

    /// Attach structured operator-domain metadata to a symbol.
    pub fn declare_operator_space(&mut self, name: lasso::Spur, metadata: OperatorSpaceMetadata) {
        self.declare_simple(name, TensorProperty::OperatorSpaceMeta(metadata));
    }

    /// Attach structured mode metadata to a symbol and add compatible legacy commutation markers.
    pub fn declare_mode(&mut self, name: lasso::Spur, metadata: ModeMetadata) {
        self.declare_simple_with_compatibility(name, TensorProperty::ModeMeta(metadata));
    }

    pub fn add_inheritance(&mut self, rule: InheritanceRule) {
        self.inheritance_rules.push(rule);
    }

    pub fn set_index_to_family(&mut self, map: HashMap<lasso::Spur, lasso::Spur>) {
        self.index_to_family = map;
    }

    pub fn set_index_families(&mut self, map: HashMap<lasso::Spur, IndexFamily>) {
        self.index_families = map;
    }

    pub fn symbols(&self) -> Vec<lasso::Spur> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for attachment in &self.attachments {
            if seen.insert(attachment.pattern.base_name) {
                out.push(attachment.pattern.base_name);
            }
        }
        out
    }

    pub fn as_legacy_hashmap(&self) -> HashMap<lasso::Spur, Vec<TensorProperty>> {
        let mut out: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
        for attachment in &self.attachments {
            out.entry(attachment.pattern.base_name)
                .or_default()
                .push(attachment.property.clone());
        }
        out
    }

    pub fn get(
        &self,
        name: lasso::Spur,
        indices: &[Index],
        index_families: &HashMap<lasso::Spur, lasso::Spur>,
    ) -> Vec<&TensorProperty> {
        self.attachments
            .iter()
            .filter(|attachment| attachment.pattern.base_name == name)
            .filter(|attachment| {
                if attachment.pattern.index_slots.is_empty() {
                    return true;
                }
                if attachment.pattern.index_slots.len() != indices.len() {
                    return false;
                }
                attachment
                    .pattern
                    .index_slots
                    .iter()
                    .zip(indices.iter())
                    .all(|(slot, index)| {
                        if slot.variance != index.variance {
                            return false;
                        }
                        match slot.family {
                            None => true,
                            Some(family) => {
                                index_families.get(&index.name).copied() == Some(family)
                            }
                        }
                    })
            })
            .map(|attachment| &attachment.property)
            .collect()
    }

    pub fn get_all(&self, name: lasso::Spur) -> Vec<&TensorProperty> {
        self.attachments
            .iter()
            .filter(|attachment| attachment.pattern.base_name == name)
            .map(|attachment| &attachment.property)
            .collect()
    }

    pub fn try_get_tensor_symmetry(
        &self,
        name: lasso::Spur,
        indices: &[Index],
        index_families: &HashMap<lasso::Spur, lasso::Spur>,
    ) -> Result<Option<TensorSymmetry>, ax_ir::SymmetryValidationError> {
        let matching = self
            .get(name, indices, index_families)
            .into_iter()
            .map(|property| property.clone())
            .collect::<Vec<_>>();

        if matching.is_empty() {
            return Ok(None);
        }

        let matching = ax_tensor::structured_curvature_properties_from_legacy(&matching)
            .0
            .into_iter()
            .collect::<Vec<_>>();

        if matching.is_empty() {
            return Ok(None);
        }

        let mut merged = TensorSymmetry {
            tableaux: Vec::new(),
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };

        for symmetry in matching {
            merged.tableaux.extend(symmetry.tableaux);
            merged.inherits_under_derivative |= symmetry.inherits_under_derivative;
            merged.inherits_under_tensor_product |= symmetry.inherits_under_tensor_product;
            merged.inherits_under_contraction |= symmetry.inherits_under_contraction;
            merged.preserves_trace_free_under_projection |=
                symmetry.preserves_trace_free_under_projection;
        }

        merged.validate()?;
        Ok(Some(merged))
    }

    pub fn get_tensor_symmetry(
        &self,
        name: lasso::Spur,
        indices: &[Index],
        index_families: &HashMap<lasso::Spur, lasso::Spur>,
    ) -> Option<TensorSymmetry> {
        self.try_get_tensor_symmetry(name, indices, index_families)
            .ok()
            .flatten()
    }

    pub fn try_get_tensor_identities(
        &self,
        name: lasso::Spur,
        indices: &[Index],
        index_families: &HashMap<lasso::Spur, lasso::Spur>,
    ) -> anyhow::Result<ax_ir::TensorIdentitySet> {
        let matching = self
            .get(name, indices, index_families)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let (_, legacy_identities) =
            ax_tensor::structured_curvature_properties_from_legacy(&matching);
        let mut identities = ax_ir::TensorIdentitySet::empty();

        for property in &matching {
            if let TensorProperty::TensorIdentities(explicit) = property {
                for identity in &explicit.multiterm {
                    if !identities.multiterm.contains(identity) {
                        identities.multiterm.push(identity.clone());
                    }
                }
            }
        }
        for identity in legacy_identities.multiterm {
            if !identities.multiterm.contains(&identity) {
                identities.multiterm.push(identity);
            }
        }

        Ok(identities)
    }

    pub fn inherits_tableau(&self, name: lasso::Spur) -> bool {
        self.get_all(name)
            .into_iter()
            .any(|prop| matches!(prop, TensorProperty::TableauSymmetry(_)))
    }

    pub fn has_property(
        &self,
        name: lasso::Spur,
        indices: &[Index],
        index_families: &HashMap<lasso::Spur, lasso::Spur>,
        check: &TensorProperty,
    ) -> bool {
        self.get(name, indices, index_families)
            .into_iter()
            .any(|prop| property_discriminant_matches(prop, check))
    }

    pub fn compute_weight(
        &self,
        expr: &Expr,
        label: &str,
        explicit_weights: &HashMap<(lasso::Spur, String), i64>,
    ) -> i64 {
        let combine = self.inheritance_rules.iter().find_map(|rule| match rule {
            InheritanceRule::WeightInherit {
                label: rule_label,
                combine,
            } if rule_label == label => Some(combine),
            _ => None,
        });

        match combine {
            Some(WeightCombine::Additive) => {
                self.compute_weight_additive(expr, label, explicit_weights)
            }
            Some(WeightCombine::Multiplicative) => {
                self.compute_weight_multiplicative(expr, label, explicit_weights)
            }
            None => 0,
        }
    }

    pub fn compute_depends(
        &self,
        expr: &Expr,
        explicit_depends: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
    ) -> Vec<lasso::Spur> {
        if !self
            .inheritance_rules
            .iter()
            .any(|rule| matches!(rule, InheritanceRule::DependsInherit))
        {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.compute_depends_inner(expr, explicit_depends, &mut seen, &mut out);
        out
    }

    pub fn migrate_from_hashmap(
        old: &HashMap<lasso::Spur, Vec<TensorProperty>>,
        index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    ) -> PropertyStore {
        let mut store = PropertyStore::new();
        for (name, props) in old {
            for prop in props {
                store.declare_simple(*name, prop.clone());
            }
        }
        store.index_to_family = index_to_family.clone();
        store.add_inheritance(InheritanceRule::DependsInherit);
        store.add_inheritance(InheritanceRule::WeightInherit {
            label: "field".to_string(),
            combine: WeightCombine::Additive,
        });
        store
    }

    fn compute_weight_additive(
        &self,
        expr: &Expr,
        label: &str,
        explicit_weights: &HashMap<(lasso::Spur, String), i64>,
    ) -> i64 {
        let _ = self;
        match expr {
            Expr::Sym(s) => explicit_weights
                .get(&(*s, label.to_string()))
                .copied()
                .unwrap_or(0),
            Expr::Mul(factors) => factors
                .iter()
                .map(|factor| self.compute_weight_additive(factor, label, explicit_weights))
                .sum(),
            Expr::Add(terms) => terms
                .first()
                .map(|term| self.compute_weight_additive(term, label, explicit_weights))
                .unwrap_or(0),
            Expr::Pow(base, exp) => {
                if let Expr::Int(n) = exp.as_ref() {
                    let power = n.to_i64().unwrap_or(0);
                    self.compute_weight_additive(base, label, explicit_weights) * power
                } else {
                    0
                }
            }
            Expr::Neg(inner) => self.compute_weight_additive(inner, label, explicit_weights),
            Expr::Call(_, args) => args
                .iter()
                .map(|arg| self.compute_weight_additive(arg, label, explicit_weights))
                .sum(),
            Expr::Indexed(base, _) => self.compute_weight_additive(base, label, explicit_weights),
            _ => 0,
        }
    }

    fn compute_weight_multiplicative(
        &self,
        expr: &Expr,
        label: &str,
        explicit_weights: &HashMap<(lasso::Spur, String), i64>,
    ) -> i64 {
        let _ = self;
        match expr {
            Expr::Sym(s) => explicit_weights
                .get(&(*s, label.to_string()))
                .copied()
                .unwrap_or(1),
            Expr::Mul(factors) => factors.iter().fold(1, |acc, factor| {
                acc * self.compute_weight_multiplicative(factor, label, explicit_weights)
            }),
            Expr::Add(terms) => terms
                .first()
                .map(|term| self.compute_weight_multiplicative(term, label, explicit_weights))
                .unwrap_or(0),
            Expr::Pow(base, exp) => {
                if let Expr::Int(n) = exp.as_ref() {
                    let base_weight =
                        self.compute_weight_multiplicative(base, label, explicit_weights);
                    let power = n.to_i64().unwrap_or(0);
                    if power < 0 {
                        0
                    } else {
                        (0..power).fold(1, |acc, _| acc * base_weight)
                    }
                } else {
                    0
                }
            }
            Expr::Neg(inner) => self.compute_weight_multiplicative(inner, label, explicit_weights),
            Expr::Call(_, args) => args.iter().fold(1, |acc, arg| {
                acc * self.compute_weight_multiplicative(arg, label, explicit_weights)
            }),
            Expr::Indexed(base, _) => {
                self.compute_weight_multiplicative(base, label, explicit_weights)
            }
            _ => 0,
        }
    }

    fn compute_depends_inner(
        &self,
        expr: &Expr,
        explicit_depends: &HashMap<lasso::Spur, Vec<lasso::Spur>>,
        seen: &mut HashSet<lasso::Spur>,
        out: &mut Vec<lasso::Spur>,
    ) {
        let _ = self;
        match expr {
            Expr::Sym(s) => {
                if let Some(depends) = explicit_depends.get(s) {
                    for dep in depends {
                        if seen.insert(*dep) {
                            out.push(*dep);
                        }
                    }
                }
            }
            Expr::Mul(factors) | Expr::Add(factors) => {
                for factor in factors {
                    self.compute_depends_inner(factor, explicit_depends, seen, out);
                }
            }
            Expr::Pow(base, _) | Expr::Neg(base) => {
                self.compute_depends_inner(base, explicit_depends, seen, out);
            }
            Expr::Call(_, args) => {
                for arg in args {
                    self.compute_depends_inner(arg, explicit_depends, seen, out);
                }
            }
            Expr::Indexed(base, _) => {
                self.compute_depends_inner(base, explicit_depends, seen, out);
            }
            _ => {}
        }
    }
}

pub fn property_discriminant_matches(a: &TensorProperty, b: &TensorProperty) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

pub fn expand_compatible_properties(property: TensorProperty) -> Vec<TensorProperty> {
    let mut properties = vec![property.clone()];
    match property {
        TensorProperty::SpinorMeta(SpinorMetadata { class, .. }) => {
            properties.push(TensorProperty::Spinor);
            match class {
                SpinorClass::Dirac => {}
                SpinorClass::Majorana => properties.push(TensorProperty::MajoranaSpinor),
                SpinorClass::Weyl => properties.push(TensorProperty::WeylSpinor),
                SpinorClass::MajoranaWeyl => {
                    properties.push(TensorProperty::MajoranaSpinor);
                    properties.push(TensorProperty::WeylSpinor);
                }
            }
        }
        TensorProperty::GammaMatrixMeta(_) => {
            properties.push(TensorProperty::GammaMatrixProp);
        }
        TensorProperty::DiracBarMeta(_) => {
            properties.push(TensorProperty::DiracBar);
        }
        TensorProperty::ModeMeta(ModeMetadata { statistics, .. }) => {
            properties.push(TensorProperty::NonCommuting);
            if matches!(statistics, ModeStatistics::Fermionic) {
                properties.push(TensorProperty::AntiCommuting);
            }
        }
        _ => {}
    }
    properties
}

fn tableau_inherit_enabled(props: &[TensorProperty]) -> bool {
    props.iter().any(|prop| {
        matches!(
            prop,
            TensorProperty::TableauInherit | TensorProperty::CovariantDerivative
        )
    })
}

fn inherited_tensor_symmetry_property(
    symmetry: &TensorSymmetry,
    offset: usize,
    composite_rank: usize,
) -> Option<TensorProperty> {
    let tableaux = symmetry
        .tableaux
        .iter()
        .filter_map(|attachment| {
            let slot_map = attachment
                .slot_map
                .iter()
                .map(|slot| slot + offset)
                .collect::<Vec<_>>();
            if slot_map.iter().any(|slot| *slot >= composite_rank) {
                return None;
            }
            Some(TableauAttachment {
                shape: attachment.shape.clone(),
                slot_map,
                multiplicity_numer: attachment.multiplicity_numer,
                multiplicity_denom: attachment.multiplicity_denom,
                duality: attachment.duality.clone(),
                restricted_mode: attachment.restricted_mode.clone(),
                trace_free: attachment.trace_free,
                dimension_guard: attachment.dimension_guard.clone(),
                source: SymmetrySource::Inherited,
                label: attachment.label.clone(),
            })
        })
        .collect::<Vec<_>>();

    if tableaux.is_empty() {
        return None;
    }

    let inherited = TensorSymmetry {
        tableaux,
        inherits_under_derivative: symmetry.inherits_under_derivative,
        inherits_under_tensor_product: symmetry.inherits_under_tensor_product,
        inherits_under_contraction: symmetry.inherits_under_contraction,
        preserves_trace_free_under_projection: symmetry.preserves_trace_free_under_projection,
    };

    if inherited.validate().is_ok() {
        Some(TensorProperty::TableauSymmetry(inherited))
    } else {
        None
    }
}

fn derived_riemann_tensor_symmetry(offset: usize) -> TensorProperty {
    let mut symmetry = ax_tensor::riemann_tensor_symmetry();
    for attachment in &mut symmetry.tableaux {
        attachment.slot_map = attachment
            .slot_map
            .iter()
            .map(|slot| slot + offset)
            .collect();
    }
    TensorProperty::TableauSymmetry(symmetry)
}

fn has_explicit_riemann_tensor_properties(props: &[TensorProperty], n_indices: usize) -> bool {
    n_indices >= 4
        && props
            .iter()
            .any(|prop| matches!(prop, TensorProperty::RiemannSymmetry))
        && props
            .iter()
            .any(|prop| matches!(prop, TensorProperty::SatisfiesBianchi { .. }))
}

fn differential_bianchi_inherited_properties(
    leader_props: &[TensorProperty],
    follower_props: &[TensorProperty],
    composite_rank: usize,
) -> Vec<TensorProperty> {
    let leader_is_covariant = leader_props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::CovariantDerivative));
    if !leader_is_covariant || composite_rank < 3 {
        return Vec::new();
    }
    if !has_explicit_riemann_tensor_properties(follower_props, composite_rank.saturating_sub(1)) {
        return Vec::new();
    }
    vec![TensorProperty::SatisfiesBianchi {
        slots: vec![0, 1, 2],
    }]
}

fn shifted_tableau_inherited_properties(
    leader_props: &[TensorProperty],
    props: &[TensorProperty],
    offset: usize,
    follower_rank: usize,
) -> Vec<TensorProperty> {
    let composite_rank = offset + follower_rank;
    let mut inherited = Vec::new();

    let has_bianchi = props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::SatisfiesBianchi { .. }));

    for prop in props {
        match prop {
            TensorProperty::TableauSymmetry(symmetry) => {
                if let Some(shifted) =
                    inherited_tensor_symmetry_property(symmetry, offset, composite_rank)
                {
                    inherited.push(shifted);
                }
            }
            TensorProperty::TensorIdentities(identities) => {
                let mut shifted = ax_ir::TensorIdentitySet::empty();
                for identity in &identities.multiterm {
                    match identity {
                        ax_ir::TensorMultitermIdentity::FirstBianchi { cyclic_slots } => {
                            let slots = cyclic_slots.map(|slot| slot + offset);
                            if slots.iter().all(|slot| *slot < composite_rank) {
                                shifted.multiterm.push(
                                    ax_ir::TensorMultitermIdentity::FirstBianchi {
                                        cyclic_slots: slots,
                                    },
                                );
                            }
                        }
                        ax_ir::TensorMultitermIdentity::CyclicSum { slots } => {
                            let shifted_slots =
                                slots.iter().map(|slot| slot + offset).collect::<Vec<_>>();
                            if shifted_slots.iter().all(|slot| *slot < composite_rank) {
                                shifted
                                    .multiterm
                                    .push(ax_ir::TensorMultitermIdentity::CyclicSum {
                                        slots: shifted_slots,
                                    });
                            }
                        }
                        ax_ir::TensorMultitermIdentity::AlternatingSum { slots } => {
                            let shifted_slots =
                                slots.iter().map(|slot| slot + offset).collect::<Vec<_>>();
                            if shifted_slots.iter().all(|slot| *slot < composite_rank) {
                                shifted.multiterm.push(
                                    ax_ir::TensorMultitermIdentity::AlternatingSum {
                                        slots: shifted_slots,
                                    },
                                );
                            }
                        }
                        ax_ir::TensorMultitermIdentity::LinearCombination { terms } => {
                            shifted.multiterm.push(
                                ax_ir::TensorMultitermIdentity::LinearCombination {
                                    terms: terms.clone(),
                                },
                            );
                        }
                    }
                }
                if !shifted.is_empty() {
                    inherited.push(TensorProperty::TensorIdentities(shifted));
                }
            }
            TensorProperty::SatisfiesBianchi { slots } => {
                let shifted = slots.iter().map(|slot| slot + offset).collect::<Vec<_>>();
                if shifted.iter().all(|slot| *slot < composite_rank) {
                    inherited.push(TensorProperty::SatisfiesBianchi { slots: shifted });
                }
            }
            TensorProperty::DimensionDependentIdentity => {
                inherited.push(TensorProperty::DimensionDependentIdentity);
            }
            TensorProperty::Traceless => {
                inherited.push(TensorProperty::Traceless);
            }
            TensorProperty::WeylTensor => {
                inherited.push(TensorProperty::Traceless);
                if offset + 4 <= composite_rank {
                    let mut symmetry = ax_tensor::weyl_tensor_symmetry();
                    for attachment in &mut symmetry.tableaux {
                        attachment.slot_map = attachment
                            .slot_map
                            .iter()
                            .map(|slot| slot + offset)
                            .collect();
                    }
                    inherited.push(TensorProperty::TableauSymmetry(symmetry));
                    inherited.push(TensorProperty::TensorIdentities(ax_ir::TensorIdentitySet {
                        multiterm: vec![ax_ir::TensorMultitermIdentity::FirstBianchi {
                            cyclic_slots: [offset + 1, offset + 2, offset + 3],
                        }],
                    }));
                }
            }
            TensorProperty::RiemannSymmetry => {
                if has_bianchi && offset + 4 <= composite_rank {
                    inherited.push(derived_riemann_tensor_symmetry(offset));
                }
            }
            _ => {}
        }
    }

    inherited.extend(differential_bianchi_inherited_properties(
        leader_props,
        props,
        composite_rank,
    ));
    inherited
}

impl ax_tensor::PropertyLookup for PropertyStore {
    fn get_properties(&self, name: lasso::Spur) -> Vec<ax_ir::TensorProperty> {
        self.get_all(name).into_iter().cloned().collect()
    }

    fn get_properties_with_indices(
        &self,
        name: lasso::Spur,
        indices: &[ax_ir::Index],
        successor: Option<(lasso::Spur, &[ax_ir::Index])>,
    ) -> Vec<ax_ir::TensorProperty> {
        let mut properties = self
            .get(name, indices, &self.index_to_family)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let leader_inherits = tableau_inherit_enabled(&properties);
        if leader_inherits {
            if let Some((follower_name, follower_indices)) = successor {
                let follower =
                    self.get_properties_with_indices(follower_name, follower_indices, None);
                properties.extend(shifted_tableau_inherited_properties(
                    &properties,
                    &follower,
                    indices.len().saturating_sub(follower_indices.len()),
                    follower_indices.len(),
                ));
            }
        }
        properties
    }

    fn has_property_kind(&self, name: lasso::Spur, kind: &ax_ir::TensorProperty) -> bool {
        self.get_all(name)
            .iter()
            .any(|p| std::mem::discriminant(*p) == std::mem::discriminant(kind))
    }

    fn declared_index_slot_families(&self, name: lasso::Spur) -> Vec<Vec<Option<lasso::Spur>>> {
        self.attachments
            .iter()
            .filter(|attachment| attachment.pattern.base_name == name)
            .filter(|attachment| !attachment.pattern.index_slots.is_empty())
            .map(|attachment| {
                attachment
                    .pattern
                    .index_slots
                    .iter()
                    .map(|slot| slot.family)
                    .collect()
            })
            .collect()
    }

    fn index_families(&self) -> Option<&HashMap<lasso::Spur, ax_ir::IndexFamily>> {
        Some(&self.index_families)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{
        DualityKind, Index, RestrictedSymmetryMode, SymmetrySource, TableauAttachment,
        TensorSymmetry,
    };

    fn simple_symmetry(slot_map: Vec<usize>) -> TensorSymmetry {
        TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![2],
                slot_map,
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: DualityKind::None,
                restricted_mode: RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        }
    }

    fn down_index(name: lasso::Spur) -> Index {
        Index {
            name,
            variance: Variance::Down,
            index_type: None,
        }
    }

    #[test]
    fn single_structured_symmetry_retrieval() {
        let interner = ax_ir::Interner::new();
        let tensor = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let mut store = PropertyStore::new();
        store.declare_simple(
            tensor,
            TensorProperty::TableauSymmetry(simple_symmetry(vec![0, 1])),
        );

        let symmetry = store
            .get_tensor_symmetry(tensor, &[down_index(a), down_index(b)], &HashMap::new())
            .expect("expected symmetry");
        assert_eq!(symmetry.tableaux.len(), 1);
        assert_eq!(symmetry.tableaux[0].slot_map, vec![0, 1]);
    }

    #[test]
    fn two_matching_attachments_merge() {
        let interner = ax_ir::Interner::new();
        let tensor = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let mut store = PropertyStore::new();
        store.declare_simple(
            tensor,
            TensorProperty::TableauSymmetry(simple_symmetry(vec![0, 1])),
        );
        store.declare_simple(
            tensor,
            TensorProperty::TableauSymmetry(TensorSymmetry {
                tableaux: vec![TableauAttachment {
                    shape: vec![1, 1],
                    slot_map: vec![0, 1],
                    multiplicity_numer: 1,
                    multiplicity_denom: 1,
                    duality: DualityKind::None,
                    restricted_mode: RestrictedSymmetryMode::FullYoung,
                    trace_free: true,
                    dimension_guard: None,
                    source: SymmetrySource::Declared,
                    label: Some("extra".to_string()),
                }],
                inherits_under_derivative: true,
                inherits_under_tensor_product: false,
                inherits_under_contraction: false,
                preserves_trace_free_under_projection: true,
            }),
        );

        let symmetry = store
            .try_get_tensor_symmetry(tensor, &[down_index(a), down_index(b)], &HashMap::new())
            .expect("validation should succeed")
            .expect("expected symmetry");
        assert_eq!(symmetry.tableaux.len(), 2);
        assert!(symmetry.inherits_under_derivative);
        assert!(symmetry.preserves_trace_free_under_projection);
    }

    #[test]
    fn no_structured_symmetry_returns_none() {
        let interner = ax_ir::Interner::new();
        let tensor = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let store = PropertyStore::new();
        assert_eq!(
            store.get_tensor_symmetry(tensor, &[down_index(a), down_index(b)], &HashMap::new()),
            None
        );
    }

    #[test]
    fn inherits_tableau_requires_structured_symmetry_property() {
        let interner = ax_ir::Interner::new();
        let t_only_inherit = interner.get_or_intern("T_only_inherit");
        let t_with_symmetry = interner.get_or_intern("T_with_symmetry");

        let mut store = PropertyStore::new();
        store.declare_simple(t_only_inherit, TensorProperty::TableauInherit);
        store.declare_simple(
            t_with_symmetry,
            TensorProperty::TableauSymmetry(simple_symmetry(vec![0, 1])),
        );

        assert!(!store.inherits_tableau(t_only_inherit));
        assert!(store.inherits_tableau(t_with_symmetry));
    }
}
