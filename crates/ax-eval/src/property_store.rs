use ax_ir::{Expr, Index, IndexFamily, TensorProperty, Variance};
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

impl ax_tensor::PropertyLookup for PropertyStore {
    fn get_properties(&self, name: lasso::Spur) -> Vec<&ax_ir::TensorProperty> {
        self.get_all(name)
    }

    fn get_properties_with_indices(
        &self,
        name: lasso::Spur,
        indices: &[ax_ir::Index],
    ) -> Vec<&ax_ir::TensorProperty> {
        self.get(name, indices, &self.index_to_family)
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
