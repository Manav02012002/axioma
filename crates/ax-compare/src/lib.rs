#![forbid(unsafe_code)]

use ax_ir::{Expr, Index, IndexFamily, Interner, TensorProperty, Variance};
use ax_tensor::PropertyLookup;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::mem::Discriminant;
use std::sync::{Mutex, OnceLock};

type Sym = lasso::Spur;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchResult {
    Exact,
    EqualUpToNames,
    Less,
    Greater,
}

#[derive(Debug, Clone, Default)]
pub struct MatchMap {
    pub symbol_map: HashMap<lasso::Spur, lasso::Spur>,
    pub index_map: HashMap<lasso::Spur, lasso::Spur>,
    pub wildcard_map: HashMap<lasso::Spur, Expr>,
    pub multiplier: Option<BigRational>,
}

#[derive(Debug, Clone)]
pub struct SubproductMatch {
    pub factor_locations: Vec<usize>,
    pub factor_moving_signs: Vec<i32>,
    pub match_map: MatchMap,
}

#[derive(Debug, Clone)]
pub struct SubsumMatch {
    pub term_locations: Vec<usize>,
    pub term_ratio: BigRational,
    pub match_map: MatchMap,
}

pub type IndexSetInfo = IndexFamily;

#[derive(Debug, Clone)]
pub struct SubstitutionRule {
    pub lhs: Expr,
    pub rhs: Expr,
    pub conditions: Option<Expr>,
    pub lhs_contains_dummies: bool,
    pub rhs_contains_dummies: bool,
}

impl SubstitutionRule {
    pub fn new(lhs: Expr, rhs: Expr, conditions: Option<Expr>) -> Self {
        Self {
            lhs_contains_dummies: !dummy_pairs(&lhs).is_empty(),
            rhs_contains_dummies: !dummy_pairs(&rhs).is_empty(),
            lhs,
            rhs,
            conditions,
        }
    }
}

static SUBSTITUTION_RULE_CACHE: OnceLock<Mutex<HashMap<String, (bool, bool)>>> = OnceLock::new();

fn substitution_rule_cache() -> &'static Mutex<HashMap<String, (bool, bool)>> {
    SUBSTITUTION_RULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn substitution_rule_cache_key(rule: &SubstitutionRule) -> String {
    format!("{:?}=>{:?}|{:?}", rule.lhs, rule.rhs, rule.conditions)
}

fn validate_substitution_rule(rule: &mut SubstitutionRule) {
    let key = substitution_rule_cache_key(rule);
    if let Some((lhs, rhs)) = substitution_rule_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(&key).copied())
    {
        rule.lhs_contains_dummies = lhs;
        rule.rhs_contains_dummies = rhs;
        return;
    }

    let lhs = !dummy_pairs(&rule.lhs).is_empty();
    let rhs = !dummy_pairs(&rule.rhs).is_empty();
    rule.lhs_contains_dummies = lhs;
    rule.rhs_contains_dummies = rhs;
    if let Ok(mut cache) = substitution_rule_cache().lock() {
        cache.insert(key, (lhs, rhs));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteError {
    NoFreshDummy(Sym),
    InvalidCondition(String),
}

impl MatchMap {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn try_bind_symbol(&mut self, pattern: lasso::Spur, target: lasso::Spur) -> bool {
        match self.symbol_map.get(&pattern) {
            Some(bound) => *bound == target,
            None => {
                self.symbol_map.insert(pattern, target);
                true
            }
        }
    }

    pub fn try_bind_index(&mut self, pattern: lasso::Spur, target: lasso::Spur) -> bool {
        match self.index_map.get(&pattern) {
            Some(bound) => *bound == target,
            None => {
                self.index_map.insert(pattern, target);
                true
            }
        }
    }

    pub fn try_bind_wildcard(&mut self, slot: lasso::Spur, expr: Expr) -> bool {
        match self.wildcard_map.get(&slot) {
            Some(bound) => *bound == expr,
            None => {
                self.wildcard_map.insert(slot, expr);
                true
            }
        }
    }

    pub fn is_consistent(&self) -> bool {
        let mut symbol_targets = HashSet::new();
        for target in self.symbol_map.values() {
            if !symbol_targets.insert(*target) {
                return false;
            }
        }

        let mut index_targets = HashSet::new();
        for target in self.index_map.values() {
            if !index_targets.insert(*target) {
                return false;
            }
        }

        true
    }

    pub fn compose(&self, other: &MatchMap) -> Option<MatchMap> {
        let mut out = self.clone();

        for (key, value) in &other.symbol_map {
            if !merge_spur_binding(&mut out.symbol_map, *key, *value) {
                return None;
            }
        }
        for (key, value) in &other.index_map {
            if !merge_spur_binding(&mut out.index_map, *key, *value) {
                return None;
            }
        }
        for (key, value) in &other.wildcard_map {
            if !merge_expr_binding(&mut out.wildcard_map, *key, value.clone()) {
                return None;
            }
        }

        match (&out.multiplier, &other.multiplier) {
            (Some(lhs), Some(rhs)) if lhs != rhs => None,
            (None, Some(rhs)) => {
                out.multiplier = Some(rhs.clone());
                Some(out)
            }
            _ => Some(out),
        }
    }
}

fn merge_spur_binding(
    map: &mut HashMap<lasso::Spur, lasso::Spur>,
    key: lasso::Spur,
    value: lasso::Spur,
) -> bool {
    match map.get(&key) {
        Some(existing) => *existing == value,
        None => {
            map.insert(key, value);
            true
        }
    }
}

fn merge_expr_binding(map: &mut HashMap<lasso::Spur, Expr>, key: lasso::Spur, value: Expr) -> bool {
    match map.get(&key) {
        Some(existing) => *existing == value,
        None => {
            map.insert(key, value);
            true
        }
    }
}

fn is_legacy_wildcard(sym: Sym, interner: &Interner) -> bool {
    interner.resolve(sym).ends_with('_')
}

fn is_single_wildcard(sym: Sym, interner: &Interner) -> bool {
    let name = interner.resolve(sym);
    name.ends_with('?') && !name.ends_with("??")
}

fn is_sequence_wildcard(sym: Sym, interner: &Interner) -> bool {
    interner.resolve(sym).ends_with("??")
}

fn is_any_wildcard(sym: Sym, interner: &Interner) -> bool {
    is_legacy_wildcard(sym, interner)
        || is_single_wildcard(sym, interner)
        || is_sequence_wildcard(sym, interner)
}

fn bind_index_capture(map: &mut MatchMap, pattern_name: Sym, target_name: Sym) -> bool {
    map.try_bind_index(pattern_name, target_name)
        && map.try_bind_wildcard(pattern_name, Expr::Sym(target_name))
}

#[derive(Debug, Clone)]
pub struct CompareConfig {
    pub match_indices_by_family: bool,
    pub match_multipliers: bool,
    pub respect_commutativity: bool,
    pub allow_wildcards: bool,
    pub respect_parent_rel: bool,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            match_indices_by_family: true,
            match_multipliers: true,
            respect_commutativity: true,
            allow_wildcards: true,
            respect_parent_rel: true,
        }
    }
}

pub struct ExprComparator<'a> {
    pub config: CompareConfig,
    pub properties: &'a dyn PropertyLookup,
    pub index_to_family: &'a HashMap<lasso::Spur, lasso::Spur>,
    pub interner: &'a Interner,
}

impl<'a> ExprComparator<'a> {
    pub fn indices_compatible(&self, pattern_idx: &Index, target_idx: &Index) -> bool {
        if self.config.respect_parent_rel && pattern_idx.variance != target_idx.variance {
            return false;
        }

        if self.config.match_indices_by_family {
            self.indices_same_set(pattern_idx.name, target_idx.name)
        } else {
            pattern_idx.name == target_idx.name
        }
    }

    pub fn symbols_property_compatible(
        &self,
        pattern_sym: lasso::Spur,
        target_sym: lasso::Spur,
    ) -> bool {
        property_discriminant_set(self.properties.get_properties(pattern_sym))
            == property_discriminant_set(self.properties.get_properties(target_sym))
    }

    pub fn factors_commute(&self, a: &Expr, b: &Expr) -> bool {
        let a_props = factor_symbol(a)
            .map(|sym| self.properties.get_properties(sym))
            .unwrap_or_default();
        let b_props = factor_symbol(b)
            .map(|sym| self.properties.get_properties(sym))
            .unwrap_or_default();
        let noncommuting = |props: &[&TensorProperty]| {
            props.iter().any(|prop| {
                matches!(
                    prop,
                    TensorProperty::NonCommuting | TensorProperty::AntiCommuting
                )
            })
        };
        !noncommuting(&a_props) && !noncommuting(&b_props)
    }

    pub fn factors_anticommute(&self, a: &Expr, b: &Expr) -> bool {
        let has_anticommuting = |expr: &Expr| {
            factor_symbol(expr)
                .map(|sym| {
                    self.properties
                        .get_properties(sym)
                        .iter()
                        .any(|prop| matches!(prop, TensorProperty::AntiCommuting))
                })
                .unwrap_or(false)
        };
        has_anticommuting(a) && has_anticommuting(b)
    }

    pub fn indices_same_set(&self, a: lasso::Spur, b: lasso::Spur) -> bool {
        match (self.index_to_family.get(&a), self.index_to_family.get(&b)) {
            (Some(a_family), Some(b_family)) => a_family == b_family,
            _ => true,
        }
    }

    pub fn subtree_compare(&self, pattern: &Expr, target: &Expr, map: &mut MatchMap) -> bool {
        if self.config.allow_wildcards {
            if let Expr::Sym(slot) = pattern {
                if is_any_wildcard(*slot, self.interner) {
                    return map.try_bind_wildcard(*slot, target.clone());
                }
            }
        }

        if self.config.match_multipliers {
            let (pattern_coeff, pattern_core) = decompose_multiplier(pattern);
            let (target_coeff, target_core) = decompose_multiplier(target);
            if pattern_core != *pattern || target_core != *target {
                if pattern_coeff == BigRational::from_integer(BigInt::from(0)) {
                    return false;
                }
                let mut trial = map.clone();
                if !self.subtree_compare_no_multiplier(&pattern_core, &target_core, &mut trial) {
                    return false;
                }
                let ratio = target_coeff / pattern_coeff;
                if !bind_multiplier(&mut trial, ratio) {
                    return false;
                }
                *map = trial;
                return true;
            }
        }

        self.subtree_compare_no_multiplier(pattern, target, map)
    }

    pub fn match_with_dummies(&self, pattern: &Expr, target: &Expr, map: &mut MatchMap) -> bool {
        let pattern_dummies = dummy_pairs(pattern);
        let target_dummies = dummy_pairs(target);
        if pattern_dummies.len() != target_dummies.len() {
            return false;
        }

        let pattern_free = free_index_occurrences(pattern);
        let target_free = free_index_occurrences(target);
        if pattern_free.len() != target_free.len() {
            return false;
        }
        for (pattern_idx, target_idx) in pattern_free.iter().zip(&target_free) {
            if !self.indices_compatible(pattern_idx, target_idx) {
                return false;
            }
        }

        let mut direct = map.clone();
        if self.subtree_compare(pattern, target, &mut direct)
            && self.verify_dummy_bindings(pattern, target, &direct)
        {
            *map = direct;
            return true;
        }

        let pattern_dummy_names: Vec<lasso::Spur> =
            pattern_dummies.iter().map(|(name, _)| *name).collect();
        let target_dummy_names: Vec<lasso::Spur> =
            target_dummies.iter().map(|(name, _)| *name).collect();
        let mut used = vec![false; target_dummy_names.len()];
        let mut rebinding = map.clone();
        self.try_match_with_dummy_bindings(
            pattern,
            target,
            &pattern_dummy_names,
            &target_dummy_names,
            0,
            &mut used,
            &mut rebinding,
            map,
        )
    }

    fn subtree_compare_no_multiplier(
        &self,
        pattern: &Expr,
        target: &Expr,
        map: &mut MatchMap,
    ) -> bool {
        match (pattern, target) {
            (Expr::Int(a), Expr::Int(b)) => a == b,
            (Expr::Rational(a), Expr::Rational(b)) => a == b,
            (Expr::Float(a), Expr::Float(b)) => a == b,
            (Expr::Sym(pattern_sym), Expr::Sym(target_sym)) => {
                if self.config.allow_wildcards && is_any_wildcard(*pattern_sym, self.interner) {
                    return map.try_bind_wildcard(*pattern_sym, Expr::Sym(*target_sym));
                }
                pattern_sym == target_sym && map.try_bind_symbol(*pattern_sym, *target_sym)
            }
            (Expr::Add(pattern_terms), Expr::Add(target_terms)) => {
                self.match_commutative_exprs(pattern_terms, target_terms, map)
            }
            (Expr::Mul(pattern_factors), Expr::Mul(target_factors)) => {
                self.match_mul(pattern_factors, target_factors, map)
            }
            (Expr::Pow(pattern_base, pattern_exp), Expr::Pow(target_base, target_exp)) => {
                self.subtree_compare(pattern_base, target_base, map)
                    && self.subtree_compare(pattern_exp, target_exp, map)
            }
            (Expr::Neg(pattern_inner), Expr::Neg(target_inner)) => {
                self.subtree_compare(pattern_inner, target_inner, map)
            }
            (Expr::Call(pattern_fn, pattern_args), Expr::Call(target_fn, target_args)) => {
                if pattern_args.len() != target_args.len() || pattern_fn != target_fn {
                    return false;
                }
                for (pattern_arg, target_arg) in pattern_args.iter().zip(target_args) {
                    if !self.subtree_compare(pattern_arg, target_arg, map) {
                        return false;
                    }
                }
                true
            }
            (
                Expr::Indexed(pattern_base, pattern_indices),
                Expr::Indexed(target_base, target_indices),
            ) => {
                if pattern_indices.len() != target_indices.len()
                    || !self.subtree_compare(pattern_base, target_base, map)
                {
                    return false;
                }
                for (pattern_idx, target_idx) in pattern_indices.iter().zip(target_indices) {
                    if !self.indices_compatible(pattern_idx, target_idx) {
                        return false;
                    }
                    if is_any_wildcard(pattern_idx.name, self.interner) {
                        if !bind_index_capture(map, pattern_idx.name, target_idx.name) {
                            return false;
                        }
                    } else if !map.try_bind_index(pattern_idx.name, target_idx.name) {
                        return false;
                    }
                }
                true
            }
            (Expr::Matrix(pattern_rows), Expr::Matrix(target_rows)) => {
                if pattern_rows.len() != target_rows.len() {
                    return false;
                }
                for (pattern_row, target_row) in pattern_rows.iter().zip(target_rows) {
                    if pattern_row.len() != target_row.len() {
                        return false;
                    }
                    for (pattern_cell, target_cell) in pattern_row.iter().zip(target_row) {
                        if !self.subtree_compare(pattern_cell, target_cell, map) {
                            return false;
                        }
                    }
                }
                true
            }
            (Expr::List(pattern_items), Expr::List(target_items)) => {
                if pattern_items.len() != target_items.len() {
                    return false;
                }
                for (pattern_item, target_item) in pattern_items.iter().zip(target_items) {
                    if !self.subtree_compare(pattern_item, target_item, map) {
                        return false;
                    }
                }
                true
            }
            (Expr::Complex(pattern_re, pattern_im), Expr::Complex(target_re, target_im)) => {
                self.subtree_compare(pattern_re, target_re, map)
                    && self.subtree_compare(pattern_im, target_im, map)
            }
            (
                Expr::Let(pattern_name, pattern_value, pattern_body),
                Expr::Let(target_name, target_value, target_body),
            ) => {
                pattern_name == target_name
                    && map.try_bind_symbol(*pattern_name, *target_name)
                    && self.subtree_compare(pattern_value, target_value, map)
                    && self.subtree_compare(pattern_body, target_body, map)
            }
            (Expr::Piecewise(pattern_branches), Expr::Piecewise(target_branches)) => {
                if pattern_branches.len() != target_branches.len() {
                    return false;
                }
                for ((pattern_value, _), (target_value, _)) in
                    pattern_branches.iter().zip(target_branches)
                {
                    if !self.subtree_compare(pattern_value, target_value, map) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn match_mul(
        &self,
        pattern_factors: &[Expr],
        target_factors: &[Expr],
        map: &mut MatchMap,
    ) -> bool {
        if pattern_factors.len() != target_factors.len() {
            return false;
        }

        if !self.config.respect_commutativity {
            return self.match_ordered_exprs(pattern_factors, target_factors, map);
        }

        let pattern_groups = self.split_commutativity(pattern_factors);
        let target_groups = self.split_commutativity(target_factors);
        if pattern_groups.len() != target_groups.len() {
            return false;
        }

        let mut trial = map.clone();
        for (pattern_group, target_group) in pattern_groups.iter().zip(&target_groups) {
            if pattern_group.kind != target_group.kind
                || pattern_group.factors.len() != target_group.factors.len()
            {
                return false;
            }

            match pattern_group.kind {
                FactorKind::NonCommuting => {
                    if !self.match_ordered_exprs(
                        &pattern_group.factors,
                        &target_group.factors,
                        &mut trial,
                    ) {
                        return false;
                    }
                }
                FactorKind::Commuting => {
                    if !self.match_commutative_exprs(
                        &pattern_group.factors,
                        &target_group.factors,
                        &mut trial,
                    ) {
                        return false;
                    }
                }
                FactorKind::AntiCommuting => {
                    if !self.match_anticommuting_exprs(
                        &pattern_group.factors,
                        &target_group.factors,
                        &mut trial,
                    ) {
                        return false;
                    }
                }
            }
        }

        *map = trial;
        true
    }

    fn match_ordered_exprs(&self, patterns: &[Expr], targets: &[Expr], map: &mut MatchMap) -> bool {
        if patterns.len() != targets.len() {
            return false;
        }
        for (pattern, target) in patterns.iter().zip(targets) {
            if !self.subtree_compare(pattern, target, map) {
                return false;
            }
        }
        true
    }

    fn match_commutative_exprs(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        map: &mut MatchMap,
    ) -> bool {
        if patterns.len() != targets.len() {
            return false;
        }

        if patterns.len() > 8 {
            let mut sorted_patterns = patterns.to_vec();
            let mut sorted_targets = targets.to_vec();
            sorted_patterns.sort_by(compare_expr_debug);
            sorted_targets.sort_by(compare_expr_debug);
            return self.match_ordered_exprs(&sorted_patterns, &sorted_targets, map);
        }

        self.try_match_unordered(patterns, targets, map)
    }

    fn match_anticommuting_exprs(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        map: &mut MatchMap,
    ) -> bool {
        if patterns.len() != targets.len() {
            return false;
        }

        if patterns.len() > 8 {
            let mut sorted_patterns = patterns.to_vec();
            let mut sorted_targets = targets.to_vec();
            sorted_patterns.sort_by(compare_expr_debug);
            sorted_targets.sort_by(compare_expr_debug);
            let sign = permutation_sign_to_sorted(targets);
            let mut trial = map.clone();
            if !self.match_ordered_exprs(&sorted_patterns, &sorted_targets, &mut trial) {
                return false;
            }
            if sign < 0 && !bind_multiplier(&mut trial, BigRational::from_integer((-1).into())) {
                return false;
            }
            *map = trial;
            return true;
        }

        let mut used = vec![false; targets.len()];
        let mut target_order = Vec::with_capacity(targets.len());
        let mut trial = map.clone();
        if self.try_match_anticommuting_inner(
            patterns,
            targets,
            0,
            &mut used,
            &mut target_order,
            &mut trial,
        ) {
            *map = trial;
            true
        } else {
            false
        }
    }

    fn try_match_unordered(&self, patterns: &[Expr], targets: &[Expr], map: &mut MatchMap) -> bool {
        let mut used = vec![false; targets.len()];
        let mut trial = map.clone();
        if self.try_match_unordered_inner(patterns, targets, 0, &mut used, &mut trial) {
            *map = trial;
            true
        } else {
            false
        }
    }

    fn try_match_unordered_inner(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        pos: usize,
        used: &mut [bool],
        map: &mut MatchMap,
    ) -> bool {
        if pos == patterns.len() {
            return true;
        }
        for target_pos in 0..targets.len() {
            if used[target_pos] {
                continue;
            }
            let mut candidate = map.clone();
            if self.subtree_compare(&patterns[pos], &targets[target_pos], &mut candidate) {
                used[target_pos] = true;
                if self.try_match_unordered_inner(patterns, targets, pos + 1, used, &mut candidate)
                {
                    *map = candidate;
                    return true;
                }
                used[target_pos] = false;
            }
        }
        false
    }

    fn try_match_anticommuting_inner(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        pos: usize,
        used: &mut [bool],
        target_order: &mut Vec<usize>,
        map: &mut MatchMap,
    ) -> bool {
        if pos == patterns.len() {
            let sign = permutation_sign(target_order);
            return sign >= 0 || bind_multiplier(map, BigRational::from_integer((-1).into()));
        }
        for target_pos in 0..targets.len() {
            if used[target_pos] {
                continue;
            }
            let mut candidate = map.clone();
            if self.subtree_compare(&patterns[pos], &targets[target_pos], &mut candidate) {
                used[target_pos] = true;
                target_order.push(target_pos);
                if self.try_match_anticommuting_inner(
                    patterns,
                    targets,
                    pos + 1,
                    used,
                    target_order,
                    &mut candidate,
                ) {
                    *map = candidate;
                    return true;
                }
                target_order.pop();
                used[target_pos] = false;
            }
        }
        false
    }

    fn split_commutativity(&self, factors: &[Expr]) -> Vec<FactorGroup> {
        let mut groups: Vec<FactorGroup> = Vec::new();
        for factor in factors {
            let kind = if self.factors_anticommute(factor, factor) {
                FactorKind::AntiCommuting
            } else if groups.last().is_some_and(|last| {
                last.factors
                    .iter()
                    .all(|prev| self.factors_commute(prev, factor))
            }) {
                FactorKind::Commuting
            } else if factor_kind(factor, self.properties) == FactorKind::Commuting {
                FactorKind::Commuting
            } else {
                FactorKind::NonCommuting
            };

            if let Some(last) = groups.last_mut() {
                let can_join = match (last.kind, kind) {
                    (FactorKind::Commuting, FactorKind::Commuting) => last
                        .factors
                        .iter()
                        .all(|prev| self.factors_commute(prev, factor)),
                    (FactorKind::AntiCommuting, FactorKind::AntiCommuting) => last
                        .factors
                        .iter()
                        .all(|prev| self.factors_anticommute(prev, factor)),
                    (FactorKind::NonCommuting, FactorKind::NonCommuting) => false,
                    _ => false,
                };
                if can_join {
                    last.factors.push(factor.clone());
                    continue;
                }
            }

            groups.push(FactorGroup {
                kind,
                factors: vec![factor.clone()],
            });
        }
        groups
    }

    fn try_match_with_dummy_bindings(
        &self,
        pattern: &Expr,
        target: &Expr,
        pattern_dummy_names: &[lasso::Spur],
        target_dummy_names: &[lasso::Spur],
        pos: usize,
        used: &mut [bool],
        current: &mut MatchMap,
        out: &mut MatchMap,
    ) -> bool {
        if pos == pattern_dummy_names.len() {
            let mut candidate = current.clone();
            if self.subtree_compare(pattern, target, &mut candidate)
                && self.verify_dummy_bindings(pattern, target, &candidate)
            {
                *out = candidate;
                return true;
            }
            return false;
        }

        let pattern_name = pattern_dummy_names[pos];
        for target_pos in 0..target_dummy_names.len() {
            if used[target_pos] {
                continue;
            }
            let target_name = target_dummy_names[target_pos];
            if !self.index_names_family_compatible(pattern_name, target_name) {
                continue;
            }
            let mut candidate = current.clone();
            if !candidate.try_bind_index(pattern_name, target_name) {
                continue;
            }
            used[target_pos] = true;
            if self.try_match_with_dummy_bindings(
                pattern,
                target,
                pattern_dummy_names,
                target_dummy_names,
                pos + 1,
                used,
                &mut candidate,
                out,
            ) {
                return true;
            }
            used[target_pos] = false;
        }

        false
    }

    fn verify_dummy_bindings(&self, pattern: &Expr, target: &Expr, map: &MatchMap) -> bool {
        let pattern_structure = contraction_structure(pattern);
        let target_structure = contraction_structure(target);
        if pattern_structure.len() != target_structure.len() {
            return false;
        }

        if pattern_structure.is_empty() {
            let target_dummy_names: HashSet<lasso::Spur> = dummy_pairs(target)
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            return dummy_pairs(pattern).into_iter().all(|(pattern_name, _)| {
                map.index_map
                    .get(&pattern_name)
                    .is_some_and(|target_name| target_dummy_names.contains(target_name))
            });
        }

        let target_keys: HashSet<(usize, usize, lasso::Spur)> = target_structure
            .into_iter()
            .map(|(a, b, name)| canonical_contraction_key(a, b, name))
            .collect();

        pattern_structure.into_iter().all(|(a, b, pattern_name)| {
            map.index_map
                .get(&pattern_name)
                .map(|target_name| canonical_contraction_key(a, b, *target_name))
                .is_some_and(|key| target_keys.contains(&key))
        })
    }

    fn index_names_family_compatible(
        &self,
        pattern_name: lasso::Spur,
        target_name: lasso::Spur,
    ) -> bool {
        if !self.config.match_indices_by_family {
            return pattern_name == target_name;
        }

        match (
            self.index_to_family.get(&pattern_name),
            self.index_to_family.get(&target_name),
        ) {
            (Some(pattern_family), Some(target_family)) => pattern_family == target_family,
            _ => true,
        }
    }
}

pub fn collect_indices(expr: &Expr) -> Vec<(lasso::Spur, Variance)> {
    let mut out = Vec::new();
    collect_indices_inner(expr, &mut out);
    out
}

fn collect_indices_inner(expr: &Expr, out: &mut Vec<(lasso::Spur, Variance)>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_indices_inner(base, out);
            out.extend(indices.iter().map(|idx| (idx.name, idx.variance.clone())));
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_indices_inner(term, out);
            }
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) => {
            collect_indices_inner(base, out);
            collect_indices_inner(exp, out);
        }
        Expr::Neg(inner) => collect_indices_inner(inner, out),
        Expr::Group(inner, _) => collect_indices_inner(inner, out),
        Expr::Call(_, args) => {
            for arg in args {
                collect_indices_inner(arg, out);
            }
        }
        Expr::Rule(lhs, rhs, _) => {
            collect_indices_inner(lhs, out);
            collect_indices_inner(rhs, out);
        }
        Expr::Piecewise(branches) => {
            for (value, _) in branches {
                collect_indices_inner(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_indices_inner(value, out);
            collect_indices_inner(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_indices_inner(cell, out);
                }
            }
        }
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::FnDef(_, _, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => {}
    }
}

pub fn classify_indices(expr: &Expr) -> (Vec<lasso::Spur>, Vec<lasso::Spur>) {
    let collected = collect_indices(expr);
    let mut by_name: HashMap<lasso::Spur, Vec<Variance>> = HashMap::new();
    let mut order = Vec::new();
    for (name, variance) in collected {
        if !by_name.contains_key(&name) {
            order.push(name);
        }
        by_name.entry(name).or_default().push(variance);
    }

    let mut free = Vec::new();
    let mut dummy = Vec::new();
    for name in order {
        let variances = &by_name[&name];
        if variances.len() == 1 {
            free.push(name);
        } else if variances.len() == 2
            && variances
                .iter()
                .any(|variance| matches!(variance, Variance::Up))
            && variances
                .iter()
                .any(|variance| matches!(variance, Variance::Down))
        {
            dummy.push(name);
        } else {
            free.push(name);
        }
    }

    (free, dummy)
}

pub fn dummy_pairs(expr: &Expr) -> Vec<(lasso::Spur, lasso::Spur)> {
    let collected = collect_indices(expr);
    let mut by_name: HashMap<lasso::Spur, Vec<Variance>> = HashMap::new();
    let mut order = Vec::new();
    for (name, variance) in collected {
        if !by_name.contains_key(&name) {
            order.push(name);
        }
        by_name.entry(name).or_default().push(variance);
    }

    order
        .into_iter()
        .filter(|name| {
            let variances = &by_name[name];
            variances.len() == 2
                && variances
                    .iter()
                    .any(|variance| matches!(variance, Variance::Up))
                && variances
                    .iter()
                    .any(|variance| matches!(variance, Variance::Down))
        })
        .map(|name| (name, name))
        .collect()
}

pub fn contraction_structure(expr: &Expr) -> Vec<(usize, usize, lasso::Spur)> {
    let Expr::Mul(factors) = expr else {
        return Vec::new();
    };

    let mut by_name: HashMap<lasso::Spur, Vec<(usize, Variance)>> = HashMap::new();
    for (factor_pos, factor) in factors.iter().enumerate() {
        for (name, variance) in collect_indices(factor) {
            by_name
                .entry(name)
                .or_default()
                .push((factor_pos, variance));
        }
    }

    let mut out = Vec::new();
    for (name, occurrences) in by_name {
        if occurrences.len() == 2
            && occurrences
                .iter()
                .any(|(_, variance)| matches!(variance, Variance::Up))
            && occurrences
                .iter()
                .any(|(_, variance)| matches!(variance, Variance::Down))
        {
            out.push((occurrences[0].0, occurrences[1].0, name));
        }
    }
    out.sort_by_key(|(a, b, name)| canonical_contraction_key(*a, *b, *name));
    out
}

fn free_index_occurrences(expr: &Expr) -> Vec<Index> {
    let (free_names, _) = classify_indices(expr);
    let free_set: HashSet<lasso::Spur> = free_names.into_iter().collect();
    let mut out = Vec::new();
    collect_free_index_occurrences_inner(expr, &free_set, &mut out);
    out
}

fn collect_free_index_occurrences_inner(
    expr: &Expr,
    free_names: &HashSet<lasso::Spur>,
    out: &mut Vec<Index>,
) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_free_index_occurrences_inner(base, free_names, out);
            out.extend(
                indices
                    .iter()
                    .filter(|idx| free_names.contains(&idx.name))
                    .cloned(),
            );
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_free_index_occurrences_inner(term, free_names, out);
            }
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) => {
            collect_free_index_occurrences_inner(base, free_names, out);
            collect_free_index_occurrences_inner(exp, free_names, out);
        }
        Expr::Neg(inner) => collect_free_index_occurrences_inner(inner, free_names, out),
        Expr::Group(inner, _) => collect_free_index_occurrences_inner(inner, free_names, out),
        Expr::Call(_, args) => {
            for arg in args {
                collect_free_index_occurrences_inner(arg, free_names, out);
            }
        }
        Expr::Rule(lhs, rhs, _) => {
            collect_free_index_occurrences_inner(lhs, free_names, out);
            collect_free_index_occurrences_inner(rhs, free_names, out);
        }
        Expr::Piecewise(branches) => {
            for (value, _) in branches {
                collect_free_index_occurrences_inner(value, free_names, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_free_index_occurrences_inner(value, free_names, out);
            collect_free_index_occurrences_inner(body, free_names, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_free_index_occurrences_inner(cell, free_names, out);
                }
            }
        }
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::FnDef(_, _, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => {}
    }
}

fn canonical_contraction_key(
    factor_a: usize,
    factor_b: usize,
    index_name: lasso::Spur,
) -> (usize, usize, lasso::Spur) {
    if factor_a <= factor_b {
        (factor_a, factor_b, index_name)
    } else {
        (factor_b, factor_a, index_name)
    }
}

fn decompose_multiplier(expr: &Expr) -> (BigRational, Expr) {
    match expr {
        Expr::Int(n) => (BigRational::from_integer(n.clone()), Expr::one()),
        Expr::Rational(r) => (r.clone(), Expr::one()),
        Expr::Neg(inner) => {
            let (coeff, core) = decompose_multiplier(inner);
            (-coeff, core)
        }
        Expr::Mul(factors) => {
            let mut coeff = BigRational::one();
            let mut start = 0usize;
            while let Some(factor) = factors.get(start) {
                match factor {
                    Expr::Int(n) => {
                        coeff *= BigRational::from_integer(n.clone());
                        start += 1;
                    }
                    Expr::Rational(r) => {
                        coeff *= r.clone();
                        start += 1;
                    }
                    Expr::Neg(inner) if start == 0 => {
                        let mut rest = factors.clone();
                        rest[0] = inner.as_ref().clone();
                        let (inner_coeff, core) = decompose_multiplier(&Expr::mul(rest));
                        return (-inner_coeff, core);
                    }
                    _ => break,
                }
            }
            let rest = factors[start..].to_vec();
            let core = match rest.len() {
                0 => Expr::one(),
                1 => rest.into_iter().next().unwrap(),
                _ => Expr::mul(rest),
            };
            (coeff, core)
        }
        _ => (BigRational::one(), expr.clone()),
    }
}

fn bind_multiplier(map: &mut MatchMap, ratio: BigRational) -> bool {
    match &map.multiplier {
        Some(existing) => *existing == ratio,
        None => {
            map.multiplier = Some(ratio);
            true
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FactorKind {
    Commuting,
    NonCommuting,
    AntiCommuting,
}

#[derive(Debug, Clone)]
struct FactorGroup {
    kind: FactorKind,
    factors: Vec<Expr>,
}

fn factor_kind(expr: &Expr, properties: &dyn PropertyLookup) -> FactorKind {
    let Some(name) = factor_symbol(expr) else {
        return FactorKind::Commuting;
    };
    let props = properties.get_properties(name);
    if props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::AntiCommuting))
    {
        FactorKind::AntiCommuting
    } else if props
        .iter()
        .any(|prop| matches!(prop, TensorProperty::NonCommuting))
    {
        FactorKind::NonCommuting
    } else {
        FactorKind::Commuting
    }
}

fn factor_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Indexed(base, _) => match base.as_ref() {
            Expr::Sym(sym) => Some(*sym),
            _ => None,
        },
        Expr::Call(sym, _) => Some(*sym),
        Expr::Neg(inner) => factor_symbol(inner),
        _ => None,
    }
}

fn compare_expr_debug(lhs: &Expr, rhs: &Expr) -> Ordering {
    format!("{lhs:?}").cmp(&format!("{rhs:?}"))
}

fn permutation_sign(order: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            if order[i] > order[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

fn permutation_sign_to_sorted(items: &[Expr]) -> i32 {
    let mut indexed: Vec<(usize, Expr)> = items.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, lhs), (_, rhs)| compare_expr_debug(lhs, rhs));
    let order: Vec<usize> = indexed.into_iter().map(|(idx, _)| idx).collect();
    permutation_sign(&order)
}

fn property_discriminant_set(props: Vec<&TensorProperty>) -> HashSet<Discriminant<TensorProperty>> {
    props.into_iter().map(std::mem::discriminant).collect()
}

pub fn pattern_match(
    pattern: &Expr,
    target: &Expr,
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Option<MatchMap> {
    pattern_match_with_config(
        pattern,
        target,
        CompareConfig::default(),
        properties,
        index_to_family,
        interner,
    )
}

pub fn pattern_match_with_config(
    pattern: &Expr,
    target: &Expr,
    config: CompareConfig,
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Option<MatchMap> {
    let comparator = ExprComparator {
        config,
        properties,
        index_to_family,
        interner,
    };
    let mut map = MatchMap::new();
    comparator
        .match_with_dummies(pattern, target, &mut map)
        .then_some(map)
}

fn build_index_to_family(properties: &dyn PropertyLookup) -> HashMap<Sym, Sym> {
    let mut out = HashMap::new();
    if let Some(families) = properties.index_families() {
        for family in families.values() {
            out.insert(family.name, family.name);
            for value in &family.values {
                out.insert(*value, family.name);
            }
        }
    }
    out
}

fn expr_contains_wildcards(expr: &Expr, interner: &Interner) -> bool {
    match expr {
        Expr::Sym(sym) => is_any_wildcard(*sym, interner),
        Expr::Indexed(base, indices) => {
            expr_contains_wildcards(base, interner)
                || indices
                    .iter()
                    .any(|idx| is_any_wildcard(idx.name, interner))
        }
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) | Expr::Call(_, items) => {
            items.iter().any(|item| expr_contains_wildcards(item, interner))
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) => {
            expr_contains_wildcards(base, interner) || expr_contains_wildcards(exp, interner)
        }
        Expr::Neg(inner) => expr_contains_wildcards(inner, interner),
        Expr::Rule(lhs, rhs, _) => {
            expr_contains_wildcards(lhs, interner) || expr_contains_wildcards(rhs, interner)
        }
        Expr::Piecewise(branches) => branches
            .iter()
            .any(|(value, _)| expr_contains_wildcards(value, interner)),
        Expr::Let(_, value, body) => {
            expr_contains_wildcards(value, interner) || expr_contains_wildcards(body, interner)
        }
        Expr::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(|cell| expr_contains_wildcards(cell, interner)),
        _ => false,
    }
}

fn pattern_match_local(
    pattern: &Expr,
    target: &Expr,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Option<MatchMap> {
    let index_to_family = build_index_to_family(properties);
    if expr_contains_wildcards(pattern, interner) {
        let comparator = ExprComparator {
            config: CompareConfig::default(),
            properties,
            index_to_family: &index_to_family,
            interner,
        };
        let mut map = MatchMap::new();
        comparator.subtree_compare(pattern, target, &mut map).then_some(map)
    } else {
        pattern_match(pattern, target, properties, &index_to_family, interner)
    }
}

pub fn apply_match_map(template: &Expr, map: &MatchMap, interner: &Interner) -> Expr {
    let applied = apply_match_map_inner(template, map, interner);
    match &map.multiplier {
        Some(multiplier) if *multiplier != BigRational::one() => {
            Expr::mul(vec![Expr::Rational(multiplier.clone()), applied])
        }
        _ => applied,
    }
}

fn apply_match_map_inner(template: &Expr, map: &MatchMap, interner: &Interner) -> Expr {
    let _ = interner;
    match template {
        Expr::Sym(sym) => {
            if let Some(captured) = map.wildcard_map.get(sym) {
                captured.clone()
            } else if let Some(mapped) = map.symbol_map.get(sym) {
                Expr::Sym(*mapped)
            } else {
                template.clone()
            }
        }
        Expr::Indexed(base, indices) => {
            let new_base = apply_match_map_inner(base, map, interner);
            let new_indices = indices
                .iter()
                .map(|idx| Index {
                    name: map.index_map.get(&idx.name).copied().unwrap_or(idx.name),
                    variance: idx.variance.clone(),
                    index_type: idx.index_type,
                })
                .collect();
            Expr::Indexed(Box::new(new_base), new_indices)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| apply_match_map_inner(term, map, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| apply_match_map_inner(factor, map, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            apply_match_map_inner(base, map, interner),
            apply_match_map_inner(exp, map, interner),
        ),
        Expr::Neg(inner) => Expr::neg(apply_match_map_inner(inner, map, interner)),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(apply_match_map_inner(inner, map, interner)), *rel)
        }
        Expr::Call(f, args) => Expr::Call(
            map.symbol_map.get(f).copied().unwrap_or(*f),
            args.iter()
                .map(|arg| apply_match_map_inner(arg, map, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(apply_match_map_inner(re, map, interner)),
            Box::new(apply_match_map_inner(im, map, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(apply_match_map_inner(lhs, map, interner)),
            Box::new(apply_match_map_inner(rhs, map, interner)),
            *trust,
        ),
        Expr::Piecewise(branches) => Expr::Piecewise(
            branches
                .iter()
                .map(|(value, condition)| {
                    (
                        apply_match_map_inner(value, map, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            map.symbol_map.get(name).copied().unwrap_or(*name),
            Box::new(apply_match_map_inner(value, map, interner)),
            Box::new(apply_match_map_inner(body, map, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| apply_match_map_inner(item, map, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| apply_match_map_inner(cell, map, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            map.symbol_map.get(name).copied().unwrap_or(*name),
            params
                .iter()
                .map(|param| map.symbol_map.get(param).copied().unwrap_or(*param))
                .collect(),
            Box::new(apply_match_map_inner(body, map, interner)),
        ),
        Expr::Assume(sym, assumptions) => Expr::Assume(
            map.symbol_map.get(sym).copied().unwrap_or(*sym),
            assumptions.clone(),
        ),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Import(_)
        | Expr::SetConvention(_, _) => template.clone(),
    }
}

fn match_expr_with_conditions(
    pattern: &Expr,
    target: &Expr,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    conditions: Option<&Expr>,
) -> Option<MatchMap> {
    let map = pattern_match_local(pattern, target, properties, interner)?;
    if let Some(cond) = conditions {
        satisfies_conditions(cond, &map, interner).ok().filter(|ok| *ok)?;
    }
    Some(map)
}

fn expr_vec_from_product(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Mul(factors) => factors.clone(),
        other => vec![other.clone()],
    }
}

fn expr_vec_from_sum(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Add(terms) => terms.clone(),
        other => vec![other.clone()],
    }
}

fn product_expr(items: Vec<Expr>) -> Expr {
    Expr::mul(items)
}

fn sum_expr(items: Vec<Expr>) -> Expr {
    Expr::add(items)
}

fn sequence_wildcard_binding(
    wildcard: Sym,
    targets: &[Expr],
    used: &[bool],
    additive: bool,
) -> Expr {
    let captured: Vec<Expr> = targets
        .iter()
        .enumerate()
        .filter(|(idx, _)| !used[*idx])
        .map(|(_, expr)| expr.clone())
        .collect();
    let _ = wildcard;
    if additive {
        sum_expr(captured)
    } else {
        product_expr(captured)
    }
}

fn compute_moving_signs(
    factors: &[Expr],
    locations: &[usize],
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Option<Vec<i32>> {
    if locations.is_empty() {
        return Some(Vec::new());
    }
    let anchor = *locations.iter().min()?;
    let mut current_positions: Vec<usize> = (0..factors.len()).collect();
    let mut signs = Vec::with_capacity(locations.len());
    for (cluster_idx, &loc) in locations.iter().enumerate() {
        let current_idx = current_positions.iter().position(|pos| *pos == loc)?;
        let desired_idx = anchor + cluster_idx;
        if desired_idx > current_idx {
            signs.push(1);
            continue;
        }
        let mut sign = 1;
        for crossed_idx in desired_idx..current_idx {
            let crossed_pos = current_positions[crossed_idx];
            let swap = ax_tensor::can_swap(
                &factors[loc],
                &factors[crossed_pos],
                ax_tensor::subtree_compare(&factors[loc], &factors[crossed_pos], properties, interner),
                properties,
                interner,
                false,
            );
            if swap == 0 {
                return None;
            }
            sign *= swap;
        }
        let moved = current_positions.remove(current_idx);
        current_positions.insert(desired_idx, moved);
        signs.push(sign);
    }
    Some(signs)
}

fn match_subproduct_inner(
    pattern_factors: &[Expr],
    target_factors: &[Expr],
    properties: &dyn PropertyLookup,
    interner: &Interner,
    conditions: Option<&Expr>,
    pos: usize,
    used: &mut [bool],
    chosen: &mut Vec<usize>,
    map: &mut MatchMap,
) -> bool {
    if pos == pattern_factors.len() {
        return conditions
            .map(|cond| satisfies_conditions(cond, map, interner).unwrap_or(false))
            .unwrap_or(true);
    }

    if let Expr::Sym(wild) = &pattern_factors[pos] {
        if is_sequence_wildcard(*wild, interner) {
            let mut candidate = map.clone();
            let binding = sequence_wildcard_binding(*wild, target_factors, used, false);
            if !candidate.try_bind_wildcard(*wild, binding) {
                return false;
            }
            if match_subproduct_inner(
                pattern_factors,
                target_factors,
                properties,
                interner,
                conditions,
                pos + 1,
                used,
                chosen,
                &mut candidate,
            ) {
                *map = candidate;
                return true;
            }
        }
    }

    for target_pos in 0..target_factors.len() {
        if used[target_pos] {
            continue;
        }
        let mut candidate = map.clone();
        let Some(term_map) = match_expr_with_conditions(
            &pattern_factors[pos],
            &target_factors[target_pos],
            properties,
            interner,
            None,
        ) else {
            continue;
        };
        let Some(composed) = candidate.compose(&term_map) else {
            continue;
        };
        candidate = composed;
        used[target_pos] = true;
        chosen.push(target_pos);
        if match_subproduct_inner(
            pattern_factors,
            target_factors,
            properties,
            interner,
            conditions,
            pos + 1,
            used,
            chosen,
            &mut candidate,
        ) {
            *map = candidate;
            return true;
        }
        chosen.pop();
        used[target_pos] = false;
    }
    false
}

pub fn match_subproduct(
    pattern_factors: &[Expr],
    target: &Expr,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    conditions: Option<&Expr>,
) -> Option<SubproductMatch> {
    let target_factors = expr_vec_from_product(target);
    if pattern_factors.is_empty() || pattern_factors.len() > target_factors.len() + 1 {
        return None;
    }
    let mut used = vec![false; target_factors.len()];
    let mut chosen = Vec::new();
    let mut map = MatchMap::new();
    match_subproduct_inner(
        pattern_factors,
        &target_factors,
        properties,
        interner,
        conditions,
        0,
        &mut used,
        &mut chosen,
        &mut map,
    )
    .then_some(())?;
    let moving = compute_moving_signs(&target_factors, &chosen, properties, interner)?;
    Some(SubproductMatch {
        factor_locations: chosen,
        factor_moving_signs: moving,
        match_map: map,
    })
}

fn match_subsum_inner(
    pattern_terms: &[Expr],
    target_terms: &[Expr],
    properties: &dyn PropertyLookup,
    interner: &Interner,
    conditions: Option<&Expr>,
    pos: usize,
    used: &mut [bool],
    chosen: &mut Vec<usize>,
    map: &mut MatchMap,
    ratio: &mut Option<BigRational>,
) -> bool {
    if pos == pattern_terms.len() {
        return conditions
            .map(|cond| satisfies_conditions(cond, map, interner).unwrap_or(false))
            .unwrap_or(true);
    }
    for target_pos in 0..target_terms.len() {
        if used[target_pos] {
            continue;
        }
        let (pattern_coeff, pattern_core) = decompose_multiplier(&pattern_terms[pos]);
        let (target_coeff, target_core) = decompose_multiplier(&target_terms[target_pos]);
        if pattern_coeff.is_zero() {
            continue;
        }
        let local_ratio = target_coeff / pattern_coeff;
        if ratio.as_ref().is_some_and(|existing| existing != &local_ratio) {
            continue;
        }
        let mut candidate = map.clone();
        let Some(term_map) =
            match_expr_with_conditions(&pattern_core, &target_core, properties, interner, None)
        else {
            continue;
        };
        let Some(composed) = candidate.compose(&term_map) else {
            continue;
        };
        candidate = composed;
        let previous_ratio = ratio.clone();
        *ratio = Some(local_ratio);
        used[target_pos] = true;
        chosen.push(target_pos);
        if match_subsum_inner(
            pattern_terms,
            target_terms,
            properties,
            interner,
            conditions,
            pos + 1,
            used,
            chosen,
            &mut candidate,
            ratio,
        ) {
            *map = candidate;
            return true;
        }
        chosen.pop();
        used[target_pos] = false;
        *ratio = previous_ratio;
    }
    false
}

pub fn match_subsum(
    pattern_terms: &[Expr],
    target: &Expr,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    conditions: Option<&Expr>,
) -> Option<SubsumMatch> {
    let target_terms = expr_vec_from_sum(target);
    if pattern_terms.is_empty() || pattern_terms.len() > target_terms.len() {
        return None;
    }
    let mut used = vec![false; target_terms.len()];
    let mut chosen = Vec::new();
    let mut map = MatchMap::new();
    let mut ratio = None;
    match_subsum_inner(
        pattern_terms,
        &target_terms,
        properties,
        interner,
        conditions,
        0,
        &mut used,
        &mut chosen,
        &mut map,
        &mut ratio,
    )
    .then_some(())?;
    Some(SubsumMatch {
        term_locations: chosen,
        term_ratio: ratio.unwrap_or_else(BigRational::one),
        match_map: map,
    })
}

fn evaluate_condition_expr(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Neg(inner) => evaluate_condition_expr(inner).map(std::ops::Neg::neg),
        _ => None,
    }
}

pub fn satisfies_conditions(
    conditions: &Expr,
    match_map: &MatchMap,
    interner: &Interner,
) -> Result<bool, String> {
    let resolved = apply_match_map(conditions, match_map, interner);
    match resolved {
        Expr::Call(f, args) => {
            let name = interner.resolve(f);
            match (name, args.as_slice()) {
                ("unequals", [lhs, rhs]) => Ok(lhs != rhs),
                ("equals", [lhs, rhs]) => Ok(lhs == rhs),
                ("greater", [lhs, rhs]) => {
                    let Some(lhs) = evaluate_condition_expr(lhs) else {
                        return Err("greater requires numeric lhs".into());
                    };
                    let Some(rhs) = evaluate_condition_expr(rhs) else {
                        return Err("greater requires numeric rhs".into());
                    };
                    Ok(lhs > rhs)
                }
                ("less", [lhs, rhs]) => {
                    let Some(lhs) = evaluate_condition_expr(lhs) else {
                        return Err("less requires numeric lhs".into());
                    };
                    let Some(rhs) = evaluate_condition_expr(rhs) else {
                        return Err("less requires numeric rhs".into());
                    };
                    Ok(lhs < rhs)
                }
                ("and", [lhs, rhs]) => Ok(
                    satisfies_conditions(lhs, match_map, interner)?
                        && satisfies_conditions(rhs, match_map, interner)?,
                ),
                ("or", [lhs, rhs]) => Ok(
                    satisfies_conditions(lhs, match_map, interner)?
                        || satisfies_conditions(rhs, match_map, interner)?,
                ),
                ("not", [inner]) => Ok(!satisfies_conditions(inner, match_map, interner)?),
                _ => Err(format!("unsupported condition: {name}")),
            }
        }
        Expr::Piecewise(_) => Err("piecewise conditions are not supported".into()),
        Expr::Sym(sym) if interner.resolve(sym) == "true" => Ok(true),
        Expr::Sym(sym) if interner.resolve(sym) == "false" => Ok(false),
        Expr::Int(n) => Ok(!n.is_zero()),
        Expr::Rational(r) => Ok(!r.is_zero()),
        other => Err(format!("unsupported condition expression: {other:?}")),
    }
}

fn collect_index_names(expr: &Expr, out: &mut HashSet<Sym>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_index_names(base, out);
            out.extend(indices.iter().map(|idx| idx.name));
        }
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) | Expr::Call(_, items) => {
            for item in items {
                collect_index_names(item, out);
            }
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) => {
            collect_index_names(base, out);
            collect_index_names(exp, out);
        }
        Expr::Neg(inner) => collect_index_names(inner, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_index_names(lhs, out);
            collect_index_names(rhs, out);
        }
        Expr::Piecewise(branches) => {
            for (value, _) in branches {
                collect_index_names(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_index_names(value, out);
            collect_index_names(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_index_names(cell, out);
                }
            }
        }
        _ => {}
    }
}

fn replacement_dummy_info(
    expr: &Expr,
    families: Option<&HashMap<Sym, IndexFamily>>,
) -> HashMap<Sym, IndexSetInfo> {
    let mut occurrences: HashMap<Sym, Vec<Index>> = HashMap::new();
    collect_dummy_occurrences(expr, &mut occurrences);
    occurrences
        .into_iter()
        .filter_map(|(name, entries): (Sym, Vec<Index>)| {
            let has_up = entries.iter().any(|idx| matches!(idx.variance, Variance::Up));
            let has_down = entries
                .iter()
                .any(|idx| matches!(idx.variance, Variance::Down));
            if !(has_up && has_down) {
                return None;
            }
            let family_name = entries
                .iter()
                .find_map(|idx| idx.index_type)
                .or_else(|| {
                    families.and_then(|fams: &HashMap<Sym, IndexFamily>| {
                        fams.values().find_map(|fam| {
                            (fam.values.contains(&name) || fam.name == name).then_some(fam.name)
                        })
                    })
                })
                .unwrap_or(name);
            let info = families
                .and_then(|fams: &HashMap<Sym, IndexFamily>| fams.get(&family_name).cloned())
                .unwrap_or(IndexFamily {
                    name: family_name,
                    values: Vec::new(),
                    position: ax_ir::IndexPosition::Free,
                    dimension: None,
                    parent: None,
                });
            Some((name, info))
        })
        .collect()
}

fn collect_dummy_occurrences(expr: &Expr, out: &mut HashMap<Sym, Vec<Index>>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_dummy_occurrences(base, out);
            for idx in indices {
                out.entry(idx.name).or_default().push(idx.clone());
            }
        }
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) | Expr::Call(_, items) => {
            for item in items {
                collect_dummy_occurrences(item, out);
            }
        }
        Expr::Pow(base, exp) | Expr::Complex(base, exp) => {
            collect_dummy_occurrences(base, out);
            collect_dummy_occurrences(exp, out);
        }
        Expr::Neg(inner) => collect_dummy_occurrences(inner, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_dummy_occurrences(lhs, out);
            collect_dummy_occurrences(rhs, out);
        }
        Expr::Piecewise(branches) => {
            for (value, _) in branches {
                collect_dummy_occurrences(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_dummy_occurrences(value, out);
            collect_dummy_occurrences(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_dummy_occurrences(cell, out);
                }
            }
        }
        _ => {}
    }
}

fn rename_index_everywhere(expr: &Expr, from: Sym, to: Sym) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(rename_index_everywhere(base, from, to)),
            indices
                .iter()
                .map(|idx| Index {
                    name: if idx.name == from { to } else { idx.name },
                    variance: idx.variance.clone(),
                    index_type: idx.index_type,
                })
                .collect(),
        ),
        Expr::Add(items) => Expr::add(items.iter().map(|e| rename_index_everywhere(e, from, to)).collect()),
        Expr::Mul(items) => Expr::mul(items.iter().map(|e| rename_index_everywhere(e, from, to)).collect()),
        Expr::Pow(base, exp) => Expr::pow(
            rename_index_everywhere(base, from, to),
            rename_index_everywhere(exp, from, to),
        ),
        Expr::Neg(inner) => Expr::neg(rename_index_everywhere(inner, from, to)),
        Expr::Call(f, args) => Expr::Call(*f, args.iter().map(|e| rename_index_everywhere(e, from, to)).collect()),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(rename_index_everywhere(re, from, to)),
            Box::new(rename_index_everywhere(im, from, to)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(rename_index_everywhere(lhs, from, to)),
            Box::new(rename_index_everywhere(rhs, from, to)),
            *trust,
        ),
        Expr::Piecewise(branches) => Expr::Piecewise(
            branches
                .iter()
                .map(|(value, cond)| (rename_index_everywhere(value, from, to), cond.clone()))
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(rename_index_everywhere(value, from, to)),
            Box::new(rename_index_everywhere(body, from, to)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(|e| rename_index_everywhere(e, from, to)).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|e| rename_index_everywhere(e, from, to)).collect())
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn fresh_dummy(
    index_set: &IndexSetInfo,
    avoid: &[&HashSet<Sym>],
    interner: &Interner,
) -> Sym {
    let prefix = interner.resolve(index_set.name);
    for i in 0..4096 {
        let candidate = interner.get_or_intern(&format!("{prefix}_rw{i}"));
        if avoid.iter().all(|set: &&HashSet<Sym>| !set.contains(&candidate)) {
            return candidate;
        }
    }
    interner.get_or_intern(&format!("{prefix}_fresh"))
}

pub fn relabel_clashing_dummies(
    replacement: &mut Expr,
    ind_forced: &HashSet<Sym>,
    ind_dummy: &HashMap<Sym, IndexSetInfo>,
    interner: &Interner,
) -> Result<(), SubstituteError> {
    let mut used = ind_forced.clone();
    collect_index_names(replacement, &mut used);
    for (dummy, info) in ind_dummy {
        if !ind_forced.contains(dummy) {
            continue;
        }
        let fresh = fresh_dummy(info, &[&used], interner);
        if used.contains(&fresh) {
            return Err(SubstituteError::NoFreshDummy(*dummy));
        }
        *replacement = rename_index_everywhere(replacement, *dummy, fresh);
        used.insert(fresh);
    }
    Ok(())
}

fn instantiate_replacement(
    rhs: &Expr,
    map: &MatchMap,
    forced: &HashSet<Sym>,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    rhs_contains_dummies: bool,
) -> Option<Expr> {
    let mut replacement = apply_match_map(rhs, map, interner);
    if rhs_contains_dummies {
        let dummy_info = replacement_dummy_info(&replacement, properties.index_families());
        relabel_clashing_dummies(&mut replacement, forced, &dummy_info, interner).ok()?;
    }
    Some(replacement)
}

fn insert_product_replacement(
    target: &Expr,
    lhs_factors: &[Expr],
    rule: &SubstitutionRule,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    partial: bool,
    forced: &HashSet<Sym>,
) -> Option<Expr> {
    let target_factors = expr_vec_from_product(target);
    if !partial && target_factors.len() != lhs_factors.len() {
        return None;
    }
    let matched = match_subproduct(lhs_factors, target, properties, interner, rule.conditions.as_ref())?;
    if !partial && matched.factor_locations.len() != target_factors.len() {
        return None;
    }
    let mut replacement = instantiate_replacement(
        &rule.rhs,
        &matched.match_map,
        forced,
        properties,
        interner,
        rule.rhs_contains_dummies,
    )?;
    let total_sign: i32 = matched.factor_moving_signs.iter().product();
    if total_sign < 0 {
        replacement = Expr::neg(replacement);
    }
    let anchor = *matched.factor_locations.iter().min()?;
    let matched_set: HashSet<usize> = matched.factor_locations.iter().copied().collect();
    let mut out = Vec::new();
    let mut replacement_slot = Some(replacement);
    for idx in 0..target_factors.len() {
        if idx == anchor {
            match replacement_slot.take().unwrap() {
                Expr::Mul(items) => out.extend(items),
                other => out.push(other),
            }
        }
        if matched_set.contains(&idx) {
            continue;
        }
        out.push(target_factors[idx].clone());
    }
    Some(Expr::mul(out))
}

fn insert_sum_replacement(
    target: &Expr,
    lhs_terms: &[Expr],
    rule: &SubstitutionRule,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    partial: bool,
    forced: &HashSet<Sym>,
) -> Option<Expr> {
    let target_terms = expr_vec_from_sum(target);
    if !partial && target_terms.len() != lhs_terms.len() {
        return None;
    }
    let matched = match_subsum(lhs_terms, target, properties, interner, rule.conditions.as_ref())?;
    if !partial && matched.term_locations.len() != target_terms.len() {
        return None;
    }
    let mut replacement = instantiate_replacement(
        &rule.rhs,
        &matched.match_map,
        forced,
        properties,
        interner,
        rule.rhs_contains_dummies,
    )?;
    if matched.term_ratio != BigRational::one() {
        replacement = Expr::mul(vec![Expr::Rational(matched.term_ratio.clone()), replacement]);
    }
    let anchor = *matched.term_locations.iter().min()?;
    let matched_set: HashSet<usize> = matched.term_locations.iter().copied().collect();
    let mut out = Vec::new();
    let mut replacement_slot = Some(replacement);
    for idx in 0..target_terms.len() {
        if idx == anchor {
            match replacement_slot.take().unwrap() {
                Expr::Add(items) => out.extend(items),
                other => out.push(other),
            }
        }
        if matched_set.contains(&idx) {
            continue;
        }
        out.push(target_terms[idx].clone());
    }
    Some(Expr::add(out))
}

fn substitute_full_in_context(
    expr: &Expr,
    rule: &SubstitutionRule,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    partial: bool,
    forced: &HashSet<Sym>,
) -> Option<Expr> {
    match (&rule.lhs, expr) {
        (Expr::Mul(lhs_factors), Expr::Mul(_)) => {
            if let Some(done) =
                insert_product_replacement(expr, lhs_factors, rule, properties, interner, partial, forced)
            {
                return Some(done);
            }
        }
        (Expr::Add(lhs_terms), Expr::Add(_)) => {
            if let Some(done) =
                insert_sum_replacement(expr, lhs_terms, rule, properties, interner, partial, forced)
            {
                return Some(done);
            }
        }
        _ => {}
    }

    if let Some(map) =
        match_expr_with_conditions(&rule.lhs, expr, properties, interner, rule.conditions.as_ref())
    {
        return instantiate_replacement(
            &rule.rhs,
            &map,
            forced,
            properties,
            interner,
            rule.rhs_contains_dummies,
        );
    }

    let recursed = match expr {
        Expr::Add(terms) => Expr::add(
            terms.iter()
                .map(|term| substitute_full_in_context(term, rule, properties, interner, partial, forced).unwrap_or_else(|| term.clone()))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors.iter()
                .map(|factor| substitute_full_in_context(factor, rule, properties, interner, partial, forced).unwrap_or_else(|| factor.clone()))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_full_in_context(base, rule, properties, interner, partial, forced).unwrap_or_else(|| base.as_ref().clone()),
            substitute_full_in_context(exp, rule, properties, interner, partial, forced).unwrap_or_else(|| exp.as_ref().clone()),
        ),
        Expr::Neg(inner) => Expr::neg(
            substitute_full_in_context(inner, rule, properties, interner, partial, forced).unwrap_or_else(|| inner.as_ref().clone()),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| substitute_full_in_context(arg, rule, properties, interner, partial, forced).unwrap_or_else(|| arg.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_full_in_context(base, rule, properties, interner, partial, forced).unwrap_or_else(|| base.as_ref().clone())),
            indices.clone(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_full_in_context(re, rule, properties, interner, partial, forced).unwrap_or_else(|| re.as_ref().clone())),
            Box::new(substitute_full_in_context(im, rule, properties, interner, partial, forced).unwrap_or_else(|| im.as_ref().clone())),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_full_in_context(lhs, rule, properties, interner, partial, forced).unwrap_or_else(|| lhs.as_ref().clone())),
            Box::new(substitute_full_in_context(rhs, rule, properties, interner, partial, forced).unwrap_or_else(|| rhs.as_ref().clone())),
            *trust,
        ),
        Expr::Piecewise(branches) => Expr::Piecewise(
            branches
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_full_in_context(value, rule, properties, interner, partial, forced)
                            .unwrap_or_else(|| value.clone()),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_full_in_context(value, rule, properties, interner, partial, forced).unwrap_or_else(|| value.as_ref().clone())),
            Box::new(substitute_full_in_context(body, rule, properties, interner, partial, forced).unwrap_or_else(|| body.as_ref().clone())),
        ),
        Expr::List(items) => Expr::List(
            items.iter()
                .map(|item| substitute_full_in_context(item, rule, properties, interner, partial, forced).unwrap_or_else(|| item.clone()))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| substitute_full_in_context(cell, rule, properties, interner, partial, forced).unwrap_or_else(|| cell.clone()))
                        .collect()
                })
                .collect(),
        ),
        _ => return None,
    };
    (recursed != *expr).then_some(recursed)
}

pub fn substitute_full(
    expr: &Expr,
    rule: &SubstitutionRule,
    properties: &dyn PropertyLookup,
    interner: &Interner,
    partial: bool,
) -> Option<Expr> {
    let mut rule = rule.clone();
    validate_substitution_rule(&mut rule);
    let mut forced = HashSet::new();
    collect_index_names(expr, &mut forced);
    substitute_full_in_context(expr, &rule, properties, interner, partial, &forced)
}

pub fn substitute_with_compare(
    expr: &Expr,
    pattern: &Expr,
    replacement: &Expr,
    properties: &dyn PropertyLookup,
    _index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Expr {
    let rule = SubstitutionRule::new(pattern.clone(), replacement.clone(), None);
    substitute_full(expr, &rule, properties, interner, true).unwrap_or_else(|| expr.clone())
}

pub fn exprs_equal_up_to_dummies(
    a: &Expr,
    b: &Expr,
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> bool {
    let Some(ab) = pattern_match(a, b, properties, index_to_family, interner) else {
        return false;
    };
    let Some(ba) = pattern_match(b, a, properties, index_to_family, interner) else {
        return false;
    };

    maps_are_inverse(&ab.symbol_map, &ba.symbol_map)
        && maps_are_inverse(&ab.index_map, &ba.index_map)
}

fn maps_are_inverse(
    forward: &HashMap<lasso::Spur, lasso::Spur>,
    backward: &HashMap<lasso::Spur, lasso::Spur>,
) -> bool {
    forward
        .iter()
        .all(|(src, dst)| backward.get(dst).is_some_and(|back| back == src))
        && backward
            .iter()
            .all(|(src, dst)| forward.get(dst).is_some_and(|fwd| fwd == src))
}

pub fn expr_canonical_order(
    a: &Expr,
    b: &Expr,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Ordering {
    let _ = properties;
    let rank_cmp = expr_kind_rank(a).cmp(&expr_kind_rank(b));
    if rank_cmp != Ordering::Equal {
        return rank_cmp;
    }

    match (a, b) {
        (Expr::Int(lhs), Expr::Int(rhs)) => lhs.cmp(rhs),
        (Expr::Rational(lhs), Expr::Rational(rhs)) => lhs.cmp(rhs),
        (Expr::Float(lhs), Expr::Float(rhs)) => lhs.total_cmp(rhs),
        (Expr::Sym(lhs), Expr::Sym(rhs)) => interner.resolve(*lhs).cmp(interner.resolve(*rhs)),
        (Expr::Add(lhs), Expr::Add(rhs))
        | (Expr::Mul(lhs), Expr::Mul(rhs))
        | (Expr::List(lhs), Expr::List(rhs)) => compare_expr_slices(lhs, rhs, properties, interner),
        (Expr::Pow(lhs_base, lhs_exp), Expr::Pow(rhs_base, rhs_exp))
        | (Expr::Complex(lhs_base, lhs_exp), Expr::Complex(rhs_base, rhs_exp)) => {
            expr_canonical_order(lhs_base, rhs_base, properties, interner)
                .then_with(|| expr_canonical_order(lhs_exp, rhs_exp, properties, interner))
        }
        (Expr::Neg(lhs), Expr::Neg(rhs)) => expr_canonical_order(lhs, rhs, properties, interner),
        (Expr::Call(lhs_f, lhs_args), Expr::Call(rhs_f, rhs_args)) => interner
            .resolve(*lhs_f)
            .cmp(interner.resolve(*rhs_f))
            .then_with(|| compare_expr_slices(lhs_args, rhs_args, properties, interner)),
        (Expr::Indexed(lhs_base, lhs_indices), Expr::Indexed(rhs_base, rhs_indices)) => {
            expr_canonical_order(lhs_base, rhs_base, properties, interner)
                .then_with(|| compare_indices(lhs_indices, rhs_indices, interner))
        }
        (Expr::Matrix(lhs_rows), Expr::Matrix(rhs_rows)) => {
            lhs_rows.len().cmp(&rhs_rows.len()).then_with(|| {
                for (lhs_row, rhs_row) in lhs_rows.iter().zip(rhs_rows) {
                    let cmp = compare_expr_slices(lhs_row, rhs_row, properties, interner);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            })
        }
        (
            Expr::FnDef(lhs_name, lhs_params, lhs_body),
            Expr::FnDef(rhs_name, rhs_params, rhs_body),
        ) => interner
            .resolve(*lhs_name)
            .cmp(interner.resolve(*rhs_name))
            .then_with(|| compare_symbols(lhs_params, rhs_params, interner))
            .then_with(|| expr_canonical_order(lhs_body, rhs_body, properties, interner)),
        (Expr::Rule(lhs_l, lhs_r, lhs_t), Expr::Rule(rhs_l, rhs_r, rhs_t)) => format!("{lhs_t:?}")
            .cmp(&format!("{rhs_t:?}"))
            .then_with(|| expr_canonical_order(lhs_l, rhs_l, properties, interner))
            .then_with(|| expr_canonical_order(lhs_r, rhs_r, properties, interner)),
        (Expr::Import(lhs), Expr::Import(rhs)) => compare_symbols(lhs, rhs, interner),
        (Expr::Assume(lhs_sym, lhs_assumptions), Expr::Assume(rhs_sym, rhs_assumptions)) => {
            interner
                .resolve(*lhs_sym)
                .cmp(interner.resolve(*rhs_sym))
                .then_with(|| format!("{lhs_assumptions:?}").cmp(&format!("{rhs_assumptions:?}")))
        }
        (Expr::SetConvention(lhs_f, lhs_v), Expr::SetConvention(rhs_f, rhs_v)) => {
            lhs_f.cmp(rhs_f).then_with(|| lhs_v.cmp(rhs_v))
        }
        (Expr::Piecewise(lhs), Expr::Piecewise(rhs)) => lhs.len().cmp(&rhs.len()).then_with(|| {
            for ((lhs_value, lhs_condition), (rhs_value, rhs_condition)) in lhs.iter().zip(rhs) {
                let cmp = expr_canonical_order(lhs_value, rhs_value, properties, interner)
                    .then_with(|| format!("{lhs_condition:?}").cmp(&format!("{rhs_condition:?}")));
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        }),
        (Expr::Let(lhs_name, lhs_value, lhs_body), Expr::Let(rhs_name, rhs_value, rhs_body)) => {
            interner
                .resolve(*lhs_name)
                .cmp(interner.resolve(*rhs_name))
                .then_with(|| expr_canonical_order(lhs_value, rhs_value, properties, interner))
                .then_with(|| expr_canonical_order(lhs_body, rhs_body, properties, interner))
        }
        _ => Ordering::Equal,
    }
}

fn expr_kind_rank(expr: &Expr) -> u8 {
    match expr {
        Expr::Int(_) => 0,
        Expr::Rational(_) => 1,
        Expr::Float(_) => 2,
        Expr::Sym(_) => 3,
        Expr::Add(_) => 4,
        Expr::Mul(_) => 5,
        Expr::Pow(_, _) => 6,
        Expr::Call(_, _) => 7,
        Expr::Indexed(_, _) => 8,
        Expr::Neg(_) => 9,
        Expr::Complex(_, _) => 10,
        Expr::List(_) => 11,
        Expr::Matrix(_) => 12,
        Expr::FnDef(_, _, _) => 13,
        Expr::Rule(_, _, _) => 14,
        Expr::Import(_) => 15,
        Expr::Assume(_, _) => 16,
        Expr::SetConvention(_, _) => 17,
        Expr::Piecewise(_) => 18,
        Expr::Let(_, _, _) => 19,
        Expr::Group(_, _) => 20,
    }
}

fn compare_expr_slices(
    lhs: &[Expr],
    rhs: &[Expr],
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> Ordering {
    lhs.len().cmp(&rhs.len()).then_with(|| {
        for (lhs_item, rhs_item) in lhs.iter().zip(rhs) {
            let cmp = expr_canonical_order(lhs_item, rhs_item, properties, interner);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    })
}

fn compare_symbols(lhs: &[lasso::Spur], rhs: &[lasso::Spur], interner: &Interner) -> Ordering {
    lhs.len().cmp(&rhs.len()).then_with(|| {
        for (lhs_sym, rhs_sym) in lhs.iter().zip(rhs) {
            let cmp = interner.resolve(*lhs_sym).cmp(interner.resolve(*rhs_sym));
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    })
}

fn compare_indices(lhs: &[Index], rhs: &[Index], interner: &Interner) -> Ordering {
    lhs.len().cmp(&rhs.len()).then_with(|| {
        for (lhs_idx, rhs_idx) in lhs.iter().zip(rhs) {
            let cmp = interner
                .resolve(lhs_idx.name)
                .cmp(interner.resolve(rhs_idx.name))
                .then_with(|| {
                    variance_rank(&lhs_idx.variance).cmp(&variance_rank(&rhs_idx.variance))
                })
                .then_with(|| compare_index_type(lhs_idx.index_type, rhs_idx.index_type, interner));
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    })
}

fn variance_rank(variance: &Variance) -> u8 {
    match variance {
        Variance::Up => 0,
        Variance::Down => 1,
    }
}

fn compare_index_type(
    lhs: Option<lasso::Spur>,
    rhs: Option<lasso::Spur>,
    interner: &Interner,
) -> Ordering {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => interner.resolve(lhs).cmp(interner.resolve(rhs)),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::IndexPosition;

    #[derive(Default)]
    struct TestProps {
        props: HashMap<Sym, Vec<TensorProperty>>,
        families: HashMap<Sym, IndexFamily>,
    }

    impl PropertyLookup for TestProps {
        fn get_properties(&self, name: Sym) -> Vec<&TensorProperty> {
            self.props
                .get(&name)
                .map(|items| items.iter().collect())
                .unwrap_or_default()
        }

        fn get_properties_with_indices(
            &self,
            name: Sym,
            _indices: &[Index],
        ) -> Vec<&TensorProperty> {
            self.get_properties(name)
        }

        fn has_property_kind(&self, name: Sym, kind: &TensorProperty) -> bool {
            self.get_properties(name)
                .into_iter()
                .any(|prop| std::mem::discriminant(prop) == std::mem::discriminant(kind))
        }

        fn index_families(&self) -> Option<&HashMap<Sym, IndexFamily>> {
            Some(&self.families)
        }
    }

    fn idx(name: Sym, variance: Variance, index_type: Option<Sym>) -> Index {
        Index {
            name,
            variance,
            index_type,
        }
    }

    fn fam(name: Sym, values: Vec<Sym>) -> IndexFamily {
        IndexFamily {
            name,
            values,
            position: IndexPosition::Free,
            dimension: None,
            parent: None,
        }
    }

    #[test]
    fn substitute_subproduct_match() {
        let interner = Interner::new();
        let d = interner.get_or_intern("D");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let e = interner.get_or_intern("E");
        let ia = interner.get_or_intern("a");
        let fam_sym = interner.get_or_intern("Lor");

        let mut props = TestProps::default();
        props.families.insert(fam_sym, fam(fam_sym, vec![ia]));

        let rule = SubstitutionRule::new(
            Expr::mul(vec![
                Expr::Indexed(Box::new(Expr::Sym(a)), vec![idx(ia, Variance::Down, Some(fam_sym))]),
                Expr::Indexed(Box::new(Expr::Sym(b)), vec![idx(ia, Variance::Up, Some(fam_sym))]),
            ]),
            Expr::Sym(c),
            None,
        );
        let expr = Expr::mul(vec![
            Expr::Sym(d),
            Expr::Indexed(Box::new(Expr::Sym(a)), vec![idx(ia, Variance::Down, Some(fam_sym))]),
            Expr::Indexed(Box::new(Expr::Sym(b)), vec![idx(ia, Variance::Up, Some(fam_sym))]),
            Expr::Sym(e),
        ]);
        let result = substitute_full(&expr, &rule, &props, &interner, true).unwrap();
        assert_eq!(result, Expr::mul(vec![Expr::Sym(d), Expr::Sym(c), Expr::Sym(e)]));
    }

    #[test]
    fn substitute_subproduct_anticommuting_sign() {
        let interner = Interner::new();
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let phi = interner.get_or_intern("Phi");

        let mut props = TestProps::default();
        props.props.insert(psi, vec![TensorProperty::AntiCommuting]);
        props.props.insert(chi, vec![TensorProperty::AntiCommuting]);

        let rule = SubstitutionRule::new(
            Expr::mul(vec![Expr::Sym(psi), Expr::Sym(chi)]),
            Expr::Sym(phi),
            None,
        );
        let expr = Expr::mul(vec![Expr::Sym(chi), Expr::Sym(psi)]);
        let result = substitute_full(&expr, &rule, &props, &interner, true).unwrap();
        assert_eq!(result, Expr::mul(vec![Expr::Int((-1).into()), Expr::Sym(phi)]));
    }

    #[test]
    fn substitute_subsum_with_ratio() {
        let interner = Interner::new();
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let rule =
            SubstitutionRule::new(Expr::add(vec![Expr::Sym(a), Expr::Sym(b)]), Expr::Sym(c), None);
        let expr = Expr::add(vec![
            Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(a)]),
            Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(b)]),
            Expr::Sym(d),
        ]);
        let result = substitute_full(&expr, &rule, &TestProps::default(), &interner, true).unwrap();
        assert_eq!(
            result,
            Expr::add(vec![
                Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(c)]),
                Expr::Sym(d)
            ])
        );
    }

    #[test]
    fn substitute_conditional_unequals() {
        let interner = Interner::new();
        let delta = interner.get_or_intern("delta");
        let a_w = interner.get_or_intern("a?");
        let b_w = interner.get_or_intern("b?");
        let equals = interner.get_or_intern("equals");
        let unequals = interner.get_or_intern("unequals");
        let one = Expr::one();
        let d = interner.get_or_intern("d");
        let mu = interner.get_or_intern("mu");

        let neq_rule = SubstitutionRule::new(
            Expr::Indexed(
                Box::new(Expr::Sym(delta)),
                vec![idx(a_w, Variance::Up, None), idx(b_w, Variance::Down, None)],
            ),
            one.clone(),
            Some(Expr::Call(unequals, vec![Expr::Sym(a_w), Expr::Sym(b_w)])),
        );
        let eq_rule = SubstitutionRule::new(
            Expr::Indexed(
                Box::new(Expr::Sym(delta)),
                vec![idx(a_w, Variance::Up, None), idx(a_w, Variance::Down, None)],
            ),
            Expr::Sym(d),
            Some(Expr::Call(equals, vec![Expr::Sym(a_w), Expr::Sym(a_w)])),
        );
        let expr = Expr::Indexed(
            Box::new(Expr::Sym(delta)),
            vec![idx(mu, Variance::Up, None), idx(mu, Variance::Down, None)],
        );
        assert!(substitute_full(&expr, &neq_rule, &TestProps::default(), &interner, false).is_none());
        assert_eq!(
            substitute_full(&expr, &eq_rule, &TestProps::default(), &interner, false).unwrap(),
            Expr::Sym(d)
        );
    }

    #[test]
    fn substitute_dummy_relabelling() {
        let interner = Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let d = interner.get_or_intern("D");
        let ia = interner.get_or_intern("a");
        let ib = interner.get_or_intern("b");
        let fam_sym = interner.get_or_intern("Lor");

        let mut props = TestProps::default();
        props.families.insert(fam_sym, fam(fam_sym, vec![ia, ib]));

        let rule = SubstitutionRule::new(
            Expr::Indexed(Box::new(Expr::Sym(a)), vec![idx(ia, Variance::Down, Some(fam_sym))]),
            Expr::mul(vec![
                Expr::Indexed(
                    Box::new(Expr::Sym(b)),
                    vec![
                        idx(ia, Variance::Down, Some(fam_sym)),
                        idx(ib, Variance::Down, Some(fam_sym)),
                    ],
                ),
                Expr::Indexed(Box::new(Expr::Sym(c)), vec![idx(ib, Variance::Up, Some(fam_sym))]),
            ]),
            None,
        );
        let expr = Expr::mul(vec![
            Expr::Indexed(Box::new(Expr::Sym(d)), vec![idx(ib, Variance::Up, Some(fam_sym))]),
            Expr::Indexed(Box::new(Expr::Sym(a)), vec![idx(ia, Variance::Down, Some(fam_sym))]),
        ]);
        let result = substitute_full(&expr, &rule, &props, &interner, true).unwrap();
        match result {
            Expr::Mul(factors) => {
                assert_eq!(factors.len(), 3);
                assert!(matches!(&factors[1], Expr::Indexed(base, _) if **base == Expr::Sym(b)));
                assert!(matches!(&factors[2], Expr::Indexed(base, _) if **base == Expr::Sym(c)));
                let Expr::Indexed(_, b_indices) = &factors[1] else { panic!("expected B factor") };
                let Expr::Indexed(_, c_indices) = &factors[2] else { panic!("expected C factor") };
                assert_ne!(b_indices[1].name, ib);
                assert_eq!(b_indices[1].name, c_indices[0].name);
            }
            other => panic!("expected rewritten product, got {other:?}"),
        }
    }

    #[test]
    fn substitute_object_wildcard() {
        let interner = Interner::new();
        let pd = interner.get_or_intern("pd");
        let gamma = interner.get_or_intern("Gamma");
        let hold = interner.get_or_intern("A??");
        let ia = interner.get_or_intern("a");
        let ib = interner.get_or_intern("b");

        let rule = SubstitutionRule::new(
            Expr::Call(pd, vec![Expr::Sym(hold)]),
            Expr::add(vec![
                Expr::Call(pd, vec![Expr::Sym(hold)]),
                Expr::mul(vec![Expr::Sym(gamma), Expr::Sym(hold)]),
            ]),
            None,
        );
        let expr = Expr::Call(
            pd,
            vec![Expr::Indexed(
                Box::new(Expr::Sym(interner.get_or_intern("T"))),
                vec![idx(ia, Variance::Down, None), idx(ib, Variance::Down, None)],
            )],
        );
        let captured = match &expr {
            Expr::Call(_, args) => args[0].clone(),
            _ => unreachable!(),
        };
        let result = substitute_full(&expr, &rule, &TestProps::default(), &interner, false).unwrap();
        match result {
            Expr::Add(terms) => {
                assert_eq!(terms.len(), 2);
                assert_eq!(terms[0], expr);
                match &terms[1] {
                    Expr::Mul(factors) => {
                        assert_eq!(factors.len(), 2);
                        assert!(factors.contains(&Expr::Sym(gamma)));
                        assert!(factors.contains(&captured));
                    }
                    other => panic!("expected gamma times captured subtree, got {other:?}"),
                }
            }
            other => panic!("expected sum after wildcard substitution, got {other:?}"),
        }
    }

    #[test]
    fn substitute_partial_product_flag() {
        let interner = Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");
        let d = interner.get_or_intern("D");
        let rule = SubstitutionRule::new(
            Expr::mul(vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::Sym(c),
            None,
        );
        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(d)]);
        assert!(substitute_full(&expr, &rule, &TestProps::default(), &interner, false).is_none());
        assert_eq!(
            substitute_full(&expr, &rule, &TestProps::default(), &interner, true).unwrap(),
            Expr::mul(vec![Expr::Sym(c), Expr::Sym(d)])
        );
    }
}
