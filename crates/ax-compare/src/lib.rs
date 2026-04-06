#![forbid(unsafe_code)]

use ax_ir::{Expr, Index, Interner, TensorProperty, Variance};
use ax_tensor::PropertyLookup;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::mem::Discriminant;

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

fn merge_expr_binding(
    map: &mut HashMap<lasso::Spur, Expr>,
    key: lasso::Spur,
    value: Expr,
) -> bool {
    match map.get(&key) {
        Some(existing) => *existing == value,
        None => {
            map.insert(key, value);
            true
        }
    }
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
                if self.interner.resolve(*slot).ends_with('_') {
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
                self.symbols_property_compatible(*pattern_sym, *target_sym)
                    && map.try_bind_symbol(*pattern_sym, *target_sym)
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
                if pattern_args.len() != target_args.len()
                    || !map.try_bind_symbol(*pattern_fn, *target_fn)
                {
                    return false;
                }
                for (pattern_arg, target_arg) in pattern_args.iter().zip(target_args) {
                    if !self.subtree_compare(pattern_arg, target_arg, map) {
                        return false;
                    }
                }
                true
            }
            (Expr::Indexed(pattern_base, pattern_indices), Expr::Indexed(target_base, target_indices)) => {
                if pattern_indices.len() != target_indices.len()
                    || !self.subtree_compare(pattern_base, target_base, map)
                {
                    return false;
                }
                for (pattern_idx, target_idx) in pattern_indices.iter().zip(target_indices) {
                    if !self.indices_compatible(pattern_idx, target_idx)
                        || !map.try_bind_index(pattern_idx.name, target_idx.name)
                    {
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
                map.try_bind_symbol(*pattern_name, *target_name)
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

    fn match_ordered_exprs(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        map: &mut MatchMap,
    ) -> bool {
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

    fn try_match_unordered(
        &self,
        patterns: &[Expr],
        targets: &[Expr],
        map: &mut MatchMap,
    ) -> bool {
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
            return sign >= 0
                || bind_multiplier(map, BigRational::from_integer((-1).into()));
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
            } else if groups
                .last()
                .is_some_and(|last| last.factors.iter().all(|prev| self.factors_commute(prev, factor)))
            {
                FactorKind::Commuting
            } else if factor_kind(factor, self.properties) == FactorKind::Commuting {
                FactorKind::Commuting
            } else {
                FactorKind::NonCommuting
            };

            if let Some(last) = groups.last_mut() {
                let can_join = match (last.kind, kind) {
                    (FactorKind::Commuting, FactorKind::Commuting) => {
                        last.factors.iter().all(|prev| self.factors_commute(prev, factor))
                    }
                    (FactorKind::AntiCommuting, FactorKind::AntiCommuting) => {
                        last.factors.iter().all(|prev| self.factors_anticommute(prev, factor))
                    }
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
            let target_dummy_names: HashSet<lasso::Spur> =
                dummy_pairs(target).into_iter().map(|(name, _)| name).collect();
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

        pattern_structure
            .into_iter()
            .all(|(a, b, pattern_name)| {
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
            && variances.iter().any(|variance| matches!(variance, Variance::Up))
            && variances.iter().any(|variance| matches!(variance, Variance::Down))
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
                && variances.iter().any(|variance| matches!(variance, Variance::Up))
                && variances.iter().any(|variance| matches!(variance, Variance::Down))
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
            by_name.entry(name).or_default().push((factor_pos, variance));
        }
    }

    let mut out = Vec::new();
    for (name, occurrences) in by_name {
        if occurrences.len() == 2
            && occurrences.iter().any(|(_, variance)| matches!(variance, Variance::Up))
            && occurrences.iter().any(|(_, variance)| matches!(variance, Variance::Down))
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
    if inversions % 2 == 0 { 1 } else { -1 }
}

fn permutation_sign_to_sorted(items: &[Expr]) -> i32 {
    let mut indexed: Vec<(usize, Expr)> = items.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, lhs), (_, rhs)| compare_expr_debug(lhs, rhs));
    let order: Vec<usize> = indexed.into_iter().map(|(idx, _)| idx).collect();
    permutation_sign(&order)
}

fn property_discriminant_set(
    props: Vec<&TensorProperty>,
) -> HashSet<Discriminant<TensorProperty>> {
    props
        .into_iter()
        .map(std::mem::discriminant)
        .collect()
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
                    (apply_match_map_inner(value, map, interner), condition.clone())
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

pub fn substitute_with_compare(
    expr: &Expr,
    pattern: &Expr,
    replacement: &Expr,
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Expr {
    if let Some(map) = pattern_match(pattern, expr, properties, index_to_family, interner) {
        return apply_match_map(replacement, &map, interner);
    }

    match expr {
        Expr::Add(terms) => {
            if let Expr::Add(pattern_terms) = pattern {
                if let Some(rewritten) = substitute_sequence_in_terms(
                    terms,
                    pattern_terms,
                    replacement,
                    true,
                    properties,
                    index_to_family,
                    interner,
                ) {
                    return rewritten;
                }
            }
            Expr::add(
                terms
                    .iter()
                    .map(|term| {
                        substitute_with_compare(
                            term,
                            pattern,
                            replacement,
                            properties,
                            index_to_family,
                            interner,
                        )
                    })
                    .collect(),
            )
        }
        Expr::Mul(factors) => {
            if let Expr::Mul(pattern_factors) = pattern {
                if let Some(rewritten) = substitute_sequence_in_terms(
                    factors,
                    pattern_factors,
                    replacement,
                    false,
                    properties,
                    index_to_family,
                    interner,
                ) {
                    return rewritten;
                }
            }
            Expr::mul(
                factors
                    .iter()
                    .map(|factor| {
                        substitute_with_compare(
                            factor,
                            pattern,
                            replacement,
                            properties,
                            index_to_family,
                            interner,
                        )
                    })
                    .collect(),
            )
        }
        Expr::Pow(base, exp) => Expr::pow(
            substitute_with_compare(base, pattern, replacement, properties, index_to_family, interner),
            substitute_with_compare(exp, pattern, replacement, properties, index_to_family, interner),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_with_compare(
            inner,
            pattern,
            replacement,
            properties,
            index_to_family,
            interner,
        )),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| {
                    substitute_with_compare(
                        arg,
                        pattern,
                        replacement,
                        properties,
                        index_to_family,
                        interner,
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_with_compare(
                base,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
            indices.clone(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_with_compare(
                re,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(substitute_with_compare(
                im,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_with_compare(
                lhs,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(substitute_with_compare(
                rhs,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
            *trust,
        ),
        Expr::Piecewise(branches) => Expr::Piecewise(
            branches
                .iter()
                .map(|(value, condition)| {
                    (
                        substitute_with_compare(
                            value,
                            pattern,
                            replacement,
                            properties,
                            index_to_family,
                            interner,
                        ),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_with_compare(
                value,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
            Box::new(substitute_with_compare(
                body,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| {
                    substitute_with_compare(
                        item,
                        pattern,
                        replacement,
                        properties,
                        index_to_family,
                        interner,
                    )
                })
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| {
                            substitute_with_compare(
                                cell,
                                pattern,
                                replacement,
                                properties,
                                index_to_family,
                                interner,
                            )
                        })
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_with_compare(
                body,
                pattern,
                replacement,
                properties,
                index_to_family,
                interner,
            )),
        ),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => expr.clone(),
    }
}

fn substitute_sequence_in_terms(
    terms: &[Expr],
    pattern_terms: &[Expr],
    replacement: &Expr,
    additive: bool,
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
) -> Option<Expr> {
    if pattern_terms.is_empty() || pattern_terms.len() > terms.len() {
        return None;
    }

    let mut used = vec![false; terms.len()];
    let mut selected = Vec::with_capacity(pattern_terms.len());
    let mut map = MatchMap::new();
    if !match_term_subset(
        pattern_terms,
        terms,
        properties,
        index_to_family,
        interner,
        0,
        &mut used,
        &mut selected,
        &mut map,
    ) {
        return None;
    }

    let replacement = apply_match_map(replacement, &map, interner);
    let mut out = Vec::new();
    out.push(replacement);
    for (idx, term) in terms.iter().enumerate() {
        if !selected.contains(&idx) {
            out.push(term.clone());
        }
    }

    Some(if additive { Expr::add(out) } else { Expr::mul(out) })
}

fn match_term_subset(
    pattern_terms: &[Expr],
    terms: &[Expr],
    properties: &dyn PropertyLookup,
    index_to_family: &HashMap<lasso::Spur, lasso::Spur>,
    interner: &Interner,
    pos: usize,
    used: &mut [bool],
    selected: &mut Vec<usize>,
    map: &mut MatchMap,
) -> bool {
    if pos == pattern_terms.len() {
        return true;
    }
    for target_pos in 0..terms.len() {
        if used[target_pos] {
            continue;
        }
        let mut candidate = map.clone();
        if pattern_match(&pattern_terms[pos], &terms[target_pos], properties, index_to_family, interner)
            .and_then(|term_map| candidate.compose(&term_map))
            .is_some_and(|composed| {
                candidate = composed;
                true
            })
        {
            used[target_pos] = true;
            selected.push(target_pos);
            if match_term_subset(
                pattern_terms,
                terms,
                properties,
                index_to_family,
                interner,
                pos + 1,
                used,
                selected,
                &mut candidate,
            ) {
                *map = candidate;
                return true;
            }
            selected.pop();
            used[target_pos] = false;
        }
    }
    false
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
        | (Expr::List(lhs), Expr::List(rhs)) => {
            compare_expr_slices(lhs, rhs, properties, interner)
        }
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
        (Expr::Matrix(lhs_rows), Expr::Matrix(rhs_rows)) => lhs_rows
            .len()
            .cmp(&rhs_rows.len())
            .then_with(|| {
                for (lhs_row, rhs_row) in lhs_rows.iter().zip(rhs_rows) {
                    let cmp = compare_expr_slices(lhs_row, rhs_row, properties, interner);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }),
        (Expr::FnDef(lhs_name, lhs_params, lhs_body), Expr::FnDef(rhs_name, rhs_params, rhs_body)) => {
            interner
                .resolve(*lhs_name)
                .cmp(interner.resolve(*rhs_name))
                .then_with(|| compare_symbols(lhs_params, rhs_params, interner))
                .then_with(|| expr_canonical_order(lhs_body, rhs_body, properties, interner))
        }
        (Expr::Rule(lhs_l, lhs_r, lhs_t), Expr::Rule(rhs_l, rhs_r, rhs_t)) => {
            format!("{lhs_t:?}").cmp(&format!("{rhs_t:?}"))
                .then_with(|| expr_canonical_order(lhs_l, rhs_l, properties, interner))
                .then_with(|| expr_canonical_order(lhs_r, rhs_r, properties, interner))
        }
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
        (Expr::Piecewise(lhs), Expr::Piecewise(rhs)) => lhs
            .len()
            .cmp(&rhs.len())
            .then_with(|| {
                for ((lhs_value, lhs_condition), (rhs_value, rhs_condition)) in lhs.iter().zip(rhs) {
                    let cmp = expr_canonical_order(lhs_value, rhs_value, properties, interner)
                        .then_with(|| {
                            format!("{lhs_condition:?}").cmp(&format!("{rhs_condition:?}"))
                        });
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
                .then_with(|| variance_rank(&lhs_idx.variance).cmp(&variance_rank(&rhs_idx.variance)))
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
