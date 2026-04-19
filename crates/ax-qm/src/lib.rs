#![forbid(unsafe_code)]
#![allow(
    clippy::manual_contains,
    clippy::manual_range_patterns,
    clippy::needless_range_loop,
    clippy::only_used_in_recursion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use ax_ir::{
    DiracBarMetadata, Expr, GammaMatrixMetadata, Index, SpinorClass, SpinorMetadata,
    TensorProperty, Variance,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::{HashMap, HashSet};

pub fn permutation_sector_dimension(shape: &[usize], n: usize) -> anyhow::Result<u64> {
    let diagram = ax_young::YoungDiagram::try_new(shape.to_vec())?;
    Ok(ax_young::dimension_of_representation(&diagram, n))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorKind {
    Creation,
    Annihilation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorStatistics {
    Bosonic,
    Fermionic,
}

#[derive(Clone, Debug)]
pub enum GammaEntry {
    Gamma(lasso::Spur),
    Index(usize),
    Gamma5,
    Identity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearPair {
    pub psi1: lasso::Spur,
    pub gamma_a: Vec<lasso::Spur>,
    pub psi2: lasso::Spur,
    pub psi3: lasso::Spur,
    pub gamma_b: Vec<lasso::Spur>,
    pub psi4: lasso::Spur,
    pub remaining_factors: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FierzError {
    NoBilinearPair,
    AmbiguousBilinears(usize),
    MalformedBilinear,
    AmbiguousSpinorOrder,
    SpinorOrderMismatch,
    IncompatibleSpinorMetadata,
    IncompatibleSpinorDimension,
    IncompatibleSpinorChirality,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QmLinearAlgebraError {
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch { left: usize, right: usize },
    #[error("basis index out of range: index={index}, dim={dim}")]
    BasisIndexOutOfRange { index: usize, dim: usize },
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("subsystem dimension mismatch: expected={expected}, actual={actual}")]
    SubsystemDimensionMismatch { expected: usize, actual: usize },
    #[error("invalid trace target: target={target}")]
    InvalidTraceTarget { target: char },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChannelError {
    #[error("empty Kraus set")]
    EmptyKrausSet,
    #[error("non-square Kraus operator at index {index}: rows={rows}, cols={cols}")]
    NonSquareKraus {
        index: usize,
        rows: usize,
        cols: usize,
    },
    #[error(
        "Kraus operator dimension mismatch at index {index}: expected={expected}, actual={actual}"
    )]
    KrausDimensionMismatch {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MeasurementError {
    #[error("projector dimension mismatch at index {index}: expected={expected}, actual={actual}")]
    ProjectorDimensionMismatch {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("state dimension mismatch: expected={expected}, actual={actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
    #[error("zero-probability outcome at index {index}")]
    ZeroProbabilityOutcome { index: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LindbladError {
    #[error("Hamiltonian not square: rows={rows}, cols={cols}")]
    HamiltonianNotSquare { rows: usize, cols: usize },
    #[error("state not square: rows={rows}, cols={cols}")]
    StateNotSquare { rows: usize, cols: usize },
    #[error("dimension mismatch for {which}: expected={expected}, actual={actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        which: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CompositeSpaceError {
    #[error("empty factor list")]
    EmptyFactorList,
    #[error("invalid factor index {index} for factor count {factor_count}")]
    InvalidFactorIndex { index: usize, factor_count: usize },
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("total dimension mismatch: expected={expected}, actual={actual}")]
    TotalDimensionMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialTraceTarget {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BipartiteDims {
    pub dim_a: usize,
    pub dim_b: usize,
}

fn qm_error_expr(name: &str, expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern(name), vec![expr.clone()])
}

fn property_sym(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Call(sym, _) => Some(*sym),
        Expr::Indexed(base, _) => property_sym(base),
        _ => None,
    }
}

fn expr_has_property(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    kind: &TensorProperty,
) -> bool {
    property_sym(expr)
        .map(|sym| properties.has_property_kind(sym, kind))
        .unwrap_or(false)
}

fn prop_sort_order(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<Vec<lasso::Spur>> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::SortOrder(order) => Some(order.clone()),
            _ => None,
        })
}

fn declared_spinor_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<SpinorMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::SpinorMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

fn declared_gamma_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaMatrixMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::GammaMatrixMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

fn declared_diracbar_metadata_of_symbol(
    sym: lasso::Spur,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    properties
        .get_properties(sym)
        .into_iter()
        .find_map(|prop| match prop {
            TensorProperty::DiracBarMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
}

/// Return the structured spinor metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When no structured metadata is present,
/// this synthesizes a best-effort fallback from the legacy `Spinor`,
/// `MajoranaSpinor`, and `WeylSpinor` markers to preserve existing behavior.
pub fn spinor_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<SpinorMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_spinor_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }

    let has_spinor = properties.has_property_kind(sym, &TensorProperty::Spinor)
        || properties.has_property_kind(sym, &TensorProperty::MajoranaSpinor)
        || properties.has_property_kind(sym, &TensorProperty::WeylSpinor);
    if !has_spinor {
        return None;
    }

    let class = match (
        properties.has_property_kind(sym, &TensorProperty::MajoranaSpinor),
        properties.has_property_kind(sym, &TensorProperty::WeylSpinor),
    ) {
        (true, true) => SpinorClass::MajoranaWeyl,
        (true, false) => SpinorClass::Majorana,
        (false, true) => SpinorClass::Weyl,
        (false, false) => SpinorClass::Dirac,
    };

    Some(SpinorMetadata {
        class,
        dimension: None,
        chirality: None,
        index_family: None,
    })
}

/// Return the structured gamma-matrix metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When absent, this falls back to the
/// legacy `GammaMatrixProp` marker and synthesizes empty metadata so older
/// declarations continue to work.
pub fn gamma_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaMatrixMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_gamma_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }
    properties
        .has_property_kind(sym, &TensorProperty::GammaMatrixProp)
        .then_some(GammaMatrixMetadata {
            dimension: None,
            metric_symbol: None,
            index_family: None,
            has_gamma5: false,
        })
}

/// Return the structured Dirac-bar metadata attached to an expression when available.
///
/// Structured metadata takes precedence. When absent, this falls back to the
/// legacy `DiracBar` marker and preserves the old default behavior of reversing
/// gamma chains under Dirac-bar expansion.
pub fn diracbar_metadata_of_expr(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    let sym = property_sym(expr)?;
    if let Some(metadata) = declared_diracbar_metadata_of_symbol(sym, properties) {
        return Some(metadata);
    }
    properties
        .has_property_kind(sym, &TensorProperty::DiracBar)
        .then_some(DiracBarMetadata {
            gamma_symbol: None,
            spinor_family: None,
            reverse_gamma_order: true,
        })
}

fn is_majorana_spinor_expr(expr: &Expr, properties: &dyn ax_tensor::PropertyLookup) -> bool {
    spinor_metadata_of_expr(expr, properties)
        .map(|metadata| {
            matches!(
                metadata.class,
                SpinorClass::Majorana | SpinorClass::MajoranaWeyl
            )
        })
        .unwrap_or(false)
}

fn is_weyl_spinor_expr(expr: &Expr, properties: &dyn ax_tensor::PropertyLookup) -> bool {
    spinor_metadata_of_expr(expr, properties)
        .map(|metadata| {
            matches!(
                metadata.class,
                SpinorClass::Weyl | SpinorClass::MajoranaWeyl
            )
        })
        .unwrap_or(false)
}

fn index_family_name(
    idx: &Index,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    idx.index_type.or_else(|| {
        properties
            .index_families()
            .and_then(|families| families.get(&idx.name).map(|family| family.name))
    })
}

fn index_family_dimension(
    idx: &Index,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    let family_name = index_family_name(idx, properties)?;
    properties
        .index_families()
        .and_then(|families| {
            families
                .get(&family_name)
                .and_then(|family| family.dimension)
        })
        .or_else(|| {
            properties
                .index_families()
                .and_then(|families| families.get(&idx.name).and_then(|family| family.dimension))
        })
}

fn collect_all_index_names(expr: &Expr, out: &mut HashSet<lasso::Spur>) {
    match expr {
        Expr::Indexed(base, indices) => {
            for idx in indices {
                out.insert(idx.name);
            }
            collect_all_index_names(base, out);
        }
        Expr::Add(items) | Expr::Mul(items) | Expr::List(items) | Expr::Call(_, items) => {
            for item in items {
                collect_all_index_names(item, out);
            }
        }
        Expr::Pow(base, exp) => {
            collect_all_index_names(base, out);
            collect_all_index_names(exp, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_all_index_names(inner, out),
        Expr::Complex(re, im) => {
            collect_all_index_names(re, out);
            collect_all_index_names(im, out);
        }
        Expr::FnDef(_, _, body) => collect_all_index_names(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_all_index_names(lhs, out);
            collect_all_index_names(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_all_index_names(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_all_index_names(value, out);
            collect_all_index_names(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_all_index_names(cell, out);
                }
            }
        }
        _ => {}
    }
}

#[derive(Clone, Debug)]
struct GammaExprData {
    head: Expr,
    #[allow(dead_code)]
    sym: Option<lasso::Spur>,
    indices: Vec<Index>,
}

fn gamma_expr_data(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<GammaExprData> {
    match expr {
        Expr::Call(sym, args) if gamma_metadata_of_expr(&Expr::Sym(*sym), properties).is_some() => {
            let indices = args
                .iter()
                .filter_map(|arg| match arg {
                    Expr::Sym(name) => Some(Index {
                        name: *name,
                        variance: Variance::Up,
                        index_type: None,
                    }),
                    Expr::Indexed(_, idxs) if idxs.len() == 1 => Some(idxs[0].clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            Some(GammaExprData {
                head: Expr::Sym(*sym),
                sym: Some(*sym),
                indices,
            })
        }
        Expr::Indexed(base, indices) if gamma_metadata_of_expr(base, properties).is_some() => {
            Some(GammaExprData {
                head: (**base).clone(),
                sym: property_sym(base),
                indices: indices.clone(),
            })
        }
        _ => None,
    }
}

fn build_gamma_expr(head: &Expr, indices: &[Index]) -> Expr {
    match head {
        Expr::Sym(sym)
            if indices
                .iter()
                .all(|idx| idx.index_type.is_none() && idx.variance == Variance::Up) =>
        {
            Expr::Call(
                *sym,
                indices.iter().map(|idx| Expr::Sym(idx.name)).collect(),
            )
        }
        _ => Expr::Indexed(Box::new(head.clone()), indices.to_vec()),
    }
}

fn build_metric_contraction(metric: &Expr, left: &Index, right: &Index) -> Expr {
    match metric {
        Expr::Sym(_) | Expr::Call(_, _) | Expr::Indexed(_, _) => Expr::Indexed(
            Box::new(metric.clone()),
            vec![
                Index {
                    name: left.name,
                    variance: Variance::Up,
                    index_type: left.index_type,
                },
                Index {
                    name: right.name,
                    variance: Variance::Up,
                    index_type: right.index_type,
                },
            ],
        ),
        _ => Expr::mul(vec![
            metric.clone(),
            Expr::Sym(left.name),
            Expr::Sym(right.name),
        ]),
    }
}

fn build_generalised_delta(uppers: &[Index], lowers: &[Index], interner: &ax_ir::Interner) -> Expr {
    let sym = interner.get_or_intern("generalised_delta");
    let mut args = Vec::with_capacity(uppers.len() + lowers.len());
    args.extend(uppers.iter().map(|idx| Expr::Sym(idx.name)));
    args.extend(lowers.iter().map(|idx| Expr::Sym(idx.name)));
    Expr::Call(sym, args)
}

fn permutation_parity(selection: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..selection.len() {
        for j in i + 1..selection.len() {
            if selection[i] > selection[j] {
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

fn combinations_of(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn helper(
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            helper(i + 1, n, k, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    let mut current = Vec::new();
    helper(0, n, k, &mut current, &mut out);
    out
}

fn factorial(n: usize) -> BigInt {
    (1..=n).fold(BigInt::one(), |acc, k| acc * BigInt::from(k))
}

fn fresh_dummy_from_family(
    family: &ax_ir::IndexFamily,
    used: &mut HashSet<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    for value in &family.values {
        if used.insert(*value) {
            return *value;
        }
    }
    let mut counter = 0usize;
    loop {
        let candidate =
            interner.get_or_intern(&format!("{}_{}", interner.resolve(family.name), counter));
        counter += 1;
        if used.insert(candidate) {
            return candidate;
        }
    }
}

impl FierzError {
    fn symbol_name(&self) -> &'static str {
        match self {
            FierzError::NoBilinearPair => "fierz_no_bilinear_pair",
            FierzError::AmbiguousBilinears(_) => "fierz_ambiguous_bilinears",
            FierzError::MalformedBilinear => "fierz_malformed_bilinear",
            FierzError::AmbiguousSpinorOrder => "fierz_ambiguous_spinor_order",
            FierzError::SpinorOrderMismatch => "fierz_spinor_order_mismatch",
            FierzError::IncompatibleSpinorMetadata => "fierz_incompatible_spinor_metadata",
            FierzError::IncompatibleSpinorDimension => "fierz_incompatible_spinor_dimension",
            FierzError::IncompatibleSpinorChirality => "fierz_incompatible_spinor_chirality",
        }
    }
}

fn operator_info(
    expr: &Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> Option<(OperatorKind, Option<Expr>, OperatorStatistics)> {
    match expr {
        Expr::Sym(sym) => operators.get(sym).copied().map(|kind| {
            (
                kind,
                Some(Expr::Sym(*sym)),
                operator_statistics
                    .get(sym)
                    .copied()
                    .unwrap_or(OperatorStatistics::Bosonic),
            )
        }),
        Expr::Call(f, args) if args.len() == 1 => match interner.resolve(*f) {
            "creation" => Some((
                OperatorKind::Creation,
                Some(args[0].clone()),
                match &args[0] {
                    Expr::Sym(sym) => operator_statistics
                        .get(sym)
                        .copied()
                        .unwrap_or(OperatorStatistics::Bosonic),
                    _ => OperatorStatistics::Bosonic,
                },
            )),
            "annihilation" => Some((
                OperatorKind::Annihilation,
                Some(args[0].clone()),
                match &args[0] {
                    Expr::Sym(sym) => operator_statistics
                        .get(sym)
                        .copied()
                        .unwrap_or(OperatorStatistics::Bosonic),
                    _ => OperatorStatistics::Bosonic,
                },
            )),
            _ => None,
        },
        _ => None,
    }
}

fn modes_match(lhs: &Option<Expr>, rhs: &Option<Expr>) -> bool {
    matches!((lhs, rhs), (Some(a), Some(b)) if a == b)
}

fn graded_reorder_mul(
    factors: Vec<Expr>,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> Expr {
    for i in 0..factors.len().saturating_sub(1) {
        let left = operator_info(&factors[i], operators, operator_statistics, interner);
        let right = operator_info(&factors[i + 1], operators, operator_statistics, interner);
        if let (
            Some((OperatorKind::Annihilation, _, left_stats)),
            Some((OperatorKind::Creation, _, right_stats)),
        ) = (left, right)
        {
            let mut swapped = factors.clone();
            swapped.swap(i, i + 1);
            let reordered = graded_reorder_mul(swapped, operators, operator_statistics, interner);
            return if left_stats == OperatorStatistics::Fermionic
                && right_stats == OperatorStatistics::Fermionic
            {
                Expr::neg(reordered)
            } else {
                reordered
            };
        }
    }
    Expr::mul(factors)
}

fn normal_order_mul(
    factors: Vec<Expr>,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> Expr {
    for i in 0..factors.len().saturating_sub(1) {
        let left = operator_info(&factors[i], operators, operator_statistics, interner);
        let right = operator_info(&factors[i + 1], operators, operator_statistics, interner);
        if let (
            Some((OperatorKind::Annihilation, left_mode, left_stats)),
            Some((OperatorKind::Creation, right_mode, right_stats)),
        ) = (left, right)
        {
            let mut swapped = factors.clone();
            swapped.swap(i, i + 1);
            let reordered = normal_order_mul(swapped, operators, operator_statistics, interner);
            let reordered = if left_stats == OperatorStatistics::Fermionic
                && right_stats == OperatorStatistics::Fermionic
            {
                Expr::neg(reordered)
            } else {
                reordered
            };
            if modes_match(&left_mode, &right_mode) && left_stats == right_stats {
                let mut remaining = factors.clone();
                remaining.remove(i + 1);
                remaining.remove(i);
                let contraction = if remaining.is_empty() {
                    Expr::one()
                } else {
                    normal_order_mul(remaining, operators, operator_statistics, interner)
                };
                return simplify_expr(Expr::add(vec![reordered, contraction]));
            }
            return simplify_expr(reordered);
        }
    }
    Expr::mul(factors)
}

pub fn normal_order_simple(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => {
            let simplified = factors
                .iter()
                .map(|factor| normal_order_simple(factor, operators, operator_statistics, interner))
                .collect();
            normal_order_mul(simplified, operators, operator_statistics, interner)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| normal_order_simple(term, operators, operator_statistics, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            normal_order_simple(base, operators, operator_statistics, interner),
            normal_order_simple(exp, operators, operator_statistics, interner),
        ),
        Expr::Neg(inner) => Expr::neg(normal_order_simple(
            inner,
            operators,
            operator_statistics,
            interner,
        )),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normal_order_simple(
                re,
                operators,
                operator_statistics,
                interner,
            )),
            Box::new(normal_order_simple(
                im,
                operators,
                operator_statistics,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| normal_order_simple(arg, operators, operator_statistics, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(normal_order_simple(
                body,
                operators,
                operator_statistics,
                interner,
            )),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(normal_order_simple(
                lhs,
                operators,
                operator_statistics,
                interner,
            )),
            Box::new(normal_order_simple(
                rhs,
                operators,
                operator_statistics,
                interner,
            )),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        normal_order_simple(value, operators, operator_statistics, interner),
                        condition.clone(),
                    )
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(normal_order_simple(
                base,
                operators,
                operator_statistics,
                interner,
            )),
            indices.clone(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(normal_order_simple(
                value,
                operators,
                operator_statistics,
                interner,
            )),
            Box::new(normal_order_simple(
                body,
                operators,
                operator_statistics,
                interner,
            )),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| normal_order_simple(item, operators, operator_statistics, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| {
                            normal_order_simple(cell, operators, operator_statistics, interner)
                        })
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn normal_order(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    normal_order_simple(expr, operators, operator_statistics, interner)
}

fn contraction_mode_key(
    expr: &Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> Option<(OperatorKind, lasso::Spur, OperatorStatistics)> {
    let (kind, mode, statistics) = operator_info(expr, operators, operator_statistics, interner)?;
    match mode {
        Some(Expr::Sym(sym)) => Some((kind, sym, statistics)),
        _ => None,
    }
}

fn fermionic_contraction_sign(
    factors: &[Expr],
    left_index: usize,
    right_index: usize,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    interner: &ax_ir::Interner,
) -> Expr {
    let Some((_, _, right_statistics)) = contraction_mode_key(
        &factors[right_index],
        operators,
        operator_statistics,
        interner,
    ) else {
        return Expr::one();
    };
    if right_statistics != OperatorStatistics::Fermionic {
        return Expr::one();
    }

    let fermionic_crossings = factors[(left_index + 1)..right_index]
        .iter()
        .filter_map(|factor| {
            contraction_mode_key(factor, operators, operator_statistics, interner)
                .map(|(_, _, statistics)| statistics)
        })
        .filter(|statistics| *statistics == OperatorStatistics::Fermionic)
        .count();

    if fermionic_crossings % 2 == 0 {
        Expr::one()
    } else {
        Expr::neg(Expr::one())
    }
}

pub fn wick_expand_single(
    factors: &[ax_ir::Expr],
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let mut terms = Vec::new();
    let mut found_contraction = false;

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let Some((_, lhs, _)) =
                contraction_mode_key(&factors[i], operators, operator_statistics, interner)
            else {
                continue;
            };
            let Some((_, rhs, _)) =
                contraction_mode_key(&factors[j], operators, operator_statistics, interner)
            else {
                continue;
            };
            if let Some(contraction) = contractions.get(&(lhs, rhs)) {
                found_contraction = true;
                let remaining = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, factor)| {
                        if idx == i || idx == j {
                            None
                        } else {
                            Some(factor.clone())
                        }
                    })
                    .collect::<Vec<_>>();
                let ordered_remaining = if remaining.is_empty() {
                    Expr::one()
                } else {
                    wick_expand(
                        &Expr::mul(remaining),
                        operators,
                        operator_statistics,
                        contractions,
                        interner,
                    )
                };
                let signed_contraction = simplify_expr(Expr::mul(vec![
                    fermionic_contraction_sign(
                        factors,
                        i,
                        j,
                        operators,
                        operator_statistics,
                        interner,
                    ),
                    contraction.clone(),
                ]));
                terms.push(Expr::mul(vec![signed_contraction, ordered_remaining]));
            }
        }
    }

    if !found_contraction {
        return normal_order_simple(
            &Expr::mul(factors.to_vec()),
            operators,
            operator_statistics,
            interner,
        );
    }

    terms.insert(
        0,
        graded_reorder_mul(factors.to_vec(), operators, operator_statistics, interner),
    );

    simplify_expr(Expr::add(terms))
}

pub fn wick_expand(
    expr: &ax_ir::Expr,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    operator_statistics: &HashMap<lasso::Spur, OperatorStatistics>,
    contractions: &HashMap<(lasso::Spur, lasso::Spur), ax_ir::Expr>,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match expr {
        Expr::Mul(factors) => wick_expand_single(
            factors,
            operators,
            operator_statistics,
            contractions,
            interner,
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    wick_expand(term, operators, operator_statistics, contractions, interner)
                })
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            wick_expand(base, operators, operator_statistics, contractions, interner),
            wick_expand(exp, operators, operator_statistics, contractions, interner),
        ),
        Expr::Neg(inner) => Expr::neg(wick_expand(
            inner,
            operators,
            operator_statistics,
            contractions,
            interner,
        )),
        _ => normal_order_simple(expr, operators, operator_statistics, interner),
    }
}

fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let mut grouped: Vec<(Expr, usize)> = Vec::new();
            for term in terms.into_iter().map(simplify_expr) {
                if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == term)
                {
                    *count += 1;
                } else {
                    grouped.push((term, 1));
                }
            }
            Expr::add(
                grouped
                    .into_iter()
                    .map(|(term, count)| {
                        if count == 1 {
                            term
                        } else {
                            Expr::mul(vec![Expr::Int((count as i64).into()), term])
                        }
                    })
                    .collect(),
            )
        }
        Expr::Mul(factors) => Expr::mul(factors.into_iter().map(simplify_expr).collect()),
        Expr::Pow(base, exp) => Expr::pow(simplify_expr(*base), simplify_expr(*exp)),
        Expr::Neg(inner) => Expr::neg(simplify_expr(*inner)),
        other => other,
    }
}

fn simplify_matrix(matrix: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    matrix
        .into_iter()
        .map(|row| row.into_iter().map(simplify_expr).collect())
        .collect()
}

fn zero_matrix(dim: usize) -> Vec<Vec<Expr>> {
    vec![vec![Expr::zero(); dim]; dim]
}

fn matrix_shape(matrix: &[Vec<Expr>]) -> Option<(usize, usize)> {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    matrix
        .iter()
        .all(|row| row.len() == cols)
        .then_some((rows, cols))
}

fn adjoint_matrix(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    ax_linalg::transpose(matrix)
        .into_iter()
        .map(|row| row.into_iter().map(|cell| conjugate_expr(&cell)).collect())
        .collect()
}

fn is_zero_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(n) if n.is_zero()) || matches!(expr, Expr::Rational(r) if r.is_zero())
}

fn validate_kraus_set(kraus: &[Vec<Vec<Expr>>]) -> Result<usize, ChannelError> {
    if kraus.is_empty() {
        return Err(ChannelError::EmptyKrausSet);
    }

    let mut expected_dim = None;
    for (index, operator) in kraus.iter().enumerate() {
        let rows = operator.len();
        let cols = operator.first().map(|row| row.len()).unwrap_or(0);
        let Some((actual_rows, actual_cols)) = matrix_shape(operator) else {
            return Err(ChannelError::NonSquareKraus { index, rows, cols });
        };
        if actual_rows != actual_cols {
            return Err(ChannelError::NonSquareKraus {
                index,
                rows: actual_rows,
                cols: actual_cols,
            });
        }

        if let Some(expected) = expected_dim {
            if actual_rows != expected {
                return Err(ChannelError::KrausDimensionMismatch {
                    expected,
                    actual: actual_rows,
                    index,
                });
            }
        } else {
            expected_dim = Some(actual_rows);
        }
    }

    Ok(expected_dim.unwrap_or(0))
}

fn validate_square_state_dimension(
    matrix: &[Vec<Expr>],
    expected: usize,
) -> Result<(), MeasurementError> {
    let (rows, cols) = matrix_shape(matrix).unwrap_or((
        matrix.len(),
        matrix.first().map(|row| row.len()).unwrap_or(0),
    ));
    if rows != cols || rows != expected {
        return Err(MeasurementError::StateDimensionMismatch {
            expected,
            actual: rows,
        });
    }
    Ok(())
}

fn validate_projector_set(
    projectors: &[Vec<Vec<Expr>>],
    expected: usize,
) -> Result<(), MeasurementError> {
    for (index, projector) in projectors.iter().enumerate() {
        let (rows, cols) = matrix_shape(projector).unwrap_or((
            projector.len(),
            projector.first().map(|row| row.len()).unwrap_or(0),
        ));
        if rows != cols || rows != expected {
            return Err(MeasurementError::ProjectorDimensionMismatch {
                expected,
                actual: rows,
                index,
            });
        }
    }
    Ok(())
}

fn validate_lindblad_square_matrix(
    matrix: &[Vec<Expr>],
    which: &'static str,
) -> Result<usize, LindbladError> {
    let (rows, cols) = matrix_shape(matrix).unwrap_or((
        matrix.len(),
        matrix.first().map(|row| row.len()).unwrap_or(0),
    ));
    if rows != cols {
        return Err(match which {
            "Hamiltonian" => LindbladError::HamiltonianNotSquare { rows, cols },
            "state" => LindbladError::StateNotSquare { rows, cols },
            _ => LindbladError::DimensionMismatch {
                expected: rows,
                actual: cols,
                which,
            },
        });
    }
    Ok(rows)
}

fn validate_lindblad_jump_ops(
    jump_ops: &[Vec<Vec<Expr>>],
    expected: usize,
) -> Result<(), LindbladError> {
    for operator in jump_ops {
        let (rows, cols) = matrix_shape(operator).unwrap_or((
            operator.len(),
            operator.first().map(|row| row.len()).unwrap_or(0),
        ));
        if rows != cols {
            return Err(LindbladError::DimensionMismatch {
                expected,
                actual: rows,
                which: "jump operator",
            });
        }
        if rows != expected {
            return Err(LindbladError::DimensionMismatch {
                expected,
                actual: rows,
                which: "jump operator",
            });
        }
    }
    Ok(())
}

pub fn pauli_x(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero()],
    ]
}

pub fn pauli_y(interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    let _ = interner;
    let i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()));
    let neg_i = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())));
    vec![vec![Expr::zero(), neg_i], vec![i, Expr::zero()]]
}

pub fn pauli_z(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::neg(Expr::one())],
    ]
}

fn imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
}

fn neg_imag_unit() -> Expr {
    Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::neg(Expr::one())))
}

pub fn gamma_matrices_dirac(_interner: &ax_ir::Interner) -> Vec<Vec<Vec<ax_ir::Expr>>> {
    let i = imag_unit();
    let neg_i = neg_imag_unit();
    vec![
        vec![
            vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
            ],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
            ],
            vec![
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
            ],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::zero(), neg_i.clone()],
            vec![Expr::zero(), Expr::zero(), i.clone(), Expr::zero()],
            vec![Expr::zero(), i, Expr::zero(), Expr::zero()],
            vec![neg_i, Expr::zero(), Expr::zero(), Expr::zero()],
        ],
        vec![
            vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
            vec![
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
                Expr::neg(Expr::one()),
            ],
            vec![
                Expr::neg(Expr::one()),
                Expr::zero(),
                Expr::zero(),
                Expr::zero(),
            ],
            vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
        ],
    ]
}

pub fn gamma5(_interner: &ax_ir::Interner) -> Vec<Vec<ax_ir::Expr>> {
    vec![
        vec![Expr::zero(), Expr::zero(), Expr::one(), Expr::zero()],
        vec![Expr::zero(), Expr::zero(), Expr::zero(), Expr::one()],
        vec![Expr::one(), Expr::zero(), Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::one(), Expr::zero(), Expr::zero()],
    ]
}

pub fn gamma_trace_recursive(
    indices: &[lasso::Spur],
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let _ = interner;
    let n = indices.len();
    if n == 0 {
        return Expr::Int(4.into());
    }
    if n % 2 != 0 {
        return Expr::zero();
    }
    if n == 2 {
        return Expr::mul(vec![
            Expr::Int(4.into()),
            Expr::Indexed(
                Box::new(Expr::Sym(metric_sym)),
                vec![
                    ax_ir::Index {
                        name: indices[0],
                        variance: ax_ir::Variance::Up,
                        index_type: None,
                    },
                    ax_ir::Index {
                        name: indices[1],
                        variance: ax_ir::Variance::Up,
                        index_type: None,
                    },
                ],
            ),
        ]);
    }

    let a1 = indices[0];
    let mut terms = Vec::new();
    for k in 1..n {
        let sign = if (k - 1) % 2 == 0 {
            Expr::one()
        } else {
            Expr::neg(Expr::one())
        };
        let metric_factor = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                ax_ir::Index {
                    name: a1,
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
                ax_ir::Index {
                    name: indices[k],
                    variance: ax_ir::Variance::Up,
                    index_type: None,
                },
            ],
        );
        let remaining = indices[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k - 1)
            .map(|(_, sym)| *sym)
            .collect::<Vec<_>>();
        let sub_trace = gamma_trace_recursive(&remaining, metric_sym, interner);
        terms.push(Expr::mul(vec![sign, metric_factor, sub_trace]));
    }
    Expr::add(terms)
}

pub fn gamma_trace(
    indices: &[GammaEntry],
    metric: &ax_tensor::SymbolicMatrix,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    let metric_sym = interner.get_or_intern("g");
    let epsilon_sym = interner.get_or_intern("epsilon");

    let mut gamma_indices = Vec::new();
    let mut numeric_indices = Vec::new();
    let mut has_symbolic_indices = false;
    let mut has_numeric_indices = false;
    let mut gamma5_count = 0usize;
    for entry in indices {
        match entry {
            GammaEntry::Gamma(sym) => {
                gamma_indices.push(*sym);
                has_symbolic_indices = true;
            }
            GammaEntry::Index(index) => {
                numeric_indices.push(*index);
                has_numeric_indices = true;
            }
            GammaEntry::Gamma5 => gamma5_count += 1,
            GammaEntry::Identity => {}
        }
    }

    if has_numeric_indices && !has_symbolic_indices && gamma5_count == 0 {
        return gamma_trace_numeric(&numeric_indices, metric);
    }

    if has_numeric_indices {
        gamma_indices.extend(numeric_indices.into_iter().map(|index| {
            let name = format!("mu{index}");
            interner.get_or_intern(&name)
        }));
    }

    if gamma5_count > 1 {
        return Expr::zero();
    }

    if gamma5_count == 1 {
        return match gamma_indices.len() {
            0 | 1 | 2 | 3 => Expr::zero(),
            4 => Expr::mul(vec![
                Expr::Int((-4).into()),
                imag_unit(),
                Expr::Indexed(
                    Box::new(Expr::Sym(epsilon_sym)),
                    gamma_indices
                        .iter()
                        .map(|sym| ax_ir::Index {
                            name: *sym,
                            variance: ax_ir::Variance::Up,
                            index_type: None,
                        })
                        .collect(),
                ),
            ]),
            _ if gamma_indices.len() % 2 != 0 => Expr::zero(),
            _ => Expr::zero(),
        };
    }

    gamma_trace_recursive(&gamma_indices, metric_sym, interner)
}

fn gamma_trace_numeric(indices: &[usize], metric: &ax_tensor::SymbolicMatrix) -> Expr {
    let n = indices.len();
    if n == 0 {
        return Expr::Int(4.into());
    }
    if n % 2 != 0 {
        return Expr::zero();
    }
    if n == 2 {
        return Expr::mul(vec![
            Expr::Int(4.into()),
            metric.get(indices[0], indices[1]).clone(),
        ]);
    }

    let first = indices[0];
    let mut terms = Vec::new();
    for k in 1..n {
        let metric_factor = metric.get(first, indices[k]).clone();
        let remaining = indices[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k - 1)
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        let term = Expr::mul(vec![metric_factor, gamma_trace_numeric(&remaining, metric)]);
        if (k - 1) % 2 == 0 {
            terms.push(term);
        } else {
            terms.push(Expr::neg(term));
        }
    }
    simplify_expr(Expr::add(terms))
}

pub fn commutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(
        &ab,
        &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &ba),
    ))
}

pub fn anticommutator(
    a: &[Vec<ax_ir::Expr>],
    b: &[Vec<ax_ir::Expr>],
    interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let ab = ax_linalg::mat_mul(a, b, interner);
    let ba = ax_linalg::mat_mul(b, a, interner);
    simplify_matrix(ax_linalg::mat_add(&ab, &ba))
}

fn half() -> Expr {
    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
}

pub fn angular_momentum_matrices(
    j: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Option<(
    Vec<Vec<ax_ir::Expr>>,
    Vec<Vec<ax_ir::Expr>>,
    Vec<Vec<ax_ir::Expr>>,
)> {
    match j {
        Expr::Rational(r) if *r == num_rational::BigRational::new(1.into(), 2.into()) => {
            let hx = ax_linalg::mat_scale(&half(), &pauli_x(interner));
            let hy = ax_linalg::mat_scale(&half(), &pauli_y(interner));
            let hz = ax_linalg::mat_scale(&half(), &pauli_z(interner));
            Some((hx, hy, hz))
        }
        Expr::Int(n) if *n == 1.into() => {
            let sqrt2 = Expr::Call(interner.get_or_intern("sqrt"), vec![Expr::Int(2.into())]);
            let jp = vec![
                vec![Expr::zero(), sqrt2.clone(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), sqrt2.clone()],
                vec![Expr::zero(), Expr::zero(), Expr::zero()],
            ];
            let jm = ax_linalg::transpose(&jp);
            let jx = ax_linalg::mat_scale(&half(), &ax_linalg::mat_add(&jp, &jm));
            let two_i = Expr::mul(vec![
                Expr::Int(2.into()),
                Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one())),
            ]);
            let jy = ax_linalg::mat_scale(
                &Expr::pow(two_i, Expr::Int((-1).into())),
                &ax_linalg::mat_add(&jp, &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &jm)),
            );
            let jz = vec![
                vec![Expr::Int(1.into()), Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), Expr::zero()],
                vec![Expr::zero(), Expr::zero(), Expr::Int((-1).into())],
            ];
            Some((jx, jy, jz))
        }
        _ => None,
    }
}

pub fn density_matrix(state: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    match try_density_matrix(state) {
        Ok(matrix) => matrix,
        Err(_) => Vec::new(),
    }
}

pub fn partial_trace(
    rho: &[Vec<ax_ir::Expr>],
    dim_a: usize,
    dim_b: usize,
    trace_over: char,
    _interner: &ax_ir::Interner,
) -> Vec<Vec<ax_ir::Expr>> {
    let target = match trace_over {
        'A' => PartialTraceTarget::A,
        'B' => PartialTraceTarget::B,
        other => {
            let _ = QmLinearAlgebraError::InvalidTraceTarget { target: other };
            return Vec::new();
        }
    };
    try_partial_trace(rho, BipartiteDims { dim_a, dim_b }, target).unwrap_or_default()
}

pub fn ket(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    match try_ket(index, dim) {
        Ok(vec) => vec,
        Err(_) => vec![Expr::zero(); dim],
    }
}

pub fn bra(index: usize, dim: usize) -> Vec<ax_ir::Expr> {
    match try_bra(index, dim) {
        Ok(vec) => vec,
        Err(_) => vec![Expr::zero(); dim],
    }
}

pub fn braket(bra: &[ax_ir::Expr], ket: &[ax_ir::Expr]) -> ax_ir::Expr {
    match try_braket(bra, ket) {
        Ok(expr) => expr,
        Err(_) => Expr::add(
            bra.iter()
                .zip(ket.iter())
                .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
                .collect(),
        ),
    }
}

pub fn outer(a: &[ax_ir::Expr], b: &[ax_ir::Expr]) -> Vec<Vec<ax_ir::Expr>> {
    match try_outer(a, b) {
        Ok(matrix) => matrix,
        Err(_) => a
            .iter()
            .map(|ai| {
                b.iter()
                    .map(|bj| Expr::mul(vec![ai.clone(), bj.clone()]))
                    .collect()
            })
            .collect(),
    }
}

/// Return the complex conjugate of an expression while preserving symbolic structure.
pub fn conjugate_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => expr.clone(),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(conjugate_expr(re)),
            Box::new(Expr::neg(conjugate_expr(im))),
        ),
        Expr::Add(items) => Expr::add(items.iter().map(conjugate_expr).collect()),
        Expr::Mul(items) => Expr::mul(items.iter().map(conjugate_expr).collect()),
        Expr::Pow(base, exp) => Expr::pow(conjugate_expr(base), conjugate_expr(exp)),
        Expr::Neg(inner) => Expr::neg(conjugate_expr(inner)),
        Expr::Call(sym, args) => Expr::Call(*sym, args.iter().map(conjugate_expr).collect()),
        Expr::FnDef(name, params, body) => {
            Expr::FnDef(*name, params.clone(), Box::new(conjugate_expr(body)))
        }
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(conjugate_expr(lhs)),
            Box::new(conjugate_expr(rhs)),
            *trust,
        ),
        Expr::Indexed(base, indices) => {
            Expr::Indexed(Box::new(conjugate_expr(base)), indices.clone())
        }
        Expr::Group(inner, rel) => Expr::Group(Box::new(conjugate_expr(inner)), *rel),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(conjugate_expr(value)),
            Box::new(conjugate_expr(body)),
        ),
        Expr::List(items) => Expr::List(items.iter().map(conjugate_expr).collect()),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| row.iter().map(conjugate_expr).collect())
                .collect(),
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (conjugate_expr(value), condition.clone()))
                .collect(),
        ),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => expr.clone(),
    }
}

/// Return the Hermitian adjoint of a state vector represented as a flat expression slice.
pub fn adjoint_vector(vec: &[Expr]) -> Vec<Expr> {
    vec.iter().map(conjugate_expr).collect()
}

/// Build a computational basis ket with bounds checking.
pub fn try_ket(index: usize, dim: usize) -> Result<Vec<Expr>, QmLinearAlgebraError> {
    if index >= dim {
        return Err(QmLinearAlgebraError::BasisIndexOutOfRange { index, dim });
    }
    let mut out = vec![Expr::zero(); dim];
    out[index] = Expr::one();
    Ok(out)
}

/// Build a computational basis bra as the adjoint of the corresponding basis ket.
pub fn try_bra(index: usize, dim: usize) -> Result<Vec<Expr>, QmLinearAlgebraError> {
    let ket = try_ket(index, dim)?;
    Ok(adjoint_vector(&ket))
}

/// Compute the inner product `⟨bra|ket⟩ = Σ_i conj(bra_i) * ket_i`.
pub fn try_braket(bra: &[Expr], ket: &[Expr]) -> Result<Expr, QmLinearAlgebraError> {
    if bra.len() != ket.len() {
        return Err(QmLinearAlgebraError::DimensionMismatch {
            left: bra.len(),
            right: ket.len(),
        });
    }
    Ok(Expr::add(
        bra.iter()
            .zip(ket.iter())
            .map(|(bra_i, ket_i)| Expr::mul(vec![conjugate_expr(bra_i), ket_i.clone()]))
            .collect(),
    ))
}

/// Compute the outer product `|ket⟩⟨bra|` with element `(i, j) = ket[i] * conj(bra[j])`.
pub fn try_outer(ket: &[Expr], bra: &[Expr]) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    Ok(ket
        .iter()
        .map(|ket_i| {
            bra.iter()
                .map(|bra_j| Expr::mul(vec![ket_i.clone(), conjugate_expr(bra_j)]))
                .collect()
        })
        .collect())
}

/// Compute the pure-state density matrix `|ψ⟩⟨ψ|`.
pub fn try_density_matrix(state: &[Expr]) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    try_outer(state, state)
}

/// Construct the computational-basis projector `|index⟩⟨index|`.
pub fn basis_projector(index: usize, dim: usize) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    let ket = try_ket(index, dim)?;
    try_outer(&ket, &ket)
}

/// Compute the Kraus completeness matrix `Σ_k K_k† K_k` for a finite-dimensional channel.
pub fn kraus_completeness_matrix(kraus: &[Vec<Vec<Expr>>]) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = validate_kraus_set(kraus)?;
    let interner = ax_ir::Interner::new();
    let mut completeness = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let term = ax_linalg::mat_mul(&adjoint, operator, &interner);
        completeness = simplify_matrix(ax_linalg::mat_add(&completeness, &term));
    }
    Ok(simplify_matrix(completeness))
}

/// Apply a Kraus channel to a density matrix via `Σ_k K_k ρ K_k†`.
pub fn apply_kraus_channel(
    kraus: &[Vec<Vec<Expr>>],
    rho: &[Vec<Expr>],
) -> Result<Vec<Vec<Expr>>, ChannelError> {
    let dim = validate_kraus_set(kraus)?;
    let (rows, cols) =
        matrix_shape(rho).unwrap_or((rho.len(), rho.first().map(|row| row.len()).unwrap_or(0)));
    if rows != cols || rows != dim {
        return Err(ChannelError::StateDimensionMismatch {
            expected: dim,
            actual: rows,
        });
    }

    let interner = ax_ir::Interner::new();
    let mut output = zero_matrix(dim);
    for operator in kraus {
        let adjoint = adjoint_matrix(operator);
        let left = ax_linalg::mat_mul(operator, rho, &interner);
        let term = ax_linalg::mat_mul(&left, &adjoint, &interner);
        output = simplify_matrix(ax_linalg::mat_add(&output, &term));
    }

    Ok(simplify_matrix(output))
}

/// Construct the finite-dimensional identity channel with a single identity Kraus operator.
pub fn identity_channel(dim: usize) -> Vec<Vec<Vec<Expr>>> {
    vec![ax_linalg::identity(dim)]
}

/// Compute projective-measurement probabilities `p_i = Tr(P_i ρ)`.
pub fn measurement_probabilities(
    projectors: &[Vec<Vec<Expr>>],
    rho: &[Vec<Expr>],
) -> Result<Vec<Expr>, MeasurementError> {
    let dim = rho.len();
    validate_square_state_dimension(rho, dim)?;
    validate_projector_set(projectors, dim)?;
    let interner = ax_ir::Interner::new();

    Ok(projectors
        .iter()
        .map(|projector| {
            let product = ax_linalg::mat_mul(projector, rho, &interner);
            simplify_expr(ax_linalg::trace(&product))
        })
        .collect())
}

/// Compute the normalized post-measurement state `ρ_i = P_i ρ P_i / p_i`.
pub fn post_measurement_state(
    projector: &[Vec<Expr>],
    rho: &[Vec<Expr>],
    outcome_index: usize,
) -> Result<Vec<Vec<Expr>>, MeasurementError> {
    let dim = rho.len();
    validate_square_state_dimension(rho, dim)?;
    validate_projector_set(&[projector.to_vec()], dim)?;
    let probability = measurement_probabilities(&[projector.to_vec()], rho)?
        .into_iter()
        .next()
        .unwrap_or_else(Expr::zero);
    if is_zero_expr(&probability) {
        return Err(MeasurementError::ZeroProbabilityOutcome {
            index: outcome_index,
        });
    }

    let interner = ax_ir::Interner::new();
    let left = ax_linalg::mat_mul(projector, rho, &interner);
    let numerator = ax_linalg::mat_mul(&left, projector, &interner);
    let inv_probability = Expr::pow(probability, Expr::Int((-1).into()));
    Ok(simplify_matrix(ax_linalg::mat_scale(
        &inv_probability,
        &numerator,
    )))
}

/// Construct the finite-dimensional Lindblad right-hand side
/// `ρ̇ = -i [H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})`.
pub fn lindblad_rhs(
    h: &[Vec<Expr>],
    rho: &[Vec<Expr>],
    jump_ops: &[Vec<Vec<Expr>>],
    interner: &ax_ir::Interner,
) -> Result<Vec<Vec<Expr>>, LindbladError> {
    let dim = validate_lindblad_square_matrix(h, "Hamiltonian")?;
    let rho_dim = validate_lindblad_square_matrix(rho, "state")?;
    if rho_dim != dim {
        return Err(LindbladError::DimensionMismatch {
            expected: dim,
            actual: rho_dim,
            which: "state",
        });
    }
    validate_lindblad_jump_ops(jump_ops, dim)?;

    let coherent = simplify_matrix(ax_linalg::mat_scale(
        &Expr::neg(imag_unit()),
        &commutator(h, rho, interner),
    ));

    let mut dissipator = zero_matrix(dim);
    for jump in jump_ops {
        let jump_dagger = adjoint_matrix(jump);
        let jump_rho = ax_linalg::mat_mul(jump, rho, interner);
        let gain = ax_linalg::mat_mul(&jump_rho, &jump_dagger, interner);
        let jump_norm = ax_linalg::mat_mul(&jump_dagger, jump, interner);
        let loss = ax_linalg::mat_scale(&half(), &anticommutator(&jump_norm, rho, interner));
        let term = ax_linalg::mat_add(&gain, &ax_linalg::mat_scale(&Expr::neg(Expr::one()), &loss));
        dissipator = simplify_matrix(ax_linalg::mat_add(&dissipator, &term));
    }

    Ok(simplify_matrix(ax_linalg::mat_add(&coherent, &dissipator)))
}

/// Convert a multi-index in row-major tensor-product order into a flattened linear index.
pub fn linear_index_from_multi(indices: &[usize], dims: &[usize]) -> usize {
    indices
        .iter()
        .zip(dims.iter())
        .fold(0usize, |acc, (index, dim)| {
            acc.saturating_mul(*dim).saturating_add(*index)
        })
}

/// Convert a flattened linear index into its row-major tensor-product multi-index.
pub fn multi_index_from_linear(index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }

    let mut remaining = index;
    let mut out = vec![0; dims.len()];
    for pos in (0..dims.len()).rev() {
        let dim = dims[pos];
        if dim == 0 {
            out[pos] = 0;
        } else {
            out[pos] = remaining % dim;
            remaining /= dim;
        }
    }
    out
}

/// Trace out one tensor factor from a density matrix over an arbitrary ordered factorization.
///
/// The output preserves the original relative order of all remaining factors. When the input has
/// a single factor, tracing that factor returns the `1x1` matrix whose only entry is `Tr(rho)`.
pub fn try_partial_trace_factor(
    rho: &[Vec<Expr>],
    factor_dims: &[usize],
    traced_factor: usize,
) -> Result<Vec<Vec<Expr>>, CompositeSpaceError> {
    if factor_dims.is_empty() {
        return Err(CompositeSpaceError::EmptyFactorList);
    }
    if traced_factor >= factor_dims.len() {
        return Err(CompositeSpaceError::InvalidFactorIndex {
            index: traced_factor,
            factor_count: factor_dims.len(),
        });
    }

    let rows = rho.len();
    let cols = rho.first().map(|row| row.len()).unwrap_or(0);
    if rho.iter().any(|row| row.len() != cols) || rows != cols {
        return Err(CompositeSpaceError::NonSquareMatrix { rows, cols });
    }

    let expected = factor_dims.iter().product::<usize>();
    if rows != expected {
        return Err(CompositeSpaceError::TotalDimensionMismatch {
            expected,
            actual: rows,
        });
    }

    if factor_dims.len() == 1 {
        return Ok(vec![vec![simplify_expr(ax_linalg::trace(rho))]]);
    }

    let traced_dim = factor_dims[traced_factor];
    let remaining_dims = factor_dims
        .iter()
        .enumerate()
        .filter_map(|(idx, dim)| (idx != traced_factor).then_some(*dim))
        .collect::<Vec<_>>();
    let remaining_total = remaining_dims.iter().product::<usize>();
    let mut out = vec![vec![Expr::zero(); remaining_total]; remaining_total];

    for out_row in 0..remaining_total {
        let row_multi = multi_index_from_linear(out_row, &remaining_dims);
        for out_col in 0..remaining_total {
            let col_multi = multi_index_from_linear(out_col, &remaining_dims);
            let terms = (0..traced_dim)
                .map(|traced_index| {
                    let mut full_row = Vec::with_capacity(factor_dims.len());
                    let mut full_col = Vec::with_capacity(factor_dims.len());
                    let mut row_cursor = 0usize;
                    let mut col_cursor = 0usize;
                    for factor_idx in 0..factor_dims.len() {
                        if factor_idx == traced_factor {
                            full_row.push(traced_index);
                            full_col.push(traced_index);
                        } else {
                            full_row.push(row_multi[row_cursor]);
                            full_col.push(col_multi[col_cursor]);
                            row_cursor += 1;
                            col_cursor += 1;
                        }
                    }
                    let row_index = linear_index_from_multi(&full_row, factor_dims);
                    let col_index = linear_index_from_multi(&full_col, factor_dims);
                    rho[row_index][col_index].clone()
                })
                .collect::<Vec<_>>();
            out[out_row][out_col] = simplify_expr(Expr::add(terms));
        }
    }

    Ok(out)
}

pub fn try_partial_trace(
    rho: &[Vec<Expr>],
    dims: BipartiteDims,
    target: PartialTraceTarget,
) -> Result<Vec<Vec<Expr>>, QmLinearAlgebraError> {
    let traced_factor = match target {
        PartialTraceTarget::A => 0,
        PartialTraceTarget::B => 1,
    };
    try_partial_trace_factor(rho, &[dims.dim_a, dims.dim_b], traced_factor).map_err(|err| match err
    {
        CompositeSpaceError::EmptyFactorList | CompositeSpaceError::InvalidFactorIndex { .. } => {
            QmLinearAlgebraError::InvalidTraceTarget { target: '?' }
        }
        CompositeSpaceError::NonSquareMatrix { rows, cols } => {
            QmLinearAlgebraError::NonSquareMatrix { rows, cols }
        }
        CompositeSpaceError::TotalDimensionMismatch { expected, actual } => {
            QmLinearAlgebraError::SubsystemDimensionMismatch { expected, actual }
        }
    })
}

/// Join (contract) a product of gamma matrices.
///
/// gamma(a) * gamma(b) → gamma(a, b) + g(a+, b+)
/// gamma(a) * gamma(b, c) → gamma(a, b, c) + g(a+, b+) * gamma(c) - g(a+, c+) * gamma(b)
///
/// Uses the recursive contraction identity.
pub fn join_gamma_pair(
    indices1: &[lasso::Spur],
    indices2: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    if indices1.is_empty() {
        return make_gamma(indices2, gamma_sym);
    }
    if indices2.is_empty() {
        return make_gamma(indices1, gamma_sym);
    }

    let a1 = indices1[0];
    let rest1 = &indices1[1..];

    if rest1.is_empty() {
        join_single_with_multi(a1, indices2, gamma_sym, metric_sym, interner)
    } else {
        let inner = join_gamma_pair(rest1, indices2, gamma_sym, metric_sym, interner);
        join_single_with_expr(a1, &inner, gamma_sym, metric_sym, interner)
    }
}

fn join_single_with_multi(
    a: lasso::Spur,
    bs: &[lasso::Spur],
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    _interner: &ax_ir::Interner,
) -> Expr {
    let mut terms = Vec::new();

    // First term: gamma with all indices combined
    let mut all = vec![a];
    all.extend_from_slice(bs);
    terms.push(make_gamma(&all, gamma_sym));

    // Contraction terms: Σ_k (-1)^k g^{a b_k} γ^{bs \ b_k}
    for k in 0..bs.len() {
        let metric = Expr::Indexed(
            Box::new(Expr::Sym(metric_sym)),
            vec![
                Index {
                    name: a,
                    variance: Variance::Up,
                    index_type: None,
                },
                Index {
                    name: bs[k],
                    variance: Variance::Up,
                    index_type: None,
                },
            ],
        );

        let remaining: Vec<lasso::Spur> = bs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .map(|(_, &b)| b)
            .collect();

        let gamma_part = if remaining.is_empty() {
            Expr::one()
        } else {
            make_gamma(&remaining, gamma_sym)
        };

        let term = Expr::mul(vec![metric, gamma_part]);
        if k % 2 == 0 {
            terms.push(term);
        } else {
            terms.push(Expr::neg(term));
        }
    }

    Expr::add(terms)
}

fn join_single_with_expr(
    a: lasso::Spur,
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_single_with_expr(a, t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Mul(factors) => {
            // Find the first gamma factor and join with it
            for (i, factor) in factors.iter().enumerate() {
                if let Expr::Call(f, args) = factor {
                    if *f == gamma_sym {
                        let gamma_indices: Vec<lasso::Spur> = args
                            .iter()
                            .filter_map(|arg| {
                                if let Expr::Sym(s) = arg {
                                    Some(*s)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let joined = join_single_with_multi(
                            a,
                            &gamma_indices,
                            gamma_sym,
                            metric_sym,
                            interner,
                        );
                        let mut rest: Vec<Expr> = factors
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, f)| f.clone())
                            .collect();
                        rest.push(joined);
                        return Expr::mul(rest);
                    }
                }
            }
            // No gamma factor found — prepend a single-index gamma
            let mut new_factors = vec![make_gamma(&[a], gamma_sym)];
            new_factors.extend(factors.iter().cloned());
            Expr::mul(new_factors)
        }
        Expr::Neg(e) => Expr::neg(join_single_with_expr(a, e, gamma_sym, metric_sym, interner)),
        _ => {
            // Check if expr itself is a gamma call
            if let Expr::Call(f, args) = expr {
                if *f == gamma_sym {
                    let gamma_indices: Vec<lasso::Spur> = args
                        .iter()
                        .filter_map(|arg| {
                            if let Expr::Sym(s) = arg {
                                Some(*s)
                            } else {
                                None
                            }
                        })
                        .collect();
                    return join_single_with_multi(
                        a,
                        &gamma_indices,
                        gamma_sym,
                        metric_sym,
                        interner,
                    );
                }
            }
            Expr::mul(vec![make_gamma(&[a], gamma_sym), expr.clone()])
        }
    }
}

fn make_gamma(indices: &[lasso::Spur], gamma_sym: lasso::Spur) -> Expr {
    Expr::Call(gamma_sym, indices.iter().map(|&i| Expr::Sym(i)).collect())
}

/// Extract gamma indices from a `gamma(a, b, ...)` Call expression.
fn gamma_indices(args: &[Expr]) -> Vec<lasso::Spur> {
    args.iter()
        .filter_map(|arg| {
            if let Expr::Sym(s) = arg {
                Some(*s)
            } else {
                None
            }
        })
        .collect()
}

/// Walk an expression and join all adjacent gamma-matrix Call nodes in products.
pub fn join_gammas_in_expr(
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            // Recursively process each factor first
            let factors: Vec<Expr> = factors
                .iter()
                .map(|f| join_gammas_in_expr(f, gamma_sym, metric_sym, interner))
                .collect();

            // Now fold adjacent gamma pairs left-to-right
            let mut result: Vec<Expr> = Vec::new();
            for factor in factors {
                if let Some(last) = result.last() {
                    if let (Expr::Call(f1, a1), Expr::Call(f2, a2)) = (last, &factor) {
                        if *f1 == gamma_sym && *f2 == gamma_sym {
                            let i1 = gamma_indices(a1);
                            let i2 = gamma_indices(a2);
                            let joined = join_gamma_pair(&i1, &i2, gamma_sym, metric_sym, interner);
                            result.pop();
                            // The joined expression may be an Add — wrap in a group
                            // by pushing the whole joined expression, then distributing
                            // remaining factors over it at the end.
                            result.push(joined);
                            continue;
                        }
                    }
                }
                result.push(factor);
            }

            if result.len() == 1 {
                result.remove(0)
            } else {
                Expr::mul(result)
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| join_gammas_in_expr(t, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(join_gammas_in_expr(e, gamma_sym, metric_sym, interner)),
        _ => expr.clone(),
    }
}

fn structured_spinor_family(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    property_sym(expr)
        .and_then(|sym| declared_spinor_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.index_family)
}

fn structured_gamma_family(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<lasso::Spur> {
    property_sym(expr)
        .and_then(|sym| declared_gamma_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.index_family)
}

fn gamma_declared_dimension(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    property_sym(expr)
        .and_then(|sym| declared_gamma_metadata_of_symbol(sym, properties))
        .and_then(|metadata| metadata.dimension)
}

fn gamma_effective_dimension(
    gam1: &Expr,
    g1: &GammaExprData,
    gam2: &Expr,
    g2: &GammaExprData,
    dimension: Option<usize>,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<usize> {
    gamma_declared_dimension(gam1, properties)
        .or_else(|| gamma_declared_dimension(gam2, properties))
        .or_else(|| {
            g1.indices
                .iter()
                .chain(g2.indices.iter())
                .filter_map(|idx| index_family_dimension(idx, properties))
                .max()
        })
        .or(dimension)
}

fn gamma_effective_families(
    expr: &Expr,
    data: &GammaExprData,
    properties: &dyn ax_tensor::PropertyLookup,
) -> HashSet<lasso::Spur> {
    let mut families = HashSet::new();
    if let Some(family) = structured_gamma_family(expr, properties) {
        families.insert(family);
    }
    families.extend(
        data.indices
            .iter()
            .filter_map(|idx| index_family_name(idx, properties)),
    );
    families
}

fn gamma_indices_have_duplicate_in_same_family(
    indices: &[Index],
    properties: &dyn ax_tensor::PropertyLookup,
) -> bool {
    let mut seen = HashSet::new();
    for idx in indices {
        let key = (idx.name, index_family_name(idx, properties));
        if !seen.insert(key) {
            return true;
        }
    }
    false
}

fn structured_diracbar_metadata(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Option<DiracBarMetadata> {
    property_sym(expr).and_then(|sym| declared_diracbar_metadata_of_symbol(sym, properties))
}

fn sort_spinor_metadata_error(
    expr: &Expr,
    bar_factor: &Expr,
    gamma_factor: Option<&Expr>,
    left_spinor: &Expr,
    right_spinor: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let left_family = structured_spinor_family(left_spinor, properties);
    let right_family = structured_spinor_family(right_spinor, properties);

    if let (Some(left), Some(right)) = (left_family, right_family) {
        if left != right {
            return Some(qm_error_expr(
                "sort_spinors_spinor_family_mismatch",
                expr,
                interner,
            ));
        }
    }

    if let Some(metadata) = structured_diracbar_metadata(bar_factor, properties) {
        if let Some(expected_gamma_symbol) = metadata.gamma_symbol {
            if gamma_factor
                .and_then(property_sym)
                .is_some_and(|actual| actual != expected_gamma_symbol)
            {
                return Some(qm_error_expr(
                    "sort_spinors_gamma_family_mismatch",
                    expr,
                    interner,
                ));
            }
        }
        if let Some(expected_spinor_family) = metadata.spinor_family {
            for actual in [left_family, right_family].into_iter().flatten() {
                if actual != expected_spinor_family {
                    return Some(qm_error_expr(
                        "sort_spinors_spinor_family_mismatch",
                        expr,
                        interner,
                    ));
                }
            }
        }
    }

    if let Some(gamma_factor) = gamma_factor {
        if let Some(gamma_family) = structured_gamma_family(gamma_factor, properties) {
            for actual in [left_family, right_family].into_iter().flatten() {
                if actual != gamma_family {
                    return Some(qm_error_expr(
                        "sort_spinors_gamma_family_mismatch",
                        expr,
                        interner,
                    ));
                }
            }
        }
    }

    None
}

pub fn sort_spinors(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Mul(factors) => {
            let mapped = factors
                .iter()
                .map(|factor| sort_spinors(factor, properties, interner))
                .collect::<Vec<_>>();

            let mut out = mapped.clone();
            for i in 0..mapped.len() {
                let bar_factor = &mapped[i];
                let Expr::Call(bar_sym, args) = bar_factor else {
                    continue;
                };
                if diracbar_metadata_of_expr(&Expr::Sym(*bar_sym), properties).is_none() {
                    continue;
                }
                if args.len() != 1 || !is_majorana_spinor_expr(&args[0], properties) {
                    continue;
                }
                let left_spinor = args[0].clone();
                let left_sym = match property_sym(&left_spinor) {
                    Some(sym) => sym,
                    None => continue,
                };

                let mut gamma_pos = None;
                let mut spinor_pos = None;
                for j in i + 1..mapped.len() {
                    let candidate = &mapped[j];
                    if diracbar_metadata_of_expr(candidate, properties).is_some() {
                        break;
                    }
                    if gamma_metadata_of_expr(candidate, properties).is_some() {
                        if gamma_pos.is_some() {
                            return qm_error_expr("sort_spinors_join_gamma_first", expr, interner);
                        }
                        gamma_pos = Some(j);
                        continue;
                    }
                    if spinor_metadata_of_expr(candidate, properties).is_some() {
                        spinor_pos = Some(j);
                        break;
                    }
                    if !matches!(candidate, Expr::Int(_) | Expr::Rational(_) | Expr::Float(_)) {
                        break;
                    }
                }

                let Some(j) = spinor_pos else {
                    continue;
                };
                let right_spinor = mapped[j].clone();
                if let Some(error) = sort_spinor_metadata_error(
                    expr,
                    bar_factor,
                    gamma_pos.map(|pos| &mapped[pos]),
                    &left_spinor,
                    &right_spinor,
                    properties,
                    interner,
                ) {
                    return error;
                }
                if !is_majorana_spinor_expr(&right_spinor, properties) {
                    return qm_error_expr("sort_spinors_second_not_majorana", expr, interner);
                }
                let right_sym = match property_sym(&right_spinor) {
                    Some(sym) => sym,
                    None => continue,
                };
                let Some(order) = prop_sort_order(left_sym, properties) else {
                    continue;
                };
                let Some(pos_left) = order.iter().position(|sym| *sym == left_sym) else {
                    continue;
                };
                let Some(pos_right) = order.iter().position(|sym| *sym == right_sym) else {
                    continue;
                };
                if pos_left <= pos_right {
                    continue;
                }

                let gamma_rank = gamma_pos
                    .and_then(|pos| {
                        gamma_expr_data(&mapped[pos], properties).map(|data| data.indices.len())
                    })
                    .unwrap_or(0);
                let majorana_sign = if ((gamma_rank * (gamma_rank + 1)) / 2) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let comparison =
                    ax_tensor::subtree_compare(&left_spinor, &right_spinor, properties, interner);
                let swap_sign = ax_tensor::can_swap(
                    &left_spinor,
                    &right_spinor,
                    comparison,
                    properties,
                    interner,
                    false,
                );
                if swap_sign == 0 {
                    continue;
                }
                let total_sign = majorana_sign * swap_sign;
                out[i] = Expr::Call(*bar_sym, vec![right_spinor.clone()]);
                out[j] = left_spinor.clone();
                let reordered = Expr::mul(out);
                return if total_sign < 0 {
                    Expr::neg(reordered)
                } else {
                    reordered
                };
            }

            Expr::mul(mapped)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| sort_spinors(term, properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(sort_spinors(inner, properties, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            sort_spinors(base, properties, interner),
            sort_spinors(exp, properties, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(sort_spinors(re, properties, interner)),
            Box::new(sort_spinors(im, properties, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| sort_spinors(arg, properties, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(sort_spinors(base, properties, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => {
            Expr::Group(Box::new(sort_spinors(inner, properties, interner)), *rel)
        }
        _ => expr.clone(),
    }
}

pub fn join_gamma_full(
    gam1: &Expr,
    gam2: &Expr,
    dimension: Option<usize>,
    expand: bool,
    use_generalised_delta: bool,
    metric: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let Some(g1) = gamma_expr_data(gam1, properties) else {
        return Expr::mul(vec![gam1.clone(), gam2.clone()]);
    };
    let Some(g2) = gamma_expr_data(gam2, properties) else {
        return Expr::mul(vec![gam1.clone(), gam2.clone()]);
    };

    let rank1 = g1.indices.len();
    let rank2 = g2.indices.len();
    let dim = gamma_effective_dimension(gam1, &g1, gam2, &g2, dimension, properties);

    let families1 = gamma_effective_families(gam1, &g1, properties);
    let families2 = gamma_effective_families(gam2, &g2, properties);
    if !families1.is_empty() && !families2.is_empty() && families1 != families2 {
        return qm_error_expr(
            "join_gamma_family_mismatch",
            &Expr::mul(vec![gam1.clone(), gam2.clone()]),
            interner,
        );
    }

    let mut terms = Vec::new();
    let max_i = rank1.min(rank2);
    for i in 0..=max_i {
        let free_rank = rank1 + rank2 - 2 * i;
        if dim.is_some_and(|d| free_rank > d) {
            continue;
        }
        let coeff = BigRational::new(
            factorial(rank1) * factorial(rank2),
            factorial(rank1 - i) * factorial(rank2 - i) * factorial(i),
        );

        if i == 0 {
            let mut free = g1.indices.clone();
            free.extend(g2.indices.clone());
            let gamma = if free.is_empty() {
                Expr::one()
            } else if gamma_indices_have_duplicate_in_same_family(&free, properties) {
                Expr::zero()
            } else {
                build_gamma_expr(&g1.head, &free)
            };
            if gamma != Expr::zero() {
                terms.push(if coeff.is_one() {
                    gamma
                } else {
                    Expr::mul(vec![Expr::Rational(coeff), gamma])
                });
            }
            continue;
        }

        let left_choices = combinations_of(rank1, i);
        let right_choices = combinations_of(rank2, i);
        let mut contracted_terms = Vec::new();
        for left in &left_choices {
            for right in &right_choices {
                let left_sign = permutation_parity(left);
                let right_sign = permutation_parity(right);
                let mut free = g1
                    .indices
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, item)| (!left.contains(&idx)).then_some(item.clone()))
                    .collect::<Vec<_>>();
                free.extend(
                    g2.indices
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, item)| (!right.contains(&idx)).then_some(item.clone())),
                );
                let gamma_part = if free.is_empty() {
                    Expr::one()
                } else if gamma_indices_have_duplicate_in_same_family(&free, properties) {
                    Expr::zero()
                } else {
                    build_gamma_expr(&g1.head, &free)
                };
                let contraction = if use_generalised_delta {
                    let uppers = left
                        .iter()
                        .map(|idx| g1.indices[*idx].clone())
                        .collect::<Vec<_>>();
                    let lowers = right
                        .iter()
                        .map(|idx| g2.indices[*idx].clone())
                        .collect::<Vec<_>>();
                    build_generalised_delta(&uppers, &lowers, interner)
                } else {
                    let metrics = left
                        .iter()
                        .zip(right.iter())
                        .map(|(li, ri)| {
                            build_metric_contraction(metric, &g1.indices[*li], &g2.indices[*ri])
                        })
                        .collect::<Vec<_>>();
                    if metrics.is_empty() {
                        Expr::one()
                    } else {
                        Expr::mul(metrics)
                    }
                };
                if gamma_part == Expr::zero() {
                    continue;
                }
                let mut term = Expr::mul(vec![gamma_part, contraction]);
                if left_sign * right_sign < 0 {
                    term = Expr::neg(term);
                }
                contracted_terms.push(term);
                if !expand {
                    break;
                }
            }
            if !expand {
                break;
            }
        }

        let contraction_sum = if contracted_terms.len() == 1 {
            contracted_terms.pop().unwrap()
        } else {
            Expr::add(contracted_terms)
        };
        if expand && !coeff.is_one() {
            match contraction_sum {
                Expr::Add(items) => {
                    terms.extend(
                        items
                            .into_iter()
                            .map(|item| Expr::mul(vec![Expr::Rational(coeff.clone()), item])),
                    );
                }
                other => terms.push(Expr::mul(vec![Expr::Rational(coeff), other])),
            }
        } else {
            terms.push(if coeff.is_one() {
                contraction_sum
            } else {
                Expr::mul(vec![Expr::Rational(coeff), contraction_sum])
            });
        }
    }

    if terms.is_empty() {
        Expr::zero()
    } else {
        Expr::add(terms)
    }
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute Fierz rearrangement coefficients for each antisymmetric gamma rank.
///
/// Returns a list of `(coefficient, rank)` pairs for ranks 0..=dim.
/// The coefficient for rank k is:
///   c_k = -(-1)^{k(k+1)/2} * C(d,k) / (k! * spinor_dim)
/// where spinor_dim = 2^(d/2).
pub fn fierz_coefficients(dim: usize) -> Vec<(num_rational::BigRational, usize)> {
    let spinor_dim = 1usize << (dim / 2); // 2^(d/2)
    let mut result = Vec::new();

    for k in 0..=dim {
        let sign = if (k * (k + 1) / 2) % 2 == 0 {
            1i64
        } else {
            -1i64
        };
        let binom = binomial(dim, k);
        let coeff = num_rational::BigRational::new(
            (sign * binom as i64).into(),
            (spinor_dim as i64).into(),
        );
        // Divide by k! for the normalisation of the antisymmetric gamma basis element
        let k_fact: i64 = (1..=k as i64).product();
        let final_coeff = num_rational::BigRational::new(
            coeff.numer().clone(),
            coeff.denom().clone() * num_bigint::BigInt::from(k_fact),
        );
        result.push((final_coeff, k));
    }

    // Overall minus sign from Fierz rearrangement
    for (c, _) in &mut result {
        *c = -c.clone();
    }

    result
}

/// Perform a Fierz rearrangement.
///
/// Given an expression of the form (ψ̄₁ Γ ψ₂)(ψ̄₃ Γ ψ₄), rearrange to
/// a sum over the Fierz basis: Σ_n c_n (ψ̄₁ Γ_n ψ₄)(ψ̄₃ Γ_n ψ₂)
pub fn fierz_rearrange(
    expr: &Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> Expr {
    fierz(expr, dim, spinor_order, interner)
}

fn is_name(name: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| name == *candidate)
}

fn has_property(
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    sym: lasso::Spur,
    property: &TensorProperty,
) -> bool {
    properties
        .map(|props| props.has_property_kind(sym, property))
        .unwrap_or(false)
}

fn expr_head_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Call(sym, _) => Some(*sym),
        Expr::Indexed(base, _) => expr_head_symbol(base),
        _ => None,
    }
}

fn is_dirac_bar_call(
    sym: lasso::Spur,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    properties
        .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(sym), props))
        .is_some()
        || is_name(
            interner.resolve(sym),
            &["dirac_bar", "diracbar", "bar", "DiracBar"],
        )
}

fn barred_spinor_symbol(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => {
            if properties
                .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(*sym), props))
                .is_some()
            {
                return Some(*sym);
            }
            let name = interner.resolve(*sym);
            if name.contains("bar")
                || name.contains("Bar")
                || name.ends_with("bar")
                || name.ends_with("_bar")
                || name.ends_with("Bar")
            {
                Some(*sym)
            } else {
                None
            }
        }
        Expr::Call(f, args) => {
            if is_dirac_bar_call(*f, properties, interner) {
                args.first().and_then(spinor_symbol)
            } else {
                None
            }
        }
        Expr::Indexed(base, _) => barred_spinor_symbol(base, properties, interner),
        _ => None,
    }
}

fn spinor_symbol(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => Some(*sym),
        Expr::Indexed(base, _) => spinor_symbol(base),
        _ => None,
    }
}

fn spinor_symbol_with_properties(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(sym) => {
            if properties
                .map(|props| {
                    spinor_metadata_of_expr(&Expr::Sym(*sym), props).is_some()
                        || props.has_property_kind(*sym, &TensorProperty::AntiCommuting)
                })
                .unwrap_or(true)
            {
                Some(*sym)
            } else {
                None
            }
        }
        Expr::Indexed(base, _) => spinor_symbol_with_properties(base, properties),
        _ => None,
    }
}

fn gamma_factor_indices(
    expr: &Expr,
    gamma_sym: Option<lasso::Spur>,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<Vec<lasso::Spur>> {
    match expr {
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            if Some(*f) == gamma_sym
                || properties
                    .and_then(|props| gamma_metadata_of_expr(&Expr::Sym(*f), props))
                    .is_some()
                || is_name(name, &["gamma", "Gamma", "γ"])
            {
                Some(
                    args.iter()
                        .filter_map(|arg| match arg {
                            Expr::Sym(sym) => Some(*sym),
                            _ => None,
                        })
                        .collect(),
                )
            } else if is_name(name, &["gamma5", "Gamma5", "γ5"]) {
                Some(vec![interner.get_or_intern("5")])
            } else {
                None
            }
        }
        Expr::Indexed(base, indices) => match base.as_ref() {
            Expr::Sym(sym)
                if Some(*sym) == gamma_sym
                    || properties
                        .and_then(|props| gamma_metadata_of_expr(&Expr::Sym(*sym), props))
                        .is_some()
                    || is_name(interner.resolve(*sym), &["gamma", "Gamma", "γ"]) =>
            {
                Some(indices.iter().map(|idx| idx.name).collect())
            }
            _ => None,
        },
        _ => None,
    }
}

fn parse_bilinear_at(
    factors: &[Expr],
    start: usize,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Option<(lasso::Spur, Vec<lasso::Spur>, lasso::Spur, usize, bool)> {
    let barred = barred_spinor_symbol(&factors[start], properties, interner)?;
    let expected_gamma_sym = properties.and_then(|props| {
        diracbar_metadata_of_expr(&factors[start], props).and_then(|metadata| metadata.gamma_symbol)
    });
    let mut gamma_indices = Vec::new();
    let mut cursor = start + 1;
    let mut saw_non_gamma_before_spinor = false;
    while cursor < factors.len() {
        let Some(mut indices) =
            gamma_factor_indices(&factors[cursor], expected_gamma_sym, properties, interner)
        else {
            break;
        };
        gamma_indices.append(&mut indices);
        cursor += 1;
    }
    if cursor >= factors.len() {
        return None;
    }
    let spinor = spinor_symbol_with_properties(&factors[cursor], properties)?;

    let mut trailing_cursor = cursor + 1;
    while trailing_cursor < factors.len() {
        if let Some(mut indices) =
            gamma_factor_indices(&factors[trailing_cursor], None, properties, interner)
        {
            gamma_indices.append(&mut indices);
            saw_non_gamma_before_spinor = true;
            trailing_cursor += 1;
        } else {
            break;
        }
    }

    Some((
        barred,
        gamma_indices,
        spinor,
        trailing_cursor,
        saw_non_gamma_before_spinor,
    ))
}

pub fn find_bilinears(expr: &Expr, interner: &ax_ir::Interner) -> Option<BilinearPair> {
    find_bilinears_impl(expr, None, interner).ok()
}

pub fn find_bilinears_with_properties(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<BilinearPair> {
    find_bilinears_impl(expr, Some(properties), interner).ok()
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedBilinear {
    barred: lasso::Spur,
    gamma_indices: Vec<lasso::Spur>,
    spinor: lasso::Spur,
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedFierzInput {
    pair: BilinearPair,
    sign: i64,
}

fn flatten_mul_factors(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Mul(factors) => {
            for factor in factors {
                flatten_mul_factors(factor, out);
            }
        }
        other => out.push(other.clone()),
    }
}

fn factor_contains_diracbar(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    match expr {
        Expr::Sym(sym) => {
            barred_spinor_symbol(expr, properties, interner).is_some()
                || properties
                    .and_then(|props| diracbar_metadata_of_expr(&Expr::Sym(*sym), props))
                    .is_some()
        }
        Expr::Call(sym, args) => {
            is_dirac_bar_call(*sym, properties, interner)
                || args
                    .iter()
                    .any(|arg| factor_contains_diracbar(arg, properties, interner))
        }
        Expr::Indexed(base, _) => factor_contains_diracbar(base, properties, interner),
        Expr::Mul(factors) | Expr::Add(factors) => factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner)),
        Expr::Neg(inner) => factor_contains_diracbar(inner, properties, interner),
        Expr::Pow(base, exp) => {
            factor_contains_diracbar(base, properties, interner)
                || factor_contains_diracbar(exp, properties, interner)
        }
        Expr::Complex(re, im) => {
            factor_contains_diracbar(re, properties, interner)
                || factor_contains_diracbar(im, properties, interner)
        }
        _ => false,
    }
}

fn is_anticommuting_spinor(
    sym: lasso::Spur,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> bool {
    properties
        .map(|props| props.has_property_kind(sym, &TensorProperty::AntiCommuting))
        .unwrap_or_else(|| {
            let name = interner.resolve(sym);
            name.starts_with("psi")
                || name.starts_with("chi")
                || name.starts_with("theta")
                || name.contains("spinor")
        })
}

fn anticommuting_reorder_sign(
    input_order: &[lasso::Spur],
    output_order: &[lasso::Spur],
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<i64, FierzError> {
    if input_order.len() != output_order.len() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let input_set: HashSet<_> = input_order.iter().copied().collect();
    let output_set: HashSet<_> = output_order.iter().copied().collect();
    if input_set != output_set || input_set.len() != input_order.len() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let mut current = input_order.to_vec();
    let mut sign = 1i64;
    for target_pos in 0..output_order.len() {
        let Some(found_pos) = current[target_pos..]
            .iter()
            .position(|sym| *sym == output_order[target_pos])
            .map(|pos| pos + target_pos)
        else {
            return Err(FierzError::SpinorOrderMismatch);
        };

        for pos in (target_pos..found_pos).rev() {
            if is_anticommuting_spinor(current[pos], properties, interner)
                && is_anticommuting_spinor(current[pos + 1], properties, interner)
            {
                sign = -sign;
            }
            current.swap(pos, pos + 1);
        }
    }
    Ok(sign)
}

fn find_bilinears_impl(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<BilinearPair, FierzError> {
    parse_fierz_input(expr, properties, interner).map(|parsed| parsed.pair)
}

fn parse_fierz_input(
    expr: &Expr,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<ParsedFierzInput, FierzError> {
    let mut factors = Vec::new();
    flatten_mul_factors(expr, &mut factors);
    if factors.len() < 4 {
        if factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner))
        {
            return Err(FierzError::MalformedBilinear);
        }
        return Err(FierzError::NoBilinearPair);
    }

    let mut bilinears: Vec<ParsedBilinear> = Vec::new();
    let mut remaining_factors = Vec::new();
    let mut reordered_within_bilinear = false;
    let mut cursor = 0usize;
    while cursor < factors.len() {
        if bilinears.len() < 2 {
            if let Some((barred, gamma_indices, spinor, next, reordered)) =
                parse_bilinear_at(&factors, cursor, properties, interner)
            {
                bilinears.push(ParsedBilinear {
                    barred,
                    gamma_indices,
                    spinor,
                });
                reordered_within_bilinear |= reordered;
                cursor = next;
                continue;
            }
        }
        remaining_factors.push(factors[cursor].clone());
        cursor += 1;
    }

    if bilinears.len() < 2 {
        if factors
            .iter()
            .any(|factor| factor_contains_diracbar(factor, properties, interner))
        {
            return Err(FierzError::MalformedBilinear);
        }
        return Err(FierzError::NoBilinearPair);
    }

    let mut probe = 0usize;
    let mut total_bilinears = 0usize;
    while probe < factors.len() {
        if let Some((_, _, _, next, _)) = parse_bilinear_at(&factors, probe, properties, interner) {
            total_bilinears += 1;
            probe = next;
        } else {
            probe += 1;
        }
    }
    if total_bilinears > 2 {
        return Err(FierzError::AmbiguousBilinears(total_bilinears));
    }

    let first = bilinears[0].clone();
    let second = bilinears[1].clone();
    let pair = BilinearPair {
        psi1: first.barred,
        gamma_a: first.gamma_indices,
        psi2: first.spinor,
        psi3: second.barred,
        gamma_b: second.gamma_indices,
        psi4: second.spinor,
        remaining_factors,
    };

    let sign = if reordered_within_bilinear { -1 } else { 1 };

    Ok(ParsedFierzInput { pair, sign })
}

fn gamma_index_count(expr: &Expr, gamma_sym: lasso::Spur) -> Option<usize> {
    match expr {
        Expr::Call(f, args) if *f == gamma_sym => Some(args.len()),
        Expr::Indexed(base, indices) if expr_head_symbol(base) == Some(gamma_sym) => {
            Some(indices.len())
        }
        _ => None,
    }
}

fn is_gamma_expr(expr: &Expr, gamma_sym: lasso::Spur) -> bool {
    gamma_index_count(expr, gamma_sym).is_some()
}

fn expand_diracbar_inner(inner: &Expr, diracbar_sym: lasso::Spur, gamma_sym: lasso::Spur) -> Expr {
    if let Expr::Neg(nested) = inner {
        return Expr::neg(expand_diracbar_inner(nested, diracbar_sym, gamma_sym));
    }

    let Expr::Mul(factors) = inner else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    if factors.len() > 1 {
        if let Expr::Int(n) = &factors[0] {
            if *n == (-1).into() {
                return Expr::neg(expand_diracbar_inner(
                    &Expr::mul(factors[1..].to_vec()),
                    diracbar_sym,
                    gamma_sym,
                ));
            }
        }
    }

    let mut gamma_chain = Vec::new();
    let mut spinor = None;
    for factor in factors {
        if is_gamma_expr(factor, gamma_sym) && spinor.is_none() {
            gamma_chain.push(factor.clone());
        } else if spinor.is_none() {
            spinor = Some(factor.clone());
        } else {
            return Expr::Call(diracbar_sym, vec![inner.clone()]);
        }
    }

    if gamma_chain.is_empty() {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    }

    let Some(spinor) = spinor else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    let total_gamma_indices: usize = gamma_chain
        .iter()
        .filter_map(|gamma| gamma_index_count(gamma, gamma_sym))
        .sum();
    let mut factors = vec![Expr::Call(diracbar_sym, vec![spinor])];
    factors.extend(gamma_chain.into_iter().rev());
    let result = Expr::mul(factors);

    if (total_gamma_indices * total_gamma_indices.saturating_sub(1) / 2) % 2 == 1 {
        Expr::neg(result)
    } else {
        result
    }
}

pub fn expand_diracbar(
    expr: &Expr,
    diracbar_sym: lasso::Spur,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = (metric_sym, interner);
    match expr {
        Expr::Call(f, args) if *f == diracbar_sym && args.len() == 1 => {
            let inner = expand_diracbar(&args[0], diracbar_sym, gamma_sym, metric_sym, interner);
            expand_diracbar_inner(&inner, diracbar_sym, gamma_sym)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| expand_diracbar(term, diracbar_sym, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| {
                    expand_diracbar(factor, diracbar_sym, gamma_sym, metric_sym, interner)
                })
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(expand_diracbar(
            inner,
            diracbar_sym,
            gamma_sym,
            metric_sym,
            interner,
        )),
        Expr::Pow(base, exp) => Expr::pow(
            expand_diracbar(base, diracbar_sym, gamma_sym, metric_sym, interner),
            expand_diracbar(exp, diracbar_sym, gamma_sym, metric_sym, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_diracbar(
                re,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
            Box::new(expand_diracbar(
                im,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| expand_diracbar(arg, diracbar_sym, gamma_sym, metric_sym, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(expand_diracbar(
                base,
                diracbar_sym,
                gamma_sym,
                metric_sym,
                interner,
            )),
            indices.clone(),
        ),
        _ => expr.clone(),
    }
}

fn expand_diracbar_full_inner(
    inner: &Expr,
    diracbar_sym: lasso::Spur,
    metadata: &DiracBarMetadata,
    properties: &dyn ax_tensor::PropertyLookup,
) -> Expr {
    if let Expr::Neg(nested) = inner {
        return Expr::neg(expand_diracbar_full_inner(
            nested,
            diracbar_sym,
            metadata,
            properties,
        ));
    }

    let Expr::Mul(factors) = inner else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    if factors.len() > 1 {
        if let Expr::Int(n) = &factors[0] {
            if *n == (-1).into() {
                return Expr::neg(expand_diracbar_full_inner(
                    &Expr::mul(factors[1..].to_vec()),
                    diracbar_sym,
                    metadata,
                    properties,
                ));
            }
        }
    }

    let mut gamma_chain = Vec::new();
    let mut spinor = None;
    for factor in factors {
        if gamma_metadata_of_expr(factor, properties).is_some() && spinor.is_none() {
            if let Some(expected_gamma_symbol) = metadata.gamma_symbol {
                if property_sym(factor) != Some(expected_gamma_symbol) {
                    return Expr::Call(diracbar_sym, vec![inner.clone()]);
                }
            }
            gamma_chain.push(factor.clone());
        } else if spinor.is_none() {
            if let Some(expected_spinor_family) = metadata.spinor_family {
                if structured_spinor_family(factor, properties)
                    .is_some_and(|family| family != expected_spinor_family)
                {
                    return Expr::Call(diracbar_sym, vec![inner.clone()]);
                }
            }
            spinor = Some(factor.clone());
        } else {
            return Expr::Call(diracbar_sym, vec![inner.clone()]);
        }
    }

    if gamma_chain.is_empty() {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    }
    let Some(spinor) = spinor else {
        return Expr::Call(diracbar_sym, vec![inner.clone()]);
    };

    let total_gamma_indices: usize = gamma_chain
        .iter()
        .filter_map(|gamma| gamma_expr_data(gamma, properties).map(|data| data.indices.len()))
        .sum();
    let ordered_gamma_chain = if metadata.reverse_gamma_order {
        gamma_chain.into_iter().rev().collect::<Vec<_>>()
    } else {
        gamma_chain
    };
    let mut factors = vec![Expr::Call(diracbar_sym, vec![spinor])];
    factors.extend(ordered_gamma_chain);
    let result = Expr::mul(factors);

    if metadata.reverse_gamma_order
        && ((total_gamma_indices * (total_gamma_indices + 1)) / 2) % 2 == 1
    {
        Expr::neg(result)
    } else {
        result
    }
}

pub fn expand_diracbar_full(
    expr: &Expr,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    match expr {
        Expr::Call(f, args) if args.len() == 1 => {
            let Some(metadata) = diracbar_metadata_of_expr(&Expr::Sym(*f), properties) else {
                return Expr::Call(
                    *f,
                    args.iter()
                        .map(|arg| expand_diracbar_full(arg, properties, interner))
                        .collect(),
                );
            };
            let inner = expand_diracbar_full(&args[0], properties, interner);
            expand_diracbar_full_inner(&inner, *f, &metadata, properties)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| expand_diracbar_full(term, properties, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| expand_diracbar_full(factor, properties, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(expand_diracbar_full(inner, properties, interner)),
        Expr::Pow(base, exp) => Expr::pow(
            expand_diracbar_full(base, properties, interner),
            expand_diracbar_full(exp, properties, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(expand_diracbar_full(re, properties, interner)),
            Box::new(expand_diracbar_full(im, properties, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| expand_diracbar_full(arg, properties, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(expand_diracbar_full(base, properties, interner)),
            indices.clone(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(expand_diracbar_full(inner, properties, interner)),
            *rel,
        ),
        _ => expr.clone(),
    }
}

pub fn diracbar_sort(
    expr: &Expr,
    diracbar_sym: lasso::Spur,
    gamma_sym: lasso::Spur,
    operators: &HashMap<lasso::Spur, OperatorKind>,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = (operators, interner);
    match expr {
        Expr::Mul(factors) => {
            let sorted = factors
                .iter()
                .map(|factor| diracbar_sort(factor, diracbar_sym, gamma_sym, operators, interner))
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            let mut cursor = 0usize;
            while cursor < sorted.len() {
                let factor = &sorted[cursor];
                if matches!(factor, Expr::Call(f, _) if *f == diracbar_sym) {
                    out.push(factor.clone());
                    cursor += 1;
                    let mut gammas = Vec::new();
                    let mut spinor = None;
                    let mut others = Vec::new();
                    while cursor < sorted.len() {
                        let next = &sorted[cursor];
                        if matches!(next, Expr::Call(f, _) if *f == diracbar_sym) {
                            break;
                        }
                        if is_gamma_expr(next, gamma_sym) {
                            gammas.push(next.clone());
                        } else if spinor.is_none() {
                            spinor = Some(next.clone());
                        } else {
                            others.push(next.clone());
                        }
                        cursor += 1;
                    }
                    out.extend(gammas);
                    if let Some(spinor) = spinor {
                        out.push(spinor);
                    }
                    out.extend(others);
                } else {
                    out.push(factor.clone());
                    cursor += 1;
                }
            }
            Expr::mul(out)
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| diracbar_sort(term, diracbar_sym, gamma_sym, operators, interner))
                .collect(),
        ),
        Expr::Neg(inner) => Expr::neg(diracbar_sort(
            inner,
            diracbar_sym,
            gamma_sym,
            operators,
            interner,
        )),
        Expr::Pow(base, exp) => Expr::pow(
            diracbar_sort(base, diracbar_sym, gamma_sym, operators, interner),
            diracbar_sort(exp, diracbar_sym, gamma_sym, operators, interner),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(diracbar_sort(
                re,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
            Box::new(diracbar_sort(
                im,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| diracbar_sort(arg, diracbar_sym, gamma_sym, operators, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(diracbar_sort(
                base,
                diracbar_sym,
                gamma_sym,
                operators,
                interner,
            )),
            indices.clone(),
        ),
        _ => expr.clone(),
    }
}

pub fn fierz_full(
    expr: &Expr,
    spinor_order: &[Expr; 4],
    dimension: usize,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let parsed = parse_fierz_input(expr, Some(properties), interner).ok()?;
    let desired = spinor_order
        .iter()
        .map(property_sym)
        .collect::<Option<Vec<_>>>()?;
    let desired = [desired[0], desired[1], desired[2], desired[3]];
    let right_order = [
        parsed.pair.psi1,
        parsed.pair.psi2,
        parsed.pair.psi3,
        parsed.pair.psi4,
    ];
    let wrong_order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    if desired == right_order {
        return None;
    }
    if desired != wrong_order {
        return None;
    }

    let spinor_dim = if dimension % 2 == 0 {
        1usize << (dimension / 2)
    } else {
        1usize << ((dimension - 1) / 2)
    };
    let use_weyl = spinor_order
        .iter()
        .all(|spinor| is_weyl_spinor_expr(spinor, properties));
    let max_rank = if use_weyl { dimension / 2 } else { dimension };
    let gamma_sym = property_sym(&Expr::Call(interner.get_or_intern("gamma"), vec![]))
        .unwrap_or_else(|| interner.get_or_intern("gamma"));

    let mut used = HashSet::new();
    collect_all_index_names(expr, &mut used);
    let family = properties
        .index_families()
        .and_then(|families| families.values().next().cloned());

    let mut terms = Vec::new();
    for rank in 0..=max_rank {
        let coeff = -BigRational::new(BigInt::one(), BigInt::from(spinor_dim) * factorial(rank));
        let gamma_indices = if let Some(info) = &family {
            (0..rank)
                .map(|_| fresh_dummy_from_family(info, &mut used, interner))
                .collect::<Vec<_>>()
        } else {
            (0..rank)
                .map(|idx| interner.get_or_intern(&format!("_fierz{rank}_{idx}")))
                .collect::<Vec<_>>()
        };

        let first_gamma = if gamma_indices.is_empty() {
            None
        } else {
            Some(Expr::Call(
                gamma_sym,
                gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
            ))
        };
        let mut second_chain = Vec::new();
        if !parsed.pair.gamma_a.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                parsed
                    .pair
                    .gamma_a
                    .iter()
                    .map(|idx| Expr::Sym(*idx))
                    .collect(),
            ));
        }
        if !gamma_indices.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
            ));
        }
        if !parsed.pair.gamma_b.is_empty() {
            second_chain.push(Expr::Call(
                gamma_sym,
                parsed
                    .pair
                    .gamma_b
                    .iter()
                    .rev()
                    .map(|idx| Expr::Sym(*idx))
                    .collect(),
            ));
        }

        let mut first_bilinear = vec![Expr::Sym(desired[0])];
        if let Some(gamma) = first_gamma {
            first_bilinear.push(gamma);
        }
        first_bilinear.push(Expr::Sym(desired[1]));

        let mut second_bilinear = vec![Expr::Sym(desired[2])];
        second_bilinear.extend(second_chain);
        second_bilinear.push(Expr::Sym(desired[3]));

        let mut term_factors = parsed.pair.remaining_factors.clone();
        term_factors.push(Expr::Rational(coeff));
        term_factors.push(Expr::mul(first_bilinear));
        term_factors.push(Expr::mul(second_bilinear));
        terms.push(Expr::mul(term_factors));
    }

    Some(Expr::add(terms))
}

pub fn split_gamma_full(
    gamma_expr: &Expr,
    on_back: bool,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    let Some(data) = gamma_expr_data(gamma_expr, properties) else {
        return gamma_expr.clone();
    };
    if data.indices.len() <= 1 {
        return gamma_expr.clone();
    }

    let metric = gamma_metadata_of_expr(gamma_expr, properties)
        .and_then(|metadata| metadata.metric_symbol)
        .map(Expr::Sym)
        .unwrap_or_else(|| Expr::Sym(interner.get_or_intern("g")));
    let (left_indices, right_indices) = if on_back {
        (
            data.indices[..data.indices.len() - 1].to_vec(),
            vec![data.indices[data.indices.len() - 1].clone()],
        )
    } else {
        (vec![data.indices[0].clone()], data.indices[1..].to_vec())
    };
    let lhs = build_gamma_expr(&data.head, &left_indices);
    let rhs = build_gamma_expr(&data.head, &right_indices);
    let product = Expr::mul(vec![lhs.clone(), rhs.clone()]);
    let joined = join_gamma_full(&lhs, &rhs, None, true, true, &metric, properties, interner);
    if matches!(joined, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_family_mismatch")
    {
        return joined;
    }

    let joined_terms = match joined {
        Expr::Add(terms) => terms,
        other => vec![other],
    };
    let original_data = gamma_expr_data(gamma_expr, properties);
    let mut rest = Vec::new();
    for term in joined_terms {
        let rank_match = match &term {
            Expr::Mul(factors) => factors.iter().any(|factor| {
                gamma_expr_data(factor, properties)
                    .map(|candidate| {
                        original_data
                            .as_ref()
                            .map(|d| candidate.indices == d.indices)
                            .unwrap_or(false)
                    })
                    .unwrap_or(false)
            }),
            _ => gamma_expr_data(&term, properties)
                .map(|candidate| {
                    original_data
                        .as_ref()
                        .map(|d| candidate.indices == d.indices)
                        .unwrap_or(false)
                })
                .unwrap_or(false),
        };
        if !rank_match {
            rest.push(Expr::neg(term));
        }
    }
    let mut out_terms = vec![product];
    out_terms.extend(rest);
    Expr::add(out_terms)
}

fn fresh_fierz_indices(
    rank: usize,
    counter: &mut usize,
    interner: &ax_ir::Interner,
) -> Vec<lasso::Spur> {
    (0..rank)
        .map(|_| {
            let name = format!("_f{}", *counter);
            *counter += 1;
            interner.get_or_intern(&name)
        })
        .collect()
}

fn bilinear_expr(
    left: lasso::Spur,
    gamma_indices: &[lasso::Spur],
    right: lasso::Spur,
    gamma_sym: lasso::Spur,
) -> Expr {
    let mut factors = vec![Expr::Sym(left)];
    if !gamma_indices.is_empty() {
        factors.push(Expr::Call(
            gamma_sym,
            gamma_indices.iter().map(|idx| Expr::Sym(*idx)).collect(),
        ));
    }
    factors.push(Expr::Sym(right));
    Expr::mul(factors)
}

fn fierz_error_expr(error: &FierzError, expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let sym = interner.get_or_intern(error.symbol_name());
    Expr::Call(sym, vec![expr.clone()])
}

fn validate_fierz_spinor_metadata(
    pair: &BilinearPair,
    properties: Option<&dyn ax_tensor::PropertyLookup>,
) -> Result<(), FierzError> {
    let Some(properties) = properties else {
        return Ok(());
    };

    let spinors = [pair.psi1, pair.psi2, pair.psi3, pair.psi4]
        .into_iter()
        .map(|sym| spinor_metadata_of_expr(&Expr::Sym(sym), properties))
        .collect::<Vec<_>>();

    let dimensions = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().and_then(|metadata| metadata.dimension))
        .collect::<HashSet<_>>();
    if dimensions.len() > 1 {
        return Err(FierzError::IncompatibleSpinorDimension);
    }

    let chiralities = spinors
        .iter()
        .filter_map(|metadata| {
            metadata
                .as_ref()
                .and_then(|metadata| metadata.chirality.clone())
        })
        .collect::<HashSet<_>>();
    if chiralities.len() > 1 {
        return Err(FierzError::IncompatibleSpinorChirality);
    }

    let classes = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().map(|metadata| metadata.class.clone()))
        .collect::<HashSet<_>>();
    if classes.len() > 1 {
        return Err(FierzError::IncompatibleSpinorMetadata);
    }

    let families = spinors
        .iter()
        .filter_map(|metadata| metadata.as_ref().and_then(|metadata| metadata.index_family))
        .collect::<HashSet<_>>();
    if families.len() > 1 {
        return Err(FierzError::IncompatibleSpinorMetadata);
    }

    Ok(())
}

fn build_fierz_sum(
    parsed: ParsedFierzInput,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: Option<&dyn ax_tensor::PropertyLookup>,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let pair = parsed.pair;
    let expected = [pair.psi1, pair.psi4, pair.psi3, pair.psi2];
    let explicit_set: HashSet<_> = spinor_order.iter().copied().collect();
    if explicit_set.len() != 4 || explicit_set != expected.iter().copied().collect() {
        return Err(FierzError::SpinorOrderMismatch);
    }

    let input_order = [pair.psi1, pair.psi2, pair.psi3, pair.psi4];
    let mut sign = anticommuting_reorder_sign(&input_order, &spinor_order, properties, interner)?;
    if parsed.sign < 0 {
        sign = -sign;
    }

    let coeffs = fierz_coefficients(dim);
    let gamma_sym = interner.get_or_intern("gamma");
    let [psi1, psi4, psi3, psi2] = spinor_order;
    let mut counter = 0usize;

    let terms = coeffs
        .into_iter()
        .map(|(coefficient, rank)| {
            let gamma_indices = fresh_fierz_indices(rank, &mut counter, interner);
            let first = bilinear_expr(psi1, &gamma_indices, psi4, gamma_sym);
            let second = bilinear_expr(psi3, &gamma_indices, psi2, gamma_sym);

            let mut factors = pair.remaining_factors.clone();
            let signed_coefficient = if sign < 0 { -coefficient } else { coefficient };
            factors.push(Expr::Rational(signed_coefficient));
            factors.push(first);
            factors.push(second);
            Expr::mul(factors)
        })
        .collect();
    Ok(ax_ir::Expr::add(terms))
}

pub fn try_fierz(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, None, interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, None)?;
    build_fierz_sum(parsed, dim, spinor_order, None, interner)
}

pub fn try_fierz_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, Some(properties), interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, Some(properties))?;
    build_fierz_sum(parsed, dim, spinor_order, Some(properties), interner)
}

/// Apply Fierz identity to a concrete product of two spinor bilinears.
pub fn fierz(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match try_fierz(expr, dim, spinor_order, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

pub fn fierz_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    spinor_order: [lasso::Spur; 4],
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> ax_ir::Expr {
    match try_fierz_with_properties(expr, dim, spinor_order, properties, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

pub fn try_fierz_auto(
    expr: &ax_ir::Expr,
    dim: usize,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, None, interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, None)?;
    let order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    build_fierz_sum(parsed, dim, order, None, interner)
}

pub fn try_fierz_auto_with_properties(
    expr: &ax_ir::Expr,
    dim: usize,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, FierzError> {
    let parsed = parse_fierz_input(expr, Some(properties), interner)?;
    validate_fierz_spinor_metadata(&parsed.pair, Some(properties))?;
    let order = [
        parsed.pair.psi1,
        parsed.pair.psi4,
        parsed.pair.psi3,
        parsed.pair.psi2,
    ];
    build_fierz_sum(parsed, dim, order, Some(properties), interner)
}

pub fn fierz_auto(expr: &ax_ir::Expr, dim: usize, interner: &ax_ir::Interner) -> ax_ir::Expr {
    match try_fierz_auto(expr, dim, interner) {
        Ok(result) => result,
        Err(error) => fierz_error_expr(&error, expr, interner),
    }
}

/// Return the abstract Fierz coefficient expansion used by the old API.
pub fn fierz_simple(dim: usize, interner: &ax_ir::Interner) -> ax_ir::Expr {
    let coeffs = fierz_coefficients(dim);
    let terms: Vec<ax_ir::Expr> = coeffs
        .iter()
        .map(|(c, k)| {
            ax_ir::Expr::mul(vec![
                ax_ir::Expr::Rational(c.clone()),
                ax_ir::Expr::Call(
                    interner.get_or_intern("gamma_basis"),
                    vec![ax_ir::Expr::Int(BigInt::from(*k))],
                ),
            ])
        })
        .collect();
    ax_ir::Expr::add(terms)
}

// ─── split_gamma ──────────────────────────────────────────────────────────────

/// Split one index off a multi-index antisymmetric gamma matrix.
///
/// Uses the join identity in reverse:
/// ```text
/// γ^{a} γ^{b…z} = γ^{a b…z} + contraction terms
/// γ^{a b…z} γ^{z} = γ^{a b…} + contraction terms
/// ```
/// So:
/// ```text
/// γ^{a b…z} = γ^{a b…} γ^{z} − (contraction terms)   [on_back = true]
/// γ^{a b…z} = γ^{a} γ^{b…z} − (contraction terms)   [on_back = false]
/// ```
///
/// Parameters:
/// - `gamma_sym`: symbol for the gamma matrix
/// - `metric_sym`: symbol for the metric used in contractions
/// - `on_back`: if `true`, split the last index; if `false`, split the first
pub fn split_gamma(
    expr: &Expr,
    gamma_sym: lasso::Spur,
    metric_sym: lasso::Spur,
    on_back: bool,
    interner: &ax_ir::Interner,
) -> Expr {
    let _ = interner;
    match expr {
        Expr::Call(f, args) if *f == gamma_sym && args.len() > 1 => {
            let indices: Vec<lasso::Spur> = args
                .iter()
                .filter_map(|a| if let Expr::Sym(s) = a { Some(*s) } else { None })
                .collect();

            if indices.len() <= 1 {
                return expr.clone();
            }

            // Choose which index to split off and what remains
            let (split_idx, remaining_indices) = if on_back {
                let last = *indices.last().unwrap();
                (last, indices[..indices.len() - 1].to_vec())
            } else {
                let first = indices[0];
                (first, indices[1..].to_vec())
            };

            // Main term: γ(remaining) * γ(split)  [on_back]
            //         or γ(split) * γ(remaining)  [on_front]
            let main = if on_back {
                Expr::mul(vec![
                    make_gamma(&remaining_indices, gamma_sym),
                    make_gamma(&[split_idx], gamma_sym),
                ])
            } else {
                Expr::mul(vec![
                    make_gamma(&[split_idx], gamma_sym),
                    make_gamma(&remaining_indices, gamma_sym),
                ])
            };

            // Contraction terms come from the join identity:
            //   γ(remaining) γ(split) = γ(full) + Σ_k (±1) g^{split rem_k} γ(remaining \ rem_k)
            // Rearranging: γ(full) = main − Σ_k (±1) g^{split rem_k} γ(remaining \ rem_k)
            //
            // Signs: k-th contraction gets (-1)^k when splitting from back,
            //        and (-1)^k when splitting from front (same rule, position counts from 0).
            let mut all_terms = vec![main];

            for (k, &rem_idx) in remaining_indices.iter().enumerate() {
                // Sign: (-1)^k  (k is 0-based position in remaining)
                let negate = k % 2 != 0;

                let metric = Expr::Indexed(
                    Box::new(Expr::Sym(metric_sym)),
                    vec![
                        Index {
                            name: split_idx,
                            variance: Variance::Up,
                            index_type: None,
                        },
                        Index {
                            name: rem_idx,
                            variance: Variance::Up,
                            index_type: None,
                        },
                    ],
                );

                // Sub-gamma: remaining indices with rem_idx removed
                let sub_remaining: Vec<lasso::Spur> = remaining_indices
                    .iter()
                    .filter(|&&i| i != rem_idx)
                    .copied()
                    .collect();

                let sub_gamma = if sub_remaining.is_empty() {
                    Expr::one()
                } else {
                    make_gamma(&sub_remaining, gamma_sym)
                };

                let contraction = Expr::mul(vec![metric, sub_gamma]);
                // Subtract the contraction: − (±contraction)
                let signed = if negate {
                    contraction
                } else {
                    Expr::neg(contraction)
                };
                all_terms.push(signed);
            }

            Expr::add(all_terms)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| split_gamma(f, gamma_sym, metric_sym, on_back, interner))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| split_gamma(t, gamma_sym, metric_sym, on_back, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(split_gamma(e, gamma_sym, metric_sym, on_back, interner)),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prop_map() -> HashMap<lasso::Spur, Vec<TensorProperty>> {
        HashMap::new()
    }

    fn operator_stats() -> HashMap<lasso::Spur, OperatorStatistics> {
        HashMap::new()
    }

    #[test]
    fn pauli_commutation() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let sy = pauli_y(&interner);
        let comm = commutator(&sx, &sy, &interner);
        let simplified = ax_eval::eval(&comm[0][0], &ax_eval::Env::new(), &interner);
        let expected = Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::Int(2.into())));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn anticommutator_pauli() {
        let interner = ax_ir::Interner::new();
        let sx = pauli_x(&interner);
        let anti = anticommutator(&sx, &sx, &interner);
        let simplified_00 = ax_eval::eval(&anti[0][0], &ax_eval::Env::new(), &interner);
        assert_eq!(simplified_00, Expr::Int(2.into()));
    }

    #[test]
    fn ket_basis_vectors() {
        let interner = ax_ir::Interner::new();
        let ket0 = vec![Expr::one(), Expr::zero()];
        let ket1 = vec![Expr::zero(), Expr::one()];
        let inner = Expr::add(
            ket0.iter()
                .zip(ket1.iter())
                .map(|(a, b)| Expr::mul(vec![a.clone(), b.clone()]))
                .collect::<Vec<_>>(),
        );
        let result = ax_eval::eval(&inner, &ax_eval::Env::new(), &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn trace_identity() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let result = gamma_trace_recursive(&[], g, &interner);
        assert_eq!(result, Expr::Int(4.into()));
    }

    #[test]
    fn trace_single_gamma_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let result = gamma_trace_recursive(&[mu], g, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn trace_two_gammas() {
        let interner = ax_ir::Interner::new();
        let g_sym = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let result = gamma_trace_recursive(&[mu, nu], g_sym, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("4") && pp.contains("g"), "got: {}", pp);
    }

    #[test]
    fn trace_odd_is_zero() {
        let interner = ax_ir::Interner::new();
        let g = interner.get_or_intern("g");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");
        let result = gamma_trace_recursive(&[mu, nu, rho], g, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn normal_order_puts_creation_first() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);
        let statistics = operator_stats();

        let expr = Expr::mul(vec![Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &statistics, &interner);
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2);
            assert_eq!(factors[0], Expr::Sym(a_dag));
            assert_eq!(factors[1], Expr::Sym(a));
        } else {
            panic!("expected Mul");
        }
    }

    #[test]
    fn normal_order_preserves_scalars() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("a");
        let a_dag = interner.get_or_intern("a_dag");

        let mut operators = HashMap::new();
        operators.insert(a, OperatorKind::Annihilation);
        operators.insert(a_dag, OperatorKind::Creation);
        let statistics = operator_stats();

        let expr = Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(a), Expr::Sym(a_dag)]);
        let result = normal_order_simple(&expr, &operators, &statistics, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("3"), "got: {}", pp);
    }

    #[test]
    fn normal_order_bosonic_same_mode_adds_plus_identity() {
        let interner = ax_ir::Interner::new();
        let operators = HashMap::new();
        let statistics = operator_stats();
        let a = interner.get_or_intern("a");
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
        ]);
        let result = normal_order_simple(&expr, &operators, &statistics, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(a)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(a)]),
            ]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn normal_order_fermionic_same_mode_adds_minus_identity_term() {
        let interner = ax_ir::Interner::new();
        let operators = HashMap::new();
        let c = interner.get_or_intern("c");
        let mut statistics = operator_stats();
        statistics.insert(c, OperatorStatistics::Fermionic);
        let expr = Expr::mul(vec![
            Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
        ]);
        let result = normal_order_simple(&expr, &operators, &statistics, &interner);
        let expected = Expr::add(vec![
            Expr::one(),
            Expr::neg(Expr::mul(vec![
                Expr::Call(interner.get_or_intern("creation"), vec![Expr::Sym(c)]),
                Expr::Call(interner.get_or_intern("annihilation"), vec![Expr::Sym(c)]),
            ])),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn join_two_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // γ^a γ^b = γ^{ab} + g^{ab}
        let result = join_gamma_pair(&[a], &[b], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                2,
                "expected γ^{{ab}} + g^{{ab}}, got {terms:?}"
            );
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_three_gammas() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // γ^a γ^{bc} = γ^{abc} + g^{ab} γ^c - g^{ac} γ^b
        let result = join_gamma_pair(&[a], &[b, c], gamma, g, &interner);
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 3, "expected 3 terms, got {terms:?}");
        } else {
            panic!("expected Add, got {result:?}");
        }
    }

    #[test]
    fn join_empty_left_is_identity() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");

        // Identity * γ^a = γ^a
        let result = join_gamma_pair(&[], &[a], gamma, g, &interner);
        assert_eq!(result, Expr::Call(gamma, vec![Expr::Sym(a)]));
    }

    #[test]
    fn join_gammas_in_product() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        // join_gammas_in_expr(gamma(a) * gamma(b)) → Add(...)
        let expr = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
        ]);
        let result = join_gammas_in_expr(&expr, gamma, g, &interner);
        assert!(
            matches!(result, Expr::Add(_)),
            "expected Add, got {result:?}"
        );
    }

    #[test]
    fn expand_bar_single_gamma() {
        // bar(gamma(a) psi) = bar(psi) gamma(a)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let inner = Expr::mul(vec![Expr::Call(gamma, vec![Expr::Sym(a)]), Expr::Sym(psi)]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let result_str = ax_ir::pretty_print(&result, &interner);
        assert!(
            result_str.contains("bar") && result_str.contains("gamma"),
            "should contain bar(psi) and gamma, got {}",
            result_str
        );
    }

    #[test]
    fn expand_bar_double_gamma_reverses() {
        // bar(gamma(a) gamma(b) psi) = -bar(psi) gamma(b) gamma(a)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let inner = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Sym(psi),
        ]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let result_str = format!("{:?}", result);
        assert!(
            result_str.contains("Neg") || result_str.contains("-1"),
            "double gamma reversal should introduce a sign, got {}",
            result_str
        );
    }

    #[test]
    fn expand_bar_multi_index_gamma_chain_reverses_with_total_rank_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let inner = Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Sym(psi),
        ]);
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let expected = Expr::neg(Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]));
        assert_eq!(
            result, expected,
            "rank-3 gamma chain should reverse and pick a minus sign"
        );
    }

    #[test]
    fn expand_bar_nested_negative_chain_keeps_transpose_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let inner = Expr::neg(Expr::mul(vec![
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Sym(psi),
        ]));
        let expr = Expr::Call(bar, vec![inner]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]);
        assert_eq!(
            result, expected,
            "explicit minus and two-gamma transpose minus should cancel"
        );
    }

    #[test]
    fn expand_bar_no_gamma() {
        // bar(psi) should stay as bar(psi)
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let expr = Expr::Call(bar, vec![Expr::Sym(psi)]);
        let result = expand_diracbar(&expr, bar, gamma, interner.get_or_intern("eta"), &interner);
        assert_eq!(result, expr, "bar(psi) with no gammas should be unchanged");
    }

    #[test]
    fn fierz_coefficients_4d() {
        let coeffs = fierz_coefficients(4);
        // ranks 0, 1, 2, 3, 4 in 4D → 5 entries
        assert_eq!(coeffs.len(), 5);
        // Verify ranks are 0..=4
        for (i, (_, rank)) in coeffs.iter().enumerate() {
            assert_eq!(*rank, i);
        }
        // spinor_dim = 4; overall minus; check signs
        // k=0: sign=(0%2==0)→+1, binom=1, coeff_raw=1/4, k!=1 → raw=1/4, after minus: -1/4
        // k=1: sign=(1%2==1)→-1, binom=4, coeff_raw=-1/1, k!=1 → raw=-1/1, after minus: 1/1
        // k=2: sign=(3%2==1)→-1, binom=6, coeff_raw=-3/2, k!=2 → raw=-3/4, after minus: 3/4
        // k=3: sign=(6%2==0)→+1, binom=4, coeff_raw=1/1, k!=6 → raw=1/6, after minus: -1/6
        // k=4: sign=(10%2==0)→+1, binom=1, coeff_raw=1/4, k!=24 → raw=1/96, after minus: -1/96
        let expected: Vec<num_rational::BigRational> = vec![
            num_rational::BigRational::new((-1i64).into(), 4i64.into()),
            num_rational::BigRational::new(1i64.into(), 1i64.into()),
            num_rational::BigRational::new(3i64.into(), 4i64.into()),
            num_rational::BigRational::new((-1i64).into(), 6i64.into()),
            num_rational::BigRational::new((-1i64).into(), 96i64.into()),
        ];
        for (i, (c, _)) in coeffs.iter().enumerate() {
            assert_eq!(c, &expected[i], "mismatch at rank {i}");
        }
    }

    #[test]
    fn fierz_coefficients_sum_check() {
        // In d=4, the 16 gamma matrix basis elements are counted by C(4,k):
        // C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1 = 16 = spinor_dim^2
        let dim = 4;
        let coeffs = fierz_coefficients(dim);
        assert!(!coeffs.is_empty());
        assert_eq!(coeffs.len(), dim + 1);
        // Completeness: sum of |c_k| * C(d,k) * k! * spinor_dim should equal total basis size
        // As a basic sanity check, verify no coefficient is zero
        for (c, _) in &coeffs {
            assert_ne!(*c, num_rational::BigRational::new(0i64.into(), 1i64.into()));
        }
    }

    #[test]
    fn fierz_4d_unit_unit() {
        // (psibar1 psi2)(psibar3 psi4) Fierz rearranged in 4D.
        let interner = ax_ir::Interner::new();
        let coeffs = fierz_coefficients(4);
        let total_basis: usize = coeffs.iter().map(|(_, k)| binomial(4, *k) as usize).sum();
        assert_eq!(total_basis, 16, "total gamma basis size in 4D should be 16");

        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");
        let expr = Expr::mul(vec![
            Expr::Sym(psibar1),
            Expr::Sym(psi2),
            Expr::Sym(psibar3),
            Expr::Sym(psi4),
        ]);
        let result = fierz(&expr, 4, [psibar1, psi4, psibar3, psi2], &interner);
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), coeffs.len()),
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    fn collect_rationals(expr: &Expr, out: &mut Vec<num_rational::BigRational>) {
        match expr {
            Expr::Rational(value) => out.push(value.clone()),
            Expr::Mul(factors) | Expr::Add(factors) => {
                for factor in factors {
                    collect_rationals(factor, out);
                }
            }
            Expr::Neg(inner) => {
                let mut nested = Vec::new();
                collect_rationals(inner, &mut nested);
                out.extend(nested.into_iter().map(|value| -value));
            }
            _ => {}
        }
    }

    #[test]
    fn fierz_detects_nontrivial_gamma_chains() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let expr = Expr::mul(vec![
            Expr::Sym(psibar1),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Call(gamma, vec![Expr::Sym(b), Expr::Sym(c)]),
            Expr::Sym(psi2),
            Expr::Sym(psibar3),
            Expr::Call(gamma, vec![Expr::Sym(d)]),
            Expr::Sym(psi4),
        ]);
        let pair = find_bilinears(&expr, &interner).expect("gamma-chain bilinears should parse");
        assert_eq!(pair.gamma_a, vec![a, b, c]);
        assert_eq!(pair.gamma_b, vec![d]);

        let result = fierz_auto(&expr, 4, &interner);
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), fierz_coefficients(4).len()),
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    #[test]
    fn fierz_auto_infers_standard_spinor_order_in_nested_product() {
        let interner = ax_ir::Interner::new();
        let scalar = interner.get_or_intern("m");
        let psibar1 = interner.get_or_intern("psibar1");
        let psi2 = interner.get_or_intern("psi2");
        let psibar3 = interner.get_or_intern("psibar3");
        let psi4 = interner.get_or_intern("psi4");

        let expr = Expr::mul(vec![
            Expr::Sym(scalar),
            Expr::mul(vec![Expr::Sym(psibar1), Expr::Sym(psi2)]),
            Expr::mul(vec![Expr::Sym(psibar3), Expr::Sym(psi4)]),
        ]);
        let result =
            try_fierz_auto(&expr, 4, &interner).expect("standard product should infer order");
        match result {
            Expr::Add(terms) => {
                assert_eq!(terms.len(), fierz_coefficients(4).len());
                assert!(
                    matches!(&terms[0], Expr::Mul(factors) if factors.contains(&Expr::Sym(scalar))),
                    "remaining scalar should be preserved"
                );
            }
            other => panic!("expected Fierz sum, got {other:?}"),
        }
    }

    #[test]
    fn fierz_ambiguous_three_bilinears_fails_clearly() {
        let interner = ax_ir::Interner::new();
        let s = ["psibar1", "psi2", "psibar3", "psi4", "psibar5", "psi6"]
            .iter()
            .map(|name| interner.get_or_intern(name))
            .collect::<Vec<_>>();
        let expr = Expr::mul(s.iter().map(|sym| Expr::Sym(*sym)).collect());
        let error = try_fierz_auto(&expr, 4, &interner).expect_err("three bilinears are ambiguous");
        assert_eq!(error, FierzError::AmbiguousBilinears(3));

        let wrapped = fierz_auto(&expr, 4, &interner);
        assert!(
            matches!(wrapped, Expr::Call(sym, _) if interner.resolve(sym) == "fierz_ambiguous_bilinears")
        );
    }

    #[test]
    fn fierz_malformed_bar_fails_clearly() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let psi = interner.get_or_intern("psi");
        let expr = Expr::mul(vec![Expr::Call(bar, vec![Expr::Sym(psi)])]);
        let error = try_fierz_auto(&expr, 4, &interner).expect_err("single bar is malformed");
        assert_eq!(error, FierzError::MalformedBilinear);
    }

    #[test]
    fn fierz_anticommuting_spinors_flip_rearrangement_sign() {
        let interner = ax_ir::Interner::new();
        let s1 = interner.get_or_intern("s1bar");
        let s2 = interner.get_or_intern("s2");
        let s3 = interner.get_or_intern("s3bar");
        let s4 = interner.get_or_intern("s4");
        let expr = Expr::mul(vec![
            Expr::Sym(s1),
            Expr::Sym(s2),
            Expr::Sym(s3),
            Expr::Sym(s4),
        ]);

        let plain = try_fierz_auto(&expr, 4, &interner).expect("plain spinors should rearrange");
        let mut props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
        for sym in [s1, s2, s3, s4] {
            props.insert(sym, vec![TensorProperty::AntiCommuting]);
        }
        let graded =
            try_fierz_auto_with_properties(&expr, 4, &props, &interner).expect("graded spinors");

        let mut plain_coeffs = Vec::new();
        collect_rationals(&plain, &mut plain_coeffs);
        let mut graded_coeffs = Vec::new();
        collect_rationals(&graded, &mut graded_coeffs);
        let mut negated_plain = plain_coeffs
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        graded_coeffs.sort_by_key(|value| format!("{value:?}"));
        negated_plain.sort_by_key(|value| format!("{value:?}"));
        assert_eq!(
            graded_coeffs, negated_plain,
            "moving the fourth anticommuting spinor through the third should flip every Fierz coefficient"
        );
    }

    #[test]
    fn fierz_structured_metadata_compatible_still_succeeds() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        for sym in [psi1, psi2, psi3, psi4] {
            props.insert(
                sym,
                vec![
                    TensorProperty::SpinorMeta(SpinorMetadata {
                        class: SpinorClass::Majorana,
                        dimension: Some(4),
                        chirality: None,
                        index_family: Some(spin),
                    }),
                    TensorProperty::AntiCommuting,
                ],
            );
        }
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let result = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect("compatible structured metadata should allow Fierz");
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn fierz_structured_metadata_dimension_mismatch_fails() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi1,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi2,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(2),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi3,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi4,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let error = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect_err("dimension mismatch should be rejected");
        assert_eq!(error, FierzError::IncompatibleSpinorDimension);
    }

    #[test]
    fn fierz_structured_metadata_chirality_mismatch_fails() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let spin = interner.get_or_intern("spin");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi1,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi2,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Right),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi3,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        props.insert(
            psi4,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Weyl,
                dimension: Some(4),
                chirality: Some(ax_ir::Chirality::Left),
                index_family: Some(spin),
            })],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi2),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi4),
        ]);
        let error = try_fierz_auto_with_properties(&expr, 4, &props, &interner)
            .expect_err("chirality mismatch should be rejected");
        assert_eq!(error, FierzError::IncompatibleSpinorChirality);
    }

    // ── split_gamma tests ─────────────────────────────────────────────────────

    #[test]
    fn split_gamma_three_indices() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // gamma(a, b, c) split from back → gamma(a,b)*gamma(c) + contractions
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert!(
                terms.len() >= 2,
                "expected main term + contraction terms, got {:?}",
                result
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_back_vs_front_differ() {
        // Splitting from back vs front should produce different expressions
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let back = split_gamma(&expr, gamma, g, true, &interner);
        let front = split_gamma(&expr, gamma, g, false, &interner);

        assert_ne!(back, front, "splitting from back vs front should differ");
    }

    #[test]
    fn split_gamma_two_indices_back() {
        // gamma(a, b) split from back → gamma(a)*gamma(b) − g^{ba} * 1
        // (2-index: remaining = [a], split = b, k=0 → sign = +, negate=false → subtract metric)
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                2,
                "gamma(a,b) split should give 2 terms: main + one contraction"
            );
            // First term should be a Mul (gamma(a) * gamma(b))
            assert!(
                matches!(&terms[0], Expr::Mul(_)),
                "first term should be a product"
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_two_indices_front() {
        // gamma(a, b) split from front → gamma(a)*gamma(b) − g^{ab} * 1
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, false, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2, "gamma(a,b) split-front should give 2 terms");
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_single_index_unchanged() {
        // gamma(a) has only one index — cannot be split, returned as-is
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");

        let expr = Expr::Call(gamma, vec![Expr::Sym(a)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);
        assert_eq!(result, expr, "single-index gamma should be unchanged");
    }

    #[test]
    fn split_gamma_four_indices_term_count() {
        // gamma(a,b,c,d) split from back has remaining=[a,b,c] → 3 contractions + 1 main = 4 terms
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");

        let expr = Expr::Call(
            gamma,
            vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c), Expr::Sym(d)],
        );
        let result = split_gamma(&expr, gamma, g, true, &interner);

        if let Expr::Add(terms) = &result {
            assert_eq!(
                terms.len(),
                4,
                "4-index gamma split should give 4 terms (1 main + 3 contractions)"
            );
        } else {
            panic!("expected Add, got {:?}", result);
        }
    }

    #[test]
    fn split_gamma_non_gamma_call_unchanged() {
        // A call to a different function should not be touched
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let f_sym = interner.get_or_intern("f");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");

        let expr = Expr::Call(f_sym, vec![Expr::Sym(a), Expr::Sym(b)]);
        let result = split_gamma(&expr, gamma, g, true, &interner);
        assert_eq!(result, expr, "non-gamma call should be unchanged");
    }

    #[test]
    fn split_gamma_distributes_in_sum() {
        let interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let g = interner.get_or_intern("g");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");

        // gamma(a,b,c) + gamma(a,b) → both are processed
        let g3 = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let g2 = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]);
        let expr = Expr::add(vec![g3, g2]);
        let result = split_gamma(&expr, gamma, g, true, &interner);

        // Result should still be an Add (may have more terms after expansion)
        assert!(
            matches!(result, Expr::Add(_)),
            "result of split on a sum should be an Add, got {:?}",
            result
        );
    }

    #[test]
    fn sort_spinors_majorana_flip_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(
            psi,
            vec![
                TensorProperty::Spinor,
                TensorProperty::MajoranaSpinor,
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::Spinor,
                TensorProperty::MajoranaSpinor,
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(chi)]),
            Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            Expr::Sym(psi),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn sort_spinors_structured_metadata_family_match_succeeds() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );

        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(chi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(psi),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn sort_spinors_structured_metadata_family_mismatch_returns_diagnostic() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let chi = interner.get_or_intern("chi");
        let spin = interner.get_or_intern("spin");
        let other_spin = interner.get_or_intern("other_spin");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );
        props.insert(
            chi,
            vec![
                TensorProperty::SpinorMeta(SpinorMetadata {
                    class: SpinorClass::Majorana,
                    dimension: Some(4),
                    chirality: None,
                    index_family: Some(other_spin),
                }),
                TensorProperty::AntiCommuting,
                TensorProperty::SortOrder(vec![chi, psi]),
            ],
        );

        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(chi),
        ]);
        let result = sort_spinors(&expr, &props, &interner);
        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "sort_spinors_spinor_family_mismatch")
        );
    }

    #[test]
    fn join_gamma_rank1_rank1_4d() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma, vec![Expr::Sym(b)]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert_eq!(terms.len(), 2),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn join_gamma_rank2_rank1_4d() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c)]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert!(terms.len() >= 3),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn join_gamma_duplicate_index_zero() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(a)]),
            &Expr::Call(gamma, vec![]),
            Some(4),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn join_gamma_generalised_delta() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let d = interner.get_or_intern("d");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c), Expr::Sym(d)]),
            Some(4),
            true,
            true,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        let printed = ax_ir::pretty_print(&result, &interner);
        assert!(printed.contains("generalised_delta"));
    }

    #[test]
    fn join_gamma_family_mismatch_returns_diagnostic() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let spin = interner.get_or_intern("spin");
        let flavor = interner.get_or_intern("flavor");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        let gamma_flavor = interner.get_or_intern("gamma_flavor");
        props.insert(
            gamma_flavor,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(flavor),
                has_gamma5: false,
            })],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a)]),
            &Expr::Call(gamma_flavor, vec![Expr::Sym(b)]),
            None,
            true,
            false,
            &Expr::Sym(interner.get_or_intern("eta")),
            &props,
            &mut interner,
        );
        assert!(
            matches!(result, Expr::Call(sym, _) if interner.resolve(sym) == "join_gamma_family_mismatch")
        );
    }

    #[test]
    fn join_gamma_dimension_comes_from_metadata_before_fallback() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let eta = interner.get_or_intern("eta");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let spin = interner.get_or_intern("spin");
        let mut props = prop_map();
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );

        let result = join_gamma_full(
            &Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b)]),
            &Expr::Call(gamma, vec![Expr::Sym(c)]),
            Some(2),
            true,
            false,
            &Expr::Sym(eta),
            &props,
            &mut interner,
        );
        match result {
            Expr::Add(terms) => assert!(
                terms.len() >= 3,
                "metadata dimension=4 should win over fallback dimension=2"
            ),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn expand_diracbar_sign() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let eps = interner.get_or_intern("eps");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(bar, vec![TensorProperty::DiracBar]);
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let odd = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a)]),
                Expr::Sym(eps),
            ])],
        );
        let even = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]),
                Expr::Sym(psi),
            ])],
        );
        let odd_result = expand_diracbar_full(&odd, &props, &interner);
        let even_result = expand_diracbar_full(&even, &props, &interner);
        let odd_str = format!("{odd_result:?}");
        assert!(matches!(odd_result, Expr::Neg(_)) || odd_str.contains("-1"));
        assert!(!matches!(even_result, Expr::Neg(_)));
    }

    #[test]
    fn expand_diracbar_full_structured_metadata_respects_reverse_gamma_order() {
        let interner = ax_ir::Interner::new();
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let spin = interner.get_or_intern("spin");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut props = prop_map();
        props.insert(
            bar,
            vec![TensorProperty::DiracBarMeta(DiracBarMetadata {
                gamma_symbol: Some(gamma),
                spinor_family: Some(spin),
                reverse_gamma_order: true,
            })],
        );
        props.insert(
            gamma,
            vec![TensorProperty::GammaMatrixMeta(GammaMatrixMetadata {
                dimension: Some(4),
                metric_symbol: None,
                index_family: Some(spin),
                has_gamma5: true,
            })],
        );
        props.insert(
            psi,
            vec![TensorProperty::SpinorMeta(SpinorMetadata {
                class: SpinorClass::Majorana,
                dimension: Some(4),
                chirality: None,
                index_family: Some(spin),
            })],
        );

        let expr = Expr::Call(
            bar,
            vec![Expr::mul(vec![
                Expr::Call(gamma, vec![Expr::Sym(a)]),
                Expr::Call(gamma, vec![Expr::Sym(b)]),
                Expr::Sym(psi),
            ])],
        );
        let result = expand_diracbar_full(&expr, &props, &interner);
        let expected = Expr::neg(Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(b)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]));
        assert_eq!(result, expected);
    }

    #[test]
    fn split_gamma_back_full() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma_full(&expr, true, &props, &mut interner);
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn split_gamma_front_full() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let c = interner.get_or_intern("c");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        let expr = Expr::Call(gamma, vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = split_gamma_full(&expr, false, &props, &mut interner);
        assert!(matches!(result, Expr::Add(_)));
    }

    #[test]
    fn fierz_full_reorders_wrong_spinor_order() {
        let mut interner = ax_ir::Interner::new();
        let gamma = interner.get_or_intern("gamma");
        let bar = interner.get_or_intern("bar");
        let psi1 = interner.get_or_intern("psi1");
        let psi2 = interner.get_or_intern("psi2");
        let psi3 = interner.get_or_intern("psi3");
        let psi4 = interner.get_or_intern("psi4");
        let mu = interner.get_or_intern("mu");
        let mut props = prop_map();
        props.insert(gamma, vec![TensorProperty::GammaMatrixProp]);
        props.insert(bar, vec![TensorProperty::DiracBar]);
        for sym in [psi1, psi2, psi3, psi4] {
            props.insert(
                sym,
                vec![TensorProperty::Spinor, TensorProperty::AntiCommuting],
            );
        }
        let expr = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi1)]),
            Expr::Call(gamma, vec![Expr::Sym(mu)]),
            Expr::Sym(psi4),
            Expr::Call(bar, vec![Expr::Sym(psi3)]),
            Expr::Sym(psi2),
        ]);
        let order = [
            Expr::Sym(psi1),
            Expr::Sym(psi2),
            Expr::Sym(psi3),
            Expr::Sym(psi4),
        ];
        let result = fierz_full(&expr, &order, 4, &props, &mut interner);
        assert!(matches!(result, Some(Expr::Add(_))));
    }
}
