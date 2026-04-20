#![forbid(unsafe_code)]
#![allow(
    clippy::get_first,
    clippy::if_same_then_else,
    clippy::incompatible_msrv,
    clippy::manual_map,
    clippy::only_used_in_recursion
)]

pub mod diagnostics;
pub mod equation;
pub mod inspect;
pub mod integrate;
pub mod limits;
pub mod property_store;
pub mod registry;
pub mod series;
pub mod simplify;
pub mod suggest;
pub mod workflows;

use ax_ir::{Assumption, Condition, Expr, Grading};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

pub use equation::{
    add_through, apply_through, contract_through, differentiate_equation, equation_to_rule,
    equation_to_subrule, get_factor, get_lhs, get_rhs, integrate_equation, is_equation, isolate,
    lower_equation, make_equation, multiply_through, multiply_through_indexed, raise_equation,
    substitute_equation, swap_sides, to_lhs, to_rhs,
};
pub use property_store::{
    property_discriminant_matches, InheritanceRule, PropertyAttachment, PropertyPattern,
    PropertyStore, SlotSpec, WeightCombine,
};
pub use registry::{
    algorithm_entries, assumption_entries, builtin_entries, callable_entries, convention_entries,
    format_tensor_property, property_entries, property_lookup_aliases, property_lookup_names,
    std_modules, syntax_rules, AlgorithmEntry, AssumptionEntry, BuiltinEntry, CallableEntry,
    ConventionEntry, EvalState, ParamDef, ParamType, PropertyEntry, StdModule, SyntaxRule,
};

fn find_tensor_symmetry(
    env: &Env,
    sym: lasso::Spur,
    indices: &[ax_ir::Index],
) -> Option<ax_ir::TensorSymmetry> {
    env.property_store
        .get_tensor_symmetry(sym, indices, &env.index_to_family)
}

#[derive(Clone, Debug, Default)]
pub struct Env {
    pub bindings: HashMap<lasso::Spur, Expr>,
    pub parent: Option<Box<Env>>,
    pub rules: Vec<ax_rewrite::RewriteRule>,
    pub assumptions: HashMap<lasso::Spur, Vec<Assumption>>,
    pub gradings: HashMap<lasso::Spur, Grading>,
    pub operators: HashMap<lasso::Spur, ax_qm::OperatorKind>,
    pub operator_statistics: HashMap<lasso::Spur, ax_qm::OperatorStatistics>,
    pub contractions: HashMap<(lasso::Spur, lasso::Spur), Expr>,
    pub coordinates: HashSet<lasso::Spur>,
    pub component_rule_symbols: HashSet<lasso::Spur>,
    pub index_families: HashMap<lasso::Spur, ax_ir::IndexFamily>,
    pub index_to_family: HashMap<lasso::Spur, lasso::Spur>,
    /// Deprecated: use property_store instead. Kept for backward compatibility with external code.
    pub tensor_properties: HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    pub property_store: crate::property_store::PropertyStore,
    pub expr_pool: Option<ax_ir::ExprPool>,
    pub expr_pool_threshold: usize,
    pub parallel: bool,
    pub spinor_labels: ax_spinor::LabelMap,
    pub graded_table: ax_graded::GradedSymbolTable,
    pub superspace_setup: Option<ax_graded::superspace::SuperspaceSetup>,
    pub brst_setup: Option<ax_graded::brst::BRSTSetup>,
    pub convention: ax_ir::Convention,
    /// Weights assigned to symbols. Map from (symbol, label) to weight value.
    /// Example: x::Weight(value=1, label=field) → weights[(x, "field")] = 1
    pub weights: HashMap<(lasso::Spur, String), i64>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
            rules: Vec::new(),
            assumptions: HashMap::new(),
            gradings: HashMap::new(),
            operators: HashMap::new(),
            operator_statistics: HashMap::new(),
            contractions: HashMap::new(),
            coordinates: HashSet::new(),
            component_rule_symbols: HashSet::new(),
            index_families: HashMap::new(),
            index_to_family: HashMap::new(),
            tensor_properties: HashMap::new(),
            property_store: crate::property_store::PropertyStore::new(),
            expr_pool: None,
            expr_pool_threshold: 256,
            parallel: false,
            spinor_labels: ax_spinor::LabelMap::new(),
            graded_table: ax_graded::GradedSymbolTable::new(),
            superspace_setup: None,
            brst_setup: None,
            convention: ax_ir::Convention::default(),
            weights: HashMap::new(),
        }
    }

    pub fn lookup(&self, sym: lasso::Spur) -> Option<&Expr> {
        self.bindings
            .get(&sym)
            .or_else(|| self.parent.as_deref().and_then(|parent| parent.lookup(sym)))
    }

    pub fn extend(&self, sym: lasso::Spur, val: Expr) -> Env {
        let mut bindings = HashMap::new();
        bindings.insert(sym, val);
        Env {
            bindings,
            parent: Some(Box::new(self.clone())),
            rules: self.rules.clone(),
            assumptions: self.assumptions.clone(),
            gradings: self.gradings.clone(),
            operators: self.operators.clone(),
            operator_statistics: self.operator_statistics.clone(),
            contractions: self.contractions.clone(),
            coordinates: self.coordinates.clone(),
            component_rule_symbols: self.component_rule_symbols.clone(),
            index_families: self.index_families.clone(),
            index_to_family: self.index_to_family.clone(),
            tensor_properties: self.tensor_properties.clone(),
            property_store: self.property_store.clone(),
            expr_pool: self.expr_pool.clone(),
            expr_pool_threshold: self.expr_pool_threshold,
            parallel: self.parallel,
            spinor_labels: self.spinor_labels.clone(),
            graded_table: self.graded_table.clone(),
            superspace_setup: self.superspace_setup.clone(),
            brst_setup: self.brst_setup.clone(),
            convention: self.convention.clone(),
            weights: self.weights.clone(),
        }
    }

    pub fn enable_pool(&mut self) {
        self.expr_pool = Some(ax_ir::ExprPool::new());
    }
}

impl ax_tensor::DummyRenameEnv for Env {
    fn index_families(&self) -> &HashMap<lasso::Spur, ax_ir::IndexFamily> {
        &self.index_families
    }

    fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur> {
        &self.index_to_family
    }
}

impl ax_tensor::ComponentEvalEnv for Env {
    fn coordinates(&self) -> Vec<lasso::Spur> {
        self.coordinates.iter().copied().collect()
    }

    fn is_coordinate(&self, s: lasso::Spur) -> bool {
        self.coordinates.contains(&s)
    }

    fn tensor_properties(&self) -> &HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>> {
        &self.tensor_properties
    }
}

fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Complex(re, im) => 1 + expr_node_count(re) + expr_node_count(im),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(expr_node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) | Expr::Rule(base, exp, _) => {
            1 + expr_node_count(base) + expr_node_count(exp)
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => 1 + expr_node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(expr_node_count).sum::<usize>(),
        Expr::Indexed(base, _) => 1 + expr_node_count(base),
        Expr::Matrix(rows) => {
            1 + rows
                .iter()
                .flat_map(|row| row.iter())
                .map(expr_node_count)
                .sum::<usize>()
        }
        Expr::Piecewise(cases) => {
            1 + cases
                .iter()
                .map(|(value, _)| expr_node_count(value))
                .sum::<usize>()
        }
        Expr::Let(_, val, body) => 1 + expr_node_count(val) + expr_node_count(body),
        Expr::FnDef(_, _, body) => 1 + expr_node_count(body),
        Expr::Import(_) | Expr::Assume(_, _) | Expr::SetConvention(_, _) => 1,
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => 1,
    }
}

fn maybe_pooled_canonicalise(
    expr: &Expr,
    env: &mut Env,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if expr_node_count(expr) >= env.expr_pool_threshold {
        if let Some(ref mut pool) = env.expr_pool {
            let id = pool.from_expr(expr);
            let result_id =
                ax_tensor::pooled_canon::canonicalise_pooled(id, pool, properties, interner);
            return pool.to_expr(result_id);
        }
    }

    if env.parallel {
        ax_tensor::canonicalise_parallel(expr, properties, interner)
    } else {
        ax_tensor::canonicalise(expr, properties, interner)
    }
}

fn maybe_pooled_meld(
    expr: &Expr,
    env: &mut Env,
    properties: &dyn ax_tensor::PropertyLookup,
    interner: &ax_ir::Interner,
) -> Expr {
    if expr_node_count(expr) >= env.expr_pool_threshold {
        if let Some(ref mut pool) = env.expr_pool {
            let id = pool.from_expr(expr);
            let result_id = ax_tensor::pooled_canon::meld_pooled(id, pool, properties, interner);
            return pool.to_expr(result_id);
        }
    }

    if env.parallel {
        ax_tensor::meld_parallel(expr, properties, interner)
    } else {
        ax_tensor::meld(expr, properties, interner)
    }
}

fn label_from_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<ax_spinor::Label> {
    match expr {
        Expr::Int(n) => n.to_u16().map(ax_spinor::Label::new),
        Expr::Sym(s) => interner
            .resolve(*s)
            .parse::<u16>()
            .ok()
            .map(ax_spinor::Label::new),
        _ => None,
    }
}

fn labels_from_exprs(args: &[Expr], interner: &ax_ir::Interner) -> Option<Vec<ax_spinor::Label>> {
    args.iter()
        .map(|arg| label_from_expr(arg, interner))
        .collect::<Option<Vec<_>>>()
}

fn labels_from_list_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<Vec<ax_spinor::Label>> {
    let Expr::List(items) = expr else {
        return None;
    };
    labels_from_exprs(items, interner)
}

fn spinor_multi_mandelstam_expr(labels: Vec<ax_spinor::Label>, interner: &ax_ir::Interner) -> Expr {
    match labels.as_slice() {
        [i, j] => spinor_to_expr(&ax_spinor::SpinorExpr::Mandelstam(*i, *j), interner),
        [i, j, k] => spinor_to_expr(&ax_spinor::SpinorExpr::Mandelstam3(*i, *j, *k), interner),
        _ => spinor_to_expr(
            &ax_spinor::SpinorExpr::Product(vec![ax_spinor::SpinorTerm::new(
                BigRational::from_integer(1.into()),
                vec![ax_spinor::SpinorFactor::Mandelstam(labels)],
            )]),
            interner,
        ),
    }
}

fn int_from_expr(expr: &Expr) -> Option<u16> {
    match expr {
        Expr::Int(n) => n.to_u16(),
        _ => None,
    }
}

fn int_expr_u16(n: u16) -> Expr {
    Expr::Int(BigInt::from(n))
}

fn usize_from_expr(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Int(n) => n.to_usize(),
        _ => None,
    }
}

fn symbol_from_expr(expr: &Expr) -> Option<lasso::Spur> {
    match expr {
        Expr::Sym(s) => Some(*s),
        _ => None,
    }
}

fn extract_sym(expr: &Expr, interner: &ax_ir::Interner) -> lasso::Spur {
    match expr {
        Expr::Sym(sym) => *sym,
        other => interner.get_or_intern(&ax_ir::pretty_print(other, interner)),
    }
}

fn creation_expr(mode: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("creation"), vec![mode])
}

fn annihilation_expr(mode: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern("annihilation"), vec![mode])
}

fn number_state_expr(mode: Expr, n: usize, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(
        interner.get_or_intern("number_state"),
        vec![mode, Expr::Int(BigInt::from(n))],
    )
}

fn vacuum_expr(mode: Expr, interner: &ax_ir::Interner) -> Expr {
    number_state_expr(mode, 0, interner)
}

fn sqrt_usize_expr(n: usize, interner: &ax_ir::Interner) -> Expr {
    match n {
        0 => Expr::zero(),
        1 => Expr::one(),
        _ => builtin_unary("sqrt", Expr::Int(BigInt::from(n)), interner),
    }
}

fn decompose_scalar_times_state(expr: &Expr, interner: &ax_ir::Interner) -> Option<(Expr, Expr)> {
    match expr {
        Expr::Call(f, args) if interner.resolve(*f) == "number_state" && args.len() == 2 => {
            Some((Expr::one(), expr.clone()))
        }
        Expr::Mul(factors) => {
            let mut scalar = Vec::new();
            let mut state = None;
            for factor in factors {
                match factor {
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "number_state" && args.len() == 2 =>
                    {
                        if state.is_some() {
                            return None;
                        }
                        state = Some(factor.clone());
                    }
                    _ => scalar.push(factor.clone()),
                }
            }
            state.map(|state_expr| {
                let scalar_expr = if scalar.is_empty() {
                    Expr::one()
                } else {
                    Expr::mul(scalar)
                };
                (scalar_expr, state_expr)
            })
        }
        _ => None,
    }
}

fn apply_abstract_qm_operator(
    operator: &Expr,
    state: &Expr,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    if let Expr::Add(terms) = state {
        return Some(Expr::add(
            terms
                .iter()
                .filter_map(|term| apply_abstract_qm_operator(operator, term, interner))
                .collect(),
        ));
    }

    let (scalar, basis_state) = decompose_scalar_times_state(state, interner)?;
    let Expr::Call(state_head, state_args) = &basis_state else {
        return None;
    };
    if interner.resolve(*state_head) != "number_state" || state_args.len() != 2 {
        return None;
    }
    let mode = state_args[0].clone();
    let n = usize_from_expr(&state_args[1])?;

    let applied = match operator {
        Expr::Call(f, args) if args.len() == 1 && interner.resolve(*f) == "creation" => {
            if args[0] != mode {
                return None;
            }
            Expr::mul(vec![
                sqrt_usize_expr(n + 1, interner),
                number_state_expr(mode.clone(), n + 1, interner),
            ])
        }
        Expr::Call(f, args) if args.len() == 1 && interner.resolve(*f) == "annihilation" => {
            if args[0] != mode {
                return None;
            }
            if n == 0 {
                Expr::zero()
            } else {
                Expr::mul(vec![
                    sqrt_usize_expr(n, interner),
                    number_state_expr(mode.clone(), n - 1, interner),
                ])
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| {
                    apply_abstract_qm_operator(term, &basis_state, interner)
                        .unwrap_or_else(|| Expr::mul(vec![term.clone(), basis_state.clone()]))
                })
                .collect(),
        ),
        Expr::Mul(factors) => {
            let mut scalar_factors = Vec::new();
            let mut current = basis_state.clone();
            let mut applied_any = false;
            for factor in factors.iter().rev() {
                if let Some(next) = apply_abstract_qm_operator(factor, &current, interner) {
                    current = next;
                    applied_any = true;
                } else {
                    scalar_factors.push(factor.clone());
                }
            }

            if !applied_any {
                return Some(Expr::mul(vec![
                    Expr::mul(factors.clone()),
                    basis_state.clone(),
                ]));
            }

            scalar_factors.reverse();
            if scalar_factors.is_empty() {
                current
            } else {
                let mut combined = scalar_factors;
                combined.push(current);
                Expr::mul(combined)
            }
        }
        _ => return None,
    };

    Some(if scalar == Expr::one() {
        applied
    } else if applied == Expr::zero() {
        Expr::zero()
    } else {
        Expr::mul(vec![scalar, applied])
    })
}

fn find_tensor_property_sym(
    env: &Env,
    property: fn(&ax_ir::TensorProperty) -> bool,
) -> Option<lasso::Spur> {
    env.tensor_properties
        .iter()
        .find_map(|(sym, props)| props.iter().any(property).then_some(*sym))
        .or_else(|| {
            env.property_store
                .symbols()
                .into_iter()
                .find(|sym| env.property_store.get_all(*sym).into_iter().any(property))
        })
}

fn find_metric_sym(env: &Env) -> Option<lasso::Spur> {
    find_tensor_property_sym(env, |prop| matches!(prop, ax_ir::TensorProperty::Metric))
}

fn find_inv_metric_sym(env: &Env) -> Option<lasso::Spur> {
    find_tensor_property_sym(env, |prop| {
        matches!(prop, ax_ir::TensorProperty::InverseMetric)
    })
}

fn symbol_list_from_expr(expr: &Expr) -> Option<Vec<lasso::Spur>> {
    match expr {
        Expr::List(items) => items.iter().map(symbol_from_expr).collect(),
        _ => None,
    }
}

fn hilbert_space_metadata_of_symbol(
    env: &Env,
    symbol: lasso::Spur,
) -> Option<ax_ir::HilbertSpaceMetadata> {
    env.property_store
        .get_all(symbol)
        .into_iter()
        .find_map(|prop| match prop {
            ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
        .or_else(|| {
            env.tensor_properties.get(&symbol).and_then(|props| {
                props.iter().find_map(|prop| match prop {
                    ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
                    _ => None,
                })
            })
        })
}

fn flatten_hilbert_space_factors(
    env: &Env,
    factors: &[lasso::Spur],
) -> Option<Vec<ax_ir::HilbertSpaceFactor>> {
    if factors.is_empty() {
        return None;
    }
    let mut flattened = Vec::new();
    for factor in factors {
        let metadata = hilbert_space_metadata_of_symbol(env, *factor)?;
        if metadata.factors.is_empty() {
            return None;
        }
        flattened.extend(metadata.factors);
    }
    Some(flattened)
}

fn parse_quantum_object_kind_expr(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_ir::QuantumObjectKind> {
    match name_from_expr(expr, interner)?
        .to_ascii_lowercase()
        .as_str()
    {
        "ket" => Some(ax_ir::QuantumObjectKind::Ket),
        "bra" => Some(ax_ir::QuantumObjectKind::Bra),
        "operator" => Some(ax_ir::QuantumObjectKind::Operator),
        "density_operator" => Some(ax_ir::QuantumObjectKind::DensityOperator),
        "projector" => Some(ax_ir::QuantumObjectKind::Projector),
        "observable" => Some(ax_ir::QuantumObjectKind::Observable),
        "channel" => Some(ax_ir::QuantumObjectKind::Channel),
        _ => None,
    }
}

fn usize_list_from_expr(expr: &Expr) -> Option<Vec<usize>> {
    match expr {
        Expr::List(items) => items.iter().map(usize_from_expr).collect(),
        _ => None,
    }
}

fn unique_factor_index(
    metadata: &ax_ir::HilbertSpaceMetadata,
    factor_symbol: lasso::Spur,
) -> Result<usize, ()> {
    let matches = metadata
        .factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (factor.symbol == factor_symbol).then_some(index))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        _ => Err(()),
    }
}

/// Attach structured Hilbert-space metadata to the evaluator environment.
pub fn apply_hilbert_space_declaration(
    env: &mut Env,
    symbol: lasso::Spur,
    metadata: ax_ir::HilbertSpaceMetadata,
) {
    env.tensor_properties
        .entry(symbol)
        .or_default()
        .push(ax_ir::TensorProperty::HilbertSpaceMeta(metadata.clone()));
    env.property_store.declare_hilbert_space(symbol, metadata);
}

/// Attach structured quantum-object metadata and required legacy compatibility markers.
pub fn apply_quantum_object_declaration(
    env: &mut Env,
    symbol: lasso::Spur,
    metadata: ax_ir::QuantumObjectMetadata,
) {
    env.tensor_properties
        .entry(symbol)
        .or_default()
        .push(ax_ir::TensorProperty::QuantumObjectMeta(metadata.clone()));
    if matches!(
        metadata.kind,
        ax_ir::QuantumObjectKind::Operator
            | ax_ir::QuantumObjectKind::DensityOperator
            | ax_ir::QuantumObjectKind::Projector
            | ax_ir::QuantumObjectKind::Observable
            | ax_ir::QuantumObjectKind::Channel
    ) {
        env.tensor_properties
            .entry(symbol)
            .or_default()
            .push(ax_ir::TensorProperty::NonCommuting);
    }
    env.property_store.declare_quantum_object(symbol, metadata);
}

fn name_from_expr<'a>(expr: &Expr, interner: &'a ax_ir::Interner) -> Option<&'a str> {
    match expr {
        Expr::Sym(s) => Some(interner.resolve(*s)),
        _ => None,
    }
}

fn theta_monomial_from_spec(
    expr: &Expr,
    setup: &ax_graded::superspace::SuperspaceSetup,
) -> Option<ax_graded::superspace::ThetaMonomial> {
    let Expr::List(items) = expr else {
        return None;
    };
    if items.len() != 2 {
        return None;
    }
    let theta_count = usize_from_expr(&items[0])?;
    let theta_bar_count = usize_from_expr(&items[1])?;
    if theta_count > setup.theta.len() || theta_bar_count > setup.theta_bar.len() {
        return None;
    }
    let mut theta_powers = vec![0; setup.theta.len()];
    let mut theta_bar_powers = vec![0; setup.theta_bar.len()];
    for power in theta_powers.iter_mut().take(theta_count) {
        *power = 1;
    }
    for power in theta_bar_powers.iter_mut().take(theta_bar_count) {
        *power = 1;
    }
    Some(ax_graded::superspace::ThetaMonomial {
        theta_powers,
        theta_bar_powers,
    })
}

fn active_superspace(
    env: &Env,
    interner: &ax_ir::Interner,
) -> (
    ax_graded::superspace::SuperspaceSetup,
    ax_graded::GradedSymbolTable,
) {
    env.superspace_setup
        .clone()
        .map(|setup| (setup, env.graded_table.clone()))
        .unwrap_or_else(|| ax_graded::superspace::setup_n1_superspace(interner))
}

fn grading_from_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<ax_graded::Grading> {
    match expr {
        Expr::Int(n) => n.to_i32().map(ax_graded::Grading::ghost),
        Expr::Sym(sym) => match interner.resolve(*sym).to_ascii_lowercase().as_str() {
            "bosonic" | "boson" | "even" => Some(ax_graded::Grading::bosonic()),
            "fermionic" | "fermion" | "odd" => Some(ax_graded::Grading::fermionic()),
            _ => None,
        },
        _ => None,
    }
}

fn perturbation_setup(
    full_field: lasso::Spur,
    background: lasso::Spur,
    inverse_background: Option<lasso::Spur>,
    perturbation: lasso::Spur,
    epsilon: lasso::Spur,
    max_order: usize,
) -> ax_perturb::PerturbationSetup {
    ax_perturb::PerturbationSetup {
        full_field,
        background,
        inverse_background,
        perturbations: vec![ax_perturb::PerturbationOrder {
            order: 1,
            field: perturbation,
        }],
        epsilon,
        max_order,
    }
}

fn expanded_to_expr_list(expanded: ax_perturb::ExpandedExpression, max_order: usize) -> Expr {
    let mut by_order = vec![Expr::zero(); max_order + 1];
    for term in expanded.orders {
        if term.order <= max_order {
            by_order[term.order] = term.expr;
        }
    }
    Expr::List(by_order)
}

fn labelled_exprs_to_list(
    items: Vec<ax_perturb::NamedEquation>,
    interner: &ax_ir::Interner,
) -> Expr {
    Expr::List(
        items
            .into_iter()
            .map(|item| {
                Expr::List(vec![
                    Expr::Sym(interner.get_or_intern(&item.label)),
                    item.expr,
                ])
            })
            .collect(),
    )
}

fn named_exprs_to_list(items: Vec<ax_perturb::NamedExpr>) -> Expr {
    Expr::List(
        items
            .into_iter()
            .map(|item| Expr::List(vec![Expr::Sym(item.name), item.expr]))
            .collect(),
    )
}

fn cpt_spec_tag(interner: &ax_ir::Interner) -> lasso::Spur {
    interner.get_or_intern("__cpt_spec__")
}

fn curvature_name(curvature: ax_perturb::SpatialCurvature) -> &'static str {
    match curvature {
        ax_perturb::SpatialCurvature::Flat => "flat",
        ax_perturb::SpatialCurvature::Closed => "closed",
        ax_perturb::SpatialCurvature::Open => "open",
    }
}

fn parse_curvature_name(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::SpatialCurvature> {
    match name_from_expr(expr, interner)? {
        "flat" => Some(ax_perturb::SpatialCurvature::Flat),
        "closed" => Some(ax_perturb::SpatialCurvature::Closed),
        "open" => Some(ax_perturb::SpatialCurvature::Open),
        _ => None,
    }
}

fn parse_cubic_channel_name(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::CubicInteractionChannel> {
    match name_from_expr(expr, interner)? {
        "scalar_scalar_scalar" => Some(ax_perturb::CubicInteractionChannel::ScalarScalarScalar),
        "tensor_tensor_tensor" => Some(ax_perturb::CubicInteractionChannel::TensorTensorTensor),
        "scalar_scalar_tensor" => Some(ax_perturb::CubicInteractionChannel::ScalarScalarTensor),
        "scalar_tensor_tensor" => Some(ax_perturb::CubicInteractionChannel::ScalarTensorTensor),
        _ => None,
    }
}

fn parse_eft_model_name(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::EftModelKind> {
    match name_from_expr(expr, interner)? {
        "canonical" => Some(ax_perturb::EftModelKind::Canonical),
        "reduced_sound_speed" => Some(ax_perturb::EftModelKind::ReducedSoundSpeed),
        "horndeski_like" => Some(ax_perturb::EftModelKind::HorndeskiLike),
        _ => None,
    }
}

fn parse_hierarchy_gauge_name(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::HierarchyGauge> {
    match name_from_expr(expr, interner)? {
        "newtonian" => Some(ax_perturb::HierarchyGauge::Newtonian),
        "synchronous" => Some(ax_perturb::HierarchyGauge::Synchronous),
        _ => None,
    }
}

fn parse_hierarchy_closure_name(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::HierarchyClosure> {
    match name_from_expr(expr, interner)? {
        "power_law" => Some(ax_perturb::HierarchyClosure::PowerLaw),
        "free_streaming" => Some(ax_perturb::HierarchyClosure::FreeStreaming),
        "user_symbolic" => Some(ax_perturb::HierarchyClosure::UserSymbolic),
        _ => None,
    }
}

fn make_harmonic_spec_expr(
    curvature: ax_perturb::SpatialCurvature,
    sector: ax_perturb::SectorKind,
    interner: &ax_ir::Interner,
) -> Expr {
    let sector_name = match sector {
        ax_perturb::SectorKind::Scalar => "scalar",
        ax_perturb::SectorKind::Vector => "vector",
        ax_perturb::SectorKind::Tensor => "tensor",
    };
    Expr::List(vec![
        Expr::Sym(cpt_spec_tag(interner)),
        Expr::Sym(interner.get_or_intern("harmonic")),
        Expr::Sym(interner.get_or_intern(sector_name)),
        Expr::Sym(interner.get_or_intern(curvature_name(curvature))),
        Expr::Sym(interner.get_or_intern("k")),
    ])
}

fn make_eft_model_expr(model: ax_perturb::EftModelKind, interner: &ax_ir::Interner) -> Expr {
    Expr::List(vec![
        Expr::Sym(cpt_spec_tag(interner)),
        Expr::Sym(interner.get_or_intern("eft_model")),
        Expr::Sym(interner.get_or_intern(ax_perturb::eft_model_name(model))),
    ])
}

fn make_background_spec_expr(
    bg: &ax_perturb::FrwBackgroundSpec,
    interner: &ax_ir::Interner,
) -> Expr {
    let curvature = match bg.spatial_curvature {
        ax_perturb::SpatialCurvature::Flat => "flat",
        ax_perturb::SpatialCurvature::Closed => "closed",
        ax_perturb::SpatialCurvature::Open => "open",
    };
    let time = match bg.time_coordinate {
        ax_perturb::TimeCoordinate::Conformal => "conformal",
        ax_perturb::TimeCoordinate::Cosmic => "cosmic",
    };
    Expr::List(vec![
        Expr::Sym(cpt_spec_tag(interner)),
        Expr::Sym(interner.get_or_intern("background")),
        Expr::Sym(bg.scale_factor),
        Expr::Sym(bg.conformal_hubble),
        Expr::Sym(bg.cosmic_hubble),
        Expr::Sym(bg.conformal_time),
        Expr::Sym(bg.cosmic_time),
        Expr::Int(BigInt::from(bg.spatial_dim)),
        Expr::Sym(interner.get_or_intern(curvature)),
        Expr::Sym(interner.get_or_intern(time)),
    ])
}

fn parse_background_spec_expr(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::FrwBackgroundSpec> {
    let Expr::List(items) = expr else {
        return None;
    };
    let [Expr::Sym(tag), Expr::Sym(kind), Expr::Sym(scale_factor), Expr::Sym(conformal_hubble), Expr::Sym(cosmic_hubble), Expr::Sym(conformal_time), Expr::Sym(cosmic_time), Expr::Int(spatial_dim), Expr::Sym(curvature), Expr::Sym(time)] =
        items.as_slice()
    else {
        return None;
    };
    if *tag != cpt_spec_tag(interner) || interner.resolve(*kind) != "background" {
        return None;
    }
    let spatial_curvature = match interner.resolve(*curvature) {
        "flat" => ax_perturb::SpatialCurvature::Flat,
        "closed" => ax_perturb::SpatialCurvature::Closed,
        "open" => ax_perturb::SpatialCurvature::Open,
        _ => return None,
    };
    let time_coordinate = match interner.resolve(*time) {
        "conformal" => ax_perturb::TimeCoordinate::Conformal,
        "cosmic" => ax_perturb::TimeCoordinate::Cosmic,
        _ => return None,
    };
    ax_perturb::FrwBackgroundSpec::new(
        *scale_factor,
        *conformal_hubble,
        *cosmic_hubble,
        *conformal_time,
        *cosmic_time,
        spatial_dim.to_usize()?,
        spatial_curvature,
        time_coordinate,
    )
    .ok()
}

fn make_gauge_spec_expr(kind: ax_perturb::GaugeKind, interner: &ax_ir::Interner) -> Expr {
    let kind_name = match kind {
        ax_perturb::GaugeKind::Newtonian => "newtonian",
        ax_perturb::GaugeKind::Synchronous => "synchronous",
        ax_perturb::GaugeKind::Comoving => "comoving",
        ax_perturb::GaugeKind::Flat => "flat",
        ax_perturb::GaugeKind::UniformDensity => "uniform_density",
        ax_perturb::GaugeKind::UniformCurvature => "uniform_curvature",
        ax_perturb::GaugeKind::Poisson => "poisson",
    };
    Expr::List(vec![
        Expr::Sym(cpt_spec_tag(interner)),
        Expr::Sym(interner.get_or_intern("gauge")),
        Expr::Sym(interner.get_or_intern(kind_name)),
    ])
}

fn parse_gauge_spec_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<ax_perturb::GaugeKind> {
    let Expr::List(items) = expr else {
        return None;
    };
    let [Expr::Sym(tag), Expr::Sym(kind), Expr::Sym(name)] = items.as_slice() else {
        return None;
    };
    if *tag != cpt_spec_tag(interner) || interner.resolve(*kind) != "gauge" {
        return None;
    }
    match interner.resolve(*name) {
        "newtonian" => Some(ax_perturb::GaugeKind::Newtonian),
        "synchronous" => Some(ax_perturb::GaugeKind::Synchronous),
        "comoving" => Some(ax_perturb::GaugeKind::Comoving),
        "flat" => Some(ax_perturb::GaugeKind::Flat),
        "uniform_density" => Some(ax_perturb::GaugeKind::UniformDensity),
        "uniform_curvature" => Some(ax_perturb::GaugeKind::UniformCurvature),
        "poisson" => Some(ax_perturb::GaugeKind::Poisson),
        _ => None,
    }
}

fn make_matter_spec_expr(kind: ax_perturb::MatterKind, interner: &ax_ir::Interner) -> Expr {
    let mut items = vec![
        Expr::Sym(cpt_spec_tag(interner)),
        Expr::Sym(interner.get_or_intern("matter")),
    ];
    match kind {
        ax_perturb::MatterKind::PerfectFluid => {
            items.push(Expr::Sym(interner.get_or_intern("perfect_fluid")))
        }
        ax_perturb::MatterKind::ImperfectFluid => {
            items.push(Expr::Sym(interner.get_or_intern("imperfect_fluid")))
        }
        ax_perturb::MatterKind::CanonicalScalar => {
            items.push(Expr::Sym(interner.get_or_intern("canonical_scalar")))
        }
        ax_perturb::MatterKind::MultiCanonicalScalar { fields } => {
            items.push(Expr::Sym(interner.get_or_intern("multi_canonical_scalar")));
            items.push(Expr::Int(BigInt::from(fields)));
        }
        ax_perturb::MatterKind::Symbolic => {
            items.push(Expr::Sym(interner.get_or_intern("symbolic")))
        }
    }
    Expr::List(items)
}

fn parse_matter_spec_expr(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_perturb::MatterKind> {
    let Expr::List(items) = expr else {
        return None;
    };
    let [Expr::Sym(tag), Expr::Sym(kind), rest @ ..] = items.as_slice() else {
        return None;
    };
    if *tag != cpt_spec_tag(interner) || interner.resolve(*kind) != "matter" {
        return None;
    }
    match rest {
        [Expr::Sym(name)] => match interner.resolve(*name) {
            "perfect_fluid" => Some(ax_perturb::MatterKind::PerfectFluid),
            "imperfect_fluid" => Some(ax_perturb::MatterKind::ImperfectFluid),
            "canonical_scalar" => Some(ax_perturb::MatterKind::CanonicalScalar),
            "symbolic" => Some(ax_perturb::MatterKind::Symbolic),
            _ => None,
        },
        [Expr::Sym(name), Expr::Int(fields)]
            if interner.resolve(*name) == "multi_canonical_scalar" =>
        {
            Some(ax_perturb::MatterKind::MultiCanonicalScalar {
                fields: fields.to_usize()?,
            })
        }
        _ => None,
    }
}

fn substitute_symbol_expr(expr: &Expr, from: lasso::Spur, to: &Expr) -> Expr {
    match expr {
        Expr::Sym(sym) if *sym == from => to.clone(),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(substitute_symbol_expr(re, from, to)),
            Box::new(substitute_symbol_expr(im, from, to)),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| substitute_symbol_expr(term, from, to))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| substitute_symbol_expr(factor, from, to))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            substitute_symbol_expr(base, from, to),
            substitute_symbol_expr(exp, from, to),
        ),
        Expr::Neg(inner) => Expr::neg(substitute_symbol_expr(inner, from, to)),
        Expr::Call(fun, args) => Expr::Call(
            *fun,
            args.iter()
                .map(|arg| substitute_symbol_expr(arg, from, to))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(substitute_symbol_expr(body, from, to)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(substitute_symbol_expr(lhs, from, to)),
            Box::new(substitute_symbol_expr(rhs, from, to)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (substitute_symbol_expr(value, from, to), condition.clone())
                })
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_symbol_expr(base, from, to)),
            indices.clone(),
        ),
        Expr::Group(inner, relation) => {
            Expr::Group(Box::new(substitute_symbol_expr(inner, from, to)), *relation)
        }
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(substitute_symbol_expr(value, from, to)),
            Box::new(substitute_symbol_expr(body, from, to)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| substitute_symbol_expr(item, from, to))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| substitute_symbol_expr(item, from, to))
                        .collect()
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn svt_decomposition_to_expr(
    decomp: ax_perturb::gauge::SVTDecomposition,
    interner: &ax_ir::Interner,
) -> Expr {
    let component_sym = |name: &str| Expr::Sym(interner.get_or_intern(name));
    let scalars = decomp
        .scalar_modes
        .into_iter()
        .map(|mode| {
            let component = match mode.component {
                ax_perturb::gauge::SVTComponent::Phi => "Phi",
                ax_perturb::gauge::SVTComponent::Psi => "Psi",
                ax_perturb::gauge::SVTComponent::B => "B",
                ax_perturb::gauge::SVTComponent::E => "E",
            };
            Expr::List(vec![Expr::Sym(mode.name), component_sym(component)])
        })
        .collect();
    let vectors = decomp
        .vector_modes
        .into_iter()
        .map(|mode| {
            let component = match mode.component {
                ax_perturb::gauge::VectorSVT::Si => "Si",
                ax_perturb::gauge::VectorSVT::Fi => "Fi",
            };
            Expr::List(vec![Expr::Sym(mode.name), component_sym(component)])
        })
        .collect();
    let tensors = decomp
        .tensor_modes
        .into_iter()
        .map(|mode| Expr::Sym(mode.name))
        .collect();
    Expr::List(vec![
        Expr::List(scalars),
        Expr::List(vectors),
        Expr::List(tensors),
    ])
}

fn regge_wheeler_decomposition_to_expr(
    decomp: ax_perturb::gauge::ReggeWheelerDecomposition,
    interner: &ax_ir::Interner,
) -> Expr {
    let named_exprs = |items: Vec<(lasso::Spur, Expr)>| {
        Expr::List(
            items
                .into_iter()
                .map(|(name, expr)| Expr::List(vec![Expr::Sym(name), expr]))
                .collect(),
        )
    };
    Expr::List(vec![
        Expr::List(vec![
            Expr::Sym(interner.get_or_intern("even_parity")),
            named_exprs(decomp.even_parity),
        ]),
        Expr::List(vec![
            Expr::Sym(interner.get_or_intern("odd_parity")),
            named_exprs(decomp.odd_parity),
        ]),
    ])
}

fn rational_or_int(r: &BigRational) -> Expr {
    if r.is_integer() {
        Expr::Int(r.to_integer())
    } else {
        Expr::Rational(r.clone())
    }
}

fn spinor_factor_to_expr(factor: &ax_spinor::SpinorFactor, interner: &ax_ir::Interner) -> Expr {
    match factor {
        ax_spinor::SpinorFactor::Angle(i, j) => Expr::Call(
            interner.get_or_intern("__angle"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
        ),
        ax_spinor::SpinorFactor::Square(i, j) => Expr::Call(
            interner.get_or_intern("__square"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
        ),
        ax_spinor::SpinorFactor::AngleSquare(i, middle, j) => Expr::Call(
            interner.get_or_intern("__angle_square_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorFactor::SquareAngle(i, middle, j) => Expr::Call(
            interner.get_or_intern("__square_angle_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorFactor::Mandelstam(labels) => Expr::Call(
            interner.get_or_intern("__mandelstam_multi"),
            labels.iter().map(|l| int_expr_u16(l.0)).collect(),
        ),
        ax_spinor::SpinorFactor::Power(base, n) => Expr::pow(
            spinor_factor_to_expr(base, interner),
            Expr::Int(BigInt::from(*n)),
        ),
        ax_spinor::SpinorFactor::Grouped(expr) => spinor_to_expr(expr, interner),
        ax_spinor::SpinorFactor::SymbolicParam(s) => Expr::Sym(*s),
    }
}

fn spinor_term_to_expr(term: &ax_spinor::SpinorTerm, interner: &ax_ir::Interner) -> Expr {
    let mut factors = Vec::new();
    if term.coefficient != BigRational::from_integer(1.into()) || term.factors.is_empty() {
        factors.push(rational_or_int(&term.coefficient));
    }
    factors.extend(
        term.factors
            .iter()
            .map(|f| spinor_factor_to_expr(f, interner)),
    );
    Expr::mul(factors)
}

fn spinor_to_expr(s: &ax_spinor::SpinorExpr, interner: &ax_ir::Interner) -> Expr {
    match s {
        ax_spinor::SpinorExpr::AngleBracket(i, j) => Expr::Call(
            interner.get_or_intern("__angle"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
        ),
        ax_spinor::SpinorExpr::SquareBracket(i, j) => Expr::Call(
            interner.get_or_intern("__square"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
        ),
        ax_spinor::SpinorExpr::AngleChain(i, middle, j) => Expr::Call(
            interner.get_or_intern("__angle_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorExpr::SquareChain(i, middle, j) => Expr::Call(
            interner.get_or_intern("__square_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorExpr::AngleSquareChain(i, middle, j) => Expr::Call(
            interner.get_or_intern("__angle_square_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorExpr::SquareAngleChain(i, middle, j) => Expr::Call(
            interner.get_or_intern("__square_angle_chain"),
            std::iter::once(int_expr_u16(i.0))
                .chain(middle.iter().map(|l| int_expr_u16(l.0)))
                .chain(std::iter::once(int_expr_u16(j.0)))
                .collect(),
        ),
        ax_spinor::SpinorExpr::Mandelstam(i, j) => Expr::Call(
            interner.get_or_intern("__mandelstam"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
        ),
        ax_spinor::SpinorExpr::Mandelstam3(i, j, k) => Expr::Call(
            interner.get_or_intern("__mandelstam3"),
            vec![int_expr_u16(i.0), int_expr_u16(j.0), int_expr_u16(k.0)],
        ),
        ax_spinor::SpinorExpr::Product(terms) => Expr::mul(
            terms
                .iter()
                .map(|t| spinor_term_to_expr(t, interner))
                .collect(),
        ),
        ax_spinor::SpinorExpr::Sum(terms) => {
            Expr::add(terms.iter().map(|t| spinor_to_expr(t, interner)).collect())
        }
        ax_spinor::SpinorExpr::Ratio(n, d) => Expr::mul(vec![
            spinor_to_expr(n, interner),
            Expr::pow(spinor_to_expr(d, interner), Expr::Int((-1).into())),
        ]),
        ax_spinor::SpinorExpr::Numeric(r) => rational_or_int(r),
        ax_spinor::SpinorExpr::Power(base, n) => {
            Expr::pow(spinor_to_expr(base, interner), Expr::Int(BigInt::from(*n)))
        }
        ax_spinor::SpinorExpr::Neg(x) => Expr::neg(spinor_to_expr(x, interner)),
    }
}

fn expr_to_spinor(e: &Expr, interner: &ax_ir::Interner) -> Option<ax_spinor::SpinorExpr> {
    match e {
        Expr::Call(f, args) if interner.resolve(*f) == "__angle" && args.len() == 2 => {
            Some(ax_spinor::SpinorExpr::AngleBracket(
                label_from_expr(&args[0], interner)?,
                label_from_expr(&args[1], interner)?,
            ))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__square" && args.len() == 2 => {
            Some(ax_spinor::SpinorExpr::SquareBracket(
                label_from_expr(&args[0], interner)?,
                label_from_expr(&args[1], interner)?,
            ))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__mandelstam" && args.len() == 2 => {
            Some(ax_spinor::SpinorExpr::Mandelstam(
                label_from_expr(&args[0], interner)?,
                label_from_expr(&args[1], interner)?,
            ))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__mandelstam3" && args.len() == 3 => {
            Some(ax_spinor::SpinorExpr::Mandelstam3(
                label_from_expr(&args[0], interner)?,
                label_from_expr(&args[1], interner)?,
                label_from_expr(&args[2], interner)?,
            ))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__mandelstam_multi" && args.len() >= 2 => {
            let labels = labels_from_exprs(args, interner)?;
            Some(ax_spinor::SpinorExpr::Product(vec![
                ax_spinor::SpinorTerm::new(
                    BigRational::from_integer(1.into()),
                    vec![ax_spinor::SpinorFactor::Mandelstam(labels)],
                ),
            ]))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__angle_chain" && args.len() >= 2 => {
            let labels = args
                .iter()
                .map(|arg| label_from_expr(arg, interner))
                .collect::<Option<Vec<_>>>()?;
            Some(ax_spinor::SpinorExpr::AngleChain(
                labels[0],
                labels[1..labels.len() - 1].to_vec(),
                labels[labels.len() - 1],
            ))
        }
        Expr::Call(f, args) if interner.resolve(*f) == "__square_chain" && args.len() >= 2 => {
            let labels = args
                .iter()
                .map(|arg| label_from_expr(arg, interner))
                .collect::<Option<Vec<_>>>()?;
            Some(ax_spinor::SpinorExpr::SquareChain(
                labels[0],
                labels[1..labels.len() - 1].to_vec(),
                labels[labels.len() - 1],
            ))
        }
        Expr::Call(f, args)
            if interner.resolve(*f) == "__angle_square_chain" && args.len() >= 2 =>
        {
            let labels = args
                .iter()
                .map(|arg| label_from_expr(arg, interner))
                .collect::<Option<Vec<_>>>()?;
            Some(ax_spinor::SpinorExpr::AngleSquareChain(
                labels[0],
                labels[1..labels.len() - 1].to_vec(),
                labels[labels.len() - 1],
            ))
        }
        Expr::Call(f, args)
            if interner.resolve(*f) == "__square_angle_chain" && args.len() >= 2 =>
        {
            let labels = args
                .iter()
                .map(|arg| label_from_expr(arg, interner))
                .collect::<Option<Vec<_>>>()?;
            Some(ax_spinor::SpinorExpr::SquareAngleChain(
                labels[0],
                labels[1..labels.len() - 1].to_vec(),
                labels[labels.len() - 1],
            ))
        }
        Expr::Add(terms) => Some(ax_spinor::SpinorExpr::Sum(
            terms
                .iter()
                .map(|t| expr_to_spinor(t, interner))
                .collect::<Option<Vec<_>>>()?,
        )),
        Expr::Neg(inner) => Some(ax_spinor::SpinorExpr::Neg(Box::new(expr_to_spinor(
            inner, interner,
        )?))),
        Expr::Pow(base, exp) => {
            let Expr::Int(n) = exp.as_ref() else {
                return None;
            };
            Some(ax_spinor::SpinorExpr::Power(
                Box::new(expr_to_spinor(base, interner)?),
                n.to_i32()?,
            ))
        }
        Expr::Rational(r) => Some(ax_spinor::SpinorExpr::Numeric(r.clone())),
        Expr::Int(n) => Some(ax_spinor::SpinorExpr::Numeric(BigRational::from_integer(
            n.clone(),
        ))),
        Expr::Mul(factors) => {
            let mut coeff = BigRational::from_integer(1.into());
            let mut spinor_factors = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Int(n) => coeff *= BigRational::from_integer(n.clone()),
                    Expr::Rational(r) => coeff *= r.clone(),
                    Expr::Call(f, args) if interner.resolve(*f) == "__angle" && args.len() == 2 => {
                        spinor_factors.push(ax_spinor::SpinorFactor::Angle(
                            label_from_expr(&args[0], interner)?,
                            label_from_expr(&args[1], interner)?,
                        ));
                    }
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__square" && args.len() == 2 =>
                    {
                        spinor_factors.push(ax_spinor::SpinorFactor::Square(
                            label_from_expr(&args[0], interner)?,
                            label_from_expr(&args[1], interner)?,
                        ));
                    }
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__mandelstam" && args.len() == 2 =>
                    {
                        spinor_factors.push(ax_spinor::SpinorFactor::Mandelstam(vec![
                            label_from_expr(&args[0], interner)?,
                            label_from_expr(&args[1], interner)?,
                        ]));
                    }
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__mandelstam_multi" && args.len() >= 2 =>
                    {
                        spinor_factors.push(ax_spinor::SpinorFactor::Mandelstam(
                            labels_from_exprs(args, interner)?,
                        ));
                    }
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__angle_square_chain" && args.len() >= 2 =>
                    {
                        let labels = labels_from_exprs(args, interner)?;
                        spinor_factors.push(ax_spinor::SpinorFactor::AngleSquare(
                            labels[0],
                            labels[1..labels.len() - 1].to_vec(),
                            labels[labels.len() - 1],
                        ));
                    }
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__square_angle_chain" && args.len() >= 2 =>
                    {
                        let labels = labels_from_exprs(args, interner)?;
                        spinor_factors.push(ax_spinor::SpinorFactor::SquareAngle(
                            labels[0],
                            labels[1..labels.len() - 1].to_vec(),
                            labels[labels.len() - 1],
                        ));
                    }
                    Expr::Sym(s) => spinor_factors.push(ax_spinor::SpinorFactor::SymbolicParam(*s)),
                    Expr::Pow(base, exp) => {
                        let Expr::Int(n) = exp.as_ref() else {
                            return None;
                        };
                        let factor_expr = expr_to_spinor(base, interner)?;
                        let factor = match factor_expr {
                            ax_spinor::SpinorExpr::AngleBracket(i, j) => {
                                ax_spinor::SpinorFactor::Angle(i, j)
                            }
                            ax_spinor::SpinorExpr::SquareBracket(i, j) => {
                                ax_spinor::SpinorFactor::Square(i, j)
                            }
                            ax_spinor::SpinorExpr::Mandelstam(i, j) => {
                                ax_spinor::SpinorFactor::Mandelstam(vec![i, j])
                            }
                            _ => return None,
                        };
                        spinor_factors.push(ax_spinor::SpinorFactor::Power(
                            Box::new(factor),
                            n.to_i32()?,
                        ));
                    }
                    _ => return None,
                }
            }
            Some(ax_spinor::SpinorExpr::Product(vec![
                ax_spinor::SpinorTerm::new(coeff, spinor_factors),
            ]))
        }
        _ => None,
    }
}

fn twistor_to_expr(t: &ax_spinor::twistor::TwistorExpr, interner: &ax_ir::Interner) -> Expr {
    use ax_spinor::twistor::{TwistorExpr, TwistorFactor};
    match t {
        TwistorExpr::FourBracket(i, j, k, l) => Expr::Call(
            interner.get_or_intern("__four_bracket"),
            vec![
                int_expr_u16(i.0),
                int_expr_u16(j.0),
                int_expr_u16(k.0),
                int_expr_u16(l.0),
            ],
        ),
        TwistorExpr::Product(terms) => Expr::mul(
            terms
                .iter()
                .map(|term| {
                    let mut factors = vec![rational_or_int(&term.coefficient)];
                    factors.extend(term.factors.iter().map(|factor| match factor {
                        TwistorFactor::FourBracket(i, j, k, l) => Expr::Call(
                            interner.get_or_intern("__four_bracket"),
                            vec![
                                int_expr_u16(i.0),
                                int_expr_u16(j.0),
                                int_expr_u16(k.0),
                                int_expr_u16(l.0),
                            ],
                        ),
                        TwistorFactor::FundamentalAngle(i, j) => Expr::Call(
                            interner.get_or_intern("__angle"),
                            vec![int_expr_u16(i.0), int_expr_u16(j.0)],
                        ),
                        TwistorFactor::Power(base, n) => Expr::pow(
                            twistor_to_expr(
                                &TwistorExpr::Product(vec![ax_spinor::twistor::TwistorTerm::new(
                                    BigRational::from_integer(1.into()),
                                    vec![base.as_ref().clone()],
                                )]),
                                interner,
                            ),
                            Expr::Int(BigInt::from(*n)),
                        ),
                    }));
                    Expr::mul(factors)
                })
                .collect(),
        ),
        TwistorExpr::Sum(terms) => {
            Expr::add(terms.iter().map(|t| twistor_to_expr(t, interner)).collect())
        }
        TwistorExpr::Ratio(n, d) => Expr::mul(vec![
            twistor_to_expr(n, interner),
            Expr::pow(twistor_to_expr(d, interner), Expr::Int((-1).into())),
        ]),
        TwistorExpr::Numeric(r) => rational_or_int(r),
        TwistorExpr::Power(base, n) => {
            Expr::pow(twistor_to_expr(base, interner), Expr::Int(BigInt::from(*n)))
        }
        TwistorExpr::Neg(x) => Expr::neg(twistor_to_expr(x, interner)),
    }
}

fn expr_to_twistor(
    e: &Expr,
    interner: &ax_ir::Interner,
) -> Option<ax_spinor::twistor::TwistorExpr> {
    use ax_spinor::twistor::{TwistorExpr, TwistorFactor, TwistorTerm};
    match e {
        Expr::Call(f, args) if interner.resolve(*f) == "__four_bracket" && args.len() == 4 => {
            Some(TwistorExpr::FourBracket(
                label_from_expr(&args[0], interner)?,
                label_from_expr(&args[1], interner)?,
                label_from_expr(&args[2], interner)?,
                label_from_expr(&args[3], interner)?,
            ))
        }
        Expr::Add(terms) => Some(TwistorExpr::Sum(
            terms
                .iter()
                .map(|term| expr_to_twistor(term, interner))
                .collect::<Option<Vec<_>>>()?,
        )),
        Expr::Neg(inner) => Some(TwistorExpr::Neg(Box::new(expr_to_twistor(
            inner, interner,
        )?))),
        Expr::Rational(r) => Some(TwistorExpr::Numeric(r.clone())),
        Expr::Int(n) => Some(TwistorExpr::Numeric(BigRational::from_integer(n.clone()))),
        Expr::Pow(base, exp) => {
            let Expr::Int(n) = exp.as_ref() else {
                return None;
            };
            Some(TwistorExpr::Power(
                Box::new(expr_to_twistor(base, interner)?),
                n.to_i32()?,
            ))
        }
        Expr::Mul(factors) => {
            let mut coeff = BigRational::from_integer(1.into());
            let mut twistor_factors = Vec::new();
            for factor in factors {
                match factor {
                    Expr::Int(n) => coeff *= BigRational::from_integer(n.clone()),
                    Expr::Rational(r) => coeff *= r.clone(),
                    Expr::Call(f, args)
                        if interner.resolve(*f) == "__four_bracket" && args.len() == 4 =>
                    {
                        twistor_factors.push(TwistorFactor::FourBracket(
                            label_from_expr(&args[0], interner)?,
                            label_from_expr(&args[1], interner)?,
                            label_from_expr(&args[2], interner)?,
                            label_from_expr(&args[3], interner)?,
                        ));
                    }
                    Expr::Call(f, args) if interner.resolve(*f) == "__angle" && args.len() == 2 => {
                        twistor_factors.push(TwistorFactor::FundamentalAngle(
                            label_from_expr(&args[0], interner)?,
                            label_from_expr(&args[1], interner)?,
                        ));
                    }
                    _ => return None,
                }
            }
            Some(TwistorExpr::Product(vec![TwistorTerm::new(
                coeff,
                twistor_factors,
            )]))
        }
        _ => None,
    }
}

pub fn apply_parallel_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "__set_parallel" || args.len() != 1 {
        return None;
    }
    let Expr::Sym(mode) = &args[0] else {
        return None;
    };
    match interner.resolve(*mode) {
        "on" | "true" => {
            env.parallel = true;
            Some("parallel mode enabled".to_string())
        }
        "off" | "false" => {
            env.parallel = false;
            Some("parallel mode disabled".to_string())
        }
        _ => None,
    }
}

pub fn apply_graded_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "graded" || args.len() != 2 {
        return None;
    }
    let Expr::Sym(sym) = &args[0] else {
        return None;
    };
    let grading = grading_from_expr(&args[1], interner)?;
    env.graded_table.declare(*sym, grading.clone());
    Some(format!(
        "declared grading for {}: {:?}",
        interner.resolve(*sym),
        grading
    ))
}

pub fn apply_superspace_setup(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "setup_superspace" || args.len() != 1 {
        return None;
    }
    match usize_from_expr(&args[0])? {
        1 => {
            let (setup, table) = ax_graded::superspace::setup_n1_superspace(interner);
            env.superspace_setup = Some(setup);
            env.graded_table = table;
            Some("initialized N=1 superspace".to_string())
        }
        _ => Some("N>1 superspace not yet implemented".to_string()),
    }
}

pub fn apply_brst_setup(expr: &Expr, env: &mut Env, interner: &ax_ir::Interner) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "setup_brst_ym" || args.len() != 5 {
        return None;
    }
    let (Some(gauge), Some(ghost), Some(antighost), Some(aux), Some(coupling)) = (
        symbol_from_expr(&args[0]),
        symbol_from_expr(&args[1]),
        symbol_from_expr(&args[2]),
        symbol_from_expr(&args[3]),
        symbol_from_expr(&args[4]),
    ) else {
        return None;
    };
    let (setup, table) =
        ax_graded::brst::setup_yang_mills_brst(gauge, ghost, antighost, aux, coupling, interner);
    env.brst_setup = Some(setup);
    env.graded_table = table;
    Some("initialized Yang-Mills BRST setup".to_string())
}

fn extract_sym_list(expr: &Expr) -> Vec<lasso::Spur> {
    match expr {
        Expr::List(items) => items
            .iter()
            .filter_map(|e| if let Expr::Sym(s) = e { Some(*s) } else { None })
            .collect(),
        Expr::Sym(s) => vec![*s],
        _ => vec![],
    }
}

pub fn parse_component_rules(rule_exprs: &[Expr]) -> Vec<ax_tensor::ComponentRule> {
    let mut rules = Vec::new();
    for rule_expr in rule_exprs {
        let Expr::List(items) = rule_expr else {
            continue;
        };
        if items.len() != 3 {
            continue;
        }
        let Expr::Sym(tensor) = items[0] else {
            continue;
        };
        let Expr::List(index_exprs) = &items[1] else {
            continue;
        };

        let mut indices = Vec::new();
        let mut valid = true;
        for item in index_exprs {
            match item {
                Expr::Indexed(_, concrete_indices) if concrete_indices.len() == 1 => {
                    indices.push((
                        concrete_indices[0].name,
                        concrete_indices[0].variance.clone(),
                    ));
                }
                Expr::Sym(sym) => {
                    indices.push((*sym, ax_ir::Variance::Down));
                }
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }

        rules.push(ax_tensor::ComponentRule {
            tensor,
            indices,
            value: items[2].clone(),
        });
    }
    rules
}

pub fn parse_component_rules_expr(expr: &Expr) -> Vec<ax_tensor::ComponentRule> {
    match expr {
        Expr::List(items) => parse_component_rules(items),
        Expr::Matrix(rows) => {
            let row_exprs = rows
                .iter()
                .map(|row| Expr::List(row.clone()))
                .collect::<Vec<_>>();
            parse_component_rules(&row_exprs)
        }
        _ => Vec::new(),
    }
}

fn grading_rank(grading: Grading) -> usize {
    match grading {
        Grading::Even => 0,
        Grading::Odd => 1,
    }
}

pub fn infer_grading(expr: &Expr, gradings: &HashMap<lasso::Spur, Grading>) -> Grading {
    match expr {
        Expr::Sym(sym) => gradings.get(sym).copied().unwrap_or(Grading::Even),
        Expr::Neg(inner) | Expr::Group(inner, _) => infer_grading(inner, gradings),
        Expr::Mul(factors) => {
            let odd_count = factors
                .iter()
                .filter(|factor| infer_grading(factor, gradings) == Grading::Odd)
                .count();
            if odd_count % 2 == 0 {
                Grading::Even
            } else {
                Grading::Odd
            }
        }
        Expr::Add(terms) => terms
            .first()
            .map(|term| infer_grading(term, gradings))
            .unwrap_or(Grading::Even),
        Expr::Pow(base, exp) => {
            if infer_grading(base, gradings) == Grading::Odd {
                match exp.as_ref() {
                    Expr::Int(n) if n.is_zero() => Grading::Even,
                    Expr::Int(n) if n.is_one() => Grading::Odd,
                    Expr::Int(_) => Grading::Even,
                    _ => Grading::Even,
                }
            } else {
                Grading::Even
            }
        }
        Expr::Complex(re, im) => {
            let re_grade = infer_grading(re, gradings);
            let im_grade = infer_grading(im, gradings);
            if grading_rank(re_grade) >= grading_rank(im_grade) {
                re_grade
            } else {
                im_grade
            }
        }
        Expr::Call(_, _)
        | Expr::FnDef(_, _, _)
        | Expr::Rule(_, _, _)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Piecewise(_)
        | Expr::Indexed(_, _)
        | Expr::Let(_, _, _)
        | Expr::List(_)
        | Expr::Matrix(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => Grading::Even,
    }
}

pub fn grassmann_simplify(
    expr: &Expr,
    gradings: &HashMap<lasso::Spur, Grading>,
    interner: &ax_ir::Interner,
) -> Expr {
    fn grassmann_key(expr: &Expr) -> String {
        format!("{expr:?}")
    }

    fn simplify_product(
        factors: &[Expr],
        gradings: &HashMap<lasso::Spur, Grading>,
        interner: &ax_ir::Interner,
    ) -> Expr {
        let mut flat = Vec::new();
        for factor in factors {
            let simplified = grassmann_simplify(factor, gradings, interner);
            match simplified {
                Expr::Mul(inner) => flat.extend(inner),
                Expr::Pow(base, exp) => match (&*base, &*exp) {
                    (_, Expr::Int(n)) if n.is_zero() => {}
                    (_, Expr::Int(n)) if n.is_one() => flat.push(*base),
                    _ if infer_grading(base.as_ref(), gradings) == Grading::Odd => {
                        if matches!(&*exp, Expr::Int(n) if *n >= 2.into()) {
                            return Expr::zero();
                        }
                        flat.push(Expr::Pow(base, exp));
                    }
                    _ => flat.push(Expr::Pow(base, exp)),
                },
                other => flat.push(other),
            }
        }

        if flat.iter().any(|factor| factor == &Expr::zero()) {
            return Expr::zero();
        }

        for i in 0..flat.len() {
            if infer_grading(&flat[i], gradings) != Grading::Odd {
                continue;
            }
            for j in (i + 1)..flat.len() {
                if infer_grading(&flat[j], gradings) == Grading::Odd && flat[i] == flat[j] {
                    return Expr::zero();
                }
            }
        }

        let mut odd_swaps = 0usize;
        for i in 0..flat.len() {
            for j in (i + 1)..flat.len() {
                if infer_grading(&flat[i], gradings) == Grading::Odd
                    && infer_grading(&flat[j], gradings) == Grading::Odd
                    && grassmann_key(&flat[i]) > grassmann_key(&flat[j])
                {
                    odd_swaps += 1;
                }
            }
        }

        flat.sort_by_key(grassmann_key);
        let product = Expr::mul(flat);
        let product = if odd_swaps % 2 == 1 {
            Expr::neg(product)
        } else {
            product
        };
        eval(&product, &Env::new(), interner)
    }

    match expr {
        Expr::Add(terms) => eval(
            &Expr::add(
                terms
                    .iter()
                    .map(|term| grassmann_simplify(term, gradings, interner))
                    .collect(),
            ),
            &Env::new(),
            interner,
        ),
        Expr::Mul(factors) => simplify_product(factors, gradings, interner),
        Expr::Pow(base, exp) => {
            let simplified_base = grassmann_simplify(base, gradings, interner);
            let simplified_exp = grassmann_simplify(exp, gradings, interner);
            if infer_grading(&simplified_base, gradings) == Grading::Odd {
                match &simplified_exp {
                    Expr::Int(n) if n.is_zero() => Expr::one(),
                    Expr::Int(n) if n.is_one() => simplified_base,
                    Expr::Int(n) if *n >= 2.into() => Expr::zero(),
                    _ => Expr::pow(simplified_base, simplified_exp),
                }
            } else {
                Expr::pow(simplified_base, simplified_exp)
            }
        }
        Expr::Neg(inner) => Expr::neg(grassmann_simplify(inner, gradings, interner)),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(grassmann_simplify(re, gradings, interner)),
            Box::new(grassmann_simplify(im, gradings, interner)),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| grassmann_simplify(arg, gradings, interner))
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(grassmann_simplify(body, gradings, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(grassmann_simplify(lhs, gradings, interner)),
            Box::new(grassmann_simplify(rhs, gradings, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, cond)| (grassmann_simplify(value, gradings, interner), cond.clone()))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(grassmann_simplify(base, gradings, interner)),
            indices.clone(),
        ),
        Expr::Let(name, value, body) => Expr::Let(
            *name,
            Box::new(grassmann_simplify(value, gradings, interner)),
            Box::new(grassmann_simplify(body, gradings, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| grassmann_simplify(item, gradings, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|item| grassmann_simplify(item, gradings, interner))
                        .collect()
                })
                .collect(),
        ),
        _ => expr.clone(),
    }
}

pub fn apply_grassmann_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "grassmann" {
        return None;
    }
    let mut declared = Vec::new();
    for arg in args {
        if let Expr::Sym(sym) = arg {
            env.gradings.insert(*sym, Grading::Odd);
            declared.push(interner.resolve(*sym).to_string());
        }
    }
    if declared.is_empty() {
        None
    } else {
        Some(format!(
            "declared Grassmann variables: {}",
            declared.join(", ")
        ))
    }
}

pub fn apply_operator_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    let kind = match interner.resolve(*f) {
        "creation" => ax_qm::OperatorKind::Creation,
        "annihilation" => ax_qm::OperatorKind::Annihilation,
        _ => return None,
    };
    let mut declared = Vec::new();
    for arg in args {
        if let Expr::Sym(sym) = arg {
            env.operators.insert(*sym, kind);
            env.operator_statistics
                .entry(*sym)
                .or_insert(ax_qm::OperatorStatistics::Bosonic);
            declared.push(interner.resolve(*sym).to_string());
        }
    }
    if declared.is_empty() {
        None
    } else {
        let kind_name = match kind {
            ax_qm::OperatorKind::Creation => "creation",
            ax_qm::OperatorKind::Annihilation => "annihilation",
        };
        Some(format!(
            "declared {kind_name} operators: {}",
            declared.join(", ")
        ))
    }
}

pub fn apply_named_operator_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "declare_operator" || !(args.len() == 2 || args.len() == 3) {
        return None;
    }

    let symbol = match &args[0] {
        Expr::Sym(sym) => *sym,
        _ => return None,
    };
    let kind = match &args[1] {
        Expr::Sym(sym) => match interner.resolve(*sym) {
            "creation" => ax_qm::OperatorKind::Creation,
            "annihilation" => ax_qm::OperatorKind::Annihilation,
            _ => return None,
        },
        _ => return None,
    };
    let statistics = match args.get(2) {
        None => ax_qm::OperatorStatistics::Bosonic,
        Some(Expr::Sym(sym)) => match interner.resolve(*sym) {
            "bosonic" => ax_qm::OperatorStatistics::Bosonic,
            "fermionic" => ax_qm::OperatorStatistics::Fermionic,
            _ => return None,
        },
        Some(_) => return None,
    };

    env.operators.insert(symbol, kind);
    env.operator_statistics.insert(symbol, statistics);
    Some(format!(
        "declared {:?} operator {} with {:?} statistics",
        kind,
        interner.resolve(symbol),
        statistics
    ))
}

pub fn apply_named_contraction_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "declare_contraction" || args.len() != 3 {
        return None;
    }

    let lhs = match &args[0] {
        Expr::Sym(sym) => *sym,
        _ => return None,
    };
    let rhs = match &args[1] {
        Expr::Sym(sym) => *sym,
        _ => return None,
    };
    let value = args[2].clone();

    env.contractions.insert((lhs, rhs), value.clone());
    Some(format!(
        "declared contraction ({}, {}) -> {}",
        interner.resolve(lhs),
        interner.resolve(rhs),
        ax_ir::pretty_print(&value, interner)
    ))
}

pub fn apply_coordinate_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "__declare_coordinates" {
        return None;
    }
    let mut declared = 0usize;
    for arg in args {
        if let Expr::Sym(sym) = arg {
            env.coordinates.insert(*sym);
            declared += 1;
        }
    }
    if declared == 0 {
        None
    } else {
        Some("declared coordinates".to_string())
    }
}

pub fn apply_property_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    fn parse_property_target(
        target: &Expr,
        env: &Env,
    ) -> Option<(lasso::Spur, Option<Vec<SlotSpec>>)> {
        match target {
            Expr::Sym(tensor) => Some((*tensor, None)),
            Expr::Indexed(base, indices) => {
                let Expr::Sym(tensor) = base.as_ref() else {
                    return None;
                };
                let slots: Vec<SlotSpec> = indices
                    .iter()
                    .map(|idx| SlotSpec {
                        variance: idx.variance.clone(),
                        family: env.index_to_family.get(&idx.name).copied(),
                    })
                    .collect();
                Some((*tensor, Some(slots)))
            }
            _ => None,
        }
    }

    fn attach_property(
        env: &mut Env,
        tensor: lasso::Spur,
        pattern_slots: &Option<Vec<SlotSpec>>,
        property: ax_ir::TensorProperty,
    ) {
        for property in crate::property_store::expand_compatible_properties(property) {
            env.tensor_properties
                .entry(tensor)
                .or_default()
                .push(property.clone());
            if let Some(index_slots) = pattern_slots {
                env.property_store.declare(
                    PropertyPattern {
                        base_name: tensor,
                        index_slots: index_slots.clone(),
                    },
                    property,
                );
            } else {
                env.property_store.declare_simple(tensor, property);
            }
        }
    }

    let (target, prop_name, prop_args): (&Expr, &str, &[Expr]) = match interner.resolve(*f) {
        "__declare_property" if args.len() == 2 => {
            let (prop_name, prop_args) = match &args[1] {
                Expr::Sym(prop) => (interner.resolve(*prop), &[][..]),
                Expr::Call(prop, prop_args) => (interner.resolve(*prop), prop_args.as_slice()),
                _ => return None,
            };
            (&args[0], prop_name, prop_args)
        }
        "riemann_tensor" if args.len() == 1 => (&args[0], "riemann_tensor", &[][..]),
        "riemann_symmetry" if args.len() == 1 => (&args[0], "riemann_symmetry", &[][..]),
        "bianchi" | "satisfies_bianchi" if !args.is_empty() => {
            (&args[0], "satisfies_bianchi", &args[1..])
        }
        "weyl" | "weyl_tensor" if args.len() == 1 => (&args[0], "weyl_tensor", &[][..]),
        "dimension_dependent_identity" if args.len() == 1 => {
            (&args[0], "dimension_dependent_identity", &[][..])
        }
        "tableau_symmetry" if args.len() == 3 => (&args[0], "tableau_symmetry", &args[1..]),
        "derivative" if args.len() == 1 => (&args[0], "derivative", &[][..]),
        "partial_derivative" if args.len() == 1 => (&args[0], "partial_derivative", &[][..]),
        "covariant_derivative" if args.len() == 1 => (&args[0], "covariant_derivative", &[][..]),
        "tableau_inherit" if args.len() == 1 => (&args[0], "tableau_inherit", &[][..]),
        "declare_spinor_meta" if args.len() == 5 => (&args[0], "declare_spinor_meta", &args[1..]),
        "declare_gamma_matrix_meta" if args.len() == 5 => {
            (&args[0], "declare_gamma_matrix_meta", &args[1..])
        }
        "declare_dirac_bar_meta" if args.len() == 4 => {
            (&args[0], "declare_dirac_bar_meta", &args[1..])
        }
        "declare_trace_space" if args.len() == 3 => (&args[0], "declare_trace_space", &args[1..]),
        "declare_hilbert_space" if args.len() == 2 => {
            (&args[0], "declare_hilbert_space", &args[1..])
        }
        "declare_composite_space" if args.len() == 2 => {
            (&args[0], "declare_composite_space", &args[1..])
        }
        "declare_quantum_object" if args.len() == 3 => {
            (&args[0], "declare_quantum_object", &args[1..])
        }
        _ => return None,
    };
    let (tensor, pattern_slots) = match parse_property_target(target, env) {
        Some(parsed) => parsed,
        None => return None,
    };
    let (tensor, pattern_slots) = (tensor, pattern_slots);
    let default_positions = vec![0, 1];
    let parse_positions = |items: &[Expr]| -> Option<Vec<usize>> {
        match items.first() {
            Some(Expr::List(entries)) => entries
                .iter()
                .map(|entry| match entry {
                    Expr::Int(n) => n.to_usize(),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let parse_usize_list = |expr: &Expr| -> Option<Vec<usize>> {
        match expr {
            Expr::List(entries) => entries
                .iter()
                .map(|entry| match entry {
                    Expr::Int(n) => n.to_usize(),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let mut add_property = |property: ax_ir::TensorProperty| {
        attach_property(env, tensor, &pattern_slots, property);
    };
    match prop_name {
        "metric" => {
            add_property(ax_ir::TensorProperty::Metric);
            add_property(ax_ir::TensorProperty::Symmetric(default_positions.clone()));
            Some(format!(
                "attached property metric (symmetric) to {}",
                interner.resolve(tensor)
            ))
        }
        "symmetric" => {
            add_property(ax_ir::TensorProperty::Symmetric(
                parse_positions(prop_args).unwrap_or_else(|| default_positions.clone()),
            ));
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "antisymmetric" => {
            add_property(ax_ir::TensorProperty::AntiSymmetric(
                parse_positions(prop_args).unwrap_or_else(|| default_positions.clone()),
            ));
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "inverse_metric" => {
            add_property(ax_ir::TensorProperty::InverseMetric);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "kronecker_delta" | "kronecker" => {
            add_property(ax_ir::TensorProperty::KroneckerDelta);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "epsilon" | "epsilon_tensor" => {
            add_property(ax_ir::TensorProperty::EpsilonTensor);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "riemann" | "riemann_symmetry" => {
            add_property(ax_ir::TensorProperty::RiemannSymmetry);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "riemann_tensor" => {
            add_property(ax_ir::TensorProperty::RiemannSymmetry);
            add_property(ax_ir::TensorProperty::SatisfiesBianchi {
                slots: vec![0, 1, 2, 3],
            });
            Some(format!(
                "attached composite property riemann_tensor to {}",
                interner.resolve(tensor)
            ))
        }
        "traceless" => {
            add_property(ax_ir::TensorProperty::Traceless);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "diagonal" => {
            add_property(ax_ir::TensorProperty::Diagonal);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "trace" => {
            add_property(ax_ir::TensorProperty::Trace);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "derivative" => {
            add_property(ax_ir::TensorProperty::Derivative);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "partial_derivative" => {
            add_property(ax_ir::TensorProperty::PartialDerivative);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "covariant_derivative" => {
            add_property(ax_ir::TensorProperty::CovariantDerivative);
            add_property(ax_ir::TensorProperty::TableauInherit);
            Some(format!(
                "attached property {} to {}",
                prop_name,
                interner.resolve(tensor)
            ))
        }
        "spinor" => {
            add_property(ax_ir::TensorProperty::Spinor);
            Some(format!(
                "attached property spinor to {}",
                interner.resolve(tensor)
            ))
        }
        "dirac_bar" | "diracbar" => {
            add_property(ax_ir::TensorProperty::DiracBar);
            Some(format!(
                "attached property dirac_bar to {}",
                interner.resolve(tensor)
            ))
        }
        "gamma_matrix" => {
            add_property(ax_ir::TensorProperty::GammaMatrixProp);
            Some(format!(
                "attached property gamma_matrix to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_spinor_meta" => {
            let metadata = ax_ir::SpinorMetadata {
                dimension: parse_optional_usize_expr(&prop_args[0], interner)?,
                class: parse_spinor_class_expr(&prop_args[1], interner)?,
                chirality: parse_optional_chirality_expr(&prop_args[2], interner)?,
                index_family: parse_optional_symbol_expr(&prop_args[3], interner)?,
            };
            add_property(ax_ir::TensorProperty::SpinorMeta(metadata));
            Some(format!(
                "attached property spinor_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_gamma_matrix_meta" => {
            let metadata = ax_ir::GammaMatrixMetadata {
                dimension: parse_optional_usize_expr(&prop_args[0], interner)?,
                metric_symbol: parse_optional_symbol_expr(&prop_args[1], interner)?,
                index_family: parse_optional_symbol_expr(&prop_args[2], interner)?,
                has_gamma5: parse_bool_like_expr(&prop_args[3], interner)?,
            };
            add_property(ax_ir::TensorProperty::GammaMatrixMeta(metadata));
            Some(format!(
                "attached property gamma_matrix_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_dirac_bar_meta" => {
            let metadata = ax_ir::DiracBarMetadata {
                gamma_symbol: parse_optional_symbol_expr(&prop_args[0], interner)?,
                spinor_family: parse_optional_symbol_expr(&prop_args[1], interner)?,
                reverse_gamma_order: parse_bool_like_expr(&prop_args[2], interner)?,
            };
            add_property(ax_ir::TensorProperty::DiracBarMeta(metadata));
            Some(format!(
                "attached property dirac_bar_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_trace_space" => {
            let metadata = ax_ir::TraceSpaceMetadata {
                space_symbol: symbol_from_expr(&prop_args[0])?,
                cyclic: parse_bool_like_expr(&prop_args[1], interner)?,
            };
            add_property(ax_ir::TensorProperty::TraceSpaceMeta(metadata));
            Some(format!(
                "attached property trace_space_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_hilbert_space" => {
            let dimension = usize_from_expr(&prop_args[0]).filter(|dim| *dim > 0)?;
            apply_hilbert_space_declaration(
                env,
                tensor,
                ax_ir::HilbertSpaceMetadata {
                    dimension,
                    factors: vec![ax_ir::HilbertSpaceFactor {
                        symbol: tensor,
                        dimension,
                    }],
                },
            );
            Some(format!(
                "attached property hilbert_space_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_composite_space" => {
            let factors = symbol_list_from_expr(&prop_args[0])?;
            let flattened = flatten_hilbert_space_factors(env, &factors)?;
            let dimension = flattened.iter().map(|factor| factor.dimension).product();
            apply_hilbert_space_declaration(
                env,
                tensor,
                ax_ir::HilbertSpaceMetadata {
                    dimension,
                    factors: flattened,
                },
            );
            Some(format!(
                "attached property hilbert_space_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "declare_quantum_object" => {
            let kind = parse_quantum_object_kind_expr(&prop_args[0], interner)?;
            let space_symbol = symbol_from_expr(&prop_args[1])?;
            hilbert_space_metadata_of_symbol(env, space_symbol)?;
            apply_quantum_object_declaration(
                env,
                tensor,
                ax_ir::QuantumObjectMetadata { kind, space_symbol },
            );
            Some(format!(
                "attached property quantum_object_meta to {}",
                interner.resolve(tensor)
            ))
        }
        "commuting" => {
            add_property(ax_ir::TensorProperty::Commuting);
            Some(format!(
                "attached property commuting to {}",
                interner.resolve(tensor)
            ))
        }
        "anticommuting" | "anti_commuting" => {
            add_property(ax_ir::TensorProperty::AntiCommuting);
            Some(format!(
                "attached property anticommuting to {}",
                interner.resolve(tensor)
            ))
        }
        "noncommuting" | "non_commuting" => {
            add_property(ax_ir::TensorProperty::NonCommuting);
            Some(format!(
                "attached property noncommuting to {}",
                interner.resolve(tensor)
            ))
        }
        "bianchi" | "satisfies_bianchi" => {
            let slots = parse_positions(prop_args)
                .filter(|positions| positions.len() == 3 || positions.len() == 4)
                .unwrap_or_else(|| vec![0, 1, 2, 3]);
            add_property(ax_ir::TensorProperty::SatisfiesBianchi { slots });
            Some(format!(
                "attached property satisfies_bianchi to {}",
                interner.resolve(tensor)
            ))
        }
        "dimension_dependent_identity" => {
            add_property(ax_ir::TensorProperty::DimensionDependentIdentity);
            Some(format!(
                "attached property dimension_dependent_identity to {}",
                interner.resolve(tensor)
            ))
        }
        "tableau_inherit" => {
            add_property(ax_ir::TensorProperty::TableauInherit);
            Some(format!(
                "attached property tableau_inherit to {}",
                interner.resolve(tensor)
            ))
        }
        "weyl" | "weyl_tensor" => {
            add_property(ax_ir::TensorProperty::WeylTensor);
            Some(format!(
                "attached property weyl_tensor to {}",
                interner.resolve(tensor)
            ))
        }
        "tableau_symmetry" => {
            if prop_args.len() == 2 {
                let shape = parse_usize_list(&prop_args[0])?;
                let indices = parse_usize_list(&prop_args[1])?;
                let symmetry = ax_ir::TensorSymmetry {
                    tableaux: vec![ax_ir::TableauAttachment {
                        shape,
                        slot_map: indices,
                        multiplicity_numer: 1,
                        multiplicity_denom: 1,
                        duality: ax_ir::DualityKind::None,
                        restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                        trace_free: false,
                        dimension_guard: None,
                        source: ax_ir::SymmetrySource::Declared,
                        label: None,
                    }],
                    inherits_under_derivative: false,
                    inherits_under_tensor_product: false,
                    inherits_under_contraction: false,
                    preserves_trace_free_under_projection: false,
                };
                if symmetry.validate().is_err() {
                    return None;
                }
                add_property(ax_ir::TensorProperty::TableauSymmetry(symmetry));
                if let Expr::Indexed(_, declared_indices) = target {
                    find_tensor_symmetry(env, tensor, declared_indices)?;
                }
                Some(format!(
                    "attached property tableau_symmetry to {}",
                    interner.resolve(tensor)
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn apply_index_declaration(
    expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    let Expr::Call(f, args) = expr else {
        return None;
    };
    if interner.resolve(*f) != "__declare_indices" || args.len() < 2 {
        return None;
    }

    let (Expr::Sym(family_name), Expr::List(index_syms)) = (&args[0], &args[1]) else {
        return None;
    };

    let mut dim = None;
    let mut values = Vec::new();
    for arg in args.iter().skip(2).take(args.len().saturating_sub(3)) {
        match arg {
            Expr::Int(n) => dim = n.to_usize(),
            Expr::List(items) => {
                values = items
                    .iter()
                    .filter_map(|item| match item {
                        Expr::Sym(sym) => Some(*sym),
                        _ => None,
                    })
                    .collect();
            }
            _ => {}
        }
    }
    let position = match args.last() {
        Some(Expr::Sym(sym)) if interner.resolve(*sym) == "fixed" => ax_ir::IndexPosition::Fixed,
        _ => ax_ir::IndexPosition::Free,
    };

    env.index_families.insert(
        *family_name,
        ax_ir::IndexFamily {
            name: *family_name,
            values,
            position,
            dimension: dim,
            parent: None,
        },
    );
    for index_expr in index_syms {
        if let Expr::Sym(sym) = index_expr {
            env.index_to_family.insert(*sym, *family_name);
        }
    }
    env.property_store
        .set_index_to_family(env.index_to_family.clone());
    env.property_store
        .set_index_families(env.index_families.clone());
    Some(format!(
        "declared index family: {}",
        interner.resolve(*family_name)
    ))
}

fn symbols_with_property<F>(env: &Env, mut predicate: F) -> HashSet<lasso::Spur>
where
    F: FnMut(&ax_ir::TensorProperty) -> bool,
{
    env.property_store
        .symbols()
        .into_iter()
        .filter(|sym| {
            env.property_store
                .get_all(*sym)
                .iter()
                .any(|prop| predicate(prop))
        })
        .collect()
}

fn explicit_depends_map(env: &Env) -> HashMap<lasso::Spur, Vec<lasso::Spur>> {
    env.property_store
        .symbols()
        .into_iter()
        .filter_map(|sym| {
            env.property_store
                .get_all(sym)
                .into_iter()
                .find_map(|prop| match prop {
                    ax_ir::TensorProperty::Depends(deps) => Some((sym, deps.clone())),
                    _ => None,
                })
        })
        .collect()
}

fn parse_metric_signature(value: &str) -> Option<ax_ir::MetricSignature> {
    match value {
        "mostly_plus" => Some(ax_ir::MetricSignature::MostlyPlus),
        "mostly_minus" => Some(ax_ir::MetricSignature::MostlyMinus),
        _ => None,
    }
}

fn parse_riemann_sign(value: &str) -> Option<ax_ir::RiemannSign> {
    match value {
        "mtw" => Some(ax_ir::RiemannSign::MTW),
        "weinberg" => Some(ax_ir::RiemannSign::Weinberg),
        _ => None,
    }
}

fn parse_ricci_contraction(value: &str) -> Option<ax_ir::RicciContraction> {
    match value {
        "first_third" => Some(ax_ir::RicciContraction::FirstThird),
        "first_fourth" => Some(ax_ir::RicciContraction::FirstFourth),
        _ => None,
    }
}

fn parse_levi_civita_norm(value: &str) -> Option<ax_ir::LeviCivitaNorm> {
    match value {
        "plus_one" => Some(ax_ir::LeviCivitaNorm::PlusOne),
        "minus_one" => Some(ax_ir::LeviCivitaNorm::MinusOne),
        "sqrt_g" => Some(ax_ir::LeviCivitaNorm::SqrtG),
        _ => None,
    }
}

fn parse_fourier_sign(value: &str) -> Option<ax_ir::FourierSign> {
    match value {
        "minus_i" => Some(ax_ir::FourierSign::MinusI),
        "plus_i" => Some(ax_ir::FourierSign::PlusI),
        _ => None,
    }
}

pub fn describe_convention(convention: &ax_ir::Convention) -> String {
    format!(
        "metric_signature={:?}, riemann_sign={:?}, ricci_contraction={:?}, levi_civita_norm={:?}, fourier_sign={:?}",
        convention.metric_signature,
        convention.riemann_sign,
        convention.ricci_contraction,
        convention.levi_civita_norm,
        convention.fourier_sign
    )
}

pub fn apply_set_convention(expr: &Expr, env: &mut Env) -> Option<String> {
    let Expr::SetConvention(field, value) = expr else {
        return None;
    };

    let applied = match field.as_str() {
        "metric_signature" => parse_metric_signature(value).map(|parsed| {
            env.convention.metric_signature = parsed;
        }),
        "riemann_sign" => parse_riemann_sign(value).map(|parsed| {
            env.convention.riemann_sign = parsed;
        }),
        "ricci_contraction" => parse_ricci_contraction(value).map(|parsed| {
            env.convention.ricci_contraction = parsed;
        }),
        "levi_civita_norm" => parse_levi_civita_norm(value).map(|parsed| {
            env.convention.levi_civita_norm = parsed;
        }),
        "fourier_sign" => parse_fourier_sign(value).map(|parsed| {
            env.convention.fourier_sign = parsed;
        }),
        _ => None,
    };

    applied.map(|_| describe_convention(&env.convention))
}

pub fn check_convention_compatible(a: &ax_ir::Convention, b: &ax_ir::Convention) -> Vec<String> {
    let mut warnings = Vec::new();
    if a.metric_signature != b.metric_signature {
        warnings.push(format!(
            "metric signature mismatch: {:?} vs {:?}",
            a.metric_signature, b.metric_signature
        ));
    }
    if a.riemann_sign != b.riemann_sign {
        warnings.push(format!(
            "Riemann sign convention mismatch: {:?} vs {:?}",
            a.riemann_sign, b.riemann_sign
        ));
    }
    if a.ricci_contraction != b.ricci_contraction {
        warnings.push(format!(
            "Ricci contraction mismatch: {:?} vs {:?}",
            a.ricci_contraction, b.ricci_contraction
        ));
    }
    if a.levi_civita_norm != b.levi_civita_norm {
        warnings.push(format!(
            "Levi-Civita normalization mismatch: {:?} vs {:?}",
            a.levi_civita_norm, b.levi_civita_norm
        ));
    }
    if a.fourier_sign != b.fourier_sign {
        warnings.push(format!(
            "Fourier sign mismatch: {:?} vs {:?}",
            a.fourier_sign, b.fourier_sign
        ));
    }
    warnings
}

fn has_assumption(env: &Env, sym: lasso::Spur, assumption: &Assumption) -> bool {
    env.assumptions
        .get(&sym)
        .is_some_and(|assumptions| assumptions.contains(assumption))
        || env
            .parent
            .as_deref()
            .is_some_and(|parent| has_assumption(parent, sym, assumption))
}

fn to_rational(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Int(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn numeric_pow(base: &Expr, exp: &Expr) -> Option<Expr> {
    let base_r = to_rational(base)?;
    match exp {
        Expr::Int(n) => {
            if let Some(pow) = n.to_u32() {
                let numer = base_r.numer().clone().pow(pow);
                let denom = base_r.denom().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else if n.is_negative() {
                let pow = (-n).to_u32()?;
                let numer = base_r.denom().clone().pow(pow);
                let denom = base_r.numer().clone().pow(pow);
                let out = BigRational::new(numer, denom);
                if out.is_integer() {
                    Some(Expr::Int(out.to_integer()))
                } else {
                    Some(Expr::Rational(out))
                }
            } else {
                None
            }
        }
        Expr::Rational(_) => None,
        _ => None,
    }
}

fn perfect_square_root(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    let root = n.sqrt();
    if &root * &root == *n {
        Some(root)
    } else {
        None
    }
}

fn one_half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn diff_call(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    let diff_sym = interner.get_or_intern("diff");
    Expr::Call(diff_sym, vec![expr.clone(), Expr::Sym(var)])
}

fn builtin_unary(name: &str, arg: Expr, interner: &ax_ir::Interner) -> Expr {
    Expr::Call(interner.get_or_intern(name), vec![arg])
}

fn is_unevaluated_integrate_check(expr: &Expr, interner: &ax_ir::Interner) -> bool {
    match expr {
        Expr::Call(f, _) => interner.resolve(*f) == "integrate",
        Expr::Add(terms) => terms
            .iter()
            .any(|t| is_unevaluated_integrate_check(t, interner)),
        Expr::Mul(factors) => factors
            .iter()
            .any(|f| is_unevaluated_integrate_check(f, interner)),
        Expr::Neg(e) => is_unevaluated_integrate_check(e, interner),
        _ => false,
    }
}

fn trig_special_from_rational(
    func: &str,
    r: &BigRational,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let two = BigRational::from_integer(2.into());
    let mut normalized = r.clone();
    while normalized < BigRational::zero() {
        normalized += two.clone();
    }
    while normalized >= two {
        normalized -= two.clone();
    }

    let half = BigRational::new(1.into(), 2.into());
    let third = BigRational::new(1.into(), 3.into());
    let quarter = BigRational::new(1.into(), 4.into());
    let sixth = BigRational::new(1.into(), 6.into());
    let one = BigRational::from_integer(1.into());
    let zero = BigRational::zero();

    match func {
        "sin" => {
            if normalized == zero || normalized == one {
                Some(Expr::Int(0.into()))
            } else if normalized == sixth {
                Some(Expr::Rational(BigRational::new(1.into(), 2.into())))
            } else if normalized == quarter {
                Some(Expr::mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    builtin_unary("sqrt", Expr::Int(2.into()), interner),
                ]))
            } else if normalized == third {
                Some(Expr::mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    builtin_unary("sqrt", Expr::Int(3.into()), interner),
                ]))
            } else if normalized == half {
                Some(Expr::Int(1.into()))
            } else {
                None
            }
        }
        "cos" => {
            if normalized == zero {
                Some(Expr::Int(1.into()))
            } else if normalized == sixth {
                Some(Expr::mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    builtin_unary("sqrt", Expr::Int(3.into()), interner),
                ]))
            } else if normalized == quarter {
                Some(Expr::mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    builtin_unary("sqrt", Expr::Int(2.into()), interner),
                ]))
            } else if normalized == third {
                Some(Expr::Rational(BigRational::new(1.into(), 2.into())))
            } else if normalized == half {
                Some(Expr::Int(0.into()))
            } else if normalized == one {
                Some(Expr::Int((-1i64).into()))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_trig_special_value(
    func: &str,
    factors: &[Expr],
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    let pi_sym = interner.get_or_intern("pi");
    let coeff = if factors.len() == 2 {
        if let Expr::Sym(s) = &factors[1] {
            if *s == pi_sym {
                factors[0].clone()
            } else {
                return None;
            }
        } else if let Expr::Float(v) = &factors[1] {
            if (*v - std::f64::consts::PI).abs() < 1e-12 {
                factors[0].clone()
            } else {
                return None;
            }
        } else if let Expr::Sym(s) = &factors[0] {
            if *s == pi_sym {
                factors[1].clone()
            } else {
                return None;
            }
        } else if let Expr::Float(v) = &factors[0] {
            if (*v - std::f64::consts::PI).abs() < 1e-12 {
                factors[1].clone()
            } else {
                return None;
            }
        } else {
            return None;
        }
    } else {
        return None;
    };

    let r = match &coeff {
        Expr::Rational(r) => r.clone(),
        Expr::Int(n) => BigRational::from_integer(n.clone()),
        _ => return None,
    };

    trig_special_from_rational(func, &r, interner)
}

fn try_trig_special_float(func: &str, value: f64, interner: &ax_ir::Interner) -> Option<Expr> {
    let ratio = value / std::f64::consts::PI;
    let specials = [
        BigRational::from_integer(0.into()),
        BigRational::new(1.into(), 6.into()),
        BigRational::new(1.into(), 4.into()),
        BigRational::new(1.into(), 3.into()),
        BigRational::new(1.into(), 2.into()),
        BigRational::from_integer(1.into()),
    ];
    for r in specials {
        if (ratio - to_f64(&Expr::Rational(r.clone()))?).abs() < 1e-12 {
            return trig_special_from_rational(func, &r, interner);
        }
    }
    None
}

fn collapse_duplicate_sum_terms(terms: Vec<Expr>) -> Expr {
    let mut grouped: Vec<(Expr, usize)> = Vec::new();

    for term in terms {
        if let Some((_, count)) = grouped.iter_mut().find(|(existing, _)| *existing == term) {
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

fn substitute_condition(
    condition: &Condition,
    target: &Expr,
    replacement: &Expr,
    interner: &ax_ir::Interner,
) -> Condition {
    match condition {
        Condition::Gt(a, b) => Condition::Gt(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::Lt(a, b) => Condition::Lt(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::Ge(a, b) => Condition::Ge(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::Le(a, b) => Condition::Le(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::Eq(a, b) => Condition::Eq(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::Ne(a, b) => Condition::Ne(
            symbolic_substitute(a, target, replacement, interner),
            symbolic_substitute(b, target, replacement, interner),
        ),
        Condition::And(a, b) => Condition::And(
            Box::new(substitute_condition(a, target, replacement, interner)),
            Box::new(substitute_condition(b, target, replacement, interner)),
        ),
        Condition::Or(a, b) => Condition::Or(
            Box::new(substitute_condition(a, target, replacement, interner)),
            Box::new(substitute_condition(b, target, replacement, interner)),
        ),
        Condition::Not(c) => Condition::Not(Box::new(substitute_condition(
            c,
            target,
            replacement,
            interner,
        ))),
        Condition::True => Condition::True,
        Condition::False => Condition::False,
    }
}

fn multi_substitute_condition(
    condition: &Condition,
    substitutions: &[(Expr, Expr)],
    interner: &ax_ir::Interner,
) -> Condition {
    match condition {
        Condition::Gt(a, b) => Condition::Gt(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::Lt(a, b) => Condition::Lt(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::Ge(a, b) => Condition::Ge(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::Le(a, b) => Condition::Le(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::Eq(a, b) => Condition::Eq(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::Ne(a, b) => Condition::Ne(
            multi_substitute(a, substitutions, interner),
            multi_substitute(b, substitutions, interner),
        ),
        Condition::And(a, b) => Condition::And(
            Box::new(multi_substitute_condition(a, substitutions, interner)),
            Box::new(multi_substitute_condition(b, substitutions, interner)),
        ),
        Condition::Or(a, b) => Condition::Or(
            Box::new(multi_substitute_condition(a, substitutions, interner)),
            Box::new(multi_substitute_condition(b, substitutions, interner)),
        ),
        Condition::Not(c) => Condition::Not(Box::new(multi_substitute_condition(
            c,
            substitutions,
            interner,
        ))),
        Condition::True => Condition::True,
        Condition::False => Condition::False,
    }
}

pub fn symbolic_substitute(
    expr: &Expr,
    target: &Expr,
    replacement: &Expr,
    interner: &ax_ir::Interner,
) -> Expr {
    if expr == target {
        return replacement.clone();
    }

    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| symbolic_substitute(term, target, replacement, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| symbolic_substitute(factor, target, replacement, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            symbolic_substitute(base, target, replacement, interner),
            symbolic_substitute(exp, target, replacement, interner),
        ),
        Expr::Neg(inner) => Expr::neg(symbolic_substitute(inner, target, replacement, interner)),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(symbolic_substitute(inner, target, replacement, interner)),
            *rel,
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| symbolic_substitute(arg, target, replacement, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(symbolic_substitute(re, target, replacement, interner)),
            Box::new(symbolic_substitute(im, target, replacement, interner)),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(symbolic_substitute(base, target, replacement, interner)),
            indices.clone(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(symbolic_substitute(val, target, replacement, interner)),
            Box::new(symbolic_substitute(body, target, replacement, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| symbolic_substitute(item, target, replacement, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| symbolic_substitute(cell, target, replacement, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(symbolic_substitute(body, target, replacement, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(symbolic_substitute(lhs, target, replacement, interner)),
            Box::new(symbolic_substitute(rhs, target, replacement, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        symbolic_substitute(value, target, replacement, interner),
                        substitute_condition(condition, target, replacement, interner),
                    )
                })
                .collect(),
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

pub fn multi_substitute(
    expr: &Expr,
    substitutions: &[(Expr, Expr)],
    interner: &ax_ir::Interner,
) -> Expr {
    for (target, replacement) in substitutions {
        if expr == target {
            return replacement.clone();
        }
    }

    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| multi_substitute(term, substitutions, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|factor| multi_substitute(factor, substitutions, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => Expr::pow(
            multi_substitute(base, substitutions, interner),
            multi_substitute(exp, substitutions, interner),
        ),
        Expr::Neg(inner) => Expr::neg(multi_substitute(inner, substitutions, interner)),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(multi_substitute(inner, substitutions, interner)),
            *rel,
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|arg| multi_substitute(arg, substitutions, interner))
                .collect(),
        ),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(multi_substitute(re, substitutions, interner)),
            Box::new(multi_substitute(im, substitutions, interner)),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(multi_substitute(base, substitutions, interner)),
            indices.clone(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            Box::new(multi_substitute(val, substitutions, interner)),
            Box::new(multi_substitute(body, substitutions, interner)),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| multi_substitute(item, substitutions, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| multi_substitute(cell, substitutions, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(multi_substitute(body, substitutions, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(multi_substitute(lhs, substitutions, interner)),
            Box::new(multi_substitute(rhs, substitutions, interner)),
            *trust,
        ),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| {
                    (
                        multi_substitute(value, substitutions, interner),
                        multi_substitute_condition(condition, substitutions, interner),
                    )
                })
                .collect(),
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

fn contains_var(expr: &Expr, var: lasso::Spur) -> bool {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => false,
        Expr::Complex(re, im) => contains_var(re, var) || contains_var(im, var),
        Expr::Sym(s) => *s == var,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(|term| contains_var(term, var))
        }
        Expr::Pow(base, exp) => contains_var(base, var) || contains_var(exp, var),
        Expr::Neg(e) | Expr::Group(e, _) => contains_var(e, var),
        Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
        Expr::FnDef(_, _, body) => contains_var(body, var),
        Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::Import(_) => false,
        Expr::Assume(_, _) => false,
        Expr::SetConvention(_, _) => false,
        Expr::Piecewise(cases) => cases.iter().any(|(value, condition)| {
            contains_var(value, var) || condition_contains_var(condition, var)
        }),
        Expr::Indexed(base, _) => contains_var(base, var),
        Expr::Let(_, val, body) => contains_var(val, var) || contains_var(body, var),
        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|cell| contains_var(cell, var))),
    }
}

fn condition_contains_var(condition: &Condition, var: lasso::Spur) -> bool {
    match condition {
        Condition::Gt(a, b)
        | Condition::Lt(a, b)
        | Condition::Ge(a, b)
        | Condition::Le(a, b)
        | Condition::Eq(a, b)
        | Condition::Ne(a, b) => contains_var(a, var) || contains_var(b, var),
        Condition::And(a, b) | Condition::Or(a, b) => {
            condition_contains_var(a, var) || condition_contains_var(b, var)
        }
        Condition::Not(c) => condition_contains_var(c, var),
        Condition::True | Condition::False => false,
    }
}

fn compare_numeric(a: &Expr, b: &Expr) -> Option<Ordering> {
    let fa = to_f64(a)?;
    let fb = to_f64(b)?;
    fa.partial_cmp(&fb)
}

fn canonical_equiv_form(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let expanded = simplify::expand(expr, interner);
    let collected = simplify::collect_terms(&expanded, interner);
    eval(&collected, &Env::new(), interner)
}

fn contains_indexed_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(contains_indexed_expr)
        }
        Expr::Pow(base, exp) => contains_indexed_expr(base) || contains_indexed_expr(exp),
        Expr::Neg(inner) => contains_indexed_expr(inner),
        Expr::Call(_, args) => args.iter().any(contains_indexed_expr),
        Expr::Complex(re, im) => contains_indexed_expr(re) || contains_indexed_expr(im),
        Expr::FnDef(_, _, body) => contains_indexed_expr(body),
        Expr::Rule(lhs, rhs, _) => contains_indexed_expr(lhs) || contains_indexed_expr(rhs),
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_indexed_expr(value)),
        Expr::Let(_, value, body) => contains_indexed_expr(value) || contains_indexed_expr(body),
        Expr::Matrix(rows) => rows.iter().flatten().any(contains_indexed_expr),
        _ => false,
    }
}

fn collect_unbound_syms(expr: &Expr, out: &mut Vec<lasso::Spur>) {
    match expr {
        Expr::Sym(s) => out.push(*s),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_unbound_syms(term, out);
            }
        }
        Expr::Pow(base, exp) => {
            collect_unbound_syms(base, out);
            collect_unbound_syms(exp, out);
        }
        Expr::Neg(inner) => collect_unbound_syms(inner, out),
        Expr::Call(_, args) => {
            for arg in args {
                collect_unbound_syms(arg, out);
            }
        }
        Expr::Complex(re, im) => {
            collect_unbound_syms(re, out);
            collect_unbound_syms(im, out);
        }
        Expr::FnDef(_, _, body) => collect_unbound_syms(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_unbound_syms(lhs, out);
            collect_unbound_syms(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_unbound_syms(value, out);
            }
        }
        Expr::Indexed(base, _) => collect_unbound_syms(base, out),
        Expr::Let(_, value, body) => {
            collect_unbound_syms(value, out);
            collect_unbound_syms(body, out);
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_unbound_syms(cell, out);
                }
            }
        }
        _ => {}
    }
}

fn equiv_sample_check(a: &Expr, b: &Expr, env: &Env, interner: &ax_ir::Interner) -> Option<bool> {
    let mut syms = Vec::new();
    collect_unbound_syms(a, &mut syms);
    collect_unbound_syms(b, &mut syms);
    let reserved = ["pi", "e", "i", "inf", "infty", "neg_inf"];
    syms.retain(|s| env.lookup(*s).is_none() && !reserved.contains(&interner.resolve(*s)));
    syms.sort();
    syms.dedup();
    if syms.is_empty() {
        return None;
    }

    fn numeric_eval_expr(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> Option<f64> {
        match expr {
            Expr::Int(n) => num_traits::ToPrimitive::to_f64(n),
            Expr::Rational(r) => Some(
                num_traits::ToPrimitive::to_f64(r.numer())?
                    / num_traits::ToPrimitive::to_f64(r.denom())?,
            ),
            Expr::Float(f) => Some(*f),
            Expr::Complex(re, im) => {
                let re = numeric_eval_expr(re, env, interner)?;
                let im = numeric_eval_expr(im, env, interner)?;
                if im == 0.0 {
                    Some(re)
                } else {
                    None
                }
            }
            Expr::Sym(s) => {
                if let Some(bound) = env.lookup(*s) {
                    numeric_eval_expr(bound, env, interner)
                } else {
                    match interner.resolve(*s) {
                        "pi" => Some(std::f64::consts::PI),
                        "e" => Some(std::f64::consts::E),
                        _ => None,
                    }
                }
            }
            Expr::Add(terms) => {
                let mut acc = 0.0;
                for term in terms {
                    acc += numeric_eval_expr(term, env, interner)?;
                }
                Some(acc)
            }
            Expr::Mul(factors) => {
                let mut acc = 1.0;
                for factor in factors {
                    acc *= numeric_eval_expr(factor, env, interner)?;
                }
                Some(acc)
            }
            Expr::Pow(base, exp) => Some(
                numeric_eval_expr(base, env, interner)?
                    .powf(numeric_eval_expr(exp, env, interner)?),
            ),
            Expr::Neg(inner) => Some(-numeric_eval_expr(inner, env, interner)?),
            Expr::Call(f, args) if args.len() == 1 => {
                let arg = numeric_eval_expr(&args[0], env, interner)?;
                match interner.resolve(*f) {
                    "sin" => Some(arg.sin()),
                    "cos" => Some(arg.cos()),
                    "tan" => Some(arg.tan()),
                    "exp" => Some(arg.exp()),
                    "log" => Some(arg.ln()),
                    "sqrt" => Some(arg.sqrt()),
                    "abs" => Some(arg.abs()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    let base = [0.5, 1.0, -1.5, 2.3, -0.7];
    let mut successes = 0usize;
    for sample in 0..5 {
        let mut sample_env = env.clone();
        for (idx, sym) in syms.iter().enumerate() {
            sample_env
                .bindings
                .insert(*sym, Expr::Float(base[(sample + idx) % base.len()]));
        }
        let (Some(lhs), Some(rhs)) = (
            numeric_eval_expr(a, &sample_env, interner),
            numeric_eval_expr(b, &sample_env, interner),
        ) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        successes += 1;
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        if (lhs - rhs).abs() / scale > 1e-10 {
            return Some(false);
        }
    }
    if successes == 5 {
        Some(true)
    } else {
        None
    }
}

fn equiv_description(a: &Expr, b: &Expr, env: &Env, interner: &ax_ir::Interner) -> String {
    let ea = eval(a, env, interner);
    let eb = eval(b, env, interner);
    if ea == eb {
        return "equal".into();
    }

    let sa = canonical_equiv_form(&ea, interner);
    let sb = canonical_equiv_form(&eb, interner);
    if sa == sb {
        return "equal".into();
    }

    let diff = canonical_equiv_form(
        &Expr::add(vec![sa.clone(), Expr::neg(sb.clone())]),
        interner,
    );
    if diff == Expr::zero() {
        return "equal".into();
    }

    let ta = simplify::trig_simplify(&sa, interner);
    let tb = simplify::trig_simplify(&sb, interner);
    if ta == tb {
        return "equal".into();
    }

    if contains_indexed_expr(&ta) || contains_indexed_expr(&tb) {
        let ca = ax_tensor::rename_dummies(
            &ax_tensor::canonicalize_indices(&ta, &env.property_store, interner),
            env,
            interner,
        );
        let cb = ax_tensor::rename_dummies(
            &ax_tensor::canonicalize_indices(&tb, &env.property_store, interner),
            env,
            interner,
        );
        if ca == cb {
            return "dummy_index_renamed".into();
        }
    }

    match equiv_sample_check(&ta, &tb, env, interner) {
        Some(true) => return "equal_under_assumptions".into(),
        Some(false) => return "not_equal".into(),
        None => {}
    }

    if ta == canonical_equiv_form(&Expr::neg(tb), interner) {
        return "convention_difference".into();
    }

    "unknown".into()
}

fn eval_condition(cond: &Condition, env: &Env, interner: &ax_ir::Interner) -> Option<bool> {
    match cond {
        Condition::True => Some(true),
        Condition::False => Some(false),
        Condition::Gt(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord == Ordering::Greater)
        }
        Condition::Lt(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord == Ordering::Less)
        }
        Condition::Ge(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord != Ordering::Less)
        }
        Condition::Le(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            compare_numeric(&ea, &eb).map(|ord| ord != Ordering::Greater)
        }
        Condition::Eq(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            Some(ea == eb)
        }
        Condition::Ne(a, b) => {
            let ea = eval(a, env, interner);
            let eb = eval(b, env, interner);
            Some(ea != eb)
        }
        Condition::And(a, b) => {
            Some(eval_condition(a, env, interner)? && eval_condition(b, env, interner)?)
        }
        Condition::Or(a, b) => {
            Some(eval_condition(a, env, interner)? || eval_condition(b, env, interner)?)
        }
        Condition::Not(c) => eval_condition(c, env, interner).map(|v| !v),
    }
}

fn expr_to_pattern(expr: &Expr, interner: &ax_ir::Interner) -> ax_rewrite::Pattern {
    match expr {
        Expr::Sym(s) => {
            let name = interner.resolve(*s);
            if name.ends_with('_') {
                ax_rewrite::Pattern::Slot(*s)
            } else {
                ax_rewrite::Pattern::Exact(expr.clone())
            }
        }
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => {
            ax_rewrite::Pattern::Exact(expr.clone())
        }
        Expr::Add(terms) => {
            ax_rewrite::Pattern::Add(terms.iter().map(|t| expr_to_pattern(t, interner)).collect())
        }
        Expr::Mul(factors) => ax_rewrite::Pattern::Mul(
            factors
                .iter()
                .map(|f| expr_to_pattern(f, interner))
                .collect(),
        ),
        Expr::Pow(base, exp) => ax_rewrite::Pattern::Pow(
            Box::new(expr_to_pattern(base, interner)),
            Box::new(expr_to_pattern(exp, interner)),
        ),
        Expr::Neg(e) => ax_rewrite::Pattern::Neg(Box::new(expr_to_pattern(e, interner))),
        Expr::Call(f, args) => ax_rewrite::Pattern::Call(
            *f,
            args.iter().map(|a| expr_to_pattern(a, interner)).collect(),
        ),
        _ => ax_rewrite::Pattern::Exact(expr.clone()),
    }
}

pub fn register_rule(
    rule_expr: &Expr,
    env: &mut Env,
    interner: &ax_ir::Interner,
) -> Option<String> {
    if let Expr::Rule(lhs, rhs, trust_level) = rule_expr {
        let rule = ax_rewrite::RewriteRule {
            name: format!("user_rule_{}", env.rules.len()),
            pattern: expr_to_pattern(lhs, interner),
            replacement: *rhs.clone(),
            condition: None,
            trust_level: *trust_level,
        };
        let name = rule.name.clone();
        env.rules.push(rule);
        Some(name)
    } else {
        None
    }
}

pub fn rewrite_with_trace(
    expr: &Expr,
    env: &Env,
    interner: &ax_ir::Interner,
) -> (Expr, ax_rewrite::RewriteTrace) {
    let mut trace = ax_rewrite::RewriteTrace::default();
    let result = if env.property_store.symbols().is_empty() {
        ax_rewrite::rewrite_fixed_point_traced(&env.rules, expr, interner, 100, &mut trace)
    } else {
        ax_rewrite::rewrite_fixed_point_with_compare(
            &env.rules,
            expr,
            &env.property_store,
            &env.index_to_family,
            interner,
            100,
        )
    };
    (result, trace)
}

fn trust_level_name(level: ax_ir::TrustLevel) -> &'static str {
    match level {
        ax_ir::TrustLevel::Exact => "exact",
        ax_ir::TrustLevel::UnderAssumptions => "under_assumptions",
        ax_ir::TrustLevel::Heuristic => "heuristic",
        ax_ir::TrustLevel::NumericallyChecked => "numerically_checked",
        ax_ir::TrustLevel::Unverified => "unverified",
    }
}

pub fn describe_rewrite_trace(trace: &ax_rewrite::RewriteTrace) -> String {
    if trace.steps.is_empty() {
        "trust: exact".to_string()
    } else {
        let overall = trust_level_name(trace.overall_trust());
        let used = trace
            .steps
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "trust: {overall} (used rule{}: {used})",
            if trace.steps.len() == 1 { "" } else { "s" }
        )
    }
}

pub fn resolve_import(
    path: &[lasso::Spur],
    interner: &ax_ir::Interner,
    search_paths: &[PathBuf],
) -> Option<PathBuf> {
    if path.is_empty() {
        return None;
    }

    let parts = path
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect::<Vec<_>>();

    let mut rel = PathBuf::new();
    for part in &parts {
        rel.push(part);
    }
    rel.set_extension("ax");

    let rel_without_std = if parts.first().is_some_and(|part| part == "std") && parts.len() > 1 {
        let mut p = PathBuf::new();
        for part in &parts[1..] {
            p.push(part);
        }
        p.set_extension("ax");
        Some(p)
    } else {
        None
    };

    for root in search_paths {
        let candidate = root.join(&rel);
        if candidate.is_file() {
            return Some(candidate);
        }

        if let Some(rel_without_std) = &rel_without_std {
            let candidate = root.join(rel_without_std);
            if candidate.is_file() {
                return Some(candidate);
            }
        }

        let config_path = root.join("axioma.toml");
        if config_path.is_file() {
            if let Ok(text) = std::fs::read_to_string(&config_path) {
                if let Ok(cfg) = toml::from_str::<ax_context::AxiomaConfig>(&text) {
                    if let Some(dep_name) = parts.first() {
                        if let Some(dep) = cfg.dependencies.get(dep_name) {
                            if let Some(dep_path) = &dep.path {
                                let dep_root = if PathBuf::from(dep_path).is_absolute() {
                                    PathBuf::from(dep_path)
                                } else {
                                    root.join(dep_path)
                                };
                                let dep_rel = if parts.len() > 1 {
                                    let mut p = PathBuf::new();
                                    for part in &parts[1..] {
                                        p.push(part);
                                    }
                                    p.set_extension("ax");
                                    p
                                } else {
                                    let mut p = PathBuf::new();
                                    p.push(dep_name);
                                    p.set_extension("ax");
                                    p
                                };
                                let candidate = dep_root.join(&dep_rel);
                                if candidate.is_file() {
                                    return Some(candidate);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn differentiate(expr: &Expr, var: lasso::Spur, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) => Expr::Int(0.into()),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(differentiate(re, var, interner)),
            Box::new(differentiate(im, var, interner)),
        ),
        Expr::Sym(s) => {
            if *s == var {
                Expr::Int(1.into())
            } else {
                Expr::Int(0.into())
            }
        }
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|term| differentiate(term, var, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(differentiate(e, var, interner)),
        Expr::Group(inner, rel) => Expr::Group(Box::new(differentiate(inner, var, interner)), *rel),
        Expr::Mul(factors) => {
            let terms = factors
                .iter()
                .enumerate()
                .map(|(i, factor)| {
                    let mut product = Vec::with_capacity(factors.len());
                    product.extend(factors[..i].iter().cloned());
                    product.push(differentiate(factor, var, interner));
                    product.extend(factors[i + 1..].iter().cloned());
                    Expr::mul(product)
                })
                .collect();
            collapse_duplicate_sum_terms(terms)
        }
        Expr::Pow(base, exp) => {
            if !contains_var(exp, var) {
                Expr::mul(vec![
                    exp.as_ref().clone(),
                    Expr::pow(
                        base.as_ref().clone(),
                        Expr::add(vec![exp.as_ref().clone(), Expr::neg(Expr::one())]),
                    ),
                    differentiate(base, var, interner),
                ])
            } else if !contains_var(base, var) {
                match base.as_ref() {
                    Expr::Sym(sym) if interner.resolve(*sym) == "e" => {
                        Expr::mul(vec![expr.clone(), differentiate(exp, var, interner)])
                    }
                    Expr::Call(f, args) if interner.resolve(*f) == "exp" && args.len() == 1 => {
                        Expr::mul(vec![expr.clone(), differentiate(exp, var, interner)])
                    }
                    _ => diff_call(expr, var, interner),
                }
            } else {
                diff_call(expr, var, interner)
            }
        }
        Expr::Call(f, args) => {
            if args.iter().all(|arg| !contains_var(arg, var)) {
                return Expr::Int(0.into());
            }

            let name = interner.resolve(*f);
            if args.len() != 1 {
                return diff_call(expr, var, interner);
            }

            let arg = args[0].clone();
            let darg = differentiate(&args[0], var, interner);
            match name {
                "sin" => Expr::mul(vec![builtin_unary("cos", arg, interner), darg]),
                "cos" => Expr::mul(vec![Expr::neg(builtin_unary("sin", arg, interner)), darg]),
                "exp" => Expr::mul(vec![builtin_unary("exp", arg, interner), darg]),
                "log" => Expr::mul(vec![Expr::pow(arg, Expr::neg(Expr::one())), darg]),
                "sqrt" => differentiate(&Expr::pow(arg, one_half()), var, interner),
                "tan" => Expr::mul(vec![
                    Expr::pow(builtin_unary("sec", arg, interner), Expr::Int(2.into())),
                    darg,
                ]),
                "sec" => Expr::mul(vec![
                    builtin_unary("sec", arg.clone(), interner),
                    builtin_unary("tan", arg, interner),
                    darg,
                ]),
                "csc" => Expr::neg(Expr::mul(vec![
                    builtin_unary("csc", arg.clone(), interner),
                    builtin_unary("cot", arg, interner),
                    darg,
                ])),
                "cot" => Expr::neg(Expr::mul(vec![
                    Expr::pow(builtin_unary("csc", arg, interner), Expr::Int(2.into())),
                    darg,
                ])),
                "asin" | "arcsin" => Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![
                            Expr::one(),
                            Expr::neg(Expr::pow(arg, Expr::Int(2.into()))),
                        ]),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
                    darg,
                ]),
                "acos" | "arccos" => Expr::neg(Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![
                            Expr::one(),
                            Expr::neg(Expr::pow(arg, Expr::Int(2.into()))),
                        ]),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
                    darg,
                ])),
                "atan" | "arctan" => Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![Expr::one(), Expr::pow(arg, Expr::Int(2.into()))]),
                        Expr::Int((-1).into()),
                    ),
                    darg,
                ]),
                "sinh" => Expr::mul(vec![builtin_unary("cosh", arg, interner), darg]),
                "cosh" => Expr::mul(vec![builtin_unary("sinh", arg, interner), darg]),
                "tanh" => Expr::mul(vec![
                    Expr::add(vec![
                        Expr::one(),
                        Expr::neg(Expr::pow(
                            builtin_unary("tanh", arg, interner),
                            Expr::Int(2.into()),
                        )),
                    ]),
                    darg,
                ]),
                "asinh" | "arcsinh" => Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![Expr::pow(arg, Expr::Int(2.into())), Expr::one()]),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
                    darg,
                ]),
                "acosh" | "arccosh" => Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![
                            Expr::pow(arg, Expr::Int(2.into())),
                            Expr::neg(Expr::one()),
                        ]),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
                    darg,
                ]),
                "atanh" | "arctanh" => Expr::mul(vec![
                    Expr::pow(
                        Expr::add(vec![
                            Expr::one(),
                            Expr::neg(Expr::pow(arg, Expr::Int(2.into()))),
                        ]),
                        Expr::Int((-1).into()),
                    ),
                    darg,
                ]),
                "abs" => Expr::mul(vec![builtin_unary("sign", arg, interner), darg]),
                _ => diff_call(expr, var, interner),
            }
        }
        Expr::FnDef(name, params, body) => Expr::FnDef(
            *name,
            params.clone(),
            Box::new(differentiate(body, var, interner)),
        ),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(
            Box::new(differentiate(lhs, var, interner)),
            Box::new(differentiate(rhs, var, interner)),
            *trust,
        ),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => Expr::Piecewise(
            cases
                .iter()
                .map(|(value, condition)| (differentiate(value, var, interner), condition.clone()))
                .collect(),
        ),
        Expr::Let(name, val, body) => Expr::Let(
            *name,
            val.clone(),
            Box::new(differentiate(body, var, interner)),
        ),
        Expr::Indexed(_, _) => diff_call(expr, var, interner),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| differentiate(item, var, interner))
                .collect(),
        ),
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| differentiate(cell, var, interner))
                        .collect()
                })
                .collect(),
        ),
    }
}

fn builtin_call(
    name: &str,
    f: lasso::Spur,
    args: Vec<Expr>,
    interner: &ax_ir::Interner,
    env: &Env,
) -> Expr {
    match name {
        "eq" | "==" => {
            if args.len() == 2 {
                equation::make_equation(args[0].clone(), args[1].clone(), interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "get_lhs" => {
            if args.len() == 1 {
                equation::get_lhs(&args[0], interner).unwrap_or_else(Expr::zero)
            } else {
                Expr::Call(f, args)
            }
        }
        "get_rhs" => {
            if args.len() == 1 {
                equation::get_rhs(&args[0], interner).unwrap_or_else(Expr::zero)
            } else {
                Expr::Call(f, args)
            }
        }
        "swap_sides" => {
            if args.len() == 1 {
                equation::swap_sides(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "multiply_through" => {
            if args.len() == 2 {
                equation::multiply_through(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "add_through" => {
            if args.len() == 2 {
                equation::add_through(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "to_rhs" => {
            if args.len() == 2 {
                equation::to_rhs(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "to_lhs" => {
            if args.len() == 2 {
                equation::to_lhs(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "isolate" => {
            if args.len() == 2 {
                equation::isolate(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "eq_to_rule" | "eq_to_subrule" => {
            if args.len() == 1 {
                equation::equation_to_rule(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "differentiate_eq" => {
            if args.len() == 2 {
                let var = extract_sym(&args[1], interner);
                equation::differentiate_equation(&args[0], var, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "integrate_eq" => {
            if args.len() == 2 {
                let var = extract_sym(&args[1], interner);
                equation::integrate_equation(&args[0], var, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "raise_eq" => {
            if args.len() == 2 {
                match (find_metric_sym(env), find_inv_metric_sym(env)) {
                    (Some(metric), Some(inv_metric)) => equation::raise_equation(
                        &args[0],
                        metric,
                        inv_metric,
                        extract_sym(&args[1], interner),
                        interner,
                    ),
                    _ => args[0].clone(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lower_eq" => {
            if args.len() == 2 {
                match (find_metric_sym(env), find_inv_metric_sym(env)) {
                    (Some(metric), Some(inv_metric)) => equation::lower_equation(
                        &args[0],
                        metric,
                        inv_metric,
                        extract_sym(&args[1], interner),
                        interner,
                    ),
                    _ => args[0].clone(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "substitute_eq" => {
            if args.len() == 3 {
                equation::substitute_equation(&args[0], &args[1], &args[2], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "angle" => {
            if args.len() == 2 {
                match (
                    label_from_expr(&args[0], interner),
                    label_from_expr(&args[1], interner),
                ) {
                    (Some(i), Some(j)) => {
                        spinor_to_expr(&ax_spinor::SpinorExpr::angle(i, j), interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "square" => {
            if args.len() == 2 {
                match (
                    label_from_expr(&args[0], interner),
                    label_from_expr(&args[1], interner),
                ) {
                    (Some(i), Some(j)) => {
                        spinor_to_expr(&ax_spinor::SpinorExpr::square(i, j), interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "angle_chain" | "square_chain" | "angle_square_chain" | "square_angle_chain" => {
            if args.len() == 3 {
                let start = label_from_expr(&args[0], interner);
                let middle = labels_from_list_expr(&args[1], interner);
                let end = label_from_expr(&args[2], interner);
                match (start, middle, end) {
                    (Some(i), Some(middle), Some(j)) => {
                        let expr = match name {
                            "angle_chain" => ax_spinor::SpinorExpr::AngleChain(i, middle, j),
                            "square_chain" => ax_spinor::SpinorExpr::SquareChain(i, middle, j),
                            "angle_square_chain" => {
                                ax_spinor::SpinorExpr::AngleSquareChain(i, middle, j)
                            }
                            "square_angle_chain" => {
                                ax_spinor::SpinorExpr::SquareAngleChain(i, middle, j)
                            }
                            _ => unreachable!(),
                        };
                        spinor_to_expr(&expr, interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "mandelstam" => {
            if args.len() == 2 {
                match (
                    label_from_expr(&args[0], interner),
                    label_from_expr(&args[1], interner),
                ) {
                    (Some(i), Some(j)) => spinor_to_expr(&ax_spinor::SpinorExpr::s(i, j), interner),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "mandelstam_multi" => {
            if args.len() == 1 {
                if let Some(labels) = labels_from_list_expr(&args[0], interner) {
                    spinor_multi_mandelstam_expr(labels, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "parke_taylor" => {
            if args.len() == 3 {
                match (
                    int_from_expr(&args[0]),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                ) {
                    (Some(n), Some(i), Some(j)) => {
                        spinor_to_expr(&ax_spinor::parke_taylor(n, i, j), interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "three_point_mhv" | "three_point_anti_mhv" => {
            if args.len() == 3 {
                match (
                    label_from_expr(&args[0], interner),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                ) {
                    (Some(i), Some(j), Some(k)) => {
                        let out = if name == "three_point_mhv" {
                            ax_spinor::three_point_mhv([i, j, k])
                        } else {
                            ax_spinor::three_point_anti_mhv([i, j, k])
                        };
                        spinor_to_expr(&out, interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_chain" | "contract_adjacent" | "expand_mandelstam" | "collect_mandelstam" => {
            if args.len() == 1 {
                if let Some(s) = expr_to_spinor(&args[0], interner) {
                    let out = match name {
                        "expand_chain" => ax_spinor::expand_chain(&s),
                        "contract_adjacent" => ax_spinor::contract_adjacent(&s),
                        "expand_mandelstam" => ax_spinor::expand_mandelstam(&s),
                        "collect_mandelstam" => ax_spinor::collect_mandelstam(&s),
                        _ => unreachable!(),
                    };
                    spinor_to_expr(&out, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "schouten" => {
            if args.len() == 5 {
                if let (Some(s), Some(a), Some(b), Some(c), Some(d)) = (
                    expr_to_spinor(&args[0], interner),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                    label_from_expr(&args[3], interner),
                    label_from_expr(&args[4], interner),
                ) {
                    spinor_to_expr(&ax_spinor::apply_schouten(&s, a, b, c, d), interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "momentum_conservation" | "spinor_simplify" => {
            if args.len() == 3 || (name == "spinor_simplify" && args.len() == 2) {
                if let (Some(s), Some(n)) =
                    (expr_to_spinor(&args[0], interner), int_from_expr(&args[1]))
                {
                    let out = if name == "spinor_simplify" {
                        ax_spinor::spinor_simplify(&s, n)
                    } else if let Some(elim) = label_from_expr(&args[2], interner) {
                        ax_spinor::apply_momentum_conservation(&s, n, elim)
                    } else {
                        return Expr::Call(f, args);
                    };
                    spinor_to_expr(&out, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bcfw_shift" => {
            if args.len() == 4 {
                if let (Some(s), Some(i), Some(j), Expr::Sym(z)) = (
                    expr_to_spinor(&args[0], interner),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                    &args[3],
                ) {
                    let shift = ax_spinor::BCFWShift {
                        shifted_angle: i,
                        shifted_square: j,
                    };
                    spinor_to_expr(
                        &ax_spinor::bcfw_shift_momentum(&s, &shift, *z, interner),
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bcfw_decomposition" => {
            if args.len() == 4 {
                if let (Some(n), Some(i), Some(j), Expr::List(helicity_exprs)) = (
                    int_from_expr(&args[0]),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                    &args[3],
                ) {
                    let helicities = helicity_exprs
                        .iter()
                        .map(|e| match e {
                            Expr::Int(v) => v.to_i8(),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(helicities) = helicities {
                        let shift = ax_spinor::BCFWShift {
                            shifted_angle: i,
                            shifted_square: j,
                        };
                        let terms = ax_spinor::bcfw_decomposition(n, &shift, &helicities)
                            .into_iter()
                            .map(|t| {
                                Expr::List(vec![
                                    Expr::List(
                                        t.left_particles
                                            .into_iter()
                                            .map(|l| int_expr_u16(l.0))
                                            .collect(),
                                    ),
                                    Expr::List(
                                        t.right_particles
                                            .into_iter()
                                            .map(|l| int_expr_u16(l.0))
                                            .collect(),
                                    ),
                                    Expr::Int(BigInt::from(t.internal_helicity)),
                                    Expr::List(
                                        t.propagator_momentum
                                            .into_iter()
                                            .map(|l| int_expr_u16(l.0))
                                            .collect(),
                                    ),
                                ])
                            })
                            .collect();
                        Expr::List(terms)
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "four_bracket" => {
            if args.len() == 4 {
                if let (Some(i), Some(j), Some(k), Some(l)) = (
                    label_from_expr(&args[0], interner),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                    label_from_expr(&args[3], interner),
                ) {
                    twistor_to_expr(
                        &ax_spinor::twistor::TwistorExpr::four_bracket(i, j, k, l),
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "plucker" => {
            if args.len() == 7 {
                if let (Some(t), Some(a), Some(b), Some(c), Some(d), Some(e), Some(g)) = (
                    expr_to_twistor(&args[0], interner),
                    label_from_expr(&args[1], interner),
                    label_from_expr(&args[2], interner),
                    label_from_expr(&args[3], interner),
                    label_from_expr(&args[4], interner),
                    label_from_expr(&args[5], interner),
                    label_from_expr(&args[6], interner),
                ) {
                    twistor_to_expr(
                        &ax_spinor::twistor::apply_plucker(&t, a, b, c, d, e, g),
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "perturb" => {
            if args.len() == 6 {
                if let (
                    Some(field),
                    Some(background),
                    Some(perturbation),
                    Some(epsilon),
                    Some(order),
                ) = (
                    symbol_from_expr(&args[1]),
                    symbol_from_expr(&args[2]),
                    symbol_from_expr(&args[3]),
                    symbol_from_expr(&args[4]),
                    usize_from_expr(&args[5]),
                ) {
                    let setup =
                        perturbation_setup(field, background, None, perturbation, epsilon, order);
                    expanded_to_expr_list(
                        ax_perturb::perturb_expand(&args[0], &setup, interner),
                        order,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "perturb_inverse" => {
            if args.len() == 6 {
                if let (
                    Some(field),
                    Some(background),
                    Some(background_inv),
                    Some(perturbation),
                    Some(epsilon),
                    Some(order),
                ) = (
                    symbol_from_expr(&args[0]),
                    symbol_from_expr(&args[1]),
                    symbol_from_expr(&args[2]),
                    symbol_from_expr(&args[3]),
                    symbol_from_expr(&args[4]),
                    usize_from_expr(&args[5]),
                ) {
                    let setup = perturbation_setup(
                        field,
                        background,
                        Some(background_inv),
                        perturbation,
                        epsilon,
                        order,
                    );
                    expanded_to_expr_list(
                        ax_perturb::perturb_inverse_metric(&setup, interner),
                        order,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "perturb_christoffel" | "perturb_riemann" | "perturb_ricci" | "perturb_einstein" => {
            if args.len() == 7 {
                if let (
                    Some(field),
                    Some(background),
                    Some(background_inv),
                    Some(perturbation),
                    Some(epsilon),
                    Some(coords),
                    Some(order),
                ) = (
                    symbol_from_expr(&args[0]),
                    symbol_from_expr(&args[1]),
                    symbol_from_expr(&args[2]),
                    symbol_from_expr(&args[3]),
                    symbol_from_expr(&args[4]),
                    symbol_list_from_expr(&args[5]),
                    usize_from_expr(&args[6]),
                ) {
                    let setup = perturbation_setup(
                        field,
                        background,
                        Some(background_inv),
                        perturbation,
                        epsilon,
                        order,
                    );
                    let expanded = match name {
                        "perturb_christoffel" => {
                            ax_perturb::perturb_christoffel(&setup, &coords, interner)
                        }
                        "perturb_riemann" => ax_perturb::perturb_riemann(&setup, &coords, interner),
                        "perturb_ricci" => ax_perturb::perturb_ricci(&setup, &coords, interner),
                        "perturb_einstein" => {
                            ax_perturb::perturb_einstein(&setup, &coords, interner)
                        }
                        _ => unreachable!(),
                    };
                    expanded_to_expr_list(expanded, order)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "linearized_einstein" => {
            if args.len() == 1 {
                if let Some(order) = usize_from_expr(&args[0]) {
                    let bg = ax_perturb::cosmology::frw_background(interner);
                    let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner)
                    else {
                        return Expr::Call(f, args);
                    };
                    let equations = match order {
                        1 => ax_perturb::cosmology::linearized_einstein_scalar(
                            &bg, &decomp, interner,
                        ),
                        2 => ax_perturb::cosmology::linearized_einstein_second_order(
                            &bg, &decomp, interner,
                        ),
                        _ => return Expr::Call(f, args),
                    };
                    match equations {
                        Ok(equations) => labelled_exprs_to_list(equations, interner),
                        Err(_) => Expr::Call(f, args),
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "mukhanov_sasaki" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let eps = interner.get_or_intern("epsilon");
                match ax_perturb::cosmology::mukhanov_sasaki_equation(&bg, eps, interner) {
                    Ok(expr) => expr,
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "linearized_einstein_vector" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::linearized_einstein_vector(&bg, &decomp, interner) {
                    Ok(equations) => labelled_exprs_to_list(equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "linearized_einstein_tensor" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::linearized_einstein_tensor(&bg, &decomp, interner) {
                    Ok(equations) => labelled_exprs_to_list(equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "second_order_einstein_vector" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::second_order_einstein_vector(&bg, &decomp, interner) {
                    Ok(equations) => labelled_exprs_to_list(equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "second_order_einstein_tensor" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::second_order_einstein_tensor(&bg, &decomp, interner) {
                    Ok(equations) => labelled_exprs_to_list(equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_mode_equation" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match ax_perturb::cosmology::tensor_mode_equation(&bg, interner) {
                    Ok(expressions) => named_exprs_to_list(expressions),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_mode_first_order" => {
            if args.len() == 1 {
                if let Some(polarization) = name_from_expr(&args[0], interner) {
                    let bg = ax_perturb::cosmology::frw_background(interner);
                    match ax_perturb::tensor_mode_first_order_system(&bg, polarization, interner) {
                        Ok(system) => Expr::List(
                            system
                                .into_iter()
                                .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                                .collect(),
                        ),
                        Err(_) => Expr::Call(f, args),
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "multifield_equations" => {
            if args.len() == 1 {
                if let Some(nfields) = usize_from_expr(&args[0]) {
                    let bg = ax_perturb::cosmology::frw_background(interner);
                    match ax_perturb::standard_multifield_symbols(nfields, interner).and_then(
                        |symbols| {
                            ax_perturb::derive_multifield_curvature_entropy_equations(
                                &bg, &symbols, interner,
                            )
                        },
                    ) {
                        Ok(system) => labelled_exprs_to_list(system.equations, interner),
                        Err(_) => Expr::Call(f, args),
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "boltzmann_bridge" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match ax_perturb::symbolic_boltzmann_bridge_system(&bg, interner) {
                    Ok(system) => Expr::List(
                        system
                            .equations
                            .into_iter()
                            .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                            .collect(),
                    ),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "boltzmann_bridge_export" => {
            if args.len() == 1 {
                if let Some(target) = name_from_expr(&args[0], interner) {
                    let bg = ax_perturb::cosmology::frw_background(interner);
                    match ax_perturb::symbolic_boltzmann_bridge_system(&bg, interner).and_then(
                        |system| {
                            ax_perturb::export_boltzmann_bridge_system(target, &system, interner)
                        },
                    ) {
                        Ok(code) => Expr::Sym(interner.get_or_intern(&code)),
                        Err(_) => Expr::Call(f, args),
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cubic_action" => {
            if args.len() == 1 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Some(channel) = parse_cubic_channel_name(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let result = match channel {
                    ax_perturb::CubicInteractionChannel::ScalarScalarScalar => {
                        ax_perturb::reduced_cubic_scalar_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::TensorTensorTensor => {
                        ax_perturb::reduced_cubic_tensor_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::ScalarScalarTensor
                    | ax_perturb::CubicInteractionChannel::ScalarTensorTensor => {
                        ax_perturb::reduced_cubic_mixed_action(channel, &bg, interner)
                    }
                };
                match result {
                    Ok(action) => action.lagrangian_density,
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cubic_kernel" => {
            if args.len() == 1 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Some(channel) = parse_cubic_channel_name(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let action = match channel {
                    ax_perturb::CubicInteractionChannel::ScalarScalarScalar => {
                        ax_perturb::reduced_cubic_scalar_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::TensorTensorTensor => {
                        ax_perturb::reduced_cubic_tensor_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::ScalarScalarTensor
                    | ax_perturb::CubicInteractionChannel::ScalarTensorTensor => {
                        ax_perturb::reduced_cubic_mixed_action(channel, &bg, interner)
                    }
                };
                match action.and_then(|built| ax_perturb::cubic_fourier_kernel(&built, interner)) {
                    Ok(kernel) => kernel.kernel,
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bispectrum_shape" => {
            if args.len() == 2 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Some(channel) = parse_cubic_channel_name(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(shape) = name_from_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let action = match channel {
                    ax_perturb::CubicInteractionChannel::ScalarScalarScalar => {
                        ax_perturb::reduced_cubic_scalar_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::TensorTensorTensor => {
                        ax_perturb::reduced_cubic_tensor_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::ScalarScalarTensor
                    | ax_perturb::CubicInteractionChannel::ScalarTensorTensor => {
                        ax_perturb::reduced_cubic_mixed_action(channel, &bg, interner)
                    }
                };
                match action
                    .and_then(|built| ax_perturb::cubic_fourier_kernel(&built, interner))
                    .and_then(|kernel| ax_perturb::bispectrum_shape(&kernel, shape, interner))
                {
                    Ok(shape_value) => shape_value.evaluated_form,
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "export_cubic_vertex" => {
            if args.len() == 2 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Some(channel) = parse_cubic_channel_name(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(target) = name_from_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let action = match channel {
                    ax_perturb::CubicInteractionChannel::ScalarScalarScalar => {
                        ax_perturb::reduced_cubic_scalar_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::TensorTensorTensor => {
                        ax_perturb::reduced_cubic_tensor_action(&bg, interner)
                    }
                    ax_perturb::CubicInteractionChannel::ScalarScalarTensor
                    | ax_perturb::CubicInteractionChannel::ScalarTensorTensor => {
                        ax_perturb::reduced_cubic_mixed_action(channel, &bg, interner)
                    }
                };
                match action
                    .and_then(|built| ax_perturb::cubic_fourier_kernel(&built, interner))
                    .and_then(|kernel| ax_perturb::export_cubic_vertex(target, &kernel, interner))
                {
                    Ok(export) => Expr::Sym(interner.get_or_intern(&export.code)),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eft_model" => {
            if args.len() == 1 {
                match parse_eft_model_name(&args[0], interner) {
                    Some(model) => make_eft_model_expr(model, interner),
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eft_quadratic_sector" => {
            if args.len() == 1 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match parse_eft_model_name(&args[0], interner)
                    .map(|model| ax_perturb::standard_eft_coefficients(model, interner))
                    .ok_or(())
                    .and_then(|coeffs| {
                        ax_perturb::eft_quadratic_sector_named(&bg, &coeffs, interner)
                            .map_err(|_| ())
                    }) {
                    Ok(items) => named_exprs_to_list(items),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eft_stability" => {
            if args.len() == 1 {
                match parse_eft_model_name(&args[0], interner)
                    .map(|model| ax_perturb::standard_eft_coefficients(model, interner))
                    .ok_or(())
                    .and_then(|coeffs| {
                        ax_perturb::eft_stability_named(&coeffs, interner).map_err(|_| ())
                    }) {
                    Ok(items) => named_exprs_to_list(items),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eft_mode_equations" => {
            if args.len() == 1 {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match parse_eft_model_name(&args[0], interner)
                    .map(|model| ax_perturb::standard_eft_coefficients(model, interner))
                    .ok_or(())
                    .and_then(|coeffs| {
                        ax_perturb::eft_mode_equations_named(&bg, &coeffs, interner).map_err(|_| ())
                    }) {
                    Ok(items) => labelled_exprs_to_list(items, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eft_export_rhs" => {
            if args.len() == 2 {
                let Some(model) = parse_eft_model_name(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(target) = name_from_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let coeffs = ax_perturb::standard_eft_coefficients(model, interner);
                match ax_perturb::export_eft_mode_rhs(target, &coeffs, interner) {
                    Ok(code) => Expr::Sym(interner.get_or_intern(&code)),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "project_scalar_harmonics" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::linearized_einstein_scalar(&bg, &decomp, interner)
                    .and_then(|equations| {
                        ax_perturb::project_scalar_equations_to_harmonic_space(
                            &equations, &bg, interner,
                        )
                    }) {
                    Ok(projected) => labelled_exprs_to_list(projected.equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "project_vector_harmonics" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::linearized_einstein_vector(&bg, &decomp, interner)
                    .and_then(|equations| {
                        ax_perturb::project_vector_equations_to_harmonic_space(
                            &equations, &bg, interner,
                        )
                    }) {
                    Ok(projected) => labelled_exprs_to_list(projected.equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "project_tensor_harmonics" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::cosmology::linearized_einstein_tensor(&bg, &decomp, interner)
                    .and_then(|equations| {
                        ax_perturb::project_tensor_equations_to_harmonic_space(
                            &equations, &bg, interner,
                        )
                    }) {
                    Ok(projected) => labelled_exprs_to_list(projected.equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "project_second_order_vector_harmonics" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match ax_perturb::derive_second_order_vector_system(&bg, interner).and_then(
                    |system| {
                        ax_perturb::project_second_order_vector_to_harmonics(&system, &bg, interner)
                    },
                ) {
                    Ok(projected) => labelled_exprs_to_list(projected.equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "project_second_order_tensor_harmonics" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                match ax_perturb::derive_second_order_tensor_system(&bg, interner).and_then(
                    |system| {
                        ax_perturb::project_second_order_tensor_to_harmonics(&system, &bg, interner)
                    },
                ) {
                    Ok(projected) => labelled_exprs_to_list(projected.equations, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "scalar_harmonic_spec" => {
            if args.len() == 1 {
                if let Some(curvature) = parse_curvature_name(&args[0], interner) {
                    make_harmonic_spec_expr(curvature, ax_perturb::SectorKind::Scalar, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "vector_harmonic_spec" => {
            if args.len() == 1 {
                if let Some(curvature) = parse_curvature_name(&args[0], interner) {
                    make_harmonic_spec_expr(curvature, ax_perturb::SectorKind::Vector, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "neutrino_hierarchy" => {
            if args.len() == 3 {
                let Some(lmax) = usize_from_expr(&args[0]) else {
                    return Expr::Call(f, args);
                };
                let Some(gauge) = parse_hierarchy_gauge_name(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(closure) = parse_hierarchy_closure_name(&args[2], interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::hierarchy_spec(lmax, gauge, closure)
                    .and_then(|spec| ax_perturb::neutrino_hierarchy_system(&spec, interner))
                {
                    Ok(system) => Expr::List(
                        system
                            .equations
                            .into_iter()
                            .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                            .collect(),
                    ),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "photon_hierarchy" => {
            if args.len() == 3 {
                let Some(lmax) = usize_from_expr(&args[0]) else {
                    return Expr::Call(f, args);
                };
                let Some(gauge) = parse_hierarchy_gauge_name(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(closure) = parse_hierarchy_closure_name(&args[2], interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::hierarchy_spec(lmax, gauge, closure)
                    .and_then(|spec| ax_perturb::photon_hierarchy_system(&spec, interner))
                {
                    Ok(system) => Expr::List(
                        system
                            .equations
                            .into_iter()
                            .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                            .collect(),
                    ),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "export_hierarchy" => {
            if args.len() == 5 {
                let Some(target) = name_from_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(species) = name_from_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(lmax) = usize_from_expr(&args[2]) else {
                    return Expr::Call(f, args);
                };
                let Some(gauge) = parse_hierarchy_gauge_name(&args[3], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(closure) = parse_hierarchy_closure_name(&args[4], interner) else {
                    return Expr::Call(f, args);
                };
                let system =
                    ax_perturb::hierarchy_spec(lmax, gauge, closure).and_then(
                        |spec| match species {
                            "neutrino" => ax_perturb::neutrino_hierarchy_system(&spec, interner),
                            "photon" => ax_perturb::photon_hierarchy_system(&spec, interner),
                            _ => Err(ax_perturb::CosmologyError::UnsupportedExternalSolverHook {
                                target: species.to_string(),
                            }),
                        },
                    );
                match system.and_then(|system| {
                    ax_perturb::export_hierarchy_system(target, &system, interner)
                }) {
                    Ok(payload) => Expr::Sym(interner.get_or_intern(&payload)),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_parity_report" => {
            if args.is_empty() {
                match ax_perturb::built_in_parity_reports(interner).and_then(|reports| {
                    serde_json::to_string_pretty(&reports).map_err(|err| {
                        ax_perturb::CosmologyError::ParityFixtureValidationFailure {
                            fixture: err.to_string(),
                        }
                    })
                }) {
                    Ok(report) => Expr::Sym(interner.get_or_intern(&report)),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_harmonic_spec" => {
            if args.len() == 1 {
                if let Some(curvature) = parse_curvature_name(&args[0], interner) {
                    make_harmonic_spec_expr(curvature, ax_perturb::SectorKind::Tensor, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "frw_background_spec" => {
            if args.len() == 3 {
                let bg = match (
                    name_from_expr(&args[0], interner),
                    name_from_expr(&args[1], interner),
                    usize_from_expr(&args[2]),
                ) {
                    (Some("conformal"), Some("flat"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Flat,
                            ax_perturb::TimeCoordinate::Conformal,
                        )
                    }
                    (Some("conformal"), Some("closed"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Closed,
                            ax_perturb::TimeCoordinate::Conformal,
                        )
                    }
                    (Some("conformal"), Some("open"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Open,
                            ax_perturb::TimeCoordinate::Conformal,
                        )
                    }
                    (Some("cosmic"), Some("flat"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Flat,
                            ax_perturb::TimeCoordinate::Cosmic,
                        )
                    }
                    (Some("cosmic"), Some("closed"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Closed,
                            ax_perturb::TimeCoordinate::Cosmic,
                        )
                    }
                    (Some("cosmic"), Some("open"), Some(dim)) => {
                        ax_perturb::FrwBackgroundSpec::new(
                            interner.get_or_intern("a"),
                            interner.get_or_intern("H"),
                            interner.get_or_intern("H_cosmic"),
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("t"),
                            dim,
                            ax_perturb::SpatialCurvature::Open,
                            ax_perturb::TimeCoordinate::Cosmic,
                        )
                    }
                    _ => return Expr::Call(f, args),
                };
                bg.map(|bg| make_background_spec_expr(&bg, interner))
                    .unwrap_or_else(|_| Expr::Call(f, args))
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_gauge" => {
            if args.len() == 1 {
                match name_from_expr(&args[0], interner) {
                    Some("newtonian") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::Newtonian, interner)
                    }
                    Some("synchronous") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::Synchronous, interner)
                    }
                    Some("comoving") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::Comoving, interner)
                    }
                    Some("flat") => make_gauge_spec_expr(ax_perturb::GaugeKind::Flat, interner),
                    Some("uniform_density") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::UniformDensity, interner)
                    }
                    Some("uniform_curvature") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::UniformCurvature, interner)
                    }
                    Some("poisson") => {
                        make_gauge_spec_expr(ax_perturb::GaugeKind::Poisson, interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_matter" => match args.as_slice() {
            [kind] => match name_from_expr(kind, interner) {
                Some("perfect_fluid") => {
                    make_matter_spec_expr(ax_perturb::MatterKind::PerfectFluid, interner)
                }
                Some("imperfect_fluid") => {
                    make_matter_spec_expr(ax_perturb::MatterKind::ImperfectFluid, interner)
                }
                Some("canonical_scalar") => {
                    make_matter_spec_expr(ax_perturb::MatterKind::CanonicalScalar, interner)
                }
                Some("symbolic") => {
                    make_matter_spec_expr(ax_perturb::MatterKind::Symbolic, interner)
                }
                _ => Expr::Call(f, args),
            },
            [kind, nfields] => match (name_from_expr(kind, interner), usize_from_expr(nfields)) {
                (Some("multi_canonical_scalar"), Some(fields)) => make_matter_spec_expr(
                    ax_perturb::MatterKind::MultiCanonicalScalar { fields },
                    interner,
                ),
                _ => Expr::Call(f, args),
            },
            _ => Expr::Call(f, args),
        },
        "cpt_linearized_einstein" => {
            if args.len() == 4 {
                let Some(order) = usize_from_expr(&args[0]) else {
                    return Expr::Call(f, args);
                };
                let Some(bg) = parse_background_spec_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(gauge) = parse_gauge_spec_expr(&args[2], interner) else {
                    return Expr::Call(f, args);
                };
                if parse_matter_spec_expr(&args[3], interner).is_none()
                    || gauge != ax_perturb::GaugeKind::Newtonian
                {
                    return Expr::Call(f, args);
                }
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                let equations = match order {
                    1 => ax_perturb::cosmology::linearized_einstein_scalar(&bg, &decomp, interner),
                    2 => ax_perturb::cosmology::linearized_einstein_second_order(
                        &bg, &decomp, interner,
                    ),
                    _ => return Expr::Call(f, args),
                };
                equations
                    .map(|items| labelled_exprs_to_list(items, interner))
                    .unwrap_or_else(|_| Expr::Call(f, args))
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_fluid_equations" => {
            if args.len() == 1 {
                let Some(bg) = parse_background_spec_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                ax_perturb::cosmology::perfect_fluid_linear_conservation(&bg, interner)
                    .map(|items| labelled_exprs_to_list(items, interner))
                    .unwrap_or_else(|_| Expr::Call(f, args))
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_quadratic_action" => {
            if args.len() == 2 {
                let Some(bg) = parse_background_spec_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                match parse_matter_spec_expr(&args[1], interner) {
                    Some(ax_perturb::MatterKind::CanonicalScalar) => {
                        let symbols = ax_perturb::standard_canonical_scalar_symbols(interner);
                        ax_perturb::canonical_scalar_reduced_quadratic_action(
                            &bg, &symbols, interner,
                        )
                        .map(|action| action.lagrangian_density)
                        .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_mukhanov_sasaki" => {
            if args.len() == 2 {
                let Some(bg) = parse_background_spec_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                match parse_matter_spec_expr(&args[1], interner) {
                    Some(ax_perturb::MatterKind::CanonicalScalar) => {
                        let symbols = ax_perturb::standard_canonical_scalar_symbols(interner);
                        ax_perturb::derive_mukhanov_sasaki_from_action(&bg, &symbols, interner)
                            .map(|derivation| derivation.fourier_space_equation)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_mukhanov_sasaki_first_order" => {
            if args.len() == 2 {
                let Some(bg) = parse_background_spec_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                match parse_matter_spec_expr(&args[1], interner) {
                    Some(ax_perturb::MatterKind::CanonicalScalar) => {
                        let symbols = ax_perturb::standard_canonical_scalar_symbols(interner);
                        match ax_perturb::mukhanov_sasaki_first_order_system(
                            &bg, &symbols, interner,
                        ) {
                            Ok(system) => Expr::List(
                                system
                                    .into_iter()
                                    .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                                    .collect(),
                            ),
                            Err(_) => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_bardeen_invariance" => {
            if args.len() == 1 {
                let Some(bg) = parse_background_spec_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let generator = ax_perturb::default_scalar_gauge_generator(interner);
                match ax_perturb::bardeen_variations(&bg, &generator, interner) {
                    Ok(items) => Expr::List(
                        items
                            .into_iter()
                            .map(|item| {
                                Expr::List(vec![
                                    Expr::Sym(item.name),
                                    item.variation,
                                    Expr::Int(BigInt::from(if item.is_invariant { 1 } else { 0 })),
                                ])
                            })
                            .collect(),
                    ),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cpt_export_mode_rhs" => {
            if args.len() == 3 {
                let Some(target) = name_from_expr(&args[0], interner) else {
                    return Expr::Call(f, args);
                };
                let Some(bg) = parse_background_spec_expr(&args[1], interner) else {
                    return Expr::Call(f, args);
                };
                match parse_matter_spec_expr(&args[2], interner) {
                    Some(ax_perturb::MatterKind::CanonicalScalar) => {
                        let symbols = ax_perturb::standard_canonical_scalar_symbols(interner);
                        let Ok(system) =
                            ax_perturb::mukhanov_sasaki_first_order_system(&bg, &symbols, interner)
                        else {
                            return Expr::Call(f, args);
                        };
                        let Some((lhs_second, rhs_second)) = system.get(1) else {
                            return Expr::Call(f, args);
                        };
                        let Expr::Sym(v1_source) = lhs_second else {
                            return Expr::Call(f, args);
                        };
                        let v1 = interner.get_or_intern("v1");
                        let rhs = substitute_symbol_expr(rhs_second, *v1_source, &Expr::Sym(v1));
                        let function_args = vec![
                            interner.get_or_intern("eta"),
                            interner.get_or_intern("v"),
                            v1,
                            interner.get_or_intern("k"),
                            interner.get_or_intern("c_s"),
                            interner.get_or_intern("a"),
                            interner.get_or_intern("epsilon"),
                        ];
                        let code = match target {
                            "python" => ax_codegen::emit_python_function(
                                "ms_rhs",
                                &function_args,
                                &rhs,
                                interner,
                            ),
                            "rust" => ax_codegen::emit_rust_function(
                                "ms_rhs",
                                &function_args,
                                &rhs,
                                interner,
                            ),
                            "cpp" => ax_codegen::emit_cpp_function(
                                "ms_rhs",
                                &function_args,
                                &rhs,
                                interner,
                            ),
                            _ => return Expr::Call(f, args),
                        };
                        Expr::Sym(interner.get_or_intern(&code))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "svt_decompose" => {
            if args.is_empty() {
                match ax_perturb::gauge::svt_decompose_perturbation(3, interner) {
                    Ok(decomp) => svt_decomposition_to_expr(decomp, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bardeen" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let Ok(decomp) = ax_perturb::gauge::svt_decompose_perturbation(3, interner) else {
                    return Expr::Call(f, args);
                };
                match ax_perturb::gauge::bardeen_variables(&decomp, &bg, interner) {
                    Ok(vars) => named_exprs_to_list(vars),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "regge_wheeler_decompose" => {
            if args.len() == 1 {
                if let Some(l) = usize_from_expr(&args[0]) {
                    regge_wheeler_decomposition_to_expr(
                        ax_perturb::gauge::regge_wheeler_decompose(l, interner),
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "zerilli" | "regge_wheeler" => {
            if args.len() == 1 {
                if let Some(l) = usize_from_expr(&args[0]) {
                    let mass = interner.get_or_intern("M");
                    if name == "zerilli" {
                        ax_perturb::gauge::zerilli_equation(l, mass, interner)
                    } else {
                        ax_perturb::gauge::regge_wheeler_equation(l, mass, interner)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "power_spectrum" => {
            if args.is_empty() {
                let bg = ax_perturb::cosmology::frw_background(interner);
                let eps = interner.get_or_intern("epsilon");
                let h_star = interner.get_or_intern("H_star");
                ax_perturb::cosmology::power_spectrum_leading(&bg, eps, h_star, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "spectral_index" => {
            if args.is_empty() {
                let eps = interner.get_or_intern("epsilon");
                let eta = interner.get_or_intern("eta_sr");
                ax_perturb::cosmology::spectral_index(eps, eta, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_scalar_ratio" => {
            if args.is_empty() {
                let eps = interner.get_or_intern("epsilon");
                ax_perturb::cosmology::tensor_to_scalar_ratio(eps, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "graded" => {
            if args.len() == 2 && symbol_from_expr(&args[0]).is_some() {
                if let Some(grading) = grading_from_expr(&args[1], interner) {
                    Expr::Sym(interner.get_or_intern(match grading {
                        ax_graded::Grading::Z2(0) => "bosonic",
                        ax_graded::Grading::Z2(_) => "fermionic",
                        ax_graded::Grading::Z(_) | ax_graded::Grading::Product(_) => "graded",
                    }))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "graded_commutator" => {
            if args.len() == 2 {
                ax_graded::graded_commutator(&args[0], &args[1], &env.graded_table, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "graded_simplify" => {
            if args.len() == 1 {
                ax_graded::graded_simplify(&args[0], &env.graded_table, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "setup_superspace" => {
            if args.len() == 1 {
                match usize_from_expr(&args[0]) {
                    Some(1) => Expr::Sym(interner.get_or_intern("superspace_n1")),
                    Some(_) => {
                        Expr::Sym(interner.get_or_intern("N_gt_1_superspace_not_yet_implemented"))
                    }
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_superfield"
        | "chiral_superfield"
        | "antichiral_superfield"
        | "vector_superfield_wz" => {
            if args.len() == 1 {
                if let Some(name_sym) = symbol_from_expr(&args[0]) {
                    let (setup, _) = active_superspace(env, interner);
                    let expansion = match name {
                        "expand_superfield" => {
                            ax_graded::superspace::expand_superfield(name_sym, &setup, interner)
                        }
                        "chiral_superfield" => {
                            let expanded = ax_graded::superspace::expand_superfield(
                                name_sym, &setup, interner,
                            );
                            ax_graded::superspace::chiral_constraint(&expanded, &setup, interner)
                        }
                        "antichiral_superfield" => {
                            let expanded = ax_graded::superspace::expand_superfield(
                                name_sym, &setup, interner,
                            );
                            ax_graded::superspace::antichiral_constraint(
                                &expanded, &setup, interner,
                            )
                        }
                        "vector_superfield_wz" => {
                            ax_graded::superspace::vector_superfield_wz_gauge(
                                name_sym, &setup, interner,
                            )
                        }
                        _ => unreachable!(),
                    };
                    ax_graded::superspace::superfield_to_expr(&expansion, &setup, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "extract_component" => {
            if args.len() == 2 {
                let (setup, table) = active_superspace(env, interner);
                if let Some(theta) = theta_monomial_from_spec(&args[1], &setup) {
                    ax_graded::superspace::extract_component(
                        &args[0], &theta, &setup, &table, interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "d_alpha" | "d_bar" => {
            if args.len() == 2 {
                let (setup, table) = active_superspace(env, interner);
                if let Some(alpha) = usize_from_expr(&args[1]) {
                    if name == "d_alpha" {
                        ax_graded::d_algebra::apply_d_alpha(
                            &args[0], alpha, &setup, &table, interner,
                        )
                    } else {
                        ax_graded::d_algebra::apply_d_bar_alpha_dot(
                            &args[0], alpha, &setup, &table, interner,
                        )
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "d_squared" | "d_bar_squared" => {
            if args.len() == 1 {
                let (setup, table) = active_superspace(env, interner);
                if name == "d_squared" {
                    ax_graded::d_algebra::d_squared(&args[0], &setup, &table, interner)
                } else {
                    ax_graded::d_algebra::d_bar_squared(&args[0], &setup, &table, interner)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "superspace_integrate" => {
            if args.len() == 2 {
                let (setup, table) = active_superspace(env, interner);
                let measure =
                    match name_from_expr(&args[1], interner).map(|s| s.to_ascii_lowercase()) {
                        Some(s) if s == "full" => {
                            ax_graded::d_algebra::SuperspaceMeasure::FullSuperspace
                        }
                        Some(s) if s == "chiral" => ax_graded::d_algebra::SuperspaceMeasure::Chiral,
                        Some(s) if s == "antichiral" || s == "anti_chiral" => {
                            ax_graded::d_algebra::SuperspaceMeasure::AntiChiral
                        }
                        _ => return Expr::Call(f, args),
                    };
                ax_graded::d_algebra::superspace_integrate(
                    &args[0], measure, &setup, &table, interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "setup_brst_ym" => {
            if args.len() == 5 {
                Expr::Sym(interner.get_or_intern("brst_yang_mills"))
            } else {
                Expr::Call(f, args)
            }
        }
        "brst" => {
            if args.len() == 1 {
                if let Some(setup) = &env.brst_setup {
                    ax_graded::brst::apply_brst(&args[0], setup, &env.graded_table, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "brst_check" => {
            if args.len() == 1 {
                if let Some(setup) = &env.brst_setup {
                    let applied =
                        ax_graded::brst::apply_brst(&args[0], setup, &env.graded_table, interner);
                    let simplified =
                        ax_graded::graded_simplify(&applied, &env.graded_table, interner);
                    Expr::Sym(interner.get_or_intern(if simplified == Expr::zero() {
                        "true"
                    } else {
                        "false"
                    }))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ghost_number" => {
            if args.len() == 1 {
                ax_graded::brst::ghost_number(&args[0], &env.graded_table)
                    .map(|n| Expr::Int(BigInt::from(n)))
                    .unwrap_or_else(|| {
                        Expr::Sym(interner.get_or_intern("inconsistent_ghost_number"))
                    })
            } else {
                Expr::Call(f, args)
            }
        }
        "filter_ghost" | "filter_ghost_number" => {
            if args.len() == 2 {
                if let Expr::Int(n) = &args[1] {
                    if let Some(target) = n.to_i32() {
                        ax_graded::brst::filter_by_ghost_number(
                            &args[0],
                            target,
                            &env.graded_table,
                            interner,
                        )
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "Re" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Complex(re, _) => re.as_ref().clone(),
                    other => other.clone(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "Im" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Complex(_, im) => im.as_ref().clone(),
                    _ => Expr::zero(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conj" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Complex(re, im) => Expr::Complex(
                        Box::new(re.as_ref().clone()),
                        Box::new(Expr::neg(im.as_ref().clone())),
                    ),
                    other => other.clone(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "N" => {
            if args.len() == 1 {
                if let Expr::Complex(re, im) = &args[0] {
                    if let (Some(re), Some(im)) = (to_f64(re), to_f64(im)) {
                        Expr::Complex(Box::new(Expr::Float(re)), Box::new(Expr::Float(im)))
                    } else {
                        Expr::Call(f, args)
                    }
                } else if let Some(v) = to_f64(&args[0]) {
                    Expr::Float(v)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sin" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        if let Some(special) = try_trig_special_float("sin", *v, interner) {
                            special
                        } else {
                            Expr::Float(v.sin())
                        }
                    }
                    Expr::Mul(factors) if factors.len() == 2 => {
                        if let Some(special) = try_trig_special_value("sin", factors, interner) {
                            special
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    Expr::Sym(s) if interner.resolve(*s) == "pi" => Expr::Int(0.into()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cos" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        if let Some(special) = try_trig_special_float("cos", *v, interner) {
                            special
                        } else {
                            Expr::Float(v.cos())
                        }
                    }
                    Expr::Mul(factors) if factors.len() == 2 => {
                        if let Some(special) = try_trig_special_value("cos", factors, interner) {
                            special
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    Expr::Sym(s) if interner.resolve(*s) == "pi" => Expr::Int((-1i64).into()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "exp" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Call(inner_f, inner_args)
                        if interner.resolve(*inner_f) == "log" && inner_args.len() == 1 =>
                    {
                        inner_args[0].clone()
                    }
                    Expr::Float(v) => Expr::Float(v.exp()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    Expr::Complex(re, im) => {
                        if let (Some(a), Some(b)) = (to_f64(re), to_f64(im)) {
                            let mag = a.exp();
                            Expr::Complex(
                                Box::new(Expr::Float(mag * b.cos())),
                                Box::new(Expr::Float(mag * b.sin())),
                            )
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "log" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Call(inner_f, inner_args)
                        if interner.resolve(*inner_f) == "exp" && inner_args.len() == 1 =>
                    {
                        inner_args[0].clone()
                    }
                    Expr::Pow(base, exp) => {
                        let base_is_e = matches!(base.as_ref(), Expr::Sym(sym) if interner.resolve(*sym) == "e");
                        if !base_is_e {
                            Expr::mul(vec![
                                exp.as_ref().clone(),
                                builtin_unary("log", base.as_ref().clone(), interner),
                            ])
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    Expr::Float(v) if *v > 0.0 => Expr::Float(v.ln()),
                    Expr::Int(n) if n.is_one() => Expr::Int(0.into()),
                    Expr::Float(v) if *v == 1.0 => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sinh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.sinh()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cosh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.cosh()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tanh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.tanh()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "asin" | "arcsin" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v >= -1.0 && *v <= 1.0 => Expr::Float(v.asin()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    Expr::Int(n) if *n == 1.into() => Expr::Float(std::f64::consts::FRAC_PI_2),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "acos" | "arccos" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v >= -1.0 && *v <= 1.0 => Expr::Float(v.acos()),
                    Expr::Int(n) if n.is_zero() => Expr::Float(std::f64::consts::FRAC_PI_2),
                    Expr::Int(n) if *n == 1.into() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "atan" | "arctan" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.atan()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "atan2" => {
            if args.len() == 2 {
                if let (Some(y), Some(x_val)) = (to_f64(&args[0]), to_f64(&args[1])) {
                    Expr::Float(y.atan2(x_val))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sec" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        let c = v.cos();
                        if c.abs() < 1e-15 {
                            Expr::Call(f, args)
                        } else {
                            Expr::Float(1.0 / c)
                        }
                    }
                    Expr::Int(n) if n.is_zero() => Expr::Int(1.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "csc" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        let s = v.sin();
                        if s.abs() < 1e-15 {
                            Expr::Call(f, args)
                        } else {
                            Expr::Float(1.0 / s)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cot" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        let t = v.tan();
                        if t.abs() < 1e-15 {
                            Expr::Call(f, args)
                        } else {
                            Expr::Float(1.0 / t)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "asinh" | "arcsinh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => Expr::Float(v.asinh()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "acosh" | "arccosh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v >= 1.0 => Expr::Float(v.acosh()),
                    Expr::Int(n) if *n == 1.into() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "atanh" | "arctanh" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if v.abs() < 1.0 => Expr::Float(v.atanh()),
                    Expr::Int(n) if n.is_zero() => Expr::Int(0.into()),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sign" | "sgn" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) => {
                        if *v > 0.0 {
                            Expr::Int(1.into())
                        } else if *v < 0.0 {
                            Expr::Int((-1i64).into())
                        } else {
                            Expr::Int(0.into())
                        }
                    }
                    Expr::Int(n) => {
                        use num_traits::Signed;
                        if n.is_positive() {
                            Expr::Int(1.into())
                        } else if n.is_negative() {
                            Expr::Int((-1i64).into())
                        } else {
                            Expr::Int(0.into())
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sqrt" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Float(v) if *v >= 0.0 => Expr::Float(v.sqrt()),
                    Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into()) => {
                        match base.as_ref() {
                            Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Positive) => {
                                Expr::Sym(*sym)
                            }
                            _ => builtin_unary("abs", base.as_ref().clone(), interner),
                        }
                    }
                    Expr::Int(n) => {
                        if let Some(root) = perfect_square_root(n) {
                            Expr::Int(root)
                        } else {
                            Expr::pow(args[0].clone(), one_half())
                        }
                    }
                    _ => Expr::pow(args[0].clone(), one_half()),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "abs" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Int(n) => Expr::Int(n.abs()),
                    Expr::Float(v) => Expr::Float(v.abs()),
                    Expr::Complex(re, im) => Expr::Call(
                        interner.get_or_intern("sqrt"),
                        vec![Expr::add(vec![
                            Expr::pow(re.as_ref().clone(), Expr::Int(2.into())),
                            Expr::pow(im.as_ref().clone(), Expr::Int(2.into())),
                        ])],
                    ),
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Positive) => {
                        Expr::Sym(*sym)
                    }
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Negative) => {
                        Expr::neg(Expr::Sym(*sym))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "arg" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Complex(re, im) => {
                        if let (Some(a), Some(b)) = (to_f64(re), to_f64(im)) {
                            Expr::Float(b.atan2(a))
                        } else {
                            Expr::Call(
                                interner.get_or_intern("arctan"),
                                vec![Expr::mul(vec![
                                    im.as_ref().clone(),
                                    Expr::pow(re.as_ref().clone(), Expr::Int((-1).into())),
                                ])],
                            )
                        }
                    }
                    _ => Expr::zero(),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "canonicalise" | "canonicalize" => {
            if args.len() == 1 {
                let mut pooled_env = env.clone();
                let properties = pooled_env.property_store.clone();
                let canonical =
                    maybe_pooled_canonicalise(&args[0], &mut pooled_env, &properties, interner);
                ax_tensor::rename_dummies(&canonical, env, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "lower_free_indices" | "lower_indices" => {
            if args.len() == 1 {
                ax_tensor::lower_free_indices(
                    &args[0],
                    &env.index_to_family,
                    &env.index_families,
                    interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "raise_free_indices" | "raise_indices" => {
            if args.len() == 1 {
                ax_tensor::raise_free_indices(
                    &args[0],
                    &env.index_to_family,
                    &env.index_families,
                    interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "meld" => {
            if args.len() == 1 {
                let mut pooled_env = env.clone();
                let properties = pooled_env.property_store.clone();
                maybe_pooled_meld(&args[0], &mut pooled_env, &properties, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "rename_dummies" => {
            if args.len() == 1 {
                ax_tensor::rename_dummies(&args[0], env, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "sort_product" => {
            if args.len() == 1 {
                ax_tensor::sort_product(&args[0], &env.property_store, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "product_rule" | "leibniz" => {
            if args.len() == 1 {
                let deriv_syms = symbols_with_property(env, |p| {
                    matches!(
                        p,
                        ax_ir::TensorProperty::Derivative
                            | ax_ir::TensorProperty::PartialDerivative
                            | ax_ir::TensorProperty::CovariantDerivative
                    )
                });
                ax_tensor::product_rule(&args[0], &deriv_syms, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "unwrap" => {
            if args.len() == 1 {
                let deriv_syms = symbols_with_property(env, |p| {
                    matches!(
                        p,
                        ax_ir::TensorProperty::Derivative
                            | ax_ir::TensorProperty::PartialDerivative
                            | ax_ir::TensorProperty::CovariantDerivative
                    )
                });
                let depends = explicit_depends_map(env);
                ax_tensor::unwrap_derivatives(&args[0], &deriv_syms, &depends, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "integrate_by_parts" | "ibp" => {
            if args.len() >= 2 {
                if let Expr::Sym(away) = &args[1] {
                    let deriv_syms = symbols_with_property(env, |p| {
                        matches!(
                            p,
                            ax_ir::TensorProperty::Derivative
                                | ax_ir::TensorProperty::PartialDerivative
                                | ax_ir::TensorProperty::CovariantDerivative
                        )
                    });
                    ax_tensor::integrate_by_parts(&args[0], *away, &deriv_syms, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_distribute" | "tdistribute" => {
            if args.len() == 1 {
                ax_tensor::tensor_distribute(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "zoom" => {
            if args.len() == 2 {
                let (zoomed, remainder) = zoom(&args[0], &args[1], interner);
                Expr::List(vec![zoomed, remainder])
            } else {
                Expr::Call(f, args)
            }
        }
        "unzoom" => {
            if args.len() == 2 {
                unzoom(&args[0], &args[1])
            } else {
                Expr::Call(f, args)
            }
        }
        "take_match" => {
            if args.len() == 2 {
                take_match(&args[0], &args[1], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "keep_weight" => {
            if args.len() >= 2 {
                if let Expr::Int(w) = &args[1] {
                    let target = num_traits::ToPrimitive::to_i64(w).unwrap_or(0);
                    let label = args
                        .get(2)
                        .and_then(|e| {
                            if let Expr::Sym(s) = e {
                                Some(interner.resolve(*s).to_string())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    ax_tensor::keep_weight(&args[0], target, &env.weights, &label, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "drop_weight" => {
            if args.len() >= 2 {
                if let Expr::Int(w) = &args[1] {
                    let target = num_traits::ToPrimitive::to_i64(w).unwrap_or(0);
                    let label = args
                        .get(2)
                        .and_then(|e| {
                            if let Expr::Sym(s) = e {
                                Some(interner.resolve(*s).to_string())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    ax_tensor::drop_weight(&args[0], target, &env.weights, &label, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "einsteinify" => {
            if !args.is_empty() {
                let metric = if args.len() >= 2 {
                    if let Expr::Sym(s) = &args[1] {
                        Some(*s)
                    } else {
                        None
                    }
                } else {
                    None
                };
                ax_tensor::einsteinify(&args[0], metric, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "split_index" => {
            if args.len() == 4 {
                let parents = extract_sym_list(&args[1]);
                let sub1 = extract_sym_list(&args[2]);
                let sub2 = extract_sym_list(&args[3]);
                ax_tensor::split_index(&args[0], &parents, &sub1, &sub2, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "eliminate_kronecker" => {
            if !args.is_empty() {
                let delta = if args.len() >= 2 {
                    if let Expr::Sym(s) = &args[1] {
                        *s
                    } else {
                        interner.get_or_intern("delta")
                    }
                } else {
                    interner.get_or_intern("delta")
                };
                ax_tensor::eliminate_kronecker(&args[0], delta, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_delta" => {
            if !args.is_empty() {
                let delta = interner.get_or_intern("delta");
                ax_tensor::expand_delta(&args[0], delta, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_dummies" => {
            if !args.is_empty() {
                let coords: Vec<lasso::Spur> = env.coordinates.iter().copied().collect();
                ax_tensor::expand_dummies(&args[0], &coords, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "explicit_indices" => {
            if !args.is_empty() {
                let implicit = symbols_with_property(env, |p| {
                    matches!(p, ax_ir::TensorProperty::GammaMatrixProp)
                });
                let n_per: HashMap<lasso::Spur, usize> = implicit.iter().map(|s| (*s, 2)).collect();
                let avail: Vec<lasso::Spur> = (0..20)
                    .map(|i| interner.get_or_intern(&format!("_impl{}", i)))
                    .collect();
                ax_tensor::explicit_indices(
                    &args[0],
                    &implicit,
                    &avail,
                    &n_per,
                    &env.property_store,
                    interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_implicit" => {
            if !args.is_empty() {
                let implicit = symbols_with_property(env, |p| {
                    matches!(p, ax_ir::TensorProperty::GammaMatrixProp)
                });
                let n_per: HashMap<lasso::Spur, usize> = implicit.iter().map(|s| (*s, 2)).collect();
                let avail: Vec<lasso::Spur> = (0..40)
                    .map(|i| interner.get_or_intern(&format!("_exp{}", i)))
                    .collect();
                ax_tensor::expand_implicit(
                    &args[0],
                    &implicit,
                    &avail,
                    &n_per,
                    &env.property_store,
                    interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "rewrite_indices" => {
            // rewrite_indices(expr, tensor_name, [down, down, ...])
            if args.len() >= 3 {
                if let (Expr::Sym(tensor), Expr::List(variances)) = (&args[1], &args[2]) {
                    let vars: Vec<ax_ir::Variance> = variances
                        .iter()
                        .map(|v| {
                            if let Expr::Sym(s) = v {
                                if interner.resolve(*s) == "up" {
                                    ax_ir::Variance::Up
                                } else {
                                    ax_ir::Variance::Down
                                }
                            } else {
                                ax_ir::Variance::Down
                            }
                        })
                        .collect();
                    let mut targets = HashMap::new();
                    targets.insert(*tensor, vars);
                    let g = interner.get_or_intern("g");
                    let ginv = interner.get_or_intern("ginv");
                    ax_tensor::rewrite_indices(&args[0], &targets, g, ginv, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "rewrite_indices_vielbein" => {
            if args.len() == 5 {
                if let (
                    Expr::Sym(e_sym),
                    Expr::Sym(e_inv_sym),
                    Expr::Sym(from_family),
                    Expr::Sym(to_family),
                ) = (&args[1], &args[2], &args[3], &args[4])
                {
                    ax_tensor::rewrite_indices_vielbein(
                        &args[0],
                        *e_sym,
                        *e_inv_sym,
                        *from_family,
                        *to_family,
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "reduce_delta" => {
            if !args.is_empty() {
                let delta = interner.get_or_intern("delta");
                let dim = interner.get_or_intern("dim");
                ax_tensor::reduce_delta(&args[0], delta, dim, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "young_project" => {
            if !args.is_empty() {
                if let Some(tableau) = args.get(1).and_then(parse_young_tableau_expr) {
                    ax_tensor::young_project(&args[0], &tableau, interner)
                } else {
                    let opts = ax_tensor::YoungProjectTensorOptions {
                        modulo_monoterm: args
                            .get(1)
                            .and_then(|arg| parse_bool_like_expr(arg, interner))
                            .unwrap_or(true),
                        canonicalize_after: args
                            .get(2)
                            .and_then(|arg| parse_bool_like_expr(arg, interner))
                            .unwrap_or(true),
                        rename_dummies_after: args
                            .get(3)
                            .and_then(|arg| parse_bool_like_expr(arg, interner))
                            .unwrap_or(true),
                    };
                    ax_tensor::young_project_tensor_with_options(
                        &args[0],
                        &env.property_store,
                        interner,
                        &opts,
                    )
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "young_project_tensor" => {
            if !args.is_empty() {
                let opts = ax_tensor::YoungProjectTensorOptions {
                    modulo_monoterm: args
                        .get(1)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    canonicalize_after: args
                        .get(2)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    rename_dummies_after: args
                        .get(3)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                };
                ax_tensor::young_project_tensor_with_options(
                    &args[0],
                    &env.property_store,
                    interner,
                    &opts,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_reduce" => {
            if !args.is_empty() {
                let target = eval(&args[0], env, interner);
                let opts = ax_tensor::TensorReduceOptions {
                    monoterm: args
                        .get(1)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    multiterm: args
                        .get(2)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    dimension_dependent: args
                        .get(3)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    meld: args
                        .get(4)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    modulo_monoterm: args
                        .get(5)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                };
                ax_tensor::tensor_reduce(&target, &env.property_store, interner, &opts)
            } else {
                Expr::Call(f, args)
            }
        }
        "abstract_tensor_reduce" | "abstract_gr_reduce" => {
            if !args.is_empty() {
                let target = eval(&args[0], env, interner);
                let opts = ax_tensor::TensorReduceOptions {
                    monoterm: args
                        .get(1)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    multiterm: args
                        .get(2)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    dimension_dependent: args
                        .get(3)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    meld: args
                        .get(4)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                    modulo_monoterm: args
                        .get(5)
                        .and_then(|arg| parse_bool_like_expr(arg, interner))
                        .unwrap_or(true),
                };
                ax_tensor::tensor_reduce(&target, &env.property_store, interner, &opts)
            } else {
                Expr::Call(f, args)
            }
        }
        "riemann_to_ricci" => {
            if args.len() == 2 || args.len() == 3 {
                let ricci_sym = match args.get(1) {
                    Some(Expr::Sym(sym)) => *sym,
                    _ => {
                        return Expr::Sym(
                            interner.get_or_intern(
                                &ax_tensor::AbstractCurvatureReduceError::MissingRicciSymbol
                                    .to_string(),
                            ),
                        )
                    }
                };
                let scalar_sym = match args.get(2) {
                    Some(Expr::Sym(sym)) => Some(*sym),
                    Some(_) => return Expr::Call(f, args),
                    None => None,
                };
                match ax_tensor::riemann_to_ricci(
                    &args[0],
                    ricci_sym,
                    scalar_sym,
                    &env.property_store,
                    interner,
                ) {
                    Ok(result) => result,
                    Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "contracted_bianchi_reduce" => {
            if args.len() == 4 || args.len() == 5 {
                let derivative_sym = match args.get(1) {
                    Some(Expr::Sym(sym)) => *sym,
                    _ => return Expr::Call(f, args),
                };
                let ricci_sym = match args.get(2) {
                    Some(Expr::Sym(sym)) => *sym,
                    _ => return Expr::Call(f, args),
                };
                let scalar_sym = match args.get(3) {
                    Some(Expr::Sym(sym)) => *sym,
                    _ => return Expr::Call(f, args),
                };
                let einstein_sym = match args.get(4) {
                    Some(Expr::Sym(sym)) => Some(*sym),
                    Some(_) => return Expr::Call(f, args),
                    None => None,
                };
                match ax_tensor::contracted_bianchi_reduce(
                    &args[0],
                    derivative_sym,
                    ricci_sym,
                    scalar_sym,
                    einstein_sym,
                    &env.property_store,
                    interner,
                ) {
                    Ok(result) => result,
                    Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "symmetrise" | "symmetrize" | "sym" => {
            if args.len() >= 2 {
                if let Expr::List(pos_list) = &args[1] {
                    let positions: Vec<usize> = pos_list
                        .iter()
                        .filter_map(|expr| {
                            if let Expr::Int(n) = expr {
                                n.to_usize()
                            } else {
                                None
                            }
                        })
                        .collect();
                    ax_tensor::symmetrise(&args[0], &positions, false, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "antisymmetrise" | "antisymmetrize" | "asym" => {
            if args.len() >= 2 {
                if let Expr::List(pos_list) = &args[1] {
                    let positions: Vec<usize> = pos_list
                        .iter()
                        .filter_map(|expr| {
                            if let Expr::Int(n) = expr {
                                n.to_usize()
                            } else {
                                None
                            }
                        })
                        .collect();
                    ax_tensor::symmetrise(&args[0], &positions, true, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eliminate_metric" => {
            if !args.is_empty() {
                let metric = interner.get_or_intern("g");
                let inv_metric = interner.get_or_intern("ginv");
                ax_tensor::eliminate_metric(&args[0], metric, inv_metric, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "eliminate_vielbein" => {
            if !args.is_empty() {
                let vb = interner.get_or_intern("e");
                let ivb = interner.get_or_intern("einv");
                ax_tensor::eliminate_vielbein(&args[0], vb, ivb, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "decompose" => {
            if args.len() >= 2 {
                if let Expr::List(basis_exprs) = &args[1] {
                    ax_tensor::decompose(&args[0], basis_exprs, &env.property_store, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "decompose_product" => {
            if !args.is_empty() {
                let dim = args
                    .get(1)
                    .and_then(|e| {
                        if let Expr::Int(n) = e {
                            n.to_usize()
                        } else {
                            None
                        }
                    })
                    .or_else(|| ax_tensor::infer_tensor_dimension(&args[0], &env.property_store));
                match dim {
                    Some(dim) => {
                        ax_tensor::decompose_product(&args[0], dim, &env.property_store, interner)
                    }
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "schouten_reduce" => {
            if !args.is_empty() {
                ax_tensor::schouten_reduce(&args[0], &env.property_store, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "epsilon_to_delta" => {
            if !args.is_empty() {
                let eps = interner.get_or_intern("epsilon");
                let delta = interner.get_or_intern("delta");
                let dim = args
                    .get(1)
                    .and_then(|expr| match expr {
                        Expr::Int(n) => n.to_usize(),
                        _ => None,
                    })
                    .unwrap_or(4);
                ax_tensor::epsilon_to_delta(&args[0], eps, delta, dim, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "evaluate" | "eval_components" => {
            if args.len() >= 2 {
                if let Expr::List(rule_exprs) = &args[1] {
                    let rules = parse_component_rules(rule_exprs);
                    let index_vals = HashMap::new();
                    let mut coords: Vec<_> = env.coordinates.iter().copied().collect();
                    coords.sort_by_key(|sym| interner.resolve(*sym).to_string());
                    let eval_env = ax_tensor::DefaultEvalEnv::new(
                        coords,
                        env.property_store.as_legacy_hashmap(),
                    );
                    ax_tensor::evaluate_components(
                        &args[0],
                        &rules,
                        &index_vals,
                        &eval_env,
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "dim" => {
            if args.len() == 1 {
                let mut units = ax_units::si_units(interner);
                units.extend(ax_units::natural_units(interner));
                match ax_units::check_dimensions(&args[0], &units, interner) {
                    Ok(unit) => ax_units::unit_to_expr(&unit, interner),
                    Err(_) => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "convert" => {
            if args.len() == 3 {
                let mut units = ax_units::si_units(interner);
                units.extend(ax_units::natural_units(interner));
                match (&args[1], &args[2]) {
                    (Expr::Sym(from), Expr::Sym(to)) => match (units.get(from), units.get(to)) {
                        (Some(from_unit), Some(to_unit)) => {
                            match ax_units::convert(&args[0], from_unit, to_unit) {
                                Ok(expr) => eval(&expr, &Env::new(), interner),
                                Err(_) => Expr::Call(f, args),
                            }
                        }
                        _ => Expr::Call(f, args),
                    },
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "check_units" => {
            if args.len() == 1 {
                let mut units = ax_units::si_units(interner);
                units.extend(ax_units::natural_units(interner));
                match ax_units::check_dimensions(&args[0], &units, interner) {
                    Ok(_) => Expr::Sym(interner.get_or_intern("ok")),
                    Err(_) => Expr::Sym(interner.get_or_intern("unit_error")),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "symmetric" | "antisymmetric" | "riemann_symmetry" | "traceless" => Expr::Call(f, args),
        "__declare_indices"
        | "__declare_coordinates"
        | "__declare_property"
        | "__declare_weight"
        | "__declare_depends"
        | "__set_parallel" => Expr::Call(f, args),
        "grassmann" => Expr::Call(f, args),
        "expand" => {
            if args.len() == 1 {
                simplify::expand(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "simplify" => {
            if args.len() == 1 {
                let simplified =
                    simplify::trig_simplify(&simplify::simplify(&args[0], interner), interner);
                if env.gradings.is_empty() {
                    simplified
                } else {
                    grassmann_simplify(&simplified, &env.gradings, interner)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "rationalize" => {
            if args.len() == 1 {
                simplify::rationalize(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "partial_fractions" | "apart" => {
            if args.len() >= 2 {
                if let Expr::Sym(var) = &args[1] {
                    simplify::apart_expr(&args[0], *var, interner)
                        .unwrap_or_else(|| args[0].clone())
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "grassmann_simplify" => {
            if args.len() == 1 {
                grassmann_simplify(&args[0], &env.gradings, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "trig_simplify" => {
            if args.len() == 1 {
                simplify::trig_simplify(&args[0], interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "factor_out" => {
            if !args.is_empty() {
                let targets: Vec<lasso::Spur> = if args.len() >= 2 {
                    match &args[1] {
                        Expr::List(syms) => syms
                            .iter()
                            .filter_map(|e| if let Expr::Sym(s) = e { Some(*s) } else { None })
                            .collect(),
                        Expr::Sym(s) => vec![*s],
                        _ => vec![],
                    }
                } else {
                    vec![]
                };
                simplify::factor_out(&args[0], &targets, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "factor_in" => {
            if !args.is_empty() {
                let targets: Vec<lasso::Spur> = if args.len() >= 2 {
                    match &args[1] {
                        Expr::List(syms) => syms
                            .iter()
                            .filter_map(|e| if let Expr::Sym(s) = e { Some(*s) } else { None })
                            .collect(),
                        Expr::Sym(s) => vec![*s],
                        _ => vec![],
                    }
                } else {
                    vec![]
                };
                simplify::factor_in(&args[0], &targets, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "subs" => {
            if args.len() == 3 {
                match (&args[1], &args[2]) {
                    (Expr::List(targets), Expr::List(replacements))
                        if targets.len() == replacements.len() =>
                    {
                        let substitutions = targets
                            .iter()
                            .cloned()
                            .zip(replacements.iter().cloned())
                            .collect::<Vec<_>>();
                        multi_substitute(&args[0], &substitutions, interner)
                    }
                    _ => {
                        let has_indices = has_any_indices(&args[0])
                            || has_any_indices(&args[1])
                            || has_any_indices(&args[2]);
                        if has_indices {
                            substitute_with_indices(&args[0], &args[1], &args[2], env, interner)
                        } else {
                            symbolic_substitute(&args[0], &args[1], &args[2], interner)
                        }
                    }
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "rewrite" => {
            if args.len() == 1 {
                rewrite_with_trace(&args[0], env, interner).0
            } else {
                Expr::Call(f, args)
            }
        }
        "equiv" => {
            if args.len() == 2 {
                Expr::Sym(
                    interner.get_or_intern(&equiv_description(&args[0], &args[1], env, interner)),
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "semantic_diff" => {
            if args.len() == 2 {
                let description = equiv_description(&args[0], &args[1], env, interner);
                Expr::Sym(interner.get_or_intern(&format!("semantic_diff_{description}")))
            } else {
                Expr::Call(f, args)
            }
        }
        "to_python" => {
            if args.len() == 1 {
                println!(
                    "{}",
                    ax_codegen::generate(&args[0], ax_codegen::Target::Python, interner, None, &[])
                );
                Expr::zero()
            } else {
                Expr::Call(f, args)
            }
        }
        "to_rust" => {
            if args.len() == 1 {
                println!(
                    "{}",
                    ax_codegen::generate(&args[0], ax_codegen::Target::Rust, interner, None, &[])
                );
                Expr::zero()
            } else {
                Expr::Call(f, args)
            }
        }
        "to_cpp" => {
            if args.len() == 1 {
                println!(
                    "{}",
                    ax_codegen::generate(&args[0], ax_codegen::Target::Cpp, interner, None, &[])
                );
                Expr::zero()
            } else {
                Expr::Call(f, args)
            }
        }
        "gradient" | "grad" => {
            if args.len() == 2 {
                if let Expr::List(vars) = &args[1] {
                    let components: Vec<Expr> = vars
                        .iter()
                        .map(|v| {
                            if let Expr::Sym(s) = v {
                                let d = differentiate(&args[0], *s, interner);
                                eval(&d, env, interner)
                            } else {
                                Expr::zero()
                            }
                        })
                        .collect();
                    Expr::List(components)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "divergence" | "div" => {
            if args.len() == 2 {
                if let (Expr::List(components), Expr::List(vars)) = (&args[0], &args[1]) {
                    if components.len() == vars.len() {
                        let terms: Vec<Expr> = components
                            .iter()
                            .zip(vars.iter())
                            .map(|(comp, v)| {
                                if let Expr::Sym(s) = v {
                                    let d = differentiate(comp, *s, interner);
                                    eval(&d, env, interner)
                                } else {
                                    Expr::zero()
                                }
                            })
                            .collect();
                        eval(&Expr::add(terms), env, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "curl" => {
            if args.len() == 2 {
                if let (Expr::List(f_vec), Expr::List(vars)) = (&args[0], &args[1]) {
                    if f_vec.len() == 3 && vars.len() == 3 {
                        let (fx, fy, fz) = (&f_vec[0], &f_vec[1], &f_vec[2]);
                        let (x, y, z) = if let (Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)) =
                            (&vars[0], &vars[1], &vars[2])
                        {
                            (*a, *b, *c)
                        } else {
                            return Expr::Call(f, args);
                        };

                        let curl_x = Expr::add(vec![
                            eval(&differentiate(fz, y, interner), env, interner),
                            Expr::neg(eval(&differentiate(fy, z, interner), env, interner)),
                        ]);
                        let curl_y = Expr::add(vec![
                            eval(&differentiate(fx, z, interner), env, interner),
                            Expr::neg(eval(&differentiate(fz, x, interner), env, interner)),
                        ]);
                        let curl_z = Expr::add(vec![
                            eval(&differentiate(fy, x, interner), env, interner),
                            Expr::neg(eval(&differentiate(fx, y, interner), env, interner)),
                        ]);

                        Expr::List(vec![
                            eval(&curl_x, env, interner),
                            eval(&curl_y, env, interner),
                            eval(&curl_z, env, interner),
                        ])
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "laplacian" => {
            if args.len() == 2 {
                if let Expr::List(vars) = &args[1] {
                    let terms: Vec<Expr> = vars
                        .iter()
                        .map(|v| {
                            if let Expr::Sym(s) = v {
                                let d1 = differentiate(&args[0], *s, interner);
                                let d2 = differentiate(&d1, *s, interner);
                                eval(&d2, env, interner)
                            } else {
                                Expr::zero()
                            }
                        })
                        .collect();
                    eval(&Expr::add(terms), env, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "jacobian" => {
            if args.len() == 2 {
                if let (Expr::List(funcs), Expr::List(vars)) = (&args[0], &args[1]) {
                    let rows: Vec<Vec<Expr>> = funcs
                        .iter()
                        .map(|fi| {
                            vars.iter()
                                .map(|v| {
                                    if let Expr::Sym(s) = v {
                                        eval(&differentiate(fi, *s, interner), env, interner)
                                    } else {
                                        Expr::zero()
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    Expr::Matrix(rows)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "hessian" => {
            if args.len() == 2 {
                if let Expr::List(vars) = &args[1] {
                    let n = vars.len();
                    let rows: Vec<Vec<Expr>> = (0..n)
                        .map(|i| {
                            (0..n)
                                .map(|j| {
                                    if let (Expr::Sym(xi), Expr::Sym(xj)) = (&vars[i], &vars[j]) {
                                        let d1 = differentiate(&args[0], *xi, interner);
                                        let d2 = differentiate(&d1, *xj, interner);
                                        eval(&d2, env, interner)
                                    } else {
                                        Expr::zero()
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    Expr::Matrix(rows)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "solve" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (expr, Expr::Sym(var)) => ax_solve::solve(expr, *var, interner),
                    (Expr::List(equations), Expr::List(vars_expr)) => {
                        let vars = vars_expr
                            .iter()
                            .map(|expr| {
                                if let Expr::Sym(sym) = expr {
                                    Some(*sym)
                                } else {
                                    None
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(vars) = vars {
                            if let Some(solution) =
                                ax_solve::solve_linear_system(equations, &vars, interner)
                            {
                                Expr::List(solution.into_iter().map(|(_, value)| value).collect())
                            } else {
                                Expr::Call(f, args)
                            }
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "det" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => ax_linalg::determinant(rows, interner),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "inv" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => {
                        if let Some(inv) = ax_linalg::inverse(rows, interner) {
                            Expr::Matrix(inv)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "transpose" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => Expr::Matrix(ax_linalg::transpose(rows)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "trace_mat" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => ax_linalg::trace(rows),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "eigenvalues" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => {
                        if rows.len() == 2 && rows.iter().all(|row| row.len() == 2) {
                            Expr::List(ax_linalg::eigenvalues_2x2(rows, interner))
                        } else {
                            ax_linalg::eigenvalues_symbolic(rows, interner)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "hermitian_eigenvalues" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => match ax_qm::hermitian_eigenvalues_small(rows, interner) {
                        Ok(values) => Expr::List(values),
                        Err(_) => Expr::Sym(interner.get_or_intern(
                            "hermitian_eigenvalues expects a square Hermitian matrix of supported dimension",
                        )),
                    },
                    _ => Expr::Sym(interner.get_or_intern(
                        "hermitian_eigenvalues expects a square Hermitian matrix of supported dimension",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "hermitian_eigenprojectors" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => match ax_qm::hermitian_eigenprojectors_small(rows, interner) {
                        Ok(projectors) => Expr::List(
                            projectors.into_iter().map(Expr::Matrix).collect(),
                        ),
                        Err(_) => Expr::Sym(interner.get_or_intern(
                            "hermitian_eigenprojectors expects a square Hermitian matrix of supported dimension with nondegenerate spectrum",
                        )),
                    },
                    _ => Expr::Sym(interner.get_or_intern(
                        "hermitian_eigenprojectors expects a square Hermitian matrix of supported dimension with nondegenerate spectrum",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "matmul" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_linalg::mat_mul(a, b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "identity" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Int(n) => {
                        if let Some(dim) = num_traits::ToPrimitive::to_usize(n) {
                            Expr::Matrix(ax_linalg::identity(dim))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "tensor_product" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_linalg::tensor_product(a, b))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "pauli_x" | "sigma_x" => {
            if args.is_empty() {
                Expr::Matrix(ax_qm::pauli_x(interner))
            } else {
                Expr::Call(f, args)
            }
        }
        "pauli_y" | "sigma_y" => {
            if args.is_empty() {
                Expr::Matrix(ax_qm::pauli_y(interner))
            } else {
                Expr::Call(f, args)
            }
        }
        "pauli_z" | "sigma_z" => {
            if args.is_empty() {
                Expr::Matrix(ax_qm::pauli_z(interner))
            } else {
                Expr::Call(f, args)
            }
        }
        "gamma" => {
            if args.is_empty() {
                Expr::List(
                    ax_qm::gamma_matrices_dirac(interner)
                        .into_iter()
                        .map(Expr::Matrix)
                        .collect(),
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "gamma5" => {
            if args.is_empty() {
                Expr::Matrix(ax_qm::gamma5(interner))
            } else {
                Expr::Call(f, args)
            }
        }
        "commutator" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_qm::commutator(a, b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "anticommutator" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(a), Expr::Matrix(b)) => {
                        Expr::Matrix(ax_qm::anticommutator(a, b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ket" => match args.as_slice() {
            [Expr::Int(n)] => {
                if let Some(index) = n.to_usize() {
                    Expr::List(ax_qm::ket(index, 2))
                } else {
                    Expr::Call(f, args)
                }
            }
            [Expr::Int(n), Expr::Int(d)] => match (n.to_usize(), d.to_usize()) {
                (Some(index), Some(dim)) => Expr::List(ax_qm::ket(index, dim)),
                _ => Expr::Call(f, args),
            },
            _ => Expr::Call(f, args),
        },
        "bra" => match args.as_slice() {
            [Expr::Int(n)] => {
                if let Some(index) = n.to_usize() {
                    Expr::List(ax_qm::bra(index, 2))
                } else {
                    Expr::Call(f, args)
                }
            }
            [Expr::Int(n), Expr::Int(d)] => match (n.to_usize(), d.to_usize()) {
                (Some(index), Some(dim)) => Expr::List(ax_qm::bra(index, dim)),
                _ => Expr::Call(f, args),
            },
            _ => Expr::Call(f, args),
        },
        "braket" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::List(a), Expr::List(b)) => ax_qm::braket(a, b),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "outer" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::List(a), Expr::List(b)) => Expr::Matrix(ax_qm::outer(a, b)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "density" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::List(state) => Expr::Matrix(ax_qm::density_matrix(state)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "partial_trace" => {
            if args.len() == 4 {
                match (&args[0], &args[1], &args[2], &args[3]) {
                    (Expr::Matrix(rho), Expr::Int(dim_a), Expr::Int(dim_b), Expr::Sym(which)) => {
                        match (dim_a.to_usize(), dim_b.to_usize(), interner.resolve(*which)) {
                            (Some(dim_a), Some(dim_b), "A") => ax_qm::try_partial_trace(
                                rho,
                                ax_qm::BipartiteDims { dim_a, dim_b },
                                ax_qm::PartialTraceTarget::A,
                            )
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args)),
                            (Some(dim_a), Some(dim_b), "B") => ax_qm::try_partial_trace(
                                rho,
                                ax_qm::BipartiteDims { dim_a, dim_b },
                                ax_qm::PartialTraceTarget::B,
                            )
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args)),
                            _ => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "partial_trace_factor" => {
            if args.len() == 3 {
                match (&args[0], usize_list_from_expr(&args[1]), &args[2]) {
                    (Expr::Matrix(rho), Some(dims), Expr::Int(factor_index)) => factor_index
                        .to_usize()
                        .and_then(|factor_index| {
                            ax_qm::try_partial_trace_factor(rho, &dims, factor_index).ok()
                        })
                        .map(Expr::Matrix)
                        .unwrap_or_else(|| Expr::Call(f, args)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "partial_transpose_factor" => {
            if args.len() == 3 {
                match (&args[0], usize_list_from_expr(&args[1]), &args[2]) {
                    (Expr::Matrix(rho), Some(dims), Expr::Int(factor_index)) => factor_index
                        .to_usize()
                        .and_then(|factor_index| {
                            ax_qm::try_partial_transpose_factor(rho, &dims, factor_index).ok()
                        })
                        .map(Expr::Matrix)
                        .unwrap_or_else(|| Expr::Call(f, args)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "permute_subsystems" => {
            if args.len() == 3 {
                match (
                    &args[0],
                    usize_list_from_expr(&args[1]),
                    usize_list_from_expr(&args[2]),
                ) {
                    (Expr::Matrix(rho), Some(dims), Some(permutation)) => {
                        ax_qm::try_permute_subsystems(rho, &dims, &permutation)
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "partial_trace_space" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (Expr::Matrix(rho), Expr::Sym(composite_space), Expr::Sym(factor_space)) => {
                        hilbert_space_metadata_of_symbol(env, *composite_space)
                            .and_then(|metadata| {
                                unique_factor_index(&metadata, *factor_space).ok().and_then(
                                    |factor_index| {
                                        let dims = metadata.factor_dimensions();
                                        ax_qm::try_partial_trace_factor(rho, &dims, factor_index)
                                            .ok()
                                    },
                                )
                            })
                            .map(Expr::Matrix)
                            .unwrap_or_else(|| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "creation" => {
            if args.len() == 1 {
                creation_expr(args[0].clone(), interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "annihilation" => {
            if args.len() == 1 {
                annihilation_expr(args[0].clone(), interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "number_state" => {
            if args.len() == 2 {
                match usize_from_expr(&args[1]) {
                    Some(n) => number_state_expr(args[0].clone(), n, interner),
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "vacuum" => {
            if args.len() == 1 {
                vacuum_expr(args[0].clone(), interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "number_operator" => {
            if args.len() == 1 {
                Expr::mul(vec![
                    creation_expr(args[0].clone(), interner),
                    annihilation_expr(args[0].clone(), interner),
                ])
            } else {
                Expr::Call(f, args)
            }
        }
        "hamiltonian_ho" => match args.as_slice() {
            [mode] => Expr::mul(vec![Expr::add(vec![
                Expr::mul(vec![
                    creation_expr(mode.clone(), interner),
                    annihilation_expr(mode.clone(), interner),
                ]),
                Expr::Rational(BigRational::new(1.into(), 2.into())),
            ])]),
            [mode, omega] => Expr::mul(vec![
                omega.clone(),
                Expr::add(vec![
                    Expr::mul(vec![
                        creation_expr(mode.clone(), interner),
                        annihilation_expr(mode.clone(), interner),
                    ]),
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                ]),
            ]),
            [mode, hbar, omega] => Expr::mul(vec![
                hbar.clone(),
                omega.clone(),
                Expr::add(vec![
                    Expr::mul(vec![
                        creation_expr(mode.clone(), interner),
                        annihilation_expr(mode.clone(), interner),
                    ]),
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                ]),
            ]),
            _ => Expr::Call(f, args),
        },
        "apply_operator" => {
            if args.len() == 2 {
                apply_abstract_qm_operator(&args[0], &args[1], interner)
                    .map(|expr| eval(&expr, env, interner))
                    .unwrap_or_else(|| Expr::Call(f, args))
            } else {
                Expr::Call(f, args)
            }
        }
        "normal_order" => {
            if args.len() == 1 {
                ax_qm::normal_order(&args[0], &env.operators, &env.operator_statistics, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "wick" => {
            if args.len() == 1 {
                ax_qm::wick_expand(
                    &args[0],
                    &env.operators,
                    &env.operator_statistics,
                    &env.contractions,
                    interner,
                )
            } else {
                Expr::Call(f, args)
            }
        }
        "fierz" => {
            if args.len() == 1 {
                match ax_qm::try_fierz_auto_with_properties(
                    &args[0],
                    4,
                    &env.property_store,
                    interner,
                ) {
                    Ok(result) => result,
                    Err(_) => ax_qm::fierz_auto(&args[0], 4, interner),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "expand_diracbar" | "expand_bar" => {
            if args.len() == 1 {
                let full = ax_qm::expand_diracbar_full(&args[0], &env.property_store, interner);
                if full != args[0] {
                    full
                } else {
                    let diracbar_sym = find_tensor_property_sym(env, |prop| {
                        matches!(prop, ax_ir::TensorProperty::DiracBar)
                    })
                    .unwrap_or_else(|| interner.get_or_intern("bar"));
                    let gamma_sym = find_tensor_property_sym(env, |prop| {
                        matches!(prop, ax_ir::TensorProperty::GammaMatrixProp)
                    })
                    .unwrap_or_else(|| interner.get_or_intern("gamma"));
                    let metric_sym =
                        find_metric_sym(env).unwrap_or_else(|| interner.get_or_intern("g"));
                    ax_qm::expand_diracbar(&args[0], diracbar_sym, gamma_sym, metric_sym, interner)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "sort_spinors" | "diracbar_sort" => {
            if args.len() == 1 {
                ax_qm::sort_spinors(&args[0], &env.property_store, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "join_gamma" | "join_gammas_in_expr" => {
            if args.len() == 1 {
                let metric_sym =
                    find_metric_sym(env).unwrap_or_else(|| interner.get_or_intern("g"));
                let metric = Expr::Sym(metric_sym);
                match &args[0] {
                    Expr::Mul(factors) if factors.len() == 2 => ax_qm::join_gamma_full(
                        &factors[0],
                        &factors[1],
                        None,
                        true,
                        false,
                        &metric,
                        &env.property_store,
                        interner,
                    ),
                    _ => {
                        let gamma_sym = find_tensor_property_sym(env, |prop| {
                            matches!(prop, ax_ir::TensorProperty::GammaMatrixProp)
                        })
                        .unwrap_or_else(|| interner.get_or_intern("gamma"));
                        ax_qm::join_gammas_in_expr(&args[0], gamma_sym, metric_sym, interner)
                    }
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "split_gamma" => {
            if !args.is_empty() {
                let on_back = args
                    .get(1)
                    .map(|a| {
                        if let Expr::Sym(s) = a {
                            interner.resolve(*s) == "back"
                        } else {
                            false
                        }
                    })
                    .unwrap_or(true);
                let full =
                    ax_qm::split_gamma_full(&args[0], on_back, &env.property_store, interner);
                if full != args[0] {
                    full
                } else {
                    let gamma_sym = find_tensor_property_sym(env, |prop| {
                        matches!(prop, ax_ir::TensorProperty::GammaMatrixProp)
                    })
                    .unwrap_or_else(|| interner.get_or_intern("gamma"));
                    let metric_sym =
                        find_metric_sym(env).unwrap_or_else(|| interner.get_or_intern("g"));
                    ax_qm::split_gamma(&args[0], gamma_sym, metric_sym, on_back, interner)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "euler_lagrange" => {
            if args.len() == 4 {
                match (&args[0], &args[1], &args[2], &args[3]) {
                    (
                        lagrangian,
                        Expr::Sym(field),
                        Expr::List(field_derivs),
                        Expr::List(coords),
                    ) => {
                        let field_derivs = field_derivs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let coords = coords
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let (Some(field_derivs), Some(coords)) = (field_derivs, coords) {
                            ax_variational::functional_derivative(
                                lagrangian,
                                *field,
                                &field_derivs,
                                &coords,
                                interner,
                            )
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "functional_derivative" => {
            if args.len() == 4 {
                match (&args[0], &args[1], &args[2], &args[3]) {
                    (
                        lagrangian,
                        Expr::Sym(field),
                        Expr::List(field_derivs),
                        Expr::List(coords),
                    ) => {
                        let field_derivs = field_derivs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let coords = coords
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let (Some(field_derivs), Some(coords)) = (field_derivs, coords) {
                            ax_variational::functional_derivative(
                                lagrangian,
                                *field,
                                &field_derivs,
                                &coords,
                                interner,
                            )
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "euler_lagrange_system" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (lagrangian, fields_expr, Expr::List(coords)) => {
                        let field_rows: Vec<Expr> = match fields_expr {
                            Expr::List(fields) => fields.clone(),
                            Expr::Matrix(rows) => {
                                rows.iter().map(|row| Expr::List(row.clone())).collect()
                            }
                            _ => return Expr::Call(f, args),
                        };
                        let fields = field_rows
                            .iter()
                            .map(|entry| match entry {
                                Expr::List(pair) if pair.len() == 2 => match (&pair[0], &pair[1]) {
                                    (Expr::Sym(field), Expr::List(derivs)) => derivs
                                        .iter()
                                        .map(|expr| match expr {
                                            Expr::Sym(sym) => Some(*sym),
                                            _ => None,
                                        })
                                        .collect::<Option<Vec<_>>>()
                                        .map(|derivs| (*field, derivs)),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let coords = coords
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let (Some(fields), Some(coords)) = (fields, coords) {
                            Expr::List(ax_variational::euler_lagrange_system(
                                lagrangian, &fields, &coords, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "vary_action" | "vary" => {
            if args.len() == 5 {
                match (&args[0], &args[1], &args[2], &args[3], &args[4]) {
                    (
                        lagrangian,
                        Expr::Sym(field),
                        Expr::Sym(variation),
                        Expr::List(field_derivs),
                        Expr::List(variation_derivs),
                    ) => {
                        let field_derivs = field_derivs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let variation_derivs = variation_derivs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let (Some(field_derivs), Some(variation_derivs)) =
                            (field_derivs, variation_derivs)
                        {
                            ax_variational::vary_action(
                                lagrangian,
                                *field,
                                *variation,
                                &field_derivs,
                                &variation_derivs,
                                interner,
                            )
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "gamma_trace" => {
            let parse_indices = |expr: &Expr| -> Option<Vec<lasso::Spur>> {
                let Expr::List(items) = expr else {
                    return None;
                };
                items
                    .iter()
                    .map(|item| match item {
                        Expr::Sym(sym) => Some(*sym),
                        _ => None,
                    })
                    .collect()
            };
            match args.as_slice() {
                [indices_expr] => {
                    if let Some(indices) = parse_indices(indices_expr) {
                        let g = interner.get_or_intern("g");
                        ax_qm::gamma_trace_recursive(&indices, g, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                }
                [indices_expr, Expr::Sym(metric_sym)] => {
                    if let Some(indices) = parse_indices(indices_expr) {
                        ax_qm::gamma_trace_recursive(&indices, *metric_sym, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
        "gamma5_trace" => {
            let parse_indices = |expr: &Expr| -> Option<Vec<lasso::Spur>> {
                let Expr::List(items) = expr else {
                    return None;
                };
                items
                    .iter()
                    .map(|item| match item {
                        Expr::Sym(sym) => Some(*sym),
                        _ => None,
                    })
                    .collect()
            };
            match args.as_slice() {
                [indices_expr] => {
                    if let Some(indices) = parse_indices(indices_expr) {
                        let entries = std::iter::once(ax_qm::GammaEntry::Gamma5)
                            .chain(indices.into_iter().map(ax_qm::GammaEntry::Gamma))
                            .collect::<Vec<_>>();
                        let metric = ax_tensor::SymbolicMatrix::from_diagonal(vec![
                            Expr::Int((-1).into()),
                            Expr::one(),
                            Expr::one(),
                            Expr::one(),
                        ]);
                        ax_qm::gamma_trace(&entries, &metric, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
        "dsolve" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (equation, Expr::Sym(y), Expr::Sym(x)) => {
                        ax_ode::solve_ode(equation, *y, *x, interner)
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "first_order_form" => {
            if args.len() >= 3 {
                if let (Expr::Sym(dep), Expr::Sym(indep)) = (&args[1], &args[2]) {
                    let system = ax_ode::first_order_form(&args[0], *dep, *indep, interner);
                    Expr::List(
                        system
                            .into_iter()
                            .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                            .collect(),
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "rk4" => {
            let parse_steps = |expr: &Expr| -> Option<usize> {
                match expr {
                    Expr::Int(n) => n.to_usize(),
                    _ => None,
                }
            };
            match args.as_slice() {
                [expr, Expr::Sym(x), Expr::Sym(y), x0, y0, x_end] => {
                    match (to_f64(x0), to_f64(y0), to_f64(x_end)) {
                        (Some(x0), Some(y0), Some(x_end)) => Expr::List(
                            ax_ode::rk4(expr, *x, *y, x0, y0, x_end, 1000, interner)
                                .into_iter()
                                .map(|(xv, yv)| Expr::List(vec![Expr::Float(xv), Expr::Float(yv)]))
                                .collect(),
                        ),
                        _ => Expr::Call(f, args),
                    }
                }
                [expr, Expr::Sym(x), Expr::Sym(y), x0, y0, x_end, steps] => {
                    match (to_f64(x0), to_f64(y0), to_f64(x_end), parse_steps(steps)) {
                        (Some(x0), Some(y0), Some(x_end), Some(steps)) => Expr::List(
                            ax_ode::rk4(expr, *x, *y, x0, y0, x_end, steps, interner)
                                .into_iter()
                                .map(|(xv, yv)| Expr::List(vec![Expr::Float(xv), Expr::Float(yv)]))
                                .collect(),
                        ),
                        _ => Expr::Call(f, args),
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
        "classify_pde" => {
            if args.len() == 3 {
                let pde_type = ax_ode::classify_pde(&args[0], &args[1], &args[2], interner);
                let type_str = match pde_type {
                    ax_ode::PdeType::Elliptic => "elliptic",
                    ax_ode::PdeType::Parabolic => "parabolic",
                    ax_ode::PdeType::Hyperbolic => "hyperbolic",
                    ax_ode::PdeType::Unknown => "unknown",
                };
                Expr::Sym(interner.get_or_intern(type_str))
            } else {
                Expr::Call(f, args)
            }
        }
        "separate_variables" | "separation" => {
            if args.len() >= 3 {
                let pde_type = if let Expr::Sym(s) = &args[0] {
                    match interner.resolve(*s) {
                        "wave" | "hyperbolic" => ax_ode::PdeType::Hyperbolic,
                        "heat" | "parabolic" | "diffusion" => ax_ode::PdeType::Parabolic,
                        "laplace" | "elliptic" => ax_ode::PdeType::Elliptic,
                        _ => ax_ode::PdeType::Unknown,
                    }
                } else {
                    ax_ode::PdeType::Unknown
                };

                if let (Expr::Sym(x), Expr::Sym(t)) = (&args[1], &args[2]) {
                    let coeff = args.get(3).cloned().unwrap_or(Expr::Int(1.into()));
                    let sol = ax_ode::separate_variables(pde_type, *x, *t, &coeff, interner);
                    Expr::List(vec![sol.spatial, sol.temporal, sol.separation_constant])
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "plot" => {
            if args.len() == 4 {
                match (&args[1], to_f64(&args[2]), to_f64(&args[3])) {
                    (Expr::Sym(var), Some(x_min), Some(x_max)) => {
                        let svg = ax_plot::plot_2d(&args[0], *var, x_min, x_max, interner);
                        match std::fs::write("axioma_plot.svg", svg) {
                            Ok(()) => {
                                println!("Plot saved to axioma_plot.svg");
                                Expr::zero()
                            }
                            Err(_) => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "wedge_1_1" => {
            if args.len() == 2 {
                match (
                    ax_forms::one_form_from_expr(&args[0]),
                    ax_forms::one_form_from_expr(&args[1]),
                ) {
                    (Some(a), Some(b)) => {
                        ax_forms::form_to_expr(&ax_forms::wedge(&a, &b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "wedge" => {
            if args.len() == 2 {
                match (
                    ax_forms::form_from_expr(&args[0]),
                    ax_forms::form_from_expr(&args[1]),
                ) {
                    (Some(a), Some(b)) => {
                        let dim = a.dim.max(b.dim);
                        let a = ax_forms::resize_form(&a, dim);
                        let b = ax_forms::resize_form(&b, dim);
                        ax_forms::form_to_expr(&ax_forms::wedge(&a, &b, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "exterior_d" | "d" => {
            let coords = if args.len() == 2 {
                match &args[1] {
                    Expr::List(coords_exprs) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            Some(coords)
                        } else {
                            return Expr::Call(f, args);
                        }
                    }
                    _ => return Expr::Call(f, args),
                }
            } else if args.len() == 1 {
                let mut coords = env.coordinates.iter().copied().collect::<Vec<_>>();
                coords.sort_by_key(|sym| interner.resolve(*sym).to_string());
                (!coords.is_empty()).then_some(coords)
            } else {
                None
            };
            if let Some(coords) = coords {
                let field = &args[0];
                let form = ax_forms::form_from_expr(field)
                    .map(|form| ax_forms::resize_form(&form, coords.len()))
                    .unwrap_or_else(|| ax_forms::scalar_form(field, coords.len()));
                ax_forms::form_to_expr(&ax_forms::exterior_derivative(&form, &coords, interner))
            } else {
                Expr::Call(f, args)
            }
        }
        "hodge_star" => {
            if args.len() == 2 {
                match (&args[0], matrix_to_symbolic(&args[1])) {
                    (field, Some(metric)) => {
                        let metric = symbolic_to_forms_matrix(&metric);
                        let form = ax_forms::form_from_expr(field)
                            .map(|form| ax_forms::resize_form(&form, metric.dim))
                            .unwrap_or_else(|| ax_forms::scalar_form(field, metric.dim));
                        ax_forms::form_to_expr(&ax_forms::hodge_dual(&form, &metric, interner))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "codifferential" => {
            if args.len() == 3 {
                match (&args[0], matrix_to_symbolic(&args[1]), &args[2]) {
                    (field, Some(metric), Expr::List(coords_exprs)) => {
                        let metric = symbolic_to_forms_matrix(&metric);
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            let form = ax_forms::form_from_expr(field)
                                .map(|form| ax_forms::resize_form(&form, metric.dim))
                                .unwrap_or_else(|| ax_forms::scalar_form(field, metric.dim));
                            ax_forms::form_to_expr(&ax_forms::codifferential(
                                &form, &metric, &coords, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "interior_product" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::List(vector), field) => {
                        let form = ax_forms::form_from_expr(field)
                            .map(|form| ax_forms::resize_form(&form, vector.len()));
                        if let Some(form) = form {
                            ax_forms::form_to_expr(&ax_forms::interior_product(
                                vector, &form, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lie_derivative_form" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (field, Expr::List(vector), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let form = ax_forms::form_from_expr(field)
                            .map(|form| ax_forms::resize_form(&form, vector.len()));
                        if let (Some(coords), Some(form)) = (coords, form) {
                            ax_forms::form_to_expr(&ax_forms::lie_derivative_form(
                                vector, &form, &coords, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "diff" => {
            if args.len() == 2 {
                if let Expr::Sym(var_sym) = args[1] {
                    let diffed = differentiate(&args[0], var_sym, interner);
                    let diff_sym = interner.get_or_intern("diff");
                    if matches!(&diffed, Expr::Call(sym, inner_args) if *sym == diff_sym && inner_args == &args)
                    {
                        diffed
                    } else {
                        eval(&diffed, &Env::new(), interner)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "christoffel" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (metric_expr, Expr::List(coords_exprs)) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            let coords = coords_exprs
                                .iter()
                                .map(|expr| {
                                    if let Expr::Sym(sym) = expr {
                                        Some(*sym)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Option<Vec<_>>>();
                            if let Some(coords) = coords {
                                expr_3d_to_list(ax_tensor::christoffel_from_metric(
                                    &metric, &coords, interner,
                                ))
                            } else {
                                Expr::Call(f, args)
                            }
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "killing_equations" => {
            if args.len() == 2 || args.len() == 3 {
                match (expr_to_3d(&args[0]), &args[1], args.get(2)) {
                    (Some(gamma), Expr::List(coords_exprs), prefix_arg) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let field_prefix = match prefix_arg {
                            Some(Expr::Sym(sym)) => interner.resolve(*sym),
                            None => "xi",
                            Some(_) => return Expr::Call(f, args),
                        };
                        match coords {
                            Some(coords) => match ax_tensor::killing_equations(
                                &gamma,
                                &coords,
                                field_prefix,
                                interner,
                            ) {
                                Ok(system) => Expr::List(vec![
                                    Expr::List(system.covector_components),
                                    Expr::List(system.equations),
                                    Expr::List(
                                        system
                                            .slot_pairs
                                            .into_iter()
                                            .map(|(a, b)| {
                                                Expr::List(vec![
                                                    Expr::Int(a.into()),
                                                    Expr::Int(b.into()),
                                                ])
                                            })
                                            .collect(),
                                    ),
                                ]),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "adm_decompose" => {
            if args.len() == 3 {
                match (matrix_to_symbolic(&args[0]), &args[1], &args[2]) {
                    (Some(metric), Expr::List(coords_exprs), Expr::Int(time_coord)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let Some(time_coord) = time_coord.to_usize() else {
                            return Expr::Call(f, args);
                        };
                        match coords {
                            Some(coords) => match ax_tensor::adm_decompose(
                                &metric, &coords, time_coord, interner,
                            ) {
                                Ok(adm) => Expr::List(vec![
                                    adm.lapse,
                                    Expr::List(adm.shift_covector),
                                    Expr::List(adm.shift_vector),
                                    Expr::Matrix(adm.spatial_metric.data),
                                    Expr::Matrix(adm.spatial_inverse_metric.data),
                                    Expr::Matrix(adm.extrinsic_curvature),
                                    adm.hamiltonian_constraint,
                                    Expr::List(adm.momentum_constraints),
                                ]),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "contorsion_tensor" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), matrix_to_symbolic(&args[1])) {
                    (Some(torsion), Some(metric)) => {
                        match ax_tensor::contorsion_tensor(&torsion, &metric, interner) {
                            Ok(contorsion) => expr_3d_to_list(contorsion),
                            Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "connection_with_torsion" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), expr_to_3d(&args[1])) {
                    (Some(gamma), Some(contorsion)) => {
                        match ax_tensor::connection_with_torsion(&gamma, &contorsion, interner) {
                            Ok(connection) => expr_3d_to_list(connection),
                            Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "spin_connection" => {
            if args.len() == 3 {
                match (
                    matrix_to_symbolic(&args[0]),
                    matrix_to_symbolic(&args[1]),
                    &args[2],
                ) {
                    (Some(vielbein), Some(metric), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::spin_connection(
                                &vielbein, &metric, &coords, interner,
                            ) {
                                Ok(omega) => expr_3d_to_list(omega),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "first_cartan_structure" => {
            if args.len() == 3 {
                match (matrix_to_symbolic(&args[0]), expr_to_3d(&args[1]), &args[2]) {
                    (Some(vielbein), Some(omega), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::first_cartan_structure(
                                &vielbein, &omega, &coords, interner,
                            ) {
                                Ok(forms) => Expr::List(
                                    forms.iter().map(ax_forms::form_to_expr).collect::<Vec<_>>(),
                                ),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "second_cartan_structure" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), &args[1]) {
                    (Some(omega), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => {
                                match ax_tensor::second_cartan_structure(&omega, &coords, interner)
                                {
                                    Ok(forms) => Expr::Matrix(
                                        forms
                                            .iter()
                                            .map(|row| {
                                                row.iter()
                                                    .map(ax_forms::form_to_expr)
                                                    .collect::<Vec<_>>()
                                            })
                                            .collect::<Vec<_>>(),
                                    ),
                                    Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                                }
                            }
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conformal_transform_metric" => {
            if args.len() == 2 {
                match (matrix_to_symbolic(&args[0]), &args[1]) {
                    (Some(metric), omega) => Expr::Matrix(
                        ax_tensor::conformal_transform_metric(&metric, omega, interner).data,
                    ),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conformal_transform_inverse_metric" => {
            if args.len() == 2 {
                match (matrix_to_symbolic(&args[0]), &args[1]) {
                    (Some(metric), omega) => Expr::Matrix(
                        ax_tensor::conformal_transform_inverse_metric(&metric, omega, interner)
                            .data,
                    ),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conformal_transform_christoffel" => {
            if args.len() == 4 {
                match (
                    expr_to_3d(&args[0]),
                    matrix_to_symbolic(&args[1]),
                    &args[2],
                    &args[3],
                ) {
                    (Some(gamma), Some(metric), omega, Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::conformal_transform_christoffel(
                                &gamma, &metric, omega, &coords, interner,
                            ) {
                                Ok(transformed) => expr_3d_to_list(transformed),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conformal_transform_ricci" => {
            if args.len() == 5 {
                match (
                    &args[0],
                    &args[1],
                    matrix_to_symbolic(&args[2]),
                    &args[3],
                    &args[4],
                ) {
                    (
                        Expr::Matrix(ricci),
                        scalar,
                        Some(metric),
                        omega,
                        Expr::List(coords_exprs),
                    ) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::conformal_transform_ricci(
                                ricci, scalar, &metric, omega, &coords, interner,
                            ) {
                                Ok(transformed) => Expr::Matrix(transformed),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conformal_transform_scalar" => {
            if args.len() == 4 {
                match (&args[0], matrix_to_symbolic(&args[1]), &args[2], &args[3]) {
                    (scalar, Some(metric), omega, Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::conformal_transform_scalar(
                                scalar, &metric, omega, &coords, interner,
                            ) {
                                Ok(transformed) => transformed,
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "null_tetrad" | "null_tetrad_from_metric" => {
            if args.len() == 2 {
                match (matrix_to_symbolic(&args[0]), &args[1]) {
                    (Some(metric), Expr::List(coords_exprs)) => {
                        let metric = simplify_symbolic_matrix(&metric, env, interner);
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => {
                                match ax_tensor::null_tetrad_from_metric(&metric, &coords, interner)
                                {
                                    Ok(tetrad) => null_tetrad_to_expr(tetrad),
                                    Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                                }
                            }
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "verify_null_tetrad" => {
            if args.len() == 2 {
                match (expr_to_null_tetrad(&args[0]), matrix_to_symbolic(&args[1])) {
                    (Some(tetrad), Some(metric)) => {
                        match ax_tensor::verify_null_tetrad(&tetrad, &metric, interner) {
                            Ok(()) => Expr::Sym(interner.get_or_intern("ok")),
                            Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "spin_coefficients" => {
            if args.len() == 4 {
                match (
                    expr_to_null_tetrad(&args[0]),
                    expr_to_3d(&args[1]),
                    matrix_to_symbolic(&args[2]),
                    &args[3],
                ) {
                    (Some(tetrad), Some(gamma), Some(metric), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::spin_coefficients(
                                &tetrad, &gamma, &metric, &coords, interner,
                            ) {
                                Ok(coeffs) => spin_coefficients_to_expr(coeffs),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "riemann" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), &args[1]) {
                    (Some(gamma), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| {
                                if let Expr::Sym(sym) = expr {
                                    Some(*sym)
                                } else {
                                    None
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            expr_4d_to_list(ax_tensor::riemann_from_christoffel(
                                &gamma,
                                &coords,
                                interner,
                                &env.convention,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ricci" => {
            if args.len() == 1 {
                if let Some(riemann) = expr_to_4d(&args[0]) {
                    let n = riemann.len();
                    aggressive_eval_simplify(
                        &Expr::Matrix(ax_tensor::ricci_from_riemann(
                            &riemann,
                            n,
                            interner,
                            &env.convention,
                        )),
                        env,
                        interner,
                    )
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "ricci_scalar" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Matrix(ricci), ginv_expr) => {
                        if let Some(ginv) = matrix_to_symbolic(ginv_expr) {
                            ax_tensor::ricci_scalar(ricci, &ginv, interner)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "einstein" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (Expr::Matrix(ricci), scalar, metric_expr) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            Expr::Matrix(ax_tensor::einstein_tensor(
                                ricci, scalar, &metric, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "weyl_from_curvature" | "weyl_from_riemann" => {
            if args.len() == 4 {
                match (
                    expr_to_4d(&args[0]),
                    &args[1],
                    &args[2],
                    matrix_to_symbolic(&args[3]),
                ) {
                    (Some(riemann), Expr::Matrix(ricci), scalar, Some(metric)) => {
                        match ax_tensor::weyl_from_curvature(
                            &riemann, ricci, scalar, &metric, interner,
                        ) {
                            Ok(weyl) => expr_4d_to_list(weyl),
                            Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "weyl_scalars" => {
            if args.len() == 3 {
                match (
                    expr_to_4d(&args[0]),
                    expr_to_null_tetrad(&args[1]),
                    matrix_to_symbolic(&args[2]),
                ) {
                    (Some(weyl), Some(tetrad), Some(metric)) => {
                        match ax_tensor::weyl_scalars(&weyl, &tetrad, &metric, interner) {
                            Ok(scalars) => weyl_scalars_to_expr(scalars),
                            Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "petrov_classify" => {
            if args.len() == 1 {
                match expr_to_weyl_scalars(&args[0]) {
                    Some(scalars) => match ax_tensor::petrov_classify(&scalars, interner) {
                        Ok(kind) => {
                            let name = match kind {
                                ax_tensor::PetrovType::I => "I",
                                ax_tensor::PetrovType::II => "II",
                                ax_tensor::PetrovType::D => "D",
                                ax_tensor::PetrovType::III => "III",
                                ax_tensor::PetrovType::N => "N",
                                ax_tensor::PetrovType::O => "O",
                            };
                            Expr::Sym(interner.get_or_intern(name))
                        }
                        Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                    },
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "cotton_from_curvature" => {
            if args.len() == 5 {
                match (
                    &args[0],
                    &args[1],
                    expr_to_3d(&args[2]),
                    matrix_to_symbolic(&args[3]),
                    &args[4],
                ) {
                    (
                        Expr::Matrix(ricci),
                        scalar,
                        Some(gamma),
                        Some(metric),
                        Expr::List(coords_exprs),
                    ) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::cotton_from_curvature(
                                ricci, scalar, &gamma, &metric, &coords, interner,
                            ) {
                                Ok(cotton) => expr_3d_to_list(cotton),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bach_from_curvature" => {
            if args.len() == 5 {
                match (
                    expr_to_4d(&args[0]),
                    &args[1],
                    expr_to_3d(&args[2]),
                    matrix_to_symbolic(&args[3]),
                    &args[4],
                ) {
                    (
                        Some(weyl),
                        Expr::Matrix(ricci),
                        Some(gamma),
                        Some(metric),
                        Expr::List(coords_exprs),
                    ) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        match coords {
                            Some(coords) => match ax_tensor::bach_from_curvature(
                                &weyl, ricci, &gamma, &metric, &coords, interner,
                            ) {
                                Ok(bach) => Expr::Matrix(bach),
                                Err(err) => Expr::Sym(interner.get_or_intern(&err.to_string())),
                            },
                            None => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "kretschner" => {
            if args.len() == 2 {
                match (expr_to_4d(&args[0]), &args[1]) {
                    (Some(riemann), metric_expr) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            ax_tensor::kretschner_scalar(&riemann, &metric, interner)
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "kretschmann_scalar_diagonal_approx" => {
            if args.len() == 2 {
                match (expr_to_4d(&args[0]), &args[1]) {
                    (Some(riemann), metric_expr) => {
                        if let Some(metric) = matrix_to_symbolic(metric_expr) {
                            ax_tensor::kretschmann_scalar_diagonal_approx(
                                &riemann, &metric, interner,
                            )
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "covariant_diff" => {
            if args.len() == 4 {
                match (&args[0], expr_to_3d(&args[1]), &args[2], &args[3]) {
                    (
                        Expr::List(v),
                        Some(gamma),
                        Expr::Int(coord_index),
                        Expr::List(coords_exprs),
                    ) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let coord_index = coord_index.to_usize();
                        match (coords, coord_index) {
                            (Some(coords), Some(coord_index)) if coord_index < coords.len() => {
                                Expr::List(ax_tensor::covariant_derivative_vector(
                                    v,
                                    &gamma,
                                    coord_index,
                                    &coords,
                                    interner,
                                ))
                            }
                            _ => Expr::Call(f, args),
                        }
                    }
                    (
                        Expr::Matrix(t),
                        Some(gamma),
                        Expr::Int(coord_index),
                        Expr::List(coords_exprs),
                    ) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        let coord_index = coord_index.to_usize();
                        match (coords, coord_index) {
                            (Some(coords), Some(coord_index)) if coord_index < coords.len() => {
                                Expr::Matrix(ax_tensor::covariant_derivative_tensor2(
                                    t,
                                    &gamma,
                                    coord_index,
                                    &coords,
                                    interner,
                                ))
                            }
                            _ => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "geodesic" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), &args[1]) {
                    (Some(gamma), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            Expr::List(ax_tensor::geodesic_equations(&gamma, &coords, interner))
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lie_derivative" => {
            if args.len() == 3 {
                match (&args[0], &args[1], &args[2]) {
                    (field, Expr::List(v), Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            match field {
                                Expr::List(w) => Expr::List(ax_tensor::lie_derivative_vector(
                                    w, v, &coords, interner,
                                )),
                                _ => ax_tensor::lie_derivative_scalar(field, v, &coords, interner),
                            }
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "metric" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Call(diag_f, diag_args) if interner.resolve(*diag_f) == "diag" => {
                        symbolic_to_matrix(&ax_tensor::SymbolicMatrix::from_diagonal(
                            diag_args.clone(),
                        ))
                    }
                    Expr::Matrix(rows) => {
                        let dim = rows.len();
                        if rows.iter().all(|row| row.len() == dim) {
                            Expr::Matrix(rows.clone())
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "vielbein" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => {
                        let dim = rows.len();
                        if rows.iter().all(|row| row.len() == dim) {
                            Expr::Matrix(rows.clone())
                        } else {
                            Expr::Call(f, args)
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "inv_vielbein" | "inverse_vielbein" => {
            if args.len() == 1 {
                if let Some(e) = matrix_to_symbolic(&args[0]) {
                    symbolic_to_matrix(&ax_tensor::inverse_vielbein(&e, interner))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "metric_from_vielbein" => {
            if args.len() == 2 {
                if let (Some(e), Some(eta)) =
                    (matrix_to_symbolic(&args[0]), matrix_to_symbolic(&args[1]))
                {
                    symbolic_to_matrix(&ax_tensor::metric_from_vielbein(&e, &eta, interner))
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "vielbein_from_metric_diagonal" => {
            if args.len() == 2 {
                if let Some(g) = matrix_to_symbolic(&args[0]) {
                    if let Expr::Sym(sig) = &args[1] {
                        if let Some(signature) = parse_metric_signature(interner.resolve(*sig)) {
                            symbolic_to_matrix(&ax_tensor::vielbein_from_metric_diagonal(
                                &g, signature, interner,
                            ))
                        } else {
                            Expr::Call(f, args)
                        }
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "diag" => symbolic_to_matrix(&ax_tensor::SymbolicMatrix::from_diagonal(args)),
        "integrate" => {
            if args.len() == 2 {
                if let Expr::Sym(var_sym) = args[1] {
                    let integrated = integrate::integrate(&args[0], var_sym, interner);
                    eval(&integrated, &Env::new(), interner)
                } else {
                    Expr::Call(f, args)
                }
            } else if args.len() == 4 {
                if let Expr::Sym(var_sym) = args[1] {
                    let integrated = integrate::integrate(&args[0], var_sym, interner);
                    let integrate_sym = interner.get_or_intern("integrate");
                    if matches!(&integrated, Expr::Call(sym, _) if *sym == integrate_sym) {
                        Expr::Call(f, args)
                    } else {
                        let mut hi_env = Env::new();
                        hi_env.bindings.insert(var_sym, args[3].clone());
                        let hi_val = eval(&integrated, &hi_env, interner);

                        let mut lo_env = Env::new();
                        lo_env.bindings.insert(var_sym, args[2].clone());
                        let lo_val = eval(&integrated, &lo_env, interner);

                        eval(
                            &Expr::add(vec![hi_val, Expr::neg(lo_val)]),
                            &Env::new(),
                            interner,
                        )
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "double_integral" | "dblint" => {
            if args.len() == 3 {
                if let (Expr::Sym(x), Expr::Sym(y)) = (&args[1], &args[2]) {
                    let inner = integrate::integrate(&args[0], *x, interner);
                    let outer = integrate::integrate(&inner, *y, interner);
                    eval(&outer, env, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "triple_integral" | "tplint" => {
            if args.len() == 4 {
                if let (Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)) = (&args[1], &args[2], &args[3]) {
                    let i1 = integrate::integrate(&args[0], *x, interner);
                    let i2 = integrate::integrate(&i1, *y, interner);
                    let i3 = integrate::integrate(&i2, *z, interner);
                    eval(&i3, env, interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "definite_integral" | "defint" => {
            if args.len() == 4 {
                if let Expr::Sym(var_sym) = &args[1] {
                    let antideriv = integrate::integrate(&args[0], *var_sym, interner);
                    if is_unevaluated_integrate_check(&antideriv, interner) {
                        Expr::Call(f, args)
                    } else {
                        let at_b = symbolic_substitute(
                            &antideriv,
                            &Expr::Sym(*var_sym),
                            &args[3],
                            interner,
                        );
                        let at_a = symbolic_substitute(
                            &antideriv,
                            &Expr::Sym(*var_sym),
                            &args[2],
                            interner,
                        );
                        eval(
                            &Expr::add(vec![
                                eval(&at_b, env, interner),
                                Expr::neg(eval(&at_a, env, interner)),
                            ]),
                            env,
                            interner,
                        )
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "limit" => {
            if args.len() == 3 {
                if let Expr::Sym(var_sym) = args[1] {
                    limits::limit(&args[0], var_sym, &args[2], interner)
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "series" => {
            if args.len() == 4 {
                if let (Expr::Sym(var_sym), Expr::Int(order)) = (&args[1], &args[3]) {
                    if let Some(order) = order.to_usize() {
                        series::taylor_series(&args[0], *var_sym, &args[2], order, interner)
                    } else {
                        Expr::Call(f, args)
                    }
                } else {
                    Expr::Call(f, args)
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "identity_channel" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Int(dim) => dim
                        .to_usize()
                        .map(ax_qm::identity_channel)
                        .map(expr_3d_to_list)
                        .unwrap_or_else(|| Expr::Call(f, args)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "basis_projector" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Int(index), Expr::Int(dim)) => {
                        match (index.to_usize(), dim.to_usize()) {
                            (Some(index), Some(dim)) => ax_qm::basis_projector(index, dim)
                                .map(Expr::Matrix)
                                .unwrap_or_else(|_| Expr::Call(f, args)),
                            _ => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "measurement_probabilities" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), expr_to_matrix(&args[1])) {
                    (Some(projectors), Some(rho)) => {
                        ax_qm::measurement_probabilities(&projectors, &rho)
                            .map(Expr::List)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "expectation_value" => {
            if args.len() == 2 {
                match (expr_to_matrix(&args[0]), expr_to_matrix(&args[1])) {
                    (Some(operator), Some(rho)) => ax_qm::expectation_value(&operator, &rho)
                        .unwrap_or_else(|_| Expr::Call(f, args)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "variance" => {
            if args.len() == 2 {
                match (expr_to_matrix(&args[0]), expr_to_matrix(&args[1])) {
                    (Some(operator), Some(rho)) => {
                        ax_qm::variance(&operator, &rho).unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "purity" => {
            if args.len() == 1 {
                match expr_to_matrix(&args[0]) {
                    Some(rho) => ax_qm::purity(&rho).unwrap_or_else(|_| Expr::Call(f, args)),
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "linear_entropy" => {
            if args.len() == 1 {
                match expr_to_matrix(&args[0]) {
                    Some(rho) => {
                        ax_qm::linear_entropy(&rho).unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "participation_ratio" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => match ax_qm::participation_ratio(rows, interner) {
                        Ok(value) => value,
                        Err(_) => Expr::Sym(interner.get_or_intern(
                            "participation_ratio expects a square density matrix",
                        )),
                    },
                    _ => Expr::Sym(interner.get_or_intern(
                        "participation_ratio expects a square density matrix",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "renyi2_entropy" => {
            if args.len() == 1 {
                match expr_to_matrix(&args[0]) {
                    Some(rho) => ax_qm::renyi2_entropy(&rho, interner)
                        .unwrap_or_else(|_| Expr::Call(f, args)),
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "renyi2_entropy_factor" => {
            if args.len() == 3 {
                match (&args[0], usize_list_from_expr(&args[1]), &args[2]) {
                    (Expr::Matrix(rho), Some(dims), Expr::Int(kept_factor)) => {
                        match kept_factor.to_usize() {
                            Some(kept_factor) => ax_qm::renyi2_entropy_factor(
                                rho,
                                &dims,
                                kept_factor,
                                interner,
                            )
                            .unwrap_or_else(|_| {
                                Expr::Sym(interner.get_or_intern(
                                    "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions",
                                ))
                            }),
                            None => Expr::Sym(interner.get_or_intern(
                                "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions",
                            )),
                        }
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "von_neumann_entropy" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::Matrix(rows) => match ax_qm::von_neumann_entropy(rows, interner) {
                        Ok(value) => value,
                        Err(_) => Expr::Sym(interner.get_or_intern(
                            "von_neumann_entropy expects a supported square Hermitian density matrix",
                        )),
                    },
                    _ => Expr::Sym(interner.get_or_intern(
                        "von_neumann_entropy expects a supported square Hermitian density matrix",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "mutual_information" => {
            if args.len() == 3 {
                match (
                    expr_to_matrix(&args[0]),
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Some(rho_ab), Some(dim_a), Some(dim_b)) => {
                        ax_qm::von_neumann_mutual_information_bipartite(
                            &rho_ab,
                            dim_a,
                            dim_b,
                            interner,
                        )
                        .unwrap_or_else(|_| {
                            Expr::Sym(interner.get_or_intern(
                                "mutual_information expects a bipartite density matrix of dimension dim_a * dim_b",
                            ))
                        })
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "mutual_information expects a bipartite density matrix of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "conditional_entropy" => {
            if args.len() == 3 {
                match (
                    expr_to_matrix(&args[0]),
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Some(rho_ab), Some(dim_a), Some(dim_b)) => {
                        ax_qm::conditional_entropy_b_given_a(&rho_ab, dim_a, dim_b, interner)
                            .unwrap_or_else(|_| {
                                Expr::Sym(interner.get_or_intern(
                                    "conditional_entropy expects a bipartite density matrix of dimension dim_a * dim_b",
                                ))
                            })
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "conditional_entropy expects a bipartite density matrix of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "entanglement_spectrum" => {
            if args.len() == 3 {
                match (
                    &args[0],
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Expr::List(state), Some(dim_a), Some(dim_b)) => {
                        match ax_qm::entanglement_spectrum_from_state(
                            state, dim_a, dim_b, interner,
                        ) {
                            Ok(values) => Expr::List(values),
                            Err(_) => Expr::Sym(interner.get_or_intern(
                                "entanglement_spectrum expects a bipartite state vector or density matrix of dimension dim_a * dim_b",
                            )),
                        }
                    }
                    (Expr::Matrix(rho), Some(dim_a), Some(dim_b)) => {
                        match ax_qm::entanglement_spectrum_from_density(
                            rho, dim_a, dim_b, 'A', interner,
                        ) {
                            Ok(values) => Expr::List(values),
                            Err(_) => Expr::Sym(interner.get_or_intern(
                                "entanglement_spectrum expects a bipartite state vector or density matrix of dimension dim_a * dim_b",
                            )),
                        }
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "entanglement_spectrum expects a bipartite state vector or density matrix of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "schmidt_coefficients" => {
            if args.len() == 3 {
                match (
                    &args[0],
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Expr::List(state), Some(dim_a), Some(dim_b)) => {
                        match ax_qm::schmidt_coefficients_from_state(state, dim_a, dim_b, interner)
                        {
                            Ok(values) => Expr::List(values),
                            Err(_) => Expr::Sym(interner.get_or_intern(
                                "schmidt_coefficients expects a bipartite pure-state vector of dimension dim_a * dim_b",
                            )),
                        }
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "schmidt_coefficients expects a bipartite pure-state vector of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "negativity" => {
            if args.len() == 3 {
                match (
                    &args[0],
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Expr::Matrix(rho_ab), Some(dim_a), Some(dim_b)) => {
                        match ax_qm::negativity_bipartite(rho_ab, dim_a, dim_b, 1, interner) {
                            Ok(value) => value,
                            Err(_) => Expr::Sym(interner.get_or_intern(
                                "negativity expects a bipartite density matrix of dimension dim_a * dim_b",
                            )),
                        }
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "negativity expects a bipartite density matrix of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "logarithmic_negativity" => {
            if args.len() == 3 {
                match (
                    &args[0],
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Expr::Matrix(rho_ab), Some(dim_a), Some(dim_b)) => {
                        match ax_qm::logarithmic_negativity_bipartite(
                            rho_ab, dim_a, dim_b, 1, interner,
                        ) {
                            Ok(value) => value,
                            Err(_) => Expr::Sym(interner.get_or_intern(
                                "logarithmic_negativity expects a bipartite density matrix of dimension dim_a * dim_b",
                            )),
                        }
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "logarithmic_negativity expects a bipartite density matrix of dimension dim_a * dim_b",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "renyi2_mutual_information" => {
            if args.len() == 3 {
                match (
                    expr_to_matrix(&args[0]),
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                ) {
                    (Some(rho_ab), Some(dim_a), Some(dim_b)) => {
                        ax_qm::renyi2_mutual_information_bipartite(&rho_ab, dim_a, dim_b, interner)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "renyi2_tripartite_information" => {
            if args.len() == 4 {
                match (
                    expr_to_matrix(&args[0]),
                    usize_from_expr(&args[1]),
                    usize_from_expr(&args[2]),
                    usize_from_expr(&args[3]),
                ) {
                    (Some(rho_abc), Some(dim_a), Some(dim_b), Some(dim_c)) => {
                        ax_qm::renyi2_tripartite_information(
                            &rho_abc,
                            [dim_a, dim_b, dim_c],
                            interner,
                        )
                        .unwrap_or_else(|_| {
                            Expr::Sym(interner.get_or_intern(
                                "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c",
                            ))
                        })
                    }
                    _ => Expr::Sym(interner.get_or_intern(
                        "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c",
                    )),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "bloch_vector" => {
            if args.len() == 1 {
                match expr_to_matrix(&args[0]) {
                    Some(rho) => ax_qm::bloch_vector(&rho)
                        .map(|components| Expr::List(components.into_iter().collect()))
                        .unwrap_or_else(|_| Expr::Call(f, args)),
                    None => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "qubit_density_from_bloch" => {
            if args.len() == 1 {
                match &args[0] {
                    Expr::List(items) if items.len() == 3 => {
                        Expr::Matrix(ax_qm::qubit_density_from_bloch([
                            items[0].clone(),
                            items[1].clone(),
                            items[2].clone(),
                        ]))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "post_measurement_state" => {
            if args.len() == 3 {
                match (&args[0], expr_to_matrix(&args[1]), &args[2]) {
                    (Expr::Matrix(projector), Some(rho), Expr::Int(outcome_index)) => outcome_index
                        .to_usize()
                        .and_then(|outcome_index| {
                            ax_qm::post_measurement_state(projector, &rho, outcome_index).ok()
                        })
                        .map(Expr::Matrix)
                        .unwrap_or_else(|| Expr::Call(f, args)),
                    (Expr::List(rows), Some(rho), Expr::Int(outcome_index)) => {
                        let projector = expr_to_matrix(&Expr::List(rows.clone()));
                        match (projector, outcome_index.to_usize()) {
                            (Some(projector), Some(outcome_index)) => {
                                ax_qm::post_measurement_state(&projector, &rho, outcome_index)
                                    .map(Expr::Matrix)
                                    .unwrap_or_else(|_| Expr::Call(f, args))
                            }
                            _ => Expr::Call(f, args),
                        }
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "apply_channel" => {
            if args.len() == 2 {
                match (expr_to_3d(&args[0]), expr_to_matrix(&args[1])) {
                    (Some(kraus), Some(rho)) => ax_qm::apply_kraus_channel(&kraus, &rho)
                        .map(Expr::Matrix)
                        .unwrap_or_else(|_| Expr::Call(f, args)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lindblad_rhs" => {
            if args.len() == 3 {
                match (
                    expr_to_matrix(&args[0]),
                    expr_to_matrix(&args[1]),
                    expr_to_3d(&args[2]),
                ) {
                    (Some(h), Some(rho), Some(jump_ops)) => {
                        ax_qm::lindblad_rhs(&h, &rho, &jump_ops, interner)
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lindblad_euler_step" => {
            if args.len() == 4 {
                match (
                    expr_to_matrix(&args[0]),
                    expr_to_matrix(&args[1]),
                    expr_to_3d(&args[2]),
                ) {
                    (Some(h), Some(rho), Some(jump_ops)) => {
                        ax_ode::lindblad_euler_step(&h, &rho, &jump_ops, &args[3], interner)
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lindblad_rk4_step" => {
            if args.len() == 4 {
                match (
                    expr_to_matrix(&args[0]),
                    expr_to_matrix(&args[1]),
                    expr_to_3d(&args[2]),
                ) {
                    (Some(h), Some(rho), Some(jump_ops)) => {
                        ax_ode::lindblad_rk4_step(&h, &rho, &jump_ops, &args[3], interner)
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "lindblad_steady_state" => {
            if args.len() == 2 {
                match (expr_to_matrix(&args[0]), expr_to_3d(&args[1])) {
                    (Some(h), Some(jump_ops)) => {
                        ax_solve::lindblad_steady_state_linear(&h, &jump_ops, interner)
                            .map(Expr::Matrix)
                            .unwrap_or_else(|_| Expr::Call(f, args))
                    }
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        _ => Expr::Call(f, args),
    }
}

/// Focus on sub-expressions matching a pattern.
/// Returns (matching_terms, remainder) from the top-level sum.
pub fn zoom(expr: &Expr, pattern: &Expr, _interner: &ax_ir::Interner) -> (Expr, Expr) {
    match expr {
        Expr::Add(terms) => {
            let mut matching = Vec::new();
            let mut non_matching = Vec::new();

            for term in terms {
                if structurally_contains(term, pattern) {
                    matching.push(term.clone());
                } else {
                    non_matching.push(term.clone());
                }
            }

            let zoomed = match matching.len() {
                0 => Expr::zero(),
                1 => matching.into_iter().next().unwrap(),
                _ => Expr::add(matching),
            };
            let remainder = match non_matching.len() {
                0 => Expr::zero(),
                1 => non_matching.into_iter().next().unwrap(),
                _ => Expr::add(non_matching),
            };

            (zoomed, remainder)
        }
        _ => (expr.clone(), Expr::zero()),
    }
}

/// Restore an expression after zoom by combining zoomed part with remainder.
pub fn unzoom(zoomed: &Expr, remainder: &Expr) -> Expr {
    if *remainder == Expr::zero() {
        zoomed.clone()
    } else {
        Expr::add(vec![zoomed.clone(), remainder.clone()])
    }
}

/// Keep only terms in a sum that match the given pattern. Discard non-matching terms.
pub fn take_match(expr: &Expr, pattern: &Expr, interner: &ax_ir::Interner) -> Expr {
    let (matched, _) = zoom(expr, pattern, interner);
    matched
}

fn structurally_contains(expr: &Expr, pattern: &Expr) -> bool {
    if structurally_matches(expr, pattern) {
        return true;
    }
    match expr {
        Expr::Mul(factors) => factors.iter().any(|f| structurally_contains(f, pattern)),
        Expr::Add(terms) => terms.iter().any(|t| structurally_contains(t, pattern)),
        Expr::Neg(e) => structurally_contains(e, pattern),
        Expr::Indexed(base, _) => structurally_contains(base, pattern),
        Expr::Call(_, args) => args.iter().any(|a| structurally_contains(a, pattern)),
        _ => false,
    }
}

fn structurally_matches(expr: &Expr, pattern: &Expr) -> bool {
    match (expr, pattern) {
        (Expr::Sym(a), Expr::Sym(b)) => a == b,
        (Expr::Indexed(base_e, _), Expr::Sym(s)) => {
            if let Expr::Sym(base_s) = base_e.as_ref() {
                base_s == s
            } else {
                false
            }
        }
        (Expr::Call(f1, _), Expr::Sym(s)) => f1 == s,
        (Expr::Call(f1, _), Expr::Call(f2, _)) => f1 == f2,
        _ => false,
    }
}

fn parse_bool_like_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<bool> {
    match expr {
        Expr::Sym(sym) => match interner.resolve(*sym) {
            "true" | "on" => Some(true),
            "false" | "off" => Some(false),
            _ => None,
        },
        Expr::Int(n) => {
            if n == &0.into() {
                Some(false)
            } else if n == &1.into() {
                Some(true)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_optional_usize_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<Option<usize>> {
    match expr {
        Expr::Int(n) => n.to_usize().map(Some),
        Expr::Sym(sym) if interner.resolve(*sym).eq_ignore_ascii_case("none") => Some(None),
        _ => None,
    }
}

fn parse_optional_symbol_expr(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<Option<lasso::Spur>> {
    match expr {
        Expr::Sym(sym) if interner.resolve(*sym).eq_ignore_ascii_case("none") => Some(None),
        Expr::Sym(sym) => Some(Some(*sym)),
        _ => None,
    }
}

fn parse_spinor_class_expr(expr: &Expr, interner: &ax_ir::Interner) -> Option<ax_ir::SpinorClass> {
    let Expr::Sym(sym) = expr else {
        return None;
    };
    match interner.resolve(*sym).to_ascii_lowercase().as_str() {
        "dirac" => Some(ax_ir::SpinorClass::Dirac),
        "majorana" => Some(ax_ir::SpinorClass::Majorana),
        "weyl" => Some(ax_ir::SpinorClass::Weyl),
        "majoranaweyl" | "majorana_weyl" => Some(ax_ir::SpinorClass::MajoranaWeyl),
        _ => None,
    }
}

fn parse_optional_chirality_expr(
    expr: &Expr,
    interner: &ax_ir::Interner,
) -> Option<Option<ax_ir::Chirality>> {
    let Expr::Sym(sym) = expr else {
        return None;
    };
    match interner.resolve(*sym).to_ascii_lowercase().as_str() {
        "none" => Some(None),
        "left" => Some(Some(ax_ir::Chirality::Left)),
        "right" => Some(Some(ax_ir::Chirality::Right)),
        _ => None,
    }
}

fn parse_young_tableau_expr(expr: &Expr) -> Option<ax_young::YoungTableau> {
    let rows = match expr {
        Expr::List(rows) => rows.clone(),
        Expr::Matrix(rows) => rows.iter().cloned().map(Expr::List).collect(),
        _ => return None,
    };

    let mut parsed_rows = Vec::with_capacity(rows.len());
    for row in &rows {
        let Expr::List(cells) = row else {
            return None;
        };
        let mut parsed_cells = Vec::with_capacity(cells.len());
        for cell in cells {
            let Expr::Int(value) = cell else {
                return None;
            };
            parsed_cells.push(value.to_usize()?);
        }
        parsed_rows.push(parsed_cells);
    }

    ax_young::YoungTableau::with_metadata(parsed_rows, num_rational::BigRational::one(), 0).ok()
}

pub fn eval(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> Expr {
    ax_ir::abort_if_cancelled();
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(eval(re, env, interner)),
            Box::new(eval(im, env, interner)),
        ),
        Expr::Sym(s) => {
            if let Some(val) = env.lookup(*s) {
                if matches!(val, Expr::Sym(bound) if *bound == *s) {
                    Expr::Sym(*s)
                } else {
                    eval(val, env, interner)
                }
            } else if interner.resolve(*s) == "i" {
                Expr::Complex(Box::new(Expr::zero()), Box::new(Expr::one()))
            } else if interner.resolve(*s) == "pi" {
                Expr::Float(std::f64::consts::PI)
            } else if interner.resolve(*s) == "e" {
                Expr::Float(std::f64::consts::E)
            } else {
                Expr::Sym(*s)
            }
        }
        Expr::Add(terms) => {
            let evaluated = terms.iter().map(|term| eval(term, env, interner)).collect();
            let result = Expr::add(evaluated);
            if env.gradings.is_empty() {
                result
            } else {
                grassmann_simplify(&result, &env.gradings, interner)
            }
        }
        Expr::Mul(factors) => {
            let evaluated = factors
                .iter()
                .map(|factor| eval(factor, env, interner))
                .collect();
            let result = Expr::mul(evaluated);
            if env.gradings.is_empty() {
                result
            } else {
                grassmann_simplify(&result, &env.gradings, interner)
            }
        }
        Expr::Pow(base, exp) => {
            let evaled_base = eval(base, env, interner);
            let evaled_exp = eval(exp, env, interner);
            let result = if let Some(out) = numeric_pow(&evaled_base, &evaled_exp) {
                out
            } else if matches!(&evaled_base, Expr::Int(n) if *n == (-1).into()) {
                match &evaled_exp {
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Even) => Expr::one(),
                    Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Odd) => {
                        Expr::Int((-1).into())
                    }
                    _ => Expr::pow(evaled_base, evaled_exp),
                }
            } else {
                Expr::pow(evaled_base, evaled_exp)
            };
            if env.gradings.is_empty() {
                result
            } else {
                grassmann_simplify(&result, &env.gradings, interner)
            }
        }
        Expr::Neg(e) => Expr::neg(eval(e, env, interner)),
        Expr::Group(inner, rel) => Expr::Group(Box::new(eval(inner, env, interner)), *rel),
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            match name {
                "dsolve" => {
                    if args.len() == 3 {
                        if let (Expr::Sym(y), Expr::Sym(x)) = (&args[1], &args[2]) {
                            return ax_ode::solve_ode(&args[0], *y, *x, interner);
                        }
                    }
                }
                "first_order_form" => {
                    if args.len() >= 3 {
                        if let (Expr::Sym(dep), Expr::Sym(indep)) = (&args[1], &args[2]) {
                            let system = ax_ode::first_order_form(&args[0], *dep, *indep, interner);
                            return Expr::List(
                                system
                                    .into_iter()
                                    .map(|(lhs, rhs)| Expr::List(vec![lhs, rhs]))
                                    .collect(),
                            );
                        }
                    }
                }
                "abstract_tensor_reduce" | "abstract_gr_reduce" => {
                    return builtin_call(name, *f, args.clone(), interner, env);
                }
                _ => {}
            }
            let evaled_args: Vec<Expr> = args.iter().map(|arg| eval(arg, env, interner)).collect();
            let result = builtin_call(name, *f, evaled_args.clone(), interner, env);
            if let Expr::Call(returned_f, _) = &result {
                if *returned_f == *f {
                    if let Some(Expr::FnDef(_, params, body)) = env.lookup(*f) {
                        if params.len() == evaled_args.len() {
                            let mut child_env = env.clone();
                            for (param, arg) in params.iter().zip(evaled_args.iter()) {
                                child_env.bindings.insert(*param, arg.clone());
                            }
                            return eval(body, &child_env, interner);
                        }
                    }
                }
            }
            result
        }
        Expr::FnDef(name, params, body) => Expr::FnDef(*name, params.clone(), body.clone()),
        Expr::Rule(lhs, rhs, trust) => Expr::Rule(lhs.clone(), rhs.clone(), *trust),
        Expr::Import(path) => Expr::Import(path.clone()),
        Expr::Assume(name, assumptions) => Expr::Assume(*name, assumptions.clone()),
        Expr::SetConvention(field, value) => Expr::SetConvention(field.clone(), value.clone()),
        Expr::Piecewise(cases) => {
            for (value, condition) in cases {
                match eval_condition(condition, env, interner) {
                    Some(true) => return eval(value, env, interner),
                    Some(false) => continue,
                    None => {
                        return Expr::Piecewise(
                            cases
                                .iter()
                                .map(|(value, condition)| {
                                    (eval(value, env, interner), condition.clone())
                                })
                                .collect(),
                        );
                    }
                }
            }
            Expr::zero()
        }
        Expr::Let(name, val, body) => {
            let evaled_val = eval(val, env, interner);
            let child = env.extend(*name, evaled_val);
            eval(body, &child, interner)
        }
        Expr::Indexed(base, indices) => {
            let typed_indices = indices
                .iter()
                .map(|idx| ax_ir::Index {
                    name: idx.name,
                    variance: idx.variance.clone(),
                    index_type: env.index_to_family.get(&idx.name).copied(),
                })
                .collect::<Vec<_>>();
            let indexed = Expr::Indexed(Box::new(eval(base, env, interner)), typed_indices);
            let canonical =
                ax_tensor::canonicalize_indices(&indexed, &env.property_store, interner);
            let _ = if let Expr::Indexed(_, idxs) = &canonical {
                ax_tensor::detect_contractions(idxs)
            } else {
                Vec::new()
            };
            ax_tensor::rename_dummies(&canonical, env, interner)
        }
        Expr::List(items) => {
            let evaled: Vec<Expr> = items.iter().map(|item| eval(item, env, interner)).collect();
            if let Some(ncols) = evaled.first().and_then(|e| {
                if let Expr::List(inner) = e {
                    Some(inner.len())
                } else {
                    None
                }
            }) {
                if evaled
                    .iter()
                    .all(|e| matches!(e, Expr::List(inner) if inner.len() == ncols))
                {
                    let rows = evaled
                        .into_iter()
                        .map(|e| {
                            if let Expr::List(inner) = e {
                                inner
                            } else {
                                unreachable!()
                            }
                        })
                        .collect();
                    return Expr::Matrix(rows);
                }
            }
            Expr::List(evaled)
        }
        Expr::Matrix(rows) => {
            let evaled_rows = rows
                .iter()
                .map(|row| row.iter().map(|cell| eval(cell, env, interner)).collect())
                .collect();
            Expr::Matrix(evaled_rows)
        }
    }
}

pub(crate) fn to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Int(n) => n.to_f64(),
        Expr::Rational(r) => Some(r.numer().to_f64()? / r.denom().to_f64()?),
        Expr::Float(f) => Some(*f),
        Expr::Complex(re, im) => {
            if to_f64(im)? == 0.0 {
                to_f64(re)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn expr_to_rows(expr: &Expr) -> Option<Vec<Vec<Expr>>> {
    match expr {
        Expr::Matrix(rows) => Some(rows.clone()),
        Expr::List(level2) => level2
            .iter()
            .map(|row| match row {
                Expr::List(level3) => Some(level3.clone()),
                Expr::Matrix(rows) if rows.len() == 1 => rows.first().cloned(),
                _ => None,
            })
            .collect(),
        Expr::Group(inner, _) => expr_to_rows(inner),
        _ => None,
    }
}

fn expr_to_3d(expr: &Expr) -> Option<Vec<Vec<Vec<Expr>>>> {
    match expr {
        Expr::List(level1) => level1.iter().map(expr_to_rows).collect(),
        Expr::Matrix(rows) => rows
            .iter()
            .map(|row| expr_to_rows(&Expr::List(row.clone())))
            .collect(),
        Expr::Group(inner, _) => expr_to_3d(inner),
        _ => None,
    }
}

fn expr_to_matrix(expr: &Expr) -> Option<Vec<Vec<Expr>>> {
    match expr {
        Expr::Matrix(rows) => Some(rows.clone()),
        Expr::List(rows) => rows
            .iter()
            .map(|row| match row {
                Expr::List(cells) => Some(cells.clone()),
                _ => None,
            })
            .collect(),
        Expr::Group(inner, _) => expr_to_matrix(inner),
        _ => None,
    }
}

fn expr_3d_to_list(data: Vec<Vec<Vec<Expr>>>) -> Expr {
    Expr::List(
        data.into_iter()
            .map(|level2| Expr::List(level2.into_iter().map(Expr::List).collect()))
            .collect(),
    )
}

fn expr_to_3d_level(expr: &Expr) -> Option<Vec<Vec<Vec<Expr>>>> {
    match expr {
        Expr::List(level2) => level2.iter().map(expr_to_rows).collect(),
        Expr::Matrix(rows) => rows
            .iter()
            .map(|row| expr_to_rows(&Expr::List(row.clone())))
            .collect(),
        Expr::Group(inner, _) => expr_to_3d_level(inner),
        _ => None,
    }
}

fn expr_to_4d(expr: &Expr) -> Option<Vec<Vec<Vec<Vec<Expr>>>>> {
    match expr {
        Expr::List(level1) => level1.iter().map(expr_to_3d_level).collect(),
        Expr::Matrix(rows) => rows
            .iter()
            .map(|row| expr_to_3d_level(&Expr::List(row.clone())))
            .collect(),
        Expr::Group(inner, _) => expr_to_4d(inner),
        _ => None,
    }
}

fn simplifier_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(simplifier_node_count).sum::<usize>()
        }
        Expr::Pow(base, exp) => 1 + simplifier_node_count(base) + simplifier_node_count(exp),
        Expr::Neg(inner) => 1 + simplifier_node_count(inner),
        Expr::Call(_, args) => 1 + args.iter().map(simplifier_node_count).sum::<usize>(),
        Expr::Complex(re, im) => 1 + simplifier_node_count(re) + simplifier_node_count(im),
        Expr::FnDef(_, _, body) => 1 + simplifier_node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + simplifier_node_count(lhs) + simplifier_node_count(rhs),
        Expr::Piecewise(cases) => {
            1 + cases
                .iter()
                .map(|(value, _)| simplifier_node_count(value))
                .sum::<usize>()
        }
        Expr::Indexed(base, _) => 1 + simplifier_node_count(base),
        Expr::Group(inner, _) => 1 + simplifier_node_count(inner),
        Expr::Let(_, val, body) => 1 + simplifier_node_count(val) + simplifier_node_count(body),
        Expr::Matrix(rows) => {
            1 + rows
                .iter()
                .flatten()
                .map(simplifier_node_count)
                .sum::<usize>()
        }
        _ => 1,
    }
}

fn aggressive_eval_simplify(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Matrix(rows) => Expr::Matrix(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| aggressive_eval_simplify(cell, env, interner))
                        .collect()
                })
                .collect(),
        ),
        Expr::List(items) => Expr::List(
            items
                .iter()
                .map(|item| aggressive_eval_simplify(item, env, interner))
                .collect(),
        ),
        Expr::Group(inner, rel) => Expr::Group(
            Box::new(aggressive_eval_simplify(inner, env, interner)),
            *rel,
        ),
        _ => {
            let mut current = eval(expr, env, interner);
            for _ in 0..6 {
                let mut step = current.clone();
                if simplifier_node_count(&step) <= 256 {
                    step = simplify::expand(&step, interner);
                }
                let collected = simplify::collect_terms(&step, interner);
                let evaled = simplify::rationalize_expanded_numerator(
                    &eval(&collected, env, interner),
                    interner,
                );
                if evaled == current {
                    break;
                }
                current = evaled;
            }
            current
        }
    }
}

fn expr_4d_to_list(data: Vec<Vec<Vec<Vec<Expr>>>>) -> Expr {
    Expr::List(
        data.into_iter()
            .map(|level2| {
                Expr::List(
                    level2
                        .into_iter()
                        .map(|level3| Expr::List(level3.into_iter().map(Expr::List).collect()))
                        .collect(),
                )
            })
            .collect(),
    )
}

pub(crate) fn expr_to_vector(expr: &Expr) -> Option<Vec<Expr>> {
    match expr {
        Expr::List(items) => Some(items.clone()),
        Expr::Group(inner, _) => expr_to_vector(inner),
        _ => None,
    }
}

pub(crate) fn expr_to_null_tetrad(expr: &Expr) -> Option<ax_tensor::NullTetrad> {
    let Expr::List(parts) = expr else {
        return None;
    };
    if parts.len() != 4 {
        return None;
    }
    Some(ax_tensor::NullTetrad {
        l: expr_to_vector(&parts[0])?,
        n: expr_to_vector(&parts[1])?,
        m: expr_to_vector(&parts[2])?,
        m_bar: expr_to_vector(&parts[3])?,
    })
}

pub(crate) fn null_tetrad_to_expr(tetrad: ax_tensor::NullTetrad) -> Expr {
    Expr::List(vec![
        Expr::List(tetrad.l),
        Expr::List(tetrad.n),
        Expr::List(tetrad.m),
        Expr::List(tetrad.m_bar),
    ])
}

pub(crate) fn spin_coefficients_to_expr(coeffs: ax_tensor::SpinCoefficients) -> Expr {
    Expr::List(vec![
        coeffs.kappa,
        coeffs.sigma,
        coeffs.lambda,
        coeffs.nu,
        coeffs.rho,
        coeffs.mu,
        coeffs.tau,
        coeffs.pi,
        coeffs.epsilon,
        coeffs.gamma,
        coeffs.alpha,
        coeffs.beta,
    ])
}

pub(crate) fn expr_to_weyl_scalars(expr: &Expr) -> Option<ax_tensor::WeylScalars> {
    let Expr::List(parts) = expr else {
        return None;
    };
    if parts.len() != 5 {
        return None;
    }
    Some(ax_tensor::WeylScalars {
        psi0: parts[0].clone(),
        psi1: parts[1].clone(),
        psi2: parts[2].clone(),
        psi3: parts[3].clone(),
        psi4: parts[4].clone(),
    })
}

pub(crate) fn weyl_scalars_to_expr(scalars: ax_tensor::WeylScalars) -> Expr {
    Expr::List(vec![
        scalars.psi0,
        scalars.psi1,
        scalars.psi2,
        scalars.psi3,
        scalars.psi4,
    ])
}

fn matrix_to_symbolic(expr: &Expr) -> Option<ax_tensor::SymbolicMatrix> {
    let Expr::Matrix(rows) = expr else {
        return None;
    };
    let dim = rows.len();
    if rows.iter().any(|row| row.len() != dim) {
        return None;
    }
    Some(ax_tensor::SymbolicMatrix {
        dim,
        data: rows.clone(),
    })
}

fn symbolic_to_matrix(m: &ax_tensor::SymbolicMatrix) -> Expr {
    Expr::Matrix(m.data.clone())
}

fn symbolic_to_forms_matrix(m: &ax_tensor::SymbolicMatrix) -> ax_forms::SymbolicMatrix {
    ax_forms::SymbolicMatrix {
        dim: m.dim,
        data: m.data.clone(),
    }
}

fn normalize_metric_entry(expr: Expr) -> Expr {
    match expr {
        Expr::Group(inner, _) => normalize_metric_entry(*inner),
        Expr::Neg(inner) => Expr::neg(normalize_metric_entry(*inner)),
        Expr::Mul(factors) => {
            let mut factors = factors
                .into_iter()
                .map(normalize_metric_entry)
                .collect::<Vec<_>>();
            if let Some(first) = factors.first() {
                match first {
                    Expr::Int(n) if *n == (-1).into() => {
                        let _ = factors.remove(0);
                        return match factors.as_slice() {
                            [] => Expr::Int((-1).into()),
                            [single] => Expr::neg(single.clone()),
                            _ => Expr::neg(Expr::mul(factors)),
                        };
                    }
                    _ => {}
                }
            }
            Expr::mul(factors)
        }
        Expr::Add(terms) => Expr::add(terms.into_iter().map(normalize_metric_entry).collect()),
        Expr::Pow(base, exp) => {
            Expr::pow(normalize_metric_entry(*base), normalize_metric_entry(*exp))
        }
        Expr::Call(sym, args) => {
            Expr::Call(sym, args.into_iter().map(normalize_metric_entry).collect())
        }
        Expr::Complex(re, im) => Expr::Complex(
            Box::new(normalize_metric_entry(*re)),
            Box::new(normalize_metric_entry(*im)),
        ),
        other => other,
    }
}

pub(crate) fn simplify_symbolic_matrix(
    matrix: &ax_tensor::SymbolicMatrix,
    _env: &Env,
    _interner: &ax_ir::Interner,
) -> ax_tensor::SymbolicMatrix {
    ax_tensor::SymbolicMatrix {
        dim: matrix.dim,
        data: matrix
            .data
            .iter()
            .map(|row| row.iter().cloned().map(normalize_metric_entry).collect())
            .collect(),
    }
}

// ─── Index-aware substitution ────────────────────────────────────────────────

fn has_any_indices(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, indices) => !indices.is_empty(),
        Expr::Mul(factors) => factors.iter().any(has_any_indices),
        Expr::Add(terms) => terms.iter().any(has_any_indices),
        Expr::Neg(e) => has_any_indices(e),
        Expr::Call(_, args) => args.iter().any(has_any_indices),
        Expr::Pow(b, e) => has_any_indices(b) || has_any_indices(e),
        _ => false,
    }
}

pub fn match_tensor_pattern(
    pattern: &Expr,
    expr: &Expr,
    env: &Env,
    _interner: &ax_ir::Interner,
) -> Option<HashMap<lasso::Spur, lasso::Spur>> {
    match (pattern, expr) {
        (Expr::Indexed(p_base, p_indices), Expr::Indexed(e_base, e_indices)) => {
            if p_base != e_base || p_indices.len() != e_indices.len() {
                return None;
            }

            let mut mapping = HashMap::new();
            for (pi, ei) in p_indices.iter().zip(e_indices.iter()) {
                if pi.variance != ei.variance {
                    return None;
                }

                let p_family = pi
                    .index_type
                    .or_else(|| env.index_to_family.get(&pi.name).copied());
                let e_family = ei
                    .index_type
                    .or_else(|| env.index_to_family.get(&ei.name).copied());
                if let (Some(pf), Some(ef)) = (p_family, e_family) {
                    if pf != ef {
                        return None;
                    }
                }

                if let Some(&existing) = mapping.get(&pi.name) {
                    if existing != ei.name {
                        return None;
                    }
                } else {
                    mapping.insert(pi.name, ei.name);
                }
            }

            Some(mapping)
        }
        (Expr::Sym(ps), Expr::Sym(es)) if ps == es => Some(HashMap::new()),
        (Expr::Mul(p_factors), Expr::Mul(e_factors)) => {
            if p_factors.len() != e_factors.len() {
                return None;
            }

            let mut combined_mapping = HashMap::new();
            for (pf, ef) in p_factors.iter().zip(e_factors.iter()) {
                let sub_match = match_tensor_pattern(pf, ef, env, _interner)?;
                for (k, v) in sub_match {
                    if let Some(&existing) = combined_mapping.get(&k) {
                        if existing != v {
                            return None;
                        }
                    } else {
                        combined_mapping.insert(k, v);
                    }
                }
            }
            Some(combined_mapping)
        }
        _ if pattern == expr => Some(HashMap::new()),
        _ => None,
    }
}

fn collect_index_names_set(expr: &Expr, names: &mut HashSet<lasso::Spur>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_index_names_set(base, names);
            for idx in indices {
                names.insert(idx.name);
            }
        }
        Expr::Mul(factors) => {
            for f in factors {
                collect_index_names_set(f, names);
            }
        }
        Expr::Add(terms) => {
            for t in terms {
                collect_index_names_set(t, names);
            }
        }
        Expr::Neg(e) => collect_index_names_set(e, names),
        Expr::Call(_, args) => {
            for a in args {
                collect_index_names_set(a, names);
            }
        }
        Expr::Pow(b, e) => {
            collect_index_names_set(b, names);
            collect_index_names_set(e, names);
        }
        _ => {}
    }
}

fn count_index_names_recursive(expr: &Expr, counts: &mut HashMap<lasso::Spur, usize>) {
    match expr {
        Expr::Indexed(base, indices) => {
            count_index_names_recursive(base, counts);
            for idx in indices {
                *counts.entry(idx.name).or_default() += 1;
            }
        }
        Expr::Mul(factors) => {
            for f in factors {
                count_index_names_recursive(f, counts);
            }
        }
        Expr::Add(terms) => {
            for t in terms {
                count_index_names_recursive(t, counts);
            }
        }
        Expr::Neg(e) => count_index_names_recursive(e, counts),
        Expr::Call(_, args) => {
            for a in args {
                count_index_names_recursive(a, counts);
            }
        }
        _ => {}
    }
}

fn find_dummy_index_names(expr: &Expr) -> Vec<lasso::Spur> {
    let mut counts: HashMap<lasso::Spur, usize> = HashMap::new();
    count_index_names_recursive(expr, &mut counts);
    counts
        .into_iter()
        .filter(|(_, c)| *c >= 2)
        .map(|(n, _)| n)
        .collect()
}

fn generate_fresh_index(
    base: lasso::Spur,
    used: &HashSet<lasso::Spur>,
    interner: &ax_ir::Interner,
) -> lasso::Spur {
    let base_str = interner.resolve(base);
    for i in 1..100 {
        let candidate = format!("{}_{}", base_str, i);
        let sym = interner.get_or_intern(&candidate);
        if !used.contains(&sym) {
            return sym;
        }
    }
    interner.get_or_intern(&format!("{}_fresh", base_str))
}

fn rename_index_everywhere(expr: &Expr, from: lasso::Spur, to: lasso::Spur) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    if idx.name == from {
                        ax_ir::Index {
                            name: to,
                            variance: idx.variance.clone(),
                            index_type: idx.index_type,
                        }
                    } else {
                        idx.clone()
                    }
                })
                .collect();
            Expr::Indexed(
                Box::new(rename_index_everywhere(base, from, to)),
                new_indices,
            )
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| rename_index_everywhere(f, from, to))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| rename_index_everywhere(t, from, to))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(rename_index_everywhere(e, from, to)),
        Expr::Sym(s) if *s == from => Expr::Sym(to),
        _ => expr.clone(),
    }
}

fn apply_index_mapping(expr: &Expr, mapping: &HashMap<lasso::Spur, lasso::Spur>) -> Expr {
    match expr {
        Expr::Indexed(base, indices) => {
            let new_indices: Vec<ax_ir::Index> = indices
                .iter()
                .map(|idx| {
                    let new_name = mapping.get(&idx.name).copied().unwrap_or(idx.name);
                    ax_ir::Index {
                        name: new_name,
                        variance: idx.variance.clone(),
                        index_type: idx.index_type,
                    }
                })
                .collect();
            Expr::Indexed(Box::new(apply_index_mapping(base, mapping)), new_indices)
        }
        Expr::Sym(s) => {
            let new_s = mapping.get(s).copied().unwrap_or(*s);
            Expr::Sym(new_s)
        }
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| apply_index_mapping(f, mapping))
                .collect(),
        ),
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| apply_index_mapping(t, mapping))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(apply_index_mapping(e, mapping)),
        Expr::Call(f, args) => {
            let new_f = mapping.get(f).copied().unwrap_or(*f);
            Expr::Call(
                new_f,
                args.iter()
                    .map(|a| apply_index_mapping(a, mapping))
                    .collect(),
            )
        }
        _ => expr.clone(),
    }
}

/// Enhanced substitution that handles index relabeling.
///
/// When substituting T_{a b} → A_{a} B_{b}, if the expression already
/// uses dummy index 'c', and the replacement introduces its own dummy 'c',
/// the replacement's dummy is renamed to avoid conflict.
pub fn substitute_with_indices(
    expr: &Expr,
    target: &Expr,
    replacement: &Expr,
    env: &Env,
    interner: &ax_ir::Interner,
) -> Expr {
    if !env.property_store.symbols().is_empty() {
        return ax_compare::substitute_with_compare(
            expr,
            target,
            replacement,
            &env.property_store,
            &env.index_to_family,
            interner,
        );
    }

    if let Some(mapping) = match_tensor_pattern(target, expr, env, interner) {
        let mut used_indices = HashSet::new();
        collect_index_names_set(expr, &mut used_indices);

        let mut renamed_replacement = replacement.clone();
        for dummy in find_dummy_index_names(&renamed_replacement) {
            if used_indices.contains(&dummy) && !mapping.values().any(|&v| v == dummy) {
                let fresh = generate_fresh_index(dummy, &used_indices, interner);
                renamed_replacement = rename_index_everywhere(&renamed_replacement, dummy, fresh);
                used_indices.insert(fresh);
            }
        }

        return apply_index_mapping(&renamed_replacement, &mapping);
    }

    match expr {
        Expr::Add(terms) => Expr::add(
            terms
                .iter()
                .map(|t| substitute_with_indices(t, target, replacement, env, interner))
                .collect(),
        ),
        Expr::Mul(factors) => Expr::mul(
            factors
                .iter()
                .map(|f| substitute_with_indices(f, target, replacement, env, interner))
                .collect(),
        ),
        Expr::Neg(e) => Expr::neg(substitute_with_indices(
            e,
            target,
            replacement,
            env,
            interner,
        )),
        Expr::Pow(b, e) => Expr::pow(
            substitute_with_indices(b, target, replacement, env, interner),
            substitute_with_indices(e, target, replacement, env, interner),
        ),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter()
                .map(|a| substitute_with_indices(a, target, replacement, env, interner))
                .collect(),
        ),
        Expr::Indexed(base, indices) => Expr::Indexed(
            Box::new(substitute_with_indices(
                base,
                target,
                replacement,
                env,
                interner,
            )),
            indices.clone(),
        ),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_ir::{Index, Variance};

    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(
            result.errors.is_empty(),
            "lower errors: {:?}",
            result.errors
        );
        if result.exprs.len() <= 1 {
            let expr = result.expr.expect("expected expression");
            let env = Env::new();
            return (eval(&expr, &env, &interner), interner);
        }
        let mut env = Env::new();
        let mut last = Expr::zero();
        for expr in result.exprs {
            if apply_set_convention(&expr, &mut env).is_some()
                || apply_parallel_declaration(&expr, &mut env, &interner).is_some()
                || apply_graded_declaration(&expr, &mut env, &interner).is_some()
                || apply_superspace_setup(&expr, &mut env, &interner).is_some()
                || apply_brst_setup(&expr, &mut env, &interner).is_some()
                || apply_named_operator_declaration(&expr, &mut env, &interner).is_some()
                || apply_named_contraction_declaration(&expr, &mut env, &interner).is_some()
                || apply_property_declaration(&expr, &mut env, &interner).is_some()
                || apply_coordinate_declaration(&expr, &mut env, &interner).is_some()
                || apply_index_declaration(&expr, &mut env, &interner).is_some()
            {
                last = Expr::zero();
                continue;
            }

            let result = eval(&expr, &env, &interner);
            if let Expr::FnDef(name, _, _) = &result {
                env.bindings.insert(*name, result.clone());
            }
            let _ = register_rule(&result, &mut env, &interner);
            let _ = apply_coordinate_declaration(&result, &mut env, &interner);
            let _ = apply_grassmann_declaration(&result, &mut env, &interner);
            let _ = apply_operator_declaration(&result, &mut env, &interner);
            let _ = apply_named_operator_declaration(&result, &mut env, &interner);
            let _ = apply_named_contraction_declaration(&result, &mut env, &interner);
            if let Expr::Assume(var, assumptions) = &result {
                env.assumptions
                    .entry(*var)
                    .or_default()
                    .extend(assumptions.clone());
            }
            if let Expr::Let(name, val, _) = &expr {
                let evaled_val = eval(val, &env, &interner);
                env.bindings.insert(*name, evaled_val.clone());
                last = evaled_val;
            } else {
                last = result;
            }
        }
        (last, interner)
    }

    fn eval_fixture(path: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture_path = manifest_dir
            .join("../../tests/fixtures/cosmology")
            .join(path);
        let source = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", fixture_path.display()));
        let interner = ax_ir::Interner::new();
        let lowered = ax_core_ir::lower(&source, &interner);
        assert!(
            lowered.errors.is_empty(),
            "lower errors in {}: {:?}",
            fixture_path.display(),
            lowered.errors
        );
        let mut env = Env::new();
        let mut outputs = Vec::new();
        for expr in lowered.exprs {
            if apply_set_convention(&expr, &mut env).is_some()
                || apply_parallel_declaration(&expr, &mut env, &interner).is_some()
                || apply_graded_declaration(&expr, &mut env, &interner).is_some()
                || apply_superspace_setup(&expr, &mut env, &interner).is_some()
                || apply_brst_setup(&expr, &mut env, &interner).is_some()
                || apply_named_operator_declaration(&expr, &mut env, &interner).is_some()
                || apply_named_contraction_declaration(&expr, &mut env, &interner).is_some()
                || apply_property_declaration(&expr, &mut env, &interner).is_some()
                || apply_coordinate_declaration(&expr, &mut env, &interner).is_some()
                || apply_index_declaration(&expr, &mut env, &interner).is_some()
            {
                continue;
            }

            let result = eval(&expr, &env, &interner);
            if let Expr::FnDef(name, _, _) = &result {
                env.bindings.insert(*name, result.clone());
            }
            let _ = register_rule(&result, &mut env, &interner);
            let _ = apply_coordinate_declaration(&result, &mut env, &interner);
            let _ = apply_grassmann_declaration(&result, &mut env, &interner);
            let _ = apply_operator_declaration(&result, &mut env, &interner);
            let _ = apply_named_operator_declaration(&result, &mut env, &interner);
            let _ = apply_named_contraction_declaration(&result, &mut env, &interner);
            if let Expr::Assume(var, assumptions) = &result {
                env.assumptions
                    .entry(*var)
                    .or_default()
                    .extend(assumptions.clone());
            }
            if let Expr::Let(name, val, _) = &expr {
                let evaled_val = eval(val, &env, &interner);
                env.bindings.insert(*name, evaled_val);
                continue;
            }
            outputs.push(result);
        }
        (Expr::List(outputs), interner)
    }

    fn contains_unresolved_cpt_call(
        expr: &Expr,
        interner: &ax_ir::Interner,
        names: &[&str],
    ) -> bool {
        match expr {
            Expr::Call(name, args) => {
                let unresolved = names
                    .iter()
                    .any(|candidate| interner.resolve(*name) == *candidate);
                unresolved
                    || args
                        .iter()
                        .any(|arg| contains_unresolved_cpt_call(arg, interner, names))
            }
            Expr::Add(items) | Expr::Mul(items) | Expr::List(items) => items
                .iter()
                .any(|item| contains_unresolved_cpt_call(item, interner, names)),
            Expr::Pow(base, exp) => {
                contains_unresolved_cpt_call(base, interner, names)
                    || contains_unresolved_cpt_call(exp, interner, names)
            }
            Expr::Neg(inner) | Expr::Group(inner, _) => {
                contains_unresolved_cpt_call(inner, interner, names)
            }
            Expr::Matrix(rows) => rows
                .iter()
                .flatten()
                .any(|item| contains_unresolved_cpt_call(item, interner, names)),
            Expr::Indexed(base, _) => contains_unresolved_cpt_call(base, interner, names),
            Expr::Rule(lhs, rhs, _) => {
                contains_unresolved_cpt_call(lhs, interner, names)
                    || contains_unresolved_cpt_call(rhs, interner, names)
            }
            Expr::Let(_, value, body) => {
                contains_unresolved_cpt_call(value, interner, names)
                    || contains_unresolved_cpt_call(body, interner, names)
            }
            Expr::FnDef(_, _, body) => contains_unresolved_cpt_call(body, interner, names),
            Expr::Assume(_, _) => false,
            Expr::Piecewise(items) => items.iter().any(|(branch, condition)| {
                contains_unresolved_cpt_call(branch, interner, names)
                    || contains_unresolved_cpt_condition(condition, interner, names)
            }),
            Expr::Import(_) | Expr::SetConvention(_, _) => false,
            _ => false,
        }
    }

    fn contains_unresolved_cpt_condition(
        condition: &ax_ir::Condition,
        interner: &ax_ir::Interner,
        names: &[&str],
    ) -> bool {
        match condition {
            ax_ir::Condition::Gt(lhs, rhs)
            | ax_ir::Condition::Lt(lhs, rhs)
            | ax_ir::Condition::Ge(lhs, rhs)
            | ax_ir::Condition::Le(lhs, rhs)
            | ax_ir::Condition::Eq(lhs, rhs)
            | ax_ir::Condition::Ne(lhs, rhs) => {
                contains_unresolved_cpt_call(lhs, interner, names)
                    || contains_unresolved_cpt_call(rhs, interner, names)
            }
            ax_ir::Condition::And(lhs, rhs) | ax_ir::Condition::Or(lhs, rhs) => {
                contains_unresolved_cpt_condition(lhs, interner, names)
                    || contains_unresolved_cpt_condition(rhs, interner, names)
            }
            ax_ir::Condition::Not(inner) => {
                contains_unresolved_cpt_condition(inner, interner, names)
            }
            ax_ir::Condition::True | ax_ir::Condition::False => false,
        }
    }

    #[test]
    fn convention_default_is_mtw() {
        let env = Env::new();
        assert_eq!(env.convention.riemann_sign, ax_ir::RiemannSign::MTW);
    }

    #[test]
    fn index_family_creation() {
        let interner = ax_ir::Interner::new();
        let spacetime = interner.get_or_intern("spacetime");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let family = ax_ir::IndexFamily {
            name: spacetime,
            values: vec![],
            position: ax_ir::IndexPosition::Free,
            dimension: Some(4),
            parent: None,
        };
        let mut env = Env::new();
        env.index_families.insert(spacetime, family);
        env.index_to_family.insert(mu, spacetime);
        env.index_to_family.insert(nu, spacetime);
        assert_eq!(env.index_to_family[&mu], spacetime);
    }

    #[test]
    fn index_family_registered() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();

        let decl = ax_core_ir::lower("indices spacetime [mu, nu, rho, sigma] dim=4", &interner);
        let expr = decl.expr.unwrap();
        let result = eval(&expr, &env, &interner);

        if let ax_ir::Expr::Call(f, args) = &result {
            assert_eq!(interner.resolve(*f), "__declare_indices");
            assert!(args.len() >= 2);
        } else {
            panic!("expected __declare_indices call");
        }
    }

    #[test]
    fn numerical_sinh() {
        let interner = ax_ir::Interner::new();
        let sinh_sym = interner.get_or_intern("sinh");
        let result = eval(
            &Expr::Call(sinh_sym, vec![Expr::Float(1.0)]),
            &Env::new(),
            &interner,
        );
        if let Expr::Float(v) = result {
            assert!((v - 1.0_f64.sinh()).abs() < 1e-10);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn numerical_arctan() {
        let interner = ax_ir::Interner::new();
        let atan_sym = interner.get_or_intern("atan");
        let result = eval(
            &Expr::Call(atan_sym, vec![Expr::Float(1.0)]),
            &Env::new(),
            &interner,
        );
        if let Expr::Float(v) = result {
            assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn log_exp_cancel() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let log_sym = interner.get_or_intern("log");
        let exp_sym = interner.get_or_intern("exp");
        let expr = Expr::Call(log_sym, vec![Expr::Call(exp_sym, vec![Expr::Sym(x)])]);
        let result = eval(&expr, &Env::new(), &interner);
        assert_eq!(result, Expr::Sym(x));
    }

    #[test]
    fn exp_log_cancel() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let log_sym = interner.get_or_intern("log");
        let exp_sym = interner.get_or_intern("exp");
        let expr = Expr::Call(exp_sym, vec![Expr::Call(log_sym, vec![Expr::Sym(x)])]);
        let result = eval(&expr, &Env::new(), &interner);
        assert_eq!(result, Expr::Sym(x));
    }

    #[test]
    fn sqrt_perfect_square() {
        let interner = ax_ir::Interner::new();
        let sqrt_sym = interner.get_or_intern("sqrt");
        let expr = Expr::Call(sqrt_sym, vec![Expr::Int(49.into())]);
        let result = eval(&expr, &Env::new(), &interner);
        assert_eq!(result, Expr::Int(7.into()));
    }

    #[test]
    fn sin_pi_over_6() {
        let interner = ax_ir::Interner::new();
        let sin_sym = interner.get_or_intern("sin");
        let pi_sym = interner.get_or_intern("pi");
        let expr = Expr::Call(
            sin_sym,
            vec![Expr::mul(vec![
                Expr::Rational(BigRational::new(1.into(), 6.into())),
                Expr::Sym(pi_sym),
            ])],
        );
        let result = eval(&expr, &Env::new(), &interner);
        assert_eq!(result, Expr::Rational(BigRational::new(1.into(), 2.into())));
    }

    #[test]
    fn cos_pi() {
        let interner = ax_ir::Interner::new();
        let cos_sym = interner.get_or_intern("cos");
        let pi_sym = interner.get_or_intern("pi");
        let expr = Expr::Call(cos_sym, vec![Expr::Sym(pi_sym)]);
        let result = eval(&expr, &Env::new(), &interner);
        assert_eq!(result, Expr::Int((-1i64).into()));
    }

    #[test]
    fn gradient_3d() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let grad_sym = interner.get_or_intern("gradient");

        let f = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::pow(Expr::Sym(y), Expr::Int(2.into())),
            Expr::pow(Expr::Sym(z), Expr::Int(2.into())),
        ]);
        let expr = Expr::Call(
            grad_sym,
            vec![
                f,
                Expr::List(vec![Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)]),
            ],
        );
        let result = eval(&expr, &env, &interner);

        if let Expr::List(components) = &result {
            assert_eq!(components.len(), 3, "gradient should have 3 components");
        } else {
            panic!("expected List, got {:?}", result);
        }
    }

    #[test]
    fn divergence_3d() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let div_sym = interner.get_or_intern("divergence");

        let expr = Expr::Call(
            div_sym,
            vec![
                Expr::List(vec![Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)]),
                Expr::List(vec![Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)]),
            ],
        );
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::Int(3.into()), "div([x,y,z]) = 3");
    }

    #[test]
    fn curl_conservative_field_is_zero() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let z = interner.get_or_intern("z");
        let curl_sym = interner.get_or_intern("curl");

        let expr = Expr::Call(
            curl_sym,
            vec![
                Expr::List(vec![Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)]),
                Expr::List(vec![Expr::Sym(x), Expr::Sym(y), Expr::Sym(z)]),
            ],
        );
        let result = eval(&expr, &env, &interner);
        if let Expr::List(components) = &result {
            for c in components {
                assert_eq!(
                    *c,
                    Expr::zero(),
                    "curl of conservative field should be zero"
                );
            }
        } else {
            panic!("expected List, got {:?}", result);
        }
    }

    #[test]
    fn laplacian_harmonic() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let lap_sym = interner.get_or_intern("laplacian");

        let f = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::neg(Expr::pow(Expr::Sym(y), Expr::Int(2.into()))),
        ]);
        let expr = Expr::Call(
            lap_sym,
            vec![f, Expr::List(vec![Expr::Sym(x), Expr::Sym(y)])],
        );
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero(), "x²-y² is harmonic");
    }

    #[test]
    fn hessian_quadratic() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let hess_sym = interner.get_or_intern("hessian");

        let f = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::mul(vec![Expr::Int(3.into()), Expr::Sym(x), Expr::Sym(y)]),
            Expr::pow(Expr::Sym(y), Expr::Int(2.into())),
        ]);
        let expr = Expr::Call(
            hess_sym,
            vec![f, Expr::List(vec![Expr::Sym(x), Expr::Sym(y)])],
        );
        let result = eval(&expr, &env, &interner);
        if let Expr::Matrix(rows) = &result {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
            assert_eq!(rows[0][0], Expr::Int(2.into()));
            assert_eq!(rows[0][1], Expr::Int(3.into()));
            assert_eq!(rows[1][0], Expr::Int(3.into()));
            assert_eq!(rows[1][1], Expr::Int(2.into()));
        } else {
            panic!("expected Matrix, got {:?}", result);
        }
    }

    #[test]
    fn definite_integral_x_squared() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let x = interner.get_or_intern("x");
        let defint_sym = interner.get_or_intern("definite_integral");

        let expr = Expr::Call(
            defint_sym,
            vec![
                Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
                Expr::Sym(x),
                Expr::Int(0.into()),
                Expr::Int(1.into()),
            ],
        );
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::Rational(BigRational::new(1.into(), 3.into())));
    }

    #[test]
    fn eval_arithmetic() {
        let (e, _) = eval_src("2 + 3 * 4;");
        assert_eq!(e, ax_ir::Expr::Int(14.into()));
    }

    #[test]
    fn eval_symbolic_stays() {
        let (e, int) = eval_src("x + 1;");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("x"), "expected x in output: {}", pp);
    }

    #[test]
    fn eval_let_binding() {
        let (e, _) = eval_src("let x = 5 in x + 3;");
        assert_eq!(e, ax_ir::Expr::Int(8.into()));
    }

    #[test]
    fn eval_nested_let() {
        let (e, _) = eval_src("let x = 2 in let y = 3 in x + y;");
        assert_eq!(e, ax_ir::Expr::Int(5.into()));
    }

    #[test]
    fn eval_sqrt_perfect_square() {
        let (e, _) = eval_src("sqrt(9);");
        assert_eq!(e, ax_ir::Expr::Int(3.into()));
    }

    #[test]
    fn eval_zero_times_anything() {
        let (e, _) = eval_src("0 * x;");
        assert_eq!(e, ax_ir::Expr::Int(0.into()));
    }

    #[test]
    fn eval_diag() {
        let (e, _) = eval_src("diag(1, 2, 3);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0][0], Expr::Int(1.into()));
                assert_eq!(rows[1][1], Expr::Int(2.into()));
                assert_eq!(rows[2][2], Expr::Int(3.into()));
                assert_eq!(rows[0][1], Expr::zero());
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn det_2x2() {
        let (e, _) = eval_src("det([[1, 2], [3, 4]]);");
        assert_eq!(e, Expr::Int((-2).into()));
    }

    #[test]
    fn det_3x3() {
        let (e, _) = eval_src("det([[1,0,0],[0,1,0],[0,0,1]]);");
        assert_eq!(e, Expr::Int(1.into()));
    }

    #[test]
    fn transpose_test() {
        let (e, _) = eval_src("transpose([[1,2],[3,4]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows[0][1], Expr::Int(3.into()));
                assert_eq!(rows[1][0], Expr::Int(2.into()));
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn trace_test() {
        let (e, _) = eval_src("trace_mat([[1,2],[3,4]]);");
        assert_eq!(e, Expr::Int(5.into()));
    }

    #[test]
    fn metric_accepts_evaluated_matrix_argument() {
        let (e, _) = eval_src("metric(diag(1,2));");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], Expr::Int(1.into()));
                assert_eq!(rows[1][1], Expr::Int(2.into()));
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn christoffel_riemann_ricci_and_geodesic_accept_evaluated_tensors() {
        let (gamma, _) = eval_src("christoffel(metric(diag(1, x^2)), [x, y]);");
        assert!(
            matches!(gamma, Expr::List(_)),
            "expected christoffel tensor, got {:?}",
            gamma
        );

        let (riemann, _) = eval_src("riemann(christoffel(metric(diag(1, x^2)), [x, y]), [x, y]);");
        assert!(
            matches!(riemann, Expr::List(_)),
            "expected riemann tensor, got {:?}",
            riemann
        );

        let (ricci, _) =
            eval_src("ricci(riemann(christoffel(metric(diag(1, x^2)), [x, y]), [x, y]));");
        assert!(
            matches!(ricci, Expr::Matrix(_)),
            "expected ricci matrix, got {:?}",
            ricci
        );

        let (geodesic, _) =
            eval_src("geodesic(christoffel(metric(diag(1, x^2)), [x, y]), [x, y]);");
        assert!(
            matches!(geodesic, Expr::List(_)),
            "expected geodesic equations, got {:?}",
            geodesic
        );
    }

    #[test]
    fn inverse_2x2() {
        let (e, _) = eval_src("inv([[1,0],[0,2]]);");
        match e {
            Expr::Matrix(rows) => {
                assert_eq!(
                    rows[1][1],
                    Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()))
                );
            }
            other => panic!("expected Matrix, got {:?}", other),
        }
    }

    #[test]
    fn diff_polynomial() {
        let (e, int) = eval_src("diff(x^3, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("3") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn diff_sum() {
        let (e, _) = eval_src("diff(x + 1, x);");
        assert_eq!(e, ax_ir::Expr::Int(1.into()));
    }

    #[test]
    fn diff_constant() {
        let (e, _) = eval_src("diff(5, x);");
        assert_eq!(e, ax_ir::Expr::Int(0.into()));
    }

    #[test]
    fn diff_product() {
        let (e, int) = eval_src("diff(x * x, x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("2") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn diff_sin() {
        let (e, int) = eval_src("diff(sin(x), x);");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("cos"), "got: {}", pp);
    }

    #[test]
    fn diff_tan() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let tan_sym = interner.get_or_intern("tan");
        let expr = Expr::Call(tan_sym, vec![Expr::Sym(x)]);
        let result = differentiate(&expr, x, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(
            pp.contains("sec"),
            "d/dx tan(x) should contain sec, got: {}",
            pp
        );
    }

    #[test]
    fn diff_sinh() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let sinh_sym = interner.get_or_intern("sinh");
        let expr = Expr::Call(sinh_sym, vec![Expr::Sym(x)]);
        let result = differentiate(&expr, x, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(
            pp.contains("cosh"),
            "d/dx sinh(x) should be cosh(x), got: {}",
            pp
        );
    }

    #[test]
    fn diff_arctan() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let atan_sym = interner.get_or_intern("atan");
        let expr = Expr::Call(atan_sym, vec![Expr::Sym(x)]);
        let result = differentiate(&expr, x, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(
            !pp.contains("diff"),
            "d/dx atan(x) should not be unevaluated, got: {}",
            pp
        );
    }

    #[test]
    fn diff_chain_tan_x_squared() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let tan_sym = interner.get_or_intern("tan");
        let expr = Expr::Call(tan_sym, vec![Expr::pow(Expr::Sym(x), Expr::Int(2.into()))]);
        let result = differentiate(&expr, x, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(
            pp.contains("sec"),
            "chain rule should produce sec²(x²) · 2x, got: {}",
            pp
        );
    }

    #[test]
    fn eval_user_function() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("f(x) = x^2", &interner);
        let f_call = ax_core_ir::lower("f(3)", &interner);

        let mut env = Env::new();
        let def_expr = f_def.expr.unwrap();
        let def_result = eval(&def_expr, &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let call_expr = f_call.expr.unwrap();
        let result = eval(&call_expr, &env, &interner);
        assert_eq!(result, Expr::Int(9.into()));
    }

    #[test]
    fn eval_user_function_two_args() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("g(x, y) = x + y", &interner);
        let f_call = ax_core_ir::lower("g(3, 4)", &interner);

        let mut env = Env::new();
        let def_result = eval(&f_def.expr.unwrap(), &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let result = eval(&f_call.expr.unwrap(), &env, &interner);
        assert_eq!(result, Expr::Int(7.into()));
    }

    #[test]
    fn diff_user_function() {
        let interner = ax_ir::Interner::new();
        let f_def = ax_core_ir::lower("f(x) = x^3", &interner);
        let mut env = Env::new();
        let def_result = eval(&f_def.expr.unwrap(), &env, &interner);
        if let Expr::FnDef(name, _, _) = &def_result {
            env.bindings.insert(*name, def_result.clone());
        }

        let diff_call = ax_core_ir::lower("diff(f(x), x)", &interner);
        let result = eval(&diff_call.expr.unwrap(), &env, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("3") && pp.contains("x"), "got: {}", pp);
    }

    #[test]
    fn diff_unknown_function_with_independent_argument_is_zero() {
        let (result, _) = eval_src("diff(a(t), x);");
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn apply_user_rule() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();

        let rule_src = "rule pythag: sin(x_)^2 + cos(x_)^2 => 1";
        let rule_expr = eval(
            &ax_core_ir::lower(rule_src, &interner).expr.unwrap(),
            &env,
            &interner,
        );
        register_rule(&rule_expr, &mut env, &interner);

        let test_src = "rewrite(sin(a)^2 + cos(a)^2)";
        let test_expr = ax_core_ir::lower(test_src, &interner).expr.unwrap();
        let result = eval(&test_expr, &env, &interner);
        assert_eq!(result, Expr::Int(1.into()));
    }

    #[test]
    fn assume_positive_sqrt() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let mut env = Env::new();
        env.assumptions.insert(x, vec![ax_ir::Assumption::Positive]);

        let expr = ax_core_ir::lower("sqrt(x^2)", &interner).expr.unwrap();
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::Sym(x));
    }

    #[test]
    fn grassmann_square_is_zero() {
        let interner = ax_ir::Interner::new();
        let theta = interner.get_or_intern("theta");
        let mut gradings = HashMap::new();
        gradings.insert(theta, ax_ir::Grading::Odd);

        let expr = Expr::mul(vec![Expr::Sym(theta), Expr::Sym(theta)]);
        let result = grassmann_simplify(&expr, &gradings, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn grassmann_anticommutation() {
        let interner = ax_ir::Interner::new();
        let t1 = interner.get_or_intern("theta1");
        let t2 = interner.get_or_intern("theta2");
        let mut gradings = HashMap::new();
        gradings.insert(t1, ax_ir::Grading::Odd);
        gradings.insert(t2, ax_ir::Grading::Odd);

        let a = Expr::mul(vec![Expr::Sym(t1), Expr::Sym(t2)]);
        let b = Expr::mul(vec![Expr::Sym(t2), Expr::Sym(t1)]);
        let sum = Expr::add(vec![
            grassmann_simplify(&a, &gradings, &interner),
            grassmann_simplify(&b, &gradings, &interner),
        ]);
        let result = eval(&sum, &Env::new(), &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn grassmann_canonical_order() {
        let interner = ax_ir::Interner::new();
        let t1 = interner.get_or_intern("theta1");
        let t2 = interner.get_or_intern("theta2");
        let mut gradings = HashMap::new();
        gradings.insert(t1, ax_ir::Grading::Odd);
        gradings.insert(t2, ax_ir::Grading::Odd);

        let expr = Expr::mul(vec![Expr::Sym(t2), Expr::Sym(t1)]);
        let result = grassmann_simplify(&expr, &gradings, &interner);
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(
            pp.contains('-') && pp.contains("theta1") && pp.contains("theta2"),
            "got: {}",
            pp
        );
    }

    #[test]
    fn imaginary_unit_squared() {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower("i^2", &interner);
        let expr = result.expr.unwrap();
        let evaled = eval(&expr, &Env::new(), &interner);
        assert_eq!(evaled, Expr::Int((-1).into()));
    }

    #[test]
    fn euler_formula() {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower("N(exp(i * pi))", &interner);
        let expr = result.expr.unwrap();
        let evaled = eval(&expr, &Env::new(), &interner);
        match evaled {
            Expr::Complex(_, _) | Expr::Float(_) => {}
            other => panic!("expected numeric, got {:?}", other),
        }
    }

    #[test]
    fn eval_if_then_else_numeric() {
        let (e, _) = eval_src("if 3 > 2 then 10 else 20");
        assert_eq!(e, Expr::Int(10.into()));
    }

    #[test]
    fn eval_if_symbolic() {
        let (e, _) = eval_src("if x > 0 then x else -x");
        assert!(matches!(e, Expr::Piecewise(_)));
    }

    #[test]
    fn subs_symbol() {
        let (e, _) = eval_src("subs(x^2 + x + 1, x, 3)");
        assert_eq!(e, Expr::Int(13.into()));
    }

    #[test]
    fn subs_expression() {
        let (e, int) = eval_src("subs(sin(x)^2 + 1, sin(x), y)");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(pp.contains("y") && !pp.contains("sin"), "got: {}", pp);
    }

    #[test]
    fn subs_multiple() {
        let (e, _) = eval_src("subs(x + y, [x, y], [1, 2])");
        assert_eq!(e, Expr::Int(3.into()));
    }

    #[test]
    fn subs_nested() {
        let (e, int) = eval_src("subs(diff(f(x), x), x, 0)");
        let pp = ax_ir::pretty_print(&e, &int);
        assert!(!pp.contains("(x") && !pp.contains(" x"), "got: {}", pp);
    }

    #[test]
    fn zoom_selects_matching_terms() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        // A + B + C, zoom on B → (B, A + C)
        let expr = Expr::add(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let (zoomed, remainder) = zoom(&expr, &Expr::Sym(b), &interner);
        assert_eq!(zoomed, Expr::Sym(b));
        assert!(
            matches!(&remainder, Expr::Add(terms) if terms.len() == 2),
            "expected Add of 2, got: {remainder:?}"
        );
    }

    #[test]
    fn unzoom_restores() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let result = unzoom(&Expr::Sym(a), &Expr::Sym(b));
        if let Expr::Add(terms) = &result {
            assert_eq!(terms.len(), 2);
        } else {
            panic!("expected Add, got: {result:?}");
        }
    }

    #[test]
    fn take_match_discards_non_matching() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        // A + B + C, take_match on A → A
        let expr = Expr::add(vec![Expr::Sym(a), Expr::Sym(b), Expr::Sym(c)]);
        let result = take_match(&expr, &Expr::Sym(a), &interner);
        assert_eq!(result, Expr::Sym(a));
    }

    #[test]
    fn zoom_non_sum_returns_whole() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");

        let expr = Expr::Sym(a);
        let (zoomed, remainder) = zoom(&expr, &Expr::Sym(a), &interner);
        assert_eq!(zoomed, expr);
        assert_eq!(remainder, Expr::zero());
    }

    #[test]
    fn unzoom_with_zero_remainder() {
        let interner = ax_ir::Interner::new();
        let a = interner.get_or_intern("A");

        let result = unzoom(&Expr::Sym(a), &Expr::zero());
        assert_eq!(result, Expr::Sym(a));
    }

    #[test]
    fn substitute_with_index_mapping() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let t = interner.get_or_intern("T");
        let a_sym = interner.get_or_intern("A");
        let b_sym = interner.get_or_intern("B");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");

        // Substitute T[a↓, b↓] → A[a↓] * B[b↓]
        // In expression T[mu↓, nu↓], should give A[mu↓] * B[nu↓]
        let target = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );
        let replacement = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(a_sym)),
                vec![ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(b_sym)),
                vec![ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                }],
            ),
        ]);
        let expression = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: None,
                },
            ],
        );

        let result = substitute_with_indices(&expression, &target, &replacement, &env, &interner);
        // Should be A[mu↓] * B[nu↓]
        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2);
            // Check that mu and nu appear as the index names
            let pp = ax_ir::pretty_print(&result, &interner);
            assert!(pp.contains("mu"), "expected mu in result, got: {pp}");
            assert!(pp.contains("nu"), "expected nu in result, got: {pp}");
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn substitute_matches_by_family() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();

        let spacetime = interner.get_or_intern("spacetime");
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        env.index_to_family.insert(a, spacetime);
        env.index_to_family.insert(b, spacetime);
        env.index_to_family.insert(mu, spacetime);
        env.index_to_family.insert(nu, spacetime);

        let t = interner.get_or_intern("T");
        let a_sym = interner.get_or_intern("A");
        let b_sym = interner.get_or_intern("B");

        let target = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                },
                ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                },
            ],
        );
        let replacement = Expr::mul(vec![
            Expr::Indexed(
                Box::new(Expr::Sym(a_sym)),
                vec![ax_ir::Index {
                    name: a,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                }],
            ),
            Expr::Indexed(
                Box::new(Expr::Sym(b_sym)),
                vec![ax_ir::Index {
                    name: b,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                }],
            ),
        ]);
        let expression = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![
                ax_ir::Index {
                    name: mu,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                },
                ax_ir::Index {
                    name: nu,
                    variance: ax_ir::Variance::Down,
                    index_type: Some(spacetime),
                },
            ],
        );

        let result = substitute_with_indices(&expression, &target, &replacement, &env, &interner);

        if let Expr::Mul(factors) = &result {
            assert_eq!(factors.len(), 2);
            for factor in factors {
                if let Expr::Indexed(_, indices) = factor {
                    assert!(
                        indices[0].name == mu || indices[0].name == nu,
                        "expected mu or nu, got {}",
                        interner.resolve(indices[0].name)
                    );
                }
            }
        } else {
            panic!("expected Mul, got {:?}", result);
        }
    }

    #[test]
    fn substitute_with_indices_no_conflict() {
        // When there are no dummy index conflicts, behavior is identical to direct substitution
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let f_sym = interner.get_or_intern("F");
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");

        let target = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![ax_ir::Index {
                name: x,
                variance: ax_ir::Variance::Up,
                index_type: None,
            }],
        );
        let replacement = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![ax_ir::Index {
                name: x,
                variance: ax_ir::Variance::Up,
                index_type: None,
            }],
        );
        let expression = Expr::Indexed(
            Box::new(Expr::Sym(f_sym)),
            vec![ax_ir::Index {
                name: y,
                variance: ax_ir::Variance::Up,
                index_type: None,
            }],
        );

        let result = substitute_with_indices(&expression, &target, &replacement, &env, &interner);
        // F[y↑] substituted by pattern F[x↑] → F[x↑], mapping x→y gives F[y↑]
        let pp = ax_ir::pretty_print(&result, &interner);
        assert!(pp.contains("y"), "expected y in result, got: {pp}");
    }

    #[test]
    fn substitute_with_indices_variance_mismatch_no_sub() {
        // T[a↑] should not match T[a↓]
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let t = interner.get_or_intern("T");
        let a = interner.get_or_intern("a");
        let x = interner.get_or_intern("x");

        let target = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![ax_ir::Index {
                name: a,
                variance: ax_ir::Variance::Up,
                index_type: None,
            }],
        );
        let replacement = Expr::Sym(x);
        let expression = Expr::Indexed(
            Box::new(Expr::Sym(t)),
            vec![ax_ir::Index {
                name: a,
                variance: ax_ir::Variance::Down,
                index_type: None,
            }],
        );

        let result = substitute_with_indices(&expression, &target, &replacement, &env, &interner);
        // Variance mismatch: expression unchanged
        assert_eq!(result, expression);
    }

    #[test]
    fn eval_sort_product_normalizes_barred_gamma_bilinear() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let sort_product = interner.get_or_intern("sort_product");
        let bar = interner.get_or_intern("bar");
        let gamma = interner.get_or_intern("gamma");
        let psi = interner.get_or_intern("psi");
        let a = interner.get_or_intern("a");
        env.property_store
            .declare_simple(bar, ax_ir::TensorProperty::DiracBar);
        env.property_store
            .declare_simple(gamma, ax_ir::TensorProperty::GammaMatrixProp);
        env.property_store
            .declare_simple(psi, ax_ir::TensorProperty::Spinor);

        let bilinear = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Sym(psi),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
        ]);
        let result = eval(&Expr::Call(sort_product, vec![bilinear]), &env, &interner);
        let expected = Expr::mul(vec![
            Expr::Call(bar, vec![Expr::Sym(psi)]),
            Expr::Call(gamma, vec![Expr::Sym(a)]),
            Expr::Sym(psi),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn join_gammas_in_expr_alias_evaluates() {
        let (result, interner) = eval_src("join_gammas_in_expr(gamma(mu) * gamma(nu));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            !rendered.contains("join_gammas_in_expr"),
            "alias should evaluate, got {rendered}"
        );
    }

    #[test]
    fn dsolve_evaluates_first_order_ode_written_with_diff_on_lhs() {
        let (result, interner) = eval_src("dsolve(diff(y, x) - y, y, x);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("solve_ode"), "got {rendered}");
        assert!(rendered.contains("exp"), "got {rendered}");
    }

    #[test]
    fn dsolve_evaluates_rhs_only_first_order_ode() {
        let (result, interner) = eval_src("dsolve(diff(y, x) - x, y, x);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("solve_ode"), "got {rendered}");
        assert!(
            rendered.contains("x²") || rendered.contains("x^2"),
            "got {rendered}"
        );
    }

    #[test]
    fn dsolve_reduces_elementary_integrating_factor() {
        let (result, interner) = eval_src("dsolve(diff(y, x) + x*y, y, x);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("solve_ode"), "got {rendered}");
        assert!(!rendered.contains("integrate"), "got {rendered}");
        assert!(rendered.contains("exp"), "got {rendered}");
    }

    #[test]
    fn functional_derivative_source_eval_reduces() {
        let (result, interner) =
            eval_src("functional_derivative(1/2 * m * x_dot^2 - 1/2 * k * x^2, x, [x_dot], [t]);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            !rendered.contains("functional_derivative"),
            "got {rendered}"
        );
        assert!(rendered.contains("d2x_dtdt"), "got {rendered}");
        assert!(
            rendered.contains("k*x") || rendered.contains("kx"),
            "got {rendered}"
        );
    }

    #[test]
    fn vary_action_source_eval_reduces() {
        let (result, interner) = eval_src(
            "vary_action(1/2 * m * x_dot^2 - 1/2 * k * x^2, x, delta_x, [x_dot], [delta_x_dot]);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("vary_action"), "got {rendered}");
        assert!(rendered.contains("delta_x"), "got {rendered}");
        assert!(rendered.contains("delta_x_dot"), "got {rendered}");
    }

    #[test]
    fn euler_lagrange_system_source_eval_reduces() {
        let (result, interner) = eval_src(
            "euler_lagrange_system(1/2 * phi_t^2 + 1/2 * chi_t^2 - g * phi * chi, [[phi, [phi_t]], [chi, [chi_t]]], [t]);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            !rendered.contains("euler_lagrange_system"),
            "got {rendered}"
        );
        assert!(rendered.contains("d2phi_dtdt"), "got {rendered}");
        assert!(rendered.contains("d2chi_dtdt"), "got {rendered}");
        assert!(
            rendered.contains("χ") || rendered.contains("chi"),
            "got {rendered}"
        );
        assert!(
            rendered.contains("φ") || rendered.contains("phi"),
            "got {rendered}"
        );
    }

    #[test]
    fn wedge_source_eval_accepts_general_public_name() {
        let (result, interner) = eval_src("wedge([P, Q], [R, S]);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("wedge("), "got {rendered}");
        assert!(
            rendered.contains("PS") || rendered.contains("P*S"),
            "got {rendered}"
        );
    }

    #[test]
    fn exterior_d_uses_declared_coordinates_when_omitted() {
        let (result, interner) = eval_src("coordinates [x, y]; d([P, Q]);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("d("), "got {rendered}");
        assert_eq!(rendered, "[[0, 0], [0, 0]]", "got {rendered}");
    }

    #[test]
    fn codifferential_and_interior_product_reduce() {
        let (result, interner) = eval_src(
            "let g = metric(diag(1, 1)); [codifferential([P, Q], g, [x, y]), interior_product([1, 0], [[0, M], [-M, 0]])];",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("codifferential"), "got {rendered}");
        assert!(!rendered.contains("interior_product"), "got {rendered}");
        assert!(rendered.contains("M"), "got {rendered}");
    }

    #[test]
    fn lie_derivative_form_reduces() {
        let (result, interner) = eval_src("lie_derivative_form([x, 0], [1, 0], [x, y]);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("lie_derivative_form"), "got {rendered}");
        assert!(rendered.contains("[1, 0]"), "got {rendered}");
    }

    #[test]
    fn ricci_accepts_tensor_bound_in_environment() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let src = "riemann(christoffel(metric(diag(1, x^2, y^2)), [x, y, z]), [x, y, z]);";
        let lowered = ax_core_ir::lower(src, &interner);
        let riem_expr = lowered.expr.expect("riemann expr");
        let riem_value = eval(&riem_expr, &env, &interner);
        let riem_sym = interner.get_or_intern("Riem");
        env.bindings.insert(riem_sym, riem_value.clone());

        let ricci_sym = interner.get_or_intern("ricci");
        let result = eval(
            &Expr::Call(ricci_sym, vec![Expr::Sym(riem_sym)]),
            &env,
            &interner,
        );
        assert!(
            matches!(result, Expr::Matrix(_)),
            "expected matrix from bound riemann tensor, got {:?}",
            result
        );
    }

    #[test]
    fn minkowski_christoffel_is_zero_tensor() {
        let (gamma, _) = eval_src("christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]);");
        match gamma {
            Expr::List(level1) => {
                for slice in level1 {
                    match slice {
                        Expr::Matrix(rows) => {
                            assert!(
                                rows.iter().flatten().all(|entry| *entry == Expr::zero()),
                                "expected all Minkowski Christoffel entries to vanish, got {:?}",
                                rows
                            );
                        }
                        Expr::List(rows) => {
                            for row in rows {
                                match row {
                                    Expr::List(entries) => {
                                        assert!(
                                            entries.iter().all(|entry| *entry == Expr::zero()),
                                            "expected all Minkowski Christoffel entries to vanish, got {:?}",
                                            entries
                                        );
                                    }
                                    other => panic!("expected row list, got {:?}", other),
                                }
                            }
                        }
                        other => panic!("expected matrix slice, got {:?}", other),
                    }
                }
            }
            other => panic!("expected Christoffel tensor list, got {:?}", other),
        }
    }

    #[test]
    fn frw_christoffel_does_not_include_spatial_derivatives_of_scale_factor() {
        let (gamma, interner) =
            eval_src("christoffel(metric(diag(-1, a(t)^2, a(t)^2, a(t)^2)), [t, x, y, z]);");
        let rendered = ax_ir::pretty_print(&gamma, &interner);
        assert!(
            !rendered.contains("diff(a(t), x)"),
            "FRW Christoffel should not depend on ∂a/∂x, got {rendered}"
        );
        assert!(
            !rendered.contains("diff(a(t), y)"),
            "FRW Christoffel should not depend on ∂a/∂y, got {rendered}"
        );
        assert!(
            !rendered.contains("diff(a(t), z)"),
            "FRW Christoffel should not depend on ∂a/∂z, got {rendered}"
        );
        assert!(
            rendered.contains("diff(a(t), t)"),
            "FRW Christoffel should retain time-derivative dependence, got {rendered}"
        );
    }

    #[test]
    fn de_sitter_christoffel_is_evaluated_tensor_not_symbolic_call() {
        let (gamma, interner) = eval_src(
            "christoffel(metric(diag(-(1 - r^2/3), 1/(1 - r^2/3), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]);",
        );
        let rendered = ax_ir::pretty_print(&gamma, &interner);
        assert!(
            !rendered.contains("christoffel("),
            "expected evaluated de Sitter Christoffel tensor, got {rendered}"
        );
        assert!(matches!(gamma, Expr::List(_)), "got {:?}", gamma);
    }

    #[test]
    fn schwarzschild_ricci_scalar_collapses_to_zero() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower(
            "ricci_scalar(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), inv(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))));",
            &interner,
        )
        .expr
        .expect("schwarzschild ricci scalar expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn schwarzschild_einstein_tensor_collapses_to_zero_matrix() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower(
            "einstein(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), inv(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)))), metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)));",
            &interner,
        )
        .expr
        .expect("schwarzschild einstein expr");
        let result = eval(&expr, &env, &interner);
        match result {
            Expr::Matrix(rows) => {
                assert!(
                    rows.iter().flatten().all(|entry| *entry == Expr::zero()),
                    "expected zero Einstein tensor, got {:?}",
                    rows
                );
            }
            other => panic!("expected Einstein tensor matrix, got {:?}", other),
        }
    }

    #[test]
    fn minkowski_weyl_from_curvature_eval_is_zero_rank4_tensor() {
        let (result, _interner) = eval_src(
            "weyl_from_curvature(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z]), ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), inv(metric(diag(-1, 1, 1, 1)))), metric(diag(-1, 1, 1, 1)));",
        );
        let weyl = expr_to_4d(&result).expect("weyl rank-4 list");
        assert_eq!(weyl.len(), 4);
        assert!(
            weyl.iter()
                .flatten()
                .flatten()
                .flatten()
                .all(|entry| *entry == Expr::zero()),
            "expected zero Weyl tensor, got {:?}",
            weyl
        );
    }

    #[test]
    fn schwarzschild_weyl_from_riemann_eval_matches_riemann_in_vacuum_componentwise() {
        let weyl_src = "weyl_from_riemann(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi]), ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), inv(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)))), metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)));";
        let riemann_src =
            "riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi]);";
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let weyl_expr = eval(
            &ax_core_ir::lower(weyl_src, &interner)
                .expr
                .expect("weyl expr"),
            &env,
            &interner,
        );
        let riemann_expr = eval(
            &ax_core_ir::lower(riemann_src, &interner)
                .expr
                .expect("riemann expr"),
            &env,
            &interner,
        );
        let weyl = expr_to_4d(&weyl_expr).expect("weyl rank-4 list");
        let riemann = expr_to_4d(&riemann_expr).expect("riemann rank-4 list");

        assert_ne!(
            weyl[0][1][0][1],
            Expr::zero(),
            "expected a nonzero Schwarzschild Weyl component"
        );
        assert_eq!(weyl[0][1][0][1], riemann[0][1][0][1]);
        assert_eq!(weyl[2][3][2][3], riemann[2][3][2][3]);
    }

    #[test]
    fn schwarzschild_cotton_from_curvature_eval_is_zero_rank3_tensor() {
        let (result, _interner) = eval_src(
            "cotton_from_curvature(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), inv(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)))), christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]);",
        );
        let cotton = expr_to_3d(&result).expect("cotton rank-3 list");
        assert_eq!(cotton.len(), 4);
        assert!(
            cotton
                .iter()
                .flatten()
                .flatten()
                .all(|entry| *entry == Expr::zero()),
            "expected zero Cotton tensor, got {:?}",
            cotton
        );
    }

    #[test]
    fn minkowski_bach_from_curvature_eval_is_zero_matrix() {
        let (result, _interner) = eval_src(
            "bach_from_curvature(weyl_from_curvature(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z]), ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), inv(metric(diag(-1, 1, 1, 1)))), metric(diag(-1, 1, 1, 1))), ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), metric(diag(-1, 1, 1, 1)), [t, x, y, z]);",
        );
        match result {
            Expr::Matrix(rows) => {
                assert!(
                    rows.iter().flatten().all(|entry| *entry == Expr::zero()),
                    "expected zero Bach tensor, got {:?}",
                    rows
                );
            }
            other => panic!("expected Bach tensor matrix, got {:?}", other),
        }
    }

    #[test]
    fn killing_equations_eval_returns_structured_system() {
        let (result, _interner) =
            eval_src("killing_equations(christoffel(metric(diag(-1, 1)), [t, x]), [t, x]);");
        let Expr::List(outer) = result else {
            panic!("expected outer list");
        };
        assert_eq!(outer.len(), 3);
        let Expr::List(unknowns) = &outer[0] else {
            panic!("expected unknown list");
        };
        let Expr::List(equations) = &outer[1] else {
            panic!("expected equation list");
        };
        let Expr::List(pairs) = &outer[2] else {
            panic!("expected pair list");
        };
        assert_eq!(unknowns.len(), 2);
        assert_eq!(equations.len(), 3);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn minkowski_adm_eval_returns_trivial_structured_output() {
        let (result, _interner) =
            eval_src("adm_decompose(metric(diag(-1, 1, 1, 1)), [t, x, y, z], 0);");
        let Expr::List(items) = result else {
            panic!("expected outer list");
        };
        assert_eq!(items.len(), 8);
        assert_eq!(items[0], Expr::one());
        let Expr::List(shift_covector) = &items[1] else {
            panic!("expected shift covector");
        };
        let Expr::List(shift_vector) = &items[2] else {
            panic!("expected shift vector");
        };
        let Expr::Matrix(extrinsic_curvature) = &items[5] else {
            panic!("expected extrinsic curvature matrix");
        };
        let Expr::List(momentum_constraints) = &items[7] else {
            panic!("expected momentum constraints");
        };
        assert_eq!(shift_covector.len(), 3);
        assert_eq!(shift_vector.len(), 3);
        assert!(shift_covector.iter().all(|entry| *entry == Expr::zero()));
        assert!(shift_vector.iter().all(|entry| *entry == Expr::zero()));
        assert!(extrinsic_curvature
            .iter()
            .flatten()
            .all(|entry| *entry == Expr::zero()));
        assert_eq!(items[6], Expr::zero());
        assert!(momentum_constraints
            .iter()
            .all(|entry| *entry == Expr::zero()));
    }

    #[test]
    fn flat_frw_adm_eval_has_zero_shift_and_nonzero_k11() {
        let (result, _interner) =
            eval_src("adm_decompose(metric(diag(-1, a(t)^2, a(t)^2, a(t)^2)), [t, x, y, z], 0);");
        let Expr::List(items) = result else {
            panic!("expected outer list");
        };
        let Expr::List(shift_covector) = &items[1] else {
            panic!("expected shift covector");
        };
        let Expr::List(shift_vector) = &items[2] else {
            panic!("expected shift vector");
        };
        let Expr::Matrix(extrinsic_curvature) = &items[5] else {
            panic!("expected extrinsic curvature matrix");
        };
        assert!(shift_covector.iter().all(|entry| *entry == Expr::zero()));
        assert!(shift_vector.iter().all(|entry| *entry == Expr::zero()));
        assert_ne!(extrinsic_curvature[0][0], Expr::zero());
    }

    #[test]
    fn conformal_transform_metric_eval_scales_by_constant_omega() {
        let (result, _interner) = eval_src("conformal_transform_metric(metric(diag(-1, 1)), 3);");
        assert_eq!(
            result,
            Expr::Matrix(vec![
                vec![Expr::Int((-9).into()), Expr::zero()],
                vec![Expr::zero(), Expr::Int(9.into())],
            ])
        );
    }

    #[test]
    fn conformal_transform_scalar_eval_keeps_minkowski_zero_for_constant_omega() {
        let (result, _interner) = eval_src(
            "conformal_transform_scalar(ricci_scalar(ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), inv(metric(diag(-1, 1, 1, 1)))), metric(diag(-1, 1, 1, 1)), 5, [t, x, y, z]);",
        );
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn cartan_contorsion_tensor_eval_is_zero_for_zero_torsion() {
        let (result, _interner) = eval_src(
            "contorsion_tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], metric(diag(1, 1)));",
        );
        let contorsion = expr_to_3d(&result).expect("contorsion rank-3 list");
        assert!(contorsion
            .iter()
            .flatten()
            .flatten()
            .all(|entry| *entry == Expr::zero()));
    }

    #[test]
    fn cartan_spin_connection_eval_is_zero_for_flat_cartesian_vielbein() {
        let (result, _interner) =
            eval_src("spin_connection([[1, 0], [0, 1]], metric(diag(1, 1)), [x, y]);");
        let omega = expr_to_3d(&result).expect("spin connection rank-3 list");
        assert!(omega
            .iter()
            .flatten()
            .flatten()
            .all(|entry| *entry == Expr::zero()));
    }

    #[test]
    fn minkowski_petrov_pipeline_returns_o() {
        let (result, interner) = eval_src(
            "petrov_classify(weyl_scalars(weyl_from_curvature(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z]), ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])), inv(metric(diag(-1, 1, 1, 1)))), metric(diag(-1, 1, 1, 1))), null_tetrad(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), metric(diag(-1, 1, 1, 1))));",
        );
        assert_eq!(result, Expr::Sym(interner.get_or_intern("O")));
    }

    #[test]
    fn schwarzschild_petrov_pipeline_returns_d() {
        let (result, interner) = eval_src(
            "petrov_classify(weyl_scalars(weyl_from_curvature(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi]), ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), ricci_scalar(ricci(riemann(christoffel(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi])), inv(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)))), metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))), null_tetrad(metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))));",
        );
        assert_eq!(result, Expr::Sym(interner.get_or_intern("D")));
    }

    #[test]
    fn covariant_diff_in_flat_space_reduces_to_partial_derivative() {
        let (result, interner) = eval_src(
            "covariant_diff([f(t, x, y, z), 0, 0, 0], christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), 0, [t, x, y, z]);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered.contains("diff(f(t, x, y, z), t)"),
            "got {rendered}"
        );
        assert!(!rendered.contains("christoffel("), "got {rendered}");
    }

    #[test]
    fn young_project_ignores_pure_symmetric_property_without_tableau_projector() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let t = interner.get_or_intern("T");
        env.property_store
            .declare_simple(t, ax_ir::TensorProperty::Symmetric(vec![0, 1]));
        let lowered = ax_core_ir::lower("young_project(T[a-,b-]);", &interner);
        let expr = lowered.expr.expect("young_project expr");
        let result = eval(&expr, &env, &interner);
        let expected = ax_core_ir::lower("T[a-,b-];", &interner)
            .expr
            .expect("expected expr");
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn tableau_symmetry_property_declaration_projects_end_to_end() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property T tableau_symmetry([1, 1], [0, 1])", &interner)
            .expr
            .expect("tableau property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(
            message.is_some(),
            "expected tableau_symmetry declaration to be applied"
        );

        let expr = ax_core_ir::lower("young_project(T[a-,b-]);", &interner)
            .expr
            .expect("young project expr");
        let result = eval(&expr, &env, &interner);
        let expected = ax_tensor::young_project_tensor(
            &ax_core_ir::lower("T[a-,b-];", &interner)
                .expr
                .expect("base expr"),
            &env.property_store,
            &interner,
        );
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn young_project_accepts_tensor_options() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property T tableau_symmetry([2], [0, 1])", &interner)
            .expr
            .expect("tableau property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(
            message.is_some(),
            "expected tableau_symmetry declaration to be applied"
        );

        let reduced_expr =
            ax_core_ir::lower("young_project(T[b-,a-], true, true, true);", &interner)
                .expr
                .expect("young project expr");
        let reduced = eval(&reduced_expr, &env, &interner);
        let expected = ax_tensor::young_project_tensor_with_options(
            &ax_core_ir::lower("T[b-,a-];", &interner)
                .expr
                .expect("base expr"),
            &env.property_store,
            &interner,
            &ax_tensor::YoungProjectTensorOptions {
                modulo_monoterm: true,
                canonicalize_after: true,
                rename_dummies_after: true,
            },
        );
        assert_eq!(reduced, expected, "got {:?}", reduced);

        let expanded_expr =
            ax_core_ir::lower("young_project(T[b-,a-], false, false, false);", &interner)
                .expr
                .expect("young project expr");
        let expanded = eval(&expanded_expr, &env, &interner);
        let expected = ax_tensor::young_project_tensor_with_options(
            &ax_core_ir::lower("T[b-,a-];", &interner)
                .expr
                .expect("base expr"),
            &env.property_store,
            &interner,
            &ax_tensor::YoungProjectTensorOptions {
                modulo_monoterm: false,
                canonicalize_after: false,
                rename_dummies_after: false,
            },
        );
        assert_eq!(expanded, expected, "got {:?}", expanded);
    }

    #[test]
    fn young_project_accepts_explicit_tableau_argument() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower("young_project(T[a-,b-], [[0], [1]]);", &interner)
            .expr
            .expect("young project expr");
        let result = eval(&expr, &env, &interner);
        let base_expr = ax_core_ir::lower("T[a-,b-];", &interner)
            .expr
            .expect("base expr");
        let tableau = ax_young::YoungTableau {
            rows: vec![vec![0], vec![1]],
            multiplicity: BigRational::one(),
            selfdual_column: 0,
        };
        let expected = ax_tensor::young_project(&base_expr, &tableau, &interner);
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn young_project_tensor_alias_is_property_driven() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property T tableau_symmetry([1, 1], [0, 1])", &interner)
            .expr
            .expect("tableau property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some(), "expected tableau_symmetry declaration");

        let expr = ax_core_ir::lower("young_project_tensor(T[a-,b-]);", &interner)
            .expr
            .expect("young project tensor expr");
        let result = eval(&expr, &env, &interner);
        let expected = ax_tensor::young_project_tensor(
            &ax_core_ir::lower("T[a-,b-];", &interner)
                .expr
                .expect("base expr"),
            &env.property_store,
            &interner,
        );
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn young_project_product_identity_via_eval() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property R satisfies_bianchi", &interner)
            .expr
            .expect("bianchi property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some(), "expected satisfies_bianchi declaration");

        let expr = ax_core_ir::lower(
            "young_project(R[a-,b-,c-,d-]*V[e-] + R[a-,c-,d-,b-]*V[e-] + R[a-,d-,b-,c-]*V[e-]);",
            &interner,
        )
        .expr
        .expect("young project expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn tensor_reduce_pipeline_identity_via_eval() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property R satisfies_bianchi", &interner)
            .expr
            .expect("bianchi property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some(), "expected satisfies_bianchi declaration");

        let expr = ax_core_ir::lower(
            "tensor_reduce(R[a-,b-,c-,d-]*V[e-] + R[a-,c-,d-,b-]*V[e-] + R[a-,d-,b-,c-]*V[e-]);",
            &interner,
        )
        .expr
        .expect("tensor reduce expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn schouten_reduce_uses_declared_index_family_dimension() {
        let (result, _interner) = eval_src(
            "indices V [a, b, c] dim=2;
             property T tableau_symmetry([1,1,1], [0,1,2]);
             property T dimension_dependent_identity;
             schouten_reduce(T[a-,b-,c-] - T[a-,c-,b-] - T[b-,a-,c-] + T[b-,c-,a-] + T[c-,a-,b-] - T[c-,b-,a-]);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn tensor_reduce_dimension_toggle_is_public() {
        let (result, _interner) = eval_src(
            "indices V [a, b, c] dim=2;
             property T tableau_symmetry([1,1,1], [0,1,2]);
             property T dimension_dependent_identity;
             tensor_reduce(T[a-,b-,c-] - T[a-,c-,b-] - T[b-,a-,c-] + T[b-,c-,a-] + T[c-,a-,b-] - T[c-,b-,a-], true, true, false, true, true);",
        );
        assert_ne!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn satisfies_bianchi_property_declaration_melds_to_zero() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property R satisfies_bianchi", &interner)
            .expr
            .expect("bianchi property decl");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some(), "expected satisfies_bianchi declaration");

        let expr = ax_core_ir::lower(
            "meld(R[a-,b-,c-,d-] + R[a-,c-,d-,b-] + R[a-,d-,b-,c-]);",
            &interner,
        )
        .expr
        .expect("meld expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn weyl_tensor_property_declaration_melds_to_zero() {
        let (result, _interner) = eval_src(
            "property C weyl_tensor;
             meld(C[a-,b-,c-,d-] + C[a-,c-,d-,b-] + C[a-,d-,b-,c-]);",
        );
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn riemann_tensor_direct_declaration_melds_to_zero() {
        let (result, _interner) = eval_src(
            "riemann_tensor(R);
             meld(R[a-,b-,c-,d-] + R[a-,c-,d-,b-] + R[a-,d-,b-,c-]);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn tableau_inherit_property_declaration_is_recorded() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("tableau_inherit(nabla);", &interner)
            .expr
            .expect("declaration");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some());
        let nabla = interner.get_or_intern("nabla");
        assert!(
            env.property_store
                .get_all(nabla)
                .into_iter()
                .any(|prop| matches!(prop, ax_ir::TensorProperty::TableauInherit)),
            "expected TableauInherit on nabla"
        );
    }

    #[test]
    fn covariant_derivative_declaration_also_enables_tableau_inherit() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let decl = ax_core_ir::lower("property nabla covariant_derivative", &interner)
            .expr
            .expect("declaration");
        let message = apply_property_declaration(&decl, &mut env, &interner);
        assert!(message.is_some());
        let nabla = interner.get_or_intern("nabla");
        let props = env.property_store.get_all(nabla);
        assert!(
            props
                .iter()
                .any(|prop| matches!(prop, ax_ir::TensorProperty::CovariantDerivative)),
            "expected CovariantDerivative on nabla"
        );
        assert!(
            props
                .iter()
                .any(|prop| matches!(prop, ax_ir::TensorProperty::TableauInherit)),
            "expected TableauInherit on nabla"
        );
    }

    #[test]
    fn abstract_tensor_reduce_reveals_bianchi_identity_publicly() {
        let (result, _interner) = eval_src(
            "riemann_tensor(R);
             abstract_tensor_reduce(R[a-,b-,c-,d-] + R[a-,c-,d-,b-] + R[a-,d-,b-,c-]);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn riemann_to_ricci_evaluates_single_contraction_publicly() {
        let (result, interner) =
            eval_src("riemann_tensor(R); riemann_to_ricci(R[a-,b-,a+,d-], Ric);");
        let expected = ax_core_ir::lower("Ric[b-,d-];", &interner)
            .expr
            .expect("expected expr");
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn riemann_to_ricci_evaluates_double_contraction_to_scalar_publicly() {
        let (result, interner) =
            eval_src("riemann_tensor(R); riemann_to_ricci(R[a-,b-,a+,b+], Ric, Scal);");
        let expected = ax_core_ir::lower("Scal;", &interner)
            .expr
            .expect("expected expr");
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn riemann_to_ricci_mixed_with_abstract_gr_reduce_keeps_ricci_unsymmetrised() {
        let (result, interner) = eval_src(
            "riemann_tensor(R);
             abstract_gr_reduce(riemann_to_ricci(R[a-,b-,a+,d-] + R[a-,d-,a+,b-], Ric), true, true, true, true, true);",
        );
        let expected = ax_core_ir::lower("Ric[b-,d-] + Ric[d-,b-];", &interner)
            .expr
            .expect("expected expr");
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn contracted_bianchi_reduce_rewrites_ricci_divergence_publicly() {
        let (result, interner) = eval_src(
            "riemann_tensor(R); covariant_derivative(nabla); contracted_bianchi_reduce(nabla[a+]*Ric[a-,b-], nabla, Ric, R);",
        );
        let env = Env::new();
        let expected = ax_tensor::canonicalise(
            &ax_core_ir::lower("(1/2)*nabla[b-]*R;", &interner)
                .expr
                .expect("expected expr"),
            &env.property_store,
            &interner,
        );
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn contracted_bianchi_reduce_rewrites_einstein_divergence_publicly() {
        let (result, _interner) = eval_src(
            "riemann_tensor(R); covariant_derivative(nabla); contracted_bianchi_reduce(nabla[a+]*G[a-,b-], nabla, Ric, R, G);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn contracted_bianchi_reduce_mixed_with_abstract_gr_reduce_goes_to_zero() {
        let (result, _interner) = eval_src(
            "riemann_tensor(R); covariant_derivative(nabla); abstract_gr_reduce(contracted_bianchi_reduce(nabla[b-]*R - 2*nabla[a+]*Ric[a-,b-], nabla, Ric, R), true, true, true, true, true);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn abstract_gr_reduce_second_bianchi_survives_ten_dimensional_index_family() {
        let (result, _interner) = eval_src(
            "indices spacetime [mu, nu, rho, sigma, lambda, alpha, beta, gamma, delta, epsilon] dim=10;
             riemann_tensor(R);
             covariant_derivative(nabla);
             abstract_gr_reduce(nabla[mu-]*R[nu-,rho-,sigma-,lambda-] + nabla[nu-]*R[rho-,mu-,sigma-,lambda-] + nabla[rho-]*R[mu-,nu-,sigma-,lambda-], true, true, true, true, true);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn abstract_gr_reduce_second_bianchi_survives_eleven_dimensional_index_family() {
        let (result, _interner) = eval_src(
            "indices spacetime [mu, nu, rho, sigma, lambda, alpha, beta, gamma, delta, epsilon, kappa] dim=11;
             riemann_tensor(R);
             covariant_derivative(nabla);
             abstract_gr_reduce(nabla[mu-]*R[nu-,rho-,sigma-,lambda-] + nabla[nu-]*R[rho-,mu-,sigma-,lambda-] + nabla[rho-]*R[mu-,nu-,sigma-,lambda-], true, true, true, true, true);",
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
    }

    #[test]
    fn tensor_reduce_direct_from_eval_env_handles_second_bianchi_identity() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        for decl_src in ["riemann_tensor(R);", "property nabla covariant_derivative;"] {
            let decl = ax_core_ir::lower(decl_src, &interner)
                .expr
                .expect("declaration");
            let message = apply_property_declaration(&decl, &mut env, &interner);
            assert!(
                message.is_some(),
                "expected declaration to apply for {decl_src}"
            );
        }

        let expr = ax_core_ir::lower(
            "nabla[mu-] * R[nu-,rho-,sigma-,lambda-] + nabla[nu-] * R[rho-,mu-,sigma-,lambda-] + nabla[rho-] * R[mu-,nu-,sigma-,lambda-];",
            &interner,
        )
        .expr
        .expect("expr");

        let nabla = interner.get_or_intern("nabla");
        let r = interner.get_or_intern("R");
        let mu = interner.get_or_intern("mu");
        let nu = interner.get_or_intern("nu");
        let rho = interner.get_or_intern("rho");
        let sigma = interner.get_or_intern("sigma");
        let lambda = interner.get_or_intern("lambda");
        let cov = |idx| {
            Expr::Indexed(
                Box::new(Expr::Sym(nabla)),
                vec![Index {
                    name: idx,
                    variance: Variance::Down,
                    index_type: None,
                }],
            )
        };
        let riem = |i0, i1, i2, i3| {
            Expr::Indexed(
                Box::new(Expr::Sym(r)),
                vec![
                    Index {
                        name: i0,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i1,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i2,
                        variance: Variance::Down,
                        index_type: None,
                    },
                    Index {
                        name: i3,
                        variance: Variance::Down,
                        index_type: None,
                    },
                ],
            )
        };
        let manual = Expr::add(vec![
            Expr::mul(vec![cov(mu), riem(nu, rho, sigma, lambda)]),
            Expr::mul(vec![cov(nu), riem(rho, mu, sigma, lambda)]),
            Expr::mul(vec![cov(rho), riem(mu, nu, sigma, lambda)]),
        ]);
        assert_eq!(expr, manual, "lowered expr mismatch: got {:?}", expr);

        let mut explicit_props = std::collections::HashMap::new();
        explicit_props.insert(nabla, vec![ax_ir::TensorProperty::CovariantDerivative]);
        explicit_props.insert(
            r,
            vec![
                ax_ir::TensorProperty::RiemannSymmetry,
                ax_ir::TensorProperty::SatisfiesBianchi {
                    slots: vec![0, 1, 2, 3],
                },
            ],
        );
        let explicit_result = ax_tensor::tensor_reduce(
            &expr,
            &explicit_props,
            &interner,
            &ax_tensor::TensorReduceOptions::default(),
        );
        assert_eq!(
            explicit_result,
            Expr::zero(),
            "explicit props got {:?}",
            explicit_result
        );
        let explicit_meld = ax_tensor::meld(&expr, &explicit_props, &interner);
        assert_eq!(
            explicit_meld,
            Expr::zero(),
            "explicit meld got {:?}",
            explicit_meld
        );

        let result = ax_tensor::tensor_reduce(
            &expr,
            &env.tensor_properties,
            &interner,
            &ax_tensor::TensorReduceOptions::default(),
        );
        assert_eq!(result, Expr::zero(), "got {:?}", result);
        let meld_result = ax_tensor::meld(&expr, &env.tensor_properties, &interner);
        assert_eq!(meld_result, Expr::zero(), "meld got {:?}", meld_result);
    }

    #[test]
    fn brst_demo_nilpotency_evaluates_to_zero() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let setup = ax_core_ir::lower("setup_brst_ym(A, c, cbar, B, e);", &interner)
            .expr
            .expect("setup expr");
        let setup_msg = apply_brst_setup(&setup, &mut env, &interner);
        assert!(setup_msg.is_some(), "expected BRST setup to initialize env");
        let expr = ax_core_ir::lower("brst(brst(A));", &interner)
            .expr
            .expect("brst expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn brst_filter_ghost_number_alias_projects_selected_sector() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let setup = ax_core_ir::lower("setup_brst_ym(A, c, cbar, B, e);", &interner)
            .expr
            .expect("setup expr");
        let setup_msg = apply_brst_setup(&setup, &mut env, &interner);
        assert!(setup_msg.is_some(), "expected BRST setup to initialize env");
        let expr = ax_core_ir::lower("filter_ghost_number(c + cbar + B, 1);", &interner)
            .expr
            .expect("filter ghost expr");
        let result = eval(&expr, &env, &interner);
        assert_eq!(result, Expr::Sym(interner.get_or_intern("c")));
    }

    #[test]
    fn qft_expand_diracbar_source_eval_reduces() {
        let (result, interner) = eval_src("expand_diracbar(bar(gamma(mu) * psi));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("expand_diracbar"), "got {rendered}");
        assert!(!rendered.contains("bar(gamma"), "got {rendered}");
        assert!(rendered.contains("bar(psi)"), "got {rendered}");
        assert!(rendered.contains("gamma(mu)"), "got {rendered}");
    }

    #[test]
    fn qft_fierz_source_eval_reduces() {
        let (result, interner) =
            eval_src("fierz(bar(psi1) * gamma(mu) * psi2 * bar(psi3) * psi4);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("fierz("), "got {rendered}");
        assert!(
            matches!(result, Expr::Add(_)),
            "expected Fierz sum, got {rendered}"
        );
    }

    #[test]
    fn qft_spinor_helicity_public_chain_and_mandelstam_aliases_reduce_cleanly() {
        let (chain, interner) = eval_src("expand_chain(angle_square_chain(1, [2], 3));");
        let chain_rendered = ax_render::to_unicode(&chain, &interner);
        assert_eq!(chain_rendered, "⟨12⟩[23]");

        let (contracted, interner) = eval_src("contract_adjacent(angle(1,2) * square(2,3));");
        let contracted_rendered = ax_render::to_unicode(&contracted, &interner);
        assert_eq!(contracted_rendered, "⟨1|2|3]");

        let (collected, interner) = eval_src("collect_mandelstam(angle(1,2) * square(2,1));");
        let collected_rendered = ax_render::to_unicode(&collected, &interner);
        assert_eq!(collected_rendered, "s_{12}");

        let (multi, interner) = eval_src("mandelstam_multi([1, 2, 3]);");
        let multi_rendered = ax_render::to_unicode(&multi, &interner);
        assert_eq!(multi_rendered, "s_{123}");

        let (expanded_multi, interner) =
            eval_src("expand_mandelstam(mandelstam_multi([1, 2, 3]));");
        let expanded_rendered = ax_render::to_unicode(&expanded_multi, &interner);
        assert!(
            expanded_rendered.contains("⟨12⟩[21]"),
            "got {expanded_rendered}"
        );
        assert!(!expanded_rendered.contains("__"), "got {expanded_rendered}");
    }

    #[test]
    fn qft_gamma_source_ops_reduce_cleanly() {
        let (split, interner) = eval_src("split_gamma(gamma(mu, nu));");
        let split_rendered = ax_ir::pretty_print(&split, &interner);
        assert!(
            !split_rendered.contains("split_gamma"),
            "got {split_rendered}"
        );
        assert!(
            split_rendered.contains("gamma(mu)") && split_rendered.contains("gamma(nu)"),
            "got {split_rendered}"
        );

        let (trace5, interner) = eval_src("gamma5_trace([mu, nu, rho, sigma]);");
        let trace5_rendered = ax_render::to_unicode(&trace5, &interner);
        assert_eq!(trace5_rendered, "-4ε^μ^ν^ρ^σi");
    }

    #[test]
    fn qft_superspace_extended_ops_reduce_after_setup() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let setup = ax_core_ir::lower("setup_superspace(1);", &interner)
            .expr
            .expect("setup superspace expr");
        let setup_msg = apply_superspace_setup(&setup, &mut env, &interner);
        assert!(
            setup_msg.is_some(),
            "expected superspace setup to initialize env"
        );

        let d_squared = ax_core_ir::lower("d_squared(chiral_superfield(Phi));", &interner)
            .expr
            .expect("d_squared expr");
        let d_squared_result = eval(&d_squared, &env, &interner);
        let d_squared_rendered = ax_ir::pretty_print(&d_squared_result, &interner);
        assert_eq!(d_squared_rendered, "-2*F_Phi(x0, x1, x2, x3)");

        let d_bar_squared =
            ax_core_ir::lower("d_bar_squared(antichiral_superfield(Phi_bar));", &interner)
                .expr
                .expect("d_bar_squared expr");
        let d_bar_squared_result = eval(&d_bar_squared, &env, &interner);
        let d_bar_squared_rendered = ax_ir::pretty_print(&d_bar_squared_result, &interner);
        assert_eq!(d_bar_squared_rendered, "2*F_bar_Phi_bar(x0, x1, x2, x3)");

        let extract = ax_core_ir::lower(
            "extract_component(vector_superfield_wz(V), [1,1]);",
            &interner,
        )
        .expr
        .expect("extract component expr");
        let extract_result = eval(&extract, &env, &interner);
        assert_eq!(extract_result, Expr::zero());
    }

    #[test]
    fn qft_brst_field_actions_reduce_after_setup() {
        let interner = ax_ir::Interner::new();
        let mut env = Env::new();
        let setup = ax_core_ir::lower("setup_brst_ym(A, c, cbar, B, g);", &interner)
            .expr
            .expect("setup brst expr");
        let setup_msg = apply_brst_setup(&setup, &mut env, &interner);
        assert!(setup_msg.is_some(), "expected BRST setup to initialize env");

        let brst_cbar = ax_core_ir::lower("brst(cbar);", &interner)
            .expr
            .expect("brst cbar expr");
        let brst_cbar_result = eval(&brst_cbar, &env, &interner);
        assert_eq!(brst_cbar_result, Expr::Sym(interner.get_or_intern("B")));

        let brst_b = ax_core_ir::lower("brst(B);", &interner)
            .expr
            .expect("brst B expr");
        let brst_b_result = eval(&brst_b, &env, &interner);
        assert_eq!(brst_b_result, Expr::zero());
    }

    #[test]
    fn cosmology_linearized_einstein_second_order_exposes_expected_labels() {
        let (result, interner) = eval_src("linearized_einstein(2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered.contains("second_order_00_constraint"),
            "got {rendered}"
        );
        assert!(
            rendered.contains("second_order_ij_traceless"),
            "got {rendered}"
        );
    }

    #[test]
    fn cosmology_bardeen_variables_include_phi_b_and_psi_b() {
        let (result, interner) = eval_src("bardeen();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("Phi_B"), "got {rendered}");
        assert!(rendered.contains("Psi_B"), "got {rendered}");
    }

    #[test]
    fn cosmology_tensor_scalar_ratio_is_16epsilon() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower("tensor_scalar_ratio();", &interner)
            .expr
            .expect("tensor scalar ratio expr");
        let result = eval(&expr, &env, &interner);
        let expected = Expr::mul(vec![
            Expr::Int(16.into()),
            Expr::Sym(interner.get_or_intern("epsilon")),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn cosmology_mukhanov_sasaki_still_evaluates_to_non_symbolic_expression() {
        let (result, interner) = eval_src("mukhanov_sasaki();");
        let rendered = ax_ir::pretty_print(&result, &interner);

        assert!(
            !rendered.contains("mukhanov_sasaki("),
            "expected evaluated Mukhanov-Sasaki equation, got {rendered}"
        );
        assert!(rendered.contains("c_s"), "got {rendered}");
        assert!(
            rendered.contains("diff(diff(v, eta), eta)"),
            "got {rendered}"
        );
    }

    #[test]
    fn cpt_background_spec_round_trips_through_eval_helpers() {
        let (result, interner) = eval_src("frw_background_spec(conformal, flat, 3);");
        let reparsed = parse_background_spec_expr(&result, &interner);
        assert!(reparsed.is_some());
        assert_eq!(
            make_background_spec_expr(
                &reparsed.unwrap_or_else(|| ax_perturb::FrwBackgroundSpec::default_flat_conformal(
                    &interner
                )),
                &interner
            ),
            result
        );
    }

    #[test]
    fn cpt_linearized_einstein_order1_returns_expected_labels() {
        let (result, interner) = eval_src(
            "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("00_constraint"));
        assert!(rendered.contains("ij_traceless"));
    }

    #[test]
    fn cpt_fluid_equations_returns_two_labels() {
        let (result, interner) =
            eval_src("cpt_fluid_equations(frw_background_spec(conformal, flat, 3));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("fluid_continuity"));
        assert!(rendered.contains("fluid_euler"));
    }

    #[test]
    fn linearized_einstein_vector_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("linearized_einstein_vector();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("vector_0x_momentum"), "got {rendered}");
        assert!(rendered.contains("vector_z_evolution"), "got {rendered}");
    }

    #[test]
    fn linearized_einstein_tensor_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("linearized_einstein_tensor();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("tensor_xx"), "got {rendered}");
        assert!(rendered.contains("tensor_zz"), "got {rendered}");
    }

    #[test]
    fn tensor_mode_equation_builtin_returns_plus_and_cross_entries() {
        let (result, interner) = eval_src("tensor_mode_equation();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("h_plus_eq"), "got {rendered}");
        assert!(rendered.contains("h_cross_eq"), "got {rendered}");
    }

    #[test]
    fn project_scalar_harmonics_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("project_scalar_harmonics();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("00_constraint"), "got {rendered}");
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
    }

    #[test]
    fn project_vector_harmonics_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("project_vector_harmonics();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("vector_0x_momentum"), "got {rendered}");
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
    }

    #[test]
    fn project_tensor_harmonics_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("project_tensor_harmonics();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("tensor_xx"), "got {rendered}");
        assert!(!rendered.contains(", x)"), "got {rendered}");
        assert!(!rendered.contains(", y)"), "got {rendered}");
        assert!(!rendered.contains(", z)"), "got {rendered}");
    }

    #[test]
    fn multifield_equations_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("multifield_equations(2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("multifield_curvature"), "got {rendered}");
        assert!(rendered.contains("multifield_entropy_1"), "got {rendered}");
    }

    #[test]
    fn boltzmann_bridge_builtin_returns_ten_pairs() {
        let (result, _) = eval_src("boltzmann_bridge();");
        match result {
            Expr::List(items) => assert_eq!(items.len(), 10),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn boltzmann_bridge_export_builtin_returns_code_string() {
        let (result, interner) = eval_src("boltzmann_bridge_export(python);");
        let Expr::Sym(code) = result else {
            panic!("expected interned code string");
        };
        assert!(interner.resolve(code).contains("def rhs_0("));
    }

    #[test]
    fn second_order_einstein_vector_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("second_order_einstein_vector();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("second_order_vector_x"), "got {rendered}");
        assert!(rendered.contains("second_order_vector_z"), "got {rendered}");
    }

    #[test]
    fn second_order_einstein_tensor_builtin_returns_labelled_list() {
        let (result, interner) = eval_src("second_order_einstein_tensor();");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered.contains("second_order_tensor_xx"),
            "got {rendered}"
        );
        assert!(
            rendered.contains("second_order_tensor_zz"),
            "got {rendered}"
        );
    }

    #[test]
    fn cubic_kernel_builtin_returns_nontrivial_expression() {
        let (result, interner) = eval_src("cubic_kernel(scalar_scalar_scalar);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("k1"), "got {rendered}");
        assert!(rendered.contains("epsilon"), "got {rendered}");
    }

    #[test]
    fn bispectrum_shape_builtin_returns_expression_without_raw_spatial_derivatives() {
        let (result, interner) = eval_src("bispectrum_shape(scalar_scalar_scalar, local);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("_x"), "got {rendered}");
        assert!(!rendered.contains("_y"), "got {rendered}");
        assert!(!rendered.contains("_z"), "got {rendered}");
        assert!(rendered.contains("p"), "got {rendered}");
        assert!(rendered.contains("q"), "got {rendered}");
    }

    #[test]
    fn export_cubic_vertex_builtin_returns_code_string() {
        let (result, interner) = eval_src("export_cubic_vertex(scalar_scalar_scalar, python);");
        let Expr::Sym(code) = result else {
            panic!("expected interned code string");
        };
        assert!(interner.resolve(code).contains("def cubic_vertex("));
    }

    #[test]
    fn eft_stability_builtin_returns_four_entries() {
        let (result, interner) = eval_src("eft_stability(canonical);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("ghost_free_scalar"), "got {rendered}");
        assert!(
            rendered.contains("gradient_stable_tensor"),
            "got {rendered}"
        );
    }

    #[test]
    fn eft_mode_equations_builtin_returns_two_labelled_equations() {
        let (result, interner) = eval_src("eft_mode_equations(reduced_sound_speed);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("eft_scalar_mode"), "got {rendered}");
        assert!(rendered.contains("eft_tensor_mode"), "got {rendered}");
    }

    #[test]
    fn eft_export_rhs_builtin_returns_code_string() {
        let (result, interner) = eval_src("eft_export_rhs(horndeski_like, python);");
        let Expr::Sym(code) = result else {
            panic!("expected interned code string");
        };
        let rendered = interner.resolve(code);
        assert!(rendered.contains("def eft_scalar_rhs("), "got {rendered}");
        assert!(rendered.contains("def eft_tensor_rhs("), "got {rendered}");
    }

    #[test]
    fn neutrino_hierarchy_builtin_returns_pairs() {
        let (result, _) = eval_src("neutrino_hierarchy(3, newtonian, power_law);");
        match result {
            Expr::List(items) => {
                assert_eq!(items.len(), 4);
                assert!(items
                    .iter()
                    .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2)));
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn photon_hierarchy_builtin_returns_pairs() {
        let (result, _) = eval_src("photon_hierarchy(3, synchronous, free_streaming);");
        match result {
            Expr::List(items) => {
                assert_eq!(items.len(), 4);
                assert!(items
                    .iter()
                    .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2)));
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn export_hierarchy_builtin_returns_payload_string() {
        let (result, interner) =
            eval_src("export_hierarchy(class_hook, neutrino, 3, newtonian, power_law);");
        let Expr::Sym(payload) = result else {
            panic!("expected interned payload string");
        };
        assert!(interner.resolve(payload).contains("\"target\":\"class\""));
    }

    #[test]
    fn cpt_parity_report_builtin_returns_nonempty_summary() {
        let (result, interner) = eval_src("cpt_parity_report();");
        let Expr::Sym(report) = result else {
            panic!("expected interned report string");
        };
        let rendered = interner.resolve(report);
        assert!(rendered.contains("ma_bertschinger_scalar_labels"));
        assert!(rendered.contains("tensor_mode_labels"));
    }

    #[test]
    fn structured_cpt_callables_remain_backward_compatible_with_simple_builtins() {
        let (simple, simple_interner) = eval_src("linearized_einstein(1);");
        let (structured, structured_interner) = eval_src(
            "cpt_linearized_einstein(1, frw_background_spec(conformal, flat, 3), cpt_gauge(newtonian), cpt_matter(symbolic));",
        );

        let Expr::List(simple_items) = simple else {
            panic!("expected simple builtin list");
        };
        let Expr::List(structured_items) = structured else {
            panic!("expected structured builtin list");
        };

        let simple_labels = simple_items
            .iter()
            .map(|item| match item {
                Expr::List(pair) if pair.len() == 2 => match &pair[0] {
                    Expr::Sym(label) => simple_interner.resolve(*label).to_string(),
                    other => panic!("expected label symbol, got {other:?}"),
                },
                other => panic!("expected labelled pair, got {other:?}"),
            })
            .collect::<Vec<_>>();
        let structured_labels = structured_items
            .iter()
            .map(|item| match item {
                Expr::List(pair) if pair.len() == 2 => match &pair[0] {
                    Expr::Sym(label) => structured_interner.resolve(*label).to_string(),
                    other => panic!("expected label symbol, got {other:?}"),
                },
                other => panic!("expected labelled pair, got {other:?}"),
            })
            .collect::<Vec<_>>();

        assert_eq!(simple_items.len(), structured_items.len());
        assert_eq!(simple_labels, structured_labels);
    }

    #[test]
    fn fixture_files_parse_and_evaluate_without_fallback() {
        let fixtures = [
            (
                "de_sitter.ax",
                vec![
                    "linearized_einstein",
                    "cpt_linearized_einstein",
                    "tensor_mode_equation",
                    "project_scalar_harmonics",
                    "mukhanov_sasaki",
                    "cpt_mukhanov_sasaki",
                ],
            ),
            (
                "radiation.ax",
                vec![
                    "cpt_linearized_einstein",
                    "cpt_fluid_equations",
                    "project_scalar_harmonics",
                ],
            ),
            (
                "matter.ax",
                vec![
                    "cpt_linearized_einstein",
                    "cpt_fluid_equations",
                    "project_scalar_harmonics",
                ],
            ),
            (
                "multifield_twofield.ax",
                vec![
                    "multifield_equations",
                    "cpt_mukhanov_sasaki",
                    "boltzmann_bridge",
                ],
            ),
        ];

        for (fixture, call_names) in fixtures {
            let (result, interner) = eval_fixture(fixture);
            assert!(
                !contains_unresolved_cpt_call(&result, &interner, &call_names),
                "fixture {fixture} left unresolved CPT callable in {:?}",
                result
            );
        }
    }

    #[test]
    fn cpt_quadratic_action_returns_non_symbolic_expr_for_canonical_scalar() {
        let (result, interner) = eval_src(
            "cpt_quadratic_action(frw_background_spec(conformal, flat, 3), cpt_matter(canonical_scalar));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(!rendered.contains("cpt_quadratic_action("));
        assert!(rendered.contains("v_eta"));
    }

    #[test]
    fn cpt_mukhanov_sasaki_first_order_returns_two_pairs() {
        let (result, _) = eval_src(
            "cpt_mukhanov_sasaki_first_order(frw_background_spec(conformal, flat, 3), cpt_matter(canonical_scalar));",
        );
        match result {
            Expr::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn cpt_export_mode_rhs_python_returns_interned_code_string_containing_def() {
        let (result, interner) = eval_src(
            "cpt_export_mode_rhs(python, frw_background_spec(conformal, flat, 3), cpt_matter(canonical_scalar));",
        );
        let Expr::Sym(code) = result else {
            panic!("expected interned code string");
        };
        assert!(interner.resolve(code).contains("def ms_rhs("));
    }

    #[test]
    fn legacy_simple_cosmology_builtins_still_work() {
        let (einstein, interner) = eval_src("linearized_einstein(1);");
        assert!(ax_ir::pretty_print(&einstein, &interner).contains("00_constraint"));

        let (ms, interner) = eval_src("mukhanov_sasaki();");
        assert!(ax_ir::pretty_print(&ms, &interner).contains("c_s"));
    }

    #[test]
    fn cosmology_black_hole_master_equations_evaluate_not_symbolic_calls() {
        let (zerilli, interner) = eval_src("zerilli(2);");
        let zerilli_rendered = ax_ir::pretty_print(&zerilli, &interner);
        assert!(
            !zerilli_rendered.contains("zerilli("),
            "expected evaluated Zerilli equation, got {zerilli_rendered}"
        );
        assert!(zerilli_rendered.contains("Psi_Z"), "got {zerilli_rendered}");

        let (regge_wheeler, interner) = eval_src("regge_wheeler(2);");
        let rw_rendered = ax_ir::pretty_print(&regge_wheeler, &interner);
        assert!(
            !rw_rendered.contains("regge_wheeler("),
            "expected evaluated Regge-Wheeler equation, got {rw_rendered}"
        );
        assert!(rw_rendered.contains("Psi_RW"), "got {rw_rendered}");
    }

    #[test]
    fn qm_pauli_commutator_matches_two_i_sigma_z() {
        let (result, interner) = eval_src("commutator(pauli_x(), pauli_y());");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "[[0 + 2*i, 0], [0, 0 + -2*i]]");
    }

    #[test]
    fn qm_bell_partial_trace_is_maximally_mixed() {
        let (result, interner) =
            eval_src("partial_trace(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2, A);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "[[1/2, 0], [0, 1/2]]");
    }

    #[test]
    fn qm_partial_trace_space_matches_partial_trace_factor() {
        let (space_result, _) = eval_src(
            "declare_hilbert_space(QA, 2); declare_hilbert_space(QB, 2); declare_composite_space(QAB, [QA, QB]); let rho = density([1/sqrt(2), 0, 0, 1/sqrt(2)]); partial_trace_space(rho, QAB, QB);",
        );
        let (factor_result, _) = eval_src(
            "let rho = density([1/sqrt(2), 0, 0, 1/sqrt(2)]); partial_trace_factor(rho, [2, 2], 1);",
        );
        assert_eq!(space_result, factor_result);
    }

    #[test]
    fn qm_lindblad_euler_step_with_zero_rhs_returns_input_matrix() {
        let (result, _) = eval_src("lindblad_euler_step([[1,0],[0,2]], [[1,0],[0,0]], [], 1/10);");
        assert_eq!(
            result,
            Expr::Matrix(vec![
                vec![Expr::one(), Expr::zero()],
                vec![Expr::zero(), Expr::zero()],
            ])
        );
    }

    #[test]
    fn qm_lindblad_rk4_step_with_zero_rhs_returns_input_matrix() {
        let (result, _) = eval_src("lindblad_rk4_step([[1,0],[0,2]], [[1,0],[0,0]], [], 1/10);");
        assert_eq!(
            result,
            Expr::Matrix(vec![
                vec![Expr::one(), Expr::zero()],
                vec![Expr::zero(), Expr::zero()],
            ])
        );
    }

    #[test]
    fn qm_lindblad_steady_state_amplitude_damping_returns_ground_state() {
        let (result, _) = eval_src("lindblad_steady_state([[0,0],[0,0]], [[[0,1],[0,0]]]);");
        assert_eq!(
            result,
            Expr::Matrix(vec![
                vec![Expr::one(), Expr::zero()],
                vec![Expr::zero(), Expr::zero()],
            ])
        );
    }

    #[test]
    fn qm_purity_of_bell_state_is_one() {
        let (result, _) = eval_src("purity(density([1/sqrt(2), 0, 0, 1/sqrt(2)]));");
        assert_eq!(result, Expr::one());
    }

    #[test]
    fn qm_hermitian_eigenvalues_pauli_z_returns_pm_one() {
        let (result, _) = eval_src("hermitian_eigenvalues([[1,0],[0,-1]]);");
        assert_eq!(
            result,
            Expr::List(vec![Expr::one(), Expr::neg(Expr::one())])
        );
    }

    #[test]
    fn qm_hermitian_eigenprojectors_pauli_z_returns_two_projectors() {
        let (result, _) = eval_src("hermitian_eigenprojectors([[1,0],[0,-1]]);");
        assert_eq!(
            result,
            Expr::List(vec![
                Expr::Matrix(vec![
                    vec![Expr::one(), Expr::zero()],
                    vec![Expr::zero(), Expr::zero()],
                ]),
                Expr::Matrix(vec![
                    vec![Expr::zero(), Expr::zero()],
                    vec![Expr::zero(), Expr::one()],
                ]),
            ])
        );
    }

    #[test]
    fn qm_hermitian_eigenvalues_nonhermitian_matrix_returns_exact_error_string() {
        let (result, interner) = eval_src("hermitian_eigenvalues([[0,1],[2,0]]);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "hermitian_eigenvalues expects a square Hermitian matrix of supported dimension"
        );
    }

    #[test]
    fn qm_linear_entropy_of_reduced_bell_state_is_one_half() {
        let (result, interner) = eval_src(
            "linear_entropy(partial_trace(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2, A));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1/2");
    }

    #[test]
    fn qm_participation_ratio_of_reduced_bell_state_is_two() {
        let (result, interner) = eval_src(
            "participation_ratio(partial_trace(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2, A));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "2");
    }

    #[test]
    fn qm_renyi2_entropy_of_reduced_bell_state_is_log_two() {
        let (result, interner) = eval_src(
            "renyi2_entropy(partial_trace(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2, A));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered == "log(2)" || rendered == "-log(1/2)" || rendered == "-1*log(1/2)");
    }

    #[test]
    fn qm_renyi2_entropy_factor_of_bell_state_is_log_two() {
        let (result, interner) = eval_src(
            "renyi2_entropy_factor(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), [2, 2], 0);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered == "log(2)" || rendered == "-log(1/2)" || rendered == "-1*log(1/2)");
    }

    #[test]
    fn qm_renyi2_entropy_factor_bad_dims_return_exact_error_string() {
        let (result, interner) = eval_src("renyi2_entropy_factor([[1,0],[0,0]], [2, 2], 0);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions"
        );
    }

    #[test]
    fn qm_von_neumann_entropy_maximally_mixed_qubit_is_log_two() {
        let (result, interner) = eval_src("von_neumann_entropy([[1/2,0],[0,1/2]]);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered == "log(2)" || rendered == "-log(1/2)" || rendered == "-1*log(1/2)");
    }

    #[test]
    fn qm_von_neumann_entropy_nonhermitian_matrix_returns_exact_error_string() {
        let (result, interner) = eval_src("von_neumann_entropy([[0,1],[2,0]]);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "von_neumann_entropy expects a supported square Hermitian density matrix"
        );
    }

    #[test]
    fn qm_mutual_information_bell_state_is_log_four() {
        let (result, interner) =
            eval_src("mutual_information(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered == "log(4)"
                || rendered == "2*log(2)"
                || rendered == "log(2) + log(2)"
                || rendered == "-2*log(1/2)"
                || rendered == "-log(1/2) + -log(1/2)"
                || rendered == "-1*log(1/2) + -1*log(1/2)"
                || rendered == "log(1) + -2*log(1/2)",
            "got {rendered}"
        );
    }

    #[test]
    fn qm_conditional_entropy_bell_state_is_minus_log_two() {
        let (result, interner) =
            eval_src("conditional_entropy(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered == "-log(2)"
                || rendered == "log(1/2)"
                || rendered == "-1*log(2)"
                || rendered == "0 + -1*log(1/2)"
                || rendered == "-1*log(1/2)",
            "got {rendered}"
        );
    }

    #[test]
    fn qm_mutual_information_bad_dims_return_exact_error_string() {
        let (result, interner) = eval_src("mutual_information([[1,0],[0,0]], 2, 2);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "mutual_information expects a bipartite density matrix of dimension dim_a * dim_b"
        );
    }

    #[test]
    fn qm_conditional_entropy_bad_dims_return_exact_error_string() {
        let (result, interner) = eval_src("conditional_entropy([[1,0],[0,0]], 2, 2);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "conditional_entropy expects a bipartite density matrix of dimension dim_a * dim_b"
        );
    }

    #[test]
    fn qm_entanglement_spectrum_bell_state_contains_half_twice() {
        let (result, _) =
            eval_src("entanglement_spectrum([1/sqrt(2), 0, 0, 1/sqrt(2)], 2, 2);");
        assert_eq!(
            result,
            Expr::List(vec![
                Expr::Rational(BigRational::new(1.into(), 2.into())),
                Expr::Rational(BigRational::new(1.into(), 2.into())),
            ])
        );
    }

    #[test]
    fn qm_schmidt_coefficients_bell_state_contain_inv_sqrt2_twice() {
        let (result, interner) =
            eval_src("schmidt_coefficients([1/sqrt(2), 0, 0, 1/sqrt(2)], 2, 2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered == "[sqrt(1/2), sqrt(1/2)]"
                || rendered == "[1/sqrt(2), 1/sqrt(2)]"
                || rendered == "[2^-1/2, 2^-1/2]",
            "got {rendered}"
        );
    }

    #[test]
    fn qm_entanglement_spectrum_bad_dimensions_return_exact_error_string() {
        let (result, interner) = eval_src("entanglement_spectrum([1,0,0], 2, 2);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "entanglement_spectrum expects a bipartite state vector or density matrix of dimension dim_a * dim_b"
        );
    }

    #[test]
    fn qm_schmidt_coefficients_bad_dimensions_return_exact_error_string() {
        let (result, interner) = eval_src("schmidt_coefficients([1,0,0], 2, 2);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "schmidt_coefficients expects a bipartite pure-state vector of dimension dim_a * dim_b"
        );
    }

    #[test]
    fn qm_negativity_bell_state_is_one_half() {
        let (result, interner) = eval_src(
            "negativity(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1/2");
    }

    #[test]
    fn qm_logarithmic_negativity_bell_state_is_log_two() {
        let (result, interner) = eval_src(
            "logarithmic_negativity(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2);",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "log(2)");
    }

    #[test]
    fn qm_renyi2_mutual_information_of_bell_state_is_log_four() {
        let (result, interner) =
            eval_src("renyi2_mutual_information(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2);");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(
            rendered == "log(4)"
                || rendered == "2*log(2)"
                || rendered == "log(2) + log(2)"
                || rendered == "-2*log(1/2)"
                || rendered == "-log(1/2) + -log(1/2)"
                || rendered == "-1*log(1/2) + -1*log(1/2)"
                || rendered == "log(1) + -2*log(1/2)",
            "got {rendered}"
        );
    }

    #[test]
    fn qm_renyi2_tripartite_information_product_state_is_zero() {
        let (result, _) =
            eval_src("renyi2_tripartite_information(density([1, 0, 0, 0, 0, 0, 0, 0]), 2, 2, 2);");
        assert_eq!(result, Expr::zero());
    }

    #[test]
    fn qm_renyi2_tripartite_information_bad_dims_return_exact_error_string() {
        let (result, interner) =
            eval_src("renyi2_tripartite_information([[1,0],[0,0]], 2, 2, 2);");
        let Expr::Sym(message) = result else {
            panic!("expected interned error string");
        };
        assert_eq!(
            interner.resolve(message),
            "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c"
        );
    }

    #[test]
    fn qm_bloch_vector_of_zero_state_is_001() {
        let (result, _) = eval_src("bloch_vector([[1,0],[0,0]]);");
        assert_eq!(
            result,
            Expr::List(vec![Expr::zero(), Expr::zero(), Expr::one()])
        );
    }

    #[test]
    fn qm_qubit_density_from_bloch_z_axis_is_zero_state() {
        let (result, _) = eval_src("qubit_density_from_bloch([0, 0, 1]);");
        assert_eq!(
            result,
            Expr::Matrix(vec![
                vec![Expr::one(), Expr::zero()],
                vec![Expr::zero(), Expr::zero()],
            ])
        );
    }

    #[test]
    fn qm_normal_order_adds_bosonic_commutator_term() {
        let (result, interner) = eval_src("normal_order(annihilation(a) * creation(a));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1 + creation(a)*annihilation(a)");
    }

    #[test]
    fn qm_normal_order_adds_fermionic_anticommutator_sign() {
        let (result, interner) = eval_src(
            "declare_operator(c, annihilation, fermionic); normal_order(annihilation(c) * creation(c));",
        );
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1 + -1*creation(c)*annihilation(c)");
    }

    #[test]
    fn qm_wick_uses_declared_contraction_term() {
        let (result, interner) =
            eval_src("declare_contraction(a, a, 1); wick(annihilation(a) * creation(a));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1 + creation(a)*annihilation(a)");
    }

    #[test]
    fn qm_wick_without_declared_contraction_reduces_to_normal_ordering() {
        let (result, interner) = eval_src("wick(annihilation(a) * creation(a));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "1 + creation(a)*annihilation(a)");
    }

    #[test]
    fn qm_abstract_number_operator_acts_diagonally_on_fock_states() {
        let (result, interner) =
            eval_src("apply_operator(number_operator(a), number_state(a, 2));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "2*number_state(a, 2)");
    }

    #[test]
    fn qm_abstract_harmonic_oscillator_hamiltonian_has_correct_energy() {
        let (result, interner) =
            eval_src("apply_operator(hamiltonian_ho(a, hbar, omega), number_state(a, 1));");
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert_eq!(rendered, "3/2*hbar*omega*number_state(a, 1)");
    }

    #[test]
    fn qm_creation_and_annihilation_act_on_vacuum_and_excited_states() {
        let (created, interner) = eval_src("apply_operator(creation(a), vacuum(a));");
        let created_rendered = ax_ir::pretty_print(&created, &interner);
        assert_eq!(created_rendered, "number_state(a, 1)");

        let (annihilated, interner) =
            eval_src("apply_operator(annihilation(a), number_state(a, 2));");
        let annihilated_rendered = ax_ir::pretty_print(&annihilated, &interner);
        assert_eq!(annihilated_rendered, "2^1/2*number_state(a, 1)");

        let (vacuum_lowered, _) = eval_src("apply_operator(annihilation(a), vacuum(a));");
        assert_eq!(vacuum_lowered, Expr::zero());
    }

    #[test]
    fn schwarzschild_ricci_demo_collapses_to_zero_matrix() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower(
            "ricci(riemann(christoffel(metric(diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi]));",
            &interner,
        )
        .expr
        .expect("schwarzschild ricci expr");
        let result = eval(&expr, &env, &interner);
        match result {
            Expr::Matrix(rows) => {
                assert!(
                    rows.iter().flatten().all(|entry| *entry == Expr::zero()),
                    "expected zero Ricci matrix, got {:?}",
                    rows
                );
            }
            other => panic!("expected Schwarzschild Ricci matrix, got {:?}", other),
        }
    }

    #[test]
    fn schwarzschild_kretschner_demo_collapses_to_closed_form() {
        let interner = ax_ir::Interner::new();
        let env = Env::new();
        let expr = ax_core_ir::lower(
            "kretschner(riemann(christoffel(metric(diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 * sin(theta)^2)), [t, r, theta, phi]), [t, r, theta, phi]), metric(diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 * sin(theta)^2)));",
            &interner,
        )
        .expr
        .expect("schwarzschild kretschner expr");
        let result = eval(&expr, &env, &interner);
        let expected = Expr::mul(vec![
            Expr::Int(48.into()),
            Expr::pow(Expr::Sym(interner.get_or_intern("M")), Expr::Int(2.into())),
            Expr::pow(
                Expr::Sym(interner.get_or_intern("r")),
                Expr::Int((-6).into()),
            ),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn metric_from_vielbein_eval_reconstructs_metric() {
        let (result, interner) = eval_src(
            "metric_from_vielbein(vielbein([[sqrt(f), 0], [0, sqrt(f^(-2))]]), metric(diag(-1, 1)));",
        );
        let expected = Expr::Matrix(vec![
            vec![
                Expr::neg(Expr::Sym(interner.get_or_intern("f"))),
                Expr::zero(),
            ],
            vec![
                Expr::zero(),
                Expr::pow(
                    Expr::Sym(interner.get_or_intern("f")),
                    Expr::Int((-2).into()),
                ),
            ],
        ]);
        assert_eq!(result, expected, "got {:?}", result);
    }

    #[test]
    fn rewrite_indices_vielbein_eval_inserts_frame_factors() {
        let (result, interner) = eval_src(
            "indices spacetime [mu] dim=1;
             indices frame [a] dim=1;
             rewrite_indices_vielbein(V[mu+], E, EInv, spacetime, frame);",
        );
        assert!(matches!(result, Expr::Mul(_)), "got {:?}", result);
        let rendered = ax_ir::pretty_print(&result, &interner);
        assert!(rendered.contains("E["), "got {rendered}");
        assert!(rendered.contains("V["), "got {rendered}");
    }
}
