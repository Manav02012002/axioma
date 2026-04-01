#![forbid(unsafe_code)]

pub mod integrate;
pub mod limits;
pub mod series;
pub mod simplify;

use ax_ir::{Assumption, Condition, Expr, Grading};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct Env {
    pub bindings: HashMap<lasso::Spur, Expr>,
    pub parent: Option<Box<Env>>,
    pub rules: Vec<ax_rewrite::RewriteRule>,
    pub assumptions: HashMap<lasso::Spur, Vec<Assumption>>,
    pub gradings: HashMap<lasso::Spur, Grading>,
    pub operators: HashMap<lasso::Spur, ax_qm::OperatorKind>,
    pub coordinates: HashSet<lasso::Spur>,
    pub index_families: HashMap<lasso::Spur, ax_ir::IndexFamily>,
    pub index_to_family: HashMap<lasso::Spur, lasso::Spur>,
    pub tensor_properties: HashMap<lasso::Spur, Vec<ax_ir::TensorProperty>>,
    pub convention: ax_ir::Convention,
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
            coordinates: HashSet::new(),
            index_families: HashMap::new(),
            index_to_family: HashMap::new(),
            tensor_properties: HashMap::new(),
            convention: ax_ir::Convention::default(),
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
            coordinates: self.coordinates.clone(),
            index_families: self.index_families.clone(),
            index_to_family: self.index_to_family.clone(),
            tensor_properties: self.tensor_properties.clone(),
            convention: self.convention.clone(),
        }
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
    fn coordinates(&self) -> &HashSet<lasso::Spur> {
        &self.coordinates
    }

    fn index_to_family(&self) -> &HashMap<lasso::Spur, lasso::Spur> {
        &self.index_to_family
    }
}

fn parse_component_rules(
    rule_exprs: &[Expr],
) -> Vec<ax_tensor::ComponentRule> {
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
                    indices.push((concrete_indices[0].name, concrete_indices[0].variance.clone()));
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

fn grading_rank(grading: Grading) -> usize {
    match grading {
        Grading::Even => 0,
        Grading::Odd => 1,
    }
}

pub fn infer_grading(expr: &Expr, gradings: &HashMap<lasso::Spur, Grading>) -> Grading {
    match expr {
        Expr::Sym(sym) => gradings.get(sym).copied().unwrap_or(Grading::Even),
        Expr::Neg(inner) => infer_grading(inner, gradings),
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
        Some(format!("declared Grassmann variables: {}", declared.join(", ")))
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
        Some(format!("declared {kind_name} operators: {}", declared.join(", ")))
    }
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
    if interner.resolve(*f) != "__declare_property" || args.len() != 2 {
        return None;
    }
    let (Expr::Sym(tensor), Expr::Sym(prop)) = (&args[0], &args[1]) else {
        return None;
    };
    let prop_name = interner.resolve(*prop);
    let entry = env.tensor_properties.entry(*tensor).or_default();
    match prop_name {
        "metric" => {
            entry.push(ax_ir::TensorProperty::Metric);
            entry.push(ax_ir::TensorProperty::Symmetric(vec![0, 1]));
            Some(format!(
                "attached property metric (symmetric) to {}",
                interner.resolve(*tensor)
            ))
        }
        "symmetric" => {
            entry.push(ax_ir::TensorProperty::Symmetric(vec![0, 1]));
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "antisymmetric" => {
            entry.push(ax_ir::TensorProperty::AntiSymmetric(vec![0, 1]));
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "inverse_metric" => {
            entry.push(ax_ir::TensorProperty::InverseMetric);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "kronecker_delta" | "kronecker" => {
            entry.push(ax_ir::TensorProperty::KroneckerDelta);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "epsilon" | "epsilon_tensor" => {
            entry.push(ax_ir::TensorProperty::EpsilonTensor);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "riemann" | "riemann_symmetry" => {
            entry.push(ax_ir::TensorProperty::RiemannSymmetry);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "traceless" => {
            entry.push(ax_ir::TensorProperty::Traceless);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "derivative" => {
            entry.push(ax_ir::TensorProperty::Derivative);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "partial_derivative" => {
            entry.push(ax_ir::TensorProperty::PartialDerivative);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
        }
        "covariant_derivative" => {
            entry.push(ax_ir::TensorProperty::CovariantDerivative);
            Some(format!(
                "attached property {} to {}",
                interner.resolve(*prop),
                interner.resolve(*tensor)
            ))
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
    Some(format!(
        "declared index family: {}",
        interner.resolve(*family_name)
    ))
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
        Condition::Not(c) => {
            Condition::Not(Box::new(substitute_condition(c, target, replacement, interner)))
        }
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
        Condition::Not(c) => {
            Condition::Not(Box::new(multi_substitute_condition(c, substitutions, interner)))
        }
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
            terms.iter()
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
            terms.iter()
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
        Expr::Neg(e) => contains_var(e, var),
        Expr::Call(_, args) => args.iter().any(|arg| contains_var(arg, var)),
        Expr::FnDef(_, _, body) => contains_var(body, var),
        Expr::Rule(lhs, rhs, _) => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::Import(_) => false,
        Expr::Assume(_, _) => false,
        Expr::SetConvention(_, _) => false,
        Expr::Piecewise(cases) => cases
            .iter()
            .any(|(value, condition)| contains_var(value, var) || condition_contains_var(condition, var)),
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
                if im == 0.0 { Some(re) } else { None }
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
            sample_env.bindings.insert(*sym, Expr::Float(base[(sample + idx) % base.len()]));
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
    if successes == 5 { Some(true) } else { None }
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

    let diff = canonical_equiv_form(&Expr::add(vec![sa.clone(), Expr::neg(sb.clone())]), interner);
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
            &ax_tensor::canonicalize_indices(&ta, &env.tensor_properties, interner),
            env,
            interner,
        );
        let cb = ax_tensor::rename_dummies(
            &ax_tensor::canonicalize_indices(&tb, &env.tensor_properties, interner),
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
        Condition::And(a, b) => Some(eval_condition(a, env, interner)? && eval_condition(b, env, interner)?),
        Condition::Or(a, b) => Some(eval_condition(a, env, interner)? || eval_condition(b, env, interner)?),
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
    let result = ax_rewrite::rewrite_fixed_point_traced(&env.rules, expr, interner, 100, &mut trace);
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
        format!("trust: {overall} (used rule{}: {used})", if trace.steps.len() == 1 { "" } else { "s" })
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
            cases.iter()
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
                    Expr::Complex(re, im) => {
                        Expr::Complex(Box::new(re.as_ref().clone()), Box::new(Expr::neg(im.as_ref().clone())))
                    }
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
                    Expr::Float(v) => Expr::Float(v.sin()),
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
                    Expr::Float(v) => Expr::Float(v.cos()),
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
                    Expr::Float(v) if *v > 0.0 => Expr::Float(v.ln()),
                    Expr::Int(n) if n.is_one() => Expr::Int(0.into()),
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
                    Expr::Int(n) => {
                        if let Some(root) = perfect_square_root(n) {
                            Expr::Int(root)
                        } else {
                            Expr::pow(args[0].clone(), one_half())
                        }
                    }
                    Expr::Pow(base, exp) if matches!(exp.as_ref(), Expr::Int(n) if *n == 2.into()) => {
                        match base.as_ref() {
                            Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Positive) => {
                                Expr::Sym(*sym)
                            }
                            Expr::Sym(sym) if has_assumption(env, *sym, &Assumption::Real) => {
                                Expr::Call(interner.get_or_intern("abs"), vec![Expr::Sym(*sym)])
                            }
                            _ => Expr::pow(args[0].clone(), one_half()),
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
                let canonical = ax_tensor::canonicalise(&args[0], &env.tensor_properties, interner);
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
                ax_tensor::meld(&args[0], &env.tensor_properties, interner)
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
                ax_tensor::sort_product(&args[0], &env.tensor_properties, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "product_rule" | "leibniz" => {
            if args.len() == 1 {
                let deriv_syms: HashSet<lasso::Spur> = env
                    .tensor_properties
                    .iter()
                    .filter(|(_, props)| {
                        props.iter().any(|p| {
                            matches!(
                                p,
                                ax_ir::TensorProperty::Derivative
                                    | ax_ir::TensorProperty::PartialDerivative
                                    | ax_ir::TensorProperty::CovariantDerivative
                            )
                        })
                    })
                    .map(|(sym, _)| *sym)
                    .collect();
                ax_tensor::product_rule(&args[0], &deriv_syms, interner)
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
        "eliminate_metric" => {
            if !args.is_empty() {
                let metric = interner.get_or_intern("g");
                let inv_metric = interner.get_or_intern("ginv");
                ax_tensor::eliminate_metric(&args[0], metric, inv_metric, interner)
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
                    ax_tensor::evaluate_components(&args[0], &rules, &index_vals, env, interner)
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
                    (Expr::Sym(from), Expr::Sym(to)) => {
                        match (units.get(from), units.get(to)) {
                            (Some(from_unit), Some(to_unit)) => {
                                match ax_units::convert(&args[0], from_unit, to_unit) {
                                    Ok(expr) => eval(&expr, &Env::new(), interner),
                                    Err(_) => Expr::Call(f, args),
                                }
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
        "symmetric" | "antisymmetric" | "riemann_symmetry" | "traceless" => {
            Expr::Call(f, args)
        }
        "__declare_indices" | "__declare_coordinates" | "__declare_property" => Expr::Call(f, args),
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
                    _ => symbolic_substitute(&args[0], &args[1], &args[2], interner),
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
                Expr::Sym(interner.get_or_intern(&equiv_description(&args[0], &args[1], env, interner)))
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
                println!("{}", ax_codegen::generate(&args[0], ax_codegen::Target::Python, interner, None, &[]));
                Expr::zero()
            } else {
                Expr::Call(f, args)
            }
        }
        "to_rust" => {
            if args.len() == 1 {
                println!("{}", ax_codegen::generate(&args[0], ax_codegen::Target::Rust, interner, None, &[]));
                Expr::zero()
            } else {
                Expr::Call(f, args)
            }
        }
        "to_cpp" => {
            if args.len() == 1 {
                println!("{}", ax_codegen::generate(&args[0], ax_codegen::Target::Cpp, interner, None, &[]));
                Expr::zero()
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
        "ket" => {
            match args.as_slice() {
                [Expr::Int(n)] => {
                    if let Some(index) = n.to_usize() {
                        Expr::List(ax_qm::ket(index, 2))
                    } else {
                        Expr::Call(f, args)
                    }
                }
                [Expr::Int(n), Expr::Int(d)] => {
                    match (n.to_usize(), d.to_usize()) {
                        (Some(index), Some(dim)) => Expr::List(ax_qm::ket(index, dim)),
                        _ => Expr::Call(f, args),
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
        "bra" => {
            match args.as_slice() {
                [Expr::Int(n)] => {
                    if let Some(index) = n.to_usize() {
                        Expr::List(ax_qm::bra(index, 2))
                    } else {
                        Expr::Call(f, args)
                    }
                }
                [Expr::Int(n), Expr::Int(d)] => {
                    match (n.to_usize(), d.to_usize()) {
                        (Some(index), Some(dim)) => Expr::List(ax_qm::bra(index, dim)),
                        _ => Expr::Call(f, args),
                    }
                }
                _ => Expr::Call(f, args),
            }
        }
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
                            (Some(dim_a), Some(dim_b), "A") => {
                                Expr::Matrix(ax_qm::partial_trace(rho, dim_a, dim_b, 'A', interner))
                            }
                            (Some(dim_a), Some(dim_b), "B") => {
                                Expr::Matrix(ax_qm::partial_trace(rho, dim_a, dim_b, 'B', interner))
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
        "creation" | "annihilation" => Expr::Call(f, args),
        "normal_order" => {
            if args.len() == 1 {
                ax_qm::normal_order(&args[0], &env.operators, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "wick" => {
            if args.len() == 1 {
                let contractions = HashMap::new();
                ax_qm::wick_expand(&args[0], &env.operators, &contractions, interner)
            } else {
                Expr::Call(f, args)
            }
        }
        "euler_lagrange" => {
            if args.len() == 4 {
                match (&args[0], &args[1], &args[2], &args[3]) {
                    (lagrangian, Expr::Sym(field), Expr::List(field_derivs), Expr::List(coords)) => {
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
        "vary" => {
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
                items.iter()
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
                items.iter()
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
                    (Some(a), Some(b)) => ax_forms::form_to_expr(&ax_forms::wedge(&a, &b, interner)),
                    _ => Expr::Call(f, args),
                }
            } else {
                Expr::Call(f, args)
            }
        }
        "exterior_d" | "d" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (field, Expr::List(coords_exprs)) => {
                        let coords = coords_exprs
                            .iter()
                            .map(|expr| match expr {
                                Expr::Sym(sym) => Some(*sym),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let Some(coords) = coords {
                            let form = if let Some(one_form) = ax_forms::one_form_from_expr(field) {
                                one_form
                            } else {
                                ax_forms::scalar_form(field, coords.len())
                            };
                            ax_forms::form_to_expr(&ax_forms::exterior_derivative(
                                &form,
                                &coords,
                                interner,
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
        "hodge_star" => {
            if args.len() == 2 {
                match (&args[0], matrix_to_symbolic(&args[1])) {
                    (field, Some(metric)) => {
                        let form = if let Some(one_form) = ax_forms::one_form_from_expr(field) {
                            one_form
                        } else if let Some(two_form) = ax_forms::two_form_from_expr(field) {
                            two_form
                        } else {
                            ax_forms::scalar_form(field, metric.dim)
                        };
                        ax_forms::form_to_expr(&ax_forms::hodge_dual(&form, &metric, interner))
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
                                &gamma, &coords, interner, &env.convention,
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
                    Expr::Matrix(ax_tensor::ricci_from_riemann(
                        &riemann,
                        n,
                        interner,
                        &env.convention,
                    ))
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
        "covariant_diff" => {
            if args.len() == 4 {
                match (&args[0], expr_to_3d(&args[1]), &args[2], &args[3]) {
                    (Expr::List(v), Some(gamma), Expr::Int(coord_index), Expr::List(coords_exprs)) => {
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
                    (Expr::Matrix(t), Some(gamma), Expr::Int(coord_index), Expr::List(coords_exprs)) => {
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
                                    w,
                                    v,
                                    &coords,
                                    interner,
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
                    _ => Expr::Call(f, args),
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
        _ => Expr::Call(f, args),
    }
}

pub fn eval(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> Expr {
    match expr {
        Expr::Int(n) => Expr::Int(n.clone()),
        Expr::Rational(r) => Expr::Rational(r.clone()),
        Expr::Float(f) => Expr::Float(*f),
        Expr::Complex(re, im) => {
            Expr::Complex(Box::new(eval(re, env, interner)), Box::new(eval(im, env, interner)))
        }
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
        Expr::Call(f, args) => {
            let evaled_args: Vec<Expr> = args.iter().map(|arg| eval(arg, env, interner)).collect();
            let name = interner.resolve(*f);
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
                            cases.iter()
                                .map(|(value, condition)| (eval(value, env, interner), condition.clone()))
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
            let canonical = ax_tensor::canonicalize_indices(&indexed, &env.tensor_properties, interner);
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

fn expr_to_3d(expr: &Expr) -> Option<Vec<Vec<Vec<Expr>>>> {
    let Expr::List(level1) = expr else {
        return None;
    };
    level1
        .iter()
        .map(|item| {
            let Expr::List(level2) = item else {
                return None;
            };
            level2
                .iter()
                .map(|row| {
                    let Expr::List(level3) = row else {
                        return None;
                    };
                    Some(level3.clone())
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

fn expr_3d_to_list(data: Vec<Vec<Vec<Expr>>>) -> Expr {
    Expr::List(
        data.into_iter()
            .map(|level2| Expr::List(level2.into_iter().map(Expr::List).collect()))
            .collect(),
    )
}

fn expr_to_4d(expr: &Expr) -> Option<Vec<Vec<Vec<Vec<Expr>>>>> {
    let Expr::List(level1) = expr else {
        return None;
    };
    level1
        .iter()
        .map(|item| {
            let Expr::List(level2) = item else {
                return None;
            };
            level2
                .iter()
                .map(|item2| {
                    let Expr::List(level3) = item2 else {
                        return None;
                    };
                    level3
                        .iter()
                        .map(|item3| {
                            let Expr::List(level4) = item3 else {
                                return None;
                            };
                            Some(level4.clone())
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_src(src: &str) -> (ax_ir::Expr, ax_ir::Interner) {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        assert!(
            result.errors.is_empty(),
            "lower errors: {:?}",
            result.errors
        );
        let expr = result.expr.expect("expected expression");
        let env = Env::new();
        (eval(&expr, &env, &interner), interner)
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
}
