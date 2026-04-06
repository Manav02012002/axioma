use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::HashMap;

pub mod twistor;

/// Particle label used by spinor-helicity expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub struct Label(pub u16);

impl Label {
    pub fn new(n: u16) -> Self {
        Self(n)
    }

    pub fn index(&self) -> u16 {
        self.0
    }
}

/// Spinor-helicity expressions for massless amplitudes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpinorExpr {
    /// Angle bracket <i j>.
    AngleBracket(Label, Label),
    /// Square bracket [i j].
    SquareBracket(Label, Label),
    /// Same-chirality angle chain <i| p_k1 p_k2 ... |j>.
    AngleChain(Label, Vec<Label>, Label),
    /// Same-chirality square chain [i| p_k1 p_k2 ... |j].
    SquareChain(Label, Vec<Label>, Label),
    /// Mixed chain <i| p_k1 ... |j].
    AngleSquareChain(Label, Vec<Label>, Label),
    /// Mixed chain [i| p_k1 ... |j>.
    SquareAngleChain(Label, Vec<Label>, Label),
    /// Two-particle Mandelstam invariant s_ij = (p_i + p_j)^2.
    Mandelstam(Label, Label),
    /// Three-particle Mandelstam invariant s_ijk = (p_i + p_j + p_k)^2.
    Mandelstam3(Label, Label, Label),
    /// Product of spinor terms.
    Product(Vec<SpinorTerm>),
    /// Sum of spinor expressions.
    Sum(Vec<SpinorExpr>),
    /// Ratio of spinor expressions.
    Ratio(Box<SpinorExpr>, Box<SpinorExpr>),
    /// Rational numerical coefficient.
    Numeric(BigRational),
    /// Integer power.
    Power(Box<SpinorExpr>, i32),
    /// Negation.
    Neg(Box<SpinorExpr>),
}

/// A coefficient times a product of spinor factors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpinorTerm {
    pub coefficient: BigRational,
    pub factors: Vec<SpinorFactor>,
}

/// Atomic factor in a spinor-helicity product.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpinorFactor {
    /// Angle bracket <i j>.
    Angle(Label, Label),
    /// Square bracket [i j].
    Square(Label, Label),
    /// Mixed chain <i|...|j].
    AngleSquare(Label, Vec<Label>, Label),
    /// Mixed chain [i|...|j>.
    SquareAngle(Label, Vec<Label>, Label),
    /// Mandelstam invariant s_{i1 i2 ...}.
    Mandelstam(Vec<Label>),
    /// Integer power of a factor.
    Power(Box<SpinorFactor>, i32),
    /// Parenthesized spinor expression used when factoring sums.
    Grouped(Box<SpinorExpr>),
    /// Symbolic scalar parameter, such as the BCFW shift parameter z.
    SymbolicParam(lasso::Spur),
}

/// Bidirectional map between symbolic particle names and compact labels.
#[derive(Clone, Debug)]
pub struct LabelMap {
    to_name: Vec<lasso::Spur>,
    from_name: HashMap<lasso::Spur, Label>,
}

impl LabelMap {
    pub fn new() -> Self {
        Self {
            to_name: Vec::new(),
            from_name: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: lasso::Spur) -> Label {
        if let Some(label) = self.from_name.get(&name) {
            return *label;
        }

        let next = self
            .to_name
            .len()
            .try_into()
            .expect("too many spinor labels to fit in u16");
        let label = Label::new(next);
        self.to_name.push(name);
        self.from_name.insert(name, label);
        label
    }

    pub fn label_for(&self, name: lasso::Spur) -> Option<Label> {
        self.from_name.get(&name).copied()
    }

    pub fn name_for(&self, label: Label) -> Option<lasso::Spur> {
        self.to_name.get(label.0 as usize).copied()
    }

    pub fn len(&self) -> usize {
        self.to_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_name.is_empty()
    }
}

impl Default for LabelMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SpinorExpr {
    pub fn angle(i: Label, j: Label) -> Self {
        Self::AngleBracket(i, j)
    }

    pub fn square(i: Label, j: Label) -> Self {
        Self::SquareBracket(i, j)
    }

    pub fn s(i: Label, j: Label) -> Self {
        Self::Mandelstam(i, j)
    }

    pub fn is_zero(&self) -> bool {
        match self {
            SpinorExpr::AngleBracket(i, j) | SpinorExpr::SquareBracket(i, j) => i == j,
            SpinorExpr::Numeric(n) => n.is_zero(),
            SpinorExpr::Neg(inner) => inner.is_zero(),
            SpinorExpr::Product(terms) => terms.iter().any(SpinorTerm::is_zero),
            SpinorExpr::Sum(terms) => terms.iter().all(SpinorExpr::is_zero),
            SpinorExpr::Power(base, n) if *n > 0 => base.is_zero(),
            _ => false,
        }
    }

    pub fn mass_dimension(&self) -> i32 {
        match self {
            SpinorExpr::AngleBracket(_, _) | SpinorExpr::SquareBracket(_, _) => 1,
            SpinorExpr::AngleChain(_, middle, _)
            | SpinorExpr::SquareChain(_, middle, _)
            | SpinorExpr::AngleSquareChain(_, middle, _)
            | SpinorExpr::SquareAngleChain(_, middle, _) => 1 + middle.len() as i32,
            SpinorExpr::Mandelstam(_, _) | SpinorExpr::Mandelstam3(_, _, _) => 2,
            SpinorExpr::Product(terms) => terms.iter().map(SpinorTerm::mass_dimension).sum(),
            SpinorExpr::Sum(terms) => terms
                .first()
                .map(SpinorExpr::mass_dimension)
                .unwrap_or_default(),
            SpinorExpr::Ratio(num, den) => num.mass_dimension() - den.mass_dimension(),
            SpinorExpr::Numeric(_) => 0,
            SpinorExpr::Power(base, n) => base.mass_dimension() * *n,
            SpinorExpr::Neg(inner) => inner.mass_dimension(),
        }
    }

    pub fn little_group_weight(&self, particle: Label) -> i32 {
        match self {
            SpinorExpr::AngleBracket(i, j) => label_hits(*i, *j, particle),
            SpinorExpr::SquareBracket(i, j) => -label_hits(*i, *j, particle),
            SpinorExpr::AngleChain(i, _, j) => label_hits(*i, *j, particle),
            SpinorExpr::SquareChain(i, _, j) => -label_hits(*i, *j, particle),
            SpinorExpr::AngleSquareChain(i, _, j) => {
                single_label_hit(*i, particle) - single_label_hit(*j, particle)
            }
            SpinorExpr::SquareAngleChain(i, _, j) => {
                -single_label_hit(*i, particle) + single_label_hit(*j, particle)
            }
            SpinorExpr::Mandelstam(_, _) | SpinorExpr::Mandelstam3(_, _, _) => 0,
            SpinorExpr::Product(terms) => terms
                .iter()
                .map(|term| term.little_group_weight(particle))
                .sum(),
            SpinorExpr::Sum(terms) => terms
                .first()
                .map(|term| term.little_group_weight(particle))
                .unwrap_or_default(),
            SpinorExpr::Ratio(num, den) => {
                num.little_group_weight(particle) - den.little_group_weight(particle)
            }
            SpinorExpr::Numeric(_) => 0,
            SpinorExpr::Power(base, n) => base.little_group_weight(particle) * *n,
            SpinorExpr::Neg(inner) => inner.little_group_weight(particle),
        }
    }
}

impl SpinorTerm {
    pub fn new(coefficient: BigRational, factors: Vec<SpinorFactor>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero() || self.factors.iter().any(SpinorFactor::is_zero)
    }

    pub fn mass_dimension(&self) -> i32 {
        self.factors.iter().map(SpinorFactor::mass_dimension).sum()
    }

    pub fn little_group_weight(&self, particle: Label) -> i32 {
        self.factors
            .iter()
            .map(|factor| factor.little_group_weight(particle))
            .sum()
    }
}

impl SpinorFactor {
    pub fn is_zero(&self) -> bool {
        match self {
            SpinorFactor::Angle(i, j) | SpinorFactor::Square(i, j) => i == j,
            SpinorFactor::Power(base, n) if *n > 0 => base.is_zero(),
            SpinorFactor::Grouped(expr) => expr.is_zero(),
            _ => false,
        }
    }

    pub fn mass_dimension(&self) -> i32 {
        match self {
            SpinorFactor::Angle(_, _) | SpinorFactor::Square(_, _) => 1,
            SpinorFactor::AngleSquare(_, middle, _) | SpinorFactor::SquareAngle(_, middle, _) => {
                1 + middle.len() as i32
            }
            SpinorFactor::Mandelstam(_) => 2,
            SpinorFactor::Power(base, n) => base.mass_dimension() * *n,
            SpinorFactor::Grouped(expr) => expr.mass_dimension(),
            SpinorFactor::SymbolicParam(_) => 0,
        }
    }

    pub fn little_group_weight(&self, particle: Label) -> i32 {
        match self {
            SpinorFactor::Angle(i, j) => label_hits(*i, *j, particle),
            SpinorFactor::Square(i, j) => -label_hits(*i, *j, particle),
            SpinorFactor::AngleSquare(i, _, j) => {
                single_label_hit(*i, particle) - single_label_hit(*j, particle)
            }
            SpinorFactor::SquareAngle(i, _, j) => {
                -single_label_hit(*i, particle) + single_label_hit(*j, particle)
            }
            SpinorFactor::Mandelstam(_) => 0,
            SpinorFactor::Power(base, n) => base.little_group_weight(particle) * *n,
            SpinorFactor::Grouped(expr) => expr.little_group_weight(particle),
            SpinorFactor::SymbolicParam(_) => 0,
        }
    }
}

pub fn canonicalise_bracket(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::AngleBracket(i, j) if i > j => neg_expr(SpinorExpr::AngleBracket(*j, *i)),
        SpinorExpr::SquareBracket(i, j) if i > j => neg_expr(SpinorExpr::SquareBracket(*j, *i)),
        SpinorExpr::Product(terms) => SpinorExpr::Product(
            terms
                .iter()
                .map(|term| {
                    let mut coefficient = term.coefficient.clone();
                    let factors = term
                        .factors
                        .iter()
                        .map(|factor| canonicalise_factor_bracket(factor, &mut coefficient))
                        .collect();
                    SpinorTerm {
                        coefficient,
                        factors,
                    }
                })
                .collect(),
        ),
        SpinorExpr::Sum(terms) => {
            let terms = terms.iter().map(canonicalise_bracket).collect();
            SpinorExpr::Sum(terms)
        }
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(canonicalise_bracket(num)),
            Box::new(canonicalise_bracket(den)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(canonicalise_bracket(base)), *n),
        SpinorExpr::Neg(inner) => match canonicalise_bracket(inner) {
            SpinorExpr::Neg(nested) => *nested,
            other => SpinorExpr::Neg(Box::new(other)),
        },
        _ => expr.clone(),
    }
}

pub fn apply_schouten(expr: &SpinorExpr, a: Label, b: Label, c: Label, d: Label) -> SpinorExpr {
    apply_schouten_for_kind(expr, a, b, c, d, BracketKind::Angle)
}

pub fn apply_schouten_square(
    expr: &SpinorExpr,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
) -> SpinorExpr {
    apply_schouten_for_kind(expr, a, b, c, d, BracketKind::Square)
}

pub fn apply_momentum_conservation(
    expr: &SpinorExpr,
    n_particles: u16,
    eliminate: Label,
) -> SpinorExpr {
    match expr {
        SpinorExpr::AngleChain(left, middle, right) => replace_chain_momentum(
            ChainKind::Angle,
            *left,
            middle,
            *right,
            n_particles,
            eliminate,
        ),
        SpinorExpr::SquareChain(left, middle, right) => replace_chain_momentum(
            ChainKind::Square,
            *left,
            middle,
            *right,
            n_particles,
            eliminate,
        ),
        SpinorExpr::AngleSquareChain(left, middle, right) => replace_chain_momentum(
            ChainKind::AngleSquare,
            *left,
            middle,
            *right,
            n_particles,
            eliminate,
        ),
        SpinorExpr::SquareAngleChain(left, middle, right) => replace_chain_momentum(
            ChainKind::SquareAngle,
            *left,
            middle,
            *right,
            n_particles,
            eliminate,
        ),
        SpinorExpr::Mandelstam(i, j) if *i == eliminate || *j == eliminate => {
            let fixed = if *i == eliminate { *j } else { *i };
            let terms = replacement_labels(n_particles, eliminate)
                .into_iter()
                .filter(|label| *label != fixed)
                .map(|label| neg_expr(SpinorExpr::Mandelstam(fixed, label)))
                .collect();
            SpinorExpr::Sum(terms)
        }
        SpinorExpr::Mandelstam3(i, j, k)
            if *i == eliminate || *j == eliminate || *k == eliminate =>
        {
            let expanded = expand_mandelstam(expr);
            apply_momentum_conservation(&expanded, n_particles, eliminate)
        }
        SpinorExpr::Product(terms) => {
            let mut out = Vec::new();
            for term in terms {
                out.push(apply_momentum_to_term(term, n_particles, eliminate));
            }
            combine_product_term_rewrites(out)
        }
        SpinorExpr::Sum(terms) => {
            if massless_mandelstam_sum_is_zero(terms, n_particles) {
                return SpinorExpr::Numeric(BigRational::zero());
            }
            SpinorExpr::Sum(
                terms
                    .iter()
                    .map(|term| apply_momentum_conservation(term, n_particles, eliminate))
                    .collect(),
            )
        }
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(apply_momentum_conservation(num, n_particles, eliminate)),
            Box::new(apply_momentum_conservation(den, n_particles, eliminate)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(
            Box::new(apply_momentum_conservation(base, n_particles, eliminate)),
            *n,
        ),
        SpinorExpr::Neg(inner) => {
            neg_expr(apply_momentum_conservation(inner, n_particles, eliminate))
        }
        _ => expr.clone(),
    }
}

pub fn expand_mandelstam(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Mandelstam(i, j) => SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Angle(*i, *j), SpinorFactor::Square(*j, *i)],
        )]),
        SpinorExpr::Mandelstam3(i, j, k) => SpinorExpr::Sum(vec![
            expand_mandelstam(&SpinorExpr::Mandelstam(*i, *j)),
            expand_mandelstam(&SpinorExpr::Mandelstam(*i, *k)),
            expand_mandelstam(&SpinorExpr::Mandelstam(*j, *k)),
        ]),
        SpinorExpr::Product(terms) => {
            let expanded_terms = terms.iter().map(expand_mandelstam_term).collect();
            combine_product_term_rewrites(expanded_terms)
        }
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(terms.iter().map(expand_mandelstam).collect()),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(expand_mandelstam(num)),
            Box::new(expand_mandelstam(den)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(expand_mandelstam(base)), *n),
        SpinorExpr::Neg(inner) => neg_expr(expand_mandelstam(inner)),
        _ => expr.clone(),
    }
}

pub fn collect_mandelstam(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Product(terms) => SpinorExpr::Product(
            terms
                .iter()
                .map(|term| {
                    let mut factors = term.factors.clone();
                    collect_mandelstam_factors(&mut factors);
                    SpinorTerm {
                        coefficient: term.coefficient.clone(),
                        factors,
                    }
                })
                .collect(),
        ),
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(terms.iter().map(collect_mandelstam).collect()),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(collect_mandelstam(num)),
            Box::new(collect_mandelstam(den)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(collect_mandelstam(base)), *n),
        SpinorExpr::Neg(inner) => neg_expr(collect_mandelstam(inner)),
        _ => expr.clone(),
    }
}

pub fn expand_chain(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::AngleChain(left, middle, right) => {
            expand_chain_labels(*left, middle, *right, ChainKind::Angle)
        }
        SpinorExpr::SquareChain(left, middle, right) => {
            expand_chain_labels(*left, middle, *right, ChainKind::Square)
        }
        SpinorExpr::AngleSquareChain(left, middle, right) => {
            expand_chain_labels(*left, middle, *right, ChainKind::AngleSquare)
        }
        SpinorExpr::SquareAngleChain(left, middle, right) => {
            expand_chain_labels(*left, middle, *right, ChainKind::SquareAngle)
        }
        SpinorExpr::Product(terms) => {
            let expanded_terms = terms.iter().map(expand_chain_term).collect();
            combine_product_term_rewrites(expanded_terms)
        }
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(terms.iter().map(expand_chain).collect()),
        SpinorExpr::Ratio(num, den) => {
            SpinorExpr::Ratio(Box::new(expand_chain(num)), Box::new(expand_chain(den)))
        }
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(expand_chain(base)), *n),
        SpinorExpr::Neg(inner) => neg_expr(expand_chain(inner)),
        _ => expr.clone(),
    }
}

pub fn contract_adjacent(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Product(terms) => SpinorExpr::Product(
            terms
                .iter()
                .map(|term| {
                    let mut factors = term.factors.clone();
                    contract_adjacent_factors(&mut factors);
                    SpinorTerm {
                        coefficient: term.coefficient.clone(),
                        factors,
                    }
                })
                .collect(),
        ),
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(terms.iter().map(contract_adjacent).collect()),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(contract_adjacent(num)),
            Box::new(contract_adjacent(den)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(contract_adjacent(base)), *n),
        SpinorExpr::Neg(inner) => neg_expr(contract_adjacent(inner)),
        _ => expr.clone(),
    }
}

pub fn spinor_simplify(expr: &SpinorExpr, _n_particles: u16) -> SpinorExpr {
    let mut current = expr.clone();
    for _ in 0..5 {
        let next = collect_common_factors(&collect_mandelstam(&simplify_structure(
            &cancel_matching_bracket_pairs(&expand_chain(&expand_mandelstam(
                &canonicalise_bracket(&current),
            ))),
        )));
        if next == current {
            return next;
        }
        current = next;
    }
    current
}

pub fn collect_common_factors(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Sum(terms) if terms.len() > 1 => {
            let product_terms: Option<Vec<SpinorTerm>> = terms
                .iter()
                .map(|term| match term {
                    SpinorExpr::Product(items) if items.len() == 1 => Some(items[0].clone()),
                    _ => None,
                })
                .collect();
            let Some(mut product_terms) = product_terms else {
                return SpinorExpr::Sum(terms.iter().map(collect_common_factors).collect());
            };
            let common = common_factors(&product_terms);
            if common.is_empty() {
                return SpinorExpr::Sum(terms.iter().map(collect_common_factors).collect());
            }

            for term in &mut product_terms {
                remove_common_factors(&mut term.factors, &common);
            }
            let remainder = SpinorExpr::Sum(
                product_terms
                    .into_iter()
                    .map(|term| SpinorExpr::Product(vec![term]))
                    .collect(),
            );
            let mut factors = common;
            factors.push(SpinorFactor::Grouped(Box::new(remainder)));
            SpinorExpr::Product(vec![SpinorTerm::new(BigRational::one(), factors)])
        }
        SpinorExpr::Sum(terms) => {
            SpinorExpr::Sum(terms.iter().map(collect_common_factors).collect())
        }
        SpinorExpr::Product(terms) => SpinorExpr::Product(terms.clone()),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(collect_common_factors(num)),
            Box::new(collect_common_factors(den)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(Box::new(collect_common_factors(base)), *n),
        SpinorExpr::Neg(inner) => neg_expr(collect_common_factors(inner)),
        _ => expr.clone(),
    }
}

pub fn parke_taylor(n: u16, neg_helicity_1: Label, neg_helicity_2: Label) -> SpinorExpr {
    validate_label_in_range(n, neg_helicity_1, "neg_helicity_1");
    validate_label_in_range(n, neg_helicity_2, "neg_helicity_2");

    SpinorExpr::Ratio(
        Box::new(SpinorExpr::Power(
            Box::new(SpinorExpr::AngleBracket(neg_helicity_1, neg_helicity_2)),
            4,
        )),
        Box::new(cyclic_bracket_product(n, BracketKind::Angle)),
    )
}

pub fn parke_taylor_conjugate(n: u16, pos_helicity_1: Label, pos_helicity_2: Label) -> SpinorExpr {
    validate_label_in_range(n, pos_helicity_1, "pos_helicity_1");
    validate_label_in_range(n, pos_helicity_2, "pos_helicity_2");

    SpinorExpr::Ratio(
        Box::new(SpinorExpr::Power(
            Box::new(SpinorExpr::SquareBracket(pos_helicity_1, pos_helicity_2)),
            4,
        )),
        Box::new(cyclic_bracket_product(n, BracketKind::Square)),
    )
}

pub fn three_point_mhv(labels: [Label; 3]) -> SpinorExpr {
    let [one, two, three] = labels;
    SpinorExpr::Ratio(
        Box::new(SpinorExpr::Power(
            Box::new(SpinorExpr::AngleBracket(one, two)),
            3,
        )),
        Box::new(SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![
                SpinorFactor::Angle(two, three),
                SpinorFactor::Angle(three, one),
            ],
        )])),
    )
}

pub fn three_point_anti_mhv(labels: [Label; 3]) -> SpinorExpr {
    let [one, two, three] = labels;
    SpinorExpr::Ratio(
        Box::new(SpinorExpr::Power(
            Box::new(SpinorExpr::SquareBracket(one, two)),
            3,
        )),
        Box::new(SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![
                SpinorFactor::Square(two, three),
                SpinorFactor::Square(three, one),
            ],
        )])),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BCFWShift {
    pub shifted_angle: Label,
    pub shifted_square: Label,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BCFWTerm {
    pub left_particles: Vec<Label>,
    pub right_particles: Vec<Label>,
    pub internal_helicity: i8,
    pub propagator_momentum: Vec<Label>,
}

pub fn bcfw_shift_momentum(
    expr: &SpinorExpr,
    shift: &BCFWShift,
    z: lasso::Spur,
    interner: &ax_ir::Interner,
) -> SpinorExpr {
    let _ = interner;
    match expr {
        SpinorExpr::AngleBracket(i, j) => shifted_angle_bracket(*i, *j, shift, z),
        SpinorExpr::SquareBracket(i, j) => shifted_square_bracket(*i, *j, shift, z),
        SpinorExpr::AngleChain(left, middle, right) => bcfw_shift_momentum(
            &expand_chain_labels(*left, middle, *right, ChainKind::Angle),
            shift,
            z,
            interner,
        ),
        SpinorExpr::SquareChain(left, middle, right) => bcfw_shift_momentum(
            &expand_chain_labels(*left, middle, *right, ChainKind::Square),
            shift,
            z,
            interner,
        ),
        SpinorExpr::AngleSquareChain(left, middle, right) => bcfw_shift_momentum(
            &expand_chain_labels(*left, middle, *right, ChainKind::AngleSquare),
            shift,
            z,
            interner,
        ),
        SpinorExpr::SquareAngleChain(left, middle, right) => bcfw_shift_momentum(
            &expand_chain_labels(*left, middle, *right, ChainKind::SquareAngle),
            shift,
            z,
            interner,
        ),
        SpinorExpr::Mandelstam(_, _) | SpinorExpr::Mandelstam3(_, _, _) => {
            bcfw_shift_momentum(&expand_mandelstam(expr), shift, z, interner)
        }
        SpinorExpr::Product(terms) => {
            let shifted = terms
                .iter()
                .map(|term| shift_term(term, shift, z))
                .collect();
            combine_product_term_rewrites(shifted)
        }
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(
            terms
                .iter()
                .map(|term| bcfw_shift_momentum(term, shift, z, interner))
                .collect(),
        ),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(bcfw_shift_momentum(num, shift, z, interner)),
            Box::new(bcfw_shift_momentum(den, shift, z, interner)),
        ),
        SpinorExpr::Power(base, n) => {
            SpinorExpr::Power(Box::new(bcfw_shift_momentum(base, shift, z, interner)), *n)
        }
        SpinorExpr::Neg(inner) => neg_expr(bcfw_shift_momentum(inner, shift, z, interner)),
        _ => expr.clone(),
    }
}

pub fn bcfw_propagator(channel_particles: &[Label], _n_particles: u16) -> SpinorExpr {
    let mut terms = Vec::new();
    for i in 0..channel_particles.len() {
        for j in (i + 1)..channel_particles.len() {
            terms.push(SpinorExpr::Mandelstam(
                channel_particles[i],
                channel_particles[j],
            ));
        }
    }

    let p_squared = match terms.len() {
        0 => SpinorExpr::Numeric(BigRational::zero()),
        1 => terms.remove(0),
        _ => SpinorExpr::Sum(terms),
    };

    SpinorExpr::Ratio(
        Box::new(SpinorExpr::Numeric(BigRational::one())),
        Box::new(p_squared),
    )
}

pub fn bcfw_decomposition(n: u16, shift: &BCFWShift, helicities: &[i8]) -> Vec<BCFWTerm> {
    assert_eq!(
        helicities.len(),
        n as usize,
        "helicities length must match n_particles"
    );
    assert!(
        helicities.iter().all(|h| *h == 1 || *h == -1),
        "helicities must be +1 or -1"
    );
    validate_label_in_range(n, shift.shifted_angle, "shifted_angle");
    validate_label_in_range(n, shift.shifted_square, "shifted_square");

    let full_mask = if n >= u32::BITS as u16 {
        panic!("BCFW decomposition supports fewer than 32 labels in this representation");
    } else {
        (1u32 << n) - 1
    };
    let angle_bit = 1u32 << shift.shifted_angle.0;
    let square_bit = 1u32 << shift.shifted_square.0;
    let mut terms = Vec::new();

    for mask in 1..full_mask {
        let complement = full_mask ^ mask;
        if mask > complement {
            continue;
        }
        if !is_cyclic_contiguous(mask, n) || !is_cyclic_contiguous(complement, n) {
            continue;
        }
        let left_size = mask.count_ones();
        let right_size = complement.count_ones();
        if left_size < 2 || right_size < 2 {
            continue;
        }
        let split_shifted_particles = ((mask & angle_bit) != 0 && (mask & square_bit) == 0)
            || ((mask & angle_bit) == 0 && (mask & square_bit) != 0);
        if !split_shifted_particles {
            continue;
        }

        let left_particles = labels_from_mask(mask, n);
        let right_particles = labels_from_mask(complement, n);
        let propagator_momentum = if left_particles.len() <= right_particles.len() {
            left_particles.clone()
        } else {
            right_particles.clone()
        };
        for internal_helicity in [-1, 1] {
            terms.push(BCFWTerm {
                left_particles: left_particles.clone(),
                right_particles: right_particles.clone(),
                internal_helicity,
                propagator_momentum: propagator_momentum.clone(),
            });
        }
    }

    terms.sort_by_key(|term| {
        (
            term.left_particles.len().min(term.right_particles.len()),
            term.left_particles.clone(),
            term.right_particles.clone(),
            term.internal_helicity,
        )
    });
    terms
}

fn validate_label_in_range(n: u16, label: Label, name: &str) {
    assert!(
        label.0 <= n,
        "{} label {} is outside the supported zero- or one-based range 0..{} / 1..{}",
        name,
        label.0,
        n.saturating_sub(1),
        n
    );
}

fn is_cyclic_contiguous(mask: u32, n: u16) -> bool {
    let labels = labels_from_mask(mask, n);
    if labels.len() <= 1 || labels.len() == n as usize {
        return true;
    }
    let positions: Vec<u16> = labels.iter().map(|label| label.0).collect();
    for start in &positions {
        let candidate = (0..positions.len())
            .map(|offset| (*start + offset as u16) % n)
            .collect::<Vec<_>>();
        if candidate.iter().all(|idx| positions.contains(idx)) {
            return true;
        }
    }
    false
}

fn cyclic_bracket_product(n: u16, kind: BracketKind) -> SpinorExpr {
    assert!(
        n >= 2,
        "Parke-Taylor amplitudes require at least two particles"
    );
    let factors = (0..n)
        .map(|i| {
            let left = Label::new(i);
            let right = Label::new((i + 1) % n);
            make_bracket_factor(left, right, kind)
        })
        .collect();
    SpinorExpr::Product(vec![SpinorTerm::new(BigRational::one(), factors)])
}

fn labels_from_mask(mask: u32, n: u16) -> Vec<Label> {
    (0..n)
        .filter(|idx| (mask & (1u32 << idx)) != 0)
        .map(Label::new)
        .collect()
}

fn shift_term(term: &SpinorTerm, shift: &BCFWShift, z: lasso::Spur) -> SpinorExpr {
    let mut expr = SpinorExpr::Product(vec![SpinorTerm::new(term.coefficient.clone(), Vec::new())]);
    for factor in &term.factors {
        expr = multiply_exprs(expr, shift_factor(factor, shift, z));
    }
    expr
}

fn shift_factor(factor: &SpinorFactor, shift: &BCFWShift, z: lasso::Spur) -> SpinorExpr {
    match factor {
        SpinorFactor::Angle(i, j) => shifted_angle_bracket(*i, *j, shift, z),
        SpinorFactor::Square(i, j) => shifted_square_bracket(*i, *j, shift, z),
        SpinorFactor::AngleSquare(i, middle, j) => bcfw_shift_momentum(
            &expand_chain_labels(*i, middle, *j, ChainKind::AngleSquare),
            shift,
            z,
            &ax_ir::Interner::new(),
        ),
        SpinorFactor::SquareAngle(i, middle, j) => bcfw_shift_momentum(
            &expand_chain_labels(*i, middle, *j, ChainKind::SquareAngle),
            shift,
            z,
            &ax_ir::Interner::new(),
        ),
        SpinorFactor::Mandelstam(labels) => bcfw_shift_momentum(
            &expand_mandelstam_factor(labels),
            shift,
            z,
            &ax_ir::Interner::new(),
        ),
        SpinorFactor::Power(base, n) => {
            SpinorExpr::Power(Box::new(shift_factor(base, shift, z)), *n)
        }
        SpinorFactor::Grouped(expr) => bcfw_shift_momentum(expr, shift, z, &ax_ir::Interner::new()),
        SpinorFactor::SymbolicParam(_) => SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![factor.clone()],
        )]),
    }
}

fn shifted_angle_bracket(i: Label, j: Label, shift: &BCFWShift, z: lasso::Spur) -> SpinorExpr {
    let left = shifted_angle_slot(i, shift, z);
    let right = shifted_angle_slot(j, shift, z);
    bracket_bilinear(left, right, BracketKind::Angle)
}

fn shifted_square_bracket(i: Label, j: Label, shift: &BCFWShift, z: lasso::Spur) -> SpinorExpr {
    let left = shifted_square_slot(i, shift, z);
    let right = shifted_square_slot(j, shift, z);
    bracket_bilinear(left, right, BracketKind::Square)
}

fn shifted_angle_slot(label: Label, shift: &BCFWShift, z: lasso::Spur) -> Vec<ShiftSlot> {
    if label == shift.shifted_angle {
        vec![
            ShiftSlot::plain(label),
            ShiftSlot {
                coefficient: BigRational::one(),
                z_power: 1,
                label: shift.shifted_square,
                z: Some(z),
            },
        ]
    } else {
        vec![ShiftSlot::plain(label)]
    }
}

fn shifted_square_slot(label: Label, shift: &BCFWShift, z: lasso::Spur) -> Vec<ShiftSlot> {
    if label == shift.shifted_square {
        vec![
            ShiftSlot::plain(label),
            ShiftSlot {
                coefficient: -BigRational::one(),
                z_power: 1,
                label: shift.shifted_angle,
                z: Some(z),
            },
        ]
    } else {
        vec![ShiftSlot::plain(label)]
    }
}

#[derive(Clone)]
struct ShiftSlot {
    coefficient: BigRational,
    z_power: i32,
    label: Label,
    z: Option<lasso::Spur>,
}

impl ShiftSlot {
    fn plain(label: Label) -> Self {
        Self {
            coefficient: BigRational::one(),
            z_power: 0,
            label,
            z: None,
        }
    }
}

fn bracket_bilinear(left: Vec<ShiftSlot>, right: Vec<ShiftSlot>, kind: BracketKind) -> SpinorExpr {
    let mut terms = Vec::new();
    for lhs in &left {
        for rhs in &right {
            let coefficient = lhs.coefficient.clone() * rhs.coefficient.clone();
            let z_power = lhs.z_power + rhs.z_power;
            let z = lhs.z.or(rhs.z);
            let factor = make_bracket_factor(lhs.label, rhs.label, kind);
            terms.push(factor_product_expr(coefficient, z, z_power, factor));
        }
    }

    match terms.len() {
        0 => SpinorExpr::Numeric(BigRational::zero()),
        1 => terms.remove(0),
        _ => SpinorExpr::Sum(terms),
    }
}

fn factor_product_expr(
    coefficient: BigRational,
    z: Option<lasso::Spur>,
    z_power: i32,
    factor: SpinorFactor,
) -> SpinorExpr {
    let mut factors = Vec::new();
    if let Some(z) = z {
        match z_power {
            0 => {}
            1 => factors.push(SpinorFactor::SymbolicParam(z)),
            n => factors.push(SpinorFactor::Power(
                Box::new(SpinorFactor::SymbolicParam(z)),
                n,
            )),
        }
    }
    factors.push(factor);
    SpinorExpr::Product(vec![SpinorTerm::new(coefficient, factors)])
}

#[derive(Clone, Copy)]
enum BracketKind {
    Angle,
    Square,
}

#[derive(Clone, Copy)]
enum ChainKind {
    Angle,
    Square,
    AngleSquare,
    SquareAngle,
}

fn canonicalise_factor_bracket(
    factor: &SpinorFactor,
    coefficient: &mut BigRational,
) -> SpinorFactor {
    match factor {
        SpinorFactor::Angle(i, j) if i > j => {
            *coefficient = -coefficient.clone();
            SpinorFactor::Angle(*j, *i)
        }
        SpinorFactor::Square(i, j) if i > j => {
            *coefficient = -coefficient.clone();
            SpinorFactor::Square(*j, *i)
        }
        SpinorFactor::Power(base, n) => {
            let mut inner_coeff = BigRational::one();
            let inner = canonicalise_factor_bracket(base, &mut inner_coeff);
            if inner_coeff == -BigRational::one() && n % 2 != 0 {
                *coefficient = -coefficient.clone();
            }
            SpinorFactor::Power(Box::new(inner), *n)
        }
        _ => factor.clone(),
    }
}

fn apply_schouten_for_kind(
    expr: &SpinorExpr,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
    kind: BracketKind,
) -> SpinorExpr {
    match expr {
        SpinorExpr::Product(terms) => {
            let mut out = Vec::new();
            for term in terms {
                out.push(apply_schouten_to_term(term, a, b, c, d, kind));
            }
            combine_product_term_rewrites(out)
        }
        SpinorExpr::Sum(terms) => SpinorExpr::Sum(
            terms
                .iter()
                .map(|term| apply_schouten_for_kind(term, a, b, c, d, kind))
                .collect(),
        ),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(apply_schouten_for_kind(num, a, b, c, d, kind)),
            Box::new(apply_schouten_for_kind(den, a, b, c, d, kind)),
        ),
        SpinorExpr::Power(base, n) => SpinorExpr::Power(
            Box::new(apply_schouten_for_kind(base, a, b, c, d, kind)),
            *n,
        ),
        SpinorExpr::Neg(inner) => neg_expr(apply_schouten_for_kind(inner, a, b, c, d, kind)),
        _ => expr.clone(),
    }
}

fn apply_schouten_to_term(
    term: &SpinorTerm,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
    kind: BracketKind,
) -> SpinorExpr {
    for first in 0..term.factors.len() {
        let Some(sign_first) = match_bracket_factor(&term.factors[first], a, b, kind) else {
            continue;
        };
        for second in 0..term.factors.len() {
            if first == second {
                continue;
            }
            let Some(sign_second) = match_bracket_factor(&term.factors[second], c, d, kind) else {
                continue;
            };

            let mut rest = Vec::with_capacity(term.factors.len());
            for (idx, factor) in term.factors.iter().enumerate() {
                if idx != first && idx != second {
                    rest.push(factor.clone());
                }
            }

            let coefficient = term.coefficient.clone()
                * BigRational::from_integer((sign_first * sign_second).into());
            let mut first_term = rest.clone();
            first_term.push(make_bracket_factor(a, c, kind));
            first_term.push(make_bracket_factor(b, d, kind));

            let mut second_term = rest;
            second_term.push(make_bracket_factor(a, d, kind));
            second_term.push(make_bracket_factor(c, b, kind));

            return SpinorExpr::Sum(vec![
                SpinorExpr::Product(vec![SpinorTerm::new(coefficient.clone(), first_term)]),
                SpinorExpr::Product(vec![SpinorTerm::new(coefficient, second_term)]),
            ]);
        }
    }

    SpinorExpr::Product(vec![term.clone()])
}

fn match_bracket_factor(
    factor: &SpinorFactor,
    left: Label,
    right: Label,
    kind: BracketKind,
) -> Option<i32> {
    match (kind, factor) {
        (BracketKind::Angle, SpinorFactor::Angle(i, j))
        | (BracketKind::Square, SpinorFactor::Square(i, j)) => {
            if *i == left && *j == right {
                Some(1)
            } else if *i == right && *j == left {
                Some(-1)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn make_bracket_factor(left: Label, right: Label, kind: BracketKind) -> SpinorFactor {
    match kind {
        BracketKind::Angle => SpinorFactor::Angle(left, right),
        BracketKind::Square => SpinorFactor::Square(left, right),
    }
}

fn replace_chain_momentum(
    kind: ChainKind,
    left: Label,
    middle: &[Label],
    right: Label,
    n_particles: u16,
    eliminate: Label,
) -> SpinorExpr {
    let Some(position) = middle.iter().position(|label| *label == eliminate) else {
        return make_chain_expr(kind, left, middle.to_vec(), right);
    };

    let terms = replacement_labels(n_particles, eliminate)
        .into_iter()
        .map(|replacement| {
            let mut replaced = middle.to_vec();
            replaced[position] = replacement;
            neg_expr(apply_momentum_conservation(
                &make_chain_expr(kind, left, replaced, right),
                n_particles,
                eliminate,
            ))
        })
        .collect();
    SpinorExpr::Sum(terms)
}

fn apply_momentum_to_term(term: &SpinorTerm, n_particles: u16, eliminate: Label) -> SpinorExpr {
    for (idx, factor) in term.factors.iter().enumerate() {
        match factor {
            SpinorFactor::AngleSquare(left, middle, right)
                if middle.iter().any(|label| *label == eliminate) =>
            {
                return expand_momentum_factor(
                    term,
                    idx,
                    ChainKind::AngleSquare,
                    *left,
                    middle,
                    *right,
                    n_particles,
                    eliminate,
                );
            }
            SpinorFactor::SquareAngle(left, middle, right)
                if middle.iter().any(|label| *label == eliminate) =>
            {
                return expand_momentum_factor(
                    term,
                    idx,
                    ChainKind::SquareAngle,
                    *left,
                    middle,
                    *right,
                    n_particles,
                    eliminate,
                );
            }
            SpinorFactor::Mandelstam(labels) if labels.contains(&eliminate) => {
                let expanded = expand_mandelstam_factor(labels);
                let replaced = apply_momentum_conservation(&expanded, n_particles, eliminate);
                return multiply_expr_by_remaining_factors(term, idx, replaced);
            }
            _ => {}
        }
    }

    SpinorExpr::Product(vec![term.clone()])
}

fn expand_momentum_factor(
    term: &SpinorTerm,
    idx: usize,
    kind: ChainKind,
    left: Label,
    middle: &[Label],
    right: Label,
    n_particles: u16,
    eliminate: Label,
) -> SpinorExpr {
    let replaced = replace_chain_momentum(kind, left, middle, right, n_particles, eliminate);
    multiply_expr_by_remaining_factors(term, idx, replaced)
}

fn multiply_expr_by_remaining_factors(
    term: &SpinorTerm,
    remove_idx: usize,
    expr: SpinorExpr,
) -> SpinorExpr {
    let rest: Vec<SpinorFactor> = term
        .factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| {
            if idx == remove_idx {
                None
            } else {
                Some(factor.clone())
            }
        })
        .collect();
    let prefix = SpinorTerm::new(term.coefficient.clone(), rest);
    multiply_exprs(SpinorExpr::Product(vec![prefix]), expr)
}

fn replacement_labels(n_particles: u16, eliminate: Label) -> Vec<Label> {
    (1..=n_particles)
        .map(Label::new)
        .filter(|label| *label != eliminate)
        .collect()
}

fn make_chain_expr(kind: ChainKind, left: Label, middle: Vec<Label>, right: Label) -> SpinorExpr {
    match kind {
        ChainKind::Angle => SpinorExpr::AngleChain(left, middle, right),
        ChainKind::Square => SpinorExpr::SquareChain(left, middle, right),
        ChainKind::AngleSquare => SpinorExpr::AngleSquareChain(left, middle, right),
        ChainKind::SquareAngle => SpinorExpr::SquareAngleChain(left, middle, right),
    }
}

fn expand_mandelstam_term(term: &SpinorTerm) -> SpinorExpr {
    for (idx, factor) in term.factors.iter().enumerate() {
        if let SpinorFactor::Mandelstam(labels) = factor {
            let expanded = expand_mandelstam_factor(labels);
            return multiply_expr_by_remaining_factors(term, idx, expanded);
        }
    }
    SpinorExpr::Product(vec![term.clone()])
}

fn expand_mandelstam_factor(labels: &[Label]) -> SpinorExpr {
    match labels {
        [i, j] => expand_mandelstam(&SpinorExpr::Mandelstam(*i, *j)),
        [i, j, k] => expand_mandelstam(&SpinorExpr::Mandelstam3(*i, *j, *k)),
        _ => {
            let mut terms = Vec::new();
            for i in 0..labels.len() {
                for j in (i + 1)..labels.len() {
                    terms.push(expand_mandelstam(&SpinorExpr::Mandelstam(
                        labels[i], labels[j],
                    )));
                }
            }
            SpinorExpr::Sum(terms)
        }
    }
}

fn collect_mandelstam_factors(factors: &mut Vec<SpinorFactor>) {
    let mut i = 0;
    while i < factors.len() {
        let SpinorFactor::Angle(a, b) = factors[i] else {
            i += 1;
            continue;
        };

        if let Some(j) = factors.iter().enumerate().find_map(|(j, factor)| {
            if i != j && matches!(factor, SpinorFactor::Square(x, y) if *x == b && *y == a) {
                Some(j)
            } else {
                None
            }
        }) {
            let hi = i.max(j);
            let lo = i.min(j);
            factors.remove(hi);
            factors.remove(lo);
            factors.push(SpinorFactor::Mandelstam(vec![a, b]));
            i = 0;
        } else {
            i += 1;
        }
    }
}

fn expand_chain_labels(left: Label, middle: &[Label], right: Label, kind: ChainKind) -> SpinorExpr {
    if middle.is_empty() {
        return make_chain_expr(kind, left, Vec::new(), right);
    }

    let (mut next_is_angle, final_is_angle) = match kind {
        ChainKind::Angle => (true, true),
        ChainKind::Square => (false, false),
        ChainKind::AngleSquare => (true, false),
        ChainKind::SquareAngle => (false, true),
    };

    let mut factors = Vec::with_capacity(middle.len() + 1);
    let mut current = left;
    for momentum in middle {
        factors.push(if next_is_angle {
            SpinorFactor::Angle(current, *momentum)
        } else {
            SpinorFactor::Square(current, *momentum)
        });
        current = *momentum;
        next_is_angle = !next_is_angle;
    }
    factors.push(if final_is_angle {
        SpinorFactor::Angle(current, right)
    } else {
        SpinorFactor::Square(current, right)
    });

    SpinorExpr::Product(vec![SpinorTerm::new(BigRational::one(), factors)])
}

fn expand_chain_term(term: &SpinorTerm) -> SpinorExpr {
    for (idx, factor) in term.factors.iter().enumerate() {
        let expanded = match factor {
            SpinorFactor::AngleSquare(left, middle, right) => Some(expand_chain_labels(
                *left,
                middle,
                *right,
                ChainKind::AngleSquare,
            )),
            SpinorFactor::SquareAngle(left, middle, right) => Some(expand_chain_labels(
                *left,
                middle,
                *right,
                ChainKind::SquareAngle,
            )),
            SpinorFactor::Grouped(expr) => Some(expand_chain(expr)),
            _ => None,
        };
        if let Some(expanded) = expanded {
            return multiply_expr_by_remaining_factors(term, idx, expanded);
        }
    }
    SpinorExpr::Product(vec![term.clone()])
}

fn contract_adjacent_factors(factors: &mut Vec<SpinorFactor>) {
    let mut i = 0;
    while i + 1 < factors.len() {
        let replacement = match (&factors[i], &factors[i + 1]) {
            (SpinorFactor::Angle(a, b), SpinorFactor::Square(c, d)) if b == c => {
                Some(SpinorFactor::AngleSquare(*a, vec![*b], *d))
            }
            (SpinorFactor::Square(a, b), SpinorFactor::Angle(c, d)) if b == c => {
                Some(SpinorFactor::SquareAngle(*a, vec![*b], *d))
            }
            _ => None,
        };
        if let Some(replacement) = replacement {
            factors.splice(i..=i + 1, [replacement]);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
}

fn cancel_matching_bracket_pairs(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Product(terms) => SpinorExpr::Product(
            terms
                .iter()
                .map(|term| {
                    let mut coefficient = term.coefficient.clone();
                    let factors = cancel_term_bracket_pairs(&term.factors, &mut coefficient);
                    SpinorTerm {
                        coefficient,
                        factors,
                    }
                })
                .collect(),
        ),
        SpinorExpr::Sum(terms) => {
            SpinorExpr::Sum(terms.iter().map(cancel_matching_bracket_pairs).collect())
        }
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(cancel_matching_bracket_pairs(num)),
            Box::new(cancel_matching_bracket_pairs(den)),
        ),
        SpinorExpr::Power(base, n) => {
            SpinorExpr::Power(Box::new(cancel_matching_bracket_pairs(base)), *n)
        }
        SpinorExpr::Neg(inner) => neg_expr(cancel_matching_bracket_pairs(inner)),
        _ => expr.clone(),
    }
}

fn cancel_term_bracket_pairs(
    factors: &[SpinorFactor],
    coefficient: &mut BigRational,
) -> Vec<SpinorFactor> {
    let mut factors = factors.to_vec();
    let mut i = 0;
    while i < factors.len() {
        let pair = match factors[i] {
            SpinorFactor::Angle(a, b) => factors.iter().enumerate().find_map(|(j, factor)| {
                if i != j && matches!(factor, SpinorFactor::Angle(x, y) if *x == b && *y == a) {
                    Some((
                        j,
                        SpinorFactor::Power(Box::new(SpinorFactor::Angle(a, b)), 2),
                    ))
                } else {
                    None
                }
            }),
            SpinorFactor::Square(a, b) => factors.iter().enumerate().find_map(|(j, factor)| {
                if i != j && matches!(factor, SpinorFactor::Square(x, y) if *x == b && *y == a) {
                    Some((
                        j,
                        SpinorFactor::Power(Box::new(SpinorFactor::Square(a, b)), 2),
                    ))
                } else {
                    None
                }
            }),
            _ => None,
        };

        if let Some((j, replacement)) = pair {
            *coefficient = -coefficient.clone();
            let hi = i.max(j);
            let lo = i.min(j);
            factors.remove(hi);
            factors.remove(lo);
            factors.push(replacement);
            i = 0;
        } else {
            i += 1;
        }
    }
    factors
}

fn simplify_structure(expr: &SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Product(terms) => {
            let terms: Vec<SpinorTerm> = terms
                .iter()
                .filter_map(|term| {
                    let mut coefficient = term.coefficient.clone();
                    let mut factors = Vec::new();
                    for factor in &term.factors {
                        push_simplified_factor(factor, &mut coefficient, &mut factors);
                    }
                    combine_duplicate_factors(&mut factors);
                    let term = SpinorTerm {
                        coefficient,
                        factors,
                    };
                    if term.is_zero() {
                        None
                    } else {
                        Some(term)
                    }
                })
                .collect();
            if terms.is_empty() {
                SpinorExpr::Numeric(BigRational::zero())
            } else {
                SpinorExpr::Product(terms)
            }
        }
        SpinorExpr::Sum(terms) => combine_like_sum_terms(
            terms
                .iter()
                .flat_map(|term| match simplify_structure(term) {
                    SpinorExpr::Sum(nested) => nested,
                    SpinorExpr::Numeric(n) if n.is_zero() => Vec::new(),
                    other if other.is_zero() => Vec::new(),
                    other => vec![other],
                })
                .collect(),
        ),
        SpinorExpr::Ratio(num, den) => SpinorExpr::Ratio(
            Box::new(simplify_structure(num)),
            Box::new(simplify_structure(den)),
        ),
        SpinorExpr::Power(base, n) => {
            let base = simplify_structure(base);
            if *n == 0 {
                SpinorExpr::Numeric(BigRational::one())
            } else if *n == 1 {
                base
            } else {
                SpinorExpr::Power(Box::new(base), *n)
            }
        }
        SpinorExpr::Neg(inner) => neg_expr(simplify_structure(inner)),
        _ => expr.clone(),
    }
}

fn push_simplified_factor(
    factor: &SpinorFactor,
    coefficient: &mut BigRational,
    out: &mut Vec<SpinorFactor>,
) {
    match factor {
        SpinorFactor::Power(_, 0) => {}
        SpinorFactor::Power(inner, 1) => push_simplified_factor(inner, coefficient, out),
        SpinorFactor::Grouped(expr) => {
            out.push(SpinorFactor::Grouped(Box::new(simplify_structure(expr))))
        }
        SpinorFactor::Angle(i, j) if i == j => *coefficient = BigRational::zero(),
        SpinorFactor::Square(i, j) if i == j => *coefficient = BigRational::zero(),
        _ => out.push(factor.clone()),
    }
}

fn combine_duplicate_factors(factors: &mut Vec<SpinorFactor>) {
    let mut i = 0;
    while i < factors.len() {
        let mut exponent = factor_exponent(&factors[i]);
        let base = factor_base(&factors[i]).clone();
        let mut j = i + 1;
        while j < factors.len() {
            if factor_base(&factors[j]) == &base {
                exponent += factor_exponent(&factors[j]);
                factors.remove(j);
            } else {
                j += 1;
            }
        }

        factors[i] = if exponent == 1 {
            base
        } else {
            SpinorFactor::Power(Box::new(base), exponent)
        };
        i += 1;
    }
}

fn factor_base(factor: &SpinorFactor) -> &SpinorFactor {
    match factor {
        SpinorFactor::Power(base, _) => base,
        _ => factor,
    }
}

fn factor_exponent(factor: &SpinorFactor) -> i32 {
    match factor {
        SpinorFactor::Power(_, n) => *n,
        _ => 1,
    }
}

fn combine_like_sum_terms(terms: Vec<SpinorExpr>) -> SpinorExpr {
    let mut products: Vec<SpinorTerm> = Vec::new();
    let mut others: Vec<SpinorExpr> = Vec::new();

    for term in terms {
        if let Some(item) = expr_as_single_term(term.clone()) {
            if let Some(existing) = products
                .iter_mut()
                .find(|existing| existing.factors == item.factors)
            {
                existing.coefficient += item.coefficient;
            } else {
                products.push(item);
            }
        } else {
            others.push(term);
        }
    }

    let mut result: Vec<SpinorExpr> = products
        .into_iter()
        .filter(|term| !term.coefficient.is_zero() && !term.is_zero())
        .map(|term| SpinorExpr::Product(vec![term]))
        .collect();
    result.extend(others);

    if schouten_sum_is_zero(&result) {
        return SpinorExpr::Numeric(BigRational::zero());
    }

    match result.len() {
        0 => SpinorExpr::Numeric(BigRational::zero()),
        1 => result.remove(0),
        _ => SpinorExpr::Sum(result),
    }
}

fn expr_as_single_term(expr: SpinorExpr) -> Option<SpinorTerm> {
    match expr {
        SpinorExpr::Product(items) if items.len() == 1 => Some(items[0].clone()),
        SpinorExpr::AngleBracket(i, j) => Some(SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Angle(i, j)],
        )),
        SpinorExpr::SquareBracket(i, j) => Some(SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Square(i, j)],
        )),
        SpinorExpr::Mandelstam(i, j) => Some(SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Mandelstam(vec![i, j])],
        )),
        SpinorExpr::Neg(inner) => expr_as_single_term(*inner).map(|mut term| {
            term.coefficient = -term.coefficient;
            term
        }),
        _ => None,
    }
}

fn common_factors(terms: &[SpinorTerm]) -> Vec<SpinorFactor> {
    let Some(first) = terms.first() else {
        return Vec::new();
    };
    let mut common = Vec::new();
    let mut used_per_term: Vec<Vec<bool>> = terms
        .iter()
        .map(|term| vec![false; term.factors.len()])
        .collect();

    for factor in &first.factors {
        let mut positions = Vec::new();
        let mut present_in_all = true;
        for (term_idx, term) in terms.iter().enumerate() {
            if let Some(pos) = term
                .factors
                .iter()
                .enumerate()
                .find_map(|(idx, candidate)| {
                    if !used_per_term[term_idx][idx] && candidate == factor {
                        Some(idx)
                    } else {
                        None
                    }
                })
            {
                positions.push((term_idx, pos));
            } else {
                present_in_all = false;
                break;
            }
        }
        if present_in_all {
            for (term_idx, pos) in positions {
                used_per_term[term_idx][pos] = true;
            }
            common.push(factor.clone());
        }
    }
    common
}

fn remove_common_factors(factors: &mut Vec<SpinorFactor>, common: &[SpinorFactor]) {
    for factor in common {
        if let Some(pos) = factors.iter().position(|candidate| candidate == factor) {
            factors.remove(pos);
        }
    }
}

fn schouten_sum_is_zero(terms: &[SpinorExpr]) -> bool {
    if terms.len() != 3 {
        return false;
    }
    let parsed: Vec<SpinorTerm> = terms
        .iter()
        .filter_map(|term| expr_as_single_term(term.clone()))
        .collect();
    if parsed.len() != 3 {
        return false;
    }

    for first_idx in 0..parsed.len() {
        for kind in [BracketKind::Angle, BracketKind::Square] {
            let Some((a, b, c, d)) = two_bracket_labels(&parsed[first_idx], kind) else {
                continue;
            };
            let q = parsed[first_idx].coefficient.clone();
            if q.is_zero() {
                continue;
            }
            let mut has_second = false;
            let mut has_third = false;
            for (idx, term) in parsed.iter().enumerate() {
                if idx == first_idx {
                    continue;
                }
                if term.coefficient == -q.clone()
                    && term_matches_two_brackets(term, a, c, b, d, kind)
                {
                    has_second = true;
                }
                if term.coefficient == q && term_matches_two_brackets(term, a, d, b, c, kind) {
                    has_third = true;
                }
            }
            if has_second && has_third {
                return true;
            }
        }
    }
    false
}

fn two_bracket_labels(
    term: &SpinorTerm,
    kind: BracketKind,
) -> Option<(Label, Label, Label, Label)> {
    if term.factors.len() != 2 {
        return None;
    }
    let (a, b) = bracket_factor_labels(&term.factors[0], kind)?;
    let (c, d) = bracket_factor_labels(&term.factors[1], kind)?;
    Some((a, b, c, d))
}

fn bracket_factor_labels(factor: &SpinorFactor, kind: BracketKind) -> Option<(Label, Label)> {
    match (kind, factor) {
        (BracketKind::Angle, SpinorFactor::Angle(a, b))
        | (BracketKind::Square, SpinorFactor::Square(a, b)) => Some((*a, *b)),
        _ => None,
    }
}

fn term_matches_two_brackets(
    term: &SpinorTerm,
    a: Label,
    b: Label,
    c: Label,
    d: Label,
    kind: BracketKind,
) -> bool {
    if term.factors.len() != 2 {
        return false;
    }
    (matches_bracket_exact(&term.factors[0], a, b, kind)
        && matches_bracket_exact(&term.factors[1], c, d, kind))
        || (matches_bracket_exact(&term.factors[0], c, d, kind)
            && matches_bracket_exact(&term.factors[1], a, b, kind))
}

fn matches_bracket_exact(factor: &SpinorFactor, a: Label, b: Label, kind: BracketKind) -> bool {
    match (kind, factor) {
        (BracketKind::Angle, SpinorFactor::Angle(x, y))
        | (BracketKind::Square, SpinorFactor::Square(x, y)) => *x == a && *y == b,
        _ => false,
    }
}

fn massless_mandelstam_sum_is_zero(terms: &[SpinorExpr], n_particles: u16) -> bool {
    if terms.len() != n_particles.saturating_sub(1) as usize {
        return false;
    }
    let Some((fixed, first_other)) = terms.first().and_then(expanded_mandelstam_pair) else {
        return false;
    };
    let mut others = vec![first_other];
    for term in &terms[1..] {
        let Some((candidate_fixed, other)) = expanded_mandelstam_pair(term) else {
            return false;
        };
        if candidate_fixed != fixed {
            return false;
        }
        others.push(other);
    }
    others.sort();
    let mut expected: Vec<Label> = (1..=n_particles)
        .map(Label::new)
        .filter(|label| *label != fixed)
        .collect();
    expected.sort();
    others == expected
}

fn expanded_mandelstam_pair(expr: &SpinorExpr) -> Option<(Label, Label)> {
    let SpinorExpr::Product(terms) = expr else {
        return None;
    };
    if terms.len() != 1 {
        return None;
    }
    let term = &terms[0];
    if term.coefficient != BigRational::one() || term.factors.len() != 2 {
        return None;
    }
    match (&term.factors[0], &term.factors[1]) {
        (SpinorFactor::Angle(a, b), SpinorFactor::Square(c, d)) if b == c && a == d => {
            Some((*a, *b))
        }
        (SpinorFactor::Square(c, d), SpinorFactor::Angle(a, b)) if b == c && a == d => {
            Some((*a, *b))
        }
        _ => None,
    }
}

fn combine_product_term_rewrites(items: Vec<SpinorExpr>) -> SpinorExpr {
    let mut iter = items.into_iter();
    let Some(first) = iter.next() else {
        return SpinorExpr::Product(Vec::new());
    };
    iter.fold(first, multiply_exprs)
}

fn multiply_exprs(left: SpinorExpr, right: SpinorExpr) -> SpinorExpr {
    match (left, right) {
        (SpinorExpr::Sum(terms), rhs) => SpinorExpr::Sum(
            terms
                .into_iter()
                .map(|term| multiply_exprs(term, rhs.clone()))
                .collect(),
        ),
        (lhs, SpinorExpr::Sum(terms)) => SpinorExpr::Sum(
            terms
                .into_iter()
                .map(|term| multiply_exprs(lhs.clone(), term))
                .collect(),
        ),
        (SpinorExpr::Neg(lhs), rhs) => neg_expr(multiply_exprs(*lhs, rhs)),
        (lhs, SpinorExpr::Neg(rhs)) => neg_expr(multiply_exprs(lhs, *rhs)),
        (SpinorExpr::Numeric(coeff), SpinorExpr::Product(mut terms))
        | (SpinorExpr::Product(mut terms), SpinorExpr::Numeric(coeff)) => {
            if let Some(first) = terms.first_mut() {
                first.coefficient *= coeff;
                SpinorExpr::Product(terms)
            } else {
                SpinorExpr::Numeric(coeff)
            }
        }
        (SpinorExpr::Product(mut lhs), SpinorExpr::Product(mut rhs)) => {
            match (lhs.len(), rhs.len()) {
                (0, _) => SpinorExpr::Product(rhs),
                (_, 0) => SpinorExpr::Product(lhs),
                (1, 1) => {
                    let mut left = lhs.remove(0);
                    let right = rhs.remove(0);
                    left.coefficient *= right.coefficient;
                    left.factors.extend(right.factors);
                    SpinorExpr::Product(vec![left])
                }
                _ => {
                    lhs.append(&mut rhs);
                    SpinorExpr::Product(lhs)
                }
            }
        }
        (SpinorExpr::Product(mut terms), rhs) => {
            if let Some(factor) = expr_to_factor(rhs) {
                if let Some(last) = terms.last_mut() {
                    last.factors.push(factor);
                } else {
                    terms.push(SpinorTerm::new(BigRational::one(), vec![factor]));
                }
                SpinorExpr::Product(terms)
            } else {
                SpinorExpr::Product(terms)
            }
        }
        (lhs, SpinorExpr::Product(mut terms)) => {
            if let Some(factor) = expr_to_factor(lhs) {
                if let Some(first) = terms.first_mut() {
                    first.factors.insert(0, factor);
                } else {
                    terms.push(SpinorTerm::new(BigRational::one(), vec![factor]));
                }
                SpinorExpr::Product(terms)
            } else {
                SpinorExpr::Product(terms)
            }
        }
        (lhs, rhs) => match (expr_to_factor(lhs), expr_to_factor(rhs)) {
            (Some(lhs_factor), Some(rhs_factor)) => SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![lhs_factor, rhs_factor],
            )]),
            (Some(lhs_factor), None) => {
                SpinorExpr::Product(vec![SpinorTerm::new(BigRational::one(), vec![lhs_factor])])
            }
            (None, Some(rhs_factor)) => {
                SpinorExpr::Product(vec![SpinorTerm::new(BigRational::one(), vec![rhs_factor])])
            }
            (None, None) => SpinorExpr::Product(Vec::new()),
        },
    }
}

fn expr_to_factor(expr: SpinorExpr) -> Option<SpinorFactor> {
    match expr {
        SpinorExpr::AngleBracket(i, j) => Some(SpinorFactor::Angle(i, j)),
        SpinorExpr::SquareBracket(i, j) => Some(SpinorFactor::Square(i, j)),
        SpinorExpr::AngleSquareChain(i, middle, j) => Some(SpinorFactor::AngleSquare(i, middle, j)),
        SpinorExpr::SquareAngleChain(i, middle, j) => Some(SpinorFactor::SquareAngle(i, middle, j)),
        SpinorExpr::Mandelstam(i, j) => Some(SpinorFactor::Mandelstam(vec![i, j])),
        SpinorExpr::Mandelstam3(i, j, k) => Some(SpinorFactor::Mandelstam(vec![i, j, k])),
        SpinorExpr::Power(base, n) => {
            expr_to_factor(*base).map(|factor| SpinorFactor::Power(Box::new(factor), n))
        }
        _ => None,
    }
}

fn neg_expr(expr: SpinorExpr) -> SpinorExpr {
    match expr {
        SpinorExpr::Neg(inner) => *inner,
        SpinorExpr::Numeric(n) => SpinorExpr::Numeric(-n),
        SpinorExpr::Product(mut terms) if terms.len() == 1 => {
            terms[0].coefficient = -terms[0].coefficient.clone();
            SpinorExpr::Product(terms)
        }
        other => SpinorExpr::Neg(Box::new(other)),
    }
}

fn single_label_hit(label: Label, particle: Label) -> i32 {
    if label == particle {
        1
    } else {
        0
    }
}

fn label_hits(i: Label, j: Label, particle: Label) -> i32 {
    single_label_hit(i, particle) + single_label_hit(j, particle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::ThreadedRodeo;
    use num_traits::{One, Zero};

    #[test]
    fn label_map_reuses_existing_labels() {
        let interner = ThreadedRodeo::new();
        let a = interner.get_or_intern("a");
        let b = interner.get_or_intern("b");
        let mut labels = LabelMap::new();

        assert_eq!(labels.register(a), Label::new(0));
        assert_eq!(labels.register(b), Label::new(1));
        assert_eq!(labels.register(a), Label::new(0));
        assert_eq!(labels.name_for(Label::new(1)), Some(b));
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn bracket_zero_and_weights() {
        let one = Label::new(1);
        let two = Label::new(2);

        assert!(SpinorExpr::angle(one, one).is_zero());
        assert_eq!(SpinorExpr::angle(one, two).mass_dimension(), 1);
        assert_eq!(SpinorExpr::angle(one, two).little_group_weight(one), 1);
        assert_eq!(SpinorExpr::square(one, two).little_group_weight(one), -1);
        assert_eq!(SpinorExpr::s(one, two).little_group_weight(one), 0);
    }

    #[test]
    fn product_dimension_and_zero() {
        let one = Label::new(1);
        let two = Label::new(2);
        let term = SpinorTerm::new(
            BigRational::one(),
            vec![
                SpinorFactor::Angle(one, two),
                SpinorFactor::Square(two, one),
            ],
        );
        let expr = SpinorExpr::Product(vec![term]);

        assert_eq!(expr.mass_dimension(), 2);
        assert_eq!(expr.little_group_weight(one), 0);
        assert!(!expr.is_zero());

        let zero_term = SpinorTerm::new(BigRational::zero(), vec![SpinorFactor::Angle(one, two)]);
        assert!(SpinorExpr::Product(vec![zero_term]).is_zero());
    }

    #[test]
    fn canonicalise_bracket_flips_sign() {
        let one = Label::new(1);
        let two = Label::new(2);

        assert_eq!(
            canonicalise_bracket(&SpinorExpr::angle(two, one)),
            SpinorExpr::Neg(Box::new(SpinorExpr::angle(one, two)))
        );

        let expr = SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Square(two, one)],
        )]);
        assert_eq!(
            canonicalise_bracket(&expr),
            SpinorExpr::Product(vec![SpinorTerm::new(
                -BigRational::one(),
                vec![SpinorFactor::Square(one, two)]
            )])
        );
    }

    #[test]
    fn schouten_rewrites_angle_product() {
        let a = Label::new(1);
        let b = Label::new(2);
        let c = Label::new(3);
        let d = Label::new(4);
        let expr = SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Angle(a, b), SpinorFactor::Angle(c, d)],
        )]);

        let result = apply_schouten(&expr, a, b, c, d);

        assert_eq!(
            result,
            SpinorExpr::Sum(vec![
                SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![SpinorFactor::Angle(a, c), SpinorFactor::Angle(b, d)]
                )]),
                SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![SpinorFactor::Angle(a, d), SpinorFactor::Angle(c, b)]
                )]),
            ])
        );
    }

    #[test]
    fn mandelstam_expand_and_collect() {
        let one = Label::new(1);
        let two = Label::new(2);

        let expanded = expand_mandelstam(&SpinorExpr::s(one, two));
        assert_eq!(
            expanded,
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![
                    SpinorFactor::Angle(one, two),
                    SpinorFactor::Square(two, one)
                ]
            )])
        );

        assert_eq!(
            collect_mandelstam(&expanded),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![SpinorFactor::Mandelstam(vec![one, two])]
            )])
        );
    }

    #[test]
    fn momentum_conservation_expands_eliminated_chain_momentum() {
        let a = Label::new(1);
        let eliminated = Label::new(2);
        let b = Label::new(3);
        let expr = SpinorExpr::AngleSquareChain(a, vec![eliminated], b);

        assert_eq!(
            apply_momentum_conservation(&expr, 3, eliminated),
            SpinorExpr::Sum(vec![
                SpinorExpr::Neg(Box::new(SpinorExpr::AngleSquareChain(a, vec![a], b))),
                SpinorExpr::Neg(Box::new(SpinorExpr::AngleSquareChain(a, vec![b], b))),
            ])
        );
    }

    #[test]
    fn expand_chain_alternates_brackets() {
        let i = Label::new(1);
        let k = Label::new(2);
        let l = Label::new(3);
        let j = Label::new(4);

        assert_eq!(
            expand_chain(&SpinorExpr::AngleChain(i, vec![k, l], j)),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![
                    SpinorFactor::Angle(i, k),
                    SpinorFactor::Square(k, l),
                    SpinorFactor::Angle(l, j),
                ]
            )])
        );

        assert_eq!(
            expand_chain(&SpinorExpr::SquareAngleChain(i, vec![k], j)),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![SpinorFactor::Square(i, k), SpinorFactor::Angle(k, j)]
            )])
        );
    }

    #[test]
    fn contract_adjacent_builds_single_momentum_chain() {
        let i = Label::new(1);
        let j = Label::new(2);
        let k = Label::new(3);
        let expr = SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Angle(i, j), SpinorFactor::Square(j, k)],
        )]);

        assert_eq!(
            contract_adjacent(&expr),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![SpinorFactor::AngleSquare(i, vec![j], k)]
            )])
        );
    }

    #[test]
    fn spinor_simplify_cancels_reversed_brackets() {
        let i = Label::new(1);
        let j = Label::new(2);
        let expr = SpinorExpr::Product(vec![SpinorTerm::new(
            BigRational::one(),
            vec![SpinorFactor::Angle(i, j), SpinorFactor::Angle(j, i)],
        )]);

        assert_eq!(
            spinor_simplify(&expr, 2),
            SpinorExpr::Product(vec![SpinorTerm::new(
                -BigRational::one(),
                vec![SpinorFactor::Power(Box::new(SpinorFactor::Angle(i, j)), 2)]
            )])
        );
    }

    #[test]
    fn collect_common_factors_factors_sum() {
        let one = Label::new(1);
        let two = Label::new(2);
        let three = Label::new(3);
        let four = Label::new(4);
        let common = SpinorFactor::Angle(one, two);
        let expr = SpinorExpr::Sum(vec![
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![common.clone(), SpinorFactor::Square(two, three)],
            )]),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![common.clone(), SpinorFactor::Square(two, four)],
            )]),
        ]);

        assert_eq!(
            collect_common_factors(&expr),
            SpinorExpr::Product(vec![SpinorTerm::new(
                BigRational::one(),
                vec![
                    common,
                    SpinorFactor::Grouped(Box::new(SpinorExpr::Sum(vec![
                        SpinorExpr::Product(vec![SpinorTerm::new(
                            BigRational::one(),
                            vec![SpinorFactor::Square(two, three)]
                        )]),
                        SpinorExpr::Product(vec![SpinorTerm::new(
                            BigRational::one(),
                            vec![SpinorFactor::Square(two, four)]
                        )]),
                    ])))
                ]
            )])
        );
    }

    #[test]
    fn parke_taylor_builds_mhv_ratio() {
        let result = parke_taylor(4, Label::new(0), Label::new(2));

        assert_eq!(
            result,
            SpinorExpr::Ratio(
                Box::new(SpinorExpr::Power(
                    Box::new(SpinorExpr::AngleBracket(Label::new(0), Label::new(2))),
                    4
                )),
                Box::new(SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![
                        SpinorFactor::Angle(Label::new(0), Label::new(1)),
                        SpinorFactor::Angle(Label::new(1), Label::new(2)),
                        SpinorFactor::Angle(Label::new(2), Label::new(3)),
                        SpinorFactor::Angle(Label::new(3), Label::new(0)),
                    ]
                )]))
            )
        );
    }

    #[test]
    fn three_point_amplitudes_use_standard_brackets() {
        let one = Label::new(0);
        let two = Label::new(1);
        let three = Label::new(2);

        assert_eq!(
            three_point_mhv([one, two, three]),
            SpinorExpr::Ratio(
                Box::new(SpinorExpr::Power(
                    Box::new(SpinorExpr::AngleBracket(one, two)),
                    3
                )),
                Box::new(SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![
                        SpinorFactor::Angle(two, three),
                        SpinorFactor::Angle(three, one)
                    ]
                )]))
            )
        );

        assert_eq!(
            three_point_anti_mhv([one, two, three]),
            SpinorExpr::Ratio(
                Box::new(SpinorExpr::Power(
                    Box::new(SpinorExpr::SquareBracket(one, two)),
                    3
                )),
                Box::new(SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![
                        SpinorFactor::Square(two, three),
                        SpinorFactor::Square(three, one)
                    ]
                )]))
            )
        );
    }

    #[test]
    fn bcfw_shift_expands_shifted_brackets() {
        let interner = ax_ir::Interner::new();
        let z = interner.get_or_intern("z");
        let shift = BCFWShift {
            shifted_angle: Label::new(0),
            shifted_square: Label::new(1),
        };

        assert_eq!(
            bcfw_shift_momentum(
                &SpinorExpr::AngleBracket(Label::new(0), Label::new(2)),
                &shift,
                z,
                &interner
            ),
            SpinorExpr::Sum(vec![
                SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![SpinorFactor::Angle(Label::new(0), Label::new(2))]
                )]),
                SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![
                        SpinorFactor::SymbolicParam(z),
                        SpinorFactor::Angle(Label::new(1), Label::new(2))
                    ]
                )]),
            ])
        );

        assert_eq!(
            bcfw_shift_momentum(
                &SpinorExpr::SquareBracket(Label::new(1), Label::new(2)),
                &shift,
                z,
                &interner
            ),
            SpinorExpr::Sum(vec![
                SpinorExpr::Product(vec![SpinorTerm::new(
                    BigRational::one(),
                    vec![SpinorFactor::Square(Label::new(1), Label::new(2))]
                )]),
                SpinorExpr::Product(vec![SpinorTerm::new(
                    -BigRational::one(),
                    vec![
                        SpinorFactor::SymbolicParam(z),
                        SpinorFactor::Square(Label::new(0), Label::new(2))
                    ]
                )]),
            ])
        );
    }

    #[test]
    fn bcfw_propagator_and_decomposition() {
        assert_eq!(
            bcfw_propagator(&[Label::new(0), Label::new(2), Label::new(3)], 4),
            SpinorExpr::Ratio(
                Box::new(SpinorExpr::Numeric(BigRational::one())),
                Box::new(SpinorExpr::Sum(vec![
                    SpinorExpr::Mandelstam(Label::new(0), Label::new(2)),
                    SpinorExpr::Mandelstam(Label::new(0), Label::new(3)),
                    SpinorExpr::Mandelstam(Label::new(2), Label::new(3)),
                ]))
            )
        );

        let terms = bcfw_decomposition(
            4,
            &BCFWShift {
                shifted_angle: Label::new(0),
                shifted_square: Label::new(1),
            },
            &[-1, 1, 1, -1],
        );
        assert!(!terms.is_empty());
        assert!(terms.iter().all(|term| {
            term.left_particles.len() >= 2
                && term.right_particles.len() >= 2
                && (term.internal_helicity == 1 || term.internal_helicity == -1)
        }));
    }
}
