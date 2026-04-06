use crate::{Assumption, Condition, Expr, Index, Interner, TrustLevel};
use num_traits::Zero;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(pub u32);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PooledExpr {
    Int(num_bigint::BigInt),
    Rational(num_rational::BigRational),
    Float(u64),
    Complex(ExprId, ExprId),
    Sym(lasso::Spur),
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Pow(ExprId, ExprId),
    Neg(ExprId),
    Call(lasso::Spur, Vec<ExprId>),
    Indexed(ExprId, Vec<Index>),
    List(Vec<ExprId>),
    Matrix(Vec<Vec<ExprId>>),
    Piecewise(Vec<(ExprId, ExprId)>),
    Let(lasso::Spur, ExprId, ExprId),
    FnDef(lasso::Spur, Vec<lasso::Spur>, ExprId),
    Rule(ExprId, ExprId, TrustLevel),
    Import(Vec<lasso::Spur>),
    Assume(lasso::Spur, Vec<Assumption>),
    SetConvention(String, String),
}

pub struct ExprPool {
    nodes: Vec<Arc<PooledExpr>>,
    dedup: HashMap<PooledExpr, ExprId>,
    interner: Interner,
}

impl Clone for ExprPool {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            dedup: self.dedup.clone(),
            interner: Interner::new(),
        }
    }
}

impl fmt::Debug for ExprPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprPool")
            .field("nodes", &self.nodes.len())
            .field("dedup", &self.dedup.len())
            .finish()
    }
}

impl ExprPool {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dedup: HashMap::new(),
            interner: Interner::new(),
        }
    }

    pub fn intern(&mut self, expr: PooledExpr) -> ExprId {
        if let Some(existing) = self.dedup.get(&expr) {
            return *existing;
        }
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(Arc::new(expr.clone()));
        self.dedup.insert(expr, id);
        id
    }

    pub fn get(&self, id: ExprId) -> &PooledExpr {
        self.nodes
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("invalid ExprId: {id:?}"))
            .as_ref()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn from_expr(&mut self, expr: &Expr) -> ExprId {
        match expr {
            Expr::Int(n) => self.intern(PooledExpr::Int(n.clone())),
            Expr::Rational(r) => self.intern(PooledExpr::Rational(r.clone())),
            Expr::Float(f) => self.intern(PooledExpr::Float(f.to_bits())),
            Expr::Complex(re, im) => {
                let re = self.from_expr(re);
                let im = self.from_expr(im);
                self.intern(PooledExpr::Complex(re, im))
            }
            Expr::Sym(sym) => self.intern(PooledExpr::Sym(*sym)),
            Expr::Add(terms) => {
                let terms = terms.iter().map(|term| self.from_expr(term)).collect();
                self.intern(PooledExpr::Add(terms))
            }
            Expr::Mul(factors) => {
                let factors = factors
                    .iter()
                    .map(|factor| self.from_expr(factor))
                    .collect();
                self.intern(PooledExpr::Mul(factors))
            }
            Expr::Pow(base, exp) => {
                let base = self.from_expr(base);
                let exp = self.from_expr(exp);
                self.intern(PooledExpr::Pow(base, exp))
            }
            Expr::Neg(inner) => {
                let inner = self.from_expr(inner);
                self.intern(PooledExpr::Neg(inner))
            }
            Expr::Call(f, args) => {
                let args = args.iter().map(|arg| self.from_expr(arg)).collect();
                self.intern(PooledExpr::Call(*f, args))
            }
            Expr::FnDef(name, params, body) => {
                let body = self.from_expr(body);
                self.intern(PooledExpr::FnDef(*name, params.clone(), body))
            }
            Expr::Rule(lhs, rhs, trust) => {
                let lhs = self.from_expr(lhs);
                let rhs = self.from_expr(rhs);
                self.intern(PooledExpr::Rule(lhs, rhs, *trust))
            }
            Expr::Import(path) => self.intern(PooledExpr::Import(path.clone())),
            Expr::Assume(sym, assumptions) => {
                self.intern(PooledExpr::Assume(*sym, assumptions.clone()))
            }
            Expr::SetConvention(field, value) => {
                self.intern(PooledExpr::SetConvention(field.clone(), value.clone()))
            }
            Expr::Piecewise(cases) => {
                let cases = cases
                    .iter()
                    .map(|(value, condition)| {
                        let value = self.from_expr(value);
                        let condition = self.condition_to_expr_id(condition);
                        (value, condition)
                    })
                    .collect();
                self.intern(PooledExpr::Piecewise(cases))
            }
            Expr::Indexed(base, indices) => {
                let base = self.from_expr(base);
                self.intern(PooledExpr::Indexed(base, indices.clone()))
            }
            Expr::Let(name, value, body) => {
                let value = self.from_expr(value);
                let body = self.from_expr(body);
                self.intern(PooledExpr::Let(*name, value, body))
            }
            Expr::List(items) => {
                let items = items.iter().map(|item| self.from_expr(item)).collect();
                self.intern(PooledExpr::List(items))
            }
            Expr::Matrix(rows) => {
                let rows = rows
                    .iter()
                    .map(|row| row.iter().map(|cell| self.from_expr(cell)).collect())
                    .collect();
                self.intern(PooledExpr::Matrix(rows))
            }
        }
    }

    pub fn to_expr(&self, id: ExprId) -> Expr {
        match self.get(id) {
            PooledExpr::Int(n) => Expr::Int(n.clone()),
            PooledExpr::Rational(r) => Expr::Rational(r.clone()),
            PooledExpr::Float(bits) => Expr::Float(f64::from_bits(*bits)),
            PooledExpr::Complex(re, im) => {
                Expr::Complex(Box::new(self.to_expr(*re)), Box::new(self.to_expr(*im)))
            }
            PooledExpr::Sym(sym) => Expr::Sym(*sym),
            PooledExpr::Add(terms) => Expr::Add(terms.iter().map(|id| self.to_expr(*id)).collect()),
            PooledExpr::Mul(factors) => {
                Expr::Mul(factors.iter().map(|id| self.to_expr(*id)).collect())
            }
            PooledExpr::Pow(base, exp) => {
                Expr::Pow(Box::new(self.to_expr(*base)), Box::new(self.to_expr(*exp)))
            }
            PooledExpr::Neg(inner) => Expr::Neg(Box::new(self.to_expr(*inner))),
            PooledExpr::Call(f, args) => self.call_to_expr(*f, args),
            PooledExpr::Indexed(base, indices) => {
                Expr::Indexed(Box::new(self.to_expr(*base)), indices.clone())
            }
            PooledExpr::List(items) => {
                Expr::List(items.iter().map(|id| self.to_expr(*id)).collect())
            }
            PooledExpr::Matrix(rows) => Expr::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|id| self.to_expr(*id)).collect())
                    .collect(),
            ),
            PooledExpr::Piecewise(cases) => Expr::Piecewise(
                cases
                    .iter()
                    .map(|(value, condition)| {
                        (self.to_expr(*value), self.expr_id_to_condition(*condition))
                    })
                    .collect(),
            ),
            PooledExpr::Let(name, value, body) => Expr::Let(
                *name,
                Box::new(self.to_expr(*value)),
                Box::new(self.to_expr(*body)),
            ),
            PooledExpr::FnDef(name, params, body) => {
                Expr::FnDef(*name, params.clone(), Box::new(self.to_expr(*body)))
            }
            PooledExpr::Rule(lhs, rhs, trust) => Expr::Rule(
                Box::new(self.to_expr(*lhs)),
                Box::new(self.to_expr(*rhs)),
                *trust,
            ),
            PooledExpr::Import(path) => Expr::Import(path.clone()),
            PooledExpr::Assume(sym, assumptions) => Expr::Assume(*sym, assumptions.clone()),
            PooledExpr::SetConvention(field, value) => {
                Expr::SetConvention(field.clone(), value.clone())
            }
        }
    }

    pub fn structural_eq(&self, a: ExprId, b: ExprId) -> bool {
        a == b
    }

    pub fn node_count(&self, id: ExprId) -> usize {
        let mut visited = HashSet::new();
        self.node_count_inner(id, &mut visited)
    }

    pub fn unique_node_count(&self, id: ExprId) -> usize {
        let mut visited = HashSet::new();
        self.visit(id, &mut visited);
        visited.len()
    }

    fn node_count_inner(&self, id: ExprId, visited: &mut HashSet<ExprId>) -> usize {
        if !visited.insert(id) {
            return 0;
        }
        1 + self
            .children(id)
            .into_iter()
            .map(|child| self.node_count_inner(child, visited))
            .sum::<usize>()
    }

    fn visit(&self, id: ExprId, visited: &mut HashSet<ExprId>) {
        if !visited.insert(id) {
            return;
        }
        for child in self.children(id) {
            self.visit(child, visited);
        }
    }

    fn children(&self, id: ExprId) -> Vec<ExprId> {
        match self.get(id) {
            PooledExpr::Complex(a, b) | PooledExpr::Pow(a, b) | PooledExpr::Rule(a, b, _) => {
                vec![*a, *b]
            }
            PooledExpr::Neg(inner)
            | PooledExpr::Indexed(inner, _)
            | PooledExpr::FnDef(_, _, inner) => {
                vec![*inner]
            }
            PooledExpr::Add(items)
            | PooledExpr::Mul(items)
            | PooledExpr::Call(_, items)
            | PooledExpr::List(items) => items.clone(),
            PooledExpr::Matrix(rows) => rows.iter().flatten().copied().collect(),
            PooledExpr::Piecewise(cases) => cases
                .iter()
                .flat_map(|(value, cond)| [*value, *cond])
                .collect(),
            PooledExpr::Let(_, value, body) => vec![*value, *body],
            PooledExpr::Int(_)
            | PooledExpr::Rational(_)
            | PooledExpr::Float(_)
            | PooledExpr::Sym(_)
            | PooledExpr::Import(_)
            | PooledExpr::Assume(_, _)
            | PooledExpr::SetConvention(_, _) => Vec::new(),
        }
    }

    fn synthetic(&mut self, name: &str) -> lasso::Spur {
        self.interner.get_or_intern(name)
    }

    fn condition_to_expr_id(&mut self, condition: &Condition) -> ExprId {
        match condition {
            Condition::Gt(a, b) => self.condition_call("__gt", &[a, b]),
            Condition::Lt(a, b) => self.condition_call("__lt", &[a, b]),
            Condition::Ge(a, b) => self.condition_call("__ge", &[a, b]),
            Condition::Le(a, b) => self.condition_call("__le", &[a, b]),
            Condition::Eq(a, b) => self.condition_call("__eq", &[a, b]),
            Condition::Ne(a, b) => self.condition_call("__ne", &[a, b]),
            Condition::And(a, b) => {
                let sym = self.synthetic("__and");
                let a = self.condition_to_expr_id(a);
                let b = self.condition_to_expr_id(b);
                self.intern(PooledExpr::Call(sym, vec![a, b]))
            }
            Condition::Or(a, b) => {
                let sym = self.synthetic("__or");
                let a = self.condition_to_expr_id(a);
                let b = self.condition_to_expr_id(b);
                self.intern(PooledExpr::Call(sym, vec![a, b]))
            }
            Condition::Not(inner) => {
                let sym = self.synthetic("__not");
                let inner = self.condition_to_expr_id(inner);
                self.intern(PooledExpr::Call(sym, vec![inner]))
            }
            Condition::True => {
                let sym = self.synthetic("__true");
                self.intern(PooledExpr::Call(sym, Vec::new()))
            }
            Condition::False => {
                let sym = self.synthetic("__false");
                self.intern(PooledExpr::Call(sym, Vec::new()))
            }
        }
    }

    fn condition_call(&mut self, name: &str, args: &[&Expr; 2]) -> ExprId {
        let sym = self.synthetic(name);
        let lhs = self.from_expr(args[0]);
        let rhs = self.from_expr(args[1]);
        self.intern(PooledExpr::Call(sym, vec![lhs, rhs]))
    }

    fn expr_id_to_condition(&self, id: ExprId) -> Condition {
        match self.get(id) {
            PooledExpr::Call(sym, args) => match (self.synthetic_name(*sym), args.as_slice()) {
                ("__gt", [a, b]) => Condition::Gt(self.to_expr(*a), self.to_expr(*b)),
                ("__lt", [a, b]) => Condition::Lt(self.to_expr(*a), self.to_expr(*b)),
                ("__ge", [a, b]) => Condition::Ge(self.to_expr(*a), self.to_expr(*b)),
                ("__le", [a, b]) => Condition::Le(self.to_expr(*a), self.to_expr(*b)),
                ("__eq", [a, b]) => Condition::Eq(self.to_expr(*a), self.to_expr(*b)),
                ("__ne", [a, b]) => Condition::Ne(self.to_expr(*a), self.to_expr(*b)),
                ("__and", [a, b]) => Condition::And(
                    Box::new(self.expr_id_to_condition(*a)),
                    Box::new(self.expr_id_to_condition(*b)),
                ),
                ("__or", [a, b]) => Condition::Or(
                    Box::new(self.expr_id_to_condition(*a)),
                    Box::new(self.expr_id_to_condition(*b)),
                ),
                ("__not", [inner]) => Condition::Not(Box::new(self.expr_id_to_condition(*inner))),
                ("__true", []) => Condition::True,
                ("__false", []) => Condition::False,
                _ => Condition::True,
            },
            _ => Condition::True,
        }
    }

    fn call_to_expr(&self, f: lasso::Spur, args: &[ExprId]) -> Expr {
        Expr::Call(f, args.iter().map(|id| self.to_expr(*id)).collect())
    }

    fn synthetic_name(&self, sym: lasso::Spur) -> &str {
        self.interner.resolve(sym)
    }
}

impl Default for ExprPool {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ExprBuilder<'a> {
    pool: &'a mut ExprPool,
}

impl<'a> ExprBuilder<'a> {
    pub fn new(pool: &'a mut ExprPool) -> Self {
        Self { pool }
    }

    pub fn int(&mut self, n: impl Into<num_bigint::BigInt>) -> ExprId {
        self.pool.intern(PooledExpr::Int(n.into()))
    }

    pub fn rational(&mut self, r: num_rational::BigRational) -> ExprId {
        self.pool.intern(PooledExpr::Rational(r))
    }

    pub fn float(&mut self, f: f64) -> ExprId {
        self.pool.intern(PooledExpr::Float(f.to_bits()))
    }

    pub fn sym(&mut self, s: lasso::Spur) -> ExprId {
        self.pool.intern(PooledExpr::Sym(s))
    }

    pub fn add(&mut self, terms: Vec<ExprId>) -> ExprId {
        let mut flat = Vec::new();
        let mut numeric = num_rational::BigRational::from_integer(0.into());
        let mut saw_numeric = false;

        for term in terms {
            match self.pool.get(term).clone() {
                PooledExpr::Add(children) => flat.extend(children),
                PooledExpr::Int(n) => {
                    numeric += num_rational::BigRational::from_integer(n);
                    saw_numeric = true;
                }
                PooledExpr::Rational(r) => {
                    numeric += r;
                    saw_numeric = true;
                }
                _ => flat.push(term),
            }
        }

        if saw_numeric && !numeric.is_zero() {
            flat.push(rational_to_id(self.pool, numeric));
        }
        flat.sort_by_key(|id| id.0);

        match flat.len() {
            0 => self.int(0),
            1 => flat[0],
            _ => self.pool.intern(PooledExpr::Add(flat)),
        }
    }

    pub fn mul(&mut self, factors: Vec<ExprId>) -> ExprId {
        let mut flat = Vec::new();
        let mut numeric = num_rational::BigRational::from_integer(1.into());
        let mut saw_numeric = false;

        for factor in factors {
            match self.pool.get(factor).clone() {
                PooledExpr::Mul(children) => flat.extend(children),
                PooledExpr::Int(n) => {
                    if n == num_bigint::BigInt::from(0) {
                        return self.int(0);
                    }
                    numeric *= num_rational::BigRational::from_integer(n);
                    saw_numeric = true;
                }
                PooledExpr::Rational(r) => {
                    if r.is_zero() {
                        return self.int(0);
                    }
                    numeric *= r;
                    saw_numeric = true;
                }
                _ => flat.push(factor),
            }
        }

        if saw_numeric && numeric.is_zero() {
            return self.int(0);
        }
        if saw_numeric && numeric != num_rational::BigRational::from_integer(1.into()) {
            flat.push(rational_to_id(self.pool, numeric));
        }
        flat.sort_by_key(|id| id.0);

        match flat.len() {
            0 => self.int(1),
            1 => flat[0],
            _ => self.pool.intern(PooledExpr::Mul(flat)),
        }
    }

    pub fn pow(&mut self, base: ExprId, exp: ExprId) -> ExprId {
        if let PooledExpr::Int(n) = self.pool.get(exp) {
            if *n == num_bigint::BigInt::from(0) {
                return self.int(1);
            }
            if *n == num_bigint::BigInt::from(1) {
                return base;
            }
        }

        if let PooledExpr::Int(n) = self.pool.get(base) {
            if *n == num_bigint::BigInt::from(0) {
                return self.int(0);
            }
            if *n == num_bigint::BigInt::from(1) {
                return self.int(1);
            }
        }

        if let Some(power) = small_positive_int(self.pool.get(exp)) {
            if let PooledExpr::Int(n) = self.pool.get(base) {
                let mut result = num_bigint::BigInt::from(1);
                for _ in 0..power {
                    result *= n;
                }
                return self.int(result);
            }
            if let PooledExpr::Rational(r) = self.pool.get(base) {
                let mut result = num_rational::BigRational::from_integer(1.into());
                for _ in 0..power {
                    result *= r.clone();
                }
                return self.rational(result);
            }
        }

        self.pool.intern(PooledExpr::Pow(base, exp))
    }

    pub fn neg(&mut self, inner: ExprId) -> ExprId {
        match self.pool.get(inner).clone() {
            PooledExpr::Neg(id) => id,
            PooledExpr::Int(n) => self.int(-n),
            PooledExpr::Rational(r) => self.rational(-r),
            _ => self.pool.intern(PooledExpr::Neg(inner)),
        }
    }

    pub fn call(&mut self, func: lasso::Spur, args: Vec<ExprId>) -> ExprId {
        self.pool.intern(PooledExpr::Call(func, args))
    }

    pub fn indexed(&mut self, base: ExprId, indices: Vec<crate::Index>) -> ExprId {
        self.pool.intern(PooledExpr::Indexed(base, indices))
    }

    pub fn complex(&mut self, re: ExprId, im: ExprId) -> ExprId {
        if matches!(self.pool.get(im), PooledExpr::Int(n) if *n == num_bigint::BigInt::from(0)) {
            return re;
        }
        self.pool.intern(PooledExpr::Complex(re, im))
    }

    pub fn list(&mut self, items: Vec<ExprId>) -> ExprId {
        self.pool.intern(PooledExpr::List(items))
    }

    pub fn matrix(&mut self, rows: Vec<Vec<ExprId>>) -> ExprId {
        self.pool.intern(PooledExpr::Matrix(rows))
    }

    pub fn let_expr(&mut self, name: lasso::Spur, value: ExprId, body: ExprId) -> ExprId {
        self.pool.intern(PooledExpr::Let(name, value, body))
    }
}

fn rational_to_id(pool: &mut ExprPool, rational: num_rational::BigRational) -> ExprId {
    if rational.is_integer() {
        pool.intern(PooledExpr::Int(rational.to_integer()))
    } else {
        pool.intern(PooledExpr::Rational(rational))
    }
}

fn small_positive_int(expr: &PooledExpr) -> Option<usize> {
    let PooledExpr::Int(n) = expr else {
        return None;
    };
    let value = num_traits::ToPrimitive::to_usize(n)?;
    (value < 100).then_some(value)
}
