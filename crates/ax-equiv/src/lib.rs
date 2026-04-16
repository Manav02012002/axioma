#![forbid(unsafe_code)]

use ax_ir::Expr;
use std::cmp::Ordering;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EquivResult {
    Equal,
    EqualUnderAssumptions(Vec<String>),
    DummyIndexRenamed,
    ConventionDifference(Vec<String>),
    NotEqual,
    Unknown,
}

pub fn equivalent_under_tensor_symmetry(a: &ax_ir::TensorSymmetry, b: &ax_ir::TensorSymmetry) -> bool {
    ax_compare::compare_tensor_symmetry(a, b) == Ordering::Equal
}

fn numeric_eval_expr(expr: &Expr, env: &ax_eval::Env, interner: &ax_ir::Interner) -> Option<f64> {
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
            numeric_eval_expr(base, env, interner)?.powf(numeric_eval_expr(exp, env, interner)?),
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

fn contains_indexed(expr: &Expr) -> bool {
    match expr {
        Expr::Indexed(_, _) => true,
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            terms.iter().any(contains_indexed)
        }
        Expr::Pow(base, exp) => contains_indexed(base) || contains_indexed(exp),
        Expr::Neg(inner) => contains_indexed(inner),
        Expr::Call(_, args) => args.iter().any(contains_indexed),
        Expr::Complex(re, im) => contains_indexed(re) || contains_indexed(im),
        Expr::FnDef(_, _, body) => contains_indexed(body),
        Expr::Rule(lhs, rhs, _) => contains_indexed(lhs) || contains_indexed(rhs),
        Expr::Piecewise(cases) => cases.iter().any(|(value, _)| contains_indexed(value)),
        Expr::Let(_, value, body) => contains_indexed(value) || contains_indexed(body),
        Expr::Group(inner, _) => contains_indexed(inner),
        Expr::Matrix(rows) => rows.iter().flatten().any(contains_indexed),
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => false,
    }
}

fn canonical_simplify(expr: &Expr, interner: &ax_ir::Interner) -> Expr {
    let expanded = ax_eval::simplify::expand(expr, interner);
    let collected = ax_eval::simplify::collect_terms(&expanded, interner);
    ax_eval::eval(&collected, &ax_eval::Env::new(), interner)
}

fn collect_syms(expr: &ax_ir::Expr, out: &mut Vec<lasso::Spur>) {
    match expr {
        ax_ir::Expr::Sym(s) => out.push(*s),
        ax_ir::Expr::Add(terms) | ax_ir::Expr::Mul(terms) | ax_ir::Expr::List(terms) => {
            for t in terms {
                collect_syms(t, out);
            }
        }
        ax_ir::Expr::Pow(base, exp) => {
            collect_syms(base, out);
            collect_syms(exp, out);
        }
        ax_ir::Expr::Neg(e) => collect_syms(e, out),
        ax_ir::Expr::Call(_, args) => {
            for a in args {
                collect_syms(a, out);
            }
        }
        ax_ir::Expr::Complex(re, im) => {
            collect_syms(re, out);
            collect_syms(im, out);
        }
        ax_ir::Expr::FnDef(_, _, body) => collect_syms(body, out),
        ax_ir::Expr::Rule(lhs, rhs, _) => {
            collect_syms(lhs, out);
            collect_syms(rhs, out);
        }
        ax_ir::Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_syms(value, out);
            }
        }
        ax_ir::Expr::Indexed(base, _) => collect_syms(base, out),
        ax_ir::Expr::Let(_, val, body) => {
            collect_syms(val, out);
            collect_syms(body, out);
        }
        ax_ir::Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_syms(cell, out);
                }
            }
        }
        _ => {}
    }
}

fn unbound_symbols(
    expr: &ax_ir::Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Vec<lasso::Spur> {
    let mut syms = Vec::new();
    collect_syms(expr, &mut syms);
    let reserved: HashSet<&str> = ["pi", "e", "i", "inf", "infty", "neg_inf"]
        .into_iter()
        .collect();
    syms.retain(|s| env.lookup(*s).is_none() && !reserved.contains(interner.resolve(*s)));
    syms.sort();
    syms.dedup();
    syms
}

fn sample_values(sample: usize) -> [f64; 5] {
    let base = [0.5, 1.0, -1.5, 2.3, -0.7];
    [
        base[sample % 5],
        base[(sample + 1) % 5],
        base[(sample + 2) % 5],
        base[(sample + 3) % 5],
        base[(sample + 4) % 5],
    ]
}

#[cfg(test)]
mod tensor_symmetry_tests {
    use super::*;
    use ax_ir::{
        DualityKind, RestrictedSymmetryMode, SymmetrySource, TableauAttachment, TensorSymmetry,
    };

    fn symmetry(slots: Vec<usize>) -> TensorSymmetry {
        TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![2],
                slot_map: slots,
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

    #[test]
    fn identical_structured_symmetries_are_equivalent() {
        assert!(equivalent_under_tensor_symmetry(
            &symmetry(vec![0, 1]),
            &symmetry(vec![0, 1])
        ));
    }

    #[test]
    fn different_slot_maps_are_not_equivalent() {
        assert!(!equivalent_under_tensor_symmetry(
            &symmetry(vec![0, 1]),
            &symmetry(vec![1, 0])
        ));
    }
}

fn numeric_env(base_env: &ax_eval::Env, syms: &[lasso::Spur], sample: usize) -> ax_eval::Env {
    let mut env = base_env.clone();
    let values = sample_values(sample);
    for (idx, sym) in syms.iter().enumerate() {
        let value = values[idx % values.len()];
        env.bindings.insert(*sym, Expr::Float(value));
    }
    env
}

fn sample_check(
    a: &ax_ir::Expr,
    b: &ax_ir::Expr,
    syms: &[lasso::Spur],
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
    n_samples: usize,
) -> Option<bool> {
    let mut successful = 0usize;
    for sample in 0..n_samples {
        let sample_env = numeric_env(env, syms, sample);
        let (Some(lhs), Some(rhs)) = (
            numeric_eval_expr(a, &sample_env, interner),
            numeric_eval_expr(b, &sample_env, interner),
        ) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        successful += 1;
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        if (lhs - rhs).abs() / scale > 1e-10 {
            return Some(false);
        }
    }
    if successful == n_samples {
        Some(true)
    } else {
        None
    }
}

pub fn check_equiv(
    a: &ax_ir::Expr,
    b: &ax_ir::Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> EquivResult {
    let ea = ax_eval::eval(a, env, interner);
    let eb = ax_eval::eval(b, env, interner);
    if ea == eb {
        return EquivResult::Equal;
    }

    let sa = canonical_simplify(&ea, interner);
    let sb = canonical_simplify(&eb, interner);
    if sa == sb {
        return EquivResult::Equal;
    }

    let diff = canonical_simplify(
        &Expr::add(vec![sa.clone(), Expr::neg(sb.clone())]),
        interner,
    );
    if diff == Expr::zero() {
        return EquivResult::Equal;
    }

    let ta = ax_eval::simplify::trig_simplify(&sa, interner);
    let tb = ax_eval::simplify::trig_simplify(&sb, interner);
    if ta == tb {
        return EquivResult::Equal;
    }

    if contains_indexed(&ta) || contains_indexed(&tb) {
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
            return EquivResult::DummyIndexRenamed;
        }
    }

    let mut syms = unbound_symbols(&ta, env, interner);
    syms.extend(unbound_symbols(&tb, env, interner));
    syms.sort();
    syms.dedup();

    if !syms.is_empty() {
        match sample_check(&ta, &tb, &syms, env, interner, 5) {
            Some(true) => {
                return EquivResult::EqualUnderAssumptions(vec![
                    "numerically verified at 5 points".into()
                ])
            }
            Some(false) => return EquivResult::NotEqual,
            None => {}
        }
    }

    let neg_tb = canonical_simplify(&Expr::neg(tb.clone()), interner);
    if ta == neg_tb {
        return EquivResult::ConventionDifference(vec![format!(
            "global sign difference may be explained by {:?} / {:?}",
            env.convention.riemann_sign, env.convention.ricci_contraction
        )]);
    }

    EquivResult::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equiv_structural() {
        let interner = ax_ir::Interner::new();
        let env = ax_eval::Env::new();
        let x = interner.get_or_intern("x");
        let a = Expr::add(vec![Expr::Sym(x), Expr::one()]);
        let b = Expr::add(vec![Expr::one(), Expr::Sym(x)]);
        assert_eq!(check_equiv(&a, &b, &env, &interner), EquivResult::Equal);
    }

    #[test]
    fn equiv_trig_identity() {
        let interner = ax_ir::Interner::new();
        let env = ax_eval::Env::new();
        let x = interner.get_or_intern("x");
        let sin_sym = interner.get_or_intern("sin");
        let cos_sym = interner.get_or_intern("cos");
        let a = Expr::add(vec![
            Expr::pow(Expr::Call(sin_sym, vec![Expr::Sym(x)]), Expr::Int(2.into())),
            Expr::pow(Expr::Call(cos_sym, vec![Expr::Sym(x)]), Expr::Int(2.into())),
        ]);
        let b = Expr::one();
        assert_eq!(check_equiv(&a, &b, &env, &interner), EquivResult::Equal);
    }

    #[test]
    fn equiv_different() {
        let interner = ax_ir::Interner::new();
        let env = ax_eval::Env::new();
        let x = interner.get_or_intern("x");
        let a = Expr::Sym(x);
        let b = Expr::pow(Expr::Sym(x), Expr::Int(2.into()));
        assert_eq!(check_equiv(&a, &b, &env, &interner), EquivResult::NotEqual);
    }

    #[test]
    fn equiv_numeric_sampling() {
        let interner = ax_ir::Interner::new();
        let env = ax_eval::Env::new();
        let x = interner.get_or_intern("x");
        let a = Expr::pow(
            Expr::add(vec![Expr::Sym(x), Expr::one()]),
            Expr::Int(2.into()),
        );
        let b = Expr::add(vec![
            Expr::pow(Expr::Sym(x), Expr::Int(2.into())),
            Expr::mul(vec![Expr::Int(2.into()), Expr::Sym(x)]),
            Expr::one(),
        ]);
        let result = check_equiv(&a, &b, &env, &interner);
        assert!(matches!(
            result,
            EquivResult::Equal | EquivResult::EqualUnderAssumptions(_)
        ));
    }
}
