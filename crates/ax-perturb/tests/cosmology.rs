use ax_ir::*;
use ax_perturb::cosmology::*;
use ax_perturb::gauge::*;
use std::collections::HashMap;

fn substitute(expr: &Expr, bindings: &HashMap<lasso::Spur, Expr>) -> Expr {
    match expr {
        Expr::Sym(sym) => bindings
            .get(sym)
            .cloned()
            .unwrap_or_else(|| Expr::Sym(*sym)),
        Expr::Add(terms) => Expr::add(terms.iter().map(|term| substitute(term, bindings)).collect()),
        Expr::Mul(factors) => {
            Expr::mul(factors.iter().map(|factor| substitute(factor, bindings)).collect())
        }
        Expr::Pow(base, exp) => Expr::pow(substitute(base, bindings), substitute(exp, bindings)),
        Expr::Neg(inner) => Expr::neg(substitute(inner, bindings)),
        Expr::Call(f, args) => Expr::Call(
            *f,
            args.iter().map(|arg| substitute(arg, bindings)).collect(),
        ),
        other => other.clone(),
    }
}

#[test]
fn spectral_index_value() {
    let interner = Interner::new();
    let eps = interner.get_or_intern("epsilon");
    let eta = interner.get_or_intern("eta");
    let ns = spectral_index(eps, eta, &interner);
    let mut bindings = HashMap::new();
    bindings.insert(eps, Expr::zero());
    bindings.insert(eta, Expr::zero());
    let at_zero = substitute(&ns, &bindings);
    assert_eq!(
        at_zero,
        Expr::one(),
        "n_s at epsilon=eta=0 should be 1, got {:?}",
        at_zero
    );
}

#[test]
fn tensor_to_scalar_value() {
    let interner = Interner::new();
    let eps = interner.get_or_intern("epsilon");
    let r = tensor_to_scalar_ratio(eps, &interner);
    let mut bindings = HashMap::new();
    bindings.insert(eps, Expr::one());
    let val = substitute(&r, &bindings);
    assert_eq!(
        val,
        Expr::Int(16.into()),
        "r at epsilon=1 should be 16, got {:?}",
        val
    );
}

#[test]
fn linearized_einstein_has_four_equations() {
    let interner = Interner::new();
    let bg = frw_background(&interner);
    let decomp = svt_decompose_perturbation(3, &interner);
    let eqs = linearized_einstein_scalar(&bg, &decomp, &interner);
    assert_eq!(
        eqs.len(),
        4,
        "should have 4 linearized Einstein equations, got {}",
        eqs.len()
    );
}

#[test]
fn bardeen_has_two_potentials() {
    let interner = Interner::new();
    let decomp = svt_decompose_perturbation(3, &interner);
    let a = interner.get_or_intern("a");
    let eta = interner.get_or_intern("eta");
    let vars = bardeen_variables(&decomp, a, eta, &interner);
    assert_eq!(
        vars.len(),
        2,
        "should have 2 Bardeen potentials, got {}",
        vars.len()
    );
}

#[test]
fn regge_wheeler_equation_structure() {
    let interner = Interner::new();
    let m = interner.get_or_intern("M");
    let eq = regge_wheeler_equation(2, m, &interner);
    assert_ne!(eq, Expr::zero(), "Regge-Wheeler equation should be nonzero");
}

#[test]
fn zerilli_equation_structure() {
    let interner = Interner::new();
    let m = interner.get_or_intern("M");
    let eq = zerilli_equation(2, m, &interner);
    assert_ne!(eq, Expr::zero(), "Zerilli equation should be nonzero");
}

#[test]
fn svt_decomposition_has_all_modes() {
    let interner = Interner::new();
    let decomp = svt_decompose_perturbation(3, &interner);
    assert_eq!(decomp.scalar_modes.len(), 4, "should have 4 scalar modes");
    assert_eq!(decomp.vector_modes.len(), 2, "should have 2 vector modes");
    assert_eq!(decomp.tensor_modes.len(), 1, "should have 1 tensor mode");
}
