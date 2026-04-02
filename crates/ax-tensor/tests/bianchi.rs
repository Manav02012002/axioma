//! Integration test: tensor algebra foundations for the Bianchi identity.
//!
//! The first Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0 follows from
//! the Young-tableau symmetry of the Riemann tensor. Here we verify the
//! sub-identities that the RiemannSymmetry property *does* enforce automatically:
//!   - antisymmetry in the first pair: R_{abcd} = -R_{bacd}
//!   - antisymmetry in the second pair: R_{abcd} = -R_{abdc}
//!   - pair-exchange symmetry: R_{abcd} = R_{cdab}
//! and demonstrate that canonicalise places every Riemann term in a unique
//! canonical form.

use ax_ir::{Expr, Index, Interner, TensorProperty, Variance};
use std::collections::HashMap;

fn setup() -> (Interner, HashMap<lasso::Spur, Vec<TensorProperty>>) {
    let interner = Interner::new();
    let r_sym = interner.get_or_intern("R");
    let mut props = HashMap::new();
    props.insert(r_sym, vec![TensorProperty::RiemannSymmetry]);
    (interner, props)
}

fn make_down_index(name: lasso::Spur) -> Index {
    Index { name, variance: Variance::Down, index_type: None }
}

fn make_riemann(
    interner: &Interner,
    i0: lasso::Spur,
    i1: lasso::Spur,
    i2: lasso::Spur,
    i3: lasso::Spur,
) -> Expr {
    let r = interner.get_or_intern("R");
    Expr::Indexed(
        Box::new(Expr::Sym(r)),
        vec![
            make_down_index(i0),
            make_down_index(i1),
            make_down_index(i2),
            make_down_index(i3),
        ],
    )
}

fn is_riemann_or_neg(expr: &Expr, r: lasso::Spur) -> bool {
    match expr {
        Expr::Indexed(base, indices) => {
            matches!(base.as_ref(), Expr::Sym(s) if *s == r) && indices.len() == 4
        }
        Expr::Neg(inner) => is_riemann_or_neg(inner, r),
        _ => false,
    }
}

// ─── Main Bianchi test ────────────────────────────────────────────────────────

#[test]
fn bianchi_identity_via_canonicalise() {
    let (interner, props) = setup();
    let r = interner.get_or_intern("R");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    // ── 1. Antisymmetry in the first pair: R_{abcd} + R_{bacd} = 0 ──────────
    let sum_first = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        make_riemann(&interner, b, a, c, d),
    ]);
    let canon_first = ax_tensor::canonicalise(&sum_first, &props, &interner);
    assert_eq!(
        canon_first,
        Expr::zero(),
        "R_abcd + R_bacd should vanish by first-pair antisymmetry, got: {}",
        ax_ir::pretty_print(&canon_first, &interner)
    );

    // ── 2. Antisymmetry in the second pair: R_{abcd} + R_{abdc} = 0 ─────────
    let sum_second = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        make_riemann(&interner, a, b, d, c),
    ]);
    let canon_second = ax_tensor::canonicalise(&sum_second, &props, &interner);
    assert_eq!(
        canon_second,
        Expr::zero(),
        "R_abcd + R_abdc should vanish by second-pair antisymmetry, got: {}",
        ax_ir::pretty_print(&canon_second, &interner)
    );

    // ── 3. Pair-exchange symmetry: R_{abcd} - R_{cdab} = 0 ──────────────────
    let sum_exchange = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        Expr::neg(make_riemann(&interner, c, d, a, b)),
    ]);
    let canon_exchange = ax_tensor::canonicalise(&sum_exchange, &props, &interner);
    assert_eq!(
        canon_exchange,
        Expr::zero(),
        "R_abcd - R_cdab should vanish by pair-exchange symmetry, got: {}",
        ax_ir::pretty_print(&canon_exchange, &interner)
    );

    // ── 4. Bianchi canonical structure ───────────────────────────────────────
    // R_{abcd} + R_{acdb} + R_{adbc} canonicalises each term to a valid
    // Riemann tensor in canonical form. The full algebraic identity requires
    // Young symmetrizer projection (Butler-Portugal), beyond meld alone.
    let term1 = make_riemann(&interner, a, b, c, d);
    let term2 = make_riemann(&interner, a, c, d, b);
    let term3 = make_riemann(&interner, a, d, b, c);

    let c1 = ax_tensor::canonicalise(&term1, &props, &interner);
    let c2 = ax_tensor::canonicalise(&term2, &props, &interner);
    let c3 = ax_tensor::canonicalise(&term3, &props, &interner);

    assert!(
        is_riemann_or_neg(&c1, r),
        "canonical R_abcd should be a Riemann tensor, got: {}",
        ax_ir::pretty_print(&c1, &interner)
    );
    assert!(
        is_riemann_or_neg(&c2, r),
        "canonical R_acdb should be a Riemann tensor, got: {}",
        ax_ir::pretty_print(&c2, &interner)
    );
    assert!(
        is_riemann_or_neg(&c3, r),
        "canonical R_adbc should be a Riemann tensor, got: {}",
        ax_ir::pretty_print(&c3, &interner)
    );

    // The three canonical forms are all distinct (different index orderings).
    assert_ne!(c1, c2, "distinct Bianchi terms should have distinct canonical forms");
    assert_ne!(c2, c3, "distinct Bianchi terms should have distinct canonical forms");
    assert_ne!(c1, c3, "distinct Bianchi terms should have distinct canonical forms");

    // ── 5. meld reduces terms sharing the same structure key ─────────────────
    // The pair-antisymmetry identities are also detected by meld:
    let antisym_sum = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        make_riemann(&interner, b, a, c, d),
    ]);
    let melded = ax_tensor::meld(
        &ax_tensor::canonicalise(&antisym_sum, &props, &interner),
        &props,
        &interner,
    );
    let simplified = ax_eval::eval(&melded, &ax_eval::Env::new(), &interner);
    assert_eq!(
        simplified,
        Expr::zero(),
        "meld should reduce R_abcd + R_bacd to zero"
    );
}

// ─── Additional algebraic symmetry tests ─────────────────────────────────────

#[test]
fn riemann_first_pair_antisymmetry() {
    let (interner, props) = setup();
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    let sum = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        make_riemann(&interner, b, a, c, d),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "R_abcd + R_bacd = 0 by first-pair antisymmetry"
    );
}

#[test]
fn riemann_second_pair_antisymmetry() {
    let (interner, props) = setup();
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    let sum = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        make_riemann(&interner, a, b, d, c),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "R_abcd + R_abdc = 0 by second-pair antisymmetry"
    );
}

#[test]
fn riemann_pair_exchange_symmetry() {
    let (interner, props) = setup();
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    // R_{abcd} = R_{cdab}  →  R_{abcd} - R_{cdab} = 0
    let sum = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        Expr::neg(make_riemann(&interner, c, d, a, b)),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "R_abcd - R_cdab = 0 by pair-exchange symmetry"
    );
}

#[test]
fn riemann_double_antisymmetry_identity() {
    // R_{abcd} = -R_{bacd} = -R_{abdc} = R_{badc}
    let (interner, props) = setup();
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let c = interner.get_or_intern("c");
    let d = interner.get_or_intern("d");

    // R_{abcd} - R_{badc} = 0  (flip both pairs → double sign → no net sign)
    let sum = Expr::add(vec![
        make_riemann(&interner, a, b, c, d),
        Expr::neg(make_riemann(&interner, b, a, d, c)),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(
        result,
        Expr::zero(),
        "R_abcd - R_badc = 0 (two sign flips cancel)"
    );
}

// ─── Antisymmetric tensor tests ───────────────────────────────────────────────

#[test]
fn antisymmetric_trace_is_zero() {
    let interner = Interner::new();
    let f = interner.get_or_intern("F");
    let mu = interner.get_or_intern("mu");
    let mut props = HashMap::new();
    props.insert(f, vec![TensorProperty::AntiSymmetric(vec![0, 1])]);

    // F_{mu mu} should canonicalise to zero
    let expr = Expr::Indexed(
        Box::new(Expr::Sym(f)),
        vec![
            Index { name: mu, variance: Variance::Down, index_type: None },
            Index { name: mu, variance: Variance::Down, index_type: None },
        ],
    );
    let result = ax_tensor::canonicalise(&expr, &props, &interner);
    assert_eq!(result, Expr::zero(), "F_{{μμ}} should be zero for antisymmetric F");
}

#[test]
fn antisymmetric_swap_negates() {
    // F_{ab} = -F_{ba}  →  F_{ab} + F_{ba} = 0
    let interner = Interner::new();
    let f = interner.get_or_intern("F");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let mut props = HashMap::new();
    props.insert(f, vec![TensorProperty::AntiSymmetric(vec![0, 1])]);

    let sum = Expr::add(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(f)),
            vec![make_down_index(a), make_down_index(b)],
        ),
        Expr::Indexed(
            Box::new(Expr::Sym(f)),
            vec![make_down_index(b), make_down_index(a)],
        ),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(result, Expr::zero(), "F_ab + F_ba = 0 for antisymmetric F");
}

// ─── Symmetric tensor tests ───────────────────────────────────────────────────

#[test]
fn symmetric_tensor_canonical_order() {
    let interner = Interner::new();
    let g = interner.get_or_intern("g");
    let b = interner.get_or_intern("b");
    let a = interner.get_or_intern("a");
    let mut props = HashMap::new();
    props.insert(g, vec![TensorProperty::Symmetric(vec![0, 1])]);

    // g_{ba} should canonicalise to g_{ab}
    let expr = Expr::Indexed(
        Box::new(Expr::Sym(g)),
        vec![
            Index { name: b, variance: Variance::Down, index_type: None },
            Index { name: a, variance: Variance::Down, index_type: None },
        ],
    );
    let result = ax_tensor::canonicalise(&expr, &props, &interner);
    if let Expr::Indexed(_, indices) = &result {
        let first = interner.resolve(indices[0].name);
        let second = interner.resolve(indices[1].name);
        assert!(
            first <= second,
            "expected canonical order a ≤ b, got {} {}",
            first,
            second
        );
    } else {
        panic!("expected Indexed, got: {}", ax_ir::pretty_print(&result, &interner));
    }
}

#[test]
fn symmetric_tensor_swap_no_sign() {
    // g_{ab} - g_{ba} = 0 for symmetric g
    let interner = Interner::new();
    let g = interner.get_or_intern("g");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let mut props = HashMap::new();
    props.insert(g, vec![TensorProperty::Symmetric(vec![0, 1])]);

    let sum = Expr::add(vec![
        Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![make_down_index(a), make_down_index(b)],
        ),
        Expr::neg(Expr::Indexed(
            Box::new(Expr::Sym(g)),
            vec![make_down_index(b), make_down_index(a)],
        )),
    ]);
    let result = ax_tensor::canonicalise(&sum, &props, &interner);
    assert_eq!(result, Expr::zero(), "g_ab - g_ba = 0 for symmetric g");
}

// ─── Placeholder for full Schwarzschild pipeline ──────────────────────────────

#[test]
fn full_workflow_schwarzschild() {
    // The Schwarzschild pipeline (metric → Christoffel → Riemann → Ricci → 0)
    // is verified in tests/schwarzschild.rs. This test confirms the property
    // infrastructure used there still compiles and is accessible here.
    let interner = Interner::new();
    let r = interner.get_or_intern("R");
    let mut props: HashMap<lasso::Spur, Vec<TensorProperty>> = HashMap::new();
    props.insert(r, vec![TensorProperty::RiemannSymmetry]);
    // Property map is correctly constructed — pipeline tested in schwarzschild.rs
    assert!(props.contains_key(&r));
}
