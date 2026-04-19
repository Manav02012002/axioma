use ax_ir::*;
use ax_render::*;

fn int() -> Interner {
    Interner::new()
}

#[test]
fn latex_fraction() {
    let interner = int();
    let expr = Expr::Rational(num_rational::BigRational::new(1.into(), 3.into()));
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains("frac") || latex.contains('1') && latex.contains('3'),
        "1/3 LaTeX should use frac, got {}",
        latex
    );
}

#[test]
fn latex_greek() {
    let interner = int();
    let alpha = interner.get_or_intern("alpha");
    let expr = Expr::Sym(alpha);
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains("\\alpha"),
        "alpha should render as \\alpha, got {}",
        latex
    );
}

#[test]
fn latex_indexed_subscript() {
    let interner = int();
    let t = interner.get_or_intern("T");
    let mu = interner.get_or_intern("mu");
    let expr = Expr::Indexed(
        Box::new(Expr::Sym(t)),
        vec![Index {
            name: mu,
            variance: Variance::Down,
            index_type: None,
        }],
    );
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains('_') || latex.contains("mu"),
        "subscript index should appear, got {}",
        latex
    );
}

#[test]
fn latex_indexed_superscript() {
    let interner = int();
    let v = interner.get_or_intern("V");
    let mu = interner.get_or_intern("mu");
    let expr = Expr::Indexed(
        Box::new(Expr::Sym(v)),
        vec![Index {
            name: mu,
            variance: Variance::Up,
            index_type: None,
        }],
    );
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains('^'),
        "superscript index should appear, got {}",
        latex
    );
}

#[test]
fn unicode_greek() {
    let interner = int();
    let omega = interner.get_or_intern("omega");
    let expr = Expr::Sym(omega);
    let unicode = to_unicode(&expr, &interner);
    assert!(
        unicode.contains('ω'),
        "omega should render as ω, got {}",
        unicode
    );
}

#[test]
fn latex_power() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let expr = Expr::pow(Expr::Sym(x), Expr::Int(3.into()));
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains('^') && latex.contains('3'),
        "x^3 should have ^ and 3, got {}",
        latex
    );
}

#[test]
fn latex_sin() {
    let interner = int();
    let sin = interner.get_or_intern("sin");
    let x = interner.get_or_intern("x");
    let expr = Expr::Call(sin, vec![Expr::Sym(x)]);
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains("sin"),
        "sin(x) should contain sin, got {}",
        latex
    );
}

#[test]
fn unicode_neg() {
    let interner = int();
    let x = interner.get_or_intern("x");
    let expr = Expr::neg(Expr::Sym(x));
    let unicode = to_unicode(&expr, &interner);
    assert!(
        unicode.contains('-') || unicode.contains('−'),
        "negation should show -, got {}",
        unicode
    );
}

#[test]
fn unicode_ket_render() {
    let interner = int();
    let ket = interner.get_or_intern("ket");
    let psi = interner.get_or_intern("psi");
    let expr = Expr::Call(ket, vec![Expr::Sym(psi)]);
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains("|psi⟩"), "got {}", unicode);
}

#[test]
fn unicode_bra_render() {
    let interner = int();
    let bra = interner.get_or_intern("bra");
    let phi = interner.get_or_intern("phi");
    let expr = Expr::Call(bra, vec![Expr::Sym(phi)]);
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains("⟨phi|"), "got {}", unicode);
}

#[test]
fn unicode_braket_render() {
    let interner = int();
    let braket = interner.get_or_intern("braket");
    let bra = interner.get_or_intern("bra");
    let ket = interner.get_or_intern("ket");
    let phi = interner.get_or_intern("phi");
    let psi = interner.get_or_intern("psi");
    let expr = Expr::Call(
        braket,
        vec![
            Expr::Call(bra, vec![Expr::Sym(phi)]),
            Expr::Call(ket, vec![Expr::Sym(psi)]),
        ],
    );
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains("⟨phi|psi⟩"), "got {}", unicode);
}

#[test]
fn unicode_dagger_render() {
    let interner = int();
    let dagger = interner.get_or_intern("dagger");
    let a = interner.get_or_intern("A");
    let expr = Expr::Call(dagger, vec![Expr::Sym(a)]);
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains('†'), "got {}", unicode);
}

#[test]
fn unicode_tensor_product_render() {
    let interner = int();
    let tensor_product = interner.get_or_intern("tensor_product");
    let a = interner.get_or_intern("A");
    let b = interner.get_or_intern("B");
    let expr = Expr::Call(tensor_product, vec![Expr::Sym(a), Expr::Sym(b)]);
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains('⊗'), "got {}", unicode);
}

#[test]
fn latex_matrix() {
    let interner = int();
    let expr = Expr::Matrix(vec![
        vec![Expr::Int(1.into()), Expr::Int(2.into())],
        vec![Expr::Int(3.into()), Expr::Int(4.into())],
    ]);
    let latex = to_latex(&expr, &interner);
    assert!(
        latex.contains("matrix") || latex.contains("pmatrix") || latex.contains("begin"),
        "matrix should render as LaTeX matrix, got {}",
        latex
    );
}

#[test]
fn latex_ket_render() {
    let interner = int();
    let ket = interner.get_or_intern("ket");
    let psi = interner.get_or_intern("psi");
    let expr = Expr::Call(ket, vec![Expr::Sym(psi)]);
    let latex = to_latex(&expr, &interner);
    assert!(latex.contains("\\left|"), "got {}", latex);
    assert!(latex.contains("\\right\\rangle"), "got {}", latex);
}

#[test]
fn latex_braket_render() {
    let interner = int();
    let braket = interner.get_or_intern("braket");
    let bra = interner.get_or_intern("bra");
    let ket = interner.get_or_intern("ket");
    let phi = interner.get_or_intern("phi");
    let psi = interner.get_or_intern("psi");
    let expr = Expr::Call(
        braket,
        vec![
            Expr::Call(bra, vec![Expr::Sym(phi)]),
            Expr::Call(ket, vec![Expr::Sym(psi)]),
        ],
    );
    let latex = to_latex(&expr, &interner);
    assert!(latex.contains("\\left\\langle"), "got {}", latex);
    assert!(
        latex.contains("\\middle|") || latex.contains("\\mid"),
        "got {}",
        latex
    );
    assert!(latex.contains("\\right\\rangle"), "got {}", latex);
}

#[test]
fn latex_dagger_render() {
    let interner = int();
    let dagger = interner.get_or_intern("dagger");
    let a = interner.get_or_intern("A");
    let expr = Expr::Call(dagger, vec![Expr::Sym(a)]);
    let latex = to_latex(&expr, &interner);
    assert!(latex.contains("\\dagger"), "got {}", latex);
}

#[test]
fn latex_tensor_product_render() {
    let interner = int();
    let tensor_product = interner.get_or_intern("tensor_product");
    let a = interner.get_or_intern("A");
    let b = interner.get_or_intern("B");
    let expr = Expr::Call(tensor_product, vec![Expr::Sym(a), Expr::Sym(b)]);
    let latex = to_latex(&expr, &interner);
    assert!(latex.contains("\\otimes"), "got {}", latex);
}

#[test]
fn unicode_outer_bra_ket_render() {
    let interner = int();
    let outer = interner.get_or_intern("outer");
    let ket = interner.get_or_intern("ket");
    let bra = interner.get_or_intern("bra");
    let a = interner.get_or_intern("a");
    let b = interner.get_or_intern("b");
    let expr = Expr::Call(
        outer,
        vec![
            Expr::Call(ket, vec![Expr::Sym(a)]),
            Expr::Call(bra, vec![Expr::Sym(b)]),
        ],
    );
    let unicode = to_unicode(&expr, &interner);
    assert!(unicode.contains("|a⟩⟨b|"), "got {}", unicode);
}

#[test]
fn renders_young_diagram_ascii() {
    assert_eq!(render_young_diagram_ascii(&[2, 1]), "[][]\n[]");
}

#[test]
fn renders_young_diagram_unicode() {
    assert_eq!(render_young_diagram_unicode(&[2, 1]), "□ □\n□");
}

#[test]
fn renders_tableau_slot_map_ascii() {
    assert_eq!(
        render_tableau_slot_map_ascii(&[2, 1], &[0, 1, 2]),
        "[0][1]\n[2]"
    );
}

#[test]
fn renders_tensor_symmetry_summary() {
    let sym = TensorSymmetry {
        tableaux: vec![TableauAttachment {
            shape: vec![2, 1],
            slot_map: vec![0, 1, 2],
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
    };

    let summary = render_tensor_symmetry_summary(&sym);
    assert!(summary.contains("tableau[0]:"));
    assert!(summary.contains("shape=[2, 1]"));
    assert!(summary.contains("slots=[0, 1, 2]"));
}

#[test]
fn renders_power_sum_expansion_lines() {
    let shape = ax_young::YoungDiagram::try_new(vec![2]).unwrap();
    let rendered = render_power_sum_expansion(&ax_young::frobenius_characteristic(&shape).unwrap());
    assert!(rendered.contains("p_[2]"));
}

#[test]
fn renders_monomial_expansion_lines() {
    let monomial = ax_young::schur_to_monomial(&ax_young::SchurExpansion::from_shape(
        ax_young::YoungDiagram::try_new(vec![2, 1]).unwrap(),
    ))
    .unwrap();
    let rendered = render_monomial_expansion(&monomial);
    assert!(rendered.contains("m_[1, 1, 1]"));
}

#[test]
fn renders_multiplicity_basis_trace_lines() {
    let trace = ax_young::multiplicity_basis_trace(
        &[
            ax_young::YoungDiagram::try_new(vec![1]).unwrap(),
            ax_young::YoungDiagram::try_new(vec![1]).unwrap(),
            ax_young::YoungDiagram::try_new(vec![1]).unwrap(),
        ],
        &ax_young::YoungDiagram::try_new(vec![2, 1]).unwrap(),
    )
    .unwrap();
    let rendered = render_multiplicity_basis_trace(&trace);
    assert!(rendered.contains("target=[2, 1]"));
    assert!(rendered.contains("left_basis="));
    assert!(rendered.contains("right_basis="));
}
