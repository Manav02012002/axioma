use ax_eval::{convention_entries, property_entries};
use ax_ir::TensorProperty;

pub const KEYWORDS: &[&str] = &[
    "let",
    "property",
    "assume",
    "rule",
    "indices",
    "coordinates",
    "convention",
    "depends",
    "weight",
    "grassmann",
    "if",
    "then",
    "else",
    "module",
    "import",
    "piecewise",
    "parallel",
    "in",
    "indexset",
    "operator",
];

pub const PROPERTY_NAMES: &[&str] = &[
    "symmetric",
    "antisymmetric",
    "riemann_symmetry",
    "riemann",
    "metric",
    "inverse_metric",
    "kronecker_delta",
    "kronecker",
    "epsilon",
    "epsilon_tensor",
    "traceless",
    "diagonal",
    "trace",
    "derivative",
    "partial_derivative",
    "covariant_derivative",
    "spinor",
    "dirac_bar",
    "gamma_matrix",
    "commuting",
    "anticommuting",
    "noncommuting",
    "sort_order",
    "tableau_symmetry",
    "bianchi",
    "satisfies_bianchi",
    "weyl",
    "weyl_tensor",
    "differential_form",
    "self_anticommuting",
    "self_noncommuting",
    "self_commuting",
    "commuting_as_product",
    "commuting_as_sum",
    "majorana",
    "implicit_index",
];

pub const GREEK_LETTERS: &[(&str, &str)] = &[
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("epsilon", "ε"),
    ("zeta", "ζ"),
    ("eta", "η"),
    ("theta", "θ"),
    ("iota", "ι"),
    ("kappa", "κ"),
    ("lambda", "λ"),
    ("mu", "μ"),
    ("nu", "ν"),
    ("xi", "ξ"),
    ("omicron", "ο"),
    ("pi", "π"),
    ("rho", "ρ"),
    ("sigma", "σ"),
    ("tau", "τ"),
    ("upsilon", "υ"),
    ("phi", "φ"),
    ("chi", "χ"),
    ("psi", "ψ"),
    ("omega", "ω"),
    ("Alpha", "Α"),
    ("Beta", "Β"),
    ("Gamma", "Γ"),
    ("Delta", "Δ"),
    ("Epsilon", "Ε"),
    ("Zeta", "Ζ"),
    ("Eta", "Η"),
    ("Theta", "Θ"),
    ("Iota", "Ι"),
    ("Kappa", "Κ"),
    ("Lambda", "Λ"),
    ("Mu", "Μ"),
    ("Nu", "Ν"),
    ("Xi", "Ξ"),
    ("Omicron", "Ο"),
    ("Pi", "Π"),
    ("Rho", "Ρ"),
    ("Sigma", "Σ"),
    ("Tau", "Τ"),
    ("Upsilon", "Υ"),
    ("Phi", "Φ"),
    ("Chi", "Χ"),
    ("Psi", "Ψ"),
    ("Omega", "Ω"),
    ("final_sigma", "ς"),
];

pub const CPT_CALLABLE_DOCS: &[(&str, &str)] = &[
    (
        "frw_background_spec",
        "Construct a compact FRW background CPT spec.",
    ),
    ("cpt_gauge", "Construct a compact CPT gauge spec."),
    ("cpt_matter", "Construct a compact CPT matter spec."),
    (
        "cpt_linearized_einstein",
        "Derive labelled linearized Einstein equations from CPT specs.",
    ),
    (
        "cpt_fluid_equations",
        "Return labelled perfect-fluid conservation equations.",
    ),
    (
        "cpt_quadratic_action",
        "Return the reduced quadratic action density for supported CPT matter.",
    ),
    (
        "cpt_mukhanov_sasaki",
        "Return the Mukhanov-Sasaki mode equation derived from the CPT action.",
    ),
    (
        "cpt_mukhanov_sasaki_first_order",
        "Return the Mukhanov-Sasaki first-order system.",
    ),
    (
        "cpt_bardeen_invariance",
        "Check Bardeen-potential gauge invariance in structured CPT form.",
    ),
    (
        "cpt_export_mode_rhs",
        "Export the Mukhanov-Sasaki mode RHS as target-language code.",
    ),
];

pub const QM_SNIPPETS: &[(&str, &str, &str)] = &[
    ("ket", "|${1:psi}>", "Dirac ket syntax."),
    ("bra", "<${1:phi}|", "Dirac bra syntax."),
    (
        "braket",
        "<${1:phi}|${2:psi}>",
        "Dirac inner-product syntax.",
    ),
    ("dagger", "${1:A}†", "Adjoint / Hermitian-conjugate syntax."),
    (
        "tensor_product",
        "${1:A} ⊗ ${2:B}",
        "Tensor-product syntax.",
    ),
];

pub fn greek_to_unicode(name: &str) -> Option<&'static str> {
    GREEK_LETTERS
        .iter()
        .find_map(|(entry, unicode)| (*entry == name).then_some(*unicode))
}

pub fn qm_snippet_documentation(name: &str) -> Option<&'static str> {
    QM_SNIPPETS
        .iter()
        .find_map(|(entry, _, documentation)| (*entry == name).then_some(*documentation))
}

pub fn property_documentation(name: &str) -> &'static str {
    if let Some(entry) = property_entries().into_iter().find(|entry| {
        let syntax = entry.syntax.to_ascii_lowercase();
        syntax.contains(&format!("property t {}", name))
            || syntax.contains(&format!("property r {}", name))
            || syntax.contains(&format!("property f {}", name))
            || syntax.contains(&format!("property psi {}", name))
            || syntax.contains(&format!("property {}", name))
    }) {
        return entry.description;
    }
    match name {
        "symmetric" => "Declare pairwise slot symmetry for a tensor.",
        "antisymmetric" => "Declare antisymmetry under slot exchange.",
        "riemann_symmetry" | "riemann" => "Apply the standard Riemann tensor symmetries.",
        "metric" => "Marks a tensor as a metric and implies symmetry.",
        "inverse_metric" => "Marks a tensor as an inverse metric used to raise indices.",
        "kronecker_delta" | "kronecker" => "Marks a tensor as a Kronecker delta.",
        "epsilon" | "epsilon_tensor" => "Marks a tensor as a Levi-Civita epsilon tensor.",
        "traceless" => "Marks a tensor as traceless.",
        "diagonal" => "Marks a tensor as diagonal.",
        "trace" => "Marks a tensor as a trace-like object.",
        "derivative" => "Marks a symbol as a derivative operator.",
        "partial_derivative" => "Marks a symbol as a partial derivative operator.",
        "covariant_derivative" => "Marks a symbol as a covariant derivative operator.",
        "spinor" => "Marks a symbol as a spinor.",
        "dirac_bar" => "Marks a symbol as a Dirac-bar object.",
        "gamma_matrix" => "Marks a symbol as a gamma matrix.",
        "commuting" => "Marks an object as commuting.",
        "anticommuting" => "Marks an object as anticommuting.",
        "noncommuting" => "Marks an object as noncommuting.",
        "sort_order" => "Declares an explicit preferred order of symbols.",
        "tableau_symmetry" => "Declares Young-tableau slot symmetry data.",
        "bianchi" | "satisfies_bianchi" => "Marks a tensor as satisfying a Bianchi identity.",
        "weyl" | "weyl_tensor" => "Marks a tensor as a Weyl tensor.",
        "differential_form" => "Associates differential-form structure with a symbol.",
        "self_anticommuting" => "Marks identical-head objects as anticommuting with themselves.",
        "self_noncommuting" => "Marks identical-head objects as noncommuting with themselves.",
        "self_commuting" => "Marks identical-head objects as commuting with themselves.",
        "commuting_as_product" => "Uses product-like commutativity rules when sorting factors.",
        "commuting_as_sum" => "Uses sum-like commutativity rules when sorting factors.",
        "majorana" => "Marks a spinor as Majorana.",
        "implicit_index" => "Marks an object as carrying implicit indices.",
        _ => "Tensor property.",
    }
}

pub fn convention_values(field: &str) -> &'static [&'static str] {
    if let Some(entry) = convention_entries()
        .into_iter()
        .find(|entry| entry.field == field)
    {
        return match entry.field {
            "metric_signature" => &["mostly_plus", "mostly_minus"],
            "riemann_sign" => &["mtw", "weinberg"],
            "ricci_contraction" => &["first_third", "first_fourth"],
            "levi_civita_norm" => &["plus_one", "minus_one", "sqrt_g"],
            "fourier_sign" => &["minus_i", "plus_i"],
            _ => &[],
        };
    }
    match field {
        "metric_signature" => &["mostly_plus", "mostly_minus"],
        "riemann_sign" => &["mtw", "weinberg"],
        "ricci_contraction" => &["first_third", "first_fourth"],
        "levi_civita_norm" => &["plus_one", "minus_one", "sqrt_g"],
        "fourier_sign" => &["minus_i", "plus_i"],
        _ => &[],
    }
}

pub fn _format_tensor_property(prop: &TensorProperty, interner: &ax_ir::Interner) -> String {
    ax_eval::registry::format_tensor_property(prop, interner)
}
