#![forbid(unsafe_code)]
#![allow(clippy::redundant_closure)]

pub mod oracle;
pub mod tableau;
pub mod unicode;

use ax_ir::{Assumption, Condition, Expr, Index, Variance};
use num_rational::BigRational;
use num_traits::Signed;

pub use oracle::render_oracle_trace;
pub use tableau::{
    render_tableau_slot_map_ascii, render_tensor_symmetry_summary, render_young_diagram_ascii,
    render_young_diagram_unicode,
};
pub use unicode::to_unicode;

pub fn render_classical_irrep_summary(
    summary: &ax_young::classical_groups::ClassicalIrrepSummary,
) -> String {
    format!(
        "family={:?} rank={} shape={:?} dimension={}",
        summary.family, summary.rank, summary.highest_weight.rows, summary.dimension
    )
}

pub fn render_mixed_tensor_symmetry_summary(sym: &ax_ir::MixedTensorSymmetry) -> String {
    sym.tableaux
        .iter()
        .enumerate()
        .map(|(index, tableau)| {
            let slots = tableau
                .slots
                .iter()
                .map(|slot| format!("({},{:?})", slot.index, slot.kind))
                .collect::<Vec<_>>()
                .join(", ");
            let label = tableau
                .label
                .as_ref()
                .map(|label| format!(" label={label}"))
                .unwrap_or_default();
            format!(
                "mixed_tableau[{index}]: shape={:?} slots=[{}]{}",
                tableau.shape, slots, label
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_delta_pairing_terms(terms: &[ax_tensor::epsilon_engine::DeltaPairingTerm]) -> String {
    terms
        .iter()
        .map(|term| format!("coeff={}; pairs={:?}", term.coefficient, term.pairings))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_linear_decomposition_terms(
    terms: &[ax_tensor::curvature_decompose::LinearDecompositionTerm],
) -> String {
    terms
        .iter()
        .map(|term| {
            format!(
                "kind={}; coeff={}/{}",
                term.kind, term.coefficient_numer, term.coefficient_denom
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_power_sum_expansion(
    exp: &ax_young::symmetric_functions::PowerSumExpansion,
) -> String {
    exp.terms
        .iter()
        .map(|(partition, coeff)| format!("{coeff} * p_{:?}", partition.rows))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_monomial_expansion(exp: &ax_young::symmetric_functions::MonomialExpansion) -> String {
    exp.terms
        .iter()
        .map(|(partition, coeff)| format!("{coeff} * m_{:?}", partition.rows))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render_multiplicity_basis_trace(trace: &ax_trace::MultiplicityBasisTrace) -> String {
    let mut lines = vec![
        format!("target={:?}", trace.target),
        format!("left_basis={:?}", trace.left_associated_basis),
        format!("right_basis={:?}", trace.right_associated_basis),
    ];
    lines.extend(trace.change_of_basis_matrix.iter().map(|row| {
        format!(
            "[{}]",
            row.iter()
                .map(|entry| entry.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }));
    lines.join("\n")
}

const PREC_TOP: u8 = 0;
const PREC_ADD_SUB: u8 = 50;
const PREC_MUL_DIV: u8 = 60;
const PREC_UNARY: u8 = 70;
const PREC_POW: u8 = 80;
const PREC_ATOM: u8 = 100;

fn needs_paren(child_prec: u8, parent_prec: u8) -> bool {
    child_prec < parent_prec
}

fn greek_latex(name: &str) -> Option<&'static str> {
    match name {
        "alpha" => Some("\\alpha"),
        "beta" => Some("\\beta"),
        "gamma" => Some("\\gamma"),
        "Gamma" => Some("\\Gamma"),
        "delta" => Some("\\delta"),
        "Delta" => Some("\\Delta"),
        "epsilon" => Some("\\epsilon"),
        "varepsilon" => Some("\\varepsilon"),
        "zeta" => Some("\\zeta"),
        "eta" => Some("\\eta"),
        "theta" => Some("\\theta"),
        "Theta" => Some("\\Theta"),
        "iota" => Some("\\iota"),
        "kappa" => Some("\\kappa"),
        "lambda" => Some("\\lambda"),
        "Lambda" => Some("\\Lambda"),
        "mu" => Some("\\mu"),
        "nu" => Some("\\nu"),
        "xi" => Some("\\xi"),
        "Xi" => Some("\\Xi"),
        "pi" => Some("\\pi"),
        "Pi" => Some("\\Pi"),
        "rho" => Some("\\rho"),
        "sigma" => Some("\\sigma"),
        "Sigma" => Some("\\Sigma"),
        "tau" => Some("\\tau"),
        "upsilon" => Some("\\upsilon"),
        "phi" => Some("\\phi"),
        "Phi" => Some("\\Phi"),
        "varphi" => Some("\\varphi"),
        "chi" => Some("\\chi"),
        "psi" => Some("\\psi"),
        "Psi" => Some("\\Psi"),
        "omega" => Some("\\omega"),
        "Omega" => Some("\\Omega"),
        "inf" | "infty" => Some("\\infty"),
        _ => None,
    }
}

fn symbol_to_latex(sym: lasso::Spur, interner: &ax_ir::Interner) -> String {
    let name = interner.resolve(sym);
    if let Some(latex) = greek_latex(name) {
        latex.to_string()
    } else if name.len() > 1 {
        format!("\\text{{{name}}}")
    } else {
        name.to_string()
    }
}

fn format_float(f: f64) -> String {
    let mut s = format!("{f:.6}");
    if let Some(dot) = s.find('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.len() == dot + 1 {
            s.pop();
        }
    }
    if s == "-0" {
        "0".to_string()
    } else {
        s
    }
}

fn wrap_if_needed(rendered: String, child_prec: u8, parent_prec: u8) -> String {
    if needs_paren(child_prec, parent_prec) {
        format!("\\left({rendered}\\right)")
    } else {
        rendered
    }
}

fn is_numeric_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(_) | Expr::Rational(_) | Expr::Float(_))
}

fn is_negative_one(expr: &Expr) -> bool {
    match expr {
        Expr::Int(n) => *n == (-1).into(),
        Expr::Rational(r) => *r == BigRational::from_integer((-1).into()),
        Expr::Float(f) => *f == -1.0,
        _ => false,
    }
}

fn render_joined(exprs: &[Expr], parent_prec: u8, interner: &ax_ir::Interner, sep: &str) -> String {
    exprs
        .iter()
        .map(|expr| render(expr, parent_prec, interner))
        .collect::<Vec<_>>()
        .join(sep)
}

fn spinor_label_latex(expr: &Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Sym(s) => symbol_to_latex(*s, interner),
        _ => render(expr, PREC_TOP, interner),
    }
}

fn render_add(terms: &[Expr], parent_prec: u8, interner: &ax_ir::Interner) -> String {
    let mut out = String::new();
    for (idx, term) in terms.iter().enumerate() {
        if idx == 0 {
            out.push_str(&render(term, PREC_ADD_SUB, interner));
            continue;
        }
        if let Expr::Neg(inner) = term {
            out.push_str(" - ");
            out.push_str(&render(inner, PREC_ADD_SUB, interner));
        } else {
            out.push_str(" + ");
            out.push_str(&render(term, PREC_ADD_SUB, interner));
        }
    }
    wrap_if_needed(out, PREC_ADD_SUB, parent_prec)
}

fn denominator_factor(factor: &Expr) -> Option<Expr> {
    match factor {
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Int(n) if *n == (-1).into() => Some(base.as_ref().clone()),
            Expr::Neg(inner) => Some(Expr::pow(base.as_ref().clone(), inner.as_ref().clone())),
            _ => None,
        },
        _ => None,
    }
}

fn render_mul_like(exprs: &[Expr], interner: &ax_ir::Interner) -> String {
    let mut out = String::new();
    let mut prev_numeric = false;
    for (idx, expr) in exprs.iter().enumerate() {
        let rendered = render(expr, PREC_MUL_DIV, interner);
        let cur_numeric = is_numeric_literal(expr);
        if idx > 0 && prev_numeric && cur_numeric {
            out.push_str(" \\cdot ");
        }
        out.push_str(&rendered);
        prev_numeric = cur_numeric;
    }
    out
}

fn render_mul(factors: &[Expr], parent_prec: u8, interner: &ax_ir::Interner) -> String {
    let mut numerator = Vec::new();
    let mut denominator = Vec::new();

    for factor in factors {
        if let Some(denom) = denominator_factor(factor) {
            denominator.push(denom);
        } else {
            numerator.push(factor.clone());
        }
    }

    if !denominator.is_empty() {
        let numer = if numerator.is_empty() {
            "1".to_string()
        } else {
            render_mul_like(&numerator, interner)
        };
        let denom = render_mul_like(&denominator, interner);
        let frac = format!("\\frac{{{numer}}}{{{denom}}}");
        return wrap_if_needed(frac, PREC_MUL_DIV, parent_prec);
    }

    let numeric_factors = factors
        .iter()
        .filter(|factor| is_numeric_literal(factor))
        .count();
    if numeric_factors == 1
        && factors.len() > 1
        && factors
            .iter()
            .find(|factor| is_numeric_literal(factor))
            .is_some_and(|factor| is_negative_one(factor))
    {
        let rest = factors
            .iter()
            .filter(|factor| !is_negative_one(factor))
            .cloned()
            .collect::<Vec<_>>();
        let rendered = format!("-{}", render_mul_like(&rest, interner));
        return wrap_if_needed(rendered, PREC_MUL_DIV, parent_prec);
    }

    let rendered = render_mul_like(factors, interner);
    wrap_if_needed(rendered, PREC_MUL_DIV, parent_prec)
}

fn render_pow(base: &Expr, exp: &Expr, parent_prec: u8, interner: &ax_ir::Interner) -> String {
    if let Expr::Rational(r) = exp {
        let half = BigRational::new(1.into(), 2.into());
        let neg_half = BigRational::new((-1).into(), 2.into());
        if *r == half {
            return format!("\\sqrt{{{}}}", render(base, PREC_TOP, interner));
        }
        if *r == neg_half {
            return format!(
                "\\frac{{1}}{{\\sqrt{{{}}}}}",
                render(base, PREC_TOP, interner)
            );
        }
    }

    if let Expr::Int(n) = exp {
        if *n == (-1).into() {
            return format!("\\frac{{1}}{{{}}}", render(base, PREC_MUL_DIV, interner));
        }
    }

    let base_rendered = match base {
        Expr::Add(_) | Expr::Mul(_) | Expr::Neg(_) => {
            format!("\\left({}\\right)", render(base, PREC_TOP, interner))
        }
        _ => render(base, PREC_POW, interner),
    };
    let exp_rendered = render(exp, PREC_TOP, interner);
    let rendered = format!("{{{base_rendered}}}^{{{exp_rendered}}}");
    wrap_if_needed(rendered, PREC_POW, parent_prec)
}

fn render_call(f: lasso::Spur, args: &[Expr], interner: &ax_ir::Interner) -> String {
    let name = interner.resolve(f);
    let rendered_args = args
        .iter()
        .map(|arg| render(arg, PREC_TOP, interner))
        .collect::<Vec<_>>();
    let spinor_labels_latex = || {
        args.iter()
            .map(|arg| spinor_label_latex(arg, interner))
            .collect::<Vec<_>>()
    };

    match (name, args.len()) {
        ("__eq", 2) => format!("{} = {}", rendered_args[0], rendered_args[1]),
        ("__angle", 2) => format!(
            "\\langle {}\\,{} \\rangle",
            spinor_label_latex(&args[0], interner),
            spinor_label_latex(&args[1], interner)
        ),
        ("__square", 2) => format!(
            "[{}\\,{}]",
            spinor_label_latex(&args[0], interner),
            spinor_label_latex(&args[1], interner)
        ),
        ("__mandelstam", 2) => format!(
            "s_{{{} {}}}",
            spinor_label_latex(&args[0], interner),
            spinor_label_latex(&args[1], interner)
        ),
        ("__mandelstam_multi", n) if n >= 2 => {
            format!("s_{{{}}}", spinor_labels_latex().join(" "))
        }
        ("__mandelstam3", 3) => format!(
            "s_{{{} {} {}}}",
            spinor_label_latex(&args[0], interner),
            spinor_label_latex(&args[1], interner),
            spinor_label_latex(&args[2], interner)
        ),
        ("__angle_chain", n) if n >= 2 => {
            let labels = spinor_labels_latex();
            format!(
                "\\langle {}\\mid {}\\mid {} \\rangle",
                labels[0],
                labels[1..n - 1].join("\\,"),
                labels[n - 1]
            )
        }
        ("__square_chain", n) if n >= 2 => {
            let labels = spinor_labels_latex();
            format!(
                "[{}\\mid {}\\mid {}]",
                labels[0],
                labels[1..n - 1].join("\\,"),
                labels[n - 1]
            )
        }
        ("__angle_square_chain", n) if n >= 2 => {
            let labels = spinor_labels_latex();
            format!(
                "\\langle {}\\mid {}\\mid {}]",
                labels[0],
                labels[1..n - 1].join("\\,"),
                labels[n - 1]
            )
        }
        ("__square_angle_chain", n) if n >= 2 => {
            let labels = spinor_labels_latex();
            format!(
                "[{}\\mid {}\\mid {} \\rangle",
                labels[0],
                labels[1..n - 1].join("\\,"),
                labels[n - 1]
            )
        }
        ("__four_bracket", 4) => format!(
            "\\langle {}\\,{}\\,{}\\,{} \\rangle",
            spinor_label_latex(&args[0], interner),
            spinor_label_latex(&args[1], interner),
            spinor_label_latex(&args[2], interner),
            spinor_label_latex(&args[3], interner)
        ),
        ("laplacian", 1) => format!("\\nabla^2 {}", rendered_args[0]),
        ("partial_i", 1) => format!("\\partial_i {}", rendered_args[0]),
        ("sin", 1)
        | ("cos", 1)
        | ("tan", 1)
        | ("cot", 1)
        | ("sec", 1)
        | ("csc", 1)
        | ("sinh", 1)
        | ("cosh", 1)
        | ("tanh", 1)
        | ("arcsin", 1)
        | ("asin", 1)
        | ("arccos", 1)
        | ("acos", 1)
        | ("arctan", 1)
        | ("atan", 1)
        | ("arcsinh", 1)
        | ("asinh", 1)
        | ("arccosh", 1)
        | ("acosh", 1)
        | ("arctanh", 1)
        | ("atanh", 1) => {
            let latex_name = match name {
                "sin" => "\\sin",
                "cos" => "\\cos",
                "tan" => "\\tan",
                "cot" => "\\cot",
                "sec" => "\\sec",
                "csc" => "\\csc",
                "asin" | "arcsin" => "\\arcsin",
                "acos" | "arccos" => "\\arccos",
                "atan" | "arctan" => "\\arctan",
                "asinh" | "arcsinh" => "\\operatorname{arcsinh}",
                "acosh" | "arccosh" => "\\operatorname{arccosh}",
                "atanh" | "arctanh" => "\\operatorname{arctanh}",
                "sinh" => "\\sinh",
                "cosh" => "\\cosh",
                "tanh" => "\\tanh",
                _ => unreachable!(),
            };
            format!("{}\\!\\left({}\\right)", latex_name, rendered_args[0])
        }
        ("sign", 1) | ("sgn", 1) => {
            format!(
                "\\operatorname{{sgn}}\\!\\left({}\\right)",
                rendered_args[0]
            )
        }
        ("atan2", 2) => {
            format!(
                "\\operatorname{{atan2}}\\!\\left({}, {}\\right)",
                rendered_args[0], rendered_args[1]
            )
        }
        ("log", 1) => format!("\\log\\!\\left({}\\right)", rendered_args[0]),
        ("ln", 1) => format!("\\ln\\!\\left({}\\right)", rendered_args[0]),
        ("exp", 1) => format!("e^{{{}}}", rendered_args[0]),
        ("sqrt", 1) => format!("\\sqrt{{{}}}", rendered_args[0]),
        ("abs", 1) => format!("\\left| {} \\right|", rendered_args[0]),
        ("diff", 2) => format!(
            "\\frac{{\\partial {}}}{{\\partial {}}}",
            rendered_args[0], rendered_args[1]
        ),
        ("diff", 3) => format!(
            "\\frac{{\\partial^{{{}}} {}}}{{\\partial {}^{{{}}}}}",
            rendered_args[2], rendered_args[0], rendered_args[1], rendered_args[2]
        ),
        ("integrate", 2) => format!("\\int {} \\, d\\,{}", rendered_args[0], rendered_args[1]),
        ("integrate", 4) => format!(
            "\\int_{{{}}}^{{{}}} {} \\, d\\,{}",
            rendered_args[2], rendered_args[3], rendered_args[0], rendered_args[1]
        ),
        ("sum", 4) => format!(
            "\\sum_{{{}={}}}^{{{}}} {}",
            rendered_args[1], rendered_args[2], rendered_args[3], rendered_args[0]
        ),
        _ => {
            let display = if let Some(greek) = greek_latex(name) {
                greek.to_string()
            } else {
                format!("\\text{{{name}}}")
            };
            format!("{display}\\!\\left({}\\right)", rendered_args.join(", "))
        }
    }
}

fn render_index_name(index: &Index, interner: &ax_ir::Interner) -> String {
    symbol_to_latex(index.name, interner)
}

fn render_assumption(assumption: &Assumption) -> &'static str {
    match assumption {
        Assumption::Real => "\\text{real}",
        Assumption::Positive => "\\text{positive}",
        Assumption::Negative => "\\text{negative}",
        Assumption::NonZero => "\\text{nonzero}",
        Assumption::Integer => "\\text{integer}",
        Assumption::Even => "\\text{even}",
        Assumption::Odd => "\\text{odd}",
    }
}

fn render_condition(condition: &Condition, interner: &ax_ir::Interner) -> String {
    match condition {
        Condition::Gt(a, b) => format!(
            "{} > {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::Lt(a, b) => format!(
            "{} < {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::Ge(a, b) => format!(
            "{} \\ge {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::Le(a, b) => format!(
            "{} \\le {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::Eq(a, b) => format!(
            "{} = {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::Ne(a, b) => format!(
            "{} \\ne {}",
            render(a, PREC_TOP, interner),
            render(b, PREC_TOP, interner)
        ),
        Condition::And(a, b) => format!(
            "\\left({}\\right) \\land \\left({}\\right)",
            render_condition(a, interner),
            render_condition(b, interner)
        ),
        Condition::Or(a, b) => format!(
            "\\left({}\\right) \\lor \\left({}\\right)",
            render_condition(a, interner),
            render_condition(b, interner)
        ),
        Condition::Not(c) => format!("\\lnot \\left({}\\right)", render_condition(c, interner)),
        Condition::True => "\\text{true}".to_string(),
        Condition::False => "\\text{false}".to_string(),
    }
}

fn render_indexed(base: &Expr, indices: &[Index], interner: &ax_ir::Interner) -> String {
    let mut out = render(base, PREC_ATOM, interner);
    let mut i = 0;
    let mut first = true;
    while i < indices.len() {
        let variance = &indices[i].variance;
        let mut group = vec![render_index_name(&indices[i], interner)];
        i += 1;
        while i < indices.len() && indices[i].variance == *variance {
            group.push(render_index_name(&indices[i], interner));
            i += 1;
        }
        let joined = group.join(" ");
        let prefix = if first { "" } else { "{}" };
        match variance {
            Variance::Down => out.push_str(&format!("{prefix}_{{{joined}}}")),
            Variance::Up => out.push_str(&format!("{prefix}^{{{joined}}}")),
        }
        first = false;
    }
    out
}

fn render(expr: &Expr, parent_prec: u8, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => {
            if n.is_negative() {
                format!("{{{n}}}")
            } else {
                n.to_string()
            }
        }
        Expr::Rational(r) => {
            let numer = r.numer();
            let denom = r.denom();
            if numer.is_negative() {
                format!("-\\frac{{{}}}{{{}}}", numer.abs(), denom)
            } else {
                format!("\\frac{{{numer}}}{{{denom}}}")
            }
        }
        Expr::Float(f) => format_float(*f),
        Expr::Complex(re, im) => {
            let re_s = render(re, PREC_ADD_SUB, interner);
            if matches!(im.as_ref(), Expr::Int(n) if *n == 0.into()) {
                re_s
            } else if matches!(re.as_ref(), Expr::Int(n) if *n == 0.into())
                && matches!(im.as_ref(), Expr::Int(n) if *n == 1.into())
            {
                "i".to_string()
            } else if matches!(re.as_ref(), Expr::Int(n) if *n == 0.into()) {
                format!("{}i", render(im, PREC_MUL_DIV, interner))
            } else if matches!(im.as_ref(), Expr::Int(n) if *n == 1.into()) {
                format!("{re_s} + i")
            } else if let Expr::Neg(inner) = im.as_ref() {
                format!("{re_s} - {}i", render(inner, PREC_MUL_DIV, interner))
            } else {
                format!("{re_s} + {}i", render(im, PREC_MUL_DIV, interner))
            }
        }
        Expr::Sym(s) => symbol_to_latex(*s, interner),
        Expr::Add(terms) => render_add(terms, parent_prec, interner),
        Expr::Mul(factors) => render_mul(factors, parent_prec, interner),
        Expr::Pow(base, exp) => render_pow(base, exp, parent_prec, interner),
        Expr::Neg(e) => {
            let inner = render(e, PREC_UNARY, interner);
            let rendered = format!("-{inner}");
            wrap_if_needed(rendered, PREC_UNARY, parent_prec)
        }
        Expr::Call(f, args) => render_call(*f, args, interner),
        Expr::FnDef(name, params, body) => format!(
            "{}\\!\\left({}\\right) = {}",
            symbol_to_latex(*name, interner),
            params
                .iter()
                .map(|param| symbol_to_latex(*param, interner))
                .collect::<Vec<_>>()
                .join(", "),
            render(body, PREC_TOP, interner)
        ),
        Expr::Rule(lhs, rhs, _) => format!(
            "{} \\Rightarrow {}",
            render(lhs, PREC_TOP, interner),
            render(rhs, PREC_TOP, interner)
        ),
        Expr::Import(path) => format!(
            "\\text{{import }} {}",
            path.iter()
                .map(|sym| symbol_to_latex(*sym, interner))
                .collect::<Vec<_>>()
                .join(".")
        ),
        Expr::Assume(name, assumptions) => format!(
            "\\text{{assume }} {}{}",
            symbol_to_latex(*name, interner),
            if assumptions.is_empty() {
                String::new()
            } else {
                format!(
                    "\\text{{ is }} {}",
                    assumptions
                        .iter()
                        .map(render_assumption)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ),
        Expr::SetConvention(field, value) => {
            format!("\\text{{convention }} {}\\;{}", field, value)
        }
        Expr::Piecewise(cases) => {
            let body = cases
                .iter()
                .map(|(value, condition)| {
                    format!(
                        "{} & \\text{{if }} {}",
                        render(value, PREC_TOP, interner),
                        render_condition(condition, interner)
                    )
                })
                .collect::<Vec<_>>()
                .join(" \\\\ ");
            format!("\\begin{{cases}} {body} \\end{{cases}}")
        }
        Expr::Indexed(base, indices) => render_indexed(base, indices, interner),
        Expr::Group(inner, _) => {
            let rendered = format!("\\left({}\\right)", render(inner, PREC_TOP, interner));
            wrap_if_needed(rendered, PREC_ATOM, parent_prec)
        }
        Expr::Let(name, val, body) => format!(
            "\\text{{let }} {} = {} \\text{{ in }} {}",
            symbol_to_latex(*name, interner),
            render(val, PREC_TOP, interner),
            render(body, PREC_TOP, interner)
        ),
        Expr::List(items) => format!(
            "\\left[ {} \\right]",
            render_joined(items, PREC_TOP, interner, ", ")
        ),
        Expr::Matrix(rows) => {
            let body = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| render(cell, PREC_TOP, interner))
                        .collect::<Vec<_>>()
                        .join(" & ")
                })
                .collect::<Vec<_>>()
                .join(" \\\\ ");
            format!("\\begin{{pmatrix}} {body} \\end{{pmatrix}}")
        }
    }
}

pub fn to_latex(expr: &Expr, interner: &ax_ir::Interner) -> String {
    render(expr, PREC_TOP, interner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_src(src: &str) -> String {
        let interner = ax_ir::Interner::new();
        let result = ax_core_ir::lower(src, &interner);
        let expr = result.expr.expect("expected expression");
        to_latex(&expr, &interner)
    }

    #[test]
    fn latex_integer() {
        assert_eq!(render_src("42;"), "42");
    }

    #[test]
    fn latex_greek() {
        let s = render_src("alpha;");
        assert_eq!(s, "\\alpha");
    }

    #[test]
    fn latex_power() {
        let s = render_src("x^2;");
        assert!(s.contains("^{2}"), "got: {}", s);
    }

    #[test]
    fn latex_fraction() {
        let interner = ax_ir::Interner::new();
        let r = ax_ir::Expr::Rational(num_rational::BigRational::new(1.into(), 3.into()));
        let s = to_latex(&r, &interner);
        assert_eq!(s, "\\frac{1}{3}");
    }

    #[test]
    fn latex_indexed_tensor() {
        let s = render_src("T[mu-, nu+];");
        assert!(s.contains("_{\\mu}"), "got: {}", s);
        assert!(s.contains("^{\\nu}"), "got: {}", s);
    }

    #[test]
    fn latex_sqrt() {
        let interner = ax_ir::Interner::new();
        let x = ax_ir::Expr::Sym(interner.get_or_intern("x"));
        let half = ax_ir::Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
        let expr = ax_ir::Expr::Pow(Box::new(x), Box::new(half));
        let s = to_latex(&expr, &interner);
        assert_eq!(s, "\\sqrt{x}");
    }

    #[test]
    fn latex_sinh() {
        let interner = ax_ir::Interner::new();
        let sinh_sym = interner.get_or_intern("sinh");
        let x = interner.get_or_intern("x");
        let expr = Expr::Call(sinh_sym, vec![Expr::Sym(x)]);
        let latex = to_latex(&expr, &interner);
        assert!(latex.contains("\\sinh"), "expected \\sinh, got: {}", latex);
    }

    #[test]
    fn classical_irrep_summary_render_contains_required_fields() {
        let shape = ax_young::YoungDiagram::try_new(vec![1]).unwrap();
        let summary = ax_young::classical_groups::summarize_classical_irrep(
            ax_young::classical_groups::ClassicalGroupFamily::Symplectic,
            2,
            &shape,
        )
        .unwrap();
        let rendered = render_classical_irrep_summary(&summary);
        assert!(rendered.contains("family=Symplectic"));
        assert!(rendered.contains("rank=2"));
        assert!(rendered.contains("shape=[1]"));
        assert!(rendered.contains("dimension=4"));
    }

    #[test]
    fn mixed_tensor_summary_render_contains_required_fields() {
        let rendered =
            render_mixed_tensor_symmetry_summary(&ax_spinor::symmetric_two_undotted_spinors());
        assert!(rendered.contains("mixed_tableau[0]:"));
        assert!(rendered.contains("shape=[2]"));
        assert!(rendered.contains("UndottedSpinor"));
    }

    #[test]
    fn delta_pairing_terms_render_rank_two_lines() {
        let rendered = render_delta_pairing_terms(
            &ax_tensor::epsilon_engine::epsilon_epsilon_to_delta_terms(2).unwrap(),
        );
        assert!(rendered.contains("coeff=1; pairs=[(0, 0), (1, 1)]"));
        assert!(rendered.contains("coeff=-1; pairs=[(0, 1), (1, 0)]"));
    }

    #[test]
    fn linear_decomposition_render_contains_riemann_terms() {
        let rendered = render_linear_decomposition_terms(
            &ax_tensor::curvature_decompose::riemann_to_weyl_ricci_scalar_coefficients(4).unwrap(),
        );
        assert!(rendered.contains("kind=weyl_rank4; coeff=1/1"));
        assert!(rendered.contains("kind=metric_ricci_rank4; coeff=1/2"));
        assert!(rendered.contains("kind=metric_scalar_rank4; coeff=-1/6"));
    }
}
