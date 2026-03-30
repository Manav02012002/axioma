use ax_ir::{Expr, Index, Variance};
use num_rational::BigRational;
const PREC_TOP: u8 = 0;
const PREC_ADD: u8 = 50;
const PREC_MUL: u8 = 60;
const PREC_UNARY: u8 = 70;
const PREC_POW: u8 = 80;

fn greek(name: &str) -> Option<&'static str> {
    match name {
        "alpha" => Some("α"),
        "beta" => Some("β"),
        "gamma" => Some("γ"),
        "delta" => Some("δ"),
        "epsilon" => Some("ε"),
        "zeta" => Some("ζ"),
        "eta" => Some("η"),
        "theta" => Some("θ"),
        "mu" => Some("μ"),
        "nu" => Some("ν"),
        "xi" => Some("ξ"),
        "pi" => Some("π"),
        "rho" => Some("ρ"),
        "sigma" => Some("σ"),
        "tau" => Some("τ"),
        "phi" => Some("φ"),
        "chi" => Some("χ"),
        "psi" => Some("ψ"),
        "omega" => Some("ω"),
        "Gamma" => Some("Γ"),
        "Delta" => Some("Δ"),
        "Theta" => Some("Θ"),
        "Lambda" => Some("Λ"),
        "Xi" => Some("Ξ"),
        "Pi" => Some("Π"),
        "Sigma" => Some("Σ"),
        "Phi" => Some("Φ"),
        "Psi" => Some("Ψ"),
        "Omega" => Some("Ω"),
        "lambda" => Some("λ"),
        "inf" | "infty" => Some("∞"),
        _ => None,
    }
}

fn sym_to_unicode(sym: lasso::Spur, interner: &ax_ir::Interner) -> String {
    let name = interner.resolve(sym);
    greek(name).unwrap_or(name).to_string()
}

fn common_fraction(r: &BigRational) -> Option<&'static str> {
    match (r.numer().to_string().as_str(), r.denom().to_string().as_str()) {
        ("1", "2") => Some("½"),
        ("1", "3") => Some("⅓"),
        ("1", "4") => Some("¼"),
        ("3", "4") => Some("¾"),
        ("-1", "2") => Some("-½"),
        ("-1", "3") => Some("-⅓"),
        ("-1", "4") => Some("-¼"),
        ("-3", "4") => Some("-¾"),
        _ => None,
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
    if s == "-0" { "0".to_string() } else { s }
}

fn superscript_digits(s: &str) -> Option<String> {
    let mut out = String::new();
    for ch in s.chars() {
        let mapped = match ch {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            '-' => '⁻',
            _ => return None,
        };
        out.push(mapped);
    }
    Some(out)
}

fn needs_paren(child_prec: u8, parent_prec: u8) -> bool {
    child_prec < parent_prec
}

fn render_with_paren(expr: &Expr, parent_prec: u8, interner: &ax_ir::Interner) -> String {
    let (text, child_prec) = render(expr, interner);
    if needs_paren(child_prec, parent_prec) {
        format!("({text})")
    } else {
        text
    }
}

fn is_number(expr: &Expr) -> bool {
    matches!(expr, Expr::Int(_) | Expr::Rational(_) | Expr::Float(_))
}

fn render_fractional_or_plain(r: &BigRational) -> String {
    if let Some(common) = common_fraction(r) {
        common.to_string()
    } else {
        format!("{}/{}", r.numer(), r.denom())
    }
}

fn render_call(name: &str, args: &[Expr], interner: &ax_ir::Interner) -> String {
    let rendered_args = args
        .iter()
        .map(|arg| render_with_paren(arg, PREC_TOP, interner))
        .collect::<Vec<_>>();

    match (name, args.len()) {
        ("sin", 1)
        | ("cos", 1)
        | ("tan", 1)
        | ("cot", 1)
        | ("sec", 1)
        | ("csc", 1)
        | ("arcsin", 1)
        | ("arccos", 1)
        | ("arctan", 1) => format!("{name}({})", rendered_args[0]),
        ("sqrt", 1) => format!("√({})", rendered_args[0]),
        ("diff", 2) => format!("∂{}/∂{}", rendered_args[0], rendered_args[1]),
        ("integrate", 2) => format!("∫ {} d{}", rendered_args[0], rendered_args[1]),
        ("integrate", 4) => format!(
            "∫_{}^{} {} d{}",
            rendered_args[2], rendered_args[3], rendered_args[0], rendered_args[1]
        ),
        ("sum", 4) => format!(
            "Σ_{{{}={}}}^{} {}",
            rendered_args[1], rendered_args[2], rendered_args[3], rendered_args[0]
        ),
        ("abs", 1) => format!("|{}|", rendered_args[0]),
        ("exp", 1) => {
            let (_, p) = render(&args[0], interner);
            let arg = if needs_paren(p, PREC_POW) {
                format!("({})", rendered_args[0])
            } else {
                rendered_args[0].clone()
            };
            format!("e^{}", arg)
        }
        _ => format!("{name}({})", rendered_args.join(", ")),
    }
}

fn render_index(index: &Index, interner: &ax_ir::Interner) -> String {
    sym_to_unicode(index.name, interner)
}

fn render_indexed(base: &Expr, indices: &[Index], interner: &ax_ir::Interner) -> String {
    let mut out = render_with_paren(base, PREC_POW, interner);
    for index in indices {
        let idx = render_index(index, interner);
        match index.variance {
            Variance::Down => out.push_str(&format!("_{idx}")),
            Variance::Up => out.push_str(&format!("^{idx}")),
        }
    }
    out
}

fn render(expr: &Expr, interner: &ax_ir::Interner) -> (String, u8) {
    match expr {
        Expr::Int(n) => (n.to_string(), PREC_POW),
        Expr::Rational(r) => (render_fractional_or_plain(r), PREC_POW),
        Expr::Float(f) => (format_float(*f), PREC_POW),
        Expr::Sym(s) => (sym_to_unicode(*s, interner), PREC_POW),
        Expr::Add(terms) => {
            let mut out = String::new();
            for (idx, term) in terms.iter().enumerate() {
                if idx == 0 {
                    out.push_str(&render_with_paren(term, PREC_ADD, interner));
                } else if let Expr::Neg(inner) = term {
                    out.push_str(" - ");
                    out.push_str(&render_with_paren(inner, PREC_ADD, interner));
                } else {
                    out.push_str(" + ");
                    out.push_str(&render_with_paren(term, PREC_ADD, interner));
                }
            }
            (out, PREC_ADD)
        }
        Expr::Mul(factors) => {
            let mut out = String::new();
            let mut prev_numeric = false;
            for (idx, factor) in factors.iter().enumerate() {
                let rendered = render_with_paren(factor, PREC_MUL, interner);
                let numeric = is_number(factor);
                if idx > 0 && prev_numeric && numeric {
                    out.push('·');
                }
                out.push_str(&rendered);
                prev_numeric = numeric;
            }
            (out, PREC_MUL)
        }
        Expr::Pow(base, exp) => {
            if let Expr::Rational(r) = exp.as_ref() {
                if *r == BigRational::new(1.into(), 2.into()) {
                    return (format!("√{}", render_with_paren(base, PREC_UNARY, interner)), PREC_UNARY);
                }
            }

            let base_text = render_with_paren(base, PREC_POW, interner);
            match exp.as_ref() {
                Expr::Int(n) => {
                    let n_str = n.to_string();
                    if let Some(sup) = superscript_digits(&n_str) {
                        (format!("{base_text}{sup}"), PREC_POW)
                    } else {
                        (format!("{base_text}^({})", render_with_paren(exp, PREC_TOP, interner)), PREC_POW)
                    }
                }
                _ => (
                    format!("{base_text}^({})", render_with_paren(exp, PREC_TOP, interner)),
                    PREC_POW,
                ),
            }
        }
        Expr::Neg(e) => (format!("-{}", render_with_paren(e, PREC_UNARY, interner)), PREC_UNARY),
        Expr::Call(f, args) => (render_call(interner.resolve(*f), args, interner), PREC_POW),
        Expr::Indexed(base, indices) => (render_indexed(base, indices, interner), PREC_POW),
        Expr::Let(name, val, body) => (
            format!(
                "let {} = {} in {}",
                sym_to_unicode(*name, interner),
                render_with_paren(val, PREC_TOP, interner),
                render_with_paren(body, PREC_TOP, interner)
            ),
            PREC_TOP,
        ),
        Expr::List(items) => (
            format!(
                "[{}]",
                items
                    .iter()
                    .map(|item| render_with_paren(item, PREC_TOP, interner))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            PREC_TOP,
        ),
        Expr::Matrix(rows) => (
            format!(
                "[{}]",
                rows.iter()
                    .map(|row| {
                        format!(
                            "[{}]",
                            row.iter()
                                .map(|cell| render_with_paren(cell, PREC_TOP, interner))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            PREC_TOP,
        ),
    }
}

pub fn to_unicode(expr: &Expr, interner: &ax_ir::Interner) -> String {
    render(expr, interner).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_greek() {
        let interner = ax_ir::Interner::new();
        let alpha = interner.get_or_intern("alpha");
        let s = to_unicode(&ax_ir::Expr::Sym(alpha), &interner);
        assert_eq!(s, "α");
    }

    #[test]
    fn unicode_power() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::pow(ax_ir::Expr::Sym(x), ax_ir::Expr::Int(2.into()));
        let s = to_unicode(&expr, &interner);
        assert_eq!(s, "x²");
    }

    #[test]
    fn unicode_sqrt() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let half = ax_ir::Expr::Rational(num_rational::BigRational::new(1.into(), 2.into()));
        let expr = ax_ir::Expr::Pow(Box::new(ax_ir::Expr::Sym(x)), Box::new(half));
        let s = to_unicode(&expr, &interner);
        assert!(s.contains("√"), "got: {}", s);
    }
}
