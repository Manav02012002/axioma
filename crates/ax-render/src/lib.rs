#![forbid(unsafe_code)]

use ax_ir::{Expr, Index, Variance};
use num_rational::BigRational;
use num_traits::Signed;

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
    exprs.iter()
        .map(|expr| render(expr, parent_prec, interner))
        .collect::<Vec<_>>()
        .join(sep)
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

    let numeric_factors = factors.iter().filter(|factor| is_numeric_literal(factor)).count();
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
            return format!("\\frac{{1}}{{\\sqrt{{{}}}}}", render(base, PREC_TOP, interner));
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

    match (name, args.len()) {
        ("sin", 1)
        | ("cos", 1)
        | ("tan", 1)
        | ("cot", 1)
        | ("sec", 1)
        | ("csc", 1)
        | ("arcsin", 1)
        | ("arccos", 1)
        | ("arctan", 1) => format!("\\{name}\\!\\left({}\\right)", rendered_args[0]),
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
        ("integrate", 2) => format!(
            "\\int {} \\, d\\,{}",
            rendered_args[0], rendered_args[1]
        ),
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
        Expr::Indexed(base, indices) => render_indexed(base, indices, interner),
        Expr::Let(name, val, body) => format!(
            "\\text{{let }} {} = {} \\text{{ in }} {}",
            symbol_to_latex(*name, interner),
            render(val, PREC_TOP, interner),
            render(body, PREC_TOP, interner)
        ),
        Expr::List(items) => format!("\\left[ {} \\right]", render_joined(items, PREC_TOP, interner, ", ")),
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
        let env = ax_eval::Env::new();
        let evaled = ax_eval::eval(&expr, &env, &interner);
        to_latex(&evaled, &interner)
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
}
