use crate::expr::{Expr, Index, Variance};
use crate::intern::Interner;

fn needs_parens_in_pow_base(expr: &Expr) -> bool {
    matches!(expr, Expr::Add(_) | Expr::Mul(_))
}

fn needs_parens_in_neg(expr: &Expr) -> bool {
    matches!(expr, Expr::Add(_))
}

fn render_index(index: &Index, interner: &Interner) -> String {
    let variance = match index.variance {
        Variance::Up => "+",
        Variance::Down => "-",
    };
    format!("{}{}", interner.resolve(index.name), variance)
}

fn render(expr: &Expr, interner: &Interner) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Float(f) => format!("{f:?}"),
        Expr::Sym(s) => interner.resolve(*s).to_string(),
        Expr::Add(terms) => {
            let mut out = String::new();
            for (idx, term) in terms.iter().enumerate() {
                if idx == 0 {
                    out.push_str(&render(term, interner));
                    continue;
                }

                if let Expr::Neg(inner) = term {
                    out.push_str(" - ");
                    let inner_s = render(inner, interner);
                    if needs_parens_in_neg(inner) {
                        out.push('(');
                        out.push_str(&inner_s);
                        out.push(')');
                    } else {
                        out.push_str(&inner_s);
                    }
                } else {
                    out.push_str(" + ");
                    out.push_str(&render(term, interner));
                }
            }
            out
        }
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| match factor {
                Expr::Add(_) => format!("({})", render(factor, interner)),
                _ => render(factor, interner),
            })
            .collect::<Vec<_>>()
            .join("*"),
        Expr::Pow(base, exp) => {
            let base_s = render(base, interner);
            let exp_s = render(exp, interner);
            let base_part = if needs_parens_in_pow_base(base) {
                format!("({base_s})")
            } else {
                base_s
            };
            format!("{base_part}^{exp_s}")
        }
        Expr::Neg(e) => {
            let inner = render(e, interner);
            if needs_parens_in_neg(e) {
                format!("-({inner})")
            } else {
                format!("-{inner}")
            }
        }
        Expr::Call(f, args) => format!(
            "{}({})",
            interner.resolve(*f),
            args.iter()
                .map(|arg| render(arg, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Indexed(base, indices) => format!(
            "{}[{}]",
            render(base, interner),
            indices
                .iter()
                .map(|index| render_index(index, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Let(name, val, body) => format!(
            "let {} = {} in {}",
            interner.resolve(*name),
            render(val, interner),
            render(body, interner)
        ),
        Expr::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| render(item, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Matrix(rows) => format!(
            "[{}]",
            rows.iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter()
                            .map(|item| render(item, interner))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub fn pretty_print(expr: &Expr, interner: &Interner) -> String {
    render(expr, interner)
}
