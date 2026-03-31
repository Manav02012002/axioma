use crate::expr::{Assumption, Condition, Expr, Index, Variance};
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

fn render_assumption(assumption: &Assumption) -> &'static str {
    match assumption {
        Assumption::Real => "real",
        Assumption::Positive => "positive",
        Assumption::Negative => "negative",
        Assumption::NonZero => "nonzero",
        Assumption::Integer => "integer",
        Assumption::Even => "even",
        Assumption::Odd => "odd",
    }
}

fn render_condition(condition: &Condition, interner: &Interner) -> String {
    match condition {
        Condition::Gt(a, b) => format!("{} > {}", render(a, interner), render(b, interner)),
        Condition::Lt(a, b) => format!("{} < {}", render(a, interner), render(b, interner)),
        Condition::Ge(a, b) => format!("{} >= {}", render(a, interner), render(b, interner)),
        Condition::Le(a, b) => format!("{} <= {}", render(a, interner), render(b, interner)),
        Condition::Eq(a, b) => format!("{} == {}", render(a, interner), render(b, interner)),
        Condition::Ne(a, b) => format!("{} != {}", render(a, interner), render(b, interner)),
        Condition::And(a, b) => format!(
            "({}) and ({})",
            render_condition(a, interner),
            render_condition(b, interner)
        ),
        Condition::Or(a, b) => format!(
            "({}) or ({})",
            render_condition(a, interner),
            render_condition(b, interner)
        ),
        Condition::Not(c) => format!("not ({})", render_condition(c, interner)),
        Condition::True => "true".to_string(),
        Condition::False => "false".to_string(),
    }
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
        Expr::FnDef(name, params, body) => format!(
            "{}({}) = {}",
            interner.resolve(*name),
            params
                .iter()
                .map(|param| interner.resolve(*param).to_string())
                .collect::<Vec<_>>()
                .join(", "),
            render(body, interner)
        ),
        Expr::Rule(lhs, rhs) => format!("{} => {}", render(lhs, interner), render(rhs, interner)),
        Expr::Import(path) => format!(
            "import {}",
            path.iter()
                .map(|sym| interner.resolve(*sym))
                .collect::<Vec<_>>()
                .join(".")
        ),
        Expr::Assume(name, assumptions) => format!(
            "assume {}{}",
            interner.resolve(*name),
            if assumptions.is_empty() {
                String::new()
            } else {
                format!(
                    " {}",
                    assumptions
                        .iter()
                        .map(render_assumption)
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
        ),
        Expr::Piecewise(cases) => format!(
            "piecewise({})",
            cases.iter()
                .map(|(value, condition)| format!(
                    "{}, {}",
                    render(value, interner),
                    render_condition(condition, interner)
                ))
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
