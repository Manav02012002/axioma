#![forbid(unsafe_code)]

use ax_ir::Expr;
use num_traits::ToPrimitive;

#[derive(Clone, Copy, Debug)]
pub enum Target {
    Python,
    Rust,
    Cpp,
}

fn render_sym(sym: lasso::Spur, interner: &ax_ir::Interner) -> String {
    interner.resolve(sym).to_string()
}

fn join_args(
    args: &[Expr],
    interner: &ax_ir::Interner,
    f: fn(&Expr, &ax_ir::Interner) -> String,
) -> String {
    args.iter()
        .map(|arg| f(arg, interner))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn to_python(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Float(f) => format!("{:.17}", f),
        Expr::Complex(re, im) => {
            format!(
                "complex({}, {})",
                to_python(re, interner),
                to_python(im, interner)
            )
        }
        Expr::Sym(s) => render_sym(*s, interner),
        Expr::Add(terms) => terms
            .iter()
            .map(|term| match term {
                Expr::Neg(inner) => format!("-({})", to_python(inner, interner)),
                _ => to_python(term, interner),
            })
            .collect::<Vec<_>>()
            .join(" + "),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| to_python(factor, interner))
            .collect::<Vec<_>>()
            .join(" * "),
        Expr::Pow(base, exp) => {
            if matches!(exp.as_ref(), Expr::Rational(r) if *r == num_rational::BigRational::new(1.into(), 2.into()))
            {
                format!("math.sqrt({})", to_python(base, interner))
            } else {
                format!(
                    "{}**{}",
                    to_python(base, interner),
                    to_python(exp, interner)
                )
            }
        }
        Expr::Neg(inner) => format!("-({})", to_python(inner, interner)),
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            let args_str = join_args(args, interner, to_python);
            match name {
                "sin" => format!("math.sin({args_str})"),
                "cos" => format!("math.cos({args_str})"),
                "exp" => format!("math.exp({args_str})"),
                "log" => format!("math.log({args_str})"),
                "sqrt" => format!("math.sqrt({args_str})"),
                "abs" => format!("abs({args_str})"),
                _ => format!("{name}({args_str})"),
            }
        }
        Expr::Matrix(rows) => {
            let rows = rows
                .iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter()
                            .map(|cell| to_python(cell, interner))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("np.array([{rows}])")
        }
        Expr::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| to_python(item, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!("# unsupported: {:?}", expr),
    }
}

pub fn to_rust(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => {
            if let Some(v) = n.to_i64() {
                format!("{v}i64")
            } else {
                n.to_string()
            }
        }
        Expr::Rational(r) => format!("{}.0 / {}.0", r.numer(), r.denom()),
        Expr::Float(f) => format!("{:.17}f64", f),
        Expr::Complex(re, im) => {
            format!(
                "num::Complex::new({}, {})",
                to_rust(re, interner),
                to_rust(im, interner)
            )
        }
        Expr::Sym(s) => render_sym(*s, interner),
        Expr::Add(terms) => terms
            .iter()
            .map(|term| to_rust(term, interner))
            .collect::<Vec<_>>()
            .join(" + "),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| to_rust(factor, interner))
            .collect::<Vec<_>>()
            .join(" * "),
        Expr::Pow(base, exp) => match exp.as_ref() {
            Expr::Int(n) => format!("({}).powi({})", to_rust(base, interner), n),
            _ => format!(
                "({}).powf({})",
                to_rust(base, interner),
                to_rust(exp, interner)
            ),
        },
        Expr::Neg(inner) => format!("-({})", to_rust(inner, interner)),
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            let args_str = join_args(args, interner, to_rust);
            match (name, args.as_slice()) {
                ("sin", [arg]) => format!("({}).sin()", to_rust(arg, interner)),
                ("cos", [arg]) => format!("({}).cos()", to_rust(arg, interner)),
                ("exp", [arg]) => format!("({}).exp()", to_rust(arg, interner)),
                ("log", [arg]) => format!("({}).ln()", to_rust(arg, interner)),
                ("sqrt", [arg]) => format!("({}).sqrt()", to_rust(arg, interner)),
                ("abs", [arg]) => format!("({}).abs()", to_rust(arg, interner)),
                _ => format!("{name}({args_str})"),
            }
        }
        Expr::Matrix(rows) => {
            let rows = rows
                .iter()
                .map(|row| {
                    format!(
                        "vec![{}]",
                        row.iter()
                            .map(|cell| to_rust(cell, interner))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("vec![{rows}]")
        }
        Expr::List(items) => format!(
            "vec![{}]",
            items
                .iter()
                .map(|item| to_rust(item, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!("/* unsupported: {:?} */", expr),
    }
}

pub fn to_cpp(expr: &ax_ir::Expr, interner: &ax_ir::Interner) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Rational(r) => format!("(double){}/{}", r.numer(), r.denom()),
        Expr::Float(f) => format!("{:.17}", f),
        Expr::Complex(re, im) => format!(
            "std::complex<double>({}, {})",
            to_cpp(re, interner),
            to_cpp(im, interner)
        ),
        Expr::Sym(s) => render_sym(*s, interner),
        Expr::Add(terms) => terms
            .iter()
            .map(|term| to_cpp(term, interner))
            .collect::<Vec<_>>()
            .join(" + "),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| to_cpp(factor, interner))
            .collect::<Vec<_>>()
            .join(" * "),
        Expr::Pow(base, exp) => {
            format!("pow({}, {})", to_cpp(base, interner), to_cpp(exp, interner))
        }
        Expr::Neg(inner) => format!("-({})", to_cpp(inner, interner)),
        Expr::Call(f, args) => {
            let name = interner.resolve(*f);
            let mapped = match name {
                "sin" => "std::sin",
                "cos" => "std::cos",
                "exp" => "std::exp",
                "log" => "std::log",
                "sqrt" => "std::sqrt",
                "abs" => "std::abs",
                _ => name,
            };
            format!("{}({})", mapped, join_args(args, interner, to_cpp))
        }
        Expr::Matrix(_) => "// matrix code generation not supported in simple C++ mode".into(),
        Expr::List(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|item| to_cpp(item, interner))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!("/* unsupported: {:?} */", expr),
    }
}

pub fn generate(
    expr: &ax_ir::Expr,
    target: Target,
    interner: &ax_ir::Interner,
    fn_name: Option<&str>,
    params: &[lasso::Spur],
) -> String {
    let body = match target {
        Target::Python => to_python(expr, interner),
        Target::Rust => to_rust(expr, interner),
        Target::Cpp => to_cpp(expr, interner),
    };

    match (target, fn_name) {
        (_, None) => body,
        (Target::Python, Some(name)) => format!(
            "import math\nimport numpy as np\n\ndef {}({}):\n    return {}\n",
            name,
            params
                .iter()
                .map(|sym| render_sym(*sym, interner))
                .collect::<Vec<_>>()
                .join(", "),
            body
        ),
        (Target::Rust, Some(name)) => format!(
            "fn {}({}) -> f64 {{\n    {}\n}}\n",
            name,
            params
                .iter()
                .map(|sym| format!("{}: f64", render_sym(*sym, interner)))
                .collect::<Vec<_>>()
                .join(", "),
            body
        ),
        (Target::Cpp, Some(name)) => format!(
            "#include <cmath>\n#include <complex>\n\ndouble {}({}) {{\n    return {};\n}}\n",
            name,
            params
                .iter()
                .map(|sym| format!("double {}", render_sym(*sym, interner)))
                .collect::<Vec<_>>()
                .join(", "),
            body
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_simple() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::pow(ax_ir::Expr::Sym(x), ax_ir::Expr::Int(2.into()));
        let code = to_python(&expr, &interner);
        assert_eq!(code, "x**2");
    }

    #[test]
    fn python_sin() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let sin = interner.get_or_intern("sin");
        let expr = ax_ir::Expr::Call(sin, vec![ax_ir::Expr::Sym(x)]);
        let code = to_python(&expr, &interner);
        assert_eq!(code, "math.sin(x)");
    }

    #[test]
    fn rust_pow() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::pow(ax_ir::Expr::Sym(x), ax_ir::Expr::Int(3.into()));
        let code = to_rust(&expr, &interner);
        assert!(code.contains("powi(3)"), "got: {}", code);
    }

    #[test]
    fn python_function_gen() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = ax_ir::Expr::add(vec![
            ax_ir::Expr::pow(ax_ir::Expr::Sym(x), ax_ir::Expr::Int(2.into())),
            ax_ir::Expr::one(),
        ]);
        let code = generate(&expr, Target::Python, &interner, Some("my_func"), &[x]);
        assert!(code.contains("def my_func(x)"), "got: {}", code);
        assert!(code.contains("return"), "got: {}", code);
    }

    #[test]
    fn cpp_complex() {
        let interner = ax_ir::Interner::new();
        let expr = ax_ir::Expr::Complex(
            Box::new(ax_ir::Expr::Int(1.into())),
            Box::new(ax_ir::Expr::Int(2.into())),
        );
        let code = to_cpp(&expr, &interner);
        assert!(code.contains("complex"), "got: {}", code);
    }
}
