#![forbid(unsafe_code)]

use anyhow::Context;
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
                "dagger" => format!("dagger({args_str})"),
                "tensor_product" => format!("tensor_product({args_str})"),
                "commutator" => format!("commutator({args_str})"),
                "anticommutator" => format!("anticommutator({args_str})"),
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

/// Return a Python NumPy prelude containing helper definitions for QM matrix operations.
pub fn python_qm_prelude() -> &'static str {
    "import numpy as np

def dagger(x):
    return np.conjugate(x).T

def tensor_product(a, b):
    return np.kron(a, b)

def commutator(a, b):
    return a @ b - b @ a

def anticommutator(a, b):
    return a @ b + b @ a"
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

pub fn emit_python_function(
    name: &str,
    args: &[lasso::Spur],
    body: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> String {
    format!(
        "def {}({}):\n    return {}",
        name,
        args.iter()
            .map(|sym| render_sym(*sym, interner))
            .collect::<Vec<_>>()
            .join(", "),
        to_python(body, interner)
    )
}

/// Emit a Python function together with the QM NumPy helper prelude it depends on.
pub fn emit_python_qm_function(
    name: &str,
    args: &[lasso::Spur],
    body: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> String {
    format!(
        "{}\n\ndef {}({}):\n    return {}",
        python_qm_prelude(),
        name,
        args.iter()
            .map(|sym| render_sym(*sym, interner))
            .collect::<Vec<_>>()
            .join(", "),
        to_python(body, interner)
    )
}

pub fn emit_rust_function(
    name: &str,
    args: &[lasso::Spur],
    body: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> String {
    format!(
        "pub fn {}({}) -> f64 {{\n    {}\n}}",
        name,
        args.iter()
            .map(|sym| format!("{}: f64", render_sym(*sym, interner)))
            .collect::<Vec<_>>()
            .join(", "),
        to_rust(body, interner)
    )
}

pub fn emit_cpp_function(
    name: &str,
    args: &[lasso::Spur],
    body: &ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> String {
    format!(
        "double {}({}) {{\n    return {};\n}}",
        name,
        args.iter()
            .map(|sym| format!("double {}", render_sym(*sym, interner)))
            .collect::<Vec<_>>()
            .join(", "),
        to_cpp(body, interner)
    )
}

pub fn emit_tensor_symmetry_json(sym: &ax_ir::TensorSymmetry) -> anyhow::Result<String> {
    serde_json::to_string(&serde_json::json!({
        "tableaux": sym.tableaux.iter().map(|tableau| serde_json::json!({
            "shape": tableau.shape,
            "slot_map": tableau.slot_map,
            "label": tableau.label,
            "trace_free": tableau.trace_free,
            "duality": format!("{:?}", tableau.duality),
        })).collect::<Vec<_>>(),
        "inherits_under_derivative": sym.inherits_under_derivative,
        "inherits_under_tensor_product": sym.inherits_under_tensor_product,
        "inherits_under_contraction": sym.inherits_under_contraction,
        "preserves_trace_free_under_projection": sym.preserves_trace_free_under_projection,
    }))
    .context("failed to serialize tensor symmetry to JSON")
}

#[cfg(test)]
mod symmetry_tests {
    use super::*;
    use ax_ir::{
        DualityKind, RestrictedSymmetryMode, SymmetrySource, TableauAttachment, TensorSymmetry,
    };

    #[test]
    fn emits_tensor_symmetry_json_with_tableaux_key() {
        let symmetry = TensorSymmetry {
            tableaux: vec![TableauAttachment {
                shape: vec![2],
                slot_map: vec![0, 1],
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

        let json = emit_tensor_symmetry_json(&symmetry).unwrap();
        assert!(json.contains("\"tableaux\""), "{json}");
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

    #[test]
    fn emit_python_function_wraps_to_python_body() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let body = ax_ir::Expr::Add(vec![ax_ir::Expr::Sym(x), ax_ir::Expr::one()]);

        let code = emit_python_function("f", &[x], &body, &interner);

        assert_eq!(code, "def f(x):\n    return x + 1");
    }

    #[test]
    fn emit_rust_function_wraps_to_rust_body() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let body = ax_ir::Expr::Add(vec![ax_ir::Expr::Sym(x), ax_ir::Expr::one()]);

        let code = emit_rust_function("f", &[x], &body, &interner);

        assert_eq!(code, "pub fn f(x: f64) -> f64 {\n    x + 1i64\n}");
    }

    #[test]
    fn emit_cpp_function_wraps_to_cpp_body() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let body = ax_ir::Expr::Add(vec![ax_ir::Expr::Sym(x), ax_ir::Expr::one()]);

        let code = emit_cpp_function("f", &[x], &body, &interner);

        assert_eq!(code, "double f(double x) {\n    return x + 1;\n}");
    }

    #[test]
    fn python_qm_prelude_contains_helpers() {
        let prelude = python_qm_prelude();
        assert!(prelude.contains("import numpy as np"));
        assert!(prelude.contains("def dagger(x):"));
        assert!(prelude.contains("def tensor_product(a, b):"));
    }

    #[test]
    fn to_python_emits_tensor_product_helper_call() {
        let interner = ax_ir::Interner::new();
        let tensor_product = interner.get_or_intern("tensor_product");
        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let expr = ax_ir::Expr::Call(
            tensor_product,
            vec![ax_ir::Expr::Sym(a), ax_ir::Expr::Sym(b)],
        );
        let code = to_python(&expr, &interner);
        assert!(code.contains("tensor_product(A, B)"));
    }

    #[test]
    fn to_python_emits_dagger_helper_call() {
        let interner = ax_ir::Interner::new();
        let dagger = interner.get_or_intern("dagger");
        let a = interner.get_or_intern("A");
        let expr = ax_ir::Expr::Call(dagger, vec![ax_ir::Expr::Sym(a)]);
        let code = to_python(&expr, &interner);
        assert!(code.contains("dagger(A)"));
    }

    #[test]
    fn emit_python_qm_function_includes_numpy_helpers() {
        let interner = ax_ir::Interner::new();
        let dagger = interner.get_or_intern("dagger");
        let a = interner.get_or_intern("A");
        let body = ax_ir::Expr::Call(dagger, vec![ax_ir::Expr::Sym(a)]);
        let code = emit_python_qm_function("f", &[a], &body, &interner);
        assert!(code.contains("import numpy as np"));
        assert!(code.contains("def dagger(x):"));
        assert!(code.contains("def tensor_product(a, b):"));
        assert!(code.contains("dagger(A)"));
    }
}
