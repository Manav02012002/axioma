use crate::{
    expr_3d_to_list, expr_4d_to_list, expr_to_3d, expr_to_4d, expr_to_kraus_list,
    expr_to_null_tetrad, expr_to_weyl_scalars, kraus_list_to_expr, matrix_to_symbolic,
    null_tetrad_to_expr, simplify_symbolic_matrix, spin_coefficients_to_expr, weyl_scalars_to_expr,
};
use ax_ir::Expr;
use num_traits::{One, ToPrimitive};

pub struct BuiltinEntry {
    pub name: &'static str,
    pub category: &'static str,
    pub signature: &'static str,
    pub description: &'static str,
    pub example: &'static str,
}

pub struct PropertyEntry {
    pub name: &'static str,
    pub syntax: &'static str,
    pub description: &'static str,
    pub enables: &'static str,
    pub example: &'static str,
}

pub struct AlgorithmEntry {
    pub name: &'static str,
    pub category: &'static str,
    pub signature: &'static str,
    pub description: &'static str,
    pub preconditions: &'static str,
    pub example: &'static str,
}

pub struct SyntaxRule {
    pub pattern: &'static str,
    pub meaning: &'static str,
    pub example: &'static str,
}

pub struct StdModule {
    pub path: &'static str,
    pub description: &'static str,
    pub provides: &'static str,
}

pub struct ConventionEntry {
    pub field: &'static str,
    pub options: &'static str,
    pub default: &'static str,
    pub description: &'static str,
}

pub struct AssumptionEntry {
    pub name: &'static str,
    pub description: &'static str,
}

pub struct CallableEntry {
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub parameters: &'static [ParamDef],
    pub handler: fn(
        args: &[serde_json::Value],
        state: &mut dyn EvalState,
    ) -> Result<serde_json::Value, String>,
}

pub struct ParamDef {
    pub name: &'static str,
    pub param_type: ParamType,
    pub required: bool,
    pub description: &'static str,
}

pub enum ParamType {
    ExprId,
    Code,
    Symbol,
    SymbolList,
    Bool,
    Integer,
    Float,
    StringEnum(&'static [&'static str]),
    Matrix,
    Optional(Box<ParamType>),
}

pub trait EvalState {
    fn interner(&self) -> &ax_ir::Interner;
    fn interner_mut(&mut self) -> &mut ax_ir::Interner;
    fn env(&self) -> &crate::Env;
    fn env_mut(&mut self) -> &mut crate::Env;
    fn store_expr(&mut self, expr: ax_ir::Expr) -> String;
    fn get_expr(&self, id: &str) -> Option<&ax_ir::Expr>;
    fn parse_code(&mut self, code: &str) -> Result<ax_ir::Expr, String>;
    fn render_latex(&self, expr: &ax_ir::Expr) -> String;
    fn render_unicode(&self, expr: &ax_ir::Expr) -> String;
    fn get_metric(&self, id: &str) -> Option<&(ax_tensor::SymbolicMatrix, Vec<lasso::Spur>)>;
    fn store_metric(
        &mut self,
        id: String,
        metric: ax_tensor::SymbolicMatrix,
        coords: Vec<lasso::Spur>,
    );
    fn get_christoffel(&self, id: &str) -> Option<&Vec<Vec<Vec<ax_ir::Expr>>>>;
    fn store_christoffel(&mut self, id: String, chris: Vec<Vec<Vec<ax_ir::Expr>>>);
    fn get_riemann(&self, id: &str) -> Option<&Vec<Vec<Vec<Vec<ax_ir::Expr>>>>>;
    fn store_riemann(&mut self, id: String, riem: Vec<Vec<Vec<Vec<ax_ir::Expr>>>>);
    fn get_ricci(&self, id: &str) -> Option<&Vec<Vec<ax_ir::Expr>>>;
    fn store_ricci(&mut self, id: String, ric: Vec<Vec<ax_ir::Expr>>);
    fn get_matrix_data(&self, id: &str) -> Option<Vec<Vec<ax_ir::Expr>>>;
    fn list_expression_ids(&self) -> Vec<String> {
        vec![]
    }
    fn list_metric_ids(&self) -> Vec<String> {
        vec![]
    }
    fn list_christoffel_ids(&self) -> Vec<String> {
        vec![]
    }
    fn list_riemann_ids(&self) -> Vec<String> {
        vec![]
    }
    fn list_ricci_ids(&self) -> Vec<String> {
        vec![]
    }
    fn list_properties(&self) -> Vec<(String, Vec<String>)> {
        vec![]
    }
    fn list_index_families(&self) -> Vec<(String, Vec<String>, Option<usize>)> {
        vec![]
    }
    fn deadline(&self) -> Option<std::time::Instant> {
        None
    }
    fn check_deadline(&self) -> Result<(), String> {
        if let Some(deadline) = self.deadline() {
            if std::time::Instant::now() > deadline {
                return Err("computation timed out".to_string());
            }
        }
        Ok(())
    }
}

fn b(
    name: &'static str,
    category: &'static str,
    signature: &'static str,
    description: &'static str,
    example: &'static str,
) -> BuiltinEntry {
    BuiltinEntry {
        name,
        category,
        signature,
        description,
        example,
    }
}

fn p(
    name: &'static str,
    syntax: &'static str,
    description: &'static str,
    enables: &'static str,
    example: &'static str,
) -> PropertyEntry {
    PropertyEntry {
        name,
        syntax,
        description,
        enables,
        example,
    }
}

fn a(
    name: &'static str,
    category: &'static str,
    signature: &'static str,
    description: &'static str,
    preconditions: &'static str,
    example: &'static str,
) -> AlgorithmEntry {
    AlgorithmEntry {
        name,
        category,
        signature,
        description,
        preconditions,
        example,
    }
}

fn s(pattern: &'static str, meaning: &'static str, example: &'static str) -> SyntaxRule {
    SyntaxRule {
        pattern,
        meaning,
        example,
    }
}

fn m(path: &'static str, description: &'static str, provides: &'static str) -> StdModule {
    StdModule {
        path,
        description,
        provides,
    }
}

fn c(
    field: &'static str,
    options: &'static str,
    default: &'static str,
    description: &'static str,
) -> ConventionEntry {
    ConventionEntry {
        field,
        options,
        default,
        description,
    }
}

fn asm(name: &'static str, description: &'static str) -> AssumptionEntry {
    AssumptionEntry { name, description }
}

pub fn builtin_entries() -> Vec<BuiltinEntry> {
    vec![
        b(
            "import",
            "syntax",
            "import(path)",
            "Import a standard-library module into the current environment.",
            "import std.conventions.mtw",
        ),
        b(
            "assume",
            "syntax",
            "assume(sym, property)",
            "Attach assumptions such as real, positive, integer, even, or odd to a symbol.",
            "assume x positive",
        ),
        b(
            "Re",
            "complex",
            "Re(z)",
            "Return the real part of a complex expression.",
            "Re(3 + 4*i)",
        ),
        b(
            "Im",
            "complex",
            "Im(z)",
            "Return the imaginary part of a complex expression.",
            "Im(3 + 4*i)",
        ),
        b(
            "conj",
            "complex",
            "conj(z)",
            "Return the complex conjugate.",
            "conj(3 + 4*i)",
        ),
        b(
            "arg",
            "complex",
            "arg(z)",
            "Return the complex argument or phase.",
            "arg(1 + i)",
        ),
        b(
            "N",
            "numeric",
            "N(expr)",
            "Evaluate an expression numerically when possible.",
            "N(sin(pi/4))",
        ),
        b(
            "sin",
            "elementary",
            "sin(x)",
            "Sine with symbolic and numeric evaluation.",
            "sin(x)",
        ),
        b(
            "cos",
            "elementary",
            "cos(x)",
            "Cosine with symbolic and numeric evaluation.",
            "cos(x)",
        ),
        b(
            "tan",
            "elementary",
            "tan(x)",
            "Tangent with symbolic and numeric evaluation.",
            "tan(x)",
        ),
        b(
            "sec",
            "elementary",
            "sec(x)",
            "Secant with symbolic and numeric evaluation.",
            "sec(x)",
        ),
        b(
            "csc",
            "elementary",
            "csc(x)",
            "Cosecant with symbolic and numeric evaluation.",
            "csc(x)",
        ),
        b(
            "cot",
            "elementary",
            "cot(x)",
            "Cotangent with symbolic and numeric evaluation.",
            "cot(x)",
        ),
        b("asin", "elementary", "asin(x)", "Inverse sine.", "asin(x)"),
        b(
            "arcsin",
            "elementary",
            "arcsin(x)",
            "Inverse sine alias.",
            "arcsin(x)",
        ),
        b(
            "acos",
            "elementary",
            "acos(x)",
            "Inverse cosine.",
            "acos(x)",
        ),
        b(
            "arccos",
            "elementary",
            "arccos(x)",
            "Inverse cosine alias.",
            "arccos(x)",
        ),
        b(
            "atan",
            "elementary",
            "atan(x)",
            "Inverse tangent.",
            "atan(x)",
        ),
        b(
            "arctan",
            "elementary",
            "arctan(x)",
            "Inverse tangent alias.",
            "arctan(x)",
        ),
        b(
            "atan2",
            "elementary",
            "atan2(y, x)",
            "Two-argument arctangent.",
            "atan2(y, x)",
        ),
        b(
            "sinh",
            "elementary",
            "sinh(x)",
            "Hyperbolic sine.",
            "sinh(x)",
        ),
        b(
            "cosh",
            "elementary",
            "cosh(x)",
            "Hyperbolic cosine.",
            "cosh(x)",
        ),
        b(
            "tanh",
            "elementary",
            "tanh(x)",
            "Hyperbolic tangent.",
            "tanh(x)",
        ),
        b(
            "asinh",
            "elementary",
            "asinh(x)",
            "Inverse hyperbolic sine.",
            "asinh(x)",
        ),
        b(
            "arcsinh",
            "elementary",
            "arcsinh(x)",
            "Inverse hyperbolic sine alias.",
            "arcsinh(x)",
        ),
        b(
            "acosh",
            "elementary",
            "acosh(x)",
            "Inverse hyperbolic cosine.",
            "acosh(x)",
        ),
        b(
            "arccosh",
            "elementary",
            "arccosh(x)",
            "Inverse hyperbolic cosine alias.",
            "arccosh(x)",
        ),
        b(
            "atanh",
            "elementary",
            "atanh(x)",
            "Inverse hyperbolic tangent.",
            "atanh(x)",
        ),
        b(
            "arctanh",
            "elementary",
            "arctanh(x)",
            "Inverse hyperbolic tangent alias.",
            "arctanh(x)",
        ),
        b(
            "exp",
            "elementary",
            "exp(x)",
            "Exponential function.",
            "exp(x)",
        ),
        b(
            "log",
            "elementary",
            "log(x)",
            "Natural logarithm.",
            "log(x)",
        ),
        b(
            "sqrt",
            "elementary",
            "sqrt(x)",
            "Square root with exact perfect-square simplification.",
            "sqrt(9)",
        ),
        b(
            "abs",
            "elementary",
            "abs(x)",
            "Absolute value or complex modulus.",
            "abs(x)",
        ),
        b("sign", "elementary", "sign(x)", "Sign function.", "sign(x)"),
        b(
            "sgn",
            "elementary",
            "sgn(x)",
            "Sign function alias.",
            "sgn(x)",
        ),
        b(
            "diff",
            "calculus",
            "diff(expr, var)",
            "Differentiate an expression symbolically.",
            "diff(sin(x^2), x)",
        ),
        b(
            "integrate",
            "calculus",
            "integrate(expr, var) or integrate(expr, var, a, b)",
            "Indefinite or definite symbolic integration.",
            "integrate(x^2, x)",
        ),
        b(
            "double_integral",
            "calculus",
            "double_integral(expr, x, y)",
            "Perform iterated double integration.",
            "double_integral(x*y, x, y)",
        ),
        b(
            "dblint",
            "calculus",
            "dblint(expr, x, y)",
            "Alias for double_integral.",
            "dblint(x*y, x, y)",
        ),
        b(
            "triple_integral",
            "calculus",
            "triple_integral(expr, x, y, z)",
            "Perform iterated triple integration.",
            "triple_integral(x*y*z, x, y, z)",
        ),
        b(
            "tplint",
            "calculus",
            "tplint(expr, x, y, z)",
            "Alias for triple_integral.",
            "tplint(x*y*z, x, y, z)",
        ),
        b(
            "definite_integral",
            "calculus",
            "definite_integral(expr, var, a, b)",
            "Compute a definite integral from an antiderivative.",
            "definite_integral(x^2, x, 0, 1)",
        ),
        b(
            "defint",
            "calculus",
            "defint(expr, var, a, b)",
            "Alias for definite_integral.",
            "defint(x^2, x, 0, 1)",
        ),
        b(
            "integrate_by_parts",
            "calculus",
            "integrate_by_parts(expr, u, v, var)",
            "Perform one integration-by-parts step using explicit u and v'.",
            "integrate_by_parts(x*exp(x), x, exp(x), x)",
        ),
        b(
            "ibp",
            "calculus",
            "ibp(expr, u, v, var)",
            "Alias for integrate_by_parts.",
            "ibp(x*exp(x), x, exp(x), x)",
        ),
        b(
            "limit",
            "calculus",
            "limit(expr, var, point)",
            "Evaluate a symbolic limit.",
            "limit(sin(x)/x, x, 0)",
        ),
        b(
            "series",
            "calculus",
            "series(expr, var, point, order)",
            "Compute a Taylor series.",
            "series(exp(x), x, 0, 4)",
        ),
        b(
            "angle",
            "spinor",
            "angle(i, j)",
            "Construct the spinor-helicity angle bracket <ij>.",
            "angle(1, 2)",
        ),
        b(
            "square",
            "spinor",
            "square(i, j)",
            "Construct the spinor-helicity square bracket [ij].",
            "square(1, 2)",
        ),
        b(
            "mandelstam",
            "spinor",
            "mandelstam(i, j)",
            "Construct the two-particle Mandelstam invariant s_ij.",
            "mandelstam(1, 2)",
        ),
        b(
            "parke_taylor",
            "spinor",
            "parke_taylor(n, i, j)",
            "Construct the n-gluon MHV Parke-Taylor amplitude.",
            "parke_taylor(4, 0, 2)",
        ),
        b(
            "three_point_mhv",
            "spinor",
            "three_point_mhv(i, j, k)",
            "Construct the three-point MHV amplitude.",
            "three_point_mhv(0, 1, 2)",
        ),
        b(
            "three_point_anti_mhv",
            "spinor",
            "three_point_anti_mhv(i, j, k)",
            "Construct the three-point anti-MHV amplitude.",
            "three_point_anti_mhv(0, 1, 2)",
        ),
        b(
            "expand_chain",
            "spinor",
            "expand_chain(expr)",
            "Expand spinor chains into angle and square bracket products.",
            "expand_chain(angle_square_chain)",
        ),
        b(
            "contract_adjacent",
            "spinor",
            "contract_adjacent(expr)",
            "Contract adjacent angle-square bracket pairs into one-momentum chains.",
            "contract_adjacent(expr)",
        ),
        b(
            "expand_mandelstam",
            "spinor",
            "expand_mandelstam(expr)",
            "Expand Mandelstam invariants into spinor brackets.",
            "expand_mandelstam(mandelstam(1,2))",
        ),
        b(
            "collect_mandelstam",
            "spinor",
            "collect_mandelstam(expr)",
            "Collect spinor-bracket products back into Mandelstam invariants.",
            "collect_mandelstam(expr)",
        ),
        b(
            "schouten",
            "spinor",
            "schouten(expr, a, b, c, d)",
            "Apply the spinor Schouten identity.",
            "schouten(expr, 1, 2, 3, 4)",
        ),
        b(
            "momentum_conservation",
            "spinor",
            "momentum_conservation(expr, n, eliminate)",
            "Apply spinor-helicity momentum conservation eliminating one particle momentum.",
            "momentum_conservation(expr, 4, 3)",
        ),
        b(
            "spinor_simplify",
            "spinor",
            "spinor_simplify(expr, n)",
            "Run the spinor-helicity simplification pipeline.",
            "spinor_simplify(expr, 4)",
        ),
        b(
            "bcfw_shift",
            "spinor",
            "bcfw_shift(expr, i, j, z)",
            "Apply a BCFW shift to a spinor expression.",
            "bcfw_shift(expr, 0, 1, z)",
        ),
        b(
            "bcfw_decomposition",
            "spinor",
            "bcfw_decomposition(n, i, j, helicities)",
            "Enumerate BCFW factorization channels.",
            "bcfw_decomposition(4, 0, 1, [-1,1,1,-1])",
        ),
        b(
            "four_bracket",
            "twistor",
            "four_bracket(i, j, k, l)",
            "Construct a momentum-twistor four-bracket.",
            "four_bracket(1, 2, 3, 4)",
        ),
        b(
            "plucker",
            "twistor",
            "plucker(expr, a, b, c, d, e, f)",
            "Apply the momentum-twistor Plucker identity.",
            "plucker(expr, 1, 2, 3, 4, 5, 6)",
        ),
        b(
            "eq",
            "equation",
            "eq(lhs, rhs)",
            "Create an equation object.",
            "eq(x + y, 3)",
        ),
        b(
            "get_lhs",
            "equation",
            "get_lhs(eq)",
            "Get the left-hand side of an equation.",
            "get_lhs(eq1)",
        ),
        b(
            "get_rhs",
            "equation",
            "get_rhs(eq)",
            "Get the right-hand side of an equation.",
            "get_rhs(eq1)",
        ),
        b(
            "swap_sides",
            "equation",
            "swap_sides(eq)",
            "Swap the left- and right-hand sides of an equation.",
            "swap_sides(eq1)",
        ),
        b(
            "multiply_through",
            "equation",
            "multiply_through(eq, factor)",
            "Multiply both sides of an equation by a factor.",
            "multiply_through(eq1, 2)",
        ),
        b(
            "add_through",
            "equation",
            "add_through(eq, term)",
            "Add a term to both sides of an equation.",
            "add_through(eq1, y)",
        ),
        b(
            "to_rhs",
            "equation",
            "to_rhs(eq, target)",
            "Move terms containing target from the LHS to the RHS.",
            "to_rhs(eq1, y)",
        ),
        b(
            "to_lhs",
            "equation",
            "to_lhs(eq, target)",
            "Move terms containing target from the RHS to the LHS.",
            "to_lhs(eq1, x)",
        ),
        b(
            "isolate",
            "equation",
            "isolate(eq, target)",
            "Solve simple equation patterns for target.",
            "isolate(eq1, x)",
        ),
        b(
            "eq_to_rule",
            "equation",
            "eq_to_rule(eq)",
            "Convert an equation to an exact rewrite rule.",
            "eq_to_rule(eq1)",
        ),
        b(
            "eq_to_subrule",
            "equation",
            "eq_to_subrule(eq)",
            "Alias for eq_to_rule.",
            "eq_to_subrule(eq1)",
        ),
        b(
            "differentiate_eq",
            "equation",
            "differentiate_eq(eq, var)",
            "Differentiate both sides of an equation.",
            "differentiate_eq(eq1, x)",
        ),
        b(
            "integrate_eq",
            "equation",
            "integrate_eq(eq, var)",
            "Integrate both sides of an equation.",
            "integrate_eq(eq1, x)",
        ),
        b(
            "substitute_eq",
            "equation",
            "substitute_eq(eq, target, replacement)",
            "Substitute in both sides of an equation.",
            "substitute_eq(eq1, x, y + 1)",
        ),
        b(
            "raise_eq",
            "equation",
            "raise_eq(eq, index)",
            "Raise an index on both sides using the active metric.",
            "raise_eq(eq1, a)",
        ),
        b(
            "lower_eq",
            "equation",
            "lower_eq(eq, index)",
            "Lower an index on both sides using the active metric.",
            "lower_eq(eq1, a)",
        ),
        b("perturb", "perturbation", "perturb(expr, field, background, perturbation, epsilon, order)", "Expand an expression in a metric perturbation series through the requested order.", "perturb(R_ab, g, g0, h, eps, 2)"),
        b("perturb_inverse", "perturbation", "perturb_inverse(field, background, background_inv, perturbation, epsilon, order)", "Expand the inverse metric perturbatively as a geometric series.", "perturb_inverse(g, g0, g0inv, h, eps, 2)"),
        b("perturb_christoffel", "perturbation", "perturb_christoffel(field, background, background_inv, perturbation, epsilon, coords, order)", "Expand the Christoffel symbol order by order in a metric perturbation.", "perturb_christoffel(g, g0, g0inv, h, eps, [t,r,theta,phi], 1)"),
        b("perturb_riemann", "perturbation", "perturb_riemann(field, background, background_inv, perturbation, epsilon, coords, order)", "Expand the Riemann tensor order by order in a metric perturbation.", "perturb_riemann(g, g0, g0inv, h, eps, [t,r,theta,phi], 1)"),
        b("perturb_ricci", "perturbation", "perturb_ricci(field, background, background_inv, perturbation, epsilon, coords, order)", "Expand the Ricci tensor order by order in a metric perturbation.", "perturb_ricci(g, g0, g0inv, h, eps, [t,r,theta,phi], 1)"),
        b("perturb_einstein", "perturbation", "perturb_einstein(field, background, background_inv, perturbation, epsilon, coords, order)", "Expand the Einstein tensor order by order in a metric perturbation.", "perturb_einstein(g, g0, g0inv, h, eps, [t,r,theta,phi], 1)"),
        b("linearized_einstein", "cosmology", "linearized_einstein(order)", "Return first- or second-order scalar perturbation Einstein equations on an FRW background.", "linearized_einstein(1)"),
        b("linearized_einstein_vector", "cosmology", "linearized_einstein_vector()", "Derived linear vector Einstein equations in FRW Poisson gauge.", "linearized_einstein_vector()"),
        b("linearized_einstein_tensor", "cosmology", "linearized_einstein_tensor()", "Derived linear tensor Einstein equations in FRW.", "linearized_einstein_tensor()"),
        b("second_order_einstein_vector", "cosmology", "second_order_einstein_vector()", "Derived second-order vector Einstein equations with quadratic source splitting.", "second_order_einstein_vector()"),
        b("second_order_einstein_tensor", "cosmology", "second_order_einstein_tensor()", "Derived second-order tensor Einstein equations with quadratic source splitting.", "second_order_einstein_tensor()"),
        b("mukhanov_sasaki", "cosmology", "mukhanov_sasaki()", "Return the Mukhanov-Sasaki equation in conformal time.", "mukhanov_sasaki()"),
        b("tensor_mode_equation", "cosmology", "tensor_mode_equation()", "Tensor polarization mode equations derived from the quadratic action.", "tensor_mode_equation()"),
        b("tensor_mode_first_order", "cosmology", "tensor_mode_first_order(polarization)", "First-order ODE system for a tensor polarization mode.", "tensor_mode_first_order(plus)"),
        b("multifield_equations", "cosmology", "multifield_equations(nfields)", "Derived multifield curvature and entropy mode equations.", "multifield_equations(2)"),
        b("boltzmann_bridge", "cosmology", "boltzmann_bridge()", "Symbolic first-order Einstein–Boltzmann bridge system in Newtonian gauge.", "boltzmann_bridge()"),
        b("boltzmann_bridge_export", "cosmology", "boltzmann_bridge_export(target)", "Export the symbolic Einstein–Boltzmann bridge system.", "boltzmann_bridge_export(python)"),
        b("cubic_action", "cosmology", "cubic_action(channel)", "Reduced cubic CPT action density for a given interaction channel.", "cubic_action(scalar_scalar_scalar)"),
        b("cubic_kernel", "cosmology", "cubic_kernel(channel)", "Fourier-space cubic interaction kernel for a given CPT channel.", "cubic_kernel(scalar_scalar_scalar)"),
        b("bispectrum_shape", "cosmology", "bispectrum_shape(channel, shape)", "Evaluate a cubic kernel on a named bispectrum shape.", "bispectrum_shape(scalar_scalar_scalar, local)"),
        b("export_cubic_vertex", "cosmology", "export_cubic_vertex(channel, target)", "Export a cubic interaction vertex as code.", "export_cubic_vertex(scalar_scalar_scalar, python)"),
        b("eft_model", "cosmology", "eft_model(kind)", "Construct a typed reduced EFT-of-inflation model.", "eft_model(canonical)"),
        b("eft_quadratic_sector", "cosmology", "eft_quadratic_sector(kind)", "Reduced scalar/tensor quadratic sector for an EFT model.", "eft_quadratic_sector(canonical)"),
        b("eft_stability", "cosmology", "eft_stability(kind)", "Ghost and gradient stability conditions for a reduced EFT model.", "eft_stability(canonical)"),
        b("eft_mode_equations", "cosmology", "eft_mode_equations(kind)", "Reduced scalar and tensor mode equations for a reduced EFT model.", "eft_mode_equations(canonical)"),
        b("eft_export_rhs", "cosmology", "eft_export_rhs(kind, target)", "Export reduced EFT mode RHS functions.", "eft_export_rhs(canonical, python)"),
        b("project_scalar_harmonics", "cosmology", "project_scalar_harmonics()", "Project derived scalar CPT equations to FRW harmonic space.", "project_scalar_harmonics()"),
        b("project_vector_harmonics", "cosmology", "project_vector_harmonics()", "Project derived vector CPT equations to FRW harmonic space.", "project_vector_harmonics()"),
        b("project_tensor_harmonics", "cosmology", "project_tensor_harmonics()", "Project derived tensor CPT equations to FRW harmonic space.", "project_tensor_harmonics()"),
        b("project_second_order_vector_harmonics", "cosmology", "project_second_order_vector_harmonics()", "Project derived second-order vector equations to harmonic space.", "project_second_order_vector_harmonics()"),
        b("project_second_order_tensor_harmonics", "cosmology", "project_second_order_tensor_harmonics()", "Project derived second-order tensor equations to harmonic space.", "project_second_order_tensor_harmonics()"),
        b("neutrino_hierarchy", "cosmology", "neutrino_hierarchy(lmax, gauge, closure)", "Construct a symbolic neutrino multipole hierarchy with explicit truncation.", "neutrino_hierarchy(3, newtonian, power_law)"),
        b("photon_hierarchy", "cosmology", "photon_hierarchy(lmax, gauge, closure)", "Construct a symbolic photon multipole hierarchy with explicit truncation.", "photon_hierarchy(3, newtonian, power_law)"),
        b("export_hierarchy", "cosmology", "export_hierarchy(target, species, lmax, gauge, closure)", "Export a symbolic hierarchy system or external-solver hook payload.", "export_hierarchy(class_hook, neutrino, 3, newtonian, power_law)"),
        b("cpt_parity_report", "cosmology", "cpt_parity_report()", "Run built-in CPT parity suites against embedded benchmark fixtures.", "cpt_parity_report()"),
        b("scalar_harmonic_spec", "cosmology", "scalar_harmonic_spec(curvature)", "Describe the scalar harmonic basis for a given FRW spatial curvature.", "scalar_harmonic_spec(flat)"),
        b("vector_harmonic_spec", "cosmology", "vector_harmonic_spec(curvature)", "Describe the vector harmonic basis for a given FRW spatial curvature.", "vector_harmonic_spec(closed)"),
        b("tensor_harmonic_spec", "cosmology", "tensor_harmonic_spec(curvature)", "Describe the tensor harmonic basis for a given FRW spatial curvature.", "tensor_harmonic_spec(open)"),
        b("svt_decompose", "cosmology", "svt_decompose()", "Return the standard scalar-vector-tensor decomposition modes.", "svt_decompose()"),
        b("bardeen", "cosmology", "bardeen()", "Return the two Bardeen gauge-invariant scalar potentials.", "bardeen()"),
        b("regge_wheeler_decompose", "cosmology", "regge_wheeler_decompose(l)", "Return symbolic even- and odd-parity Schwarzschild perturbation sectors.", "regge_wheeler_decompose(2)"),
        b("zerilli", "cosmology", "zerilli(l)", "Return the Zerilli master equation for even-parity Schwarzschild perturbations.", "zerilli(2)"),
        b("regge_wheeler", "cosmology", "regge_wheeler(l)", "Return the Regge-Wheeler master equation for odd-parity Schwarzschild perturbations.", "regge_wheeler(2)"),
        b("power_spectrum", "cosmology", "power_spectrum()", "Return the leading-order scalar curvature power spectrum.", "power_spectrum()"),
        b("spectral_index", "cosmology", "spectral_index()", "Return the leading slow-roll scalar spectral index.", "spectral_index()"),
        b("tensor_scalar_ratio", "cosmology", "tensor_scalar_ratio()", "Return the leading single-field tensor-to-scalar ratio.", "tensor_scalar_ratio()"),
        b("graded", "graded-algebra", "graded(sym, bosonic|fermionic|n)", "Declare a Z2 or integer grading on a symbol.", "graded(theta, fermionic)"),
        b("graded_commutator", "graded-algebra", "graded_commutator(a, b)", "Compute the graded commutator using the active graded symbol table.", "graded_commutator(Q, theta)"),
        b("graded_simplify", "graded-algebra", "graded_simplify(expr)", "Simplify expressions with graded-commutation and nilpotency rules.", "graded_simplify(theta*theta)"),
        b("setup_superspace", "superspace", "setup_superspace(N)", "Initialize N=1 superspace coordinates and Grassmann gradings.", "setup_superspace(1)"),
        b("expand_superfield", "superspace", "expand_superfield(name)", "Expand a generic N=1 superfield into theta components.", "expand_superfield(Phi)"),
        b("chiral_superfield", "superspace", "chiral_superfield(name)", "Construct a chiral N=1 superfield expansion.", "chiral_superfield(Phi)"),
        b("antichiral_superfield", "superspace", "antichiral_superfield(name)", "Construct an antichiral N=1 superfield expansion.", "antichiral_superfield(Phi_bar)"),
        b("vector_superfield_wz", "superspace", "vector_superfield_wz(name)", "Construct a Wess-Zumino gauge vector superfield.", "vector_superfield_wz(V)"),
        b("extract_component", "superspace", "extract_component(expr, [theta, theta_bar])", "Extract a theta-sector component from a superspace expression.", "extract_component(Phi, [2,0])"),
        b("d_alpha", "superspace", "d_alpha(expr, alpha)", "Apply the N=1 supercovariant derivative D_alpha.", "d_alpha(Phi, 0)"),
        b("d_bar", "superspace", "d_bar(expr, alpha_dot)", "Apply the conjugate supercovariant derivative D_bar.", "d_bar(Phi, 1)"),
        b("d_squared", "superspace", "d_squared(expr)", "Apply D^2 to a superspace expression.", "d_squared(Phi)"),
        b("d_bar_squared", "superspace", "d_bar_squared(expr)", "Apply D_bar^2 to a superspace expression.", "d_bar_squared(Phi)"),
        b("superspace_integrate", "superspace", "superspace_integrate(expr, full|chiral|antichiral)", "Extract the component corresponding to a superspace integration measure.", "superspace_integrate(Phi_bar*Phi, full)"),
        b("setup_brst_ym", "brst", "setup_brst_ym(A, c, cbar, B, g)", "Initialize Yang-Mills BRST fields, ghost numbers, and transformation rules.", "setup_brst_ym(A, c, cbar, B, g)"),
        b("brst", "brst", "brst(expr)", "Apply the active BRST operator as a graded derivation.", "brst(A)"),
        b("brst_check", "brst", "brst_check(expr)", "Check whether an expression is BRST-closed.", "brst_check(F)"),
        b("ghost_number", "brst", "ghost_number(expr)", "Compute the ghost number of an expression.", "ghost_number(cbar*B)"),
        b("filter_ghost", "brst", "filter_ghost(expr, n)", "Keep only terms with the requested ghost number.", "filter_ghost(expr, 0)"),
        b(
            "gradient",
            "vector-calculus",
            "gradient(f, [x, y, z])",
            "Return the gradient vector.",
            "gradient(x^2 + y^2 + z^2, [x, y, z])",
        ),
        b(
            "grad",
            "vector-calculus",
            "grad(f, [x, y, z])",
            "Alias for gradient.",
            "grad(x^2 + y^2, [x, y])",
        ),
        b(
            "divergence",
            "vector-calculus",
            "divergence([Fx, Fy, Fz], [x, y, z])",
            "Return the divergence of a vector field.",
            "divergence([x, y, z], [x, y, z])",
        ),
        b(
            "div",
            "vector-calculus",
            "div([Fx, Fy, Fz], [x, y, z])",
            "Alias for divergence.",
            "div([x, y], [x, y])",
        ),
        b(
            "curl",
            "vector-calculus",
            "curl([Fx, Fy, Fz], [x, y, z])",
            "Return the three-dimensional curl.",
            "curl([x, y, z], [x, y, z])",
        ),
        b(
            "laplacian",
            "vector-calculus",
            "laplacian(f, [x, y, z])",
            "Return the Laplacian.",
            "laplacian(x^2 - y^2, [x, y])",
        ),
        b(
            "jacobian",
            "vector-calculus",
            "jacobian([f1, ...], [x1, ...])",
            "Return the Jacobian matrix.",
            "jacobian([x^2, x*y], [x, y])",
        ),
        b(
            "hessian",
            "vector-calculus",
            "hessian(f, [x1, ...])",
            "Return the Hessian matrix.",
            "hessian(x^2 + 3*x*y + y^2, [x, y])",
        ),
        b(
            "expand",
            "simplify",
            "expand(expr)",
            "Distribute products and expand small powers.",
            "expand((x + 1)^2)",
        ),
        b(
            "simplify",
            "simplify",
            "simplify(expr)",
            "Run the full simplification pipeline.",
            "simplify(sin(x)^2 + cos(x)^2)",
        ),
        b(
            "rationalize",
            "simplify",
            "rationalize(expr)",
            "Put sums over a common denominator and cancel common factors.",
            "rationalize(1/x + 1/x^2)",
        ),
        b(
            "partial_fractions",
            "simplify",
            "partial_fractions(expr, var)",
            "Decompose a rational function into partial fractions when supported.",
            "partial_fractions(1/(x*(x+1)), x)",
        ),
        b(
            "apart",
            "simplify",
            "apart(expr, var)",
            "Alias for partial_fractions.",
            "apart(1/(x*(x+1)), x)",
        ),
        b(
            "trig_simplify",
            "simplify",
            "trig_simplify(expr)",
            "Apply exact trigonometric rewrite rules.",
            "trig_simplify(sin(x)^2 + cos(x)^2)",
        ),
        b(
            "factor_out",
            "simplify",
            "factor_out(expr[, targets])",
            "Factor common factors from a sum.",
            "factor_out(a*x + a*y, [a])",
        ),
        b(
            "factor_in",
            "simplify",
            "factor_in(expr[, targets])",
            "Group terms that share common prefactors.",
            "factor_in(a*x + a*y, [a])",
        ),
        b(
            "subs",
            "rewrite",
            "subs(expr, target, replacement)",
            "Perform symbolic substitution with index-aware matching when needed.",
            "subs(f(x), x, y)",
        ),
        b(
            "rewrite",
            "rewrite",
            "rewrite(expr)",
            "Apply user-defined rewrite rules to an expression.",
            "rewrite(expr)",
        ),
        b(
            "zoom",
            "rewrite",
            "zoom(expr, pattern)",
            "Split an expression into matching and non-matching parts.",
            "zoom(a + b + c, a + b)",
        ),
        b(
            "unzoom",
            "rewrite",
            "unzoom(focus, remainder)",
            "Recombine a focused expression with its remainder.",
            "unzoom(a + b, c)",
        ),
        b(
            "take_match",
            "rewrite",
            "take_match(expr, pattern)",
            "Keep only the parts of a sum that match a pattern.",
            "take_match(a + b + c, a_)",
        ),
        b(
            "equiv",
            "analysis",
            "equiv(lhs, rhs)",
            "Describe whether two expressions are semantically equivalent.",
            "equiv(x + x, 2*x)",
        ),
        b(
            "semantic_diff",
            "analysis",
            "semantic_diff(lhs, rhs)",
            "Return a semantic-difference descriptor.",
            "semantic_diff(x + x, 2*x)",
        ),
        b(
            "canonicalise",
            "tensor",
            "canonicalise(expr)",
            "Canonicalize tensor indices using declared tensor properties.",
            "canonicalise(R[a-,b-,c-,d-] + R[a-,c-,d-,b-])",
        ),
        b(
            "canonicalize",
            "tensor",
            "canonicalize(expr)",
            "Alias for canonicalise.",
            "canonicalize(T[a-, b-])",
        ),
        b(
            "lower_free_indices",
            "tensor",
            "lower_free_indices(expr)",
            "Lower free upper indices using the active metric family.",
            "lower_free_indices(V[mu+])",
        ),
        b(
            "lower_indices",
            "tensor",
            "lower_indices(expr)",
            "Alias for lower_free_indices.",
            "lower_indices(V[mu+])",
        ),
        b(
            "raise_free_indices",
            "tensor",
            "raise_free_indices(expr)",
            "Raise free lower indices using the active inverse metric family.",
            "raise_free_indices(V[mu-])",
        ),
        b(
            "raise_indices",
            "tensor",
            "raise_indices(expr)",
            "Alias for raise_free_indices.",
            "raise_indices(V[mu-])",
        ),
        b(
            "meld",
            "tensor",
            "meld(expr)",
            "Detect multi-term tensor identities using Young projection and linear dependence.",
            "meld(R[a-,b-,c-,d-] + R[a-,c-,d-,b-] + R[a-,d-,b-,c-])",
        ),
        b(
            "rename_dummies",
            "tensor",
            "rename_dummies(expr)",
            "Rename dummy indices to a canonical fresh naming scheme.",
            "rename_dummies(T[a-, a+])",
        ),
        b(
            "sort_product",
            "tensor",
            "sort_product(expr)",
            "Sort tensor products using symmetry-aware canonicalization.",
            "sort_product(B[a-] * A[a-])",
        ),
        b(
            "product_rule",
            "tensor",
            "product_rule(expr)",
            "Apply the Leibniz rule to an indexed product.",
            "product_rule(partial(A*B))",
        ),
        b(
            "leibniz",
            "tensor",
            "leibniz(expr)",
            "Alias for product_rule.",
            "leibniz(partial(A*B))",
        ),
        b(
            "unwrap",
            "tensor",
            "unwrap(expr)",
            "Flatten nested additive and multiplicative structure.",
            "unwrap((a + b) + c)",
        ),
        b(
            "tensor_distribute",
            "tensor",
            "tensor_distribute(expr)",
            "Distribute products over sums in tensor expressions.",
            "tensor_distribute(A*(B + C))",
        ),
        b(
            "tdistribute",
            "tensor",
            "tdistribute(expr)",
            "Alias for tensor_distribute.",
            "tdistribute(A*(B + C))",
        ),
        b(
            "keep_weight",
            "tensor",
            "keep_weight(expr, label, value)",
            "Filter terms by a recorded symbolic weight.",
            "keep_weight(expr, field, 1)",
        ),
        b(
            "drop_weight",
            "tensor",
            "drop_weight(expr, label, value)",
            "Remove terms with a recorded symbolic weight.",
            "drop_weight(expr, field, 0)",
        ),
        b(
            "einsteinify",
            "tensor",
            "einsteinify(expr)",
            "Insert implicit Einstein summation contractions.",
            "einsteinify(A[mu-] * B[mu+])",
        ),
        b(
            "split_index",
            "tensor",
            "split_index(expr, old, [new...])",
            "Split one abstract index family into several fixed values.",
            "split_index(T[a-], a, [0,1,2])",
        ),
        b(
            "eliminate_kronecker",
            "tensor",
            "eliminate_kronecker(expr)",
            "Contract Kronecker deltas through an expression.",
            "eliminate_kronecker(delta[mu+, nu-] * T[nu+, rho-])",
        ),
        b(
            "expand_delta",
            "tensor",
            "expand_delta(expr)",
            "Expand delta contractions into explicit sums when possible.",
            "expand_delta(delta[mu+, nu-] * V[nu+])",
        ),
        b(
            "expand_dummies",
            "tensor",
            "expand_dummies(expr)",
            "Expand dummy sums over the declared coordinate set.",
            "expand_dummies(T[mu-, mu+])",
        ),
        b(
            "explicit_indices",
            "tensor",
            "explicit_indices(expr)",
            "Make implicit repeated indices explicit.",
            "explicit_indices(A * B)",
        ),
        b(
            "expand_implicit",
            "tensor",
            "expand_implicit(expr)",
            "Expand implicit tensor contractions and index conventions.",
            "expand_implicit(A[mu-] B[mu+])",
        ),
        b(
            "rewrite_indices",
            "tensor",
            "rewrite_indices(expr)",
            "Rewrite index names while preserving variance and families.",
            "rewrite_indices(T[a-, b+])",
        ),
        b(
            "rewrite_indices_vielbein",
            "tensor",
            "rewrite_indices_vielbein(expr, e, einv, from_family, to_family)",
            "Rewrite tensor indices between coordinate and frame families using explicit vielbein factors.",
            "rewrite_indices_vielbein(V[mu+], e, einv, spacetime, frame)",
        ),
        b(
            "reduce_delta",
            "tensor",
            "reduce_delta(expr)",
            "Simplify explicit delta-expanded expressions back to compact form.",
            "reduce_delta(expr)",
        ),
        b(
            "young_project",
            "tensor",
            "young_project(expr, tableau) or young_project(expr, modulo_monoterm=true, canonicalize_after=true, rename_dummies_after=true)",
            "Either project with an explicit tableau `[[...], ...]` or use the tensor's declared symmetry properties with optional post-projection monoterm canonicalisation and dummy renaming.",
            "young_project(T[a-,b-,c-])",
        ),
        b(
            "tensor_reduce",
            "tensor",
            "tensor_reduce(expr, monoterm=true, multiterm=true, dimension_dependent=true, meld=true, modulo_monoterm=true)",
            "Run the finished tensor reduction pipeline: monoterm canonicalisation, multi-term Young projection, dimension-dependent reduction, dummy renaming, and optional meld.",
            "tensor_reduce(R[a-,b-,c-,d-]*V[e-] + R[a-,c-,d-,b-]*V[e-] + R[a-,d-,b-,c-]*V[e-])",
        ),
        b(
            "abstract_tensor_reduce",
            "tensor",
            "abstract_tensor_reduce(expr, monoterm=true, multiterm=true, dimension_dependent=true, meld=true, modulo_monoterm=true)",
            "Run the user-facing abstract tensor reduction pipeline for declared tensor symmetries and inherited covariant-derivative identities.",
            "abstract_tensor_reduce(nabla[mu-]*R[nu-,rho-,sigma-,lambda-] + nabla[nu-]*R[rho-,mu-,sigma-,lambda-] + nabla[rho-]*R[mu-,nu-,sigma-,lambda-])",
        ),
        b(
            "abstract_gr_reduce",
            "tensor",
            "abstract_gr_reduce(expr, monoterm=true, multiterm=true, dimension_dependent=true, meld=true, modulo_monoterm=true)",
            "Alias for abstract_tensor_reduce aimed at abstract GR workflows.",
            "abstract_gr_reduce(nabla[mu-]*R[nu-,rho-,sigma-,lambda-] + nabla[nu-]*R[rho-,mu-,sigma-,lambda-] + nabla[rho-]*R[mu-,nu-,sigma-,lambda-])",
        ),
        b(
            "contracted_bianchi_reduce",
            "tensor",
            "contracted_bianchi_reduce(expr, nabla, Ric, R, G?)",
            "Abstract contracted-Bianchi reducer for Ricci/scalar and optional Einstein-tensor divergence identities.",
            "contracted_bianchi_reduce(nabla[a+]*Ric[a-,b-], nabla, Ric, R)",
        ),
        b(
            "schouten_reduce",
            "tensor",
            "schouten_reduce(expr)",
            "Apply dimension-dependent tensor reduction using inferred index-family dimension metadata.",
            "schouten_reduce(A[a-]*B[b-]*C[c-] - A[a-]*B[c-]*C[b-] + A[b-]*B[c-]*C[a-] - A[b-]*B[a-]*C[c-] + A[c-]*B[a-]*C[b-] - A[c-]*B[b-]*C[a-])",
        ),
        b(
            "symmetrise",
            "tensor",
            "symmetrise(expr, [positions])",
            "Symmetrise over listed slots.",
            "symmetrise(T[a-, b-], [0,1])",
        ),
        b(
            "symmetrize",
            "tensor",
            "symmetrize(expr, [positions])",
            "Alias for symmetrise.",
            "symmetrize(T[a-, b-], [0,1])",
        ),
        b(
            "sym",
            "tensor",
            "sym(expr, [positions])",
            "Short alias for symmetrise.",
            "sym(T[a-, b-], [0,1])",
        ),
        b(
            "antisymmetrise",
            "tensor",
            "antisymmetrise(expr, [positions])",
            "Antisymmetrise over listed slots.",
            "antisymmetrise(F[a-, b-], [0,1])",
        ),
        b(
            "antisymmetrize",
            "tensor",
            "antisymmetrize(expr, [positions])",
            "Alias for antisymmetrise.",
            "antisymmetrize(F[a-, b-], [0,1])",
        ),
        b(
            "asym",
            "tensor",
            "asym(expr, [positions])",
            "Short alias for antisymmetrise.",
            "asym(F[a-, b-], [0,1])",
        ),
        b(
            "eliminate_metric",
            "tensor",
            "eliminate_metric(expr)",
            "Use the metric or inverse metric to raise or lower contracted indices.",
            "eliminate_metric(g[mu-, nu-] * V[nu+])",
        ),
        b(
            "eliminate_vielbein",
            "tensor",
            "eliminate_vielbein(expr)",
            "Simplify vielbein contractions into metric data when possible.",
            "eliminate_vielbein(e[a-,mu-] * e[b+,mu+])",
        ),
        b(
            "decompose",
            "tensor",
            "decompose(expr)",
            "Decompose a tensor into symmetry-adapted pieces.",
            "decompose(T[a-, b-])",
        ),
        b(
            "decompose_product",
            "tensor",
            "decompose_product(expr)",
            "Decompose a tensor product using known tensor properties.",
            "decompose_product(g[a-,b-] * T[b+,c-])",
        ),
        b(
            "epsilon_to_delta",
            "tensor",
            "epsilon_to_delta(expr)",
            "Convert epsilon-tensor contractions into generalized Kronecker deltas.",
            "epsilon_to_delta(epsilon[a-,b-,c-] * epsilon[a+,d+,e+])",
        ),
        b(
            "evaluate",
            "tensor",
            "evaluate(expr, rules)",
            "Evaluate tensor components using declared component rules.",
            "evaluate(g[mu-, nu-], rules)",
        ),
        b(
            "eval_components",
            "tensor",
            "eval_components(expr, rules)",
            "Alias for evaluate component expressions.",
            "eval_components(g[mu-, nu-], rules)",
        ),
        b(
            "dim",
            "units",
            "dim(expr)",
            "Return the dimension of a units-aware expression.",
            "dim(force)",
        ),
        b(
            "convert",
            "units",
            "convert(expr, units)",
            "Convert an expression between compatible units.",
            "convert(1*m, cm)",
        ),
        b(
            "check_units",
            "units",
            "check_units(expr)",
            "Verify that a units expression is dimensionally consistent.",
            "check_units(force == mass*acceleration)",
        ),
        b(
            "metric",
            "properties",
            "metric(tensor)",
            "Declare a tensor as a metric and attach the symmetric metric property.",
            "metric(g)",
        ),
        b(
            "symmetric",
            "properties",
            "symmetric(tensor)",
            "Property marker used in property declarations and metadata.",
            "symmetric(g)",
        ),
        b(
            "antisymmetric",
            "properties",
            "antisymmetric(tensor)",
            "Property marker used in property declarations and metadata.",
            "antisymmetric(F)",
        ),
        b(
            "inverse_metric",
            "properties",
            "inverse_metric(tensor)",
            "Declare a tensor as an inverse metric.",
            "inverse_metric(ginv)",
        ),
        b(
            "kronecker_delta",
            "properties",
            "kronecker_delta(tensor)",
            "Declare a tensor as a Kronecker delta.",
            "kronecker_delta(delta)",
        ),
        b(
            "kronecker",
            "properties",
            "kronecker(tensor)",
            "Alias for kronecker_delta.",
            "kronecker(delta)",
        ),
        b(
            "epsilon",
            "properties",
            "epsilon(tensor)",
            "Declare a tensor as an epsilon or Levi-Civita tensor.",
            "epsilon(eps)",
        ),
        b(
            "epsilon_tensor",
            "properties",
            "epsilon_tensor(tensor)",
            "Alias for epsilon.",
            "epsilon_tensor(eps)",
        ),
        b(
            "riemann",
            "properties",
            "riemann(tensor)",
            "Declare Riemann slot symmetries on a tensor.",
            "riemann(R)",
        ),
        b(
            "riemann_symmetry",
            "properties",
            "riemann_symmetry(tensor)",
            "Property marker for Riemann-like slot symmetries.",
            "riemann_symmetry(R)",
        ),
        b(
            "riemann_tensor",
            "properties",
            "riemann_tensor(tensor)",
            "Declare a tensor as an abstract Riemann tensor, attaching Riemann symmetry plus the first Bianchi identity.",
            "riemann_tensor(R)",
        ),
        b(
            "traceless",
            "properties",
            "traceless(tensor)",
            "Property marker for traceless tensors.",
            "traceless(T)",
        ),
        b(
            "derivative",
            "properties",
            "derivative(op)",
            "Declare a symbol as a derivative operator.",
            "derivative(D)",
        ),
        b(
            "partial_derivative",
            "properties",
            "partial_derivative(op)",
            "Declare a symbol as a partial derivative operator.",
            "partial_derivative(partial)",
        ),
        b(
            "covariant_derivative",
            "properties",
            "covariant_derivative(op)",
            "Declare a symbol as a covariant derivative operator.",
            "covariant_derivative(nabla)",
        ),
        b(
            "spinor",
            "properties",
            "spinor(tensor)",
            "Declare a tensor as carrying spinor indices.",
            "spinor(psi)",
        ),
        b(
            "dirac_bar",
            "properties",
            "dirac_bar(symbol)",
            "Declare a symbol as a Dirac-bar object.",
            "dirac_bar(psibar)",
        ),
        b(
            "diracbar",
            "properties",
            "diracbar(symbol)",
            "Alias for dirac_bar.",
            "diracbar(psibar)",
        ),
        b(
            "gamma_matrix",
            "properties",
            "gamma_matrix(symbol)",
            "Declare a symbol as a gamma matrix.",
            "gamma_matrix(gamma)",
        ),
        b(
            "declare_spinor_meta",
            "properties",
            "declare_spinor_meta(symbol, dim, class, chirality, family)",
            "Attach structured spinor metadata and compatible legacy markers.",
            "declare_spinor_meta(psi, 4, Majorana, none, spin)",
        ),
        b(
            "declare_gamma_matrix_meta",
            "properties",
            "declare_gamma_matrix_meta(symbol, dim, metric, family, has_gamma5)",
            "Attach structured gamma-matrix metadata and the legacy gamma-matrix marker.",
            "declare_gamma_matrix_meta(gamma, 4, eta, spin, true)",
        ),
        b(
            "declare_gamma_convention",
            "properties",
            "declare_gamma_convention(symbol, signature, clifford, dimension)",
            "Attach structured gamma-matrix convention metadata.",
            "declare_gamma_convention(gamma, mostly_plus, plus_two_g, 4)",
        ),
        b(
            "declare_gamma5_convention",
            "properties",
            "declare_gamma5_convention(symbol, signature, clifford, gamma5_kind, epsilon_symbol, dimension)",
            "Attach structured gamma5 convention metadata.",
            "declare_gamma5_convention(gamma, mostly_plus, plus_two_g, levi_civita, epsilon, 4)",
        ),
        b(
            "declare_dirac_bar_meta",
            "properties",
            "declare_dirac_bar_meta(symbol, gamma_symbol, family, reverse_gamma_order)",
            "Attach structured Dirac-bar metadata and the legacy DiracBar marker.",
            "declare_dirac_bar_meta(psibar, gamma, spin, true)",
        ),
        b(
            "declare_trace_space",
            "properties",
            "declare_trace_space(symbol, space_symbol, cyclic)",
            "Attach structured trace-space metadata to a trace-like symbol.",
            "declare_trace_space(Tr, color, true)",
        ),
        b(
            "declare_hilbert_space",
            "properties",
            "declare_hilbert_space(symbol, dim)",
            "Attach structured finite-dimensional Hilbert-space metadata to a symbol.",
            "declare_hilbert_space(H, 2)",
        ),
        b(
            "declare_composite_space",
            "properties",
            "declare_composite_space(symbol, factors)",
            "Declare a composite Hilbert space from previously declared factor spaces.",
            "declare_composite_space(HAB, [HA, HB])",
        ),
        b(
            "declare_quantum_object",
            "properties",
            "declare_quantum_object(symbol, kind, space_symbol)",
            "Attach structured quantum-object metadata and legacy operator compatibility markers when required.",
            "declare_quantum_object(A, operator, H)",
        ),
        b(
            "declare_operator_space",
            "properties",
            "declare_operator_space(symbol, domain_space, codomain_space)",
            "Attach structured operator domain/codomain metadata using previously declared Hilbert spaces.",
            "declare_operator_space(U, HA, HB)",
        ),
        b(
            "declare_mode",
            "properties",
            "declare_mode(symbol, statistics, mode_index)",
            "Attach structured mode metadata and the legacy commutation markers implied by the mode statistics.",
            "declare_mode(a, bosonic, 0)",
        ),
        b(
            "compose_operators",
            "qm",
            "compose_operators(left, right)",
            "Build a symbolic operator composition, checking compatible codomain/domain metadata when available.",
            "compose_operators(U, V)",
        ),
        b(
            "declare_mode_in_subsystem",
            "properties",
            "declare_mode_in_subsystem(symbol, statistics, subsystem, mode_index)",
            "Attach structured mode metadata with an explicit subsystem tag and compatible legacy commutation markers.",
            "declare_mode_in_subsystem(a0, bosonic, QA, 0)",
        ),
        b(
            "declare_mode_with_label",
            "properties",
            "declare_mode_with_label(symbol, statistics, subsystem, mode_index, label)",
            "Attach structured mode metadata with subsystem and symbolic label metadata plus compatible legacy commutation markers.",
            "declare_mode_with_label(mode0, fermionic, reg, 0, a)",
        ),
        b(
            "declare_bosonic_truncated_mode",
            "properties",
            "declare_bosonic_truncated_mode(symbol, mode_index, nmax)",
            "Attach bosonic ModeMeta and remember a finite occupation cutoff for later Fock-space declarations.",
            "declare_bosonic_truncated_mode(a0, 0, 3)",
        ),
        b(
            "declare_fermionic_mode",
            "properties",
            "declare_fermionic_mode(symbol, mode_index)",
            "Attach fermionic ModeMeta for a mode used in Fock-space declarations.",
            "declare_fermionic_mode(c0, 0)",
        ),
        b(
            "declare_fock_space",
            "properties",
            "declare_fock_space(symbol, mode_symbols)",
            "Attach structured Fock-space metadata from previously declared mode symbols, preserving the listed occupation-basis order.",
            "declare_fock_space(F, [a0, a1])",
        ),
        b(
            "bosonic_fock_basis_state",
            "qm",
            "bosonic_fock_basis_state(space_symbol, occupations)",
            "Build a bosonic occupation-basis state for a declared Fock space after validating mode count and truncations.",
            "bosonic_fock_basis_state(F, [1, 0])",
        ),
        b(
            "fermionic_fock_basis_state",
            "qm",
            "fermionic_fock_basis_state(space_symbol, occupations)",
            "Build a fermionic occupation-basis state for a declared Fock space after validating mode count and 0/1 occupations.",
            "fermionic_fock_basis_state(Ff, [1, 0, 1])",
        ),
        b(
            "commuting",
            "properties",
            "commuting(symbol)",
            "Declare an object as commuting.",
            "commuting(A)",
        ),
        b(
            "anticommuting",
            "properties",
            "anticommuting(symbol)",
            "Declare an object as anticommuting.",
            "anticommuting(psi)",
        ),
        b(
            "anti_commuting",
            "properties",
            "anti_commuting(symbol)",
            "Alias for anticommuting.",
            "anti_commuting(psi)",
        ),
        b(
            "noncommuting",
            "properties",
            "noncommuting(symbol)",
            "Declare an object as noncommuting.",
            "noncommuting(A)",
        ),
        b(
            "non_commuting",
            "properties",
            "non_commuting(symbol)",
            "Alias for noncommuting.",
            "non_commuting(A)",
        ),
        b(
            "bianchi",
            "properties",
            "bianchi(tensor)",
            "Declare that a tensor satisfies a Bianchi identity.",
            "bianchi(R)",
        ),
        b(
            "satisfies_bianchi",
            "properties",
            "satisfies_bianchi(tensor)",
            "Alias for bianchi.",
            "satisfies_bianchi(R)",
        ),
        b(
            "weyl",
            "properties",
            "weyl(tensor)",
            "Declare a tensor as a Weyl tensor.",
            "weyl(C)",
        ),
        b(
            "weyl_tensor",
            "properties",
            "weyl_tensor(tensor)",
            "Alias for weyl.",
            "weyl_tensor(C)",
        ),
        b(
            "tableau_symmetry",
            "properties",
            "tableau_symmetry(tensor, shape, indices)",
            "Declare Young-tableau symmetry data on a tensor.",
            "tableau_symmetry(T, [2,1], [0,1,2])",
        ),
        b(
            "grassmann",
            "quantum",
            "grassmann(sym)",
            "Declare a Grassmann-odd symbol.",
            "grassmann theta eta",
        ),
        b(
            "grassmann_simplify",
            "quantum",
            "grassmann_simplify(expr)",
            "Simplify using Grassmann anticommutation.",
            "grassmann_simplify(theta*theta)",
        ),
        b(
            "solve",
            "algebra",
            "solve(expr, var) or solve([eqs], [vars])",
            "Solve one polynomial equation or a linear system.",
            "solve(x^2 - 5*x + 6, x)",
        ),
        b(
            "det",
            "linear-algebra",
            "det(matrix)",
            "Determinant of a matrix.",
            "det([[1,2],[3,4]])",
        ),
        b(
            "inv",
            "linear-algebra",
            "inv(matrix)",
            "Inverse of a matrix.",
            "inv([[1,0],[0,2]])",
        ),
        b(
            "transpose",
            "linear-algebra",
            "transpose(matrix)",
            "Transpose a matrix.",
            "transpose([[1,2],[3,4]])",
        ),
        b(
            "trace_mat",
            "linear-algebra",
            "trace_mat(matrix)",
            "Trace of a matrix.",
            "trace_mat([[1,2],[3,4]])",
        ),
        b(
            "eigenvalues",
            "linear-algebra",
            "eigenvalues(matrix)",
            "Eigenvalues of a small symbolic or numeric matrix.",
            "eigenvalues([[1,0],[0,2]])",
        ),
        b(
            "hermitian_eigenvalues",
            "quantum",
            "hermitian_eigenvalues(matrix)",
            "Exact small Hermitian eigenvalues for supported 2x2 and diagonal 3x3 matrices.",
            "hermitian_eigenvalues(pauli_z())",
        ),
        b(
            "hermitian_eigenprojectors",
            "quantum",
            "hermitian_eigenprojectors(matrix)",
            "Exact small Hermitian spectral projectors for supported nondegenerate 2x2 and diagonal 3x3 matrices.",
            "hermitian_eigenprojectors(pauli_z())",
        ),
        b(
            "first_order_energy_shift",
            "quantum",
            "first_order_energy_shift(H0, V, n)",
            "Exact nondegenerate stationary perturbation-theory first-order energy shift in the eigenbasis of H0.",
            "first_order_energy_shift([[1,0],[0,2]], [[a,0],[0,b]], 0)",
        ),
        b(
            "second_order_energy_shift",
            "quantum",
            "second_order_energy_shift(H0, V, n)",
            "Exact nondegenerate stationary perturbation-theory second-order energy shift in the eigenbasis of H0.",
            "second_order_energy_shift([[1,0],[0,2]], [[0,g],[g,0]], 0)",
        ),
        b(
            "degenerate_effective_perturbation",
            "quantum",
            "degenerate_effective_perturbation(H0, V, subspace)",
            "Exact effective perturbation matrix inside a chosen degenerate basis-state subspace of a diagonal H0.",
            "degenerate_effective_perturbation([[1,0,0],[0,1,0],[0,0,2]], [[a,b,0],[c,d,0],[0,0,1]], [0,1])",
        ),
        b(
            "degenerate_first_order_splittings",
            "quantum",
            "degenerate_first_order_splittings(H0, V, subspace)",
            "Exact first-order splittings obtained by diagonalizing the effective perturbation inside a chosen degenerate subspace.",
            "degenerate_first_order_splittings([[1,0,0],[0,1,0],[0,0,2]], [[0,g,0],[g,0,0],[0,0,1]], [0,1])",
        ),
        b(
            "berry_connection",
            "quantum",
            "berry_connection(psi, parameter)",
            "Construct the symbolic Berry-connection one-form component i <psi | d/dparameter | psi>.",
            "berry_connection(psi(theta), theta)",
        ),
        b(
            "geometric_phase",
            "quantum",
            "geometric_phase(A, parameter)",
            "Construct the symbolic geometric phase as a contour-style integral of a Berry connection.",
            "geometric_phase(berry_connection(psi(theta), theta), theta)",
        ),
        b(
            "matmul",
            "linear-algebra",
            "matmul(a, b)",
            "Matrix multiplication.",
            "matmul(A, B)",
        ),
        b(
            "identity",
            "linear-algebra",
            "identity(n)",
            "n×n identity matrix.",
            "identity(3)",
        ),
        b(
            "tensor_product",
            "linear-algebra",
            "tensor_product(a, b)",
            "Kronecker or tensor product of arrays or operators.",
            "tensor_product(A, B)",
        ),
        b(
            "pauli_x",
            "quantum",
            "pauli_x()",
            "Pauli sigma_x matrix.",
            "pauli_x()",
        ),
        b(
            "sigma_x",
            "quantum",
            "sigma_x()",
            "Alias for pauli_x.",
            "sigma_x()",
        ),
        b(
            "pauli_y",
            "quantum",
            "pauli_y()",
            "Pauli sigma_y matrix.",
            "pauli_y()",
        ),
        b(
            "sigma_y",
            "quantum",
            "sigma_y()",
            "Alias for pauli_y.",
            "sigma_y()",
        ),
        b(
            "pauli_z",
            "quantum",
            "pauli_z()",
            "Pauli sigma_z matrix.",
            "pauli_z()",
        ),
        b(
            "jz",
            "quantum",
            "jz(two_j)",
            "Spin-j angular-momentum J_z matrix using the exact integer label two_j = 2j.",
            "jz(1)",
        ),
        b(
            "jplus",
            "quantum",
            "jplus(two_j)",
            "Spin-j angular-momentum raising operator J_+ using the exact integer label two_j = 2j.",
            "jplus(2)",
        ),
        b(
            "jminus",
            "quantum",
            "jminus(two_j)",
            "Spin-j angular-momentum lowering operator J_- using the exact integer label two_j = 2j.",
            "jminus(2)",
        ),
        b(
            "jx",
            "quantum",
            "jx(two_j)",
            "Spin-j angular-momentum J_x matrix using the exact integer label two_j = 2j.",
            "jx(1)",
        ),
        b(
            "jy",
            "quantum",
            "jy(two_j)",
            "Spin-j angular-momentum J_y matrix using the exact integer label two_j = 2j.",
            "jy(1)",
        ),
        b(
            "singlet_state_2spinhalf",
            "quantum",
            "singlet_state_2spinhalf()",
            "Explicit two-spin-1/2 singlet state in the basis |↑↑>, |↑↓>, |↓↑>, |↓↓>.",
            "singlet_state_2spinhalf()",
        ),
        b(
            "triplet_states_2spinhalf",
            "quantum",
            "triplet_states_2spinhalf()",
            "Explicit two-spin-1/2 triplet states in the basis |↑↑>, |↑↓>, |↓↑>, |↓↓>.",
            "triplet_states_2spinhalf()",
        ),
        b(
            "singlet_projector_2spinhalf",
            "quantum",
            "singlet_projector_2spinhalf()",
            "Projector onto the explicit two-spin-1/2 singlet subspace.",
            "singlet_projector_2spinhalf()",
        ),
        b(
            "triplet_projector_2spinhalf",
            "quantum",
            "triplet_projector_2spinhalf()",
            "Projector onto the explicit two-spin-1/2 triplet subspace.",
            "triplet_projector_2spinhalf()",
        ),
        b(
            "time_evolution_operator",
            "quantum",
            "time_evolution_operator(H, t)",
            "Constant-Hamiltonian unitary propagator U(t) = exp(-i t H) for supported small Hermitian matrices.",
            "time_evolution_operator(pauli_z(), t)",
        ),
        b(
            "schrodinger_evolve",
            "quantum",
            "schrodinger_evolve(H, psi0, t)",
            "Schrodinger-picture pure-state evolution psi(t) = U(t) psi0 for supported finite-dimensional Hermitian Hamiltonians.",
            "schrodinger_evolve(pauli_z(), [1, 0], t)",
        ),
        b(
            "heisenberg_evolve",
            "quantum",
            "heisenberg_evolve(H, O0, t)",
            "Heisenberg-picture operator evolution O(t) = U†(t) O0 U(t) for supported finite-dimensional Hermitian Hamiltonians.",
            "heisenberg_evolve(pauli_z(), pauli_x(), t)",
        ),
        b(
            "liouville_rhs",
            "quantum",
            "liouville_rhs(H, rho)",
            "Closed-system density-matrix right-hand side ρ̇ = -i [H, ρ] for square Hamiltonian and density matrices of matching dimension.",
            "liouville_rhs([[1, 0], [0, 2]], [[1, 0], [0, 0]])",
        ),
        b(
            "dyson_series",
            "quantum",
            "dyson_series(Ht, order)",
            "Finite-order symbolic Dyson expansion for a time-dependent Hamiltonian using canonical time-order and nested integral placeholders.",
            "dyson_series(t * pauli_z(), 2)",
        ),
        b(
            "magnus_expansion",
            "quantum",
            "magnus_expansion(Ht, order)",
            "Finite-order symbolic Magnus expansion for a time-dependent Hamiltonian using nested commutator integrals.",
            "magnus_expansion(t * pauli_z(), 3)",
        ),
        b(
            "kubo_response",
            "quantum",
            "kubo_response(A, B, rho0, t)",
            "Construct the symbolic Kubo linear-response function -i theta(t) Tr(rho0 [A(t), B(0)]).",
            "kubo_response(x, p, rho0, t)",
        ),
        b(
            "susceptibility_fourier",
            "quantum",
            "susceptibility_fourier(chi_t, omega)",
            "Construct the symbolic Fourier susceptibility integral ∫ dt exp(i omega t) chi_t using canonical integral bounds.",
            "susceptibility_fourier(kubo_response(x, p, rho0, t), omega)",
        ),
        b(
            "projector_left",
            "quantum",
            "projector_left()",
            "Construct the canonical left chiral projector P_L = (1 - gamma5)/2.",
            "projector_left()",
        ),
        b(
            "projector_right",
            "quantum",
            "projector_right()",
            "Construct the canonical right chiral projector P_R = (1 + gamma5)/2.",
            "projector_right()",
        ),
        b(
            "simplify_chiral",
            "quantum",
            "simplify_chiral(expr)",
            "Simplify chiral projector algebra and projector actions on Weyl spinors with explicit chirality metadata.",
            "simplify_chiral(projector_left() * projector_right())",
        ),
        b(
            "simplify_spinor_bilinears",
            "quantum",
            "simplify_spinor_bilinears(expr)",
            "Apply conservative metadata-driven 4D Majorana and Weyl bilinear selection rules, returning zero only for proven forbidden bilinears.",
            "simplify_spinor_bilinears(bar(psi) * gamma(mu) * psi)",
        ),
        b(
            "sigma",
            "quantum",
            "sigma(mu, nu)",
            "Construct the canonical sigma^{mu nu} spin-generator basis element with sigma^{mu nu} = (i/2)[gamma^mu, gamma^nu].",
            "sigma(mu, nu)",
        ),
        b(
            "sigma_to_gamma",
            "quantum",
            "sigma_to_gamma(expr)",
            "Expand sigma(mu,nu) into the exact gamma commutator form (i/2)(gamma(mu) gamma(nu) - gamma(nu) gamma(mu)).",
            "sigma_to_gamma(sigma(mu, nu))",
        ),
        b(
            "gamma_to_sigma",
            "quantum",
            "gamma_to_sigma(expr)",
            "Convert an exact canonical gamma commutator pattern back to -2i sigma(mu,nu), leaving nonmatching input unchanged.",
            "gamma_to_sigma(gamma(mu) * gamma(nu) - gamma(nu) * gamma(mu))",
        ),
        b(
            "sigma_z",
            "quantum",
            "sigma_z()",
            "Alias for pauli_z.",
            "sigma_z()",
        ),
        b(
            "gamma",
            "quantum",
            "gamma(index)",
            "Dirac gamma matrix for an index.",
            "gamma(mu)",
        ),
        b(
            "gamma5",
            "quantum",
            "gamma5()",
            "Dirac gamma_5 matrix.",
            "gamma5()",
        ),
        b(
            "commutator",
            "quantum",
            "commutator(a, b)",
            "Operator commutator [a, b].",
            "commutator(A, B)",
        ),
        b(
            "anticommutator",
            "quantum",
            "anticommutator(a, b)",
            "Operator anticommutator {a, b}.",
            "anticommutator(A, B)",
        ),
        b(
            "displacement_series",
            "quantum",
            "displacement_series(alpha, mode, order)",
            "Construct a truncated symbolic power series for the bosonic displacement operator.",
            "displacement_series(alpha, a, 2)",
        ),
        b(
            "squeezing_series",
            "quantum",
            "squeezing_series(zeta, mode, order)",
            "Construct a truncated symbolic power series for the bosonic squeezing operator.",
            "squeezing_series(zeta, a, 2)",
        ),
        b(
            "ket",
            "quantum",
            "ket(label)",
            "Construct a ket vector.",
            "ket(0)",
        ),
        b(
            "bra",
            "quantum",
            "bra(label)",
            "Construct a bra vector.",
            "bra(1)",
        ),
        b(
            "braket",
            "quantum",
            "braket(bra, ket)",
            "Inner product of a bra and a ket.",
            "braket(ket(0), ket(1))",
        ),
        b(
            "outer",
            "quantum",
            "outer(ket, bra)",
            "Outer product operator.",
            "outer(ket(0), ket(0))",
        ),
        b(
            "density",
            "quantum",
            "density(state)",
            "Density matrix of a pure state.",
            "density([1/sqrt(2), 0, 0, 1/sqrt(2)])",
        ),
        b(
            "partial_trace",
            "quantum",
            "partial_trace(rho, dim_a, dim_b, subsystem)",
            "Partial trace over a subsystem.",
            "partial_trace(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2, A)",
        ),
        b(
            "partial_trace_factor",
            "quantum",
            "partial_trace_factor(rho, dims, factor_index)",
            "Partial trace over one factor of a general composite Hilbert space.",
            "partial_trace_factor(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), [2, 2], 1)",
        ),
        b(
            "partial_transpose_factor",
            "quantum",
            "partial_transpose_factor(rho, dims, factor_index)",
            "Partial transpose on one factor of a general composite Hilbert space.",
            "partial_transpose_factor(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), [2, 2], 1)",
        ),
        b(
            "permute_subsystems",
            "quantum",
            "permute_subsystems(rho, dims, permutation)",
            "Permute subsystem order by exact composite-basis relabeling.",
            "permute_subsystems(density([0, 1, 0, 0]), [2, 2], [1, 0])",
        ),
        b(
            "partial_trace_space",
            "quantum",
            "partial_trace_space(rho, composite_space_symbol, factor_space_symbol)",
            "Partial trace using declared composite-space metadata.",
            "partial_trace_space(rho, QAB, QB)",
        ),
        b(
            "basis_projector",
            "quantum",
            "basis_projector(index, dim)",
            "Projector onto a computational-basis state.",
            "basis_projector(0, 2)",
        ),
        b(
            "measurement_probabilities",
            "quantum",
            "measurement_probabilities(projectors, rho)",
            "Projective-measurement probabilities for a density matrix.",
            "measurement_probabilities([basis_projector(0, 2), basis_projector(1, 2)], [[1,0],[0,0]])",
        ),
        b(
            "expectation_value",
            "quantum",
            "expectation_value(operator, rho)",
            "Expectation value Tr(rho * operator) for a finite-dimensional observable.",
            "expectation_value(pauli_z(), density_matrix([1, 0]))",
        ),
        b(
            "variance",
            "quantum",
            "variance(operator, rho)",
            "Variance Tr(rho * operator^2) - Tr(rho * operator)^2 for a finite-dimensional observable.",
            "variance(pauli_z(), density_matrix([1, 0]))",
        ),
        b(
            "purity",
            "quantum",
            "purity(rho)",
            "Purity Tr(rho^2) of a finite-dimensional density matrix.",
            "purity(density_matrix([1, 0]))",
        ),
        b(
            "linear_entropy",
            "quantum",
            "linear_entropy(rho)",
            "Linear entropy 1 - Tr(rho^2) of a finite-dimensional density matrix.",
            "linear_entropy([[1/2,0],[0,1/2]])",
        ),
        b(
            "renyi2_entropy",
            "quantum",
            "renyi2_entropy(rho)",
            "Renyi-2 entropy -log(Tr(rho^2)) of a finite-dimensional density matrix.",
            "renyi2_entropy([[1/2,0],[0,1/2]])",
        ),
        b(
            "renyi2_entropy_factor",
            "quantum",
            "renyi2_entropy_factor(rho, dims, kept_factor)",
            "Renyi-2 entropy of the reduced state obtained by keeping one tensor factor and tracing out the rest.",
            "renyi2_entropy_factor(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), [2, 2], 0)",
        ),
        b(
            "von_neumann_entropy",
            "quantum",
            "von_neumann_entropy(rho)",
            "Von Neumann entropy -Tr(rho log rho) of a supported finite-dimensional Hermitian density matrix.",
            "von_neumann_entropy([[1/2,0],[0,1/2]])",
        ),
        b(
            "mutual_information",
            "quantum",
            "mutual_information(rho_ab, dim_a, dim_b)",
            "Bipartite von Neumann mutual information S(rho_A) + S(rho_B) - S(rho_AB).",
            "mutual_information(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2)",
        ),
        b(
            "conditional_entropy",
            "quantum",
            "conditional_entropy(rho_ab, dim_a, dim_b)",
            "Bipartite conditional entropy S(B|A) = S(rho_AB) - S(rho_A).",
            "conditional_entropy(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2)",
        ),
        b(
            "participation_ratio",
            "quantum",
            "participation_ratio(rho)",
            "Participation ratio 1 / Tr(rho^2) of a finite-dimensional density matrix.",
            "participation_ratio([[1/2,0],[0,1/2]])",
        ),
        b(
            "entanglement_spectrum",
            "quantum",
            "entanglement_spectrum(state_or_rho, dim_a, dim_b)",
            "Bipartite entanglement spectrum for a pure-state vector or bipartite density matrix, keeping subsystem A.",
            "entanglement_spectrum([1/sqrt(2), 0, 0, 1/sqrt(2)], 2, 2)",
        ),
        b(
            "schmidt_coefficients",
            "quantum",
            "schmidt_coefficients(state, dim_a, dim_b)",
            "Schmidt coefficients of a bipartite pure-state vector.",
            "schmidt_coefficients([1/sqrt(2), 0, 0, 1/sqrt(2)], 2, 2)",
        ),
        b(
            "negativity",
            "quantum",
            "negativity(rho_ab, dim_a, dim_b)",
            "Bipartite negativity from the supported partial-transpose spectrum.",
            "negativity(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2)",
        ),
        b(
            "logarithmic_negativity",
            "quantum",
            "logarithmic_negativity(rho_ab, dim_a, dim_b)",
            "Bipartite logarithmic negativity log(1 + 2 N(rho)).",
            "logarithmic_negativity(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2)",
        ),
        b(
            "renyi2_mutual_information",
            "quantum",
            "renyi2_mutual_information(rho_ab, dim_a, dim_b)",
            "Bipartite Renyi-2 mutual information S2(rho_A) + S2(rho_B) - S2(rho_AB).",
            "renyi2_mutual_information(density([1/sqrt(2), 0, 0, 1/sqrt(2)]), 2, 2)",
        ),
        b(
            "renyi2_tripartite_information",
            "quantum",
            "renyi2_tripartite_information(rho, dim_a, dim_b, dim_c)",
            "Tripartite Renyi-2 information S2(A) + S2(B) + S2(C) - S2(AB) - S2(AC) - S2(BC) + S2(ABC).",
            "renyi2_tripartite_information(density([1, 0, 0, 0, 0, 0, 0, 0]), 2, 2, 2)",
        ),
        b(
            "bloch_vector",
            "quantum",
            "bloch_vector(rho)",
            "Bloch-vector components [x, y, z] for a 2x2 density matrix.",
            "bloch_vector([[1,0],[0,0]])",
        ),
        b(
            "qubit_density_from_bloch",
            "quantum",
            "qubit_density_from_bloch([x, y, z])",
            "Qubit density matrix 1/2 (I + x sigma_x + y sigma_y + z sigma_z) from a Bloch vector.",
            "qubit_density_from_bloch([0, 0, 1])",
        ),
        b(
            "post_measurement_state",
            "quantum",
            "post_measurement_state(projector, rho, outcome_index)",
            "Normalized post-measurement state for a selected outcome.",
            "post_measurement_state(basis_projector(0, 2), [[1,0],[0,0]], 0)",
        ),
        b(
            "identity_channel",
            "quantum",
            "identity_channel(dim)",
            "Construct a finite-dimensional identity Kraus channel.",
            "identity_channel(2)",
        ),
        b(
            "depolarizing_channel",
            "quantum",
            "depolarizing_channel(p)",
            "Construct the canonical qubit depolarizing channel with Kraus operators sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, and sqrt(p/3) Z.",
            "depolarizing_channel(p)",
        ),
        b(
            "dephasing_channel",
            "quantum",
            "dephasing_channel(p)",
            "Construct the canonical qubit dephasing channel with Kraus operators sqrt(1-p) I and sqrt(p) Z.",
            "dephasing_channel(p)",
        ),
        b(
            "amplitude_damping_channel",
            "quantum",
            "amplitude_damping_channel(gamma)",
            "Construct the canonical qubit amplitude-damping channel with Kraus operators [[1,0],[0,sqrt(1-gamma)]] and [[0,sqrt(gamma)],[0,0]].",
            "amplitude_damping_channel(gamma)",
        ),
        b(
            "bit_flip_channel",
            "quantum",
            "bit_flip_channel(p)",
            "Construct the canonical qubit bit-flip channel with Kraus operators sqrt(1-p) I and sqrt(p) X.",
            "bit_flip_channel(p)",
        ),
        b(
            "phase_flip_channel",
            "quantum",
            "phase_flip_channel(p)",
            "Construct the canonical qubit phase-flip channel with Kraus operators sqrt(1-p) I and sqrt(p) Z.",
            "phase_flip_channel(p)",
        ),
        b(
            "bit_phase_flip_channel",
            "quantum",
            "bit_phase_flip_channel(p)",
            "Construct the canonical qubit bit-phase-flip channel with Kraus operators sqrt(1-p) I and sqrt(p) Y.",
            "bit_phase_flip_channel(p)",
        ),
        b(
            "apply_channel",
            "quantum",
            "apply_channel(kraus, rho)",
            "Apply a Kraus channel to a density matrix.",
            "apply_channel(identity_channel(2), [[1,0],[0,0]])",
        ),
        b(
            "compose_channels",
            "quantum",
            "compose_channels(left, right)",
            "Compose two Kraus channels so the right channel acts first and the left channel acts second.",
            "compose_channels(identity_channel(2), identity_channel(2))",
        ),
        b(
            "tensor_product_channel",
            "quantum",
            "tensor_product_channel(left, right)",
            "Form the tensor-product Kraus channel whose operators are L_i tensor R_j.",
            "tensor_product_channel(identity_channel(2), identity_channel(2))",
        ),
        b(
            "choi_distance",
            "quantum",
            "choi_distance(left, right)",
            "Compute the Frobenius distance between two channels using their Choi matrices.",
            "choi_distance(identity_channel(2), identity_channel(2))",
        ),
        b(
            "trace_preserving_residual",
            "quantum",
            "trace_preserving_residual(kraus)",
            "Compute the exact trace-preserving residual Σ_k K_k† K_k - I for a Kraus channel.",
            "trace_preserving_residual(identity_channel(2))",
        ),
        b(
            "is_trace_preserving",
            "quantum",
            "is_trace_preserving(kraus)",
            "Check whether a Kraus channel is exactly trace preserving.",
            "is_trace_preserving(identity_channel(2))",
        ),
        b(
            "unital_residual",
            "quantum",
            "unital_residual(kraus)",
            "Compute the exact unital residual Σ_k K_k K_k† - I for a Kraus channel.",
            "unital_residual(identity_channel(2))",
        ),
        b(
            "is_unital",
            "quantum",
            "is_unital(kraus)",
            "Check whether a Kraus channel is exactly unital.",
            "is_unital(identity_channel(2))",
        ),
        b(
            "lindblad_rhs",
            "quantum",
            "lindblad_rhs(H, rho, jumps)",
            "Construct the finite-dimensional Lindblad right-hand side for a density matrix.",
            "lindblad_rhs([[1,0],[0,2]], [[1/2,0],[0,1/2]], [])",
        ),
        b(
            "lindblad_euler_step",
            "quantum",
            "lindblad_euler_step(H, rho, jumps, dt)",
            "Take one explicit Euler step for finite-dimensional Lindblad evolution.",
            "lindblad_euler_step([[1,0],[0,2]], [[1,0],[0,0]], [], 1/10)",
        ),
        b(
            "lindblad_rk4_step",
            "quantum",
            "lindblad_rk4_step(H, rho, jumps, dt)",
            "Take one classical RK4 step for finite-dimensional Lindblad evolution.",
            "lindblad_rk4_step([[1,0],[0,2]], [[1,0],[0,0]], [], 1/10)",
        ),
        b(
            "lindblad_steady_state",
            "quantum",
            "lindblad_steady_state(H, jumps)",
            "Solve lindblad_rhs(H, rho, jumps) = 0 together with Tr(rho) = 1 for a finite-dimensional steady state.",
            "lindblad_steady_state([[0,0],[0,0]], [[[0,1],[0,0]]])",
        ),
        b(
            "lindbladian_superoperator",
            "quantum",
            "lindbladian_superoperator(H, jumps)",
            "Construct the exact Lindbladian superoperator acting on column-major vec(rho).",
            "lindbladian_superoperator([[0,0],[0,0]], [])",
        ),
        b(
            "lindbladian_eigenvalues",
            "quantum",
            "lindbladian_eigenvalues(H, jumps)",
            "Return exact low-dimensional Lindbladian eigenvalues for supported superoperators.",
            "lindbladian_eigenvalues([[0,0],[0,0]], [])",
        ),
        b(
            "sparse_steady_state",
            "quantum",
            "sparse_steady_state(H, jumps, tolerance, max_iterations)",
            "Solve a numeric Lindblad steady-state problem using the sparse plugin backend when the backend selector chooses sparse mode.",
            "sparse_steady_state([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], [], 1e-8, 1000)",
        ),
        b(
            "sparse_lindbladian_spectrum",
            "quantum",
            "sparse_lindbladian_spectrum(H, jumps, k, which, tolerance, max_iterations)",
            "Compute selected eigenvalues of a numeric Lindbladian superoperator using the sparse plugin backend.",
            "sparse_lindbladian_spectrum([[0,0],[0,0]], [], 2, LR, 1e-8, 1000)",
        ),
        b(
            "creation",
            "quantum",
            "creation(sym)",
            "Construct an abstract creation operator. As a top-level statement it also declares the symbol as a creation operator with bosonic normal-ordering metadata.",
            "creation(a)",
        ),
        b(
            "annihilation",
            "quantum",
            "annihilation(sym)",
            "Construct an abstract annihilation operator. As a top-level statement it also declares the symbol as an annihilation operator with bosonic normal-ordering metadata.",
            "annihilation(a)",
        ),
        b(
            "number_state",
            "quantum",
            "number_state(mode, n)",
            "Construct an abstract Fock number state |n> for the given oscillator mode.",
            "number_state(a, 2)",
        ),
        b(
            "fock_state",
            "quantum",
            "fock_state(occupations)",
            "Construct a canonical multimode bosonic occupation-basis state `fock_state([n0, n1, ...])`.",
            "fock_state([2, 0, 1])",
        ),
        b(
            "fermion_state",
            "quantum",
            "fermion_state(occupations)",
            "Construct a canonical multimode fermionic occupation-basis state `fermion_state([n0, n1, ...])`.",
            "fermion_state([1, 0, 0])",
        ),
        b(
            "bosonic_creation_action",
            "quantum",
            "bosonic_creation_action(mode, occupations)",
            "Apply a bosonic creation operator to one mode of a multimode occupation-basis state.",
            "bosonic_creation_action(1, [2, 0, 1])",
        ),
        b(
            "fermionic_creation_action",
            "quantum",
            "fermionic_creation_action(mode, occupations)",
            "Apply a fermionic creation operator to one mode of a multimode occupation-basis state with exact Jordan-Wigner signs.",
            "fermionic_creation_action(1, [1, 0, 0])",
        ),
        b(
            "bosonic_annihilation_action",
            "quantum",
            "bosonic_annihilation_action(mode, occupations)",
            "Apply a bosonic annihilation operator to one mode of a multimode occupation-basis state.",
            "bosonic_annihilation_action(0, [2, 0, 1])",
        ),
        b(
            "fermionic_annihilation_action",
            "quantum",
            "fermionic_annihilation_action(mode, occupations)",
            "Apply a fermionic annihilation operator to one mode of a multimode occupation-basis state with exact Jordan-Wigner signs.",
            "fermionic_annihilation_action(2, [1, 1, 1])",
        ),
        b(
            "vacuum",
            "quantum",
            "vacuum(mode)",
            "Construct the oscillator vacuum state |0> for the given mode.",
            "vacuum(a)",
        ),
        b(
            "number_operator",
            "quantum",
            "number_operator(mode)",
            "Construct the abstract harmonic-oscillator number operator a† a.",
            "number_operator(a)",
        ),
        b(
            "hamiltonian_ho",
            "quantum",
            "hamiltonian_ho(mode, [hbar], [omega])",
            "Construct the harmonic-oscillator Hamiltonian proportional to N + 1/2.",
            "hamiltonian_ho(a, hbar, omega)",
        ),
        b(
            "apply_operator",
            "quantum",
            "apply_operator(op, state)",
            "Apply an abstract oscillator operator expression to a Fock state.",
            "apply_operator(creation(a), vacuum(a))",
        ),
        b(
            "normal_order",
            "quantum",
            "normal_order(expr)",
            "Reorder ladder operators into graded normal order, preferring structured mode metadata and falling back to declared bosonic or fermionic operator statistics.",
            "normal_order(annihilation(a) * creation(a))",
        ),
        b(
            "time_order",
            "quantum",
            "time_order(expr)",
            "Wrap an expression in the canonical symbolic time-ordering operator without expanding it.",
            "time_order(a * b)",
        ),
        b(
            "anti_time_order",
            "quantum",
            "anti_time_order(expr)",
            "Wrap an expression in the canonical symbolic anti-time-ordering operator without expanding it.",
            "anti_time_order(a * b)",
        ),
        b(
            "bch",
            "quantum",
            "bch(A, B, order)",
            "Construct the finite-order symbolic Baker-Campbell-Hausdorff expansion for log(exp(A) exp(B)).",
            "bch(A, B, 2)",
        ),
        b(
            "wick",
            "quantum",
            "wick(expr)",
            "Expand products using declared Wick contractions, including fermionic pairing signs from operator swaps and crossing parity while preferring structured mode metadata over legacy operator statistics when needed.",
            "wick(annihilation(a) * creation(a))",
        ),
        b(
            "simplify_ccr_car",
            "quantum",
            "simplify_ccr_car(expr)",
            "Apply explicit bosonic CCR and fermionic CAR rewrites using structured mode metadata when available.",
            "simplify_ccr_car(annihilation(a) * creation(a))",
        ),
        b(
            "declare_contraction",
            "quantum",
            "declare_contraction(lhs, rhs, value)",
            "Declare a Wick contraction value for an ordered pair of operator modes.",
            "declare_contraction(a, a, 1)",
        ),
        b(
            "join_gamma",
            "quantum",
            "join_gamma(expr)",
            "Join adjacent gamma matrices into a compact gamma chain.",
            "join_gamma(gamma(mu) * gamma(nu))",
        ),
        b(
            "split_gamma",
            "quantum",
            "split_gamma(expr)",
            "Split compact gamma-chain structures into explicit factors.",
            "split_gamma(expr)",
        ),
        b(
            "gamma_trace",
            "quantum",
            "gamma_trace(expr)",
            "Trace over a chain of gamma matrices.",
            "gamma_trace([mu, nu])",
        ),
        b(
            "gamma5_trace",
            "quantum",
            "gamma5_trace(expr)",
            "Trace a gamma-chain with gamma_5 inserted.",
            "gamma5_trace([mu, nu, rho, sigma])",
        ),
        b(
            "euler_lagrange",
            "variational",
            "euler_lagrange(L, field, field_derivs, coords)",
            "Compute Euler-Lagrange equations.",
            "euler_lagrange(L, phi, [phi_t, phi_x], [t, x])",
        ),
        b(
            "functional_derivative",
            "variational",
            "functional_derivative(L, field, field_derivs, coords)",
            "Compute the Euler-Lagrange functional derivative with respect to a field.",
            "functional_derivative(L, phi, [phi_t, phi_x], [t, x])",
        ),
        b(
            "euler_lagrange_system",
            "variational",
            "euler_lagrange_system(L, [[field, derivs], ...], coords)",
            "Compute Euler-Lagrange equations for several fields at once.",
            "euler_lagrange_system(L, [[phi, [phi_t]], [chi, [chi_t]]], [t])",
        ),
        b(
            "vary_action",
            "variational",
            "vary_action(L, field, variation, field_derivs, variation_derivs)",
            "Take the first variation of an action density before integration by parts.",
            "vary_action(L, phi, delta_phi, [phi_t, phi_x], [delta_phi_t, delta_phi_x])",
        ),
        b(
            "vary",
            "variational",
            "vary(L, field, variation, field_derivs, variation_derivs)",
            "Alias for vary_action.",
            "vary(L, phi, delta_phi, [phi_t, phi_x], [delta_phi_t, delta_phi_x])",
        ),
        b(
            "dsolve",
            "ode",
            "dsolve(eq, y, x)",
            "Solve a supported first-order ODE symbolically.",
            "dsolve(diff(y, x) - y, y, x)",
        ),
        b(
            "first_order_form",
            "ode",
            "first_order_form(ode, dep, indep)",
            "Convert a higher-order ODE to a first-order system.",
            "first_order_form(diff(diff(x,t),t)+x, x, t)",
        ),
        b(
            "rk4",
            "ode",
            "rk4(f, x, y, x0, y0, x1[, steps])",
            "Numerically integrate an ODE with fourth-order Runge-Kutta.",
            "rk4(y, x, y, 0, 1, 1, 100)",
        ),
        b(
            "classify_pde",
            "pde",
            "classify_pde(A, B, C)",
            "Classify a second-order PDE as elliptic, parabolic, or hyperbolic.",
            "classify_pde(1, 0, -1)",
        ),
        b(
            "separate_variables",
            "pde",
            "separate_variables(type, x, t[, coeff])",
            "Return a standard separated solution ansatz for a supported PDE family.",
            "separate_variables(wave, x, t, c)",
        ),
        b(
            "separation",
            "pde",
            "separation(type, x, t[, coeff])",
            "Alias for separate_variables.",
            "separation(heat, x, t, alpha)",
        ),
        b(
            "plot",
            "plotting",
            "plot(expr, var, xmin, xmax)",
            "Plot a one-dimensional expression to SVG.",
            "plot(sin(x), x, 0, 6.28)",
        ),
        b(
            "wedge",
            "forms",
            "wedge(a, b)",
            "Wedge product of differential forms of any supported degree.",
            "wedge(A, B)",
        ),
        b(
            "wedge_1_1",
            "forms",
            "wedge_1_1(a, b)",
            "Wedge product of two 1-forms.",
            "wedge_1_1(A, B)",
        ),
        b(
            "exterior_d",
            "forms",
            "exterior_d(form[, coords])",
            "Exterior derivative of a differential form.",
            "exterior_d(A, [x, y])",
        ),
        b(
            "d",
            "forms",
            "d(form[, coords])",
            "Alias for exterior_d in the forms subsystem.",
            "d(A, [x, y])",
        ),
        b(
            "hodge_star",
            "forms",
            "hodge_star(form, metric)",
            "Hodge dual of a differential form.",
            "hodge_star(F, g)",
        ),
        b(
            "codifferential",
            "forms",
            "codifferential(form, metric, coords)",
            "Codifferential of a form via Hodge dual and exterior derivative.",
            "codifferential(A, g, [x, y])",
        ),
        b(
            "interior_product",
            "forms",
            "interior_product(vector, form)",
            "Interior product of a vector field with a differential form.",
            "interior_product([1, 0], F)",
        ),
        b(
            "lie_derivative_form",
            "forms",
            "lie_derivative_form(form, vector, coords)",
            "Lie derivative of a differential form via Cartan's formula.",
            "lie_derivative_form(A, [1, 0], [x, y])",
        ),
        b(
            "christoffel",
            "gr",
            "christoffel(metric, coords)",
            "Christoffel symbols from a metric.",
            "christoffel(g, [t,r,theta,phi])",
        ),
        b(
            "riemann",
            "gr",
            "riemann(christoffel, coords)",
            "Riemann tensor from a connection.",
            "riemann(Gamma, [t,r,theta,phi])",
        ),
        b(
            "ricci",
            "gr",
            "ricci(riemann)",
            "Ricci tensor from the Riemann tensor.",
            "ricci(R)",
        ),
        b(
            "ricci_scalar",
            "gr",
            "ricci_scalar(ricci, inverse_metric)",
            "Ricci scalar curvature.",
            "ricci_scalar(Ric, inv(g))",
        ),
        b(
            "einstein",
            "gr",
            "einstein(ricci, scalar, metric)",
            "Einstein tensor.",
            "einstein(Ric, R, g)",
        ),
        b(
            "weyl_from_curvature",
            "gr",
            "weyl_from_curvature(riemann, ricci, scalar, metric)",
            "Dimension-generic component Weyl tensor from curvature inputs: identically zero for n <= 3 and the standard n-dimensional decomposition for n >= 4, distinct from the abstract property declaration `weyl_tensor(C)`.",
            "weyl_from_curvature(R, Ric, Scal, g)",
        ),
        b(
            "weyl_from_riemann",
            "gr",
            "weyl_from_riemann(riemann, ricci, scalar, metric)",
            "Alias for weyl_from_curvature(riemann, ricci, scalar, metric).",
            "weyl_from_riemann(R, Ric, Scal, g)",
        ),
        b(
            "cotton_from_curvature",
            "gr",
            "cotton_from_curvature(ricci, scalar, gamma, metric, [coords...])",
            "Dimension-generic component Cotton tensor from Ricci, scalar curvature, Levi-Civita connection, metric, and coordinates, defined here for n >= 3.",
            "cotton_from_curvature(Ric, R, Gamma, g, [t, x, y])",
        ),
        b(
            "bach_from_curvature",
            "gr",
            "bach_from_curvature(weyl, ricci, gamma, metric, [coords...])",
            "Dimension-generic component Bach tensor from Weyl, Ricci, Levi-Civita connection, metric, and coordinates, defined here for n >= 4.",
            "bach_from_curvature(C, Ric, Gamma, g, [t, r, theta, phi])",
        ),
        b(
            "contorsion_tensor",
            "gr",
            "contorsion_tensor(T, g)",
            "Compute the contorsion tensor from a torsion tensor and metric.",
            "contorsion_tensor(T, g)",
        ),
        b(
            "connection_with_torsion",
            "gr",
            "connection_with_torsion(Gamma, K)",
            "Compose a torsionful connection from Levi-Civita Christoffel symbols and contorsion.",
            "connection_with_torsion(Gamma, K)",
        ),
        b(
            "spin_connection",
            "gr",
            "spin_connection(e, g, [coords...])",
            "Compute the torsion-free spin connection from a vielbein and metric.",
            "spin_connection(e, g, [t, r, theta, phi])",
        ),
        b(
            "first_cartan_structure",
            "gr",
            "first_cartan_structure(e, omega, [coords...])",
            "Compute the first Cartan structure equations T^a = de^a + omega^a_b wedge e^b as differential forms.",
            "first_cartan_structure(e, omega, [x, y])",
        ),
        b(
            "second_cartan_structure",
            "gr",
            "second_cartan_structure(omega, [coords...])",
            "Compute the second Cartan structure equations R^a_b = d omega^a_b + omega^a_c wedge omega^c_b as differential forms.",
            "second_cartan_structure(omega, [x, y])",
        ),
        b(
            "conformal_transform_metric",
            "gr",
            "conformal_transform_metric(metric, Omega)",
            "Conformally rescale a metric by Omega^2 as a component-level transformation.",
            "conformal_transform_metric(metric(diag(-1, 1, 1, 1)), a(t))",
        ),
        b(
            "conformal_transform_inverse_metric",
            "gr",
            "conformal_transform_inverse_metric(inverse_metric, Omega)",
            "Conformally rescale an inverse metric by Omega^-2 as a component-level transformation.",
            "conformal_transform_inverse_metric(inv(metric(diag(-1, 1, 1, 1))), a(t))",
        ),
        b(
            "conformal_transform_christoffel",
            "gr",
            "conformal_transform_christoffel(Gamma, g, Omega, [coords...])",
            "Transform Christoffel symbols under g_tilde = Omega^2 g using the component conformal-connection formula.",
            "conformal_transform_christoffel(Gamma, g, Omega, [t, x, y, z])",
        ),
        b(
            "conformal_transform_ricci",
            "gr",
            "conformal_transform_ricci(Ric, R, g, Omega, [coords...])",
            "Transform the Ricci tensor under g_tilde = Omega^2 g as a component-level curvature formula.",
            "conformal_transform_ricci(Ric, R, g, Omega, [t, x, y, z])",
        ),
        b(
            "conformal_transform_scalar",
            "gr",
            "conformal_transform_scalar(R, g, Omega, [coords...])",
            "Transform the scalar curvature under g_tilde = Omega^2 g as a component-level curvature formula.",
            "conformal_transform_scalar(R, g, Omega, [t, x, y, z])",
        ),
        b(
            "killing_equations",
            "gr",
            "killing_equations(Gamma, [coords...], xi?)",
            "Generate the symmetric Killing system for unknown covector components on a background connection.",
            "killing_equations(christoffel(metric(diag(-1, 1, 1, 1)), [t, x, y, z]), [t, x, y, z])",
        ),
        b(
            "adm_decompose",
            "gr",
            "adm_decompose(metric, [coords...], time_coord)",
            "Compute the ADM lapse, shift, spatial metric, extrinsic curvature, and constraints for a component metric in any dimension d >= 2 with one chosen time coordinate.",
            "adm_decompose(metric(diag(-1, 1, 1, 1)), [t, x, y, z], 0)",
        ),
        b(
            "null_tetrad",
            "gr",
            "null_tetrad(metric, [coords...])",
            "Auto-construct a Newman-Penrose null tetrad for a diagonal Lorentzian 4-metric.",
            "null_tetrad(metric(diag(-f(r), 1/f(r), r^2, r^2*sin(theta)^2)), [t, r, theta, phi])",
        ),
        b(
            "null_tetrad_from_metric",
            "gr",
            "null_tetrad_from_metric(metric, [coords...])",
            "Alias for null_tetrad(metric, [coords...]) using the tensor algorithm's public Rust function name.",
            "null_tetrad_from_metric(metric(diag(-f(r), 1/f(r), r^2, r^2*sin(theta)^2)), [t, r, theta, phi])",
        ),
        b(
            "verify_null_tetrad",
            "gr",
            "verify_null_tetrad(tetrad, metric)",
            "Verify NP null-tetrad normalization and orthogonality against a metric.",
            "verify_null_tetrad(T, g)",
        ),
        b(
            "spin_coefficients",
            "gr",
            "spin_coefficients(tetrad, Gamma, metric, [coords...])",
            "Compute the 12 Newman-Penrose spin coefficients from a null tetrad and Levi-Civita connection.",
            "spin_coefficients(T, Gamma, g, [t, r, theta, phi])",
        ),
        b(
            "weyl_scalars",
            "gr",
            "weyl_scalars(C, tetrad, metric)",
            "Compute the five Newman-Penrose Weyl scalars from a Weyl tensor and null tetrad.",
            "weyl_scalars(C, T, g)",
        ),
        b(
            "petrov_classify",
            "gr",
            "petrov_classify(weyl_scalars(...))",
            "Classify the Weyl tensor algebraically from its Newman-Penrose scalars.",
            "petrov_classify(weyl_scalars(C, T, g))",
        ),
        b(
            "kretschner",
            "gr",
            "kretschner(riemann, metric)",
            "Kretschmann scalar.",
            "kretschner(R, g)",
        ),
        b(
            "covariant_diff",
            "gr",
            "covariant_diff(expr, christoffel, coord_index, coords)",
            "Covariant derivative.",
            "covariant_diff(V, Gamma, 0, [t,r,theta,phi])",
        ),
        b(
            "geodesic",
            "gr",
            "geodesic(christoffel, coords)",
            "Geodesic equations from a Levi-Civita connection.",
            "geodesic(Gamma, [t,r,theta,phi])",
        ),
        b(
            "lie_derivative",
            "gr",
            "lie_derivative(field, vector, coords)",
            "Lie derivative of a scalar or vector field.",
            "lie_derivative(T, V, [x,y,z])",
        ),
        b(
            "metric",
            "gr",
            "metric(diag(...))",
            "Construct a symbolic metric tensor from a diagonal form.",
            "metric(diag(-1, 1, 1, 1))",
        ),
        b(
            "vielbein",
            "gr",
            "vielbein([[...], ...])",
            "Construct a symbolic vielbein matrix.",
            "vielbein([[sqrt(f), 0], [0, 1/sqrt(f)]])",
        ),
        b(
            "inv_vielbein",
            "gr",
            "inv_vielbein(e)",
            "Construct the inverse vielbein matrix.",
            "inv_vielbein(e)",
        ),
        b(
            "metric_from_vielbein",
            "gr",
            "metric_from_vielbein(e, eta)",
            "Construct the coordinate-frame metric from a vielbein and frame metric.",
            "metric_from_vielbein(e, eta)",
        ),
        b(
            "vielbein_from_metric_diagonal",
            "gr",
            "vielbein_from_metric_diagonal(g, signature)",
            "Construct a diagonal vielbein from a diagonal metric and signature convention.",
            "vielbein_from_metric_diagonal(metric(diag(-(f), 1/f)), mostly_plus)",
        ),
        b(
            "diag",
            "linear-algebra",
            "diag(a, b, ...)",
            "Construct a diagonal matrix.",
            "diag(1, 2, 3)",
        ),
        b(
            "to_python",
            "codegen",
            "to_python(expr)",
            "Print Python code for an expression.",
            "to_python(sin(x)^2)",
        ),
        b(
            "to_rust",
            "codegen",
            "to_rust(expr)",
            "Print Rust code for an expression.",
            "to_rust(sin(x)^2)",
        ),
        b(
            "to_cpp",
            "codegen",
            "to_cpp(expr)",
            "Print C++ code for an expression.",
            "to_cpp(sin(x)^2)",
        ),
    ]
}

pub fn property_entries() -> Vec<PropertyEntry> {
    vec![
        p("Symmetric", "property T symmetric([positions])", "Indices are symmetric under exchange of the listed slots.", "build_generating_set, canonicalize_indices, symmetry_tableaux_from_properties, handle_factor symmetry lookup", "property g symmetric"),
        p("AntiSymmetric", "property T antisymmetric([positions])", "Indices are antisymmetric under exchange of the listed slots.", "build_generating_set, canonicalize_indices, symmetry_tableaux_from_properties, handle_factor symmetry lookup", "property F antisymmetric"),
        p("RiemannSymmetry", "property R riemann_symmetry", "Apply the standard pair antisymmetry and pair-exchange symmetry of a Riemann tensor.", "build_generating_set, canonicalize_indices, symmetry_tableaux_from_properties, handle_factor symmetry lookup", "property R riemann_symmetry"),
        p("RiemannTensor", "property R riemann_tensor", "Composite abstract-GR declaration attaching RiemannSymmetry plus SatisfiesBianchi([0,1,2,3]).", "public declaration layer for meld, tensor_reduce, young_project_tensor, and inherited covariant-derivative differential-Bianchi projection", "property R riemann_tensor"),
        p("Traceless", "property T traceless", "Marks a tensor as traceless.", "canonicalise/canonicalize_indices fast-zero trace detection", "property T traceless"),
        p("Diagonal", "property D diagonal", "Marks a tensor as diagonal in numerical components.", "canonicalise/canonicalize_indices numerical diagonal fast-zero detection", "property D diagonal"),
        p("Trace", "property Tr trace", "Marks a call symbol as a trace wrapper over implicit-index products.", "explicit_indices trace wrapper handling", "property Tr trace"),
        p("Metric", "property g metric", "Marks a tensor as a metric used to lower indices and define dummy-pair symmetry.", "canonicalise slot symmetry, property-aware metric contraction, lower_free_indices, eliminate_metric", "property g metric"),
        p("InverseMetric", "property g inverse_metric", "Marks a tensor as an inverse metric used to raise indices and define dummy-pair symmetry.", "canonicalise slot symmetry, property-aware inverse-metric contraction, raise_free_indices, eliminate_metric", "property g inverse_metric"),
        p("KroneckerDelta", "property d kronecker_delta", "Marks a tensor as a Kronecker delta.", "canonicalise slot symmetry, property-aware delta contraction, eliminate_kronecker, expand_delta, reduce_delta", "property delta kronecker_delta"),
        p("EpsilonTensor", "property eps epsilon_tensor", "Marks a tensor as a Levi-Civita epsilon tensor.", "canonicalise antisymmetric slot/sign handling, epsilon_to_delta, handle_epsilon component evaluation", "property epsilon epsilon_tensor"),
        p("Derivative", "property D derivative", "Marks a symbol as a derivative operator.", "canonicalise dummy-boundary classification, sort_product noncommuting barrier, component derivative handling", "property D derivative"),
        p("PartialDerivative", "property D partial_derivative", "Marks a symbol as a partial derivative operator.", "canonicalise dummy-boundary classification, sort_product noncommuting barrier, component derivative handling", "property partial partial_derivative"),
        p("CovariantDerivative", "property nabla covariant_derivative", "Marks a symbol as a covariant derivative operator.", "canonicalise dummy-boundary classification and sort_product noncommuting barrier", "property nabla covariant_derivative"),
        p("TableauInherit", "property nabla tableau_inherit", "Marks a derivative-like head as inheriting tableau-style symmetry metadata from the immediately following tensor factor, with inherited slot numbers shifted by the derivative-slot count.", "composite derivative*tensor projector lookup for inherited TableauSymmetry and SatisfiesBianchi metadata", "property nabla tableau_inherit"),
        p("Depends", "depends T [x, y, ...]", "Declares that a tensor depends on listed symbols.", "stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it", "depends phi [x, t]"),
        p("Spinor", "property psi spinor", "Marks a tensor as carrying spinor indices.", "canonicalise_product dummy classification via metric_symmetry_for_slots", "property psi spinor"),
        p("SpinorMeta", "declare_spinor_meta(psi, dim, class, chirality, family)", "Attaches structured spinor metadata including representation class, optional dimension, chirality, and index family.", "stored by the evaluator property store; automatically also attaches compatible legacy spinor markers", "declare_spinor_meta(psi, 4, Majorana, none, spin)"),
        p("DiracBar", "property psibar dirac_bar", "Marks a symbol as a Dirac-bar object.", "canonicalize_indices local argument canonicalisation, sort_product barred-bilinear normalization/barrier handling, and ax-qm DiracBar expansion/sorting", "property psibar dirac_bar"),
        p("DiracBarMeta", "declare_dirac_bar_meta(psibar, gamma_symbol, family, reverse_gamma_order)", "Attaches structured Dirac-bar metadata linking a bar symbol to its gamma family and spinor family.", "stored by the evaluator property store; automatically also attaches the legacy DiracBar marker", "declare_dirac_bar_meta(psibar, gamma, spin, true)"),
        p("GammaMatrixProp", "property gamma gamma_matrix", "Marks a symbol as a gamma matrix.", "canonicalize_indices antisymmetric gamma-call slots, sort_product local barred-bilinear gamma placement/barrier handling, and ax-qm gamma algorithms", "property gamma gamma_matrix"),
        p("GammaMatrixMeta", "declare_gamma_matrix_meta(gamma, dim, metric, family, has_gamma5)", "Attaches structured gamma-matrix metadata including Clifford dimension, metric symbol, family, and gamma5 support.", "stored by the evaluator property store; automatically also attaches the legacy GammaMatrixProp marker", "declare_gamma_matrix_meta(gamma, 4, eta, spin, true)"),
        p("GammaConventionMeta", "declare_gamma_convention(gamma, signature, clifford, dimension) or declare_gamma5_convention(gamma, signature, clifford, gamma5_kind, epsilon, dimension)", "Attaches structured gamma-matrix convention metadata including metric signature, Clifford sign, optional gamma5 convention, epsilon symbol, and dimension.", "stored by the evaluator property store for convention-aware Clifford and gamma5 algorithms", "declare_gamma5_convention(gamma, mostly_plus, plus_two_g, levi_civita, epsilon, 4)"),
        p("Commuting", "property A commuting", "Marks an object as commuting.", "stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it", "property A commuting"),
        p("AntiCommuting", "property psi anticommuting", "Marks an object as anticommuting.", "canonicalise signed dummy/factor exchange and sort_product graded sign handling", "property psi anticommuting"),
        p("NonCommuting", "property A noncommuting", "Marks an object as noncommuting.", "canonicalise/sort_product factor-order barrier; identical noncommuting tensor factors are not exchanged", "property A noncommuting"),
        p("CommutingWith", "property A CommutingWith(B)", "Marks an object as explicitly commuting with listed objects.", "sort_product pairwise commutativity lookup", "property A CommutingWith(B)"),
        p("AntiCommutingWith", "property psi AntiCommutingWith(chi)", "Marks an object as explicitly anticommuting with listed objects.", "sort_product pairwise commutativity sign lookup", "property psi AntiCommutingWith(chi)"),
        p("NonCommutingWith", "property A NonCommutingWith(B)", "Marks an object as explicitly noncommuting with listed objects.", "sort_product pairwise commutativity barrier lookup", "property A NonCommutingWith(B)"),
        p("SelfAntiCommuting", "property gamma self_anticommuting", "Marks identical-head objects as anticommuting with themselves.", "sort_product self-commutation sign lookup", "property gamma self_anticommuting"),
        p("SelfNonCommuting", "property X self_noncommuting", "Marks identical-head objects as noncommuting with themselves.", "sort_product self-commutation barrier lookup", "property X self_noncommuting"),
        p("SelfCommuting", "property X self_commuting", "Marks identical-head objects as commuting with themselves.", "sort_product self-commutation lookup", "property X self_commuting"),
        p("CommutingAsProduct", "property F commuting_as_product", "Determines commutativity by checking product-like components pairwise.", "sort_product product-like commutativity lookup", "property F commuting_as_product"),
        p("CommutingAsSum", "property F commuting_as_sum", "Determines commutativity by checking sum-like components termwise.", "sort_product sum-like commutativity lookup", "property F commuting_as_sum"),
        p("MajoranaSpinor", "property psi majorana_spinor", "Marks a spinor as Majorana.", "stored by ax-tensor metadata", "property psi majorana_spinor"),
        p("WeylSpinor", "property psi weyl_spinor", "Marks a spinor as Weyl.", "stored by ax-tensor metadata", "property psi weyl_spinor"),
        p("ImplicitIndex", "property T implicit_index", "Marks an object as carrying implicit indices.", "sort_product implicit-index commutativity barrier lookup", "property T implicit_index"),
        p("SortOrder", "property T sort_order([...])", "Declares an explicit preferred order of symbols.", "stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it", "property T sort_order([A, B, C])"),
        p("TableauSymmetry", "property T tableau_symmetry([shape], [indices])", "Declares a Young-tableau symmetry shape and slot assignment.", "canonicalise slot/sign handling, meld symmetry_tableaux_from_properties, young_project_tensor", "property T tableau_symmetry([2,1], [0,1,2])"),
        p("SatisfiesBianchi", "property R satisfies_bianchi([0,1,2,3])", "Marks a tensor as satisfying a cyclic Bianchi identity on the specified three or four slots.", "young_project_tensor, meld, and symmetry projector lookup", "property R satisfies_bianchi([0,1,2,3])"),
        p("DimensionDependentIdentity", "property T dimension_dependent_identity", "Marks a tensor as carrying dimension-dependent identities relevant to Schouten-style reductions.", "dimension-reduction metadata and public property inspection", "property T dimension_dependent_identity"),
        p("WeylTensor", "property C weyl_tensor", "Marks a tensor as a Weyl tensor, i.e. RiemannSymmetry plus SatisfiesBianchi plus tracelessness.", "canonicalise Riemann-like slot symmetries, traceless fast-zero handling, and meld/young-project Bianchi hooks", "property C weyl_tensor"),
        p("DifferentialFormDegree", "property F differential_form_degree(n)", "Declares the degree of a differential form.", "stored by ax-tensor metadata; differential-form algorithms live outside ax-tensor", "property F differential_form_degree(2)"),
        p("TraceSpaceMeta", "declare_trace_space(Tr, space_symbol, cyclic)", "Attaches structured metadata describing the trace space represented by a trace-like symbol.", "stored by the evaluator property store for help, inspection, and future space-aware trace algorithms", "declare_trace_space(Tr, color, true)"),
        p("HilbertSpaceMeta", "declare_hilbert_space(H, 2) or declare_composite_space(HAB, [HA, HB])", "Attaches structured metadata describing a finite-dimensional Hilbert space and its ordered tensor-product factors.", "stored by the evaluator property store for quantum-space declarations and future space-aware algorithms", "declare_composite_space(HAB, [HA, HB])"),
        p("FockSpaceMeta", "declare_fock_space(F, [a0, a1])", "Attaches structured metadata describing a Fock space, its declared ordered modes, and the explicit occupation-basis ordering.", "stored by the evaluator property store for Fock-space declarations and basis-state validation", "declare_fock_space(F, [a0, a1])"),
        p("QuantumObjectMeta", "declare_quantum_object(A, operator, H)", "Attaches structured metadata describing a quantum object's kind and Hilbert space.", "stored by the evaluator property store; operator-like kinds also attach the legacy NonCommuting marker", "declare_quantum_object(rho, density_operator, H)"),
        p("OperatorSpaceMeta", "declare_operator_space(U, HA, HB)", "Attaches structured metadata describing an operator's domain and codomain Hilbert spaces.", "used by symbolic operator composition, dagger, tensor-product, and subsystem-aware operator-domain checks", "declare_operator_space(U, HA, HB)"),
        p("ModeMeta", "declare_mode(a, bosonic, 0), declare_mode_in_subsystem(a0, bosonic, QA, 0), or declare_mode_with_label(mode0, fermionic, reg, 0, a)", "Attaches structured mode metadata describing statistics, optional subsystem grouping, zero-based canonical mode index, and an optional symbolic label.", "stored by the evaluator property store; bosonic and spin modes also attach legacy NonCommuting, while fermionic modes also attach legacy AntiCommuting", "declare_mode_with_label(mode0, fermionic, reg, 0, a)"),
    ]
}

pub fn algorithm_entries() -> Vec<AlgorithmEntry> {
    vec![
        a("lower_free_indices", "tensor", "lower_free_indices(expr: &Expr, index_to_family: &HashMap<Spur, Spur>, index_families: &HashMap<Spur, IndexFamily>, interner: &Interner) -> Expr", "Flip free upper indices to lower variance without inserting an explicit metric.", "Index families should be declared when only some families are free-position indices; otherwise all singly-occurring upper indices are lowered.", "lower_indices(V[mu+])"),
        a("raise_free_indices", "tensor", "raise_free_indices(expr: &Expr, index_to_family: &HashMap<Spur, Spur>, index_families: &HashMap<Spur, IndexFamily>, interner: &Interner) -> Expr", "Flip free lower indices to upper variance without inserting an explicit inverse metric.", "Index families should be declared when only some families are free-position indices; otherwise all singly-occurring lower indices are raised.", "raise_indices(V[mu-])"),
        a("canonicalise", "tensor", "canonicalise(expr: &Expr, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Canonicalize tensor monomials and sums using declared slot symmetries, property-aware metric/delta contraction, epsilon signs, tableau symmetries, and dummy-index canonicalization.", "Tensor properties must be present in tensor_properties for anything beyond lexicographic index ordering.", "canonicalise(R[a-,b-,c-,d-] + R[a-,c-,d-,b-])"),
        a("meld", "tensor", "meld(expr: &Expr, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Detect multi-term tensor cancellations by property-aware canonicalization, Young projection, and rational linear dependence testing.", "Best results require symmetry properties such as Symmetric, AntiSymmetric, RiemannSymmetry, TableauSymmetry, Metric, KroneckerDelta, or EpsilonTensor on the factors involved.", "meld(R[a-,b-,c-,d-] + R[a-,c-,d-,b-] + R[a-,d-,b-,c-])"),
        a("evaluate_components", "tensor", "evaluate_components<E: ComponentEvalEnv>(expr: &Expr, rules: &[ComponentRule], index_values: &HashMap<Spur, Vec<Spur>>, env: &E, interner: &Interner) -> Expr", "Evaluate tensor expressions into explicit component expressions, including dummy summations, derivative handling, symmetry-aware lookups, and epsilon components.", "Concrete component rules and coordinates must be available through the evaluation environment; tensor_properties are used for symmetry and epsilon handling.", "evaluate(g[mu-,nu-], [[g,[t,t],-1],[g,[r,r],1]])"),
        a("sort_product", "tensor", "sort_product(expr: &Expr, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Sort multiplicative factors into deterministic local regions with graded signs, noncommuting/derivative/gamma/DiracBar barriers, and local barred-bilinear gamma placement.", "Tensor properties determine anticommuting signs, barred spinor/gamma windows, and which factors cannot be crossed.", "sort_product(B[a-] * A[a-])"),
        a("product_rule", "tensor", "product_rule(expr: &Expr, derivative_syms: &HashSet<Spur>, interner: &Interner) -> Expr", "Expand derivative operators over products and sums using the Leibniz rule.", "The derivative operator symbols must be listed in derivative_syms.", "leibniz(partial(A*B))"),
        a("tensor_distribute", "tensor", "tensor_distribute(expr: &Expr, interner: &Interner) -> Expr", "Distribute tensor products over sums, including sums that appear in indexed bases.", "No extra setup is required.", "tdistribute(A*(B + C))"),
        a("epsilon_to_delta", "tensor", "epsilon_to_delta(expr: &Expr, epsilon_sym: Spur, delta_sym: Spur, dim: usize, interner: &Interner) -> Expr", "Rewrite products of epsilon tensors into factorial factors times generalized Kronecker deltas.", "The epsilon symbol and target delta symbol must be provided, and epsilon factors must carry exactly dim indices.", "epsilon_to_delta(epsilon[a-,b-,c-] * epsilon[a+,d+,e+])"),
        a("expand_delta", "tensor", "expand_delta(expr: &Expr, delta_sym: Spur, interner: &Interner) -> Expr", "Expand a generalized Kronecker delta into a signed sum of ordinary two-index deltas.", "The delta symbol must identify an indexed tensor with an even number of slots split into equal up/down sets.", "expand_delta(Delta[a+,b+,c-,d-])"),
        a("symmetrise", "tensor", "symmetrise(expr: &Expr, positions: &[usize], antisymmetric: bool, interner: &Interner) -> Expr", "Symmetrize or antisymmetrize an expression over specific index slots by averaging over permutations.", "The listed positions must refer to valid index slots in the target indexed factor or flattened product ordering.", "symmetrize(T[a-,b-], [0,1])"),
        a("canonicalize_indices", "tensor", "canonicalize_indices(expr: &Expr, properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Apply local index-slot canonicalization from declared tensor properties before product-level canonicalization.", "Useful properties such as Symmetric, AntiSymmetric, RiemannSymmetry, WeylTensor, Metric, InverseMetric, KroneckerDelta, EpsilonTensor, GammaMatrixProp, DiracBar, or TableauSymmetry must be declared on tensor symbols.", "canonicalize_indices(F[b-,a-])"),
        a("rename_dummies", "tensor", "rename_dummies<E: DummyRenameEnv>(expr: &Expr, env: &E, interner: &Interner) -> Expr", "Rename dummy indices to deterministic family-aware placeholders so alpha-equivalent contractions compare equal.", "Index-family data improves the generated names; without it, generic _dN names are used.", "rename_dummies(T[a-,a+] + T[b-,b+])"),
        a("young_project", "tensor", "young_project(expr: &Expr, tableau: &YoungTableau, interner: &Interner) -> Expr OR young_project(expr: &Expr, modulo_monoterm: bool=true, canonicalize_after: bool=true, rename_dummies_after: bool=true) -> Expr", "Either apply an explicit Young tableau given as a nested slot list like [[0,1],[2]], or project using the tensor's declared symmetry properties with optional monoterm simplification.", "For the explicit form, supply a valid tableau cell layout in slot-number form. For the property-driven form, the tensor should carry TableauSymmetry, RiemannSymmetry, SatisfiesBianchi, or WeylTensor metadata.", "young_project(T[a-,b-,c-], [[0,1],[2]])"),
        a("young_project_tensor", "tensor", "young_project_tensor_with_options(expr: &Expr, properties: &dyn PropertyLookup, interner: &Interner, opts: &YoungProjectTensorOptions) -> Expr", "Apply declared Young-tableau symmetry and optionally simplify modulo monoterm symmetries by distributing, canonicalizing slots/products, renaming dummies, and collecting duplicates.", "The relevant tensor symbol must carry TableauSymmetry, RiemannSymmetry, SatisfiesBianchi, or WeylTensor properties; enabling modulo_monoterm is most useful when ordinary monoterm symmetries are also declared.", "young_project_tensor(T[a-,b-,c-], true, true, true)"),
        a("tensor_reduce", "tensor", "tensor_reduce(expr: &Expr, properties: &dyn PropertyLookup, interner: &Interner, opts: &TensorReduceOptions) -> Expr", "Run the finished tensor-reduction pipeline in order: monoterm canonicalisation, Cadabra-style multi-term Young projection on products, dimension-dependent reduction, dummy renaming, and optional meld.", "The expression should carry the relevant tensor properties for each enabled phase; dimension-dependent reduction additionally needs DimensionDependentIdentity plus inferable index-family dimension metadata.", "tensor_reduce(R[a-,b-,c-,d-]*V[e-] + R[a-,c-,d-,b-]*V[e-] + R[a-,d-,b-,c-]*V[e-])"),
        a("abstract_tensor_reduce", "tensor", "abstract_tensor_reduce(expr: &Expr, properties: &dyn PropertyLookup, interner: &Interner, opts: &TensorReduceOptions) -> Expr", "User-facing abstract tensor reduction pipeline for declared symmetries and inherited covariant-derivative identities such as the second Bianchi identity of a declared Riemann tensor.", "Use riemann_tensor(...) or property R riemann_tensor together with covariant_derivative(...) for abstract GR workflows.", "abstract_tensor_reduce(nabla[mu-]*R[nu-,rho-,sigma-,lambda-] + nabla[nu-]*R[rho-,mu-,sigma-,lambda-] + nabla[rho-]*R[mu-,nu-,sigma-,lambda-])"),
        a("contracted_bianchi_reduce", "tensor", "contracted_bianchi_reduce(expr: &Expr, derivative_sym: Spur, ricci_sym: Spur, scalar_sym: Spur, einstein_sym: Option<Spur>, properties: &dyn PropertyLookup, interner: &Interner) -> Result<Expr, ContractedBianchiError>", "Reduce abstract contracted-Bianchi identities only. This reducer performs no metric insertion, no component computation, and complements rather than replaces `abstract_gr_reduce`.", "The derivative product must already contain an explicit up/down contracted dummy index; this routine rewrites only abstract local patterns and does not raise or lower indices.", "contracted_bianchi_reduce(nabla[a+]*Ric[a-,b-], nabla, Ric, R)"),
        a("riemann_to_ricci", "tensor", "riemann_to_ricci(expr: &Expr, ricci_sym: Spur, scalar_sym: Option<Spur>, properties: &dyn PropertyLookup, interner: &Interner) -> Result<Expr, AbstractCurvatureReduceError>", "Rewrite internal abstract Riemann contractions inside a single factor into Ricci or scalar-curvature factors. This is abstract/internal contraction only, performs no component computation, and does not contract indices across distinct factors in a product. Typical uses are riemann_to_ricci(R[a-,b-,a+,d-], Ric) and riemann_to_ricci(R[a-,b-,a+,b+], Ric, Scal).", "The target indexed factor must be Riemann-like under the current property lookup; cross-factor contractions remain the responsibility of canonicalise/tensor_reduce.", "riemann_to_ricci(R[a-,b-,a+,d-], Ric)"),
        a("reduce_delta", "tensor", "reduce_delta(expr: &Expr, delta_sym: Spur, dim_sym: Spur, interner: &Interner) -> Expr", "Iteratively contract products and traces of Kronecker deltas back to simpler delta or dimension factors.", "The delta symbol and the symbol representing the dimension must be supplied.", "reduce_delta(Delta[a+,b-] * Delta[b+,c-])"),
        a("eliminate_kronecker", "tensor", "eliminate_kronecker(expr: &Expr, delta_sym: Spur, interner: &Interner) -> Expr", "Use Kronecker deltas to substitute contracted indices and remove delta factors from products.", "The delta symbol must identify a two-index Kronecker delta with one up and one down slot.", "eliminate_kronecker(delta[mu+,nu-] * T[nu+,rho-])"),
        a("eliminate_metric", "tensor", "eliminate_metric(expr: &Expr, metric_sym: Spur, inv_metric_sym: Spur, interner: &Interner) -> Expr", "Use metric or inverse-metric factors to raise or lower contracted indices and remove those metric factors.", "Metric components must use two down indices and inverse-metric components two up indices.", "eliminate_metric(g[mu-,nu-] * V[nu+])"),
        a("eliminate_vielbein", "tensor", "eliminate_vielbein(expr: &Expr, vielbein_sym: Spur, inv_vielbein_sym: Spur, interner: &Interner) -> Expr", "Use vielbein or inverse-vielbein factors to convert contracted indices between two families and remove the conversion factors.", "Vielbein factors must appear as indexed two-slot tensors with one contractible index matching another factor.", "eliminate_vielbein(e[a-,mu-] * V[mu+])"),
        a("rewrite_indices_vielbein", "tensor", "rewrite_indices_vielbein(expr: &Expr, e_sym: Spur, e_inv_sym: Spur, from_family: Spur, to_family: Spur, interner: &Interner) -> Expr", "Insert vielbein or inverse-vielbein factors so indices are rewritten from one index family into another frame family.", "The expression should carry explicit index-family metadata; e is assumed to map `from_family` coordinate indices into `to_family` frame indices via e[a+,mu-], with the inverse map carried by e_inv[mu+,a-].", "rewrite_indices_vielbein(V[mu+], e, einv, spacetime, frame)"),
        a("christoffel_from_metric", "gr", "christoffel_from_metric(g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Vec<Vec<Vec<Expr>>>", "Compute Christoffel symbols from a symbolic metric by the standard Levi-Civita formula in any metric dimension.", "The metric must be square and coords.len() must equal g.dim; the routine uses the symbolic inverse of g and is dimension-generic.", "christoffel(metric(diag(-1, 1)), [t, r])"),
        a("riemann_from_christoffel", "gr", "riemann_from_christoffel(gamma: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner, convention: &Convention) -> Vec<Vec<Vec<Vec<Expr>>>>", "Compute the Riemann tensor from Christoffel symbols in any dimension, respecting the active sign convention.", "The connection array dimensions must match coords.len(); the Convention determines MTW versus Weinberg sign.", "riemann(Gamma, [t, r, theta, phi])"),
        a("ricci_from_riemann", "gr", "ricci_from_riemann(riemann: &[Vec<Vec<Vec<Expr>>>], n: usize, interner: &Interner, convention: &Convention) -> Vec<Vec<Expr>>", "Contract a Riemann tensor into the Ricci tensor using the configured Ricci-contraction convention in any dimension.", "n must match the tensor dimensions; the Convention selects first-third or first-fourth contraction.", "ricci(R)"),
        a("ricci_scalar", "gr", "ricci_scalar(ricci: &[Vec<Expr>], ginv: &SymbolicMatrix, interner: &Interner) -> Expr", "Contract the Ricci tensor with the inverse metric to obtain the scalar curvature in any dimension.", "The inverse metric dimension must match the Ricci tensor dimensions.", "ricci_scalar(ginv, Ric)"),
        a("einstein_tensor", "gr", "einstein_tensor(ricci: &[Vec<Expr>], scalar: &Expr, g: &SymbolicMatrix, interner: &Interner) -> Vec<Vec<Expr>>", "Build the Einstein tensor G_ab = R_ab - 1/2 g_ab R in any metric dimension.", "The metric dimension must match the Ricci tensor dimensions.", "einstein(g, Ric, R)"),
        a("weyl_from_curvature", "gr", "weyl_from_curvature(riemann: &[Vec<Vec<Vec<Expr>>>], ricci: &[Vec<Expr>], scalar: &Expr, g: &SymbolicMatrix, interner: &Interner) -> Result<Vec<Vec<Vec<Vec<Expr>>>>, WeylError>", "Compute the Weyl tensor as a component-computation algorithm from already-computed Riemann, Ricci, scalar curvature, and metric data. This is distinct from the abstract property declaration `weyl_tensor(C)`.", "The inputs must already be concrete component tensors with consistent dimensions. The algorithm is dimension-generic: it returns the identically zero Weyl tensor for n <= 3 and uses the standard n-dimensional decomposition for n >= 4.", "weyl_from_curvature(R, Ric, Scal, g)"),
        a("weyl_from_riemann", "gr", "weyl_from_curvature(riemann: &[Vec<Vec<Vec<Expr>>>], ricci: &[Vec<Expr>], scalar: &Expr, g: &SymbolicMatrix, interner: &Interner) -> Result<Vec<Vec<Vec<Vec<Expr>>>>, WeylError>", "Alias for the component-computation algorithm `weyl_from_curvature(...)`, distinct from the abstract property declaration `weyl_tensor(C)`.", "Use this alias when you want the same dimension-generic component Weyl computation under a name that emphasizes the Riemann-curvature input.", "weyl_from_riemann(R, Ric, Scal, g)"),
        a("cotton_from_curvature", "gr", "cotton_from_curvature(ricci: &[Vec<Expr>], scalar: &Expr, gamma: &[Vec<Vec<Expr>>], g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<Vec<Expr>>>, ConformalCurvatureError>", "Compute the Cotton tensor as a component algorithm from Ricci, scalar curvature, Christoffel symbols, metric, and coordinates.", "The Ricci tensor, connection, metric, and coordinate list must have consistent dimensions. This component-computation routine is dimension-generic for n >= 3 and rejects n < 3 exactly.", "cotton_from_curvature(Ric, R, Gamma, g, [t, x, y])"),
        a("bach_from_curvature", "gr", "bach_from_curvature(weyl: &[Vec<Vec<Vec<Expr>>>], ricci: &[Vec<Expr>], gamma: &[Vec<Vec<Expr>>], g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<Expr>>, ConformalCurvatureError>", "Compute the Bach tensor as a component algorithm from Weyl, Ricci, Christoffel symbols, metric, and coordinates.", "The Weyl tensor, Ricci tensor, connection, metric, and coordinate list must have consistent dimensions. This component-computation routine is dimension-generic for n >= 4 and rejects n < 4 exactly.", "bach_from_curvature(C, Ric, Gamma, g, [t, r, theta, phi])"),
        a("contorsion_tensor", "gr", "contorsion_tensor(torsion: &[Vec<Vec<Expr>>], g: &SymbolicMatrix, interner: &Interner) -> Result<Vec<Vec<Vec<Expr>>>, CartanError>", "Compute the contorsion tensor K^a_bc from a torsion tensor T^a_bc and the metric.", "The torsion tensor must be a consistently shaped rank-3 array and the metric dimension must match it.", "contorsion_tensor(T, g)"),
        a("connection_with_torsion", "gr", "connection_with_torsion(christoffel: &[Vec<Vec<Expr>>], contorsion: &[Vec<Vec<Expr>>], interner: &Interner) -> Result<Vec<Vec<Vec<Expr>>>, CartanError>", "Compose a torsionful affine connection from Levi-Civita Christoffel symbols and contorsion.", "Both rank-3 tensors must have the same dimension and consistent shape.", "connection_with_torsion(Gamma, K)"),
        a("spin_connection", "gr", "spin_connection(vielbein: &SymbolicMatrix, g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<Vec<Expr>>>, CartanError>", "Compute the torsion-free spin connection omega_mu^a_b from a vielbein and metric in any dimension.", "The vielbein must be square and invertible, the metric dimension must match it, and coords.len() must equal that dimension.", "spin_connection(e, g, [t, r, theta, phi])"),
        a("first_cartan_structure", "gr", "first_cartan_structure(vielbein: &SymbolicMatrix, spin_connection: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner) -> Result<Vec<DiffForm>, CartanError>", "Compute the first Cartan structure equations T^a = de^a + omega^a_b ∧ e^b as component differential forms in any dimension.", "This routine converts the coframe and spin connection into `ax_forms::DiffForm` values and reuses the forms wedge and exterior-derivative machinery.", "first_cartan_structure(e, omega, [x, y])"),
        a("second_cartan_structure", "gr", "second_cartan_structure(spin_connection: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<DiffForm>>, CartanError>", "Compute the second Cartan structure equations R^a_b = d omega^a_b + omega^a_c ∧ omega^c_b as component differential forms in any dimension.", "This routine reuses `ax_forms::DiffForm`, `wedge`, and `exterior_derivative` rather than duplicating form algebra in the tensor crate.", "second_cartan_structure(omega, [x, y])"),
        a("conformal_transform_metric", "gr", "conformal_transform_metric(g: &SymbolicMatrix, omega: &Expr, interner: &Interner) -> SymbolicMatrix", "Conformally rescale a metric by Omega^2 as a component-level transformation in any metric dimension.", "This routine acts directly on metric components and does not compute abstract tensor properties.", "conformal_transform_metric(metric(diag(-1,1,1,1)), a(t))"),
        a("conformal_transform_inverse_metric", "gr", "conformal_transform_inverse_metric(g_inv: &SymbolicMatrix, omega: &Expr, interner: &Interner) -> SymbolicMatrix", "Conformally rescale an inverse metric by Omega^-2 as a component-level transformation in any metric dimension.", "This routine acts directly on inverse-metric components and does not compute abstract tensor properties.", "conformal_transform_inverse_metric(inv(metric(diag(-1,1,1,1))), a(t))"),
        a("conformal_transform_christoffel", "gr", "conformal_transform_christoffel(gamma: &[Vec<Vec<Expr>>], g: &SymbolicMatrix, omega: &Expr, coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<Vec<Expr>>>, ConformalError>", "Transform Christoffel symbols under the component conformal rescaling g_tilde = Omega^2 g.", "The metric, Christoffel symbols, and coordinate list must have consistent dimensions; the formula is dimension-generic and uses the original metric to raise the conformal-gradient index.", "conformal_transform_christoffel(Gamma, g, Omega, [t, x, y, z])"),
        a("conformal_transform_ricci", "gr", "conformal_transform_ricci(ricci: &[Vec<Expr>], scalar: &Expr, g: &SymbolicMatrix, omega: &Expr, coords: &[Spur], interner: &Interner) -> Result<Vec<Vec<Expr>>, ConformalError>", "Transform the Ricci tensor under the component conformal rescaling g_tilde = Omega^2 g.", "The metric, Ricci tensor, and coordinate list must have consistent dimensions; the implementation is dimension-generic and builds the needed covariant derivatives from the original Levi-Civita connection.", "conformal_transform_ricci(Ric, R, g, Omega, [t, x, y, z])"),
        a("conformal_transform_scalar", "gr", "conformal_transform_scalar(scalar: &Expr, g: &SymbolicMatrix, omega: &Expr, coords: &[Spur], interner: &Interner) -> Result<Expr, ConformalError>", "Transform the scalar curvature under the component conformal rescaling g_tilde = Omega^2 g.", "The metric and coordinate list must have consistent dimensions; the implementation is dimension-generic and uses the original metric to build U_a, U^a, and ∇_a U^a.", "conformal_transform_scalar(R, g, Omega, [t, x, y, z])"),
        a("killing_equations", "gr", "killing_equations(gamma: &[Vec<Vec<Expr>>], coords: &[Spur], field_prefix: &str, interner: &Interner) -> Result<KillingSystem, KillingError>", "Generate the symmetric Killing system ∇_a ξ_b + ∇_b ξ_a = 0 for unknown covector components.", "The connection dimensions must match the coordinate list, and the routine returns the unknown covector components, the independent symmetric equations, and their slot-pair labels.", "killing_equations(Gamma, [t, r, theta, phi], k)"),
        a("adm_decompose", "gr", "adm_decompose(g: &SymbolicMatrix, coords: &[Spur], time_coord: usize, interner: &Interner) -> Result<ADMDecomposition, ADMError>", "Compute the ADM decomposition of a component metric into lapse, shift, spatial metric, spatial inverse metric, extrinsic curvature, and the Hamiltonian and momentum constraints.", "The metric must be square with dimension at least two, coords.len() must match the metric dimension, and time_coord must choose one valid coordinate slot. This implementation is dimension-generic for any d >= 2.", "adm_decompose(metric(diag(-1, 1, 1, 1)), [t, x, y, z], 0)"),
        a("spatial_christoffel", "gr", "spatial_christoffel(gamma_ij: &SymbolicMatrix, spatial_coords: &[Spur], interner: &Interner) -> Vec<Vec<Vec<Expr>>>", "Compute Christoffel symbols for the spatial metric induced by an ADM split in any spatial dimension.", "The spatial metric must be square and its dimension must match the spatial coordinate list length.", "adm_decompose(g, [t, r, theta, phi], 0)"),
        a("spatial_ricci_tensor", "gr", "spatial_ricci_tensor(gamma_ij: &SymbolicMatrix, spatial_coords: &[Spur], interner: &Interner) -> Vec<Vec<Expr>>", "Compute the Ricci tensor of the spatial metric appearing in an ADM decomposition in any spatial dimension.", "The spatial metric must be square and compatible with the supplied spatial coordinates.", "adm_decompose(g, [t, r, theta, phi], 0)"),
        a("spatial_ricci_scalar", "gr", "spatial_ricci_scalar(gamma_ij: &SymbolicMatrix, spatial_coords: &[Spur], interner: &Interner) -> Expr", "Compute the Ricci scalar of the spatial metric appearing in an ADM decomposition in any spatial dimension.", "The spatial metric must be square and compatible with the supplied spatial coordinates.", "adm_decompose(g, [t, r, theta, phi], 0)"),
        a("verify_null_tetrad", "gr", "verify_null_tetrad(tetrad: &NullTetrad, g: &SymbolicMatrix, interner: &Interner) -> Result<(), NewmanPenroseError>", "Verify that a Newman-Penrose tetrad is null, normalized, and mutually orthogonal with respect to the supplied metric.", "The tetrad vectors must be contravariant 4-vectors compatible with the metric dimension; Newman-Penrose support is intentionally restricted to 4D.", "verify_null_tetrad(T, g)"),
        a("null_tetrad_from_metric", "gr", "null_tetrad_from_metric(g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Result<NullTetrad, NewmanPenroseError>", "Auto-construct a Newman-Penrose null tetrad as a component algorithm from a diagonal Lorentzian 4-metric.", "Auto-construction is limited to diagonal Lorentzian 4-metrics; this is independent of the abstract property declaration system.", "null_tetrad(metric(diag(-f(r), 1/f(r), r^2, r^2*sin(theta)^2)), [t, r, theta, phi])"),
        a("spin_coefficients", "gr", "spin_coefficients(tetrad: &NullTetrad, gamma: &[Vec<Vec<Expr>>], g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Result<SpinCoefficients, NewmanPenroseError>", "Compute the 12 Newman-Penrose spin coefficients from a null tetrad, Levi-Civita connection, metric, and coordinates.", "The tetrad, connection, metric, and coordinate list must all be 4-dimensional and mutually compatible.", "spin_coefficients(T, Gamma, g, [t, r, theta, phi])"),
        a("weyl_scalars", "gr", "weyl_scalars(weyl: &[Vec<Vec<Vec<Expr>>>], tetrad: &NullTetrad, g: &SymbolicMatrix, interner: &Interner) -> Result<WeylScalars, NewmanPenroseError>", "Compute the Newman-Penrose Weyl scalars by contracting a component Weyl tensor with a null tetrad.", "The Weyl tensor, tetrad, and metric must all be 4-dimensional; this is a component algorithm distinct from abstract Weyl-property declarations.", "weyl_scalars(C, T, g)"),
        a("petrov_classify", "gr", "petrov_classify(scalars: &WeylScalars, interner: &Interner) -> Result<PetrovType, NewmanPenroseError>", "Classify the Weyl tensor algebraically from the exact vanishing pattern and invariants of the Newman-Penrose Weyl scalars.", "Classification requires exact simplification enough to decide the necessary zero and nonzero conditions.", "petrov_classify(weyl_scalars(C, T, g))"),
        a("kretschner_scalar", "gr", "kretschner_scalar(riemann: &[Vec<Vec<Vec<Expr>>>], g: &SymbolicMatrix, interner: &Interner) -> Expr", "Compute the full Kretschmann scalar by contracting two Riemann tensors with four inverse metrics.", "The metric must be invertible; Axioma keeps a Schwarzschild closed-form shortcut as an optimization, and also provides a separate kretschmann_scalar_diagonal_approx helper when a diagonal-only approximation is desired.", "kretschner(g, R)"),
        a("kretschmann_scalar_diagonal_approx", "gr", "kretschmann_scalar_diagonal_approx(riemann: &[Vec<Vec<Vec<Expr>>>], g: &SymbolicMatrix, interner: &Interner) -> Expr", "Compute the old diagonal-only approximation to the Kretschmann scalar by contracting only matching diagonal inverse-metric entries against squared Riemann components.", "This is only exact for diagonal metrics in bases where the contraction really reduces that way; for the physical invariant use kretschner_scalar instead.", "kretschmann_scalar_diagonal_approx(R, g)"),
        a("inverse_vielbein", "gr", "inverse_vielbein(e: &SymbolicMatrix, interner: &Interner) -> SymbolicMatrix", "Compute the inverse vielbein matrix symbolically.", "The vielbein matrix must be square and invertible.", "inv_vielbein(e)"),
        a("metric_from_vielbein", "gr", "metric_from_vielbein(e: &SymbolicMatrix, eta: &SymbolicMatrix, interner: &Interner) -> SymbolicMatrix", "Build g_{mu nu} = eta_{ab} e^a_mu e^b_nu from a vielbein and frame metric.", "Both matrices must be square with the same dimension.", "metric_from_vielbein(e, eta)"),
        a("vielbein_from_metric_diagonal", "gr", "vielbein_from_metric_diagonal(g: &SymbolicMatrix, signature: Signature, interner: &Interner) -> SymbolicMatrix", "Construct a diagonal vielbein whose frame metric is fixed by the chosen signature convention.", "This helper assumes the input metric is already diagonal in the chosen coordinates.", "vielbein_from_metric_diagonal(g, mostly_plus)"),
        a("covariant_derivative_vector", "gr", "covariant_derivative_vector(v: &[Expr], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute ∇_coord_index v for a contravariant vector field in any dimension.", "The vector length, connection dimensions, and coordinate list length must agree.", "covariant_diff(V, g, [t, r])"),
        a("covariant_derivative_covector", "gr", "covariant_derivative_covector(w: &[Expr], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute ∇_coord_index w for a covector field in any dimension.", "The covector length, connection dimensions, and coordinate list length must agree.", "covariant_diff(W, g, [t, r])"),
        a("geodesic_equations", "gr", "geodesic_equations(gamma: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Construct the geodesic equations ẍ^i = -Γ^i_jk ẋ^j ẋ^k in symbolic form in any dimension.", "Connection dimensions must match the coordinate list.", "geodesic(g, [t, r, theta, phi], lambda)"),
        a("lie_derivative_scalar", "gr", "lie_derivative_scalar(f: &Expr, v: &[Expr], coords: &[Spur], interner: &Interner) -> Expr", "Compute the Lie derivative of a scalar along a vector field.", "The vector field length must match the coordinate list length.", "lie_derivative(phi, V, [x, y, z])"),
        a("lie_derivative_vector", "gr", "lie_derivative_vector(w: &[Expr], v: &[Expr], coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute the Lie derivative of a vector field along another vector field.", "Both vectors must have the same length as coords.", "lie_derivative(W, V, [x, y, z])"),
        a("unwrap_derivatives", "tensor", "unwrap_derivatives(expr: &Expr, derivative_syms: &HashSet<Spur>, depends: &HashMap<Spur, Vec<Spur>>, interner: &Interner) -> Expr", "Pull factors that do not depend on the differentiation variables outside derivative operators, and kill derivatives of constants.", "Derivative symbols must be listed explicitly, and dependence information should be populated for symbols that are not constant.", "unwrap(diff(a * phi, x))"),
        a("integrate_by_parts", "tensor", "integrate_by_parts(expr: &Expr, away_from: Spur, derivative_syms: &HashSet<Spur>, interner: &Interner) -> Expr", "Perform one integration-by-parts rewrite by moving a derivative off the factor containing away_from.", "The expression should contain a derivative operator from derivative_syms acting on a factor that contains away_from; boundary terms are assumed to vanish.", "ibp(diff(phi, x) * psi, phi, partial, x)"),
        a("keep_weight", "tensor", "keep_weight(expr: &Expr, target_weight: i64, weights: &HashMap<(Spur, String), i64>, label: &str, interner: &Interner) -> Expr", "Keep only the terms whose computed symbolic weight equals target_weight.", "Weight assignments for symbols under the chosen label must be present when nonzero weights are required.", "keep_weight(expr, field, 1)"),
        a("drop_weight", "tensor", "drop_weight(expr: &Expr, target_weight: i64, weights: &HashMap<(Spur, String), i64>, label: &str, interner: &Interner) -> Expr", "Remove terms whose computed symbolic weight equals target_weight.", "Weight assignments for symbols under the chosen label must be present when nonzero weights are required.", "drop_weight(expr, field, 0)"),
        a("complete_inverse_metric", "tensor", "complete_inverse_metric(metric_rules: &[ComponentRule], metric_sym: Spur, inv_metric_sym: Spur, coordinates: &[Spur], interner: &Interner) -> Vec<ComponentRule>", "Construct inverse-metric component rules from metric component rules by symbolic matrix inversion.", "Metric component rules must define an invertible square metric over the supplied coordinate list.", "complete_inverse_metric(metric_rules, g, ginv, [t, r])"),
        a("einsteinify", "tensor", "einsteinify(expr: &Expr, metric_sym: Option<Spur>, interner: &Interner) -> Expr", "Fix repeated-index pairs that have the same variance by flipping one slot so Einstein summation becomes well-formed.", "Useful on products where the same abstract index appears twice with both slots up or both slots down.", "einsteinify(T[mu+,mu+])"),
        a("split_index", "tensor", "split_index(expr: &Expr, parent_indices: &[Spur], sub1_indices: &[Spur], sub2_indices: &[Spur], interner: &Interner) -> Expr", "Replace occurrences of a parent index family by sums over two sub-families.", "The parent index names to split must be listed, and each target subfamily list should be non-empty if it is intended to contribute terms.", "split_index(T[a-], [a], [i], [a0])"),
        a("expand_dummies", "tensor", "expand_dummies(expr: &Expr, coordinates: &[Spur], interner: &Interner) -> Expr", "Replace each dummy index pair by an explicit sum over the supplied coordinate labels.", "A coordinate list must be supplied; abstract dummy names not already in that list are expanded.", "expand_dummies(T[mu-,mu+])"),
        a("explicit_indices", "tensor", "explicit_indices(expr: &Expr, implicit_index_tensors: &HashSet<Spur>, available_indices: &[Spur], n_indices_per_tensor: &HashMap<Spur, usize>, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Insert explicit indices for implicit-index tensors by building deterministic contraction graph components inside products.", "Tensor ranks are read from n_indices_per_tensor or tensor properties; explicit slots are preserved, scalar barriers split disconnected graph components, and trace wrappers close graph edges.", "explicit_indices(A * B)"),
        a("rewrite_indices", "tensor", "rewrite_indices(expr: &Expr, target_tensors: &HashMap<Spur, Vec<Variance>>, metric_sym: Spur, inv_metric_sym: Spur, interner: &Interner) -> Expr", "Insert metric or inverse-metric factors so selected tensors end up with requested slot variances.", "Each target tensor must have a full desired-variance specification per slot, and metric symbols must be provided.", "rewrite_indices(T[a+], targets)"),
        a("decompose", "tensor", "decompose(expr: &Expr, basis: &[Expr], tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Express a tensor expression as a rational linear combination of a supplied canonical basis plus any residual unmatched terms.", "The basis should span the intended subspace, and tensor_properties should contain the symmetries needed for canonical matching.", "decompose(expr, [basis1, basis2])"),
        a("decompose_product", "tensor", "decompose_product(expr: &Expr, dim: usize, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Decompose indexed tensor products by associative Littlewood-Richardson tableau composition and Young projection.", "The input should be a product containing at least two indexed tensors with inferable shapes; TableauSymmetry, Symmetric, AntiSymmetric, RiemannSymmetry, and generic indexed slots drive shape inference, multiplicities are preserved, and unsupported or inconsistent shapes return a diagnostic expression. When dim is omitted in the public evaluator, Axioma now tries to infer it from index-family metadata.", "decompose_product(T[a-,b-] * S[c-,d-] * V[e-], 4)"),
        a("schouten_reduce", "tensor", "schouten_reduce(expr: &Expr, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Apply dimension-dependent Schouten-style tensor reduction by inferring a unique index-family dimension, decomposing products into Young irreps, discarding dimensionally forbidden shapes, then canonicalising and melding the result.", "At least one tensor in the expression should carry the DimensionDependentIdentity property, and the expression must carry enough index-family metadata to infer a unique dimension; ambiguous or missing dimensions return a diagnostic expression.", "schouten_reduce(A[a-]*B[b-]*C[c-] - A[a-]*B[c-]*C[b-] + A[b-]*B[c-]*C[a-] - A[b-]*B[a-]*C[c-] + A[c-]*B[a-]*C[b-] - A[c-]*B[b-]*C[a-])"),
        a("expand_implicit", "tensor", "expand_implicit(expr: &Expr, implicit_index_tensors: &HashSet<Spur>, available_indices: &[Spur], n_indices_per_tensor: &HashMap<Spur, usize>, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Recursively make implicit tensor contraction graphs explicit across sums, products, trace wrappers, and call arguments.", "Tensor ranks are read from n_indices_per_tensor or tensor properties; each sum branch receives disjoint fresh graph indices.", "expand_implicit(A * B + C * D)"),
        a("normal_order", "qm", "normal_order(expr: &Expr, operators: &HashMap<Spur, OperatorKind>, operator_statistics: &HashMap<Spur, OperatorStatistics>, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Reorder products of operators into graded normal order using declared creation/annihilation kinds while preferring structured mode metadata over legacy operator-statistics fallbacks.", "Operator kinds must be declared for raw operator symbols; structured ModeMeta metadata on modes takes precedence when inferring bosonic or fermionic swap signs.", "normal_order(annihilation(a) * creation(a))"),
        a("time_ordered", "qm", "time_ordered(expr: Expr, interner: &Interner) -> Expr", "Wrap an expression in the canonical symbolic time-ordering form `time_order(expr)` without performing any time-order expansion.", "This is a symbolic constructor only in the current evaluator surface.", "time_order(a * b)"),
        a("anti_time_ordered", "qm", "anti_time_ordered(expr: Expr, interner: &Interner) -> Expr", "Wrap an expression in the canonical symbolic anti-time-ordering form `anti_time_order(expr)` without performing any expansion.", "This is a symbolic constructor only in the current evaluator surface.", "anti_time_order(a * b)"),
        a("kubo_response_function", "qm", "kubo_response_function(a_op: Expr, b_op: Expr, rho0: Expr, t: Expr, interner: &Interner) -> Expr", "Construct the symbolic Kubo response function -i theta(t) trace(rho0 * commutator(A(t), B(0))).", "This is a symbolic constructor only in the current evaluator surface; it does not expand commutators or traces.", "kubo_response(A, B, rho0, t)"),
        a("susceptibility_fourier", "qm", "susceptibility_fourier(response: Expr, omega: Expr, interner: &Interner) -> Expr", "Construct the symbolic Fourier susceptibility integral integral(t, neg_inf, inf, exp(i*omega*t) * response).", "This is a symbolic constructor only in the current evaluator surface.", "susceptibility_fourier(chi_t, omega)"),
        a("projector_left", "qm", "projector_left(interner: &Interner) -> Expr", "Construct the canonical left chiral projector P_L = (1 - gamma5)/2.", "No setup required.", "projector_left()"),
        a("projector_right", "qm", "projector_right(interner: &Interner) -> Expr", "Construct the canonical right chiral projector P_R = (1 + gamma5)/2.", "No setup required.", "projector_right()"),
        a("simplify_chiral_projectors", "qm", "simplify_chiral_projectors(expr: &Expr, props: &dyn PropertyLookup, interner: &Interner) -> Expr", "Simplify chiral projector idempotence, orthogonality, completeness, and Weyl-spinor chirality actions.", "Weyl spinor action requires SpinorMeta metadata with explicit chirality.", "simplify_chiral(projector_left() * psi_L)"),
        a("simplify_spinor_bilinear_selection_rules", "qm", "simplify_spinor_bilinear_selection_rules(expr: &Expr, props: &dyn PropertyLookup, interner: &Interner) -> Expr", "Apply supported 4D metadata-driven Majorana and Weyl bilinear selection rules, simplifying proven forbidden bilinears to zero.", "Requires structured SpinorMeta and convention metadata; insufficient metadata leaves the expression unchanged.", "simplify_spinor_bilinears(bar(psi) * gamma(mu) * psi)"),
        a("insert_explicit_spinor_indices", "qm", "insert_explicit_spinor_indices(expr: &Expr, props: &dyn PropertyLookup, interner: &Interner) -> Expr", "Expand supported implicit spinor bilinears and gamma chains into explicit spinor-index contractions.", "Requires structured DiracBarMeta, SpinorMeta, GammaMatrixMeta, and a common spinor index family.", "insert_explicit_spinor_indices(bar(psi) * gamma(mu) * psi)"),
        a("remove_trivial_spinor_indices", "qm", "remove_trivial_spinor_indices(expr: &Expr, props: &dyn PropertyLookup, interner: &Interner) -> Expr", "Collapse unambiguous canonical spinor-index contractions back to implicit bilinear or gamma-chain form.", "Only contractions matching the bridge's canonical pattern are removed.", "remove_trivial_spinor_indices(insert_explicit_spinor_indices(bar(psi) * gamma(mu) * psi))"),
        a("sigma_matrix", "qm", "sigma_matrix(mu: Expr, nu: Expr, interner: &Interner) -> Expr", "Construct the canonical Lorentz-generator spin matrix basis element sigma(mu,nu).", "No setup required.", "sigma(mu, nu)"),
        a("sigma_to_gamma_commutator", "qm", "sigma_to_gamma_commutator(expr: &Expr, interner: &Interner) -> Expr", "Expand sigma(mu,nu) to (i/2)(gamma(mu) gamma(nu) - gamma(nu) gamma(mu)).", "The input should contain canonical sigma(mu,nu) calls.", "sigma_to_gamma(sigma(mu, nu))"),
        a("gamma_commutator_to_sigma", "qm", "gamma_commutator_to_sigma(expr: &Expr, interner: &Interner) -> Expr", "Convert exact gamma commutator calls or two-term product differences to -2i sigma(mu,nu).", "The input must match the exact canonical gamma commutator pattern.", "gamma_to_sigma(gamma(mu) * gamma(nu) - gamma(nu) * gamma(mu))"),
        a("displacement_operator_series", "qm", "displacement_operator_series(alpha: Expr, mode: Expr, order: usize, interner: &Interner) -> Expr", "Construct the truncated symbolic series for the bosonic displacement operator exp(alpha a† - conj(alpha) a).", "The truncation order must be a nonnegative integer; the result is returned as a raw symbolic series without simplification.", "displacement_series(alpha, a, 2)"),
        a("squeezing_operator_series", "qm", "squeezing_operator_series(zeta: Expr, mode: Expr, order: usize, interner: &Interner) -> Expr", "Construct the truncated symbolic series for the bosonic squeezing operator exp(1/2 (zeta a† a† - conj(zeta) a a)).", "The truncation order must be a nonnegative integer; the result is returned as a raw symbolic series without simplification.", "squeezing_series(zeta, a, 2)"),
        a("bch_expand", "qm", "bch_expand(a: Expr, b: Expr, order: usize, interner: &Interner) -> Expr", "Construct the finite-order symbolic Baker-Campbell-Hausdorff expansion through the requested truncation order.", "The truncation order must be a nonnegative integer; orders above four currently return the order-four truncation.", "bch(A, B, 4)"),
        a("simplify_ccr_car", "qm", "simplify_ccr_car_full(expr: &Expr, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Apply explicit same-mode and distinct-mode CCR/CAR rewrites to ladder-operator products until a fixed point.", "The expression should use creation(mode) and annihilation(mode) forms; structured ModeMeta metadata is used when available to distinguish same and distinct modes.", "simplify_ccr_car(annihilation(a) * creation(a))"),
        a("wick_expand", "qm", "wick_expand(expr: &Expr, operators: &HashMap<Spur, OperatorKind>, operator_statistics: &HashMap<Spur, OperatorStatistics>, properties: &dyn PropertyLookup, contractions: &HashMap<(Spur, Spur), Expr>, interner: &Interner) -> Expr", "Expand operator products into graded normal-ordered terms plus all declared Wick contraction patterns, with fermionic signs determined by the required operator swaps and pairing crossings.", "Operator kinds and any nonzero contraction values must be provided explicitly; structured ModeMeta metadata takes precedence when determining fermionic swap signs.", "wick(psi * psibar)"),
        a("gamma_trace", "qm", "gamma_trace(indices: &[GammaEntry], metric: &SymbolicMatrix, interner: &Interner) -> Expr", "Trace a gamma-matrix chain, including the special gamma5 epsilon-tensor case.", "The input must already be parsed into GammaEntry values; the implementation assumes the standard four-dimensional Dirac trace normalization.", "gamma_trace([gamma(mu), gamma(nu)])"),
        a("join_gammas_in_expr", "qm", "join_gammas_in_expr(expr: &Expr, gamma_sym: Spur, metric_sym: Spur, interner: &Interner) -> Expr", "Join adjacent gamma-matrix factors into antisymmetrized multi-index gamma objects plus metric contractions.", "Gamma factors must be represented as Call(gamma_sym, [...]) nodes and use a compatible metric symbol.", "join_gamma(gamma(mu) * gamma(nu))"),
        a("split_gamma", "qm", "split_gamma(expr: &Expr, gamma_sym: Spur, metric_sym: Spur, on_back: bool, interner: &Interner) -> Expr", "Split a multi-index antisymmetric gamma matrix into a shorter chain plus contraction terms.", "The input must contain gamma_sym calls with more than one index.", "split_gamma(gamma(mu, nu))"),
        a("expand_diracbar", "qm", "expand_diracbar(expr: &Expr, diracbar_sym: Spur, gamma_sym: Spur, metric_sym: Spur, interner: &Interner) -> Expr", "Expand Dirac bars through gamma-spinor products, reversing gamma chains and applying the Majorana transpose sign, including nested negative/barred subexpressions.", "The input should use diracbar_sym calls around products of gamma factors followed by a spinor; tensor canonicalisation also uses DiracBar/GammaMatrixProp for legal local bilinear normalization.", "expand_diracbar(bar(gamma(mu) * psi))"),
        a("diracbar_sort", "qm", "diracbar_sort(expr: &Expr, diracbar_sym: Spur, gamma_sym: Spur, operators: &HashMap<Spur, OperatorKind>, interner: &Interner) -> Expr", "Reorder products into barred-spinor gamma spinor bilinear form, matching the local normalization used by tensor sort_product where properties are declared.", "The routine groups factors following a DiracBar call into gamma factors and the next spinor-like factor.", "sort_spinors(bar(psi) * chi * gamma(mu))"),
        a("fierz", "qm", "fierz(expr: &Expr, dim: usize, interner: &Interner) -> Expr", "Perform a concrete Fierz rearrangement of two detected spinor bilinears, with automatic common-case spinor-order inference.", "The parser handles mixed and nested products, gamma chains, DiracBar-style barred spinors, graded anticommuting signs, and returns an explicit diagnostic expression for ambiguous or malformed inputs.", "fierz((psibar1 * gamma(mu) * psi2) * (psibar3 * psi4), 4)"),
        a("commutator", "qm", "commutator(a: &[Vec<Expr>], b: &[Vec<Expr>], interner: &Interner) -> Vec<Vec<Expr>>", "Compute the matrix commutator AB - BA.", "The matrices must be dimensionally compatible for multiplication.", "commutator(pauli_x(), pauli_y())"),
        a("anticommutator", "qm", "anticommutator(a: &[Vec<Expr>], b: &[Vec<Expr>], interner: &Interner) -> Vec<Vec<Expr>>", "Compute the matrix anticommutator AB + BA.", "The matrices must be dimensionally compatible for multiplication.", "anticommutator(pauli_x(), pauli_x())"),
        a("density_matrix", "qm", "density_matrix(state: &[Expr]) -> Vec<Vec<Expr>>", "Build the rank-one density matrix |psi><psi| from a state vector.", "The state should be given as a finite component vector.", "density([a, b])"),
        a("partial_trace", "qm", "partial_trace(rho: &[Vec<Expr>], dim_a: usize, dim_b: usize, trace_over: char, interner: &Interner) -> Vec<Vec<Expr>>", "Trace out subsystem A or B from a bipartite density matrix.", "rho must be arranged as a (dim_a*dim_b) square matrix, and trace_over must be 'A' or 'B'.", "partial_trace(rho, 2, 2, B)"),
        a("partial_trace_factor", "qm", "try_partial_trace_factor(rho: &[Vec<Expr>], factor_dims: &[usize], traced_factor: usize) -> Result<Vec<Vec<Expr>>, CompositeSpaceError>", "Trace out one factor from a general finite-dimensional tensor-product space while preserving the order of the remaining factors.", "rho must be square with dimension equal to the product of factor_dims, and traced_factor must be a valid factor index.", "partial_trace_factor(rho, [2, 2], 1)"),
        a("partial_transpose_factor", "qm", "try_partial_transpose_factor(rho: &[Vec<Expr>], factor_dims: &[usize], transposed_factor: usize) -> Result<Vec<Vec<Expr>>, CompositeSpaceError>", "Partially transpose one subsystem by swapping the chosen factor's bra and ket indices in lexicographic tensor-product order.", "rho must be square with dimension equal to the product of factor_dims, and transposed_factor must be a valid factor index.", "partial_transpose_factor(rho, [2, 2], 1)"),
        a("permute_subsystems", "qm", "try_permute_subsystems(rho: &[Vec<Expr>], factor_dims: &[usize], permutation: &[usize]) -> Result<Vec<Vec<Expr>>, CompositeSpaceError>", "Permute subsystem order by exact basis relabeling of both row and column tensor-product indices.", "rho must be square with dimension equal to the product of factor_dims, and permutation must contain each factor index exactly once.", "permute_subsystems(rho, [2, 2], [1, 0])"),
        a("renyi2_entropy_factor", "qm", "renyi2_entropy_factor(rho: &[Vec<Expr>], factor_dims: &[usize], kept_factor: usize, interner: &Interner) -> Result<Expr, CompositeSpaceError>", "Compute the Renyi-2 entropy of the reduced state obtained by keeping one tensor factor and tracing out the rest.", "rho must be square with dimension equal to the product of factor_dims, and kept_factor must be a valid factor index.", "renyi2_entropy_factor(rho, [2, 2], 0)"),
        a("von_neumann_mutual_information_bipartite", "qm", "von_neumann_mutual_information_bipartite(rho_ab: &[Vec<Expr>], dim_a: usize, dim_b: usize, interner: &Interner) -> Result<Expr, EntropyError>", "Compute bipartite von Neumann mutual information S(rho_A) + S(rho_B) - S(rho_AB).", "rho_ab must be a supported Hermitian bipartite density matrix of dimension dim_a * dim_b.", "mutual_information(rho, 2, 2)"),
        a("conditional_entropy_b_given_a", "qm", "conditional_entropy_b_given_a(rho_ab: &[Vec<Expr>], dim_a: usize, dim_b: usize, interner: &Interner) -> Result<Expr, EntropyError>", "Compute bipartite conditional entropy S(B|A) = S(rho_AB) - S(rho_A).", "rho_ab must be a supported Hermitian bipartite density matrix of dimension dim_a * dim_b.", "conditional_entropy(rho, 2, 2)"),
        a("renyi2_tripartite_information", "qm", "renyi2_tripartite_information(rho_abc: &[Vec<Expr>], dims: [usize; 3], interner: &Interner) -> Result<Expr, CompositeSpaceError>", "Compute tripartite Renyi-2 information S2(A) + S2(B) + S2(C) - S2(AB) - S2(AC) - S2(BC) + S2(ABC).", "rho_abc must be a square density matrix of dimension dim_a * dim_b * dim_c.", "renyi2_tripartite_information(rho, 2, 2, 2)"),
        a("expectation_value", "qm", "expectation_value(operator: &[Vec<Expr>], rho: &[Vec<Expr>]) -> Result<Expr, ObservableError>", "Compute the observable expectation value Tr(rho * operator) for a finite-dimensional density matrix.", "Both operator and rho must be square matrices of the same dimension.", "expectation_value(pauli_z(), density_matrix([1, 0]))"),
        a("variance", "qm", "variance(operator: &[Vec<Expr>], rho: &[Vec<Expr>]) -> Result<Expr, ObservableError>", "Compute the observable variance Tr(rho * operator^2) - Tr(rho * operator)^2 for a finite-dimensional density matrix.", "Both operator and rho must be square matrices of the same dimension.", "variance(pauli_z(), density_matrix([1, 0]))"),
        a("braket", "qm", "braket(bra: &[Expr], ket: &[Expr]) -> Expr", "Compute the inner product of a bra and ket by componentwise contraction.", "The two vectors should have the same length.", "braket([1, 0], [0, 1])"),
        a("wedge", "forms", "wedge(a: &DiffForm, b: &DiffForm, interner: &Interner) -> DiffForm", "Compute the antisymmetric wedge product of two differential forms.", "Both forms must have the same ambient dimension.", "wedge_1_1(A, B)"),
        a("exterior_derivative", "forms", "exterior_derivative(form: &DiffForm, coords: &[Spur], interner: &Interner) -> DiffForm", "Take the exterior derivative of a differential form by differentiating components and wedging in basis one-forms.", "form.dim must equal coords.len().", "exterior_d(A)"),
        a("hodge_dual", "forms", "hodge_dual(form: &DiffForm, g: &SymbolicMatrix, interner: &Interner) -> DiffForm", "Take the Hodge dual of a differential form with respect to a symbolic metric.", "The metric dimension must equal the form dimension; the implementation uses the symbolic inverse and determinant of g.", "hodge_star(F, g)"),
        a("functional_derivative", "variational", "functional_derivative(lagrangian: &Expr, field: Spur, field_derivs: &[Spur], coords: &[Spur], interner: &Interner) -> Expr", "Compute the Euler-Lagrange functional derivative δL/δfield for first-derivative Lagrangians.", "field_derivs and coords should be aligned so each derivative symbol corresponds to differentiation with respect to the matching coordinate.", "functional_derivative(L, phi, [phi_t, phi_x], [t, x])"),
        a("euler_lagrange_system", "variational", "euler_lagrange_system(lagrangian: &Expr, fields: &[(Spur, Vec<Spur>)], coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute the Euler-Lagrange equations for several fields at once.", "Each field entry must provide derivative symbols aligned with coords.", "euler_lagrange(L, [phi, chi], [t, x])"),
        a("vary_action", "variational", "vary_action(lagrangian: &Expr, field: Spur, variation: Spur, field_derivs: &[Spur], variation_derivs: &[Spur], interner: &Interner) -> Expr", "Form the first variation of an action density before integrating by parts.", "field_derivs and variation_derivs must be aligned term-by-term.", "vary(S, phi)"),
        a("differentiate", "calculus", "differentiate(expr: &Expr, var: Spur, interner: &Interner) -> Expr", "Take a symbolic derivative with chain, product, and builtin function rules.", "The differentiation variable must be a symbol.", "diff(sin(x^2), x)"),
        a("symbolic_substitute", "rewrite", "symbolic_substitute(expr: &Expr, target: &Expr, replacement: &Expr, interner: &Interner) -> Expr", "Replace exact symbolic subexpressions recursively.", "Best suited to scalar expressions without tensor-index matching requirements.", "subs(f(x), x, y)"),
        a("multi_substitute", "rewrite", "multi_substitute(expr: &Expr, substitutions: &[(Expr, Expr)], interner: &Interner) -> Expr", "Apply several exact substitutions in one pass.", "Targets are applied structurally rather than by solving matching ambiguities.", "subs(expr, [x, y], [a, b])"),
        a("match_tensor_pattern", "rewrite", "match_tensor_pattern(pattern: &Expr, expr: &Expr, env: &Env, interner: &Interner) -> Option<HashMap<Spur, Spur>>", "Match indexed tensor patterns using variance and index-family compatibility rather than literal index names.", "Index-family information in env improves matching across renamed abstract indices.", "subs(T[mu-,nu-], T[a-,b-], A[a-]*B[b-])"),
        a("substitute_with_indices", "rewrite", "substitute_with_indices(expr: &Expr, target: &Expr, replacement: &Expr, env: &Env, interner: &Interner) -> Expr", "Perform substitution while renaming bound dummy indices to avoid capture and preserving index-family matches.", "Use when the expression or rule contains indexed tensors.", "subs(T[mu-,nu-], T[a-,b-], A[a-]*B[b-])"),
        a("rewrite_with_trace", "rewrite", "rewrite_with_trace(expr: &Expr, env: &Env, interner: &Interner) -> (Expr, Vec<RewriteStep>)", "Apply registered rewrite rules and return both the rewritten expression and a trace of the applied rules.", "Rewrite rules must be registered in the environment.", "rewrite(expr)"),
        a("describe_rewrite_trace", "rewrite", "describe_rewrite_trace(trace: &[RewriteStep], interner: &Interner) -> String", "Render a human-readable summary of a rewrite trace.", "A trace from rewrite_with_trace is required.", "describe_rewrite_trace(trace)"),
        a("resolve_import", "syntax", "resolve_import(path: &str) -> Option<PathBuf>", "Resolve a std-module import path to the corresponding .ax file on disk.", "The imported module must exist under std/ or another supported search root.", "import std.gr.schwarzschild"),
        a("eval", "syntax", "eval(expr: &Expr, env: &Env, interner: &Interner) -> Expr", "Evaluate an expression by dispatching builtins, rewrite rules, declarations, and symbolic simplifications.", "Environment declarations, rules, coordinates, and tensor properties affect the result.", "simplify(expr)"),
        a("zoom", "rewrite", "zoom(expr: &Expr, pattern: &Expr, interner: &Interner) -> (Expr, Expr)", "Split a sum into matching and nonmatching parts with respect to a pattern.", "Most useful on additive expressions.", "zoom(a + b + c, a_)"),
        a("unzoom", "rewrite", "unzoom(focus: &Expr, remainder: &Expr, interner: &Interner) -> Expr", "Recombine a focused expression with its saved remainder.", "The focus and remainder should come from a compatible zoom step.", "unzoom(a + b, c)"),
        a("take_match", "rewrite", "take_match(expr: &Expr, pattern: &Expr, interner: &Interner) -> Expr", "Keep only the subterms of a sum that match a pattern.", "Most useful on additive expressions.", "take_match(a + b + c, a_)"),
        a("grassmann_simplify", "qm", "grassmann_simplify(expr: &Expr, gradings: &HashMap<Spur, Grading>, interner: &Interner) -> Expr", "Simplify products of commuting and anticommuting symbols using the stored gradings.", "Grassmann or operator gradings must be present in the environment.", "grassmann_simplify(theta*theta)"),
        a("solve", "solve", "solve(equation: &Expr, var: Spur, interner: &Interner) -> Expr", "Solve a univariate polynomial equation when its coefficients can be extracted.", "The equation must reduce to a polynomial in var; otherwise the function returns an unevaluated solve call.", "solve(x^3 - 6*x^2 + 11*x - 6, x)"),
        a("solve_linear_system", "solve", "solve_linear_system(equations: &[Expr], vars: &[Spur], interner: &Interner) -> Option<Vec<(Spur, Expr)>>", "Solve a linear system over exact rationals by Gaussian elimination.", "Every equation must be linear in the listed variables, and the system must have a unique consistent solution.", "solve([x + y - 3, x - y - 1], [x, y])"),
        a("classify_pde", "ode", "classify_pde(a: &Expr, b: &Expr, c: &Expr, interner: &Interner) -> PdeType", "Classify a second-order PDE from its A, B, C coefficients via the discriminant B^2 - A*C.", "The discriminant must simplify to a numeric sign to get a definite classification; otherwise the result is Unknown.", "classify_pde(1, 0, -1)"),
        a("separate_variables", "ode", "separate_variables(pde_type: PdeType, spatial_var: Spur, temporal_var: Spur, coefficient: &Expr, interner: &Interner) -> SeparatedSolution", "Return a standard separated-variables ansatz for hyperbolic, parabolic, or elliptic PDE families.", "This is a template generator for standard wave, heat, and Laplace-type equations rather than an automatic PDE parser.", "separate_variables(wave, x, t, c)"),
        a("solve_ode", "ode", "solve_ode(equation: &Expr, y_sym: Spur, x_sym: Spur, interner: &Interner) -> Expr", "Solve simple separable or first-order linear ODEs symbolically.", "The ODE right-hand side must match one of the supported separable or linear forms; otherwise an unevaluated solve_ode call is returned.", "dsolve(y - x, y, x)"),
        a("rk4", "ode", "rk4(f: &Expr, x_sym: Spur, y_sym: Spur, x0: f64, y0: f64, x_end: f64, n_steps: usize, interner: &Interner) -> Vec<(f64, f64)>", "Numerically integrate a scalar first-order ODE y' = f(x, y) with fourth-order Runge-Kutta.", "f must evaluate numerically for the supplied bindings, and n_steps must be nonzero.", "rk4(y, x, y, 0, 1, 1, 100)"),
        a("rk4_system", "ode", "rk4_system(fs: &[Expr], x_sym: Spur, y_syms: &[Spur], x0: f64, y0s: &[f64], x_end: f64, n_steps: usize, interner: &Interner) -> Vec<Vec<f64>>", "Numerically integrate a coupled first-order ODE system with fourth-order Runge-Kutta.", "The numbers of equations, dependent variables, and initial values must match, and each expression must evaluate numerically.", "rk4_system([y, -x], t, [x, y], 0, [1, 0], 10, 1000)"),
        a("parallel_transport", "gr", "parallel_transport(initial_vector: &[f64], curve: &[Vec<f64>], gamma_numeric: &dyn Fn(&[f64]) -> Vec<Vec<Vec<f64>>>) -> Result<Vec<Vec<f64>>, NumericalGRError>", "Numerically parallel-transport a contravariant vector along a discrete curve using the GR transport equation and the shared RK4 system integrator.", "This is currently a library-level/native-callback API rather than ordinary source syntax; the Christoffel callback must be provided from Rust, and the implementation is dimension-generic over the callback output shape.", "parallel_transport(initial_vector, curve, gamma_numeric)"),
        a("integrate_geodesic", "gr", "integrate_geodesic(gamma_numeric: &dyn Fn(&[f64]) -> Vec<Vec<Vec<f64>>>, initial_position: &[f64], initial_velocity: &[f64], tau_range: (f64, f64), n_steps: usize) -> Result<Vec<(f64, Vec<f64>, Vec<f64>)>, NumericalGRError>", "Numerically integrate the first-order geodesic system (x^mu, v^mu) using the shared RK4 system integrator.", "This is currently a library-level/native-callback API rather than ordinary source syntax; the Christoffel callback must be provided from Rust, and the implementation is dimension-generic over the callback output shape.", "integrate_geodesic(gamma_numeric, x0, v0, [0.0, 1.0], 1000)"),
        a("first_order_form", "ode", "first_order_form(ode: &Expr, dependent_var: Spur, independent_var: Spur, interner: &Interner) -> Vec<(Expr, Expr)>", "Convert a higher-order ODE into a first-order system by introducing auxiliary derivative variables.", "The ODE should contain nested diff calls with respect to independent_var, or else it is treated as the right-hand side of a second-order equation.", "first_order_form(diff(diff(x,t),t) + x, x, t)"),
        a("evaluate_components_v2", "tensor", "evaluate_components_v2(expr: &Expr, rules: &[ComponentRule], env: &dyn ComponentEvalEnv, interner: &Interner) -> Expr", "Evaluate tensor component algebra across sums, products, traces, derivatives, deltas, epsilon tensors, metrics, inverse metrics, and symmetry-aware sparse rules.", "Component rules, coordinates, and tensor properties must be available through env; dummy contractions are assigned before lookup, missing sparse components evaluate to zero, and generated inverse-metric components are collected with downstream terms.", "evaluate(g[mu-,nu-] * ginv[nu+,mu+], rules)"),
        a("rename_dummy_indices", "tensor", "rename_dummy_indices(expr: &Expr, prefix: &str, interner: &Interner) -> Expr", "Rename repeated contracted indices to fresh deterministic names with the chosen prefix.", "Useful when preparing expressions for display or comparison.", "rename_dummy_indices(T[a-,a+], d)"),
        a("diff_component", "tensor", "diff_component(expr: &Expr, var: Spur, interner: &Interner) -> Expr", "Differentiate a component expression with tensor-aware fallback handling.", "The variable should be a coordinate or scalar symbol.", "diff_component(r^2, r)"),
        a("covariant_derivative_tensor2", "gr", "covariant_derivative_tensor2(t: &[Vec<Expr>], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Vec<Expr>>", "Compute the covariant derivative of a rank-2 covariant tensor.", "Tensor dimensions, connection dimensions, and coordinate count must agree.", "covariant_diff(T, Gamma, 0, [t, r])"),
        a("compute_weight", "tensor", "compute_weight(expr: &Expr, weights: &HashMap<(Spur, String), i64>, label: &str) -> i64", "Compute the total symbolic weight of an expression under a chosen label.", "Weight assignments should be declared for the participating symbols.", "compute_weight(expr, weights, field)"),
        a("pauli_x", "qm", "pauli_x(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_x matrix.", "No extra setup is required.", "pauli_x()"),
        a("pauli_y", "qm", "pauli_y(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_y matrix.", "No extra setup is required.", "pauli_y()"),
        a("pauli_z", "qm", "pauli_z(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_z matrix.", "No extra setup is required.", "pauli_z()"),
        a("jz_matrix", "qm", "jz_matrix(two_j: usize, interner: &Interner) -> Result<Vec<Vec<Expr>>, SpinError>", "Construct the exact spin-j J_z matrix in the standard m = j, j-1, ..., -j basis.", "The argument two_j must be a nonnegative integer representing 2j.", "jz(1)"),
        a("jplus_matrix", "qm", "jplus_matrix(two_j: usize, interner: &Interner) -> Result<Vec<Vec<Expr>>, SpinError>", "Construct the exact spin-j raising-operator matrix J_+ in the standard m = j, j-1, ..., -j basis.", "The argument two_j must be a nonnegative integer representing 2j.", "jplus(2)"),
        a("jminus_matrix", "qm", "jminus_matrix(two_j: usize, interner: &Interner) -> Result<Vec<Vec<Expr>>, SpinError>", "Construct the exact spin-j lowering-operator matrix J_- in the standard m = j, j-1, ..., -j basis.", "The argument two_j must be a nonnegative integer representing 2j.", "jminus(2)"),
        a("jx_matrix", "qm", "jx_matrix(two_j: usize, interner: &Interner) -> Result<Vec<Vec<Expr>>, SpinError>", "Construct the exact spin-j Cartesian operator J_x = (J_+ + J_-)/2.", "The argument two_j must be a nonnegative integer representing 2j.", "jx(1)"),
        a("jy_matrix", "qm", "jy_matrix(two_j: usize, interner: &Interner) -> Result<Vec<Vec<Expr>>, SpinError>", "Construct the exact spin-j Cartesian operator J_y = (J_+ - J_-)/(2i).", "The argument two_j must be a nonnegative integer representing 2j.", "jy(1)"),
        a("two_spin_half_singlet_state", "qm", "two_spin_half_singlet_state(interner: &Interner) -> Vec<Expr>", "Return the explicit two-spin-1/2 singlet state in the computational basis.", "No extra setup is required.", "singlet_state_2spinhalf()"),
        a("two_spin_half_triplet_states", "qm", "two_spin_half_triplet_states(interner: &Interner) -> [Vec<Expr>; 3]", "Return the explicit two-spin-1/2 triplet states in the computational basis.", "No extra setup is required.", "triplet_states_2spinhalf()"),
        a("two_spin_half_singlet_projector", "qm", "two_spin_half_singlet_projector(interner: &Interner) -> Vec<Vec<Expr>>", "Return the exact singlet projector |S><S| for two spin-1/2 systems.", "No extra setup is required.", "singlet_projector_2spinhalf()"),
        a("two_spin_half_triplet_projector", "qm", "two_spin_half_triplet_projector(interner: &Interner) -> Vec<Vec<Expr>>", "Return the exact triplet projector for two spin-1/2 systems.", "No extra setup is required.", "triplet_projector_2spinhalf()"),
        a("gamma5", "qm", "gamma5(interner: &Interner) -> Vec<Vec<Expr>>", "Return the standard Dirac gamma_5 matrix.", "No extra setup is required.", "gamma5()"),
        a("outer", "qm", "outer(ket: &[Expr], bra: &[Expr]) -> Vec<Vec<Expr>>", "Build the outer-product operator |ket><bra| from two vectors.", "The two vectors should have finite explicit components.", "outer([1,0], [0,1])"),
        a("determinant", "linalg", "determinant(matrix: &[Vec<Expr>], interner: &Interner) -> Expr", "Compute the determinant of a symbolic square matrix.", "The matrix should be square; symbolic simplification is applied recursively by minors.", "det([[1, 2], [3, 4]])"),
        a("inverse", "linalg", "inverse(matrix: &[Vec<Expr>], interner: &Interner) -> Option<Vec<Vec<Expr>>>", "Compute the symbolic inverse of a square matrix by adjugate over determinant.", "The matrix must be square and have nonzero determinant.", "inv([[1, 0], [0, 2]])"),
        a("trace", "linalg", "trace(matrix: &[Vec<Expr>]) -> Expr", "Compute the trace of a square matrix.", "The matrix should be square.", "trace_mat([[1, 2], [3, 4]])"),
        a("eigenvalues_symbolic", "linalg", "eigenvalues_symbolic(matrix: &[Vec<Expr>], interner: &Interner) -> Expr", "Return the characteristic polynomial det(A - lambda I) for a symbolic matrix.", "The matrix should be square; solving that polynomial is a separate step.", "eigenvalues([[a, b], [c, d]])"),
        a("tensor_product", "linalg", "tensor_product(a: &[Vec<Expr>], b: &[Vec<Expr>]) -> Vec<Vec<Expr>>", "Compute the Kronecker product of two matrices.", "Both inputs must be rectangular matrices.", "tensor_product([[1,0],[0,1]], [[0,1],[1,0]])"),
    ]
}

pub fn syntax_rules() -> Vec<SyntaxRule> {
    vec![
        s("module name;", "Declare a module name; the core lowering step accepts and ignores it, while frontends preserve it as a file-level declaration.", "module demo;"),
        s("import std.path.name", "Import a dotted module path.", "import std.gr.schwarzschild"),
        s("let x = expr", "Create a top-level binding. As a bare statement it lowers to Let(x, expr, x).", "let x = 5"),
        s("let x = expr in body", "Create a local binding scoped to body.", "let x = 2 in x + 3"),
        s("f(x, y) = expr", "Define a function with identifier parameters.", "f(x, y) = x^2 + y"),
        s("assume x real positive integer", "Attach one or more assumptions to a symbol.", "assume n integer positive"),
        s("grassmann theta eta", "Declare one or more Grassmann variables.", "grassmann theta eta"),
        s("indices family [a, b, c] dim=4 values=[i, j, k] position=fixed", "Declare an index family with optional dimension, explicit values, and free/fixed position metadata.", "indices spacetime [mu, nu, rho, sigma] dim=4"),
        s("coordinates [t, r, theta, phi]", "Declare the active coordinate labels.", "coordinates [t, r, theta, phi]"),
        s("property T metric", "Declare a tensor property on a symbol.", "property g metric"),
        s("depends T [x, t] or depends T x", "Declare explicit symbol dependencies.", "depends phi [t, x]"),
        s("weight A -1 label=field", "Assign an integer symbolic weight with an optional label.", "weight psi 1 label=field"),
        s("convention key value", "Set an active convention entry such as metric_signature or riemann_sign.", "convention riemann_sign mtw"),
        s("rule name: lhs => rhs", "Define a rewrite rule.", "rule pythag: sin(x_)^2 + cos(x_)^2 => 1"),
        s("rule [exact] name: lhs => rhs", "Define a rewrite rule with a trust level.", "rule [exact] pythag: sin(x_)^2 + cos(x_)^2 => 1"),
        s("if cond then a else b", "Conditional expression lowered to a two-branch piecewise form.", "if x > 0 then x else -x"),
        s("piecewise(v1, cond1, v2, cond2, ...)", "Explicit piecewise constructor.", "piecewise(x, x > 0, -x, true)"),
        s("a + b, a - b, a * b, a / b, a ^ b", "Arithmetic with standard precedence and right-associative exponentiation.", "x + y*z^2"),
        s("-a", "Unary negation.", "-x^2"),
        s("(expr)", "Parenthesized grouping.", "(x + 1)^2"),
        s("name(args...)", "Function or builtin call syntax.", "integrate(x^2, x)"),
        s("T[mu-, nu+]", "ASCII indexed-tensor syntax with explicit variance markers.", "T[mu-, nu+]"),
        s("[a, b, c]", "List literal syntax.", "[t, r, theta, phi]"),
        s("[[a, b], [c, d]]", "Nested list syntax commonly used for matrices.", "[[1, 2], [3, 4]]"),
        s("x > y, x >= y, x < y, x <= y, x == y, x != y, a and b, a or b, not a", "Condition syntax used by if/then/else and piecewise.", "if x >= 0 then x else -x"),
        s("ident", "Identifier syntax: leading ASCII letter, then letters, digits, or underscores.", "alpha1"),
        s("123, 3.14", "Integer and floating-point literals.", "1 + 2.5"),
        s("// comment", "Line comment syntax recognized by core lowering and the lightweight syntax lexer.", "// this is a comment"),
        s("/* comment */", "Block comment syntax recognized by the lightweight syntax lexer.", "1 /* note */ + 2"),
        s("R_{a b c d}, T^{a}_{b}", "LaTeX-style tensor indices accepted by the LaTeX translation path and converted into ASCII indexed syntax.", "R_{a b c d}"),
        s("\\frac{a}{b}, \\sqrt{x}, \\partial_{a}", "Supported LaTeX command fragments translated before lowering.", "\\frac{1}{2} \\partial_{a} phi"),
    ]
}

pub fn std_modules() -> Vec<StdModule> {
    vec![
        m("algebra", "Notes the standard algebra operations used for expansion and simplification.", "documentation comments only"),
        m("calculus", "Documents the standard calculus builtins for differentiation, integration, series, and limits.", "documentation comments only"),
        m("conventions/landau", "Sets Landau-Lifshitz sign and curvature conventions.", "convention metric_signature mostly_minus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one"),
        m("conventions/mtw", "Sets Misner-Thorne-Wheeler general-relativity conventions.", "convention metric_signature mostly_plus, convention riemann_sign mtw, convention ricci_contraction first_third, convention levi_civita_norm plus_one"),
        m("conventions/particle_physics", "Sets particle-physics sign conventions.", "convention metric_signature mostly_plus, convention riemann_sign mtw, convention fourier_sign minus_i"),
        m("conventions/weinberg", "Sets Weinberg general-relativity conventions.", "convention metric_signature mostly_plus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one"),
        m("gr/de_sitter", "Builds the de Sitter metric and its Christoffel symbols in static coordinates.", "let f, let g, let coords, let Gamma"),
        m("gr/frw", "Builds a flat FRW metric with symbolic scale factor and computes Christoffel symbols.", "let g, let coords, let Gamma"),
        m("gr/kerr_newman", "Defines symbolic Kerr-Newman metric component expressions in Boyer-Lindquist coordinates.", "let Sigma_expr, let Delta_expr, let g_tt, let g_rr, let g_theta_theta, let g_phi_phi, let g_t_phi"),
        m("gr/minkowski", "Builds flat Minkowski spacetime and its vanishing Christoffel symbols.", "let g, let coords, let Gamma"),
        m("gr/abstract_tensor", "Abstract GR tensor algebra with declared Riemann tensors, covariant derivatives, and the finished reduction pipeline.", "indices spacetime [mu,nu,rho,sigma,lambda] dim=11, riemann_tensor(R), covariant_derivative(nabla), abstract_tensor_reduce(...)"),
        m("gr/perturbation", "Metric perturbation theory: expansion of inverse metric, Christoffel symbols, Riemann, Ricci, and Einstein tensors to arbitrary order in a perturbation parameter.", "perturb, perturb_inverse, perturb_christoffel, perturb_riemann, perturb_ricci, perturb_einstein"),
        m("gr/schwarzschild", "Builds the Schwarzschild metric, Christoffel symbols, Riemann tensor, and Ricci tensor.", "let g, let coords, let Gamma, let R, let Ric"),
        m("cosmology/perturbation", "Cosmological perturbation theory: SVT decomposition, Bardeen variables, structured CPT specs and workflows, linearized Einstein equations, Mukhanov-Sasaki equation, power spectrum, spectral index.", "linearized_einstein, mukhanov_sasaki, svt_decompose, bardeen, frw_background_spec, cpt_gauge, cpt_matter, cpt_linearized_einstein, cpt_fluid_equations, cpt_mukhanov_sasaki, cpt_mukhanov_sasaki_first_order, cpt_export_mode_rhs, power_spectrum, spectral_index, tensor_scalar_ratio"),
        m("gr/black_hole_perturbation", "Black hole perturbation theory: Regge-Wheeler and Zerilli master equations for Schwarzschild perturbations.", "regge_wheeler, zerilli, regge_wheeler_decompose"),
        m("physics/classical_mechanics", "Builds free-particle, harmonic-oscillator, and pendulum Lagrangians and computes their Euler-Lagrange equations.", "let L_free, let free_particle, let L_ho, let harmonic_oscillator, let L_pendulum, let pendulum"),
        m("physics/differential_forms", "Builds one-forms, wedge products, exterior derivatives, Hodge duals, codifferentials, and Lie derivatives of forms.", "coordinates [x, y], let A, let B, let F, let g, let A_wedge_B, let dA, let star_A, let delta_A, let lie_A"),
        m("physics/klein_gordon", "Sets up a Klein-Gordon Lagrangian and computes its Euler-Lagrange equation.", "let dphi_dt, let dphi_dx, let dphi_dy, let dphi_dz, let L, let EOM"),
        m("physics/maxwell", "Builds a flat-space Maxwell Lagrangian from symbolic field-strength components and computes Gauss-law and Ampere-type Euler-Lagrange equations.", "let F_tx, let F_ty, let F_tz, let F_xy, let F_xz, let F_yz, let L, let gauss_law, let ampere_x"),
        m("physics/variational", "Builds single-field and coupled-field variational examples, including functional derivatives, first variations, and Euler-Lagrange systems.", "let L_mech, let eom_mech, let delta_L_mech, let L_fields, let eoms_fields"),
        m("qft/bilinears", "Majorana and Weyl bilinear selection rules plus explicit/implicit spinor-index bridge examples.", "simplify_spinor_bilinears, insert_explicit_spinor_indices, remove_trivial_spinor_indices"),
        m("qft/chiral_projectors", "Chiral projector algebra and Weyl-spinor chirality actions under structured metadata.", "projector_left, projector_right, simplify_chiral, simplify_spinor_bilinears"),
        m("qft/dirac", "Dirac / spinor manipulations: Dirac-bar expansion, spinor sorting, sigma matrices, gamma conversion, and spinor-index bridge examples.", "expand_diracbar, sort_spinors, sigma_to_gamma, gamma_to_sigma, insert_explicit_spinor_indices"),
        m("qft/fierz", "Fierz rearrangement examples under explicit 4D spinor and gamma metadata.", "fierz, sort_product, join_gamma, split_gamma"),
        m("qft/gamma", "Gamma-matrix algebra: joining/splitting chains and Dirac traces.", "join_gammas_in_expr, split_gamma, gamma_trace, gamma5_trace"),
        m("qft/gamma_trace", "Convention-aware gamma traces, gamma5 traces, and gamma-chain join/split examples.", "declare_gamma5_convention, gamma_trace, gamma5_trace, join_gamma, split_gamma"),
        m("qft/normal_ordering", "Bosonic normal ordering, Wick expansion, and abstract oscillator actions.", "normal_order, wick, apply_operator"),
        m("qft/scalar_field", "Free scalar-field Klein-Gordon Lagrangian and Euler-Lagrange equation.", "let dphi_dt, let dphi_dx, let dphi_dy, let dphi_dz, let L, let EOM"),
        m("qft/spinors", "Consolidated spinor/gamma workflows: conventions, chiral projectors, gamma traces, Fierz rearrangements, and bilinear selection rules.", "declare_spinor_meta, declare_gamma_convention, declare_gamma5_convention, simplify_chiral, gamma_trace, fierz, simplify_spinor_bilinears"),
        m("qft/spinor_helicity", "Spinor-helicity formalism: angle/square brackets, Mandelstam invariants, Parke-Taylor amplitudes, BCFW recursion, momentum twistors.", "angle, square, mandelstam, parke_taylor, bcfw_shift, bcfw_decomposition, four_bracket"),
        m("qft/superspace", "N=1 superspace: supercovariant derivatives, chiral/antichiral superfields, Wess-Zumino gauge vector superfields, D-algebra, superspace integration.", "setup_superspace, expand_superfield, chiral_superfield, d_alpha, d_squared, superspace_integrate"),
        m("qft/brst", "BRST cohomology: ghost grading, Yang-Mills BRST setup, nilpotency, and ghost-sector projection.", "setup_brst_ym, brst, ghost_number, brst_check, filter_ghost_number"),
        m("qm/bell", "Constructs a Bell state, its density matrix, and a reduced density matrix by partial trace.", "let up, let down, let phi_plus, let rho, let rho_A"),
        m("qm/channels", "Finite-dimensional Kraus-channel examples including the identity channel, canonical qubit noise channels, channel composition, tensor-product channels, Choi-matrix distance surrogates, and exact trace-preserving and unital checks.", "let I, depolarizing_channel, dephasing_channel, amplitude_damping_channel, bit_flip_channel, phase_flip_channel, bit_phase_flip_channel, compose_channels, choi_distance, tensor_product_channel, trace_preserving_residual, is_trace_preserving, unital_residual, is_unital"),
        m("qm/dynamics", "Quantum dynamics examples: propagators, Schrödinger and Heisenberg evolution, Liouville/Lindblad generators, and steady states.", "let H_qubit, let U_t, let psi_t, let x_t, let liouville_toy, let lindblad_amp, let rho_steady"),
        m("qm/fock", "Shows symbolic bosonic displacement and squeezing series examples in Fock-space notation.", "let disp_series, let squeeze_series"),
        m("qm/harmonic_oscillator", "Builds an abstract harmonic-oscillator annihilation operator, creation operator, number operator, Hamiltonian, and sample Fock-state actions.", "let a_op, let adag_op, let n_op, let h_op, let vac, let one, let two, let lowered_two, let number_on_two, let energy_on_one, let normal_reordered"),
        m("qm/info", "Quantum-information examples: entropies, mutual information, entanglement spectrum, Schmidt data, and negativity diagnostics.", "let s_vn_mm, let s_renyi2_mm, let s_vn_A, let mutual_info_bell, let negativity_bell, let log_negativity_bell"),
        m("qm/spin", "Builds spin-1/2 and spin-1 angular-momentum matrices, including Pauli and arbitrary spin-j examples.", "let sigma_x, let sigma_y, let sigma_z, let jz_half, let jplus_one, let jz_one"),
        m("qm/states", "Quantum-state examples: basis kets/bras, density matrices, Bell-state reductions, subsystem partial traces, and Bloch vectors.", "let ket0, let ket1, let rho0, let rho_plus, let bell_rho, let rho_A, let bloch_rho0"),
        m("tensor/index", "Documents index notation and contraction conventions for tensors.", "documentation comments only"),
        m("tensor/symmetry", "Documents tensor-symmetry declarations and examples.", "documentation comments only"),
        m("trig", "Defines standard exact trigonometric rewrite rules.", "rule pythag, rule pythag_alt1, rule pythag_alt2, rule double_sin, rule double_cos"),
        m("units/cgs", "Documents the CGS unit system and derived units.", "documentation comments only"),
        m("units/natural", "Documents the natural-unit system convention.", "documentation comments only"),
        m("units/si", "Documents the SI unit system import and usage.", "documentation comments only"),
    ]
}

pub fn convention_entries() -> Vec<ConventionEntry> {
    vec![
        c(
            "metric_signature",
            "MostlyPlus, MostlyMinus",
            "MostlyPlus",
            "Chooses the sign convention for the metric signature.",
        ),
        c(
            "riemann_sign",
            "MTW, Weinberg",
            "MTW",
            "Chooses the sign convention for the Riemann tensor definition.",
        ),
        c(
            "ricci_contraction",
            "FirstThird, FirstFourth",
            "FirstThird",
            "Chooses which Riemann slots are contracted to form the Ricci tensor.",
        ),
        c(
            "levi_civita_norm",
            "PlusOne, MinusOne, SqrtG",
            "PlusOne",
            "Chooses the normalization convention for the Levi-Civita tensor.",
        ),
        c(
            "fourier_sign",
            "MinusI, PlusI",
            "MinusI",
            "Chooses the sign convention in Fourier-transform exponentials.",
        ),
    ]
}

pub fn assumption_entries() -> Vec<AssumptionEntry> {
    vec![
        asm("Real", "Expression is assumed to take real values."),
        asm("Positive", "Expression is assumed strictly positive."),
        asm("Negative", "Expression is assumed strictly negative."),
        asm("NonZero", "Expression is assumed not equal to zero."),
        asm("Integer", "Expression is assumed to be an integer."),
        asm("Even", "Expression is assumed to be an even integer."),
        asm("Odd", "Expression is assumed to be an odd integer."),
    ]
}

fn pdef(
    name: &'static str,
    param_type: ParamType,
    required: bool,
    description: &'static str,
) -> ParamDef {
    ParamDef {
        name,
        param_type,
        required,
        description,
    }
}

fn centry(
    name: &'static str,
    description: &'static str,
    parameters: &'static [ParamDef],
    handler: fn(&[serde_json::Value], &mut dyn EvalState) -> Result<serde_json::Value, String>,
) -> CallableEntry {
    let category = match name {
        "list_expressions"
        | "list_metrics"
        | "list_properties"
        | "list_index_families"
        | "get_state_summary" => "state",
        "diff" | "check_properties" | "explain" => "diagnostics",
        "workflow" | "list_workflows" => "workflow",
        _ => "general",
    };
    CallableEntry {
        name,
        category,
        description,
        parameters,
        handler,
    }
}

fn handle_eval_syntax_entry(
    _args: &[serde_json::Value],
    _state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Err(
        "This registry entry is source-level Axioma syntax; call eval with the corresponding code."
            .to_string(),
    )
}

fn require_arg<'a>(
    args: &'a [serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<&'a serde_json::Value, String> {
    args.get(idx)
        .ok_or_else(|| format!("missing required argument '{name}'"))
}

fn expr_from_id(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<ax_ir::Expr, String> {
    let id = require_arg(args, idx, name)?
        .as_str()
        .ok_or_else(|| format!("argument '{name}' must be a string expression id"))?;
    state
        .get_expr(id)
        .cloned()
        .ok_or_else(|| format!("unknown expression id '{id}'"))
}

fn code_expr(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<ax_ir::Expr, String> {
    let code = require_arg(args, idx, name)?
        .as_str()
        .ok_or_else(|| format!("argument '{name}' must be a code string"))?;
    state.parse_code(code)
}

fn tableau_from_expr(expr: &ax_ir::Expr) -> Result<ax_young::YoungTableau, String> {
    let rows = match expr {
        ax_ir::Expr::List(rows) => rows.clone(),
        ax_ir::Expr::Matrix(rows) => rows.iter().cloned().map(ax_ir::Expr::List).collect(),
        _ => return Err("tableau must be a nested list such as [[0,1],[2]]".to_string()),
    };

    let mut parsed_rows = Vec::with_capacity(rows.len());
    for row in &rows {
        let ax_ir::Expr::List(cells) = row else {
            return Err("each tableau row must be a list of slot numbers".to_string());
        };
        let mut parsed_cells = Vec::with_capacity(cells.len());
        for cell in cells {
            let ax_ir::Expr::Int(value) = cell else {
                return Err("tableau cells must be non-negative integers".to_string());
            };
            let as_usize = value
                .to_usize()
                .ok_or_else(|| "tableau cells must be non-negative integers".to_string())?;
            parsed_cells.push(as_usize);
        }
        parsed_rows.push(parsed_cells);
    }

    ax_young::YoungTableau::with_metadata(parsed_rows, num_rational::BigRational::one(), 0)
        .map_err(|err| err.to_string())
}

fn symbol_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<lasso::Spur, String> {
    let sym_name = require_arg(args, idx, name)?
        .as_str()
        .ok_or_else(|| format!("argument '{name}' must be a symbol string"))?;
    Ok(state.interner_mut().get_or_intern(sym_name))
}

fn symbol_list_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<Vec<lasso::Spur>, String> {
    let value = require_arg(args, idx, name)?;
    let list = value
        .as_array()
        .ok_or_else(|| format!("argument '{name}' must be an array of symbol strings"))?;
    list.iter()
        .map(|item| {
            item.as_str()
                .map(|s| state.interner_mut().get_or_intern(s))
                .ok_or_else(|| format!("argument '{name}' contains a non-string item"))
        })
        .collect()
}

fn optional_symbol_list_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<Vec<lasso::Spur>, String> {
    match args.get(idx) {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::String(s)) => Ok(vec![state.interner_mut().get_or_intern(s)]),
        Some(serde_json::Value::Array(_)) => symbol_list_arg(args, idx, name, state),
        Some(_) => Err(format!(
            "argument '{name}' must be null, a symbol string, or an array of symbols"
        )),
    }
}

fn int_arg(args: &[serde_json::Value], idx: usize, name: &str) -> Result<i64, String> {
    require_arg(args, idx, name)?
        .as_i64()
        .ok_or_else(|| format!("argument '{name}' must be an integer"))
}

fn float_arg(args: &[serde_json::Value], idx: usize, name: &str) -> Result<f64, String> {
    require_arg(args, idx, name)?
        .as_f64()
        .ok_or_else(|| format!("argument '{name}' must be a float"))
}

fn bool_arg(args: &[serde_json::Value], idx: usize, name: &str) -> Result<bool, String> {
    require_arg(args, idx, name)?
        .as_bool()
        .ok_or_else(|| format!("argument '{name}' must be a boolean"))
}

fn string_arg<'a>(
    args: &'a [serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<&'a str, String> {
    require_arg(args, idx, name)?
        .as_str()
        .ok_or_else(|| format!("argument '{name}' must be a string"))
}

fn string_list_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<Vec<String>, String> {
    let arr = require_arg(args, idx, name)?
        .as_array()
        .ok_or_else(|| format!("argument '{name}' must be an array of strings"))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| format!("argument '{name}' contains a non-string item"))
        })
        .collect()
}

fn optional_integer_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<Option<usize>, String> {
    match args.get(idx) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .ok_or_else(|| format!("argument '{name}' must be an integer or null"))
            .and_then(|n| {
                usize::try_from(n)
                    .map(Some)
                    .map_err(|_| format!("argument '{name}' must be non-negative"))
            }),
    }
}

fn optional_symbol_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<Option<lasso::Spur>, String> {
    match args.get(idx) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(_)) => Ok(Some(symbol_arg(args, idx, name, state)?)),
        Some(_) => Err(format!("argument '{name}' must be a symbol string or null")),
    }
}

fn parse_spinor_class_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<ax_ir::SpinorClass, String> {
    match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
        "dirac" => Ok(ax_ir::SpinorClass::Dirac),
        "majorana" => Ok(ax_ir::SpinorClass::Majorana),
        "weyl" => Ok(ax_ir::SpinorClass::Weyl),
        "majoranaweyl" | "majorana_weyl" => Ok(ax_ir::SpinorClass::MajoranaWeyl),
        _ => Err(format!(
            "argument '{name}' must be one of: dirac, majorana, weyl, majorana_weyl"
        )),
    }
}

fn parse_optional_chirality_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<Option<ax_ir::Chirality>, String> {
    match args.get(idx) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(_) => match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
            "left" => Ok(Some(ax_ir::Chirality::Left)),
            "right" => Ok(Some(ax_ir::Chirality::Right)),
            "none" => Ok(None),
            _ => Err(format!(
                "argument '{name}' must be 'left', 'right', or null"
            )),
        },
    }
}

fn attach_compatible_property(
    state: &mut dyn EvalState,
    symbol: lasso::Spur,
    property: ax_ir::TensorProperty,
) {
    for property in crate::property_store::expand_compatible_properties(property) {
        state
            .env_mut()
            .tensor_properties
            .entry(symbol)
            .or_default()
            .push(property.clone());
        state
            .env_mut()
            .property_store
            .declare_simple(symbol, property);
    }
}

fn hilbert_space_metadata_for_symbol(
    state: &mut dyn EvalState,
    symbol: lasso::Spur,
) -> Option<ax_ir::HilbertSpaceMetadata> {
    state
        .env()
        .property_store
        .get_all(symbol)
        .into_iter()
        .find_map(|prop| match prop {
            ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
        .or_else(|| {
            state
                .env()
                .tensor_properties
                .get(&symbol)
                .and_then(|props| {
                    props.iter().find_map(|prop| match prop {
                        ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => Some(metadata.clone()),
                        _ => None,
                    })
                })
        })
}

fn flatten_declared_hilbert_factors(
    state: &mut dyn EvalState,
    factors: &[lasso::Spur],
) -> Option<Vec<ax_ir::HilbertSpaceFactor>> {
    if factors.is_empty() {
        return None;
    }
    let mut flattened = Vec::new();
    for factor in factors {
        let metadata = hilbert_space_metadata_for_symbol(state, *factor)?;
        if metadata.factors.is_empty() {
            return None;
        }
        flattened.extend(metadata.factors);
    }
    Some(flattened)
}

fn mode_metadata_for_symbol(
    state: &mut dyn EvalState,
    symbol: lasso::Spur,
) -> Option<ax_ir::ModeMetadata> {
    state
        .env()
        .property_store
        .get_all(symbol)
        .into_iter()
        .find_map(|prop| match prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => Some(metadata.clone()),
            _ => None,
        })
        .or_else(|| {
            state
                .env()
                .tensor_properties
                .get(&symbol)
                .and_then(|props| {
                    props.iter().find_map(|prop| match prop {
                        ax_ir::TensorProperty::ModeMeta(metadata) => Some(metadata.clone()),
                        _ => None,
                    })
                })
        })
}

fn build_fock_space_metadata_for_state(
    state: &mut dyn EvalState,
    symbol: lasso::Spur,
    mode_symbols: &[lasso::Spur],
) -> Option<ax_ir::FockSpaceMetadata> {
    if mode_symbols.is_empty() {
        return None;
    }
    let modes = mode_symbols
        .iter()
        .map(|mode_symbol| {
            let metadata = mode_metadata_for_symbol(state, *mode_symbol)?;
            Some(ax_ir::FockModeFactor {
                symbol: *mode_symbol,
                statistics: metadata.statistics,
                truncation: state.env().fock_mode_truncations.get(mode_symbol).copied(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ax_ir::FockSpaceMetadata {
        symbol,
        modes,
        basis_order: mode_symbols.to_vec(),
    })
}

fn factor_dimensions_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<Vec<usize>, String> {
    let values = require_arg(args, idx, name)?
        .as_array()
        .ok_or_else(|| format!("argument '{name}' must be an array of integers"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_i64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("argument '{name}' contains a non-integer item"))
        })
        .collect()
}

fn unique_factor_index_in_metadata(
    metadata: &ax_ir::HilbertSpaceMetadata,
    factor_symbol: lasso::Spur,
) -> Result<usize, &'static str> {
    let matches = metadata
        .factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (factor.symbol == factor_symbol).then_some(index))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Err("partial_trace_space factor symbol not found in composite space"),
        [index] => Ok(*index),
        _ => Err("partial_trace_space factor symbol must occur exactly once in composite space"),
    }
}

fn parse_quantum_object_kind_name(kind: &str) -> Option<ax_ir::QuantumObjectKind> {
    match kind.to_ascii_lowercase().as_str() {
        "ket" => Some(ax_ir::QuantumObjectKind::Ket),
        "bra" => Some(ax_ir::QuantumObjectKind::Bra),
        "operator" => Some(ax_ir::QuantumObjectKind::Operator),
        "density_operator" => Some(ax_ir::QuantumObjectKind::DensityOperator),
        "projector" => Some(ax_ir::QuantumObjectKind::Projector),
        "observable" => Some(ax_ir::QuantumObjectKind::Observable),
        "channel" => Some(ax_ir::QuantumObjectKind::Channel),
        _ => None,
    }
}

fn parse_gamma_metric_signature_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<ax_ir::MetricSignature, String> {
    match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
        "mostly_plus" => Ok(ax_ir::MetricSignature::MostlyPlus),
        "mostly_minus" => Ok(ax_ir::MetricSignature::MostlyMinus),
        "euclidean" => Ok(ax_ir::MetricSignature::Euclidean),
        _ => Err(
            "gamma convention signature must be one of: mostly_plus, mostly_minus, euclidean"
                .to_string(),
        ),
    }
}

fn parse_clifford_convention_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<ax_ir::CliffordConvention, String> {
    match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
        "plus_two_g" => Ok(ax_ir::CliffordConvention::PlusTwoG),
        "minus_two_g" => Ok(ax_ir::CliffordConvention::MinusTwoG),
        _ => Err(
            "gamma convention clifford sign must be one of: plus_two_g, minus_two_g".to_string(),
        ),
    }
}

fn parse_gamma5_convention_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<ax_ir::GammaFiveConvention, String> {
    match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
        "levi_civita" => Ok(ax_ir::GammaFiveConvention::LeviCivita),
        "abstract_chiral" => Ok(ax_ir::GammaFiveConvention::AbstractChiral),
        _ => Err("gamma5 convention kind must be one of: levi_civita, abstract_chiral".to_string()),
    }
}

fn positive_gamma_dimension_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<usize, String> {
    require_arg(args, idx, name)?
        .as_i64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| "gamma convention dimension must be a positive integer".to_string())
}

fn matrix_code_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<Vec<Vec<String>>, String> {
    let rows = require_arg(args, idx, name)?
        .as_array()
        .ok_or_else(|| format!("argument '{name}' must be a 2D array of code strings"))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| format!("argument '{name}' must be a 2D array of code strings"))
                .and_then(|cells| {
                    cells
                        .iter()
                        .map(|cell| {
                            cell.as_str().map(|s| s.to_string()).ok_or_else(|| {
                                format!("argument '{name}' contains a non-string matrix entry")
                            })
                        })
                        .collect()
                })
        })
        .collect()
}

fn matrix_from_expr(expr: &ax_ir::Expr) -> Option<Vec<Vec<ax_ir::Expr>>> {
    match expr {
        ax_ir::Expr::Matrix(rows) => Some(rows.clone()),
        ax_ir::Expr::List(rows) => rows
            .iter()
            .map(|row| match row {
                ax_ir::Expr::List(cells) => Some(cells.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn matrix_from_id(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<Vec<Vec<ax_ir::Expr>>, String> {
    let id = string_arg(args, idx, name)?;
    if let Some(matrix) = state.get_matrix_data(id) {
        return Ok(matrix);
    }
    let expr = state
        .get_expr(id)
        .ok_or_else(|| format!("unknown expression id '{id}'"))?;
    matrix_from_expr(expr).ok_or_else(|| format!("expression '{id}' is not a matrix"))
}

fn list_from_expr(expr: &ax_ir::Expr) -> Option<Vec<ax_ir::Expr>> {
    match expr {
        ax_ir::Expr::List(items) => Some(items.clone()),
        _ => None,
    }
}

fn list_from_id(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    state: &mut dyn EvalState,
) -> Result<Vec<ax_ir::Expr>, String> {
    let expr = expr_from_id(args, idx, name, state)?;
    list_from_expr(&expr)
        .ok_or_else(|| format!("argument '{name}' must reference a list expression"))
}

fn symbolic_matrix_from_rows(
    rows: Vec<Vec<ax_ir::Expr>>,
) -> Result<ax_tensor::SymbolicMatrix, String> {
    let dim = rows.len();
    if dim == 0 {
        return Ok(ax_tensor::SymbolicMatrix::new(0));
    }
    if rows.iter().any(|row| row.len() != dim) {
        return Err("matrix must be square".to_string());
    }
    let mut m = ax_tensor::SymbolicMatrix::new(dim);
    for (i, row) in rows.into_iter().enumerate() {
        for (j, cell) in row.into_iter().enumerate() {
            m.set(i, j, cell);
        }
    }
    Ok(m)
}

fn matrix_response(
    matrix: Vec<Vec<ax_ir::Expr>>,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = ax_ir::Expr::Matrix(matrix.clone());
    let expr_id = state.store_expr(expr);
    let rendered = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| state.render_latex(cell))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rows = matrix.len();
    let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
    Ok(serde_json::json!({
        "status": "ok",
        "expr_id": expr_id,
        "matrix": rendered,
        "dimensions": [rows, cols]
    }))
}

fn list_response(
    items: Vec<ax_ir::Expr>,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = ax_ir::Expr::List(items.clone());
    let expr_id = state.store_expr(expr);
    Ok(serde_json::json!({
        "status": "ok",
        "expr_id": expr_id,
        "components": items.iter().map(|item| state.render_latex(item)).collect::<Vec<_>>()
    }))
}

fn points_response(points: Vec<(f64, f64)>) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "status": "ok",
        "points": points.into_iter().map(|(x, y)| serde_json::json!({"x": x, "y": y})).collect::<Vec<_>>()
    }))
}

pub fn format_tensor_property(prop: &ax_ir::TensorProperty, interner: &ax_ir::Interner) -> String {
    use ax_ir::TensorProperty;

    match prop {
        TensorProperty::Symmetric(slots) => format!("Symmetric(slots: {:?})", slots),
        TensorProperty::AntiSymmetric(slots) => format!("AntiSymmetric(slots: {:?})", slots),
        TensorProperty::RiemannSymmetry => "RiemannSymmetry".to_string(),
        TensorProperty::Traceless => "Traceless".to_string(),
        TensorProperty::Diagonal => "Diagonal".to_string(),
        TensorProperty::Trace => "Trace".to_string(),
        TensorProperty::Metric => "Metric".to_string(),
        TensorProperty::InverseMetric => "InverseMetric".to_string(),
        TensorProperty::KroneckerDelta => "KroneckerDelta".to_string(),
        TensorProperty::EpsilonTensor => "EpsilonTensor".to_string(),
        TensorProperty::Derivative => "Derivative".to_string(),
        TensorProperty::PartialDerivative => "PartialDerivative".to_string(),
        TensorProperty::CovariantDerivative => "CovariantDerivative".to_string(),
        TensorProperty::TableauInherit => "TableauInherit".to_string(),
        TensorProperty::Depends(syms) => format!(
            "Depends({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::Spinor => "Spinor".to_string(),
        TensorProperty::SpinorMeta(metadata) => format!(
            "SpinorMeta(class: {:?}, dimension: {:?}, chirality: {:?}, index_family: {:?})",
            metadata.class,
            metadata.dimension,
            metadata.chirality,
            metadata
                .index_family
                .map(|sym| interner.resolve(sym).to_string())
        ),
        TensorProperty::DiracBar => "DiracBar".to_string(),
        TensorProperty::DiracBarMeta(metadata) => format!(
            "DiracBarMeta(gamma_symbol: {:?}, spinor_family: {:?}, reverse_gamma_order: {})",
            metadata
                .gamma_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata
                .spinor_family
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.reverse_gamma_order
        ),
        TensorProperty::GammaMatrixProp => "GammaMatrix".to_string(),
        TensorProperty::GammaMatrixMeta(metadata) => format!(
            "GammaMatrixMeta(dimension: {:?}, metric_symbol: {:?}, index_family: {:?}, has_gamma5: {})",
            metadata.dimension,
            metadata
                .metric_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata
                .index_family
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.has_gamma5
        ),
        TensorProperty::GammaConventionMeta(metadata) => format!(
            "GammaConventionMeta(signature: {:?}, clifford: {:?}, gamma5: {:?}, epsilon_symbol: {:?}, dimension: {:?})",
            metadata.signature,
            metadata.clifford,
            metadata.gamma5,
            metadata
                .epsilon_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.dimension
        ),
        TensorProperty::Commuting => "Commuting".to_string(),
        TensorProperty::AntiCommuting => "AntiCommuting".to_string(),
        TensorProperty::NonCommuting => "NonCommuting".to_string(),
        TensorProperty::CommutingWith(syms) => format!(
            "CommutingWith({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::AntiCommutingWith(syms) => format!(
            "AntiCommutingWith({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::NonCommutingWith(syms) => format!(
            "NonCommutingWith({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::SelfAntiCommuting => "SelfAntiCommuting".to_string(),
        TensorProperty::SelfNonCommuting => "SelfNonCommuting".to_string(),
        TensorProperty::SelfCommuting => "SelfCommuting".to_string(),
        TensorProperty::CommutingAsProduct => "CommutingAsProduct".to_string(),
        TensorProperty::CommutingAsSum => "CommutingAsSum".to_string(),
        TensorProperty::MajoranaSpinor => "MajoranaSpinor".to_string(),
        TensorProperty::WeylSpinor => "WeylSpinor".to_string(),
        TensorProperty::ImplicitIndex => "ImplicitIndex".to_string(),
        TensorProperty::SortOrder(syms) => format!(
            "SortOrder({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::TableauSymmetry(symmetry) => {
            format!("TableauSymmetry(tableaux: {:?})", symmetry.tableaux)
        }
        TensorProperty::MixedTableauSymmetry(symmetry) => {
            format!("MixedTableauSymmetry(tableaux: {:?})", symmetry.tableaux)
        }
        TensorProperty::GradedParity(values) => format!("GradedParity({values:?})"),
        TensorProperty::TensorIdentities(identities) => {
            format!("TensorIdentities(multiterm: {:?})", identities.multiterm)
        }
        TensorProperty::SatisfiesBianchi { slots } => {
            format!("SatisfiesBianchi(slots: {:?})", slots)
        }
        TensorProperty::DimensionDependentIdentity => "DimensionDependentIdentity".to_string(),
        TensorProperty::WeylTensor => "WeylTensor".to_string(),
        TensorProperty::DifferentialFormDegree(d) => {
            format!("DifferentialForm(degree: {})", d)
        }
        TensorProperty::HilbertSpaceMeta(metadata) => format!(
            "HilbertSpaceMeta(dimension: {}, factors: [{}])",
            metadata.dimension,
            metadata
                .factors
                .iter()
                .map(|factor| format!("{}:{}", interner.resolve(factor.symbol), factor.dimension))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::FockSpaceMeta(metadata) => format!(
            "FockSpaceMeta(symbol: {}, modes: [{}], basis_order: [{}])",
            interner.resolve(metadata.symbol),
            metadata
                .modes
                .iter()
                .map(|mode| {
                    format!(
                        "{}:{:?}:{:?}",
                        interner.resolve(mode.symbol),
                        mode.statistics,
                        mode.truncation
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
            metadata
                .basis_order
                .iter()
                .map(|sym| interner.resolve(*sym).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::QuantumObjectMeta(metadata) => format!(
            "QuantumObjectMeta(kind: {:?}, space_symbol: {})",
            metadata.kind,
            interner.resolve(metadata.space_symbol)
        ),
        TensorProperty::OperatorSpaceMeta(metadata) => format!(
            "OperatorSpaceMeta(domain_space: {}, codomain_space: {})",
            interner.resolve(metadata.domain_space),
            interner.resolve(metadata.codomain_space)
        ),
        TensorProperty::ModeMeta(metadata) => format!(
            "ModeMeta(statistics: {:?}, subsystem: {:?}, mode_index: {}, label: {:?})",
            metadata.statistics,
            metadata
                .subsystem
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.mode_index,
            metadata.label.map(|sym| interner.resolve(sym).to_string())
        ),
        TensorProperty::BackgroundClass(sym) => {
            format!("BackgroundClass({})", interner.resolve(*sym))
        }
        TensorProperty::PerturbationFamily { family, order } => format!(
            "PerturbationFamily(family: {}, order: {})",
            interner.resolve(*family),
            order
        ),
        TensorProperty::SectorTag(sym) => {
            format!("SectorTag({})", interner.resolve(*sym))
        }
        TensorProperty::GaugeTag {
            gauge,
            invariant,
            generator,
        } => format!(
            "GaugeTag(gauge: {}, invariant: {}, generator: {})",
            interner.resolve(*gauge),
            invariant,
            generator
        ),
        TensorProperty::HarmonicTag { basis, wave_symbol } => format!(
            "HarmonicTag(basis: {}, wave_symbol: {})",
            interner.resolve(*basis),
            wave_symbol
                .map(|sym| interner.resolve(sym).to_string())
                .unwrap_or_else(|| "None".to_string())
        ),
        TensorProperty::MatterTag(sym) => {
            format!("MatterTag({})", interner.resolve(*sym))
        }
        TensorProperty::TraceSpaceMeta(metadata) => format!(
            "TraceSpaceMeta(space_symbol: {}, cyclic: {})",
            interner.resolve(metadata.space_symbol),
            metadata.cyclic
        ),
    }
}

pub fn property_lookup_aliases(name: &str) -> &'static [&'static str] {
    match name {
        "Spinor" => &["spinor"],
        "SpinorMeta" => &["spinor_meta", "declare_spinor_meta"],
        "DiracBar" => &["dirac_bar", "diracbar"],
        "DiracBarMeta" => &["dirac_bar_meta", "declare_dirac_bar_meta"],
        "GammaMatrixProp" => &["gamma_matrix", "gamma"],
        "GammaMatrixMeta" => &["gamma_matrix_meta", "declare_gamma_matrix_meta"],
        "GammaConventionMeta" => &[
            "gamma_convention_meta",
            "declare_gamma_convention",
            "declare_gamma5_convention",
        ],
        "TraceSpaceMeta" => &["trace_space", "declare_trace_space"],
        "HilbertSpaceMeta" => &[
            "hilbert_space",
            "declare_hilbert_space",
            "declare_composite_space",
        ],
        "FockSpaceMeta" => &["fock_space", "declare_fock_space"],
        "QuantumObjectMeta" => &["quantum_object", "declare_quantum_object"],
        "OperatorSpaceMeta" => &["operator_space", "declare_operator_space"],
        "ModeMeta" => &[
            "mode_meta",
            "declare_mode",
            "declare_mode_in_subsystem",
            "declare_mode_with_label",
        ],
        "MajoranaSpinor" => &["majorana_spinor"],
        "WeylSpinor" => &["weyl_spinor"],
        _ => &[],
    }
}

pub fn property_lookup_names() -> Vec<&'static str> {
    let mut names = Vec::new();
    for entry in property_entries() {
        names.push(entry.name);
        names.extend_from_slice(property_lookup_aliases(entry.name));
    }
    names.sort_unstable();
    names.dedup();
    names
}

fn expr_or_struct_response(
    expr: ax_ir::Expr,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    match expr {
        ax_ir::Expr::Matrix(rows) => matrix_response(rows, state),
        ax_ir::Expr::List(items) => list_response(items, state),
        other => expr_response(other, state),
    }
}

fn annotate_success_response(
    mut response: serde_json::Value,
    status: &str,
    changed: bool,
    message: String,
) -> Result<serde_json::Value, String> {
    let obj = response
        .as_object_mut()
        .ok_or_else(|| "success response must be a JSON object".to_string())?;
    obj.insert("status".to_string(), serde_json::json!(status));
    obj.insert("changed".to_string(), serde_json::json!(changed));
    obj.insert("message".to_string(), serde_json::json!(message));
    Ok(response)
}

fn ensure_not_timeout(
    expr: ax_ir::Expr,
    interner: &ax_ir::Interner,
) -> Result<ax_ir::Expr, String> {
    if ax_tensor::is_timeout_expr(&expr, interner) {
        Err("computation timed out".to_string())
    } else {
        Ok(expr)
    }
}

fn evaluate_matrix(
    matrix: Vec<Vec<ax_ir::Expr>>,
    state: &mut dyn EvalState,
) -> Vec<Vec<ax_ir::Expr>> {
    matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| crate::eval(&cell, state.env(), state.interner()))
                .map(|cell| crate::simplify::simplify(&cell, state.interner()))
                .collect()
        })
        .collect()
}

fn evaluate_matrix_lightweight(
    matrix: Vec<Vec<ax_ir::Expr>>,
    state: &mut dyn EvalState,
) -> Result<Vec<Vec<ax_ir::Expr>>, String> {
    matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| {
                    state.check_deadline()?;
                    let evaluated = crate::eval(&cell, state.env(), state.interner());
                    Ok(crate::simplify::rationalize_expanded_numerator(
                        &evaluated,
                        state.interner(),
                    ))
                })
                .collect()
        })
        .collect()
}

fn evaluate_list(items: Vec<ax_ir::Expr>, state: &mut dyn EvalState) -> Vec<ax_ir::Expr> {
    items
        .into_iter()
        .map(|item| crate::eval(&item, state.env(), state.interner()))
        .collect()
}

fn parse_property_string(
    value: &str,
    state: &mut dyn EvalState,
) -> Result<ax_ir::TensorProperty, String> {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    let parse_usize_list = |body: &str| -> Result<Vec<usize>, String> {
        if body.trim().is_empty() {
            return Ok(Vec::new());
        }
        body.split(',')
            .map(|item| {
                item.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("invalid integer slot '{item}' in property '{trimmed}'"))
            })
            .collect()
    };
    let parse_sym_list = |body: &str, state: &mut dyn EvalState| -> Vec<lasso::Spur> {
        body.split(',')
            .filter_map(|item| {
                let s = item.trim();
                (!s.is_empty()).then(|| state.interner_mut().get_or_intern(s))
            })
            .collect()
    };

    if lower == "riemannsymmetry" || lower == "riemann_symmetry" {
        return Ok(ax_ir::TensorProperty::RiemannSymmetry);
    }
    if lower == "traceless" {
        return Ok(ax_ir::TensorProperty::Traceless);
    }
    if lower == "diagonal" {
        return Ok(ax_ir::TensorProperty::Diagonal);
    }
    if lower == "trace" {
        return Ok(ax_ir::TensorProperty::Trace);
    }
    if lower == "metric" {
        return Ok(ax_ir::TensorProperty::Metric);
    }
    if lower == "inversemetric" || lower == "inverse_metric" {
        return Ok(ax_ir::TensorProperty::InverseMetric);
    }
    if lower == "kroneckerdelta" || lower == "kronecker_delta" {
        return Ok(ax_ir::TensorProperty::KroneckerDelta);
    }
    if lower == "epsilontensor" || lower == "epsilon_tensor" {
        return Ok(ax_ir::TensorProperty::EpsilonTensor);
    }
    if lower == "derivative" {
        return Ok(ax_ir::TensorProperty::Derivative);
    }
    if lower == "partialderivative" || lower == "partial_derivative" {
        return Ok(ax_ir::TensorProperty::PartialDerivative);
    }
    if lower == "covariantderivative" || lower == "covariant_derivative" {
        return Ok(ax_ir::TensorProperty::CovariantDerivative);
    }
    if lower == "tableauinherit" || lower == "tableau_inherit" {
        return Ok(ax_ir::TensorProperty::TableauInherit);
    }
    if lower == "spinor" {
        return Ok(ax_ir::TensorProperty::Spinor);
    }
    if lower == "diracbar" || lower == "dirac_bar" {
        return Ok(ax_ir::TensorProperty::DiracBar);
    }
    if lower == "gammamatrixprop" || lower == "gamma_matrix" || lower == "gammamatrix" {
        return Ok(ax_ir::TensorProperty::GammaMatrixProp);
    }
    if lower == "commuting" {
        return Ok(ax_ir::TensorProperty::Commuting);
    }
    if lower == "anticommuting" || lower == "anti_commuting" {
        return Ok(ax_ir::TensorProperty::AntiCommuting);
    }
    if lower == "noncommuting" || lower == "non_commuting" {
        return Ok(ax_ir::TensorProperty::NonCommuting);
    }
    if let Some(body) = trimmed
        .strip_prefix("CommutingWith(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::CommutingWith(parse_sym_list(
            body, state,
        )));
    }
    if let Some(body) = trimmed
        .strip_prefix("AntiCommutingWith(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::AntiCommutingWith(parse_sym_list(
            body, state,
        )));
    }
    if let Some(body) = trimmed
        .strip_prefix("NonCommutingWith(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::NonCommutingWith(parse_sym_list(
            body, state,
        )));
    }
    if lower == "selfanticommuting" || lower == "self_anticommuting" {
        return Ok(ax_ir::TensorProperty::SelfAntiCommuting);
    }
    if lower == "selfnoncommuting" || lower == "self_noncommuting" {
        return Ok(ax_ir::TensorProperty::SelfNonCommuting);
    }
    if lower == "selfcommuting" || lower == "self_commuting" {
        return Ok(ax_ir::TensorProperty::SelfCommuting);
    }
    if lower == "commutingasproduct" || lower == "commuting_as_product" {
        return Ok(ax_ir::TensorProperty::CommutingAsProduct);
    }
    if lower == "commutingassum" || lower == "commuting_as_sum" {
        return Ok(ax_ir::TensorProperty::CommutingAsSum);
    }
    if lower == "majoranaspinor" || lower == "majorana_spinor" {
        return Ok(ax_ir::TensorProperty::MajoranaSpinor);
    }
    if lower == "weylspinor" || lower == "weyl_spinor" {
        return Ok(ax_ir::TensorProperty::WeylSpinor);
    }
    if lower == "implicitindex" || lower == "implicit_index" {
        return Ok(ax_ir::TensorProperty::ImplicitIndex);
    }
    if lower == "dimensiondependentidentity" || lower == "dimension_dependent_identity" {
        return Ok(ax_ir::TensorProperty::DimensionDependentIdentity);
    }
    if lower == "satisfiesbianchi" || lower == "satisfies_bianchi" || lower == "bianchi" {
        return Ok(ax_ir::TensorProperty::SatisfiesBianchi {
            slots: vec![0, 1, 2, 3],
        });
    }
    if lower == "weyltensor" || lower == "weyl_tensor" || lower == "weyl" {
        return Ok(ax_ir::TensorProperty::WeylTensor);
    }
    if let Some(body) = trimmed
        .strip_prefix("Symmetric(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::Symmetric(parse_usize_list(body)?));
    }
    if let Some(body) = trimmed
        .strip_prefix("AntiSymmetric(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::AntiSymmetric(parse_usize_list(
            body,
        )?));
    }
    if let Some(body) = trimmed
        .strip_prefix("Depends(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::Depends(parse_sym_list(body, state)));
    }
    if let Some(body) = trimmed
        .strip_prefix("SortOrder(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return Ok(ax_ir::TensorProperty::SortOrder(parse_sym_list(
            body, state,
        )));
    }
    if let Some(body) = trimmed
        .strip_prefix("DifferentialFormDegree(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let n = body
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid differential form degree in '{trimmed}'"))?;
        return Ok(ax_ir::TensorProperty::DifferentialFormDegree(n));
    }
    if let Some(body) = trimmed
        .strip_prefix("TableauSymmetry(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let mut shape = Vec::new();
        let mut indices = Vec::new();
        for part in body.split(';') {
            let part = part.trim();
            if let Some(rest) = part.strip_prefix("shape=") {
                shape = parse_usize_list(rest)?;
            } else if let Some(rest) = part.strip_prefix("indices=") {
                indices = parse_usize_list(rest)?;
            }
        }
        let symmetry = ax_ir::TensorSymmetry {
            tableaux: vec![ax_ir::TableauAttachment {
                shape,
                slot_map: indices,
                multiplicity_numer: 1,
                multiplicity_denom: 1,
                duality: ax_ir::DualityKind::None,
                restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                trace_free: false,
                dimension_guard: None,
                source: ax_ir::SymmetrySource::Declared,
                label: None,
            }],
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        };
        symmetry.validate().map_err(|err| err.to_string())?;
        return Ok(ax_ir::TensorProperty::TableauSymmetry(symmetry));
    }
    if let Some(body) = trimmed
        .strip_prefix("SatisfiesBianchi(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let slots = parse_usize_list(body)?;
        if slots.len() != 3 && slots.len() != 4 {
            return Err(format!(
                "SatisfiesBianchi requires exactly three or four slots in '{trimmed}'"
            ));
        }
        return Ok(ax_ir::TensorProperty::SatisfiesBianchi { slots });
    }
    match lower.as_str() {
        "symmetric" => Ok(ax_ir::TensorProperty::Symmetric(vec![0, 1])),
        "antisymmetric" => Ok(ax_ir::TensorProperty::AntiSymmetric(vec![0, 1])),
        _ => Err(format!("unknown tensor property '{value}'")),
    }
}

fn convention_value_to_json(env: &crate::Env) -> serde_json::Value {
    serde_json::json!({
        "metric_signature": format!("{:?}", env.convention.metric_signature),
        "riemann_sign": format!("{:?}", env.convention.riemann_sign),
        "ricci_contraction": format!("{:?}", env.convention.ricci_contraction),
        "levi_civita_norm": format!("{:?}", env.convention.levi_civita_norm),
        "fourier_sign": format!("{:?}", env.convention.fourier_sign),
    })
}

fn expr_response(
    expr: ax_ir::Expr,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr_id = state.store_expr(expr.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "expr_id": expr_id,
        "latex": state.render_latex(&expr),
        "unicode": state.render_unicode(&expr)
    }))
}

fn expr_response_with_change(
    input_expr: &ax_ir::Expr,
    output_expr: ax_ir::Expr,
    algorithm_name: &str,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let changed = input_expr != &output_expr;
    let expr_id = state.store_expr(output_expr.clone());
    let status = if changed { "ok" } else { "unchanged" };
    let message = if changed {
        format!("{algorithm_name} applied successfully")
    } else {
        format!("{algorithm_name} did not change the expression")
    };
    Ok(serde_json::json!({
        "status": status,
        "changed": changed,
        "message": message,
        "expr_id": expr_id,
        "latex": state.render_latex(&output_expr),
        "unicode": state.render_unicode(&output_expr),
    }))
}

fn zoom_response(
    focus: ax_ir::Expr,
    remainder: ax_ir::Expr,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let focus_id = state.store_expr(focus.clone());
    let remainder_id = state.store_expr(remainder.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "focus_id": focus_id,
        "focus_latex": state.render_latex(&focus),
        "focus_unicode": state.render_unicode(&focus),
        "remainder_id": remainder_id,
        "remainder_latex": state.render_latex(&remainder),
        "remainder_unicode": state.render_unicode(&remainder)
    }))
}

fn expr_or_struct_response_with_change(
    input_expr: &ax_ir::Expr,
    output_expr: ax_ir::Expr,
    algorithm_name: &str,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let changed = input_expr != &output_expr;
    let status = if changed { "ok" } else { "unchanged" };
    let message = if changed {
        format!("{algorithm_name} applied successfully")
    } else {
        format!("{algorithm_name} did not change the expression")
    };
    let response = match output_expr {
        ax_ir::Expr::Matrix(rows) => matrix_response(rows, state)?,
        ax_ir::Expr::List(items) => list_response(items, state)?,
        other => return expr_response_with_change(input_expr, other, algorithm_name, state),
    };
    annotate_success_response(response, status, changed, message)
}

fn expr_or_struct_response_named(
    output_expr: ax_ir::Expr,
    algorithm_name: &str,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    annotate_success_response(
        expr_or_struct_response(output_expr, state)?,
        "ok",
        true,
        format!("{algorithm_name} applied successfully"),
    )
}

fn call_named(name: &str, call_args: Vec<ax_ir::Expr>, state: &mut dyn EvalState) -> ax_ir::Expr {
    let sym = state.interner_mut().get_or_intern(name);
    crate::eval(
        &ax_ir::Expr::Call(sym, call_args),
        state.env(),
        state.interner(),
    )
}

fn has_indices(expr: &ax_ir::Expr) -> bool {
    match expr {
        ax_ir::Expr::Indexed(_, _) => true,
        ax_ir::Expr::Add(terms) | ax_ir::Expr::Mul(terms) | ax_ir::Expr::List(terms) => {
            terms.iter().any(has_indices)
        }
        ax_ir::Expr::Matrix(rows) => rows.iter().flatten().any(has_indices),
        ax_ir::Expr::Pow(base, exp) => has_indices(base) || has_indices(exp),
        ax_ir::Expr::Neg(inner) | ax_ir::Expr::Group(inner, _) => has_indices(inner),
        ax_ir::Expr::Complex(re, im) => has_indices(re) || has_indices(im),
        ax_ir::Expr::Call(_, args) => args.iter().any(has_indices),
        ax_ir::Expr::FnDef(_, _, body) => has_indices(body),
        ax_ir::Expr::Rule(lhs, rhs, _) => has_indices(lhs) || has_indices(rhs),
        ax_ir::Expr::Piecewise(cases) => cases.iter().any(|(value, _)| has_indices(value)),
        ax_ir::Expr::Let(_, value, body) => has_indices(value) || has_indices(body),
        ax_ir::Expr::Import(_)
        | ax_ir::Expr::Assume(_, _)
        | ax_ir::Expr::SetConvention(_, _)
        | ax_ir::Expr::Sym(_)
        | ax_ir::Expr::Int(_)
        | ax_ir::Expr::Rational(_)
        | ax_ir::Expr::Float(_) => false,
    }
}

fn derivative_syms(state: &dyn EvalState) -> std::collections::HashSet<lasso::Spur> {
    state
        .env()
        .tensor_properties
        .iter()
        .filter(|(_, props)| {
            props.iter().any(|p| {
                matches!(
                    p,
                    ax_ir::TensorProperty::Derivative
                        | ax_ir::TensorProperty::PartialDerivative
                        | ax_ir::TensorProperty::CovariantDerivative
                )
            })
        })
        .map(|(sym, _)| *sym)
        .collect()
}

fn handle_diff(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    expr_response_with_change(
        &expr,
        crate::differentiate(&expr, var, state.interner()),
        "differentiate",
        state,
    )
}

fn handle_integrate(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    expr_response_with_change(
        &expr,
        crate::integrate::integrate(&expr, var, state.interner()),
        "integrate",
        state,
    )
}

fn handle_double_integral(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let x = symbol_arg(args, 1, "x", state)?;
    let y = symbol_arg(args, 2, "y", state)?;
    let inner = crate::integrate::integrate(&expr, x, state.interner());
    let outer = crate::integrate::integrate(&inner, y, state.interner());
    expr_response_with_change(
        &expr,
        crate::eval(&outer, state.env(), state.interner()),
        "double_integral",
        state,
    )
}

fn handle_triple_integral(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let x = symbol_arg(args, 1, "x", state)?;
    let y = symbol_arg(args, 2, "y", state)?;
    let z = symbol_arg(args, 3, "z", state)?;
    let i1 = crate::integrate::integrate(&expr, x, state.interner());
    let i2 = crate::integrate::integrate(&i1, y, state.interner());
    let i3 = crate::integrate::integrate(&i2, z, state.interner());
    expr_response_with_change(
        &expr,
        crate::eval(&i3, state.env(), state.interner()),
        "triple_integral",
        state,
    )
}

fn handle_definite_integral(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    let lower = code_expr(args, 2, "lower_bound", state)?;
    let upper = code_expr(args, 3, "upper_bound", state)?;
    let antideriv = crate::integrate::integrate(&expr, var, state.interner());
    let at_b =
        crate::symbolic_substitute(&antideriv, &ax_ir::Expr::Sym(var), &upper, state.interner());
    let at_a =
        crate::symbolic_substitute(&antideriv, &ax_ir::Expr::Sym(var), &lower, state.interner());
    let result = crate::eval(
        &ax_ir::Expr::add(vec![at_b, ax_ir::Expr::neg(at_a)]),
        state.env(),
        state.interner(),
    );
    expr_response_with_change(&expr, result, "definite_integral", state)
}

fn handle_integrate_by_parts(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let away = symbol_arg(args, 1, "away", state)?;
    let deriv_syms = derivative_syms(state);
    expr_response_with_change(
        &expr,
        ax_tensor::integrate_by_parts(&expr, away, &deriv_syms, state.interner()),
        "integrate_by_parts",
        state,
    )
}

fn handle_limit(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    let point = code_expr(args, 2, "point", state)?;
    expr_response_with_change(
        &expr,
        crate::limits::limit(&expr, var, &point, state.interner()),
        "limit",
        state,
    )
}

fn handle_series(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    let point = code_expr(args, 2, "point", state)?;
    let order = int_arg(args, 3, "order")?;
    if order < 0 {
        return Err("argument 'order' must be non-negative".to_string());
    }
    expr_response_with_change(
        &expr,
        crate::series::taylor_series(&expr, var, &point, order as usize, state.interner()),
        "series",
        state,
    )
}

fn handle_simplify(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::simplify_checked(&expr, state.interner())?,
        "simplify",
        state,
    )
}

fn handle_expand(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::expand(&expr, state.interner()),
        "expand",
        state,
    )
}

fn handle_collect_terms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::collect_terms(&expr, state.interner()),
        "collect_terms",
        state,
    )
}

fn handle_rationalize(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::rationalize(&expr, state.interner()),
        "rationalize",
        state,
    )
}

fn handle_partial_fractions(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    let result =
        crate::simplify::apart_expr(&expr, var, state.interner()).unwrap_or_else(|| expr.clone());
    expr_response_with_change(&expr, result, "partial_fractions", state)
}

fn handle_trig_simplify(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::trig_simplify(&expr, state.interner()),
        "trig_simplify",
        state,
    )
}

fn handle_factor_out(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let targets = optional_symbol_list_arg(args, 1, "targets", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::factor_out(&expr, &targets, state.interner()),
        "factor_out",
        state,
    )
}

fn handle_factor_in(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let targets = optional_symbol_list_arg(args, 1, "targets", state)?;
    expr_response_with_change(
        &expr,
        crate::simplify::factor_in(&expr, &targets, state.interner()),
        "factor_in",
        state,
    )
}

fn handle_subs(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let target = code_expr(args, 1, "target", state)?;
    let replacement = code_expr(args, 2, "replacement", state)?;
    let result = if has_indices(&expr) || has_indices(&target) || has_indices(&replacement) {
        crate::substitute_with_indices(&expr, &target, &replacement, state.env(), state.interner())
    } else {
        crate::symbolic_substitute(&expr, &target, &replacement, state.interner())
    };
    expr_response_with_change(
        &expr,
        crate::eval(&result, state.env(), state.interner()),
        "substitute",
        state,
    )
}

fn handle_rewrite(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        crate::rewrite_with_trace(&expr, state.env(), state.interner()).0,
        "rewrite",
        state,
    )
}

fn handle_zoom(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let pattern = code_expr(args, 1, "pattern", state)?;
    let (focus, remainder) = crate::zoom(&expr, &pattern, state.interner());
    zoom_response(focus, remainder, state)
}

fn handle_unzoom(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let focus = expr_from_id(args, 0, "focus", state)?;
    let remainder = expr_from_id(args, 1, "remainder", state)?;
    expr_response_with_change(&focus, crate::unzoom(&focus, &remainder), "unzoom", state)
}

fn handle_take_match(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let pattern = code_expr(args, 1, "pattern", state)?;
    expr_response_with_change(
        &expr,
        crate::take_match(&expr, &pattern, state.interner()),
        "take_match",
        state,
    )
}

fn unary_expr_builtin(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(name, vec![expr.clone()], state);
    expr_response_with_change(&expr, result, name, state)
}

fn binary_expr_builtin(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lhs = expr_from_id(args, 0, "lhs", state)?;
    let rhs = expr_from_id(args, 1, "rhs", state)?;
    let result = call_named(name, vec![lhs.clone(), rhs.clone()], state);
    let changed =
        ax_ir::Expr::Call(state.interner_mut().get_or_intern(name), vec![lhs, rhs]) != result;
    let mut response = expr_or_struct_response(result, state)?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert(
            "status".to_string(),
            serde_json::json!(if changed { "ok" } else { "unchanged" }),
        );
        obj.insert("changed".to_string(), serde_json::json!(changed));
        obj.insert(
            "message".to_string(),
            serde_json::json!(if changed {
                format!("{name} applied successfully")
            } else {
                format!("{name} did not change the expression")
            }),
        );
    }
    Ok(response)
}

fn handle_sin(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sin", args, state)
}
fn handle_cos(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("cos", args, state)
}
fn handle_tan(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("tan", args, state)
}
fn handle_sec(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sec", args, state)
}
fn handle_csc(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("csc", args, state)
}
fn handle_cot(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("cot", args, state)
}
fn handle_asin(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("asin", args, state)
}
fn handle_arcsin(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arcsin", args, state)
}
fn handle_acos(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("acos", args, state)
}
fn handle_arccos(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arccos", args, state)
}
fn handle_atan(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("atan", args, state)
}
fn handle_arctan(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arctan", args, state)
}
fn handle_atan2(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_expr_builtin("atan2", args, state)
}
fn handle_sinh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sinh", args, state)
}
fn handle_cosh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("cosh", args, state)
}
fn handle_tanh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("tanh", args, state)
}
fn handle_asinh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("asinh", args, state)
}
fn handle_arcsinh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arcsinh", args, state)
}
fn handle_acosh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("acosh", args, state)
}
fn handle_arccosh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arccosh", args, state)
}
fn handle_atanh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("atanh", args, state)
}
fn handle_arctanh(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arctanh", args, state)
}
fn handle_exp(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("exp", args, state)
}
fn handle_log(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("log", args, state)
}
fn handle_sqrt(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sqrt", args, state)
}
fn handle_abs(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("abs", args, state)
}
fn handle_sign(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sign", args, state)
}
fn handle_sgn(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("sgn", args, state)
}
fn handle_re(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("Re", args, state)
}
fn handle_im(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("Im", args, state)
}
fn handle_conj(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("conj", args, state)
}
fn handle_arg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("arg", args, state)
}
fn handle_n(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_expr_builtin("N", args, state)
}

fn handle_gradient(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named(
        "gradient",
        vec![expr.clone(), ax_ir::Expr::List(vars)],
        state,
    );
    expr_response_with_change(&expr, result, "gradient", state)
}

fn handle_grad(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named("grad", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
    expr_response_with_change(&expr, result, "grad", state)
}

fn handle_divergence(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named(
        "divergence",
        vec![expr.clone(), ax_ir::Expr::List(vars)],
        state,
    );
    expr_response_with_change(&expr, result, "divergence", state)
}

fn handle_div(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named("div", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
    expr_response_with_change(&expr, result, "div", state)
}

fn handle_curl(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named("curl", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
    expr_response_with_change(&expr, result, "curl", state)
}

fn handle_laplacian(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named(
        "laplacian",
        vec![expr.clone(), ax_ir::Expr::List(vars)],
        state,
    );
    expr_response_with_change(&expr, result, "laplacian", state)
}

fn handle_jacobian(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named(
        "jacobian",
        vec![expr.clone(), ax_ir::Expr::List(vars)],
        state,
    );
    expr_response_with_change(&expr, result, "jacobian", state)
}

fn handle_hessian(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let vars = symbol_list_arg(args, 1, "variables", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    let result = call_named(
        "hessian",
        vec![expr.clone(), ax_ir::Expr::List(vars)],
        state,
    );
    expr_response_with_change(&expr, result, "hessian", state)
}

fn unary_named_expr_response(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(name, vec![expr.clone()], state);
    expr_or_struct_response_with_change(&expr, result, name, state)
}

fn binary_named_expr_response(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lhs = expr_from_id(args, 0, "lhs", state)?;
    let rhs = expr_from_id(args, 1, "rhs", state)?;
    expr_or_struct_response_named(call_named(name, vec![lhs, rhs], state), name, state)
}

fn handle_equation_ternary(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let target = code_expr(args, 1, "target", state)?;
    let replacement = code_expr(args, 2, "replacement", state)?;
    let result = call_named(name, vec![expr.clone(), target, replacement], state);
    expr_response_with_change(&expr, result, name, state)
}

fn handle_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("eq", args, state)
}

fn handle_get_lhs_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("get_lhs", args, state)
}

fn handle_get_rhs_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("get_rhs", args, state)
}

fn handle_swap_sides_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("swap_sides", args, state)
}

fn handle_multiply_through_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("multiply_through", args, state)
}

fn handle_add_through_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("add_through", args, state)
}

fn handle_to_rhs_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("to_rhs", args, state)
}

fn handle_to_lhs_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("to_lhs", args, state)
}

fn handle_isolate_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("isolate", args, state)
}

fn handle_eq_to_rule_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("eq_to_rule", args, state)
}

fn handle_eq_to_subrule_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("eq_to_subrule", args, state)
}

fn handle_differentiate_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let var = symbol_arg(args, 1, "var", state)?;
    let result = call_named(
        "differentiate_eq",
        vec![expr.clone(), ax_ir::Expr::Sym(var)],
        state,
    );
    expr_response_with_change(&expr, result, "differentiate_eq", state)
}

fn handle_integrate_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let var = symbol_arg(args, 1, "var", state)?;
    let result = call_named(
        "integrate_eq",
        vec![expr.clone(), ax_ir::Expr::Sym(var)],
        state,
    );
    expr_response_with_change(&expr, result, "integrate_eq", state)
}

fn handle_substitute_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_equation_ternary("substitute_eq", args, state)
}

fn handle_raise_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let index = symbol_arg(args, 1, "index", state)?;
    let result = call_named(
        "raise_eq",
        vec![expr.clone(), ax_ir::Expr::Sym(index)],
        state,
    );
    expr_response_with_change(&expr, result, "raise_eq", state)
}

fn handle_lower_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let index = symbol_arg(args, 1, "index", state)?;
    let result = call_named(
        "lower_eq",
        vec![expr.clone(), ax_ir::Expr::Sym(index)],
        state,
    );
    expr_response_with_change(&expr, result, "lower_eq", state)
}

#[allow(dead_code)]
fn list_builtin_response(
    name: &str,
    expr: ax_ir::Expr,
    vars: Vec<lasso::Spur>,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let vars = vars.into_iter().map(ax_ir::Expr::Sym).collect::<Vec<_>>();
    expr_or_struct_response(
        call_named(name, vec![expr, ax_ir::Expr::List(vars)], state),
        state,
    )
}

#[allow(dead_code)]
fn eval_call_response(
    name: &str,
    args: Vec<ax_ir::Expr>,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response(call_named(name, args, state), state)
}

fn handle_canonicalise(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let sym = state.interner_mut().get_or_intern("canonicalise");
    let result = crate::eval(
        &ax_ir::Expr::Call(sym, vec![expr.clone()]),
        state.env(),
        state.interner(),
    );
    expr_response_with_change(
        &expr,
        ensure_not_timeout(result, state.interner())?,
        "canonicalise",
        state,
    )
}
fn handle_canonicalize_indices(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        ensure_not_timeout(
            ax_tensor::canonicalize_indices(&expr, &state.env().property_store, state.interner()),
            state.interner(),
        )?,
        "canonicalize_indices",
        state,
    )
}
fn handle_meld(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let sym = state.interner_mut().get_or_intern("meld");
    let result = crate::eval(
        &ax_ir::Expr::Call(sym, vec![expr.clone()]),
        state.env(),
        state.interner(),
    );
    expr_response_with_change(
        &expr,
        ensure_not_timeout(result, state.interner())?,
        "meld",
        state,
    )
}
fn handle_sort_product(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("sort_product", args, state)
}
fn handle_product_rule_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("product_rule", args, state)
}
fn handle_tensor_distribute(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("tensor_distribute", args, state)
}
fn handle_eliminate_kronecker(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("eliminate_kronecker", args, state)
}
fn handle_eliminate_metric(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("eliminate_metric", args, state)
}
fn handle_eliminate_vielbein(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("eliminate_vielbein", args, state)
}
fn handle_epsilon_to_delta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("epsilon_to_delta", args, state)
}
fn handle_expand_delta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("expand_delta", args, state)
}
fn handle_expand_dummies(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("expand_dummies", args, state)
}
fn handle_explicit_indices(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("explicit_indices", args, state)
}
fn handle_expand_implicit(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("expand_implicit", args, state)
}
fn handle_einsteinify(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("einsteinify", args, state)
}
fn handle_rename_dummies(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("rename_dummies", args, state)
}
fn handle_young_project(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    if let Some(tableau_arg) = args.get(1) {
        if !tableau_arg.is_null() && !tableau_arg.is_boolean() {
            let tableau_expr = code_expr(args, 1, "tableau", state)?;
            let tableau = tableau_from_expr(&tableau_expr)?;
            let result = ax_tensor::young_project(&expr, &tableau, state.interner());
            return expr_or_struct_response_with_change(&expr, result, "young_project", state);
        }
    }
    let opts = ax_tensor::YoungProjectTensorOptions {
        modulo_monoterm: match args.get(1) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 1, "modulo_monoterm")?,
        },
        canonicalize_after: match args.get(2) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 2, "canonicalize_after")?,
        },
        rename_dummies_after: match args.get(3) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 3, "rename_dummies_after")?,
        },
    };
    let result = ax_tensor::young_project_tensor_with_options(
        &expr,
        &state.env().property_store,
        state.interner(),
        &opts,
    );
    expr_or_struct_response_with_change(&expr, result, "young_project", state)
}

fn handle_young_project_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let opts = ax_tensor::YoungProjectTensorOptions {
        modulo_monoterm: match args.get(1) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 1, "modulo_monoterm")?,
        },
        canonicalize_after: match args.get(2) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 2, "canonicalize_after")?,
        },
        rename_dummies_after: match args.get(3) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 3, "rename_dummies_after")?,
        },
    };
    let result = ax_tensor::young_project_tensor_with_options(
        &expr,
        &state.env().property_store,
        state.interner(),
        &opts,
    );
    expr_or_struct_response_with_change(&expr, result, "young_project_tensor", state)
}

fn handle_tensor_reduce(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let opts = ax_tensor::TensorReduceOptions {
        monoterm: match args.get(1) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 1, "monoterm")?,
        },
        multiterm: match args.get(2) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 2, "multiterm")?,
        },
        dimension_dependent: match args.get(3) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 3, "dimension_dependent")?,
        },
        meld: match args.get(4) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 4, "meld")?,
        },
        modulo_monoterm: match args.get(5) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 5, "modulo_monoterm")?,
        },
    };
    let result =
        ax_tensor::tensor_reduce(&expr, &state.env().property_store, state.interner(), &opts);
    expr_or_struct_response_with_change(&expr, result, "tensor_reduce", state)
}

fn handle_abstract_tensor_reduce(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let opts = ax_tensor::TensorReduceOptions {
        monoterm: match args.get(1) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 1, "monoterm")?,
        },
        multiterm: match args.get(2) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 2, "multiterm")?,
        },
        dimension_dependent: match args.get(3) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 3, "dimension_dependent")?,
        },
        meld: match args.get(4) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 4, "meld")?,
        },
        modulo_monoterm: match args.get(5) {
            Some(serde_json::Value::Null) | None => true,
            Some(_) => bool_arg(args, 5, "modulo_monoterm")?,
        },
    };
    let result = ax_tensor::tensor_reduce(
        &expr,
        &state.env().tensor_properties,
        state.interner(),
        &opts,
    );
    expr_or_struct_response_with_change(&expr, result, "abstract_tensor_reduce", state)
}

fn handle_riemann_to_ricci_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let ricci_sym = symbol_arg(args, 1, "ricci_sym", state)?;
    let mut call_args = vec![expr.clone(), ax_ir::Expr::Sym(ricci_sym)];
    if !matches!(args.get(2), None | Some(serde_json::Value::Null)) {
        let scalar_sym = symbol_arg(args, 2, "scalar_sym", state)?;
        call_args.push(ax_ir::Expr::Sym(scalar_sym));
    }
    let result = call_named("riemann_to_ricci", call_args, state);
    expr_or_struct_response_with_change(&expr, result, "riemann_to_ricci", state)
}

fn handle_contracted_bianchi_reduce_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let derivative_sym = symbol_arg(args, 1, "derivative_sym", state)?;
    let ricci_sym = symbol_arg(args, 2, "ricci_sym", state)?;
    let scalar_sym = symbol_arg(args, 3, "scalar_sym", state)?;
    let mut call_args = vec![
        expr.clone(),
        ax_ir::Expr::Sym(derivative_sym),
        ax_ir::Expr::Sym(ricci_sym),
        ax_ir::Expr::Sym(scalar_sym),
    ];
    if !matches!(args.get(4), None | Some(serde_json::Value::Null)) {
        let einstein_sym = symbol_arg(args, 4, "einstein_sym", state)?;
        call_args.push(ax_ir::Expr::Sym(einstein_sym));
    }
    let result = call_named("contracted_bianchi_reduce", call_args, state);
    expr_or_struct_response_with_change(&expr, result, "contracted_bianchi_reduce", state)
}

fn handle_reduce_delta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("reduce_delta", args, state)
}
fn handle_unwrap_derivatives_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("unwrap", args, state)
}
fn handle_drop_weight_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let label = string_arg(args, 1, "label")?;
    let value = int_arg(args, 2, "value")?;
    expr_response_with_change(
        &expr,
        ax_tensor::drop_weight(&expr, value, &state.env().weights, label, state.interner()),
        "drop_weight",
        state,
    )
}
fn handle_keep_weight_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let label = string_arg(args, 1, "label")?;
    let value = int_arg(args, 2, "value")?;
    expr_response_with_change(
        &expr,
        ax_tensor::keep_weight(&expr, value, &state.env().weights, label, state.interner()),
        "keep_weight",
        state,
    )
}
fn handle_lower_free_indices(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("lower_free_indices", args, state)
}
fn handle_raise_free_indices(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("raise_free_indices", args, state)
}

fn handle_symmetrise_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let positions = require_arg(args, 1, "positions")?
        .as_array()
        .ok_or_else(|| "argument 'positions' must be an array of integers".to_string())?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| "positions must be integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    expr_response_with_change(
        &expr,
        ax_tensor::symmetrise(&expr, &positions, false, state.interner()),
        "symmetrise",
        state,
    )
}

fn handle_antisymmetrise_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let positions = require_arg(args, 1, "positions")?
        .as_array()
        .ok_or_else(|| "argument 'positions' must be an array of integers".to_string())?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| "positions must be integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    expr_response_with_change(
        &expr,
        ax_tensor::symmetrise(&expr, &positions, true, state.interner()),
        "antisymmetrise",
        state,
    )
}

fn handle_split_index_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let parent = optional_symbol_list_arg(args, 1, "parent_indices", state)?;
    let sub1 = optional_symbol_list_arg(args, 2, "subfamily_one", state)?;
    let sub2 = optional_symbol_list_arg(args, 3, "subfamily_two", state)?;
    expr_response_with_change(
        &expr,
        ax_tensor::split_index(&expr, &parent, &sub1, &sub2, state.interner()),
        "split_index",
        state,
    )
}

fn handle_rewrite_indices_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let tensor = symbol_arg(args, 1, "tensor", state)?;
    let variances = require_arg(args, 2, "variances")?
        .as_array()
        .ok_or_else(|| "argument 'variances' must be an array".to_string())?
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .ok_or_else(|| "variances must be strings".to_string())?;
            match s.to_ascii_lowercase().as_str() {
                "up" | "+" => Ok(ax_ir::Variance::Up),
                "down" | "-" => Ok(ax_ir::Variance::Down),
                _ => Err(format!("unknown variance '{s}'")),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut target_tensors = std::collections::HashMap::new();
    target_tensors.insert(tensor, variances);
    let g = state.interner_mut().get_or_intern("g");
    let ginv = state.interner_mut().get_or_intern("ginv");
    expr_response_with_change(
        &expr,
        ax_tensor::rewrite_indices(&expr, &target_tensors, g, ginv, state.interner()),
        "rewrite_indices",
        state,
    )
}

fn handle_rewrite_indices_vielbein_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let e_sym = symbol_arg(args, 1, "vielbein", state)?;
    let e_inv_sym = symbol_arg(args, 2, "inverse_vielbein", state)?;
    let from_family = symbol_arg(args, 3, "from_family", state)?;
    let to_family = symbol_arg(args, 4, "to_family", state)?;
    expr_response_with_change(
        &expr,
        ax_tensor::rewrite_indices_vielbein(
            &expr,
            e_sym,
            e_inv_sym,
            from_family,
            to_family,
            state.interner(),
        ),
        "rewrite_indices_vielbein",
        state,
    )
}

fn handle_evaluate_components_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let rules_expr = expr_from_id(args, 1, "rules", state)?;
    let rules = crate::parse_component_rules_expr(&rules_expr);
    for rule in &rules {
        state.env_mut().component_rule_symbols.insert(rule.tensor);
    }
    let env = ax_tensor::DefaultEvalEnv::new(
        state.env().coordinates.iter().copied().collect(),
        state.env().tensor_properties.clone(),
    );
    expr_response_with_change(
        &expr,
        ensure_not_timeout(
            ax_tensor::evaluate_components_v2(&expr, &rules, &env, state.interner()),
            state.interner(),
        )?,
        "evaluate_components",
        state,
    )
}

fn handle_complete_inverse_metric(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rules_expr = expr_from_id(args, 0, "rules", state)?;
    let metric = symbol_arg(args, 1, "metric", state)?;
    let inv_metric = symbol_arg(args, 2, "inverse_metric", state)?;
    let coords = symbol_list_arg(args, 3, "coordinates", state)?;
    let rules = if let ax_ir::Expr::List(items) = rules_expr {
        crate::parse_component_rules(&items)
    } else {
        Vec::new()
    };
    let completed =
        ax_tensor::complete_inverse_metric(&rules, metric, inv_metric, &coords, state.interner());
    let as_expr = ax_ir::Expr::List(
        completed
            .into_iter()
            .map(|rule| {
                let lhs = ax_ir::Expr::Indexed(
                    Box::new(ax_ir::Expr::Sym(rule.tensor)),
                    rule.indices
                        .iter()
                        .map(|(name, variance)| ax_ir::Index {
                            name: *name,
                            variance: variance.clone(),
                            index_type: None,
                        })
                        .collect(),
                );
                ax_ir::Expr::Rule(
                    Box::new(lhs),
                    Box::new(rule.value),
                    ax_ir::TrustLevel::Exact,
                )
            })
            .collect(),
    );
    expr_or_struct_response_named(as_expr, "euler_lagrange_system", state)
}

fn handle_diff_component_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    expr_response_with_change(
        &expr,
        ax_tensor::diff_component(&expr, var, state.interner()),
        "diff_component",
        state,
    )
}

fn handle_decompose_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let basis = list_from_id(args, 1, "basis", state)?;
    expr_response_with_change(
        &expr,
        ax_tensor::decompose(&expr, &basis, &state.env().property_store, state.interner()),
        "decompose",
        state,
    )
}

fn handle_decompose_product_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let dim = args
        .get(1)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .or_else(|| ax_tensor::infer_tensor_dimension(&expr, &state.env().property_store))
        .ok_or_else(|| {
            "could not infer a unique dimension from the expression; pass 'dim' explicitly or declare index-family dimensions".to_string()
        })?;
    expr_response_with_change(
        &expr,
        ensure_not_timeout(
            ax_tensor::decompose_product(&expr, dim, &state.env().property_store, state.interner()),
            state.interner(),
        )?,
        "decompose_product",
        state,
    )
}

fn handle_schouten_reduce_tensor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        ensure_not_timeout(
            ax_tensor::schouten_reduce(&expr, &state.env().property_store, state.interner()),
            state.interner(),
        )?,
        "schouten_reduce",
        state,
    )
}

fn handle_metric_pipeline_christoffel(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_id = string_arg(args, 0, "metric_id")?;
    let (metric, coords) = state
        .get_metric(metric_id)
        .cloned()
        .ok_or_else(|| format!("unknown metric '{metric_id}'"))?;
    let gamma = ax_tensor::christoffel_from_metric(&metric, &coords, state.interner());
    let nonzero_count = gamma
        .iter()
        .flatten()
        .flatten()
        .filter(|entry| crate::eval(entry, state.env(), state.interner()) != ax_ir::Expr::zero())
        .count();
    state.store_christoffel(metric_id.to_string(), gamma.clone());
    let mut response = expr_or_struct_response(
        ax_ir::Expr::List(
            gamma
                .into_iter()
                .map(|plane| ax_ir::Expr::List(plane.into_iter().map(ax_ir::Expr::List).collect()))
                .collect(),
        ),
        state,
    )?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
        obj.insert("christoffel_id".to_string(), serde_json::json!(metric_id));
        obj.insert(
            "nonzero_count".to_string(),
            serde_json::json!(nonzero_count),
        );
    }
    Ok(response)
}

fn handle_riemann_from_christoffel(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let id = string_arg(args, 0, "christoffel_id")?;
    let gamma = state
        .get_christoffel(id)
        .cloned()
        .ok_or_else(|| format!("unknown christoffel '{id}'"))?;
    let coords = state
        .get_metric(id)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("no coordinates recorded for '{id}'"))?;
    let riem = ax_tensor::riemann_from_christoffel(
        &gamma,
        &coords,
        state.interner(),
        &state.env().convention,
    );
    state.store_riemann(id.to_string(), riem.clone());
    let expr = ax_ir::Expr::List(
        riem.into_iter()
            .map(|cube| {
                ax_ir::Expr::List(
                    cube.into_iter()
                        .map(|plane| {
                            ax_ir::Expr::List(plane.into_iter().map(ax_ir::Expr::List).collect())
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    let mut response = expr_response(expr, state)?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
        obj.insert("riemann_id".to_string(), serde_json::json!(id));
    }
    Ok(response)
}

fn handle_ricci_from_riemann(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let id = string_arg(args, 0, "riemann_id")?;
    let riem = state
        .get_riemann(id)
        .cloned()
        .ok_or_else(|| format!("unknown riemann '{id}'"))?;
    let ric =
        ax_tensor::ricci_from_riemann(&riem, riem.len(), state.interner(), &state.env().convention);
    state.store_ricci(id.to_string(), ric.clone());
    let mut response = matrix_response(evaluate_matrix_lightweight(ric, state)?, state)?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
        obj.insert("ricci_id".to_string(), serde_json::json!(id));
        if let Some(matrix) = obj.get("matrix").cloned() {
            obj.insert("components".to_string(), matrix);
        }
    }
    Ok(response)
}

fn handle_ricci_scalar_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let ricci = matrix_from_id(args, 0, "ricci", state)?;
    let metric = matrix_from_id(args, 1, "metric_inverse", state)?;
    let ginv = symbolic_matrix_from_rows(metric)?;
    let input_expr = ax_ir::Expr::Matrix(ricci.clone());
    expr_response_with_change(
        &input_expr,
        crate::eval(
            &ax_tensor::ricci_scalar(&ricci, &ginv, state.interner()),
            state.env(),
            state.interner(),
        ),
        "ricci_scalar",
        state,
    )
}

fn handle_scalar_curvature_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let id = string_arg(args, 0, "ricci_id")?;
    let ricci = state
        .get_ricci(id)
        .cloned()
        .ok_or_else(|| format!("unknown ricci '{id}'"))?;
    let metric = state
        .get_metric(id)
        .map(|(m, _)| m.clone())
        .ok_or_else(|| format!("unknown metric '{id}'"))?;
    let input_expr = ax_ir::Expr::Matrix(ricci.clone());
    expr_response_with_change(
        &input_expr,
        crate::eval(
            &ax_tensor::ricci_scalar(
                &ricci,
                &metric.symbolic_inverse(state.interner()),
                state.interner(),
            ),
            state.env(),
            state.interner(),
        ),
        "scalar_curvature",
        state,
    )
}

fn handle_einstein_tensor_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let has_id_mode = args.get(0).and_then(serde_json::Value::as_str).is_some()
        && args.get(1).and_then(serde_json::Value::as_str).is_some()
        && args.get(2).map(|value| value.is_null()).unwrap_or(true);
    let (ricci, scalar, metric) = if has_id_mode {
        let ricci_id = string_arg(args, 0, "ricci_id")?;
        let metric_id = string_arg(args, 1, "metric_id")?;
        let ricci = state
            .get_ricci(ricci_id)
            .cloned()
            .ok_or_else(|| format!("unknown ricci '{ricci_id}'"))?;
        let metric = state
            .get_metric(metric_id)
            .map(|(m, _)| m.clone())
            .ok_or_else(|| format!("unknown metric '{metric_id}'"))?;
        let scalar = crate::eval(
            &ax_tensor::ricci_scalar(
                &ricci,
                &metric.symbolic_inverse(state.interner()),
                state.interner(),
            ),
            state.env(),
            state.interner(),
        );
        (ricci, scalar, metric)
    } else {
        let ricci = matrix_from_id(args, 2, "ricci", state)?;
        let scalar = expr_from_id(args, 3, "scalar", state)?;
        let metric = symbolic_matrix_from_rows(matrix_from_id(args, 4, "metric", state)?)?;
        (ricci, scalar, metric)
    };
    matrix_response(
        evaluate_matrix(
            ax_tensor::einstein_tensor(&ricci, &scalar, &metric, state.interner()),
            state,
        ),
        state,
    )
}

fn handle_weyl_from_curvature_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let riemann_expr = expr_from_id(args, 0, "riemann", state)?;
    let riemann = expr_to_4d(&riemann_expr)
        .ok_or_else(|| "argument 'riemann' must be a rank-4 list tensor".to_string())?;
    let ricci_expr = expr_from_id(args, 1, "ricci", state)?;
    let ricci = match &ricci_expr {
        ax_ir::Expr::Matrix(rows) => rows.clone(),
        _ => return Err("argument 'ricci' must be a matrix expression".to_string()),
    };
    let scalar = expr_from_id(args, 2, "scalar", state)?;
    let metric_expr = expr_from_id(args, 3, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;

    match ax_tensor::weyl_from_curvature(&riemann, &ricci, &scalar, &metric, state.interner()) {
        Ok(weyl) => expr_response(expr_4d_to_list(weyl), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_cotton_from_curvature_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let ricci_expr = expr_from_id(args, 0, "ricci", state)?;
    let ricci = match &ricci_expr {
        ax_ir::Expr::Matrix(rows) => rows.clone(),
        _ => return Err("argument 'ricci' must be a matrix expression".to_string()),
    };
    let scalar = expr_from_id(args, 1, "scalar", state)?;
    let gamma_expr = expr_from_id(args, 2, "gamma", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'gamma' must be a rank-3 list tensor".to_string())?;
    let metric_expr = expr_from_id(args, 3, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 4, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::cotton_from_curvature(
        &ricci,
        &scalar,
        &gamma,
        &metric,
        &coords,
        state.interner(),
    ) {
        Ok(cotton) => expr_response(expr_3d_to_list(cotton), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_bach_from_curvature_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let weyl_expr = expr_from_id(args, 0, "weyl", state)?;
    let weyl = expr_to_4d(&weyl_expr)
        .ok_or_else(|| "argument 'weyl' must be a rank-4 list tensor".to_string())?;
    let ricci_expr = expr_from_id(args, 1, "ricci", state)?;
    let ricci = match &ricci_expr {
        ax_ir::Expr::Matrix(rows) => rows.clone(),
        _ => return Err("argument 'ricci' must be a matrix expression".to_string()),
    };
    let gamma_expr = expr_from_id(args, 2, "gamma", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'gamma' must be a rank-3 list tensor".to_string())?;
    let metric_expr = expr_from_id(args, 3, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 4, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::bach_from_curvature(&weyl, &ricci, &gamma, &metric, &coords, state.interner())
    {
        Ok(bach) => expr_response(ax_ir::Expr::Matrix(bach), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_contorsion_tensor_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let torsion_expr = expr_from_id(args, 0, "torsion", state)?;
    let torsion = expr_to_3d(&torsion_expr)
        .ok_or_else(|| "argument 'torsion' must be a rank-3 list tensor".to_string())?;
    let metric_expr = expr_from_id(args, 1, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    match ax_tensor::contorsion_tensor(&torsion, &metric, state.interner()) {
        Ok(contorsion) => expr_response(expr_3d_to_list(contorsion), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_connection_with_torsion_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gamma_expr = expr_from_id(args, 0, "christoffel", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'christoffel' must be a rank-3 list tensor".to_string())?;
    let contorsion_expr = expr_from_id(args, 1, "contorsion", state)?;
    let contorsion = expr_to_3d(&contorsion_expr)
        .ok_or_else(|| "argument 'contorsion' must be a rank-3 list tensor".to_string())?;
    match ax_tensor::connection_with_torsion(&gamma, &contorsion, state.interner()) {
        Ok(connection) => expr_response(expr_3d_to_list(connection), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_spin_connection_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let vielbein_expr = expr_from_id(args, 0, "vielbein", state)?;
    let vielbein = matrix_to_symbolic(&vielbein_expr)
        .ok_or_else(|| "argument 'vielbein' must be a square matrix".to_string())?;
    let metric_expr = expr_from_id(args, 1, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let coords = symbol_list_arg(args, 2, "coords", state)?;
    match ax_tensor::spin_connection(&vielbein, &metric, &coords, state.interner()) {
        Ok(omega) => expr_response(expr_3d_to_list(omega), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_first_cartan_structure_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let vielbein_expr = expr_from_id(args, 0, "vielbein", state)?;
    let vielbein = matrix_to_symbolic(&vielbein_expr)
        .ok_or_else(|| "argument 'vielbein' must be a square matrix".to_string())?;
    let omega_expr = expr_from_id(args, 1, "spin_connection", state)?;
    let omega = expr_to_3d(&omega_expr)
        .ok_or_else(|| "argument 'spin_connection' must be a rank-3 list tensor".to_string())?;
    let coords = symbol_list_arg(args, 2, "coords", state)?;
    match ax_tensor::first_cartan_structure(&vielbein, &omega, &coords, state.interner()) {
        Ok(forms) => expr_response(
            ax_ir::Expr::List(forms.iter().map(ax_forms::form_to_expr).collect()),
            state,
        ),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_second_cartan_structure_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let omega_expr = expr_from_id(args, 0, "spin_connection", state)?;
    let omega = expr_to_3d(&omega_expr)
        .ok_or_else(|| "argument 'spin_connection' must be a rank-3 list tensor".to_string())?;
    let coords = symbol_list_arg(args, 1, "coords", state)?;
    match ax_tensor::second_cartan_structure(&omega, &coords, state.interner()) {
        Ok(forms) => expr_response(
            ax_ir::Expr::Matrix(
                forms
                    .iter()
                    .map(|row| row.iter().map(ax_forms::form_to_expr).collect())
                    .collect(),
            ),
            state,
        ),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_conformal_transform_metric_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let omega = expr_from_id(args, 1, "omega", state)?;
    expr_response(
        ax_ir::Expr::Matrix(
            ax_tensor::conformal_transform_metric(&metric, &omega, state.interner()).data,
        ),
        state,
    )
}

fn handle_conformal_transform_inverse_metric_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "inverse_metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'inverse_metric' must be a square matrix".to_string())?;
    let omega = expr_from_id(args, 1, "omega", state)?;
    expr_response(
        ax_ir::Expr::Matrix(
            ax_tensor::conformal_transform_inverse_metric(&metric, &omega, state.interner()).data,
        ),
        state,
    )
}

fn handle_conformal_transform_christoffel_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gamma_expr = expr_from_id(args, 0, "gamma", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'gamma' must be a rank-3 list tensor".to_string())?;
    let metric_expr = expr_from_id(args, 1, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let omega = expr_from_id(args, 2, "omega", state)?;
    let coords_expr = expr_from_id(args, 3, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::conformal_transform_christoffel(
        &gamma,
        &metric,
        &omega,
        &coords,
        state.interner(),
    ) {
        Ok(out) => expr_response(expr_3d_to_list(out), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_conformal_transform_ricci_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let ricci_expr = expr_from_id(args, 0, "ricci", state)?;
    let ricci = match &ricci_expr {
        ax_ir::Expr::Matrix(rows) => rows.clone(),
        _ => return Err("argument 'ricci' must be a matrix expression".to_string()),
    };
    let scalar = expr_from_id(args, 1, "scalar", state)?;
    let metric_expr = expr_from_id(args, 2, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let omega = expr_from_id(args, 3, "omega", state)?;
    let coords_expr = expr_from_id(args, 4, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::conformal_transform_ricci(
        &ricci,
        &scalar,
        &metric,
        &omega,
        &coords,
        state.interner(),
    ) {
        Ok(out) => expr_response(ax_ir::Expr::Matrix(out), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_conformal_transform_scalar_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let scalar = expr_from_id(args, 0, "scalar", state)?;
    let metric_expr = expr_from_id(args, 1, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let omega = expr_from_id(args, 2, "omega", state)?;
    let coords_expr = expr_from_id(args, 3, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::conformal_transform_scalar(&scalar, &metric, &omega, &coords, state.interner())
    {
        Ok(out) => expr_response(out, state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_killing_equations_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gamma_expr = expr_from_id(args, 0, "gamma", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'gamma' must be a rank-3 list tensor".to_string())?;
    let coords_expr = expr_from_id(args, 1, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };
    let field_prefix = match args.get(2) {
        Some(serde_json::Value::Null) | None => "xi",
        Some(_) => string_arg(args, 2, "field_prefix")?,
    };
    match ax_tensor::killing_equations(&gamma, &coords, field_prefix, state.interner()) {
        Ok(system) => expr_response(
            ax_ir::Expr::List(vec![
                ax_ir::Expr::List(system.covector_components),
                ax_ir::Expr::List(system.equations),
                ax_ir::Expr::List(
                    system
                        .slot_pairs
                        .into_iter()
                        .map(|(a, b)| {
                            ax_ir::Expr::List(vec![
                                ax_ir::Expr::Int(a.into()),
                                ax_ir::Expr::Int(b.into()),
                            ])
                        })
                        .collect(),
                ),
            ]),
            state,
        ),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_adm_decompose_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 1, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };
    let time_coord_i64 = int_arg(args, 2, "time_coord")?;
    let Some(time_coord) = time_coord_i64.to_usize() else {
        return Err("argument 'time_coord' must be a non-negative integer".to_string());
    };

    match ax_tensor::adm_decompose(&metric, &coords, time_coord, state.interner()) {
        Ok(adm) => expr_response(
            ax_ir::Expr::List(vec![
                adm.lapse,
                ax_ir::Expr::List(adm.shift_covector),
                ax_ir::Expr::List(adm.shift_vector),
                ax_ir::Expr::Matrix(adm.spatial_metric.data),
                ax_ir::Expr::Matrix(adm.spatial_inverse_metric.data),
                ax_ir::Expr::Matrix(adm.extrinsic_curvature),
                adm.hamiltonian_constraint,
                ax_ir::Expr::List(adm.momentum_constraints),
            ]),
            state,
        ),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_spatial_christoffel_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "gamma_ij", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'gamma_ij' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 1, "spatial_coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'spatial_coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'spatial_coords' must be a list of symbols".to_string()),
    };
    expr_response(
        expr_3d_to_list(ax_tensor::spatial_christoffel(
            &metric,
            &coords,
            state.interner(),
        )),
        state,
    )
}

fn handle_spatial_ricci_tensor_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "gamma_ij", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'gamma_ij' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 1, "spatial_coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'spatial_coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'spatial_coords' must be a list of symbols".to_string()),
    };
    expr_response(
        ax_ir::Expr::Matrix(ax_tensor::spatial_ricci_tensor(
            &metric,
            &coords,
            state.interner(),
        )),
        state,
    )
}

fn handle_spatial_ricci_scalar_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "gamma_ij", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'gamma_ij' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 1, "spatial_coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'spatial_coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'spatial_coords' must be a list of symbols".to_string()),
    };
    expr_response(
        ax_tensor::spatial_ricci_scalar(&metric, &coords, state.interner()),
        state,
    )
}

fn handle_null_tetrad_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metric_expr = expr_from_id(args, 0, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let metric = simplify_symbolic_matrix(&metric, state.env(), state.interner());
    let coords_expr = expr_from_id(args, 1, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::null_tetrad_from_metric(&metric, &coords, state.interner()) {
        Ok(tetrad) => expr_response(null_tetrad_to_expr(tetrad), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_verify_null_tetrad_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let tetrad_expr = expr_from_id(args, 0, "tetrad", state)?;
    let tetrad = expr_to_null_tetrad(&tetrad_expr)
        .ok_or_else(|| "argument 'tetrad' must be a 4-list of vector components".to_string())?;
    let metric_expr = expr_from_id(args, 1, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;

    ax_tensor::verify_null_tetrad(&tetrad, &metric, state.interner())
        .map_err(|err| err.to_string())?;
    expr_response(
        ax_ir::Expr::Sym(state.interner().get_or_intern("ok")),
        state,
    )
}

fn handle_spin_coefficients_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let tetrad_expr = expr_from_id(args, 0, "tetrad", state)?;
    let tetrad = expr_to_null_tetrad(&tetrad_expr)
        .ok_or_else(|| "argument 'tetrad' must be a 4-list of vector components".to_string())?;
    let gamma_expr = expr_from_id(args, 1, "gamma", state)?;
    let gamma = expr_to_3d(&gamma_expr)
        .ok_or_else(|| "argument 'gamma' must be a rank-3 list tensor".to_string())?;
    let metric_expr = expr_from_id(args, 2, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;
    let coords_expr = expr_from_id(args, 3, "coords", state)?;
    let coords = match coords_expr {
        ax_ir::Expr::List(items) => items
            .iter()
            .map(|expr| match expr {
                ax_ir::Expr::Sym(sym) => Some(*sym),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "argument 'coords' must be a list of symbols".to_string())?,
        _ => return Err("argument 'coords' must be a list of symbols".to_string()),
    };

    match ax_tensor::spin_coefficients(&tetrad, &gamma, &metric, &coords, state.interner()) {
        Ok(coeffs) => expr_response(spin_coefficients_to_expr(coeffs), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_weyl_scalars_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let weyl_expr = expr_from_id(args, 0, "weyl", state)?;
    let weyl = expr_to_4d(&weyl_expr)
        .ok_or_else(|| "argument 'weyl' must be a rank-4 list tensor".to_string())?;
    let tetrad_expr = expr_from_id(args, 1, "tetrad", state)?;
    let tetrad = expr_to_null_tetrad(&tetrad_expr)
        .ok_or_else(|| "argument 'tetrad' must be a 4-list of vector components".to_string())?;
    let metric_expr = expr_from_id(args, 2, "metric", state)?;
    let metric = matrix_to_symbolic(&metric_expr)
        .ok_or_else(|| "argument 'metric' must be a square matrix".to_string())?;

    match ax_tensor::weyl_scalars(&weyl, &tetrad, &metric, state.interner()) {
        Ok(scalars) => expr_response(weyl_scalars_to_expr(scalars), state),
        Err(err) => Err(err.to_string()),
    }
}

fn handle_petrov_classify_expr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let scalars_expr = expr_from_id(args, 0, "scalars", state)?;
    let scalars = expr_to_weyl_scalars(&scalars_expr)
        .ok_or_else(|| "argument 'scalars' must be a 5-list of Weyl scalars".to_string())?;
    let kind =
        ax_tensor::petrov_classify(&scalars, state.interner()).map_err(|err| err.to_string())?;
    let label = match kind {
        ax_tensor::PetrovType::I => "I",
        ax_tensor::PetrovType::II => "II",
        ax_tensor::PetrovType::D => "D",
        ax_tensor::PetrovType::III => "III",
        ax_tensor::PetrovType::N => "N",
        ax_tensor::PetrovType::O => "O",
    };
    expr_response(
        ax_ir::Expr::Sym(state.interner().get_or_intern(label)),
        state,
    )
}

fn handle_weyl_curvature_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let riemann_id = string_arg(args, 0, "riemann_id")?;
    let riem = state
        .get_riemann(riemann_id)
        .cloned()
        .ok_or_else(|| format!("unknown riemann '{riemann_id}'"))?;
    let ricci = state
        .get_ricci(riemann_id)
        .cloned()
        .ok_or_else(|| format!("unknown ricci '{riemann_id}'"))?;
    let metric = state
        .get_metric(riemann_id)
        .map(|(m, _)| m.clone())
        .ok_or_else(|| format!("unknown metric '{riemann_id}'"))?;
    let scalar = crate::eval(
        &ax_tensor::ricci_scalar(
            &ricci,
            &metric.symbolic_inverse(state.interner()),
            state.interner(),
        ),
        state.env(),
        state.interner(),
    );
    let out = ax_tensor::weyl_from_curvature(&riem, &ricci, &scalar, &metric, state.interner())
        .map_err(|err| err.to_string())?;
    let expr = ax_ir::Expr::List(
        out.into_iter()
            .map(|cube| {
                ax_ir::Expr::List(
                    cube.into_iter()
                        .map(|plane| {
                            ax_ir::Expr::List(plane.into_iter().map(ax_ir::Expr::List).collect())
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    expr_response(expr, state)
}

fn handle_kretschner_scalar_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let id = string_arg(args, 0, "riemann_id")?;
    let riem = state
        .get_riemann(id)
        .cloned()
        .ok_or_else(|| format!("unknown riemann '{id}'"))?;
    let metric = state
        .get_metric(id)
        .map(|(m, _)| m.clone())
        .ok_or_else(|| format!("unknown metric '{id}'"))?;
    let input_expr = ax_ir::Expr::List(
        riem.iter()
            .cloned()
            .map(|cube| {
                ax_ir::Expr::List(
                    cube.into_iter()
                        .map(|plane| {
                            ax_ir::Expr::List(plane.into_iter().map(ax_ir::Expr::List).collect())
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    expr_response_with_change(
        &input_expr,
        crate::eval(
            &ax_tensor::kretschner_scalar(&riem, &metric, state.interner()),
            state.env(),
            state.interner(),
        ),
        "kretschner_scalar",
        state,
    )
}

fn handle_kretschmann_scalar_diagonal_approx_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let id = string_arg(args, 0, "riemann_id")?;
    let riem = state
        .get_riemann(id)
        .cloned()
        .ok_or_else(|| format!("unknown riemann '{id}'"))?;
    let metric = state
        .get_metric(id)
        .map(|(m, _)| m.clone())
        .ok_or_else(|| format!("unknown metric '{id}'"))?;
    let input_expr = ax_ir::Expr::List(
        riem.iter()
            .cloned()
            .map(|cube| {
                ax_ir::Expr::List(
                    cube.into_iter()
                        .map(|plane| {
                            ax_ir::Expr::List(plane.into_iter().map(ax_ir::Expr::List).collect())
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    expr_response_with_change(
        &input_expr,
        crate::eval(
            &ax_tensor::kretschmann_scalar_diagonal_approx(&riem, &metric, state.interner()),
            state.env(),
            state.interner(),
        ),
        "kretschmann_scalar_diagonal_approx",
        state,
    )
}

fn handle_inverse_vielbein_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "vielbein", state)?;
    let rows = matrix_from_expr(&expr)
        .ok_or_else(|| "argument 'vielbein' must reference a matrix expression".to_string())?;
    let matrix = symbolic_matrix_from_rows(rows)?;
    expr_response_with_change(
        &expr,
        ax_ir::Expr::Matrix(ax_tensor::inverse_vielbein(&matrix, state.interner()).data),
        "inverse_vielbein",
        state,
    )
}

fn handle_vielbein_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "vielbein", state)?;
    let rows = matrix_from_expr(&expr)
        .ok_or_else(|| "argument 'vielbein' must reference a matrix expression".to_string())?;
    let _ = symbolic_matrix_from_rows(rows)?;
    expr_response_with_change(&expr, expr.clone(), "vielbein", state)
}

fn handle_metric_from_vielbein_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let e_expr = expr_from_id(args, 0, "vielbein", state)?;
    let eta_expr = expr_from_id(args, 1, "eta", state)?;
    let e =
        symbolic_matrix_from_rows(matrix_from_expr(&e_expr).ok_or_else(|| {
            "argument 'vielbein' must reference a matrix expression".to_string()
        })?)?;
    let eta = symbolic_matrix_from_rows(
        matrix_from_expr(&eta_expr)
            .ok_or_else(|| "argument 'eta' must reference a matrix expression".to_string())?,
    )?;
    expr_response_with_change(
        &e_expr,
        ax_ir::Expr::Matrix(ax_tensor::metric_from_vielbein(&e, &eta, state.interner()).data),
        "metric_from_vielbein",
        state,
    )
}

fn handle_vielbein_from_metric_diagonal_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let g_expr = expr_from_id(args, 0, "metric", state)?;
    let g = symbolic_matrix_from_rows(
        matrix_from_expr(&g_expr)
            .ok_or_else(|| "argument 'metric' must reference a matrix expression".to_string())?,
    )?;
    let signature = match string_arg(args, 1, "signature")? {
        "mostly_plus" => ax_ir::MetricSignature::MostlyPlus,
        "mostly_minus" => ax_ir::MetricSignature::MostlyMinus,
        other => {
            return Err(format!(
                "signature must be 'mostly_plus' or 'mostly_minus', got '{other}'"
            ));
        }
    };
    expr_response_with_change(
        &g_expr,
        ax_ir::Expr::Matrix(
            ax_tensor::vielbein_from_metric_diagonal(&g, signature, state.interner()).data,
        ),
        "vielbein_from_metric_diagonal",
        state,
    )
}

fn handle_covariant_derivative_vector_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let v = list_from_id(args, 0, "vector", state)?;
    let gamma_id = string_arg(args, 1, "christoffel_id")?;
    let coord_index = int_arg(args, 2, "coord_index")? as usize;
    let gamma = state
        .get_christoffel(gamma_id)
        .cloned()
        .ok_or_else(|| format!("unknown christoffel '{gamma_id}'"))?;
    let coords = state
        .get_metric(gamma_id)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("no coordinates recorded for '{gamma_id}'"))?;
    list_response(
        evaluate_list(
            ax_tensor::covariant_derivative_vector(
                &v,
                &gamma,
                coord_index,
                &coords,
                state.interner(),
            ),
            state,
        ),
        state,
    )
}

fn handle_covariant_derivative_covector_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let w = list_from_id(args, 0, "covector", state)?;
    let gamma_id = string_arg(args, 1, "christoffel_id")?;
    let coord_index = int_arg(args, 2, "coord_index")? as usize;
    let gamma = state
        .get_christoffel(gamma_id)
        .cloned()
        .ok_or_else(|| format!("unknown christoffel '{gamma_id}'"))?;
    let coords = state
        .get_metric(gamma_id)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("no coordinates recorded for '{gamma_id}'"))?;
    list_response(
        evaluate_list(
            ax_tensor::covariant_derivative_covector(
                &w,
                &gamma,
                coord_index,
                &coords,
                state.interner(),
            ),
            state,
        ),
        state,
    )
}

fn handle_covariant_derivative_tensor2_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let t = matrix_from_id(args, 0, "tensor", state)?;
    let gamma_id = string_arg(args, 1, "christoffel_id")?;
    let coord_index = int_arg(args, 2, "coord_index")? as usize;
    let gamma = state
        .get_christoffel(gamma_id)
        .cloned()
        .ok_or_else(|| format!("unknown christoffel '{gamma_id}'"))?;
    let coords = state
        .get_metric(gamma_id)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("no coordinates recorded for '{gamma_id}'"))?;
    matrix_response(
        evaluate_matrix(
            ax_tensor::covariant_derivative_tensor2(
                &t,
                &gamma,
                coord_index,
                &coords,
                state.interner(),
            ),
            state,
        ),
        state,
    )
}

fn handle_geodesic_equations_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gamma_id = string_arg(args, 0, "christoffel_id")?;
    let gamma = state
        .get_christoffel(gamma_id)
        .cloned()
        .ok_or_else(|| format!("unknown christoffel '{gamma_id}'"))?;
    let coords = state
        .get_metric(gamma_id)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("no coordinates recorded for '{gamma_id}'"))?;
    list_response(
        evaluate_list(
            ax_tensor::geodesic_equations(&gamma, &coords, state.interner()),
            state,
        ),
        state,
    )
}

fn handle_lie_derivative_scalar_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let f = expr_from_id(args, 0, "expr", state)?;
    let v = list_from_id(args, 1, "vector", state)?;
    let coords = symbol_list_arg(args, 2, "coordinates", state)?;
    expr_response_with_change(
        &f,
        crate::eval(
            &ax_tensor::lie_derivative_scalar(&f, &v, &coords, state.interner()),
            state.env(),
            state.interner(),
        ),
        "lie_derivative_scalar",
        state,
    )
}

fn handle_lie_derivative_vector_gr(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let w = list_from_id(args, 0, "field", state)?;
    let v = list_from_id(args, 1, "vector", state)?;
    let coords = symbol_list_arg(args, 2, "coordinates", state)?;
    list_response(
        evaluate_list(
            ax_tensor::lie_derivative_vector(&w, &v, &coords, state.interner()),
            state,
        ),
        state,
    )
}

fn handle_pauli_x(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(ax_qm::pauli_x(state.interner()), state)
}
fn handle_pauli_y(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(ax_qm::pauli_y(state.interner()), state)
}
fn handle_pauli_z(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(ax_qm::pauli_z(state.interner()), state)
}

fn spin_two_j_arg(args: &[serde_json::Value]) -> Result<usize, String> {
    let two_j = int_arg(args, 0, "two_j")?;
    usize::try_from(two_j)
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())
}

fn handle_jz(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let two_j = spin_two_j_arg(args)?;
    let matrix = ax_qm::jz_matrix(two_j, state.interner())
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())?;
    matrix_response(matrix, state)
}

fn handle_jplus(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let two_j = spin_two_j_arg(args)?;
    let matrix = ax_qm::jplus_matrix(two_j, state.interner())
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())?;
    matrix_response(matrix, state)
}

fn handle_jminus(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let two_j = spin_two_j_arg(args)?;
    let matrix = ax_qm::jminus_matrix(two_j, state.interner())
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())?;
    matrix_response(matrix, state)
}

fn handle_jx(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let two_j = spin_two_j_arg(args)?;
    let matrix = ax_qm::jx_matrix(two_j, state.interner())
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())?;
    matrix_response(matrix, state)
}

fn handle_jy(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let two_j = spin_two_j_arg(args)?;
    let matrix = ax_qm::jy_matrix(two_j, state.interner())
        .map_err(|_| "spin operator constructors expect a nonnegative integer two_j".to_string())?;
    matrix_response(matrix, state)
}

fn handle_singlet_state_2spinhalf(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    list_response(ax_qm::two_spin_half_singlet_state(state.interner()), state)
}

fn handle_triplet_states_2spinhalf(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    list_response(
        ax_qm::two_spin_half_triplet_states(state.interner())
            .into_iter()
            .map(ax_ir::Expr::List)
            .collect(),
        state,
    )
}

fn handle_singlet_projector_2spinhalf(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(
        ax_qm::two_spin_half_singlet_projector(state.interner()),
        state,
    )
}

fn handle_triplet_projector_2spinhalf(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(
        ax_qm::two_spin_half_triplet_projector(state.interner()),
        state,
    )
}

fn handle_time_evolution_operator(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let t = expr_from_id(args, 1, "t", state)?;
    let matrix = ax_qm::time_evolution_operator(&h, t, state.interner()).map_err(|_| {
        "time_evolution_operator expects a supported square Hermitian Hamiltonian".to_string()
    })?;
    matrix_response(matrix, state)
}

fn handle_schrodinger_evolve(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let psi0 = list_from_id(args, 1, "psi0", state)?;
    let t = expr_from_id(args, 2, "t", state)?;
    let evolved = ax_qm::schrodinger_evolve_state(&h, &psi0, t, state.interner()).map_err(
        |_| {
            "schrodinger_evolve expects a supported Hermitian Hamiltonian and a state vector of matching dimension"
                .to_string()
        },
    )?;
    list_response(evolved, state)
}

fn handle_heisenberg_evolve(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let op0 = matrix_from_id(args, 1, "O0", state)?;
    let t = expr_from_id(args, 2, "t", state)?;
    let evolved = ax_qm::heisenberg_evolve_operator(&h, &op0, t, state.interner()).map_err(
        |_| {
            "heisenberg_evolve expects a supported Hermitian Hamiltonian and an operator matrix of matching dimension"
                .to_string()
        },
    )?;
    matrix_response(evolved, state)
}

fn handle_liouville_rhs(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let rhs = ax_qm::liouville_von_neumann_rhs(&h, &rho, state.interner()).map_err(|_| {
        "liouville_rhs expects square Hamiltonian and density matrices of matching dimension"
            .to_string()
    })?;
    matrix_response(rhs, state)
}

fn handle_dyson_series(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h_t = expr_from_id(args, 0, "Ht", state)?;
    let order = int_arg(args, 1, "order")?;
    let Some(order) = usize::try_from(order).ok() else {
        return Err("dyson_series expects a nonnegative integer truncation order".to_string());
    };
    expr_response(ax_qm::dyson_series(h_t, order, state.interner()), state)
}

fn handle_magnus_expansion(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h_t = expr_from_id(args, 0, "Ht", state)?;
    let order = int_arg(args, 1, "order")?;
    let Some(order) = usize::try_from(order).ok() else {
        return Err("magnus_expansion expects a nonnegative integer truncation order".to_string());
    };
    expr_response(ax_qm::magnus_expansion(h_t, order, state.interner()), state)
}

fn handle_kubo_response_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = expr_from_id(args, 0, "A", state)?;
    let b = expr_from_id(args, 1, "B", state)?;
    let rho0 = expr_from_id(args, 2, "rho0", state)?;
    let t = expr_from_id(args, 3, "t", state)?;
    expr_response(
        ax_qm::kubo_response_function(a, b, rho0, t, state.interner()),
        state,
    )
}

fn handle_susceptibility_fourier_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let response = expr_from_id(args, 0, "chi_t", state)?;
    let omega = expr_from_id(args, 1, "omega", state)?;
    expr_response(
        ax_qm::susceptibility_fourier(response, omega, state.interner()),
        state,
    )
}

fn handle_projector_left_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    expr_response(ax_qm::projector_left(state.interner()), state)
}

fn handle_projector_right_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    expr_response(ax_qm::projector_right(state.interner()), state)
}

fn handle_simplify_chiral_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response(
        ax_qm::simplify_chiral_projectors(&expr, &state.env().property_store, state.interner()),
        state,
    )
}

fn handle_simplify_spinor_bilinears_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        ax_qm::simplify_spinor_bilinear_selection_rules(
            &expr,
            &state.env().property_store,
            state.interner(),
        ),
        "simplify_spinor_bilinears",
        state,
    )
}

fn handle_insert_explicit_spinor_indices_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        ax_qm::insert_explicit_spinor_indices(&expr, &state.env().property_store, state.interner()),
        "insert_explicit_spinor_indices",
        state,
    )
}

fn handle_remove_trivial_spinor_indices_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response_with_change(
        &expr,
        ax_qm::remove_trivial_spinor_indices(&expr, &state.env().property_store, state.interner()),
        "remove_trivial_spinor_indices",
        state,
    )
}

fn handle_sigma_matrix_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mu = expr_from_id(args, 0, "mu", state)?;
    let nu = expr_from_id(args, 1, "nu", state)?;
    expr_response(ax_qm::sigma_matrix(mu, nu, state.interner()), state)
}

fn handle_sigma_to_gamma_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response(
        ax_qm::sigma_to_gamma_commutator(&expr, state.interner()),
        state,
    )
}

fn handle_gamma_to_sigma_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_response(
        ax_qm::gamma_commutator_to_sigma(&expr, state.interner()),
        state,
    )
}

fn handle_gamma5(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let _ = args;
    matrix_response(ax_qm::gamma5(state.interner()), state)
}
fn handle_gamma_trace_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("gamma_trace", args, state)
}
fn handle_join_gamma_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("join_gamma", args, state)
}
fn handle_split_gamma_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("split_gamma", args, state)
}
fn handle_expand_diracbar_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("expand_diracbar", args, state)
}
fn handle_sort_spinors_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("sort_spinors", args, state)
}
fn handle_fierz_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("fierz", args, state)
}
fn handle_commutator_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("commutator", args, state)
}
fn handle_anticommutator_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("anticommutator", args, state)
}
fn handle_normal_order_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("normal_order", args, state)
}

fn handle_time_order_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("time_order", args, state)
}

fn handle_anti_time_order_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("anti_time_order", args, state)
}

fn handle_bch_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = expr_from_id(args, 0, "A", state)?;
    let b = expr_from_id(args, 1, "B", state)?;
    let order = int_arg(args, 2, "order")?;
    let order = usize::try_from(order)
        .map_err(|_| "bch expects a nonnegative integer truncation order".to_string())?;
    expr_or_struct_response_named(
        ax_qm::bch_expand(a, b, order, state.interner()),
        "bch",
        state,
    )
}

fn handle_displacement_series_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let alpha = expr_from_id(args, 0, "alpha", state)?;
    let mode = expr_from_id(args, 1, "mode", state)?;
    let order = int_arg(args, 2, "order")?;
    let order = usize::try_from(order).map_err(|_| {
        "displacement_series and squeezing_series expect a nonnegative integer truncation order"
            .to_string()
    })?;
    expr_or_struct_response_named(
        ax_qm::displacement_operator_series(alpha, mode, order, state.interner()),
        "displacement_series",
        state,
    )
}

fn handle_squeezing_series_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let zeta = expr_from_id(args, 0, "zeta", state)?;
    let mode = expr_from_id(args, 1, "mode", state)?;
    let order = int_arg(args, 2, "order")?;
    let order = usize::try_from(order).map_err(|_| {
        "displacement_series and squeezing_series expect a nonnegative integer truncation order"
            .to_string()
    })?;
    expr_or_struct_response_named(
        ax_qm::squeezing_operator_series(zeta, mode, order, state.interner()),
        "squeezing_series",
        state,
    )
}

fn handle_simplify_ccr_car_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("simplify_ccr_car", args, state)
}

fn handle_wick_expand_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    expr_or_struct_response_named(
        ax_qm::wick_expand(
            &expr,
            &state.env().operators,
            &state.env().operator_statistics,
            &state.env().property_store,
            &state.env().contractions,
            state.interner(),
        ),
        "wick",
        state,
    )
}
fn handle_grassmann_simplify_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("grassmann_simplify", args, state)
}

fn handle_number_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = symbol_arg(args, 0, "mode", state)?;
    let n = int_arg(args, 1, "n")?;
    expr_or_struct_response_named(
        call_named(
            "number_state",
            vec![ax_ir::Expr::Sym(mode), ax_ir::Expr::Int(n.into())],
            state,
        ),
        "number_state",
        state,
    )
}

fn handle_fock_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let occupations = expr_from_id(args, 0, "occupations", state)?;
    expr_or_struct_response_named(
        call_named("fock_state", vec![occupations], state),
        "fock_state",
        state,
    )
}

fn handle_fermion_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let occupations = expr_from_id(args, 0, "occupations", state)?;
    expr_or_struct_response_named(
        call_named("fermion_state", vec![occupations], state),
        "fermion_state",
        state,
    )
}

fn handle_bosonic_creation_action_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = int_arg(args, 0, "mode")?;
    let occupations = expr_from_id(args, 1, "occupations", state)?;
    expr_or_struct_response_named(
        call_named(
            "bosonic_creation_action",
            vec![ax_ir::Expr::Int(mode.into()), occupations],
            state,
        ),
        "bosonic_creation_action",
        state,
    )
}

fn handle_fermionic_creation_action_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = int_arg(args, 0, "mode")?;
    let occupations = expr_from_id(args, 1, "occupations", state)?;
    expr_or_struct_response_named(
        call_named(
            "fermionic_creation_action",
            vec![ax_ir::Expr::Int(mode.into()), occupations],
            state,
        ),
        "fermionic_creation_action",
        state,
    )
}

fn handle_bosonic_annihilation_action_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = int_arg(args, 0, "mode")?;
    let occupations = expr_from_id(args, 1, "occupations", state)?;
    expr_or_struct_response_named(
        call_named(
            "bosonic_annihilation_action",
            vec![ax_ir::Expr::Int(mode.into()), occupations],
            state,
        ),
        "bosonic_annihilation_action",
        state,
    )
}

fn handle_fermionic_annihilation_action_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = int_arg(args, 0, "mode")?;
    let occupations = expr_from_id(args, 1, "occupations", state)?;
    expr_or_struct_response_named(
        call_named(
            "fermionic_annihilation_action",
            vec![ax_ir::Expr::Int(mode.into()), occupations],
            state,
        ),
        "fermionic_annihilation_action",
        state,
    )
}

fn handle_bosonic_fock_basis_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let space_symbol = symbol_arg(args, 0, "space_symbol", state)?;
    let occupations = factor_dimensions_arg(args, 1, "occupations").map_err(|_| {
        "bosonic_fock_basis_state occupation list does not match the declared Fock space"
            .to_string()
    })?;
    let expr = crate::bosonic_fock_basis_state_for_space(
        state.env(),
        space_symbol,
        &occupations,
        state.interner(),
    )?;
    expr_or_struct_response_named(expr, "bosonic_fock_basis_state", state)
}

fn handle_fermionic_fock_basis_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let space_symbol = symbol_arg(args, 0, "space_symbol", state)?;
    let occupations = factor_dimensions_arg(args, 1, "occupations").map_err(|_| {
        "fermionic_fock_basis_state occupation list does not match the declared Fock space"
            .to_string()
    })?;
    let expr = crate::fermionic_fock_basis_state_for_space(
        state.env(),
        space_symbol,
        &occupations,
        state.interner(),
    )?;
    expr_or_struct_response_named(expr, "fermionic_fock_basis_state", state)
}

fn handle_vacuum_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = symbol_arg(args, 0, "mode", state)?;
    expr_or_struct_response_named(
        call_named("vacuum", vec![ax_ir::Expr::Sym(mode)], state),
        "vacuum",
        state,
    )
}

fn handle_number_operator_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = symbol_arg(args, 0, "mode", state)?;
    expr_or_struct_response_named(
        call_named("number_operator", vec![ax_ir::Expr::Sym(mode)], state),
        "number_operator",
        state,
    )
}

fn handle_hamiltonian_ho_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mode = symbol_arg(args, 0, "mode", state)?;
    let mut call_args = vec![ax_ir::Expr::Sym(mode)];
    if let Some(arg) = args.get(1) {
        if !arg.is_null() {
            call_args.push(code_expr(args, 1, "hbar", state)?);
        }
    }
    if let Some(arg) = args.get(2) {
        if !arg.is_null() {
            call_args.push(code_expr(args, 2, "omega", state)?);
        }
    }
    expr_or_struct_response_named(
        call_named("hamiltonian_ho", call_args, state),
        "hamiltonian_ho",
        state,
    )
}

fn handle_apply_operator_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let op = expr_from_id(args, 0, "op", state)?;
    let state_expr = expr_from_id(args, 1, "state", state)?;
    expr_or_struct_response_named(
        call_named("apply_operator", vec![op, state_expr], state),
        "apply_operator",
        state,
    )
}

fn handle_density_matrix_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let state_vec = list_from_id(args, 0, "state", state)?;
    matrix_response(ax_qm::density_matrix(&state_vec), state)
}

fn handle_partial_trace_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let dim_a = int_arg(args, 1, "dim_a")? as usize;
    let dim_b = int_arg(args, 2, "dim_b")? as usize;
    let which = string_arg(args, 3, "which")?;
    let which = match which {
        "A" => ax_qm::PartialTraceTarget::A,
        "B" => ax_qm::PartialTraceTarget::B,
        _ => return Err("argument 'which' must be 'A' or 'B'".to_string()),
    };
    let reduced = ax_qm::try_partial_trace(&rho, ax_qm::BipartiteDims { dim_a, dim_b }, which)
        .map_err(|err| match err {
            ax_qm::QmLinearAlgebraError::NonSquareMatrix { .. } => {
                "partial_trace expects a square matrix".to_string()
            }
            ax_qm::QmLinearAlgebraError::SubsystemDimensionMismatch { .. } => {
                "partial_trace matrix dimension does not match dim_a * dim_b".to_string()
            }
            ax_qm::QmLinearAlgebraError::InvalidTraceTarget { .. } => {
                "argument 'which' must be 'A' or 'B'".to_string()
            }
            _ => "partial_trace expects a square matrix".to_string(),
        })?;
    matrix_response(reduced, state)
}

fn handle_partial_trace_factor_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let dims = factor_dimensions_arg(args, 1, "dims").map_err(|_| {
        "partial_trace_factor expects a non-empty factor-dimension list".to_string()
    })?;
    let factor_index = usize::try_from(int_arg(args, 2, "factor_index")?)
        .map_err(|_| "partial_trace_factor factor index is out of range".to_string())?;
    let reduced =
        ax_qm::try_partial_trace_factor(&rho, &dims, factor_index).map_err(|err| match err {
            ax_qm::CompositeSpaceError::EmptyFactorList => {
                "partial_trace_factor expects a non-empty factor-dimension list".to_string()
            }
            ax_qm::CompositeSpaceError::InvalidFactorIndex { .. } => {
                "partial_trace_factor factor index is out of range".to_string()
            }
            ax_qm::CompositeSpaceError::InvalidPermutationLength { .. }
            | ax_qm::CompositeSpaceError::InvalidPermutationEntry { .. }
            | ax_qm::CompositeSpaceError::DuplicatePermutationEntry { .. } => {
                "partial_trace_factor factor index is out of range".to_string()
            }
            ax_qm::CompositeSpaceError::NonSquareMatrix { .. }
            | ax_qm::CompositeSpaceError::TotalDimensionMismatch { .. } => {
                "partial_trace_factor matrix dimension does not match the factor dimensions"
                    .to_string()
            }
        })?;
    matrix_response(reduced, state)
}

fn handle_partial_trace_space_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let composite_space = symbol_arg(args, 1, "composite_space_symbol", state)?;
    let factor_space = symbol_arg(args, 2, "factor_space_symbol", state)?;
    let metadata = hilbert_space_metadata_for_symbol(state, composite_space).ok_or_else(|| {
        "partial_trace_space requires a declared composite Hilbert space".to_string()
    })?;
    if metadata.factors.len() <= 1 {
        return Err("partial_trace_space requires a declared composite Hilbert space".to_string());
    }
    let factor_index = unique_factor_index_in_metadata(&metadata, factor_space)
        .map_err(|message| message.to_string())?;
    let dims = metadata.factor_dimensions();
    let reduced =
        ax_qm::try_partial_trace_factor(&rho, &dims, factor_index).map_err(|err| match err {
            ax_qm::CompositeSpaceError::EmptyFactorList
            | ax_qm::CompositeSpaceError::InvalidFactorIndex { .. } => {
                "partial_trace_space requires a declared composite Hilbert space".to_string()
            }
            ax_qm::CompositeSpaceError::InvalidPermutationLength { .. }
            | ax_qm::CompositeSpaceError::InvalidPermutationEntry { .. }
            | ax_qm::CompositeSpaceError::DuplicatePermutationEntry { .. } => {
                "partial_trace_space requires a declared composite Hilbert space".to_string()
            }
            ax_qm::CompositeSpaceError::NonSquareMatrix { .. }
            | ax_qm::CompositeSpaceError::TotalDimensionMismatch { .. } => {
                "partial_trace_factor matrix dimension does not match the factor dimensions"
                    .to_string()
            }
        })?;
    matrix_response(reduced, state)
}

fn handle_partial_transpose_factor_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let dims = factor_dimensions_arg(args, 1, "dims")
        .map_err(|_| "partial_transpose_factor factor index is out of range".to_string())?;
    let factor_index = usize::try_from(int_arg(args, 2, "factor_index")?)
        .map_err(|_| "partial_transpose_factor factor index is out of range".to_string())?;
    let reduced = ax_qm::try_partial_transpose_factor(&rho, &dims, factor_index).map_err(
        |err| match err {
            ax_qm::CompositeSpaceError::InvalidFactorIndex { .. }
            | ax_qm::CompositeSpaceError::EmptyFactorList => {
                "partial_transpose_factor factor index is out of range".to_string()
            }
            ax_qm::CompositeSpaceError::NonSquareMatrix { .. }
            | ax_qm::CompositeSpaceError::TotalDimensionMismatch { .. } => {
                "partial_trace_factor matrix dimension does not match the factor dimensions"
                    .to_string()
            }
            ax_qm::CompositeSpaceError::InvalidPermutationLength { .. }
            | ax_qm::CompositeSpaceError::InvalidPermutationEntry { .. }
            | ax_qm::CompositeSpaceError::DuplicatePermutationEntry { .. } => {
                "partial_transpose_factor factor index is out of range".to_string()
            }
        },
    )?;
    matrix_response(reduced, state)
}

fn handle_permute_subsystems_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let dims = factor_dimensions_arg(args, 1, "dims").map_err(|_| {
        "permute_subsystems permutation length must match the number of factors".to_string()
    })?;
    let permutation = factor_dimensions_arg(args, 2, "permutation").map_err(|_| {
        "permute_subsystems permutation must contain each factor index exactly once".to_string()
    })?;
    let reduced =
        ax_qm::try_permute_subsystems(&rho, &dims, &permutation).map_err(|err| match err {
            ax_qm::CompositeSpaceError::InvalidPermutationLength { .. } => {
                "permute_subsystems permutation length must match the number of factors".to_string()
            }
            ax_qm::CompositeSpaceError::InvalidPermutationEntry { .. }
            | ax_qm::CompositeSpaceError::DuplicatePermutationEntry { .. } => {
                "permute_subsystems permutation must contain each factor index exactly once"
                    .to_string()
            }
            ax_qm::CompositeSpaceError::EmptyFactorList => {
                "permute_subsystems permutation length must match the number of factors".to_string()
            }
            ax_qm::CompositeSpaceError::InvalidFactorIndex { .. } => {
                "permute_subsystems permutation must contain each factor index exactly once"
                    .to_string()
            }
            ax_qm::CompositeSpaceError::NonSquareMatrix { .. }
            | ax_qm::CompositeSpaceError::TotalDimensionMismatch { .. } => {
                "partial_trace_factor matrix dimension does not match the factor dimensions"
                    .to_string()
            }
        })?;
    matrix_response(reduced, state)
}

fn handle_braket_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let bra = list_from_id(args, 0, "bra", state)?;
    let ket = list_from_id(args, 1, "ket", state)?;
    expr_or_struct_response_named(ax_qm::braket(&bra, &ket), "braket", state)
}

fn handle_outer_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = list_from_id(args, 0, "left", state)?;
    let b = list_from_id(args, 1, "right", state)?;
    matrix_response(ax_qm::outer(&a, &b), state)
}

fn handle_basis_projector_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let index = int_arg(args, 0, "index")? as usize;
    let dim = int_arg(args, 1, "dim")? as usize;
    let projector = ax_qm::basis_projector(index, dim).map_err(|err| err.to_string())?;
    matrix_response(projector, state)
}

fn handle_measurement_probabilities_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let projectors_expr = expr_from_id(args, 0, "projectors", state)?;
    let projectors = expr_to_3d(&projectors_expr)
        .ok_or_else(|| "argument 'projectors' must reference a rank-3 nested list".to_string())?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let probabilities =
        ax_qm::measurement_probabilities(&projectors, &rho).map_err(|err| match err {
            ax_qm::MeasurementError::ProjectorDimensionMismatch { .. }
            | ax_qm::MeasurementError::StateDimensionMismatch { .. } => {
                "measurement projectors must match the state dimension".to_string()
            }
            ax_qm::MeasurementError::ZeroProbabilityOutcome { .. } => {
                "post_measurement_state encountered a zero-probability outcome".to_string()
            }
        })?;
    list_response(probabilities, state)
}

fn handle_expectation_value_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let operator = matrix_from_id(args, 0, "operator", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let value = ax_qm::expectation_value(&operator, &rho).map_err(|_| {
        "expectation_value expects square operator and density matrices of the same dimension"
            .to_string()
    })?;
    expr_or_struct_response_named(value, "expectation_value", state)
}

fn handle_variance_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let operator = matrix_from_id(args, 0, "operator", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let value = ax_qm::variance(&operator, &rho).map_err(|_| {
        "variance expects square operator and density matrices of the same dimension".to_string()
    })?;
    expr_or_struct_response_named(value, "variance", state)
}

fn handle_purity_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value =
        ax_qm::purity(&rho).map_err(|_| "purity expects a square density matrix".to_string())?;
    expr_or_struct_response_named(value, "purity", state)
}

fn handle_linear_entropy_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value = ax_qm::linear_entropy(&rho)
        .map_err(|_| "linear_entropy expects a square density matrix".to_string())?;
    expr_or_struct_response_named(value, "linear_entropy", state)
}

fn handle_participation_ratio_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value = ax_qm::participation_ratio(&rho, state.interner())
        .map_err(|_| "participation_ratio expects a square density matrix".to_string())?;
    expr_or_struct_response_named(value, "participation_ratio", state)
}

fn handle_renyi2_entropy_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value = ax_qm::renyi2_entropy(&rho, state.interner())
        .map_err(|_| "renyi2_entropy expects a square density matrix".to_string())?;
    expr_or_struct_response_named(value, "renyi2_entropy", state)
}

fn handle_renyi2_entropy_factor_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let dims = factor_dimensions_arg(args, 1, "dims").map_err(|_| {
        "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions"
            .to_string()
    })?;
    let kept_factor = usize::try_from(int_arg(args, 2, "kept_factor")?).map_err(|_| {
        "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions"
            .to_string()
    })?;
    let value = ax_qm::renyi2_entropy_factor(&rho, &dims, kept_factor, state.interner())
        .map_err(|_| {
            "renyi2_entropy_factor expects a square matrix whose dimension matches the factor dimensions"
                .to_string()
        })?;
    expr_or_struct_response_named(value, "renyi2_entropy_factor", state)
}

fn handle_renyi2_mutual_information_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_ab = matrix_from_id(args, 0, "rho_ab", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let value = ax_qm::renyi2_mutual_information_bipartite(&rho_ab, dim_a, dim_b, state.interner())
        .map_err(|_| {
            "renyi2_mutual_information matrix dimension does not match dim_a * dim_b".to_string()
        })?;
    expr_or_struct_response_named(value, "renyi2_mutual_information", state)
}

fn handle_renyi2_tripartite_information_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_abc = matrix_from_id(args, 0, "rho", state)?;
    let dim_a = usize::try_from(int_arg(args, 1, "dim_a")?).map_err(|_| {
        "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c"
            .to_string()
    })?;
    let dim_b = usize::try_from(int_arg(args, 2, "dim_b")?).map_err(|_| {
        "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c"
            .to_string()
    })?;
    let dim_c = usize::try_from(int_arg(args, 3, "dim_c")?).map_err(|_| {
        "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c"
            .to_string()
    })?;
    let value = ax_qm::renyi2_tripartite_information(&rho_abc, [dim_a, dim_b, dim_c], state.interner())
        .map_err(|_| {
            "renyi2_tripartite_information expects a tripartite density matrix of dimension dim_a * dim_b * dim_c"
                .to_string()
        })?;
    expr_or_struct_response_named(value, "renyi2_tripartite_information", state)
}

fn handle_von_neumann_entropy_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value = ax_qm::von_neumann_entropy(&rho, state.interner()).map_err(|_| {
        "von_neumann_entropy expects a supported square Hermitian density matrix".to_string()
    })?;
    expr_or_struct_response_named(value, "von_neumann_entropy", state)
}

fn handle_mutual_information_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_ab = matrix_from_id(args, 0, "rho_ab", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let value =
        ax_qm::von_neumann_mutual_information_bipartite(&rho_ab, dim_a, dim_b, state.interner())
            .map_err(|_| {
                "mutual_information expects a bipartite density matrix of dimension dim_a * dim_b"
                    .to_string()
            })?;
    expr_or_struct_response_named(value, "mutual_information", state)
}

fn handle_conditional_entropy_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_ab = matrix_from_id(args, 0, "rho_ab", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let value = ax_qm::conditional_entropy_b_given_a(&rho_ab, dim_a, dim_b, state.interner())
        .map_err(|_| {
            "conditional_entropy expects a bipartite density matrix of dimension dim_a * dim_b"
                .to_string()
        })?;
    expr_or_struct_response_named(value, "conditional_entropy", state)
}

fn handle_entanglement_spectrum_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "state_or_rho", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let values = match &expr {
        ax_ir::Expr::List(state_vector) => {
            ax_qm::entanglement_spectrum_from_state(state_vector, dim_a, dim_b, state.interner())
        }
        ax_ir::Expr::Matrix(rho_ab) => {
            ax_qm::entanglement_spectrum_from_density(rho_ab, dim_a, dim_b, 'A', state.interner())
        }
        _ => Err(ax_qm::EntanglementError::StateDimensionMismatch {
            expected: dim_a.saturating_mul(dim_b),
            actual: 0,
        }),
    }
    .map_err(|_| {
        "entanglement_spectrum expects a bipartite state vector or density matrix of dimension dim_a * dim_b".to_string()
    })?;
    list_response(values, state)
}

fn handle_schmidt_coefficients_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let state_vector = list_from_id(args, 0, "state", state).map_err(|_| {
        "schmidt_coefficients expects a bipartite pure-state vector of dimension dim_a * dim_b"
            .to_string()
    })?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let values = ax_qm::schmidt_coefficients_from_state(
        &state_vector,
        dim_a,
        dim_b,
        state.interner(),
    )
    .map_err(|_| {
        "schmidt_coefficients expects a bipartite pure-state vector of dimension dim_a * dim_b"
            .to_string()
    })?;
    list_response(values, state)
}

fn handle_negativity_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_ab = matrix_from_id(args, 0, "rho_ab", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let value =
        ax_qm::negativity_bipartite(&rho_ab, dim_a, dim_b, 1, state.interner()).map_err(|_| {
            "negativity expects a bipartite density matrix of dimension dim_a * dim_b".to_string()
        })?;
    expr_or_struct_response_named(value, "negativity", state)
}

fn handle_logarithmic_negativity_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho_ab = matrix_from_id(args, 0, "rho_ab", state)?;
    let dim_a = int_arg(args, 1, "dim_a").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_a' must be non-negative".to_string())
    })?;
    let dim_b = int_arg(args, 2, "dim_b").and_then(|n| {
        usize::try_from(n).map_err(|_| "argument 'dim_b' must be non-negative".to_string())
    })?;
    let value = ax_qm::logarithmic_negativity_bipartite(&rho_ab, dim_a, dim_b, 1, state.interner())
        .map_err(|_| {
            "logarithmic_negativity expects a bipartite density matrix of dimension dim_a * dim_b"
                .to_string()
        })?;
    expr_or_struct_response_named(value, "logarithmic_negativity", state)
}

fn handle_bloch_vector_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let rho = matrix_from_id(args, 0, "rho", state)?;
    let value = ax_qm::bloch_vector(&rho)
        .map_err(|_| "bloch_vector expects a 2x2 density matrix".to_string())?;
    list_response(value.into_iter().collect(), state)
}

fn handle_qubit_density_from_bloch_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let vector_expr = expr_from_id(args, 0, "r", state)?;
    let ax_ir::Expr::List(items) = vector_expr else {
        return Err("qubit_density_from_bloch expects a length-3 list".to_string());
    };
    let [x, y, z] = items.as_slice() else {
        return Err("qubit_density_from_bloch expects a length-3 list".to_string());
    };
    let matrix = ax_qm::qubit_density_from_bloch([x.clone(), y.clone(), z.clone()]);
    matrix_response(matrix, state)
}

fn handle_post_measurement_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let projector = matrix_from_id(args, 0, "projector", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let outcome_index = int_arg(args, 2, "outcome_index")? as usize;
    let state_out = ax_qm::post_measurement_state(&projector, &rho, outcome_index).map_err(
        |err| match err {
            ax_qm::MeasurementError::ProjectorDimensionMismatch { .. }
            | ax_qm::MeasurementError::StateDimensionMismatch { .. } => {
                "measurement projectors must match the state dimension".to_string()
            }
            ax_qm::MeasurementError::ZeroProbabilityOutcome { .. } => {
                "post_measurement_state encountered a zero-probability outcome".to_string()
            }
        },
    )?;
    matrix_response(state_out, state)
}

fn handle_identity_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let dim = int_arg(args, 0, "dim")? as usize;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::identity_channel(dim)),
        "identity_channel",
        state,
    )
}

fn handle_depolarizing_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let p = expr_from_id(args, 0, "p", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::depolarizing_channel_qubit(p, state.interner())),
        "depolarizing_channel",
        state,
    )
}

fn handle_dephasing_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let p = expr_from_id(args, 0, "p", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::dephasing_channel_qubit(p, state.interner())),
        "dephasing_channel",
        state,
    )
}

fn handle_amplitude_damping_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gamma = expr_from_id(args, 0, "gamma", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::amplitude_damping_channel_qubit(
            gamma,
            state.interner(),
        )),
        "amplitude_damping_channel",
        state,
    )
}

fn handle_bit_flip_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let p = expr_from_id(args, 0, "p", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::bit_flip_channel_qubit(p, state.interner())),
        "bit_flip_channel",
        state,
    )
}

fn handle_phase_flip_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let p = expr_from_id(args, 0, "p", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::phase_flip_channel_qubit(p, state.interner())),
        "phase_flip_channel",
        state,
    )
}

fn handle_bit_phase_flip_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let p = expr_from_id(args, 0, "p", state)?;
    expr_or_struct_response_named(
        kraus_list_to_expr(ax_qm::bit_phase_flip_channel_qubit(p, state.interner())),
        "bit_phase_flip_channel",
        state,
    )
}

fn handle_compose_channels_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = crate::diagnostics::qm_diag(
        ax_diagnostics::QuantumDiagnosticKind::InvalidChannel,
        "compose_channels expects two non-empty Kraus lists of square matrices with matching dimension",
    )
    .to_string();
    let left_expr = expr_from_id(args, 0, "left", state)?;
    let right_expr = expr_from_id(args, 1, "right", state)?;
    let left = expr_to_kraus_list(&left_expr).ok_or_else(|| error.clone())?;
    let right = expr_to_kraus_list(&right_expr).ok_or_else(|| error.clone())?;
    let result = ax_qm::compose_kraus_channels(&left, &right).map_err(|_| error.clone())?;
    expr_or_struct_response_named(kraus_list_to_expr(result), "compose_channels", state)
}

fn handle_tensor_product_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "tensor_product_channel expects two non-empty Kraus lists of square matrices";
    let left_expr = expr_from_id(args, 0, "left", state)?;
    let right_expr = expr_from_id(args, 1, "right", state)?;
    let left = expr_to_kraus_list(&left_expr).ok_or_else(|| error.to_string())?;
    let right = expr_to_kraus_list(&right_expr).ok_or_else(|| error.to_string())?;
    let result =
        ax_qm::tensor_product_kraus_channels(&left, &right).map_err(|_| error.to_string())?;
    expr_or_struct_response_named(kraus_list_to_expr(result), "tensor_product_channel", state)
}

fn handle_choi_distance_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = crate::diagnostics::qm_diag(
        ax_diagnostics::QuantumDiagnosticKind::InvalidChannel,
        "choi_distance expects two channels of matching dimension",
    )
    .to_string();
    let left_expr = expr_from_id(args, 0, "left", state)?;
    let right_expr = expr_from_id(args, 1, "right", state)?;
    let left = expr_to_kraus_list(&left_expr).ok_or_else(|| error.clone())?;
    let right = expr_to_kraus_list(&right_expr).ok_or_else(|| error.clone())?;
    let distance = ax_qm::choi_frobenius_distance(&left, &right, state.interner())
        .map_err(|_| error.clone())?;
    expr_or_struct_response_named(distance, "choi_distance", state)
}

fn handle_trace_preserving_residual_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "trace_preserving_residual expects a non-empty Kraus list of square matrices";
    let kraus_expr = expr_from_id(args, 0, "kraus", state)?;
    let kraus = expr_to_kraus_list(&kraus_expr).ok_or_else(|| error.to_string())?;
    let residual = ax_qm::trace_preserving_residual(&kraus, state.interner())
        .map_err(|_| error.to_string())?;
    expr_or_struct_response_named(Expr::Matrix(residual), "trace_preserving_residual", state)
}

fn handle_is_trace_preserving_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "is_trace_preserving expects a non-empty Kraus list of square matrices";
    let kraus_expr = expr_from_id(args, 0, "kraus", state)?;
    let kraus = expr_to_kraus_list(&kraus_expr).ok_or_else(|| error.to_string())?;
    let value = ax_qm::is_trace_preserving_exact(&kraus, state.interner())
        .map_err(|_| error.to_string())?;
    expr_or_struct_response_named(
        Expr::Sym(
            state
                .interner_mut()
                .get_or_intern(if value { "true" } else { "false" }),
        ),
        "is_trace_preserving",
        state,
    )
}

fn handle_unital_residual_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "unital_residual expects a non-empty Kraus list of square matrices";
    let kraus_expr = expr_from_id(args, 0, "kraus", state)?;
    let kraus = expr_to_kraus_list(&kraus_expr).ok_or_else(|| error.to_string())?;
    let residual =
        ax_qm::unital_residual(&kraus, state.interner()).map_err(|_| error.to_string())?;
    expr_or_struct_response_named(Expr::Matrix(residual), "unital_residual", state)
}

fn handle_is_unital_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "is_unital expects a non-empty Kraus list of square matrices";
    let kraus_expr = expr_from_id(args, 0, "kraus", state)?;
    let kraus = expr_to_kraus_list(&kraus_expr).ok_or_else(|| error.to_string())?;
    let value = ax_qm::is_unital_exact(&kraus, state.interner()).map_err(|_| error.to_string())?;
    expr_or_struct_response_named(
        Expr::Sym(
            state
                .interner_mut()
                .get_or_intern(if value { "true" } else { "false" }),
        ),
        "is_unital",
        state,
    )
}

fn handle_apply_channel_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let kraus_expr = expr_from_id(args, 0, "kraus", state)?;
    let kraus = expr_to_kraus_list(&kraus_expr)
        .ok_or_else(|| "argument 'kraus' must reference a rank-3 nested list".to_string())?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let result = ax_qm::apply_kraus_channel(&kraus, &rho).map_err(|err| match err {
        ax_qm::ChannelError::EmptyKrausSet => {
            "apply_channel expects a non-empty Kraus list".to_string()
        }
        ax_qm::ChannelError::InvalidKrausSet => {
            "apply_channel Kraus list must describe a non-empty positive-dimensional channel"
                .to_string()
        }
        ax_qm::ChannelError::UnsupportedChoiRecovery
        | ax_qm::ChannelError::InvalidChoiDimension { .. }
        | ax_qm::ChannelError::NonNumericChoiMatrix
        | ax_qm::ChannelError::UnsupportedCompletePositivityCheck { .. } => {
            "apply_channel received an unexpected Choi-recovery error".to_string()
        }
        ax_qm::ChannelError::NonSquareKraus { .. }
        | ax_qm::ChannelError::KrausDimensionMismatch { .. } => crate::diagnostics::qm_diag(
            ax_diagnostics::QuantumDiagnosticKind::InvalidChannel,
            "apply_channel Kraus operators must be square and share a common dimension",
        )
        .to_string(),
        ax_qm::ChannelError::CompositionDimensionMismatch { .. } => crate::diagnostics::qm_diag(
            ax_diagnostics::QuantumDiagnosticKind::InvalidChannel,
            "apply_channel received an unexpected channel composition error",
        )
        .to_string(),
        ax_qm::ChannelError::StateDimensionMismatch { .. } => crate::diagnostics::qm_diag(
            ax_diagnostics::QuantumDiagnosticKind::InvalidChannel,
            "apply_channel state dimension does not match Kraus dimension",
        )
        .to_string(),
    })?;
    matrix_response(result, state)
}

fn handle_lindblad_rhs_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let jumps_expr = expr_from_id(args, 2, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let result =
        ax_qm::lindblad_rhs(&h, &rho, &jump_ops, state.interner()).map_err(|err| match err {
            ax_qm::LindbladError::HamiltonianNotSquare { .. }
            | ax_qm::LindbladError::StateNotSquare { .. } => {
                "lindblad_rhs expects square Hamiltonian and density matrices".to_string()
            }
            ax_qm::LindbladError::DimensionMismatch { .. } => {
                "lindblad_rhs operator dimensions do not agree".to_string()
            }
        })?;
    matrix_response(result, state)
}

fn handle_lindblad_euler_step_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let jumps_expr = expr_from_id(args, 2, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let dt = expr_from_id(args, 3, "dt", state)?;
    let result =
        ax_ode::lindblad_euler_step(&h, &rho, &jump_ops, &dt, state.interner()).map_err(|err| {
            match err {
                ax_ode::QuantumOdeError::ZeroTimeStep => {
                    "lindblad step expects a nonzero dt".to_string()
                }
                ax_ode::QuantumOdeError::Lindblad(_)
                | ax_ode::QuantumOdeError::Liouville(_)
                | ax_ode::QuantumOdeError::StateEvolutionDimensionMismatch => {
                    "lindblad step expects square operators with matching dimensions".to_string()
                }
            }
        })?;
    matrix_response(result, state)
}

fn handle_lindblad_rk4_step_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let rho = matrix_from_id(args, 1, "rho", state)?;
    let jumps_expr = expr_from_id(args, 2, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let dt = expr_from_id(args, 3, "dt", state)?;
    let result =
        ax_ode::lindblad_rk4_step(&h, &rho, &jump_ops, &dt, state.interner()).map_err(|err| {
            match err {
                ax_ode::QuantumOdeError::ZeroTimeStep => {
                    "lindblad step expects a nonzero dt".to_string()
                }
                ax_ode::QuantumOdeError::Lindblad(_)
                | ax_ode::QuantumOdeError::Liouville(_)
                | ax_ode::QuantumOdeError::StateEvolutionDimensionMismatch => {
                    "lindblad step expects square operators with matching dimensions".to_string()
                }
            }
        })?;
    matrix_response(result, state)
}

fn handle_lindblad_steady_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let jumps_expr = expr_from_id(args, 1, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let result =
        ax_solve::lindblad_steady_state_linear(&h, &jump_ops, state.interner()).map_err(|err| {
            match err {
                ax_solve::LindbladSteadyStateError::HamiltonianNotSquare { .. }
                | ax_solve::LindbladSteadyStateError::JumpOperatorNotSquare { .. }
                | ax_solve::LindbladSteadyStateError::DimensionMismatch { .. } => {
                    "lindblad_steady_state expects a square Hamiltonian and square jump operators of matching dimension".to_string()
                }
                ax_solve::LindbladSteadyStateError::UnderdeterminedSteadyState => {
                    "lindblad_steady_state generator has non-unique steady states".to_string()
                }
                ax_solve::LindbladSteadyStateError::InconsistentSteadyStateSystem => {
                    "lindblad_steady_state system is inconsistent".to_string()
                }
            }
    })?;
    matrix_response(result, state)
}

fn handle_lindbladian_superoperator_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let jumps_expr = expr_from_id(args, 1, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let result =
        ax_qm::lindbladian_superoperator(&h, &jump_ops, state.interner()).map_err(|_| {
            "lindbladian_superoperator expects square operators with matching dimensions"
                .to_string()
        })?;
    matrix_response(result, state)
}

fn handle_lindbladian_eigenvalues_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let jumps_expr = expr_from_id(args, 1, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let result =
        ax_qm::lindbladian_eigenvalues_small(&h, &jump_ops, state.interner()).map_err(|_| {
            "lindbladian_eigenvalues currently supports only low-dimensional cases".to_string()
        })?;
    expr_or_struct_response_named(Expr::List(result), "lindbladian_eigenvalues", state)
}

fn handle_sparse_steady_state_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let jumps_expr = expr_from_id(args, 1, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let tolerance = float_arg(args, 2, "tolerance")?;
    let max_iterations = usize::try_from(int_arg(args, 3, "max_iterations")?)
        .map_err(|_| "argument 'max_iterations' must be a nonnegative integer".to_string())?;
    let result = crate::sparse_steady_state_expr(
        &h,
        &jump_ops,
        tolerance,
        max_iterations,
        state.interner(),
    )?;
    expr_or_struct_response_named(result, "sparse_steady_state", state)
}

fn handle_sparse_lindbladian_spectrum_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h = matrix_from_id(args, 0, "H", state)?;
    let jumps_expr = expr_from_id(args, 1, "jumps", state)?;
    let jump_ops = expr_to_3d(&jumps_expr)
        .ok_or_else(|| "argument 'jumps' must reference a rank-3 nested list".to_string())?;
    let k = usize::try_from(int_arg(args, 2, "k")?)
        .map_err(|_| "argument 'k' must be a nonnegative integer".to_string())?;
    let which = string_arg(args, 3, "which")?;
    let tolerance = float_arg(args, 4, "tolerance")?;
    let max_iterations = usize::try_from(int_arg(args, 5, "max_iterations")?)
        .map_err(|_| "argument 'max_iterations' must be a nonnegative integer".to_string())?;
    let result = crate::sparse_lindbladian_spectrum_expr(
        &h,
        &jump_ops,
        k,
        which,
        tolerance,
        max_iterations,
        state.interner(),
    )?;
    expr_or_struct_response_named(result, "sparse_lindbladian_spectrum", state)
}

fn handle_wedge_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("wedge_1_1", args, state)
}
fn handle_exterior_derivative_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("exterior_d", args, state)
}
fn handle_hodge_dual_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("hodge_star", args, state)
}

fn handle_codifferential_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let form = expr_from_id(args, 0, "form", state)?;
    let metric = expr_from_id(args, 1, "metric", state)?;
    let coords = symbol_list_arg(args, 2, "coords", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    expr_or_struct_response_named(
        call_named(
            "codifferential",
            vec![form, metric, ax_ir::Expr::List(coords)],
            state,
        ),
        "codifferential",
        state,
    )
}

fn handle_interior_product_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let vector = expr_from_id(args, 0, "vector", state)?;
    let form = expr_from_id(args, 1, "form", state)?;
    expr_or_struct_response_named(
        call_named("interior_product", vec![vector, form], state),
        "interior_product",
        state,
    )
}

fn handle_lie_derivative_form_forms(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let form = expr_from_id(args, 0, "form", state)?;
    let vector = expr_from_id(args, 1, "vector", state)?;
    let coords = symbol_list_arg(args, 2, "coords", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect::<Vec<_>>();
    expr_or_struct_response_named(
        call_named(
            "lie_derivative_form",
            vec![form, vector, ax_ir::Expr::List(coords)],
            state,
        ),
        "lie_derivative_form",
        state,
    )
}

fn handle_functional_derivative_variational(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lagrangian = expr_from_id(args, 0, "lagrangian", state)?;
    let field = symbol_arg(args, 1, "field", state)?;
    let field_derivs = symbol_list_arg(args, 2, "field_derivatives", state)?;
    let coords = symbol_list_arg(args, 3, "coordinates", state)?;
    expr_response_with_change(
        &lagrangian,
        ax_variational::functional_derivative(
            &lagrangian,
            field,
            &field_derivs,
            &coords,
            state.interner(),
        ),
        "functional_derivative",
        state,
    )
}

fn handle_euler_lagrange_system_variational(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lagrangian = expr_from_id(args, 0, "lagrangian", state)?;
    let fields = require_arg(args, 1, "fields")?
        .as_array()
        .ok_or_else(|| "argument 'fields' must be an array".to_string())?
        .iter()
        .map(|item| {
            let pair = item
                .as_array()
                .ok_or_else(|| "each field entry must be [field, derivs]".to_string())?;
            if pair.len() != 2 {
                return Err("each field entry must be [field, derivs]".to_string());
            }
            let field = pair[0]
                .as_str()
                .ok_or_else(|| "field name must be a string".to_string())?;
            let derivs = pair[1]
                .as_array()
                .ok_or_else(|| "derivative list must be an array".to_string())?;
            let derivs = derivs
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| state.interner_mut().get_or_intern(s))
                        .ok_or_else(|| "derivative names must be strings".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((state.interner_mut().get_or_intern(field), derivs))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let coords = symbol_list_arg(args, 2, "coordinates", state)?;
    list_response(
        ax_variational::euler_lagrange_system(&lagrangian, &fields, &coords, state.interner()),
        state,
    )
}

fn handle_vary_action_variational(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lagrangian = expr_from_id(args, 0, "lagrangian", state)?;
    let field = symbol_arg(args, 1, "field", state)?;
    let variation = symbol_arg(args, 2, "variation", state)?;
    let field_derivs = symbol_list_arg(args, 3, "field_derivatives", state)?;
    let variation_derivs = symbol_list_arg(args, 4, "variation_derivatives", state)?;
    expr_response_with_change(
        &lagrangian,
        ax_variational::vary_action(
            &lagrangian,
            field,
            variation,
            &field_derivs,
            &variation_derivs,
            state.interner(),
        ),
        "vary_action",
        state,
    )
}

fn handle_solve_general(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let equation = code_expr(args, 0, "equation", state)?;
    let var = symbol_arg(args, 1, "variable", state)?;
    expr_or_struct_response_named(
        ax_solve::solve(&equation, var, state.interner()),
        "solve",
        state,
    )
}

fn handle_solve_linear_system_general(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let equations = string_list_arg(args, 0, "equations")?
        .into_iter()
        .map(|code| state.parse_code(&code))
        .collect::<Result<Vec<_>, _>>()?;
    let vars = symbol_list_arg(args, 1, "variables", state)?;
    let solutions =
        ax_solve::solve_linear_system(&equations, &vars, state.interner()).unwrap_or_default();
    let expr = ax_ir::Expr::List(
        solutions
            .into_iter()
            .map(|(sym, rhs)| {
                ax_ir::Expr::Rule(
                    Box::new(ax_ir::Expr::Sym(sym)),
                    Box::new(rhs),
                    ax_ir::TrustLevel::Exact,
                )
            })
            .collect(),
    );
    expr_or_struct_response_named(expr, "solve_linear_system", state)
}

fn handle_solve_ode_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let equation = expr_from_id(args, 0, "equation", state)?;
    let dependent = symbol_arg(args, 1, "dependent", state)?;
    let independent = symbol_arg(args, 2, "independent", state)?;
    expr_response_with_change(
        &equation,
        ax_ode::solve_ode(&equation, dependent, independent, state.interner()),
        "solve_ode",
        state,
    )
}

fn handle_rk4_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let f = expr_from_id(args, 0, "f", state)?;
    let x = symbol_arg(args, 1, "x", state)?;
    let y = symbol_arg(args, 2, "y", state)?;
    let x0 = float_arg(args, 3, "x0")?;
    let y0 = float_arg(args, 4, "y0")?;
    let x_end = float_arg(args, 5, "x_end")?;
    let steps = args.get(6).and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
    points_response(ax_ode::rk4(
        &f,
        x,
        y,
        x0,
        y0,
        x_end,
        steps,
        state.interner(),
    ))
}

fn handle_rk4_system_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let fs = list_from_id(args, 0, "functions", state)?;
    let x = symbol_arg(args, 1, "independent", state)?;
    let ys = symbol_list_arg(args, 2, "dependents", state)?;
    let x0 = float_arg(args, 3, "x0")?;
    let y0s = require_arg(args, 4, "y0s")?
        .as_array()
        .ok_or_else(|| "argument 'y0s' must be an array of floats".to_string())?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "initial values must be numeric".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_end = float_arg(args, 5, "x_end")?;
    let steps = args.get(6).and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
    let values = ax_ode::rk4_system(&fs, x, &ys, x0, &y0s, x_end, steps, state.interner());
    Ok(serde_json::json!({ "status": "ok", "values": values }))
}

fn handle_first_order_form_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let ode = expr_from_id(args, 0, "ode", state)?;
    let dependent = symbol_arg(args, 1, "dependent", state)?;
    let independent = symbol_arg(args, 2, "independent", state)?;
    let system = ax_ode::first_order_form(&ode, dependent, independent, state.interner());
    let expr = ax_ir::Expr::List(
        system
            .into_iter()
            .map(|(lhs, rhs)| ax_ir::Expr::List(vec![lhs, rhs]))
            .collect(),
    );
    expr_or_struct_response_with_change(&ode, expr, "first_order_form", state)
}

fn handle_parallel_transport_native_only(
    _args: &[serde_json::Value],
    _state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Err("parallel_transport is currently a library-level API that requires a native numeric Christoffel callback and is not exposed through ordinary source syntax".to_string())
}

fn handle_integrate_geodesic_native_only(
    _args: &[serde_json::Value],
    _state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Err("integrate_geodesic is currently a library-level API that requires a native numeric Christoffel callback and is not exposed through ordinary source syntax".to_string())
}

fn handle_classify_pde_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = expr_from_id(args, 0, "A", state)?;
    let b = expr_from_id(args, 1, "B", state)?;
    let c = expr_from_id(args, 2, "C", state)?;
    let kind = match ax_ode::classify_pde(&a, &b, &c, state.interner()) {
        ax_ode::PdeType::Elliptic => "Elliptic",
        ax_ode::PdeType::Parabolic => "Parabolic",
        ax_ode::PdeType::Hyperbolic => "Hyperbolic",
        ax_ode::PdeType::Unknown => "Unknown",
    };
    Ok(serde_json::json!({ "status": "ok", "kind": kind }))
}

fn handle_separate_variables_ode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let kind = string_arg(args, 0, "pde_type")?;
    let pde_type = match kind.to_ascii_lowercase().as_str() {
        "wave" | "hyperbolic" => ax_ode::PdeType::Hyperbolic,
        "heat" | "parabolic" | "diffusion" => ax_ode::PdeType::Parabolic,
        "laplace" | "elliptic" => ax_ode::PdeType::Elliptic,
        _ => ax_ode::PdeType::Unknown,
    };
    let x = symbol_arg(args, 1, "spatial", state)?;
    let t = symbol_arg(args, 2, "temporal", state)?;
    let coeff = if args.len() > 3 {
        code_expr(args, 3, "coefficient", state)?
    } else {
        ax_ir::Expr::one()
    };
    let sol = ax_ode::separate_variables(pde_type, x, t, &coeff, state.interner());
    list_response(
        vec![sol.spatial, sol.temporal, sol.separation_constant],
        state,
    )
}

fn handle_determinant_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    expr_response_with_change(
        &ax_ir::Expr::Matrix(matrix.clone()),
        ax_linalg::determinant(&matrix, state.interner()),
        "determinant",
        state,
    )
}

fn handle_inverse_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    let inv = ax_linalg::inverse(&matrix, state.interner())
        .ok_or_else(|| "matrix is singular".to_string())?;
    matrix_response(inv, state)
}

fn handle_trace_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    expr_response_with_change(
        &ax_ir::Expr::Matrix(matrix.clone()),
        ax_linalg::trace(&matrix),
        "trace",
        state,
    )
}

fn handle_eigenvalues_symbolic_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    expr_response_with_change(
        &ax_ir::Expr::Matrix(matrix.clone()),
        ax_linalg::eigenvalues_symbolic(&matrix, state.interner()),
        "eigenvalues_symbolic",
        state,
    )
}

fn handle_hermitian_eigenvalues_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    let values = ax_qm::hermitian_eigenvalues_small(&matrix, state.interner()).map_err(|_| {
        "hermitian_eigenvalues expects a square Hermitian matrix of supported dimension".to_string()
    })?;
    list_response(values, state)
}

fn handle_hermitian_eigenprojectors_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    let projectors =
        ax_qm::hermitian_eigenprojectors_small(&matrix, state.interner()).map_err(|_| {
            "hermitian_eigenprojectors expects a square Hermitian matrix of supported dimension with nondegenerate spectrum".to_string()
        })?;
    list_response(
        projectors.into_iter().map(ax_ir::Expr::Matrix).collect(),
        state,
    )
}

fn handle_first_order_energy_shift_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h0 = matrix_from_id(args, 0, "H0", state)?;
    let v = matrix_from_id(args, 1, "V", state)?;
    let state_index = usize::try_from(int_arg(args, 2, "n")?)
        .map_err(|_| "argument 'n' must be non-negative".to_string())?;
    let shift = ax_qm::first_order_energy_shift(&h0, &v, state_index, state.interner()).map_err(
        |_| {
            "perturbation-theory helpers expect supported nondegenerate Hermitian H0 and matching-dimension V"
                .to_string()
        },
    )?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state
                .interner_mut()
                .get_or_intern("first_order_energy_shift"),
            vec![
                ax_ir::Expr::Matrix(h0.clone()),
                ax_ir::Expr::Matrix(v.clone()),
                ax_ir::Expr::Int((state_index as i64).into()),
            ],
        ),
        shift,
        "first_order_energy_shift",
        state,
    )
}

fn handle_second_order_energy_shift_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h0 = matrix_from_id(args, 0, "H0", state)?;
    let v = matrix_from_id(args, 1, "V", state)?;
    let state_index = usize::try_from(int_arg(args, 2, "n")?)
        .map_err(|_| "argument 'n' must be non-negative".to_string())?;
    let shift =
        ax_qm::second_order_energy_shift(&h0, &v, state_index, state.interner()).map_err(
            |_| {
                "perturbation-theory helpers expect supported nondegenerate Hermitian H0 and matching-dimension V"
                    .to_string()
            },
        )?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state
                .interner_mut()
                .get_or_intern("second_order_energy_shift"),
            vec![
                ax_ir::Expr::Matrix(h0.clone()),
                ax_ir::Expr::Matrix(v.clone()),
                ax_ir::Expr::Int((state_index as i64).into()),
            ],
        ),
        shift,
        "second_order_energy_shift",
        state,
    )
}

fn handle_degenerate_effective_perturbation_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h0 = matrix_from_id(args, 0, "H0", state)?;
    let v = matrix_from_id(args, 1, "V", state)?;
    let subspace_expr = expr_from_id(args, 2, "subspace", state)?;
    let subspace = list_from_expr(&subspace_expr)
        .and_then(|items| {
            items.into_iter()
                .map(|item| match item {
                    ax_ir::Expr::Int(value) => value.to_usize(),
                    _ => None,
                })
                .collect()
        })
        .filter(|values: &Vec<usize>| !values.is_empty())
        .ok_or_else(|| {
            "degenerate perturbation theory expects a non-empty list of degenerate basis-state indices"
                .to_string()
        })?;
    let effective = ax_qm::degenerate_subspace_effective_perturbation(
        &h0,
        &v,
        &subspace,
        state.interner(),
    )
    .map_err(|err| match err {
        ax_qm::PerturbationError::SelectedSubspaceNotDegenerate => {
            "selected subspace is not degenerate in H0".to_string()
        }
        _ => {
            "degenerate perturbation theory expects a non-empty list of degenerate basis-state indices"
                .to_string()
        }
    })?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state
                .interner_mut()
                .get_or_intern("degenerate_effective_perturbation"),
            vec![
                ax_ir::Expr::Matrix(h0.clone()),
                ax_ir::Expr::Matrix(v.clone()),
                subspace_expr.clone(),
            ],
        ),
        ax_ir::Expr::Matrix(effective),
        "degenerate_effective_perturbation",
        state,
    )
}

fn handle_degenerate_first_order_splittings_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let h0 = matrix_from_id(args, 0, "H0", state)?;
    let v = matrix_from_id(args, 1, "V", state)?;
    let subspace_expr = expr_from_id(args, 2, "subspace", state)?;
    let subspace = list_from_expr(&subspace_expr)
        .and_then(|items| {
            items.into_iter()
                .map(|item| match item {
                    ax_ir::Expr::Int(value) => value.to_usize(),
                    _ => None,
                })
                .collect()
        })
        .filter(|values: &Vec<usize>| !values.is_empty())
        .ok_or_else(|| {
            "degenerate perturbation theory expects a non-empty list of degenerate basis-state indices"
                .to_string()
        })?;
    let splittings =
        ax_qm::degenerate_first_order_splittings(&h0, &v, &subspace, state.interner()).map_err(
            |err| match err {
                ax_qm::PerturbationError::SelectedSubspaceNotDegenerate => {
                    "selected subspace is not degenerate in H0".to_string()
                }
                _ => {
                    "degenerate perturbation theory expects a non-empty list of degenerate basis-state indices"
                        .to_string()
                }
            },
        )?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state
                .interner_mut()
                .get_or_intern("degenerate_first_order_splittings"),
            vec![
                ax_ir::Expr::Matrix(h0.clone()),
                ax_ir::Expr::Matrix(v.clone()),
                subspace_expr.clone(),
            ],
        ),
        ax_ir::Expr::List(splittings),
        "degenerate_first_order_splittings",
        state,
    )
}

fn handle_berry_connection_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let psi = expr_from_id(args, 0, "psi", state)?;
    let parameter = expr_from_id(args, 1, "parameter", state)?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state.interner_mut().get_or_intern("berry_connection"),
            vec![psi.clone(), parameter.clone()],
        ),
        ax_qm::berry_connection(psi, parameter, state.interner()),
        "berry_connection",
        state,
    )
}

fn handle_geometric_phase_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = expr_from_id(args, 0, "A", state)?;
    let parameter = expr_from_id(args, 1, "parameter", state)?;
    expr_response_with_change(
        &ax_ir::Expr::Call(
            state.interner_mut().get_or_intern("geometric_phase"),
            vec![a.clone(), parameter.clone()],
        ),
        ax_qm::geometric_phase(a, parameter, state.interner()),
        "geometric_phase",
        state,
    )
}

fn handle_tensor_product_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = matrix_from_id(args, 0, "a", state)?;
    let b = matrix_from_id(args, 1, "b", state)?;
    matrix_response(ax_linalg::tensor_product(&a, &b), state)
}

fn handle_matmul_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let a = matrix_from_id(args, 0, "a", state)?;
    let b = matrix_from_id(args, 1, "b", state)?;
    matrix_response(ax_linalg::mat_mul(&a, &b, state.interner()), state)
}

fn handle_transpose_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let matrix = matrix_from_id(args, 0, "matrix", state)?;
    matrix_response(ax_linalg::transpose(&matrix), state)
}

fn handle_identity_matrix_linalg(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let n = int_arg(args, 0, "n")?;
    if n < 0 {
        return Err("argument 'n' must be non-negative".to_string());
    }
    matrix_response(ax_linalg::identity(n as usize), state)
}

fn handle_declare_property(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let prop = parse_property_string(string_arg(args, 1, "property")?, state)?;
    attach_compatible_property(state, symbol, prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format!("{:?}", prop)
    }))
}

fn handle_declare_spinor_meta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let metadata = ax_ir::SpinorMetadata {
        dimension: optional_integer_arg(args, 1, "dim")?,
        class: parse_spinor_class_arg(args, 2, "class")?,
        chirality: parse_optional_chirality_arg(args, 3, "chirality")?,
        index_family: optional_symbol_arg(args, 4, "family", state)?,
    };
    let prop = ax_ir::TensorProperty::SpinorMeta(metadata);
    attach_compatible_property(state, symbol, prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_gamma_matrix_meta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let metadata = ax_ir::GammaMatrixMetadata {
        dimension: optional_integer_arg(args, 1, "dim")?,
        metric_symbol: optional_symbol_arg(args, 2, "metric", state)?,
        index_family: optional_symbol_arg(args, 3, "family", state)?,
        has_gamma5: bool_arg(args, 4, "has_gamma5")?,
    };
    let prop = ax_ir::TensorProperty::GammaMatrixMeta(metadata);
    attach_compatible_property(state, symbol, prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_gamma_convention(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let metadata = ax_ir::GammaConventionMetadata {
        signature: parse_gamma_metric_signature_arg(args, 1, "signature")?,
        clifford: parse_clifford_convention_arg(args, 2, "clifford")?,
        gamma5: None,
        epsilon_symbol: None,
        dimension: Some(positive_gamma_dimension_arg(args, 3, "dimension")?),
    };
    let prop = ax_ir::TensorProperty::GammaConventionMeta(metadata.clone());
    crate::apply_gamma_convention_declaration(state.env_mut(), symbol, metadata);
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_gamma5_convention(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let epsilon_symbol = symbol_arg(args, 4, "epsilon_symbol", state)?;
    let metadata = ax_ir::GammaConventionMetadata {
        signature: parse_gamma_metric_signature_arg(args, 1, "signature")?,
        clifford: parse_clifford_convention_arg(args, 2, "clifford")?,
        gamma5: Some(parse_gamma5_convention_arg(args, 3, "gamma5_kind")?),
        epsilon_symbol: Some(epsilon_symbol),
        dimension: Some(positive_gamma_dimension_arg(args, 5, "dimension")?),
    };
    let prop = ax_ir::TensorProperty::GammaConventionMeta(metadata.clone());
    crate::apply_gamma_convention_declaration(state.env_mut(), symbol, metadata);
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_dirac_bar_meta(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let metadata = ax_ir::DiracBarMetadata {
        gamma_symbol: optional_symbol_arg(args, 1, "gamma_symbol", state)?,
        spinor_family: optional_symbol_arg(args, 2, "family", state)?,
        reverse_gamma_order: bool_arg(args, 3, "reverse_gamma_order")?,
    };
    let prop = ax_ir::TensorProperty::DiracBarMeta(metadata);
    attach_compatible_property(state, symbol, prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_trace_space(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let metadata = ax_ir::TraceSpaceMetadata {
        space_symbol: symbol_arg(args, 1, "space_symbol", state)?,
        cyclic: bool_arg(args, 2, "cyclic")?,
    };
    let prop = ax_ir::TensorProperty::TraceSpaceMeta(metadata);
    attach_compatible_property(state, symbol, prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_hilbert_space(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let dim = require_arg(args, 1, "dim")?
        .as_i64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|dim| *dim > 0)
        .ok_or_else(|| "declare_hilbert_space expects a positive integer dimension".to_string())?;
    let prop = ax_ir::TensorProperty::HilbertSpaceMeta(ax_ir::HilbertSpaceMetadata {
        dimension: dim,
        factors: vec![ax_ir::HilbertSpaceFactor {
            symbol,
            dimension: dim,
        }],
    });
    crate::apply_hilbert_space_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_composite_space(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let factors = require_arg(args, 1, "factors")?
        .as_array()
        .ok_or_else(|| {
            "declare_composite_space expects a list of previously declared Hilbert spaces"
                .to_string()
        })?
        .iter()
        .map(|item| {
            item.as_str()
                .map(|value| state.interner_mut().get_or_intern(value))
                .ok_or_else(|| {
                    "declare_composite_space expects a list of previously declared Hilbert spaces"
                        .to_string()
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flattened = flatten_declared_hilbert_factors(state, &factors).ok_or_else(|| {
        "declare_composite_space expects a list of previously declared Hilbert spaces".to_string()
    })?;
    let dimension = flattened.iter().map(|factor| factor.dimension).product();
    let prop = ax_ir::TensorProperty::HilbertSpaceMeta(ax_ir::HilbertSpaceMetadata {
        dimension,
        factors: flattened,
    });
    crate::apply_hilbert_space_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_quantum_object(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let kind_name = string_arg(args, 1, "kind")?;
    let kind = parse_quantum_object_kind_name(kind_name).ok_or_else(|| {
        "declare_quantum_object kind must be one of: ket, bra, operator, density_operator, projector, observable, channel".to_string()
    })?;
    let space_symbol = symbol_arg(args, 2, "space_symbol", state)?;
    hilbert_space_metadata_for_symbol(state, space_symbol).ok_or_else(|| {
        "declare_quantum_object expects a previously declared Hilbert space".to_string()
    })?;
    let prop = ax_ir::TensorProperty::QuantumObjectMeta(ax_ir::QuantumObjectMetadata {
        kind,
        space_symbol,
    });
    crate::apply_quantum_object_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::QuantumObjectMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_operator_space(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let domain_space = symbol_arg(args, 1, "domain_space", state)?;
    let codomain_space = symbol_arg(args, 2, "codomain_space", state)?;
    if hilbert_space_metadata_for_symbol(state, domain_space).is_none()
        || hilbert_space_metadata_for_symbol(state, codomain_space).is_none()
    {
        return Err(
            "declare_operator_space expects previously declared Hilbert spaces".to_string(),
        );
    }
    let metadata = ax_ir::OperatorSpaceMetadata {
        domain_space,
        codomain_space,
    };
    let prop = ax_ir::TensorProperty::OperatorSpaceMeta(metadata.clone());
    crate::apply_operator_space_declaration(state.env_mut(), symbol, metadata);
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_compose_operators_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let left = expr_from_id(args, 0, "left", state)?;
    let right = expr_from_id(args, 1, "right", state)?;
    let expr = ax_ir::Expr::Call(
        state.interner_mut().get_or_intern("compose_operators"),
        vec![left, right],
    );
    crate::propagate_operator_space_metadata(state.env(), &expr, state.interner())?;
    expr_or_struct_response_named(expr, "compose_operators", state)
}

fn parse_mode_statistics_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<ax_ir::ModeStatistics, String> {
    match string_arg(args, idx, name)?.to_ascii_lowercase().as_str() {
        "bosonic" => Ok(ax_ir::ModeStatistics::Bosonic),
        "fermionic" => Ok(ax_ir::ModeStatistics::Fermionic),
        "spin" => Ok(ax_ir::ModeStatistics::Spin),
        _ => Err("declare_mode statistics must be one of: bosonic, fermionic, spin".to_string()),
    }
}

fn parse_mode_index_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
) -> Result<usize, String> {
    require_arg(args, idx, name)?
        .as_i64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "declare_mode expects a nonnegative integer mode index".to_string())
}

fn parse_fock_mode_index_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    error: &str,
) -> Result<usize, String> {
    require_arg(args, idx, name)?
        .as_i64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| error.to_string())
}

fn parse_positive_truncation_arg(
    args: &[serde_json::Value],
    idx: usize,
    name: &str,
    error: &str,
) -> Result<usize, String> {
    require_arg(args, idx, name)?
        .as_i64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| error.to_string())
}

fn handle_declare_mode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let statistics = parse_mode_statistics_arg(args, 1, "statistics")?;
    let mode_index = parse_mode_index_arg(args, 2, "mode_index")?;
    state.env_mut().fock_mode_truncations.remove(&symbol);
    let prop = ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
        statistics,
        subsystem: None,
        mode_index,
        label: None,
    });
    crate::apply_mode_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_mode_in_subsystem(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let statistics = parse_mode_statistics_arg(args, 1, "statistics")?;
    let subsystem = optional_symbol_arg(args, 2, "subsystem", state)?.ok_or_else(|| {
        "declare_mode_in_subsystem expects a symbol naming the subsystem".to_string()
    })?;
    let mode_index = parse_mode_index_arg(args, 3, "mode_index")?;
    state.env_mut().fock_mode_truncations.remove(&symbol);
    let prop = ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
        statistics,
        subsystem: Some(subsystem),
        mode_index,
        label: None,
    });
    crate::apply_mode_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_mode_with_label(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let statistics = parse_mode_statistics_arg(args, 1, "statistics")?;
    let subsystem = optional_symbol_arg(args, 2, "subsystem", state)?.ok_or_else(|| {
        "declare_mode_in_subsystem expects a symbol naming the subsystem".to_string()
    })?;
    let mode_index = parse_mode_index_arg(args, 3, "mode_index")?;
    let label = optional_symbol_arg(args, 4, "label", state)?
        .ok_or_else(|| "declare_mode_with_label expects a symbol label".to_string())?;
    state.env_mut().fock_mode_truncations.remove(&symbol);
    let prop = ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
        statistics,
        subsystem: Some(subsystem),
        mode_index,
        label: Some(label),
    });
    crate::apply_mode_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_bosonic_truncated_mode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error =
        "declare_bosonic_truncated_mode expects a nonnegative mode index and positive truncation";
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let mode_index = parse_fock_mode_index_arg(args, 1, "mode_index", error)?;
    let truncation = parse_positive_truncation_arg(args, 2, "nmax", error)?;
    state
        .env_mut()
        .fock_mode_truncations
        .insert(symbol, truncation);
    let prop = ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
        statistics: ax_ir::ModeStatistics::Bosonic,
        subsystem: None,
        mode_index,
        label: None,
    });
    crate::apply_mode_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_fermionic_mode(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "declare_fermionic_mode expects a nonnegative mode index";
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let mode_index = parse_fock_mode_index_arg(args, 1, "mode_index", error)?;
    state.env_mut().fock_mode_truncations.remove(&symbol);
    let prop = ax_ir::TensorProperty::ModeMeta(ax_ir::ModeMetadata {
        statistics: ax_ir::ModeStatistics::Fermionic,
        subsystem: None,
        mode_index,
        label: None,
    });
    crate::apply_mode_declaration(
        state.env_mut(),
        symbol,
        match &prop {
            ax_ir::TensorProperty::ModeMeta(metadata) => metadata.clone(),
            _ => unreachable!(),
        },
    );
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_declare_fock_space(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let error = "declare_fock_space expects a non-empty list of previously declared mode symbols";
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let mode_symbols = require_arg(args, 1, "mode_symbols")?
        .as_array()
        .ok_or_else(|| error.to_string())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(|name| state.interner_mut().get_or_intern(name))
        })
        .collect::<Option<Vec<_>>>()
        .filter(|items| !items.is_empty())
        .ok_or_else(|| error.to_string())?;
    let metadata = build_fock_space_metadata_for_state(state, symbol, &mode_symbols)
        .ok_or_else(|| error.to_string())?;
    let prop = ax_ir::TensorProperty::FockSpaceMeta(metadata.clone());
    crate::apply_fock_space_declaration(state.env_mut(), symbol, metadata);
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format_tensor_property(&prop, state.interner()),
    }))
}

fn handle_riemann_tensor_declaration(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let props = [
        ax_ir::TensorProperty::RiemannSymmetry,
        ax_ir::TensorProperty::SatisfiesBianchi {
            slots: vec![0, 1, 2, 3],
        },
    ];
    for prop in props.iter().cloned() {
        state
            .env_mut()
            .tensor_properties
            .entry(symbol)
            .or_default()
            .push(prop.clone());
        state.env_mut().property_store.declare_simple(symbol, prop);
    }
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "properties": ["RiemannSymmetry", "SatisfiesBianchi([0,1,2,3])"]
    }))
}

fn handle_declare_indices(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let family = symbol_arg(args, 0, "family", state)?;
    let indices = symbol_list_arg(args, 1, "indices", state)?;
    let dimension = int_arg(args, 2, "dimension").ok().map(|n| n as usize);
    let family_data = ax_ir::IndexFamily {
        name: family,
        values: indices.clone(),
        position: ax_ir::IndexPosition::Free,
        dimension,
        parent: None,
    };
    state
        .env_mut()
        .index_families
        .insert(family, family_data.clone());
    for idx in indices.iter().copied() {
        state.env_mut().index_to_family.insert(idx, family);
    }
    Ok(serde_json::json!({
        "status": "ok",
        "family": state.interner().resolve(family),
        "indices": indices.iter().map(|s| state.interner().resolve(*s)).collect::<Vec<_>>(),
        "dimension": family_data.dimension
    }))
}

fn handle_declare_coordinates(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let coords = symbol_list_arg(args, 0, "coordinates", state)?;
    state.env_mut().coordinates = coords.iter().copied().collect();
    Ok(serde_json::json!({
        "status": "ok",
        "coordinates": coords.iter().map(|s| state.interner().resolve(*s)).collect::<Vec<_>>()
    }))
}

fn handle_declare_assumption(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let assumption_name = string_arg(args, 1, "assumption")?.to_ascii_lowercase();
    let assumption = match assumption_name.as_str() {
        "real" => ax_ir::Assumption::Real,
        "positive" => ax_ir::Assumption::Positive,
        "negative" => ax_ir::Assumption::Negative,
        "nonzero" | "non_zero" => ax_ir::Assumption::NonZero,
        "integer" => ax_ir::Assumption::Integer,
        "even" => ax_ir::Assumption::Even,
        "odd" => ax_ir::Assumption::Odd,
        _ => return Err(format!("unknown assumption '{assumption_name}'")),
    };
    state
        .env_mut()
        .assumptions
        .entry(symbol)
        .or_default()
        .push(assumption.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "assumption": format!("{:?}", assumption)
    }))
}

fn handle_declare_grassmann(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    state.env_mut().gradings.insert(symbol, ax_ir::Grading::Odd);
    Ok(
        serde_json::json!({ "status": "ok", "symbol": state.interner().resolve(symbol), "grading": "Odd" }),
    )
}

fn handle_declare_operator(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let kind = match string_arg(args, 1, "kind")?.to_ascii_lowercase().as_str() {
        "creation" => ax_qm::OperatorKind::Creation,
        "annihilation" => ax_qm::OperatorKind::Annihilation,
        _ => return Err("operator kind must be 'creation' or 'annihilation'".to_string()),
    };
    let explicit_statistics = match args.get(2) {
        None | Some(serde_json::Value::Null) => None,
        Some(_) => Some(
            match string_arg(args, 2, "statistics")?
                .to_ascii_lowercase()
                .as_str()
            {
                "bosonic" => ax_qm::OperatorStatistics::Bosonic,
                "fermionic" => ax_qm::OperatorStatistics::Fermionic,
                _ => return Err("operator statistics must be 'bosonic' or 'fermionic'".to_string()),
            },
        ),
    };
    let statistics = if let Some(metadata) = mode_metadata_for_symbol(state, symbol) {
        let inferred = match metadata.statistics {
            ax_ir::ModeStatistics::Bosonic => Some(ax_qm::OperatorStatistics::Bosonic),
            ax_ir::ModeStatistics::Fermionic => Some(ax_qm::OperatorStatistics::Fermionic),
            ax_ir::ModeStatistics::Spin => None,
        };
        if let Some(explicit) = explicit_statistics {
            if inferred != Some(explicit) {
                return Err(
                    "declare_operator statistics disagree with previously declared mode metadata"
                        .to_string(),
                );
            }
            explicit
        } else {
            inferred.unwrap_or(ax_qm::OperatorStatistics::Bosonic)
        }
    } else {
        explicit_statistics.unwrap_or(ax_qm::OperatorStatistics::Bosonic)
    };
    state.env_mut().operators.insert(symbol, kind);
    state
        .env_mut()
        .operator_statistics
        .insert(symbol, statistics);
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "operator": format!("{:?}", kind),
        "statistics": format!("{:?}", statistics),
    }))
}

fn handle_declare_contraction(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lhs = symbol_arg(args, 0, "lhs", state)?;
    let rhs = symbol_arg(args, 1, "rhs", state)?;
    let value = code_expr(args, 2, "value", state)?;
    state
        .env_mut()
        .contractions
        .insert((lhs, rhs), value.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "lhs": state.interner().resolve(lhs),
        "rhs": state.interner().resolve(rhs),
        "value": ax_ir::pretty_print(&value, state.interner()),
    }))
}

fn handle_set_convention(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let key = string_arg(args, 0, "field")?.to_ascii_lowercase();
    let value = string_arg(args, 1, "value")?.to_ascii_lowercase();
    match key.as_str() {
        "metric_signature" => {
            state.env_mut().convention.metric_signature = match value.as_str() {
                "mostlyplus" | "mostly_plus" => ax_ir::MetricSignature::MostlyPlus,
                "mostlyminus" | "mostly_minus" => ax_ir::MetricSignature::MostlyMinus,
                "euclidean" => ax_ir::MetricSignature::Euclidean,
                _ => return Err(format!("unknown metric_signature '{value}'")),
            };
        }
        "riemann_sign" => {
            state.env_mut().convention.riemann_sign = match value.as_str() {
                "mtw" => ax_ir::RiemannSign::MTW,
                "weinberg" => ax_ir::RiemannSign::Weinberg,
                _ => return Err(format!("unknown riemann_sign '{value}'")),
            };
        }
        "ricci_contraction" => {
            state.env_mut().convention.ricci_contraction = match value.as_str() {
                "firstthird" | "first_third" => ax_ir::RicciContraction::FirstThird,
                "firstfourth" | "first_fourth" => ax_ir::RicciContraction::FirstFourth,
                _ => return Err(format!("unknown ricci_contraction '{value}'")),
            };
        }
        "levi_civita_norm" => {
            state.env_mut().convention.levi_civita_norm = match value.as_str() {
                "plusone" | "plus_one" => ax_ir::LeviCivitaNorm::PlusOne,
                "minusone" | "minus_one" => ax_ir::LeviCivitaNorm::MinusOne,
                "sqrtg" | "sqrt_g" => ax_ir::LeviCivitaNorm::SqrtG,
                _ => return Err(format!("unknown levi_civita_norm '{value}'")),
            };
        }
        "fourier_sign" => {
            state.env_mut().convention.fourier_sign = match value.as_str() {
                "minusi" | "minus_i" => ax_ir::FourierSign::MinusI,
                "plusi" | "plus_i" => ax_ir::FourierSign::PlusI,
                _ => return Err(format!("unknown fourier_sign '{value}'")),
            };
        }
        _ => return Err(format!("unknown convention field '{key}'")),
    }
    Ok(convention_value_to_json(state.env()))
}

fn handle_define_rule(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let name = string_arg(args, 0, "name")?;
    let lhs = code_expr(args, 1, "lhs", state)?;
    let rhs = code_expr(args, 2, "rhs", state)?;
    let rule = ax_rewrite::RewriteRule {
        name: name.to_string(),
        pattern: ax_rewrite::Pattern::Exact(lhs),
        replacement: rhs,
        condition: None,
        trust_level: ax_ir::TrustLevel::Exact,
    };
    state.env_mut().rules.push(rule);
    Ok(serde_json::json!({ "status": "ok", "name": name, "trust": "Exact" }))
}

fn handle_define_metric(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let name = string_arg(args, 0, "name")?.to_string();
    let rows = matrix_code_arg(args, 1, "components")?
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| state.parse_code(&cell))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coords = symbol_list_arg(args, 2, "coordinates", state)?;
    let matrix = symbolic_matrix_from_rows(rows)?;
    let sym = state.interner_mut().get_or_intern(&name);
    state.env_mut().coordinates = coords.iter().copied().collect();
    state
        .env_mut()
        .tensor_properties
        .entry(sym)
        .or_default()
        .push(ax_ir::TensorProperty::Metric);
    state.store_metric(name.clone(), matrix, coords.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "metric_id": name,
        "coordinates": coords.iter().map(|s| state.interner().resolve(*s)).collect::<Vec<_>>()
    }))
}

fn handle_to_python_codegen(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    Ok(
        serde_json::json!({ "status": "ok", "code": ax_codegen::to_python(&expr, state.interner()) }),
    )
}

fn handle_to_rust_codegen(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    Ok(serde_json::json!({ "status": "ok", "code": ax_codegen::to_rust(&expr, state.interner()) }))
}

fn handle_to_cpp_codegen(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    Ok(serde_json::json!({ "status": "ok", "code": ax_codegen::to_cpp(&expr, state.interner()) }))
}

fn handle_equiv_analysis(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("equiv", args, state)
}
fn handle_semantic_diff_analysis(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    binary_named_expr_response("semantic_diff", args, state)
}

fn handle_inspect_analysis(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = crate::inspect::inspect_expr(&expr, state.env(), state.interner());
    Ok(serde_json::json!({
        "status": "ok",
        "kind": result.kind,
        "free_indices": result.free_indices,
        "dummy_pairs": result.dummy_pairs,
        "properties": result.properties,
        "symbols": result.symbols,
        "node_count": result.node_count
    }))
}

fn handle_suggest_analysis(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let goal = args.get(1).and_then(|v| v.as_str());
    let result = crate::suggest::suggest_for_expr(&expr, state.env(), state.interner(), goal);
    Ok(serde_json::json!({
        "status": "ok",
        "goal": goal,
        "note": result.note,
        "suggestions": result.suggestions.into_iter().map(|s| serde_json::json!({"algorithm": s.algorithm, "reason": s.reason})).collect::<Vec<_>>(),
        "missing": result.missing.into_iter().map(|m| serde_json::json!({"symbol": m.symbol, "suggestion": m.suggestion})).collect::<Vec<_>>()
    }))
}

const DIAGNOSTIC_ALGORITHMS: &[&str] = &[
    "canonicalise",
    "meld",
    "collect_terms",
    "simplify",
    "sort_product",
    "eliminate_metric",
    "eliminate_kronecker",
    "rename_dummies",
    "evaluate_components",
    "epsilon_to_delta",
    "expand_delta",
    "reduce_delta",
    "distribute",
    "unwrap",
    "product_rule",
    "integrate_by_parts",
    "factor_out",
    "factor_in",
];

fn handle_diff_diagnostics(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr_a = expr_from_id(args, 0, "expr_a", state)?;
    let expr_b = expr_from_id(args, 1, "expr_b", state)?;
    let mut result = crate::diagnostics::diff_expressions(&expr_a, &expr_b, state.interner());
    if let Some(obj) = result.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
    }
    Ok(result)
}

fn handle_check_properties_diagnostics(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let algorithm = require_arg(args, 1, "algorithm")?
        .as_str()
        .ok_or_else(|| "argument 'algorithm' must be a string".to_string())?;
    let mut result =
        crate::diagnostics::check_properties(&expr, algorithm, state.env(), state.interner());
    if let Some(obj) = result.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
    }
    Ok(result)
}

fn handle_explain_diagnostics(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let algorithm = require_arg(args, 0, "algorithm")?
        .as_str()
        .ok_or_else(|| "argument 'algorithm' must be a string".to_string())?;
    let expr = expr_from_id(args, 1, "expr", state)?;
    Ok(serde_json::json!({
        "status": "ok",
        "explanation": crate::diagnostics::explain_algorithm(algorithm, &expr, state.env(), state.interner()),
    }))
}

fn handle_workflow_lookup(
    args: &[serde_json::Value],
    _state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let goal = require_arg(args, 0, "goal")?
        .as_str()
        .ok_or_else(|| "argument 'goal' must be a string".to_string())?;
    if let Some(workflow) = crate::workflows::lookup_workflow(goal) {
        return Ok(serde_json::json!({
            "status": "ok",
            "goal": workflow.goal,
            "description": workflow.description,
            "steps": workflow.steps.iter().map(|step| serde_json::json!({
                "tool": step.tool,
                "params_template": step.params_template,
                "description": step.description,
                "output_key": step.output_key,
            })).collect::<Vec<_>>(),
            "notes": workflow.notes,
        }));
    }
    Ok(serde_json::json!({
        "status": "ok",
        "message": format!("No exact workflow found for '{}'. Available workflows:", goal),
        "available": crate::workflows::list_workflows().into_iter().map(|(goal, description)| serde_json::json!({
            "goal": goal,
            "description": description,
        })).collect::<Vec<_>>(),
    }))
}

fn handle_list_workflows(
    _args: &[serde_json::Value],
    _state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "status": "ok",
        "available": crate::workflows::list_workflows().into_iter().map(|(goal, description)| serde_json::json!({
            "goal": goal,
            "description": description,
        })).collect::<Vec<_>>(),
    }))
}

fn handle_list_expressions_state(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expressions = state
        .list_expression_ids()
        .into_iter()
        .filter_map(|id| {
            state.get_expr(&id).map(|expr| {
                serde_json::json!({
                    "id": id,
                    "latex": state.render_latex(expr),
                    "unicode": state.render_unicode(expr),
                })
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "status": "ok",
        "expressions": expressions,
    }))
}

fn handle_list_metrics_state(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let metrics = state
        .list_metric_ids()
        .into_iter()
        .filter_map(|id| {
            state.get_metric(&id).map(|(_, coords)| {
                serde_json::json!({
                    "id": id,
                    "coordinates": coords.iter().map(|c| state.interner().resolve(*c).to_string()).collect::<Vec<_>>(),
                    "dimension": coords.len(),
                })
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "status": "ok",
        "metrics": metrics,
    }))
}

fn handle_list_properties_state(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "status": "ok",
        "properties": state.list_properties().into_iter().map(|(symbol, properties)| serde_json::json!({
            "symbol": symbol,
            "properties": properties,
        })).collect::<Vec<_>>(),
    }))
}

fn handle_list_index_families_state(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "status": "ok",
        "families": state.list_index_families().into_iter().map(|(name, indices, dimension)| serde_json::json!({
            "name": name,
            "indices": indices,
            "dimension": dimension,
        })).collect::<Vec<_>>(),
    }))
}

fn handle_get_state_summary_state(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expressions = state
        .list_expression_ids()
        .into_iter()
        .filter_map(|id| {
            state.get_expr(&id).map(|expr| {
                serde_json::json!({
                    "id": id,
                    "latex": state.render_latex(expr),
                    "unicode": state.render_unicode(expr),
                })
            })
        })
        .collect::<Vec<_>>();
    let metrics = state
        .list_metric_ids()
        .into_iter()
        .filter_map(|id| {
            state.get_metric(&id).map(|(_, coords)| {
                serde_json::json!({
                    "id": id,
                    "coordinates": coords.iter().map(|c| state.interner().resolve(*c).to_string()).collect::<Vec<_>>(),
                    "dimension": coords.len(),
                })
            })
        })
        .collect::<Vec<_>>();
    let properties = state
        .list_properties()
        .into_iter()
        .map(|(symbol, properties)| {
            serde_json::json!({
                "symbol": symbol,
                "properties": properties,
            })
        })
        .collect::<Vec<_>>();
    let index_families = state
        .list_index_families()
        .into_iter()
        .map(|(name, indices, dimension)| {
            serde_json::json!({
                "name": name,
                "indices": indices,
                "dimension": dimension,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "status": "ok",
        "expression_count": expressions.len(),
        "expressions": expressions,
        "metric_count": metrics.len(),
        "metrics": metrics,
        "properties": properties,
        "index_families": index_families,
        "christoffel_ids": state.list_christoffel_ids(),
        "riemann_ids": state.list_riemann_ids(),
        "ricci_ids": state.list_ricci_ids(),
    }))
}

fn handle_eval_code(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let code = string_arg(args, 0, "code")?;
    let expr = state.parse_code(code)?;
    expr_or_struct_response(expr, state)
}

fn int_expr_arg(args: &[serde_json::Value], idx: usize, name: &str) -> Result<ax_ir::Expr, String> {
    Ok(ax_ir::Expr::Int(int_arg(args, idx, name)?.into()))
}

fn handle_angle_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "angle",
            vec![int_expr_arg(args, 0, "i")?, int_expr_arg(args, 1, "j")?],
            state,
        ),
        "angle",
        state,
    )
}

fn handle_square_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "square",
            vec![int_expr_arg(args, 0, "i")?, int_expr_arg(args, 1, "j")?],
            state,
        ),
        "square",
        state,
    )
}

fn handle_mandelstam_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "mandelstam",
            vec![int_expr_arg(args, 0, "i")?, int_expr_arg(args, 1, "j")?],
            state,
        ),
        "mandelstam",
        state,
    )
}

fn handle_parke_taylor_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "parke_taylor",
            vec![
                int_expr_arg(args, 0, "n")?,
                int_expr_arg(args, 1, "i")?,
                int_expr_arg(args, 2, "j")?,
            ],
            state,
        ),
        "parke_taylor",
        state,
    )
}

fn handle_three_point_mhv_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "three_point_mhv",
            vec![
                int_expr_arg(args, 0, "i")?,
                int_expr_arg(args, 1, "j")?,
                int_expr_arg(args, 2, "k")?,
            ],
            state,
        ),
        "three_point_mhv",
        state,
    )
}

fn handle_three_point_anti_mhv_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "three_point_anti_mhv",
            vec![
                int_expr_arg(args, 0, "i")?,
                int_expr_arg(args, 1, "j")?,
                int_expr_arg(args, 2, "k")?,
            ],
            state,
        ),
        "three_point_anti_mhv",
        state,
    )
}

fn handle_spinor_unary_named(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(name, vec![expr.clone()], state);
    expr_response_with_change(&expr, result, name, state)
}

fn handle_expand_chain_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_spinor_unary_named("expand_chain", args, state)
}
fn handle_contract_adjacent_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_spinor_unary_named("contract_adjacent", args, state)
}
fn handle_expand_mandelstam_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_spinor_unary_named("expand_mandelstam", args, state)
}
fn handle_collect_mandelstam_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_spinor_unary_named("collect_mandelstam", args, state)
}

fn handle_schouten_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(
        "schouten",
        vec![
            expr.clone(),
            int_expr_arg(args, 1, "a")?,
            int_expr_arg(args, 2, "b")?,
            int_expr_arg(args, 3, "c")?,
            int_expr_arg(args, 4, "d")?,
        ],
        state,
    );
    expr_response_with_change(&expr, result, "schouten", state)
}

fn handle_momentum_conservation_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(
        "momentum_conservation",
        vec![
            expr.clone(),
            int_expr_arg(args, 1, "n")?,
            int_expr_arg(args, 2, "eliminate")?,
        ],
        state,
    );
    expr_response_with_change(&expr, result, "momentum_conservation", state)
}

fn handle_spinor_simplify_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(
        "spinor_simplify",
        vec![expr.clone(), int_expr_arg(args, 1, "n")?],
        state,
    );
    expr_response_with_change(&expr, result, "spinor_simplify", state)
}

fn handle_bcfw_shift_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let z = ax_ir::Expr::Sym(symbol_arg(args, 3, "z", state)?);
    let result = call_named(
        "bcfw_shift",
        vec![
            expr.clone(),
            int_expr_arg(args, 1, "i")?,
            int_expr_arg(args, 2, "j")?,
            z,
        ],
        state,
    );
    expr_response_with_change(&expr, result, "bcfw_shift", state)
}

fn handle_bcfw_decomposition_spinor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let helicities = require_arg(args, 3, "helicities")?
        .as_array()
        .ok_or_else(|| "argument 'helicities' must be an array of integers".to_string())?
        .iter()
        .map(|v| {
            v.as_i64()
                .map(|n| ax_ir::Expr::Int(n.into()))
                .ok_or_else(|| "helicities must contain integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    expr_or_struct_response_named(
        call_named(
            "bcfw_decomposition",
            vec![
                int_expr_arg(args, 0, "n")?,
                int_expr_arg(args, 1, "i")?,
                int_expr_arg(args, 2, "j")?,
                ax_ir::Expr::List(helicities),
            ],
            state,
        ),
        "bcfw_decomposition",
        state,
    )
}

fn handle_four_bracket_twistor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "four_bracket",
            vec![
                int_expr_arg(args, 0, "i")?,
                int_expr_arg(args, 1, "j")?,
                int_expr_arg(args, 2, "k")?,
                int_expr_arg(args, 3, "l")?,
            ],
            state,
        ),
        "four_bracket",
        state,
    )
}

fn handle_plucker_twistor(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(
        "plucker",
        vec![
            expr.clone(),
            int_expr_arg(args, 1, "a")?,
            int_expr_arg(args, 2, "b")?,
            int_expr_arg(args, 3, "c")?,
            int_expr_arg(args, 4, "d")?,
            int_expr_arg(args, 5, "e")?,
            int_expr_arg(args, 6, "f")?,
        ],
        state,
    );
    expr_response_with_change(&expr, result, "plucker", state)
}

fn handle_perturb_general(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let result = call_named(
        "perturb",
        vec![
            expr.clone(),
            ax_ir::Expr::Sym(symbol_arg(args, 1, "field", state)?),
            ax_ir::Expr::Sym(symbol_arg(args, 2, "background", state)?),
            ax_ir::Expr::Sym(symbol_arg(args, 3, "perturbation", state)?),
            ax_ir::Expr::Sym(symbol_arg(args, 4, "epsilon", state)?),
            int_expr_arg(args, 5, "order")?,
        ],
        state,
    );
    expr_response_with_change(&expr, result, "perturb", state)
}

fn handle_perturb_inverse(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "perturb_inverse",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "field", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "background", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 2, "background_inv", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 3, "perturbation", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 4, "epsilon", state)?),
                int_expr_arg(args, 5, "order")?,
            ],
            state,
        ),
        "perturb_inverse",
        state,
    )
}

fn handle_perturb_tensor_named(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let coords = symbol_list_arg(args, 5, "coords", state)?
        .into_iter()
        .map(ax_ir::Expr::Sym)
        .collect();
    expr_or_struct_response_named(
        call_named(
            name,
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "field", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "background", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 2, "background_inv", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 3, "perturbation", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 4, "epsilon", state)?),
                ax_ir::Expr::List(coords),
                int_expr_arg(args, 6, "order")?,
            ],
            state,
        ),
        name,
        state,
    )
}

fn handle_perturb_christoffel(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_perturb_tensor_named("perturb_christoffel", args, state)
}
fn handle_perturb_riemann(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_perturb_tensor_named("perturb_riemann", args, state)
}
fn handle_perturb_ricci(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_perturb_tensor_named("perturb_ricci", args, state)
}
fn handle_perturb_einstein(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_perturb_tensor_named("perturb_einstein", args, state)
}

fn handle_linearized_einstein_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "linearized_einstein",
            vec![int_expr_arg(args, 0, "order")?],
            state,
        ),
        "linearized_einstein",
        state,
    )
}

fn handle_nullary_named(
    name: &str,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(call_named(name, Vec::new(), state), name, state)
}

fn handle_mukhanov_sasaki_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("mukhanov_sasaki", state)
}

fn handle_linearized_einstein_vector_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("linearized_einstein_vector", state)
}

fn handle_linearized_einstein_tensor_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("linearized_einstein_tensor", state)
}

fn handle_second_order_einstein_vector_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("second_order_einstein_vector", state)
}

fn handle_second_order_einstein_tensor_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("second_order_einstein_tensor", state)
}

fn handle_tensor_mode_equation_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("tensor_mode_equation", state)
}

fn handle_multifield_equations_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "multifield_equations",
            vec![int_expr_arg(args, 0, "nfields")?],
            state,
        ),
        "multifield_equations",
        state,
    )
}

fn handle_boltzmann_bridge_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("boltzmann_bridge", state)
}

fn handle_boltzmann_bridge_export_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "boltzmann_bridge_export",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "target", state)?)],
            state,
        ),
        "boltzmann_bridge_export",
        state,
    )
}

fn handle_cubic_action_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cubic_action",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "channel", state)?)],
            state,
        ),
        "cubic_action",
        state,
    )
}

fn handle_cubic_kernel_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cubic_kernel",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "channel", state)?)],
            state,
        ),
        "cubic_kernel",
        state,
    )
}

fn handle_bispectrum_shape_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "bispectrum_shape",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "channel", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "shape", state)?),
            ],
            state,
        ),
        "bispectrum_shape",
        state,
    )
}

fn handle_export_cubic_vertex_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "export_cubic_vertex",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "channel", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "target", state)?),
            ],
            state,
        ),
        "export_cubic_vertex",
        state,
    )
}

fn handle_eft_model_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "eft_model",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)],
            state,
        ),
        "eft_model",
        state,
    )
}

fn handle_eft_quadratic_sector_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "eft_quadratic_sector",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)],
            state,
        ),
        "eft_quadratic_sector",
        state,
    )
}

fn handle_eft_stability_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "eft_stability",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)],
            state,
        ),
        "eft_stability",
        state,
    )
}

fn handle_eft_mode_equations_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "eft_mode_equations",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)],
            state,
        ),
        "eft_mode_equations",
        state,
    )
}

fn handle_eft_export_rhs_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "eft_export_rhs",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "target", state)?),
            ],
            state,
        ),
        "eft_export_rhs",
        state,
    )
}

fn handle_project_scalar_harmonics_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("project_scalar_harmonics", state)
}

fn handle_project_vector_harmonics_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("project_vector_harmonics", state)
}

fn handle_project_tensor_harmonics_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("project_tensor_harmonics", state)
}

fn handle_project_second_order_vector_harmonics_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("project_second_order_vector_harmonics", state)
}

fn handle_project_second_order_tensor_harmonics_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("project_second_order_tensor_harmonics", state)
}

fn handle_neutrino_hierarchy_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "neutrino_hierarchy",
            vec![
                int_expr_arg(args, 0, "lmax")?,
                ax_ir::Expr::Sym(symbol_arg(args, 1, "gauge", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 2, "closure", state)?),
            ],
            state,
        ),
        "neutrino_hierarchy",
        state,
    )
}

fn handle_photon_hierarchy_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "photon_hierarchy",
            vec![
                int_expr_arg(args, 0, "lmax")?,
                ax_ir::Expr::Sym(symbol_arg(args, 1, "gauge", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 2, "closure", state)?),
            ],
            state,
        ),
        "photon_hierarchy",
        state,
    )
}

fn handle_export_hierarchy_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "export_hierarchy",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "target", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "species", state)?),
                int_expr_arg(args, 2, "lmax")?,
                ax_ir::Expr::Sym(symbol_arg(args, 3, "gauge", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 4, "closure", state)?),
            ],
            state,
        ),
        "export_hierarchy",
        state,
    )
}

fn handle_cpt_parity_report_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named("cpt_parity_report", vec![], state),
        "cpt_parity_report",
        state,
    )
}

fn handle_scalar_harmonic_spec_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "scalar_harmonic_spec",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "curvature", state)?)],
            state,
        ),
        "scalar_harmonic_spec",
        state,
    )
}

fn handle_vector_harmonic_spec_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "vector_harmonic_spec",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "curvature", state)?)],
            state,
        ),
        "vector_harmonic_spec",
        state,
    )
}

fn handle_tensor_harmonic_spec_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "tensor_harmonic_spec",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "curvature", state)?)],
            state,
        ),
        "tensor_harmonic_spec",
        state,
    )
}

fn handle_tensor_mode_first_order_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "tensor_mode_first_order",
            vec![ax_ir::Expr::Sym(symbol_arg(
                args,
                0,
                "polarization",
                state,
            )?)],
            state,
        ),
        "tensor_mode_first_order",
        state,
    )
}

fn handle_frw_background_spec_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "frw_background_spec",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "time", state)?),
                ax_ir::Expr::Sym(symbol_arg(args, 1, "curvature", state)?),
                int_expr_arg(args, 2, "spatial_dim")?,
            ],
            state,
        ),
        "frw_background_spec",
        state,
    )
}

fn handle_cpt_gauge_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_gauge",
            vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)],
            state,
        ),
        "cpt_gauge",
        state,
    )
}

fn handle_cpt_matter_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let mut call_args = vec![ax_ir::Expr::Sym(symbol_arg(args, 0, "kind", state)?)];
    if let Some(value) = args.get(1) {
        if !value.is_null() {
            call_args.push(int_expr_arg(args, 1, "nfields")?);
        }
    }
    expr_or_struct_response_named(
        call_named("cpt_matter", call_args, state),
        "cpt_matter",
        state,
    )
}

fn handle_cpt_linearized_einstein_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_linearized_einstein",
            vec![
                int_expr_arg(args, 0, "order")?,
                expr_from_id(args, 1, "background", state)?,
                expr_from_id(args, 2, "gauge", state)?,
                expr_from_id(args, 3, "matter", state)?,
            ],
            state,
        ),
        "cpt_linearized_einstein",
        state,
    )
}

fn handle_cpt_fluid_equations_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_fluid_equations",
            vec![expr_from_id(args, 0, "background", state)?],
            state,
        ),
        "cpt_fluid_equations",
        state,
    )
}

fn handle_cpt_quadratic_action_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_quadratic_action",
            vec![
                expr_from_id(args, 0, "background", state)?,
                expr_from_id(args, 1, "matter", state)?,
            ],
            state,
        ),
        "cpt_quadratic_action",
        state,
    )
}

fn handle_cpt_mukhanov_sasaki_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_mukhanov_sasaki",
            vec![
                expr_from_id(args, 0, "background", state)?,
                expr_from_id(args, 1, "matter", state)?,
            ],
            state,
        ),
        "cpt_mukhanov_sasaki",
        state,
    )
}

fn handle_cpt_mukhanov_sasaki_first_order_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_mukhanov_sasaki_first_order",
            vec![
                expr_from_id(args, 0, "background", state)?,
                expr_from_id(args, 1, "matter", state)?,
            ],
            state,
        ),
        "cpt_mukhanov_sasaki_first_order",
        state,
    )
}

fn handle_cpt_bardeen_invariance_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_bardeen_invariance",
            vec![expr_from_id(args, 0, "background", state)?],
            state,
        ),
        "cpt_bardeen_invariance",
        state,
    )
}

fn handle_cpt_export_mode_rhs_cosmology(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "cpt_export_mode_rhs",
            vec![
                ax_ir::Expr::Sym(symbol_arg(args, 0, "target", state)?),
                expr_from_id(args, 1, "background", state)?,
                expr_from_id(args, 2, "matter", state)?,
            ],
            state,
        ),
        "cpt_export_mode_rhs",
        state,
    )
}
fn handle_svt_decompose_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("svt_decompose", state)
}
fn handle_bardeen_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("bardeen", state)
}
fn handle_regge_wheeler_decompose_gauge(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named(
            "regge_wheeler_decompose",
            vec![int_expr_arg(args, 0, "l")?],
            state,
        ),
        "regge_wheeler_decompose",
        state,
    )
}
fn handle_power_spectrum_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("power_spectrum", state)
}
fn handle_spectral_index_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("spectral_index", state)
}
fn handle_tensor_scalar_ratio_cosmology(
    _args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_nullary_named("tensor_scalar_ratio", state)
}

fn handle_zerilli_gauge(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named("zerilli", vec![int_expr_arg(args, 0, "l")?], state),
        "zerilli",
        state,
    )
}
fn handle_regge_wheeler_gauge(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        call_named("regge_wheeler", vec![int_expr_arg(args, 0, "l")?], state),
        "regge_wheeler",
        state,
    )
}

fn handle_graded_declare(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "symbol", state)?;
    let grading = match require_arg(args, 1, "grading")? {
        serde_json::Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "bosonic" | "boson" | "even" => ax_graded::Grading::bosonic(),
            "fermionic" | "fermion" | "odd" => ax_graded::Grading::fermionic(),
            other => other
                .parse::<i32>()
                .map(ax_graded::Grading::ghost)
                .map_err(|_| format!("unknown grading '{s}'"))?,
        },
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(|v| ax_graded::Grading::ghost(v as i32))
            .ok_or_else(|| "numeric grading must be an integer".to_string())?,
        _ => return Err("grading must be a string or integer".to_string()),
    };
    state
        .env_mut()
        .graded_table
        .declare(symbol, grading.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "grading": format!("{:?}", grading)
    }))
}

fn handle_graded_commutator_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let lhs = expr_from_id(args, 0, "lhs", state)?;
    let rhs = expr_from_id(args, 1, "rhs", state)?;
    let out = ax_graded::graded_commutator(&lhs, &rhs, &state.env().graded_table, state.interner());
    let mut response = expr_or_struct_response(out, state)?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert("status".to_string(), serde_json::json!("ok"));
        obj.insert("changed".to_string(), serde_json::json!(true));
        obj.insert(
            "message".to_string(),
            serde_json::json!("graded_commutator applied successfully"),
        );
    }
    Ok(response)
}

fn handle_graded_simplify_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let out = ax_graded::graded_simplify(&expr, &state.env().graded_table, state.interner());
    expr_response_with_change(&expr, out, "graded_simplify", state)
}

fn active_superspace_for_state(
    state: &dyn EvalState,
) -> (
    ax_graded::superspace::SuperspaceSetup,
    ax_graded::GradedSymbolTable,
) {
    state
        .env()
        .superspace_setup
        .clone()
        .map(|setup| (setup, state.env().graded_table.clone()))
        .unwrap_or_else(|| ax_graded::superspace::setup_n1_superspace(state.interner()))
}

fn theta_monomial_from_json(
    value: &serde_json::Value,
    setup: &ax_graded::superspace::SuperspaceSetup,
) -> Result<ax_graded::superspace::ThetaMonomial, String> {
    let items = value
        .as_array()
        .ok_or_else(|| "theta_spec must be [theta_count, theta_bar_count]".to_string())?;
    if items.len() != 2 {
        return Err("theta_spec must contain exactly two integers".to_string());
    }
    let theta_count = items[0]
        .as_u64()
        .ok_or_else(|| "theta count must be an integer".to_string())?
        as usize;
    let theta_bar_count = items[1]
        .as_u64()
        .ok_or_else(|| "theta_bar count must be an integer".to_string())?
        as usize;
    if theta_count > setup.theta.len() || theta_bar_count > setup.theta_bar.len() {
        return Err("theta_spec exceeds available N=1 theta coordinates".to_string());
    }
    let mut theta_powers = vec![0; setup.theta.len()];
    let mut theta_bar_powers = vec![0; setup.theta_bar.len()];
    for power in theta_powers.iter_mut().take(theta_count) {
        *power = 1;
    }
    for power in theta_bar_powers.iter_mut().take(theta_bar_count) {
        *power = 1;
    }
    Ok(ax_graded::superspace::ThetaMonomial {
        theta_powers,
        theta_bar_powers,
    })
}

fn handle_setup_superspace_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let n = int_arg(args, 0, "N")?;
    if n != 1 {
        return Err("N>1 superspace not yet implemented".to_string());
    }
    let (setup, table) = ax_graded::superspace::setup_n1_superspace(state.interner());
    state.env_mut().superspace_setup = Some(setup);
    state.env_mut().graded_table = table;
    Ok(serde_json::json!({ "status": "ok", "N": 1 }))
}

fn superfield_expr_response(
    expansion: ax_graded::superspace::SuperfieldExpansion,
    setup: &ax_graded::superspace::SuperspaceSetup,
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    expr_or_struct_response_named(
        ax_graded::superspace::superfield_to_expr(&expansion, setup, state.interner()),
        "superfield",
        state,
    )
}

fn handle_superfield_named(
    name: &str,
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let symbol = symbol_arg(args, 0, "name", state)?;
    let (setup, _) = active_superspace_for_state(state);
    let expansion = match name {
        "expand_superfield" => {
            ax_graded::superspace::expand_superfield(symbol, &setup, state.interner())
        }
        "chiral_superfield" => {
            let expanded =
                ax_graded::superspace::expand_superfield(symbol, &setup, state.interner());
            ax_graded::superspace::chiral_constraint(&expanded, &setup, state.interner())
        }
        "antichiral_superfield" => {
            let expanded =
                ax_graded::superspace::expand_superfield(symbol, &setup, state.interner());
            ax_graded::superspace::antichiral_constraint(&expanded, &setup, state.interner())
        }
        "vector_superfield_wz" => {
            ax_graded::superspace::vector_superfield_wz_gauge(symbol, &setup, state.interner())
        }
        _ => unreachable!(),
    };
    superfield_expr_response(expansion, &setup, state)
}

fn handle_expand_superfield_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_superfield_named("expand_superfield", args, state)
}
fn handle_chiral_superfield_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_superfield_named("chiral_superfield", args, state)
}
fn handle_antichiral_superfield_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_superfield_named("antichiral_superfield", args, state)
}
fn handle_vector_superfield_wz_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    handle_superfield_named("vector_superfield_wz", args, state)
}

fn handle_extract_component_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let (setup, table) = active_superspace_for_state(state);
    let theta = theta_monomial_from_json(require_arg(args, 1, "theta_spec")?, &setup)?;
    expr_response_with_change(
        &expr,
        ax_graded::superspace::extract_component(&expr, &theta, &setup, &table, state.interner()),
        "extract_component",
        state,
    )
}

fn handle_d_alpha_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let alpha = int_arg(args, 1, "alpha")? as usize;
    let (setup, table) = active_superspace_for_state(state);
    expr_response_with_change(
        &expr,
        ax_graded::d_algebra::apply_d_alpha(&expr, alpha, &setup, &table, state.interner()),
        "d_alpha",
        state,
    )
}

fn handle_d_bar_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let alpha = int_arg(args, 1, "alpha_dot")? as usize;
    let (setup, table) = active_superspace_for_state(state);
    expr_response_with_change(
        &expr,
        ax_graded::d_algebra::apply_d_bar_alpha_dot(&expr, alpha, &setup, &table, state.interner()),
        "d_bar",
        state,
    )
}

fn handle_d_squared_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let (setup, table) = active_superspace_for_state(state);
    expr_response_with_change(
        &expr,
        ax_graded::d_algebra::d_squared(&expr, &setup, &table, state.interner()),
        "d_squared",
        state,
    )
}

fn handle_d_bar_squared_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let (setup, table) = active_superspace_for_state(state);
    expr_response_with_change(
        &expr,
        ax_graded::d_algebra::d_bar_squared(&expr, &setup, &table, state.interner()),
        "d_bar_squared",
        state,
    )
}

fn handle_superspace_integrate_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let measure = match string_arg(args, 1, "measure")?
        .to_ascii_lowercase()
        .as_str()
    {
        "full" => ax_graded::d_algebra::SuperspaceMeasure::FullSuperspace,
        "chiral" => ax_graded::d_algebra::SuperspaceMeasure::Chiral,
        "antichiral" | "anti_chiral" => ax_graded::d_algebra::SuperspaceMeasure::AntiChiral,
        other => return Err(format!("unknown superspace measure '{other}'")),
    };
    let (setup, table) = active_superspace_for_state(state);
    expr_response_with_change(
        &expr,
        ax_graded::d_algebra::superspace_integrate(
            &expr,
            measure,
            &setup,
            &table,
            state.interner(),
        ),
        "superspace_integrate",
        state,
    )
}

fn handle_setup_brst_ym_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let gauge = symbol_arg(args, 0, "A", state)?;
    let ghost = symbol_arg(args, 1, "c", state)?;
    let antighost = symbol_arg(args, 2, "cbar", state)?;
    let aux = symbol_arg(args, 3, "B", state)?;
    let coupling = symbol_arg(args, 4, "g", state)?;
    let (setup, table) = ax_graded::brst::setup_yang_mills_brst(
        gauge,
        ghost,
        antighost,
        aux,
        coupling,
        state.interner(),
    );
    state.env_mut().brst_setup = Some(setup);
    state.env_mut().graded_table = table;
    Ok(serde_json::json!({ "status": "ok", "theory": "yang_mills_brst" }))
}

fn handle_brst_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let setup = state
        .env()
        .brst_setup
        .clone()
        .ok_or_else(|| "BRST setup is not initialized; call setup_brst_ym first".to_string())?;
    expr_response_with_change(
        &expr,
        ax_graded::brst::apply_brst(&expr, &setup, &state.env().graded_table, state.interner()),
        "brst",
        state,
    )
}

fn handle_brst_check_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let setup = state
        .env()
        .brst_setup
        .clone()
        .ok_or_else(|| "BRST setup is not initialized; call setup_brst_ym first".to_string())?;
    let applied =
        ax_graded::brst::apply_brst(&expr, &setup, &state.env().graded_table, state.interner());
    let simplified =
        ax_graded::graded_simplify(&applied, &state.env().graded_table, state.interner());
    Ok(
        serde_json::json!({ "status": "ok", "closed": simplified == ax_ir::Expr::zero(), "result": state.render_unicode(&simplified) }),
    )
}

fn handle_ghost_number_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    match ax_graded::brst::ghost_number(&expr, &state.env().graded_table) {
        Some(n) => Ok(serde_json::json!({ "status": "ok", "ghost_number": n })),
        None => Err("expression has inconsistent ghost number".to_string()),
    }
}

fn handle_filter_ghost_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "expr", state)?;
    let target = int_arg(args, 1, "n")? as i32;
    expr_response_with_change(
        &expr,
        ax_graded::brst::filter_by_ghost_number(
            &expr,
            target,
            &state.env().graded_table,
            state.interner(),
        ),
        "filter_ghost_number",
        state,
    )
}

pub fn callable_entries() -> Vec<CallableEntry> {
    let ps =
        |params: Vec<ParamDef>| -> &'static [ParamDef] { Box::leak(params.into_boxed_slice()) };
    vec![
        centry("eval", "Parse and evaluate an Axioma code snippet.", ps(vec![pdef("code", ParamType::Code, true, "Axioma code or expression.")]), handle_eval_code),
        centry("import", "Import an Axioma std module via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Import declaration code.")]), handle_eval_syntax_entry),
        centry("assume", "Declare assumptions via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Assumption declaration code.")]), handle_eval_syntax_entry),
        centry("differentiate", "Symbolic differentiation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Differentiation variable.")]), handle_diff),
        centry("integrate", "Indefinite symbolic integration.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Integration variable.")]), handle_integrate),
        centry("double_integral", "Iterated double integration.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("x", ParamType::Symbol, true, "Inner integration variable."), pdef("y", ParamType::Symbol, true, "Outer integration variable.")]), handle_double_integral),
        centry("dblint", "Alias for double_integral.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("x", ParamType::Symbol, true, "Inner integration variable."), pdef("y", ParamType::Symbol, true, "Outer integration variable.")]), handle_double_integral),
        centry("triple_integral", "Iterated triple integration.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("x", ParamType::Symbol, true, "First integration variable."), pdef("y", ParamType::Symbol, true, "Second integration variable."), pdef("z", ParamType::Symbol, true, "Third integration variable.")]), handle_triple_integral),
        centry("tplint", "Alias for triple_integral.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("x", ParamType::Symbol, true, "First integration variable."), pdef("y", ParamType::Symbol, true, "Second integration variable."), pdef("z", ParamType::Symbol, true, "Third integration variable.")]), handle_triple_integral),
        centry("definite_integral", "Definite symbolic integration from lower to upper bounds.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Integration variable."), pdef("lower_bound", ParamType::Code, true, "Lower bound expression."), pdef("upper_bound", ParamType::Code, true, "Upper bound expression.")]), handle_definite_integral),
        centry("defint", "Alias for definite_integral.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Integration variable."), pdef("lower_bound", ParamType::Code, true, "Lower bound expression."), pdef("upper_bound", ParamType::Code, true, "Upper bound expression.")]), handle_definite_integral),
        centry("integrate_by_parts", "One-step integration by parts using a chosen differentiation variable to integrate away.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("away", ParamType::Symbol, true, "Variable to move derivatives away from.")]), handle_integrate_by_parts),
        centry("ibp", "Alias for integrate_by_parts.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("away", ParamType::Symbol, true, "Variable to move derivatives away from.")]), handle_integrate_by_parts),
        centry("limit", "Symbolic limit.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Limit variable."), pdef("point", ParamType::Code, true, "Limit point expression.")]), handle_limit),
        centry("series", "Taylor series expansion.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Expansion variable."), pdef("point", ParamType::Code, true, "Expansion point."), pdef("order", ParamType::Integer, true, "Series order.")]), handle_series),
        centry("simplify", "Full simplification pipeline.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify),
        centry("expand", "Algebraic expansion.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand),
        centry("collect_terms", "Collect like terms.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_collect_terms),
        centry("rationalize", "Common-denominator rational simplification.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_rationalize),
        centry("partial_fractions", "Partial fraction decomposition.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Decomposition variable.")]), handle_partial_fractions),
        centry("apart", "Alias for partial_fractions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Decomposition variable.")]), handle_partial_fractions),
        centry("trig_simplify", "Exact trigonometric simplification.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_trig_simplify),
        centry("factor_out", "Factor common symbols from a sum.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("targets", ParamType::SymbolList, false, "Optional target symbols to factor.")]), handle_factor_out),
        centry("collect_factors", "Factor common symbols from a sum.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("targets", ParamType::SymbolList, false, "Optional target symbols to factor.")]), handle_factor_out),
        centry("factor_in", "Group terms with common prefactors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("targets", ParamType::SymbolList, false, "Optional target symbols to factor.")]), handle_factor_in),
        centry("eq", "Create an equation object.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Left-hand side expression id."), pdef("rhs", ParamType::ExprId, true, "Right-hand side expression id.")]), handle_eq_entry),
        centry("get_lhs", "Get the left-hand side of an equation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Equation expression id.")]), handle_get_lhs_entry),
        centry("get_rhs", "Get the right-hand side of an equation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Equation expression id.")]), handle_get_rhs_entry),
        centry("swap_sides", "Swap the left- and right-hand sides of an equation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Equation expression id.")]), handle_swap_sides_entry),
        centry("multiply_through", "Multiply both sides of an equation by a factor.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Equation expression id."), pdef("rhs", ParamType::ExprId, true, "Factor expression id.")]), handle_multiply_through_entry),
        centry("add_through", "Add a term to both sides of an equation.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Equation expression id."), pdef("rhs", ParamType::ExprId, true, "Term expression id.")]), handle_add_through_entry),
        centry("to_rhs", "Move terms containing target from the LHS to the RHS.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Equation expression id."), pdef("rhs", ParamType::ExprId, true, "Target expression id.")]), handle_to_rhs_entry),
        centry("to_lhs", "Move terms containing target from the RHS to the LHS.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Equation expression id."), pdef("rhs", ParamType::ExprId, true, "Target expression id.")]), handle_to_lhs_entry),
        centry("isolate", "Solve simple equation patterns for target.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Equation expression id."), pdef("rhs", ParamType::ExprId, true, "Target expression id.")]), handle_isolate_entry),
        centry("eq_to_rule", "Convert an equation to an exact rewrite rule.", ps(vec![pdef("expr", ParamType::ExprId, true, "Equation expression id.")]), handle_eq_to_rule_entry),
        centry("eq_to_subrule", "Alias for eq_to_rule.", ps(vec![pdef("expr", ParamType::ExprId, true, "Equation expression id.")]), handle_eq_to_subrule_entry),
        centry("differentiate_eq", "Differentiate both sides of an equation.", ps(vec![pdef("eq", ParamType::ExprId, true, "Equation expression id."), pdef("var", ParamType::Symbol, true, "Differentiation variable.")]), handle_differentiate_eq_entry),
        centry("integrate_eq", "Integrate both sides of an equation.", ps(vec![pdef("eq", ParamType::ExprId, true, "Equation expression id."), pdef("var", ParamType::Symbol, true, "Integration variable.")]), handle_integrate_eq_entry),
        centry("substitute_eq", "Substitute in both sides of an equation.", ps(vec![pdef("eq", ParamType::ExprId, true, "Equation expression id."), pdef("target", ParamType::Code, true, "Target expression."), pdef("replacement", ParamType::Code, true, "Replacement expression.")]), handle_substitute_eq_entry),
        centry("raise_eq", "Raise an index on both sides using the active metric.", ps(vec![pdef("eq", ParamType::ExprId, true, "Equation expression id."), pdef("index", ParamType::Symbol, true, "Index to raise.")]), handle_raise_eq_entry),
        centry("lower_eq", "Lower an index on both sides using the active metric.", ps(vec![pdef("eq", ParamType::ExprId, true, "Equation expression id."), pdef("index", ParamType::Symbol, true, "Index to lower.")]), handle_lower_eq_entry),
        centry("subs", "Symbolic substitution.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("target", ParamType::Code, true, "Target pattern expression."), pdef("replacement", ParamType::Code, true, "Replacement expression.")]), handle_subs),
        centry("symbolic_substitute", "Symbolic substitution.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("target", ParamType::Code, true, "Target pattern expression."), pdef("replacement", ParamType::Code, true, "Replacement expression.")]), handle_subs),
        centry("multi_substitute", "Repeated symbolic substitution via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Substitution expression code.")]), handle_eval_syntax_entry),
        centry("substitute_with_indices", "Index-aware symbolic substitution.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("target", ParamType::Code, true, "Target pattern expression."), pdef("replacement", ParamType::Code, true, "Replacement expression.")]), handle_subs),
        centry("rewrite", "Apply registered rewrite rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_rewrite),
        centry("zoom", "Split an expression into matching and remainder parts.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("pattern", ParamType::Code, true, "Pattern expression.")]), handle_zoom),
        centry("unzoom", "Recombine a focus expression and its remainder.", ps(vec![pdef("focus", ParamType::ExprId, true, "Focus expression id."), pdef("remainder", ParamType::ExprId, true, "Remainder expression id.")]), handle_unzoom),
        centry("take_match", "Keep only matching summands.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("pattern", ParamType::Code, true, "Pattern expression.")]), handle_take_match),
        centry("sin", "Sine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sin),
        centry("cos", "Cosine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_cos),
        centry("tan", "Tangent.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_tan),
        centry("sec", "Secant.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sec),
        centry("csc", "Cosecant.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_csc),
        centry("cot", "Cotangent.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_cot),
        centry("asin", "Inverse sine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_asin),
        centry("arcsin", "Inverse sine alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arcsin),
        centry("acos", "Inverse cosine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_acos),
        centry("arccos", "Inverse cosine alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arccos),
        centry("atan", "Inverse tangent.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_atan),
        centry("arctan", "Inverse tangent alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arctan),
        centry("atan2", "Two-argument arctangent.", ps(vec![pdef("lhs", ParamType::ExprId, true, "First stored expression id."), pdef("rhs", ParamType::ExprId, true, "Second stored expression id.")]), handle_atan2),
        centry("sinh", "Hyperbolic sine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sinh),
        centry("cosh", "Hyperbolic cosine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_cosh),
        centry("tanh", "Hyperbolic tangent.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_tanh),
        centry("asinh", "Inverse hyperbolic sine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_asinh),
        centry("arcsinh", "Inverse hyperbolic sine alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arcsinh),
        centry("acosh", "Inverse hyperbolic cosine.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_acosh),
        centry("arccosh", "Inverse hyperbolic cosine alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arccosh),
        centry("atanh", "Inverse hyperbolic tangent.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_atanh),
        centry("arctanh", "Inverse hyperbolic tangent alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arctanh),
        centry("exp", "Exponential.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_exp),
        centry("log", "Natural logarithm.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_log),
        centry("sqrt", "Square root.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sqrt),
        centry("abs", "Absolute value.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_abs),
        centry("sign", "Sign function.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sign),
        centry("sgn", "Sign alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sgn),
        centry("Re", "Real part.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_re),
        centry("Im", "Imaginary part.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_im),
        centry("conj", "Complex conjugate.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_conj),
        centry("arg", "Complex argument.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_arg),
        centry("N", "Numeric evaluation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_n),
        centry("gradient", "Gradient.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_gradient),
        centry("grad", "Gradient alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_grad),
        centry("divergence", "Divergence.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored vector expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_divergence),
        centry("div", "Divergence alias.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored vector expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_div),
        centry("curl", "Curl.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored vector expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_curl),
        centry("laplacian", "Laplacian.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_laplacian),
        centry("jacobian", "Jacobian matrix.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored list expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_jacobian),
        centry("hessian", "Hessian matrix.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("variables", ParamType::SymbolList, true, "Variables list.")]), handle_hessian),
        centry("angle", "Construct a spinor-helicity angle bracket.", ps(vec![pdef("i", ParamType::Integer, true, "Left particle label."), pdef("j", ParamType::Integer, true, "Right particle label.")]), handle_angle_spinor),
        centry("square", "Construct a spinor-helicity square bracket.", ps(vec![pdef("i", ParamType::Integer, true, "Left particle label."), pdef("j", ParamType::Integer, true, "Right particle label.")]), handle_square_spinor),
        centry("mandelstam", "Construct a Mandelstam invariant.", ps(vec![pdef("i", ParamType::Integer, true, "First particle label."), pdef("j", ParamType::Integer, true, "Second particle label.")]), handle_mandelstam_spinor),
        centry("parke_taylor", "Construct a Parke-Taylor MHV amplitude.", ps(vec![pdef("n", ParamType::Integer, true, "Number of gluons."), pdef("i", ParamType::Integer, true, "First negative-helicity label."), pdef("j", ParamType::Integer, true, "Second negative-helicity label.")]), handle_parke_taylor_spinor),
        centry("three_point_mhv", "Construct the three-point MHV amplitude.", ps(vec![pdef("i", ParamType::Integer, true, "First negative-helicity label."), pdef("j", ParamType::Integer, true, "Second negative-helicity label."), pdef("k", ParamType::Integer, true, "Positive-helicity label.")]), handle_three_point_mhv_spinor),
        centry("three_point_anti_mhv", "Construct the three-point anti-MHV amplitude.", ps(vec![pdef("i", ParamType::Integer, true, "First positive-helicity label."), pdef("j", ParamType::Integer, true, "Second positive-helicity label."), pdef("k", ParamType::Integer, true, "Negative-helicity label.")]), handle_three_point_anti_mhv_spinor),
        centry("expand_chain", "Expand spinor chains into bracket products.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id.")]), handle_expand_chain_spinor),
        centry("contract_adjacent", "Contract adjacent brackets into one-momentum chains.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id.")]), handle_contract_adjacent_spinor),
        centry("expand_mandelstam", "Expand Mandelstam invariants into spinor brackets.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id.")]), handle_expand_mandelstam_spinor),
        centry("collect_mandelstam", "Collect bracket products into Mandelstam invariants.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id.")]), handle_collect_mandelstam_spinor),
        centry("schouten", "Apply the spinor Schouten identity.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id."), pdef("a", ParamType::Integer, true, "Label a."), pdef("b", ParamType::Integer, true, "Label b."), pdef("c", ParamType::Integer, true, "Label c."), pdef("d", ParamType::Integer, true, "Label d.")]), handle_schouten_spinor),
        centry("momentum_conservation", "Apply spinor momentum conservation.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id."), pdef("n", ParamType::Integer, true, "Number of particles."), pdef("eliminate", ParamType::Integer, true, "Particle label to eliminate.")]), handle_momentum_conservation_spinor),
        centry("spinor_simplify", "Simplify spinor-helicity expressions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id."), pdef("n", ParamType::Integer, true, "Number of particles.")]), handle_spinor_simplify_spinor),
        centry("bcfw_shift", "Apply a BCFW shift.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored spinor expression id."), pdef("i", ParamType::Integer, true, "Shifted angle label."), pdef("j", ParamType::Integer, true, "Shifted square label."), pdef("z", ParamType::Symbol, true, "Shift parameter symbol.")]), handle_bcfw_shift_spinor),
        centry("bcfw_decomposition", "Enumerate BCFW factorization terms.", ps(vec![pdef("n", ParamType::Integer, true, "Number of particles."), pdef("i", ParamType::Integer, true, "Shifted angle label."), pdef("j", ParamType::Integer, true, "Shifted square label."), pdef("helicities", ParamType::Code, true, "Array of +1/-1 helicities.")]), handle_bcfw_decomposition_spinor),
        centry("four_bracket", "Construct a momentum-twistor four-bracket.", ps(vec![pdef("i", ParamType::Integer, true, "First label."), pdef("j", ParamType::Integer, true, "Second label."), pdef("k", ParamType::Integer, true, "Third label."), pdef("l", ParamType::Integer, true, "Fourth label.")]), handle_four_bracket_twistor),
        centry("plucker", "Apply the momentum-twistor Plucker identity.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored twistor expression id."), pdef("a", ParamType::Integer, true, "Label a."), pdef("b", ParamType::Integer, true, "Label b."), pdef("c", ParamType::Integer, true, "Label c."), pdef("d", ParamType::Integer, true, "Label d."), pdef("e", ParamType::Integer, true, "Label e."), pdef("f", ParamType::Integer, true, "Label f.")]), handle_plucker_twistor),
        centry("perturb", "Expand an expression in a metric perturbation series.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("field", ParamType::Symbol, true, "Full field symbol."), pdef("background", ParamType::Symbol, true, "Background field symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_general),
        centry("perturb_inverse", "Expand the inverse metric perturbatively.", ps(vec![pdef("field", ParamType::Symbol, true, "Full metric symbol."), pdef("background", ParamType::Symbol, true, "Background metric symbol."), pdef("background_inv", ParamType::Symbol, true, "Background inverse metric symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_inverse),
        centry("perturb_christoffel", "Expand the Christoffel symbol perturbatively.", ps(vec![pdef("field", ParamType::Symbol, true, "Full metric symbol."), pdef("background", ParamType::Symbol, true, "Background metric symbol."), pdef("background_inv", ParamType::Symbol, true, "Background inverse metric symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_christoffel),
        centry("perturb_riemann", "Expand the Riemann tensor perturbatively.", ps(vec![pdef("field", ParamType::Symbol, true, "Full metric symbol."), pdef("background", ParamType::Symbol, true, "Background metric symbol."), pdef("background_inv", ParamType::Symbol, true, "Background inverse metric symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_riemann),
        centry("perturb_ricci", "Expand the Ricci tensor perturbatively.", ps(vec![pdef("field", ParamType::Symbol, true, "Full metric symbol."), pdef("background", ParamType::Symbol, true, "Background metric symbol."), pdef("background_inv", ParamType::Symbol, true, "Background inverse metric symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_ricci),
        centry("perturb_einstein", "Expand the Einstein tensor perturbatively.", ps(vec![pdef("field", ParamType::Symbol, true, "Full metric symbol."), pdef("background", ParamType::Symbol, true, "Background metric symbol."), pdef("background_inv", ParamType::Symbol, true, "Background inverse metric symbol."), pdef("perturbation", ParamType::Symbol, true, "First-order perturbation symbol."), pdef("epsilon", ParamType::Symbol, true, "Expansion parameter."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols."), pdef("order", ParamType::Integer, true, "Maximum perturbative order.")]), handle_perturb_einstein),
        centry("linearized_einstein", "Return first- or second-order scalar perturbation Einstein equations.", ps(vec![pdef("order", ParamType::Integer, true, "Perturbation order, currently 1 or 2.")]), handle_linearized_einstein_cosmology),
        centry("linearized_einstein_vector", "Derived linear vector Einstein equations in FRW Poisson gauge.", ps(vec![]), handle_linearized_einstein_vector_cosmology),
        centry("linearized_einstein_tensor", "Derived linear tensor Einstein equations in FRW.", ps(vec![]), handle_linearized_einstein_tensor_cosmology),
        centry("second_order_einstein_vector", "Derived second-order vector Einstein equations with quadratic source splitting.", ps(vec![]), handle_second_order_einstein_vector_cosmology),
        centry("second_order_einstein_tensor", "Derived second-order tensor Einstein equations with quadratic source splitting.", ps(vec![]), handle_second_order_einstein_tensor_cosmology),
        centry("mukhanov_sasaki", "Return the Mukhanov-Sasaki equation.", ps(vec![]), handle_mukhanov_sasaki_cosmology),
        centry("tensor_mode_equation", "Tensor polarization mode equations derived from the quadratic action.", ps(vec![]), handle_tensor_mode_equation_cosmology),
        centry("tensor_mode_first_order", "First-order ODE system for a tensor polarization mode.", ps(vec![
            pdef("polarization", ParamType::StringEnum(&["plus", "cross"]), true, "Tensor polarization mode."),
        ]), handle_tensor_mode_first_order_cosmology),
        centry("multifield_equations", "Derived multifield curvature and entropy mode equations.", ps(vec![
            pdef("nfields", ParamType::Integer, true, "Number of canonical scalar fields."),
        ]), handle_multifield_equations_cosmology),
        centry("boltzmann_bridge", "Symbolic first-order Einstein–Boltzmann bridge system in Newtonian gauge.", ps(vec![]), handle_boltzmann_bridge_cosmology),
        centry("boltzmann_bridge_export", "Export the symbolic Einstein–Boltzmann bridge system.", ps(vec![
            pdef("target", ParamType::StringEnum(&["python", "rust", "cpp", "json"]), true, "Code-generation target."),
        ]), handle_boltzmann_bridge_export_cosmology),
        centry("cubic_action", "Reduced cubic CPT action density for a given interaction channel.", ps(vec![
            pdef("channel", ParamType::StringEnum(&["scalar_scalar_scalar", "tensor_tensor_tensor", "scalar_scalar_tensor", "scalar_tensor_tensor"]), true, "Cubic interaction channel."),
        ]), handle_cubic_action_cosmology),
        centry("cubic_kernel", "Fourier-space cubic interaction kernel for a given CPT channel.", ps(vec![
            pdef("channel", ParamType::StringEnum(&["scalar_scalar_scalar", "tensor_tensor_tensor", "scalar_scalar_tensor", "scalar_tensor_tensor"]), true, "Cubic interaction channel."),
        ]), handle_cubic_kernel_cosmology),
        centry("bispectrum_shape", "Evaluate a cubic kernel on a named bispectrum shape.", ps(vec![
            pdef("channel", ParamType::StringEnum(&["scalar_scalar_scalar", "tensor_tensor_tensor", "scalar_scalar_tensor", "scalar_tensor_tensor"]), true, "Cubic interaction channel."),
            pdef("shape", ParamType::StringEnum(&["local", "equilateral", "squeezed"]), true, "Named bispectrum shape."),
        ]), handle_bispectrum_shape_cosmology),
        centry("export_cubic_vertex", "Export a cubic interaction vertex as code.", ps(vec![
            pdef("channel", ParamType::StringEnum(&["scalar_scalar_scalar", "tensor_tensor_tensor", "scalar_scalar_tensor", "scalar_tensor_tensor"]), true, "Cubic interaction channel."),
            pdef("target", ParamType::StringEnum(&["python", "rust", "cpp"]), true, "Code-generation target."),
        ]), handle_export_cubic_vertex_cosmology),
        centry("eft_model", "Construct a typed reduced EFT-of-inflation model.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["canonical", "reduced_sound_speed", "horndeski_like"]), true, "Reduced EFT model kind."),
        ]), handle_eft_model_cosmology),
        centry("eft_quadratic_sector", "Reduced scalar/tensor quadratic sector for an EFT model.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["canonical", "reduced_sound_speed", "horndeski_like"]), true, "Reduced EFT model kind."),
        ]), handle_eft_quadratic_sector_cosmology),
        centry("eft_stability", "Ghost and gradient stability conditions for a reduced EFT model.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["canonical", "reduced_sound_speed", "horndeski_like"]), true, "Reduced EFT model kind."),
        ]), handle_eft_stability_cosmology),
        centry("eft_mode_equations", "Reduced scalar and tensor mode equations for a reduced EFT model.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["canonical", "reduced_sound_speed", "horndeski_like"]), true, "Reduced EFT model kind."),
        ]), handle_eft_mode_equations_cosmology),
        centry("eft_export_rhs", "Export reduced EFT mode RHS functions.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["canonical", "reduced_sound_speed", "horndeski_like"]), true, "Reduced EFT model kind."),
            pdef("target", ParamType::StringEnum(&["python", "rust", "cpp", "json"]), true, "Code-generation target."),
        ]), handle_eft_export_rhs_cosmology),
        centry("project_scalar_harmonics", "Project derived scalar CPT equations to FRW harmonic space.", ps(vec![]), handle_project_scalar_harmonics_cosmology),
        centry("project_vector_harmonics", "Project derived vector CPT equations to FRW harmonic space.", ps(vec![]), handle_project_vector_harmonics_cosmology),
        centry("project_tensor_harmonics", "Project derived tensor CPT equations to FRW harmonic space.", ps(vec![]), handle_project_tensor_harmonics_cosmology),
        centry("project_second_order_vector_harmonics", "Project derived second-order vector equations to harmonic space.", ps(vec![]), handle_project_second_order_vector_harmonics_cosmology),
        centry("project_second_order_tensor_harmonics", "Project derived second-order tensor equations to harmonic space.", ps(vec![]), handle_project_second_order_tensor_harmonics_cosmology),
        centry("neutrino_hierarchy", "Construct a symbolic neutrino multipole hierarchy with explicit truncation.", ps(vec![
            pdef("lmax", ParamType::Integer, true, "Maximum multipole order."),
            pdef("gauge", ParamType::StringEnum(&["newtonian", "synchronous"]), true, "Hierarchy gauge."),
            pdef("closure", ParamType::StringEnum(&["power_law", "free_streaming", "user_symbolic"]), true, "Hierarchy closure relation."),
        ]), handle_neutrino_hierarchy_cosmology),
        centry("photon_hierarchy", "Construct a symbolic photon multipole hierarchy with explicit truncation.", ps(vec![
            pdef("lmax", ParamType::Integer, true, "Maximum multipole order."),
            pdef("gauge", ParamType::StringEnum(&["newtonian", "synchronous"]), true, "Hierarchy gauge."),
            pdef("closure", ParamType::StringEnum(&["power_law", "free_streaming", "user_symbolic"]), true, "Hierarchy closure relation."),
        ]), handle_photon_hierarchy_cosmology),
        centry("export_hierarchy", "Export a symbolic hierarchy system or external-solver hook payload.", ps(vec![
            pdef("target", ParamType::StringEnum(&["python", "rust", "cpp", "json", "class_hook", "camb_hook"]), true, "Export target."),
            pdef("species", ParamType::StringEnum(&["neutrino", "photon"]), true, "Hierarchy species."),
            pdef("lmax", ParamType::Integer, true, "Maximum multipole order."),
            pdef("gauge", ParamType::StringEnum(&["newtonian", "synchronous"]), true, "Hierarchy gauge."),
            pdef("closure", ParamType::StringEnum(&["power_law", "free_streaming", "user_symbolic"]), true, "Hierarchy closure relation."),
        ]), handle_export_hierarchy_cosmology),
        centry("cpt_parity_report", "Run built-in CPT parity suites against embedded benchmark fixtures.", ps(vec![]), handle_cpt_parity_report_cosmology),
        centry("scalar_harmonic_spec", "Describe the scalar harmonic basis for a given FRW spatial curvature.", ps(vec![
            pdef("curvature", ParamType::StringEnum(&["flat", "closed", "open"]), true, "Spatial curvature choice."),
        ]), handle_scalar_harmonic_spec_cosmology),
        centry("vector_harmonic_spec", "Describe the vector harmonic basis for a given FRW spatial curvature.", ps(vec![
            pdef("curvature", ParamType::StringEnum(&["flat", "closed", "open"]), true, "Spatial curvature choice."),
        ]), handle_vector_harmonic_spec_cosmology),
        centry("tensor_harmonic_spec", "Describe the tensor harmonic basis for a given FRW spatial curvature.", ps(vec![
            pdef("curvature", ParamType::StringEnum(&["flat", "closed", "open"]), true, "Spatial curvature choice."),
        ]), handle_tensor_harmonic_spec_cosmology),
        centry("frw_background_spec", "Return a compact CPT background spec expression.", ps(vec![
            pdef("time", ParamType::StringEnum(&["conformal", "cosmic"]), true, "Time coordinate choice."),
            pdef("curvature", ParamType::StringEnum(&["flat", "closed", "open"]), true, "Spatial curvature choice."),
            pdef("spatial_dim", ParamType::Integer, true, "Number of spatial dimensions."),
        ]), handle_frw_background_spec_cosmology),
        centry("cpt_gauge", "Return a compact CPT gauge spec expression.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["newtonian", "synchronous", "comoving", "flat", "uniform_density", "uniform_curvature", "poisson"]), true, "Gauge choice."),
        ]), handle_cpt_gauge_cosmology),
        centry("cpt_matter", "Return a compact CPT matter spec expression.", ps(vec![
            pdef("kind", ParamType::StringEnum(&["perfect_fluid", "imperfect_fluid", "canonical_scalar", "symbolic", "multi_canonical_scalar"]), true, "Matter model choice."),
            pdef("nfields", ParamType::Optional(Box::new(ParamType::Integer)), false, "Field count for multi_canonical_scalar."),
        ]), handle_cpt_matter_cosmology),
        centry("cpt_linearized_einstein", "Return the labelled CPT linearized Einstein equation list.", ps(vec![
            pdef("order", ParamType::Integer, true, "Perturbation order, currently 1 or 2."),
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
            pdef("gauge", ParamType::ExprId, true, "Stored gauge spec expression id."),
            pdef("matter", ParamType::ExprId, true, "Stored matter spec expression id."),
        ]), handle_cpt_linearized_einstein_cosmology),
        centry("cpt_fluid_equations", "Return the labelled perfect-fluid conservation equation list.", ps(vec![
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
        ]), handle_cpt_fluid_equations_cosmology),
        centry("cpt_quadratic_action", "Return the CPT reduced quadratic action density expression.", ps(vec![
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
            pdef("matter", ParamType::ExprId, true, "Stored matter spec expression id."),
        ]), handle_cpt_quadratic_action_cosmology),
        centry("cpt_mukhanov_sasaki", "Return the CPT Fourier-space Mukhanov-Sasaki equation.", ps(vec![
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
            pdef("matter", ParamType::ExprId, true, "Stored matter spec expression id."),
        ]), handle_cpt_mukhanov_sasaki_cosmology),
        centry("cpt_mukhanov_sasaki_first_order", "Return the CPT Mukhanov-Sasaki first-order system as [[lhs, rhs], ...].", ps(vec![
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
            pdef("matter", ParamType::ExprId, true, "Stored matter spec expression id."),
        ]), handle_cpt_mukhanov_sasaki_first_order_cosmology),
        centry("cpt_bardeen_invariance", "Return [name, variation, invariant_flag] entries for the Bardeen potentials.", ps(vec![
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
        ]), handle_cpt_bardeen_invariance_cosmology),
        centry("cpt_export_mode_rhs", "Return the exported CPT mode RHS code payload as an interned plain-text symbol string.", ps(vec![
            pdef("target", ParamType::StringEnum(&["python", "rust", "cpp"]), true, "Code-generation target."),
            pdef("background", ParamType::ExprId, true, "Stored FRW background spec expression id."),
            pdef("matter", ParamType::ExprId, true, "Stored matter spec expression id."),
        ]), handle_cpt_export_mode_rhs_cosmology),
        centry("svt_decompose", "Return the standard SVT decomposition modes.", ps(vec![]), handle_svt_decompose_cosmology),
        centry("bardeen", "Return the two Bardeen potentials.", ps(vec![]), handle_bardeen_cosmology),
        centry("regge_wheeler_decompose", "Return symbolic even- and odd-parity Schwarzschild perturbation sectors.", ps(vec![pdef("l", ParamType::Integer, true, "Angular momentum quantum number.")]), handle_regge_wheeler_decompose_gauge),
        centry("zerilli", "Return the Zerilli master equation.", ps(vec![pdef("l", ParamType::Integer, true, "Angular momentum quantum number.")]), handle_zerilli_gauge),
        centry("regge_wheeler", "Return the Regge-Wheeler master equation.", ps(vec![pdef("l", ParamType::Integer, true, "Angular momentum quantum number.")]), handle_regge_wheeler_gauge),
        centry("power_spectrum", "Return the leading scalar power spectrum.", ps(vec![]), handle_power_spectrum_cosmology),
        centry("spectral_index", "Return the leading slow-roll spectral index.", ps(vec![]), handle_spectral_index_cosmology),
        centry("tensor_scalar_ratio", "Return the leading tensor-to-scalar ratio.", ps(vec![]), handle_tensor_scalar_ratio_cosmology),
        centry("graded", "Declare a grading on a symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol."), pdef("grading", ParamType::Code, true, "bosonic, fermionic, or integer ghost number.")]), handle_graded_declare),
        centry("graded_commutator", "Compute the graded commutator.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Left expression id."), pdef("rhs", ParamType::ExprId, true, "Right expression id.")]), handle_graded_commutator_entry),
        centry("graded_simplify", "Simplify using graded algebra rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_graded_simplify_entry),
        centry("setup_superspace", "Initialize N=1 superspace.", ps(vec![pdef("N", ParamType::Integer, true, "Supersymmetry count; currently only 1.")]), handle_setup_superspace_entry),
        centry("expand_superfield", "Expand a generic superfield.", ps(vec![pdef("name", ParamType::Symbol, true, "Superfield symbol.")]), handle_expand_superfield_entry),
        centry("chiral_superfield", "Construct a chiral superfield.", ps(vec![pdef("name", ParamType::Symbol, true, "Superfield symbol.")]), handle_chiral_superfield_entry),
        centry("antichiral_superfield", "Construct an antichiral superfield.", ps(vec![pdef("name", ParamType::Symbol, true, "Superfield symbol.")]), handle_antichiral_superfield_entry),
        centry("vector_superfield_wz", "Construct a Wess-Zumino gauge vector superfield.", ps(vec![pdef("name", ParamType::Symbol, true, "Vector superfield symbol.")]), handle_vector_superfield_wz_entry),
        centry("extract_component", "Extract a theta component.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id."), pdef("theta_spec", ParamType::Code, true, "JSON array [theta_count, theta_bar_count].")]), handle_extract_component_entry),
        centry("d_alpha", "Apply D_alpha.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id."), pdef("alpha", ParamType::Integer, true, "Spinor index 0 or 1.")]), handle_d_alpha_entry),
        centry("d_bar", "Apply D_bar alpha-dot.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id."), pdef("alpha_dot", ParamType::Integer, true, "Dotted spinor index 0 or 1.")]), handle_d_bar_entry),
        centry("d_squared", "Apply D squared.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id.")]), handle_d_squared_entry),
        centry("d_bar_squared", "Apply D-bar squared.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id.")]), handle_d_bar_squared_entry),
        centry("superspace_integrate", "Integrate over a superspace measure.", ps(vec![pdef("expr", ParamType::ExprId, true, "Superspace expression id."), pdef("measure", ParamType::StringEnum(&["full", "chiral", "antichiral"]), true, "Integration measure.")]), handle_superspace_integrate_entry),
        centry("setup_brst_ym", "Initialize Yang-Mills BRST.", ps(vec![pdef("A", ParamType::Symbol, true, "Gauge field."), pdef("c", ParamType::Symbol, true, "Ghost."), pdef("cbar", ParamType::Symbol, true, "Antighost."), pdef("B", ParamType::Symbol, true, "Nakanishi-Lautrup field."), pdef("g", ParamType::Symbol, true, "Coupling.")]), handle_setup_brst_ym_entry),
        centry("brst", "Apply the BRST operator.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_brst_entry),
        centry("brst_check", "Check BRST closure.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_brst_check_entry),
        centry("ghost_number", "Compute ghost number.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_ghost_number_entry),
        centry("filter_ghost", "Filter a sum by ghost number.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("n", ParamType::Integer, true, "Ghost number.")]), handle_filter_ghost_entry),
        centry("canonicalise", "Canonical tensor simplification.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_canonicalise),
        centry("canonicalize", "Canonical tensor simplification.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_canonicalise),
        centry("canonicalize_indices", "Canonicalize index ordering with tensor properties.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_canonicalize_indices),
        centry("meld", "Combine indexed sum terms using symmetry identities.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_meld),
        centry("sort_product", "Sort tensor-product factors canonically.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sort_product),
        centry("product_rule", "Apply the tensor Leibniz rule.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_product_rule_tensor),
        centry("leibniz", "Apply the tensor Leibniz rule.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_product_rule_tensor),
        centry("tensor_distribute", "Distribute tensor products over sums.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_tensor_distribute),
        centry("distribute", "Distribute tensor products over sums.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_tensor_distribute),
        centry("tdistribute", "Distribute tensor products over sums.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_tensor_distribute),
        centry("eliminate_kronecker", "Contract Kronecker deltas.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_eliminate_kronecker),
        centry("eliminate_metric", "Contract metric or inverse-metric factors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_eliminate_metric),
        centry("eliminate_vielbein", "Simplify vielbein contractions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_eliminate_vielbein),
        centry("rewrite_indices_vielbein", "Rewrite tensor indices between coordinate and frame families using vielbeins.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("vielbein", ParamType::Symbol, true, "Vielbein symbol."),
            pdef("inverse_vielbein", ParamType::Symbol, true, "Inverse-vielbein symbol."),
            pdef("from_family", ParamType::Symbol, true, "Source index family."),
            pdef("to_family", ParamType::Symbol, true, "Target index family."),
        ]), handle_rewrite_indices_vielbein_tensor),
        centry("epsilon_to_delta", "Convert epsilon contractions to generalized deltas.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_epsilon_to_delta),
        centry("expand_delta", "Expand generalized delta expressions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_delta),
        centry("expand_dummies", "Expand abstract dummy contractions to coordinates.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_dummies),
        centry("explicit_indices", "Insert explicit indices for implicit-index tensors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_explicit_indices),
        centry("expand_implicit", "Expand implicit contractions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_implicit),
        centry("einsteinify", "Repair Einstein contractions by fixing dummy variances.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_einsteinify),
        centry("split_index", "Split one index family into two subfamilies.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("parent_indices", ParamType::SymbolList, true, "Parent-family indices."), pdef("subfamily_one", ParamType::SymbolList, true, "First subfamily symbols."), pdef("subfamily_two", ParamType::SymbolList, true, "Second subfamily symbols.")]), handle_split_index_tensor),
        centry("rename_dummies", "Rename dummy indices canonically.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_rename_dummies),
        centry("young_project", "Project with an explicit tableau or with declared Young-tableau symmetry plus optional monoterm simplification.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("tableau_or_modulo_monoterm", ParamType::Optional(Box::new(ParamType::Code)), false, "Either a tableau code string like [[0,1],[2]] or the modulo_monoterm boolean."),
            pdef("canonicalize_after", ParamType::Bool, false, "Whether to canonicalize indices after projection. Defaults to true."),
            pdef("rename_dummies_after", ParamType::Bool, false, "Whether to rename dummy indices after projection. Defaults to true."),
        ]), handle_young_project),
        centry("young_project_tensor", "Project onto Young-tableau symmetry with optional monoterm simplification.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("modulo_monoterm", ParamType::Bool, false, "Whether to simplify modulo declared monoterm symmetries after projection. Defaults to true."),
            pdef("canonicalize_after", ParamType::Bool, false, "Whether to canonicalize indices after projection. Defaults to true."),
            pdef("rename_dummies_after", ParamType::Bool, false, "Whether to rename dummy indices after projection. Defaults to true."),
        ]), handle_young_project_tensor),
        centry("tensor_reduce", "Run the finished tensor reduction pipeline.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("monoterm", ParamType::Bool, false, "Whether to run monoterm canonicalisation first. Defaults to true."),
            pdef("multiterm", ParamType::Bool, false, "Whether to run Cadabra-style multi-term Young projection on products. Defaults to true."),
            pdef("dimension_dependent", ParamType::Bool, false, "Whether to run dimension-dependent reduction when metadata permits it. Defaults to true."),
            pdef("meld", ParamType::Bool, false, "Whether to run final basis reduction with meld. Defaults to true."),
            pdef("modulo_monoterm", ParamType::Bool, false, "Whether the multi-term stage should simplify modulo monoterm symmetries. Defaults to true."),
        ]), handle_tensor_reduce),
        centry("abstract_tensor_reduce", "Run the abstract tensor reduction pipeline.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("monoterm", ParamType::Bool, false, "Whether to run monoterm canonicalisation first. Defaults to true."),
            pdef("multiterm", ParamType::Bool, false, "Whether to run Cadabra-style multi-term Young projection on products. Defaults to true."),
            pdef("dimension_dependent", ParamType::Bool, false, "Whether to run dimension-dependent reduction when metadata permits it. Defaults to true."),
            pdef("meld", ParamType::Bool, false, "Whether to run final basis reduction with meld. Defaults to true."),
            pdef("modulo_monoterm", ParamType::Bool, false, "Whether the multi-term stage should simplify modulo monoterm symmetries. Defaults to true."),
        ]), handle_abstract_tensor_reduce),
        centry("abstract_gr_reduce", "Run the abstract GR reduction pipeline.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("monoterm", ParamType::Bool, false, "Whether to run monoterm canonicalisation first. Defaults to true."),
            pdef("multiterm", ParamType::Bool, false, "Whether to run Cadabra-style multi-term Young projection on products. Defaults to true."),
            pdef("dimension_dependent", ParamType::Bool, false, "Whether to run dimension-dependent reduction when metadata permits it. Defaults to true."),
            pdef("meld", ParamType::Bool, false, "Whether to run final basis reduction with meld. Defaults to true."),
            pdef("modulo_monoterm", ParamType::Bool, false, "Whether the multi-term stage should simplify modulo monoterm symmetries. Defaults to true."),
        ]), handle_abstract_tensor_reduce),
        centry("contracted_bianchi_reduce", "Reduce abstract contracted-Bianchi Ricci/scalar and optional Einstein-divergence identities without inserting metrics.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("derivative_sym", ParamType::Symbol, true, "Abstract covariant-derivative symbol."),
            pdef("ricci_sym", ParamType::Symbol, true, "Ricci tensor symbol."),
            pdef("scalar_sym", ParamType::Symbol, true, "Scalar-curvature symbol."),
            pdef("einstein_sym", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional Einstein tensor symbol."),
        ]), handle_contracted_bianchi_reduce_tensor),
        centry("riemann_to_ricci", "Rewrite internal abstract Riemann contractions into Ricci or scalar-curvature factors.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("ricci_sym", ParamType::Symbol, true, "Ricci tensor symbol."),
            pdef("scalar_sym", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional scalar-curvature symbol for traced Ricci collapse."),
        ]), handle_riemann_to_ricci_tensor),
        centry("reduce_delta", "Reduce expanded deltas back to compact form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_reduce_delta),
        centry("symmetrise", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("symmetrize", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("sym", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("antisymmetrise", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("antisymmetrize", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("asym", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("decompose", "Decompose an expression in a supplied basis.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("basis", ParamType::ExprId, true, "Stored list of basis expressions.")]), handle_decompose_tensor),
        centry("decompose_product", "Decompose a tensor product by dimension.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("dim", ParamType::Integer, false, "Optional dimension; if omitted, infer from index-family metadata.")]), handle_decompose_product_tensor),
        centry("schouten_reduce", "Apply dimension-dependent Schouten-style tensor reduction.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_schouten_reduce_tensor),
        centry("unwrap_derivatives", "Pull constant factors out of derivative operators.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_unwrap_derivatives_tensor),
        centry("unwrap", "Pull constant factors out of derivative operators.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_unwrap_derivatives_tensor),
        centry("drop_weight", "Drop terms with a chosen symbolic weight.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("label", ParamType::Code, true, "Weight label."), pdef("value", ParamType::Integer, true, "Weight value to drop.")]), handle_drop_weight_tensor),
        centry("keep_weight", "Keep only terms with a chosen symbolic weight.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("label", ParamType::Code, true, "Weight label."), pdef("value", ParamType::Integer, true, "Weight value to keep.")]), handle_keep_weight_tensor),
        centry("lower_free_indices", "Lower free indices.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_lower_free_indices),
        centry("lower_indices", "Lower free indices.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_lower_free_indices),
        centry("raise_free_indices", "Raise free indices.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_raise_free_indices),
        centry("raise_indices", "Raise free indices.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_raise_free_indices),
        centry("rewrite_indices", "Rewrite free-index variances for a tensor.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("tensor", ParamType::Symbol, true, "Target tensor symbol."), pdef("variances", ParamType::Code, true, "Array of variance strings up/down.")]), handle_rewrite_indices_tensor),
        centry("evaluate_components", "Evaluate tensor components from rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("rules", ParamType::ExprId, true, "Stored rules list.")]), handle_evaluate_components_tensor),
        centry("evaluate", "Evaluate tensor components from rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("rules", ParamType::ExprId, true, "Stored rules list.")]), handle_evaluate_components_tensor),
        centry("eval_components", "Evaluate tensor components from rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("rules", ParamType::ExprId, true, "Stored rules list.")]), handle_evaluate_components_tensor),
        centry("complete_inverse_metric", "Complete inverse-metric component rules.", ps(vec![pdef("rules", ParamType::ExprId, true, "Stored component rules list."), pdef("metric", ParamType::Symbol, true, "Metric symbol."), pdef("inverse_metric", ParamType::Symbol, true, "Inverse-metric symbol."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_complete_inverse_metric),
        centry("diff_component", "Differentiate a tensor component expression.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("variable", ParamType::Symbol, true, "Differentiation variable.")]), handle_diff_component_tensor),
        centry("christoffel", "Compute Christoffel symbols from a stored metric.", ps(vec![pdef("metric_id", ParamType::Code, true, "Stored metric id.")]), handle_metric_pipeline_christoffel),
        centry("christoffel_from_metric", "Compute Christoffel symbols from a stored metric.", ps(vec![pdef("metric_id", ParamType::Code, true, "Stored metric id.")]), handle_metric_pipeline_christoffel),
        centry("riemann", "Compute a Riemann tensor from stored Christoffel symbols.", ps(vec![pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id.")]), handle_riemann_from_christoffel),
        centry("riemann_from_christoffel", "Compute a Riemann tensor from stored Christoffel symbols.", ps(vec![pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id.")]), handle_riemann_from_christoffel),
        centry("ricci", "Contract a stored Riemann tensor to the Ricci tensor.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_ricci_from_riemann),
        centry("ricci_from_riemann", "Contract a stored Riemann tensor to the Ricci tensor.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_ricci_from_riemann),
        centry("scalar_curvature", "Contract a stored Ricci tensor with the inverse of the stored metric.", ps(vec![pdef("ricci_id", ParamType::Code, true, "Stored ricci id.")]), handle_scalar_curvature_gr),
        centry("ricci_scalar", "Contract a Ricci tensor with an inverse metric.", ps(vec![pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."), pdef("metric_inverse", ParamType::ExprId, true, "Stored inverse metric matrix expression id.")]), handle_ricci_scalar_gr),
        centry("einstein_tensor", "Build the Einstein tensor from metric, Ricci tensor, and Ricci scalar, or from stored Ricci/metric ids.", ps(vec![
            pdef("ricci_id", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional stored ricci id."),
            pdef("metric_id", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional stored metric id."),
            pdef("ricci", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored scalar expression id."),
            pdef("metric", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored metric matrix expression id."),
        ]), handle_einstein_tensor_gr),
        centry("einstein", "Build the Einstein tensor from metric, Ricci tensor, and Ricci scalar, or from stored Ricci/metric ids.", ps(vec![
            pdef("ricci_id", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional stored ricci id."),
            pdef("metric_id", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional stored metric id."),
            pdef("ricci", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored scalar expression id."),
            pdef("metric", ParamType::Optional(Box::new(ParamType::ExprId)), false, "Stored metric matrix expression id."),
        ]), handle_einstein_tensor_gr),
        centry("weyl_from_curvature", "Compute the component Weyl tensor from Riemann, Ricci, scalar curvature, and metric inputs.", ps(vec![
            pdef("riemann", ParamType::ExprId, true, "Stored rank-4 Riemann tensor list expression id."),
            pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::ExprId, true, "Stored scalar-curvature expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
        ]), handle_weyl_from_curvature_expr),
        centry("weyl_from_riemann", "Alias for the component Weyl-tensor computation from curvature inputs.", ps(vec![
            pdef("riemann", ParamType::ExprId, true, "Stored rank-4 Riemann tensor list expression id."),
            pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::ExprId, true, "Stored scalar-curvature expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
        ]), handle_weyl_from_curvature_expr),
        centry("cotton_from_curvature", "Compute the component Cotton tensor from Ricci, scalar curvature, Christoffel symbols, metric, and coordinates.", ps(vec![
            pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::ExprId, true, "Stored scalar-curvature expression id."),
            pdef("gamma", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_cotton_from_curvature_expr),
        centry("bach_from_curvature", "Compute the component Bach tensor from Weyl, Ricci, Christoffel symbols, metric, and coordinates.", ps(vec![
            pdef("weyl", ParamType::ExprId, true, "Stored rank-4 Weyl tensor list expression id."),
            pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."),
            pdef("gamma", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_bach_from_curvature_expr),
        centry("contorsion_tensor", "Compute the contorsion tensor from torsion and metric inputs.", ps(vec![
            pdef("torsion", ParamType::ExprId, true, "Stored rank-3 torsion tensor list expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
        ]), handle_contorsion_tensor_expr),
        centry("connection_with_torsion", "Compose a torsionful connection from Christoffel symbols and contorsion.", ps(vec![
            pdef("christoffel", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("contorsion", ParamType::ExprId, true, "Stored rank-3 contorsion tensor list expression id."),
        ]), handle_connection_with_torsion_expr),
        centry("spin_connection", "Compute the torsion-free spin connection from a vielbein and metric.", ps(vec![
            pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_spin_connection_expr),
        centry("first_cartan_structure", "Compute the first Cartan structure equations as differential forms.", ps(vec![
            pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id."),
            pdef("spin_connection", ParamType::ExprId, true, "Stored rank-3 spin-connection tensor list expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_first_cartan_structure_expr),
        centry("second_cartan_structure", "Compute the second Cartan structure equations as differential forms.", ps(vec![
            pdef("spin_connection", ParamType::ExprId, true, "Stored rank-3 spin-connection tensor list expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_second_cartan_structure_expr),
        centry("conformal_transform_metric", "Conformally rescale a metric by Omega^2.", ps(vec![
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("omega", ParamType::ExprId, true, "Stored conformal-factor expression id."),
        ]), handle_conformal_transform_metric_expr),
        centry("conformal_transform_inverse_metric", "Conformally rescale an inverse metric by Omega^-2.", ps(vec![
            pdef("inverse_metric", ParamType::ExprId, true, "Stored inverse-metric matrix expression id."),
            pdef("omega", ParamType::ExprId, true, "Stored conformal-factor expression id."),
        ]), handle_conformal_transform_inverse_metric_expr),
        centry("conformal_transform_christoffel", "Transform Christoffel symbols under g_tilde = Omega^2 g.", ps(vec![
            pdef("gamma", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("omega", ParamType::ExprId, true, "Stored conformal-factor expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_conformal_transform_christoffel_expr),
        centry("conformal_transform_ricci", "Transform the Ricci tensor under g_tilde = Omega^2 g.", ps(vec![
            pdef("ricci", ParamType::ExprId, true, "Stored Ricci matrix expression id."),
            pdef("scalar", ParamType::ExprId, true, "Stored scalar-curvature expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("omega", ParamType::ExprId, true, "Stored conformal-factor expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_conformal_transform_ricci_expr),
        centry("conformal_transform_scalar", "Transform the scalar curvature under g_tilde = Omega^2 g.", ps(vec![
            pdef("scalar", ParamType::ExprId, true, "Stored scalar-curvature expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("omega", ParamType::ExprId, true, "Stored conformal-factor expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_conformal_transform_scalar_expr),
        centry("killing_equations", "Generate the symmetric Killing system for unknown covector components from a connection and coordinate list.", ps(vec![
            pdef("gamma", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
            pdef("field_prefix", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional prefix for the unknown covector components."),
        ]), handle_killing_equations_expr),
        centry("adm_decompose", "Compute the ADM decomposition of a metric into lapse, shift, spatial metric, extrinsic curvature, and constraints.", ps(vec![
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
            pdef("time_coord", ParamType::Integer, true, "Index of the time coordinate."),
        ]), handle_adm_decompose_expr),
        centry("spatial_christoffel", "Compute Christoffel symbols for a spatial metric.", ps(vec![
            pdef("gamma_ij", ParamType::ExprId, true, "Stored spatial metric matrix expression id."),
            pdef("spatial_coords", ParamType::ExprId, true, "Stored spatial-coordinate-list expression id."),
        ]), handle_spatial_christoffel_expr),
        centry("spatial_ricci_tensor", "Compute the Ricci tensor of a spatial metric.", ps(vec![
            pdef("gamma_ij", ParamType::ExprId, true, "Stored spatial metric matrix expression id."),
            pdef("spatial_coords", ParamType::ExprId, true, "Stored spatial-coordinate-list expression id."),
        ]), handle_spatial_ricci_tensor_expr),
        centry("spatial_ricci_scalar", "Compute the Ricci scalar of a spatial metric.", ps(vec![
            pdef("gamma_ij", ParamType::ExprId, true, "Stored spatial metric matrix expression id."),
            pdef("spatial_coords", ParamType::ExprId, true, "Stored spatial-coordinate-list expression id."),
        ]), handle_spatial_ricci_scalar_expr),
        centry("null_tetrad", "Auto-construct a Newman-Penrose null tetrad for a diagonal Lorentzian 4-metric.", ps(vec![
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_null_tetrad_expr),
        centry("null_tetrad_from_metric", "Alias for null_tetrad(metric, [coords...]) using the public tensor-algorithm name.", ps(vec![
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_null_tetrad_expr),
        centry("verify_null_tetrad", "Verify NP null-tetrad normalization and orthogonality against a metric.", ps(vec![
            pdef("tetrad", ParamType::ExprId, true, "Stored 4-list null-tetrad expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
        ]), handle_verify_null_tetrad_expr),
        centry("spin_coefficients", "Compute the Newman-Penrose spin coefficients from a tetrad, connection, metric, and coordinates.", ps(vec![
            pdef("tetrad", ParamType::ExprId, true, "Stored 4-list null-tetrad expression id."),
            pdef("gamma", ParamType::ExprId, true, "Stored rank-3 Christoffel tensor list expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("coords", ParamType::ExprId, true, "Stored coordinate-list expression id."),
        ]), handle_spin_coefficients_expr),
        centry("weyl_scalars", "Compute the Newman-Penrose Weyl scalars from a Weyl tensor and null tetrad.", ps(vec![
            pdef("weyl", ParamType::ExprId, true, "Stored rank-4 Weyl tensor list expression id."),
            pdef("tetrad", ParamType::ExprId, true, "Stored 4-list null-tetrad expression id."),
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
        ]), handle_weyl_scalars_expr),
        centry("petrov_classify", "Classify the Weyl tensor algebraically from Newman-Penrose scalars.", ps(vec![
            pdef("scalars", ParamType::ExprId, true, "Stored 5-list Weyl-scalars expression id."),
        ]), handle_petrov_classify_expr),
        centry("kretschner_scalar", "Compute the Kretschmann scalar.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_kretschner_scalar_gr),
        centry("kretschner", "Compute the Kretschmann scalar.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_kretschner_scalar_gr),
        centry("kretschmann_scalar_diagonal_approx", "Compute the diagonal-only Kretschmann approximation.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_kretschmann_scalar_diagonal_approx_gr),
        centry("vielbein", "Validate and return a vielbein matrix.", ps(vec![pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id.")]), handle_vielbein_gr),
        centry("inverse_vielbein", "Invert a vielbein matrix.", ps(vec![pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id.")]), handle_inverse_vielbein_gr),
        centry("inv_vielbein", "Invert a vielbein matrix.", ps(vec![pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id.")]), handle_inverse_vielbein_gr),
        centry("metric_from_vielbein", "Build the metric from a vielbein and frame metric.", ps(vec![
            pdef("vielbein", ParamType::ExprId, true, "Stored vielbein matrix expression id."),
            pdef("eta", ParamType::ExprId, true, "Stored frame metric matrix expression id."),
        ]), handle_metric_from_vielbein_gr),
        centry("vielbein_from_metric_diagonal", "Construct a diagonal vielbein from a diagonal metric.", ps(vec![
            pdef("metric", ParamType::ExprId, true, "Stored metric matrix expression id."),
            pdef("signature", ParamType::StringEnum(&["mostly_plus", "mostly_minus"]), true, "Frame-signature convention."),
        ]), handle_vielbein_from_metric_diagonal_gr),
        centry("weyl_curvature", "Compute the Weyl curvature tensor from stored curvature data.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_weyl_curvature_gr),
        centry("covariant_derivative_vector", "Covariant derivative of a vector.", ps(vec![pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id."), pdef("coord_index", ParamType::Integer, true, "Coordinate slot.")]), handle_covariant_derivative_vector_gr),
        centry("covariant_diff", "Covariant derivative of a vector.", ps(vec![pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id."), pdef("coord_index", ParamType::Integer, true, "Coordinate slot.")]), handle_covariant_derivative_vector_gr),
        centry("covariant_derivative_covector", "Covariant derivative of a covector.", ps(vec![pdef("covector", ParamType::ExprId, true, "Stored covector expression id."), pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id."), pdef("coord_index", ParamType::Integer, true, "Coordinate slot.")]), handle_covariant_derivative_covector_gr),
        centry("covariant_derivative_tensor2", "Covariant derivative of a rank-2 tensor.", ps(vec![pdef("tensor", ParamType::ExprId, true, "Stored matrix expression id."), pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id."), pdef("coord_index", ParamType::Integer, true, "Coordinate slot.")]), handle_covariant_derivative_tensor2_gr),
        centry("geodesic_equations", "Geodesic equations from a connection.", ps(vec![pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id.")]), handle_geodesic_equations_gr),
        centry("geodesic", "Geodesic equations from a connection.", ps(vec![pdef("christoffel_id", ParamType::Code, true, "Stored christoffel id.")]), handle_geodesic_equations_gr),
        centry("lie_derivative_scalar", "Lie derivative of a scalar field.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_lie_derivative_scalar_gr),
        centry("lie_derivative", "Lie derivative of a scalar field.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored scalar expression id."), pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_lie_derivative_scalar_gr),
        centry("lie_derivative_vector", "Lie derivative of a vector field.", ps(vec![pdef("field", ParamType::ExprId, true, "Stored vector expression id."), pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_lie_derivative_vector_gr),
        centry("pauli_x", "Return the Pauli sigma_x matrix.", ps(vec![]), handle_pauli_x),
        centry("sigma_x", "Return the Pauli sigma_x matrix.", ps(vec![]), handle_pauli_x),
        centry("pauli_y", "Return the Pauli sigma_y matrix.", ps(vec![]), handle_pauli_y),
        centry("sigma_y", "Return the Pauli sigma_y matrix.", ps(vec![]), handle_pauli_y),
        centry("pauli_z", "Return the Pauli sigma_z matrix.", ps(vec![]), handle_pauli_z),
        centry("sigma_z", "Return the Pauli sigma_z matrix.", ps(vec![]), handle_pauli_z),
        centry("jz", "Return the exact spin-j J_z matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jz),
        centry("jz_matrix", "Return the exact spin-j J_z matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jz),
        centry("jplus", "Return the exact spin-j raising operator J_+.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jplus),
        centry("jplus_matrix", "Return the exact spin-j raising operator J_+.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jplus),
        centry("jminus", "Return the exact spin-j lowering operator J_-.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jminus),
        centry("jminus_matrix", "Return the exact spin-j lowering operator J_-.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jminus),
        centry("jx", "Return the exact spin-j J_x matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jx),
        centry("jx_matrix", "Return the exact spin-j J_x matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jx),
        centry("jy", "Return the exact spin-j J_y matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jy),
        centry("jy_matrix", "Return the exact spin-j J_y matrix.", ps(vec![pdef("two_j", ParamType::Integer, true, "Exact integer value 2j.")]), handle_jy),
        centry("singlet_state_2spinhalf", "Return the explicit two-spin-1/2 singlet state.", ps(vec![]), handle_singlet_state_2spinhalf),
        centry("triplet_states_2spinhalf", "Return the explicit two-spin-1/2 triplet states.", ps(vec![]), handle_triplet_states_2spinhalf),
        centry("singlet_projector_2spinhalf", "Return the explicit two-spin-1/2 singlet projector.", ps(vec![]), handle_singlet_projector_2spinhalf),
        centry("triplet_projector_2spinhalf", "Return the explicit two-spin-1/2 triplet projector.", ps(vec![]), handle_triplet_projector_2spinhalf),
        centry("two_spin_half_singlet_state", "Return the explicit two-spin-1/2 singlet state.", ps(vec![]), handle_singlet_state_2spinhalf),
        centry("two_spin_half_triplet_states", "Return the explicit two-spin-1/2 triplet states.", ps(vec![]), handle_triplet_states_2spinhalf),
        centry("two_spin_half_singlet_projector", "Return the explicit two-spin-1/2 singlet projector.", ps(vec![]), handle_singlet_projector_2spinhalf),
        centry("two_spin_half_triplet_projector", "Return the explicit two-spin-1/2 triplet projector.", ps(vec![]), handle_triplet_projector_2spinhalf),
        centry("time_evolution_operator", "Return the constant-Hamiltonian propagator U(t) = exp(-i t H) for supported small Hermitian matrices.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix id."), pdef("t", ParamType::ExprId, true, "Stored time expression id.")]), handle_time_evolution_operator),
        centry("schrodinger_evolve", "Evolve a pure state vector via psi(t) = U(t) psi0 for supported finite-dimensional Hermitian Hamiltonians.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix id."), pdef("psi0", ParamType::ExprId, true, "Stored initial state-vector expression id."), pdef("t", ParamType::ExprId, true, "Stored time expression id.")]), handle_schrodinger_evolve),
        centry("heisenberg_evolve", "Evolve an operator via O(t) = U†(t) O0 U(t) for supported finite-dimensional Hermitian Hamiltonians.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix id."), pdef("O0", ParamType::ExprId, true, "Stored initial operator-matrix expression id."), pdef("t", ParamType::ExprId, true, "Stored time expression id.")]), handle_heisenberg_evolve),
        centry("liouville_rhs", "Return the closed-system density-matrix right-hand side ρ̇ = -i [H, ρ].", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_liouville_rhs),
        centry("dyson_series", "Return the finite-order symbolic Dyson expansion for a time-dependent Hamiltonian.", ps(vec![pdef("Ht", ParamType::ExprId, true, "Stored time-dependent Hamiltonian expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_dyson_series),
        centry("magnus_expansion", "Return the finite-order symbolic Magnus expansion for a time-dependent Hamiltonian.", ps(vec![pdef("Ht", ParamType::ExprId, true, "Stored time-dependent Hamiltonian expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_magnus_expansion),
        centry("kubo_response", "Construct the symbolic Kubo linear-response function.", ps(vec![pdef("A", ParamType::ExprId, true, "Stored operator expression id for A."), pdef("B", ParamType::ExprId, true, "Stored operator expression id for B."), pdef("rho0", ParamType::ExprId, true, "Stored reference density operator expression id."), pdef("t", ParamType::ExprId, true, "Stored time expression id.")]), handle_kubo_response_qm),
        centry("kubo_response_function", "Construct the symbolic Kubo linear-response function.", ps(vec![pdef("A", ParamType::ExprId, true, "Stored operator expression id for A."), pdef("B", ParamType::ExprId, true, "Stored operator expression id for B."), pdef("rho0", ParamType::ExprId, true, "Stored reference density operator expression id."), pdef("t", ParamType::ExprId, true, "Stored time expression id.")]), handle_kubo_response_qm),
        centry("susceptibility_fourier", "Construct the symbolic Fourier susceptibility integral.", ps(vec![pdef("chi_t", ParamType::ExprId, true, "Stored time-domain response expression id."), pdef("omega", ParamType::ExprId, true, "Stored frequency expression id.")]), handle_susceptibility_fourier_qm),
        centry("projector_left", "Construct the canonical left chiral projector P_L = (1 - gamma5)/2.", ps(vec![]), handle_projector_left_qm),
        centry("P_L", "Construct the canonical left chiral projector P_L = (1 - gamma5)/2.", ps(vec![]), handle_projector_left_qm),
        centry("projector_right", "Construct the canonical right chiral projector P_R = (1 + gamma5)/2.", ps(vec![]), handle_projector_right_qm),
        centry("P_R", "Construct the canonical right chiral projector P_R = (1 + gamma5)/2.", ps(vec![]), handle_projector_right_qm),
        centry("simplify_chiral", "Simplify chiral projector algebra and Weyl-spinor projector actions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify_chiral_qm),
        centry("simplify_chiral_projectors", "Simplify chiral projector algebra and Weyl-spinor projector actions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify_chiral_qm),
        centry("simplify_spinor_bilinears", "Apply metadata-driven 4D Majorana and Weyl bilinear selection rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify_spinor_bilinears_qm),
        centry("simplify_spinor_bilinear_selection_rules", "Apply metadata-driven 4D Majorana and Weyl bilinear selection rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify_spinor_bilinears_qm),
        centry("insert_explicit_spinor_indices", "Insert explicit spinor indices into supported implicit bilinears and gamma chains.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_insert_explicit_spinor_indices_qm),
        centry("remove_trivial_spinor_indices", "Collapse unambiguous explicit spinor-index contractions back to implicit form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_remove_trivial_spinor_indices_qm),
        centry("sigma", "Construct the canonical sigma(mu,nu) spin-generator basis element.", ps(vec![pdef("mu", ParamType::ExprId, true, "Stored first Lorentz-index expression id."), pdef("nu", ParamType::ExprId, true, "Stored second Lorentz-index expression id.")]), handle_sigma_matrix_qm),
        centry("sigma_matrix", "Construct the canonical sigma(mu,nu) spin-generator basis element.", ps(vec![pdef("mu", ParamType::ExprId, true, "Stored first Lorentz-index expression id."), pdef("nu", ParamType::ExprId, true, "Stored second Lorentz-index expression id.")]), handle_sigma_matrix_qm),
        centry("sigma_to_gamma", "Expand sigma(mu,nu) to its gamma commutator definition.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sigma_to_gamma_qm),
        centry("sigma_to_gamma_commutator", "Expand sigma(mu,nu) to its gamma commutator definition.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sigma_to_gamma_qm),
        centry("gamma_to_sigma", "Convert an exact gamma commutator pattern to the sigma basis.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_gamma_to_sigma_qm),
        centry("gamma_commutator_to_sigma", "Convert an exact gamma commutator pattern to the sigma basis.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_gamma_to_sigma_qm),
        centry("gamma5", "Return the Dirac gamma_5 matrix.", ps(vec![]), handle_gamma5),
        centry("gamma_trace", "Trace a gamma-matrix chain.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored gamma expression or index list id.")]), handle_gamma_trace_qm),
        centry("gamma5_trace", "Trace a gamma-matrix chain.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored gamma expression or index list id.")]), handle_gamma_trace_qm),
        centry("join_gamma", "Join adjacent gamma factors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_join_gamma_qm),
        centry("join_gammas_in_expr", "Join adjacent gamma factors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_join_gamma_qm),
        centry("split_gamma", "Split compact gamma chains.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_split_gamma_qm),
        centry("expand_diracbar", "Expand Dirac-barred gamma-spinor products.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_diracbar_qm),
        centry("expand_bar", "Expand Dirac-barred gamma-spinor products.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_diracbar_qm),
        centry("sort_spinors", "Sort spinor bilinears into Dirac-bar gamma spinor order.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sort_spinors_qm),
        centry("diracbar_sort", "Sort spinor bilinears into Dirac-bar gamma spinor order.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_sort_spinors_qm),
        centry("fierz", "Perform a Fierz rearrangement.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_fierz_qm),
        centry("commutator", "Operator commutator.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored expression id."), pdef("rhs", ParamType::ExprId, true, "Stored expression id.")]), handle_commutator_qm),
        centry("anticommutator", "Operator anticommutator.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored expression id."), pdef("rhs", ParamType::ExprId, true, "Stored expression id.")]), handle_anticommutator_qm),
        centry("density_matrix", "Build a density matrix from a state vector.", ps(vec![pdef("state", ParamType::ExprId, true, "Stored state-vector expression id.")]), handle_density_matrix_qm),
        centry("density", "Build a density matrix from a state vector.", ps(vec![pdef("state", ParamType::ExprId, true, "Stored state-vector expression id.")]), handle_density_matrix_qm),
        centry("number_state", "Construct a bosonic number state.", ps(vec![pdef("mode", ParamType::Symbol, true, "Oscillator mode symbol."), pdef("n", ParamType::Integer, true, "Occupation number.")]), handle_number_state_qm),
        centry("bosonic_fock_basis_state", "Construct a validated bosonic basis state for a declared Fock space.", ps(vec![
            pdef("space_symbol", ParamType::Symbol, true, "Declared Fock-space symbol."),
            pdef("occupations", ParamType::Code, true, "JSON array of occupations."),
        ]), handle_bosonic_fock_basis_state_qm),
        centry("fermionic_fock_basis_state", "Construct a validated fermionic basis state for a declared Fock space.", ps(vec![
            pdef("space_symbol", ParamType::Symbol, true, "Declared Fock-space symbol."),
            pdef("occupations", ParamType::Code, true, "JSON array of 0/1 occupations."),
        ]), handle_fermionic_fock_basis_state_qm),
        centry("fock_state", "Construct a multimode bosonic occupation-basis state.", ps(vec![pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing occupations.")]), handle_fock_state_qm),
        centry("fermion_state", "Construct a multimode fermionic occupation-basis state.", ps(vec![pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing 0/1 occupations.")]), handle_fermion_state_qm),
        centry("bosonic_creation_action", "Apply bosonic creation to one mode of a multimode basis state.", ps(vec![pdef("mode", ParamType::Integer, true, "Zero-based mode index."), pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing occupations.")]), handle_bosonic_creation_action_qm),
        centry("fermionic_creation_action", "Apply fermionic creation to one mode of a multimode basis state.", ps(vec![pdef("mode", ParamType::Integer, true, "Zero-based mode index."), pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing 0/1 occupations.")]), handle_fermionic_creation_action_qm),
        centry("bosonic_annihilation_action", "Apply bosonic annihilation to one mode of a multimode basis state.", ps(vec![pdef("mode", ParamType::Integer, true, "Zero-based mode index."), pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing occupations.")]), handle_bosonic_annihilation_action_qm),
        centry("fermionic_annihilation_action", "Apply fermionic annihilation to one mode of a multimode basis state.", ps(vec![pdef("mode", ParamType::Integer, true, "Zero-based mode index."), pdef("occupations", ParamType::ExprId, true, "Stored list expression id containing 0/1 occupations.")]), handle_fermionic_annihilation_action_qm),
        centry("vacuum", "Construct the oscillator vacuum state.", ps(vec![pdef("mode", ParamType::Symbol, true, "Oscillator mode symbol.")]), handle_vacuum_qm),
        centry("number_operator", "Construct the oscillator number operator.", ps(vec![pdef("mode", ParamType::Symbol, true, "Oscillator mode symbol.")]), handle_number_operator_qm),
        centry("hamiltonian_ho", "Construct the harmonic-oscillator Hamiltonian.", ps(vec![pdef("mode", ParamType::Symbol, true, "Oscillator mode symbol."), pdef("hbar", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional Planck constant expression."), pdef("omega", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional angular-frequency expression.")]), handle_hamiltonian_ho_qm),
        centry("apply_operator", "Apply an abstract operator to a state.", ps(vec![pdef("op", ParamType::ExprId, true, "Stored operator expression id."), pdef("state", ParamType::ExprId, true, "Stored state expression id.")]), handle_apply_operator_qm),
        centry("partial_trace", "Take a subsystem partial trace.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("dim_a", ParamType::Integer, true, "Subsystem A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem B dimension."), pdef("which", ParamType::StringEnum(&["A", "B"]), true, "Subsystem to trace out.")]), handle_partial_trace_qm),
        centry("partial_trace_factor", "Take a factor-based partial trace using explicit tensor-product dimensions.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("dims", ParamType::Code, true, "JSON array of factor dimensions."), pdef("factor_index", ParamType::Integer, true, "Factor index to trace out.")]), handle_partial_trace_factor_qm),
        centry("partial_transpose_factor", "Take a factor-based partial transpose using explicit tensor-product dimensions.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("dims", ParamType::Code, true, "JSON array of factor dimensions."), pdef("factor_index", ParamType::Integer, true, "Factor index to transpose.")]), handle_partial_transpose_factor_qm),
        centry("permute_subsystems", "Permute subsystem order using explicit tensor-product dimensions.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("dims", ParamType::Code, true, "JSON array of factor dimensions."), pdef("permutation", ParamType::Code, true, "JSON array describing the new subsystem order.")]), handle_permute_subsystems_qm),
        centry("partial_trace_space", "Take a factor-based partial trace using declared composite-space metadata.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("composite_space_symbol", ParamType::Symbol, true, "Declared composite Hilbert-space symbol."), pdef("factor_space_symbol", ParamType::Symbol, true, "Factor Hilbert-space symbol to trace out.")]), handle_partial_trace_space_qm),
        centry("braket", "Bra-ket inner product.", ps(vec![pdef("bra", ParamType::ExprId, true, "Stored bra/list expression id."), pdef("ket", ParamType::ExprId, true, "Stored ket/list expression id.")]), handle_braket_qm),
        centry("outer", "Outer-product operator.", ps(vec![pdef("left", ParamType::ExprId, true, "Stored vector id."), pdef("right", ParamType::ExprId, true, "Stored vector id.")]), handle_outer_qm),
        centry("basis_projector", "Projector onto a computational-basis state.", ps(vec![pdef("index", ParamType::Integer, true, "Basis-state index."), pdef("dim", ParamType::Integer, true, "Hilbert-space dimension.")]), handle_basis_projector_qm),
        centry("measurement_probabilities", "Projective-measurement probabilities for a density matrix.", ps(vec![pdef("projectors", ParamType::ExprId, true, "Stored rank-3 projector-list expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_measurement_probabilities_qm),
        centry("expectation_value", "Expectation value Tr(rho * operator) for a density matrix.", ps(vec![pdef("operator", ParamType::ExprId, true, "Stored observable matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_expectation_value_qm),
        centry("variance", "Observable variance for a density matrix.", ps(vec![pdef("operator", ParamType::ExprId, true, "Stored observable matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_variance_qm),
        centry("purity", "Purity Tr(rho^2) for a density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_purity_qm),
        centry("linear_entropy", "Linear entropy 1 - Tr(rho^2) for a density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_linear_entropy_qm),
        centry("participation_ratio", "Participation ratio 1 / Tr(rho^2) for a density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_participation_ratio_qm),
        centry("renyi2_entropy", "Renyi-2 entropy -log(Tr(rho^2)) for a density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_renyi2_entropy_qm),
        centry("renyi2_entropy_factor", "Renyi-2 entropy of the reduced state obtained by keeping one tensor factor.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id."), pdef("dims", ParamType::Code, true, "JSON array of factor dimensions."), pdef("kept_factor", ParamType::Integer, true, "Factor index to keep.")]), handle_renyi2_entropy_factor_qm),
        centry("von_neumann_entropy", "Von Neumann entropy -Tr(rho log rho) for a supported finite-dimensional Hermitian density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_von_neumann_entropy_qm),
        centry("mutual_information", "Bipartite von Neumann mutual information from a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_mutual_information_qm),
        centry("conditional_entropy", "Bipartite conditional entropy from a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_conditional_entropy_qm),
        centry("von_neumann_mutual_information_bipartite", "Bipartite von Neumann mutual information from a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_mutual_information_qm),
        centry("conditional_entropy_b_given_a", "Bipartite conditional entropy from a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_conditional_entropy_qm),
        centry("entanglement_spectrum", "Bipartite entanglement spectrum for a pure-state vector or bipartite density matrix, keeping subsystem A.", ps(vec![pdef("state_or_rho", ParamType::ExprId, true, "Stored bipartite state-vector or density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_entanglement_spectrum_qm),
        centry("schmidt_coefficients", "Schmidt coefficients of a bipartite pure-state vector.", ps(vec![pdef("state", ParamType::ExprId, true, "Stored bipartite pure-state vector expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_schmidt_coefficients_qm),
        centry("negativity", "Bipartite negativity from the supported partial-transpose spectrum of a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_negativity_qm),
        centry("logarithmic_negativity", "Bipartite logarithmic negativity log(1 + 2 N(rho)) from a supported density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_logarithmic_negativity_qm),
        centry("renyi2_mutual_information", "Bipartite Renyi-2 mutual information from a density matrix.", ps(vec![pdef("rho_ab", ParamType::ExprId, true, "Stored bipartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension.")]), handle_renyi2_mutual_information_qm),
        centry("renyi2_tripartite_information", "Tripartite Renyi-2 information from a density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored tripartite density-matrix expression id."), pdef("dim_a", ParamType::Integer, true, "Subsystem-A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem-B dimension."), pdef("dim_c", ParamType::Integer, true, "Subsystem-C dimension.")]), handle_renyi2_tripartite_information_qm),
        centry("hermitian_eigenvalues", "Exact small Hermitian eigenvalues for supported matrices.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_hermitian_eigenvalues_qm),
        centry("hermitian_eigenprojectors", "Exact small Hermitian spectral projectors for supported nondegenerate matrices.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_hermitian_eigenprojectors_qm),
        centry("first_order_energy_shift", "Exact nondegenerate stationary perturbation-theory first-order energy shift in the eigenbasis of H0.", ps(vec![pdef("H0", ParamType::ExprId, true, "Stored unperturbed Hamiltonian matrix id."), pdef("V", ParamType::ExprId, true, "Stored perturbation matrix id."), pdef("n", ParamType::Integer, true, "Zero-based unperturbed state index.")]), handle_first_order_energy_shift_qm),
        centry("second_order_energy_shift", "Exact nondegenerate stationary perturbation-theory second-order energy shift in the eigenbasis of H0.", ps(vec![pdef("H0", ParamType::ExprId, true, "Stored unperturbed Hamiltonian matrix id."), pdef("V", ParamType::ExprId, true, "Stored perturbation matrix id."), pdef("n", ParamType::Integer, true, "Zero-based unperturbed state index.")]), handle_second_order_energy_shift_qm),
        centry("degenerate_effective_perturbation", "Exact effective perturbation matrix inside a chosen degenerate basis-state subspace of a diagonal H0.", ps(vec![pdef("H0", ParamType::ExprId, true, "Stored unperturbed Hamiltonian matrix id."), pdef("V", ParamType::ExprId, true, "Stored perturbation matrix id."), pdef("subspace", ParamType::ExprId, true, "Stored non-empty list of degenerate basis-state indices.")]), handle_degenerate_effective_perturbation_qm),
        centry("degenerate_first_order_splittings", "Exact first-order splittings from diagonalizing the effective perturbation inside a chosen degenerate basis-state subspace.", ps(vec![pdef("H0", ParamType::ExprId, true, "Stored unperturbed Hamiltonian matrix id."), pdef("V", ParamType::ExprId, true, "Stored perturbation matrix id."), pdef("subspace", ParamType::ExprId, true, "Stored non-empty list of degenerate basis-state indices.")]), handle_degenerate_first_order_splittings_qm),
        centry("berry_connection", "Construct the symbolic Berry-connection one-form component.", ps(vec![pdef("psi", ParamType::ExprId, true, "Stored state expression id."), pdef("parameter", ParamType::ExprId, true, "Stored parameter expression id.")]), handle_berry_connection_qm),
        centry("geometric_phase", "Construct the symbolic geometric phase as a contour-style integral.", ps(vec![pdef("A", ParamType::ExprId, true, "Stored Berry-connection or line-integrand expression id."), pdef("parameter", ParamType::ExprId, true, "Stored parameter expression id.")]), handle_geometric_phase_qm),
        centry("bloch_vector", "Bloch-vector components for a 2x2 density matrix.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_bloch_vector_qm),
        centry("qubit_density_from_bloch", "Qubit density matrix from a Bloch vector.", ps(vec![pdef("r", ParamType::ExprId, true, "Stored length-3 list expression id.")]), handle_qubit_density_from_bloch_qm),
        centry("post_measurement_state", "Normalized post-measurement state for an outcome projector.", ps(vec![pdef("projector", ParamType::ExprId, true, "Stored projector matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id."), pdef("outcome_index", ParamType::Integer, true, "Outcome label used for diagnostics.")]), handle_post_measurement_state_qm),
        centry("identity_channel", "Construct a finite-dimensional identity Kraus channel.", ps(vec![pdef("dim", ParamType::Integer, true, "Hilbert-space dimension.")]), handle_identity_channel_qm),
        centry("depolarizing_channel", "Construct the canonical qubit depolarizing channel.", ps(vec![pdef("p", ParamType::ExprId, true, "Stored depolarizing probability expression id.")]), handle_depolarizing_channel_qm),
        centry("dephasing_channel", "Construct the canonical qubit dephasing channel.", ps(vec![pdef("p", ParamType::ExprId, true, "Stored dephasing probability expression id.")]), handle_dephasing_channel_qm),
        centry("amplitude_damping_channel", "Construct the canonical qubit amplitude-damping channel.", ps(vec![pdef("gamma", ParamType::ExprId, true, "Stored damping parameter expression id.")]), handle_amplitude_damping_channel_qm),
        centry("bit_flip_channel", "Construct the canonical qubit bit-flip channel.", ps(vec![pdef("p", ParamType::ExprId, true, "Stored flip probability expression id.")]), handle_bit_flip_channel_qm),
        centry("phase_flip_channel", "Construct the canonical qubit phase-flip channel.", ps(vec![pdef("p", ParamType::ExprId, true, "Stored flip probability expression id.")]), handle_phase_flip_channel_qm),
        centry("bit_phase_flip_channel", "Construct the canonical qubit bit-phase-flip channel.", ps(vec![pdef("p", ParamType::ExprId, true, "Stored flip probability expression id.")]), handle_bit_phase_flip_channel_qm),
        centry("compose_channels", "Compose two Kraus channels so the right channel acts first and the left channel acts second.", ps(vec![pdef("left", ParamType::ExprId, true, "Stored Kraus-list expression id for the outer channel."), pdef("right", ParamType::ExprId, true, "Stored Kraus-list expression id for the inner channel.")]), handle_compose_channels_qm),
        centry("tensor_product_channel", "Form the tensor-product Kraus channel whose operators are L_i tensor R_j.", ps(vec![pdef("left", ParamType::ExprId, true, "Stored Kraus-list expression id for the left channel."), pdef("right", ParamType::ExprId, true, "Stored Kraus-list expression id for the right channel.")]), handle_tensor_product_channel_qm),
        centry("choi_distance", "Compute the Frobenius distance between two channels using their Choi matrices.", ps(vec![pdef("left", ParamType::ExprId, true, "Stored Kraus-list expression id for the left channel."), pdef("right", ParamType::ExprId, true, "Stored Kraus-list expression id for the right channel.")]), handle_choi_distance_qm),
        centry("trace_preserving_residual", "Compute the exact trace-preserving residual Σ_k K_k† K_k - I for a Kraus channel.", ps(vec![pdef("kraus", ParamType::ExprId, true, "Stored Kraus-list expression id for the channel.")]), handle_trace_preserving_residual_qm),
        centry("is_trace_preserving", "Check whether a Kraus channel is exactly trace preserving.", ps(vec![pdef("kraus", ParamType::ExprId, true, "Stored Kraus-list expression id for the channel.")]), handle_is_trace_preserving_qm),
        centry("unital_residual", "Compute the exact unital residual Σ_k K_k K_k† - I for a Kraus channel.", ps(vec![pdef("kraus", ParamType::ExprId, true, "Stored Kraus-list expression id for the channel.")]), handle_unital_residual_qm),
        centry("is_unital", "Check whether a Kraus channel is exactly unital.", ps(vec![pdef("kraus", ParamType::ExprId, true, "Stored Kraus-list expression id for the channel.")]), handle_is_unital_qm),
        centry("apply_channel", "Apply a Kraus channel to a density matrix.", ps(vec![pdef("kraus", ParamType::ExprId, true, "Stored rank-3 Kraus-list expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id.")]), handle_apply_channel_qm),
        centry("lindblad_rhs", "Construct the finite-dimensional Lindblad right-hand side.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id.")]), handle_lindblad_rhs_qm),
        centry("lindblad_euler_step", "Take one explicit Euler step for finite-dimensional Lindblad evolution.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id."), pdef("dt", ParamType::ExprId, true, "Stored scalar timestep expression id.")]), handle_lindblad_euler_step_qm),
        centry("lindblad_rk4_step", "Take one classical RK4 step for finite-dimensional Lindblad evolution.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("rho", ParamType::ExprId, true, "Stored density-matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id."), pdef("dt", ParamType::ExprId, true, "Stored scalar timestep expression id.")]), handle_lindblad_rk4_step_qm),
        centry("lindblad_steady_state", "Solve for a finite-dimensional Lindblad steady state.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id.")]), handle_lindblad_steady_state_qm),
        centry("lindbladian_superoperator", "Construct the exact Lindbladian superoperator.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id.")]), handle_lindbladian_superoperator_qm),
        centry("lindbladian_eigenvalues", "Return supported low-dimensional Lindbladian eigenvalues.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id.")]), handle_lindbladian_eigenvalues_qm),
        centry("sparse_steady_state", "Plugin-backed sparse Lindblad steady state for numeric operators.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id."), pdef("tolerance", ParamType::Float, true, "Solver tolerance."), pdef("max_iterations", ParamType::Integer, true, "Maximum solver iterations.")]), handle_sparse_steady_state_qm),
        centry("sparse_lindbladian_spectrum", "Plugin-backed sparse Lindbladian spectrum for numeric operators.", ps(vec![pdef("H", ParamType::ExprId, true, "Stored Hamiltonian matrix expression id."), pdef("jumps", ParamType::ExprId, true, "Stored rank-3 jump-operator list expression id."), pdef("k", ParamType::Integer, true, "Number of requested eigenvalues."), pdef("which", ParamType::StringEnum(&["LR", "SR", "LM", "SM"]), true, "Eigenvalue selection rule."), pdef("tolerance", ParamType::Float, true, "Solver tolerance."), pdef("max_iterations", ParamType::Integer, true, "Maximum solver iterations.")]), handle_sparse_lindbladian_spectrum_qm),
        centry("normal_order", "Normal-order creation and annihilation operators.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_normal_order_qm),
        centry("time_order", "Wrap an expression in the symbolic time-ordering operator.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_time_order_qm),
        centry("time_ordered", "Wrap an expression in the symbolic time-ordering operator.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_time_order_qm),
        centry("anti_time_order", "Wrap an expression in the symbolic anti-time-ordering operator.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_anti_time_order_qm),
        centry("anti_time_ordered", "Wrap an expression in the symbolic anti-time-ordering operator.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_anti_time_order_qm),
        centry("displacement_series", "Construct a truncated symbolic displacement-operator series.", ps(vec![pdef("alpha", ParamType::ExprId, true, "Stored displacement amplitude expression id."), pdef("mode", ParamType::ExprId, true, "Stored mode expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_displacement_series_qm),
        centry("displacement_operator_series", "Construct a truncated symbolic displacement-operator series.", ps(vec![pdef("alpha", ParamType::ExprId, true, "Stored displacement amplitude expression id."), pdef("mode", ParamType::ExprId, true, "Stored mode expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_displacement_series_qm),
        centry("squeezing_series", "Construct a truncated symbolic squeezing-operator series.", ps(vec![pdef("zeta", ParamType::ExprId, true, "Stored squeezing parameter expression id."), pdef("mode", ParamType::ExprId, true, "Stored mode expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_squeezing_series_qm),
        centry("squeezing_operator_series", "Construct a truncated symbolic squeezing-operator series.", ps(vec![pdef("zeta", ParamType::ExprId, true, "Stored squeezing parameter expression id."), pdef("mode", ParamType::ExprId, true, "Stored mode expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_squeezing_series_qm),
        centry("bch", "Construct a finite-order symbolic BCH expansion.", ps(vec![pdef("A", ParamType::ExprId, true, "Stored left operator expression id."), pdef("B", ParamType::ExprId, true, "Stored right operator expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_bch_qm),
        centry("bch_expand", "Construct a finite-order symbolic BCH expansion.", ps(vec![pdef("A", ParamType::ExprId, true, "Stored left operator expression id."), pdef("B", ParamType::ExprId, true, "Stored right operator expression id."), pdef("order", ParamType::Integer, true, "Nonnegative truncation order.")]), handle_bch_qm),
        centry("simplify_ccr_car", "Apply explicit CCR/CAR rewrites to ladder-operator products.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_simplify_ccr_car_qm),
        centry("wick_expand", "Apply Wick expansion.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_wick_expand_qm),
        centry("wick", "Apply Wick expansion.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_wick_expand_qm),
        centry("grassmann_simplify", "Simplify with Grassmann grading rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_grassmann_simplify_qm),
        centry("grassmann", "Simplify with Grassmann grading rules.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_grassmann_simplify_qm),
        centry("wedge", "Wedge product of differential forms.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored form expression id."), pdef("rhs", ParamType::ExprId, true, "Stored form expression id.")]), handle_wedge_forms),
        centry("wedge_1_1", "Wedge product of differential forms.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored form expression id."), pdef("rhs", ParamType::ExprId, true, "Stored form expression id.")]), handle_wedge_forms),
        centry("exterior_derivative", "Exterior derivative of a differential form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored form expression id.")]), handle_exterior_derivative_forms),
        centry("exterior_d", "Exterior derivative of a differential form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored form expression id.")]), handle_exterior_derivative_forms),
        centry("d", "Exterior derivative of a differential form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored form expression id.")]), handle_exterior_derivative_forms),
        centry("hodge_dual", "Hodge dual with respect to a metric.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored form expression id."), pdef("rhs", ParamType::ExprId, true, "Stored metric expression id.")]), handle_hodge_dual_forms),
        centry("hodge_star", "Hodge dual with respect to a metric.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored form expression id."), pdef("rhs", ParamType::ExprId, true, "Stored metric expression id.")]), handle_hodge_dual_forms),
        centry("codifferential", "Codifferential of a differential form.", ps(vec![pdef("form", ParamType::ExprId, true, "Stored form expression id."), pdef("metric", ParamType::ExprId, true, "Stored metric expression id."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_codifferential_forms),
        centry("interior_product", "Interior product of a vector with a form.", ps(vec![pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("form", ParamType::ExprId, true, "Stored form expression id.")]), handle_interior_product_forms),
        centry("lie_derivative_form", "Lie derivative of a differential form.", ps(vec![pdef("form", ParamType::ExprId, true, "Stored form expression id."), pdef("vector", ParamType::ExprId, true, "Stored vector expression id."), pdef("coords", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_lie_derivative_form_forms),
        centry("functional_derivative", "Functional derivative with respect to a field.", ps(vec![pdef("lagrangian", ParamType::ExprId, true, "Stored Lagrangian id."), pdef("field", ParamType::Symbol, true, "Field symbol."), pdef("field_derivatives", ParamType::SymbolList, true, "Field derivative symbols."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_functional_derivative_variational),
        centry("euler_lagrange", "Functional derivative with respect to a field.", ps(vec![pdef("lagrangian", ParamType::ExprId, true, "Stored Lagrangian id."), pdef("field", ParamType::Symbol, true, "Field symbol."), pdef("field_derivatives", ParamType::SymbolList, true, "Field derivative symbols."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_functional_derivative_variational),
        centry("euler_lagrange_system", "Euler-Lagrange equations for several fields.", ps(vec![pdef("lagrangian", ParamType::ExprId, true, "Stored Lagrangian id."), pdef("fields", ParamType::Code, true, "JSON array of [field, derivs] entries."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_euler_lagrange_system_variational),
        centry("vary_action", "Formal variation of an action density.", ps(vec![pdef("lagrangian", ParamType::ExprId, true, "Stored Lagrangian id."), pdef("field", ParamType::Symbol, true, "Field symbol."), pdef("variation", ParamType::Symbol, true, "Variation symbol."), pdef("field_derivatives", ParamType::SymbolList, true, "Field derivative symbols."), pdef("variation_derivatives", ParamType::SymbolList, true, "Variation derivative symbols.")]), handle_vary_action_variational),
        centry("vary", "Formal variation of an action density.", ps(vec![pdef("lagrangian", ParamType::ExprId, true, "Stored Lagrangian id."), pdef("field", ParamType::Symbol, true, "Field symbol."), pdef("variation", ParamType::Symbol, true, "Variation symbol."), pdef("field_derivatives", ParamType::SymbolList, true, "Field derivative symbols."), pdef("variation_derivatives", ParamType::SymbolList, true, "Variation derivative symbols.")]), handle_vary_action_variational),
        centry("solve", "Solve a univariate equation.", ps(vec![pdef("equation", ParamType::Code, true, "Equation expression."), pdef("variable", ParamType::Symbol, true, "Unknown symbol.")]), handle_solve_general),
        centry("solve_linear_system", "Solve a linear system.", ps(vec![pdef("equations", ParamType::Code, true, "JSON array of equation strings."), pdef("variables", ParamType::SymbolList, true, "Unknown symbols.")]), handle_solve_linear_system_general),
        centry("solve_ode", "Solve a supported first-order ODE.", ps(vec![pdef("equation", ParamType::ExprId, true, "Stored ODE rhs/expression id."), pdef("dependent", ParamType::Symbol, true, "Dependent variable."), pdef("independent", ParamType::Symbol, true, "Independent variable.")]), handle_solve_ode_ode),
        centry("dsolve", "Solve a supported first-order ODE.", ps(vec![pdef("equation", ParamType::ExprId, true, "Stored ODE rhs/expression id."), pdef("dependent", ParamType::Symbol, true, "Dependent variable."), pdef("independent", ParamType::Symbol, true, "Independent variable.")]), handle_solve_ode_ode),
        centry("rk4", "Numerical fourth-order Runge-Kutta integration.", ps(vec![pdef("f", ParamType::ExprId, true, "Stored RHS expression id."), pdef("x", ParamType::Symbol, true, "Independent variable."), pdef("y", ParamType::Symbol, true, "Dependent variable."), pdef("x0", ParamType::Float, true, "Initial x."), pdef("y0", ParamType::Float, true, "Initial y."), pdef("x_end", ParamType::Float, true, "Final x."), pdef("steps", ParamType::Integer, false, "Optional step count.")]), handle_rk4_ode),
        centry("rk4_system", "Numerical fourth-order Runge-Kutta for coupled systems.", ps(vec![pdef("functions", ParamType::ExprId, true, "Stored list of RHS expressions."), pdef("independent", ParamType::Symbol, true, "Independent variable."), pdef("dependents", ParamType::SymbolList, true, "Dependent variables."), pdef("x0", ParamType::Float, true, "Initial x."), pdef("y0s", ParamType::Code, true, "JSON array of initial values."), pdef("x_end", ParamType::Float, true, "Final x."), pdef("steps", ParamType::Integer, false, "Optional step count.")]), handle_rk4_system_ode),
        centry("parallel_transport", "Library-level numerical parallel transport using a native Christoffel callback; ordinary source syntax does not currently expose that callback mechanism.", ps(vec![
            pdef("initial_vector", ParamType::Code, false, "Native callback mode only: numeric initial contravariant vector."),
            pdef("curve", ParamType::Code, false, "Native callback mode only: numeric curve samples."),
            pdef("gamma_numeric", ParamType::Code, false, "Native callback mode only: Christoffel callback handle."),
        ]), handle_parallel_transport_native_only),
        centry("integrate_geodesic", "Library-level numerical geodesic integration using a native Christoffel callback; ordinary source syntax does not currently expose that callback mechanism.", ps(vec![
            pdef("gamma_numeric", ParamType::Code, false, "Native callback mode only: Christoffel callback handle."),
            pdef("initial_position", ParamType::Code, false, "Native callback mode only: numeric initial position."),
            pdef("initial_velocity", ParamType::Code, false, "Native callback mode only: numeric initial velocity."),
            pdef("tau_range", ParamType::Code, false, "Native callback mode only: numeric tau interval."),
            pdef("steps", ParamType::Integer, false, "Native callback mode only: integration step count."),
        ]), handle_integrate_geodesic_native_only),
        centry("first_order_form", "Convert a higher-order ODE into first-order form.", ps(vec![pdef("ode", ParamType::ExprId, true, "Stored ODE expression id."), pdef("dependent", ParamType::Symbol, true, "Dependent variable."), pdef("independent", ParamType::Symbol, true, "Independent variable.")]), handle_first_order_form_ode),
        centry("classify_pde", "Classify a second-order PDE by its discriminant.", ps(vec![pdef("A", ParamType::ExprId, true, "Coefficient A."), pdef("B", ParamType::ExprId, true, "Coefficient B."), pdef("C", ParamType::ExprId, true, "Coefficient C.")]), handle_classify_pde_ode),
        centry("separate_variables", "Return a separated-variables ansatz.", ps(vec![pdef("pde_type", ParamType::Code, true, "PDE family name."), pdef("spatial", ParamType::Symbol, true, "Spatial variable."), pdef("temporal", ParamType::Symbol, true, "Temporal variable."), pdef("coefficient", ParamType::Code, false, "Optional PDE coefficient.")]), handle_separate_variables_ode),
        centry("separation", "Return a separated-variables ansatz.", ps(vec![pdef("pde_type", ParamType::Code, true, "PDE family name."), pdef("spatial", ParamType::Symbol, true, "Spatial variable."), pdef("temporal", ParamType::Symbol, true, "Temporal variable."), pdef("coefficient", ParamType::Code, false, "Optional PDE coefficient.")]), handle_separate_variables_ode),
        centry("determinant", "Determinant of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_determinant_linalg),
        centry("det", "Determinant of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_determinant_linalg),
        centry("inverse", "Inverse of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_inverse_linalg),
        centry("inv", "Inverse of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_inverse_linalg),
        centry("trace", "Trace of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_trace_linalg),
        centry("trace_mat", "Trace of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_trace_linalg),
        centry("eigenvalues_symbolic", "Characteristic polynomial of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_eigenvalues_symbolic_linalg),
        centry("eigenvalues", "Characteristic polynomial of a symbolic matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix expression id.")]), handle_eigenvalues_symbolic_linalg),
        centry("tensor_product", "Kronecker product of two matrices.", ps(vec![pdef("a", ParamType::ExprId, true, "Stored matrix id."), pdef("b", ParamType::ExprId, true, "Stored matrix id.")]), handle_tensor_product_linalg),
        centry("matmul", "Matrix multiplication.", ps(vec![pdef("a", ParamType::ExprId, true, "Stored matrix id."), pdef("b", ParamType::ExprId, true, "Stored matrix id.")]), handle_matmul_linalg),
        centry("transpose", "Transpose of a matrix.", ps(vec![pdef("matrix", ParamType::ExprId, true, "Stored matrix id.")]), handle_transpose_linalg),
        centry("identity_matrix", "Identity matrix of size n.", ps(vec![pdef("n", ParamType::Integer, true, "Matrix dimension.")]), handle_identity_matrix_linalg),
        centry("identity", "Identity matrix of size n.", ps(vec![pdef("n", ParamType::Integer, true, "Matrix dimension.")]), handle_identity_matrix_linalg),
        centry("metric", "Define and store a symbolic metric with coordinates, or declare a metric property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Metric declaration code.")]), handle_eval_syntax_entry),
        centry("diag", "Construct a diagonal matrix via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Diagonal matrix expression code.")]), handle_eval_syntax_entry),
        centry("dim", "Declare an index-family dimension via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Dimension declaration code.")]), handle_eval_syntax_entry),
        centry("convert", "Convert units via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Unit-conversion expression code.")]), handle_eval_syntax_entry),
        centry("check_units", "Check dimensional consistency via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Unit-check expression code.")]), handle_eval_syntax_entry),
        centry("plot", "Plot an expression via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Plot expression code.")]), handle_eval_syntax_entry),
        centry("symmetric", "Declare a symmetric tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("antisymmetric", "Declare an antisymmetric tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("inverse_metric", "Declare an inverse-metric property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("kronecker_delta", "Declare a Kronecker-delta property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("kronecker", "Declare a Kronecker-delta property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("epsilon", "Declare an epsilon-tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("epsilon_tensor", "Declare an epsilon-tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("riemann_symmetry", "Declare Riemann tensor symmetry via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("traceless", "Declare a traceless tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("derivative", "Declare a derivative operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("partial_derivative", "Declare a partial-derivative operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("covariant_derivative", "Declare a covariant-derivative operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("spinor", "Declare a spinor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("dirac_bar", "Declare a Dirac-bar property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("diracbar", "Declare a Dirac-bar property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("gamma_matrix", "Declare a gamma-matrix property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("gamma", "Construct or declare gamma-matrix expressions via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Gamma expression code.")]), handle_eval_syntax_entry),
        centry("commuting", "Declare a commuting operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("anticommuting", "Declare an anticommuting operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("anti_commuting", "Declare an anticommuting operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("noncommuting", "Declare a noncommuting operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("non_commuting", "Declare a noncommuting operator property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("bianchi", "Declare a Bianchi-identity property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("satisfies_bianchi", "Declare a Bianchi-identity property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("weyl", "Declare a Weyl tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("weyl_tensor", "Declare a Weyl tensor property via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("tableau_symmetry", "Declare Young-tableau symmetry via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Property declaration code.")]), handle_eval_syntax_entry),
        centry("ket", "Construct a ket via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Ket expression code.")]), handle_eval_syntax_entry),
        centry("bra", "Construct a bra via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Bra expression code.")]), handle_eval_syntax_entry),
        centry("creation", "Declare a creation operator via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Operator declaration code.")]), handle_eval_syntax_entry),
        centry("annihilation", "Declare an annihilation operator via source syntax.", ps(vec![pdef("code", ParamType::Code, false, "Operator declaration code.")]), handle_eval_syntax_entry),
        centry("declare_property", "Declare a tensor property on a symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol."), pdef("property", ParamType::Code, true, "Property string.")]), handle_declare_property),
        centry("declare_spinor_meta", "Attach structured spinor metadata and compatible legacy markers.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("dim", ParamType::Optional(Box::new(ParamType::Integer)), false, "Optional spinor dimension."),
            pdef("class", ParamType::StringEnum(&["dirac", "majorana", "weyl", "majorana_weyl"]), true, "Spinor class."),
            pdef("chirality", ParamType::Optional(Box::new(ParamType::StringEnum(&["left", "right"]))), false, "Optional chirality."),
            pdef("family", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional spinor index family."),
        ]), handle_declare_spinor_meta),
        centry("declare_gamma_matrix_meta", "Attach structured gamma-matrix metadata and the legacy gamma marker.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("dim", ParamType::Optional(Box::new(ParamType::Integer)), false, "Optional Clifford dimension."),
            pdef("metric", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional metric symbol."),
            pdef("family", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional spinor index family."),
            pdef("has_gamma5", ParamType::Bool, true, "Whether the family includes gamma5."),
        ]), handle_declare_gamma_matrix_meta),
        centry("declare_gamma_convention", "Attach structured gamma-matrix convention metadata.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("signature", ParamType::StringEnum(&["mostly_plus", "mostly_minus", "euclidean"]), true, "Metric signature convention."),
            pdef("clifford", ParamType::StringEnum(&["plus_two_g", "minus_two_g"]), true, "Clifford relation sign convention."),
            pdef("dimension", ParamType::Integer, true, "Positive Clifford dimension."),
        ]), handle_declare_gamma_convention),
        centry("declare_gamma5_convention", "Attach structured gamma5 convention metadata.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("signature", ParamType::StringEnum(&["mostly_plus", "mostly_minus", "euclidean"]), true, "Metric signature convention."),
            pdef("clifford", ParamType::StringEnum(&["plus_two_g", "minus_two_g"]), true, "Clifford relation sign convention."),
            pdef("gamma5_kind", ParamType::StringEnum(&["levi_civita", "abstract_chiral"]), true, "Gamma5 convention kind."),
            pdef("epsilon_symbol", ParamType::Symbol, true, "Levi-Civita epsilon symbol associated with gamma5."),
            pdef("dimension", ParamType::Integer, true, "Positive Clifford dimension."),
        ]), handle_declare_gamma5_convention),
        centry("declare_dirac_bar_meta", "Attach structured Dirac-bar metadata and the legacy DiracBar marker.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("gamma_symbol", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional gamma symbol family."),
            pdef("family", ParamType::Optional(Box::new(ParamType::Symbol)), false, "Optional spinor family."),
            pdef("reverse_gamma_order", ParamType::Bool, true, "Whether bar expansion reverses gamma order."),
        ]), handle_declare_dirac_bar_meta),
        centry("declare_mode", "Attach structured mode metadata and compatible legacy commutation markers.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target mode symbol."),
            pdef("statistics", ParamType::StringEnum(&["bosonic", "fermionic", "spin"]), true, "Mode statistics."),
            pdef("mode_index", ParamType::Integer, true, "Zero-based canonical mode position."),
        ]), handle_declare_mode),
        centry("declare_mode_in_subsystem", "Attach structured mode metadata within a named subsystem and compatible legacy markers.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target mode symbol."),
            pdef("statistics", ParamType::StringEnum(&["bosonic", "fermionic", "spin"]), true, "Mode statistics."),
            pdef("subsystem", ParamType::Symbol, true, "Subsystem or register symbol."),
            pdef("mode_index", ParamType::Integer, true, "Zero-based canonical mode position inside the subsystem."),
        ]), handle_declare_mode_in_subsystem),
        centry("declare_mode_with_label", "Attach structured mode metadata with subsystem and label aliases plus compatible legacy markers.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target mode symbol."),
            pdef("statistics", ParamType::StringEnum(&["bosonic", "fermionic", "spin"]), true, "Mode statistics."),
            pdef("subsystem", ParamType::Symbol, true, "Subsystem or register symbol."),
            pdef("mode_index", ParamType::Integer, true, "Zero-based canonical mode position inside the subsystem."),
            pdef("label", ParamType::Symbol, true, "Optional symbolic mode alias."),
        ]), handle_declare_mode_with_label),
        centry("declare_bosonic_truncated_mode", "Attach bosonic ModeMeta and store a finite truncation for later Fock-space declarations.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target mode symbol."),
            pdef("mode_index", ParamType::Integer, true, "Zero-based canonical mode position."),
            pdef("nmax", ParamType::Integer, true, "Positive maximum bosonic occupation."),
        ]), handle_declare_bosonic_truncated_mode),
        centry("declare_fermionic_mode", "Attach fermionic ModeMeta for a Fock-space mode.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target mode symbol."),
            pdef("mode_index", ParamType::Integer, true, "Zero-based canonical mode position."),
        ]), handle_declare_fermionic_mode),
        centry("declare_fock_space", "Attach structured Fock-space metadata from previously declared modes.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target Fock-space symbol."),
            pdef("mode_symbols", ParamType::SymbolList, true, "Non-empty ordered list of previously declared mode symbols."),
        ]), handle_declare_fock_space),
        centry("declare_trace_space", "Attach structured trace-space metadata.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target trace-like symbol."),
            pdef("space_symbol", ParamType::Symbol, true, "Trace space label."),
            pdef("cyclic", ParamType::Bool, true, "Whether traces in this space are cyclic."),
        ]), handle_declare_trace_space),
        centry("declare_hilbert_space", "Attach structured finite-dimensional Hilbert-space metadata.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target Hilbert-space symbol."),
            pdef("dim", ParamType::Integer, true, "Positive total dimension."),
        ]), handle_declare_hilbert_space),
        centry("declare_composite_space", "Declare a composite Hilbert space from previously declared factors.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target composite-space symbol."),
            pdef("factors", ParamType::SymbolList, true, "Ordered factor-space symbols."),
        ]), handle_declare_composite_space),
        centry("declare_quantum_object", "Attach structured quantum-object metadata and compatible legacy operator markers.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target quantum-object symbol."),
            pdef("kind", ParamType::StringEnum(&["ket", "bra", "operator", "density_operator", "projector", "observable", "channel"]), true, "Quantum-object kind."),
            pdef("space_symbol", ParamType::Symbol, true, "Previously declared Hilbert-space symbol."),
        ]), handle_declare_quantum_object),
        centry("declare_operator_space", "Attach structured operator domain/codomain metadata from previously declared Hilbert spaces.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target operator symbol."),
            pdef("domain_space", ParamType::Symbol, true, "Previously declared domain Hilbert-space symbol."),
            pdef("codomain_space", ParamType::Symbol, true, "Previously declared codomain Hilbert-space symbol."),
        ]), handle_declare_operator_space),
        centry("riemann_tensor", "Declare a symbol as an abstract Riemann tensor.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target tensor symbol.")]), handle_riemann_tensor_declaration),
        centry("declare_indices", "Declare an index family.", ps(vec![pdef("family", ParamType::Symbol, true, "Family name."), pdef("indices", ParamType::SymbolList, true, "Index symbols."), pdef("dimension", ParamType::Integer, false, "Optional family dimension.")]), handle_declare_indices),
        centry("declare_coordinates", "Declare active coordinate symbols.", ps(vec![pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_declare_coordinates),
        centry("declare_assumption", "Declare an assumption on a symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol."), pdef("assumption", ParamType::Code, true, "Assumption name.")]), handle_declare_assumption),
        centry("declare_grassmann", "Declare a Grassmann-odd symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol.")]), handle_declare_grassmann),
        centry("declare_operator", "Declare a creation or annihilation operator, optionally with bosonic or fermionic statistics.", ps(vec![
            pdef("symbol", ParamType::Symbol, true, "Target symbol."),
            pdef("kind", ParamType::StringEnum(&["creation", "annihilation"]), true, "Operator kind."),
            pdef("statistics", ParamType::Optional(Box::new(ParamType::StringEnum(&["bosonic", "fermionic"]))), false, "Optional operator statistics; defaults to bosonic."),
        ]), handle_declare_operator),
        centry("compose_operators", "Build a symbolic operator composition and validate compatible codomain/domain metadata when available.", ps(vec![
            pdef("left", ParamType::ExprId, true, "Stored outer operator expression id."),
            pdef("right", ParamType::ExprId, true, "Stored inner operator expression id."),
        ]), handle_compose_operators_qm),
        centry("declare_contraction", "Declare a Wick contraction value for an ordered operator-mode pair.", ps(vec![
            pdef("lhs", ParamType::Symbol, true, "Left operator mode symbol."),
            pdef("rhs", ParamType::Symbol, true, "Right operator mode symbol."),
            pdef("value", ParamType::Code, true, "Contraction value."),
        ]), handle_declare_contraction),
        centry("set_convention", "Set one active convention field.", ps(vec![pdef("field", ParamType::Code, true, "Convention field name."), pdef("value", ParamType::Code, true, "Convention option name.")]), handle_set_convention),
        centry("define_rule", "Define a rewrite rule.", ps(vec![pdef("name", ParamType::Code, true, "Rule name."), pdef("lhs", ParamType::Code, true, "Left-hand-side code."), pdef("rhs", ParamType::Code, true, "Right-hand-side code.")]), handle_define_rule),
        centry("define_metric", "Define and store a symbolic metric with coordinates.", ps(vec![pdef("name", ParamType::Code, true, "Metric identifier."), pdef("components", ParamType::Matrix, true, "2D array of code strings."), pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_define_metric),
        centry("list_expressions", "List stored expression ids with LaTeX and Unicode renderings.", ps(vec![]), handle_list_expressions_state),
        centry("list_metrics", "List stored metric ids with coordinates and dimension.", ps(vec![]), handle_list_metrics_state),
        centry("list_properties", "List declared tensor properties grouped by symbol.", ps(vec![]), handle_list_properties_state),
        centry("list_index_families", "List declared index families and their values.", ps(vec![]), handle_list_index_families_state),
        centry("get_state_summary", "Return a combined summary of stored expressions, metrics, properties, and index families.", ps(vec![]), handle_get_state_summary_state),
        centry("to_python", "Generate Python code.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_to_python_codegen),
        centry("to_rust", "Generate Rust code.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_to_rust_codegen),
        centry("to_cpp", "Generate C++ code.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_to_cpp_codegen),
        centry("equiv", "Test semantic equivalence.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored expression id."), pdef("rhs", ParamType::ExprId, true, "Stored expression id.")]), handle_equiv_analysis),
        centry("semantic_diff", "Summarize semantic differences.", ps(vec![pdef("lhs", ParamType::ExprId, true, "Stored expression id."), pdef("rhs", ParamType::ExprId, true, "Stored expression id.")]), handle_semantic_diff_analysis),
        centry("diff", "Compare two expressions structurally.", ps(vec![pdef("expr_a", ParamType::ExprId, true, "First stored expression id."), pdef("expr_b", ParamType::ExprId, true, "Second stored expression id.")]), handle_diff_diagnostics),
        centry("check_properties", "Check whether an expression has the properties and index declarations an algorithm expects.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("algorithm", ParamType::StringEnum(DIAGNOSTIC_ALGORITHMS), true, "Algorithm to diagnose.")]), handle_check_properties_diagnostics),
        centry("explain", "Explain what an algorithm does and why it might not change an expression.", ps(vec![pdef("algorithm", ParamType::StringEnum(DIAGNOSTIC_ALGORITHMS), true, "Algorithm to explain."), pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_explain_diagnostics),
        centry("workflow", "Look up a recommended MCP tool sequence for a physics or algebra task.", ps(vec![pdef("goal", ParamType::Code, true, "Workflow name or natural-language goal.")]), handle_workflow_lookup),
        centry("list_workflows", "List the available workflow templates.", ps(vec![]), handle_list_workflows),
        centry("inspect", "Inspect an expression structurally.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_inspect_analysis),
        centry("suggest", "Suggest next algorithms for an expression, optionally prioritised for a goal.", ps(vec![
            pdef("expr", ParamType::ExprId, true, "Stored expression id."),
            pdef("goal", ParamType::Optional(Box::new(ParamType::Code)), false, "Optional goal description (e.g. 'simplify', 'prove vanishes', 'evaluate components')."),
        ]), handle_suggest_analysis),
    ]
}
