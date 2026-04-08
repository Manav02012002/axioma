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
        b("mukhanov_sasaki", "cosmology", "mukhanov_sasaki()", "Return the Mukhanov-Sasaki equation in conformal time.", "mukhanov_sasaki()"),
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
            "reduce_delta",
            "tensor",
            "reduce_delta(expr)",
            "Simplify explicit delta-expanded expressions back to compact form.",
            "reduce_delta(expr)",
        ),
        b(
            "young_project",
            "tensor",
            "young_project(expr)",
            "Project a tensor onto a declared Young-tableau symmetry.",
            "young_project(T[a-,b-,c-])",
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
            "creation",
            "quantum",
            "creation(sym)",
            "Construct an abstract harmonic-oscillator creation operator. As a top-level statement it also declares the symbol as a creation operator for normal-ordering metadata.",
            "creation(a)",
        ),
        b(
            "annihilation",
            "quantum",
            "annihilation(sym)",
            "Construct an abstract harmonic-oscillator annihilation operator. As a top-level statement it also declares the symbol as an annihilation operator for normal-ordering metadata.",
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
            "Reorder ladder operators into normal order.",
            "normal_order(annihilation(a) * creation(a))",
        ),
        b(
            "wick",
            "quantum",
            "wick(expr)",
            "Expand products using Wick contraction rules.",
            "wick(psi*psibar)",
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
            "euler_lagrange(L, field, coords)",
            "Compute Euler-Lagrange equations.",
            "euler_lagrange(L, phi, [t, x])",
        ),
        b(
            "vary",
            "variational",
            "vary(expr, field)",
            "Take a formal variation with respect to a field.",
            "vary(S, phi)",
        ),
        b(
            "dsolve",
            "ode",
            "dsolve(eq, y, x)",
            "Solve a supported first-order ODE symbolically.",
            "dsolve(y, y, x)",
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
            "wedge_1_1",
            "forms",
            "wedge_1_1(a, b)",
            "Wedge product of two 1-forms.",
            "wedge_1_1(A, B)",
        ),
        b(
            "exterior_d",
            "forms",
            "exterior_d(form)",
            "Exterior derivative of a differential form.",
            "exterior_d(A)",
        ),
        b(
            "d",
            "forms",
            "d(form)",
            "Alias for exterior_d in the forms subsystem.",
            "d(A)",
        ),
        b(
            "hodge_star",
            "forms",
            "hodge_star(form, metric)",
            "Hodge dual of a differential form.",
            "hodge_star(F, g)",
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
        p("Symmetric", "property T symmetric([positions])", "Indices are symmetric under exchange of the listed slots.", "build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup", "property g symmetric"),
        p("AntiSymmetric", "property T antisymmetric([positions])", "Indices are antisymmetric under exchange of the listed slots.", "build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup", "property F antisymmetric"),
        p("RiemannSymmetry", "property R riemann_symmetry", "Apply the standard pair antisymmetry and pair-exchange symmetry of a Riemann tensor.", "build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup", "property R riemann_symmetry"),
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
        p("Depends", "depends T [x, y, ...]", "Declares that a tensor depends on listed symbols.", "stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it", "depends phi [x, t]"),
        p("Spinor", "property psi spinor", "Marks a tensor as carrying spinor indices.", "canonicalise_product dummy classification via metric_symmetry_for_slots", "property psi spinor"),
        p("DiracBar", "property psibar dirac_bar", "Marks a symbol as a Dirac-bar object.", "canonicalize_indices local argument canonicalisation, sort_product barred-bilinear normalization/barrier handling, and ax-qm DiracBar expansion/sorting", "property psibar dirac_bar"),
        p("GammaMatrixProp", "property gamma gamma_matrix", "Marks a symbol as a gamma matrix.", "canonicalize_indices antisymmetric gamma-call slots, sort_product local barred-bilinear gamma placement/barrier handling, and ax-qm gamma algorithms", "property gamma gamma_matrix"),
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
        p("TableauSymmetry", "property T tableau_symmetry([shape], [indices])", "Declares a Young-tableau symmetry shape and slot assignment.", "canonicalise slot/sign handling, meld tableaux_from_properties, young_project_tensor", "property T tableau_symmetry([2,1], [0,1,2])"),
        p("SatisfiesBianchi", "property R satisfies_bianchi", "Marks a tensor as satisfying a Bianchi identity.", "meld tableaux_from_properties Bianchi cancellation hook", "property R satisfies_bianchi"),
        p("WeylTensor", "property C weyl_tensor", "Marks a tensor as a Weyl tensor.", "canonicalise Riemann-like slot symmetries, traceless fast-zero handling, and meld Bianchi-style tableau hooks", "property C weyl_tensor"),
        p("DifferentialFormDegree", "property F differential_form_degree(n)", "Declares the degree of a differential form.", "stored by ax-tensor metadata; differential-form algorithms live outside ax-tensor", "property F differential_form_degree(2)"),
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
        a("young_project", "tensor", "young_project(expr: &Expr, tableau: &YoungTableau, interner: &Interner) -> Expr", "Project an expression with a specific Young tableau by antisymmetrizing columns and symmetrizing rows.", "A valid tableau cell layout must be supplied in slot-number form.", "young_project(T[a-,b-,c-])"),
        a("young_project_tensor", "tensor", "young_project_tensor(expr: &Expr, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Apply a declared TableauSymmetry property directly to a tensor expression.", "The relevant tensor symbol must carry a TableauSymmetry property in tensor_properties.", "young_project(T[a-,b-,c-])"),
        a("reduce_delta", "tensor", "reduce_delta(expr: &Expr, delta_sym: Spur, dim_sym: Spur, interner: &Interner) -> Expr", "Iteratively contract products and traces of Kronecker deltas back to simpler delta or dimension factors.", "The delta symbol and the symbol representing the dimension must be supplied.", "reduce_delta(Delta[a+,b-] * Delta[b+,c-])"),
        a("eliminate_kronecker", "tensor", "eliminate_kronecker(expr: &Expr, delta_sym: Spur, interner: &Interner) -> Expr", "Use Kronecker deltas to substitute contracted indices and remove delta factors from products.", "The delta symbol must identify a two-index Kronecker delta with one up and one down slot.", "eliminate_kronecker(delta[mu+,nu-] * T[nu+,rho-])"),
        a("eliminate_metric", "tensor", "eliminate_metric(expr: &Expr, metric_sym: Spur, inv_metric_sym: Spur, interner: &Interner) -> Expr", "Use metric or inverse-metric factors to raise or lower contracted indices and remove those metric factors.", "Metric components must use two down indices and inverse-metric components two up indices.", "eliminate_metric(g[mu-,nu-] * V[nu+])"),
        a("eliminate_vielbein", "tensor", "eliminate_vielbein(expr: &Expr, vielbein_sym: Spur, inv_vielbein_sym: Spur, interner: &Interner) -> Expr", "Use vielbein or inverse-vielbein factors to convert contracted indices between two families and remove the conversion factors.", "Vielbein factors must appear as indexed two-slot tensors with one contractible index matching another factor.", "eliminate_vielbein(e[a-,mu-] * V[mu+])"),
        a("christoffel_from_metric", "gr", "christoffel_from_metric(g: &SymbolicMatrix, coords: &[Spur], interner: &Interner) -> Vec<Vec<Vec<Expr>>>", "Compute Christoffel symbols from a symbolic metric by the standard Levi-Civita formula.", "The metric must be square and coords.len() must equal g.dim; the routine uses the symbolic inverse of g.", "christoffel(metric(diag(-1, 1)), [t, r])"),
        a("riemann_from_christoffel", "gr", "riemann_from_christoffel(gamma: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner, convention: &Convention) -> Vec<Vec<Vec<Vec<Expr>>>>", "Compute the Riemann tensor from Christoffel symbols, respecting the active sign convention.", "The connection array dimensions must match coords.len(); the Convention determines MTW versus Weinberg sign.", "riemann(Gamma, [t, r, theta, phi])"),
        a("ricci_from_riemann", "gr", "ricci_from_riemann(riemann: &[Vec<Vec<Vec<Expr>>>], n: usize, interner: &Interner, convention: &Convention) -> Vec<Vec<Expr>>", "Contract a Riemann tensor into the Ricci tensor using the configured Ricci-contraction convention.", "n must match the tensor dimensions; the Convention selects first-third or first-fourth contraction.", "ricci(R)"),
        a("ricci_scalar", "gr", "ricci_scalar(ricci: &[Vec<Expr>], ginv: &SymbolicMatrix, interner: &Interner) -> Expr", "Contract the Ricci tensor with the inverse metric to obtain the scalar curvature.", "The inverse metric dimension must match the Ricci tensor dimensions.", "ricci_scalar(ginv, Ric)"),
        a("einstein_tensor", "gr", "einstein_tensor(ricci: &[Vec<Expr>], scalar: &Expr, g: &SymbolicMatrix, interner: &Interner) -> Vec<Vec<Expr>>", "Build the Einstein tensor G_ab = R_ab - 1/2 g_ab R.", "The metric dimension must match the Ricci tensor dimensions.", "einstein(g, Ric, R)"),
        a("kretschner_scalar", "gr", "kretschner_scalar(riemann: &[Vec<Vec<Vec<Expr>>>], g: &SymbolicMatrix, interner: &Interner) -> Expr", "Compute a diagonal-metric approximation to the Kretschmann scalar from the squared Riemann components.", "The metric must be invertible; this implementation contracts using diagonal entries of g and g^{-1}.", "kretschner(g, R)"),
        a("covariant_derivative_vector", "gr", "covariant_derivative_vector(v: &[Expr], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute ∇_coord_index v for a contravariant vector field.", "The vector length, connection dimensions, and coordinate list length must agree.", "covariant_diff(V, g, [t, r])"),
        a("covariant_derivative_covector", "gr", "covariant_derivative_covector(w: &[Expr], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Compute ∇_coord_index w for a covector field.", "The covector length, connection dimensions, and coordinate list length must agree.", "covariant_diff(W, g, [t, r])"),
        a("geodesic_equations", "gr", "geodesic_equations(gamma: &[Vec<Vec<Expr>>], coords: &[Spur], interner: &Interner) -> Vec<Expr>", "Construct the geodesic equations ẍ^i = -Γ^i_jk ẋ^j ẋ^k in symbolic form.", "Connection dimensions must match the coordinate list.", "geodesic(g, [t, r, theta, phi], lambda)"),
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
        a("decompose_product", "tensor", "decompose_product(expr: &Expr, dim: usize, tensor_properties: &HashMap<Spur, Vec<TensorProperty>>, interner: &Interner) -> Expr", "Decompose indexed tensor products by associative Littlewood-Richardson tableau composition and Young projection.", "The input should be a product containing at least two indexed tensors with inferable shapes; TableauSymmetry, Symmetric, AntiSymmetric, RiemannSymmetry, and generic indexed slots drive shape inference, multiplicities are preserved, and unsupported or inconsistent shapes return a diagnostic expression.", "decompose_product(T[a-,b-] * S[c-,d-] * V[e-], 4)"),
        a("expand_implicit", "tensor", "expand_implicit(expr: &Expr, implicit_index_tensors: &HashSet<Spur>, available_indices: &[Spur], n_indices_per_tensor: &HashMap<Spur, usize>, properties: &dyn PropertyLookup, interner: &Interner) -> Expr", "Recursively make implicit tensor contraction graphs explicit across sums, products, trace wrappers, and call arguments.", "Tensor ranks are read from n_indices_per_tensor or tensor properties; each sum branch receives disjoint fresh graph indices.", "expand_implicit(A * B + C * D)"),
        a("normal_order", "qm", "normal_order(expr: &Expr, operators: &HashMap<Spur, OperatorKind>, interner: &Interner) -> Expr", "Reorder products of operators into normal order using the declared creation/annihilation kinds.", "Operator kinds must be declared for the symbols that should reorder.", "normal_order(a * creation(a))"),
        a("wick_expand", "qm", "wick_expand(expr: &Expr, operators: &HashMap<Spur, OperatorKind>, contractions: &HashMap<(Spur, Spur), Expr>, interner: &Interner) -> Expr", "Expand operator products into normal-ordered terms plus single contractions.", "Operator kinds and any nonzero contraction values must be provided explicitly.", "wick(psi * psibar)"),
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
        a("first_order_form", "ode", "first_order_form(ode: &Expr, dependent_var: Spur, independent_var: Spur, interner: &Interner) -> Vec<(Expr, Expr)>", "Convert a higher-order ODE into a first-order system by introducing auxiliary derivative variables.", "The ODE should contain nested diff calls with respect to independent_var, or else it is treated as the right-hand side of a second-order equation.", "first_order_form(diff(diff(x,t),t) + x, x, t)"),
        a("evaluate_components_v2", "tensor", "evaluate_components_v2(expr: &Expr, rules: &[ComponentRule], env: &dyn ComponentEvalEnv, interner: &Interner) -> Expr", "Evaluate tensor component algebra across sums, products, traces, derivatives, deltas, epsilon tensors, metrics, inverse metrics, and symmetry-aware sparse rules.", "Component rules, coordinates, and tensor properties must be available through env; dummy contractions are assigned before lookup, missing sparse components evaluate to zero, and generated inverse-metric components are collected with downstream terms.", "evaluate(g[mu-,nu-] * ginv[nu+,mu+], rules)"),
        a("rename_dummy_indices", "tensor", "rename_dummy_indices(expr: &Expr, prefix: &str, interner: &Interner) -> Expr", "Rename repeated contracted indices to fresh deterministic names with the chosen prefix.", "Useful when preparing expressions for display or comparison.", "rename_dummy_indices(T[a-,a+], d)"),
        a("diff_component", "tensor", "diff_component(expr: &Expr, var: Spur, interner: &Interner) -> Expr", "Differentiate a component expression with tensor-aware fallback handling.", "The variable should be a coordinate or scalar symbol.", "diff_component(r^2, r)"),
        a("covariant_derivative_tensor2", "gr", "covariant_derivative_tensor2(t: &[Vec<Expr>], gamma: &[Vec<Vec<Expr>>], coord_index: usize, coords: &[Spur], interner: &Interner) -> Vec<Vec<Expr>>", "Compute the covariant derivative of a rank-2 covariant tensor.", "Tensor dimensions, connection dimensions, and coordinate count must agree.", "covariant_diff(T, Gamma, 0, [t, r])"),
        a("compute_weight", "tensor", "compute_weight(expr: &Expr, weights: &HashMap<(Spur, String), i64>, label: &str) -> i64", "Compute the total symbolic weight of an expression under a chosen label.", "Weight assignments should be declared for the participating symbols.", "compute_weight(expr, weights, field)"),
        a("pauli_x", "qm", "pauli_x(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_x matrix.", "No extra setup is required.", "pauli_x()"),
        a("pauli_y", "qm", "pauli_y(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_y matrix.", "No extra setup is required.", "pauli_y()"),
        a("pauli_z", "qm", "pauli_z(interner: &Interner) -> Vec<Vec<Expr>>", "Return the Pauli sigma_z matrix.", "No extra setup is required.", "pauli_z()"),
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
        m("gr/perturbation", "Metric perturbation theory: expansion of inverse metric, Christoffel symbols, Riemann, Ricci, and Einstein tensors to arbitrary order in a perturbation parameter.", "perturb, perturb_inverse, perturb_christoffel, perturb_riemann, perturb_ricci, perturb_einstein"),
        m("gr/schwarzschild", "Builds the Schwarzschild metric, Christoffel symbols, Riemann tensor, and Ricci tensor.", "let g, let coords, let Gamma, let R, let Ric"),
        m("cosmology/perturbation", "Cosmological perturbation theory: SVT decomposition, Bardeen variables, linearized Einstein equations, Mukhanov-Sasaki equation, power spectrum, spectral index.", "linearized_einstein, mukhanov_sasaki, svt_decompose, bardeen, power_spectrum, spectral_index, tensor_scalar_ratio"),
        m("gr/black_hole_perturbation", "Black hole perturbation theory: Regge-Wheeler and Zerilli master equations for Schwarzschild perturbations.", "regge_wheeler, zerilli, regge_wheeler_decompose"),
        m("physics/classical_mechanics", "Notes the intended Euler-Lagrange workflow for classical mechanics.", "documentation comments only"),
        m("physics/klein_gordon", "Sets up a Klein-Gordon Lagrangian and computes its Euler-Lagrange equation.", "let dphi_dt, let dphi_dx, let dphi_dy, let dphi_dz, let L, let EOM"),
        m("physics/maxwell", "Placeholder notes for Maxwell theory setup from a Lagrangian.", "documentation comments only"),
        m("qft/dirac", "Summarizes the Dirac equation and basic gamma-trace identities.", "documentation comments only"),
        m("qft/gamma", "Introduces symbolic gamma and eta objects for gamma-matrix algebra experiments.", "let gamma, let eta"),
        m("qft/normal_ordering", "Documents normal ordering and Wick expansion usage.", "documentation comments only"),
        m("qft/scalar_field", "Summarizes the free scalar-field Lagrangian and Klein-Gordon equation.", "documentation comments only"),
        m("qft/spinor_helicity", "Spinor-helicity formalism: angle/square brackets, Mandelstam invariants, Parke-Taylor amplitudes, BCFW recursion, momentum twistors.", "angle, square, mandelstam, parke_taylor, bcfw_shift, bcfw_decomposition, four_bracket"),
        m("qft/superspace", "N=1 superspace: supercovariant derivatives, chiral/antichiral superfields, Wess-Zumino gauge vector superfields, D-algebra, superspace integration.", "setup_superspace, expand_superfield, chiral_superfield, d_alpha, d_squared, superspace_integrate"),
        m("qft/brst", "BRST cohomology: ghost number grading, BRST operator, nilpotency check, Yang-Mills BRST setup.", "setup_brst_ym, brst, ghost_number, brst_check"),
        m("qm/bell", "Constructs a Bell state, its density matrix, and a reduced density matrix by partial trace.", "let up, let down, let phi_plus, let rho, let rho_A"),
        m("qm/harmonic_oscillator", "Builds an abstract harmonic-oscillator annihilation operator, creation operator, number operator, Hamiltonian, and sample Fock-state actions.", "let a_op, let adag_op, let n_op, let h_op, let vac, let one, let two, let lowered_two, let number_on_two, let energy_on_one, let normal_reordered"),
        m("qm/spin", "Builds Pauli matrices and their commutator as a spin-1/2 algebra example.", "let sigma_x, let sigma_y, let sigma_z, let comm_xy"),
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
        TensorProperty::Depends(syms) => format!(
            "Depends({})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TensorProperty::Spinor => "Spinor".to_string(),
        TensorProperty::DiracBar => "DiracBar".to_string(),
        TensorProperty::GammaMatrixProp => "GammaMatrix".to_string(),
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
        TensorProperty::TableauSymmetry { shape, indices } => {
            format!("TableauSymmetry(shape: {:?}, indices: {:?})", shape, indices)
        }
        TensorProperty::SatisfiesBianchi => "SatisfiesBianchi".to_string(),
        TensorProperty::WeylTensor => "WeylTensor".to_string(),
        TensorProperty::DifferentialFormDegree(d) => {
            format!("DifferentialForm(degree: {})", d)
        }
    }
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

fn ensure_not_timeout(expr: ax_ir::Expr, interner: &ax_ir::Interner) -> Result<ax_ir::Expr, String> {
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
    if lower == "satisfiesbianchi" || lower == "satisfies_bianchi" || lower == "bianchi" {
        return Ok(ax_ir::TensorProperty::SatisfiesBianchi);
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
        return Ok(ax_ir::TensorProperty::TableauSymmetry { shape, indices });
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
    expr_response_with_change(&expr, crate::simplify::expand(&expr, state.interner()), "expand", state)
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
    let result = crate::simplify::apart_expr(&expr, var, state.interner()).unwrap_or_else(|| expr.clone());
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
    let changed = ax_ir::Expr::Call(state.interner_mut().get_or_intern(name), vec![lhs, rhs]) != result;
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
    let result = call_named("gradient", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
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
    let result = call_named("divergence", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
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
    let result = call_named("laplacian", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
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
    let result = call_named("jacobian", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
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
    let result = call_named("hessian", vec![expr.clone(), ax_ir::Expr::List(vars)], state);
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
    let result = call_named("differentiate_eq", vec![expr.clone(), ax_ir::Expr::Sym(var)], state);
    expr_response_with_change(&expr, result, "differentiate_eq", state)
}

fn handle_integrate_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let var = symbol_arg(args, 1, "var", state)?;
    let result = call_named("integrate_eq", vec![expr.clone(), ax_ir::Expr::Sym(var)], state);
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
    let result = call_named("raise_eq", vec![expr.clone(), ax_ir::Expr::Sym(index)], state);
    expr_response_with_change(&expr, result, "raise_eq", state)
}

fn handle_lower_eq_entry(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    let expr = expr_from_id(args, 0, "eq", state)?;
    let index = symbol_arg(args, 1, "index", state)?;
    let result = call_named("lower_eq", vec![expr.clone(), ax_ir::Expr::Sym(index)], state);
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
    expr_response_with_change(&expr, ensure_not_timeout(result, state.interner())?, "meld", state)
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
    unary_named_expr_response("young_project", args, state)
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
        .unwrap_or(4);
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
            &ax_tensor::ricci_scalar(&ricci, &metric.symbolic_inverse(state.interner()), state.interner()),
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
    let has_id_mode = args
        .get(0)
        .and_then(serde_json::Value::as_str)
        .is_some()
        && args.get(1).and_then(serde_json::Value::as_str).is_some()
        && args
            .get(2)
            .map(|value| value.is_null())
            .unwrap_or(true);
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
            &ax_tensor::ricci_scalar(&ricci, &metric.symbolic_inverse(state.interner()), state.interner()),
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
    let dim = metric.dim;
    if dim < 3 {
        return Err("weyl curvature is only defined for dimension >= 3".to_string());
    }
    let scalar = crate::eval(
        &ax_tensor::ricci_scalar(&ricci, &metric.symbolic_inverse(state.interner()), state.interner()),
        state.env(),
        state.interner(),
    );
    let denom1 = ax_ir::Expr::Rational(num_rational::BigRational::new(1.into(), (dim as i64 - 2).into()));
    let denom2 = ax_ir::Expr::Rational(num_rational::BigRational::new(1.into(), (((dim - 1) * (dim - 2)) as i64).into()));
    let mut out = vec![vec![vec![vec![ax_ir::Expr::zero(); dim]; dim]; dim]; dim];
    for a in 0..dim {
        for b in 0..dim {
            for c in 0..dim {
                for d in 0..dim {
                    let r = riem[a][b][c][d].clone();
                    let rab = &ricci;
                    let gac = metric.data[a][c].clone();
                    let gad = metric.data[a][d].clone();
                    let gbc = metric.data[b][c].clone();
                    let gbd = metric.data[b][d].clone();
                    let term1 = ax_ir::Expr::mul(vec![
                        denom1.clone(),
                        ax_ir::Expr::add(vec![
                            ax_ir::Expr::mul(vec![gac, rab[d][b].clone()]),
                            ax_ir::Expr::neg(ax_ir::Expr::mul(vec![gad, rab[c][b].clone()])),
                            ax_ir::Expr::neg(ax_ir::Expr::mul(vec![gbc, rab[d][a].clone()])),
                            ax_ir::Expr::mul(vec![gbd, rab[c][a].clone()]),
                        ]),
                    ]);
                    let term2 = ax_ir::Expr::mul(vec![
                        denom2.clone(),
                        scalar.clone(),
                        ax_ir::Expr::add(vec![
                            ax_ir::Expr::mul(vec![metric.data[a][c].clone(), metric.data[d][b].clone()]),
                            ax_ir::Expr::neg(ax_ir::Expr::mul(vec![metric.data[a][d].clone(), metric.data[c][b].clone()])),
                        ]),
                    ]);
                    out[a][b][c][d] = crate::eval(
                        &ax_ir::Expr::add(vec![r, ax_ir::Expr::neg(term1), term2]),
                        state.env(),
                        state.interner(),
                    );
                }
            }
        }
    }
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
fn handle_wick_expand_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("wick", args, state)
}
fn handle_grassmann_simplify_qm(
    args: &[serde_json::Value],
    state: &mut dyn EvalState,
) -> Result<serde_json::Value, String> {
    unary_named_expr_response("grassmann_simplify", args, state)
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
        "A" | "a" => 'A',
        "B" | "b" => 'B',
        _ => return Err("argument 'which' must be 'A' or 'B'".to_string()),
    };
    matrix_response(
        ax_qm::partial_trace(&rho, dim_a, dim_b, which, state.interner()),
        state,
    )
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
    expr_or_struct_response_named(ax_solve::solve(&equation, var, state.interner()), "solve", state)
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
    state
        .env_mut()
        .tensor_properties
        .entry(symbol)
        .or_default()
        .push(prop.clone());
    Ok(serde_json::json!({
        "status": "ok",
        "symbol": state.interner().resolve(symbol),
        "property": format!("{:?}", prop)
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
    Ok(serde_json::json!({ "status": "ok", "symbol": state.interner().resolve(symbol), "grading": "Odd" }))
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
    state.env_mut().operators.insert(symbol, kind);
    Ok(
        serde_json::json!({ "status": "ok", "symbol": state.interner().resolve(symbol), "operator": format!("{:?}", kind) }),
    )
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
    Ok(serde_json::json!({ "status": "ok", "code": ax_codegen::to_python(&expr, state.interner()) }))
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
        .map(|(symbol, properties)| serde_json::json!({
            "symbol": symbol,
            "properties": properties,
        }))
        .collect::<Vec<_>>();
    let index_families = state
        .list_index_families()
        .into_iter()
        .map(|(name, indices, dimension)| serde_json::json!({
            "name": name,
            "indices": indices,
            "dimension": dimension,
        }))
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
        centry("mukhanov_sasaki", "Return the Mukhanov-Sasaki equation.", ps(vec![]), handle_mukhanov_sasaki_cosmology),
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
        centry("epsilon_to_delta", "Convert epsilon contractions to generalized deltas.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_epsilon_to_delta),
        centry("expand_delta", "Expand generalized delta expressions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_delta),
        centry("expand_dummies", "Expand abstract dummy contractions to coordinates.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_dummies),
        centry("explicit_indices", "Insert explicit indices for implicit-index tensors.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_explicit_indices),
        centry("expand_implicit", "Expand implicit contractions.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_expand_implicit),
        centry("einsteinify", "Repair Einstein contractions by fixing dummy variances.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_einsteinify),
        centry("split_index", "Split one index family into two subfamilies.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("parent_indices", ParamType::SymbolList, true, "Parent-family indices."), pdef("subfamily_one", ParamType::SymbolList, true, "First subfamily symbols."), pdef("subfamily_two", ParamType::SymbolList, true, "Second subfamily symbols.")]), handle_split_index_tensor),
        centry("rename_dummies", "Rename dummy indices canonically.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_rename_dummies),
        centry("young_project", "Project onto Young-tableau symmetry.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_young_project),
        centry("young_project_tensor", "Project onto Young-tableau symmetry.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_young_project),
        centry("reduce_delta", "Reduce expanded deltas back to compact form.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_reduce_delta),
        centry("symmetrise", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("symmetrize", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("sym", "Symmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_symmetrise_tensor),
        centry("antisymmetrise", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("antisymmetrize", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("asym", "Antisymmetrise selected tensor slots.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("positions", ParamType::Code, true, "JSON integer array positions.")]), handle_antisymmetrise_tensor),
        centry("decompose", "Decompose an expression in a supplied basis.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("basis", ParamType::ExprId, true, "Stored list of basis expressions.")]), handle_decompose_tensor),
        centry("decompose_product", "Decompose a tensor product by dimension.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id."), pdef("dim", ParamType::Integer, false, "Optional dimension.")]), handle_decompose_product_tensor),
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
        centry("kretschner_scalar", "Compute the Kretschmann scalar.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_kretschner_scalar_gr),
        centry("kretschner", "Compute the Kretschmann scalar.", ps(vec![pdef("riemann_id", ParamType::Code, true, "Stored riemann id.")]), handle_kretschner_scalar_gr),
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
        centry("partial_trace", "Take a subsystem partial trace.", ps(vec![pdef("rho", ParamType::ExprId, true, "Stored density-matrix id."), pdef("dim_a", ParamType::Integer, true, "Subsystem A dimension."), pdef("dim_b", ParamType::Integer, true, "Subsystem B dimension."), pdef("which", ParamType::StringEnum(&["A", "B"]), true, "Subsystem to trace out.")]), handle_partial_trace_qm),
        centry("braket", "Bra-ket inner product.", ps(vec![pdef("bra", ParamType::ExprId, true, "Stored bra/list expression id."), pdef("ket", ParamType::ExprId, true, "Stored ket/list expression id.")]), handle_braket_qm),
        centry("outer", "Outer-product operator.", ps(vec![pdef("left", ParamType::ExprId, true, "Stored vector id."), pdef("right", ParamType::ExprId, true, "Stored vector id.")]), handle_outer_qm),
        centry("normal_order", "Normal-order creation and annihilation operators.", ps(vec![pdef("expr", ParamType::ExprId, true, "Stored expression id.")]), handle_normal_order_qm),
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
        centry("declare_indices", "Declare an index family.", ps(vec![pdef("family", ParamType::Symbol, true, "Family name."), pdef("indices", ParamType::SymbolList, true, "Index symbols."), pdef("dimension", ParamType::Integer, false, "Optional family dimension.")]), handle_declare_indices),
        centry("declare_coordinates", "Declare active coordinate symbols.", ps(vec![pdef("coordinates", ParamType::SymbolList, true, "Coordinate symbols.")]), handle_declare_coordinates),
        centry("declare_assumption", "Declare an assumption on a symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol."), pdef("assumption", ParamType::Code, true, "Assumption name.")]), handle_declare_assumption),
        centry("declare_grassmann", "Declare a Grassmann-odd symbol.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol.")]), handle_declare_grassmann),
        centry("declare_operator", "Declare a creation or annihilation operator.", ps(vec![pdef("symbol", ParamType::Symbol, true, "Target symbol."), pdef("kind", ParamType::StringEnum(&["creation", "annihilation"]), true, "Operator kind.")]), handle_declare_operator),
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
