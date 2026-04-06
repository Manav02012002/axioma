# Axioma Language Reference (LLM Context)

> This file is auto-generated. It is the complete reference for the Axioma scientific computing language. Inject this into your LLM system prompt or tool description when working with .ax files.

## Syntax
|pattern|meaning|example|
|---|---|---|
|module name;|Declare a module name; the core lowering step accepts and ignores it, while frontends preserve it as a file-level declaration.|module demo;|
|import std.path.name|Import a dotted module path.|import std.gr.schwarzschild|
|let x = expr|Create a top-level binding. As a bare statement it lowers to Let(x, expr, x).|let x = 5|
|let x = expr in body|Create a local binding scoped to body.|let x = 2 in x + 3|
|f(x, y) = expr|Define a function with identifier parameters.|f(x, y) = x^2 + y|
|assume x real positive integer|Attach one or more assumptions to a symbol.|assume n integer positive|
|grassmann theta eta|Declare one or more Grassmann variables.|grassmann theta eta|
|indices family [a, b, c] dim=4 values=[i, j, k] position=fixed|Declare an index family with optional dimension, explicit values, and free/fixed position metadata.|indices spacetime [mu, nu, rho, sigma] dim=4|
|coordinates [t, r, theta, phi]|Declare the active coordinate labels.|coordinates [t, r, theta, phi]|
|property T metric|Declare a tensor property on a symbol.|property g metric|
|depends T [x, t] or depends T x|Declare explicit symbol dependencies.|depends phi [t, x]|
|weight A -1 label=field|Assign an integer symbolic weight with an optional label.|weight psi 1 label=field|
|convention key value|Set an active convention entry such as metric_signature or riemann_sign.|convention riemann_sign mtw|
|rule name: lhs => rhs|Define a rewrite rule.|rule pythag: sin(x_)^2 + cos(x_)^2 => 1|
|rule [exact] name: lhs => rhs|Define a rewrite rule with a trust level.|rule [exact] pythag: sin(x_)^2 + cos(x_)^2 => 1|
|if cond then a else b|Conditional expression lowered to a two-branch piecewise form.|if x > 0 then x else -x|
|piecewise(v1, cond1, v2, cond2, etc)|Explicit piecewise constructor.|piecewise(x, x > 0, -x, true)|
|a + b, a - b, a * b, a / b, a ^ b|Arithmetic with standard precedence and right-associative exponentiation.|x + y*z^2|
|-a|Unary negation.|-x^2|
|(expr)|Parenthesized grouping.|(x + 1)^2|
|name(argsetc)|Function or builtin call syntax.|integrate(x^2, x)|
|T[mu-, nu+]|ASCII indexed-tensor syntax with explicit variance markers.|T[mu-, nu+]|
|[a, b, c]|List literal syntax.|[t, r, theta, phi]|
|[[a, b], [c, d]]|Nested list syntax commonly used for matrices.|[[1, 2], [3, 4]]|
|x > y, x >= y, x < y, x <= y, x == y, x != y, a and b, a or b, not a|Condition syntax used by if/then/else and piecewise.|if x >= 0 and y != 0 then x/y else 0|
|ident|Identifier syntax: leading ASCII letter, then letters, digits, or underscores.|alpha1|
|123, 3.14|Integer and floating-point literals.|1 + 2.5|
|// comment|Line comment syntax recognized by core lowering and the lightweight syntax lexer.|// this is a comment|
|/* comment */|Block comment syntax recognized by the lightweight syntax lexer.|/* note */|
|R_{a b c d}, T^{a}_{b}|LaTeX-style tensor indices accepted by the LaTeX translation path and converted into ASCII indexed syntax.|R_{a b c d}|
|\frac{a}{b}, \sqrt{x}, \partial_{a}|Supported LaTeX command fragments translated before lowering.|\frac{1}{2} \partial_{a} phi|

## Built-in Functions
### algebra
|name|signature|description|
|---|---|---|
|solve|solve(expr, var)|Solve one polynomial equation or a linear system.|

### analysis
|name|signature|description|
|---|---|---|
|equiv|equiv(lhs, rhs)|Describe whether two exprs are semantically equivalent.|
|semantic_diff|semantic_diff(lhs, rhs)|Return a semantic-difference descriptor.|

### calculus
|name|signature|description|
|---|---|---|
|dblint|dblint(expr, x, y)|Alias for double_integral.|
|definite_integral|definite_integral(expr, var, a, b)|Compute a definite integral from an antiderivative.|
|defint|defint(expr, var, a, b)|Alias for definite_integral.|
|diff|diff(expr, var)|Differentiate an expr symically.|
|double_integral|double_integral(expr, x, y)|Perform iterated double integration.|
|ibp|ibp(expr, u, v, var)|Alias for integrate_by_parts.|
|integrate|integrate(expr, var)|Indefinite or definite symic integration.|
|integrate_by_parts|integrate_by_parts(expr, u, v, var)|Perform one integration-by-parts step via explicit u & v'.|
|limit|limit(expr, var, point)|Evaluate a symic limit.|
|series|series(expr, var, point, order)|Compute a Taylor series.|
|tplint|tplint(expr, x, y, z)|Alias for triple_integral.|
|triple_integral|triple_integral(expr, x, y, z)|Perform iterated triple integration.|

### codegen
|name|signature|description|
|---|---|---|
|to_cpp|to_cpp(expr)|Print C++ code for an expr.|
|to_python|to_python(expr)|Print Python code for an expr.|
|to_rust|to_rust(expr)|Print Rust code for an expr.|

### complex
|name|signature|description|
|---|---|---|
|Im|Im(z)|Return imaginary part of a complex expr.|
|Re|Re(z)|Return real part of a complex expr.|
|arg|arg(z)|Return complex argument or phase.|
|conj|conj(z)|Return complex conjugate.|

### elementary
|name|signature|description|
|---|---|---|
|abs|abs(x)|Absolute value or complex modulus.|
|acos|acos(x)|Inverse cosine.|
|acosh|acosh(x)|Inverse hyperbolic cosine.|
|arccos|arccos(x)|Inverse cosine alias.|
|arccosh|arccosh(x)|Inverse hyperbolic cosine alias.|
|arcsin|arcsin(x)|Inverse sine alias.|
|arcsinh|arcsinh(x)|Inverse hyperbolic sine alias.|
|arctan|arctan(x)|Inverse tangent alias.|
|arctanh|arctanh(x)|Inverse hyperbolic tangent alias.|
|asin|asin(x)|Inverse sine.|
|asinh|asinh(x)|Inverse hyperbolic sine.|
|atan|atan(x)|Inverse tangent.|
|atan2|atan2(y, x)|Two-argument arctangent.|
|atanh|atanh(x)|Inverse hyperbolic tangent.|
|cos|cos(x)|Cosine w/ symic & numeric evaluation.|
|cosh|cosh(x)|Hyperbolic cosine.|
|cot|cot(x)|Cotangent w/ symic & numeric evaluation.|
|csc|csc(x)|Cosecant w/ symic & numeric evaluation.|
|exp|exp(x)|Exponential function.|
|log|log(x)|Natural logarithm.|
|sec|sec(x)|Secant w/ symic & numeric evaluation.|
|sgn|sgn(x)|Sign function alias.|
|sign|sign(x)|Sign function.|
|sin|sin(x)|Sine w/ symic & numeric evaluation.|
|sinh|sinh(x)|Hyperbolic sine.|
|sqrt|sqrt(x)|Square root w/ exact perfect-square simplification.|
|tan|tan(x)|Tangent w/ symic & numeric evaluation.|
|tanh|tanh(x)|Hyperbolic tangent.|

### forms
|name|signature|description|
|---|---|---|
|d|d(form)|Alias for exterior_d in forms subsystem.|
|exterior_d|exterior_d(form)|Exterior derivative of a differential form.|
|hodge_star|hodge_star(form, metric)|Hodge dual of a differential form.|
|wedge_1_1|wedge_1_1(a, b)|Wedge product of two 1-forms.|

### gr
|name|signature|description|
|---|---|---|
|christoffel|christoffel(metric, coords)|Christoffel syms from a metric.|
|covariant_diff|covariant_diff(expr, metric, coords)|Covariant derivative.|
|einstein|einstein(metric, ricci, scalar)|Einstein tensor.|
|geodesic|geodesic(metric, coords, param)|Geodesic equations for a metric.|
|kretschner|kretschner(metric, riemann)|Kretschmann scalar.|
|lie_derivative|lie_derivative(field, vector, coords)|Lie derivative of a scalar or vector field.|
|metric|metric(diag(etc))|Construct a symic metric tensor from a diagonal form.|
|ricci|ricci(riemann)|Ricci tensor from Riemann tensor.|
|ricci_scalar|ricci_scalar(metric, ricci)|Ricci scalar curvature.|
|riemann|riemann(christoffel, coords)|Riemann tensor from a connection.|

### linear-algebra
|name|signature|description|
|---|---|---|
|det|det(matrix)|Determinant of a mat.|
|diag|diag(a, b, etc)|Construct a diagonal mat.|
|eigenvalues|eigenvalues(matrix)|Eigenvalues of a small symic or numeric mat.|
|identity|identity(n)|n×n identity mat.|
|inv|inv(matrix)|Inverse of a mat.|
|matmul|matmul(a, b)|Matrix multiplication.|
|tensor_product|tensor_product(a, b)|Kronecker or tensor product of arrays or operators.|
|trace_mat|trace_mat(matrix)|Trace of a mat.|
|transpose|transpose(matrix)|Transpose a mat.|

### numeric
|name|signature|description|
|---|---|---|
|N|N(expr)|Evaluate an expr numerically when possible.|

### ode
|name|signature|description|
|---|---|---|
|dsolve|dsolve(eq, y, x)|Solve a supported first-order ODE symically.|
|first_order_form|first_order_form(ode, dep, indep)|Convert a higher-order ODE to a first-order system.|
|rk4|rk4(f, x, y, x0, y0, x1[, steps])|Numerically integrate an ODE w/ fourth-order Runge-Kutta.|

### pde
|name|signature|description|
|---|---|---|
|classify_pde|classify_pde(A, B, C)|Classify a second-order PDE as elliptic, parabolic, or hyperbolic.|
|separate_variables|separate_variables(type, x, t[, coeff])|Return a separated solution ansatz for a supported PDE family.|
|separation|separation(type, x, t[, coeff])|Alias for separate_variables.|

### plotting
|name|signature|description|
|---|---|---|
|plot|plot(expr, var, xmin, xmax)|Plot a one-dimal expr to SVG.|

### properties
|name|signature|description|
|---|---|---|
|anti_commuting|anti_commuting(symbol)|Alias for anticommuting.|
|anticommuting|anticommuting(symbol)|Declare an object as anticommuting.|
|antisymmetric|antisymmetric(tensor)|Property marker used in prop declarations & metadata.|
|bianchi|bianchi(tensor)|Declare that a tensor satisfies a Bianchi identity.|
|commuting|commuting(symbol)|Declare an object as commuting.|
|covariant_derivative|covariant_derivative(op)|Declare a sym as a covariant derivative operator.|
|derivative|derivative(op)|Declare a sym as a derivative operator.|
|dirac_bar|dirac_bar(symbol)|Declare a sym as a Dirac-bar object.|
|diracbar|diracbar(symbol)|Alias for dirac_bar.|
|epsilon|epsilon(tensor)|Declare a tensor as an epsilon or Levi-Civita tensor.|
|epsilon_tensor|epsilon_tensor(tensor)|Alias for epsilon.|
|gamma_matrix|gamma_matrix(symbol)|Declare a sym as a gamma mat.|
|inverse_metric|inverse_metric(tensor)|Declare a tensor as an inverse metric.|
|kronecker|kronecker(tensor)|Alias for kronecker_delta.|
|kronecker_delta|kronecker_delta(tensor)|Declare a tensor as a Kronecker delta.|
|metric|metric(tensor)|Declare a tensor as a metric & attach symmetric metric prop.|
|non_commuting|non_commuting(symbol)|Alias for noncommuting.|
|noncommuting|noncommuting(symbol)|Declare an object as noncommuting.|
|partial_derivative|partial_derivative(op)|Declare a sym as a partial derivative operator.|
|riemann|riemann(tensor)|Declare Riemann slot symmetries on a tensor.|
|riemann_symmetry|riemann_symmetry(tensor)|Property marker for Riemann-like slot symmetries.|
|satisfies_bianchi|satisfies_bianchi(tensor)|Alias for bianchi.|
|spinor|spinor(tensor)|Declare a tensor as carrying spinor idx.|
|symmetric|symmetric(tensor)|Property marker used in prop declarations & metadata.|
|tableau_symmetry|tableau_symmetry(tensor, shape, indices)|Declare Young-tableau symmetry data on a tensor.|
|traceless|traceless(tensor)|Property marker for traceless tensors.|
|weyl|weyl(tensor)|Declare a tensor as a Weyl tensor.|
|weyl_tensor|weyl_tensor(tensor)|Alias for weyl.|

### quantum
|name|signature|description|
|---|---|---|
|annihilation|annihilation(sym)|Declare or mark an annihilation operator.|
|anticommutator|anticommutator(a, b)|Operator anticommutator {a, b}.|
|bra|bra(label)|Construct a bra vector.|
|braket|braket(bra, ket)|Inner product of a bra & a ket.|
|commutator|commutator(a, b)|Operator commutator [a, b].|
|creation|creation(sym)|Declare or mark a creation operator.|
|density|density(state)|Density mat of a pure state.|
|gamma|gamma(index)|Dirac gamma mat for an idx.|
|gamma5|gamma5()|Dirac gamma_5 mat.|
|gamma5_trace|gamma5_trace(expr)|Trace a gamma-chain w/ gamma_5 inserted.|
|gamma_trace|gamma_trace(expr)|Trace over a chain of gamma mats.|
|grassmann|grassmann(sym)|Declare a Grassmann-odd sym.|
|grassmann_simplify|grassmann_simplify(expr)|Simplify via Grassmann anticommutation.|
|join_gamma|join_gamma(expr)|Join adjacent gamma mats into a compact gamma chain.|
|ket|ket(label)|Construct a ket vector.|
|normal_order|normal_order(expr)|Reorder ladder operators into normal order.|
|outer|outer(ket, bra)|Outer product operator.|
|partial_trace|partial_trace(rho, subsystem)|Partial trace over a subsystem.|
|pauli_x|pauli_x()|Pauli sigma_x mat.|
|pauli_y|pauli_y()|Pauli sigma_y mat.|
|pauli_z|pauli_z()|Pauli sigma_z mat.|
|sigma_x|sigma_x()|Alias for pauli_x.|
|sigma_y|sigma_y()|Alias for pauli_y.|
|sigma_z|sigma_z()|Alias for pauli_z.|
|split_gamma|split_gamma(expr)|Split compact gamma-chain structures into explicit factors.|
|wick|wick(expr)|Expand products via Wick contraction rules.|

### rewrite
|name|signature|description|
|---|---|---|
|rewrite|rewrite(expr)|Apply user-defined rewrite rules to an expr.|
|subs|subs(expr, target, replacement)|Perform symic substitution w/ idx-aware matching when needed.|
|take_match|take_match(expr, pattern)|Keep only parts of a sum that match a pattern.|
|unzoom|unzoom(focus, remainder)|Recombine a focused expr w/ its remainder.|
|zoom|zoom(expr, pattern)|Split an expr into matching & non-matching parts.|

### simplify
|name|signature|description|
|---|---|---|
|apart|apart(expr, var)|Alias for partial_fractions.|
|expand|expand(expr)|Distribute products & expand small powers.|
|factor_in|factor_in(expr[, targets])|Group terms that share common prefactors.|
|factor_out|factor_out(expr[, targets])|Factor common factors from a sum.|
|partial_fractions|partial_fractions(expr, var)|Decompose a rational function into partial fractions when supported.|
|rationalize|rationalize(expr)|Put sums over a common denominator & cancel common factors.|
|simplify|simplify(expr)|Run full simplification pipeline.|
|trig_simplify|trig_simplify(expr)|Apply exact trigonometric rewrite rules.|

### syntax
|name|signature|description|
|---|---|---|
|assume|assume(sym, property)|Attach assumptions such as real, positive, integer, even, or odd to a sym.|
|import|import(path)|Import a standard-library module into current env.|

### tensor
|name|signature|description|
|---|---|---|
|antisymmetrise|antisymmetrise(expr, [positions])|Antisymmetrise over listed slots.|
|antisymmetrize|antisymmetrize(expr, [positions])|Alias for antisymmetrise.|
|asym|asym(expr, [positions])|Short alias for antisymmetrise.|
|canonicalise|canonicalise(expr)|Canonicalize tensor idx via decl. tensor props.|
|canonicalize|canonicalize(expr)|Alias for canonicalise.|
|decompose|decompose(expr)|Decompose a tensor into symmetry-adapted pieces.|
|decompose_product|decompose_product(expr)|Decompose a tensor product via known tensor props.|
|drop_weight|drop_weight(expr, label, value)|Remove terms w/ a recorded symic weight.|
|einsteinify|einsteinify(expr)|Insert implicit Einstein summation contractions.|
|eliminate_kronecker|eliminate_kronecker(expr)|Contract Kronecker deltas via an expr.|
|eliminate_metric|eliminate_metric(expr)|Use metric or inverse metric to raise or lower contracted idx.|
|eliminate_vielbein|eliminate_vielbein(expr)|Simplify vielbein contractions into metric data when possible.|
|epsilon_to_delta|epsilon_to_delta(expr)|Convert epsilon-tensor contractions into gen. Kronecker deltas.|
|eval_components|eval_components(expr, rules)|Alias for evaluate component exprs.|
|evaluate|evaluate(expr, rules)|Evaluate tensor components via decl. component rules.|
|expand_delta|expand_delta(expr)|Expand delta contractions into explicit sums when possible.|
|expand_dummies|expand_dummies(expr)|Expand dummy sums over decl. coord set.|
|expand_implicit|expand_implicit(expr)|Expand implicit tensor contractions & idx conventions.|
|explicit_indices|explicit_indices(expr)|Make implicit repeated idx explicit.|
|keep_weight|keep_weight(expr, label, value)|Filter terms by a recorded symic weight.|
|leibniz|leibniz(expr)|Alias for product_rule.|
|lower_free_indices|lower_free_indices(expr)|Lower free upper idx via active metric family.|
|lower_indices|lower_indices(expr)|Alias for lower_free_idx.|
|meld|meld(expr)|Detect multi-term tensor identities via Young projection & linear dependence.|
|product_rule|product_rule(expr)|Apply Leibniz rule to an idxed product.|
|raise_free_indices|raise_free_indices(expr)|Raise free lower idx via active inverse metric family.|
|raise_indices|raise_indices(expr)|Alias for raise_free_idx.|
|reduce_delta|reduce_delta(expr)|Simplify explicit delta-expanded exprs back to compact form.|
|rename_dummies|rename_dummies(expr)|Rename dummy idx to a canonical fresh naming scheme.|
|rewrite_indices|rewrite_indices(expr)|Rewrite idx names while preserving variance & families.|
|sort_product|sort_product(expr)|Sort tensor products via symmetry-aware canonicalization.|
|split_index|split_index(expr, old, [newetc])|Split one abstract idx family into several fixed values.|
|sym|sym(expr, [positions])|Short alias for symmetrise.|
|symmetrise|symmetrise(expr, [positions])|Symmetrise over listed slots.|
|symmetrize|symmetrize(expr, [positions])|Alias for symmetrise.|
|tdistribute|tdistribute(expr)|Alias for tensor_distribute.|
|tensor_distribute|tensor_distribute(expr)|Distribute products over sums in tensor exprs.|
|unwrap|unwrap(expr)|Flatten nested additive & multiplicative structure.|
|young_project|young_project(expr)|Project a tensor onto a decl. Young-tableau symmetry.|

### units
|name|signature|description|
|---|---|---|
|check_units|check_units(expr)|Verify that a units expr is dimally consistent.|
|convert|convert(expr, units)|Convert an expr between units.|
|dim|dim(expr)|Return dim of a units-aware expr.|

### variational
|name|signature|description|
|---|---|---|
|euler_lagrange|euler_lagrange(L, field, coords)|Compute Euler-Lagrange equations.|
|vary|vary(expr, field)|Take a formal variation w/ respect to a field.|

### vector-calculus
|name|signature|description|
|---|---|---|
|curl|curl([Fx, Fy, Fz], [x, y, z])|Return three-dimal curl.|
|div|div([Fx, Fy, Fz], [x, y, z])|Alias for divergence.|
|divergence|divergence([Fx, Fy, Fz], [x, y, z])|Return divergence of a vector field.|
|grad|grad(f, [x, y, z])|Alias for gradient.|
|gradient|gradient(f, [x, y, z])|Return gradient vector.|
|hessian|hessian(f, [x1, etc])|Return Hessian mat.|
|jacobian|jacobian([f1, etc], [x1, etc])|Return Jacobian mat.|
|laplacian|laplacian(f, [x, y, z])|Return Laplacian.|

## Tensor Properties
|property|syntax|description|enables|
|---|---|---|---|
|Symmetric|property T symmetric([positions])|Indices are symmetric under exchange of listed slots.|build_generating_set, canonicalize_idx, tableaux_from_props, handle_factor symmetry lookup|
|AntiSymmetric|property T antisymmetric([positions])|Indices are antisymmetric under exchange of listed slots.|build_generating_set, canonicalize_idx, tableaux_from_props, handle_factor symmetry lookup|
|RiemannSymmetry|property R riemann_symmetry|Apply pair antisymmetry & pair-exchange symmetry of a Riemann tensor.|build_generating_set, canonicalize_idx, tableaux_from_props, handle_factor symmetry lookup|
|Traceless|property T traceless|Marks a tensor as traceless.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|Metric|property g metric|Marks a tensor as a metric used to lower idx & define dummy-pair symmetry.|lower_free_idx, eliminate_metric, metric_symmetry_for_slots|
|InverseMetric|property g inverse_metric|Marks a tensor as an inverse metric used to raise idx & define dummy-pair symmetry.|raise_free_idx, eliminate_metric, metric_symmetry_for_slots|
|KroneckerDelta|property d kronecker_delta|Marks a tensor as a Kronecker delta.|eliminate_kronecker, expand_delta, reduce_delta|
|EpsilonTensor|property eps epsilon_tensor|Marks a tensor as a Levi-Civita epsilon tensor.|epsilon_to_delta, handle_epsilon component evaluation|
|Derivative|property D derivative|Marks a sym as a derivative operator.|stored by ax-tensor metadata; derivative handling keys off names instead|
|PartialDerivative|property D partial_derivative|Marks a sym as a partial derivative operator.|stored by ax-tensor metadata; derivative handling keys off names instead|
|CovariantDerivative|property nabla covariant_derivative|Marks a sym as a covariant derivative operator.|stored by ax-tensor metadata; derivative handling keys off names instead|
|Depends|property T depends([x, y, etc])|Declares that a tensor depends on listed syms.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|Spinor|property psi spinor|Marks a tensor as carrying spinor idx.|canonicalise_product dummy classification via metric_symmetry_for_slots|
|DiracBar|property psibar dirac_bar|Marks a sym as a Dirac-bar object.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|GammaMatrixProp|property gamma gamma_matrix|Marks a sym as a gamma mat.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|Commuting|property A commuting|Marks an object as commuting.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|AntiCommuting|property psi anticommuting|Marks an object as anticommuting.|canonicalise_product dummy classification via metric_symmetry_for_slots|
|NonCommuting|property A noncommuting|Marks an object as noncommuting.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|SortOrder|property T sort_order([etc])|Declares an explicit preferred order of syms.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|TableauSymmetry|property T tableau_symmetry(shape=[etc], indices=[etc])|Declares a Young-tableau symmetry shape & slot assignment.|young_project_tensor|
|SatisfiesBianchi|property R satisfies_bianchi|Marks a tensor as satisfying a Bianchi identity.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|WeylTensor|property C weyl_tensor|Marks a tensor as a Weyl tensor.|stored by ax-tensor metadata; no direct ax-tensor algorithm consults it|
|DifferentialFormDegree|property F differential_form_degree(n)|Declares degree of a differential form.|stored by ax-tensor metadata; differential-form algorithms live outside ax-tensor|

## Conventions
|field|options|default|description|
|---|---|---|---|
|metric_signature|MostlyPlus, MostlyMinus|MostlyPlus|Chooses sign convention for metric signature.|
|riemann_sign|MTW, Weinberg|MTW|Chooses sign convention for Riemann tensor definition.|
|ricci_contraction|FirstThird, FirstFourth|FirstThird|Chooses which Riemann slots are contracted to form Ricci tensor.|
|levi_civita_norm|PlusOne, MinusOne, SqrtG|PlusOne|Chooses normalization convention for Levi-Civita tensor.|
|fourier_sign|MinusI, PlusI|MinusI|Chooses sign convention in Fourier-transform exponentials.|

## Assumptions
|name|description|
|---|---|
|Real|Expression is assumed to take real values.|
|Positive|Expression is assumed strictly positive.|
|Negative|Expression is assumed strictly negative.|
|NonZero|Expression is assumed not equal to zero.|
|Integer|Expression is assumed to be an integer.|
|Even|Expression is assumed to be an even integer.|
|Odd|Expression is assumed to be an odd integer.|

## Algorithms
### calculus
|name|signature|preconditions|description|
|---|---|---|---|
|differentiate|differentiate(expr, var)|differentiation variable must be a sym.|Take a symic derivative w/ chain, product, & builtin function rules.|

### forms
|name|signature|preconditions|description|
|---|---|---|---|
|exterior_derivative|exterior_derivative(form, coords)|form.dim must equal coords.len().|Take exterior derivative of a differential form by differentiating components & wedging in basis one-forms.|
|hodge_dual|hodge_dual(form, g)|metric dim must equal form dim; impl uses symic inverse & determinant of g.|Take Hodge dual of a differential form w/ respect to a symic metric.|
|wedge|wedge(a, b)|Both forms must have same ambient dim.|Compute antisymmetric wedge product of two differential forms.|

### gr
|name|signature|preconditions|description|
|---|---|---|---|
|christoffel_from_metric|christoffel_from_metric(g, coords)|metric must be square & coords.len() must equal g.dim; routine uses symic inverse of g.|Compute Christoffel syms from a symic metric by Levi-Civita formula.|
|covariant_derivative_covector|covariant_derivative_covector(w, gamma, coord_index, coords)|covector length, connection dims, & coord list length must agree.|Compute ∇_coord_idx w for a covector field.|
|covariant_derivative_tensor2|covariant_derivative_tensor2(t, gamma, coord_index, coords)|Tensor dims, connection dims, & coord count must agree.|Compute covariant derivative of a rank-2 covariant tensor.|
|covariant_derivative_vector|covariant_derivative_vector(v, gamma, coord_index, coords)|vector length, connection dims, & coord list length must agree.|Compute ∇_coord_idx v for a contravariant vector field.|
|einstein_tensor|einstein_tensor(ricci, scalar, g)|metric dim must match Ricci tensor dims.|Build Einstein tensor G_ab = R_ab - 1/2 g_ab R.|
|geodesic_equations|geodesic_equations(gamma, coords)|Connection dims must match coord list.|Construct geodesic equations ẍ^i = -Γ^i_jk ẋ^j ẋ^k in symic form.|
|kretschner_scalar|kretschner_scalar(riemann, g)|metric must be invertible; this impl contracts via diagonal entries of g & g^{-1}.|Compute a diagonal-metric approximation to Kretschmann scalar from squared Riemann components.|
|lie_derivative_scalar|lie_derivative_scalar(f, v, coords)|vector field length must match coord list length.|Compute Lie derivative of a scalar along a vector field.|
|lie_derivative_vector|lie_derivative_vector(w, v, coords)|Both vectors must have same length as coords.|Compute Lie derivative of a vector field along another vector field.|
|ricci_from_riemann|ricci_from_riemann(riemann, n, convention)|n must match tensor dims; Convention selects first-third or first-fourth contraction.|Contract a Riemann tensor into Ricci tensor via configured Ricci-contraction convention.|
|ricci_scalar|ricci_scalar(ricci, ginv)|inverse metric dim must match Ricci tensor dims.|Contract Ricci tensor w/ inverse metric to obtain scalar curvature.|
|riemann_from_christoffel|riemann_from_christoffel(gamma, coords, convention)|connection array dims must match coords.len(); Convention determines MTW versus Weinberg sign.|Compute Riemann tensor from Christoffel syms, respecting active sign convention.|

### linalg
|name|signature|preconditions|description|
|---|---|---|---|
|determinant|determinant(matrix)|mat square; symic simplification is applied recursively by minors.|Compute determinant of a symic square mat.|
|eigenvalues_symbolic|eigenvalues_symbolic(matrix)|mat square; solving that polynomial is a separate step.|Return characteristic polynomial det(A - lambda I) for a symic mat.|
|inverse|inverse(matrix)|mat must be square & have nonzero determinant.|Compute symic inverse of a square mat by adjugate over determinant.|
|tensor_product|tensor_product(a, b)|Both inputs must be rectangular mats.|Compute Kronecker product of two mats.|
|trace|trace(matrix)|mat square.|Compute trace of a square mat.|

### ode
|name|signature|preconditions|description|
|---|---|---|---|
|classify_pde|classify_pde(a, b, c)|discriminant must simplify to a numeric sign to get a definite classification; otherwise result is Unknown.|Classify a second-order PDE from its A, B, C coefficients via discriminant B^2 - A*C.|
|first_order_form|first_order_form(ode, dependent_var, independent_var, interner) -> Vec<(Expr, Expr)|ODE should contain nested diff calls w/ respect to independent_var, or else it is treated as right-hand side of a second-order equation.|Convert a higher-order ODE into a first-order system by introducing aux derivative variables.|
|rk4|rk4(f, x_sym, y_sym, x0, y0, x_end, n_steps, interner) -> Vec<(f64, f64)|f must evaluate numerically for supplied bindings, & n_steps must be nonzero.|Numerically integrate a scalar first-order ODE y' = f(x, y) w/ fourth-order Runge-Kutta.|
|rk4_system|rk4_system(fs, x_sym, y_syms, x0, y0s, x_end, n_steps)|numbers of equations, dependent variables, & initial values must match, & each expr must evaluate numerically.|Numerically integrate a coupled first-order ODE system w/ fourth-order Runge-Kutta.|
|separate_variables|separate_variables(pde_type, spatial_var, temporal_var, coefficient)|This is a template generator for wave, heat, & Laplace-type equations rather than an automatic PDE parser.|Return a separated-variables ansatz for hyperbolic, parabolic, or elliptic PDE families.|
|solve_ode|solve_ode(equation, y_sym, x_sym)|ODE right-hand side must match one of supported separable or linear forms; otherwise an unevaluated solve_ode call is returned.|Solve simple separable or first-order linear ODEs symically.|

### qm
|name|signature|preconditions|description|
|---|---|---|---|
|anticommutator|anticommutator(a, b)|mats must be dimally for multiplication.|Compute mat anticommutator AB + BA.|
|braket|braket(bra, ket)|two vectors should have same length.|Compute inner product of a bra & ket by componentwise contraction.|
|commutator|commutator(a, b)|mats must be dimally for multiplication.|Compute mat commutator AB - BA.|
|density_matrix|density_matrix(state)|state given as a finite component vector.|Build rank-one density mat \|psi><psi\| from a state vector.|
|fierz|fierz(expr, dim)|routine produces abstract gamma_basis expansion rather than fully rearranging a concrete spinor expr.|Return formal Fierz-basis expansion coefficients for given spacetime dim.|
|gamma5|gamma5()|No extra setup is needed.|Return Dirac gamma_5 mat.|
|gamma_trace|gamma_trace(indices, metric)|input must already be parsed into GammaEntry values; impl assumes four-dimal Dirac trace normalization.|Trace a gamma-mat chain, including special gamma5 epsilon-tensor case.|
|grassmann_simplify|grassmann_simplify(expr, gradings)|Grassmann or operator gradings must be present in env.|Simplify products of commuting & anticommuting syms via stored gradings.|
|join_gammas_in_expr|join_gammas_in_expr(expr)|Gamma factors must be represented as Call(gamma_sym, [etc]) nodes & use a metric sym.|Join adjacent gamma-mat factors into antisymmetrized multi-idx gamma objects plus metric contractions.|
|normal_order|normal_order(expr)|Operator kinds must be decl. for syms that should reorder.|Reorder products of operators into normal order via decl. creation/annihilation kinds.|
|outer|outer(ket, bra)|two vectors should have finite explicit components.|Build outer-product operator \|ket><bra\| from two vectors.|
|partial_trace|partial_trace(rho, dim_a, dim_b, trace_over)|rho must be arranged as a (dim_a*dim_b) square mat, & trace_over must be 'A' or 'B'.|Trace out subsystem A or B from a bipartite density mat.|
|pauli_x|pauli_x()|No extra setup is needed.|Return Pauli sigma_x mat.|
|pauli_y|pauli_y()|No extra setup is needed.|Return Pauli sigma_y mat.|
|pauli_z|pauli_z()|No extra setup is needed.|Return Pauli sigma_z mat.|
|split_gamma|split_gamma(expr, on_back)|input must contain gamma_sym calls w/ more than one idx.|Split a multi-idx antisymmetric gamma mat into a shorter chain plus contraction terms.|
|wick_expand|wick_expand(expr)|Operator kinds & any nonzero contraction values must be supplied explicitly.|Expand operator products into normal-ordered terms plus single contractions.|

### rewrite
|name|signature|preconditions|description|
|---|---|---|---|
|describe_rewrite_trace|describe_rewrite_trace(trace)|A trace from rewrite_with_trace is needed.|Render a human-readable summary of a rewrite trace.|
|match_tensor_pattern|match_tensor_pattern(pattern, expr)|Index-family information in env improves matching across renamed abstract idx.|Match idxed tensor patterns via variance & idx-family compatibility rather than literal idx names.|
|multi_substitute|multi_substitute(expr, substitutions)|Targets are applied structurally rather than by solving matching ambiguities.|Apply several exact substitutions in one pass.|
|rewrite_with_trace|rewrite_with_trace(expr, interner) -> (Expr, Vec<RewriteStep>)|Rewrite rules must be registered in env.|Apply registered rewrite rules & return both rewritten expr & a trace of applied rules.|
|substitute_with_indices|substitute_with_indices(expr, target, replacement)|Use when expr or rule contains idxed tensors.|Perform substitution while renaming bound dummy idx to avoid capture & preserving idx-family matches.|
|symbolic_substitute|symbolic_substitute(expr, target, replacement)|Best suited to scalar exprs w/o tensor-idx matching requirements.|Replace exact symic subexprs recursively.|
|take_match|take_match(expr, pattern)|Most useful on additive exprs.|Keep only subterms of a sum that match a pattern.|
|unzoom|unzoom(focus, remainder)|focus & remainder should come from a zoom step.|Recombine a focused expr w/ its saved remainder.|
|zoom|zoom(expr, pattern, interner) -> (Expr, Expr)|Most useful on additive exprs.|Split a sum into matching & nonmatching parts w/ respect to a pattern.|

### solve
|name|signature|preconditions|description|
|---|---|---|---|
|solve|solve(equation, var)|equation must reduce to a polynomial in var; otherwise function returns an unevaluated solve call.|Solve a univariate polynomial equation when its coefficients can be extracted.|
|solve_linear_system|solve_linear_system(equations, vars, interner) -> Option<Vec<(Spur, Expr)|Every equation must be linear in listed variables, & system must have a unique consistent solution.|Solve a linear system over exact rationals by Gaussian elimination.|

### syntax
|name|signature|preconditions|description|
|---|---|---|---|
|eval|eval(expr)|Environment declarations, rules, coords, & tensor props affect result.|Evaluate an expr by dispatching builtins, rewrite rules, declarations, & symic simplifications.|
|resolve_import|resolve_import(path)|imported module must exist under std/ or another supported search root.|Resolve a std-module import path to corresponding .ax file on disk.|

### tensor
|name|signature|preconditions|description|
|---|---|---|---|
|canonicalise|canonicalise(expr)|Tensor symmetries must be present in tensor_props for anything beyond lexicographic idx ordering.|Canonicalize tensor monomials & sums via decl. slot symmetries & dummy-idx canonicalization.|
|canonicalize_indices|canonicalize_indices(expr)|Useful symmetry props such as Symmetric, AntiSymmetric, or RiemannSymmetry must be decl. on tensor syms.|Apply local idx-slot canonicalization from decl. tensor symmetries before product-level canonicalization.|
|complete_inverse_metric|complete_inverse_metric(metric_rules, inv_metric_sym, coordinates)|Metric component rules must define an invertible square metric over supplied coord list.|Construct inverse-metric component rules from metric component rules by symic mat inversion.|
|compute_weight|compute_weight(expr)|Weight assignments decl. for participating syms.|Compute total symic weight of an expr under a chosen label.|
|decompose|decompose(expr, basis)|basis should span intended subspace, & tensor_props should contain symmetries needed for canonical matching.|Express a tensor expr as a rational linear combination of a supplied canonical basis plus any residual unmatched terms.|
|decompose_product|decompose_product(expr, dim)|input a product of two rank-2 idxed tensors.|Decompose a rank-2 tensor product into symmetric, antisymmetric, & trace metric-built pieces.|
|diff_component|diff_component(expr, var)|variable a coord or scalar sym.|Differentiate a component expr w/ tensor-aware fallback handling.|
|drop_weight|drop_weight(expr, target_weight)|Weight assignments for syms under chosen label must be present when nonzero weights are needed.|Remove terms whose computed symic weight equals target_weight.|
|einsteinify|einsteinify(expr)|Useful on products where same abstract idx appears twice w/ both slots up or both slots down.|Fix repeated-idx pairs that have same variance by flipping one slot so Einstein summation becomes well-formed.|
|eliminate_kronecker|eliminate_kronecker(expr)|delta sym must identify a two-idx Kronecker delta w/ one up & one down slot.|Use Kronecker deltas to substitute contracted idx & remove delta factors from products.|
|eliminate_metric|eliminate_metric(expr, inv_metric_sym)|Metric components must use two down idx & inverse-metric components two up idx.|Use metric or inverse-metric factors to raise or lower contracted idx & remove those metric factors.|
|eliminate_vielbein|eliminate_vielbein(expr, vielbein_sym, inv_vielbein_sym)|Vielbein factors must appear as idxed two-slot tensors w/ one contractible idx matching another factor.|Use vielbein or inverse-vielbein factors to convert contracted idx between two families & remove conversion factors.|
|epsilon_to_delta|epsilon_to_delta(expr, dim)|epsilon sym & target delta sym must be supplied, & epsilon factors must carry dim idx.|Rewrite products of epsilon tensors into factorial factors times gen. Kronecker deltas.|
|evaluate_components|evaluate_components(expr, rules, index_values)|Concrete component rules & coords must be available via evaluation env; tensor_props are used for symmetry & epsilon handling.|Evaluate tensor exprs into explicit component exprs, including dummy summations, derivative handling, symmetry-aware lookups, & epsilon components.|
|evaluate_components_v2|evaluate_components_v2(expr, rules)|Component rules, coords, & tensor props must be available via env.|Evaluate tensor components w/ newer handler-based evaluation pipeline.|
|expand_delta|expand_delta(expr)|delta sym must identify an idxed tensor w/ an even number of slots split into equal up/down sets.|Expand a gen. Kronecker delta into a signed sum of ordinary two-idx deltas.|
|expand_dummies|expand_dummies(expr, coordinates)|A coord list must be supplied; abstract dummy names not already in that list are expanded.|Replace each dummy idx pair by an explicit sum over supplied coord labels.|
|expand_implicit|expand_implicit(expr, implicit_index_tensors, available_indices, n_indices_per_tensor)|Implicit-idx tensor names & their slot counts must be decl.; disjoint fresh idx available.|Recursively make implicit tensor contractions explicit across sums, products, & call arguments.|
|explicit_indices|explicit_indices(expr, implicit_index_tensors, available_indices, n_indices_per_tensor)|Tensor names that should receive implicit idx must be listed, & current impl only expands common two-idx case.|Insert explicit mat-style idx for implicit-idx tensors inside products.|
|integrate_by_parts|integrate_by_parts(expr, away_from, derivative_syms)|expr should contain a derivative operator from derivative_syms acting on a factor that contains away_from; boundary terms are assumed to vanish.|Perform one integration-by-parts rewrite by moving a derivative off factor containing away_from.|
|keep_weight|keep_weight(expr, target_weight)|Weight assignments for syms under chosen label must be present when nonzero weights are needed.|Keep only terms whose computed symic weight equals target_weight.|
|lower_free_indices|lower_free_indices(expr)|Index families decl. when only some families are free-position idx; otherwise all singly-occurring upper idx are lowered.|Flip free upper idx to lower variance w/o inserting an explicit metric.|
|meld|meld(expr)|Best results require symmetry props such as Symmetric, AntiSymmetric, RiemannSymmetry, or TableauSymmetry on factors involved.|Detect multi-term tensor cancellations by canonicalization, Young projection, & rational linear dependence testing.|
|product_rule|product_rule(expr, derivative_syms)|derivative operator syms must be listed in derivative_syms.|Expand derivative operators over products & sums via Leibniz rule.|
|raise_free_indices|raise_free_indices(expr)|Index families decl. when only some families are free-position idx; otherwise all singly-occurring lower idx are raised.|Flip free lower idx to upper variance w/o inserting an explicit inverse metric.|
|reduce_delta|reduce_delta(expr)|delta sym & sym representing dim must be supplied.|Iteratively contract products & traces of Kronecker deltas back to simpler delta or dim factors.|
|rename_dummies|rename_dummies(expr)|Index-family data improves generated names; w/o it, generic _dN names are used.|Rename dummy idx to stable family-aware names so alpha-equivalent contractions compare equal.|
|rename_dummy_indices|rename_dummy_indices(expr, prefix)|Useful when preparing exprs for display or comparison.|Rename repeated contracted idx to fresh stable names w/ chosen prefix.|
|rewrite_indices|rewrite_indices(expr, target_tensors, inv_metric_sym)|Each target tensor must have a full desired-variance specification per slot, & metric syms must be supplied.|Insert metric or inverse-metric factors so selected tensors end up w/ requested slot variances.|
|sort_product|sort_product(expr)|No special setup is needed; tensor_props is not consulted by sorter.|Sort multiplicative factors into a stable order for tensor exprs.|
|split_index|split_index(expr, parent_indices, sub1_indices, sub2_indices)|parent idx names to split must be listed, & each target subfamily list non-empty if it is intended to contribute terms.|Replace occurrences of a parent idx family by sums over two sub-families.|
|symmetrise|symmetrise(expr, positions, antisymmetric)|listed positions must refer to valid idx slots in target idxed factor or flattened product ordering.|Symmetrize or antisymmetrize an expr over specific idx slots by averaging over permutations.|
|tensor_distribute|tensor_distribute(expr)|No extra setup is needed.|Distribute tensor products over sums, including sums that appear in idxed bases.|
|unwrap_derivatives|unwrap_derivatives(expr, derivative_syms, depends)|Derivative syms must be listed explicitly, & dependence information populated for syms that are not constant.|Pull factors that do not depend on differentiation variables outside derivative operators, & kill derivatives of constants.|
|young_project|young_project(expr, tableau)|A valid tableau cell layout must be supplied in slot-number form.|Project an expr w/ a specific Young tableau by antisymmetrizing columns & symmetrizing rows.|
|young_project_tensor|young_project_tensor(expr)|tensor sym must carry a TableauSymmetry prop in tensor_props.|Apply a decl. TableauSymmetry prop directly to a tensor expr.|

### variational
|name|signature|preconditions|description|
|---|---|---|---|
|euler_lagrange_system|euler_lagrange_system(lagrangian, fields, coords)|Each field entry must provide derivative syms aligned w/ coords.|Compute Euler-Lagrange equations for several fields at once.|
|functional_derivative|functional_derivative(lagrangian, field, field_derivs, coords)|field_derivs & coords aligned so each derivative sym corresponds to differentiation w/ respect to matching coord.|Compute Euler-Lagrange functional derivative δL/δfield for first-derivative Lagrangians.|
|vary_action|vary_action(lagrangian, field, variation, field_derivs, variation_derivs)|field_derivs & variation_derivs must be aligned term-by-term.|Form first variation of an action density before integrating by parts.|

## Standard Library
|module|description|provides|
|---|---|---|
|algebra|Notes algebra operations used for expansion & simplification.|documentation comments only|
|calculus|Documents calculus builtins for differentiation, integration, series, & limits.|documentation comments only|
|conventions/landau|Sets Landau-Lifshitz sign & curvature conventions.|convention metric_signature mostly_minus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|conventions/mtw|Sets Misner-Thorne-Wheeler general-relativity conventions.|convention metric_signature mostly_plus, convention riemann_sign mtw, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|conventions/particle_physics|Sets particle-physics sign conventions.|convention metric_signature mostly_plus, convention riemann_sign mtw, convention fourier_sign minus_i|
|conventions/weinberg|Sets Weinberg general-relativity conventions.|convention metric_signature mostly_plus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|gr/de_sitter|Builds de Sitter metric & its Christoffel syms in static coords.|let f, let g, let coords, let Gamma|
|gr/frw|Builds a flat FRW metric w/ symic scale factor & computes Christoffel syms.|let g, let coords, let Gamma|
|gr/kerr_newman|Defines symic Kerr-Newman metric component exprs in Boyer-Lindquist coords.|let Sigma_expr, let Delta_expr, let g_tt, let g_rr, let g_theta_theta, let g_phi_phi, let g_t_phi|
|gr/minkowski|Builds flat Minkowski spacetime & its vanishing Christoffel syms.|let g, let coords, let Gamma|
|gr/schwarzschild|Builds Schwarzschild metric, Christoffel syms, Riemann tensor, & Ricci tensor.|let g, let coords, let Gamma, let R, let Ric|
|physics/classical_mechanics|Notes intended Euler-Lagrange workflow for classical mechanics.|documentation comments only|
|physics/klein_gordon|Sets up a Klein-Gordon Lagrangian & computes its Euler-Lagrange equation.|let dphi_dt, let dphi_dx, let dphi_dy, let dphi_dz, let L, let EOM|
|physics/maxwell|Notes for Maxwell theory setup from a Lagrangian.|documentation comments only|
|qft/dirac|Summarizes Dirac equation & basic gamma-trace identities.|documentation comments only|
|qft/gamma|Introduces symic gamma & eta objects for gamma-mat algebra experiments.|let gamma, let eta|
|qft/normal_ordering|Documents normal ordering & Wick expansion usage.|documentation comments only|
|qft/scalar_field|Summarizes free scalar-field Lagrangian & Klein-Gordon equation.|documentation comments only|
|qm/bell|Constructs a Bell state, its density mat, & a reduced density mat by partial trace.|let up, let down, let phi_plus, let rho, let rho_A|
|qm/harmonic_oscillator|Documents intended harmonic-oscillator operator setup.|documentation comments only|
|qm/spin|Builds Pauli mats & their commutator as a spin-1/2 algebra example.|let sigma_x, let sigma_y, let sigma_z, let comm_xy|
|tensor/index|Documents idx notation & contraction conventions for tensors.|documentation comments only|
|tensor/symmetry|Documents tensor-symmetry declarations & examples.|documentation comments only|
|trig|Defines exact trigonometric rewrite rules.|rule pythag, rule pythag_alt1, rule pythag_alt2, rule double_sin, rule double_cos|
|units/cgs|Documents CGS unit system & derived units.|documentation comments only|
|units/natural|Documents natural-unit system convention.|documentation comments only|
|units/si|Documents SI unit system import & usage.|documentation comments only|

## Common Workflows
### Schwarzschild Ricci tensor
```ax
// Schwarzschild metric verification
// Computes Christoffel symbols, Riemann tensor, and Ricci tensor
// for the Schwarzschild spacetime, verifying it's a vacuum solution.
let g = metric(diag(-(1 - 2/r), 1/(1 - 2/r), r^2, r^2 * sin(theta)^2))
let coords = [t, r, theta, phi]
let Gamma = christoffel(g, coords)
let R = riemann(Gamma, coords)
let Ric = ricci(R)
Ric
```

### QM harmonic oscillator
```ax
// Quantum harmonic oscillator
// Creation and annihilation operators (matrix representation for truncated Hilbert space)
// For N-level truncation:
// a|n⟩ = √n |n-1⟩
// a†|n⟩ = √(n+1) |n+1⟩
// Number operator: N = a†a
// Hamiltonian: H = ℏω(N + 1/2)
```

### Calculus demo
```ax
// Calculus Demo
// =============
// Differentiation
diff(x^3 + sin(x), x)
// Integration
integrate(x^2, x)
// Taylor series of e^x around 0
series(exp(x), x, 0, 6)
// Limit
limit(sin(x)/x, x, 0)
// Solve polynomial
solve(x^2 - 5*x + 6, x)
```
