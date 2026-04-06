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
|piecewise(v1, cond1, v2, cond2, ...)|Explicit piecewise constructor.|piecewise(x, x > 0, -x, true)|
|a + b, a - b, a * b, a / b, a ^ b|Arithmetic with standard precedence and right-associative exponentiation.|x + y*z^2|
|-a|Unary negation.|-x^2|
|(expr)|Parenthesized grouping.|(x + 1)^2|
|name(args...)|Function or builtin call syntax.|integrate(x^2, x)|
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
|equiv|equiv(lhs, rhs)|Describe whether two expressions are semantically equivalent.|
|semantic_diff|semantic_diff(lhs, rhs)|Return a semantic-difference descriptor.|

### calculus
|name|signature|description|
|---|---|---|
|dblint|dblint(expr, x, y)|Alias for double_integral.|
|definite_integral|definite_integral(expr, var, a, b)|Compute a definite integral from an antiderivative.|
|defint|defint(expr, var, a, b)|Alias for definite_integral.|
|diff|diff(expr, var)|Differentiate an expression symbolically.|
|double_integral|double_integral(expr, x, y)|Perform iterated double integration.|
|ibp|ibp(expr, u, v, var)|Alias for integrate_by_parts.|
|integrate|integrate(expr, var)|Indefinite or definite symbolic integration.|
|integrate_by_parts|integrate_by_parts(expr, u, v, var)|Perform one integration-by-parts step using explicit u and v'.|
|limit|limit(expr, var, point)|Evaluate a symbolic limit.|
|series|series(expr, var, point, order)|Compute a Taylor series.|
|tplint|tplint(expr, x, y, z)|Alias for triple_integral.|
|triple_integral|triple_integral(expr, x, y, z)|Perform iterated triple integration.|

### codegen
|name|signature|description|
|---|---|---|
|to_cpp|to_cpp(expr)|Print C++ code for an expression.|
|to_python|to_python(expr)|Print Python code for an expression.|
|to_rust|to_rust(expr)|Print Rust code for an expression.|

### complex
|name|signature|description|
|---|---|---|
|Im|Im(z)|Return the imaginary part of a complex expression.|
|Re|Re(z)|Return the real part of a complex expression.|
|arg|arg(z)|Return the complex argument or phase.|
|conj|conj(z)|Return the complex conjugate.|

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
|cos|cos(x)|Cosine with symbolic and numeric evaluation.|
|cosh|cosh(x)|Hyperbolic cosine.|
|cot|cot(x)|Cotangent with symbolic and numeric evaluation.|
|csc|csc(x)|Cosecant with symbolic and numeric evaluation.|
|exp|exp(x)|Exponential function.|
|log|log(x)|Natural logarithm.|
|sec|sec(x)|Secant with symbolic and numeric evaluation.|
|sgn|sgn(x)|Sign function alias.|
|sign|sign(x)|Sign function.|
|sin|sin(x)|Sine with symbolic and numeric evaluation.|
|sinh|sinh(x)|Hyperbolic sine.|
|sqrt|sqrt(x)|Square root with exact perfect-square simplification.|
|tan|tan(x)|Tangent with symbolic and numeric evaluation.|
|tanh|tanh(x)|Hyperbolic tangent.|

### forms
|name|signature|description|
|---|---|---|
|d|d(form)|Alias for exterior_d in the forms subsystem.|
|exterior_d|exterior_d(form)|Exterior derivative of a differential form.|
|hodge_star|hodge_star(form, metric)|Hodge dual of a differential form.|
|wedge_1_1|wedge_1_1(a, b)|Wedge product of two 1-forms.|

### gr
|name|signature|description|
|---|---|---|
|christoffel|christoffel(metric, coords)|Christoffel symbols from a metric.|
|covariant_diff|covariant_diff(expr, metric, coords)|Covariant derivative.|
|einstein|einstein(metric, ricci, scalar)|Einstein tensor.|
|geodesic|geodesic(metric, coords, param)|Geodesic equations for a metric.|
|kretschner|kretschner(metric, riemann)|Kretschmann scalar.|
|lie_derivative|lie_derivative(field, vector, coords)|Lie derivative of a scalar or vector field.|
|metric|metric(diag(...))|Construct a symbolic metric tensor from a diagonal form.|
|ricci|ricci(riemann)|Ricci tensor from the Riemann tensor.|
|ricci_scalar|ricci_scalar(metric, ricci)|Ricci scalar curvature.|
|riemann|riemann(christoffel, coords)|Riemann tensor from a connection.|

### linear-algebra
|name|signature|description|
|---|---|---|
|det|det(matrix)|Determinant of a matrix.|
|diag|diag(a, b, ...)|Construct a diagonal matrix.|
|eigenvalues|eigenvalues(matrix)|Eigenvalues of a small symbolic or numeric matrix.|
|identity|identity(n)|n×n identity matrix.|
|inv|inv(matrix)|Inverse of a matrix.|
|matmul|matmul(a, b)|Matrix multiplication.|
|tensor_product|tensor_product(a, b)|Kronecker or tensor product of arrays or operators.|
|trace_mat|trace_mat(matrix)|Trace of a matrix.|
|transpose|transpose(matrix)|Transpose a matrix.|

### numeric
|name|signature|description|
|---|---|---|
|N|N(expr)|Evaluate an expression numerically when possible.|

### ode
|name|signature|description|
|---|---|---|
|dsolve|dsolve(eq, y, x)|Solve a supported first-order ODE symbolically.|
|first_order_form|first_order_form(ode, dep, indep)|Convert a higher-order ODE to a first-order system.|
|rk4|rk4(f, x, y, x0, y0, x1[, steps])|Numerically integrate an ODE with fourth-order Runge-Kutta.|

### pde
|name|signature|description|
|---|---|---|
|classify_pde|classify_pde(A, B, C)|Classify a second-order PDE as elliptic, parabolic, or hyperbolic.|
|separate_variables|separate_variables(type, x, t[, coeff])|Return a standard separated solution ansatz for a supported PDE family.|
|separation|separation(type, x, t[, coeff])|Alias for separate_variables.|

### plotting
|name|signature|description|
|---|---|---|
|plot|plot(expr, var, xmin, xmax)|Plot a one-dimensional expression to SVG.|

### properties
|name|signature|description|
|---|---|---|
|anti_commuting|anti_commuting(symbol)|Alias for anticommuting.|
|anticommuting|anticommuting(symbol)|Declare an object as anticommuting.|
|antisymmetric|antisymmetric(tensor)|Property marker used in property declarations and metadata.|
|bianchi|bianchi(tensor)|Declare that a tensor satisfies a Bianchi identity.|
|commuting|commuting(symbol)|Declare an object as commuting.|
|covariant_derivative|covariant_derivative(op)|Declare a symbol as a covariant derivative operator.|
|derivative|derivative(op)|Declare a symbol as a derivative operator.|
|dirac_bar|dirac_bar(symbol)|Declare a symbol as a Dirac-bar object.|
|diracbar|diracbar(symbol)|Alias for dirac_bar.|
|epsilon|epsilon(tensor)|Declare a tensor as an epsilon or Levi-Civita tensor.|
|epsilon_tensor|epsilon_tensor(tensor)|Alias for epsilon.|
|gamma_matrix|gamma_matrix(symbol)|Declare a symbol as a gamma matrix.|
|inverse_metric|inverse_metric(tensor)|Declare a tensor as an inverse metric.|
|kronecker|kronecker(tensor)|Alias for kronecker_delta.|
|kronecker_delta|kronecker_delta(tensor)|Declare a tensor as a Kronecker delta.|
|metric|metric(tensor)|Declare a tensor as a metric and attach the symmetric metric property.|
|non_commuting|non_commuting(symbol)|Alias for noncommuting.|
|noncommuting|noncommuting(symbol)|Declare an object as noncommuting.|
|partial_derivative|partial_derivative(op)|Declare a symbol as a partial derivative operator.|
|riemann|riemann(tensor)|Declare Riemann slot symmetries on a tensor.|
|riemann_symmetry|riemann_symmetry(tensor)|Property marker for Riemann-like slot symmetries.|
|satisfies_bianchi|satisfies_bianchi(tensor)|Alias for bianchi.|
|spinor|spinor(tensor)|Declare a tensor as carrying spinor indices.|
|symmetric|symmetric(tensor)|Property marker used in property declarations and metadata.|
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
|braket|braket(bra, ket)|Inner product of a bra and a ket.|
|commutator|commutator(a, b)|Operator commutator [a, b].|
|creation|creation(sym)|Declare or mark a creation operator.|
|density|density(state)|Density matrix of a pure state.|
|gamma|gamma(index)|Dirac gamma matrix for an index.|
|gamma5|gamma5()|Dirac gamma_5 matrix.|
|gamma5_trace|gamma5_trace(expr)|Trace a gamma-chain with gamma_5 inserted.|
|gamma_trace|gamma_trace(expr)|Trace over a chain of gamma matrices.|
|grassmann|grassmann(sym)|Declare a Grassmann-odd symbol.|
|grassmann_simplify|grassmann_simplify(expr)|Simplify using Grassmann anticommutation.|
|join_gamma|join_gamma(expr)|Join adjacent gamma matrices into a compact gamma chain.|
|ket|ket(label)|Construct a ket vector.|
|normal_order|normal_order(expr)|Reorder ladder operators into normal order.|
|outer|outer(ket, bra)|Outer product operator.|
|partial_trace|partial_trace(rho, subsystem)|Partial trace over a subsystem.|
|pauli_x|pauli_x()|Pauli sigma_x matrix.|
|pauli_y|pauli_y()|Pauli sigma_y matrix.|
|pauli_z|pauli_z()|Pauli sigma_z matrix.|
|sigma_x|sigma_x()|Alias for pauli_x.|
|sigma_y|sigma_y()|Alias for pauli_y.|
|sigma_z|sigma_z()|Alias for pauli_z.|
|split_gamma|split_gamma(expr)|Split compact gamma-chain structures into explicit factors.|
|wick|wick(expr)|Expand products using Wick contraction rules.|

### rewrite
|name|signature|description|
|---|---|---|
|rewrite|rewrite(expr)|Apply user-defined rewrite rules to an expression.|
|subs|subs(expr, target, replacement)|Perform symbolic substitution with index-aware matching when needed.|
|take_match|take_match(expr, pattern)|Keep only the parts of a sum that match a pattern.|
|unzoom|unzoom(focus, remainder)|Recombine a focused expression with its remainder.|
|zoom|zoom(expr, pattern)|Split an expression into matching and non-matching parts.|

### simplify
|name|signature|description|
|---|---|---|
|apart|apart(expr, var)|Alias for partial_fractions.|
|expand|expand(expr)|Distribute products and expand small powers.|
|factor_in|factor_in(expr[, targets])|Group terms that share common prefactors.|
|factor_out|factor_out(expr[, targets])|Factor common factors from a sum.|
|partial_fractions|partial_fractions(expr, var)|Decompose a rational function into partial fractions when supported.|
|rationalize|rationalize(expr)|Put sums over a common denominator and cancel common factors.|
|simplify|simplify(expr)|Run the full simplification pipeline.|
|trig_simplify|trig_simplify(expr)|Apply exact trigonometric rewrite rules.|

### syntax
|name|signature|description|
|---|---|---|
|assume|assume(sym, property)|Attach assumptions such as real, positive, integer, even, or odd to a symbol.|
|import|import(path)|Import a standard-library module into the current environment.|

### tensor
|name|signature|description|
|---|---|---|
|antisymmetrise|antisymmetrise(expr, [positions])|Antisymmetrise over listed slots.|
|antisymmetrize|antisymmetrize(expr, [positions])|Alias for antisymmetrise.|
|asym|asym(expr, [positions])|Short alias for antisymmetrise.|
|canonicalise|canonicalise(expr)|Canonicalize tensor indices using declared tensor properties.|
|canonicalize|canonicalize(expr)|Alias for canonicalise.|
|decompose|decompose(expr)|Decompose a tensor into symmetry-adapted pieces.|
|decompose_product|decompose_product(expr)|Decompose a tensor product using known tensor properties.|
|drop_weight|drop_weight(expr, label, value)|Remove terms with a recorded symbolic weight.|
|einsteinify|einsteinify(expr)|Insert implicit Einstein summation contractions.|
|eliminate_kronecker|eliminate_kronecker(expr)|Contract Kronecker deltas through an expression.|
|eliminate_metric|eliminate_metric(expr)|Use the metric or inverse metric to raise or lower contracted indices.|
|eliminate_vielbein|eliminate_vielbein(expr)|Simplify vielbein contractions into metric data when possible.|
|epsilon_to_delta|epsilon_to_delta(expr)|Convert epsilon-tensor contractions into generalized Kronecker deltas.|
|eval_components|eval_components(expr, rules)|Alias for evaluate component expressions.|
|evaluate|evaluate(expr, rules)|Evaluate tensor components using declared component rules.|
|expand_delta|expand_delta(expr)|Expand delta contractions into explicit sums when possible.|
|expand_dummies|expand_dummies(expr)|Expand dummy sums over the declared coordinate set.|
|expand_implicit|expand_implicit(expr)|Expand implicit tensor contractions and index conventions.|
|explicit_indices|explicit_indices(expr)|Make implicit repeated indices explicit.|
|keep_weight|keep_weight(expr, label, value)|Filter terms by a recorded symbolic weight.|
|leibniz|leibniz(expr)|Alias for product_rule.|
|lower_free_indices|lower_free_indices(expr)|Lower free upper indices using the active metric family.|
|lower_indices|lower_indices(expr)|Alias for lower_free_indices.|
|meld|meld(expr)|Detect multi-term tensor identities using Young projection and linear dependence.|
|product_rule|product_rule(expr)|Apply the Leibniz rule to an indexed product.|
|raise_free_indices|raise_free_indices(expr)|Raise free lower indices using the active inverse metric family.|
|raise_indices|raise_indices(expr)|Alias for raise_free_indices.|
|reduce_delta|reduce_delta(expr)|Simplify explicit delta-expanded expressions back to compact form.|
|rename_dummies|rename_dummies(expr)|Rename dummy indices to a canonical fresh naming scheme.|
|rewrite_indices|rewrite_indices(expr)|Rewrite index names while preserving variance and families.|
|sort_product|sort_product(expr)|Sort tensor products using symmetry-aware canonicalization.|
|split_index|split_index(expr, old, [new...])|Split one abstract index family into several fixed values.|
|sym|sym(expr, [positions])|Short alias for symmetrise.|
|symmetrise|symmetrise(expr, [positions])|Symmetrise over listed slots.|
|symmetrize|symmetrize(expr, [positions])|Alias for symmetrise.|
|tdistribute|tdistribute(expr)|Alias for tensor_distribute.|
|tensor_distribute|tensor_distribute(expr)|Distribute products over sums in tensor expressions.|
|unwrap|unwrap(expr)|Flatten nested additive and multiplicative structure.|
|young_project|young_project(expr)|Project a tensor onto a declared Young-tableau symmetry.|

### units
|name|signature|description|
|---|---|---|
|check_units|check_units(expr)|Verify that a units expression is dimensionally consistent.|
|convert|convert(expr, units)|Convert an expression between compatible units.|
|dim|dim(expr)|Return the dimension of a units-aware expression.|

### variational
|name|signature|description|
|---|---|---|
|euler_lagrange|euler_lagrange(L, field, coords)|Compute Euler-Lagrange equations.|
|vary|vary(expr, field)|Take a formal variation with respect to a field.|

### vector-calculus
|name|signature|description|
|---|---|---|
|curl|curl([Fx, Fy, Fz], [x, y, z])|Return the three-dimensional curl.|
|div|div([Fx, Fy, Fz], [x, y, z])|Alias for divergence.|
|divergence|divergence([Fx, Fy, Fz], [x, y, z])|Return the divergence of a vector field.|
|grad|grad(f, [x, y, z])|Alias for gradient.|
|gradient|gradient(f, [x, y, z])|Return the gradient vector.|
|hessian|hessian(f, [x1, ...])|Return the Hessian matrix.|
|jacobian|jacobian([f1, ...], [x1, ...])|Return the Jacobian matrix.|
|laplacian|laplacian(f, [x, y, z])|Return the Laplacian.|

## Tensor Properties
|property|syntax|description|enables|
|---|---|---|---|
|Symmetric|property T symmetric([positions])|Indices are symmetric under exchange of the listed slots.|build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup|
|AntiSymmetric|property T antisymmetric([positions])|Indices are antisymmetric under exchange of the listed slots.|build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup|
|RiemannSymmetry|property R riemann_symmetry|Apply the standard pair antisymmetry and pair-exchange symmetry of a Riemann tensor.|build_generating_set, canonicalize_indices, tableaux_from_properties, handle_factor symmetry lookup|
|Traceless|property T traceless|Marks a tensor as traceless.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|Metric|property g metric|Marks a tensor as a metric used to lower indices and define dummy-pair symmetry.|lower_free_indices, eliminate_metric, metric_symmetry_for_slots|
|InverseMetric|property g inverse_metric|Marks a tensor as an inverse metric used to raise indices and define dummy-pair symmetry.|raise_free_indices, eliminate_metric, metric_symmetry_for_slots|
|KroneckerDelta|property d kronecker_delta|Marks a tensor as a Kronecker delta.|eliminate_kronecker, expand_delta, reduce_delta|
|EpsilonTensor|property eps epsilon_tensor|Marks a tensor as a Levi-Civita epsilon tensor.|epsilon_to_delta, handle_epsilon component evaluation|
|Derivative|property D derivative|Marks a symbol as a derivative operator.|stored by ax-tensor metadata; derivative handling currently keys off names instead|
|PartialDerivative|property D partial_derivative|Marks a symbol as a partial derivative operator.|stored by ax-tensor metadata; derivative handling currently keys off names instead|
|CovariantDerivative|property nabla covariant_derivative|Marks a symbol as a covariant derivative operator.|stored by ax-tensor metadata; derivative handling currently keys off names instead|
|Depends|property T depends([x, y, ...])|Declares that a tensor depends on listed symbols.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|Spinor|property psi spinor|Marks a tensor as carrying spinor indices.|canonicalise_product dummy classification via metric_symmetry_for_slots|
|DiracBar|property psibar dirac_bar|Marks a symbol as a Dirac-bar object.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|GammaMatrixProp|property gamma gamma_matrix|Marks a symbol as a gamma matrix.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|Commuting|property A commuting|Marks an object as commuting.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|AntiCommuting|property psi anticommuting|Marks an object as anticommuting.|canonicalise_product dummy classification via metric_symmetry_for_slots|
|NonCommuting|property A noncommuting|Marks an object as noncommuting.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|SortOrder|property T sort_order([...])|Declares an explicit preferred order of symbols.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|TableauSymmetry|property T tableau_symmetry(shape=[...], indices=[...])|Declares a Young-tableau symmetry shape and slot assignment.|young_project_tensor|
|SatisfiesBianchi|property R satisfies_bianchi|Marks a tensor as satisfying a Bianchi identity.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|WeylTensor|property C weyl_tensor|Marks a tensor as a Weyl tensor.|stored by ax-tensor metadata; no direct ax-tensor algorithm currently consults it|
|DifferentialFormDegree|property F differential_form_degree(n)|Declares the degree of a differential form.|stored by ax-tensor metadata; differential-form algorithms live outside ax-tensor|

## Conventions
|field|options|default|description|
|---|---|---|---|
|metric_signature|MostlyPlus, MostlyMinus|MostlyPlus|Chooses the sign convention for the metric signature.|
|riemann_sign|MTW, Weinberg|MTW|Chooses the sign convention for the Riemann tensor definition.|
|ricci_contraction|FirstThird, FirstFourth|FirstThird|Chooses which Riemann slots are contracted to form the Ricci tensor.|
|levi_civita_norm|PlusOne, MinusOne, SqrtG|PlusOne|Chooses the normalization convention for the Levi-Civita tensor.|
|fourier_sign|MinusI, PlusI|MinusI|Chooses the sign convention in Fourier-transform exponentials.|

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
|differentiate|differentiate(expr, var)|The differentiation variable must be a symbol.|Take a symbolic derivative with chain, product, and builtin function rules.|

### forms
|name|signature|preconditions|description|
|---|---|---|---|
|exterior_derivative|exterior_derivative(form, coords)|form.dim must equal coords.len().|Take the exterior derivative of a differential form by differentiating components and wedging in basis one-forms.|
|hodge_dual|hodge_dual(form, g)|The metric dimension must equal the form dimension; the implementation uses the symbolic inverse and determinant of g.|Take the Hodge dual of a differential form with respect to a symbolic metric.|
|wedge|wedge(a, b)|Both forms must have the same ambient dimension.|Compute the antisymmetric wedge product of two differential forms.|

### gr
|name|signature|preconditions|description|
|---|---|---|---|
|christoffel_from_metric|christoffel_from_metric(g, coords)|The metric must be square and coords.len() must equal g.dim; the routine uses the symbolic inverse of g.|Compute Christoffel symbols from a symbolic metric by the standard Levi-Civita formula.|
|covariant_derivative_covector|covariant_derivative_covector(w, gamma, coord_index, coords)|The covector length, connection dimensions, and coordinate list length must agree.|Compute ∇_coord_index w for a covector field.|
|covariant_derivative_tensor2|covariant_derivative_tensor2(t, gamma, coord_index, coords)|Tensor dimensions, connection dimensions, and coordinate count must agree.|Compute the covariant derivative of a rank-2 covariant tensor.|
|covariant_derivative_vector|covariant_derivative_vector(v, gamma, coord_index, coords)|The vector length, connection dimensions, and coordinate list length must agree.|Compute ∇_coord_index v for a contravariant vector field.|
|einstein_tensor|einstein_tensor(ricci, scalar, g)|The metric dimension must match the Ricci tensor dimensions.|Build the Einstein tensor G_ab = R_ab - 1/2 g_ab R.|
|geodesic_equations|geodesic_equations(gamma, coords)|Connection dimensions must match the coordinate list.|Construct the geodesic equations ẍ^i = -Γ^i_jk ẋ^j ẋ^k in symbolic form.|
|kretschner_scalar|kretschner_scalar(riemann, g)|The metric must be invertible; this implementation contracts using diagonal entries of g and g^{-1}.|Compute a diagonal-metric approximation to the Kretschmann scalar from the squared Riemann components.|
|lie_derivative_scalar|lie_derivative_scalar(f, v, coords)|The vector field length must match the coordinate list length.|Compute the Lie derivative of a scalar along a vector field.|
|lie_derivative_vector|lie_derivative_vector(w, v, coords)|Both vectors must have the same length as coords.|Compute the Lie derivative of a vector field along another vector field.|
|ricci_from_riemann|ricci_from_riemann(riemann, n, convention)|n must match the tensor dimensions; the Convention selects first-third or first-fourth contraction.|Contract a Riemann tensor into the Ricci tensor using the configured Ricci-contraction convention.|
|ricci_scalar|ricci_scalar(ricci, ginv)|The inverse metric dimension must match the Ricci tensor dimensions.|Contract the Ricci tensor with the inverse metric to obtain the scalar curvature.|
|riemann_from_christoffel|riemann_from_christoffel(gamma, coords, convention)|The connection array dimensions must match coords.len(); the Convention determines MTW versus Weinberg sign.|Compute the Riemann tensor from Christoffel symbols, respecting the active sign convention.|

### linalg
|name|signature|preconditions|description|
|---|---|---|---|
|determinant|determinant(matrix)|The matrix should be square; symbolic simplification is applied recursively by minors.|Compute the determinant of a symbolic square matrix.|
|eigenvalues_symbolic|eigenvalues_symbolic(matrix)|The matrix should be square; solving that polynomial is a separate step.|Return the characteristic polynomial det(A - lambda I) for a symbolic matrix.|
|inverse|inverse(matrix)|The matrix must be square and have nonzero determinant.|Compute the symbolic inverse of a square matrix by adjugate over determinant.|
|tensor_product|tensor_product(a, b)|Both inputs must be rectangular matrices.|Compute the Kronecker product of two matrices.|
|trace|trace(matrix)|The matrix should be square.|Compute the trace of a square matrix.|

### ode
|name|signature|preconditions|description|
|---|---|---|---|
|classify_pde|classify_pde(a, b, c)|The discriminant must simplify to a numeric sign to get a definite classification; otherwise the result is Unknown.|Classify a second-order PDE from its A, B, C coefficients via the discriminant B^2 - A*C.|
|first_order_form|first_order_form(ode, dependent_var, independent_var, interner) -> Vec<(Expr, Expr)|The ODE should contain nested diff calls with respect to independent_var, or else it is treated as the right-hand side of a second-order equation.|Convert a higher-order ODE into a first-order system by introducing auxiliary derivative variables.|
|rk4|rk4(f, x_sym, y_sym, x0, y0, x_end, n_steps, interner) -> Vec<(f64, f64)|f must evaluate numerically for the supplied bindings, and n_steps must be nonzero.|Numerically integrate a scalar first-order ODE y' = f(x, y) with fourth-order Runge-Kutta.|
|rk4_system|rk4_system(fs, x_sym, y_syms, x0, y0s, x_end, n_steps)|The numbers of equations, dependent variables, and initial values must match, and each expression must evaluate numerically.|Numerically integrate a coupled first-order ODE system with fourth-order Runge-Kutta.|
|separate_variables|separate_variables(pde_type, spatial_var, temporal_var, coefficient)|This is a template generator for standard wave, heat, and Laplace-type equations rather than an automatic PDE parser.|Return a standard separated-variables ansatz for hyperbolic, parabolic, or elliptic PDE families.|
|solve_ode|solve_ode(equation, y_sym, x_sym)|The ODE right-hand side must match one of the supported separable or linear forms; otherwise an unevaluated solve_ode call is returned.|Solve simple separable or first-order linear ODEs symbolically.|

### qm
|name|signature|preconditions|description|
|---|---|---|---|
|anticommutator|anticommutator(a, b)|The matrices must be dimensionally compatible for multiplication.|Compute the matrix anticommutator AB + BA.|
|braket|braket(bra, ket)|The two vectors should have the same length.|Compute the inner product of a bra and ket by componentwise contraction.|
|commutator|commutator(a, b)|The matrices must be dimensionally compatible for multiplication.|Compute the matrix commutator AB - BA.|
|density_matrix|density_matrix(state)|The state should be given as a finite component vector.|Build the rank-one density matrix \|psi><psi\| from a state vector.|
|fierz|fierz(expr, dim)|The routine currently produces the abstract gamma_basis expansion rather than fully rearranging a concrete spinor expression.|Return the formal Fierz-basis expansion coefficients for the given spacetime dimension.|
|gamma5|gamma5()|No extra setup is required.|Return the standard Dirac gamma_5 matrix.|
|gamma_trace|gamma_trace(indices, metric)|The input must already be parsed into GammaEntry values; the implementation assumes the standard four-dimensional Dirac trace normalization.|Trace a gamma-matrix chain, including the special gamma5 epsilon-tensor case.|
|grassmann_simplify|grassmann_simplify(expr, gradings)|Grassmann or operator gradings must be present in the environment.|Simplify products of commuting and anticommuting symbols using the stored gradings.|
|join_gammas_in_expr|join_gammas_in_expr(expr)|Gamma factors must be represented as Call(gamma_sym, [...]) nodes and use a compatible metric symbol.|Join adjacent gamma-matrix factors into antisymmetrized multi-index gamma objects plus metric contractions.|
|normal_order|normal_order(expr)|Operator kinds must be declared for the symbols that should reorder.|Reorder products of operators into normal order using the declared creation/annihilation kinds.|
|outer|outer(ket, bra)|The two vectors should have finite explicit components.|Build the outer-product operator \|ket><bra\| from two vectors.|
|partial_trace|partial_trace(rho, dim_a, dim_b, trace_over)|rho must be arranged as a (dim_a*dim_b) square matrix, and trace_over must be 'A' or 'B'.|Trace out subsystem A or B from a bipartite density matrix.|
|pauli_x|pauli_x()|No extra setup is required.|Return the Pauli sigma_x matrix.|
|pauli_y|pauli_y()|No extra setup is required.|Return the Pauli sigma_y matrix.|
|pauli_z|pauli_z()|No extra setup is required.|Return the Pauli sigma_z matrix.|
|split_gamma|split_gamma(expr, on_back)|The input must contain gamma_sym calls with more than one index.|Split a multi-index antisymmetric gamma matrix into a shorter chain plus contraction terms.|
|wick_expand|wick_expand(expr)|Operator kinds and any nonzero contraction values must be provided explicitly.|Expand operator products into normal-ordered terms plus single contractions.|

### rewrite
|name|signature|preconditions|description|
|---|---|---|---|
|describe_rewrite_trace|describe_rewrite_trace(trace)|A trace from rewrite_with_trace is required.|Render a human-readable summary of a rewrite trace.|
|match_tensor_pattern|match_tensor_pattern(pattern, expr)|Index-family information in env improves matching across renamed abstract indices.|Match indexed tensor patterns using variance and index-family compatibility rather than literal index names.|
|multi_substitute|multi_substitute(expr, substitutions)|Targets are applied structurally rather than by solving matching ambiguities.|Apply several exact substitutions in one pass.|
|rewrite_with_trace|rewrite_with_trace(expr, interner) -> (Expr, Vec<RewriteStep>)|Rewrite rules must be registered in the environment.|Apply registered rewrite rules and return both the rewritten expression and a trace of the applied rules.|
|substitute_with_indices|substitute_with_indices(expr, target, replacement)|Use when the expression or rule contains indexed tensors.|Perform substitution while renaming bound dummy indices to avoid capture and preserving index-family matches.|
|symbolic_substitute|symbolic_substitute(expr, target, replacement)|Best suited to scalar expressions without tensor-index matching requirements.|Replace exact symbolic subexpressions recursively.|
|take_match|take_match(expr, pattern)|Most useful on additive expressions.|Keep only the subterms of a sum that match a pattern.|
|unzoom|unzoom(focus, remainder)|The focus and remainder should come from a compatible zoom step.|Recombine a focused expression with its saved remainder.|
|zoom|zoom(expr, pattern, interner) -> (Expr, Expr)|Most useful on additive expressions.|Split a sum into matching and nonmatching parts with respect to a pattern.|

### solve
|name|signature|preconditions|description|
|---|---|---|---|
|solve|solve(equation, var)|The equation must reduce to a polynomial in var; otherwise the function returns an unevaluated solve call.|Solve a univariate polynomial equation when its coefficients can be extracted.|
|solve_linear_system|solve_linear_system(equations, vars, interner) -> Option<Vec<(Spur, Expr)|Every equation must be linear in the listed variables, and the system must have a unique consistent solution.|Solve a linear system over exact rationals by Gaussian elimination.|

### syntax
|name|signature|preconditions|description|
|---|---|---|---|
|eval|eval(expr)|Environment declarations, rules, coordinates, and tensor properties affect the result.|Evaluate an expression by dispatching builtins, rewrite rules, declarations, and symbolic simplifications.|
|resolve_import|resolve_import(path)|The imported module must exist under std/ or another supported search root.|Resolve a std-module import path to the corresponding .ax file on disk.|

### tensor
|name|signature|preconditions|description|
|---|---|---|---|
|canonicalise|canonicalise(expr)|Tensor symmetries must be present in tensor_properties for anything beyond lexicographic index ordering.|Canonicalize tensor monomials and sums using declared slot symmetries and dummy-index canonicalization.|
|canonicalize_indices|canonicalize_indices(expr)|Useful symmetry properties such as Symmetric, AntiSymmetric, or RiemannSymmetry must be declared on tensor symbols.|Apply local index-slot canonicalization from declared tensor symmetries before product-level canonicalization.|
|complete_inverse_metric|complete_inverse_metric(metric_rules, inv_metric_sym, coordinates)|Metric component rules must define an invertible square metric over the supplied coordinate list.|Construct inverse-metric component rules from metric component rules by symbolic matrix inversion.|
|compute_weight|compute_weight(expr)|Weight assignments should be declared for the participating symbols.|Compute the total symbolic weight of an expression under a chosen label.|
|decompose|decompose(expr, basis)|The basis should span the intended subspace, and tensor_properties should contain the symmetries needed for canonical matching.|Express a tensor expression as a rational linear combination of a supplied canonical basis plus any residual unmatched terms.|
|decompose_product|decompose_product(expr, dim)|The input should be a product of exactly two rank-2 indexed tensors.|Decompose a rank-2 tensor product into symmetric, antisymmetric, and trace metric-built pieces.|
|diff_component|diff_component(expr, var)|The variable should be a coordinate or scalar symbol.|Differentiate a component expression with tensor-aware fallback handling.|
|drop_weight|drop_weight(expr, target_weight)|Weight assignments for symbols under the chosen label must be present when nonzero weights are required.|Remove terms whose computed symbolic weight equals target_weight.|
|einsteinify|einsteinify(expr)|Useful on products where the same abstract index appears twice with both slots up or both slots down.|Fix repeated-index pairs that have the same variance by flipping one slot so Einstein summation becomes well-formed.|
|eliminate_kronecker|eliminate_kronecker(expr)|The delta symbol must identify a two-index Kronecker delta with one up and one down slot.|Use Kronecker deltas to substitute contracted indices and remove delta factors from products.|
|eliminate_metric|eliminate_metric(expr, inv_metric_sym)|Metric components must use two down indices and inverse-metric components two up indices.|Use metric or inverse-metric factors to raise or lower contracted indices and remove those metric factors.|
|eliminate_vielbein|eliminate_vielbein(expr, vielbein_sym, inv_vielbein_sym)|Vielbein factors must appear as indexed two-slot tensors with one contractible index matching another factor.|Use vielbein or inverse-vielbein factors to convert contracted indices between two families and remove the conversion factors.|
|epsilon_to_delta|epsilon_to_delta(expr, dim)|The epsilon symbol and target delta symbol must be provided, and epsilon factors must carry exactly dim indices.|Rewrite products of epsilon tensors into factorial factors times generalized Kronecker deltas.|
|evaluate_components|evaluate_components(expr, rules, index_values)|Concrete component rules and coordinates must be available through the evaluation environment; tensor_properties are used for symmetry and epsilon handling.|Evaluate tensor expressions into explicit component expressions, including dummy summations, derivative handling, symmetry-aware lookups, and epsilon components.|
|evaluate_components_v2|evaluate_components_v2(expr, rules)|Component rules, coordinates, and tensor properties must be available through env.|Evaluate tensor components with the newer handler-based evaluation pipeline.|
|expand_delta|expand_delta(expr)|The delta symbol must identify an indexed tensor with an even number of slots split into equal up/down sets.|Expand a generalized Kronecker delta into a signed sum of ordinary two-index deltas.|
|expand_dummies|expand_dummies(expr, coordinates)|A coordinate list must be supplied; abstract dummy names not already in that list are expanded.|Replace each dummy index pair by an explicit sum over the supplied coordinate labels.|
|expand_implicit|expand_implicit(expr, implicit_index_tensors, available_indices, n_indices_per_tensor)|Implicit-index tensor names and their slot counts must be declared; disjoint fresh indices should be available.|Recursively make implicit tensor contractions explicit across sums, products, and call arguments.|
|explicit_indices|explicit_indices(expr, implicit_index_tensors, available_indices, n_indices_per_tensor)|Tensor names that should receive implicit indices must be listed, and the current implementation only expands the common two-index case.|Insert explicit matrix-style indices for implicit-index tensors inside products.|
|integrate_by_parts|integrate_by_parts(expr, away_from, derivative_syms)|The expression should contain a derivative operator from derivative_syms acting on a factor that contains away_from; boundary terms are assumed to vanish.|Perform one integration-by-parts rewrite by moving a derivative off the factor containing away_from.|
|keep_weight|keep_weight(expr, target_weight)|Weight assignments for symbols under the chosen label must be present when nonzero weights are required.|Keep only the terms whose computed symbolic weight equals target_weight.|
|lower_free_indices|lower_free_indices(expr)|Index families should be declared when only some families are free-position indices; otherwise all singly-occurring upper indices are lowered.|Flip free upper indices to lower variance without inserting an explicit metric.|
|meld|meld(expr)|Best results require symmetry properties such as Symmetric, AntiSymmetric, RiemannSymmetry, or TableauSymmetry on the factors involved.|Detect multi-term tensor cancellations by canonicalization, Young projection, and rational linear dependence testing.|
|product_rule|product_rule(expr, derivative_syms)|The derivative operator symbols must be listed in derivative_syms.|Expand derivative operators over products and sums using the Leibniz rule.|
|raise_free_indices|raise_free_indices(expr)|Index families should be declared when only some families are free-position indices; otherwise all singly-occurring lower indices are raised.|Flip free lower indices to upper variance without inserting an explicit inverse metric.|
|reduce_delta|reduce_delta(expr)|The delta symbol and the symbol representing the dimension must be supplied.|Iteratively contract products and traces of Kronecker deltas back to simpler delta or dimension factors.|
|rename_dummies|rename_dummies(expr)|Index-family data improves the generated names; without it, generic _dN names are used.|Rename dummy indices to deterministic family-aware placeholders so alpha-equivalent contractions compare equal.|
|rename_dummy_indices|rename_dummy_indices(expr, prefix)|Useful when preparing expressions for display or comparison.|Rename repeated contracted indices to fresh deterministic names with the chosen prefix.|
|rewrite_indices|rewrite_indices(expr, target_tensors, inv_metric_sym)|Each target tensor must have a full desired-variance specification per slot, and metric symbols must be provided.|Insert metric or inverse-metric factors so selected tensors end up with requested slot variances.|
|sort_product|sort_product(expr)|No special setup is required; tensor_properties is currently not consulted by the sorter.|Sort multiplicative factors into a deterministic order for tensor expressions.|
|split_index|split_index(expr, parent_indices, sub1_indices, sub2_indices)|The parent index names to split must be listed, and each target subfamily list should be non-empty if it is intended to contribute terms.|Replace occurrences of a parent index family by sums over two sub-families.|
|symmetrise|symmetrise(expr, positions, antisymmetric)|The listed positions must refer to valid index slots in the target indexed factor or flattened product ordering.|Symmetrize or antisymmetrize an expression over specific index slots by averaging over permutations.|
|tensor_distribute|tensor_distribute(expr)|No extra setup is required.|Distribute tensor products over sums, including sums that appear in indexed bases.|
|unwrap_derivatives|unwrap_derivatives(expr, derivative_syms, depends)|Derivative symbols must be listed explicitly, and dependence information should be populated for symbols that are not constant.|Pull factors that do not depend on the differentiation variables outside derivative operators, and kill derivatives of constants.|
|young_project|young_project(expr, tableau)|A valid tableau cell layout must be supplied in slot-number form.|Project an expression with a specific Young tableau by antisymmetrizing columns and symmetrizing rows.|
|young_project_tensor|young_project_tensor(expr)|The relevant tensor symbol must carry a TableauSymmetry property in tensor_properties.|Apply a declared TableauSymmetry property directly to a tensor expression.|

### variational
|name|signature|preconditions|description|
|---|---|---|---|
|euler_lagrange_system|euler_lagrange_system(lagrangian, fields, coords)|Each field entry must provide derivative symbols aligned with coords.|Compute the Euler-Lagrange equations for several fields at once.|
|functional_derivative|functional_derivative(lagrangian, field, field_derivs, coords)|field_derivs and coords should be aligned so each derivative symbol corresponds to differentiation with respect to the matching coordinate.|Compute the Euler-Lagrange functional derivative δL/δfield for first-derivative Lagrangians.|
|vary_action|vary_action(lagrangian, field, variation, field_derivs, variation_derivs)|field_derivs and variation_derivs must be aligned term-by-term.|Form the first variation of an action density before integrating by parts.|

## Standard Library
|module|description|provides|
|---|---|---|
|algebra|Notes the standard algebra operations used for expansion and simplification.|documentation comments only|
|calculus|Documents the standard calculus builtins for differentiation, integration, series, and limits.|documentation comments only|
|conventions/landau|Sets Landau-Lifshitz sign and curvature conventions.|convention metric_signature mostly_minus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|conventions/mtw|Sets Misner-Thorne-Wheeler general-relativity conventions.|convention metric_signature mostly_plus, convention riemann_sign mtw, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|conventions/particle_physics|Sets particle-physics sign conventions.|convention metric_signature mostly_plus, convention riemann_sign mtw, convention fourier_sign minus_i|
|conventions/weinberg|Sets Weinberg general-relativity conventions.|convention metric_signature mostly_plus, convention riemann_sign weinberg, convention ricci_contraction first_third, convention levi_civita_norm plus_one|
|gr/de_sitter|Builds the de Sitter metric and its Christoffel symbols in static coordinates.|let f, let g, let coords, let Gamma|
|gr/frw|Builds a flat FRW metric with symbolic scale factor and computes Christoffel symbols.|let g, let coords, let Gamma|
|gr/kerr_newman|Defines symbolic Kerr-Newman metric component expressions in Boyer-Lindquist coordinates.|let Sigma_expr, let Delta_expr, let g_tt, let g_rr, let g_theta_theta, let g_phi_phi, let g_t_phi|
|gr/minkowski|Builds flat Minkowski spacetime and its vanishing Christoffel symbols.|let g, let coords, let Gamma|
|gr/schwarzschild|Builds the Schwarzschild metric, Christoffel symbols, Riemann tensor, and Ricci tensor.|let g, let coords, let Gamma, let R, let Ric|
|physics/classical_mechanics|Notes the intended Euler-Lagrange workflow for classical mechanics.|documentation comments only|
|physics/klein_gordon|Sets up a Klein-Gordon Lagrangian and computes its Euler-Lagrange equation.|let dphi_dt, let dphi_dx, let dphi_dy, let dphi_dz, let L, let EOM|
|physics/maxwell|Placeholder notes for Maxwell theory setup from a Lagrangian.|documentation comments only|
|qft/dirac|Summarizes the Dirac equation and basic gamma-trace identities.|documentation comments only|
|qft/gamma|Introduces symbolic gamma and eta objects for gamma-matrix algebra experiments.|let gamma, let eta|
|qft/normal_ordering|Documents normal ordering and Wick expansion usage.|documentation comments only|
|qft/scalar_field|Summarizes the free scalar-field Lagrangian and Klein-Gordon equation.|documentation comments only|
|qm/bell|Constructs a Bell state, its density matrix, and a reduced density matrix by partial trace.|let up, let down, let phi_plus, let rho, let rho_A|
|qm/harmonic_oscillator|Documents the intended harmonic-oscillator operator setup.|documentation comments only|
|qm/spin|Builds Pauli matrices and their commutator as a spin-1/2 algebra example.|let sigma_x, let sigma_y, let sigma_z, let comm_xy|
|tensor/index|Documents index notation and contraction conventions for tensors.|documentation comments only|
|tensor/symmetry|Documents tensor-symmetry declarations and examples.|documentation comments only|
|trig|Defines standard exact trigonometric rewrite rules.|rule pythag, rule pythag_alt1, rule pythag_alt2, rule double_sin, rule double_cos|
|units/cgs|Documents the CGS unit system and derived units.|documentation comments only|
|units/natural|Documents the natural-unit system convention.|documentation comments only|
|units/si|Documents the SI unit system import and usage.|documentation comments only|

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

### QM spin algebra
```ax
// Spin-1/2 algebra
let sigma_x = pauli_x()
let sigma_y = pauli_y()
let sigma_z = pauli_z()
// Verify Pauli algebra: [sigma_i, sigma_j] = 2i * epsilon_ijk * sigma_k
let comm_xy = commutator(sigma_x, sigma_y)
// Should equal 2i * sigma_z
comm_xy
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
