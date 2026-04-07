use serde_json::{json, Value};
use std::sync::LazyLock;

pub struct WorkflowStep {
    pub tool: &'static str,
    pub params_template: Value,
    pub description: &'static str,
    pub output_key: &'static str,
}

pub struct Workflow {
    pub goal: &'static str,
    pub description: &'static str,
    pub steps: Vec<WorkflowStep>,
    pub notes: &'static str,
}

fn step(
    tool: &'static str,
    params_template: Value,
    description: &'static str,
    output_key: &'static str,
) -> WorkflowStep {
    WorkflowStep {
        tool,
        params_template,
        description,
        output_key,
    }
}

fn workflow(
    goal: &'static str,
    description: &'static str,
    steps: Vec<WorkflowStep>,
    notes: &'static str,
) -> Workflow {
    Workflow {
        goal,
        description,
        steps,
        notes,
    }
}

pub static WORKFLOWS: LazyLock<Vec<Workflow>> = LazyLock::new(|| {
    vec![
        workflow(
            "compute_curvature",
            "Compute the Christoffel symbols, Riemann tensor, Ricci tensor, and Ricci scalar for a metric.",
            vec![
                step(
                    "axioma_define_metric",
                    json!({"name": "<metric_name>", "components": "<4x4_matrix>", "coordinates": "<coordinate_list>"}),
                    "Define the spacetime metric with its components and coordinate labels.",
                    "metric_id",
                ),
                step(
                    "axioma_christoffel",
                    json!({"metric_id": "<from_step_1>"}),
                    "Compute the Christoffel symbols of the second kind.",
                    "christoffel_id",
                ),
                step(
                    "axioma_riemann",
                    json!({"christoffel_id": "<from_step_2>"}),
                    "Compute the Riemann curvature tensor.",
                    "riemann_id",
                ),
                step(
                    "axioma_ricci",
                    json!({"riemann_id": "<from_step_3>"}),
                    "Compute the Ricci tensor by contracting the Riemann tensor.",
                    "ricci_id",
                ),
                step(
                    "axioma_scalar_curvature",
                    json!({"ricci_id": "<from_step_4>"}),
                    "Compute the Ricci scalar by contracting the Ricci tensor with the inverse of the stored metric.",
                    "expr_id",
                ),
            ],
            "Provide metric components as a square matrix of strings. Coordinates must match the metric dimension.",
        ),
        workflow(
            "compute_einstein_tensor",
            "Compute the Einstein tensor for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<4x4_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_riemann", json!({"christoffel_id": "<from_step_2>"}), "Compute the Riemann tensor.", "riemann_id"),
                step("axioma_ricci", json!({"riemann_id": "<from_step_3>"}), "Compute the Ricci tensor.", "ricci_id"),
                step("axioma_einstein", json!({"ricci_id": "<from_step_4>", "metric_id": "<from_step_1>"}), "Compute the Einstein tensor from the stored Ricci tensor and metric.", "expr_id"),
            ],
            "This is the standard GR curvature pipeline with the final Einstein-tensor contraction step.",
        ),
        workflow(
            "compute_geodesics",
            "Compute geodesic equations for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<metric_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_geodesic", json!({"christoffel_id": "<from_step_2>"}), "Build the geodesic equations from the Christoffel symbols.", "expr_id"),
            ],
            "Use coordinates in the same order you want the geodesic equations to be returned.",
        ),
        workflow(
            "compute_kretschner",
            "Compute the Kretschmann scalar for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<metric_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_riemann", json!({"christoffel_id": "<from_step_2>"}), "Compute the Riemann tensor.", "riemann_id"),
                step("axioma_kretschner", json!({"riemann_id": "<from_step_3>"}), "Contract the Riemann tensor to obtain the Kretschmann scalar.", "expr_id"),
            ],
            "The tool name follows the existing Axioma spelling `kretschner`.",
        ),
        workflow(
            "compute_weyl",
            "Compute the Weyl tensor for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<metric_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_riemann", json!({"christoffel_id": "<from_step_2>"}), "Compute the Riemann tensor.", "riemann_id"),
                step("axioma_ricci", json!({"riemann_id": "<from_step_3>"}), "Compute the Ricci tensor.", "ricci_id"),
                step("axioma_weyl", json!({"riemann_id": "<from_step_3>"}), "Construct the Weyl tensor from the stored curvature data.", "expr_id"),
            ],
            "Weyl needs the metric, Riemann tensor, and Ricci contraction data from the same background.",
        ),
        workflow(
            "simplify_tensor_expression",
            "Run the standard tensor simplification pipeline on an indexed expression.",
            vec![
                step("axioma_canonicalise", json!({"expr": "<expr_id>"}), "Canonicalise tensor index order using declared symmetries.", "expr_id"),
                step("axioma_rename_dummies", json!({"expr": "<from_step_1>"}), "Rename dummy indices to a canonical naming scheme.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_2>"}), "Collect algebraically identical terms.", "expr_id"),
                step("axioma_meld", json!({"expr": "<from_step_3>"}), "Use tensor symmetries to merge equivalent terms in sums.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_4>"}), "Run general algebraic simplification on the resulting expression.", "expr_id"),
            ],
            "This is the usual first-pass cleanup for tensor identities and derived expressions.",
        ),
        workflow(
            "evaluate_tensor_components",
            "Declare tensor metadata and then evaluate an indexed tensor expression into explicit components.",
            vec![
                step("axioma_declare_indices", json!({"family": "<family_name>", "indices": "<index_symbol_list>", "dimension": "<optional_dimension>"}), "Declare the index family used by the tensor expression.", "status"),
                step("axioma_declare_coordinates", json!({"coordinates": "<coordinate_list>"}), "Declare the active coordinates used by component evaluation.", "status"),
                step("axioma_eval", json!({"code": "<component_rule_list_expression>"}), "Store the component rules as an Axioma list or matrix expression.", "expr_id"),
                step("axioma_evaluate_components", json!({"expr": "<expr_id>", "rules": "<from_step_3>"}), "Evaluate the indexed tensor expression using the supplied component rules.", "expr_id"),
            ],
            "The rule object should be a stored expression like `[[T, [t, t], 1], [T, [x, x], 2]]` before calling `axioma_evaluate_components`.",
        ),
        workflow(
            "contract_indices",
            "Contract metric and delta factors, then clean up dummy naming and term collection.",
            vec![
                step("axioma_eliminate_metric", json!({"expr": "<expr_id>"}), "Use metric factors to raise or lower contracted indices.", "expr_id"),
                step("axioma_eliminate_kronecker", json!({"expr": "<from_step_1>"}), "Contract Kronecker deltas through the expression.", "expr_id"),
                step("axioma_rename_dummies", json!({"expr": "<from_step_2>"}), "Rename dummy indices after contraction.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_3>"}), "Collect the resulting identical terms.", "expr_id"),
            ],
            "This is the standard contraction cleanup sequence after expanding tensor products.",
        ),
        workflow(
            "fierz_rearrangement",
            "Perform a Fierz rearrangement of spinor bilinears and simplify the resulting gamma algebra.",
            vec![
                step("axioma_join_gamma", json!({"expr": "<expr_id>"}), "Join adjacent gamma matrices before the rearrangement.", "expr_id"),
                step("axioma_fierz", json!({"expr": "<from_step_1>"}), "Apply the Fierz rearrangement to the spinor bilinears.", "expr_id"),
                step("axioma_split_gamma", json!({"expr": "<from_step_2>"}), "Split gamma products back into a preferred basis if needed.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_3>"}), "Simplify the resulting sum of bilinears.", "expr_id"),
            ],
            "Use this when the expression already has declared spinor, Dirac-bar, and gamma-matrix properties.",
        ),
        workflow(
            "gamma_algebra",
            "Simplify products of gamma matrices and graded spinor factors.",
            vec![
                step("axioma_join_gamma", json!({"expr": "<expr_id>"}), "Collapse adjacent gamma matrices into antisymmetrised sums.", "expr_id"),
                step("axioma_sort_product", json!({"expr": "<from_step_1>"}), "Sort product factors using SortOrder and commutativity properties.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_2>"}), "Collect identical gamma structures.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_3>"}), "Simplify coefficients and remaining algebraic structure.", "expr_id"),
            ],
            "This sequence is the usual first pass before traces, Fierz moves, or component evaluation.",
        ),
        workflow(
            "compute_ricci_scalar",
            "Compute the Ricci scalar for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<metric_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_riemann", json!({"christoffel_id": "<from_step_2>"}), "Compute the Riemann tensor.", "riemann_id"),
                step("axioma_ricci", json!({"riemann_id": "<from_step_3>"}), "Compute the Ricci tensor.", "ricci_id"),
                step("axioma_scalar_curvature", json!({"ricci_id": "<from_step_4>"}), "Contract the Ricci tensor with the inverse of the stored metric.", "expr_id"),
            ],
            "This is the scalar-only version of the full curvature workflow.",
        ),
        workflow(
            "prove_identity_zero",
            "Try the standard tensor-canonicalisation route for proving an expression vanishes.",
            vec![
                step("axioma_canonicalise", json!({"expr": "<expr_id>"}), "Canonicalise tensor slots and signs.", "expr_id"),
                step("axioma_rename_dummies", json!({"expr": "<from_step_1>"}), "Rename dummy pairs to canonical names.", "expr_id"),
                step("axioma_meld", json!({"expr": "<from_step_2>"}), "Use tensor symmetries to combine equivalent terms.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_3>"}), "Collect the remaining like terms before checking for zero.", "expr_id"),
            ],
            "After the last step, inspect whether the resulting expression is literally zero.",
        ),
        workflow(
            "expand_and_simplify",
            "Expand an algebraic expression and then simplify the result.",
            vec![
                step("axioma_tensor_distribute", json!({"expr": "<expr_id>"}), "Distribute tensor products over sums.", "expr_id"),
                step("axioma_expand", json!({"expr": "<from_step_1>"}), "Expand the resulting algebraic products.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_2>"}), "Collect like terms after expansion.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_3>"}), "Run general simplification on the collected expression.", "expr_id"),
            ],
            "Use this for scalar or tensor expressions when you need a fully expanded intermediate form.",
        ),
        workflow(
            "integration_by_parts",
            "Perform one integration-by-parts step and simplify the result.",
            vec![
                step("axioma_integrate_by_parts", json!({"expr": "<expr_id>", "away": "<symbol>"}), "Apply a one-step integration-by-parts move away from the chosen variable.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_1>"}), "Simplify the resulting terms.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_2>"}), "Collect like terms after simplification.", "expr_id"),
            ],
            "Choose `away` to be the derivative variable you want to move off a particular factor.",
        ),
        workflow(
            "young_decompose",
            "Decompose a tensor product into Young-tableau sectors and simplify the result.",
            vec![
                step("axioma_decompose_product", json!({"expr": "<expr_id>"}), "Decompose the tensor product into tableau sectors.", "expr_id"),
                step("axioma_collect_terms", json!({"expr": "<from_step_1>"}), "Collect repeated tableau sectors.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_2>"}), "Simplify coefficients and remaining algebraic structure.", "expr_id"),
            ],
            "This workflow assumes the relevant Young/projector properties are already declared.",
        ),
        workflow(
            "compute_riemann_tensor",
            "Compute the Riemann tensor for a metric.",
            vec![
                step("axioma_define_metric", json!({"name": "<metric_name>", "components": "<metric_matrix>", "coordinates": "<coordinate_list>"}), "Define the metric.", "metric_id"),
                step("axioma_christoffel", json!({"metric_id": "<from_step_1>"}), "Compute the Christoffel symbols.", "christoffel_id"),
                step("axioma_riemann", json!({"christoffel_id": "<from_step_2>"}), "Compute the Riemann tensor.", "riemann_id"),
            ],
            "Use this when you only need the full curvature tensor and not its contractions.",
        ),
        workflow(
            "normalise_spinor_bilinear",
            "Put a spinor bilinear into a canonical order before further gamma manipulations.",
            vec![
                step("axioma_sort_spinors", json!({"expr": "<expr_id>"}), "Order Majorana bilinears using SortOrder and graded flip signs.", "expr_id"),
                step("axioma_join_gamma", json!({"expr": "<from_step_1>"}), "Join adjacent gamma matrices after the spinors are ordered.", "expr_id"),
                step("axioma_simplify", json!({"expr": "<from_step_2>"}), "Simplify the resulting bilinear expression.", "expr_id"),
            ],
            "This is useful before traces, Fierz steps, or manual matching against basis bilinears.",
        ),
    ]
});

pub fn lookup_workflow(goal: &str) -> Option<&'static Workflow> {
    let query = goal.trim().to_lowercase();
    WORKFLOWS.iter().find(|w| {
        let target = w.goal.to_lowercase();
        target == query
            || query.contains(&target)
            || target.contains(&query)
            || fuzzy_match(&query, &target)
    })
}

pub fn list_workflows() -> Vec<(&'static str, &'static str)> {
    WORKFLOWS
        .iter()
        .map(|w| (w.goal, w.description))
        .collect::<Vec<_>>()
}

fn fuzzy_match(query: &str, target: &str) -> bool {
    let q = query.to_lowercase();
    let t = target.to_lowercase();
    t.split('_').all(|word| q.contains(word))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_lookup_exact_match() {
        let wf = lookup_workflow("compute_curvature").unwrap();
        assert_eq!(wf.steps.len(), 5);
        assert_eq!(wf.steps[0].tool, "axioma_define_metric");
    }

    #[test]
    fn workflow_lookup_fuzzy_match() {
        let wf = lookup_workflow("curvature").unwrap();
        assert_eq!(wf.goal, "compute_curvature");
    }

    #[test]
    fn workflow_lookup_returns_none_for_unknown() {
        assert!(lookup_workflow("quantum_gravity_loop_corrections").is_none());
    }

    #[test]
    fn list_workflows_returns_all() {
        let all = list_workflows();
        assert!(all.len() >= 15);
    }

    #[test]
    fn workflow_steps_have_valid_tool_names() {
        for wf in &*WORKFLOWS {
            for step in &wf.steps {
                assert!(
                    step.tool.starts_with("axioma_"),
                    "tool {} in workflow {} doesn't start with axioma_",
                    step.tool,
                    wf.goal
                );
            }
        }
    }

    #[test]
    fn workflow_steps_reference_registered_tools() {
        let entries = crate::callable_entries();
        for wf in &*WORKFLOWS {
            for step in &wf.steps {
                let name = step.tool.strip_prefix("axioma_").unwrap();
                assert!(
                    entries.iter().any(|entry| entry.name == name),
                    "workflow {} references unregistered tool {}",
                    wf.goal,
                    step.tool
                );
            }
        }
    }
}
