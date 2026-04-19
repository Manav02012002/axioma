use ax_eval::{callable_entries, Env, EvalState};
use ax_ir::{Expr, Interner};
use std::collections::HashMap;

struct TestState {
    interner: Interner,
    env: Env,
    exprs: HashMap<String, Expr>,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            env: Env::new(),
            exprs: HashMap::new(),
        }
    }
}

impl EvalState for TestState {
    fn interner(&self) -> &Interner {
        &self.interner
    }

    fn interner_mut(&mut self) -> &mut Interner {
        &mut self.interner
    }

    fn env(&self) -> &Env {
        &self.env
    }

    fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    fn store_expr(&mut self, expr: Expr) -> String {
        let id = format!("expr{}", self.exprs.len());
        self.exprs.insert(id.clone(), expr);
        id
    }

    fn get_expr(&self, id: &str) -> Option<&Expr> {
        self.exprs.get(id)
    }

    fn parse_code(&mut self, code: &str) -> Result<Expr, String> {
        Err(format!(
            "parse_code is unavailable in TestState for input: {code}"
        ))
    }

    fn render_latex(&self, expr: &Expr) -> String {
        ax_ir::pretty_print(expr, &self.interner)
    }

    fn render_unicode(&self, expr: &Expr) -> String {
        ax_ir::pretty_print(expr, &self.interner)
    }

    fn get_metric(&self, _id: &str) -> Option<&(ax_tensor::SymbolicMatrix, Vec<lasso::Spur>)> {
        None
    }

    fn store_metric(
        &mut self,
        _id: String,
        _metric: ax_tensor::SymbolicMatrix,
        _coords: Vec<lasso::Spur>,
    ) {
    }

    fn get_christoffel(&self, _id: &str) -> Option<&Vec<Vec<Vec<Expr>>>> {
        None
    }

    fn store_christoffel(&mut self, _id: String, _chris: Vec<Vec<Vec<Expr>>>) {}

    fn get_riemann(&self, _id: &str) -> Option<&Vec<Vec<Vec<Vec<Expr>>>>> {
        None
    }

    fn store_riemann(&mut self, _id: String, _riem: Vec<Vec<Vec<Vec<Expr>>>>) {}

    fn get_ricci(&self, _id: &str) -> Option<&Vec<Vec<Expr>>> {
        None
    }

    fn store_ricci(&mut self, _id: String, _ric: Vec<Vec<Expr>>) {}

    fn get_matrix_data(&self, _id: &str) -> Option<Vec<Vec<Expr>>> {
        None
    }
}

fn call_registry(
    state: &mut TestState,
    name: &str,
    args: Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let entry = callable_entries()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("missing callable entry for {name}"));
    (entry.handler)(&args, state)
}

#[test]
fn lindblad_steady_state_amplitude_damping_returns_ground_state() {
    let mut state = TestState::default();
    let h_id = state.store_expr(Expr::Matrix(vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ]));
    let jumps_id = state.store_expr(Expr::List(vec![Expr::Matrix(vec![
        vec![Expr::zero(), Expr::one()],
        vec![Expr::zero(), Expr::zero()],
    ])]));

    let response = call_registry(
        &mut state,
        "lindblad_steady_state",
        vec![serde_json::json!(h_id), serde_json::json!(jumps_id)],
    )
    .expect("steady-state solve should succeed");

    let expr_id = response["expr_id"]
        .as_str()
        .expect("response should contain expr id");
    assert_eq!(
        state.get_expr(expr_id),
        Some(&Expr::Matrix(vec![
            vec![Expr::one(), Expr::zero()],
            vec![Expr::zero(), Expr::zero()],
        ]))
    );
}

#[test]
fn lindblad_steady_state_zero_generator_reports_non_unique_error() {
    let mut state = TestState::default();
    let h_id = state.store_expr(Expr::Matrix(vec![
        vec![Expr::zero(), Expr::zero()],
        vec![Expr::zero(), Expr::zero()],
    ]));
    let jumps_id = state.store_expr(Expr::List(vec![]));

    let err = call_registry(
        &mut state,
        "lindblad_steady_state",
        vec![serde_json::json!(h_id), serde_json::json!(jumps_id)],
    )
    .expect_err("zero generator should be underdetermined");

    assert_eq!(
        err,
        "lindblad_steady_state generator has non-unique steady states"
    );
}
