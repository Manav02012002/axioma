pub fn render_oracle_trace(trace: &ax_trace::OracleCaseTrace) -> String {
    [
        format!("case={}", trace.case_name),
        format!("kind={}", trace.kind),
        format!("passed={}", trace.passed),
        format!("expected={}", trace.expected),
        format!("actual={}", trace.actual),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendered_oracle_trace_has_required_lines() {
        let trace = ax_trace::OracleCaseTrace {
            case_name: "case_a".into(),
            kind: "canonicalize".into(),
            expected: "expected".into(),
            actual: "actual".into(),
            passed: true,
        };
        let rendered = render_oracle_trace(&trace);
        assert_eq!(
            rendered,
            "case=case_a\nkind=canonicalize\npassed=true\nexpected=expected\nactual=actual"
        );
    }
}
