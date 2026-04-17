use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct YoungOracleCase {
    name: String,
    kind: String,
    input: serde_json::Value,
    expected: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CurvatureOracleCase {
    name: String,
    dimension: usize,
    input_kind: String,
    expected_terms: Vec<CurvatureExpectedTerm>,
}

#[derive(Debug, Deserialize)]
struct CurvatureExpectedTerm {
    kind: String,
    numer: i64,
    denom: i64,
}

#[derive(Debug, Deserialize)]
struct CanonicalizeInput {
    shape: Vec<usize>,
    slot_map: Vec<usize>,
    slots: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CanonicalizeExpected {
    canonical: String,
    sign: String,
}

#[derive(Debug, Deserialize)]
struct LrDecomposeInput {
    factors: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct LrDecomposeExpected {
    shapes: Vec<Vec<usize>>,
    multiplicities: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct CharacterInput {
    shape: Vec<usize>,
    cycle: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct CharacterExpected {
    character: String,
}

#[derive(Debug, Deserialize)]
struct MultitermReduceInput {
    symbol: String,
    identity: String,
    slots: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MultitermReduceExpected {
    reduced: String,
}

fn corpus_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/corpora")
        .join(name)
}

fn first_bianchi_identity(name: &str) -> ax_ir::TensorMultitermIdentity {
    match name {
        "first_bianchi" => ax_ir::TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [1, 2, 3],
        },
        other => panic!("unsupported oracle identity: {other}"),
    }
}

#[test]
fn young_oracle_corpus_cases_all_pass() {
    let text = std::fs::read_to_string(corpus_path("young_tableaux_oracle.json")).unwrap();
    let cases: Vec<YoungOracleCase> = serde_json::from_str(&text).unwrap();

    for case in cases {
        match case.kind.as_str() {
            "canonicalize" => {
                let input: CanonicalizeInput = serde_json::from_value(case.input).unwrap();
                let expected: CanonicalizeExpected = serde_json::from_value(case.expected).unwrap();
                let (canonical, sign) = ax_tensor::oracle_canonicalize_string(
                    &input.shape,
                    &input.slot_map,
                    &input.slots,
                )
                .unwrap();
                assert_eq!(canonical, expected.canonical, "case {}", case.name);
                assert_eq!(sign.to_string(), expected.sign, "case {}", case.name);
            }
            "lr_decompose" => {
                let input: LrDecomposeInput = serde_json::from_value(case.input).unwrap();
                let expected: LrDecomposeExpected = serde_json::from_value(case.expected).unwrap();
                let (shapes, multiplicities) =
                    ax_tensor::oracle_lr_decompose(&input.factors).unwrap();
                assert_eq!(shapes, expected.shapes, "case {}", case.name);
                assert_eq!(
                    multiplicities, expected.multiplicities,
                    "case {}",
                    case.name
                );
            }
            "character" => {
                let input: CharacterInput = serde_json::from_value(case.input).unwrap();
                let expected: CharacterExpected = serde_json::from_value(case.expected).unwrap();
                let diagram = ax_young::YoungDiagram::try_new(input.shape).unwrap();
                let actual = ax_young::symmetric_group_character(&diagram, &input.cycle)
                    .unwrap()
                    .to_string();
                assert_eq!(actual, expected.character, "case {}", case.name);
            }
            "multiterm_reduce" => {
                let input: MultitermReduceInput = serde_json::from_value(case.input).unwrap();
                let expected: MultitermReduceExpected =
                    serde_json::from_value(case.expected).unwrap();
                let actual = ax_tensor::oracle_multiterm_reduce_string(
                    &input.symbol,
                    &input.slots,
                    &first_bianchi_identity(&input.identity),
                )
                .unwrap();
                assert_eq!(actual, expected.reduced, "case {}", case.name);
            }
            other => panic!("unsupported young oracle kind: {other}"),
        }
    }
}

#[test]
fn curvature_oracle_corpus_cases_all_pass() {
    let text = std::fs::read_to_string(corpus_path("curvature_oracle.json")).unwrap();
    let cases: Vec<CurvatureOracleCase> = serde_json::from_str(&text).unwrap();

    for case in cases {
        let actual = ax_tensor::oracle_curvature_terms(&case.input_kind, case.dimension).unwrap();
        let expected = case
            .expected_terms
            .into_iter()
            .map(|term| (term.kind, term.numer, term.denom))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "case {}", case.name);
    }
}
